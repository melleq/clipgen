"""FastAPI app for the clipgen web UI.

A small surface over the file-based project store and the background pipeline runner:
- list / get / delete projects and stream their source media,
- create a project from a URL or an uploaded file (which queues a pipeline job),
- subscribe to a job's live progress over Server-Sent Events.

There is no CLI workflow — this app is the only entry point. Single-user and local, so the
store is a directory scan, CORS is fully open, and one worker runs one job at a time.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import FileResponse, PlainTextResponse, StreamingResponse

from clipgen.clips import adjust_clip, ground, load_clips, merge_clips, save_clips
from clipgen.env import load_env
from clipgen.export import export_clip
from clipgen.judge import parse_clips
from clipgen.lines import build_lines, roleify_speakers
from clipgen.paths import Workspace
from clipgen.projects import (
    ProjectMeta,
    delete_project,
    list_projects,
    load_meta,
    new_project_id,
    save_meta,
)
from clipgen.server.jobs import JobRunner
from clipgen.server.progress import JobBus, ProgressEvent
from clipgen.transcribe import load_transcript

load_env()  # GEMINI_API_KEY / HF_TOKEN / model paths from the repo-root .env, before any job runs

app = FastAPI(title="clipgen")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local single-user tool
    allow_methods=["*"],
    allow_headers=["*"],
)

WORKSPACE = Workspace.default()
BUS = JobBus()
RUNNER = JobRunner(WORKSPACE, BUS, transcribe_model=os.environ.get("TRANSCRIBE_MODEL"))


class SubmitUrlBody(BaseModel):
    url: str
    # Skip the Gemini pass: ingest → transcribe → export the transcription JSON only.
    transcribe_only: bool = False


def _project_response(project_id: str) -> dict:
    """Meta + grounded clips for one project, or 404 if it doesn't exist."""
    meta = load_meta(WORKSPACE, project_id)
    if meta is None:
        raise HTTPException(404, f"project '{project_id}' not found")
    clips = load_clips(WORKSPACE.clips_path(project_id))
    summary = meta.to_dict()
    summary["n_clips"] = len(clips)
    summary["has_video"] = WORKSPACE.video_path(project_id).is_file()
    return {"project": summary, "clips": [c.to_dict() for c in clips]}


@app.get("/api/projects")
def get_projects() -> list[dict]:
    """Every project as a summary row (meta + n_clips), newest first."""
    return list_projects(WORKSPACE)


@app.post("/api/projects", status_code=202)
def create_project_from_url(body: SubmitUrlBody) -> dict:
    """Create a project from a video URL and queue its pipeline run.

    ``transcribe_only`` skips Gemini: the run stops after transcription and exports the JSON.
    """
    url = body.url.strip()
    if not url:
        raise HTTPException(400, "url is required")
    pid = new_project_id()
    save_meta(WORKSPACE, ProjectMeta(id=pid, source_url=url, status="queued"))
    mode = "transcribe" if body.transcribe_only else "full"
    job = RUNNER.submit(pid, "url", url, mode=mode).to_dict()
    return {"project_id": pid, "job_id": job["id"], "job": job}


@app.post("/api/projects/upload", status_code=202)
async def create_project_from_file(file: UploadFile, transcribe_only: bool = False) -> dict:
    """Create a project from an uploaded local video file and queue its pipeline run.

    The upload is streamed to a temp file the worker copies into the project directory (and then
    deletes), so a partially-uploaded file never lands in the project as ``video.mp4``.
    ``transcribe_only`` (query param) skips Gemini: stop after transcription and export the JSON.
    """
    if not (file.content_type or "").startswith("video"):
        raise HTTPException(400, "only video files are supported")
    pid = new_project_id()
    suffix = Path(file.filename or "").suffix or ".mp4"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="clipgen-upload-")
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := await file.read(1 << 20):  # 1 MiB
                out.write(chunk)
    except Exception:
        os.unlink(tmp_path)
        raise
    title = Path(file.filename or "").stem
    save_meta(WORKSPACE, ProjectMeta(id=pid, title=title, status="queued"))
    mode = "transcribe" if transcribe_only else "full"
    job = RUNNER.submit(pid, "file", tmp_path, mode=mode).to_dict()
    return {"project_id": pid, "job_id": job["id"], "job": job}


@app.post("/api/projects/{project_id}/reprompt", status_code=202)
def reprompt_project(project_id: str) -> dict:
    """Re-run Gemini + grounding on the cached transcript, overwriting the clip list.

    No re-download or re-transcribe — reuses ``transcript.json``. 404 if the project or its
    transcript is missing (nothing to judge yet).
    """
    if load_meta(WORKSPACE, project_id) is None:
        raise HTTPException(404, f"project '{project_id}' not found")
    if not WORKSPACE.transcript_path(project_id).is_file():
        raise HTTPException(404, "project has no transcript yet — nothing to re-prompt")
    job = RUNNER.reprompt(project_id).to_dict()
    return {"project_id": project_id, "job_id": job["id"], "job": job}


class JudgeBody(BaseModel):
    # 0-based indices into the source's start-sorted chapters to send to the judge; null/omitted or
    # covering every chapter sends the whole transcript. This resumes a run paused after transcribe
    # for chapter selection.
    selected_chapters: list[int] | None = None


@app.post("/api/projects/{project_id}/judge", status_code=202)
def judge_project(project_id: str, body: JudgeBody) -> dict:
    """Judge the cached transcript, optionally restricted to a subset of the video's chapters.

    Queued as a judge-only run over ``transcript.json`` (no re-download/transcribe). It resumes a
    full run that paused after transcribe so the operator could pick chapters. 404 if the project
    or its transcript is missing.
    """
    if load_meta(WORKSPACE, project_id) is None:
        raise HTTPException(404, f"project '{project_id}' not found")
    if not WORKSPACE.transcript_path(project_id).is_file():
        raise HTTPException(404, "project has no transcript yet — nothing to judge")
    job = RUNNER.reprompt(project_id, body.selected_chapters).to_dict()
    return {"project_id": project_id, "job_id": job["id"], "job": job}


@app.post("/api/projects/{project_id}/export", status_code=202)
def export_project(project_id: str) -> dict:
    """Re-export the whole video as a DaVinci-ready ``.mov`` + transcription JSON to Downloads.

    Queued as a background job over the cached ``video.mp4`` + ``transcript.json`` (no
    re-download/transcribe/Gemini) — the re-export path for no-Gemini (transcript-only) projects.
    404 if the project, its media, or its transcript is missing.
    """
    if load_meta(WORKSPACE, project_id) is None:
        raise HTTPException(404, f"project '{project_id}' not found")
    if not WORKSPACE.video_path(project_id).is_file():
        raise HTTPException(404, "no media for this project to export from")
    if not WORKSPACE.transcript_path(project_id).is_file():
        raise HTTPException(404, "project has no transcript yet — nothing to export")
    job = RUNNER.export_project(project_id).to_dict()
    return {"project_id": project_id, "job_id": job["id"], "job": job}


@app.get("/api/projects/{project_id}/prompt", response_class=PlainTextResponse)
def get_prompt(project_id: str) -> str:
    """The exact judge prompt (instructions + numbered transcript) — paste into a web LLM.

    Its reply, in the same JSON schema, can be pasted back via the import endpoint. 404 until a
    judge run has produced the prompt.
    """
    path = WORKSPACE.judge_prompt_path(project_id)
    if not path.is_file():
        raise HTTPException(404, "no prompt yet — run the judge (or re-prompt) first")
    return path.read_text(encoding="utf-8")


class ImportClipsBody(BaseModel):
    clips_json: str


@app.post("/api/projects/{project_id}/clips/import")
def import_clips(project_id: str, body: ImportClipsBody) -> dict:
    """Ground clips pasted as judge-format JSON and merge them into the project's clip list.

    The pasted JSON is the judge schema (``{"clips":[{start_line/end_line or segments, hook,
    scores, …}]}``, a bare array is also accepted) — typically an external LLM's answer to this
    project's prompt. Clips are resolved against the transcript's numbered lines (so they land on
    real timestamps) and merged with the existing list, ranked by score with overlaps de-duped.
    The hook/payoff gate is **not** applied to imports — the operator is adding them deliberately.
    """
    if load_meta(WORKSPACE, project_id) is None:
        raise HTTPException(404, f"project '{project_id}' not found")
    transcript = load_transcript(WORKSPACE, project_id)
    if transcript is None:
        raise HTTPException(404, "project has no transcript yet — nothing to ground against")

    raw = body.clips_json.strip()
    if not raw:
        raise HTTPException(400, "paste some JSON first")
    if raw.startswith("["):  # tolerate a bare clips array
        raw = '{"clips": ' + raw + "}"
    judged = parse_clips(raw)
    if not judged:
        raise HTTPException(
            400,
            "no valid clips found — expected the judge JSON schema "
            '({"clips": [{"start_line": …, "end_line": …, "scores": {…}}, …]}).',
        )

    lines = build_lines(transcript, speaker_map=roleify_speakers(transcript))
    new_items = ground(judged, lines, top_k=len(judged), apply_gate=False, source="imported")
    if not new_items:
        raise HTTPException(
            400, "the pasted clips didn't resolve to valid line ranges within the length bounds."
        )
    clips_path = WORKSPACE.clips_path(project_id)
    merged = merge_clips(load_clips(clips_path), new_items)
    save_clips(clips_path, merged)

    imported_kept = sum(1 for c in merged if c.source == "imported")
    return {"imported": imported_kept, "total": len(merged), **_project_response(project_id)}


class AdjustClipBody(BaseModel):
    edge: str  # "start" | "end"
    unit: str  # "word" | "sentence"
    direction: str  # "expand" | "trim"
    count: int = 1


@app.post("/api/projects/{project_id}/clips/{index}/adjust")
def adjust_clip_endpoint(project_id: str, index: int, body: AdjustClipBody) -> dict:
    """Nudge one clip's start/end by a word or a sentence, snapped to the transcript, and persist.

    The clip is addressed by its position in the current list (stable — adjusting doesn't reorder).
    Returns the updated project + full clip list.
    """
    if load_meta(WORKSPACE, project_id) is None:
        raise HTTPException(404, f"project '{project_id}' not found")
    transcript = load_transcript(WORKSPACE, project_id)
    if transcript is None:
        raise HTTPException(404, "project has no transcript to snap boundaries to")
    clips_path = WORKSPACE.clips_path(project_id)
    items = load_clips(clips_path)
    if not 0 <= index < len(items):
        raise HTTPException(404, f"clip index {index} out of range")
    try:
        items[index] = adjust_clip(
            items[index], transcript.words,
            edge=body.edge, unit=body.unit, direction=body.direction, count=body.count,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    save_clips(clips_path, items)  # order preserved — index stays valid
    return _project_response(project_id)


@app.post("/api/projects/{project_id}/clips/{index}/export")
def export_clip_endpoint(project_id: str, index: int) -> dict:
    """Render one clip to a DaVinci-ready ``.mov`` (+ sidecar transcription JSON) in Downloads.

    The free DaVinci Resolve on Linux can't decode H.264/H.265, so the clip is transcoded to
    DNxHR + PCM audio (compound clips concatenated to match the preview) and named after its hook.
    A ``<name>.json`` with the clip's word-level transcription (clip-relative timings) is written
    alongside it. Returns the written paths. 404 if project/clip missing; 500 if ffmpeg fails.
    """
    if load_meta(WORKSPACE, project_id) is None:
        raise HTTPException(404, f"project '{project_id}' not found")
    video = WORKSPACE.video_path(project_id)
    if not video.is_file():
        raise HTTPException(404, "no media for this project to export from")
    items = load_clips(WORKSPACE.clips_path(project_id))
    if not 0 <= index < len(items):
        raise HTTPException(404, f"clip index {index} out of range")
    transcript = load_transcript(WORKSPACE, project_id)  # sidecar JSON is skipped if this is None
    try:
        result = export_clip(video, items[index], transcript=transcript)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else ""
        raise HTTPException(500, f"ffmpeg export failed: {stderr[-500:] or exc}") from exc
    return {
        "path": str(result.video),
        "filename": result.video.name,
        "transcript_path": str(result.transcript) if result.transcript else None,
        "transcript_filename": result.transcript.name if result.transcript else None,
    }


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict:
    """A project's meta + its grounded clips."""
    return _project_response(project_id)


@app.delete("/api/projects/{project_id}")
def remove_project(project_id: str) -> dict:
    """Delete a project's entire directory (media, transcript, clips)."""
    if not delete_project(WORKSPACE, project_id):
        raise HTTPException(404, f"project '{project_id}' not found")
    return {"deleted": project_id}


@app.get("/api/projects/{project_id}/media")
def get_media(project_id: str) -> FileResponse:
    """Serve a project's source video (Range-served, for the clip preview player)."""
    path = WORKSPACE.video_path(project_id)
    if not path.is_file():
        raise HTTPException(404, "no media for this project")
    return FileResponse(path)  # Starlette handles HTTP Range (seekable)


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return RUNNER.list_jobs()


def _sse_event_name(event: ProgressEvent) -> str:
    if event.detail.get("clips_ready"):
        return "clips_ready"
    return "progress"


def _sse_frame(event: ProgressEvent) -> str:
    return f"event: {_sse_event_name(event)}\ndata: {json.dumps(event.to_dict())}\n\n"


@app.get("/api/jobs/{job_id}/events")
def job_events(job_id: str) -> StreamingResponse:
    """Server-Sent Events stream of a job's progress, with last-state replay on connect."""
    if RUNNER.get_job(job_id) is None and BUS.last(job_id) is None:
        raise HTTPException(404, f"job '{job_id}' not found")

    # Terminal for this job's stream: done/error, or "awaiting" (a full run paused for the operator
    # to pick chapters — no further events arrive until a follow-up /judge job resumes the run).
    stream_end = {"done", "error", "awaiting"}

    def stream() -> Iterator[str]:
        subscriber, last = BUS.subscribe(job_id)
        try:
            if last is not None:
                yield _sse_frame(last)
                if last.detail.get("job_status") in stream_end:
                    return
            while True:
                try:
                    event = subscriber.get(timeout=15)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                yield _sse_frame(event)
                if event.detail.get("job_status") in stream_end:
                    return
        finally:
            BUS.unsubscribe(job_id, subscriber)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
