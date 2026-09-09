"""Serial background job runner for the video → clips pipeline.

The web UI starts a pipeline run from a URL or an uploaded file without introducing
Celery/Redis/Docker. ``JobRunner`` owns one daemon worker thread and runs one job at a time,
which also serializes the GPU-heavy transcription stage. Stage functions persist their normal
artifacts (media, transcript, clips.json) and patch the project's ``meta.json``; job state is
only live UI metadata delivered over a ``JobBus``.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.error
import uuid
from dataclasses import dataclass, field

from clipgen import clips as clips_mod
from clipgen.export import export_full_video
from clipgen.ingest import ingest_file, ingest_url
from clipgen.judge import classify_content, judge_transcript, refine_clips
from clipgen.lines import build_lines, roleify_speakers, select_chapter_lines
from clipgen.paths import Workspace
from clipgen.projects import load_meta, update_meta
from clipgen.server.progress import JobBus, ProgressEvent, Stage, Status
from clipgen.transcribe import load_transcript, transcribe_project

# ASR model: a local faster-whisper dir (offline) or a bare model name whisperx downloads.
# Overridable via the ``TRANSCRIBE_MODEL`` env var (see server/app.py).
DEFAULT_TRANSCRIBE_MODEL = "large-v3"

# How many clips to keep after the verify pass + grounding — a quality bar, not a quota: a run
# may return fewer. The judge over-generates ~2x candidates for the verifier to cut down.
TOP_K = 10


@dataclass
class JobRecord:
    """Lightweight UI state for one submitted pipeline job."""

    id: str
    project_id: str
    source_kind: str  # "url" | "file"
    source: str  # the URL, or a temp path to the uploaded file
    # "full" (ingest→transcribe→judge) | "judge" (re-prompt on cached txt) |
    # "transcribe" (ingest→transcribe→export video+JSON, no Gemini) |
    # "export" (re-export video+JSON from the cached media+transcript)
    mode: str = "full"
    # 0-based indices (into the source's start-sorted chapters) to send to the judge; None = all.
    # Set on a judge-only run resumed from the chapter-selection pause.
    selected_chapters: list[int] | None = None
    status: str = "queued"
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "source_kind": self.source_kind,
            "mode": self.mode,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobRunner:
    """One-worker pipeline runner that emits progress to a ``JobBus``."""

    def __init__(
        self,
        workspace: Workspace,
        bus: JobBus,
        *,
        transcribe_model: str | None = None,
    ) -> None:
        self.workspace = workspace
        self.bus = bus
        self.transcribe_model = transcribe_model or DEFAULT_TRANSCRIBE_MODEL
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._worker = threading.Thread(
            target=self._work_loop, name="clipgen-job-runner", daemon=True
        )
        self._worker.start()

    def submit(
        self, project_id: str, source_kind: str, source: str, mode: str = "full"
    ) -> JobRecord:
        """Queue a pipeline run for a project and return its job record.

        ``mode`` is ``"full"`` (ingest→transcribe→judge) or ``"transcribe"`` (ingest→transcribe→
        export the transcription JSON, skipping the Gemini pass).
        """
        job_id = uuid.uuid4().hex[:12]
        record = JobRecord(
            id=job_id, project_id=project_id, source_kind=source_kind, source=source, mode=mode
        )
        with self._lock:
            self._jobs[job_id] = record
        self.bus.publish(
            ProgressEvent(job_id, "ingest", "queued", "waiting for the worker", 0.0,
                          {"project_id": project_id, "mode": mode})
        )
        self._queue.put(job_id)
        return record

    def reprompt(self, project_id: str, selected_chapters: list[int] | None = None) -> JobRecord:
        """Queue a judge-only run: reuse the cached transcript, judge, (re)write clips.json.

        ``selected_chapters`` restricts the transcript to those chapters (see
        :func:`clipgen.lines.select_chapter_lines`); ``None`` judges the whole transcript. This
        path serves both the manual "re-prompt" button and the resume after the chapter-selection
        pause.
        """
        job_id = uuid.uuid4().hex[:12]
        record = JobRecord(
            id=job_id, project_id=project_id, source_kind="url", source="", mode="judge",
            selected_chapters=selected_chapters,
        )
        with self._lock:
            self._jobs[job_id] = record
        self.bus.publish(
            ProgressEvent(job_id, "judge", "queued", "waiting for the worker", 0.0,
                          {"project_id": project_id})
        )
        self._queue.put(job_id)
        return record

    def export_project(self, project_id: str) -> JobRecord:
        """Queue a re-export: transcode the cached video + rewrite the transcription JSON.

        No re-download/transcribe/Gemini — reuses the cached ``video.mp4`` + ``transcript.json``.
        Serves the "Export video + transcript" action for no-Gemini (transcript-only) projects.
        """
        job_id = uuid.uuid4().hex[:12]
        record = JobRecord(
            id=job_id, project_id=project_id, source_kind="url", source="", mode="export"
        )
        with self._lock:
            self._jobs[job_id] = record
        self.bus.publish(
            ProgressEvent(job_id, "judge", "queued", "waiting for the worker", 0.0,
                          {"project_id": project_id, "mode": "export"})
        )
        self._queue.put(job_id)
        return record

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict]:
        """Return current/recent jobs, newest first."""
        with self._lock:
            jobs = [job.to_dict() for job in self._jobs.values()]
        return sorted(jobs, key=lambda job: job["created_at"], reverse=True)

    def _update_job(self, job_id: str, **updates: object) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            for key, value in updates.items():
                setattr(record, key, value)
            record.updated_at = time.time()

    def _emit(
        self,
        job_id: str,
        project_id: str,
        stage: Stage,
        status: Status,
        message: str,
        progress: float | None = None,
        detail: dict[str, object] | None = None,
    ) -> None:
        # Carry the job's mode on every event so the UI can label the third stage (Judge vs Export)
        # correctly even when only the last event is replayed on reconnect.
        record = self.get_job(job_id)
        payload: dict[str, object] = {"project_id": project_id}
        if record is not None:
            payload["mode"] = record.mode
        payload.update(detail or {})
        self.bus.publish(ProgressEvent(job_id, stage, status, message, progress, payload))

    def _work_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._run_job(job_id)
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        record = self.get_job(job_id)
        if record is None:
            return
        pid = record.project_id
        # A judge-only re-run / re-export never touches ingest/transcribe; start the stage tracker
        # at the final row so an early failure is reported against the right stage.
        cached = record.mode in ("judge", "export")
        stage: Stage = "judge" if cached else "ingest"
        try:
            self.workspace.ensure_project_dir(pid)
            self._update_job(job_id, status="running")

            if cached:
                # Reuse the cached media + transcript; show the first two stages as already done so
                # the three-row progress UI stays coherent.
                self._emit(job_id, pid, "ingest", "done", "using cached media", 1.0)
                self._emit(job_id, pid, "transcribe", "done", "using existing transcript", 1.0)
            else:
                # 1) Ingest — download a URL or copy the uploaded file, then extract audio.
                stage = "ingest"
                update_meta(self.workspace, pid, status="ingesting")
                self._emit(job_id, pid, "ingest", "running", "starting ingest", 0.0)
                on_ingest = lambda m, p: self._emit(  # noqa: E731
                    job_id, pid, "ingest", "running", m, p
                )
                if record.source_kind == "url":
                    ingest_url(pid, record.source, self.workspace, on_progress=on_ingest)
                else:
                    ingest_file(pid, record.source, self.workspace, on_progress=on_ingest)
                    # The upload was staged to a temp path we own; drop it now it's copied in.
                    try:
                        os.unlink(record.source)
                    except OSError:
                        pass
                self._emit(job_id, pid, "ingest", "done", "media ready", 1.0)

                # 2) Transcribe — WhisperX ASR + align (+ diarize when an HF token is present).
                stage = "transcribe"
                update_meta(self.workspace, pid, status="transcribing")
                hf_token = os.environ.get("HF_TOKEN")
                diarize = bool(hf_token)
                self._emit(
                    job_id, pid, "transcribe", "running",
                    "starting transcription (detecting speakers)" if diarize
                    else "starting transcription",
                    0.0,
                )
                transcribe_project(
                    pid,
                    str(self.workspace.audio_path(pid)),
                    self.workspace,
                    model=self.transcribe_model,
                    diarize=diarize,
                    hf_token=hf_token,
                    on_progress=lambda m, p: self._emit(
                        job_id, pid, "transcribe", "running", m, p
                    ),
                )
                self._emit(job_id, pid, "transcribe", "done", "transcript persisted", 1.0)

            # 3) Finish. transcribe/export modes render the DaVinci video + JSON and stop (no
            # Gemini). Otherwise: a judge-only re-run carries its own chapter selection; a full run
            # with source chapters pauses so the operator can pick chapters before the Gemini call.
            stage = "judge"
            if record.mode in ("transcribe", "export"):
                self._export_full_video_stage(job_id, pid)
            elif record.mode == "judge":
                self._judge_stage(job_id, pid, selected_chapters=record.selected_chapters)
            elif self._pause_for_chapters(job_id, pid):
                return  # awaiting the operator's selection; a follow-up /judge job resumes the run
            else:
                self._judge_stage(job_id, pid)
        except Exception as exc:  # noqa: BLE001 - jobs must report errors through SSE
            message = self._friendly_error(exc, stage)
            update_meta(self.workspace, pid, status="error", error=message)
            self._update_job(job_id, status="error", error=message)
            self._emit(job_id, pid, stage, "error", message, None, {"job_status": "error"})

    def _export_full_video_stage(self, job_id: str, pid: str) -> None:
        """No-Gemini finish: transcode the whole video to a DaVinci ``.mov`` + write the transcript
        JSON to Downloads. Serves both transcribe-only runs and the re-export action.

        Reuses the third ("judge") progress row, which the UI labels "Export" in these modes.
        """
        self._emit(job_id, pid, "judge", "running",
                   "transcoding video (DNxHR) + writing transcript JSON…", 0.0)
        transcript = load_transcript(self.workspace, pid)
        if transcript is None:
            raise ValueError("no transcript to export — transcribe the project first")
        video = self.workspace.video_path(pid)
        if not video.is_file():
            raise FileNotFoundError("no media for this project to export from")
        meta = load_meta(self.workspace, pid)
        res = export_full_video(
            video,
            transcript,
            title=meta.title if meta else "",
            source_url=meta.source_url if meta else "",
            duration_s=meta.duration_s if meta else 0.0,
            chapters=meta.chapters if meta else [],
        )
        update_meta(self.workspace, pid, status="done", error="")
        self._update_job(job_id, status="done")
        json_name = res.transcript.name if res.transcript else ""
        self._emit(
            job_id, pid, "judge", "done", f"exported → {res.video.name} + {json_name}", 1.0,
            {
                "export_ready": True,
                "video_filename": res.video.name,
                "video_path": str(res.video),
                "transcript_filename": json_name,
                "transcript_path": str(res.transcript) if res.transcript else None,
                "job_status": "done",
            },
        )

    def _pause_for_chapters(self, job_id: str, pid: str) -> bool:
        """After transcribe, pause a full run so the operator can pick which chapters to send.

        Returns True when it paused (source chapters exist): the job ends in an ``awaiting`` state
        and a follow-up ``/judge`` call resumes it with a selection. Returns False when there are
        no chapters to choose from, so the caller should judge the whole transcript straight away.
        """
        meta = load_meta(self.workspace, pid)
        chapters = meta.chapters if meta else []
        if not chapters:
            return False
        update_meta(self.workspace, pid, status="awaiting")
        self._update_job(job_id, status="awaiting")
        self._emit(
            job_id, pid, "judge", "queued",
            f"transcribed — pick which of {len(chapters)} chapters to send to Gemini", None,
            {
                "awaiting_selection": True,
                "chapters": chapters,
                "duration_s": meta.duration_s if meta else 0.0,
                "job_status": "awaiting",
            },
        )
        return True

    def _judge_stage(
        self, job_id: str, pid: str, *, selected_chapters: list[int] | None = None
    ) -> None:
        """Run Gemini over the cached transcript and ground the result into ``clips.json``.

        Shared by the full pipeline and the judge-only re-prompt path. Reads the transcript +
        chapters from disk, so it is safe to call without having just transcribed.
        ``selected_chapters`` (0-based indices into the start-sorted chapters, or ``None`` for all)
        restricts the transcript to those chapters before judging.
        """
        update_meta(self.workspace, pid, status="judging")
        self._emit(job_id, pid, "judge", "running", "starting analysis", 0.0)
        transcript = load_transcript(self.workspace, pid)
        if transcript is None:
            raise ValueError("no transcript to judge — transcribe the project first")
        lines = build_lines(transcript, speaker_map=roleify_speakers(transcript))
        meta = load_meta(self.workspace, pid)
        chapters = meta.chapters if meta else []
        if selected_chapters is not None and chapters:
            lines, chapters = select_chapter_lines(lines, chapters, selected_chapters)
            self._emit(job_id, pid, "judge", "running",
                       f"sending {len(chapters)} selected chapter(s) to Gemini", None)
        judge_progress = lambda m, p: self._emit(  # noqa: E731
            job_id, pid, "judge", "running", m, p
        )
        content_info = classify_content(lines, on_progress=judge_progress)
        if chapters:
            self._emit(job_id, pid, "judge", "running",
                       f"using {len(chapters)} video chapters as segmentation hints", None)
        judged = judge_transcript(
            lines,
            content_info=content_info,
            target=max(TOP_K * 2, 20),  # over-generate; the verify pass + grounding cut down
            chapters=chapters,
            on_progress=judge_progress,
            prompt_path=self.workspace.judge_prompt_path(pid),
            response_path=self.workspace.judge_response_path(pid),
        )
        # Pass B: re-judge every candidate against its actual excerpt — kills clickbait and
        # merely-informative moments, tightens boundaries, re-scores. Fail-open on API trouble.
        rejects: list[dict] = []
        judged = refine_clips(
            judged,
            lines,
            rejects=rejects,
            on_progress=judge_progress,
            prompt_path=self.workspace.refine_prompt_path(pid),
            response_path=self.workspace.refine_response_path(pid),
        )
        items = clips_mod.ground(judged, lines, top_k=TOP_K, rejects=rejects)
        clips_mod.save_clips(self.workspace.clips_path(pid), items)
        self.workspace.judge_rejects_path(pid).write_text(
            json.dumps({"rejects": rejects}, indent=2)
        )

        update_meta(self.workspace, pid, status="done", error="")
        self._update_job(job_id, status="done")
        self._emit(
            job_id, pid, "judge", "done", f"clips ready ({len(items)})", 1.0,
            {"n_clips": len(items), "clips_ready": True, "job_status": "done"},
        )

    def _friendly_error(self, exc: Exception, stage: Stage) -> str:
        """Translate common failures into UI-readable errors."""
        message = str(exc)
        if stage == "judge" and "GEMINI_API_KEY" in message:
            return "Gemini API key missing — set GEMINI_API_KEY in the repo-root .env."
        if stage == "judge" and (
            isinstance(exc, urllib.error.URLError) or "Connection refused" in message
        ):
            return "The Gemini judge is unreachable — needs internet + a valid GEMINI_API_KEY."
        if stage == "transcribe" and ("gated repo" in message or "restricted" in message):
            return (
                "Speaker diarization is blocked: your HF token can't access the gated pyannote "
                "model. Accept the terms at "
                "huggingface.co/pyannote/speaker-diarization-community-1 with the same account "
                "as your HF_TOKEN, then retry."
            )
        if stage == "transcribe" and ("401" in message or "invalid" in message.lower()) and (
            "huggingface" in message.lower() or "token" in message.lower()
        ):
            return "Hugging Face rejected the diarization token — check HF_TOKEN in the .env."
        return message or exc.__class__.__name__
