"""Ingest: get a source video into a project directory, from a URL or a local file.

Two entry points share the audio-extraction tail:
- :func:`ingest_url` pulls a video with yt-dlp (height-capped, non-AV1 preferred so ffmpeg can
  decode it) and reads its title/duration from the info dict.
- :func:`ingest_file` copies an already-downloaded local file into the project and probes its
  title/duration with ffprobe.

Both write ``video.mp4`` + ``audio.wav`` under the project dir and update the project's
``meta.json`` (title, source_url, duration_s). The download/ffmpeg steps are thin I/O wrappers.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from clipgen.paths import Workspace
from clipgen.projects import update_meta

if TYPE_CHECKING:
    from collections.abc import Callable

    ProgressCb = Callable[[str, float | None], None]


def _extract_audio(video_path: str, audio_path: str) -> None:
    """ffmpeg-extract a 16 kHz mono WAV for ASR (I/O)."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", "-vn", audio_path],
        check=True,
        capture_output=True,
    )


def _probe(video_path: str) -> dict:
    """ffprobe a local file for duration + a title tag (best-effort)."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", video_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        fmt = json.loads(out.stdout).get("format", {})
        duration = float(fmt.get("duration") or 0.0)
        title = (fmt.get("tags", {}) or {}).get("title", "")
        return {"duration_s": duration, "title": title}
    except (subprocess.CalledProcessError, ValueError, OSError):
        return {"duration_s": 0.0, "title": ""}


def _chapters_from_info(info: dict) -> list[dict]:
    """Normalize yt-dlp's chapter list to ``[{start_s, title}, …]`` (empty if none).

    yt-dlp gives ``[{"start_time", "end_time", "title"}, …]`` on chaptered videos and ``None``
    otherwise. We keep only the start time + title — the judge places its own boundaries; chapters
    are just a hint about where coherent topics begin.
    """
    chapters: list[dict] = []
    for ch in info.get("chapters") or []:
        if not isinstance(ch, dict):
            continue
        start = ch.get("start_time")
        if start is None:
            continue
        chapters.append({"start_s": float(start), "title": str(ch.get("title", "")).strip()})
    chapters.sort(key=lambda c: c["start_s"])
    return chapters


def _human_rate(bytes_per_s: float | None) -> str:
    """Format a download speed (bytes/s) as a clean ``KiB/s``/``MiB/s`` string."""
    if not bytes_per_s or bytes_per_s <= 0:
        return "?"
    units = ["B/s", "KiB/s", "MiB/s", "GiB/s"]
    rate = float(bytes_per_s)
    i = 0
    while rate >= 1024 and i < len(units) - 1:
        rate /= 1024
        i += 1
    return f"{rate:.1f} {units[i]}"


def _human_eta(seconds: float | None) -> str:
    """Format a remaining-time estimate (seconds) as ``m:ss`` (or ``h:mm:ss``)."""
    if seconds is None or seconds < 0:
        return "?"
    total = int(seconds)
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _download_media(
    url: str,
    out_path: str,
    *,
    max_height: int = 1440,
    on_progress: ProgressCb | None = None,
) -> dict:
    """Probe + download the (height-capped) video to ``out_path`` (I/O). Returns the info dict."""
    import yt_dlp

    def _hook(d: dict) -> None:
        if d.get("status") == "downloading":
            # Build the message from numeric fields, not yt-dlp's pre-rendered `_percent_str`
            # (those carry ANSI codes that show as garbage in the UI).
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            frac = d.get("downloaded_bytes", 0) / total if total else None
            frac = max(0.0, min(1.0, frac)) if frac is not None else None
            pct = f"{frac * 100:.1f}%" if frac is not None else "?"
            spd = _human_rate(d.get("speed"))
            eta = _human_eta(d.get("eta"))
            if on_progress:
                on_progress(f"downloading {pct} at {spd} · ETA {eta}", frac)
        elif d.get("status") == "finished" and on_progress:
            on_progress("download finished, merging…", 0.75)

    if on_progress:
        on_progress("probing video metadata", 0.02)
    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)

    # Resolution first, codec second. YouTube serves avc1 (H.264) only up to 1080p; 1440p/2160p
    # are VP9 or AV1 only. ffmpeg can't decode AV1 here, but VP9 decodes fine, so take the best
    # non-AV1 stream at the cap first (real 1440p/2160p), fall back to avc1, then AV1 last.
    fmt = (
        f"bv*[vcodec!*=av01][height<={max_height}]+ba/"
        f"b[vcodec!*=av01][height<={max_height}]/"
        f"bv*[vcodec^=avc1][height<={max_height}]+ba/"
        f"b[vcodec^=avc1][height<={max_height}]/"
        f"bv*[height<={max_height}]+ba/b[height<={max_height}]/bv*+ba/b"
    )
    dl_opts = {
        "quiet": True,
        "noprogress": True,
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": out_path,
        "overwrites": True,
        "progress_hooks": [_hook],
    }
    with yt_dlp.YoutubeDL(dl_opts) as ydl:
        ydl.download([url])
    return info


def ingest_url(
    project_id: str,
    url: str,
    workspace: Workspace | None = None,
    *,
    max_height: int = 1440,
    on_progress: ProgressCb | None = None,
) -> None:
    """Download a URL into the project and extract audio; update the project's ``meta.json``."""
    workspace = workspace or Workspace.default()
    workspace.ensure_project_dir(project_id)
    video_path = str(workspace.video_path(project_id))

    info = _download_media(url, video_path, max_height=max_height, on_progress=on_progress)
    if on_progress:
        on_progress("extracting audio (16 kHz mono)…", 0.85)
    _extract_audio(video_path, str(workspace.audio_path(project_id)))

    update_meta(
        workspace,
        project_id,
        title=info.get("title", "") or "",
        source_url=info.get("webpage_url", url) or url,
        duration_s=float(info.get("duration") or 0.0),
        chapters=_chapters_from_info(info),
    )
    if on_progress:
        on_progress("media ready", 1.0)


def ingest_file(
    project_id: str,
    src_path: str,
    workspace: Workspace | None = None,
    *,
    on_progress: ProgressCb | None = None,
) -> None:
    """Copy a local video file into the project and extract audio; update ``meta.json``.

    The file is copied (not referenced) so the project directory is self-contained and safe to
    delete without touching the operator's original file.
    """
    workspace = workspace or Workspace.default()
    workspace.ensure_project_dir(project_id)
    src = Path(src_path)
    if not src.is_file():
        raise FileNotFoundError(f"source video not found: {src_path}")
    video_path = workspace.video_path(project_id)

    if on_progress:
        on_progress("copying source file", 0.1)
    shutil.copyfile(src, video_path)
    if on_progress:
        on_progress("extracting audio (16 kHz mono)…", 0.6)
    _extract_audio(str(video_path), str(workspace.audio_path(project_id)))

    probe = _probe(str(video_path))
    update_meta(
        workspace,
        project_id,
        title=probe["title"] or src.stem,
        source_url="",
        duration_s=probe["duration_s"],
    )
    if on_progress:
        on_progress("media ready", 1.0)
