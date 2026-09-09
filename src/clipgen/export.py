"""Export a grounded clip as a DaVinci Resolve-friendly file.

The free DaVinci Resolve build on Linux can't decode H.264/H.265/AAC (those are Studio-only there
and need a licensed GPU path), so a raw MP4 clip won't import. Each export is transcoded to
**DNxHR in a QuickTime ``.mov`` with PCM audio** — the edit intermediate the free Linux build reads
natively. Compound clips are cut per spoken segment and concatenated, so the exported file matches
the in-app preview (the spliced-out filler is gone). Files land in the operator's Downloads folder,
named after the judge's hook.

Only the ffmpeg calls touch I/O; the name/path helpers are pure and unit-tested.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from clipgen.lines import roleify_speakers

if TYPE_CHECKING:
    from clipgen.clips import ClipItem
    from clipgen.transcript import Transcript

_CONTAINER_SUFFIX = ".mov"
_MAX_NAME_LEN = 120

# DNxHR profile → the pixel format that profile requires. lb/sq/hq are 8-bit 4:2:2; hqx/444 are
# 10-bit. dnxhr_sq ("standard quality") is the default: edit-ready and roughly half the size of hq.
_PROFILE_PIX_FMT = {
    "dnxhr_lb": "yuv422p",
    "dnxhr_sq": "yuv422p",
    "dnxhr_hq": "yuv422p",
    "dnxhr_hqx": "yuv422p10le",
    "dnxhr_444": "yuv444p10le",
}
_DEFAULT_PROFILE = "dnxhr_sq"


@dataclass
class ExportResult:
    """Where an export landed: the rendered ``.mov`` and its sidecar transcription JSON (if any)."""

    video: Path
    transcript: Path | None = None


def export_dir() -> Path:
    """Where exports land — Downloads by default (override with ``CLIPGEN_EXPORT_DIR``)."""
    override = os.environ.get("CLIPGEN_EXPORT_DIR")
    return Path(override).expanduser() if override else Path.home() / "Downloads"


def _profile() -> str:
    """The DNxHR profile to encode with (``CLIPGEN_DNXHR_PROFILE`` env, validated)."""
    requested = os.environ.get("CLIPGEN_DNXHR_PROFILE", _DEFAULT_PROFILE)
    return requested if requested in _PROFILE_PIX_FMT else _DEFAULT_PROFILE


def safe_filename(title: str, *, fallback: str = "clip") -> str:
    """Turn a clip hook into a filesystem-safe base name (no extension).

    Strips characters illegal on common filesystems, collapses whitespace, drops leading/trailing
    dots and spaces, and caps the length. Falls back to ``fallback`` when the title is empty or
    sanitizes away to nothing.
    """
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "", title)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if len(cleaned) > _MAX_NAME_LEN:
        cleaned = cleaned[:_MAX_NAME_LEN].rstrip(" .")
    return cleaned or fallback


def unique_path(directory: Path, base: str, suffix: str = _CONTAINER_SUFFIX) -> Path:
    """A non-colliding path in ``directory``: ``base.suffix``, then ``base (2).suffix``, …."""
    candidate = directory / f"{base}{suffix}"
    n = 2
    while candidate.exists():
        candidate = directory / f"{base} ({n}){suffix}"
        n += 1
    return candidate


def _encode_args() -> list[str]:
    """ffmpeg codec args for a DaVinci-ready ``.mov``: DNxHR video + PCM audio."""
    profile = _profile()
    return [
        "-c:v", "dnxhd", "-profile:v", profile, "-pix_fmt", _PROFILE_PIX_FMT[profile],
        "-c:a", "pcm_s16le", "-ar", "48000",
    ]


def _cut_segment(video: Path, start_s: float, end_s: float, out: Path) -> None:
    """Transcode ``[start_s, end_s)`` of ``video`` to a DNxHR/PCM ``.mov`` at ``out`` (I/O).

    ``-ss`` before ``-i`` with a re-encode is frame-accurate (ffmpeg decodes from the preceding
    keyframe and drops the earlier frames) and fast; ``-t`` bounds the duration.
    """
    duration = max(0.0, end_s - start_s)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", f"{start_s:.3f}", "-i", str(video), "-t", f"{duration:.3f}",
            *_encode_args(),
            str(out),
        ],
        check=True,
        capture_output=True,
    )


def _transcode(video: Path, out: Path) -> None:
    """Transcode the whole of ``video`` to a DNxHR/PCM ``.mov`` at ``out`` (I/O)."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), *_encode_args(), str(out)],
        check=True,
        capture_output=True,
    )


def _concat(parts: list[Path], out: Path) -> None:
    """Concatenate identically-encoded ``.mov`` parts into ``out`` via the concat demuxer (I/O)."""
    list_file = parts[0].parent / "concat.txt"
    list_file.write_text("".join(f"file '{p.resolve()}'\n" for p in parts), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c", "copy", str(out)],
        check=True,
        capture_output=True,
    )


def clip_transcript_dict(clip: ClipItem, transcript: Transcript) -> dict:
    """The clip's word-level transcription, timed to the exported ``.mov`` (pure).

    Word timings are **clip-relative**: for a compound clip the spliced-out gaps are removed, so the
    timeline matches the concatenated export (a word's time is its offset from the clip's start, not
    from the source video). Speaker labels are the same neutral ``Person A/B/…`` names the UI shows.
    ``source_segments`` keeps the absolute source spans for reference.
    """
    segments = clip.segments or [[clip.start_s, clip.end_s]]
    speaker_map = roleify_speakers(transcript)
    words: list[dict] = []
    offset = 0.0
    for raw_start, raw_end in segments:
        seg_start, seg_end = float(raw_start), float(raw_end)
        for w in transcript.words_between(seg_start, seg_end):
            start = max(w.start_s, seg_start) - seg_start + offset
            end = min(w.end_s, seg_end) - seg_start + offset
            entry: dict = {"text": w.text, "start_s": round(start, 3), "end_s": round(end, 3)}
            speaker = speaker_map.get(w.speaker, speaker_map.get(None))
            if speaker is not None:
                entry["speaker"] = speaker
            words.append(entry)
        offset += seg_end - seg_start
    return {
        "hook": clip.hook,
        "hook_sentence": clip.hook_sentence,
        "reason": clip.reason,
        "duration_s": round(offset, 3),
        "timing": "clip-relative seconds (spliced gaps removed) — aligned to the exported .mov",
        "source_segments": [[round(float(s), 3), round(float(e), 3)] for s, e in segments],
        "scores": dict(clip.scores),
        "text": clip.text,
        "words": words,
    }


def transcript_dict(
    transcript: Transcript,
    *,
    title: str = "",
    source_url: str = "",
    duration_s: float = 0.0,
    chapters: list[dict] | None = None,
) -> dict:
    """The whole video's word-level transcription as a plain dict (pure).

    Timings are **absolute** source-video seconds (there are no clips to be relative to). Speaker
    labels are the neutral ``Person A/B/…`` names the UI shows. Used by the transcribe-only mode,
    which skips the Gemini pass entirely.
    """
    speaker_map = roleify_speakers(transcript)
    words: list[dict] = []
    for w in transcript.words:
        entry: dict = {"text": w.text, "start_s": round(w.start_s, 3), "end_s": round(w.end_s, 3)}
        speaker = speaker_map.get(w.speaker, speaker_map.get(None))
        if speaker is not None:
            entry["speaker"] = speaker
        words.append(entry)
    return {
        "title": title,
        "source_url": source_url,
        "duration_s": round(float(duration_s), 3),
        "timing": "absolute source-video seconds",
        "chapters": [
            {"start_s": round(float(c["start_s"]), 3), "title": c.get("title", "")}
            for c in (chapters or [])
        ],
        "text": " ".join(w.text for w in transcript.words).strip(),
        "words": words,
    }


def export_full_video(
    video: Path,
    transcript: Transcript,
    *,
    title: str = "",
    source_url: str = "",
    duration_s: float = 0.0,
    chapters: list[dict] | None = None,
    dest_dir: Path | None = None,
) -> ExportResult:
    """Transcode all of ``video`` to a DaVinci-ready ``.mov`` + write its transcription JSON (I/O).

    The transcribe-only / "direct" counterpart to :func:`export_clip`: no Gemini, no clip cutting —
    the entire video is transcoded to DNxHR + PCM (so the free DaVinci Resolve on Linux can import
    it) and paired with a ``<name>.json`` of the whole transcript (absolute timings). Both are named
    after the title and land in ``dest_dir`` (Downloads). Returns the written paths.
    """
    if not video.is_file():
        raise FileNotFoundError(f"source video not found: {video}")
    dest_dir = dest_dir or export_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_video = unique_path(dest_dir, safe_filename(title, fallback="video"), _CONTAINER_SUFFIX)
    _transcode(video, out_video)
    out_json = out_video.with_suffix(".json")  # same (already-unique) stem, kept paired
    payload = transcript_dict(
        transcript, title=title, source_url=source_url, duration_s=duration_s, chapters=chapters
    )
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return ExportResult(video=out_video, transcript=out_json)


def export_clip(
    video: Path,
    clip: ClipItem,
    dest_dir: Path | None = None,
    *,
    transcript: Transcript | None = None,
) -> ExportResult:
    """Render ``clip`` from ``video`` to a DaVinci-ready ``.mov`` in ``dest_dir`` (I/O).

    Compound clips are cut per spoken segment and concatenated so the file matches the preview. The
    output is named after the clip's hook (see :func:`safe_filename`) and de-duplicated within the
    directory. When ``transcript`` is given, a sidecar ``<name>.json`` with the clip's word-level
    transcription (see :func:`clip_transcript_dict`) is written alongside it. Returns the paths.
    """
    if not video.is_file():
        raise FileNotFoundError(f"source video not found: {video}")
    segments = clip.segments or [[clip.start_s, clip.end_s]]
    dest_dir = dest_dir or export_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = safe_filename(clip.hook or clip.hook_sentence)
    out_path = unique_path(dest_dir, base)

    with tempfile.TemporaryDirectory(prefix="clipgen-export-") as tmp:
        parts: list[Path] = []
        for i, (start_s, end_s) in enumerate(segments):
            part = Path(tmp) / f"seg{i:03d}{_CONTAINER_SUFFIX}"
            _cut_segment(video, float(start_s), float(end_s), part)
            parts.append(part)
        if len(parts) == 1:
            shutil.move(str(parts[0]), str(out_path))
        else:
            _concat(parts, out_path)

    transcript_path: Path | None = None
    if transcript is not None:
        # Same stem as the .mov (which is already unique), so the two files stay paired.
        transcript_path = out_path.with_suffix(".json")
        transcript_path.write_text(
            json.dumps(clip_transcript_dict(clip, transcript), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return ExportResult(video=out_path, transcript=transcript_path)
