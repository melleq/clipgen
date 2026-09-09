"""Transcript backbone via WhisperX.

Three sequential stages — ASR → forced alignment (word timestamps) → optional diarization —
produce the word-level ``Transcript``. The WhisperX-result → ``Word`` mapping is pure and
unit-tested; model loading/inference is I/O. Diarization is opt-in (pyannote weights are gated
behind a Hugging Face token); without it words carry ``speaker = None``.

The transcript is persisted as a per-project file (``data/<id>/transcript.json``).
"""

from __future__ import annotations

import gc
import json
import os
from typing import TYPE_CHECKING

from clipgen.paths import Workspace
from clipgen.transcript import Transcript, Word

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    ProgressCb = Callable[[str, float | None], None]

DEFAULT_MODEL = "large-v3"

# Recall / anti-hallucination tuning layered on top of WhisperX's own defaults (it merges these
# via ``dict.update``, so we only override the keys we care about). Capping the high-temperature
# fallback ladder keeps a failing chunk's low-temperature decode instead of hallucinating a hotter
# one; ``condition_on_previous_text`` stays off so one bad chunk can't poison the next.
DEFAULT_ASR_OPTIONS: dict = {
    "temperatures": [0.0, 0.2],
    "condition_on_previous_text": False,
}

# Slightly more inclusive speech-region boundaries than WhisperX's defaults so soft or trailing
# words at the edges of a turn (common right around crosstalk) aren't clipped before ASR sees them.
DEFAULT_VAD_OPTIONS: dict = {
    "vad_onset": 0.45,
    "vad_offset": 0.30,
}


def words_from_whisperx(result: Mapping) -> list[Word]:
    """Map a WhisperX aligned result into our ``Word`` list.

    Iterates ``result["segments"][*]["words"]``. WhisperX occasionally drops a word's
    ``start``/``end`` (unalignable tokens); such words are kept with a zero-duration timing
    carried from the previous word's end rather than dropped, so no text is lost. Picks up the
    per-word ``speaker`` field when diarization ran.
    """
    words: list[Word] = []
    last_end = 0.0
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            text = str(w.get("word", "")).strip()
            if not text:
                continue
            start = w.get("start")
            end = w.get("end")
            if start is None or end is None:
                start = end = last_end
            start, end = float(start), float(end)
            words.append(Word(text=text, start_s=start, end_s=end, speaker=w.get("speaker")))
            last_end = end
    return words


def _free(*objs: object) -> None:
    """Drop model references and clear the CUDA cache between stages (tight VRAM)."""
    for _ in objs:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - cache cleanup is best-effort
        pass


def _transcribe_with_oom_retry(
    load_asr: Callable[[], object],
    audio: object,
    batch_size: int,
    on_progress: ProgressCb | None,
) -> dict:
    """Run batched ASR, halving ``batch_size`` on CUDA OOM (down to 1) before giving up.

    The model is **reloaded** on each retry: an OOM leaves ctranslate2's CUDA context corrupted,
    so reusing the same model object fails with an unrelated error; a fresh load gets a clean one.
    """
    bs = batch_size
    while True:
        asr = load_asr()
        try:
            result = asr.transcribe(audio, batch_size=bs)  # type: ignore[attr-defined]
        except RuntimeError as e:
            del asr
            _free()  # drop the poisoned CUDA context before reloading
            if "out of memory" not in str(e).lower() or bs <= 1:
                raise
            bs = max(1, bs // 2)
            if on_progress:
                on_progress(f"low VRAM — reloading, retrying at batch_size={bs}", 0.25)
            continue
        del asr
        _free()
        return result


def transcribe(
    audio_path: str,
    *,
    model: str = DEFAULT_MODEL,
    device: str | None = None,
    compute_type: str | None = None,
    language: str | None = None,
    batch_size: int = 8,
    asr_options: Mapping | None = None,
    vad_options: Mapping | None = None,
    diarize: bool = False,
    hf_token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    on_progress: ProgressCb | None = None,
) -> Transcript:
    """Transcribe + align (+ optionally diarize) one audio file (I/O).

    Stages run sequentially, freeing each model before the next so a tight GPU isn't overloaded.
    ``device``/``compute_type`` default to CUDA+float16 when a GPU is present, else CPU+int8.
    Diarization requires a Hugging Face token (``hf_token`` or ``HF_TOKEN``).
    """
    # Fail-fast on a stalled model download instead of hanging forever.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

    import whisperx
    import whisperx.diarize  # submodule isn't auto-imported by `import whisperx`

    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        has_cuda = False
    device = device or ("cuda" if has_cuda else "cpu")
    # ``WHISPER_COMPUTE_TYPE`` lets a memory-tight GPU drop to e.g. ``int8_float16`` without a code
    # change — useful when the GPU also drives the desktop and float16 large-v3 won't fit.
    compute_type = (
        compute_type
        or os.environ.get("WHISPER_COMPUTE_TYPE")
        or ("float16" if device == "cuda" else "int8")
    )

    if on_progress:
        on_progress("loading audio", 0.02)
    audio = whisperx.load_audio(audio_path)

    # 1) ASR  (first run downloads the model from HF unless a local dir is given)
    if on_progress:
        on_progress(f"loading ASR model '{model}' on {device}", 0.08)

    def _load_asr() -> object:
        return whisperx.load_model(
            model,
            device,
            compute_type=compute_type,
            language=language,
            asr_options={**DEFAULT_ASR_OPTIONS, **(asr_options or {})},
            vad_options={**DEFAULT_VAD_OPTIONS, **(vad_options or {})},
        )

    if on_progress:
        on_progress("transcribing audio", 0.25)
    # The retry loop owns loading (it reloads on OOM) and frees the model before returning.
    result = _transcribe_with_oom_retry(_load_asr, audio, batch_size, on_progress)
    lang = result["language"]
    if on_progress:
        on_progress("ASR complete", 0.5)

    # 2) Forced alignment -> word timestamps
    if on_progress:
        on_progress(f"aligning words (lang={lang})", 0.6)
    align_model, meta = whisperx.load_align_model(language_code=lang, device=device)
    result = whisperx.align(
        result["segments"], align_model, meta, audio, device, return_char_alignments=False
    )
    _free(align_model)
    if on_progress:
        on_progress("word alignment complete", 0.82)

    # 3) Diarization (optional, gated)
    if diarize:
        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "diarization needs a Hugging Face token (pass hf_token or set HF_TOKEN) "
                "and acceptance of the pyannote model terms"
            )
        if on_progress:
            on_progress("diarizing speakers", 0.9)
        diarizer = whisperx.diarize.DiarizationPipeline(token=token, device=device)
        diarize_segments = diarizer(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        _free(diarizer)

    transcript = Transcript(words=words_from_whisperx(result))
    if on_progress:
        on_progress(f"transcript ready ({len(transcript.words)} words)", 1.0)
    return transcript


def save_transcript(workspace: Workspace, project_id: str, transcript: Transcript) -> None:
    """Persist a transcript to ``data/<id>/transcript.json``."""
    workspace.ensure_project_dir(project_id)
    workspace.transcript_path(project_id).write_text(json.dumps(transcript.to_dict()))


def load_transcript(workspace: Workspace, project_id: str) -> Transcript | None:
    """Load a previously saved transcript, or ``None`` if absent."""
    path = workspace.transcript_path(project_id)
    if not path.exists():
        return None
    return Transcript.from_dict(json.loads(path.read_text()))


def transcribe_project(
    project_id: str,
    audio_path: str,
    workspace: Workspace | None = None,
    **kwargs: object,
) -> Transcript:
    """Transcribe a project's audio and persist the transcript (orchestrator, I/O)."""
    workspace = workspace or Workspace.default()
    transcript = transcribe(audio_path, **kwargs)  # type: ignore[arg-type]
    save_transcript(workspace, project_id, transcript)
    return transcript
