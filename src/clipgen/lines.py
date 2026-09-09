"""Numbered-line transcript model — the contract the Gemini judge reads and writes against.

The judge sees the *whole* transcript as numbered sentences (``line | mm:ss | speaker | text``)
and references moments by integer line ranges, never timestamps — a small integer is hard to
corrupt and trivial to validate. This module turns the canonical word list into those lines,
formats them for the prompt, and resolves a line range back into seconds from the same words.

All functions are pure and unit-tested; there is no ML/ASR or network dependency here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clipgen.transcript import Transcript

_SENTENCE_END = ".?!"
# Fallback line caps for sparse-punctuation transcripts (e.g. the faster-whisper `base`
# model) so a "line" never balloons into a paragraph the judge can't address precisely.
DEFAULT_MAX_LINE_S = 12.0
DEFAULT_MAX_WORDS = 40


@dataclass
class Line:
    """One numbered line — a sentence (or a capped span when punctuation is sparse)."""

    index: int  # 1-based, sequential
    start_s: float
    end_s: float
    speaker: str
    text: str
    # False when the line was flushed by the duration/word cap or a speaker change without
    # terminal punctuation — i.e. it ends mid-sentence. Grounding uses this to refuse clip
    # boundaries inside a run-on (see clips.ground).
    complete: bool = True


def roleify_speakers(transcript: Transcript) -> dict[str | None, str]:
    """Map raw diarization labels to neutral, readable names for the judge.

    Distinct speakers become ``Person A``, ``Person B``, … in first-appearance order —
    deliberately role-neutral: diarization only tells voices apart, it does not know who is
    host vs guest, so we don't guess. With no diarization there is a single (or ``None``)
    speaker, which becomes ``Person A``.
    """
    names = [f"Person {chr(ord('A') + i)}" for i in range(26)]  # Person A … Person Z
    mapping: dict[str | None, str] = {}
    for spk in transcript.speakers:
        n = len(mapping)
        mapping[spk] = names[n] if n < len(names) else f"Person {n + 1}"
    if not mapping:  # no speaker labels at all
        mapping[None] = "Person A"
    return mapping


def build_lines(
    transcript: Transcript,
    *,
    speaker_map: dict[str | None, str] | None = None,
    max_line_s: float = DEFAULT_MAX_LINE_S,
    max_words: int = DEFAULT_MAX_WORDS,
) -> list[Line]:
    """Group the canonical word list into numbered lines.

    A line breaks on the first of: terminal punctuation (``.?!`` → one sentence per line, the
    normal case), a speaker change, or the duration/word-count caps (the fallback that keeps
    unpunctuated transcripts addressable). ``speaker_map`` (see :func:`roleify_speakers`)
    relabels speakers; when omitted, raw labels are used.
    """
    words = transcript.words
    if not words:
        return []

    def role(spk: str | None) -> str:
        if speaker_map is not None:
            return speaker_map.get(spk, speaker_map.get(None, ""))
        return spk or ""

    lines: list[Line] = []
    buf = [words[0]]
    for w in words[1:]:
        prev = buf[-1]
        span = w.end_s - buf[0].start_s
        ends_sentence = bool(prev.text) and prev.text[-1] in _SENTENCE_END
        speaker_changed = w.speaker != buf[0].speaker
        too_long = span > max_line_s or len(buf) >= max_words
        if ends_sentence or speaker_changed or too_long:
            lines.append(_flush(buf, len(lines) + 1, role))
            buf = [w]
        else:
            buf.append(w)
    lines.append(_flush(buf, len(lines) + 1, role))
    return lines


def _flush(buf: list, index: int, role) -> Line:  # noqa: ANN001 - internal closure helper
    text = " ".join(w.text for w in buf).strip()
    return Line(
        index=index,
        start_s=buf[0].start_s,
        end_s=buf[-1].end_s,
        speaker=role(buf[0].speaker),
        text=text,
        complete=bool(text) and text[-1] in _SENTENCE_END,
    )


def _mmss(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


def format_transcript(lines: list[Line], chapters: list[dict] | None = None) -> str:
    """Render lines for the judge prompt: a schema header + ``line | mm:ss | speaker | text``.

    When ``chapters`` (``[{start_s, title}, …]``, from the source video) is given, a
    ``### CHAPTER`` marker is inserted before the first transcript line at or after each chapter's
    start time. The markers carry no line number, so they never disturb the integer line-range
    contract; they only hint to the judge where the video's own topic boundaries fall, so it can
    keep each clip inside a single coherent segment instead of straddling two.
    """
    header = (
        "Lines are numbered sentences. Columns: line | mm:ss | speaker | text.\n"
        "Reference moments by start_line and end_line.\n"
    )
    if chapters:
        header += (
            "Lines beginning with '### CHAPTER' mark the video's own chapter boundaries "
            "(not transcript lines) — prefer clips that stay within a single chapter.\n"
        )

    pending = sorted(
        ({"start_s": float(c["start_s"]), "title": c.get("title", "")} for c in chapters or []),
        key=lambda c: c["start_s"],
    )
    rows: list[str] = []
    for ln in lines:
        while pending and pending[0]["start_s"] <= ln.start_s:
            ch = pending.pop(0)
            title = ch["title"] or "(untitled)"
            rows.append(f"### CHAPTER ({_mmss(ch['start_s'])}): {title}")
        rows.append(f"{ln.index} | {_mmss(ln.start_s)} | {ln.speaker} | {ln.text}")
    return header + "\n" + "\n".join(rows)


def select_chapter_lines(
    lines: list[Line],
    chapters: list[dict],
    selected: list[int] | None,
) -> tuple[list[Line], list[dict]]:
    """Restrict the transcript to the operator-selected chapters, re-numbered 1..M.

    ``selected`` holds 0-based indices into ``chapters`` **sorted by start time** — the order the
    UI shows them in. ``None``, an empty/invalid selection, or a selection covering every chapter
    all return the inputs unchanged (send the whole transcript). Each line is assigned to the last
    chapter whose start time it is at or after (the first chapter therefore absorbs any pre-chapter
    intro); lines in unselected chapters are dropped and the survivors are re-indexed so the
    ``ln.index`` == list-position contract that :func:`resolve_lines` / grounding rely on still
    holds. The returned chapter list is likewise pruned to the selection, for the prompt markers.
    """
    if not chapters or selected is None:
        return lines, chapters
    chs = sorted(chapters, key=lambda c: float(c["start_s"]))
    keep = {i for i in selected if 0 <= i < len(chs)}
    if not keep or keep >= set(range(len(chs))):
        return lines, chapters  # nothing valid, or everything — no filtering needed
    starts = [float(c["start_s"]) for c in chs]
    kept: list[Line] = []
    for ln in lines:
        idx = 0
        for i, start in enumerate(starts):
            if ln.start_s >= start:
                idx = i
            else:
                break
        if idx in keep:
            kept.append(ln)
    filtered = [replace(ln, index=n + 1) for n, ln in enumerate(kept)]
    return filtered, [chs[i] for i in sorted(keep)]


def resolve_lines(lines: list[Line], start_line: int, end_line: int) -> tuple[float, float]:
    """Resolve an (inclusive) line range to ``(start_s, end_s)`` from the line bounds.

    Out-of-range indices are clamped into the available lines and reordered if inverted, so a
    slightly-off range from the judge still yields a sane span rather than raising.
    """
    if not lines:
        raise ValueError("no lines to resolve against")
    n = len(lines)
    lo, hi = sorted((start_line, end_line))
    lo = max(1, min(lo, n))
    hi = max(1, min(hi, n))
    return lines[lo - 1].start_s, lines[hi - 1].end_s
