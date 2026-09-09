"""Grounding — turn Gemini's judged line-ranges into a ranked list of concrete clips.

The judge returns moments as line ranges + per-dimension scores (never timestamps). This module
resolves each range to real seconds, verifies the judge's claims locally (the hook line opens the
clip, the payoff line is inside it), enforces the hard length contract in sentence-line units,
gates on hook+payoff+virality, suppresses overlaps, ranks by a content score, and emits a plain
list of :class:`ClipItem` persisted as ``clips.json``. Candidates that fail any check are
**dropped, never patched into fragments** — a run with 4 clean clips beats one with 10 broken
ones; the drops (with reasons) go to the caller's ``rejects`` list for ``judge_rejects.json``.

A clip may be **compound**: 2-3 segments the judge spliced together to skip filler between a hook
and a later payoff (see ``judge.JUDGE_PROMPT_TEMPLATE``). A simple clip is just the single-segment
case. Numbered lines *usually* break on sentence punctuation, but sparse-punctuation ASR regions
break on duration/word caps instead (``Line.complete`` is False there) — so boundaries are snapped
outward until both edges sit on complete thoughts rather than trusting line numbers blindly. All
logic here is pure and unit-tested with synthetic ``JudgedClip`` / ``Line`` inputs.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING

from clipgen.judge import DIMENSIONS, MAX_CLIP_S, MIN_CLIP_S
from clipgen.lines import resolve_lines

if TYPE_CHECKING:
    from pathlib import Path

    from clipgen.judge import JudgedClip
    from clipgen.lines import Line
    from clipgen.transcript import Word

# Content score = weighted mean of Gemini's per-dimension scores. hook, payoff, and virality
# co-lead — a clip must stop the scroll, deliver, AND be worth sharing; the rest modulate.
CONTENT_WEIGHTS: dict[str, float] = {
    "hook": 1.3,
    "payoff": 1.3,
    "virality": 1.3,
    "emotion": 1.0,
    "quotability": 0.9,
    "controversy": 0.5,
    "standalone": 0.6,
}

# A clip must clear this on hook AND payoff AND virality (0-10) to qualify. The weighted mean
# alone would let a 10-hook/2-payoff clip through; the triple floor won't. 7.0 is meaningful
# because the judge prompt calibrates scores (median ≈ 6, 9+ rare — see judge.SCORING_RUBRIC).
GATE_FLOOR = 7.0
_GATE_DIMS = ("hook", "payoff", "virality")

# Sentence-completion snapping may run a clip this far past max_len_s before it is dropped
# instead — finishing a thought beats an exact ceiling, but not by much.
MAX_OVERRUN_S = 10.0


@dataclass
class ClipItem:
    """One grounded clip: real bounds, the judge's copy, and its scores.

    ``segments`` are the clip's spoken spans in seconds — one for a simple clip, several for a
    compound clip (the gaps between them are the spliced-out filler). ``start_s``/``end_s`` are the
    overall span (first segment start → last segment end); ``duration_s`` is the *spoken* time
    (sum of segment lengths), which is what the length contract bounds.
    """

    start_s: float
    end_s: float
    duration_s: float
    hook: str = ""
    hook_sentence: str = ""
    payoff_sentence: str = ""
    payoff: str = ""
    share_trigger: str = ""
    reason: str = ""
    text: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    content_score: float = 0.0
    clip_type: str = "simple"
    segments: list[list[float]] = field(default_factory=list)
    splice_reason: str = ""
    source: str = "gemini"  # "gemini" (the judge run) | "imported" (pasted from an external LLM)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ClipItem:
        segments = [[float(s), float(e)] for s, e in d.get("segments", [])]
        return cls(
            start_s=float(d["start_s"]),
            end_s=float(d["end_s"]),
            duration_s=float(d.get("duration_s") or (d["end_s"] - d["start_s"])),
            hook=d.get("hook", ""),
            hook_sentence=d.get("hook_sentence", ""),
            payoff_sentence=d.get("payoff_sentence", ""),
            payoff=d.get("payoff", ""),
            share_trigger=d.get("share_trigger", ""),
            reason=d.get("reason", ""),
            text=d.get("text", ""),
            scores={dim: float(d.get("scores", {}).get(dim, 0.0)) for dim in DIMENSIONS},
            content_score=float(d.get("content_score") or 0.0),
            clip_type=d.get("clip_type", "simple"),
            segments=segments or [[float(d["start_s"]), float(d["end_s"])]],
            splice_reason=d.get("splice_reason", ""),
            source=d.get("source", "gemini"),
        )


def content_score(scores: dict[str, float]) -> float:
    """Weighted mean of the per-dimension scores (0-10)."""
    total = sum(CONTENT_WEIGHTS.values())
    return sum(w * scores.get(dim, 0.0) for dim, w in CONTENT_WEIGHTS.items()) / total


def _passes_gate(scores: dict[str, float]) -> bool:
    """A clip qualifies only if hook AND payoff AND virality all clear the floor."""
    return min(scores.get(dim, 0.0) for dim in _GATE_DIMS) >= GATE_FLOOR


_NORM_RE = re.compile(r"[^a-z0-9 ]+")


def _norm(text: str) -> str:
    """Normalize a sentence for fuzzy matching: lowercase, alphanumerics + single spaces."""
    return " ".join(_NORM_RE.sub(" ", text.lower().replace("…", " ")).split())


def _match_line(
    lines: list[Line], lo: int, hi: int, sentence: str, *, prefer_last: bool = False
) -> int | None:
    """Find the line in ``[lo, hi]`` best matching a judge-quoted sentence; ``None`` if nothing
    plausibly matches.

    The judge quotes lines back imperfectly (ellipses, ASR drift, partial quotes), so matching is
    containment either way, then word-overlap ≥ 0.6 as a fallback. Ties keep the first hit for a
    hook (earliest open) or the last for a payoff (latest landing).
    """
    target = _norm(sentence)
    if len(target) < 8:
        return None
    t_words = set(target.split())
    best, best_score = None, 0.0
    for i in range(max(1, lo), min(len(lines), hi) + 1):
        lt = _norm(lines[i - 1].text)
        if not lt:
            continue
        if target in lt or (len(lt) >= 8 and lt in target):
            score = 1.0
        elif t_words:
            score = len(t_words & set(lt.split())) / len(t_words)
        else:
            continue
        if score > best_score or (prefer_last and score == best_score and best is not None):
            best, best_score = i, score
    return best if best_score >= 0.6 else None


def _starts_thought(lines: list[Line], a: int) -> bool:
    """Line ``a`` begins a thought: it's the first line, follows a completed sentence, or a
    speaker change (an interruption starts the new speaker's own utterance)."""
    prev = lines[a - 2] if a > 1 else None
    return prev is None or prev.complete or prev.speaker != lines[a - 1].speaker


def _ends_thought(lines: list[Line], b: int) -> bool:
    """Line ``b`` finishes a thought: terminal punctuation, the last line, or the next line is a
    different speaker (the thought was yielded, not cap-split)."""
    nxt = lines[b] if b < len(lines) else None
    return lines[b - 1].complete or nxt is None or nxt.speaker != lines[b - 1].speaker


def _snap_out(lines: list[Line], a: int, b: int) -> tuple[int, int]:
    """Extend ``[a, b]`` outward until both edges sit on complete thoughts.

    Cap-split run-ons (``Line.complete`` False) mean a line number is not a safe boundary; a clip
    must never open or close mid-sentence, so the edge walks out of the run-on it landed in.
    """
    while a > 1 and not _starts_thought(lines, a):
        a -= 1
    while b < len(lines) and not _ends_thought(lines, b):
        b += 1
    return a, b


def _normalize_segments(
    lines: list[Line], segments: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Clamp to valid indices, order, snap each segment to complete thoughts, merge overlaps."""
    n = len(lines)
    norm: list[tuple[int, int]] = []
    for start, end in segments:
        a = max(1, min(start, n))
        b = max(1, min(end, n))
        if a > b:
            a, b = b, a
        norm.append(_snap_out(lines, a, b))
    norm.sort()
    merged: list[tuple[int, int]] = []
    for a, b in norm:
        if merged and a <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))  # snapping joined the splice
        else:
            merged.append((a, b))
    return merged


def _enforce_segments(
    lines: list[Line],
    segs: list[tuple[int, int]],
    min_s: float,
    max_s: float,
    payoff_line: int | None = None,
) -> tuple[list[tuple[int, int]] | None, str]:
    """Fit normalized segments into ``[min_s, max_s]`` of spoken time on complete-thought
    boundaries; ``(None, reason)`` when that's impossible.

    Over-long clips trim whole lines from the tail (the hook lives at the head), but only onto
    ends that finish a thought and **never past the payoff line** — shipping a clip that cut its
    own payoff is exactly the slop this exists to prevent. A bounded overrun (``MAX_OVERRUN_S``)
    is tolerated to finish a sentence. Too-short clips extend outward to the nearest complete
    thoughts; if the minimum still can't be met, the clip is dropped, not shipped short.
    """
    n = len(lines)
    if n == 0 or not segs:
        return None, "empty range"

    def dur(a: int, b: int) -> float:
        return lines[b - 1].end_s - lines[a - 1].start_s

    def total() -> float:
        return sum(dur(a, b) for a, b in segs)

    # Over-long: walk the tail back to the previous complete thought, repeatedly.
    while total() > max_s:
        a, b = segs[-1]
        has_payoff = payoff_line is not None and a <= payoff_line <= b
        floor = max(a, payoff_line) if has_payoff else a
        nb = b - 1
        while nb >= floor and not _ends_thought(lines, nb):
            nb -= 1
        if floor <= nb < b:
            segs[-1] = (a, nb)
        elif len(segs) > 1 and not has_payoff:
            segs.pop()
        else:
            break  # can't trim further without cutting the payoff / the last thought
    if total() > max_s + MAX_OVERRUN_S:
        return None, f"cannot fit {max_s:.0f}s on complete thoughts without cutting the payoff"

    # Too-short: extend the last segment forward, then the first backward, whole thoughts at a
    # time (a cap-split run-on is swallowed in one step so the edge stays clean).
    while total() < min_s and segs[-1][1] < n:
        a, b = segs[-1]
        nb = b + 1
        while nb < n and not _ends_thought(lines, nb):
            nb += 1
        segs[-1] = (a, nb)
    while total() < min_s and segs[0][0] > 1:
        a, b = segs[0]
        na = a - 1
        while na > 1 and not _starts_thought(lines, na):
            na -= 1
        segs[0] = (na, b)
    if total() < min_s:
        return None, f"cannot reach {min_s:.0f}s of spoken time"
    return segs, ""


# Trailing reaction noises that should never close a clip — the payoff line should.
_FILLER_WORDS = frozenset(
    "yeah right okay ok yes no exactly sure wow totally absolutely true mhm hmm mm "
    "uh huh oh so and well i know thats that's".split()
)


def _is_filler(text: str) -> bool:
    words = _norm(text).split()
    return bool(words) and len(words) <= 2 and all(w in _FILLER_WORDS for w in words)


def _trim_filler_tail(
    lines: list[Line],
    segs: list[tuple[int, int]],
    min_s: float,
    payoff_line: int | None,
) -> list[tuple[int, int]]:
    """Drop trailing filler lines ("Yeah.", "Right.") so the payoff closes the clip."""
    a, b = segs[-1]
    while (
        b > a
        and _is_filler(lines[b - 1].text)
        and (payoff_line is None or b - 1 >= payoff_line)
        and _ends_thought(lines, b - 1)
    ):
        candidate = [*segs[:-1], (a, b - 1)]
        if sum(lines[e - 1].end_s - lines[s - 1].start_s for s, e in candidate) < min_s:
            break
        b -= 1
        segs = candidate
    return segs


def ground(
    judged: list[JudgedClip],
    lines: list[Line],
    *,
    top_k: int = 10,
    min_len_s: float = MIN_CLIP_S,
    max_len_s: float = MAX_CLIP_S,
    apply_gate: bool = True,
    source: str = "gemini",
    rejects: list[dict] | None = None,
) -> list[ClipItem]:
    """Resolve judged line-ranges into ranked, length-valid, non-overlapping clips.

    Per candidate, in order: the triple score gate (hook+payoff+virality — skipped with
    ``apply_gate`` False, e.g. clips imported from an external LLM that the operator is adding
    deliberately); re-anchor the start onto the judge's own ``hook_sentence`` so no preamble
    precedes the hook; locate ``payoff_sentence`` inside the range (a judge-sourced clip whose
    payoff can't be found is clickbait — dropped); snap/fit boundaries onto complete thoughts
    within the length contract; trim trailing filler. Highest content first, so the stronger clip
    wins any overlap cluster. Never pads: returns at most ``top_k`` survivors, ranked by content
    score. Every drop is appended to ``rejects`` (when given) with a one-line reason.
    """
    if not lines:
        return []
    ordered = sorted(judged, key=lambda j: content_score(j.scores), reverse=True)

    def reject(j: JudgedClip, reason: str) -> None:
        if rejects is not None:
            rejects.append(
                {
                    "hook": j.hook,
                    "lines": f"{j.start_line}-{j.end_line}",
                    "stage": "ground",
                    "reason": reason,
                }
            )

    items: list[ClipItem] = []
    for j in ordered:
        if apply_gate and not _passes_gate(j.scores):
            weak = ", ".join(f"{d}={j.scores.get(d, 0.0):.0f}" for d in _GATE_DIMS)
            reject(j, f"below the {GATE_FLOOR:.0f}-point gate on hook/payoff/virality ({weak})")
            continue
        segs = _normalize_segments(lines, [(s.start_line, s.end_line) for s in j.segments])
        if not segs:
            reject(j, "empty line range")
            continue

        # Open ON the hook: if the judge's quoted hook line sits later in the range, the lines
        # before it are preamble — snap the start forward onto it.
        if j.hook_sentence:
            a, b = segs[0]
            hit = _match_line(lines, a, b, j.hook_sentence)
            if hit is not None and hit > a:
                start, _ = _snap_out(lines, hit, hit)
                if start > a:
                    segs[0] = (start, b)

        # The payoff must actually be inside the clip — otherwise the hook is a promise the clip
        # never keeps. Imported clips (apply_gate False) don't carry the field; skip them.
        payoff_line: int | None = None
        if j.payoff_sentence:
            for a, b in segs:
                hit = _match_line(lines, a, b, j.payoff_sentence, prefer_last=True)
                if hit is not None:
                    payoff_line = hit
        if payoff_line is None and apply_gate:
            reject(j, "payoff_sentence not found inside the clip range (clickbait guard)")
            continue

        enforced, why = _enforce_segments(lines, segs, min_len_s, max_len_s, payoff_line)
        if enforced is None:
            reject(j, why)
            continue
        enforced = _trim_filler_tail(lines, enforced, min_len_s, payoff_line)

        resolved = [resolve_lines(lines, a, b) for a, b in enforced]
        start_s, end_s = resolved[0][0], resolved[-1][1]
        if any(start_s < it.end_s and end_s > it.start_s for it in items):
            reject(j, "overlaps a higher-scored clip")
            continue
        text = " ".join(
            lines[i].text for a, b in enforced for i in range(a - 1, b)
        ).strip()
        spoken = sum(e - s for s, e in resolved)
        items.append(
            ClipItem(
                start_s=start_s,
                end_s=end_s,
                duration_s=round(spoken, 3),
                hook=j.hook,
                hook_sentence=j.hook_sentence,
                payoff_sentence=j.payoff_sentence,
                payoff=j.payoff,
                share_trigger=j.share_trigger,
                reason=j.reason,
                text=text,
                scores=dict(j.scores),
                content_score=round(content_score(j.scores), 3),
                clip_type="compound" if len(enforced) > 1 else "simple",
                segments=[[round(s, 3), round(e, 3)] for s, e in resolved],
                splice_reason=j.splice_reason if len(enforced) > 1 else "",
                source=source,
            )
        )
    items.sort(key=lambda it: it.content_score, reverse=True)
    for extra in items[top_k:]:
        if rejects is not None:
            rejects.append(
                {"hook": extra.hook, "stage": "ground", "reason": f"ranked below top {top_k}"}
            )
    return items[:top_k]


def merge_clips(existing: list[ClipItem], extra: list[ClipItem]) -> list[ClipItem]:
    """Add imported clips to the existing list — additive, never destructive.

    The two lists are pooled and re-ranked by content score. Nothing is dropped for merely
    overlapping (different LLMs producing distinct takes of the same moment are all kept, so they
    can be compared); only an **exact** duplicate span is collapsed, so re-importing the same clip
    doesn't stack copies. Order-independent and pure.
    """
    kept: list[ClipItem] = []
    seen: set[tuple[float, float]] = set()
    for it in sorted([*existing, *extra], key=lambda c: c.content_score, reverse=True):
        key = (round(it.start_s, 2), round(it.end_s, 2))
        if key in seen:
            continue  # exact same span already present — keep the higher-scored one only
        seen.add(key)
        kept.append(it)
    return kept


_EPS = 1e-6


def _new_boundary(units: list, seg_start: float, seg_end: float, edge: str, direction: str,
                  count: int) -> float:
    """Compute a segment edge's new time by moving it `count` units (words or sentences).

    ``units`` are things with ``start_s``/``end_s`` (the transcript's words, or its sentence
    lines). Expanding the end reaches forward to the ``count``-th unit after it; trimming pulls it
    back over the last ``count`` units inside the segment (never below one). Start is symmetric.
    Returns the current boundary unchanged when there's nothing to move into.
    """
    if edge == "end":
        if direction == "expand":
            after = [u for u in units if u.end_s > seg_end + _EPS]
            return after[min(count, len(after)) - 1].end_s if after else seg_end
        inside = [u for u in units if u.start_s >= seg_start - _EPS and u.end_s <= seg_end + _EPS]
        if len(inside) <= 1:
            return seg_end  # keep at least one unit
        return inside[max(1, len(inside) - count) - 1].end_s
    # edge == "start"
    if direction == "expand":
        before = [u for u in units if u.start_s < seg_start - _EPS]
        return before[max(0, len(before) - count)].start_s if before else seg_start
    inside = [u for u in units if u.start_s >= seg_start - _EPS and u.end_s <= seg_end + _EPS]
    if len(inside) <= 1:
        return seg_start
    return inside[min(count, len(inside) - 1)].start_s


def _rebuild_after_edit(clip: ClipItem, segs: list[list[float]], words: list[Word]) -> ClipItem:
    """Re-derive a clip's bounds/duration/text from edited segments (scores/type unchanged)."""
    clean = [(round(s, 3), round(e, 3)) for s, e in segs if e - s > _EPS]
    if not clean:
        return clip  # refuse to collapse a clip to nothing
    parts = [
        " ".join(w.text for w in words if w.end_s > s + _EPS and w.start_s < e - _EPS)
        for s, e in clean
    ]
    return replace(
        clip,
        start_s=clean[0][0],
        end_s=clean[-1][1],
        duration_s=round(sum(e - s for s, e in clean), 3),
        segments=[[s, e] for s, e in clean],
        text=" ".join(p for p in parts if p).strip(),
    )


def adjust_clip(
    clip: ClipItem, words: list[Word], *, edge: str, unit: str, direction: str, count: int = 1
) -> ClipItem:
    """Expand or trim a clip's start/end by a word or a sentence, snapped to the transcript.

    ``edge`` is ``"start"``/``"end"``, ``unit`` is ``"word"``/``"sentence"``, ``direction`` is
    ``"expand"``/``"trim"``. Only the outer edge moves (the last segment's end, or the first
    segment's start), so compound clips keep their splice. Manual edits are not length-gated — the
    operator decides. Returns a new ``ClipItem`` (unchanged if there's nothing to move into).
    """
    if edge not in ("start", "end") or unit not in ("word", "sentence"):
        raise ValueError("edge must be start/end and unit must be word/sentence")
    if direction not in ("expand", "trim"):
        raise ValueError("direction must be expand/trim")
    ordered = sorted(words, key=lambda w: (w.start_s, w.end_s))
    if not ordered:
        return clip

    if unit == "sentence":
        from clipgen.lines import build_lines
        from clipgen.transcript import Transcript

        units: list = build_lines(Transcript(words=list(ordered)))
    else:
        units = ordered

    segs = [list(s) for s in (clip.segments or [[clip.start_s, clip.end_s]])]
    count = max(1, int(count))
    if edge == "end":
        segs[-1][1] = _new_boundary(units, segs[-1][0], segs[-1][1], "end", direction, count)
    else:
        segs[0][0] = _new_boundary(units, segs[0][0], segs[0][1], "start", direction, count)
    return _rebuild_after_edit(clip, segs, ordered)


def save_clips(path: Path, items: list[ClipItem]) -> None:
    """Persist grounded clips to ``clips.json`` (a ``{"clips": [...]}`` envelope)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"clips": [it.to_dict() for it in items]}
    path.write_text(json.dumps(payload, indent=2))


def load_clips(path: Path) -> list[ClipItem]:
    """Load grounded clips from ``clips.json``; ``[]`` if the file is absent or malformed."""
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError):
        return []
    if not isinstance(data, dict):
        return []
    return [ClipItem.from_dict(d) for d in data.get("clips", []) if isinstance(d, dict)]
