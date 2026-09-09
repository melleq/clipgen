"""The editorial judge — Gemini selects clip-worthy moments by line range, in two passes.

Gemini's 1M context holds an entire episode, so there is no windowing or synopsis: the whole
numbered transcript goes in one call (**pass A**) and the model returns clip moments as **line
ranges + per-dimension scores** — never raw timestamps. Each candidate must name its payoff line
and a concrete share trigger, so weak moments can be checked rather than trusted. A second call
(**pass B**, :func:`refine_clips`) re-judges each surviving candidate against its actual transcript
excerpt: it keeps/drops, tightens boundaries, and re-scores — this is where clickbait (hook with no
payoff in range) and merely-informative moments die. The local layer (``lines.resolve_lines`` +
``clips.ground``) places the actual cut points.

A cheap **content-type + density** pre-pass (one tiny call) conditions the main prompt, and the
prompt repeats hard length bounds + a FORBIDDEN block (the failure mode that produced 3-7s clips).
Prompt structure and criteria are adapted from the OSS SamurAIGPT / "ViralVadoo" and OpenShorts
projects. Formatting and parsing are pure and unit-tested; only ``_call_gemini`` touches the
network. The API key is read from ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) at call time.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from clipgen.lines import format_transcript

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from clipgen.lines import Line

    ProgressCb = Callable[[str, float | None], None]

NAME = "gemini"
# Free-tier Gemini (June 2026): Pro is paid-only; Flash + Flash-Lite are free. The judge does the
# hard subjective call over a whole episode, so it gets the strongest free model; the content-type
# pre-pass is trivial classification, so it stays on the cheaper/higher-RPM Flash-Lite. Both track
# the newest release via the "-latest" aliases and are overridable via env.
DEFAULT_JUDGE_MODEL = "gemini-flash-latest"  # currently Gemini 3.5 Flash
DEFAULT_CLASSIFY_MODEL = "gemini-flash-lite-latest"  # currently Gemini 3.1 Flash-Lite
DEFAULT_MODEL = DEFAULT_JUDGE_MODEL  # back-compat alias


def _judge_model() -> str:
    return os.environ.get("GEMINI_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL


def _classify_model() -> str:
    return os.environ.get("GEMINI_CLASSIFY_MODEL") or DEFAULT_CLASSIFY_MODEL

# Hard clip-length contract (also enforced in clips.ground, so the model can't slip past it) and
# how many candidates to over-generate before the refine pass + grounding trim to top_k. A clip
# needs room for its full arc — hook, connective tissue, payoff — so 20-90s of spoken time with a
# 30-60s sweet spot; the old 35s ceiling chopped payoffs mid-sentence.
MIN_CLIP_S = 20.0
MAX_CLIP_S = 90.0
DEFAULT_TARGET_CLIPS = 20

# The scoring axes Gemini returns per clip (0-10). Order is the display/column order too.
# hook = does the open stop the scroll; payoff = is there a real reward (insight/punchline/
# revelation/emotional beat); virality = would a nameable audience actually share/argue/repost.
# A clip must be strong on ALL THREE — see clips.GATE_FLOOR.
DIMENSIONS = ("hook", "payoff", "virality", "emotion", "quotability", "controversy", "standalone")

VIRALITY_CRITERIA = (
    "You are hunting for clips that TRAVEL — that get shared, argued about, and sent to a "
    "friend — not just clips that are 'good' or 'informative'. An interesting moment that "
    "nobody would stop scrolling for or forward to someone is NOT a clip.\n\n"
    "A clip that travels needs THREE things, and it needs ALL THREE:\n"
    "A) A HOOK — the opening line creates instant curiosity, stakes, or tension in the first ~3 "
    'seconds ("The secret is…", "Nobody talks about…", "I was completely wrong about…", '
    '"Here\'s what actually happened…"). It earns the next few seconds of attention.\n'
    "B) A PAYOFF — the clip actually DELIVERS on the hook: a real insight, punchline, revelation, "
    "confession, or emotional peak lands before it ends. The viewer leaves with something.\n"
    "C) A SHARE TRIGGER — a nameable audience has a concrete reason to REACT: argue in the "
    "comments, tag a friend, quote it, repost it. If you cannot name who shares it and why, it "
    "is not a clip.\n"
    "A grabby hook that goes nowhere is clickbait, not a clip. A great payoff nobody sticks around "
    "for is buried. A complete, well-said point nobody would send to anyone is filler. You want "
    "the moments that have all three.\n\n"
    "What makes a strong PAYOFF (and strong content generally):\n"
    "1. REVELATION — a surprising fact, stat, or confession that reframes how the viewer thinks.\n"
    "2. EMOTIONAL PEAK — genuine surprise, laughter, anger, vulnerability, excitement; a raw "
    "unscripted reaction that makes a viewer FEEL something.\n"
    "3. OPINION BOMB — a strong, polarizing, or counter-intuitive claim people want to agree "
    "with, argue with, or quote.\n"
    "4. STORY PEAK — the climax or twist of an anecdote.\n"
    "5. CONCRETE VALUE — a specific tip, hack, or insight the viewer can immediately apply.\n"
    "6. QUOTABLE LINE — a sentence that works as a standalone quote card.\n\n"
    "NOT VIRAL — do not return these, no matter how complete or well-said:\n"
    "- Generic advice ('work hard', 'be consistent', 'surround yourself with good people').\n"
    "- History lessons, trivia, or product/tech explainers with no stakes, conflict, or surprise.\n"
    "- Smart-sounding points a viewer nods along to but would never send to anyone.\n"
    "- Promotional talk, polite agreement, book-report summaries of well-known ideas.\n"
    "VIRAL raw material looks like: conflict and confrontation, confessions, hot takes someone "
    "will angrily disagree with, absurd stakes, specific shocking numbers, raw emotion caught "
    "live, status flexes, forbidden or counter-intuitive claims.\n\n"
    "THE SHARE TEST: for every clip, ask — would someone stop mid-scroll for the open AND feel it "
    "was worth it by the end? Would they send it to a friend, argue in the comments, or repost it? "
    "If the hook, the payoff, or the share trigger is weak, drop it.\n"
)

SCORING_RUBRIC = (
    "SCORING CALIBRATION (0-10 per axis — score like a harsh editor, not a fan):\n"
    "- 9-10: exceptional, you would bet the channel on it. RARE: at most 2 clips in your entire "
    "answer may carry ANY score of 9 or above.\n"
    "- 7-8: genuinely strong — clearly above the episode's average moment.\n"
    "- 5-6: decent but ordinary; most watchable moments land here. Your MEDIAN candidate should "
    "score about 6.\n"
    "- 0-4: weak on this axis.\n"
    "Axes: hook (does the FIRST line stop the scroll), payoff (does the promised reward land "
    "INSIDE the clip), virality (would the named audience actually share/argue/repost it), "
    "emotion, quotability, controversy, standalone (a stranger with zero context fully gets it — "
    "10 only if truly nothing is missing).\n"
)

CONTENT_TYPE_PROMPT = (
    "Classify this video transcript sample.\n"
    "content_type: one of podcast, interview, tutorial, lecture, commentary, debate, vlog, other.\n"
    "density: low (mostly filler/chit-chat), medium, or high (dense info/stories).\n"
    'Respond ONLY with JSON: {"content_type": "…", "density": "…"}.\n\nSAMPLE:\n'
)

JUDGE_PROMPT_TEMPLATE = (
    "You are an elite short-form video editor who has studied thousands of viral clips on "
    "TikTok, Instagram Reels, and YouTube Shorts. You live and die by one metric: does a "
    "clip stop the scroll in the first 3 seconds and make people share it? You are NOT "
    "cataloguing the best parts of the episode — you are hunting for the handful of moments "
    "that could actually blow up.\n\n"
    "Content type: {content_type} | Density: {density}\n\n"
    "{criteria}\n"
    "{rubric}\n"
    "Your task: comb the FULL numbered transcript below for up to {n} moments with genuine "
    "VIRAL POTENTIAL and score each one. A verification pass and a human review follow, so "
    "surface every moment that truly qualifies — but NEVER pad: returning 6 real moments beats "
    "20 mediocre ones. Order best-to-worst by viral potential.\n\n"
    "RULES:\n"
    "- FULL ARC: the line range must OPEN on the hook line itself — the very first line of the "
    "clip is the scroll-stopper, with NO warm-up lines before it — and it must END only after "
    "the payoff has landed, never before. Include the connective tissue the payoff needs; cut "
    "preamble before the hook and trailing chatter after the payoff. End 1-2 lines after the "
    "payoff line, not 10.\n"
    "- HOOK **AND** PAYOFF **AND** SHARE TRIGGER — all three are required. A strong hook with no "
    "payoff is clickbait. A strong payoff behind a slow open is buried. A complete point without "
    "a share trigger is filler. Reject anything that misses one.\n"
    "- For every clip you MUST name: the exact line where the payoff lands (payoff_sentence, "
    "verbatim from the transcript, inside your range), what the viewer concretely gets (payoff), "
    "and who shares it and why (share_trigger). If you cannot fill one of these honestly, the "
    "moment is not a clip — skip it.\n"
    "- OMIT THE INTRO — if the episode opens with an intro section (channel/show intro, host "
    "welcome or housekeeping, 'what's coming up' teaser, cold-open sizzle), skip it entirely and "
    "pull NO clips from it; start hunting once the real content begins. The only exception is if "
    "the intro itself lands a genuine standalone hook AND payoff.\n"
    "- Every clip must stand alone — understandable with zero prior context; no dangling "
    'references ("that", "he", "as I said").\n'
    "- Never start or end mid-sentence or mid-thought. CAUTION: the transcript is auto-generated "
    "and some regions are unpunctuated run-ons where a numbered line itself begins or ends "
    "mid-sentence — check that your start line's text actually BEGINS a thought and your end "
    "line's text actually FINISHES one; extend to the nearest line that does.\n"
    "- Reference each moment by line ranges (start_line, end_line). Do NOT emit timestamps.\n"
    "- DURATION: each clip MUST total between {min_s:.0f} and {max_s:.0f} seconds of spoken "
    "time. Sweet spot 30-60s; run past 60s only when a story arc genuinely needs the room. Use "
    "the mm:ss column to judge length.\n"
    "- COMPOUND CLIPS (optional): if the hook and its payoff are separated by filler, you MAY "
    'return the clip as 2-3 "segments" (line ranges) that skip the filler between them — e.g. the '
    "setup line, then the payoff that lands a bit later. All segments MUST stay within the SAME "
    "chapter (see '### CHAPTER' markers), be whole sentences, keep their transcript order, and "
    "the COMBINED spoken time must still fit the bounds. Most clips are a single segment; only "
    "splice when it genuinely tightens the moment. Give a one-line 'splice_reason' when you do.\n"
    "- Score every axis under the calibration above. Be strict on 'hook', 'payoff', and "
    "'virality' — those three gate the clip.\n\n"
    "FORBIDDEN (these will be rejected):\n"
    "- Any clip whose total spoken time is under {min_s:.0f}s or over {max_s:.0f}s.\n"
    "- One-sided clips: great hook / no payoff, great payoff / weak hook, or complete-but-boring "
    "(no share trigger).\n"
    "- A range that cuts before its own payoff_sentence.\n"
    "- The episode's intro section, generic intros/outros, ad reads, or sponsorship segments "
    "(unless they contain a real hook AND payoff).\n"
    "- Clips that significantly overlap each other.\n\n"
    "Respond ONLY with valid JSON (no markdown, no commentary). Use start_line/end_line for a "
    "simple clip, OR a 'segments' array for a compound one:\n"
    '{{"clips": [{{"start_line": <int>, "end_line": <int>, '
    '"segments": [{{"start_line": <int>, "end_line": <int>}}], '
    '"hook": "<scroll-stopping title, max ~10 words>", '
    '"hook_sentence": "<the exact opening line that makes someone stop scrolling>", '
    '"payoff_sentence": "<the verbatim transcript line where the payoff lands — inside the '
    'range>", '
    '"payoff": "<one sentence: what the viewer concretely learns or feels>", '
    '"share_trigger": "<one sentence: WHO shares/argues about this and WHY>", '
    '"reason": "<one line: why the hook + payoff work>", '
    '"splice_reason": "<one line: why these segments were spliced, or omit for a simple clip>", '
    '"scores": {{"hook": <0-10>, "payoff": <0-10>, "virality": <0-10>, "emotion": <0-10>, '
    '"quotability": <0-10>, "controversy": <0-10>, "standalone": <0-10>}}}}]}}\n'
    "For a simple clip, use start_line/end_line and omit 'segments' (or give it one range). "
    "If nothing is clip-worthy, return an empty clips list."
)


@dataclass
class LineRange:
    """An inclusive ``[start_line, end_line]`` reference into the numbered transcript."""

    start_line: int
    end_line: int


@dataclass
class JudgedClip:
    """One moment Gemini selected, with its per-dimension scores.

    ``segments`` is a list to leave room for compound (multi-range) clips later; this round the
    judge only emits the single-segment case, so ``clip_type`` is always ``"simple"``.
    """

    segments: list[LineRange]
    hook: str = ""
    hook_sentence: str = ""
    payoff_sentence: str = ""
    payoff: str = ""
    share_trigger: str = ""
    reason: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    clip_type: str = "simple"
    splice_reason: str = ""

    @property
    def start_line(self) -> int:
        return self.segments[0].start_line

    @property
    def end_line(self) -> int:
        return self.segments[-1].end_line


def _clamp_score(value: object) -> float:
    """Coerce to float and clamp to 0-10; non-numeric -> 0.0."""
    try:
        return max(0.0, min(10.0, float(value)))  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return 0.0


def _loads_loose(text: str) -> object | None:
    """Parse JSON tolerantly: strip ```` ```json ```` fences, else grab the outermost braces."""
    stripped = re.sub(r"^```(?:json)?\s*", "", text.strip())
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except (ValueError, TypeError):
        start, end = stripped.find("{"), stripped.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except (ValueError, TypeError):
                return None
        return None


def _segments_from(item: dict) -> list[LineRange]:
    """Pull line ranges from a clip item, accepting either flat keys or a ``segments`` list."""
    raw = item.get("segments")
    pairs: list[tuple[object, object]] = []
    if isinstance(raw, list) and raw:
        for seg in raw:
            if isinstance(seg, dict):
                pairs.append((seg.get("start_line"), seg.get("end_line")))
    else:
        pairs.append((item.get("start_line"), item.get("end_line")))
    out: list[LineRange] = []
    for s, e in pairs:
        try:
            out.append(LineRange(int(s), int(e)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return out


def parse_clips(text: str) -> list[JudgedClip]:
    """Parse Gemini's JSON reply into ``JudgedClip``s. Garbage / wrong shape -> ``[]``.

    Tolerant of markdown fences / wrapping prose. Each clip must carry at least one valid line
    range; scores are clamped to 0-10 with missing axes defaulting to 0.
    """
    data = _loads_loose(text)
    if not isinstance(data, dict) or not isinstance(data.get("clips"), list):
        return []
    out: list[JudgedClip] = []
    for item in data["clips"]:
        if not isinstance(item, dict):
            continue
        segments = _segments_from(item)
        if not segments:
            continue
        raw_scores = item.get("scores") if isinstance(item.get("scores"), dict) else {}
        scores = {dim: _clamp_score(raw_scores.get(dim, 0)) for dim in DIMENSIONS}
        out.append(
            JudgedClip(
                segments=segments,
                hook=str(item.get("hook", "")),
                hook_sentence=str(item.get("hook_sentence", "")),
                payoff_sentence=str(item.get("payoff_sentence", "")),
                payoff=str(item.get("payoff", "")),
                share_trigger=str(item.get("share_trigger", "")),
                reason=str(item.get("reason", "")),
                scores=scores,
                clip_type="compound" if len(segments) > 1 else "simple",
                splice_reason=str(item.get("splice_reason", "")),
            )
        )
    return out


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set — the Gemini judge needs an API key "
            "(export GEMINI_API_KEY=… or set it in the repo-root .env)."
        )
    return key


def _call_gemini(prompt: str, *, model: str = DEFAULT_MODEL, api_key: str | None = None) -> str:
    """One Gemini ``generate_content`` call, JSON-forced; return the raw text (I/O)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key or _api_key())
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return resp.text or ""


def classify_content(
    lines: list[Line],
    *,
    model: str | None = None,
    api_key: str | None = None,
    on_progress: ProgressCb | None = None,
) -> dict[str, str]:
    """Cheap pre-pass: classify content_type + density to condition the judge prompt (I/O).

    Best-effort — any failure (or no key) falls back to a neutral classification.
    """
    model = model or _classify_model()
    fallback = {"content_type": "other", "density": "medium"}
    if not lines:
        return fallback
    sample = format_transcript(lines[:30])[:3000]
    if on_progress:
        on_progress("classifying content type…", None)
    try:
        raw = _call_gemini(CONTENT_TYPE_PROMPT + sample, model=model, api_key=api_key)
        data = _loads_loose(raw)
    except Exception:  # noqa: BLE001 - pre-pass is best-effort
        return fallback
    if not isinstance(data, dict):
        return fallback
    return {
        "content_type": str(data.get("content_type", fallback["content_type"])),
        "density": str(data.get("density", fallback["density"])),
    }


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_prompt(
    lines: list[Line],
    *,
    content_info: dict[str, str] | None,
    target: int,
    chapters: list[dict] | None = None,
) -> str:
    """Assemble the full judge prompt for the whole transcript.

    ``chapters`` (the source video's chapter list, if any) is interleaved into the numbered
    transcript as ``### CHAPTER`` markers so the judge can respect the video's own topic
    boundaries — see :func:`clipgen.lines.format_transcript`.
    """
    info = content_info or {"content_type": "other", "density": "medium"}
    system = JUDGE_PROMPT_TEMPLATE.format(
        criteria=VIRALITY_CRITERIA,
        rubric=SCORING_RUBRIC,
        content_type=info.get("content_type", "other"),
        density=info.get("density", "medium"),
        n=target,
        min_s=MIN_CLIP_S,
        max_s=MAX_CLIP_S,
    )
    return system + "\n\nTRANSCRIPT:\n" + format_transcript(lines, chapters)


def judge_transcript(
    lines: list[Line],
    *,
    model: str | None = None,
    api_key: str | None = None,
    content_info: dict[str, str] | None = None,
    target: int = DEFAULT_TARGET_CLIPS,
    chapters: list[dict] | None = None,
    on_progress: ProgressCb | None = None,
    prompt_path: Path | None = None,
    response_path: Path | None = None,
) -> list[JudgedClip]:
    """Format the whole transcript, make one Gemini call, parse the moments (I/O).

    On a parse miss, retries once with a stricter JSON instruction. When ``prompt_path`` /
    ``response_path`` are given, the exact prompt is written **before** the call (so it survives a
    failed call, for review) and the raw reply on success (so it can be re-parsed without paying
    for another call). ``chapters`` (if any) are interleaved into the transcript as segmentation
    hints.
    """
    if not lines:
        return []
    model = model or _judge_model()
    prompt = build_prompt(lines, content_info=content_info, target=target, chapters=chapters)
    if prompt_path is not None:
        _write_text(prompt_path, prompt)
    if on_progress:
        on_progress(f"judging episode with {model}…", None)
    raw = _call_gemini(prompt, model=model, api_key=api_key)
    clips = parse_clips(raw)
    if not clips and raw.strip():
        if on_progress:
            on_progress("judge output unparseable; retrying with stricter format…", None)
        strict = prompt + (
            "\n\nIMPORTANT: Return ONLY valid JSON with a top-level 'clips' array. "
            "No markdown fences, no commentary."
        )
        raw = _call_gemini(strict, model=model, api_key=api_key)
        clips = parse_clips(raw)
    if response_path is not None:
        _write_text(response_path, raw)
    if on_progress:
        on_progress(f"judge selected {len(clips)} moments", None)
    return clips


# --- Pass B: verify/refine each candidate against its actual excerpt --------------------------
#
# Pass A judges the whole episode at once, which is where clickbait slips through: the model
# remembers the *idea* of a moment, not what its lines actually contain. Pass B re-presents each
# candidate as its verbatim excerpt (plus context) and forces a KEEP/DROP verdict judged only on
# what is visibly there — the payoff must be in the excerpt, the boundaries must be clean, and
# the moment must survive the share test a second time.

REFINE_CONTEXT_LINES = 8  # transcript lines of context shown on each side of a candidate

REFINE_PROMPT_TEMPLATE = (
    "You are the final quality gate for short-form clips cut from a longer video. Below are "
    "candidate clips a first-pass editor selected. Each candidate shows the editor's claims "
    "(hook / payoff / share trigger) and the ACTUAL transcript excerpt — the claimed clip lines "
    "between the >>> CLIP STARTS and <<< CLIP ENDS markers, plus a few surrounding context "
    "lines. Judge each candidate ONLY by what is actually in its excerpt, never by the claim.\n\n"
    "{rubric}\n"
    "For each candidate decide KEEP or DROP.\n"
    "DROP when ANY of these hold:\n"
    "- CLICKBAIT: the hook promises something the clip lines never deliver — the explanation, "
    "punchline, or revelation is not actually inside the clip (being elsewhere in the episode "
    "does not count).\n"
    "- Needs prior context a cold viewer doesn't have, or opens/closes on dangling references.\n"
    "- Starts or ends mid-thought and the visible context lines cannot fix it.\n"
    "- MERELY INFORMATIVE: complete and well-said, but no nameable audience would stop, watch "
    "to the end, and react (share / argue / quote / tag a friend). Generic advice, history "
    "lessons, tech explainers: DROP.\n"
    "For KEEP: tighten start_line/end_line to the exact arc — open ON the hook line, end 1-2 "
    "lines after the payoff lands; strip preamble and trailing chatter; you may use the context "
    "lines to fix a boundary. Total spoken time {min_s:.0f}-{max_s:.0f}s, sweet spot 30-60s. "
    "Restate the verbatim payoff_sentence from the excerpt, and re-score EVERY axis under the "
    "calibration above, judged against the excerpt alone — do not inherit the first pass's "
    "scores.\n\n"
    "Respond ONLY with valid JSON (no markdown, no commentary), one entry per candidate id, in "
    "order:\n"
    '{{"clips": [{{"id": <int>, "verdict": "KEEP" | "DROP", '
    '"reason": "<one line: why kept/dropped>", '
    '"start_line": <int>, "end_line": <int>, '
    '"payoff_sentence": "<verbatim line where the payoff lands>", '
    '"scores": {{"hook": <0-10>, "payoff": <0-10>, "virality": <0-10>, "emotion": <0-10>, '
    '"quotability": <0-10>, "controversy": <0-10>, "standalone": <0-10>}}}}]}}\n\n'
    "CANDIDATES:\n{candidates}"
)


def _fmt_line(ln: Line) -> str:
    s = max(0, int(ln.start_s))
    return f"{ln.index} | {s // 60:02d}:{s % 60:02d} | {ln.speaker} | {ln.text}"


def _spoken_s(lines: list[Line], segments: list[LineRange]) -> float:
    n = len(lines)
    total = 0.0
    for seg in segments:
        a = max(1, min(seg.start_line, n))
        b = max(1, min(seg.end_line, n))
        if a > b:
            a, b = b, a
        total += lines[b - 1].end_s - lines[a - 1].start_s
    return total


def build_refine_prompt(candidates: list[JudgedClip], lines: list[Line]) -> str:
    """Assemble the pass-B prompt: every candidate with its verbatim excerpt + context (pure)."""
    n = len(lines)
    blocks: list[str] = []
    for i, cand in enumerate(candidates, start=1):
        first, last = cand.start_line, cand.end_line
        lo = max(1, min(first, n) - REFINE_CONTEXT_LINES)
        hi = min(n, max(last, 1) + REFINE_CONTEXT_LINES)
        rows: list[str] = []
        for idx in range(lo, hi + 1):
            if idx == min(first, n):
                rows.append(">>> CLIP STARTS")
            rows.append(_fmt_line(lines[idx - 1]))
            if idx == min(last, n):
                rows.append("<<< CLIP ENDS")
        ranges = ", ".join(f"{seg.start_line}-{seg.end_line}" for seg in cand.segments)
        blocks.append(
            f"## Candidate {i}\n"
            f"Claimed lines: {ranges} (~{_spoken_s(lines, cand.segments):.0f}s spoken)\n"
            f"Hook title: {cand.hook}\n"
            f"Claimed hook line: {cand.hook_sentence}\n"
            f"Claimed payoff: {cand.payoff}\n"
            f"Claimed payoff line: {cand.payoff_sentence}\n"
            f"Claimed share trigger: {cand.share_trigger}\n"
            "Excerpt:\n" + "\n".join(rows)
        )
    return REFINE_PROMPT_TEMPLATE.format(
        rubric=SCORING_RUBRIC,
        min_s=MIN_CLIP_S,
        max_s=MAX_CLIP_S,
        candidates="\n\n".join(blocks),
    )


def parse_refined(
    text: str,
    candidates: list[JudgedClip],
    rejects: list[dict] | None = None,
) -> list[JudgedClip] | None:
    """Apply pass-B verdicts to the candidates. Garbage / wrong shape -> ``None`` (pure).

    KEEP verdicts update the candidate in place (tightened range, restated payoff line, fresh
    scores — per-axis fallback to the pass-A score, so a partially-scored reply can't zero a clip
    at the gate). DROP verdicts remove it, recording the verifier's reason in ``rejects``.
    Candidates the reply never mentions are kept unchanged (fail-open: a truncated reply must not
    silently kill clips). ``None`` means the reply was unparseable and the caller should retry or
    fall back to the candidates as-is.
    """
    data = _loads_loose(text)
    if not isinstance(data, dict) or not isinstance(data.get("clips"), list):
        return None
    dropped: set[int] = set()
    updated: dict[int, JudgedClip] = {}
    for item in data["clips"]:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("id"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if not 1 <= idx <= len(candidates):
            continue
        cand = candidates[idx - 1]
        reason = str(item.get("reason", ""))
        if str(item.get("verdict", "")).strip().upper() == "DROP":
            dropped.add(idx)
            if rejects is not None:
                rejects.append({"hook": cand.hook, "stage": "refine", "reason": reason})
            continue
        segments = _segments_from(item) or cand.segments
        raw_scores = item.get("scores") if isinstance(item.get("scores"), dict) else {}
        scores = {
            dim: _clamp_score(raw_scores[dim]) if dim in raw_scores else cand.scores.get(dim, 0.0)
            for dim in DIMENSIONS
        }
        payoff_sentence = str(item.get("payoff_sentence", "")) or cand.payoff_sentence
        updated[idx] = JudgedClip(
            segments=segments,
            hook=cand.hook,
            hook_sentence=cand.hook_sentence,
            payoff_sentence=payoff_sentence,
            payoff=cand.payoff,
            share_trigger=cand.share_trigger,
            reason=reason or cand.reason,
            scores=scores,
            clip_type="compound" if len(segments) > 1 else "simple",
            splice_reason=cand.splice_reason,
        )
    return [
        updated.get(i, cand)
        for i, cand in enumerate(candidates, start=1)
        if i not in dropped
    ]


def refine_clips(
    candidates: list[JudgedClip],
    lines: list[Line],
    *,
    model: str | None = None,
    api_key: str | None = None,
    rejects: list[dict] | None = None,
    on_progress: ProgressCb | None = None,
    prompt_path: Path | None = None,
    response_path: Path | None = None,
) -> list[JudgedClip]:
    """Pass B: one Gemini call that re-judges each candidate against its excerpt (I/O).

    Mirrors :func:`judge_transcript`'s I/O contract — the exact prompt is persisted before the
    call, the raw reply on success, and a parse miss gets one stricter retry. The pass is
    fail-open: if the reply is unusable twice, the candidates come back unchanged rather than
    losing the run to the extra call.
    """
    if not candidates or not lines:
        return candidates
    model = model or _judge_model()
    prompt = build_refine_prompt(candidates, lines)
    if prompt_path is not None:
        _write_text(prompt_path, prompt)
    if on_progress:
        on_progress(f"verifying {len(candidates)} candidates with {model}…", None)
    raw = _call_gemini(prompt, model=model, api_key=api_key)
    refined = parse_refined(raw, candidates, rejects)
    if refined is None and raw.strip():
        if on_progress:
            on_progress("verifier output unparseable; retrying with stricter format…", None)
        strict = prompt + (
            "\n\nIMPORTANT: Return ONLY valid JSON with a top-level 'clips' array. "
            "No markdown fences, no commentary."
        )
        raw = _call_gemini(strict, model=model, api_key=api_key)
        refined = parse_refined(raw, candidates, rejects)
    if response_path is not None:
        _write_text(response_path, raw)
    if refined is None:
        if on_progress:
            on_progress("verifier reply unusable — keeping first-pass candidates", None)
        return candidates
    if on_progress:
        on_progress(f"verifier kept {len(refined)} of {len(candidates)} candidates", None)
    return refined
