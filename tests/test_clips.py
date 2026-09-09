"""Tests for the grounding stage (clipgen.clips): line-ranges → ranked, length-valid clips.

Length contract is 20-90s (see judge.MIN_CLIP_S/MAX_CLIP_S); a clip must clear the triple
hook+payoff+virality floor (clips.GATE_FLOOR = 7) to qualify, and a judge-sourced clip must
contain its own payoff line. Compound clips keep their spliced segments.
"""

from __future__ import annotations

from clipgen.clips import ClipItem, adjust_clip, content_score, ground, merge_clips
from clipgen.judge import JudgedClip, LineRange
from clipgen.lines import Line
from clipgen.transcript import Word


def _text(i: int) -> str:
    # Distinct words around the index so fuzzy sentence matching can't confuse line 1 with
    # line 10 ("… 1 end." is not a prefix of "… 10 end.").
    return f"this is sentence {i} end."


def _lines(
    n: int,
    *,
    step: float = 10.0,
    speaker: str = "Person A",
    incomplete: set[int] | None = None,
    texts: dict[int, str] | None = None,
) -> list[Line]:
    """n sentence-lines, each `step` seconds long and back-to-back. Lines whose index is in
    ``incomplete`` are cap-split run-ons (no terminal punctuation); ``texts`` overrides text."""
    out = []
    for i in range(1, n + 1):
        run_on = bool(incomplete and i in incomplete)
        text = (texts or {}).get(i) or (_text(i)[:-1] if run_on else _text(i))
        out.append(
            Line(index=i, start_s=(i - 1) * step, end_s=i * step, speaker=speaker,
                 text=text, complete=not run_on)
        )
    return out


# All dimensions default to 7 → clears the hook/payoff/virality floor of 7, so a clip qualifies
# unless a test explicitly lowers a gated axis.
_BASE = {"hook": 7.0, "payoff": 7.0, "virality": 7.0, "emotion": 7.0, "quotability": 7.0,
         "controversy": 7.0, "standalone": 7.0}


def _judged(start: int, end: int, *, payoff_line: int | None = None,
            hook_sentence: str = "", **scores: float) -> JudgedClip:
    return JudgedClip(
        segments=[LineRange(start, end)],
        hook=f"clip {start}-{end}",
        hook_sentence=hook_sentence,
        payoff_sentence=_text(payoff_line if payoff_line is not None else start),
        scores={**_BASE, **scores},
    )


def _judged_multi(segments: list[tuple[int, int]], *, payoff_line: int | None = None,
                  **scores: float) -> JudgedClip:
    return JudgedClip(
        segments=[LineRange(a, b) for a, b in segments],
        hook="compound",
        splice_reason="skip filler",
        payoff_sentence=_text(payoff_line if payoff_line is not None else segments[-1][0]),
        scores={**_BASE, **scores},
    )


def test_resolves_line_range_to_seconds() -> None:
    lines = _lines(10)  # each line 10s
    # lines 2..4 inclusive → 10s .. 40s = 30s (inside 20-90)
    items = ground([_judged(2, 4)], lines)
    assert len(items) == 1
    assert items[0].start_s == 10.0
    assert items[0].end_s == 40.0
    assert items[0].duration_s == 30.0
    assert items[0].clip_type == "simple"
    assert items[0].segments == [[10.0, 40.0]]
    assert items[0].text == f"{_text(2)} {_text(3)} {_text(4)}"


def test_short_range_extended_to_minimum_length() -> None:
    lines = _lines(10, step=6.0)  # each 6s → one line is far below the 20s minimum
    items = ground([_judged(3, 3)], lines)
    assert len(items) == 1
    assert items[0].duration_s >= 20.0


def test_unreachable_minimum_is_dropped() -> None:
    lines = _lines(1, step=6.0)  # one 6s line, nothing to extend into → below 20s min
    rejects: list[dict] = []
    assert ground([_judged(1, 1)], lines, rejects=rejects) == []
    assert rejects and "20" in rejects[0]["reason"]


def test_overlong_range_trimmed_from_end() -> None:
    lines = _lines(20)  # 200s total; each 10s
    items = ground([_judged(1, 20, payoff_line=1)], lines)  # default max 90s
    assert items[0].duration_s <= 90.0
    assert items[0].start_s == 0.0  # hook kept at the start


def test_overlap_suppression_keeps_higher_content() -> None:
    lines = _lines(12)
    strong = _judged(1, 3, quotability=10.0)  # higher content, span 0-30
    weak = _judged(3, 5, quotability=5.0)  # overlaps (starts at 20 < 30)
    items = ground([weak, strong], lines)
    assert len(items) == 1
    assert items[0].hook == "clip 1-3"


def test_ranked_by_content_score_descending() -> None:
    lines = _lines(40)
    a = _judged(1, 3, quotability=5.0)
    b = _judged(10, 12, quotability=10.0)
    c = _judged(20, 22, quotability=7.0)
    items = ground([a, b, c], lines)
    scores = [it.content_score for it in items]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limit() -> None:
    lines = _lines(60)
    judged = [_judged(i, i + 2) for i in range(1, 40, 6)]  # non-overlapping, all pass the gate
    items = ground(judged, lines, top_k=3)
    assert len(items) == 3


def test_no_padding_below_top_k() -> None:
    lines = _lines(20)
    items = ground([_judged(1, 3), _judged(8, 10)], lines)  # only 2 qualify, default top_k=10
    assert len(items) == 2


# --- hook + payoff + virality gate ----------------------------------------------------------


def test_gate_drops_weak_hook() -> None:
    lines = _lines(10)
    assert ground([_judged(1, 3, hook=3.0, payoff=9.0)], lines) == []


def test_gate_drops_weak_payoff() -> None:
    lines = _lines(10)
    assert ground([_judged(1, 3, hook=9.0, payoff=3.0)], lines) == []


def test_gate_drops_weak_virality() -> None:
    # Complete and well-formed but not shareable — the slop the gate exists to stop.
    lines = _lines(10)
    rejects: list[dict] = []
    assert ground([_judged(1, 3, hook=9.0, payoff=9.0, virality=4.0)], lines,
                  rejects=rejects) == []
    assert rejects and "virality" in rejects[0]["reason"]


def test_gate_passes_when_all_three_clear_floor() -> None:
    lines = _lines(10)
    assert len(ground([_judged(1, 3, hook=7.0, payoff=7.0, virality=7.0)], lines)) == 1


def test_apply_gate_false_keeps_one_sided_clips() -> None:
    lines = _lines(10)
    # A weak-payoff clip is normally gated out, but imports bypass the gate.
    weak = _judged(1, 3, hook=9.0, payoff=1.0)
    assert ground([weak], lines) == []
    kept = ground([weak], lines, apply_gate=False, source="imported")
    assert len(kept) == 1
    assert kept[0].source == "imported"


def test_imported_clip_without_payoff_sentence_kept() -> None:
    lines = _lines(10)
    imported = JudgedClip(segments=[LineRange(1, 3)], hook="pasted", scores=dict(_BASE))
    assert len(ground([imported], lines, apply_gate=False, source="imported")) == 1


# --- payoff containment (clickbait guard) ---------------------------------------------------


def test_payoff_not_in_range_drops_clip() -> None:
    lines = _lines(10)
    bait = JudgedClip(
        segments=[LineRange(1, 3)], hook="clickbait",
        payoff_sentence="totally different words about nothing here",
        scores=dict(_BASE),
    )
    rejects: list[dict] = []
    assert ground([bait], lines, rejects=rejects) == []
    assert rejects and "payoff" in rejects[0]["reason"]


def test_missing_payoff_sentence_drops_judge_clip() -> None:
    lines = _lines(10)
    no_payoff = JudgedClip(segments=[LineRange(1, 3)], hook="no payoff", scores=dict(_BASE))
    assert ground([no_payoff], lines) == []


def test_trim_that_would_cut_payoff_drops_clip() -> None:
    # 150s range with the payoff on the last line: fitting 90s would cut the payoff → drop.
    lines = _lines(15)
    rejects: list[dict] = []
    assert ground([_judged(1, 15, payoff_line=15)], lines, rejects=rejects) == []
    assert rejects and "payoff" in rejects[0]["reason"]


# --- boundary hygiene: run-ons, hook re-anchoring, filler tails -----------------------------


def test_start_snapped_out_of_run_on() -> None:
    # Lines 2-3 are cap-split run-ons; a clip starting at 4 starts mid-sentence → walk back to 2.
    lines = _lines(6, incomplete={2, 3})
    items = ground([_judged(4, 5, payoff_line=5)], lines)
    assert len(items) == 1
    assert items[0].start_s == 10.0  # snapped back to line 2, where the thought begins


def test_end_snapped_out_of_run_on() -> None:
    # Lines 3-4 are run-ons; a clip ending at 3 ends mid-sentence → walk forward to 5.
    lines = _lines(6, incomplete={3, 4})
    items = ground([_judged(2, 3, payoff_line=2)], lines)
    assert len(items) == 1
    assert items[0].end_s == 50.0  # extended to line 5, where the thought completes


def test_hook_reanchors_start_to_hook_sentence() -> None:
    # The judge included two preamble lines before its own hook line → start snaps forward.
    lines = _lines(10)
    items = ground([_judged(1, 5, hook_sentence=_text(3), payoff_line=4)], lines)
    assert len(items) == 1
    assert items[0].start_s == 20.0  # opens ON the hook line (line 3), preamble dropped


def test_filler_tail_trimmed() -> None:
    lines = _lines(4, texts={4: "Yeah."})
    items = ground([_judged(1, 4, payoff_line=3)], lines)
    assert len(items) == 1
    assert items[0].end_s == 30.0  # the trailing "Yeah." is cut; the payoff closes the clip
    assert "Yeah." not in items[0].text


# --- merge (import) -------------------------------------------------------------------------


def test_merge_clips_expands_and_ranks() -> None:
    lines = _lines(40)
    existing = ground([_judged(1, 3, quotability=6.0)], lines)  # span 0-30
    extra = ground([_judged(10, 12, quotability=9.0)], lines, source="imported")  # span 90-120
    merged = merge_clips(existing, extra)
    assert len(merged) == 2  # non-overlapping → both kept
    assert merged[0].content_score >= merged[1].content_score  # ranked by score
    assert {c.source for c in merged} == {"gemini", "imported"}


def test_merge_clips_keeps_overlapping_distinct_takes() -> None:
    lines = _lines(20)
    existing = ground([_judged(1, 3, quotability=4.0)], lines)  # span 0-30
    imported = ground([_judged(2, 4, quotability=10.0)], lines, source="imported")  # overlaps 0-30
    merged = merge_clips(existing, imported)
    # Additive: overlapping-but-distinct takes are both kept (compare different LLM results).
    assert len(merged) == 2
    assert {c.source for c in merged} == {"gemini", "imported"}
    assert merged[0].content_score >= merged[1].content_score  # still ranked


def test_merge_clips_collapses_exact_duplicate_span() -> None:
    lines = _lines(20)
    a = ground([_judged(1, 3)], lines)  # span 0-30
    b = ground([_judged(1, 3, quotability=10.0)], lines, source="imported")  # identical span
    merged = merge_clips(a, b)
    assert len(merged) == 1  # exact same span → not stacked
    assert merged[0].source == "imported"  # the higher-scored copy is kept


def test_merge_clips_is_never_destructive() -> None:
    lines = _lines(30)
    existing = ground([_judged(1, 3), _judged(6, 8), _judged(11, 13)], lines)  # 3 clips
    imported = ground([_judged(2, 4, quotability=9.0)], lines, source="imported")  # overlaps one
    merged = merge_clips(existing, imported)
    assert len(merged) == 4  # all 3 existing kept + 1 imported added (nothing overwritten)


# --- compound clips -------------------------------------------------------------------------


def test_compound_keeps_two_segments_and_skips_gap() -> None:
    lines = _lines(10)  # 10s lines
    # segments 1-2 (0-20s) and 5-6 (40-60s): a 20s gap is spliced out
    items = ground([_judged_multi([(1, 2), (5, 6)])], lines)
    assert len(items) == 1
    c = items[0]
    assert c.clip_type == "compound"
    assert c.segments == [[0.0, 20.0], [40.0, 60.0]]
    assert c.start_s == 0.0 and c.end_s == 60.0
    assert c.duration_s == 40.0  # spoken time = 20 + 20, not the 60s span
    assert c.splice_reason == "skip filler"
    assert c.text == f"{_text(1)} {_text(2)} {_text(5)} {_text(6)}"


def test_compound_overlapping_segments_merge_to_simple() -> None:
    lines = _lines(10)
    # second segment overlaps the first → merged into one span, leaving a simple clip
    items = ground([_judged_multi([(1, 3), (2, 4)], payoff_line=4)], lines)
    assert len(items) == 1
    assert items[0].clip_type == "simple"
    assert len(items[0].segments) == 1


def test_compound_total_over_max_trimmed() -> None:
    lines = _lines(20)  # 10s lines
    # segments 1-2 (20s) + 5-12 (80s) = 100s total, over the 90s max → trimmed from the tail
    items = ground([_judged_multi([(1, 2), (5, 12)], payoff_line=5)], lines)
    assert len(items) == 1
    assert items[0].duration_s <= 90.0
    assert items[0].start_s == 0.0  # hook segment kept


# --- content score --------------------------------------------------------------------------


def test_content_score_gated_axes_co_lead() -> None:
    only_hook = content_score({"hook": 10})
    only_payoff = content_score({"payoff": 10})
    only_virality = content_score({"virality": 10})
    only_controversy = content_score({"controversy": 10})
    assert only_hook == only_payoff == only_virality  # equal weights
    assert only_hook > only_controversy  # and all outweigh a modulating axis


def test_empty_inputs() -> None:
    assert ground([], _lines(5)) == []
    assert ground([_judged(1, 2)], []) == []


# --- manual expand / trim -------------------------------------------------------------------


def _words(n: int, step: float = 1.0, sentence_every: int = 3) -> list[Word]:
    """n 1s words; every `sentence_every`-th word ends a sentence (trailing period)."""
    return [
        Word(text=f"w{i}" + ("." if (i + 1) % sentence_every == 0 else ""),
             start_s=i * step, end_s=(i + 1) * step)
        for i in range(n)
    ]


def _clip(start: float, end: float) -> ClipItem:
    return ClipItem(start_s=start, end_s=end, duration_s=end - start, segments=[[start, end]])


def test_expand_end_by_word() -> None:
    words = _words(9)  # 0..9s
    out = adjust_clip(_clip(0, 3), words, edge="end", unit="word", direction="expand")
    assert out.end_s == 4.0  # next word (w3) pulled in
    assert out.duration_s == 4.0
    assert out.segments == [[0.0, 4.0]]
    assert out.text.endswith("w3")


def test_expand_end_by_sentence() -> None:
    words = _words(9)  # sentences end at w2(3s), w5(6s), w8(9s)
    out = adjust_clip(_clip(0, 3), words, edge="end", unit="sentence", direction="expand")
    assert out.end_s == 6.0  # extends through the next full sentence


def test_trim_end_by_word() -> None:
    words = _words(9)
    out = adjust_clip(_clip(0, 3), words, edge="end", unit="word", direction="trim")
    assert out.end_s == 2.0  # last word dropped


def test_expand_start_by_word() -> None:
    words = _words(9)
    out = adjust_clip(_clip(3, 6), words, edge="start", unit="word", direction="expand")
    assert out.start_s == 2.0  # previous word (w2) pulled in


def test_trim_start_by_sentence() -> None:
    words = _words(9)  # clip covers two sentences (0-6); trim first sentence
    out = adjust_clip(_clip(0, 6), words, edge="start", unit="sentence", direction="trim")
    assert out.start_s == 3.0


def test_expand_end_noop_at_transcript_end() -> None:
    words = _words(9)
    out = adjust_clip(_clip(6, 9), words, edge="end", unit="word", direction="expand")
    assert out.end_s == 9.0  # nothing after — unchanged


def test_compound_expand_only_moves_outer_edge() -> None:
    words = _words(20)
    clip = ClipItem(start_s=0, end_s=3, duration_s=6, clip_type="compound",
                    segments=[[0.0, 3.0], [10.0, 13.0]])
    out = adjust_clip(clip, words, edge="end", unit="word", direction="expand")
    assert out.segments[0] == [0.0, 3.0]  # first segment untouched
    assert out.segments[1][1] == 14.0  # last segment's end extended by one word
    assert out.end_s == 14.0
