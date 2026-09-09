"""Tests for the judge's pure layer: prompt assembly and the tolerant reply parsers."""

from __future__ import annotations

from clipgen.judge import (
    DIMENSIONS,
    JudgedClip,
    LineRange,
    build_prompt,
    build_refine_prompt,
    parse_clips,
    parse_refined,
)
from clipgen.lines import Line


def test_parses_plain_json() -> None:
    raw = (
        '{"clips": [{"start_line": 3, "end_line": 9, "hook": "H", '
        '"hook_sentence": "opening", "reason": "why", '
        '"scores": {"hook": 9, "payoff": 6, "emotion": 4, "quotability": 7, '
        '"controversy": 2, "standalone": 8}}]}'
    )
    clips = parse_clips(raw)
    assert len(clips) == 1
    c = clips[0]
    assert (c.start_line, c.end_line) == (3, 9)
    assert c.hook == "H"
    assert c.scores["hook"] == 9.0
    assert c.scores["payoff"] == 6.0
    assert c.clip_type == "simple"
    assert set(c.scores) == set(DIMENSIONS)


def test_parses_compound_segments() -> None:
    raw = (
        '{"clips": [{"segments": [{"start_line": 3, "end_line": 5}, '
        '{"start_line": 9, "end_line": 11}], "hook": "H", '
        '"splice_reason": "skip the tangent", '
        '"scores": {"hook": 8, "payoff": 8}}]}'
    )
    c = parse_clips(raw)[0]
    assert c.clip_type == "compound"
    assert [(s.start_line, s.end_line) for s in c.segments] == [(3, 5), (9, 11)]
    assert c.start_line == 3 and c.end_line == 11
    assert c.splice_reason == "skip the tangent"


def test_strips_markdown_fences() -> None:
    raw = '```json\n{"clips": [{"start_line": 1, "end_line": 2}]}\n```'
    assert len(parse_clips(raw)) == 1


def test_scores_clamped_and_defaulted() -> None:
    raw = '{"clips": [{"start_line": 1, "end_line": 2, "scores": {"hook": 99, "emotion": -3}}]}'
    c = parse_clips(raw)[0]
    assert c.scores["hook"] == 10.0
    assert c.scores["emotion"] == 0.0
    assert c.scores["quotability"] == 0.0  # missing axis defaults to 0


def test_garbage_returns_empty() -> None:
    assert parse_clips("not json at all") == []
    assert parse_clips("") == []
    assert parse_clips('{"nope": 1}') == []


def test_item_without_line_range_skipped() -> None:
    raw = '{"clips": [{"hook": "no range"}, {"start_line": 1, "end_line": 2}]}'
    assert len(parse_clips(raw)) == 1


def test_parses_payoff_and_share_trigger_fields() -> None:
    raw = (
        '{"clips": [{"start_line": 1, "end_line": 4, '
        '"payoff_sentence": "the exact landing line", '
        '"payoff": "viewer learns the trick", '
        '"share_trigger": "devs will argue about it", '
        '"scores": {"virality": 8}}]}'
    )
    c = parse_clips(raw)[0]
    assert c.payoff_sentence == "the exact landing line"
    assert c.payoff == "viewer learns the trick"
    assert c.share_trigger == "devs will argue about it"
    assert c.scores["virality"] == 8.0
    assert "virality" in DIMENSIONS


# --- prompt assembly ------------------------------------------------------------------------


def _line(i: int, text: str = "", *, start: float | None = None) -> Line:
    s = start if start is not None else (i - 1) * 10.0
    return Line(index=i, start_s=s, end_s=s + 10.0, speaker="Person A",
                text=text or f"this is sentence {i} end.")


def test_build_prompt_carries_the_new_contract() -> None:
    prompt = build_prompt([_line(1), _line(2)], content_info=None, target=5)
    assert "between 20 and 90 seconds" in prompt
    assert "SCORING CALIBRATION" in prompt
    assert "NOT VIRAL" in prompt
    assert "payoff_sentence" in prompt
    assert "share_trigger" in prompt
    assert "run-ons" in prompt  # warns about unpunctuated ASR regions


def _candidate(start: int, end: int) -> JudgedClip:
    return JudgedClip(
        segments=[LineRange(start, end)],
        hook="The hook title",
        hook_sentence="this is sentence 3 end.",
        payoff_sentence="this is sentence 5 end.",
        payoff="viewer gets the point",
        share_trigger="finance twitter argues",
        scores={dim: 7.0 for dim in DIMENSIONS},
    )


def test_build_refine_prompt_shows_excerpt_with_markers() -> None:
    lines = [_line(i) for i in range(1, 21)]
    prompt = build_refine_prompt([_candidate(3, 5)], lines)
    assert "## Candidate 1" in prompt
    assert ">>> CLIP STARTS" in prompt
    assert "<<< CLIP ENDS" in prompt
    assert "this is sentence 3 end." in prompt  # the clip lines are shown verbatim
    assert "this is sentence 1 end." in prompt  # …with surrounding context
    assert "finance twitter argues" in prompt  # the claims are shown for verification
    assert "KEEP" in prompt and "DROP" in prompt


# --- pass-B verdict application -------------------------------------------------------------


def test_parse_refined_keep_tightens_and_rescores() -> None:
    cand = _candidate(3, 9)
    raw = (
        '{"clips": [{"id": 1, "verdict": "KEEP", "reason": "tightened", '
        '"start_line": 4, "end_line": 7, "payoff_sentence": "restated line", '
        '"scores": {"hook": 9}}]}'
    )
    out = parse_refined(raw, [cand])
    assert out is not None and len(out) == 1
    c = out[0]
    assert (c.start_line, c.end_line) == (4, 7)
    assert c.payoff_sentence == "restated line"
    assert c.scores["hook"] == 9.0
    assert c.scores["payoff"] == 7.0  # unscored axes fall back to pass-A, not zero
    assert c.share_trigger == cand.share_trigger  # claims carry through


def test_parse_refined_drop_removes_and_records_reason() -> None:
    rejects: list[dict] = []
    raw = '{"clips": [{"id": 1, "verdict": "DROP", "reason": "clickbait, no payoff in range"}]}'
    out = parse_refined(raw, [_candidate(3, 9)], rejects)
    assert out == []
    assert rejects == [
        {"hook": "The hook title", "stage": "refine", "reason": "clickbait, no payoff in range"}
    ]


def test_parse_refined_unmentioned_candidates_fail_open() -> None:
    cands = [_candidate(3, 9), _candidate(12, 15)]
    raw = '{"clips": [{"id": 1, "verdict": "DROP", "reason": "weak"}]}'
    out = parse_refined(raw, cands)
    assert out is not None and len(out) == 1
    assert out[0].start_line == 12  # the unmentioned candidate survives unchanged


def test_parse_refined_garbage_returns_none() -> None:
    assert parse_refined("not json", [_candidate(1, 3)]) is None
    assert parse_refined('{"nope": 1}', [_candidate(1, 3)]) is None
