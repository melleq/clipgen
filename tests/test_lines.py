"""Tests for the numbered-line transcript model (clipgen.lines)."""

from __future__ import annotations

from clipgen.lines import (
    build_lines,
    format_transcript,
    resolve_lines,
    roleify_speakers,
    select_chapter_lines,
)
from clipgen.transcript import Transcript, Word


def _t(*words: tuple[str, float, float, str | None]) -> Transcript:
    return Transcript(words=[Word(text=t, start_s=a, end_s=b, speaker=s) for t, a, b, s in words])


def test_breaks_on_sentence_punctuation() -> None:
    t = _t(("Hello.", 0.0, 1.0, None), ("World.", 1.0, 2.0, None))
    lines = build_lines(t)
    assert [ln.text for ln in lines] == ["Hello.", "World."]
    assert [ln.index for ln in lines] == [1, 2]


def test_breaks_on_speaker_change() -> None:
    t = _t(("Hi", 0.0, 1.0, "S1"), ("there", 1.0, 2.0, "S2"))
    lines = build_lines(t)
    assert len(lines) == 2


def test_complete_flag_tracks_terminal_punctuation() -> None:
    t = _t(("Hello.", 0.0, 1.0, None), ("world", 1.0, 2.0, None), ("again", 2.0, 3.0, None))
    lines = build_lines(t, max_words=2)
    assert lines[0].complete is True  # "Hello." ends a sentence
    assert lines[-1].complete is False  # "world again" was cap-flushed mid-sentence


def test_cap_split_marks_run_on_incomplete() -> None:
    # Four unpunctuated words with a 2-word cap → two lines, both mid-sentence.
    t = _t(*((f"w{i}", float(i), float(i + 1), None) for i in range(4)))
    lines = build_lines(t, max_words=2)
    assert len(lines) == 2
    assert [ln.complete for ln in lines] == [False, False]


def test_speaker_change_flush_marks_incomplete() -> None:
    t = _t(("Hi", 0.0, 1.0, "S1"), ("there.", 1.0, 2.0, "S2"))
    lines = build_lines(t)
    assert lines[0].complete is False  # "Hi" was cut off by the speaker change
    assert lines[1].complete is True


def test_roleify_maps_speakers_to_persons() -> None:
    t = _t(("a", 0.0, 1.0, "SPEAKER_00"), ("b", 1.0, 2.0, "SPEAKER_01"))
    mapping = roleify_speakers(t)
    assert mapping["SPEAKER_00"] == "Person A"
    assert mapping["SPEAKER_01"] == "Person B"


def test_roleify_no_diarization() -> None:
    t = _t(("a", 0.0, 1.0, None))
    assert roleify_speakers(t)[None] == "Person A"


def test_resolve_lines_clamps_and_orders() -> None:
    t = _t(("One.", 0.0, 2.0, None), ("Two.", 2.0, 4.0, None), ("Three.", 4.0, 6.0, None))
    lines = build_lines(t)
    assert resolve_lines(lines, 1, 3) == (0.0, 6.0)
    # inverted + out of range still resolves sanely
    assert resolve_lines(lines, 9, -1) == (0.0, 6.0)


def test_format_transcript_has_header_and_rows() -> None:
    t = _t(("Hello.", 0.0, 65.0, None))
    out = format_transcript(build_lines(t))
    assert "line | mm:ss | speaker | text" in out
    assert "1 | 00:00 | Person" not in out  # no speaker_map applied → raw label ("")
    assert "Hello." in out


def test_format_transcript_interleaves_chapters() -> None:
    t = _t(
        ("Intro.", 0.0, 5.0, None),
        ("Body.", 10.0, 15.0, None),
        ("End.", 20.0, 25.0, None),
    )
    lines = build_lines(t)
    chapters = [
        {"start_s": 0.0, "title": "Welcome"},
        {"start_s": 12.0, "title": "Main topic"},  # falls at/after the 3rd line (starts 20s)
    ]
    out = format_transcript(lines, chapters)
    assert "prefer clips that stay within a single chapter" in out
    lines_out = out.splitlines()
    # First chapter marker precedes the first transcript row.
    first_marker = next(i for i, s in enumerate(lines_out) if s.startswith("### CHAPTER"))
    first_row = next(i for i, s in enumerate(lines_out) if s.startswith("1 |"))
    assert first_marker < first_row
    assert "### CHAPTER (00:00): Welcome" in out
    assert "### CHAPTER (00:12): Main topic" in out
    # Chapter markers carry no line number, so the integer line-range contract is intact:
    # exactly three numbered transcript rows, numbered 1..3.
    numbered = [s for s in lines_out if s[:1].isdigit() and " | " in s]
    assert [s.split(" | ", 1)[0] for s in numbered] == ["1", "2", "3"]


def test_format_transcript_no_chapters_unchanged() -> None:
    t = _t(("Hello.", 0.0, 5.0, None))
    assert format_transcript(build_lines(t), []) == format_transcript(build_lines(t))
    assert "### CHAPTER" not in format_transcript(build_lines(t), None)


def _six_line_transcript() -> list:
    # One line every 10s: starts at 0,10,20,30,40,50.
    t = _t(*[(f"Line {i}.", i * 10.0, i * 10.0 + 9.0, None) for i in range(6)])
    return build_lines(t)


_CHAPTERS = [
    {"start_s": 0.0, "title": "Intro"},  # [0, 20)  -> lines at 0,10
    {"start_s": 20.0, "title": "Main"},  # [20, 40) -> lines at 20,30
    {"start_s": 40.0, "title": "Outro"},  # [40, ∞) -> lines at 40,50
]


def test_select_chapter_lines_subset_reindexes() -> None:
    lines = _six_line_transcript()
    filtered, chapters = select_chapter_lines(lines, _CHAPTERS, [1])  # Main only
    assert [ln.text for ln in filtered] == ["Line 2.", "Line 3."]
    assert [ln.index for ln in filtered] == [1, 2]  # re-numbered from 1 for grounding
    assert [c["title"] for c in chapters] == ["Main"]
    # The re-indexed list still resolves positionally (grounding contract holds).
    assert resolve_lines(filtered, 1, 2) == (20.0, 39.0)


def test_select_chapter_lines_drops_intro() -> None:
    lines = _six_line_transcript()
    filtered, chapters = select_chapter_lines(lines, _CHAPTERS, [1, 2])  # everything but Intro
    assert [ln.text for ln in filtered] == ["Line 2.", "Line 3.", "Line 4.", "Line 5."]
    assert [ln.index for ln in filtered] == [1, 2, 3, 4]
    assert [c["title"] for c in chapters] == ["Main", "Outro"]


def test_select_chapter_lines_all_or_none_is_passthrough() -> None:
    lines = _six_line_transcript()
    # None (no selection), every chapter, and an empty/invalid selection all send everything.
    for sel in (None, [0, 1, 2], [], [9]):
        filtered, chapters = select_chapter_lines(lines, _CHAPTERS, sel)
        assert len(filtered) == 6
        assert chapters == _CHAPTERS
