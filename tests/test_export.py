"""Tests for the pure name/path helpers of the clip exporter (clipgen.export)."""

from __future__ import annotations

from pathlib import Path

from clipgen.clips import ClipItem
from clipgen.export import clip_transcript_dict, safe_filename, unique_path
from clipgen.transcript import Transcript, Word


def test_safe_filename_strips_illegal_chars() -> None:
    assert safe_filename('The "secret" hack: do/this?') == "The secret hack dothis"


def test_safe_filename_collapses_whitespace_and_trims_dots() -> None:
    assert safe_filename("  ...Nobody   talks  about this.  ") == "Nobody talks about this"


def test_safe_filename_falls_back_when_empty() -> None:
    assert safe_filename("") == "clip"
    assert safe_filename("///", fallback="clip-3") == "clip-3"


def test_safe_filename_caps_length() -> None:
    out = safe_filename("word " * 60)  # 300 chars pre-trim
    assert len(out) <= 120


def test_clip_transcript_dict_compound_is_clip_relative() -> None:
    # A compound clip: [5,9] + [20,24]. Word timings must be relative to the exported clip (the
    # 9..20 gap spliced out), speakers roleified, and gap words excluded.
    words = [
        Word("The", 6.0, 6.3, "SPEAKER_00"),
        Word("gap", 12.0, 12.5, "SPEAKER_00"),  # in the spliced-out gap → dropped
        Word("this", 21.0, 21.4, "SPEAKER_01"),  # 2nd segment → offset by the 4s first segment
    ]
    clip = ClipItem(
        start_s=5.0, end_s=24.0, duration_s=8.0, hook="H",
        clip_type="compound", segments=[[5.0, 9.0], [20.0, 24.0]], text="The this",
    )
    out = clip_transcript_dict(clip, Transcript(words=words))
    assert out["duration_s"] == 8.0
    assert out["source_segments"] == [[5.0, 9.0], [20.0, 24.0]]
    assert [(w["text"], w["start_s"], w["speaker"]) for w in out["words"]] == [
        ("The", 1.0, "Person A"),
        ("this", 5.0, "Person B"),  # 21.0 - 20.0 + 4.0 (first segment length)
    ]


def test_unique_path_avoids_collisions(tmp_path: Path) -> None:
    first = unique_path(tmp_path, "My Clip")
    assert first == tmp_path / "My Clip.mov"
    first.write_bytes(b"")  # occupy it
    second = unique_path(tmp_path, "My Clip")
    assert second == tmp_path / "My Clip (2).mov"
    second.write_bytes(b"")
    assert unique_path(tmp_path, "My Clip") == tmp_path / "My Clip (3).mov"
