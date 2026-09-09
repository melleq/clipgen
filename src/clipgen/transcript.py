"""Transcript contracts: word-level, optionally speaker-labeled.

The word-level transcript is the backbone of the pipeline — clip boundaries, line grouping,
and speaker handling all derive from it. These are pure data shapes with no ASR/ML dependency;
the transcription stage populates them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Word:
    """A single transcribed word with timing and optional speaker label.

    Times are seconds from the start of the source video.
    """

    text: str
    start_s: float
    end_s: float
    speaker: str | None = None

    def to_dict(self) -> dict:
        d: dict = {"text": self.text, "start_s": self.start_s, "end_s": self.end_s}
        if self.speaker is not None:
            d["speaker"] = self.speaker
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Word:
        return cls(
            text=d["text"],
            start_s=float(d["start_s"]),
            end_s=float(d["end_s"]),
            speaker=d.get("speaker"),
        )


@dataclass(frozen=True)
class SpeakerTurn:
    """A contiguous span attributed to one speaker."""

    speaker: str | None
    start_s: float
    end_s: float


@dataclass
class Transcript:
    """An ordered sequence of words spanning a source video."""

    words: list[Word] = field(default_factory=list)

    @property
    def speakers(self) -> list[str]:
        """Distinct speaker labels, in first-appearance order."""
        seen: list[str] = []
        for w in self.words:
            if w.speaker is not None and w.speaker not in seen:
                seen.append(w.speaker)
        return seen

    def words_between(self, start_s: float, end_s: float) -> list[Word]:
        """Words whose timing overlaps the ``[start_s, end_s]`` window."""
        return [w for w in self.words if w.end_s >= start_s and w.start_s <= end_s]

    def speaker_turns(self) -> list[SpeakerTurn]:
        """Collapse consecutive same-speaker words into turns."""
        turns: list[SpeakerTurn] = []
        for w in self.words:
            if turns and turns[-1].speaker == w.speaker:
                last = turns[-1]
                turns[-1] = SpeakerTurn(last.speaker, last.start_s, w.end_s)
            else:
                turns.append(SpeakerTurn(w.speaker, w.start_s, w.end_s))
        return turns

    def to_dict(self) -> dict:
        return {"words": [w.to_dict() for w in self.words]}

    @classmethod
    def from_dict(cls, d: dict) -> Transcript:
        return cls(words=[Word.from_dict(w) for w in d.get("words", [])])
