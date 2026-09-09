"""Workspace layout — the single owner of on-disk paths.

All project data lives under one workspace root (default ``./data``), one directory per
project (= one ingested video and its clips). No other module hardcodes paths; they go
through ``Workspace``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path("data")


@dataclass(frozen=True)
class Workspace:
    """Filesystem layout for clipgen's data."""

    root: Path = DEFAULT_ROOT

    @classmethod
    def default(cls) -> Workspace:
        return cls(DEFAULT_ROOT)

    def project_dir(self, project_id: str) -> Path:
        return self.root / project_id

    def video_path(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "video.mp4"

    def audio_path(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "audio.wav"

    def meta_path(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "meta.json"

    def transcript_path(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "transcript.json"

    def clips_path(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "clips.json"

    def judge_prompt_path(self, project_id: str) -> Path:
        """The exact prompt sent to the editorial judge (for review)."""
        return self.project_dir(project_id) / "judge_prompt.txt"

    def judge_response_path(self, project_id: str) -> Path:
        """The judge's raw reply (for re-parsing / retries without a new call)."""
        return self.project_dir(project_id) / "judge_response.json"

    def refine_prompt_path(self, project_id: str) -> Path:
        """The exact pass-B (verify/refine) prompt sent to the judge (for review)."""
        return self.project_dir(project_id) / "refine_prompt.txt"

    def refine_response_path(self, project_id: str) -> Path:
        """The pass-B verifier's raw reply (for re-parsing without a new call)."""
        return self.project_dir(project_id) / "refine_response.json"

    def judge_rejects_path(self, project_id: str) -> Path:
        """Candidates dropped by the verifier or grounding, with one-line reasons."""
        return self.project_dir(project_id) / "judge_rejects.json"

    def ensure_project_dir(self, project_id: str) -> Path:
        """Create and return the per-project directory."""
        d = self.project_dir(project_id)
        d.mkdir(parents=True, exist_ok=True)
        return d
