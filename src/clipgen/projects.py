"""File-based project store — no SQLite.

A **project** is one ingested video and its resulting clips, materialised as a directory under
the workspace root (``data/<project_id>/``) holding ``meta.json``, ``video.mp4``, ``audio.wav``,
``transcript.json`` and ``clips.json``. This module owns the lightweight ``meta.json`` row and
listing/lookup/deletion; the heavy artifacts are written by the ingest/transcribe/judge stages.

Listing is a directory scan — cheap for a single-user local tool and there is no schema to
migrate. All functions here are thin filesystem I/O around :class:`ProjectMeta`.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field

from clipgen.paths import Workspace

# Lifecycle of a project, mirroring the three pipeline stages plus terminal states.
Status = str  # "queued" | "ingesting" | "transcribing" | "judging" | "done" | "error"


@dataclass
class ProjectMeta:
    """The small, always-present description of a project (its ``meta.json`` row).

    ``chapters`` is the video's own chapter list (``[{start_s, title}, …]``) when the source
    provides one (YouTube, etc.); empty for a local file or an unchaptered video. The judge uses
    it as a segmentation prior — see :func:`clipgen.lines.format_transcript`.
    """

    id: str
    title: str = ""
    source_url: str = ""  # empty for a locally-uploaded file
    duration_s: float = 0.0
    status: Status = "queued"
    error: str = ""
    chapters: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ProjectMeta:
        return cls(
            id=str(d["id"]),
            title=d.get("title", ""),
            source_url=d.get("source_url", ""),
            duration_s=float(d.get("duration_s") or 0.0),
            status=d.get("status", "queued"),
            error=d.get("error", ""),
            chapters=list(d.get("chapters") or []),
            created_at=float(d.get("created_at") or time.time()),
        )


def new_project_id() -> str:
    """A fresh, filesystem-safe project id."""
    return uuid.uuid4().hex[:12]


def save_meta(workspace: Workspace, meta: ProjectMeta) -> ProjectMeta:
    """Persist a project's ``meta.json`` (creating its directory)."""
    workspace.ensure_project_dir(meta.id)
    workspace.meta_path(meta.id).write_text(json.dumps(meta.to_dict(), indent=2))
    return meta


def load_meta(workspace: Workspace, project_id: str) -> ProjectMeta | None:
    """Load a project's ``meta.json``, or ``None`` if it doesn't exist."""
    path = workspace.meta_path(project_id)
    if not path.is_file():
        return None
    return ProjectMeta.from_dict(json.loads(path.read_text()))


def update_meta(workspace: Workspace, project_id: str, **fields: object) -> ProjectMeta | None:
    """Patch specific fields on a project's ``meta.json`` and return the updated row.

    A no-op (returns ``None``) if the project doesn't exist — the caller decides whether that's
    an error (e.g. a job whose project was deleted mid-run).
    """
    meta = load_meta(workspace, project_id)
    if meta is None:
        return None
    for key, value in fields.items():
        setattr(meta, key, value)
    return save_meta(workspace, meta)


def _n_clips(workspace: Workspace, project_id: str) -> int:
    """How many clips have been grounded for a project (0 if none yet)."""
    path = workspace.clips_path(project_id)
    if not path.is_file():
        return 0
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError):
        return 0
    return len(data.get("clips", [])) if isinstance(data, dict) else 0


def list_projects(workspace: Workspace) -> list[dict]:
    """Every project as a summary dict (meta + ``n_clips`` + ``has_video``), newest first."""
    root = workspace.root
    if not root.is_dir():
        return []
    out: list[dict] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        meta = load_meta(workspace, child.name)
        if meta is None:
            continue
        summary = meta.to_dict()
        summary["n_clips"] = _n_clips(workspace, meta.id)
        summary["has_video"] = workspace.video_path(meta.id).is_file()
        out.append(summary)
    out.sort(key=lambda d: d.get("created_at", 0.0), reverse=True)
    return out


def delete_project(workspace: Workspace, project_id: str) -> bool:
    """Remove a project's directory entirely. Returns ``False`` if it didn't exist."""
    d = workspace.project_dir(project_id)
    if not d.is_dir():
        return False
    shutil.rmtree(d, ignore_errors=True)
    return True
