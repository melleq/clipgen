"""Progress events and replayable in-memory delivery for the web UI.

The server is a local, single-user process, so progress does not need durable queue
infrastructure. ``JobBus`` keeps the last event for each job and fan-outs new events to active
SSE subscribers; reconnecting clients receive the last known state before the live stream
continues.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Literal

Stage = Literal["ingest", "transcribe", "judge"]
Status = Literal["queued", "running", "done", "error"]


@dataclass(frozen=True)
class ProgressEvent:
    """One pipeline progress update emitted by a background job."""

    job_id: str
    stage: Stage
    status: Status
    message: str = ""
    progress: float | None = None
    detail: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.progress is not None and not 0.0 <= self.progress <= 1.0:
            raise ValueError("progress must be between 0.0 and 1.0")

    def to_dict(self) -> dict:
        """JSON-ready event payload for REST/SSE responses."""
        return {
            "job_id": self.job_id,
            "stage": self.stage,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "detail": dict(self.detail),
        }


class JobBus:
    """Thread-safe pub/sub bus with last-state replay per job."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last: dict[str, ProgressEvent] = {}
        self._subscribers: dict[str, list[queue.Queue[ProgressEvent]]] = {}

    def publish(self, event: ProgressEvent) -> None:
        """Record ``event`` as the latest job state and deliver it to subscribers."""
        with self._lock:
            self._last[event.job_id] = event
            subscribers = list(self._subscribers.get(event.job_id, []))
        for subscriber in subscribers:
            subscriber.put(event)

    def last(self, job_id: str) -> ProgressEvent | None:
        """Return the most recent event for ``job_id``, if any."""
        with self._lock:
            return self._last.get(job_id)

    def subscribe(self, job_id: str) -> tuple[queue.Queue[ProgressEvent], ProgressEvent | None]:
        """Create a subscriber queue and return it with the replay event, if present."""
        subscriber: queue.Queue[ProgressEvent] = queue.Queue()
        with self._lock:
            self._subscribers.setdefault(job_id, []).append(subscriber)
            last = self._last.get(job_id)
        return subscriber, last

    def unsubscribe(self, job_id: str, subscriber: queue.Queue[ProgressEvent]) -> None:
        """Remove a subscriber queue created by ``subscribe``."""
        with self._lock:
            subscribers = self._subscribers.get(job_id)
            if not subscribers:
                return
            try:
                subscribers.remove(subscriber)
            except ValueError:
                return
            if not subscribers:
                del self._subscribers[job_id]
