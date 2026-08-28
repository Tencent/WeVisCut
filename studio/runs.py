"""Run registry and WebSocket fan-out.

Holds one record per inference run: its status, its event log, and the latest
multi-track progress snapshot. The event log exists so a browser that connects
late — or reloads mid-run — can replay history and land in the correct state
instead of showing an empty timeline for a run that is already half done.

Progress snapshots are throttled: the agent emits one after every tool call, and
a long cut would otherwise push hundreds of full track payloads at the browser.
Status transitions and errors are never throttled.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger("aicut_studio.runs")

TERMINAL_STATUSES = {"completed", "completed_with_errors", "cancelled", "error"}

# Events that must always reach the client, even under throttling.
CRITICAL_EVENTS = {
    "run_started", "run_complete", "run_paused", "run_resumed",
    "error", "pipeline_complete", "stage_update",
}


@dataclass
class Run:
    id: str
    input_text: str
    status: str = "queued"
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_monotonic: float = field(default_factory=time.monotonic)
    finished_at: str = ""

    events: deque = field(default_factory=lambda: deque(maxlen=5000))
    raw_outputs: deque = field(default_factory=lambda: deque(maxlen=400))
    progress: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    project_dir: str = ""

    task: asyncio.Task | None = None
    orchestrator: Any | None = None

    @property
    def elapsed(self) -> float:
        return round(time.monotonic() - self.started_monotonic, 2)

    @property
    def is_active(self) -> bool:
        return self.status not in TERMINAL_STATUSES

    def summary(self) -> dict[str, Any]:
        title = (self.input_text or "").strip().split("\n")[0][:60] or "(无标题)"
        return {
            "id": self.id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": self.elapsed,
            "title": title,
            "overall_pct": self.progress.get("overall_pct", 0),
            "project_dir": self.project_dir,
            "final_video_path": self.result.get("final_video_path", ""),
            "error_count": len(self.errors),
        }

    def detail(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "input_text": self.input_text,
            "progress": self.progress,
            "result": self.result,
            "errors": self.errors,
            "event_count": len(self.events),
        }


class RunStore:
    """In-memory run registry with per-run WebSocket subscribers."""

    def __init__(self, max_events: int = 5000, throttle_ms: int = 120,
                 raw_buffer: int = 400) -> None:
        self.runs: dict[str, Run] = {}
        self.sockets: dict[str, list[Any]] = {}
        self.max_events = max_events
        self.throttle = max(0.0, throttle_ms / 1000.0)
        self.raw_buffer = raw_buffer
        self._last_progress_push: dict[str, float] = {}

    # ------------------------------------------------------------------
    def create(self, input_text: str, run_id: str = "") -> Run:
        rid = run_id or f"run_{datetime.now().strftime('%m%d_%H%M%S')}_{uuid.uuid4().hex[:5]}"
        run = Run(id=rid, input_text=input_text, status="running")
        run.events = deque(maxlen=self.max_events)
        run.raw_outputs = deque(maxlen=self.raw_buffer)
        self.runs[rid] = run
        return run

    def get(self, run_id: str) -> Run | None:
        return self.runs.get(run_id)

    def list(self) -> list[dict[str, Any]]:
        return [
            r.summary()
            for r in sorted(self.runs.values(), key=lambda x: x.started_at, reverse=True)
        ]

    def active_count(self) -> int:
        return sum(1 for r in self.runs.values() if r.is_active)

    def delete(self, run_id: str) -> bool:
        run = self.runs.get(run_id)
        if run is None or run.is_active:
            return False
        self.runs.pop(run_id, None)
        self.sockets.pop(run_id, None)
        self._last_progress_push.pop(run_id, None)
        return True

    # ------------------------------------------------------------------
    def subscribe(self, run_id: str, websocket: Any) -> None:
        self.sockets.setdefault(run_id, []).append(websocket)

    def unsubscribe(self, run_id: str, websocket: Any) -> None:
        conns = self.sockets.get(run_id)
        if not conns:
            return
        self.sockets[run_id] = [ws for ws in conns if ws is not websocket]

    async def replay(self, run_id: str, websocket: Any) -> None:
        """Bring a newly connected client up to the run's current state."""
        run = self.runs.get(run_id)
        if run is None:
            return
        await websocket.send_json({"type": "replay_start", "event_count": len(run.events)})
        for event in list(run.events):
            try:
                await websocket.send_json(event)
            except Exception:
                return
        if run.progress:
            await websocket.send_json({"type": "progress", "progress": run.progress})
        for line in list(run.raw_outputs):
            await websocket.send_json({"type": "raw_output", **line})
        await websocket.send_json({"type": "replay_end"})
        if not run.is_active:
            await websocket.send_json({
                "type": "run_complete",
                "status": run.status,
                "elapsed_seconds": run.elapsed,
                "result": run.result,
                "progress": run.progress,
            })

    # ------------------------------------------------------------------
    async def emit(self, run_id: str, event: dict[str, Any]) -> None:
        """Record an event and fan it out to subscribers."""
        run = self.runs.get(run_id)
        etype = str(event.get("type", "") or "")

        if run is not None:
            self._record(run, etype, event)

        if etype == "progress" and self._should_throttle(run_id, etype):
            return

        for ws in list(self.sockets.get(run_id, [])):
            try:
                await ws.send_json(event)
            except Exception:
                self.unsubscribe(run_id, ws)

    def _record(self, run: Run, etype: str, event: dict[str, Any]) -> None:
        if etype == "progress":
            run.progress = event.get("progress", {}) or {}
            return
        if etype == "raw_output":
            run.raw_outputs.append({
                "text": event.get("text", ""),
                "step": event.get("step", 0),
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })
            return
        if etype == "error":
            msg = str(event.get("message", "") or "")
            if msg:
                run.errors.append(msg)
        if etype == "console_log":
            event.setdefault("timestamp", datetime.now().strftime("%H:%M:%S"))
        run.events.append(event)

    def _should_throttle(self, run_id: str, etype: str) -> bool:
        if etype in CRITICAL_EVENTS or self.throttle <= 0:
            return False
        now = time.monotonic()
        last = self._last_progress_push.get(run_id, 0.0)
        if now - last < self.throttle:
            return True
        self._last_progress_push[run_id] = now
        return False

    async def flush_progress(self, run_id: str) -> None:
        """Force-push the stored progress snapshot, bypassing throttling.

        Called at run end so the final state is never the one a throttle
        happened to drop.
        """
        run = self.runs.get(run_id)
        if run is None or not run.progress:
            return
        self._last_progress_push.pop(run_id, None)
        for ws in list(self.sockets.get(run_id, [])):
            try:
                await ws.send_json({"type": "progress", "progress": run.progress})
            except Exception:
                self.unsubscribe(run_id, ws)
