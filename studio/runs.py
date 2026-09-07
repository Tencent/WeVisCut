"""Run registry, event journal and WebSocket fan-out.

Holds one record per inference run: its status, its event log, and the latest
multi-track progress snapshot. The event log exists so a browser that connects
late — or reloads mid-run — can replay history and land in the correct state
instead of showing an empty timeline for a run that is already half done.

The same log is journalled to disk under ``<workspace>/runs/<run_id>/``. Without
it the whole narrative lived in this process only, so a restart — or just
coming back tomorrow — left the user staring at an empty column for a project
that had clearly been made. ``rehydrate()`` reloads those journals at startup
and ``replay()`` falls back to them when a run is no longer in memory.

Progress snapshots are throttled: the agent emits one after every tool call, and
a long cut would otherwise push hundreds of full track payloads at the browser.
Status transitions and errors are never throttled.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("aicut_studio.runs")

EVENTS_FILENAME = "events.jsonl"
META_FILENAME = "meta.json"
# Journals are read back in full on replay, so cap what one run can grow to.
MAX_JOURNAL_EVENTS = 8000

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
    # Who started it. Kept on the run itself because a run has an owner before
    # it has a project directory to look the owner up from.
    owner: str = ""

    task: asyncio.Task | None = None
    orchestrator: Any | None = None

    # Journal state. `restored` runs came back from disk and have no live task,
    # so their elapsed time is whatever was recorded, not a running clock.
    journal_dir: Path | None = None
    journal_count: int = 0
    restored: bool = False
    elapsed_recorded: float = 0.0

    @property
    def elapsed(self) -> float:
        if self.restored:
            return self.elapsed_recorded
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
            "owner": self.owner,
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
                 raw_buffer: int = 400, journal_root: str | Path | None = None) -> None:
        self.runs: dict[str, Run] = {}
        self.sockets: dict[str, list[Any]] = {}
        self.max_events = max_events
        self.throttle = max(0.0, throttle_ms / 1000.0)
        self.raw_buffer = raw_buffer
        self._last_progress_push: dict[str, float] = {}
        self.journal_root = Path(journal_root).expanduser() if journal_root else None

    # ------------------------------------------------------------------
    def create(self, input_text: str, run_id: str = "") -> Run:
        rid = run_id or f"run_{datetime.now().strftime('%m%d_%H%M%S')}_{uuid.uuid4().hex[:5]}"
        run = Run(id=rid, input_text=input_text, status="running")
        run.events = deque(maxlen=self.max_events)
        run.raw_outputs = deque(maxlen=self.raw_buffer)
        if self.journal_root is not None:
            run.journal_dir = self.journal_root / rid
            try:
                run.journal_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning("cannot create run journal %s: %s", run.journal_dir, exc)
                run.journal_dir = None
        self.runs[rid] = run
        self.save_meta(run)
        return run

    # ---- journal -----------------------------------------------------
    def save_meta(self, run: Run) -> None:
        """Persist the run summary. Cheap enough to call on every transition."""
        if run.journal_dir is None:
            return
        payload = {
            **run.summary(),
            "input_text": run.input_text,
            "progress": run.progress,
            "result": run.result,
            "errors": run.errors,
            "elapsed_recorded": run.elapsed,
        }
        try:
            tmp = run.journal_dir / (META_FILENAME + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(run.journal_dir / META_FILENAME)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("cannot write run meta for %s: %s", run.id, exc)

    def _journal(self, run: Run, event: dict[str, Any]) -> None:
        if run.journal_dir is None or run.journal_count >= MAX_JOURNAL_EVENTS:
            return
        try:
            with (run.journal_dir / EVENTS_FILENAME).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            run.journal_count += 1
        except (OSError, TypeError, ValueError) as exc:
            # Losing the journal must never fail a run that is otherwise fine.
            logger.warning("cannot journal event for %s: %s", run.id, exc)
            run.journal_dir = None

    def read_journal(self, run_id: str) -> list[dict[str, Any]]:
        """Events for a run that is no longer in memory."""
        if self.journal_root is None:
            return []
        path = self.journal_root / run_id / EVENTS_FILENAME
        if not path.is_file():
            return []
        out: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue          # a torn last line after a hard kill
        except OSError as exc:
            logger.warning("cannot read run journal %s: %s", path, exc)
        return out

    def rehydrate(self, limit: int = 60) -> int:
        """Reload finished runs from disk so history survives a restart."""
        if self.journal_root is None or not self.journal_root.is_dir():
            return 0
        metas: list[tuple[float, Path]] = []
        for d in self.journal_root.iterdir():
            meta = d / META_FILENAME
            if meta.is_file():
                try:
                    metas.append((meta.stat().st_mtime, meta))
                except OSError:
                    continue
        metas.sort(reverse=True)

        loaded = 0
        for _, meta in metas[:limit]:
            try:
                payload = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            rid = str(payload.get("id") or meta.parent.name)
            if not rid or rid in self.runs:
                continue
            status = str(payload.get("status") or "")
            run = Run(
                id=rid,
                input_text=str(payload.get("input_text") or ""),
                # An "active" status on disk means the process died mid-run;
                # nothing is driving it now, so present it as interrupted.
                status=status if status in TERMINAL_STATUSES else "cancelled",
                started_at=str(payload.get("started_at") or ""),
                finished_at=str(payload.get("finished_at") or ""),
            )
            run.progress = payload.get("progress") or {}
            run.result = payload.get("result") or {}
            run.errors = list(payload.get("errors") or [])
            run.project_dir = str(payload.get("project_dir") or "")
            run.owner = str(payload.get("owner") or "")
            run.journal_dir = meta.parent
            run.restored = True
            try:
                run.elapsed_recorded = float(payload.get("elapsed_recorded") or 0.0)
            except (TypeError, ValueError):
                run.elapsed_recorded = 0.0
            self.runs[rid] = run
            loaded += 1
        if loaded:
            logger.info("restored %d run(s) from %s", loaded, self.journal_root)
        return loaded

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
        journal = run.journal_dir
        self.runs.pop(run_id, None)
        self.sockets.pop(run_id, None)
        self._last_progress_push.pop(run_id, None)
        if journal is not None:
            # Leave the journal behind and the run reappears on next restart.
            shutil.rmtree(journal, ignore_errors=True)
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
        # A restored run has its summary in memory but its narrative on disk.
        events = list(run.events) or (self.read_journal(run_id) if run.restored else [])
        await websocket.send_json({"type": "replay_start", "event_count": len(events)})
        for event in events:
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
        if etype == "thinking_delta":
            # Live typing only: hundreds per step, and the settled `thinking`
            # event already carries the full text for replay.
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
        self._journal(run, event)
        # The narrative is only useful next to the outcome, so keep the summary
        # in step with it at the transitions that change what the summary says.
        if etype in ("run_started", "run_complete", "error", "pipeline_complete"):
            self.save_meta(run)

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
