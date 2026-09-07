"""Run a declared worklist across a pool of machines, observably.

``batch.BatchRunner`` globs a directory and points every subprocess at one
endpoint. A 9k-episode corpus annotated for publication needs more than that:

- **a fixed roster.** Episodes come from a :mod:`~longvideo_polar.curation`
  worklist, so the set that ran is the set that was declared, and folder names
  carry the source video's identity.
- **many machines.** Each episode is leased a slot on the least-loaded healthy
  endpoint and released on exit, so concurrency is the pool's capacity rather
  than a fixed ``workers`` number, and machines can be added or drained mid-run.
- **progress that means something.** A rollout takes tens of minutes; "running"
  is not a useful status for that long. Each live episode reports the step and
  tool it is on, read from the pipeline's own ``agent_trace.jsonl``.
- **an ETA that adapts.** Remaining time is projected from a moving average of
  recent completions against current concurrency, not from a nominal rate.
- **interruption.** A batch spanning days must survive being paused, stopped and
  resumed, and must never re-run an episode whose trajectories are already on
  disk.

The status file is the contract with the UI: one JSON document, rewritten
atomically, holding everything the front end renders.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from longvideo_polar.config import deep_merge
from longvideo_polar.curation import WorkItem, read_worklist
from longvideo_polar.endpoints import Endpoint, EndpointPool

logger = logging.getLogger(__name__)

STATUS_FILENAME = "campaign_status.json"
EPISODES_DIRNAME = "episodes"
STATE_DIRNAME = "_state"

# How often live episodes are re-inspected for their current step. Each check
# touches a shared filesystem, so this trades status latency against IO load.
PROGRESS_POLL_SECONDS = 10.0

# Completions averaged for the ETA. Long enough to absorb one slow outlier,
# short enough to react when machines are added or removed.
ETA_WINDOW = 40


@dataclass
class EpisodeState:
    """One episode's progress through the campaign."""

    index: int
    episode_id: str
    asset_id: str
    bv_id: str
    script_path: str
    output_dir: str
    script_chars: int = 0
    duration_sec: float = 0.0

    status: str = "pending"  # pending | running | done | failed | skipped
    endpoint: str = ""
    step: int = 0
    max_steps: int = 0
    stage: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    elapsed_seconds: float = 0.0
    error: str = ""
    final_video_path: str = ""
    llm_calls: int = 0
    attempts: int = 0

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["progress_pct"] = (
            round(100.0 * self.step / self.max_steps, 1) if self.max_steps else 0.0
        )
        if self.status == "running" and self.started_at:
            payload["elapsed_seconds"] = round(time.time() - self.started_at, 1)
        return payload


@dataclass
class CampaignSettings:
    """Everything the UI can set before pressing start."""

    #: ``text_only`` harvests trajectories with media tools off (fast, no 成片);
    #: ``full_media`` produces real videos; ``trajectory_plus_compose`` is the
    #: middle ground that composes a timeline but skips rendering.
    mode: str = "text_only"
    profile: str = "all_qwen"
    max_steps: int = 48
    #: Rollouts per machine; multiplied by the number of healthy machines.
    capacity_per_endpoint: int = 4
    resume: bool = True
    retry_failed: bool = False
    max_attempts: int = 1
    #: 0 uses the whole worklist.
    limit: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Tool switches per mode. ``text_only`` still exercises every LLM decision — the
#: orchestrator, generator and critic all run — it only skips the media work whose
#: cost dominates wall time.
MODE_TOOL_SWITCHES: dict[str, dict[str, bool]] = {
    "text_only": {
        "tts": False,
        "visual_retrieval": False,
        "bgm_search": False,
        "timeline_compose": False,
        "hyperframes_render": False,
        "final_stitch": False,
    },
    "trajectory_plus_compose": {
        "tts": True,
        "visual_retrieval": True,
        "bgm_search": True,
        "timeline_compose": True,
        "hyperframes_render": False,
        "final_stitch": False,
    },
    "full_media": {
        "tts": True,
        "visual_retrieval": True,
        "bgm_search": True,
        "timeline_compose": True,
        "hyperframes_render": True,
        "final_stitch": True,
    },
}

MODE_LABELS_ZH: dict[str, str] = {
    "text_only": "纯文本轨迹（不出片，最快）",
    "trajectory_plus_compose": "轨迹 + 时间线合成（不渲染）",
    "full_media": "全量出片（TTS + 检索 + 渲染 + 拼接）",
}


class CampaignRunner:
    """Drive a worklist across an endpoint pool with live, resumable status."""

    def __init__(
        self,
        *,
        worklist_path: str | Path,
        output_root: str | Path,
        config_path: str | Path,
        pool: EndpointPool,
        settings: CampaignSettings | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.worklist_path = Path(str(worklist_path)).expanduser()
        self.output_root = Path(str(output_root)).expanduser()
        self.config_path = str(Path(str(config_path)).expanduser().resolve())
        self.pool = pool
        self.settings = settings or CampaignSettings()
        self.on_update = on_update

        self.episodes_dir = self.output_root / EPISODES_DIRNAME
        self.state_dir = self.output_root / STATE_DIRNAME
        self.status_path = self.state_dir / STATUS_FILENAME

        self.items: list[WorkItem] = []
        self.header: dict[str, Any] = {}
        self.states: dict[str, EpisodeState] = {}
        #: Written by :meth:`prepare_session_config`; what workers actually load.
        self.session_config_path = self.state_dir / "session_config.yaml"

        self._lock = asyncio.Lock()
        self._pause = asyncio.Event()
        self._pause.set()  # set == running
        self._stop = False
        self._started_at = 0.0
        self._finished_at = 0.0
        self._phase = "idle"  # idle | running | paused | stopping | finished
        self._durations: deque[float] = deque(maxlen=ETA_WINDOW)
        self._project_dirs: dict[str, Path] = {}
        self._tasks: set[asyncio.Task] = set()

    # -- lifecycle -------------------------------------------------------

    def load(self) -> None:
        """Read the worklist and reconcile it with whatever is already on disk."""
        self.items, self.header = read_worklist(self.worklist_path)
        if self.settings.limit and self.settings.limit > 0:
            self.items = self.items[: self.settings.limit]
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.states = {}
        for item in self.items:
            output_dir = self.episodes_dir / item.episode_id
            self.states[item.episode_id] = EpisodeState(
                index=item.index,
                episode_id=item.episode_id,
                asset_id=item.asset_id,
                bv_id=item.bv_id,
                script_path=item.script_path,
                output_dir=str(output_dir),
                script_chars=item.script_chars,
                duration_sec=item.duration_sec,
                max_steps=self.settings.max_steps,
            )
        if self.settings.resume:
            self._reconcile_completed()
        self.prepare_session_config()
        self._write_status()

    def prepare_session_config(self) -> Path:
        """Freeze the run's settings into a config file the workers load.

        Mode, step budget, output root and the decoration policy are written to
        disk rather than passed as environment variables: the tool switches are
        booleans, and ``"0"`` read back from the environment is a truthy string.
        Freezing them also means the exact configuration a batch ran under is
        recoverable afterwards, which a published corpus needs.
        """
        base = yaml.safe_load(Path(self.config_path).read_text(encoding="utf-8")) or {}
        switches = MODE_TOOL_SWITCHES.get(self.settings.mode)
        if switches is None:
            raise ValueError(
                f"unknown mode {self.settings.mode!r}; "
                f"expected one of {sorted(MODE_TOOL_SWITCHES)}"
            )

        session = deep_merge(
            base,
            {
                "tool_switches": dict(switches),
                "agent": {"max_steps": self.settings.max_steps},
                "recording": {"output_root": str(self.episodes_dir)},
                "annotation_profile": self.settings.profile,
                "decoration": {
                    "ledger_path": str(self.state_dir / "template_usage.db"),
                    "slot_cache_path": str(self.state_dir / "slot_schema_cache.json"),
                },
                "_campaign": {
                    "mode": self.settings.mode,
                    "worklist": str(self.worklist_path),
                    "base_config": self.config_path,
                    "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            },
        )
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.session_config_path.write_text(
            yaml.safe_dump(session, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        return self.session_config_path

    def _reconcile_completed(self) -> None:
        """Mark episodes whose trajectories already exist so they are not re-run.

        A rollout counts as complete when its manifest exists and it captured
        trajectories. In ``full_media`` mode delivering the video is also required,
        because a media run that stopped short is exactly the one worth retrying.
        """
        require_video = self.settings.mode == "full_media"
        recovered = 0
        for state in self.states.values():
            manifest_path = Path(state.output_dir) / "episode.json"
            manifest = _read_json(manifest_path)
            if not manifest:
                continue
            has_video = bool(manifest.get("final_video_exists"))
            roles = manifest.get("roles") or {}
            calls = sum(int((info or {}).get("llm_calls", 0)) for info in roles.values())
            if require_video and not has_video:
                continue
            if calls <= 0:
                continue
            state.status = "skipped"
            state.llm_calls = calls
            state.final_video_path = str(manifest.get("final_video_path") or "")
            state.elapsed_seconds = float(manifest.get("elapsed_seconds") or 0.0)
            state.step = int(manifest.get("orchestrator_steps") or 0)
            recovered += 1
        if recovered:
            logger.info("resume: %d episodes already complete", recovered)

    async def run(self) -> dict[str, Any]:
        """Execute the campaign until finished or stopped."""
        if not self.items:
            self.load()
        self._started_at = time.time()
        self._phase = "running"
        self._stop = False
        self._write_status()

        progress_task = asyncio.create_task(self._progress_loop())
        try:
            await self._dispatch_loop()
        finally:
            progress_task.cancel()
            self._finished_at = time.time()
            self._phase = "stopped" if self._stop else "finished"
            self._write_status()

        summary = self.summary()
        logger.info(
            "campaign %s: %d done, %d failed, %d skipped of %d",
            self._phase, summary["done"], summary["failed"],
            summary["skipped"], summary["total"],
        )
        return summary

    async def _dispatch_loop(self) -> None:
        """Lease a machine per pending episode, spawning as capacity frees up."""
        pending = [
            state for state in self._ordered_states()
            if state.status in {"pending", "failed"}
            and not (state.status == "failed" and not self.settings.retry_failed)
        ]
        queue = deque(pending)
        running: set[asyncio.Task] = set()

        while queue or running:
            if self._stop:
                break
            await self._pause.wait()

            endpoint = self.pool.acquire() if queue else None
            if endpoint is not None:
                state = queue.popleft()
                task = asyncio.create_task(self._run_episode(state, endpoint))
                running.add(task)
                task.add_done_callback(running.discard)
                continue

            if not running:
                if not queue:
                    break
                # Nothing in flight and no machine available: the pool is down.
                # Re-probe rather than spin, so a restarted vLLM is picked up.
                logger.warning("no healthy endpoint available; re-probing in 30s")
                await asyncio.sleep(30)
                await asyncio.to_thread(self.pool.probe_all)
                self._write_status()
                continue

            # Wait for a slot to free up.
            await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)

        if running:
            await asyncio.gather(*running, return_exceptions=True)

    def _ordered_states(self) -> list[EpisodeState]:
        return [self.states[item.episode_id] for item in self.items]

    # -- one episode -----------------------------------------------------

    async def _run_episode(self, state: EpisodeState, endpoint: Endpoint) -> None:
        state.status = "running"
        state.endpoint = endpoint.base_url
        state.started_at = time.time()
        state.attempts += 1
        state.step = 0
        state.stage = "starting"
        state.error = ""
        self._write_status()

        output_dir = Path(state.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        code, stderr_tail = await self._spawn(state, endpoint)
        state.elapsed_seconds = round(time.monotonic() - started, 1)
        state.finished_at = time.time()

        manifest = _read_json(output_dir / "episode.json")
        roles = manifest.get("roles") or {}
        state.llm_calls = sum(int((info or {}).get("llm_calls", 0)) for info in roles.values())
        state.final_video_path = str(manifest.get("final_video_path") or "")
        state.step = int(manifest.get("orchestrator_steps") or state.step)

        # A trajectory is the deliverable in text modes, so an episode that
        # recorded calls succeeded even if the pipeline reported step-limit errors.
        # Requiring a clean exit code here would fail most of the corpus and hide
        # the runs that genuinely produced nothing.
        produced = bool(manifest) and state.llm_calls > 0
        if self.settings.mode == "full_media":
            produced = produced and bool(manifest.get("final_video_exists"))

        if produced:
            state.status = "done"
            self._durations.append(state.elapsed_seconds)
        else:
            state.status = "failed"
            reasons = [str(item) for item in manifest.get("errors", []) if str(item)]
            state.error = ("; ".join(reasons) or stderr_tail or f"exit code {code}")[:600]
            # A transport failure implicates the machine, not the script.
            if not manifest and _looks_like_endpoint_failure(stderr_tail):
                self.pool.mark_unhealthy(endpoint.base_url, stderr_tail[:200])

        self.pool.release(endpoint.base_url, success=state.status == "done")
        self._write_status()
        logger.info(
            "[%d/%d] %s %s on %s (%.0fs, %d calls)",
            state.index, len(self.items), state.status, state.episode_id,
            endpoint.display_name(), state.elapsed_seconds, state.llm_calls,
        )

    async def _spawn(self, state: EpisodeState, endpoint: Endpoint) -> tuple[int, str]:
        """Run one rollout as an isolated ``longvideo-polar run`` subprocess."""
        command = [
            sys.executable, "-m", "longvideo_polar.cli", "run",
            "--config", str(self.session_config_path),
            "--script-file", state.script_path,
            "--output", state.output_dir,
            "--episode-id", state.episode_id,
            "--base-url", endpoint.base_url,
            "--max-steps", str(self.settings.max_steps),
        ]
        if endpoint.model:
            command.extend(["--model", endpoint.model])
        if self.settings.profile:
            command.extend(["--profile", self.settings.profile])

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["OPENAI_BASE_URL"] = endpoint.base_url
        env["OPENAI_API_KEY"] = endpoint.api_key or "EMPTY"
        # The corporate proxy answers IDC addresses with 403, which surfaces as a
        # dead model rather than a blocked request. Exempt every pool host.
        env["no_proxy"] = env["NO_PROXY"] = _no_proxy_value(self.pool, env.get("no_proxy", ""))
        env.update(_import_tuning_env(env))

        log_path = Path(state.output_dir) / "worker.log"
        with log_path.open("w", encoding="utf-8") as handle:
            process = await asyncio.create_subprocess_exec(
                *command, stdout=handle, stderr=asyncio.subprocess.PIPE, env=env
            )
            _, stderr = await process.communicate()

        stderr_text = (stderr or b"").decode("utf-8", errors="replace")
        if stderr_text:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("\n--- stderr ---\n")
                handle.write(stderr_text)
        return process.returncode or 0, stderr_text.strip()[-800:]

    # -- live progress ---------------------------------------------------

    async def _progress_loop(self) -> None:
        """Refresh step/stage for running episodes and rewrite the status file."""
        while True:
            try:
                await asyncio.sleep(PROGRESS_POLL_SECONDS)
                running = [s for s in self.states.values() if s.status == "running"]
                if running:
                    await asyncio.to_thread(self._refresh_running, running)
                self._write_status()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - status must never kill a batch
                logger.debug("progress refresh skipped: %s", exc)

    def _refresh_running(self, running: list[EpisodeState]) -> None:
        for state in running:
            trace = self._agent_trace_path(state)
            if trace is None:
                continue
            step, tool = _tail_step(trace)
            if step:
                state.step = step
                state.stage = tool

    def _agent_trace_path(self, state: EpisodeState) -> Path | None:
        """Locate the pipeline's trace, caching the project dir it lives in.

        The pipeline writes into ``<output>/<project_id>/`` with a project id it
        chooses at runtime, so the first lookup has to search; caching it keeps
        later polls to a single stat on a slow shared filesystem.
        """
        cached = self._project_dirs.get(state.episode_id)
        if cached is not None and cached.is_file():
            return cached
        root = Path(state.output_dir)
        for candidate in (root / "agent_trace.jsonl", *root.glob("*/agent_trace.jsonl")):
            if candidate.is_file():
                self._project_dirs[state.episode_id] = candidate
                return candidate
        return None

    # -- control ---------------------------------------------------------

    def pause(self) -> None:
        self._pause.clear()
        self._phase = "paused"
        self._write_status()

    def resume(self) -> None:
        self._pause.set()
        self._phase = "running"
        self._write_status()

    def stop(self) -> None:
        """Stop dispatching. Episodes already in flight are allowed to finish."""
        self._stop = True
        self._pause.set()
        self._phase = "stopping"
        self._write_status()

    @property
    def phase(self) -> str:
        return self._phase

    # -- reporting -------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        counts = {"pending": 0, "running": 0, "done": 0, "failed": 0, "skipped": 0}
        for state in self.states.values():
            counts[state.status] = counts.get(state.status, 0) + 1
        total = len(self.states)
        settled = counts["done"] + counts["failed"] + counts["skipped"]
        return {
            "phase": self._phase,
            "total": total,
            "settled": settled,
            "progress_pct": round(100.0 * settled / total, 2) if total else 0.0,
            **counts,
            "eta_seconds": self._eta_seconds(),
            "mean_episode_seconds": (
                round(sum(self._durations) / len(self._durations), 1) if self._durations else 0.0
            ),
            "throughput_per_hour": self._throughput_per_hour(),
            "started_at": self._started_at,
            "finished_at": self._finished_at,
            "elapsed_seconds": round(
                (self._finished_at or time.time()) - self._started_at, 1
            ) if self._started_at else 0.0,
        }

    def _throughput_per_hour(self) -> float:
        if not self._started_at:
            return 0.0
        done = sum(1 for s in self.states.values() if s.status == "done")
        hours = max(1e-6, ((self._finished_at or time.time()) - self._started_at) / 3600.0)
        return round(done / hours, 2)

    def _eta_seconds(self) -> float:
        """Project remaining wall time from recent completions and live width.

        Uses observed concurrency rather than nominal capacity, so an ETA computed
        while half the pool is unhealthy reflects the machines actually working.
        """
        remaining = sum(
            1 for s in self.states.values() if s.status in {"pending", "running"}
        )
        if not remaining or not self._durations:
            return 0.0
        mean_seconds = sum(self._durations) / len(self._durations)
        concurrency = max(
            1,
            sum(1 for s in self.states.values() if s.status == "running")
            or self.pool.total_capacity,
        )
        return round(remaining * mean_seconds / concurrency, 0)

    def status_document(self) -> dict[str, Any]:
        """The single document the UI renders."""
        return {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "worklist_path": str(self.worklist_path),
            "output_root": str(self.output_root),
            "settings": self.settings.as_dict(),
            "mode_label": MODE_LABELS_ZH.get(self.settings.mode, self.settings.mode),
            "selection": self.header.get("selection", {}),
            "summary": self.summary(),
            "pool": self.pool.describe(),
            "episodes": [state.as_dict() for state in self._ordered_states()],
        }

    def _write_status(self) -> None:
        document = self.status_document()
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            temp_path = self.status_path.with_suffix(".tmp")
            temp_path.write_text(
                json.dumps(document, ensure_ascii=False), encoding="utf-8"
            )
            temp_path.replace(self.status_path)
        except OSError as exc:
            logger.debug("status write failed: %s", exc)
        if self.on_update is not None:
            try:
                self.on_update(document)
            except Exception as exc:  # noqa: BLE001
                logger.debug("status callback failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _tail_step(path: Path, *, tail_bytes: int = 65536) -> tuple[int, str]:
    """Read the latest ``step``/``tool_name`` pair from a growing trace file.

    Only the tail is read: these files reach megabytes, and the status loop
    revisits them every few seconds.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > tail_bytes:
                handle.seek(size - tail_bytes)
                handle.readline()  # discard the partial line
            chunk = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return 0, ""

    step, tool = 0, ""
    for line in chunk.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or "step" not in row:
            continue
        try:
            value = int(row["step"])
        except (TypeError, ValueError):
            continue
        if value >= step:
            step, tool = value, str(row.get("tool_name") or row.get("tool") or "")
    return step, tool


def _looks_like_endpoint_failure(stderr_tail: str) -> bool:
    """Whether a failure points at the machine rather than the script."""
    text = (stderr_tail or "").lower()
    return any(
        marker in text
        for marker in (
            "connecterror", "connecttimeout", "readtimeout", "connection refused",
            "forbidden host", "503", "502", "all connection attempts failed",
        )
    )


def _no_proxy_value(pool: EndpointPool, existing: str) -> str:
    """Add every pool host to ``no_proxy``, preserving what was already there."""
    from urllib.parse import urlparse

    hosts = {"127.0.0.1", "localhost", "0.0.0.0", "::1"}
    hosts.update(
        host for host in (urlparse(ep.base_url).hostname for ep in pool.all()) if host
    )
    hosts.update(part.strip() for part in (existing or "").split(",") if part.strip())
    return ",".join(sorted(hosts))


# Metadata latency on the shared mount runs ~1-2s per stat. A source path ahead of
# the stdlib makes the interpreter probe CephFS for every `import asyncio`-style
# lookup it does, which measured at 61s of pure import time per worker; moving the
# path behind the stdlib and parking bytecode on local disk brings that to ~18s.
_LOCAL_PYCACHE = Path("/dev/shm/longvideo_polar_pycache")


def _import_tuning_env(env: Mapping[str, str]) -> dict[str, str]:
    """Keep CephFS out of the stdlib import path and cache bytecode locally."""
    tuned: dict[str, str] = {}

    entries = [part for part in (env.get("PYTHONPATH", "").split(os.pathsep)) if part]
    shared, local = [], []
    for entry in entries:
        (shared if entry.startswith("/mnt/") else local).append(entry)
    if shared and local != entries:
        tuned["PYTHONPATH"] = os.pathsep.join([*local, *shared])

    if not env.get("PYTHONPYCACHEPREFIX"):
        try:
            _LOCAL_PYCACHE.mkdir(parents=True, exist_ok=True)
            tuned["PYTHONPYCACHEPREFIX"] = str(_LOCAL_PYCACHE)
        except OSError:  # tmpfs unavailable; the slower path still works
            pass
    return tuned
