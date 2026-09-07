"""Batch annotation: many 口播稿 files in, many three-part trajectories out.

Each rollout runs in its own subprocess. That is deliberate rather than
convenient: AICutPipeline's agent loop temporarily monkey-patches
``pipeline.graph._get_llm`` and the optional render-split patch mutates module
globals, so two rollouts sharing one interpreter would either serialize on that
lock or contend over global state. Process isolation also means one crashed
rollout cannot take the batch down, which matters over hundreds of scripts.

The manager keeps a resumable status file, so re-running the same command picks up
where it left off, and aggregates the per-episode dumps into combined SFT/RL
shards at the end.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from longvideo_polar.config import LongVideoConfig
from longvideo_polar.roles import ALL_ROLES, Role

logger = logging.getLogger(__name__)

STATUS_FILENAME = "batch_status.json"


@dataclass
class BatchItem:
    """One script's progress through the batch."""

    index: int
    script_path: str
    episode_id: str
    output_dir: str
    status: str = "pending"  # pending | running | done | failed
    final_video_path: str = ""
    error: str = ""
    elapsed_seconds: float = 0.0
    started_at: str = ""
    finished_at: str = ""


@dataclass
class BatchReport:
    """Aggregate outcome of a batch run."""

    total: int = 0
    done: int = 0
    failed: int = 0
    skipped: int = 0
    items: list[BatchItem] = field(default_factory=list)
    aggregate_dir: str = ""

    def summary(self) -> str:
        return (
            f"batch: {self.done}/{self.total} succeeded, {self.failed} failed, "
            f"{self.skipped} skipped (resumed)"
        )


class BatchRunner:
    """Run many rollouts concurrently, each in an isolated subprocess."""

    def __init__(
        self,
        config: LongVideoConfig,
        config_path: str | Path,
        *,
        profile: str = "",
    ) -> None:
        self.config = config
        self.config_path = str(Path(config_path).expanduser().resolve())
        # Workers re-load the config from disk, so the profile has to travel with
        # them explicitly or every subprocess would silently use the default mix.
        self.profile = profile or config.profile

    def discover_scripts(self) -> list[Path]:
        """Find input 口播稿 files per ``batch.input_dir`` / ``batch.input_glob``."""
        settings = self.config.batch
        if not str(settings.input_dir).strip():
            raise ValueError("batch.input_dir is not set")
        root = Path(settings.input_dir).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"batch.input_dir does not exist: {root}")
        found = sorted(path for path in root.glob(settings.input_glob) if path.is_file())
        if settings.limit and settings.limit > 0:
            found = found[: settings.limit]
        if not found:
            raise FileNotFoundError(
                f"no inputs matched {settings.input_glob!r} under {root}"
            )
        return found

    async def run(self, *, output_root: str | Path | None = None) -> BatchReport:
        """Execute the batch and aggregate the results."""
        settings = self.config.batch
        scripts = self.discover_scripts()
        root = Path(output_root or self.config.recording.output_root).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        status_path = root / STATUS_FILENAME

        items = [
            BatchItem(
                index=index,
                script_path=str(path),
                episode_id=path.stem,
                output_dir=str(root / path.stem),
            )
            for index, path in enumerate(scripts)
        ]

        report = BatchReport(total=len(items), items=items)
        semaphore = asyncio.Semaphore(max(1, settings.workers))
        lock = asyncio.Lock()

        async def run_one(item: BatchItem) -> None:
            # Resume skips only episodes that actually delivered a video. A dumped
            # but failed episode is worth retrying, and its trajectory is already
            # safe on disk either way.
            if settings.resume:
                manifest = _read_json(Path(item.output_dir) / "episode.json")
                if manifest.get("final_video_exists"):
                    item.status = "done"
                    item.final_video_path = str(manifest.get("final_video_path") or "")
                    async with lock:
                        report.skipped += 1
                        report.done += 1
                        _write_status(status_path, report)
                    logger.info(
                        "[%d/%d] skip (resume) %s", item.index + 1, report.total, item.episode_id
                    )
                    return

            async with semaphore:
                item.status = "running"
                item.started_at = _now()
                async with lock:
                    _write_status(status_path, report)

                started = time.monotonic()
                code, stderr_tail = await self._spawn(item)
                item.elapsed_seconds = round(time.monotonic() - started, 1)
                item.finished_at = _now()

                manifest = _read_json(Path(item.output_dir) / "episode.json")
                if code == 0 and manifest:
                    item.status = "done"
                    item.final_video_path = str(manifest.get("final_video_path") or "")
                else:
                    item.status = "failed"
                    # The manifest names the actual failure; stderr is just log noise
                    # unless the rollout died before it could write one.
                    reasons = [str(reason) for reason in manifest.get("errors", []) if str(reason)]
                    item.error = (
                        "; ".join(reasons)[:800]
                        or stderr_tail
                        or f"exit code {code}"
                    )

                async with lock:
                    if item.status == "done":
                        report.done += 1
                    else:
                        report.failed += 1
                    _write_status(status_path, report)

                logger.info(
                    "[%d/%d] %s %s (%.1fs)",
                    item.index + 1, report.total, item.status, item.episode_id,
                    item.elapsed_seconds,
                )
                if item.status == "failed" and not settings.continue_on_error:
                    raise RuntimeError(f"rollout failed for {item.script_path}: {item.error}")

        await asyncio.gather(*(run_one(item) for item in items))

        report.aggregate_dir = aggregate_episodes(root)
        _write_status(status_path, report)
        logger.info("%s", report.summary())
        return report

    async def _spawn(self, item: BatchItem) -> tuple[int, str]:
        """Run one rollout as ``python -m longvideo_polar.cli run``."""
        Path(item.output_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(item.output_dir) / "worker.log"
        command = [
            sys.executable, "-m", "longvideo_polar.cli", "run",
            "--config", self.config_path,
            "--script-file", item.script_path,
            "--output", item.output_dir,
            "--episode-id", item.episode_id,
        ]
        if self.profile:
            command.extend(["--profile", self.profile])
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")

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


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_episodes(root: str | Path) -> str:
    """Combine per-episode dumps into batch-level SFT/RL shards.

    Walks every ``<root>/*/episode.json`` and concatenates the three role streams
    into ``<root>/aggregate``:

    - ``orchestrator_sft.jsonl`` — one multi-turn episode per line (主流程)
    - ``generator_sft.jsonl`` / ``critic_sft.jsonl`` — one skill sample per line
    - ``interactions.jsonl`` — every cross-role handoff, episode-tagged
    - ``rewards.jsonl`` — flat reward rows
    - ``index.json`` — per-episode manifest roll-up

    Returns:
        The aggregate directory path, or ``""`` when no episodes were found.
    """
    root_path = Path(root).expanduser()
    manifests = sorted(root_path.glob("*/episode.json"))
    if not manifests:
        logger.warning("no episode.json found under %s; nothing to aggregate", root_path)
        return ""

    out_dir = root_path / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    index: list[dict[str, Any]] = []
    role_rows: dict[str, list[dict[str, Any]]] = {role.value: [] for role in ALL_ROLES}
    interactions: list[dict[str, Any]] = []
    rewards: list[dict[str, Any]] = []

    for manifest_path in manifests:
        manifest = _read_json(manifest_path)
        if not manifest:
            continue
        episode_id = str(manifest.get("episode_id") or manifest_path.parent.name)
        index.append({
            "episode_id": episode_id,
            "status": manifest.get("status"),
            "annotation_profile": manifest.get("metadata", {}).get("annotation_profile", ""),
            "final_video_path": manifest.get("final_video_path"),
            "final_video_exists": manifest.get("final_video_exists"),
            "orchestrator_steps": manifest.get("orchestrator_steps"),
            "roles": manifest.get("roles"),
            "interactions": manifest.get("interactions"),
            "reward_summary": manifest.get("reward_summary"),
            "errors": manifest.get("errors"),
            "elapsed_seconds": manifest.get("elapsed_seconds"),
            "output_dir": str(manifest_path.parent),
        })

        traj_dir = manifest_path.parent / "trajectories"

        roles_info = manifest.get("roles") if isinstance(manifest.get("roles"), dict) else {}
        orchestrator = _read_json(traj_dir / "orchestrator.jsonl")
        if orchestrator:
            role_rows[Role.ORCHESTRATOR.value].append({
                "episode_id": episode_id,
                "messages": orchestrator.get("sft_messages", []),
                "metadata": {
                    "episode_id": episode_id,
                    "num_tool_calls": orchestrator.get("num_tool_calls", 0),
                    "reward_summary": orchestrator.get("reward_summary", {}),
                    "status": manifest.get("status"),
                    # Skill rows carry a per-sample provider; a 主流程 row spans the
                    # whole loop, so it records the mix instead. Shards from an
                    # all-Qwen run and a Gemini-critic run must stay separable after
                    # they are concatenated here.
                    "annotation_profile": str(
                        manifest.get("metadata", {}).get("annotation_profile") or ""
                    ),
                    "providers": {
                        role: (info or {}).get("providers_used") or []
                        for role, info in roles_info.items()
                    },
                },
            })
            for row in orchestrator.get("reward_scores", []) or []:
                if isinstance(row, dict):
                    rewards.append({"episode_id": episode_id, **row})

        for role in (Role.GENERATOR, Role.CRITIC):
            payload = _read_json(traj_dir / f"{role.value}.jsonl")
            for sample in (payload.get("samples", []) if payload else []):
                if not isinstance(sample, dict):
                    continue
                metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
                role_rows[role.value].append({
                    "episode_id": episode_id,
                    "messages": sample.get("messages", []),
                    "metadata": {**metadata, "episode_id": episode_id},
                })

        for row in _read_lines(traj_dir / "interactions.jsonl"):
            interactions.append({"episode_id": episode_id, **row})

    _write_lines(out_dir / "orchestrator_sft.jsonl", role_rows[Role.ORCHESTRATOR.value])
    _write_lines(out_dir / "generator_sft.jsonl", role_rows[Role.GENERATOR.value])
    _write_lines(out_dir / "critic_sft.jsonl", role_rows[Role.CRITIC.value])
    _write_lines(out_dir / "interactions.jsonl", interactions)
    _write_lines(out_dir / "rewards.jsonl", rewards)

    succeeded = sum(1 for item in index if item.get("final_video_exists"))
    (out_dir / "index.json").write_text(
        json.dumps(
            {
                "generated_at": _now(),
                "root": str(root_path),
                "episode_count": len(index),
                "succeeded": succeeded,
                "samples": {
                    "orchestrator": len(role_rows[Role.ORCHESTRATOR.value]),
                    "generator": len(role_rows[Role.GENERATOR.value]),
                    "critic": len(role_rows[Role.CRITIC.value]),
                },
                "interactions": len(interactions),
                "reward_rows": len(rewards),
                "episodes": index,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(
        "aggregated %d episodes (%d with video) → %s",
        len(index), succeeded, out_dir,
    )
    return str(out_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, TypeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _write_lines(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_status(path: Path, report: BatchReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": _now(),
        "total": report.total,
        "done": report.done,
        "failed": report.failed,
        "skipped": report.skipped,
        "aggregate_dir": report.aggregate_dir,
        "items": [asdict(item) for item in report.items],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
