"""End-to-end rollout driver: one 口播稿 in, one 成片 plus three trajectories out.

:class:`LongVideoRunner` owns a single rollout:

1. Build the recorder and the role-routed model client (which the recorder is
   wired into, so every model call is captured).
2. Bootstrap AICutPipeline and construct its real ``InferenceOrchestrator``.
3. Await ``run_agentic_pipeline`` — the actual 主流程 loop, dispatching to the
   generator and critic and to the media tools.
4. Merge the pipeline's own ``agent_trace.jsonl`` for full-fidelity tool payloads,
   then write the orchestrator / generator / critic streams and the manifest.

The rollout is recorded whether it succeeds or fails. Failed runs carry the
reflection-and-repair signal that makes them worth keeping, so
``recording.save_on_failure`` defaults on.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from longvideo_polar.config import LongVideoConfig
from longvideo_polar.model_client import RoleRoutedModelClient
from longvideo_polar.pipeline_bridge import build_pipeline_context
from longvideo_polar.recorder import TrajectoryRecorder
from longvideo_polar.roles import Role, load_tool_schema
from longvideo_polar.switch_policy import expected_skip_error

logger = logging.getLogger(__name__)

TASK_DESCRIPTION = "将口播稿转化为完整成片：分段 → 镜头规划 → 素材/TTS → 装饰 → 配乐 → 合成 → 质检修复 → 拼接"

# Stages a complete trajectory passes through on its way to stitching. Every one
# of them is the sole producer of a training signal — the dimension judge is the
# only source of ``quality_judge`` critic samples — so a run that skips one is
# still valid but teaches less than it should.
_EXPECTED_ARC = (
    "Tool_SegmentArticle",
    "Tool_PlanShots",
    "Tool_BatchDecorateSpotlights",
    "Tool_Stage4DimensionJudge",
    "Tool_BatchTimelineCompose",
)


@dataclass
class RolloutResult:
    """Outcome of one rollout."""

    episode_id: str
    output_dir: str
    success: bool
    final_video_path: str = ""
    errors: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def summary(self) -> str:
        roles = self.manifest.get("roles", {})
        counts = ", ".join(
            f"{name}={info.get('llm_calls', 0)}"
            f"{'@' + ','.join(info.get('providers_used') or []) if info.get('providers_used') else ''}"
            for name, info in roles.items()
        )
        status = "OK" if self.success else "FAILED"
        profile = self.manifest.get("metadata", {}).get("annotation_profile") or "-"
        return (
            f"[{status}] {self.episode_id} profile={profile} "
            f"steps={self.manifest.get('orchestrator_steps', 0)} "
            f"llm_calls({counts}) video={self.final_video_path or '-'} "
            f"elapsed={self.elapsed_seconds:.1f}s"
        )


class LongVideoRunner:
    """Drive AICutPipeline's long-horizon agent loop with full trajectory capture."""

    def __init__(self, config: LongVideoConfig) -> None:
        config.validate()
        self.config = config

    async def run(
        self,
        *,
        script: str,
        episode_id: str | None = None,
        output_dir: str | Path | None = None,
        source: str = "",
    ) -> RolloutResult:
        """Execute one rollout.

        Args:
            script: The 口播稿 text.
            episode_id: Stable id for this episode; generated when omitted.
            output_dir: Where to write artifacts. Defaults to
                ``recording.output_root/<episode_id>``.
            source: Provenance label (e.g. the input filename) for the manifest.

        Returns:
            The :class:`RolloutResult`, with artifacts already on disk.
        """
        text = str(script or "").strip()
        if not text:
            raise ValueError("script is empty")

        episode = episode_id or _make_episode_id()
        target = Path(output_dir) if output_dir else Path(self.config.recording.output_root) / episode
        target.mkdir(parents=True, exist_ok=True)

        recording = self.config.recording
        tool_schema = load_tool_schema(strict=False)

        recorder = TrajectoryRecorder(
            episode_id=episode,
            output_dir=target,
            task_description=TASK_DESCRIPTION,
            input_script=text,
            max_raw_output_chars=recording.max_raw_output_chars,
            store_full_prompts=recording.store_full_prompts,
            metadata={
                "source": source,
                "config_path": self.config.raw.get("_config_path", ""),
                "annotation_profile": self.config.profile,
                "agent_mode": self.config.agent.agent_mode,
                "max_agent_steps": self.config.agent.max_steps,
                "tool_switches": self.config.tool_switches.as_dict(),
                "models": self.config.describe_models(),
            },
            require_final_video=self.config.tool_switches.final_stitch,
            benign_error=lambda message: expected_skip_error(
                message, self.config.tool_switches
            ),
        )

        client = RoleRoutedModelClient(
            endpoints=self.config.models,
            tool_schema=tool_schema,
            recorder=recorder,
        )
        events = _EventSink(recorder, tool_schema)

        started = time.monotonic()
        run_result: dict[str, Any] = {}

        # TTS preflight: probe health, auto-start if down (batch-mode safe).
        # Skipped when tool_switches.tts is False (pure trajectory harvest).
        if self.config.tool_switches.tts:
            try:
                from longvideo_polar.tts_lifecycle import preflight_tts
                tts_status = preflight_tts(
                    service_url=self.config.pipeline.tts_service_url,
                )
                if not tts_status.healthy:
                    logger.warning(
                        "TTS not healthy at %s — rollout continues but "
                        "StitchVideo may fail",
                        self.config.pipeline.tts_service_url,
                    )
            except Exception as tts_exc:  # noqa: BLE001
                logger.warning("TTS preflight error (non-fatal): %s", tts_exc)

        try:
            context = build_pipeline_context(
                self.config,
                model_client=client,
                broadcast_fn=events.broadcast,
                episode_id=episode,
            )
            # Re-resolve strictly now that AICutPipeline is importable: a drifted
            # role split would silently mislabel every stream.
            strict_schema = load_tool_schema(strict=recording.strict_tool_schema)
            client._tool_schema = strict_schema  # noqa: SLF001 - same package
            events.tool_schema = strict_schema
            recorder.update_metadata(tool_schema=strict_schema.describe())

            run_result = await context.orchestrator.run_agentic_pipeline(
                text, episode, str(target)
            )
        except Exception as exc:  # noqa: BLE001 - persist partial trajectories
            message = f"{type(exc).__name__}: {exc}"
            logger.exception("rollout %s failed", episode)
            recorder.add_error(message)
            if not recording.save_on_failure:
                raise
        finally:
            client.unload()

        events.flush()
        merged = _merge_agent_trace(recorder, target, run_result)
        if merged:
            logger.debug("merged %d agent_trace rows for %s", merged, episode)

        artifacts = recorder.save(result=run_result)
        manifest = _read_manifest(artifacts.get("episode", ""))
        elapsed = time.monotonic() - started

        errors = [str(item) for item in manifest.get("errors", [])]
        result = RolloutResult(
            episode_id=episode,
            output_dir=str(target),
            success=self._rollout_succeeded(manifest, errors),
            final_video_path=str(manifest.get("final_video_path") or ""),
            errors=errors,
            artifacts=artifacts,
            manifest=manifest,
            elapsed_seconds=elapsed,
        )
        logger.info("%s", result.summary())
        return result

    def _rollout_succeeded(self, manifest: dict[str, Any], errors: list[str]) -> bool:
        """Whether this rollout produced what the configured mode asked for.

        A finished video is the right bar only when the run was asked to make
        one. With ``final_stitch`` off the goal is the trajectory itself, so
        success means the agent walked the whole arc and ended by calling
        ``Tool_StitchVideo``, with no error beyond the switched-off stages
        reporting themselves.
        """
        switches = self.config.tool_switches
        real = [item for item in errors if not expected_skip_error(item, switches)]
        if switches.final_stitch:
            return bool(manifest.get("final_video_exists")) and not real
        return self._reached_terminal_tool(manifest) and not real

    def _reached_terminal_tool(self, manifest: dict[str, Any]) -> bool:
        """Whether the loop ended by stitching, and warn if it got there too fast.

        Reaching the terminal tool is the success bar, but reaching it *early* is
        the failure this mode is most prone to: skipping the quality arc still
        produces a well-formed episode, just one whose critic stream carries no
        ``quality_judge`` samples at all. Naming the gap here means a regression
        shows up per episode instead of at export time.
        """
        stages = {
            str(item).split(":", 1)[-1] for item in manifest.get("stages_completed") or []
        }
        if not any(name == "Tool_StitchVideo" for name in stages):
            return False
        skipped = [name for name in _EXPECTED_ARC if name not in stages]
        if skipped:
            logger.warning(
                "%s stitched without %s — the trajectory is short of the full arc",
                manifest.get("episode_id", "episode"), ", ".join(skipped),
            )
        return True


class _EventSink:
    """Translate AICutPipeline broadcast events into orchestrator step records.

    ``run_agentic_pipeline`` emits ``tool_call`` immediately before dispatch and
    ``tool_result`` immediately after, so pairing them yields exactly one record
    per executed step. ``tool_call`` also pins the authoritative step index and
    tool name, which is what attributes the generator/critic calls made during
    that dispatch back to the decision that triggered them.
    """

    def __init__(self, recorder: TrajectoryRecorder, tool_schema: Any) -> None:
        self.recorder = recorder
        self.tool_schema = tool_schema
        self._pending: dict[int, dict[str, Any]] = {}

    async def broadcast(self, run_id: str, event: dict[str, Any]) -> None:  # noqa: ARG002
        if not isinstance(event, dict):
            return
        kind = str(event.get("type") or "")
        if kind == "tool_call":
            self._on_tool_call(event)
        elif kind == "tool_result":
            self._on_tool_result(event)
        elif kind == "error":
            self.recorder.add_error(
                f"[{event.get('stage', '')}] {event.get('message', '')}".strip()
            )
        elif kind == "pipeline_complete":
            self.flush()

    def _on_tool_call(self, event: dict[str, Any]) -> None:
        step = _as_int(event.get("step"))
        tool = str(event.get("tool_name") or "")
        self.recorder.begin_step(step, tool)
        self._pending[step] = {
            "tool": tool,
            "stage": str(event.get("stage") or ""),
            "arguments": event.get("arguments") if isinstance(event.get("arguments"), dict) else {},
            "precondition_missing": [
                str(item) for item in (event.get("precondition_missing") or [])
            ],
        }

    def _on_tool_result(self, event: dict[str, Any]) -> None:
        step = _as_int(event.get("step"))
        pending = self._pending.pop(step, None)
        tool = str(event.get("tool_name") or (pending or {}).get("tool") or "")
        observation = event.get("observation")
        output: dict[str, Any] = {"status": str(event.get("status") or "unknown")}
        if isinstance(observation, dict):
            output.update(observation)
            output.setdefault("status", str(event.get("status") or "unknown"))
        self.recorder.record_tool_call(
            step=step,
            tool=tool,
            stage=str(event.get("stage") or (pending or {}).get("stage") or ""),
            arguments=(pending or {}).get("arguments") or {},
            output=output,
            precondition_missing=(pending or {}).get("precondition_missing") or [],
            external=self.tool_schema.is_external(tool),
            executed_by=self.tool_schema.role_for_tool(tool),
        )

    def flush(self) -> None:
        """Record any step that was dispatched but never reported a result."""
        for step in sorted(self._pending):
            pending = self._pending[step]
            self.recorder.record_tool_call(
                step=step,
                tool=pending["tool"],
                stage=pending["stage"],
                arguments=pending["arguments"],
                output={"status": "incomplete", "message": "no tool_result event received"},
                precondition_missing=pending["precondition_missing"],
                external=self.tool_schema.is_external(pending["tool"]),
                executed_by=self.tool_schema.role_for_tool(pending["tool"]),
            )
        self._pending.clear()


def _merge_agent_trace(
    recorder: TrajectoryRecorder, output_dir: Path, run_result: dict[str, Any]
) -> int:
    """Merge full tool payloads from the pipeline's own trace file."""
    candidates = [output_dir / "agent_trace.jsonl"]
    project_id = str(run_result.get("project_id") or "")
    project_dir = str(run_result.get("output_dir") or "")
    if project_dir:
        candidates.append(Path(project_dir) / "agent_trace.jsonl")
    elif project_id:
        candidates.append(output_dir / project_id / "agent_trace.jsonl")

    for path in candidates:
        if not path.is_file():
            continue
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
        if rows:
            recorder.update_metadata(agent_trace_path=str(path))
            return recorder.merge_agent_trace(rows)
    return 0


def _read_manifest(path: str) -> dict[str, Any]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.is_file():
        return {}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, TypeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _make_episode_id() -> str:
    return f"lvp_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
