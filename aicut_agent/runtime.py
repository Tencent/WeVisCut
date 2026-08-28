"""Per-run runtime bridge between the deepagents ReAct loop and the AICut engine.

The RL policy was trained against a very specific contract: every decision step
is a *single-turn* call ``(system_prompt, build_agentic_state_json(state),
tools_schema)`` where the runtime state — not the chat history — is the source
of truth (``agent_contract.runtime_is_state_authority: true``). The deepagents
harness, by contrast, drives a multi-turn message list.

``AICutRuntime`` reconciles the two. It owns the authoritative pipeline state
(via an :class:`AutonomousOrchestrator` instance) and exposes the three
operations the harness needs:

* ``build_user_payload()`` — rebuild the exact training user-JSON each step
  (consumed by :class:`~aicut_agent.chat_model.AICutChatModel`);
* ``execute_tool()`` — run one tool through the real pipeline and update state
  (consumed by the LangChain tool wrappers in ``tools.py``);
* the guardrail helpers used by ``middleware.AICutGuardrailMiddleware``
  (precondition blocking, repeat/failure/Stage4 budgets, stitch termination).

All of the cost guardrails *terminate or refuse* a call — they never rewrite the
model's decision, mirroring ``AutonomousOrchestrator.run_autonomous`` exactly.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from studio import tracks
from studio.config import install_import_paths

install_import_paths()

from server.infer_graph import (  # noqa: E402
    LONG_HORIZON_ACTION_SPACE,
    LONG_HORIZON_PHASE_LABELS,
    LONG_HORIZON_TOOL_NAME_ALIASES,
)

logger = logging.getLogger("aicut_online.runtime")

_STAGE4_REPAIR_TOOL = "Tool_Stage4DimensionalRepair"
_STAGE4_JUDGE_TOOL = "Tool_Stage4DimensionJudge"
_STAGE4_TOOLS = frozenset({
    _STAGE4_JUDGE_TOOL, _STAGE4_REPAIR_TOOL, "Tool_TextQuickJudge",
})


def _trunc_text(text: str, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + "…"


class AICutRuntime:
    """Holds one run's authoritative state and guardrail counters.

    A fresh runtime (and therefore a fresh agent graph) is created per run. The
    heavy model loader is shared across runs; only the state is per-run.
    """

    def __init__(
        self,
        orchestrator: Any,
        *,
        input_script: str,
        run_id: str,
        output_dir: str,
        track_order: list[str],
        broadcast_fn: Callable[..., Any],
        progress_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.input_script = input_script
        self.run_id = run_id
        self.track_order = list(track_order or tracks.ALL_TRACKS)
        self.broadcast_fn = broadcast_fn
        self.progress_fn = progress_fn

        # Termination / result bookkeeping (mirrors run_autonomous locals).
        self.results: dict[str, Any] = {
            "project_id": "",
            "output_dir": "",
            "stages_completed": [],
            "final_video_path": "",
            "errors": [],
            "agent_mode": getattr(orchestrator, "agent_mode", ""),
            "autonomous": bool(getattr(orchestrator, "fully_autonomous", True)),
            "steps_used": 0,
        }
        self.terminated_reason = ""
        self.stitch_succeeded = False

        # Guardrail counters (cost bounds only — never redirects).
        self.same_tool_count = 0
        self.last_tool = ""
        self.consecutive_parse_failures = 0
        self.consecutive_blocked = 0
        self.identical_failures: dict[str, int] = {}
        self.repeat_calls: dict[str, int] = {}
        self.stage4_attempts = 0
        self.stage4_repairs = 0
        self.stage4_refusals = 0
        self.stage4_judge_stalls = 0
        self.tool_calls_used = 0
        self.last_error = ""
        self.step_idx = 0
        # Set by the chat model / tools each step for the middleware to consume.
        self.last_prediction: Any = None
        self.last_tool_name = ""
        self.last_tool_output: dict[str, Any] = {}

        # Guardrail ceilings, read from the orchestrator's autonomy config.
        orch = orchestrator
        self.max_consecutive_parse_failures = int(getattr(orch, "max_consecutive_parse_failures", 4))
        self.max_consecutive_blocked = int(getattr(orch, "max_consecutive_blocked", 6))
        self.max_identical_failures = int(getattr(orch, "max_identical_failures", 3))
        self.max_identical_calls = int(getattr(orch, "max_identical_calls", 3))
        self.max_stage4_repairs = int(getattr(orch, "max_stage4_repairs", 2))
        self.max_stage4_attempts = int(getattr(orch, "max_stage4_attempts", 6))
        self.max_stage4_refusals = int(getattr(orch, "max_stage4_refusals", 2))
        self.max_tool_calls = int(getattr(orch, "max_tool_calls", 30))
        self.max_same_tool = int(getattr(orch, "max_same_tool", 10))
        self.max_agent_steps = int(getattr(orch, "max_agent_steps", 48))

        # Initialise the authoritative pipeline state + project directory.
        self.project_id, self.project_dir, self.state = self._init_state(output_dir)
        self.results["project_id"] = self.project_id
        self.results["output_dir"] = str(self.project_dir)
        orchestrator.state = self.state
        orchestrator.project_dir = self.project_dir
        orchestrator._track_order_cache = list(self.track_order)
        self.trace_path = self.project_dir / "agent_trace.jsonl"

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _init_state(self, output_dir: str) -> tuple[str, Path, dict[str, Any]]:
        """Create the project dir and the authoritative state dict.

        Mirrors the state initialisation in ``run_autonomous`` exactly.
        """
        from pipeline.graph import _make_project_id  # noqa: E402

        project_id = _make_project_id()
        project_dir = Path(output_dir) / project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        state: dict[str, Any] = {
            "mode": "agent_long_horizon",
            "project_id": project_id,
            "output_dir": str(project_dir),
            "config": self.orchestrator.pipeline_config,
            "input_text": self.input_script,
            "segments": [],
            "shot_plan": [],
            "shot_assets": {},
            "bgm_tracks": [],
            "severity_scores": {},
            "severity_ranked_shots": [],
            "current_stage": "agent_loop",
            "plan_approved": False,
            "reflect_round": 0,
            "errors": [],
            "trajectory": [],
        }
        return project_id, project_dir, state

    @property
    def system_prompt(self) -> str:
        """The exact training system prompt (held by the orchestrator)."""
        return str(getattr(self.orchestrator, "system_prompt", "") or "")

    # ------------------------------------------------------------------
    # model-facing payload (training-format)
    # ------------------------------------------------------------------
    def build_user_payload(self) -> str:
        """Rebuild the exact training user-JSON from the authoritative state."""
        return self.orchestrator.build_agentic_state_json(self.input_script, self.state)

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------
    async def broadcast(self, event: dict[str, Any]) -> None:
        if self.broadcast_fn is not None:
            await self.broadcast_fn(self.run_id, event)

    async def emit_progress(self, phase: str) -> None:
        snapshot = tracks.build_snapshot(self.state, self.track_order)
        snapshot["phase"] = phase
        if self.progress_fn is not None:
            await self.progress_fn(self.run_id, snapshot)
        else:
            await self.broadcast({"type": "progress", "progress": snapshot})

    async def broadcast_thinking(self, prediction: Any, tool_name: str) -> None:
        """Emit the 'thinking' event the frontend renders for each decision."""
        try:
            thinking = self.orchestrator._thinking_event(
                stage=self.phase_for_tool(tool_name), prediction=prediction, step=self.step_idx,
            )
        except Exception:
            thinking = {"type": "thinking", "step": self.step_idx}
        thinking["tool_name"] = tool_name
        await self.broadcast(thinking)

    # ------------------------------------------------------------------
    # feedback (the field the model was trained to read)
    # ------------------------------------------------------------------
    def set_feedback(self, payload: dict[str, Any]) -> None:
        self.orchestrator._set_feedback(payload)

    def phase_for_tool(self, tool_name: str) -> str:
        return self.orchestrator._phase_for_tool(tool_name)

    def observation_text(self, tool_name: str, output: dict[str, Any]) -> str:
        """Serialise the tool observation for the harness's ToolMessage content.

        The model never reads this (its input is rebuilt from runtime state), so
        this is purely for the harness's bookkeeping and the frontend's event
        stream. We return the same observation the runtime recorded.
        """
        obs = self.orchestrator._make_agent_observation(tool_name, output)
        try:
            return json.dumps(obs, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(obs)

    def record_parse_failure(self, raw_tool: str, prediction: Any, *, unknown: bool) -> bool:
        """Record an unparseable/unknown-tool decision as an observation.

        Returns True if the run should stop (ceiling reached).
        """
        self.consecutive_parse_failures += 1
        if unknown:
            self.set_feedback({
                "status": "unknown_tool",
                "requested": raw_tool,
                "allowed_tools": LONG_HORIZON_ACTION_SPACE,
                "hint": "只能调用 allowed_tools 中的工具名",
            })
        else:
            self.set_feedback({
                "status": "invalid_output",
                "reason": "未能解析出单个合法 tool_call",
                "hint": "本轮只输出一个 <tool_call> JSON，参数使用 allowed_tool_args 白名单内的键",
                "attempt": self.consecutive_parse_failures,
            })
        return self.consecutive_parse_failures >= self.max_consecutive_parse_failures

    def reset_parse_failures(self) -> None:
        self.consecutive_parse_failures = 0

    # ------------------------------------------------------------------
    # tool execution (called by the LangChain tool wrappers)
    # ------------------------------------------------------------------
    async def execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
        """Run one tool through the real pipeline and fold the result into state.

        This is the body the LangChain tool wrappers call. It assumes the
        guardrail middleware has already approved the call.
        """
        orch = self.orchestrator
        await orch._wait_if_paused(self.run_id)

        flags_before = orch._completion_flags()
        tool_output = await orch._dispatch_tool(tool_name, tool_args, self.state, self.run_id)
        if isinstance(tool_output, dict):
            # Still applied on failure: a partially-successful batch tool writes
            # real assets into state and we want to keep them.
            orch.update_state_from_respond(tool_name, tool_output)
            orch._apply_tool_output(tool_name, tool_output, self.state)
        else:
            tool_output = {"status": "done"}

        tool_status = str(tool_output.get("status") or "")
        if tool_status == "error":
            restored = orch._restore_completion_flags(flags_before)
            if restored:
                logger.warning("%s 失败，撤销被误置的完成标记: %s", tool_name, restored)
                await self.broadcast({
                    "type": "console_log", "level": "warn",
                    "text": f"  {tool_name} 失败，已撤销误置的完成标记: {', '.join(restored)}",
                })

        orch._record_agent_history(tool_name, tool_output)
        self.results["stages_completed"].append(f"{self.step_idx:02d}:{tool_name}")
        orch._append_trace(self.trace_path, {
            "step": self.step_idx,
            "tool_name": tool_name,
            "arguments": tool_args,
            "output": tool_output,
            "timestamp": datetime.now().isoformat(),
        })
        self.last_tool_output = tool_output
        return tool_output

    # ------------------------------------------------------------------
    # guardrail hooks (called by the middleware)
    # ------------------------------------------------------------------
    def check_preconditions(self, tool_name: str) -> list[str]:
        return self.orchestrator._agent_precondition_missing(tool_name, self.state)

    def note_tool_selected(self, tool_name: str) -> str | None:
        """Track consecutive same-tool selections. Returns a termination reason."""
        if tool_name == self.last_tool:
            self.same_tool_count += 1
        else:
            self.same_tool_count = 1
            self.last_tool = tool_name
        if self.same_tool_count > self.max_same_tool:
            reason = f"模型连续 {self.same_tool_count} 次选择 {tool_name}，判定为死循环，停止执行"
            if self.last_error:
                reason += f"。该工具最近一次失败原因: {self.last_error}"
            return reason
        return None

    def args_signature(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        try:
            return "%s::%s" % (
                tool_name,
                json.dumps(tool_args, sort_keys=True, ensure_ascii=False, default=str),
            )
        except (TypeError, ValueError):
            return f"{tool_name}::{tool_args!r}"

    # ------------------------------------------------------------------
    # finalisation
    # ------------------------------------------------------------------
    def finalize(self) -> dict[str, Any]:
        """Persist final_state.json and return the run results."""
        self.results["tool_calls_used"] = self.tool_calls_used
        self.results["stage4_repairs"] = self.stage4_repairs
        self.results["final_video_path"] = str(self.state.get("final_video_path", "") or "")
        if self.terminated_reason:
            self.results["errors"].append(self.terminated_reason)
        self._write_final_state()
        return self.results

    def _write_final_state(self) -> None:
        payload = {
            "schema_version": 1,
            "project_id": self.project_id,
            "input_text": self.input_script,
            "segments": self.state.get("segments", []),
            "shot_plan": self.state.get("shot_plan", []),
            "shot_assets": self.state.get("shot_assets", {}),
            "bgm_tracks": self.state.get("bgm_tracks", []),
            "stitch_result": self.state.get("stitch_result", {}),
            "music_style": self.state.get("music_style", ""),
            "final_video_path": self.state.get("final_video_path", ""),
            "stages_completed": self.results.get("stages_completed", []),
            "errors": self.results.get("errors", []),
            "agent_mode": getattr(self.orchestrator, "agent_mode", ""),
            "autonomous": bool(getattr(self.orchestrator, "fully_autonomous", True)),
            "saved_at": datetime.now().isoformat(),
        }
        try:
            (self.project_dir / "final_state.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover - disk failure path
            logger.warning("final_state.json 落盘失败（不影响推理结果）: %s", exc)
