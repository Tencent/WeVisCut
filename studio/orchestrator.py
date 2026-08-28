"""Autonomous agent loop for the RL-trained policy.

The reference implementation (``server/infer_graph.py``) wraps its
agent loop in seven "扶正" (hand-holding) redirects that silently rewrite the
model's chosen tool: precondition failures get redirected, repeats get
redirected, and once every precondition happens to be satisfied the loop force
-injects ``Tool_StitchVideo``. Those exist because an SFT-only policy strands
itself. A DAPO-trained policy has been optimised against the real runtime and
walks the flow on its own, so this loop **never rewrites the model's decision**.

What replaces the redirects: every rejection is reported back through
``last_observation`` — the same field the model saw during training — so the
model self-corrects from an in-distribution signal instead of having its action
swapped out from under it. That keeps the trajectory honest and makes failures
visible in the UI rather than papered over.

The safety rails that remain are the ones that bound cost, not correctness:
a step ceiling, a repeat ceiling, and a consecutive-parse-failure ceiling. All
three end the run with a reported error instead of steering it.

Set ``agent.autonomy.fully_autonomous: false`` in ``studio_config.yaml`` to
restore the reference behaviour.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from studio import tracks
from studio.config import install_import_paths

install_import_paths()

from server.infer_graph import (  # noqa: E402  (path set up above)
    InferenceOrchestrator,
    LONG_HORIZON_ACTION_SPACE,
    LONG_HORIZON_PHASE_LABELS,
    LONG_HORIZON_TOOL_NAME_ALIASES,
)

logger = logging.getLogger("aicut_studio.orchestrator")

# Cost bounds. These stop a run; they never redirect one.
DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES = 4
DEFAULT_MAX_CONSECUTIVE_BLOCKED = 6
# A tool failing the same way twice is an environment fault; more attempts only
# cost time and bury the cause.
DEFAULT_MAX_IDENTICAL_FAILURES = 3
# Re-sending byte-identical arguments means nothing in the response changed the
# model's mind. Tracked separately from failures because the Stage4 repair loop
# returns `completed_with_warnings`, not `error`.
DEFAULT_MAX_IDENTICAL_CALLS = 3
# Repairing the same shots a third time does not converge in practice.
DEFAULT_MAX_STAGE4_REPAIRS = 2
DEFAULT_MAX_STAGE4_ATTEMPTS = 6
# How many times we decline an over-budget repair before ending the run. Each
# refusal still costs a model call, and dragging on to the generic repeat
# ceiling would hide why the run actually stopped.
DEFAULT_MAX_STAGE4_REFUSALS = 2
# Total executed tool calls. A complete flow needs ~12; this is a cost backstop.
DEFAULT_MAX_TOOL_CALLS = 30

_STAGE4_REPAIR_TOOL = "Tool_Stage4DimensionalRepair"
_STAGE4_JUDGE_TOOL = "Tool_Stage4DimensionJudge"
_STAGE4_TOOLS = frozenset({
    _STAGE4_JUDGE_TOOL, _STAGE4_REPAIR_TOOL, "Tool_TextQuickJudge",
})


def _trunc_text(text: str, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + "…"


class AutonomousOrchestrator(InferenceOrchestrator):
    """`InferenceOrchestrator` with the guardrail redirects removed.

    Reuses the parent's prompt construction, tool dispatch, state accumulation
    and preview machinery unchanged — only the control flow differs.
    """

    def __init__(self, *args: Any, progress_fn: Callable | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        autonomy = self.infer_config.get("autonomy")
        self.autonomy: dict[str, Any] = autonomy if isinstance(autonomy, dict) else {}
        self.fully_autonomous = bool(self.autonomy.get("fully_autonomous", True))
        self.report_invalid_as_observation = bool(
            self.autonomy.get("report_invalid_as_observation", True)
        )
        self.max_consecutive_parse_failures = int(
            self.autonomy.get("max_consecutive_parse_failures",
                              DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES) or DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES
        )
        self.max_consecutive_blocked = int(
            self.autonomy.get("max_consecutive_blocked",
                              DEFAULT_MAX_CONSECUTIVE_BLOCKED) or DEFAULT_MAX_CONSECUTIVE_BLOCKED
        )
        self.max_identical_failures = int(
            self.autonomy.get("max_identical_tool_failures",
                              DEFAULT_MAX_IDENTICAL_FAILURES) or DEFAULT_MAX_IDENTICAL_FAILURES
        )
        self.max_identical_calls = int(
            self.autonomy.get("max_identical_tool_calls",
                              DEFAULT_MAX_IDENTICAL_CALLS) or DEFAULT_MAX_IDENTICAL_CALLS
        )
        self.max_stage4_repairs = int(
            self.autonomy.get("max_stage4_repairs",
                              DEFAULT_MAX_STAGE4_REPAIRS) or DEFAULT_MAX_STAGE4_REPAIRS
        )
        self.max_stage4_attempts = int(
            self.autonomy.get("max_stage4_attempts",
                              DEFAULT_MAX_STAGE4_ATTEMPTS) or DEFAULT_MAX_STAGE4_ATTEMPTS
        )
        self.max_tool_calls = int(
            self.autonomy.get("max_tool_calls",
                              DEFAULT_MAX_TOOL_CALLS) or DEFAULT_MAX_TOOL_CALLS
        )

        # Optional side-channel for track snapshots (kept separate from broadcast
        # so the server can throttle high-frequency progress independently).
        self._progress_fn = progress_fn

        # Exposed so the editor can pick up where the run left off.
        self.state: dict[str, Any] = {}
        self.project_dir: Path | None = None

    # ------------------------------------------------------------------
    # guardrail neutralisation
    # ------------------------------------------------------------------
    def _next_missing_agent_tool(self, state: dict[str, Any]) -> str:
        """Return "" so every inherited redirect site becomes a no-op.

        All call sites in the parent are guarded by ``if next_tool and
        next_tool != tool_name``, so an empty string disables them without
        touching the parent's code.
        """
        if self.fully_autonomous:
            return ""
        return super()._next_missing_agent_tool(state)

    def _agent_precondition_missing(self, tool_name: str, state: dict[str, Any]) -> list[str]:
        """Relax preconditions that conflict with the RL-trained policy.

        The parent requires ``quality_checked`` (Stage4DimensionJudge) before
        ``Tool_BatchTimelineCompose``, but the RL policy was trained on a flow
        where quality checking is optional or happens after rendering. Blocking
        it causes the model to enter a Stage4 repair loop that never terminates.
        """
        missing = super()._agent_precondition_missing(tool_name, state)
        if not self.fully_autonomous:
            return missing
        # The RL model may skip Stage4 quality check and go straight to render.
        if tool_name == "Tool_BatchTimelineCompose" and "dimension_quality_check" in missing:
            missing.remove("dimension_quality_check")
        return missing

    def _force_tool_for_stage(self, stage: str, tool_name: str, state: dict[str, Any]) -> str:
        """Never rewrite the model's tool choice."""
        if self.fully_autonomous:
            return tool_name
        return super()._force_tool_for_stage(stage, tool_name, state)

    # ------------------------------------------------------------------
    # feedback channel
    # ------------------------------------------------------------------
    def _set_feedback(self, payload: dict[str, Any]) -> None:
        """Route a rejection into the field the model was trained to read."""
        self.last_observation = payload

    def _make_agent_observation(self, tool_name: str, output: dict[str, Any]) -> dict[str, Any]:
        """Carry the failure reason through to the model.

        The parent builds ``summary`` from a fixed key whitelist that does not
        include ``error``, so a crashed tool reaches the model as
        ``{"status": "error", "summary": {"tts_success_count": 0}}`` — a failure
        with no cause. Retrying verbatim is then a perfectly sensible action,
        which is exactly how a dead service turns into a tool loop.

        ``message`` is already in the parent's whitelist, so putting the reason
        there keeps the observation in the shape the policy saw during training.
        """
        obs = super()._make_agent_observation(tool_name, output)
        if not isinstance(output, dict):
            return obs

        reason = str(output.get("error") or output.get("message") or "").strip()
        if obs.get("status") == "error" and reason:
            summary = obs.setdefault("summary", {})
            summary["message"] = reason[:400]
            summary["error"] = reason[:400]
            obs["hint"] = (
                "该工具执行失败，不是参数问题就是运行环境问题。"
                "重复同样的调用不会有不同结果：请先改变调用方式或改选其他工具。"
            )
        return obs

    # ------------------------------------------------------------------
    # completion flags
    # ------------------------------------------------------------------
    def _completion_flags(self) -> dict[str, bool]:
        """Snapshot the ``*_done`` stage markers the parent keeps on self."""
        return {
            name: value for name, value in vars(self).items()
            if name.endswith("_done") and isinstance(value, bool)
        }

    def _restore_completion_flags(self, before: dict[str, bool]) -> list[str]:
        """Undo stage markers that flipped to done during a failed call."""
        reverted: list[str] = []
        for name, was in before.items():
            if was is False and getattr(self, name, False) is True:
                setattr(self, name, False)
                reverted.append(name)
        return reverted

    # ------------------------------------------------------------------
    # Stage4 repair budget
    # ------------------------------------------------------------------
    def _accept_current_quality(self) -> None:
        """Satisfy the wrap-up preconditions so the model can stitch.

        Flipping these does not stop the model from choosing repair again — it
        only removes the excuse for not finishing. The refusal below is what
        actually ends the loop.
        """
        self.quality_checked = True
        self.repaired_done = True

    async def _refuse_tool(
        self,
        run_id: str,
        trace_path: Path,
        step_idx: int,
        phase: str,
        tool_name: str,
        tool_args: dict[str, Any],
        *,
        reason: str,
        hint: str,
    ) -> None:
        """Decline to execute a call and report it back as an observation.

        Mirrors the precondition path: the model's choice is not rewritten, it
        just does not get run, and the reason goes back through the field the
        policy was trained to read.
        """
        tool_output = {
            "status": "budget_exhausted",
            "tool": tool_name,
            "message": reason,
        }
        # History first: `_record_agent_history` overwrites `last_observation`
        # with its own summary, so setting the feedback before it would drop the
        # hint the model needs to read.
        self._record_agent_history(tool_name, tool_output)
        self._set_feedback({
            "status": "budget_exhausted",
            "tool": tool_name,
            "reason": reason,
            "hint": hint,
        })
        await self.broadcast(run_id, {
            "type": "console_log",
            "level": "warn",
            "text": f"  拒绝执行 {tool_name}: {reason}",
        })
        await self.broadcast(run_id, {
            "type": "tool_result",
            "stage": phase,
            "step": step_idx,
            "tool_name": tool_name,
            "status": "budget_exhausted",
            "observation": self.last_observation,
        })
        self._append_trace(trace_path, {
            "step": step_idx,
            "tool_name": tool_name,
            "arguments": tool_args,
            "output": tool_output,
            "refused": True,
            "timestamp": datetime.now().isoformat(),
        })

    async def _emit_progress(self, run_id: str, state: dict[str, Any], phase: str) -> None:
        snapshot = tracks.build_snapshot(state, self._track_order)
        snapshot["phase"] = phase
        if self._progress_fn is not None:
            await self._progress_fn(run_id, snapshot)
        else:
            await self.broadcast(run_id, {"type": "progress", "progress": snapshot})

    @property
    def _track_order(self) -> list[str]:
        return list(getattr(self, "_track_order_cache", None) or tracks.ALL_TRACKS)

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    async def run_autonomous(
        self,
        input_script: str,
        run_id: str,
        output_dir: str,
        track_order: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run the agent until it stitches a video, errors, or hits a ceiling."""
        self._track_order_cache = list(track_order or tracks.ALL_TRACKS)

        from pipeline.graph import _make_project_id  # noqa: E402

        project_id = _make_project_id()
        project_dir = Path(output_dir) / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        self.project_dir = project_dir

        state: dict[str, Any] = {
            "mode": "agent_long_horizon",
            "project_id": project_id,
            "output_dir": str(project_dir),
            "config": self.pipeline_config,
            "input_text": input_script,
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
        self.state = state

        results: dict[str, Any] = {
            "project_id": project_id,
            "output_dir": str(project_dir),
            "stages_completed": [],
            "final_video_path": "",
            "errors": [],
            "agent_mode": self.agent_mode,
            "autonomous": self.fully_autonomous,
            "steps_used": 0,
        }

        trace_path = project_dir / "agent_trace.jsonl"

        await self.broadcast(run_id, {
            "type": "stage_update",
            "stage": "agent_loop",
            "status": "started",
            "message": f"Agent 自主执行启动 · {project_id}",
            "allowed_tools": LONG_HORIZON_ACTION_SPACE,
            "progress_pct": 0,
        })
        await self.broadcast(run_id, {
            "type": "console_log",
            "level": "system",
            "text": (
                "完全自主模式：不做工具改写 / 不强制收尾，"
                "非法调用只作为 observation 回灌给模型"
                if self.fully_autonomous else
                "兼容模式：保留参考实现的路由校正"
            ),
        })
        await self._emit_progress(run_id, state, "agent_loop")

        same_tool_count = 0
        last_tool = ""
        consecutive_parse_failures = 0
        consecutive_blocked = 0
        identical_failures: dict[str, int] = {}
        repeat_calls: dict[str, int] = {}
        stage4_attempts = 0
        stage4_repairs = 0
        stage4_refusals = 0
        stage4_judge_stalls = 0
        tool_calls_used = 0
        last_error = ""
        terminated_reason = ""

        for step_idx in range(1, self.max_agent_steps + 1):
            results["steps_used"] = step_idx
            await self._wait_if_paused(run_id)

            user_state_json = self.build_agentic_state_json(input_script, state)
            prediction = await asyncio.to_thread(
                self.model.predict_orchestrator if self._is_multi_model else self.model.predict,
                self.system_prompt,
                user_state_json,
                stage="agent_loop",
                allowed_tools=LONG_HORIZON_ACTION_SPACE,
            )
            await self._wait_if_paused(run_id)

            raw_tool = str(prediction.tool_name or "")
            tool_name = LONG_HORIZON_TOOL_NAME_ALIASES.get(raw_tool, raw_tool)
            tool_args = prediction.arguments if isinstance(prediction.arguments, dict) else {}
            phase = self._phase_for_tool(tool_name)

            agent_state = (
                prediction.agent_trace.get("agent_state", {})
                if isinstance(getattr(prediction, "agent_trace", None), dict) else {}
            )
            if isinstance(agent_state, dict) and agent_state.get("phase"):
                phase = str(agent_state["phase"])

            thinking = self._thinking_event(stage=phase, prediction=prediction, step=step_idx)
            thinking["tool_name"] = tool_name
            await self.broadcast(run_id, thinking)

            # ---------- 1. unparseable output: report, do not substitute ----------
            if not prediction.parse_ok:
                consecutive_parse_failures += 1
                self._set_feedback({
                    "status": "invalid_output",
                    "reason": "未能解析出单个合法 tool_call",
                    "hint": "本轮只输出一个 <tool_call> JSON，参数使用 allowed_tool_args 白名单内的键",
                    "attempt": consecutive_parse_failures,
                })
                await self.broadcast(run_id, {
                    "type": "console_log",
                    "level": "warn",
                    "text": (
                        f"step {step_idx} 输出无法解析 "
                        f"({consecutive_parse_failures}/{self.max_consecutive_parse_failures})，"
                        "已回灌错误让模型自行纠正"
                    ),
                })
                if consecutive_parse_failures >= self.max_consecutive_parse_failures:
                    terminated_reason = (
                        f"模型连续 {consecutive_parse_failures} 次输出无法解析，停止执行"
                    )
                    break
                continue

            # ---------- 2. unknown tool: report, do not substitute ----------
            if tool_name not in LONG_HORIZON_ACTION_SPACE:
                consecutive_parse_failures += 1
                self._set_feedback({
                    "status": "unknown_tool",
                    "requested": raw_tool,
                    "allowed_tools": LONG_HORIZON_ACTION_SPACE,
                    "hint": "只能调用 allowed_tools 中的工具名",
                })
                await self.broadcast(run_id, {
                    "type": "console_log",
                    "level": "warn",
                    "text": f"step {step_idx} 未知工具 {raw_tool!r}，已回灌合法工具列表",
                })
                if consecutive_parse_failures >= self.max_consecutive_parse_failures:
                    terminated_reason = f"模型反复请求未知工具 {raw_tool!r}，停止执行"
                    break
                continue

            consecutive_parse_failures = 0

            # ---------- 3. repeat ceiling (cost bound, not a redirect) ----------
            if tool_name == last_tool:
                same_tool_count += 1
            else:
                same_tool_count = 1
                last_tool = tool_name

            if same_tool_count > self.max_same_tool:
                terminated_reason = (
                    f"模型连续 {same_tool_count} 次选择 {tool_name}，判定为死循环，停止执行"
                )
                # Without this the UI reports a loop and hides the tool error
                # that caused it, which points debugging at the wrong layer.
                if last_error:
                    terminated_reason += f"。该工具最近一次失败原因: {last_error}"
                break

            await self.broadcast(run_id, {
                "type": "stage_update",
                "stage": phase,
                "status": "started",
                "message": LONG_HORIZON_PHASE_LABELS.get(phase, phase),
                "progress_pct": tracks.build_overall_progress(state),
            })

            # ---------- 4. preconditions: report only ----------
            missing = self._agent_precondition_missing(tool_name, state)
            await self.broadcast(run_id, {
                "type": "tool_call",
                "stage": phase,
                "step": step_idx,
                "tool_name": tool_name,
                "arguments": tool_args,
                "precondition_missing": missing,
            })

            if missing:
                consecutive_blocked += 1
                tool_output: dict[str, Any] = {
                    "status": "blocked_precondition",
                    "missing": missing,
                    "message": f"{tool_name} 前置条件不足: {', '.join(missing)}",
                }
                # Order matters: `_record_agent_history` overwrites
                # `last_observation`, so the hint has to be written after it.
                self._record_agent_history(tool_name, tool_output)
                self._set_feedback({
                    "status": "blocked_precondition",
                    "tool": tool_name,
                    "missing": missing,
                    "hint": "请先满足上述前置条件，或改选一个当前可执行的工具",
                })
                await self.broadcast(run_id, {
                    "type": "console_log",
                    "level": "warn",
                    "text": (
                        f"{tool_name} 前置条件不足（缺 {', '.join(missing)}），"
                        f"不做重定向，交回模型决策 "
                        f"({consecutive_blocked}/{self.max_consecutive_blocked})"
                    ),
                })
                await self.broadcast(run_id, {
                    "type": "tool_result",
                    "stage": phase,
                    "step": step_idx,
                    "tool_name": tool_name,
                    "status": "blocked_precondition",
                    "observation": self.last_observation,
                })
                self._append_trace(trace_path, {
                    "step": step_idx,
                    "tool_name": tool_name,
                    "arguments": tool_args,
                    "output": tool_output,
                    "blocked": True,
                    "timestamp": datetime.now().isoformat(),
                })
                if consecutive_blocked >= self.max_consecutive_blocked:
                    terminated_reason = (
                        f"模型连续 {consecutive_blocked} 次选择前置条件不满足的工具，停止执行"
                    )
                    break
                continue

            consecutive_blocked = 0

            # ---------- 4b. Stage4 repair budget: refuse, don't execute ----------
            # Repair returns `completed_with_warnings` even when it changes
            # nothing, so executing it again is free from the model's point of
            # view and it will keep asking. Declining the call is what breaks
            # the loop; the flags let it wrap up instead.
            if tool_name == _STAGE4_REPAIR_TOOL and stage4_repairs >= self.max_stage4_repairs:
                self._accept_current_quality()
                stage4_refusals += 1
                await self._refuse_tool(
                    run_id, trace_path, step_idx, phase, tool_name, tool_args,
                    reason=(
                        f"修复次数已达上限 {self.max_stage4_repairs} 次，"
                        f"剩余告警不再修复，当前质量按通过处理"
                        f"（第 {stage4_refusals}/{DEFAULT_MAX_STAGE4_REFUSALS} 次拒绝）"
                    ),
                    hint="质检与修复已收尾，请调用 Tool_StitchVideo 完成拼接",
                )
                # Refusals are cheap but not free — each one costs a model call.
                # Dragging to `max_same_tool` would also bury the real cause
                # under a generic loop message.
                if stage4_refusals >= DEFAULT_MAX_STAGE4_REFUSALS:
                    terminated_reason = (
                        f"Stage4 修复预算（{self.max_stage4_repairs} 次）已用尽，"
                        f"模型仍连续 {stage4_refusals} 次请求修复而不收尾，停止执行。"
                        f"镜头素材与渲染均已完成，可在编辑器中直接拼接成片"
                    )
                    break
                continue

            if tool_calls_used >= self.max_tool_calls:
                terminated_reason = (
                    f"工具调用总数已达上限 max_tool_calls={self.max_tool_calls}，"
                    f"模型未完成收尾，停止执行"
                )
                break

            # ---------- 5. execute ----------
            tool_calls_used += 1
            await self._wait_if_paused(run_id)
            flags_before = self._completion_flags()
            tool_output = await self._dispatch_tool(tool_name, tool_args, state, run_id)
            if isinstance(tool_output, dict):
                # Still applied on failure: a partially-successful batch tool
                # writes real assets into state and we want to keep them.
                self.update_state_from_respond(tool_name, tool_output)
                self._apply_tool_output(tool_name, tool_output, state)
            else:
                tool_output = {"status": "done"}

            tool_status = str(tool_output.get("status") or "")
            tool_error = str(tool_output.get("error") or "").strip()

            if tool_status == "error":
                # Both state-application paths mark the stage complete without
                # looking at status, so a crashed tool would otherwise leave the
                # model reading "visual_audio done" next to an empty asset map.
                restored = self._restore_completion_flags(flags_before)
                if restored:
                    logger.warning("%s 失败，撤销被误置的完成标记: %s", tool_name, restored)
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "level": "warn",
                        "text": f"  {tool_name} 失败，已撤销误置的完成标记: {', '.join(restored)}",
                    })

            self._record_agent_history(tool_name, tool_output)
            results["stages_completed"].append(f"{step_idx:02d}:{tool_name}")

            self._append_trace(trace_path, {
                "step": step_idx,
                "tool_name": tool_name,
                "arguments": tool_args,
                "output": tool_output,
                "raw_prediction": (getattr(prediction, "raw_output", "") or "")[:2000],
                "completion_tokens": getattr(prediction, "completion_tokens", 0),
                "timestamp": datetime.now().isoformat(),
            })

            await self.broadcast(run_id, {
                "type": "tool_result",
                "stage": phase,
                "step": step_idx,
                "tool_name": tool_name,
                "status": tool_output.get("status", "unknown"),
                "observation": self.last_observation,
            })
            await self.broadcast(run_id, {
                "type": "preview_update",
                "stage": phase,
                "preview": self._build_preview(state),
            })
            await self._emit_progress(run_id, state, phase)

            # ---------- 5b. repeated identical calls ----------
            # Status-independent on purpose. The Stage4 repair loop answers
            # `completed_with_warnings`, so keying this off `error` let the model
            # re-send one byte-identical repair call indefinitely while the
            # failure counters were cleared on every pass.
            try:
                args_sig = "%s::%s" % (
                    tool_name,
                    json.dumps(tool_args, sort_keys=True, ensure_ascii=False, default=str),
                )
            except (TypeError, ValueError):
                args_sig = f"{tool_name}::{tool_args!r}"
            repeat_calls[args_sig] = repeat_calls.get(args_sig, 0) + 1
            if repeat_calls[args_sig] >= self.max_identical_calls:
                # Stage4 judge frequently returns `completed_with_warnings` with
                # unchanged args while still being "successful". Hard-stopping
                # here causes avoidable agent_loop failures right before stitch.
                # Give the model an explicit wrap-up hint first, then cap stalls.
                if (
                    tool_name == _STAGE4_JUDGE_TOOL
                    and tool_status in {"completed_with_warnings", "success"}
                ):
                    stage4_judge_stalls += 1
                    self._accept_current_quality()
                    self._set_feedback({
                        "status": "quality_checked_with_warnings",
                        "tool": tool_name,
                        "hint": (
                            "Stage4 判定已多次重复且未收敛。"
                            "当前质量按通过处理，请优先调用 Tool_StitchVideo 收尾；"
                            "若必须修复，仅针对失败镜头做一次最小修复后立即收尾。"
                        ),
                    })
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "level": "warn",
                        "text": (
                            f"  {tool_name} 同参重复 {self.max_identical_calls} 次且仍为 "
                            f"{tool_status}，判定为 Stage4 停滞，"
                            "已按带告警通过并提示收尾"
                            f"（第 {stage4_judge_stalls}/{DEFAULT_MAX_STAGE4_REFUSALS} 次）"
                        ),
                    })
                    # Reset this exact signature counter once we have fed back
                    # the stop-hint; otherwise the next identical judge call
                    # would immediately re-trigger before the model can switch.
                    repeat_calls[args_sig] = 0
                    if stage4_judge_stalls >= DEFAULT_MAX_STAGE4_REFUSALS:
                        terminated_reason = (
                            f"Stage4 判定连续 {stage4_judge_stalls} 次停滞仍未收尾，停止执行。"
                            "镜头素材与渲染已完成，可在编辑器中直接拼接成片。"
                        )
                        break
                    continue
                terminated_reason = (
                    f"{tool_name} 已被以完全相同的参数执行 {repeat_calls[args_sig]} 次，"
                    f"返回状态为 {tool_status or 'unknown'}，模型未从结果中获得新信息，"
                    f"判定为死循环，停止执行"
                )
                if last_error:
                    terminated_reason += f"。该工具最近一次失败原因: {last_error}"
                break

            # ---------- 5c. a tool that keeps failing the same way ----------
            # A dead service fails identically no matter what the model passes,
            # so letting it burn the whole repeat budget only delays the report
            # and buries the cause under a generic "死循环".
            if tool_status == "error":
                sig = f"{tool_name}::{tool_error[:200]}"
                identical_failures[sig] = identical_failures.get(sig, 0) + 1
                last_error = tool_error or f"{tool_name} 执行失败"
                await self.broadcast(run_id, {
                    "type": "console_log",
                    "level": "warn",
                    "text": (
                        f"  {tool_name} 失败 "
                        f"({identical_failures[sig]}/{self.max_identical_failures}): "
                        f"{_trunc_text(last_error, 200)}"
                    ),
                })
                if identical_failures[sig] >= self.max_identical_failures:
                    terminated_reason = (
                        f"{tool_name} 连续 {identical_failures[sig]} 次以相同原因失败，"
                        f"判定为环境故障而非模型决策问题，停止执行。原因: {last_error}"
                    )
                    break
            else:
                identical_failures.clear()

            # ---------- 5d. Stage4 quality loop ceilings ----------
            # Judge and repair can alternate forever when quality never fully
            # passes (hallucinated shot ids, warnings repair cannot fix). Two
            # ceilings: one on repairs alone, one on the combined loop.
            if tool_name in _STAGE4_TOOLS:
                stage4_attempts += 1
                if tool_name == _STAGE4_REPAIR_TOOL:
                    stage4_repairs += 1
                    if stage4_repairs >= self.max_stage4_repairs:
                        self._accept_current_quality()
                        await self.broadcast(run_id, {
                            "type": "console_log",
                            "level": "warn",
                            "text": (
                                f"  Stage4 修复已用完 {stage4_repairs}/"
                                f"{self.max_stage4_repairs} 次预算，"
                                "后续修复调用将被拒绝，请收尾"
                            ),
                        })
                if stage4_attempts >= self.max_stage4_attempts:
                    logger.warning(
                        "Stage4 质量循环已达 %d 次上限，跳过质检继续执行", stage4_attempts
                    )
                    self._accept_current_quality()
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "level": "warn",
                        "text": (
                            f"  Stage4 质检+修复合计已达 {stage4_attempts} 次上限，"
                            "质检无法完全通过，跳过继续生成"
                        ),
                    })

            # ---------- 6. terminate on the model's own stitch ----------
            if tool_name == "Tool_StitchVideo":
                final_path = str(state.get("final_video_path", "") or "")
                if final_path:
                    self.stitched_done = True
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "level": "success",
                        "text": f"模型自主完成收尾，成片: {final_path}",
                    })
                    break

                stitch = tool_output if isinstance(tool_output, dict) else {}
                detail = f"拼接未产出成片 (concat_status={stitch.get('concat_status', 'unknown')})"
                errors = stitch.get("validation_errors") or []
                if isinstance(errors, list) and errors:
                    reasons = [
                        f"{e.get('shot_id', '?')}: {','.join(e.get('reasons', []))}"
                        for e in errors[:5] if isinstance(e, dict)
                    ]
                    detail += " · " + "; ".join(reasons)
                # Report and hand control back: the model may repair and retry.
                self._set_feedback({
                    "status": "stitch_failed",
                    "message": detail,
                    "hint": "定位失败镜头并修复后可再次尝试收尾",
                })
                results["errors"].append(detail)
                await self.broadcast(run_id, {
                    "type": "error", "stage": "final_stitch", "message": detail,
                })
        else:
            terminated_reason = (
                f"达到步数上限 max_steps={self.max_agent_steps}，模型未完成收尾"
            )

        if terminated_reason:
            results["errors"].append(terminated_reason)
            await self.broadcast(run_id, {
                "type": "error", "stage": "agent_loop", "message": terminated_reason,
            })

        results["tool_calls_used"] = tool_calls_used
        results["stage4_repairs"] = stage4_repairs
        results["final_video_path"] = str(state.get("final_video_path", "") or "")
        self._write_final_state(project_dir, project_id, input_script, state, results)

        await self._emit_progress(run_id, state, "done")
        await self.broadcast(run_id, {
            "type": "pipeline_complete",
            "status": "completed" if not results["errors"] else "completed_with_errors",
            "results": results,
            "preview": self._build_preview(state),
        })
        return results

    # ------------------------------------------------------------------
    def _write_final_state(
        self,
        project_dir: Path,
        project_id: str,
        input_script: str,
        state: dict[str, Any],
        results: dict[str, Any],
    ) -> None:
        """Persist the editable project. A write failure must not fail the run."""
        payload = {
            "schema_version": 1,
            "project_id": project_id,
            "input_text": input_script,
            "segments": state.get("segments", []),
            "shot_plan": state.get("shot_plan", []),
            "shot_assets": state.get("shot_assets", {}),
            "bgm_tracks": state.get("bgm_tracks", []),
            "stitch_result": state.get("stitch_result", {}),
            "music_style": state.get("music_style", ""),
            "final_video_path": state.get("final_video_path", ""),
            "stages_completed": results.get("stages_completed", []),
            "errors": results.get("errors", []),
            "agent_mode": self.agent_mode,
            "autonomous": self.fully_autonomous,
            "saved_at": datetime.now().isoformat(),
        }
        try:
            (project_dir / "final_state.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover - disk failure path
            logger.warning("final_state.json 落盘失败（不影响推理结果）: %s", exc)
