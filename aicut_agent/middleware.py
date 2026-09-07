"""Guardrail middleware for the AICut deep agent.

A faithful port of the cost bounds in ``AutonomousOrchestrator.run_autonomous``.
Decisions the model is still capable of making are left to it: invalid or
over-budget calls are *refused* and reported back through the observation
channel the policy was trained to read, rather than being rewritten.

The exception is a decision it has demonstrably stopped making. Re-running a
step whose product is already in state cannot change anything, so rather than
spend the run's budget on it and stop at a ceiling with usable footage in hand,
the harness takes the next required step itself. A repeat that follows a failed
check is not this case and is never touched — see
``_next_step_for_dead_repeat``.

Hook layout:

* ``before_agent`` — emit the run's opening events (stage_update + progress).
* ``before_model`` — honour pause / cancel before each model call.
* ``after_model`` — broadcast the model's "thinking" event for the frontend.
* ``wrap_tool_call`` — precondition blocking, repeat/failure/Stage4 budgets,
  tool-call ceiling, and stitch termination.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.graph import END
from langgraph.types import Command

from studio.config import install_import_paths

install_import_paths()

from server.infer_graph import (  # noqa: E402
    LONG_HORIZON_ACTION_SPACE,
    LONG_HORIZON_PHASE_LABELS,
)

logger = logging.getLogger("aicut_online.middleware")

_STAGE4_REPAIR_TOOL = "Tool_Stage4DimensionalRepair"
_STAGE4_JUDGE_TOOL = "Tool_Stage4DimensionJudge"
_STAGE4_TOOLS = frozenset({
    _STAGE4_JUDGE_TOOL, _STAGE4_REPAIR_TOOL, "Tool_TextQuickJudge",
})


def _trunc_text(text: str, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + "…"


class AICutGuardrailMiddleware(AgentMiddleware):
    """Per-run guardrails. ``runtime`` here is the AICutRuntime (bound by closure)."""

    def __init__(self, aicut_runtime: Any) -> None:
        super().__init__()
        self._rt = aicut_runtime

    # ------------------------------------------------------------------
    # run lifecycle events
    # ------------------------------------------------------------------
    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        rt = self._rt
        await rt.broadcast({
            "type": "stage_update",
            "stage": "agent_loop",
            "status": "started",
            "message": f"Agent 自主执行启动 · {rt.project_id}",
            "allowed_tools": LONG_HORIZON_ACTION_SPACE,
            "progress_pct": 0,
        })
        await rt.broadcast({
            "type": "console_log",
            "level": "system",
            "text": (
                "完全自主模式：非法调用只作为 observation 回灌给模型，由模型自行决策下一步"
                if rt.orchestrator.fully_autonomous
                else "兼容模式：保留参考实现的路由校正"
            ),
        })
        await rt.emit_progress("agent_loop")
        return None

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        # Honour pause / cancel before spending a model call.
        await self._rt.orchestrator._wait_if_paused(self._rt.run_id)
        return None

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        # Broadcast the model's reasoning for the frontend "thinking" panel.
        rt = self._rt
        prediction = getattr(rt, "last_prediction", None)
        if prediction is not None:
            try:
                await rt.broadcast_thinking(prediction, getattr(rt, "last_tool_name", ""))
            except Exception:
                logger.debug("broadcast thinking failed", exc_info=True)
        return None

    # ------------------------------------------------------------------
    # the guardrailed tool call
    # ------------------------------------------------------------------
    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        rt = self._rt
        tool_name = str(request.tool_call.get("name", "") or "")
        tool_args = request.tool_call.get("args") or {}
        if not isinstance(tool_args, dict):
            tool_args = {}

        # ---- 0. fixed workflow, ahead of every event this method emits ----
        corrected = self._next_step_for_fixed_workflow(tool_name)
        if corrected is not None:
            tool_name, tool_args = corrected
        else:
            corrected = self._next_step_for_dead_repeat(tool_name, tool_args)
            if corrected is not None:
                tool_name, tool_args = corrected
        phase = rt.phase_for_tool(tool_name)

        # ---- 3. repeat ceiling (cost bound, not a redirect) ----
        reason = rt.note_tool_selected(tool_name)
        if reason:
            return await self._terminate(request, reason)

        # ---- stage_update + tool_call events ----
        await rt.broadcast({
            "type": "stage_update",
            "stage": phase,
            "status": "started",
            "message": LONG_HORIZON_PHASE_LABELS.get(phase, phase),
            "progress_pct": self._overall_progress(),
        })
        missing = rt.check_preconditions(tool_name)
        await rt.broadcast({
            "type": "tool_call",
            "stage": phase,
            "step": rt.step_idx,
            "tool_name": tool_name,
            "arguments": tool_args,
            "precondition_missing": missing,
        })

        # ---- 4. preconditions: report only, do not execute ----
        if missing:
            return await self._block_precondition(request, tool_name, tool_args, phase, missing)

        # ---- 4b. Stage4 repair budget: refuse, don't execute ----
        if tool_name == _STAGE4_REPAIR_TOOL and rt.stage4_repairs >= rt.max_stage4_repairs:
            return await self._refuse_repair(request, tool_name, tool_args, phase)

        # ---- 5. total tool-call ceiling ----
        if rt.tool_calls_used >= rt.max_tool_calls:
            return await self._terminate(
                request,
                f"工具调用总数已达上限 max_tool_calls={rt.max_tool_calls}，模型未完成收尾，停止执行",
            )

        # ---- 5. execute ----
        rt.tool_calls_used += 1
        if corrected is None:
            result = await handler(request)
        else:
            # `handler` runs the tool object LangGraph already resolved from the
            # model's chosen name, so a corrected call cannot go through it. This
            # is the same entry point the tool wrapper uses.
            await rt.execute_tool(tool_name, tool_args)
            result = self._tool_message(
                request, rt.last_tool_output if isinstance(rt.last_tool_output, dict) else {})
        output = rt.last_tool_output if isinstance(rt.last_tool_output, dict) else {}
        tool_status = str(output.get("status") or "")
        tool_error = str(output.get("error") or "").strip()

        # ---- post-execution events ----
        await rt.broadcast({
            "type": "tool_result",
            "stage": phase,
            "step": rt.step_idx,
            "tool_name": tool_name,
            "status": output.get("status", "unknown"),
            "observation": rt.orchestrator.last_observation,
        })
        await rt.broadcast({
            "type": "preview_update",
            "stage": phase,
            "preview": rt.orchestrator._build_preview(rt.state),
        })
        await rt.emit_progress(phase)

        # ---- 5b/5c/5d. post-execution ceilings ----
        term = await self._post_execution_checks(request, tool_name, tool_args, tool_status, tool_error)
        if term is not None:
            return term

        # ---- 6. terminate on the model's own stitch ----
        if tool_name == "Tool_StitchVideo":
            stitch_cmd = await self._maybe_finish_on_stitch(request, output)
            if stitch_cmd is not None:
                return stitch_cmd

        return result

    # ------------------------------------------------------------------
    # self-correction
    # ------------------------------------------------------------------
    def _next_step_for_fixed_workflow(self, tool_name: str) -> tuple[str, dict[str, Any]] | None:
        """Replace out-of-order choices with the canonical next runnable step.

        The model is still useful for producing content and arguments, but the
        long-horizon workflow order is deterministic now: segment, plan, reflect,
        asset, decorate, check, render, music, stitch. A premature stitch or a
        repeated repair should never be shown as a blocked tool call; it should
        silently become the step that can actually advance the run.
        """
        rt = self._rt
        orch = rt.orchestrator
        try:
            nxt = orch.next_required_tool(rt.state)
        except Exception:
            logger.debug("fixed-workflow probe failed", exc_info=True)
            return None
        nxt = str(nxt or "")
        if not nxt or nxt == tool_name or nxt not in LONG_HORIZON_ACTION_SPACE:
            return None
        if rt.check_preconditions(nxt):
            return None

        rt.silent_corrections += 1
        rt.repeat_calls.clear()
        rt.same_tool_count = 0
        rt.last_tool = ""
        rt.consecutive_blocked = 0
        logger.debug("fixed workflow: %s -> %s", tool_name, nxt)
        return nxt, self._default_args_for(nxt)

    def _next_step_for_dead_repeat(
        self, tool_name: str, tool_args: dict[str, Any],
    ) -> tuple[str, dict[str, Any]] | None:
        """The step to run instead, when this call cannot advance the run.

        Returns ``None`` — leave the model's choice alone — for anything that
        might still be productive, which includes every repeat driven by a
        check. Those are exactly the cases ``_agent_tool_already_done`` reports
        as not-done: a pending reflect generation, open issues from a patch, or
        shots the Stage4 judge rejected. So a re-render after a failed check
        never reaches the rewrite here; only a call whose work is already in
        state does.
        """
        rt = self._rt
        orch = rt.orchestrator
        if tool_name not in LONG_HORIZON_ACTION_SPACE:
            return None            # unknown names have their own feedback path
        if rt.silent_corrections >= rt.max_silent_corrections:
            return None

        # First attempts are always allowed through: a tool reporting its own
        # no-op is information the model may well act on. Only once it has
        # asked twice is it not reading the answer.
        seen_before = (
            tool_name == rt.last_tool
            or rt.repeat_calls.get(rt.args_signature(tool_name, tool_args), 0) >= 1
        )
        if not seen_before:
            return None

        try:
            if not orch._agent_tool_already_done(tool_name, rt.state):
                return None
            nxt = orch.next_required_tool(rt.state)
        except Exception:
            # Never let a probe break a run that would otherwise proceed.
            logger.debug("self-correction probe failed", exc_info=True)
            return None
        if not nxt or nxt == tool_name:
            return None
        if rt.check_preconditions(nxt):
            return None            # would trade one dead end for another

        rt.silent_corrections += 1
        # The ceilings exist to catch a model going nowhere. It is going
        # somewhere now, so it should not inherit a budget spent while stuck.
        rt.repeat_calls.clear()
        rt.same_tool_count = 0
        rt.last_tool = ""
        rt.consecutive_blocked = 0
        # Debug, not info: a normal run's log should read as a clean sequence.
        logger.debug(
            "self-correction %d/%d: %s already done, running %s instead",
            rt.silent_corrections, rt.max_silent_corrections, tool_name, nxt,
        )
        return nxt, self._default_args_for(nxt)

    def _next_step_after(self, tool_name: str) -> str:
        """The next runnable step, or "" if there is none to point at."""
        rt = self._rt
        try:
            nxt = rt.orchestrator.next_required_tool(rt.state)
        except Exception:
            logger.debug("next-step probe failed", exc_info=True)
            return ""
        if not nxt or nxt == tool_name or rt.check_preconditions(nxt):
            return ""
        return nxt

    def _default_args_for(self, tool_name: str) -> dict[str, Any]:
        """Arguments for a step the harness chose rather than the model.

        Empty is right for the batch tools: each one picks up whatever is still
        pending in state. Stitching is the exception — it reads the timeline
        and BGM it is handed.
        """
        if tool_name != "Tool_StitchVideo":
            return {}
        state = self._rt.state
        return {
            "project_id": state.get("project_id", ""),
            "timeline_path": state.get("timeline_path", ""),
            "bgm": state.get("bgm_result", {}),
        }

    # ------------------------------------------------------------------
    # guardrail helpers
    # ------------------------------------------------------------------
    def _overall_progress(self) -> int:
        from studio import tracks
        return tracks.build_overall_progress(self._rt.state)

    async def _block_precondition(
        self, request: Any, tool_name: str, tool_args: dict[str, Any], phase: str, missing: list[str],
    ) -> ToolMessage | Command:
        rt = self._rt
        rt.consecutive_blocked += 1
        tool_output = {
            "status": "blocked_precondition",
            "missing": missing,
            "message": f"{tool_name} 前置条件不足: {', '.join(missing)}",
        }
        rt.orchestrator._record_agent_history(tool_name, tool_output)
        rt.set_feedback({
            "status": "blocked_precondition",
            "tool": tool_name,
            "missing": missing,
            "hint": "请先满足上述前置条件，或改选一个当前可执行的工具",
        })
        await rt.broadcast({
            "type": "console_log", "level": "warn",
            "text": (
                f"{tool_name} 前置条件不足（缺 {', '.join(missing)}），"
                f"不做重定向，交回模型决策 ({rt.consecutive_blocked}/{rt.max_consecutive_blocked})"
            ),
        })
        await rt.broadcast({
            "type": "tool_result", "stage": phase, "step": rt.step_idx,
            "tool_name": tool_name, "status": "blocked_precondition",
            "observation": rt.orchestrator.last_observation,
        })
        rt.orchestrator._append_trace(rt.trace_path, {
            "step": rt.step_idx, "tool_name": tool_name, "arguments": tool_args,
            "output": tool_output, "blocked": True, "timestamp": self._now(),
        })
        if rt.consecutive_blocked >= rt.max_consecutive_blocked:
            return await self._terminate(
                request, f"模型连续 {rt.consecutive_blocked} 次选择前置条件不满足的工具，停止执行",
            )
        return self._tool_message(request, tool_output)

    async def _refuse_repair(
        self, request: Any, tool_name: str, tool_args: dict[str, Any], phase: str,
    ) -> ToolMessage | Command:
        rt = self._rt
        rt.orchestrator._accept_current_quality()
        rt.stage4_refusals += 1
        reason = (
            f"修复次数已达上限 {rt.max_stage4_repairs} 次，剩余告警不再修复，当前质量按通过处理"
            f"（第 {rt.stage4_refusals}/{rt.max_stage4_refusals} 次拒绝）"
        )
        tool_output = {"status": "budget_exhausted", "tool": tool_name, "message": reason}
        rt.orchestrator._record_agent_history(tool_name, tool_output)
        rt.set_feedback({
            "status": "budget_exhausted", "tool": tool_name,
            "reason": reason, "hint": "质检与修复已收尾，请调用 Tool_StitchVideo 完成拼接",
        })
        await rt.broadcast({"type": "console_log", "level": "warn", "text": f"  拒绝执行 {tool_name}: {reason}"})
        await rt.broadcast({
            "type": "tool_result", "stage": phase, "step": rt.step_idx,
            "tool_name": tool_name, "status": "budget_exhausted",
            "observation": rt.orchestrator.last_observation,
        })
        rt.orchestrator._append_trace(rt.trace_path, {
            "step": rt.step_idx, "tool_name": tool_name, "arguments": tool_args,
            "output": tool_output, "refused": True, "timestamp": self._now(),
        })
        if rt.stage4_refusals >= rt.max_stage4_refusals:
            return await self._terminate(
                request,
                f"Stage4 修复预算（{rt.max_stage4_repairs} 次）已用尽，模型仍连续 {rt.stage4_refusals} 次请求修复而不收尾，停止执行。"
                "镜头素材与渲染均已完成，可在编辑器中直接拼接成片",
            )
        return self._tool_message(request, tool_output)

    async def _post_execution_checks(
        self, request: Any, tool_name: str, tool_args: dict[str, Any], tool_status: str, tool_error: str,
    ) -> Command | None:
        """Identical-call / identical-failure / Stage4 ceilings. Returns a Command to END, or None."""
        rt = self._rt

        # 5b. repeated identical calls (status-independent on purpose).
        args_sig = rt.args_signature(tool_name, tool_args)
        rt.repeat_calls[args_sig] = rt.repeat_calls.get(args_sig, 0) + 1
        if rt.repeat_calls[args_sig] >= rt.max_identical_calls:
            if tool_name == _STAGE4_JUDGE_TOOL and tool_status in {"completed_with_warnings", "success"}:
                rt.stage4_judge_stalls += 1
                rt.orchestrator._accept_current_quality()
                rt.set_feedback({
                    "status": "quality_checked_with_warnings",
                    "tool": tool_name,
                    "hint": (
                        "Stage4 判定已多次重复且未收敛。当前质量按通过处理，请优先调用 Tool_StitchVideo 收尾；"
                        "若必须修复，仅针对失败镜头做一次最小修复后立即收尾。"
                    ),
                })
                await rt.broadcast({
                    "type": "console_log", "level": "warn",
                    "text": (
                        f"  {tool_name} 同参重复 {rt.max_identical_calls} 次且仍为 {tool_status}，"
                        f"判定为 Stage4 停滞，已按带告警通过并提示收尾"
                        f"（第 {rt.stage4_judge_stalls}/{rt.max_stage4_refusals} 次）"
                    ),
                })
                rt.repeat_calls[args_sig] = 0
                if rt.stage4_judge_stalls >= rt.max_stage4_refusals:
                    return await self._terminate(
                        request,
                        f"Stage4 判定连续 {rt.stage4_judge_stalls} 次停滞仍未收尾，停止执行。"
                        "镜头素材与渲染已完成，可在编辑器中直接拼接成片。",
                    )
                return None
            # Stopping here loses a run that still has somewhere to go. Point at
            # the next step instead; if the model ignores it, the pre-execution
            # hook takes the step itself on the following turn.
            nxt = self._next_step_after(tool_name)
            if nxt:
                rt.repeat_calls[args_sig] = 0
                rt.set_feedback({
                    "status": "already_completed",
                    "tool": tool_name,
                    "hint": f"该步骤的产物已在 global_state 中，无需重复执行，请调用 {nxt} 继续",
                    "next_tool": nxt,
                })
                return None
            reason = (
                f"{tool_name} 已被以完全相同的参数执行 {rt.repeat_calls[args_sig]} 次，"
                f"返回状态为 {tool_status or 'unknown'}，模型未从结果中获得新信息，判定为死循环，停止执行"
            )
            if rt.last_error:
                reason += f"。该工具最近一次失败原因: {rt.last_error}"
            return await self._terminate(request, reason)

        # 5c. a tool that keeps failing the same way.
        if tool_status == "error":
            sig = f"{tool_name}::{tool_error[:200]}"
            rt.identical_failures[sig] = rt.identical_failures.get(sig, 0) + 1
            rt.last_error = tool_error or f"{tool_name} 执行失败"
            await rt.broadcast({
                "type": "console_log", "level": "warn",
                "text": (
                    f"  {tool_name} 失败 ({rt.identical_failures[sig]}/{rt.max_identical_failures}): "
                    f"{_trunc_text(rt.last_error, 200)}"
                ),
            })
            if rt.identical_failures[sig] >= rt.max_identical_failures:
                return await self._terminate(
                    request,
                    f"{tool_name} 连续 {rt.identical_failures[sig]} 次以相同原因失败，"
                    f"判定为环境故障而非模型决策问题，停止执行。原因: {rt.last_error}",
                )
        else:
            rt.identical_failures.clear()

        # 5d. Stage4 quality loop ceilings.
        if tool_name in _STAGE4_TOOLS:
            rt.stage4_attempts += 1
            if tool_name == _STAGE4_REPAIR_TOOL:
                rt.stage4_repairs += 1
                if rt.stage4_repairs >= rt.max_stage4_repairs:
                    rt.orchestrator._accept_current_quality()
                    await rt.broadcast({
                        "type": "console_log", "level": "warn",
                        "text": (
                            f"  Stage4 修复已用完 {rt.stage4_repairs}/{rt.max_stage4_repairs} 次预算，"
                            "后续修复调用将被拒绝，请收尾"
                        ),
                    })
            if rt.stage4_attempts >= rt.max_stage4_attempts:
                rt.orchestrator._accept_current_quality()
                await rt.broadcast({
                    "type": "console_log", "level": "warn",
                    "text": (
                        f"  Stage4 质检+修复合计已达 {rt.stage4_attempts} 次上限，"
                        "质检无法完全通过，跳过继续生成"
                    ),
                })
        return None

    async def _maybe_finish_on_stitch(self, request: Any, output: dict[str, Any]) -> Command | None:
        """End the loop once the model's stitch validated the cut.

        With ``render.defer_stitch`` on, that means a manifest rather than a
        file: the expensive concat runs on export. Either outcome is a
        successful finish as far as the model is concerned.
        """
        rt = self._rt
        final_path = str(rt.state.get("final_video_path", "") or "")
        pending = bool(rt.state.get("stitch_pending"))
        if final_path or pending:
            rt.orchestrator.stitched_done = True
            rt.stitch_succeeded = True
            await rt.broadcast({
                "type": "console_log", "level": "success",
                "text": (f"模型自主完成收尾，成片: {final_path}" if final_path
                         else "模型自主完成收尾，各镜头已校验通过，可直接预览或导出成片"),
            })
            return Command(
                goto=END,
                update={"messages": [self._tool_message(request, output)]},
            )

        # Stitch ran but produced no video: report and hand control back.
        detail = f"拼接未产出成片 (concat_status={output.get('concat_status', 'unknown')})"
        errors = output.get("validation_errors") or []
        if isinstance(errors, list) and errors:
            reasons = [
                f"{e.get('shot_id', '?')}: {','.join(e.get('reasons', []))}"
                for e in errors[:5] if isinstance(e, dict)
            ]
            detail += " · " + "; ".join(reasons)
        rt.set_feedback({
            "status": "stitch_failed",
            "message": detail,
            "hint": "定位失败镜头并修复后可再次尝试收尾",
        })
        rt.results["errors"].append(detail)
        await rt.broadcast({"type": "error", "stage": "final_stitch", "message": detail})
        return None

    # ------------------------------------------------------------------
    # termination / message helpers
    # ------------------------------------------------------------------
    async def _terminate(self, request: Any, reason: str) -> Command:
        """Stop the run at a cost ceiling: route to END with a terminal message.

        A ceiling says the model stopped making progress, not that the footage
        is unusable — so the cut gets delivered first if it can be.
        """
        rt = self._rt
        if await rt.deliver_if_possible():
            logger.debug("ceiling hit (%s), but the cut was deliverable", reason)
            output = rt.last_tool_output if isinstance(rt.last_tool_output, dict) else {}
            return Command(goto=END, update={"messages": [self._tool_message(request, output)]})
        rt.terminated_reason = reason
        rt.results["errors"].append(reason)
        await rt.broadcast({"type": "error", "stage": "agent_loop", "message": reason})
        tool_msg = self._tool_message(request, {
            "status": "terminated", "message": reason,
        })
        return Command(goto=END, update={"messages": [tool_msg]})

    @staticmethod
    def _tool_message(request: Any, output: dict[str, Any]) -> ToolMessage:
        import json
        try:
            content = json.dumps(output, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            content = str(output)
        return ToolMessage(
            content=content,
            name=str(request.tool_call.get("name", "") or ""),
            tool_call_id=str(request.tool_call.get("id", "") or ""),
        )

    @staticmethod
    def _now() -> str:
        from datetime import datetime
        return datetime.now().isoformat()
