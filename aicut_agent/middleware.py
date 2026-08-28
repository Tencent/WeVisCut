"""Guardrail middleware for the AICut deep agent.

A faithful port of the cost bounds in ``AutonomousOrchestrator.run_autonomous``.
The philosophy is unchanged: the model's decisions are never rewritten — invalid
or over-budget calls are *refused* and reported back through the observation
channel the policy was trained to read, and only hard cost ceilings *terminate*
the run (by routing the tool node to ``END`` instead of back to the model).

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
                "完全自主模式：不做工具改写 / 不强制收尾，非法调用只作为 observation 回灌给模型"
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
        result = await handler(request)
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
        """End the loop when the model's own stitch produced a final video."""
        rt = self._rt
        final_path = str(rt.state.get("final_video_path", "") or "")
        if final_path:
            rt.orchestrator.stitched_done = True
            rt.stitch_succeeded = True
            await rt.broadcast({
                "type": "console_log", "level": "success",
                "text": f"模型自主完成收尾，成片: {final_path}",
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
        """Stop the run at a cost ceiling: route to END with a terminal message."""
        rt = self._rt
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
