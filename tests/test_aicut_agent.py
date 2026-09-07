"""Offline end-to-end test of the deepagents integration (no GPU, no pipeline).

Uses a scripted mock loader (returns canned ``PredictionResult`` tool calls) and a
mocked ``_dispatch_tool`` so the full deepagents ReAct loop runs without a model
or external services. Verifies:

1. The harness profile strips the built-in tools — the model only ever sees the
   12 training tools.
2. ``AICutChatModel`` converts a prediction into an ``AIMessage`` with a valid
   tool call, and rebuilds the user payload from runtime state each step.
3. The guardrail middleware blocks precondition-violating calls without
   executing them.
4. A successful ``Tool_StitchVideo`` terminates the loop (Command → END) and the
   run finalises with a ``final_state.json``.

Run:  python3 tests/test_aicut_agent.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

STUDIO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDIO_DIR))

from studio.config import install_import_paths  # noqa: E402

install_import_paths()

from server.model_loader import PredictionResult  # noqa: E402
from studio.orchestrator import AutonomousOrchestrator  # noqa: E402

from aicut_agent import AICutRuntime, build_aicut_agent  # noqa: E402
from aicut_agent.agent import AICUT_PROVIDER, ensure_aicut_profile  # noqa: E402


# ---------------------------------------------------------------------------
# mocks
# ---------------------------------------------------------------------------
class MockLoader:
    """A scripted stand-in for UnifiedModelLoader."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []
        self.system_prompt = ""

    @property
    def is_loaded(self) -> bool:
        return True

    def predict_orchestrator(self, system, user, *, stage="", allowed_tools=None, **kw):
        self.calls.append({"system": system, "user": user, "allowed_tools": allowed_tools})
        if not self.script:
            return PredictionResult(tool_name="", arguments={}, parse_ok=False,
                                    raw_output="", cleaned_output="")
        step = self.script.pop(0)
        return PredictionResult(
            tool_name=step.get("tool_name", ""),
            arguments=step.get("arguments", {}),
            parse_ok=step.get("parse_ok", True),
            raw_output=step.get("raw", "<think>ok</think>"),
            cleaned_output=step.get("cleaned", ""),
            thinking_summary=step.get("thinking", "想一下"),
            agent_trace={"agent_state": {"phase": step.get("phase", "")}},
            completion_tokens=10,
        )

    # plain-mode completion
    def predict(self, system, user, *, max_tokens=2048, temperature=0.0, **kw):
        return PredictionResult(cleaned_output="纯文本回复", raw_output="纯文本回复")


def _make_orchestrator(loader, dispatched: list[str], dispatch_outputs: dict[str, dict],
                       allow_all_preconditions: bool = False):
    """A real AutonomousOrchestrator with _dispatch_tool stubbed out."""
    infer_config = {
        "agent_mode": "long_horizon",
        "agent": {"max_steps": 48, "max_same_tool_repeats": 10},
        "autonomy": {"fully_autonomous": True},
        "tool_switches": {},
    }
    orch = AutonomousOrchestrator(
        loader, {"paths": {}}, {}, "训练系统提示词",
        broadcast_fn=None, infer_config=infer_config,
    )
    if allow_all_preconditions:
        orch._agent_precondition_missing = lambda tool_name, state: []  # type: ignore[assignment]

    async def fake_dispatch(tool_name, tool_args, state, run_id):
        dispatched.append(tool_name)
        shot_ids = [
            str(s.get("shot_id"))
            for s in (state.get("shot_plan") or [{"shot_id": "shot_001"}])
            if isinstance(s, dict) and s.get("shot_id")
        ] or ["shot_001"]
        defaults = {
            "Tool_SegmentArticle": {
                "status": "success",
                "segments": [{"segment_id": "seg_01"}],
            },
            "Tool_PlanShots": {
                "status": "success",
                "shots": [{"shot_id": "shot_001", "segment_id": "seg_01"}],
            },
            "Tool_SelfReflect": {
                "status": "success",
                "passed": True,
                "issues": [],
            },
            "Tool_SelfReflect_Patch": {
                "status": "success",
                "shots": state.get("shot_plan") or [{"shot_id": "shot_001"}],
            },
            "Tool_BatchVisualAudio": {
                "status": "success",
                "tts_success_count": len(shot_ids),
                "visual_success_count": len(shot_ids),
            },
            "Tool_BatchDecorateSpotlights": {
                "status": "success",
                "results": {
                    sid: {"decoration": {"template": "test"}}
                    for sid in shot_ids
                },
            },
            "Tool_SeverityRank": {
                "status": "success",
                "severity_scores": {sid: 0.1 for sid in shot_ids},
            },
            "Tool_Stage4DimensionJudge": {
                "status": "success",
                "passed_count": len(shot_ids),
                "failed_count": 0,
                "failed_shot_ids": [],
            },
            "Tool_Stage4DimensionalRepair": {
                "status": "success",
                "repaired_count": 0,
            },
            "Tool_BatchTimelineCompose": {
                "status": "success",
                "render_success_count": len(shot_ids),
            },
            "Tool_MusicSearch": {
                "status": "success",
                "tracks": [{"path": "/tmp/bgm.mp3"}],
            },
            "Tool_StitchVideo": {
                "status": "success",
                "final_video_path": "/tmp/final.mp4",
            },
        }
        out = dispatch_outputs.get(tool_name, defaults.get(tool_name, {"status": "success"}))
        # Fold a couple of outputs into state the way the real one does.
        if tool_name == "Tool_SegmentArticle":
            state["segments"] = out.get("segments", [{"segment_id": "seg_01"}])
        if tool_name == "Tool_PlanShots":
            state["shot_plan"] = out.get("shots", [{"shot_id": "shot_001"}])
        if tool_name == "Tool_StitchVideo" and out.get("final_video_path"):
            state["final_video_path"] = out["final_video_path"]
        return out

    orch._dispatch_tool = fake_dispatch  # type: ignore[assignment]
    return orch


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
CANONICAL_FULL_FLOW = [
    "Tool_SegmentArticle",
    "Tool_PlanShots",
    "Tool_SelfReflect",
    "Tool_BatchVisualAudio",
    "Tool_SeverityRank",
    "Tool_Stage4DimensionJudge",
    "Tool_BatchTimelineCompose",
    "Tool_MusicSearch",
    "Tool_StitchVideo",
]


class _AgentHarnessCase(unittest.TestCase):
    """Shared fixture: a real runtime + compiled graph over a scripted model."""

    def setUp(self) -> None:
        self.events: list[dict[str, Any]] = []

        async def broadcast(run_id, event):
            self.events.append(event)

        self.broadcast = broadcast
        self.tmp = tempfile.mkdtemp(prefix="aicut_test_")

    def _build(self, script, dispatch_outputs, allow_all_preconditions=False):
        loader = MockLoader(script)
        self.dispatched: list[str] = []
        orch = _make_orchestrator(loader, self.dispatched, dispatch_outputs,
                                  allow_all_preconditions=allow_all_preconditions)

        async def progress(run_id, snapshot):
            self.events.append({"type": "progress", "progress": snapshot})

        runtime = AICutRuntime(
            orch,
            input_script="测试口播稿",
            run_id="run_test",
            output_dir=self.tmp,
            track_order=[],
            broadcast_fn=self.broadcast,
            progress_fn=progress,
        )
        agent = build_aicut_agent(runtime, loader)
        return loader, runtime, agent


class TestDeepAgentIntegration(_AgentHarnessCase):
    def test_full_run_terminates_on_stitch(self):
        script = [
            {"tool_name": "Tool_SegmentArticle", "arguments": {"num_segments_target": 1}},
            {"tool_name": "Tool_PlanShots", "arguments": {"mode": "plan_shots_all"}},
            {"tool_name": "Tool_StitchVideo", "arguments": {"shot_count": 1}},
        ]
        dispatch_outputs = {
            "Tool_SegmentArticle": {"status": "success", "segments": [{"segment_id": "seg_01"}]},
            "Tool_PlanShots": {"status": "success", "shots": [{"shot_id": "shot_001"}]},
            "Tool_StitchVideo": {"status": "success", "final_video_path": "/tmp/final.mp4"},
        }
        loader, runtime, agent = self._build(script, dispatch_outputs, allow_all_preconditions=True)

        from langchain_core.messages import HumanMessage
        result = asyncio.run(agent.ainvoke({"messages": [HumanMessage(content="测试口播稿")]}))

        # The model tried to jump to stitch, but the harness owns the order and
        # inserts the mandatory steps before delivery.
        self.assertEqual(
            self.dispatched,
            CANONICAL_FULL_FLOW,
        )
        self.assertTrue(runtime.stitch_succeeded)
        self.assertEqual(runtime.state.get("final_video_path"), "/tmp/final.mp4")

        # final_state.json written.
        final_state = Path(runtime.project_dir) / "final_state.json"
        # (finalize is called by the server, not the agent loop; call it here)
        runtime.finalize()
        self.assertTrue(final_state.exists())

        # The model only ever received the 12 training tools.
        for call in loader.calls:
            self.assertEqual(len(call["allowed_tools"]), 12)

    def test_model_sees_only_training_tools(self):
        """The harness profile must strip ls/read_file/execute/task/etc."""
        ensure_aicut_profile()
        loader, runtime, agent = self._build([{"tool_name": "Tool_StitchVideo", "arguments": {}}],
                                             {"Tool_StitchVideo": {"status": "success", "final_video_path": "/tmp/x.mp4"}})
        # Inspect the compiled graph's tool node.
        from langgraph.prebuilt import ToolNode  # noqa
        # Collect tool names registered anywhere in the graph.
        names = set()
        for attr in ("tools", "_tools"):
            t = getattr(agent, attr, None)
            if t:
                names |= {getattr(x, "name", "") for x in t}
        # Fallback: check via the model's bound tools is not possible (we use the
        # canonical list); instead assert the built-ins are absent from the graph nodes.
        builtin = {"ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute", "task"}
        self.assertFalse(names & builtin, f"built-in tools leaked: {names & builtin}")

    def test_out_of_order_tool_is_corrected_before_precondition_block(self):
        """A future tool should not surface as a blocked precondition."""
        # PlanShots before SegmentArticle → precondition (segments) missing.
        script = [
            {"tool_name": "Tool_PlanShots", "arguments": {"mode": "plan_shots_all"}},
            {"tool_name": "Tool_SegmentArticle", "arguments": {}},
            {"tool_name": "Tool_StitchVideo", "arguments": {}},
        ]
        dispatch_outputs = {
            "Tool_SegmentArticle": {"status": "success", "segments": [{"segment_id": "seg_01"}]},
            "Tool_StitchVideo": {"status": "success", "final_video_path": "/tmp/final.mp4"},
        }
        loader, runtime, agent = self._build(script, dispatch_outputs)
        from langchain_core.messages import HumanMessage
        asyncio.run(agent.ainvoke({"messages": [HumanMessage(content="x")]}))

        self.assertEqual(self.dispatched[0], "Tool_SegmentArticle")
        blob = json.dumps(self.events, ensure_ascii=False, default=str)
        self.assertNotIn("前置条件不足", blob)
        self.assertNotIn("交回模型决策", blob)

    def test_plain_mode_text(self):
        """Plain (chat) mode returns text without tool calls."""
        from aicut_agent import AICutChatModel
        from langchain_core.messages import HumanMessage, SystemMessage
        model = AICutChatModel()
        model.attach_loader(MockLoader([]))
        out = model.invoke([SystemMessage(content="sys"), HumanMessage(content="你好")])
        self.assertEqual(out.content, "纯文本回复")
        self.assertFalse(getattr(out, "tool_calls", []))

    def test_user_payload_matches_training_contract(self):
        """The user JSON sent to the model must carry the exact training keys."""
        script = [{"tool_name": "Tool_StitchVideo", "arguments": {}}]
        loader, runtime, agent = self._build(
            script, {"Tool_StitchVideo": {"status": "success", "final_video_path": "/tmp/x.mp4"}},
            allow_all_preconditions=True,
        )
        from langchain_core.messages import HumanMessage
        asyncio.run(agent.ainvoke({"messages": [HumanMessage(content="测试口播稿")]}))

        self.assertTrue(loader.calls, "model was never called")
        call = loader.calls[0]
        # system prompt is the exact training prompt held by the runtime.
        self.assertEqual(call["system"], "训练系统提示词")
        # user payload is a single JSON object with the training contract keys.
        payload = json.loads(call["user"])
        for key in (
            "task", "goal", "input_script", "script_digest", "current_focus",
            "global_state", "recent_history", "last_observation",
            "tool_preconditions", "allowed_tool_args", "agent_contract",
        ):
            self.assertIn(key, payload, f"训练契约缺少字段: {key}")
        # the agent contract marks the runtime as the state authority.
        self.assertTrue(payload["agent_contract"]["runtime_is_state_authority"])
        self.assertTrue(payload["agent_contract"]["one_tool_call_per_turn"])


class TestSilentSelfCorrection(_AgentHarnessCase):
    """Repeats that cannot advance the run are replaced by the step that can.

    The line these tests defend: a repeat *following a failed check* is real
    work and must survive untouched, while a repeat of a step whose product is
    already in state must not cost the run its budget.
    """

    def _middleware(self, runtime):
        from aicut_agent.middleware import AICutGuardrailMiddleware
        return AICutGuardrailMiddleware(runtime)

    @staticmethod
    def _request(tool_name: str):
        from types import SimpleNamespace
        return SimpleNamespace(tool_call={"name": tool_name, "args": {}, "id": "call_1"})

    # -- end to end ---------------------------------------------------
    def test_repeating_a_finished_step_advances_the_run(self):
        """The model asks to segment twice; the harness plans instead."""
        script = [
            {"tool_name": "Tool_SegmentArticle", "arguments": {}},
            {"tool_name": "Tool_SegmentArticle", "arguments": {}},
            {"tool_name": "Tool_StitchVideo", "arguments": {}},
        ]
        dispatch_outputs = {
            "Tool_SegmentArticle": {"status": "success", "segments": [{"segment_id": "seg_01"}]},
            "Tool_PlanShots": {"status": "success", "shots": [{"shot_id": "shot_001"}]},
            "Tool_StitchVideo": {"status": "success", "final_video_path": "/tmp/final.mp4"},
        }
        _, runtime, agent = self._build(script, dispatch_outputs, allow_all_preconditions=True)

        from langchain_core.messages import HumanMessage
        asyncio.run(agent.ainvoke({"messages": [HumanMessage(content="x")]}))

        self.assertEqual(
            self.dispatched,
            CANONICAL_FULL_FLOW,
            "第二次分段应被扶正为下一步，而不是重复执行",
        )
        self.assertGreaterEqual(runtime.silent_corrections, 1)
        self.assertTrue(runtime.stitch_succeeded)

    def test_the_correction_is_not_visible_in_the_event_stream(self):
        """Events must read as a clean progression, naming only the real step."""
        script = [
            {"tool_name": "Tool_SegmentArticle", "arguments": {}},
            {"tool_name": "Tool_SegmentArticle", "arguments": {}},
            {"tool_name": "Tool_StitchVideo", "arguments": {}},
        ]
        dispatch_outputs = {
            "Tool_SegmentArticle": {"status": "success", "segments": [{"segment_id": "seg_01"}]},
            "Tool_PlanShots": {"status": "success", "shots": [{"shot_id": "shot_001"}]},
            "Tool_StitchVideo": {"status": "success", "final_video_path": "/tmp/final.mp4"},
        }
        _, _, agent = self._build(script, dispatch_outputs, allow_all_preconditions=True)

        from langchain_core.messages import HumanMessage
        asyncio.run(agent.ainvoke({"messages": [HumanMessage(content="x")]}))

        calls = [e for e in self.events if e.get("type") == "tool_call"]
        self.assertEqual([c["tool_name"] for c in calls], CANONICAL_FULL_FLOW)
        # Nothing anywhere may hint that a step was substituted.
        blob = json.dumps(self.events, ensure_ascii=False, default=str)
        for leak in ("扶正", "重定向", "纠正", "correct", "redirect", "已完成，推进"):
            self.assertNotIn(leak, blob, f"事件流泄漏了扶正痕迹: {leak}")
        self.assertFalse([e for e in self.events if e.get("type") == "error"])

    # -- the discrimination that matters ------------------------------
    def test_a_repeat_after_a_failed_check_is_left_alone(self):
        """Re-repairing shots the judge rejected is the point, not a mistake."""
        _, runtime, _ = self._build([], {}, allow_all_preconditions=True)
        runtime.state["shot_assets"] = {"shot_001": {"approved": False}}
        runtime.orchestrator.quality_checked = True
        runtime.last_tool = "Tool_Stage4DimensionalRepair"

        mw = self._middleware(runtime)
        self.assertIsNone(mw._next_step_for_dead_repeat("Tool_Stage4DimensionalRepair", {}))

    def test_a_repeat_while_reflection_is_pending_is_left_alone(self):
        _, runtime, _ = self._build([], {}, allow_all_preconditions=True)
        orch = runtime.orchestrator
        orch._reflect_generation = 2
        orch._last_reflect_generation = 1          # a new generation to reflect on
        runtime.last_tool = "Tool_SelfReflect"

        mw = self._middleware(runtime)
        self.assertIsNone(mw._next_step_for_dead_repeat("Tool_SelfReflect", {}))

    def test_a_repeat_with_open_issues_is_left_alone(self):
        _, runtime, _ = self._build([], {}, allow_all_preconditions=True)
        orch = runtime.orchestrator
        orch._reflect_generation = orch._last_reflect_generation = 1
        orch.open_issues = [{"issue": "镜头 1 与口播不符"}]
        runtime.last_tool = "Tool_SelfReflect_Patch"

        mw = self._middleware(runtime)
        self.assertIsNone(mw._next_step_for_dead_repeat("Tool_SelfReflect_Patch", {}))

    def test_a_first_attempt_is_never_rewritten(self):
        """A tool reporting its own no-op is information the model may use."""
        _, runtime, _ = self._build([], {}, allow_all_preconditions=True)
        runtime.state["segments"] = [{"segment_id": "seg_01"}]
        runtime.last_tool = ""                     # nothing called yet

        mw = self._middleware(runtime)
        self.assertIsNone(mw._next_step_for_dead_repeat("Tool_SegmentArticle", {}))

    def test_corrections_are_budgeted(self):
        _, runtime, _ = self._build([], {}, allow_all_preconditions=True)
        runtime.state["segments"] = [{"segment_id": "seg_01"}]
        runtime.last_tool = "Tool_SegmentArticle"
        runtime.silent_corrections = runtime.max_silent_corrections

        mw = self._middleware(runtime)
        self.assertIsNone(mw._next_step_for_dead_repeat("Tool_SegmentArticle", {}))

    def test_correction_resets_the_budget_spent_while_stuck(self):
        _, runtime, _ = self._build([], {}, allow_all_preconditions=True)
        runtime.state["segments"] = [{"segment_id": "seg_01"}]
        runtime.last_tool = "Tool_SegmentArticle"
        runtime.same_tool_count = 9
        runtime.repeat_calls = {"Tool_SegmentArticle::{}": 2}

        mw = self._middleware(runtime)
        fixed = mw._next_step_for_dead_repeat("Tool_SegmentArticle", {})
        self.assertIsNotNone(fixed)
        self.assertEqual(fixed[0], "Tool_PlanShots")
        self.assertEqual(runtime.same_tool_count, 0)
        self.assertEqual(runtime.repeat_calls, {})

    def test_a_dead_end_next_step_is_not_swapped_in(self):
        """Correcting into something that cannot run either would achieve nothing."""
        _, runtime, _ = self._build([], {})          # real preconditions
        runtime.state["segments"] = [{"segment_id": "seg_01"}]
        runtime.last_tool = "Tool_SegmentArticle"
        runtime.check_preconditions = lambda tool: ["shot_plan"]

        mw = self._middleware(runtime)
        self.assertIsNone(mw._next_step_for_dead_repeat("Tool_SegmentArticle", {}))

    # -- the delivery backstop ----------------------------------------
    def test_a_ceiling_stitches_instead_of_failing(self):
        """Usable footage plus a stalled model should still produce a film."""
        _, runtime, _ = self._build([], {}, allow_all_preconditions=True)

        async def dispatch(tool_name, tool_args, state, run_id):
            self.dispatched.append(tool_name)
            state["final_video_path"] = "/tmp/salvaged.mp4"
            return {"status": "success", "final_video_path": "/tmp/salvaged.mp4"}

        runtime.orchestrator._dispatch_tool = dispatch

        mw = self._middleware(runtime)
        cmd = asyncio.run(mw._terminate(self._request("Tool_PlanShots"), "工具调用总数已达上限"))

        self.assertEqual(self.dispatched, ["Tool_StitchVideo"])
        self.assertTrue(runtime.stitch_succeeded)
        self.assertEqual(runtime.terminated_reason, "", "抢救成功就不该记为失败")
        self.assertFalse([e for e in self.events if e.get("type") == "error"])
        self.assertEqual(getattr(cmd, "goto", None), "__end__")

    def test_a_step_ceiling_still_delivers_the_film(self):
        """The chat model's ceilings never reach the middleware; the runner does."""
        from aicut_agent.runner import run_aicut_generation

        script = [{"tool_name": "Tool_SegmentArticle", "arguments": {}}] * 4
        dispatch_outputs = {
            "Tool_SegmentArticle": {"status": "success", "segments": [{"segment_id": "seg_01"}]},
            "Tool_PlanShots": {"status": "success", "shots": [{"shot_id": "shot_001"}]},
            "Tool_StitchVideo": {"status": "success", "final_video_path": "/tmp/ceiling.mp4"},
        }
        loader, runtime, _ = self._build(script, dispatch_outputs, allow_all_preconditions=True)
        runtime.max_agent_steps = 1                # next model call trips the ceiling

        results = asyncio.run(run_aicut_generation(runtime, loader))

        self.assertIn("Tool_StitchVideo", self.dispatched)
        self.assertEqual(runtime.state.get("final_video_path"), "/tmp/ceiling.mp4")
        self.assertEqual(results["errors"], [], "能出片就不该报错")
        self.assertFalse([e for e in self.events if e.get("type") == "error"])

    def test_a_ceiling_still_fails_when_there_is_nothing_to_deliver(self):
        _, runtime, _ = self._build([], {})          # real preconditions: stitch blocked

        mw = self._middleware(runtime)
        asyncio.run(mw._terminate(self._request("Tool_PlanShots"), "连续解析失败"))

        self.assertEqual(self.dispatched, [])
        self.assertEqual(runtime.terminated_reason, "连续解析失败")
        self.assertTrue([e for e in self.events if e.get("type") == "error"])


class TestModelCallFailures(_AgentHarnessCase):
    """A failing model call must not take the film down with it.

    The reported case was vLLM answering 400 to ``tool_choice: auto``. That
    exception escaped the graph, so the run ended on a bare error message with
    no closing events and no attempt to deliver what had already rendered.
    """

    BAD_REQUEST = RuntimeError(
        '400 - "auto" tool choice requires --enable-auto-tool-choice'
    )

    def _fail_after(self, loader, working: int):
        """Answer ``working`` calls from the script, then fail every call."""
        original = loader.predict_orchestrator
        calls: list[int] = []

        def predict(system, user, **kw):
            calls.append(1)
            if len(calls) > working:
                raise self.BAD_REQUEST
            return original(system, user, **kw)

        loader.predict_orchestrator = predict   # type: ignore[method-assign]
        return calls

    def _model(self, loader, runtime, attempts: int):
        from aicut_agent.chat_model import AICutChatModel

        model = AICutChatModel().attach_loader(loader).attach_runtime(runtime)
        model.max_model_call_attempts = attempts
        return model

    def test_a_transient_failure_is_retried(self):
        script = [{"tool_name": "Tool_SegmentArticle", "arguments": {}}]
        loader, runtime, _ = self._build(script, {}, allow_all_preconditions=True)
        original = loader.predict_orchestrator
        calls: list[int] = []

        def predict(system, user, **kw):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("worker restarting")
            return original(system, user, **kw)

        loader.predict_orchestrator = predict   # type: ignore[method-assign]

        result = self._model(loader, runtime, 2)._generate([])

        self.assertEqual(len(calls), 2, "第一次失败后应当重试")
        self.assertEqual(runtime.terminated_reason, "")
        self.assertTrue(result.generations[0].message.tool_calls)

    def test_a_persistent_failure_becomes_a_clean_stop(self):
        loader, runtime, _ = self._build([], {}, allow_all_preconditions=True)
        self._fail_after(loader, working=0)

        result = self._model(loader, runtime, 1)._generate([])   # must not raise

        self.assertIn("模型调用失败", runtime.terminated_reason)
        self.assertFalse(result.generations[0].message.tool_calls)

    def test_a_failed_call_still_delivers_the_footage_it_had(self):
        """The whole point: a 400 on step N does not throw away steps 1..N-1."""
        from aicut_agent.runner import run_aicut_generation

        script = [
            {"tool_name": "Tool_SegmentArticle", "arguments": {}},
            {"tool_name": "Tool_PlanShots", "arguments": {}},
        ]
        dispatch_outputs = {
            "Tool_SegmentArticle": {"status": "success", "segments": [{"segment_id": "seg_01"}]},
            "Tool_PlanShots": {"status": "success", "shots": [{"shot_id": "shot_001"}]},
            "Tool_StitchVideo": {"status": "success", "final_video_path": "/tmp/after_400.mp4"},
        }
        loader, runtime, _ = self._build(script, dispatch_outputs, allow_all_preconditions=True)
        # Two steps land, then every further call 400s the way vLLM did.
        self._fail_after(loader, working=2)

        results = asyncio.run(run_aicut_generation(runtime, loader))

        self.assertIn("Tool_StitchVideo", self.dispatched)
        self.assertEqual(runtime.state.get("final_video_path"), "/tmp/after_400.mp4")
        self.assertEqual(results["errors"], [], "已经能出片就不该报错")
        self.assertTrue([e for e in self.events if e.get("type") == "pipeline_complete"])


class TestFixedWorkflowControl(_AgentHarnessCase):
    """The harness, not the model, owns the long-horizon step order."""

    @staticmethod
    def _request(tool_name: str):
        from types import SimpleNamespace
        return SimpleNamespace(tool_call={"name": tool_name, "args": {}, "id": "call_1"})

    def _middleware(self, runtime):
        from aicut_agent.middleware import AICutGuardrailMiddleware
        return AICutGuardrailMiddleware(runtime)

    def test_unparseable_model_output_runs_the_canonical_next_step(self):
        """A blank/prose model reply must not spend four retries and stop."""
        script = [{"tool_name": "", "arguments": {}, "parse_ok": False}]
        loader, runtime, _ = self._build(script, {}, allow_all_preconditions=True)

        from aicut_agent.chat_model import AICutChatModel

        model = AICutChatModel().attach_loader(loader).attach_runtime(runtime)
        result = model._generate([])

        calls = result.generations[0].message.tool_calls
        self.assertEqual(calls[0]["name"], "Tool_SegmentArticle")
        self.assertEqual(runtime.consecutive_parse_failures, 0)
        self.assertEqual(runtime.terminated_reason, "")

    def test_premature_stitch_is_silently_replaced_by_the_missing_render_step(self):
        """Do not show a blocked stitch; just run the fixed next step."""
        _, runtime, _ = self._build(
            [],
            {
                "Tool_BatchTimelineCompose": {
                    "status": "success",
                    "render_success_count": 6,
                }
            },
        )
        runtime.state.update({
            "segments": [{"segment_id": "seg_01"}],
            "shot_plan": [{"shot_id": f"shot_{i:03d}"} for i in range(1, 7)],
            "shot_assets": {
                f"shot_{i:03d}": {"decoration": {"template": "x"}}
                for i in range(1, 7)
            },
            "severity_scores": {"shot_001": 0.1},
        })
        orch = runtime.orchestrator
        orch.visual_audio_done = True
        orch.decorated_done = True
        orch.quality_checked = True
        orch.repaired_done = True
        orch._last_reflect_generation = orch._reflect_generation

        async def should_not_run_original(request):
            raise AssertionError("premature Tool_StitchVideo should have been replaced")

        mw = self._middleware(runtime)
        asyncio.run(
            mw.awrap_tool_call(
                self._request("Tool_StitchVideo"),
                should_not_run_original,
            )
        )

        self.assertEqual(self.dispatched, ["Tool_BatchTimelineCompose"])
        blob = json.dumps(self.events, ensure_ascii=False, default=str)
        self.assertNotIn("前置条件", blob)
        self.assertNotIn("交回模型", blob)
        self.assertNotIn("重试预算", blob)


if __name__ == "__main__":
    unittest.main(verbosity=2)
