"""Unit tests for think extraction / empty-think handling.

Run:  python3 tests/test_thinking_parse.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

STUDIO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDIO_DIR))

from server.model_loader import (  # noqa: E402
    UnifiedModelLoader,
    _PlainRequestConfig,
    _build_prediction_result,
    _extract_thinking,
    _with_thinking_extra_body,
)


class ThinkingParseTests(unittest.TestCase):
    def test_extracts_sft_style_think(self):
        raw = (
            "<think>\n观察：未分段。\n判断：先切章节。\n下一步：调用 Tool_SegmentArticle。\n"
            "</think>\n<tool_call>\n{\"name\":\"Tool_SegmentArticle\",\"arguments\":{}}\n</tool_call>"
        )
        self.assertIn("观察：未分段", _extract_thinking(raw))

    def test_empty_think_prefix_is_ignored(self):
        raw = "<think>\n\n</think>\n\n<tool_call>\n{\"name\":\"Tool_PlanShots\"}\n</tool_call>"
        self.assertEqual(_extract_thinking(raw), "")

    def test_reasoning_content_fallback(self):
        self.assertEqual(
            _extract_thinking("<tool_call>{}</tool_call>", "先规划镜头再装饰"),
            "先规划镜头再装饰",
        )

    def test_prediction_reads_reasoning_content(self):
        msg = SimpleNamespace(
            content="<tool_call>\n{\"name\":\"Tool_StitchVideo\",\"arguments\":{}}\n</tool_call>",
            reasoning_content="质检已过，收尾拼接。",
            tool_calls=None,
        )
        resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)
        result = _build_prediction_result([resp])
        self.assertEqual(result.thinking_summary, "质检已过，收尾拼接。")
        self.assertIn("<think>", result.raw_output)
        self.assertIn("Tool_StitchVideo", result.cleaned_output)

    def test_openai_extra_body_gets_enable_thinking(self):
        merged = _with_thinking_extra_body({"top_k": 1}, True)
        self.assertEqual(merged["chat_template_kwargs"]["enable_thinking"], True)
        self.assertEqual(merged["top_k"], 1)


class _FakeCompletions:
    """Records calls and raises whatever ``errors`` says for each one."""

    def __init__(self, errors: list[Exception | None]) -> None:
        self.errors = list(errors)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        err = self.errors[min(len(self.calls) - 1, len(self.errors) - 1)]
        if err is not None:
            raise err
        return SimpleNamespace(choices=[], usage=None)


class _FakeEngine:
    def __init__(self, content: str) -> None:
        self.content = content
        self.requests: list[dict] = []

    def infer(self, infer_requests, request_config=None):
        self.requests.append(infer_requests[0])
        msg = SimpleNamespace(
            content=self.content,
            reasoning_content="",
            tool_calls=None,
        )
        return [SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)]


class _FakeStreamingCompletions:
    def __init__(self, chunks) -> None:
        self.chunks = chunks
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        if not kwargs.get("stream"):
            raise AssertionError("stream=True was not sent")
        for chunk in self.chunks:
            delta = SimpleNamespace(
                content=chunk.get("content", ""),
                reasoning_content=chunk.get("reasoning_content", ""),
                tool_calls=None,
            )
            yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None)


class _Boom(Exception):
    """An error shaped like the SDK's: a ``message`` and a ``status_code``."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(f"Error code: {status_code} - {{'message': '{message}'}}")
        self.message = message
        self.status_code = status_code


class OpenAIEngineTests(unittest.TestCase):
    """The remote engine has to survive a server without tool-calling flags.

    vLLM 400s on ``tool_choice: auto`` unless it was started with
    ``--enable-auto-tool-choice`` and a ``--tool-call-parser``. The policy
    writes ``<tool_call>`` blocks as plain text, so the run does not need
    native tool calling and must not die over it.
    """

    TOOL_CHOICE_400 = (
        '"auto" tool choice requires --enable-auto-tool-choice and '
        "--tool-call-parser to be set"
    )

    def _engine(self, errors, **kw):
        from server.model_loader import _OpenAICompatEngine

        engine = _OpenAICompatEngine(
            base_url="http://127.0.0.1:8099/v1", api_key="x", model="m", **kw
        )
        fake = _FakeCompletions(errors)
        engine._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
        return engine, fake

    def _request(self):
        return [{
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "Tool_PlanShots"}}],
            "chat_template_kwargs": {"enable_thinking": False},
        }]

    def test_tool_choice_rejection_retries_without_it(self):
        engine, fake = self._engine([_Boom(self.TOOL_CHOICE_400), None])
        engine.infer(self._request())

        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[0]["tool_choice"], "auto")
        self.assertNotIn("tool_choice", fake.calls[1])
        self.assertIn("tools", fake.calls[1])   # schemas still shape the template

    def test_tools_are_never_dropped_to_make_a_call_succeed(self):
        """Without schemas in the template the policy stops emitting tool calls.

        That failure surfaces steps later as unparseable output, so a server
        that refuses ``tools`` has to be reported as the misconfiguration it is.
        """
        engine, fake = self._engine(
            [_Boom(self.TOOL_CHOICE_400), _Boom("tools are not supported"), None]
        )
        with self.assertRaises(RuntimeError) as caught:
            engine.infer(self._request())

        self.assertEqual(len(fake.calls), 2, "不该再试一次没有 tools 的请求")
        self.assertIn("--enable-auto-tool-choice", str(caught.exception))

    def test_the_lesson_is_remembered_for_later_calls(self):
        engine, fake = self._engine([_Boom(self.TOOL_CHOICE_400), None])
        engine.infer(self._request())
        engine.infer(self._request())

        # Three calls, not four: the second inference does not re-learn it.
        self.assertEqual(len(fake.calls), 3)
        self.assertNotIn("tool_choice", fake.calls[2])

    def test_an_unrelated_error_is_raised(self):
        engine, fake = self._engine([_Boom("context length exceeded")])
        with self.assertRaises(_Boom):
            engine.infer(self._request())
        self.assertEqual(len(fake.calls), 1)

    def test_a_transport_failure_is_not_mistaken_for_a_bad_shape(self):
        engine, fake = self._engine([_Boom("tool choice unavailable", status_code=503)])
        with self.assertRaises(_Boom):
            engine.infer(self._request())
        self.assertEqual(len(fake.calls), 1)

    def test_chat_template_kwargs_reach_the_server(self):
        """Without this, ``enable_thinking=False`` was silently a no-op."""
        engine, fake = self._engine([None])
        engine.infer(self._request())

        kwargs = fake.calls[0]["extra_body"]["chat_template_kwargs"]
        self.assertIs(kwargs["enable_thinking"], False)

    def test_per_call_thinking_overrides_the_configured_default(self):
        engine, fake = self._engine(
            [None], extra_body=_with_thinking_extra_body({"top_k": 20}, True)
        )
        engine.infer(self._request())

        extra = fake.calls[0]["extra_body"]
        self.assertIs(extra["chat_template_kwargs"]["enable_thinking"], False)
        self.assertEqual(extra["top_k"], 20)   # other settings survive

    def test_configured_thinking_still_applies_without_a_per_call_value(self):
        engine, fake = self._engine(
            [None], extra_body=_with_thinking_extra_body({}, True)
        )
        request = self._request()
        request[0].pop("chat_template_kwargs")
        engine.infer(request)

        extra = fake.calls[0]["extra_body"]
        self.assertIs(extra["chat_template_kwargs"]["enable_thinking"], True)

    def test_streaming_emits_thinking_deltas_and_keeps_final_parse(self):
        from server.model_loader import _OpenAICompatEngine, _build_prediction_result

        engine = _OpenAICompatEngine(
            base_url="http://127.0.0.1:8099/v1", api_key="x", model="m"
        )
        fake = _FakeStreamingCompletions([
            {"reasoning_content": "先理解需求，"},
            {"reasoning_content": "再规划镜头。"},
            {"content": "<tool_call>\n"},
            {"content": '{"name":"Tool_PlanShots","arguments":{}}\n</tool_call>'},
        ])
        engine._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
        deltas: list[str] = []
        request = [{
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"enable_thinking": True},
            "stream_callback": lambda text, source="": deltas.append(text),
        }]

        resp = engine.infer(request)
        result = _build_prediction_result(resp, stage="agent_loop")

        self.assertEqual(deltas, ["先理解需求，", "再规划镜头。"])
        self.assertTrue(fake.calls[0]["stream"])
        self.assertEqual(result.tool_name, "Tool_PlanShots")
        self.assertTrue(result.parse_ok)


class RemoteOrchestratorPromptTests(unittest.TestCase):
    """Remote vLLM should be plain text generation, not native tool calling."""

    def _loader(self):
        loader = UnifiedModelLoader({})
        loader.engine = _FakeEngine(
            '<tool_call>\n{"name":"Tool_SegmentArticle","arguments":{}}\n</tool_call>'
        )
        loader._request_config_cls = _PlainRequestConfig
        loader._loaded = True
        loader._infer_backend = "openai"
        loader._enable_thinking = True
        loader._default_max_tokens = 2048
        loader._temperature = 0.0
        loader._top_p = 1.0
        loader._top_k = None
        loader._repetition_penalty = None
        loader._stop_words = ["</tool_call>"]
        loader._max_tokens_by_role = {}
        loader._max_tokens_by_stage = {}
        return loader

    def test_remote_orchestrator_inlines_tools_instead_of_sending_them(self):
        loader = self._loader()
        result = loader.predict_orchestrator(
            "系统提示",
            '{"task":"script_to_long_video"}',
            allowed_tools=["Tool_SegmentArticle"],
        )

        request = loader.engine.requests[0]
        self.assertIsNone(request["tools"])
        system = request["messages"][0]["content"]
        self.assertIn("【可用工具定义】", system)
        self.assertIn("Tool_SegmentArticle", system)
        self.assertIn("<tool_call>", system)
        self.assertEqual(result.tool_name, "Tool_SegmentArticle")
        self.assertTrue(result.parse_ok)

    def test_inprocess_orchestrator_still_sends_native_tools_to_template(self):
        loader = self._loader()
        loader._infer_backend = "transformers"

        loader.predict_orchestrator(
            "系统提示",
            '{"task":"script_to_long_video"}',
            allowed_tools=["Tool_SegmentArticle"],
        )

        request = loader.engine.requests[0]
        self.assertIsInstance(request["tools"], list)
        self.assertIn("Tool_SegmentArticle", request["tools"][0]["function"]["name"])
        self.assertEqual(request["messages"][0]["content"], "系统提示")


if __name__ == "__main__":
    unittest.main()
