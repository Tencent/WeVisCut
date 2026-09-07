"""Role-routed model client: response parsing, role injection, recording."""

from __future__ import annotations

import json

from longvideo_polar.config import ModelEndpoint
from longvideo_polar.model_client import (
    RoleRoutedModelClient,
    _ensure_role_field,
    _parse_response,
    _parse_tool_json,
)
from longvideo_polar.recorder import TrajectoryRecorder
from longvideo_polar.roles import Role, load_tool_schema


def test_native_tool_call_is_parsed():
    result = _parse_response(
        {
            "choices": [{
                "message": {
                    "content": "<think>plan</think>",
                    "tool_calls": [{
                        "function": {
                            "name": "Tool_PlanShots",
                            "arguments": '{"segment_id": "seg_001", "chunk_index": 0}',
                        }
                    }],
                }
            }],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        },
        latency_ms=100.0,
    )
    assert result.parse_ok
    assert result.tool_name == "Tool_PlanShots"
    assert result.arguments["segment_id"] == "seg_001"
    assert result.thinking_summary == "plan"
    assert result.cleaned_output == ""
    assert result.prompt_tokens == 11
    assert result.tokens_per_sec > 0


def test_json_in_text_tool_call_is_parsed_as_fallback():
    result = _parse_response(
        {"choices": [{"message": {
            "content": '```json\n{"name": "Tool_StitchVideo", "arguments": {"shot_count": 12}}\n```'
        }}]},
        latency_ms=10.0,
    )
    assert result.parse_ok
    assert result.tool_name == "Tool_StitchVideo"
    assert result.arguments == {"shot_count": 12}


def test_reasoning_content_field_is_used_as_thinking():
    result = _parse_response(
        {"choices": [{"message": {"content": "{}", "reasoning_content": "deliberating"}}]},
        latency_ms=1.0,
    )
    assert result.thinking_summary == "deliberating"


def test_skill_output_keeps_json_and_reports_no_tool():
    result = _parse_response(
        {"choices": [{"message": {"content": '{"segments": [{"segment_id": "seg_001"}]}'}}]},
        latency_ms=1.0,
    )
    assert not result.parse_ok  # no tool name -> not a tool decision
    assert json.loads(result.cleaned_output)["segments"]


def test_malformed_and_empty_responses_do_not_raise():
    assert _parse_response({}, latency_ms=1.0).raw_output == ""
    assert _parse_response({"choices": []}, latency_ms=1.0).raw_output == ""
    assert _parse_response(
        {"choices": [{"message": {"content": "not json at all"}}]}, latency_ms=1.0
    ).parse_ok is False


def test_parse_tool_json_stops_at_first_complete_object():
    parsed = _parse_tool_json('{"a": {"b": 1}} trailing noise {"c": 2}')
    assert parsed == {"a": {"b": 1}}
    assert _parse_tool_json("no braces here") is None


def test_role_field_injected_into_skill_payload():
    injected = json.loads(_ensure_role_field('{"stage_task": "plan_shots"}', "generator"))
    assert injected["role"] == "generator"
    # An explicit role wins.
    kept = json.loads(_ensure_role_field('{"role": "critic"}', "generator"))
    assert kept["role"] == "critic"
    # Non-JSON payloads pass through untouched.
    assert _ensure_role_field("plain text", "generator") == "plain text"
    assert json.loads(_ensure_role_field("", "critic"))["role"] == "critic"


def test_engine_for_tool_matches_role_split():
    client = RoleRoutedModelClient(
        endpoints={"default": ModelEndpoint(base_url="http://x/v1", model="m")},
        tool_schema=load_tool_schema(strict=False),
    )
    assert client.engine_for_tool("Tool_PlanShots") == "generator"
    assert client.engine_for_tool("Tool_SelfReflect") == "critic"
    assert client.engine_for_tool("Tool_StitchVideo") == "orchestrator"


def test_client_exposes_unified_model_loader_surface():
    """InferenceOrchestrator only enables role routing when these exist."""
    client = RoleRoutedModelClient(
        endpoints={"default": ModelEndpoint(base_url="http://x/v1", model="m")},
        tool_schema=load_tool_schema(strict=False),
    )
    for name in (
        "predict", "predict_orchestrator", "predict_generator", "predict_critic",
        "engine_for_tool", "load", "unload", "shutdown",
    ):
        assert callable(getattr(client, name)), name
    for name in ("orchestrator_prompt", "skill_prompt", "is_loaded"):
        assert hasattr(client, name), name
    # hasattr(loader, "predict_orchestrator") is the multi-role switch.
    assert hasattr(client, "predict_orchestrator")


def test_transport_failure_is_recorded_and_does_not_raise(tmp_path, monkeypatch):
    recorder = TrajectoryRecorder(episode_id="ep", output_dir=tmp_path)
    client = RoleRoutedModelClient(
        endpoints={"default": ModelEndpoint(
            base_url="http://127.0.0.1:1/v1", model="m", max_retries=1, timeout=0.05,
        )},
        tool_schema=load_tool_schema(strict=False),
        recorder=recorder,
    )
    monkeypatch.setattr("time.sleep", lambda _: None)

    result = client.predict_generator('{"stage_task": "plan_shots"}')
    client.unload()

    assert result.parse_ok is False
    calls = recorder.llm_calls
    assert len(calls) == 1
    assert calls[0].role is Role.GENERATOR
    assert calls[0].error, "transport error must be recorded, not swallowed"


def test_predict_records_and_returns_parsed_tool_call(tmp_path):
    recorder = TrajectoryRecorder(episode_id="ep", output_dir=tmp_path)
    client = RoleRoutedModelClient(
        endpoints={"default": ModelEndpoint(base_url="http://x/v1", model="served-m")},
        tool_schema=load_tool_schema(strict=False),
        recorder=recorder,
        orchestrator_system_prompt="ORCH",
    )
    client.load()

    class _Response:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict:
            return {
                "choices": [{"message": {
                    "content": "<think>segment first</think>",
                    "tool_calls": [{"function": {
                        "name": "Tool_SegmentArticle",
                        "arguments": '{"num_segments_target": 6}',
                    }}],
                }}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            }

    captured: dict = {}

    def fake_post(url, json=None, timeout=None):  # noqa: A002 - mirrors httpx
        captured["url"] = url
        captured["payload"] = json
        return _Response()

    client._clients["default"].post = fake_post  # noqa: SLF001 - test seam

    result = client.predict_orchestrator("", "{}", stage="agent_loop", allowed_tools=["Tool_SegmentArticle"])
    client.unload()

    assert result.tool_name == "Tool_SegmentArticle"
    assert captured["url"] == "/chat/completions"
    assert captured["payload"]["model"] == "served-m"
    assert captured["payload"]["tools"][0]["function"]["name"] == "Tool_SegmentArticle"
    # An empty system prompt falls back to the configured orchestrator prompt.
    assert captured["payload"]["messages"][0]["content"] == "ORCH"

    calls = recorder.llm_calls
    assert len(calls) == 1
    assert calls[0].role is Role.ORCHESTRATOR
    assert calls[0].step == 1  # an agent_loop decision opens a step
    assert calls[0].model == "served-m"
