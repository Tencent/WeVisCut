"""Wire-level checks for the smoke harness and provider compatibility layer.

These run against the scripted server only (no AICutPipeline import), so they are
fast. The full end-to-end scenarios live in ``longvideo_polar.smoke`` and are run via
``longvideo-polar smoke``; ``test_full_scenarios_are_runnable`` opts into them when
``LVP_SMOKE_PIPELINE_ROOT`` is set.
"""

from __future__ import annotations

import json
import os

import pytest

from longvideo_polar.config import ModelEndpoint
from longvideo_polar.model_client import RoleRoutedModelClient, build_http_client
from longvideo_polar.recorder import TrajectoryRecorder
from longvideo_polar.roles import Role, load_tool_schema
from longvideo_polar.smoke import FakeModelServer, _segments_output, _shots_output


@pytest.fixture
def fake_server():
    server = FakeModelServer(model="fake-model")
    server.start()
    yield server
    server.stop()


@pytest.fixture
def recorder(tmp_path):
    return TrajectoryRecorder(episode_id="ep", output_dir=tmp_path)


def _endpoints(base_url: str, **critic: object) -> dict[str, ModelEndpoint]:
    default = ModelEndpoint.from_dict({
        "base_url": base_url, "model": "fake-model", "provider": "qwen", "max_retries": 1,
    })
    return {
        "default": default,
        "orchestrator": default,
        "generator": default,
        "critic": default.merged_with(dict(critic)) if critic else default,
    }


def _client(endpoints, recorder=None) -> RoleRoutedModelClient:
    client = RoleRoutedModelClient(
        endpoints=endpoints,
        tool_schema=load_tool_schema(strict=False),
        recorder=recorder,
    )
    client.load("system prompt")
    return client


# ---------------------------------------------------------------------------
# Scripted server contract
# ---------------------------------------------------------------------------


def test_orchestrator_requests_get_a_scripted_tool_call(fake_server):
    client = _client(_endpoints(fake_server.base_url))

    first = client.predict_orchestrator("sys", "state 1")
    second = client.predict_orchestrator("sys", "state 2")

    assert (first.tool_name, second.tool_name) == ("Tool_SegmentArticle", "Tool_PlanShots")
    assert first.parse_ok and second.parse_ok
    assert first.arguments == {"num_segments_target": 3}


def test_skill_requests_are_routed_by_the_injected_role_field(fake_server):
    client = _client(_endpoints(fake_server.base_url))

    client.predict_generator(json.dumps({"stage_task": "segment_article", "input_script": "一句话。"}))
    client.predict_critic(json.dumps({"stage_task": "self_reflect_plan"}))

    roles = [item.role for item in fake_server.snapshot()]
    assert roles == [Role.GENERATOR.value, Role.CRITIC.value]


def test_skill_output_is_marked_parseable_even_without_a_tool_name(fake_server, recorder):
    """A skill turn returns a bare JSON object; that must count as parsed."""
    client = _client(_endpoints(fake_server.base_url), recorder=recorder)

    result = client.predict_critic(json.dumps({"stage_task": "self_reflect_plan"}))

    assert result.tool_name == ""
    assert result.parse_ok is True
    assert [call.parse_ok for call in recorder.llm_calls] == [True]


def test_unparseable_skill_output_is_flagged(fake_server, recorder, monkeypatch):
    client = _client(_endpoints(fake_server.base_url), recorder=recorder)
    monkeypatch.setattr(
        "longvideo_polar.smoke._skill_output", lambda *_a, **_k: "抱歉，我无法完成。"
    )

    result = client.predict_critic(json.dumps({"stage_task": "self_reflect_plan"}))

    assert result.parse_ok is False
    assert recorder.llm_calls[0].parse_ok is False


# ---------------------------------------------------------------------------
# Third-party gateway compatibility
# ---------------------------------------------------------------------------


def test_bare_auth_header_and_extra_headers_reach_the_server(fake_server):
    endpoint = ModelEndpoint.from_dict({
        "base_url": fake_server.base_url, "model": "fake-model", "provider": "gemini",
        "api_key": "raw-key", "auth_header_prefix": "",
        "extra_headers": {"X-Trace": "abc"}, "max_retries": 1,
    })
    fake_server.extra_header_name = "X-Trace"
    client = _client({"default": endpoint, "critic": endpoint})

    client.predict_critic(json.dumps({"stage_task": "self_reflect_plan"}))

    request = fake_server.snapshot()[-1]
    assert request.auth_header == "raw-key"
    assert request.extra_header_seen == "abc"


def test_bearer_prefix_is_the_default(fake_server):
    client = _client(_endpoints(fake_server.base_url, api_key="sk-1"))

    client.predict_critic(json.dumps({"stage_task": "quality_judge"}))

    assert fake_server.snapshot()[-1].auth_header == "Bearer sk-1"


def test_drop_params_removes_rejected_sampling_params(fake_server):
    client = _client(_endpoints(fake_server.base_url, drop_params=["top_p", "stream"]))

    client.predict_critic(json.dumps({"stage_task": "quality_judge"}))

    params = fake_server.snapshot()[-1].params
    assert "top_p" not in params and "stream" not in params
    assert "temperature" in params  # only the listed keys are dropped


def test_native_tool_calls_off_sends_no_tool_schema(fake_server):
    endpoint = ModelEndpoint.from_dict({
        "base_url": fake_server.base_url, "model": "fake-model",
        "native_tool_calls": False, "max_retries": 1,
    })
    client = _client({"default": endpoint, "orchestrator": endpoint})

    client.predict_orchestrator("sys", "state")

    assert fake_server.snapshot()[-1].has_tools is False


def test_provider_is_recorded_per_call(fake_server, recorder):
    client = _client(
        _endpoints(fake_server.base_url, provider="claude", model="claude-opus-4"),
        recorder=recorder,
    )

    client.predict_generator(json.dumps({"stage_task": "segment_article"}))
    client.predict_critic(json.dumps({"stage_task": "self_reflect_plan"}))

    tagged = {(call.role.value, call.provider, call.model) for call in recorder.llm_calls}
    assert tagged == {
        ("generator", "qwen", "fake-model"),
        ("critic", "claude", "claude-opus-4"),
    }


def test_http_error_body_is_kept_in_the_error_message(fake_server, recorder):
    """A bare '400 Bad Request' hides which param the gateway rejected."""
    # This server refuses critic traffic, answering 403 with a JSON body.
    fake_server.serve_roles = ("orchestrator", "generator")
    client = _client(_endpoints(fake_server.base_url), recorder=recorder)

    client.predict_critic(json.dumps({"stage_task": "self_reflect_plan"}))

    error = recorder.llm_calls[0].error
    assert "403" in error
    assert "not served" in error


def test_loopback_endpoints_ignore_environment_proxies(monkeypatch):
    """A corporate HTTP_PROXY must not intercept a local vLLM."""
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.invalid:3128")
    local = ModelEndpoint.from_dict({"base_url": "http://127.0.0.1:8000/v1", "model": "m"})
    remote = ModelEndpoint.from_dict({"base_url": "https://api.302ai.cn/v1", "model": "m"})

    with build_http_client(local) as local_client, build_http_client(remote) as remote_client:
        assert local_client.trust_env is False
        assert remote_client.trust_env is True


def test_endpoint_path_override_is_used(fake_server):
    endpoint = ModelEndpoint.from_dict({
        "base_url": fake_server.base_url, "model": "fake-model",
        "endpoint_path": "/v1beta/chat/completions", "max_retries": 1,
    })
    client = _client({"default": endpoint, "critic": endpoint})

    client.predict_critic(json.dumps({"stage_task": "quality_judge"}))

    assert fake_server.snapshot()[-1].path.endswith("/v1beta/chat/completions")


# ---------------------------------------------------------------------------
# Canned skill outputs must satisfy the pipeline's preconditions
# ---------------------------------------------------------------------------


def test_canned_segments_cover_every_source_unit():
    payload = {
        "constraints": {"target_num_segments": 3},
        "memory": {"article_digest": {"source_units": [
            {"unit_id": f"u{i:02d}", "text": f"第{i}句。"} for i in range(1, 8)
        ]}},
    }

    segments = _segments_output(payload)["segments"]

    covered = [uid for segment in segments for uid in segment["coverage_unit_ids"]]
    assert len(segments) == 3
    assert covered == [f"u{i:02d}" for i in range(1, 8)]
    # Shot budgeting reads narration text; empty segments would plan zero shots.
    assert all(segment["narration_script"] for segment in segments)


def test_canned_shots_cover_every_segment():
    """Tool_SelfReflect is blocked unless a shot plan exists for the segments."""
    payload = {"memory": {
        "segment_anchors": [{"segment_id": "seg_001"}, {"segment_id": "seg_002"}],
        "coverage_units": [{"unit_id": "u01", "text": "甲。"}, {"unit_id": "u02", "text": "乙。"}],
    }}

    shots = _shots_output(payload)["shots"]

    assert [shot["segment_id"] for shot in shots] == ["seg_001", "seg_002"]
    assert all(shot["shot_id"] and shot["vocal_script"] for shot in shots)
    assert any(shot["is_spotlight"] for shot in shots)


# ---------------------------------------------------------------------------
# Opt-in full end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("LVP_SMOKE_PIPELINE_ROOT"),
    reason="set LVP_SMOKE_PIPELINE_ROOT to run the full pipeline scenarios (~3 min each)",
)
@pytest.mark.parametrize("scenario", ["all_qwen", "split_critic"])
async def test_full_scenarios_are_runnable(scenario, tmp_path):
    from longvideo_polar.smoke import run_scenario

    report = await run_scenario(
        scenario=scenario,
        pipeline_root=os.environ["LVP_SMOKE_PIPELINE_ROOT"],
        output_dir=tmp_path / scenario,
    )

    assert report.passed, report.render()
