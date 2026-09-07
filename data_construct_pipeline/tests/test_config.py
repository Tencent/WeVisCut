"""Config loading, env expansion and per-role endpoint resolution."""

from __future__ import annotations

import textwrap

import pytest

from longvideo_polar.config import (
    DEFAULT_CONFIG_PATH,
    LongVideoConfig,
    ModelEndpoint,
    deep_merge,
    expand_env,
)


def test_expand_env_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv("LVP_TEST_VAR", raising=False)
    assert expand_env("${LVP_TEST_VAR:-fallback}") == "fallback"
    assert expand_env("${LVP_TEST_VAR}") == ""


def test_expand_env_prefers_environment(monkeypatch):
    monkeypatch.setenv("LVP_TEST_VAR", "actual")
    assert expand_env("${LVP_TEST_VAR:-fallback}") == "actual"
    assert expand_env({"a": ["${LVP_TEST_VAR}"]}) == {"a": ["actual"]}


def test_deep_merge_recurses_into_mappings():
    merged = deep_merge(
        {"a": {"b": 1, "c": 2}, "d": 3},
        {"a": {"c": 99, "e": 4}},
    )
    assert merged == {"a": {"b": 1, "c": 99, "e": 4}, "d": 3}


def test_roles_inherit_default_endpoint_then_override():
    config = LongVideoConfig.from_dict({
        "models": {
            "default": {"base_url": "http://x/v1", "model": "m", "max_tokens": 100},
            "orchestrator": {"max_tokens": 24576},
            "critic": {"model": "critic-model"},
        }
    })
    assert config.endpoint_for("orchestrator").max_tokens == 24576
    assert config.endpoint_for("orchestrator").base_url == "http://x/v1"
    assert config.endpoint_for("generator").max_tokens == 100
    assert config.endpoint_for("critic").model == "critic-model"
    assert config.endpoint_for("critic").base_url == "http://x/v1"


def test_unknown_role_falls_back_to_default():
    config = LongVideoConfig.from_dict({
        "models": {"default": {"base_url": "http://x/v1", "model": "m"}}
    })
    assert config.endpoint_for("something-else").model == "m"


def test_endpoint_validation_flags_missing_fields():
    with pytest.raises(ValueError, match="base_url"):
        ModelEndpoint(model="m").validate("generator")
    with pytest.raises(ValueError, match="model"):
        ModelEndpoint(base_url="http://x/v1").validate("generator")


def test_unknown_keys_are_ignored():
    config = LongVideoConfig.from_dict({"agent": {"max_steps": 7, "not_a_field": 1}})
    assert config.agent.max_steps == 7


def test_validate_rejects_missing_pipeline_root():
    config = LongVideoConfig.from_dict({
        "pipeline": {"root": "/definitely/not/here"},
        "models": {"default": {"base_url": "http://x/v1", "model": "m"}},
    })
    with pytest.raises(ValueError, match="does not exist"):
        config.validate()


def test_load_applies_overrides(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        textwrap.dedent("""
            agent:
              max_steps: 10
            models:
              default:
                base_url: http://a/v1
                model: base
        """),
        encoding="utf-8",
    )
    config = LongVideoConfig.load(path, overrides={"agent": {"max_steps": 99}})
    assert config.agent.max_steps == 99
    assert config.endpoint_for("generator").model == "base"
    assert config.raw["_config_path"] == str(path)


def test_load_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        LongVideoConfig.load(tmp_path / "nope.yaml")


def test_packaged_reference_config_is_loadable():
    assert DEFAULT_CONFIG_PATH.is_file(), f"missing reference config: {DEFAULT_CONFIG_PATH}"
    config = LongVideoConfig.load(DEFAULT_CONFIG_PATH)
    assert config.agent.agent_mode == "long_horizon"
    assert config.tool_switches.final_stitch is True
    assert config.recording.strict_tool_schema is True
    # All three roles must resolve to a usable endpoint out of the box.
    for role in ("orchestrator", "generator", "critic"):
        assert config.endpoint_for(role).base_url
        assert config.endpoint_for(role).model
