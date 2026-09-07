"""Model-mix profiles: 全 Qwen vs 反思修复走 Gemini/Opus.

These tests pin the behaviour that decides *which model annotated which sample*, so
they care about precedence and provenance rather than transport.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from longvideo_polar.config import (
    DEFAULT_CONFIG_PATH,
    LongVideoConfig,
    ModelEndpoint,
    resolve_profile,
)

BASE_DOC = {
    "models": {
        "default": {"base_url": "http://local:8000/v1", "model": "qwen", "provider": "qwen"},
    },
    "profiles": {
        "all_qwen": {},
        "critic_gemini": {
            "models": {
                "critic": {
                    "provider": "gemini",
                    "base_url": "https://gw/v1",
                    "api_key": "k",
                    "model": "gemini-3.1-pro-preview",
                    "drop_params": ["top_p"],
                }
            }
        },
    },
}


def _doc():
    import copy

    return copy.deepcopy(BASE_DOC)


def test_all_qwen_keeps_every_role_on_the_local_model():
    config = LongVideoConfig.from_dict(_doc(), profile="all_qwen")

    assert config.profile == "all_qwen"
    providers = config.describe_models()["providers"]
    assert providers == {"orchestrator": "qwen", "generator": "qwen", "critic": "qwen"}
    assert config.describe_models()["mixed_providers"] is False


def test_critic_profile_moves_only_the_critic_to_the_gateway():
    config = LongVideoConfig.from_dict(_doc(), profile="critic_gemini")

    critic = config.endpoint_for("critic")
    assert critic.provider == "gemini"
    assert critic.model == "gemini-3.1-pro-preview"
    assert critic.drop_params == ["top_p"]
    # The roles being trained must stay on the local model.
    for role in ("orchestrator", "generator"):
        assert config.endpoint_for(role).base_url == "http://local:8000/v1"
    assert config.describe_models()["mixed_providers"] is True


def test_profile_name_comes_from_the_environment_when_not_passed(monkeypatch):
    monkeypatch.setenv("LVP_PROFILE", "critic_gemini")

    config = LongVideoConfig.from_dict(_doc())

    assert config.profile == "critic_gemini"
    assert config.endpoint_for("critic").provider == "gemini"


def test_annotation_profile_in_the_document_expands_env_vars(monkeypatch):
    monkeypatch.delenv("LVP_PROFILE", raising=False)
    doc = _doc()
    doc["annotation_profile"] = "${LVP_PROFILE:-critic_gemini}"

    config = LongVideoConfig.from_dict(doc)

    assert config.profile == "critic_gemini"


def test_unknown_profile_names_the_available_ones():
    with pytest.raises(ValueError, match="all_qwen, critic_gemini"):
        LongVideoConfig.from_dict(_doc(), profile="critic_llama")


def test_cli_override_wins_over_the_profile(tmp_path, monkeypatch):
    """Precedence is base < profile < explicit override."""
    monkeypatch.delenv("LVP_PROFILE", raising=False)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_doc()), encoding="utf-8")

    config = LongVideoConfig.load(
        path,
        profile="critic_gemini",
        overrides={"models": {"critic": {"model": "explicit-override"}}},
    )

    assert config.profile == "critic_gemini"
    assert config.endpoint_for("critic").model == "explicit-override"
    # The rest of the profile still applies.
    assert config.endpoint_for("critic").provider == "gemini"


def test_resolve_profile_is_idempotent():
    once, name = resolve_profile(_doc(), "critic_gemini")
    twice, name_again = resolve_profile(once, "critic_gemini")

    assert (name, name_again) == ("critic_gemini", "critic_gemini")
    assert twice == once


def test_shipped_config_defines_the_documented_profiles(monkeypatch):
    monkeypatch.delenv("LVP_PROFILE", raising=False)
    doc = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))

    assert set(doc["profiles"]) >= {"all_qwen", "critic_gemini", "critic_opus"}
    # Default must be the fully local mix: no third-party call without opting in.
    default = LongVideoConfig.from_dict(doc)
    assert default.profile == "all_qwen"
    assert default.describe_models()["mixed_providers"] is False


@pytest.mark.parametrize("profile", ["critic_gemini", "critic_opus", "skills_gemini"])
def test_shipped_third_party_profiles_only_change_skill_roles(profile, monkeypatch):
    monkeypatch.setenv("LVP_GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LVP_OPUS_API_KEY", "test-key")
    doc = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))

    config = LongVideoConfig.from_dict(doc, profile=profile)

    # 主流程 is the policy being trained; it must never be silently outsourced.
    assert config.endpoint_for("orchestrator").provider == "qwen"
    assert config.endpoint_for("critic").provider in {"gemini", "claude"}
    # Third-party gateways get no tool schema and no top_p.
    critic = config.endpoint_for("critic")
    assert critic.native_tool_calls is False
    assert "top_p" in critic.drop_params


def test_missing_third_party_key_fails_validation_instead_of_at_request_time(monkeypatch):
    monkeypatch.delenv("LVP_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("AICUT_302AI_API_KEY", raising=False)
    doc = _doc()
    doc["profiles"]["critic_gemini"]["models"]["critic"]["api_key"] = ""

    config = LongVideoConfig.from_dict(doc, profile="critic_gemini")

    with pytest.raises(ValueError, match="models.critic.api_key is empty"):
        config.endpoint_for("critic").validate("critic")


def test_endpoint_describe_omits_the_api_key():
    endpoint = ModelEndpoint.from_dict({
        "provider": "claude", "base_url": "https://gw/v1",
        "api_key": "sk-super-secret", "model": "claude-opus-4",
    })

    described = endpoint.describe()

    assert "sk-super-secret" not in str(described)
    assert described["provider"] == "claude"
    assert described["model"] == "claude-opus-4"


def test_batch_workers_inherit_the_profile(tmp_path):
    """A worker re-loads the config from disk, so the profile must be passed along."""
    from longvideo_polar.batch import BatchRunner

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_doc()), encoding="utf-8")
    config = LongVideoConfig.load(path, profile="critic_gemini")

    assert BatchRunner(config, path).profile == "critic_gemini"


def test_aggregated_shards_stay_traceable_to_their_model_mix(tmp_path):
    """Concatenating runs must not erase which model annotated what."""
    import json

    from longvideo_polar.batch import aggregate_episodes

    episode = tmp_path / "ep1"
    (episode / "trajectories").mkdir(parents=True)
    (episode / "episode.json").write_text(json.dumps({
        "episode_id": "ep1",
        "status": "ok",
        "metadata": {"annotation_profile": "critic_gemini"},
        "roles": {
            "orchestrator": {"providers_used": ["qwen"]},
            "generator": {"providers_used": ["qwen"]},
            "critic": {"providers_used": ["gemini"]},
        },
    }), encoding="utf-8")
    (episode / "trajectories" / "orchestrator.jsonl").write_text(json.dumps({
        "sft_messages": [{"role": "user", "content": "x"}], "num_tool_calls": 1,
    }), encoding="utf-8")
    (episode / "trajectories" / "critic.jsonl").write_text(json.dumps({
        "samples": [{
            "messages": [{"role": "user", "content": "x"}],
            "metadata": {"provider": "gemini", "model": "gemini-3.1-pro-preview"},
        }],
    }), encoding="utf-8")

    out_dir = Path(aggregate_episodes(tmp_path))

    orchestrator_row = json.loads((out_dir / "orchestrator_sft.jsonl").read_text())
    assert orchestrator_row["metadata"]["annotation_profile"] == "critic_gemini"
    assert orchestrator_row["metadata"]["providers"]["critic"] == ["gemini"]
    critic_row = json.loads((out_dir / "critic_sft.jsonl").read_text())
    assert critic_row["metadata"]["provider"] == "gemini"
    index = json.loads((out_dir / "index.json").read_text())
    assert index["episodes"][0]["annotation_profile"] == "critic_gemini"


def test_config_yaml_profile_block_is_documented():
    """The profile switch is only usable if the config explains how to flip it."""
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    assert "LVP_PROFILE" in text
    assert "--profile" in text
    assert textwrap.dedent("annotation_profile:") in text
