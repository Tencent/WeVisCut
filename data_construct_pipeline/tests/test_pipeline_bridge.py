"""Path bootstrapping and AICutPipeline config translation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from longvideo_polar.config import LongVideoConfig
from longvideo_polar.pipeline_bridge import (
    apply_pipeline_overrides,
    bootstrap_pipeline_paths,
    build_infer_config,
    load_pipeline_assets,
)


def make_checkout(tmp_path, *, with_root_server: bool = True):
    """Build a minimal tree shaped like a real AICutPipeline checkout."""
    root = tmp_path / "AICutPipeline"
    (root / "pipeline").mkdir(parents=True)
    (root / "pipeline" / "__init__.py").write_text("", encoding="utf-8")

    infer_server = root / "Infer_Vis_new" / "server"
    infer_server.mkdir(parents=True)
    (infer_server / "__init__.py").write_text("", encoding="utf-8")
    (infer_server / "infer_graph.py").write_text("", encoding="utf-8")
    (infer_server / "tool_schema.py").write_text("MARKER = 'infer'\n", encoding="utf-8")

    if with_root_server:
        # AICutPipeline really does ship a second `server` package (the TTS
        # service). It must not win the name.
        tts_server = root / "server"
        tts_server.mkdir()
        (tts_server / "__init__.py").write_text("", encoding="utf-8")

    (root / "config.yaml").write_text(
        "execution:\n  mode: fake\nllm:\n  model: old\ntts:\n  backend: qwen3\n",
        encoding="utf-8",
    )
    prompts = root / "preprocess_ori_data" / "configs"
    prompts.mkdir(parents=True)
    (prompts / "unified_v5_system_prompt.txt").write_text("SYSTEM\n", encoding="utf-8")
    (prompts / "stage_allowed_tools.yaml").write_text(
        "allowed_tools:\n  agent_loop: [Tool_SegmentArticle]\n", encoding="utf-8"
    )
    return root


@pytest.fixture(autouse=True)
def restore_sys_state():
    original_path = list(sys.path)
    original_modules = dict(sys.modules)
    yield
    sys.path[:] = original_path
    sys.modules.clear()
    sys.modules.update(original_modules)


def make_config(root) -> LongVideoConfig:
    return LongVideoConfig.from_dict({
        "pipeline": {
            "root": str(root),
            "pipeline_config": "config.yaml",
            "system_prompt_file": "preprocess_ori_data/configs/unified_v5_system_prompt.txt",
            "stage_config_file": "preprocess_ori_data/configs/stage_allowed_tools.yaml",
        },
        "models": {"default": {"base_url": "http://x/v1", "model": "m"}},
    })


def test_infer_server_wins_over_the_tts_server_package(tmp_path):
    """The checkout has two `server` packages; only one has the agent loop."""
    root = make_checkout(tmp_path)
    resolved_root, infer_dir = bootstrap_pipeline_paths(root)

    assert resolved_root == root.resolve()
    assert sys.path.index(str(infer_dir)) < sys.path.index(str(resolved_root))

    import server.tool_schema as tool_schema  # noqa: PLC0415 - import order is the test

    assert tool_schema.MARKER == "infer"


def test_stale_server_module_is_evicted(tmp_path):
    root = make_checkout(tmp_path)
    sys.path.insert(0, str(root))
    import server as tts_server  # noqa: PLC0415 - deliberately import the wrong one

    assert "tool_schema" not in dir(tts_server)

    bootstrap_pipeline_paths(root)
    import server.tool_schema as tool_schema  # noqa: PLC0415

    assert tool_schema.MARKER == "infer"


def test_bootstrap_is_idempotent(tmp_path):
    root = make_checkout(tmp_path)
    bootstrap_pipeline_paths(root)
    bootstrap_pipeline_paths(root)
    _, infer_dir = bootstrap_pipeline_paths(root)
    assert sys.path.count(str(infer_dir)) == 1
    assert sys.path.count(str(root.resolve())) == 1


def test_bootstrap_rejects_a_tree_without_the_agent_loop(tmp_path):
    root = tmp_path / "not_a_checkout"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="pipeline/ dir"):
        bootstrap_pipeline_paths(root)

    (root / "pipeline").mkdir()
    with pytest.raises(FileNotFoundError, match="infer_graph.py"):
        bootstrap_pipeline_paths(root)


def make_weviscut_checkout(tmp_path):
    """Build a minimal tree shaped like this WeVisCut repository."""
    root = tmp_path / "WeVisCut"
    (root / "pipeline").mkdir(parents=True)
    (root / "pipeline" / "__init__.py").write_text("", encoding="utf-8")
    server = root / "server"
    server.mkdir()
    (server / "__init__.py").write_text("", encoding="utf-8")
    (server / "infer_graph.py").write_text("", encoding="utf-8")
    (server / "tool_schema.py").write_text("MARKER = 'weviscut'\n", encoding="utf-8")
    (root / "configs").mkdir()
    (root / "configs" / "unified_v5_system_prompt.txt").write_text("SYSTEM\n", encoding="utf-8")
    (root / "configs" / "stage_allowed_tools.yaml").write_text(
        "allowed_tools:\n  agent_loop: [Tool_SegmentArticle]\n", encoding="utf-8"
    )
    return root


def test_bootstrap_accepts_weviscut_layout(tmp_path):
    root = make_weviscut_checkout(tmp_path)
    resolved_root, infer_dir = bootstrap_pipeline_paths(root)

    assert resolved_root == root.resolve()
    assert infer_dir == root.resolve()
    assert sys.path.count(str(resolved_root)) == 1

    import server.tool_schema as tool_schema  # noqa: PLC0415

    assert tool_schema.MARKER == "weviscut"


def test_load_pipeline_assets_from_weviscut_layout(tmp_path):
    root = make_weviscut_checkout(tmp_path)
    pkg_runtime = (
        Path(__file__).resolve().parents[1] / "config" / "pipeline_runtime.yaml"
    )
    if not pkg_runtime.is_file():
        pytest.skip("pipeline_runtime.yaml not packaged yet")
    config = LongVideoConfig.from_dict({
        "pipeline": {
            "root": str(root),
            "pipeline_config": "config/pipeline_runtime.yaml",
            "system_prompt_file": "configs/unified_v5_system_prompt.txt",
            "stage_config_file": "configs/stage_allowed_tools.yaml",
        },
        "models": {"default": {"base_url": "http://x/v1", "model": "m"}},
    })
    pipeline_config, prompt, stage_config = load_pipeline_assets(config)
    assert prompt == "SYSTEM"
    assert stage_config["allowed_tools"]["agent_loop"] == ["Tool_SegmentArticle"]
    assert isinstance(pipeline_config, dict)


def test_load_pipeline_assets_reads_config_prompt_and_stages(tmp_path):
    root = make_checkout(tmp_path)
    pipeline_config, prompt, stage_config = load_pipeline_assets(make_config(root))
    assert pipeline_config["execution"]["mode"] == "fake"
    assert prompt == "SYSTEM"
    assert stage_config["allowed_tools"]["agent_loop"] == ["Tool_SegmentArticle"]


def test_missing_system_prompt_is_fatal(tmp_path):
    root = make_checkout(tmp_path)
    (root / "preprocess_ori_data" / "configs" / "unified_v5_system_prompt.txt").unlink()
    with pytest.raises(FileNotFoundError, match="system prompt"):
        load_pipeline_assets(make_config(root))


def test_overrides_force_real_execution_and_retarget_the_llm(tmp_path):
    root = make_checkout(tmp_path)
    config = LongVideoConfig.from_dict({
        "pipeline": {
            "root": str(root),
            "config_overrides": {"stage3_trajectory": {"max_workers": 7}},
            "render_split": {"enabled": True, "overlay_format": "webm"},
            "tts_service_url": "http://tts:9000",
        },
        "models": {
            "default": {"base_url": "http://x/v1", "model": "m"},
            "generator": {"model": "gen-m"},
        },
    })
    pipeline_config, _, _ = load_pipeline_assets(config)
    merged = apply_pipeline_overrides(pipeline_config, config)

    assert merged["execution"]["mode"] == "real"
    assert merged["stage3_trajectory"]["max_workers"] == 7
    assert merged["tts"]["service_url"] == "http://tts:9000"
    assert merged["tts"]["backend"] == "qwen3"  # untouched while tts stays enabled
    # Tools that build their own LLMClient must reach the same served model.
    assert merged["llm"]["model"] == "gen-m"
    assert merged["llm"]["base_url"] == "http://x/v1"
    assert merged["render_split"]["overlay_format"] == "webm"


def test_disabled_switches_translate_to_pipeline_flags(tmp_path):
    root = make_checkout(tmp_path)
    config = LongVideoConfig.from_dict({
        "pipeline": {"root": str(root)},
        "models": {"default": {"base_url": "http://x/v1", "model": "m"}},
        "tool_switches": {"tts": False, "hyperframes_render": False, "final_stitch": False},
    })
    pipeline_config, _, _ = load_pipeline_assets(config)
    merged = apply_pipeline_overrides(pipeline_config, config)

    assert merged["tts"]["backend"] == "none"
    assert merged["execution"]["fake_mode"]["skip_hyperframes_render"] is True
    assert merged["execution"]["fake_mode"]["final_hyperframes_render"] is False
    assert merged["execution"]["fake_mode"]["skip_final_stitch"] is True


def test_infer_config_matches_orchestrator_expectations(tmp_path):
    """These exact keys are what InferenceOrchestrator.__init__ reads."""
    root = make_checkout(tmp_path)
    config = LongVideoConfig.from_dict({
        "pipeline": {"root": str(root)},
        "models": {"default": {"base_url": "http://x/v1", "model": "m"}},
        "agent": {"max_steps": 30, "max_same_tool_repeats": 5},
    })
    infer = build_infer_config(config)
    assert infer["agent_mode"] == "long_horizon"
    assert infer["agent"] == {"max_steps": 30, "max_same_tool_repeats": 5}
    assert set(infer["tool_switches"]) == {
        "tts", "visual_retrieval", "bgm_search",
        "timeline_compose", "hyperframes_render", "final_stitch",
    }
