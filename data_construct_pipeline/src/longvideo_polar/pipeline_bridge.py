"""Bridge into the long-horizon agent loop.

The loop lives in ``server.infer_graph.InferenceOrchestrator``. Two checkout
layouts are supported:

- **WevisionCut** (this repository): ``pipeline/`` and ``server/`` sit at the
  repo root. ``pipeline.root`` can point here and nothing else is required.
- **Legacy AICutPipeline**: the inference package lives under
  ``Infer_Vis_new/server`` and a second ``server`` package (TTS) sits at the
  repo root. The Infer_Vis_new tree is put first on ``sys.path`` so the agent
  loop wins the name.

:func:`bootstrap_pipeline_paths` puts the right roots on ``sys.path`` so the
loop can be imported and driven from here.

Nothing in AICutPipeline is modified. The only substitution is the model loader:
:class:`~longvideo_polar.model_client.RoleRoutedModelClient` stands in for
``UnifiedModelLoader``, which is what lets the run happen over HTTP (through
Polar's gateway) and be recorded per role.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from longvideo_polar.config import LongVideoConfig, deep_merge, expand_env

logger = logging.getLogger(__name__)

_INFER_DIR_NAME = "Infer_Vis_new"


def _infer_dir_for(root: Path) -> Path:
    """Return the directory that owns ``server.infer_graph``.

    Prefer the legacy ``Infer_Vis_new`` tree when it exists — that checkout
    also has a TTS ``server`` package at the repo root, and only the nested
    one is the agent loop. Fall back to the WevisionCut layout, where
    ``server/`` at the repo root *is* the loop.
    """
    legacy = root / _INFER_DIR_NAME
    if (legacy / "server" / "infer_graph.py").is_file():
        return legacy
    if (root / "server" / "infer_graph.py").is_file():
        return root
    raise FileNotFoundError(
        f"missing server/infer_graph.py under {root} "
        f"(looked at {root / 'server'} and {legacy / 'server'}); "
        "the long-horizon agent loop lives there"
    )


def bootstrap_pipeline_paths(pipeline_root: str | Path) -> tuple[Path, Path]:
    """Put the video-editing environment and its inference server on ``sys.path``.

    Args:
        pipeline_root: Path to a WevisionCut or AICutPipeline checkout.

    Returns:
        ``(pipeline_root, infer_dir)`` as resolved absolute paths.

    Raises:
        FileNotFoundError: If the checkout or its inference server is missing.
    """
    root = Path(pipeline_root).expanduser().resolve()
    if not (root / "pipeline").is_dir():
        raise FileNotFoundError(
            f"{root} does not look like a WevisionCut / AICutPipeline checkout "
            "(no pipeline/ dir)"
        )
    infer_dir = _infer_dir_for(root)
    # Insert root first so infer_dir lands at index 0 when the two differ.
    # When they are the same (WevisionCut), insert once.
    ordered = [str(root)]
    if str(infer_dir) != str(root):
        ordered.append(str(infer_dir))
    for path in ordered:
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    # A `server` imported before this call (or from a different checkout)
    # would keep shadowing the inference package for the rest of the process.
    stale = sys.modules.get("server")
    stale_paths = list(getattr(stale, "__path__", []) or [])
    if stale is not None and str(infer_dir / "server") not in stale_paths:
        for name in [key for key in sys.modules if key == "server" or key.startswith("server.")]:
            del sys.modules[name]
        logger.debug("dropped shadowing `server` package from %s", stale_paths)
    return root, infer_dir


@dataclass
class PipelineContext:
    """A configured, ready-to-run AICutPipeline agent loop."""

    orchestrator: Any
    pipeline_config: dict[str, Any]
    infer_config: dict[str, Any]
    stage_config: dict[str, Any]
    system_prompt: str
    pipeline_root: Path


def load_pipeline_assets(config: LongVideoConfig) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Load the runtime config, unified system prompt and stage table.

    Returns:
        ``(pipeline_config, system_prompt, stage_config)``.
    """
    settings = config.pipeline
    config_path = settings.resolve_existing(
        settings.pipeline_config,
        "config/pipeline_runtime.yaml",
        "data_construct_pipeline/config/pipeline_runtime.yaml",
        "config.yaml",
        required=True,
        kind="pipeline config",
    )
    pipeline_config = expand_env(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
    if not isinstance(pipeline_config, dict):
        raise ValueError(f"pipeline config must be a mapping: {config_path}")

    prompt_path = settings.resolve_existing(
        settings.system_prompt_file,
        "configs/unified_v5_system_prompt.txt",
        "preprocess_ori_data/configs/unified_v5_system_prompt.txt",
        required=True,
        kind="system prompt",
    )
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    stage_path = settings.resolve_existing(
        settings.stage_config_file,
        "configs/stage_allowed_tools.yaml",
        "preprocess_ori_data/configs/stage_allowed_tools.yaml",
        required=False,
        kind="stage config",
    )
    stage_config: dict[str, Any] = {}
    if stage_path is not None and stage_path.is_file():
        loaded = yaml.safe_load(stage_path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            stage_config = loaded
    else:
        logger.warning("stage config not found, continuing without it")

    return pipeline_config, system_prompt, stage_config


def apply_pipeline_overrides(
    pipeline_config: dict[str, Any], config: LongVideoConfig
) -> dict[str, Any]:
    """Turn longvideo_polar settings into AICutPipeline config mutations.

    Mirrors what ``Infer_Vis_new/server/infer_server.py`` does at startup: force
    real execution, translate tool switches into ``execution.fake_mode`` flags, and
    point the pipeline's own LLM client at the served endpoint so any tool that
    builds an ``LLMClient`` directly still reaches the same model.
    """
    merged = deep_merge(pipeline_config, config.pipeline.config_overrides or {})

    execution = merged.setdefault("execution", {})
    execution["mode"] = "real"

    switches = config.tool_switches
    fake_mode = execution.setdefault("fake_mode", {})
    fake_mode["skip_final_stitch"] = not switches.final_stitch
    fake_mode["final_hyperframes_render"] = switches.hyperframes_render
    fake_mode["skip_hyperframes_render"] = not switches.hyperframes_render

    tts = merged.setdefault("tts", {})
    if config.pipeline.tts_service_url:
        tts["service_url"] = config.pipeline.tts_service_url
    if not switches.tts:
        tts["backend"] = "none"

    # Direct LLMClient users (e.g. VLM judging) should hit the same endpoint the
    # roles use, so traffic stays observable at one place.
    generator = config.endpoint_for("generator")
    llm = merged.setdefault("llm", {})
    llm.update({
        "base_url": generator.base_url,
        "api_key": generator.api_key,
        "model": generator.model,
        "timeout": generator.timeout,
    })

    if config.pipeline.render_split:
        merged.setdefault("render_split", {}).update(config.pipeline.render_split)
    return merged


def build_infer_config(config: LongVideoConfig) -> dict[str, Any]:
    """Build the ``infer_config`` mapping ``InferenceOrchestrator`` expects."""
    return {
        "agent_mode": config.agent.agent_mode,
        "agent": {
            "max_steps": config.agent.max_steps,
            "max_same_tool_repeats": config.agent.max_same_tool_repeats,
        },
        "tool_switches": config.tool_switches.as_dict(),
        "tts_service_url": config.pipeline.tts_service_url,
        "render_split": dict(config.pipeline.render_split or {}),
    }


def install_render_split(pipeline_config: dict[str, Any]) -> bool:
    """Enable the transparent-overlay render acceleration, if configured.

    Returns:
        Whether the patch was installed. Failures are non-fatal: the pipeline
        falls back to full-scene rendering.
    """
    settings = pipeline_config.get("render_split") or {}
    if not settings.get("enabled"):
        return False
    try:
        from server.render_split import install  # type: ignore[import-not-found]

        installed = bool(install(pipeline_config))
        if installed:
            logger.info("render-split overlay pipeline active")
        return installed
    except Exception as exc:  # noqa: BLE001 - optimization only
        logger.warning("render-split install skipped: %s", exc)
        return False


def build_pipeline_context(
    config: LongVideoConfig,
    *,
    model_client: Any,
    broadcast_fn: Any,
    episode_id: str = "",
) -> PipelineContext:
    """Assemble a ready-to-run :class:`PipelineContext`.

    Args:
        config: Resolved longvideo_polar config.
        model_client: A ``UnifiedModelLoader``-compatible client, normally
            :class:`~longvideo_polar.model_client.RoleRoutedModelClient`.
        broadcast_fn: ``async (run_id, event) -> None`` sink for pipeline events.
        episode_id: Tags this episode's template choices in the shared coverage
            ledger.

    Returns:
        The configured context, with ``orchestrator.run_agentic_pipeline`` ready
        to await.
    """
    root, infer_dir = bootstrap_pipeline_paths(config.pipeline.root)
    logger.info("AICutPipeline bootstrapped: root=%s infer=%s", root, infer_dir)

    pipeline_config, system_prompt, stage_config = load_pipeline_assets(config)
    pipeline_config = apply_pipeline_overrides(pipeline_config, config)
    infer_config = build_infer_config(config)
    install_render_split(pipeline_config)

    from server.infer_graph import InferenceOrchestrator  # type: ignore[import-not-found]

    # Tool_StitchVideo is the only tool that breaks the agent loop, and it demands
    # a timeline_render flag that a disabled compose stage never sets. Let calling
    # a switched-off stage stand in for finishing it, so the loop can terminate
    # without losing the ordering constraints that force the quality arc to run.
    from longvideo_polar.switch_policy import install_switch_aware_preconditions

    install_switch_aware_preconditions(InferenceOrchestrator, config.tool_switches)

    # Normalize the template registry and govern how spotlight templates are
    # chosen and filled. Without it, `template_retrieve` sees 90 template ids whose
    # `keywords` and `name` are empty (the registry stores `use_for` and
    # `display_name`), so every shot picks from an unlabeled list with no memory of
    # earlier choices — measured on a real rollout, 32 retrievals collapsed onto 7
    # distinct templates.
    decoration = config.decoration
    if decoration.enabled:
        from longvideo_polar.decoration_policy import install_decoration_policy

        install_decoration_policy(
            InferenceOrchestrator,
            ledger_path=decoration.ledger_path or None,
            slot_cache_path=decoration.slot_cache_path or None,
            episode_id=episode_id,
            pool_size=decoration.pool_size,
            semantic_slice=decoration.semantic_slice,
            coverage_slice=decoration.coverage_slice,
        )
    else:
        # Fall back to the older, narrower fix so disabling the policy does not
        # reintroduce empty candidate metadata.
        from longvideo_polar.template_enricher import install_template_enricher

        install_template_enricher(InferenceOrchestrator)

    if hasattr(model_client, "load"):
        model_client.load(orchestrator_system_prompt=system_prompt)

    orchestrator = InferenceOrchestrator(
        model_loader=model_client,
        pipeline_config=pipeline_config,
        stage_config=stage_config,
        system_prompt=system_prompt,
        broadcast_fn=broadcast_fn,
        infer_config=infer_config,
    )
    if not orchestrator.is_long_horizon_agent():
        raise ValueError(
            f"agent.agent_mode={config.agent.agent_mode!r} does not select the "
            "long-horizon loop; use 'long_horizon'"
        )
    return PipelineContext(
        orchestrator=orchestrator,
        pipeline_config=pipeline_config,
        infer_config=infer_config,
        stage_config=stage_config,
        system_prompt=system_prompt,
        pipeline_root=root,
    )
