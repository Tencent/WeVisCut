"""Studio configuration: load, validate, and project onto downstream configs.

`studio_config.yaml` is the single user-facing config. This module turns it into
the three dicts the existing AICut machinery expects:

  * ``infer_config``    — what ``server.model_loader`` / ``InferenceOrchestrator`` read
  * ``pipeline_config`` — what ``pipeline.tools.*`` read (services, paths, tts, ...)
  * ``studio_config``   — studio-only settings (progress tracks, editor, chat_edit)

Nothing else in the studio reads YAML or hardcodes a path, so retargeting a
deployment is a single-file change.

Run ``python3 -m studio.config --check`` for a GPU-free sanity check.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

STUDIO_DIR = Path(__file__).resolve().parents[1]
# The inference engine (server/) and the pipeline tools (pipeline/) both live at
# the repo root, so there is no external checkout to point at.
PIPELINE_ROOT = STUDIO_DIR
REFERENCE_IMPL_DIR = STUDIO_DIR

DEFAULT_CONFIG_PATH = STUDIO_DIR / "studio_config.yaml"

# `${VAR}` / `${VAR:-fallback}` inside any string value in studio_config.yaml.
_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^{}]*))?\}")
_MAX_INTERPOLATION_PASSES = 5

# Tools the studio can toggle. Maps studio service keys -> orchestrator switches.
_SERVICE_TO_SWITCH = {
    "tts": "tts",
    "visual_retrieval": "visual_retrieval",
    "bgm": "bgm_search",
    "hyperframes": "hyperframes_render",
}


class ConfigError(RuntimeError):
    """Raised when the config is structurally unusable."""


@dataclass
class Issue:
    level: str  # "error" | "warn"
    where: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - display only
        mark = "ERROR" if self.level == "error" else "WARN "
        return f"[{mark}] {self.where}: {self.message}"


@dataclass
class StudioConfig:
    """Resolved configuration bundle."""

    raw: dict[str, Any]
    path: Path
    infer_config: dict[str, Any] = field(default_factory=dict)
    pipeline_config: dict[str, Any] = field(default_factory=dict)
    system_prompt: str = ""
    stage_config: dict[str, Any] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)

    # ---- convenience accessors (studio-only sections) ----
    @property
    def studio(self) -> dict[str, Any]:
        return _as_dict(self.raw.get("studio"))

    @property
    def server(self) -> dict[str, Any]:
        return _as_dict(self.raw.get("server"))

    @property
    def progress(self) -> dict[str, Any]:
        return _as_dict(self.raw.get("progress"))

    @property
    def editor(self) -> dict[str, Any]:
        return _as_dict(self.raw.get("editor"))

    @property
    def chat_edit(self) -> dict[str, Any]:
        return _as_dict(self.raw.get("chat_edit"))

    @property
    def autonomy(self) -> dict[str, Any]:
        return _as_dict(_as_dict(self.raw.get("agent")).get("autonomy"))

    @property
    def workspace_root(self) -> Path:
        return Path(
            self.studio.get("workspace_root")
            or _as_dict(self.raw.get("paths")).get("output_root")
            or "/tmp/aicut_studio"
        )

    @property
    def media_roots(self) -> list[Path]:
        roots = self.server.get("media_roots") or []
        out = [Path(str(r)).resolve() for r in roots if str(r).strip()]
        # The workspace always serves its own artifacts.
        ws = self.workspace_root.resolve()
        if ws not in out:
            out.append(ws)
        return out

    @property
    def active_profile(self) -> str:
        return str(_as_dict(self.raw.get("model")).get("active_profile", "") or "")

    @property
    def chat_enabled(self) -> bool:
        return bool(self.chat_edit.get("enabled", False))

    @property
    def editor_enabled(self) -> bool:
        return bool(self.editor.get("enabled", False))

    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warn"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _resolve(base: Path, value: Any) -> str:
    """Resolve a possibly-relative path against `base`."""
    text = str(value or "").strip()
    if not text:
        return ""
    p = Path(text)
    return str(p if p.is_absolute() else (base / p).resolve())


def interpolate(value: Any, variables: dict[str, str]) -> Any:
    """Expand `${VAR}` and `${VAR:-fallback}` throughout a parsed YAML tree.

    Lookup order per name is `variables` (which carries the injected
    `REPO_ROOT`) then the process environment. Unset names with no fallback
    expand to an empty string, which the validator then reports as a missing
    setting rather than letting a literal `${...}` reach a filesystem call.
    Expansion repeats until stable so a fallback may itself reference
    `${REPO_ROOT}`.
    """
    if isinstance(value, dict):
        return {k: interpolate(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate(v, variables) for v in value]
    if not isinstance(value, str) or "${" not in value:
        return value

    def _sub(match: re.Match[str]) -> str:
        name, fallback = match.group(1), match.group(2)
        resolved = variables.get(name)
        if resolved is None:
            resolved = os.environ.get(name)
        if resolved is None or resolved == "":
            resolved = fallback if fallback is not None else ""
        return resolved

    text = value
    for _ in range(_MAX_INTERPOLATION_PASSES):
        expanded = _ENV_REF_RE.sub(_sub, text)
        if expanded == text:
            break
        text = expanded
    return text


# ---------------------------------------------------------------------------
# load + project
# ---------------------------------------------------------------------------
def load(path: str | Path | None = None) -> StudioConfig:
    """Load studio_config.yaml and project it onto downstream configs."""
    cfg_path = Path(
        path
        or os.environ.get("AICUT_STUDIO_CONFIG")
        or DEFAULT_CONFIG_PATH
    ).resolve()

    if not cfg_path.exists():
        raise ConfigError(f"config not found: {cfg_path}")

    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {cfg_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path} must contain a YAML mapping at the top level")

    base = cfg_path.parent
    # `${REPO_ROOT}` is the directory holding the config, so the shipped
    # defaults work from a fresh clone with no environment set up at all.
    raw = interpolate(raw, {"REPO_ROOT": str(base)})

    cfg = StudioConfig(raw=raw, path=cfg_path)

    cfg.infer_config = _build_infer_config(cfg, base)
    cfg.pipeline_config = _build_pipeline_config(cfg)
    cfg.system_prompt = _read_prompt_file(cfg, base)
    cfg.stage_config = _read_stage_config(cfg, base)

    _validate(cfg)
    return cfg


def _build_infer_config(cfg: StudioConfig, base: Path) -> dict[str, Any]:
    """Project studio config onto the shape `InferenceOrchestrator` expects."""
    model = _as_dict(cfg.raw.get("model"))
    profiles = _as_dict(model.get("profiles"))

    active = os.environ.get("AICUT_MODEL_PROFILE", "").strip() or str(
        model.get("active_profile", "") or ""
    )
    profile = _as_dict(profiles.get(active))

    merged = {k: v for k, v in model.items() if k != "profiles"}
    merged.update(profile)
    merged["profiles"] = profiles
    merged["active_profile"] = active
    merged.setdefault("infer_backend", "transformers")
    # Qwen3.5 default is empty-think prefix unless this is on. Match SFT/DAPO.
    merged.setdefault("enable_thinking", True)
    # model_loader reads max_new_tokens_by_stage for the agent loop
    merged.setdefault(
        "max_new_tokens_by_stage",
        {"agent_loop": int(merged.get("max_new_tokens", 24576) or 24576)},
    )

    services = _as_dict(cfg.raw.get("services"))
    tts = _as_dict(services.get("tts"))

    switches = {
        switch: bool(_as_dict(services.get(svc)).get("enabled", True))
        for svc, switch in _SERVICE_TO_SWITCH.items()
    }
    # These two have no dedicated service block; they follow the render config.
    switches["timeline_compose"] = True
    switches["final_stitch"] = True

    agent = _as_dict(cfg.raw.get("agent"))
    render = _as_dict(cfg.raw.get("render"))

    return {
        "deployment_mode": "single_model",
        "model": merged,
        "agent_mode": str(agent.get("mode", "long_horizon") or "long_horizon"),
        "agent": {
            "max_steps": int(agent.get("max_steps", 48) or 48),
            "max_same_tool_repeats": int(agent.get("max_same_tool_repeats", 10) or 10),
        },
        "autonomy": _as_dict(agent.get("autonomy")),
        "tool_switches": switches,
        "tts_service_url": str(tts.get("service_url", "http://127.0.0.1:7865")),
        "render_split": _as_dict(render.get("split")),
        "server": {
            "host": str(cfg.server.get("host", "0.0.0.0")),
            "port": int(cfg.server.get("port", 8095) or 8095),
        },
        "system_prompt_file": _resolve(base, model.get("system_prompt_file")),
        "stage_config_file": _resolve(base, model.get("stage_config_file")),
    }


def _build_pipeline_config(cfg: StudioConfig) -> dict[str, Any]:
    """Project studio config onto the shape `pipeline.tools.*` expect."""
    services = _as_dict(cfg.raw.get("services"))
    paths = _as_dict(cfg.raw.get("paths"))
    render = _as_dict(cfg.raw.get("render"))
    quality = _as_dict(cfg.raw.get("quality"))

    tts = _as_dict(services.get("tts"))
    retrieval = _as_dict(services.get("visual_retrieval"))
    bgm = _as_dict(services.get("bgm"))
    hyperframes = copy.deepcopy(_as_dict(services.get("hyperframes")))
    ffmpeg = _as_dict(services.get("ffmpeg"))

    # hyperframes wants ffmpeg/ffprobe inside its own block
    hyperframes.setdefault("ffmpeg_path", ffmpeg.get("ffmpeg_path", ""))
    hyperframes.setdefault("ffprobe_path", ffmpeg.get("ffprobe_path", ""))
    hyperframes.pop("enabled", None)

    pipeline_cfg: dict[str, Any] = {
        "paths": {
            "output_root": str(paths.get("output_root", "")),
            "storage_root": str(paths.get("storage_root", "")),
            "asset_library": str(paths.get("asset_library", "")),
            "gold_template_root": str(paths.get("gold_template_root", "")),
            "gold_template_policies_file": str(paths.get("gold_template_policies_file", "")),
            "instructions_file": str(paths.get("instructions_file", "")),
        },
        "services": {
            "hyperframes": hyperframes,
        },
        # `llm` is rewritten at startup to point at the studio's own /v1 endpoint,
        # so the 9B policy model also serves the pipeline's internal LLM calls.
        "llm": {},
        "execution": {
            "mode": "real",
            "fake_mode": {
                "skip_hyperframes_render": not bool(hyperframes and services.get("hyperframes", {}).get("enabled", True)),
                "final_hyperframes_render": bool(_as_dict(services.get("hyperframes")).get("enabled", True)),
                "skip_final_stitch": False,
                "skip_bgm_search": not bool(bgm.get("enabled", True)),
                "fake_vlm_analysis": False,
                "hitl_mode": "auto",
            },
        },
        "tts": {
            "backend": str(tts.get("backend", "qwen3_service")),
            "service_url": str(tts.get("service_url", "http://127.0.0.1:7865")),
            "timeout_sec": int(tts.get("timeout_sec", 1200) or 1200),
            "voice": str(tts.get("voice", "Serena")),
            "language": str(tts.get("language", "Chinese")),
            "instruct": str(tts.get("instruct", "")),
            "speed": float(tts.get("speed", 1.0) or 1.0),
            "dispatch_mode": str(tts.get("dispatch_mode", "batch")),
            "batch_size": int(tts.get("batch_size", 64) or 64),
            "batch_concurrency": int(tts.get("batch_concurrency", 1) or 1),
            "health_probe_timeout_sec": int(tts.get("health_probe_timeout_sec", 3) or 3),
            "require_true_batch": True,
        },
        # Footage source. `bgm_provider` defaults to the same value so a
        # single-knob config keeps working; see docs/providers.md.
        "retrieval_provider": str(
            services.get("retrieval_provider", "asset_retrieval_tool")
            or "asset_retrieval_tool"
        ),
        "bgm_provider": str(
            services.get("bgm_provider")
            or services.get("retrieval_provider", "asset_retrieval_tool")
            or "asset_retrieval_tool"
        ),
        "providers": _as_dict(services.get("providers")),
        "retrieval": {
            k: v for k, v in retrieval.items() if k != "enabled"
        },
        "asset_retrieval": _as_dict(services.get("asset_retrieval")),
        "music": {
            "enabled": bool(bgm.get("enabled", True)),
            **{k: v for k, v in bgm.items() if k != "enabled"},
        },
        "timeline": {
            "audio_tail_padding_sec": 0.15,
            "rendered_duration_tolerance_sec": 0.35,
            "decorations": {"burn_in_text_overlay": False},
            "subtitles": _as_dict(render.get("subtitles")),
        },
        "stitch": _as_dict(render.get("stitch")),
        "render_split": _as_dict(render.get("split")),
        # `pipeline.graph.node_severity_rank` reads its knobs from here. Without
        # this block every lookup fell through to the code defaults, and
        # `allow_heuristic_fallback` defaulted to False — so any imperfect LLM
        # reply raised, the orchestrator swallowed it, and every shot ended up
        # scored 0.0 with an empty top list (Stage4 then had nothing to judge).
        "stage3_trajectory": {
            "severity_rank": _severity_rank_config(quality),
        },
        "hitl": {"enabled": True, "mode": "auto", "num_options": 4, "decision_timeout_sec": 0},
        "modal_tools_preflight": {"enabled": False},
    }
    return pipeline_cfg


def _severity_rank_config(quality: dict[str, Any]) -> dict[str, Any]:
    """Build `stage3_trajectory.severity_rank` from the studio `quality` block."""
    raw = _as_dict(quality.get("severity_rank"))
    return {
        "enable": bool(raw.get("enable", True)),
        "top_ratio": float(raw.get("top_ratio", 0.25) or 0.25),
        "min_shots_to_process": int(raw.get("min_shots_to_process", 4) or 4),
        "llm_temperature": float(raw.get("llm_temperature", 0.1) or 0.1),
        "max_retries": int(raw.get("max_retries", 2) or 2),
        "allow_heuristic_fallback": bool(raw.get("allow_heuristic_fallback", True)),
    }


def _read_prompt_file(cfg: StudioConfig, base: Path) -> str:
    path = cfg.infer_config.get("system_prompt_file", "")
    if path and Path(path).exists():
        return Path(path).read_text(encoding="utf-8").strip()
    return ""


def _read_stage_config(cfg: StudioConfig, base: Path) -> dict[str, Any]:
    path = cfg.infer_config.get("stage_config_file", "")
    if path and Path(path).exists():
        try:
            return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            return {}
    return {}


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def _validate(cfg: StudioConfig) -> None:
    """Collect issues. Missing model weights are warnings, not errors, so the
    studio still boots on a machine where only some profiles are materialised."""
    add = cfg.issues.append
    model = _as_dict(cfg.raw.get("model"))
    profiles = _as_dict(model.get("profiles"))

    # --- model ---
    if not profiles:
        add(Issue("error", "model.profiles", "no profiles defined"))
    active = cfg.active_profile
    if not active:
        add(Issue("error", "model.active_profile", "not set"))
    elif active not in profiles:
        add(Issue(
            "error", "model.active_profile",
            f"{active!r} is not in profiles ({', '.join(profiles) or 'none'})",
        ))
    else:
        resolved = _as_dict(cfg.infer_config.get("model"))
        base_model = str(resolved.get("base_model", "") or "")
        if not base_model:
            add(Issue("error", f"model.profiles.{active}.base_model", "empty"))
        elif not Path(base_model).exists():
            add(Issue("warn", f"model.profiles.{active}.base_model", f"path missing: {base_model}"))
        adapter = str(resolved.get("adapter", "") or "")
        if adapter and not Path(adapter).exists():
            add(Issue("warn", f"model.profiles.{active}.adapter", f"path missing: {adapter}"))
        backend = str(resolved.get("infer_backend", ""))
        if backend not in {"vllm", "transformers"}:
            add(Issue("error", f"model.profiles.{active}.infer_backend",
                      f"must be vllm|transformers, got {backend!r}"))

    # --- training/inference alignment ---
    if not cfg.system_prompt:
        add(Issue("error", "model.system_prompt_file",
                  f"unreadable: {cfg.infer_config.get('system_prompt_file') or '(unset)'}"
                  " — the agent must use the same system prompt as training"))
    if not cfg.stage_config.get("allowed_tools"):
        add(Issue("warn", "model.stage_config_file",
                  "no allowed_tools found; long-horizon mode does not need it, "
                  "but staged mode would"))

    # --- required directories ---
    for key in ("gold_template_root", "asset_library"):
        val = str(_as_dict(cfg.raw.get("paths")).get(key, "") or "")
        if not val:
            add(Issue("error", f"paths.{key}", "empty"))
        elif not Path(val).exists():
            add(Issue("warn", f"paths.{key}", f"path missing: {val}"))

    # --- asset providers ---
    _validate_providers(cfg, add)

    # --- external binaries ---
    hf = _as_dict(_as_dict(cfg.raw.get("services")).get("hyperframes"))
    if hf.get("enabled", True):
        for key in ("hf_bin", "browser_path"):
            val = str(hf.get(key, "") or "")
            if not val:
                add(Issue("warn", f"services.hyperframes.{key}", "empty"))
            elif not Path(val).exists():
                add(Issue("warn", f"services.hyperframes.{key}", f"missing: {val}"))
    ff = _as_dict(_as_dict(cfg.raw.get("services")).get("ffmpeg"))
    for key in ("ffmpeg_path", "ffprobe_path"):
        val = str(ff.get(key, "") or "")
        if val and not Path(val).exists():
            add(Issue("warn", f"services.ffmpeg.{key}", f"missing: {val}"))

    # --- subtitle font (ffmpeg drawtext fails hard without it) ---
    font = str(_as_dict(_as_dict(cfg.raw.get("render")).get("subtitles")).get("fontfile", "") or "")
    if font and not Path(font).exists():
        add(Issue("warn", "render.subtitles.fontfile", f"missing: {font}"))

    # --- server ---
    port = cfg.server.get("port")
    if not isinstance(port, int) or not (1 <= port <= 65535):
        add(Issue("error", "server.port", f"invalid: {port!r}"))
    if not cfg.media_roots:
        add(Issue("error", "server.media_roots", "at least one root required to serve media"))

    # --- progress tracks ---
    known_tracks = {
        "segments", "shots", "narration", "subtitle",
        "footage", "spotlight", "bgm", "render",
    }
    for t in cfg.progress.get("tracks") or []:
        if str(t) not in known_tracks:
            add(Issue("warn", "progress.tracks", f"unknown track {t!r} will be ignored"))

    # --- chat_edit ---
    if cfg.chat_enabled:
        chat = cfg.chat_edit
        backend = str(chat.get("backend", "inproc"))
        if backend not in {"inproc", "openai"}:
            add(Issue("error", "chat_edit.backend", f"must be inproc|openai, got {backend!r}"))
        if backend == "openai" and not str(chat.get("openai_base_url", "") or ""):
            add(Issue("error", "chat_edit.openai_base_url", "required when backend=openai"))
        ctx = _as_dict(chat.get("context"))
        budget = int(ctx.get("max_history_tokens", 0) or 0)
        summary = int(ctx.get("summary_target_tokens", 0) or 0)
        if budget and summary and summary >= budget:
            add(Issue("error", "chat_edit.context",
                      "summary_target_tokens must be well below max_history_tokens"))

    # --- reference implementation must be importable ---
    if not (REFERENCE_IMPL_DIR / "server" / "infer_graph.py").exists():
        add(Issue("error", "internal",
                  f"agent engine not found at {REFERENCE_IMPL_DIR / 'server'}"))


def _validate_providers(cfg: StudioConfig, add: Any) -> None:
    """Check that the configured footage/BGM providers actually resolve.

    A typo here is otherwise silent until the first retrieval call, halfway
    through a generation run.
    """
    pipeline = cfg.pipeline_config
    builtin_video = {"mmfinder", "asset_retrieval_tool"}
    builtin_bgm = {"mmfinder", "asset_retrieval_tool"}

    try:
        install_import_paths()
        from pipeline.providers import (  # noqa: PLC0415 - optional at config time
            list_bgm_providers,
            list_video_providers,
            load_provider_plugins,
        )
    except Exception as exc:  # noqa: BLE001
        add(Issue("warn", "services.providers", f"provider registry unavailable: {exc}"))
        return

    plugins = _as_dict(pipeline.get("providers")).get("plugins")
    requested = [str(p).strip() for p in (plugins or []) if str(p).strip()]
    loaded = set(load_provider_plugins(plugins))
    for spec in requested:
        if spec not in loaded:
            add(Issue("warn", "services.providers.plugins", f"could not import {spec!r}"))

    video = str(pipeline.get("retrieval_provider", "") or "")
    if video and video not in builtin_video and video not in list_video_providers():
        add(Issue(
            "error", "services.retrieval_provider",
            f"{video!r} is neither a built-in backend ({', '.join(sorted(builtin_video))}) "
            f"nor registered ({', '.join(list_video_providers()) or 'none'})",
        ))

    bgm = str(pipeline.get("bgm_provider", "") or "")
    if bgm and bgm not in builtin_bgm and bgm not in list_bgm_providers():
        add(Issue(
            "error", "services.bgm_provider",
            f"{bgm!r} is neither a built-in backend ({', '.join(sorted(builtin_bgm))}) "
            f"nor registered ({', '.join(list_bgm_providers()) or 'none'})",
        ))


def install_import_paths() -> None:
    """Put the project root on sys.path so the bundled packages import cleanly.

    The inference engine (``server/``), the pipeline tools (``pipeline/``), the
    backend (``studio/``) and the deepagents integration layer
    (``aicut_agent/``) all live under the repo root, so one entry is enough.
    """
    root = str(STUDIO_DIR)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _first(value: Any, fallback: str = "-") -> str:
    return str(value[0]) if isinstance(value, list) and value else fallback


def _print_interfaces(cfg: StudioConfig) -> None:
    """Show the resolved external interfaces.

    Printed at startup because the most common failure mode is editing the
    config and not noticing it had no effect.
    """
    pipeline = cfg.pipeline_config
    services = _as_dict(cfg.raw.get("services"))

    def state(name: str) -> str:
        block = _as_dict(services.get(name))
        return "on " if block.get("enabled", True) else "off"

    video_provider = str(pipeline.get("retrieval_provider", "") or "-")
    bgm_provider = str(pipeline.get("bgm_provider", "") or "-")
    # backend_order is the mmfinder cascade; every other provider replaces it,
    # so printing it unconditionally would suggest a backend that never runs.
    cascade = (
        f" · {_first(retrieval.get('backend_order'))}"
        if video_provider == "mmfinder"
        else ""
    )
    tts = _as_dict(pipeline.get("tts"))
    retrieval = _as_dict(pipeline.get("retrieval"))
    music = _as_dict(pipeline.get("music"))
    hyperframes = _as_dict(_as_dict(pipeline.get("services")).get("hyperframes"))
    paths = _as_dict(pipeline.get("paths"))
    model = _as_dict(cfg.infer_config.get("model"))

    rows = [
        ("model", f"{cfg.active_profile} · {model.get('infer_backend', '')}"),
        ("  weights", str(model.get("base_model", "") or "-")),
        ("  adapter", str(model.get("adapter", "") or "-")),
        ("  thinking", "on" if model.get("enable_thinking", True) else "off (empty <think>)"),
        (f"tts [{state('tts')}]",
         f"{tts.get('service_url', '') or '-'} · voice={tts.get('voice', '')}"),
        (f"footage [{state('visual_retrieval')}]",
         f"provider={video_provider}{cascade} · top_k={retrieval.get('top_k', '-')}"),
        (f"bgm [{state('bgm')}]",
         f"provider={bgm_provider} · top_k={music.get('top_k', '-')} · volume={music.get('bgm_volume', '-')}"),
        (f"hyperframes [{state('hyperframes')}]", str(hyperframes.get("hf_bin", "") or "-")),
        ("templates", str(paths.get("gold_template_root", "") or "-")),
        ("asset library", str(paths.get("asset_library", "") or "-")),
        ("workspace", str(cfg.workspace_root)),
    ]
    rows.append(("registered providers", _describe_registered_providers(pipeline)))
    width = max(len(k) for k, _ in rows)
    print("---- resolved external interfaces ----")
    for key, value in rows:
        print(f"  {key.ljust(width)} : {value}")
    print("")


def _describe_registered_providers(pipeline: dict[str, Any]) -> str:
    """List the pluggable providers available after loading configured plugins."""
    try:
        install_import_paths()
        from pipeline.providers import (  # noqa: PLC0415 - optional at config time
            list_bgm_providers,
            list_video_providers,
            load_provider_plugins,
        )
    except Exception as exc:  # noqa: BLE001 - summary output must never fail
        return f"(unavailable: {exc})"

    load_provider_plugins(_as_dict(pipeline.get("providers")).get("plugins"))
    return (
        f"video=[{', '.join(list_video_providers()) or '-'}] "
        f"bgm=[{', '.join(list_bgm_providers()) or '-'}]"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate studio_config.yaml")
    parser.add_argument("--config", default=None)
    parser.add_argument("--check", action="store_true", help="validate and exit")
    parser.add_argument("--summary", action="store_true",
                        help="print the resolved external interfaces and exit")
    parser.add_argument("--dump", choices=["infer", "pipeline"], help="print a projected config")
    args = parser.parse_args(argv)

    try:
        cfg = load(args.config)
    except ConfigError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if args.dump:
        target = cfg.infer_config if args.dump == "infer" else cfg.pipeline_config
        print(yaml.safe_dump(target, allow_unicode=True, sort_keys=False))
        return 0

    if args.summary:
        _print_interfaces(cfg)
        for issue in cfg.warnings():
            print(str(issue))
        return 1 if cfg.errors() else 0

    print(f"config      : {cfg.path}")
    print(f"profile     : {cfg.active_profile}")
    print(f"backend     : {_as_dict(cfg.infer_config.get('model')).get('infer_backend')}")
    print(f"agent mode  : {cfg.infer_config.get('agent_mode')}")
    print(f"autonomy    : fully_autonomous={cfg.autonomy.get('fully_autonomous')}")
    print(f"system promp: {len(cfg.system_prompt)} chars")
    print(f"workspace   : {cfg.workspace_root}")
    print(f"editor      : {'on' if cfg.editor_enabled else 'off'}")
    print(f"chat edit   : {'on' if cfg.chat_enabled else 'off'}")
    print(f"tracks      : {', '.join(str(t) for t in cfg.progress.get('tracks') or [])}")
    print("")

    if not cfg.issues:
        print("No issues found.")
        return 0

    for issue in cfg.issues:
        print(str(issue))

    n_err = len(cfg.errors())
    print("")
    print(f"{n_err} error(s), {len(cfg.warnings())} warning(s)")
    return 1 if n_err else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
