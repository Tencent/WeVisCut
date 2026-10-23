"""Configuration for longvideo_polar rollouts.

One YAML file drives everything: where AICutPipeline lives, which model endpoint
each of the three roles talks to, which media tools are enabled, and how much of
each trajectory gets written to disk. See ``config/longvideo_polar.yaml`` for the
annotated reference copy.

Values support ``${VAR}`` and ``${VAR:-default}`` environment expansion, which is
how a Polar runtime injects its gateway endpoint (``OPENAI_BASE_URL`` /
``OPENAI_API_KEY``) without the config file needing to know the port.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# `[^{}]*` for the default keeps this matching the *innermost* placeholder, so
# repeated application resolves nested forms from the inside out.
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^{}]*))?\}")

# Nesting depth allowed in `${A:-${B:-fallback}}`. Three levels covers
# "role-specific key, then shared key, then literal" and stops runaway expansion if
# an environment value happens to look like a placeholder.
_MAX_ENV_PASSES = 3

# <pkg-root>/src/longvideo_polar/config.py -> <pkg-root>/config/longvideo_polar.yaml
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "longvideo_polar.yaml"
# data_construct_pipeline/src/longvideo_polar/config.py -> WeVisCut repo root
_WEVISCUT_REPO_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in strings.

    Defaults may nest, as in ``${LVP_GEMINI_API_KEY:-${AICUT_302AI_API_KEY:-}}``,
    which reads as "role-specific key, else the shared gateway key, else empty".
    Without nesting support the outer placeholder would resolve to the literal text
    ``${AICUT_302AI_API_KEY:-}`` and get sent as an API key.
    """
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            return os.environ.get(name, default if default is not None else "")

        for _ in range(_MAX_ENV_PASSES):
            expanded = _ENV_PATTERN.sub(replace, value)
            if expanded == value:
                break
            value = expanded
        return value
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    return value


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Merge ``patch`` into ``base``, recursing into nested mappings."""
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


_ENDPOINT_FIELDS: tuple[str, ...] = (
    "provider", "base_url", "api_key", "model", "temperature", "top_p", "max_tokens",
    "timeout", "connect_timeout", "max_retries", "max_concurrency", "native_tool_calls",
    "endpoint_path", "auth_header_prefix", "extra_headers", "extra_body", "drop_params",
)


@dataclass
class ModelEndpoint:
    """An OpenAI-compatible chat endpoint for one role.

    "OpenAI-compatible" covers more than vLLM: 302AI-style gateways front Gemini and
    Claude on the same protocol, which is how the critic (反思修复) can run on a
    stronger third-party model while the other two roles stay on a local Qwen. The
    fields below exist because those gateways differ in small, breaking ways —
    auth header shape, endpoint path, and which sampling params they reject.
    """

    # Free-form label (``qwen`` / ``gemini`` / ``claude`` / ``openai``). It does not
    # change the wire protocol; it is recorded on every sample so mixed-provider
    # annotation stays traceable — you must be able to tell which rows came from
    # Opus and which from your own Qwen before training on them.
    provider: str = "qwen"
    base_url: str = ""
    api_key: str = "EMPTY"
    model: str = ""
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 8192
    timeout: float = 600.0
    connect_timeout: float = 20.0
    max_retries: int = 3
    # Cap in-flight requests per role; the pipeline fans out skill calls with
    # asyncio.gather, which can otherwise overwhelm a single served model — or blow
    # through a third-party rate limit.
    max_concurrency: int = 8
    # vLLM/SGLang serve tool calling natively; set false to force JSON-in-text.
    native_tool_calls: bool = True
    endpoint_path: str = "/chat/completions"
    # Some gateways want the raw key with no "Bearer " prefix; set "" for those.
    auth_header_prefix: str = "Bearer"
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    # Sampling params this endpoint rejects (e.g. some Gemini/Claude gateways 400 on
    # `top_p` alongside `temperature`). Listed keys are omitted from the payload.
    drop_params: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelEndpoint":
        kwargs = {key: payload[key] for key in _ENDPOINT_FIELDS if key in payload}
        for key in ("extra_headers", "extra_body"):
            if key in kwargs and not isinstance(kwargs[key], dict):
                kwargs[key] = {}
        if "drop_params" in kwargs:
            raw = kwargs["drop_params"]
            kwargs["drop_params"] = (
                [str(item) for item in raw] if isinstance(raw, (list, tuple)) else []
            )
        return cls(**kwargs)  # type: ignore[arg-type]

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "connect_timeout": self.connect_timeout,
            "max_retries": self.max_retries,
            "max_concurrency": self.max_concurrency,
            "native_tool_calls": self.native_tool_calls,
            "endpoint_path": self.endpoint_path,
            "auth_header_prefix": self.auth_header_prefix,
            "extra_headers": dict(self.extra_headers),
            "extra_body": dict(self.extra_body),
            "drop_params": list(self.drop_params),
        }

    def merged_with(self, override: dict[str, Any]) -> "ModelEndpoint":
        """Return a copy with ``override`` applied on top."""
        return ModelEndpoint.from_dict(deep_merge(self.as_dict(), override or {}))

    def describe(self) -> dict[str, Any]:
        """Secret-free provenance record for manifests and sample metadata."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "native_tool_calls": self.native_tool_calls,
        }

    def validate(self, label: str) -> None:
        if not str(self.base_url).strip():
            raise ValueError(
                f"models.{label}.base_url is empty — set it in the config or export "
                "OPENAI_BASE_URL (Polar injects this automatically)"
            )
        if not str(self.model).strip():
            raise ValueError(f"models.{label}.model is empty")
        if not str(self.api_key).strip():
            raise ValueError(
                f"models.{label}.api_key is empty — third-party providers "
                f"(provider={self.provider!r}) need a real key; use 'EMPTY' for a "
                "local vLLM"
            )


@dataclass
class PipelineSettings:
    """Where the video-editing environment lives and which configs to use."""

    root: str = ""
    pipeline_config: str = "config/pipeline_runtime.yaml"
    system_prompt_file: str = "configs/unified_v5_system_prompt.txt"
    stage_config_file: str = "configs/stage_allowed_tools.yaml"
    # Overrides deep-merged into the runtime config before the run.
    config_overrides: dict[str, Any] = field(default_factory=dict)
    render_split: dict[str, Any] = field(default_factory=dict)
    tts_service_url: str = "http://127.0.0.1:7865"

    def resolved_root(self) -> Path:
        text = str(self.root or "").strip()
        if not text:
            if (_WEVISCUT_REPO_ROOT / "pipeline").is_dir() and (
                (_WEVISCUT_REPO_ROOT / "server" / "infer_graph.py").is_file()
                or (_WEVISCUT_REPO_ROOT / "Infer_Vis_new" / "server" / "infer_graph.py").is_file()
            ):
                return _WEVISCUT_REPO_ROOT
            raise ValueError(
                "pipeline.root is empty — point it at the WeVisCut repo "
                "(the directory that contains pipeline/ and server/) or export "
                "AICUT_PIPELINE_ROOT / WEVISCUT_ROOT"
            )
        path = Path(text).expanduser()
        if not path.is_dir():
            raise ValueError(f"pipeline.root does not exist: {path}")
        return path.resolve()

    def resolve(self, relative: str) -> Path:
        """Resolve a path relative to the environment root, then this package."""
        candidate = Path(relative).expanduser()
        if candidate.is_absolute():
            return candidate
        under_root = self.resolved_root() / relative
        if under_root.exists():
            return under_root
        under_pkg = _PACKAGE_ROOT / relative
        if under_pkg.exists():
            return under_pkg
        return under_root

    def resolve_existing(
        self,
        *relatives: str,
        required: bool = True,
        kind: str = "file",
    ) -> Path | None:
        """Return the first existing candidate among ``relatives``.

        Each name is tried via :meth:`resolve` (environment root, then this
        package). Empty names are skipped.
        """
        tried: list[str] = []
        for relative in relatives:
            name = str(relative or "").strip()
            if not name:
                continue
            path = self.resolve(name)
            tried.append(str(path))
            if path.is_file():
                return path
        if not required:
            return None
        raise FileNotFoundError(
            f"{kind} not found; looked at: {', '.join(tried) or '(no candidates)'}"
        )


@dataclass
class AgentSettings:
    """Long-horizon loop bounds."""

    max_steps: int = 48
    max_same_tool_repeats: int = 10
    agent_mode: str = "long_horizon"


@dataclass
class ToolSwitches:
    """Media tool enable flags, independent of AICutPipeline's own config."""

    tts: bool = True
    visual_retrieval: bool = True
    bgm_search: bool = True
    timeline_compose: bool = True
    hyperframes_render: bool = True
    final_stitch: bool = True

    def as_dict(self) -> dict[str, bool]:
        return {
            "tts": self.tts,
            "visual_retrieval": self.visual_retrieval,
            "bgm_search": self.bgm_search,
            "timeline_compose": self.timeline_compose,
            "hyperframes_render": self.hyperframes_render,
            "final_stitch": self.final_stitch,
        }


@dataclass
class RecordingSettings:
    """What the recorder writes and where."""

    output_root: str = "./rollouts/longvideo_polar"
    max_raw_output_chars: int = 200_000
    store_full_prompts: bool = True
    # Persist whatever was captured even when the run crashes mid-way. Failed
    # trajectories are the useful ones for reflection/repair training, so this
    # defaults on.
    save_on_failure: bool = True
    strict_tool_schema: bool = True


@dataclass
class BatchSettings:
    """Batch annotation over a directory of 口播稿 files."""

    workers: int = 4
    input_dir: str = ""
    input_glob: str = "*.txt"
    limit: int = 0
    resume: bool = True
    continue_on_error: bool = True


@dataclass
class DecorationSettings:
    """Spotlight template selection and slot-filling policy.

    Left disabled, ``Infer_Vis_new`` offers every shot all 90 templates with their
    ``keywords`` and ``name`` empty and no memory of prior choices, which collapses
    a corpus onto a handful of templates. Enabling this narrows each shot to a
    ranked, coverage-balanced pool and rewrites the fill contract so slot text is
    derived from the shot's visual query and its segment summary rather than copied
    out of the narration.
    """

    enabled: bool = True
    #: Candidates offered per shot. Training ranked to 12; 14 leaves room for the
    #: coverage and stratification slices without drifting from that distribution.
    pool_size: int = 14
    #: Of the pool, how many come from semantic ranking.
    semantic_slice: int = 6
    #: How many are pulled from the globally least-used templates. Raising this
    #: flattens the corpus histogram at some cost to per-shot relevance.
    coverage_slice: int = 5
    #: SQLite ledger of cross-episode template usage. Empty disables the global
    #: coverage term, leaving only per-episode de-duplication.
    ledger_path: str = ""
    #: Consolidated slot-schema cache; avoids 90 shared-filesystem reads per
    #: rollout subprocess. Build it with ``longvideo-polar prepare``.
    slot_cache_path: str = ""


ROLE_NAMES: tuple[str, ...] = ("orchestrator", "generator", "critic")

# Marks a document whose profile overlay is already merged in. Re-applying it would
# undo any CLI override that intentionally contradicts the profile.
_PROFILE_RESOLVED_KEY = "_profile_resolved"


def resolve_profile(
    payload: dict[str, Any], profile: str | None = None
) -> tuple[dict[str, Any], str]:
    """Apply a named ``profiles`` overlay to a raw config document.

    Profiles are how the model mix is chosen without editing the config: the
    ``all_qwen`` overlay keeps every role on the local model, while the
    ``critic_*`` overlays move only 反思修复 onto Gemini or Claude. Overlays are
    deep-merged, so they only need to state what differs.

    Precedence is base < profile < CLI overrides, so this runs before overrides are
    merged and is a no-op on an already-resolved document.

    Args:
        payload: Raw config mapping (pre env-expansion).
        profile: Profile name. Falls back to ``$LVP_PROFILE``, then the document's
            ``annotation_profile``.

    Returns:
        ``(merged_document, resolved_profile_name)``. An empty name means no
        overlay was applied.

    Raises:
        ValueError: If a profile was requested but is not defined.
    """
    if payload.get(_PROFILE_RESOLVED_KEY):
        return payload, str(payload.get("annotation_profile") or "")

    profiles = payload.get("profiles")
    profiles = profiles if isinstance(profiles, dict) else {}
    # Profile selection runs before the document is env-expanded, so the name has to
    # be expanded on its own — `annotation_profile: ${LVP_PROFILE:-all_qwen}` is the
    # documented way to set the default.
    name = str(expand_env(
        profile or os.environ.get("LVP_PROFILE") or payload.get("annotation_profile") or ""
    )).strip()
    if not name:
        payload[_PROFILE_RESOLVED_KEY] = True
        return payload, ""

    overlay = profiles.get(name)
    if overlay is None:
        available = ", ".join(sorted(profiles)) or "(none defined)"
        raise ValueError(f"unknown profile {name!r}; available: {available}")
    if not isinstance(overlay, dict):
        raise ValueError(f"profiles.{name} must be a mapping")
    merged = deep_merge(payload, overlay)
    merged["annotation_profile"] = name
    merged[_PROFILE_RESOLVED_KEY] = True
    return merged, name


@dataclass
class LongVideoConfig:
    """Fully resolved configuration for one or many rollouts."""

    pipeline: PipelineSettings = field(default_factory=PipelineSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    tool_switches: ToolSwitches = field(default_factory=ToolSwitches)
    recording: RecordingSettings = field(default_factory=RecordingSettings)
    batch: BatchSettings = field(default_factory=BatchSettings)
    decoration: DecorationSettings = field(default_factory=DecorationSettings)
    models: dict[str, ModelEndpoint] = field(default_factory=dict)
    profile: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def endpoint_for(self, role: str) -> ModelEndpoint:
        """Return the endpoint for ``role``, falling back to ``default``."""
        if role in self.models:
            return self.models[role]
        if "default" in self.models:
            return self.models["default"]
        raise ValueError(f"no model endpoint configured for role {role!r} and no default")

    def describe_models(self) -> dict[str, Any]:
        """Secret-free per-role model provenance for manifests."""
        return {
            "profile": self.profile,
            "roles": {role: self.endpoint_for(role).describe() for role in ROLE_NAMES},
            "providers": {role: self.endpoint_for(role).provider for role in ROLE_NAMES},
            "mixed_providers": len({
                self.endpoint_for(role).provider for role in ROLE_NAMES
            }) > 1,
        }

    def validate(self) -> None:
        self.pipeline.resolved_root()
        for role in ROLE_NAMES:
            self.endpoint_for(role).validate(role)
        if self.agent.max_steps < 1:
            raise ValueError("agent.max_steps must be >= 1")

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, profile: str | None = None) -> "LongVideoConfig":
        merged, resolved_profile = resolve_profile(copy.deepcopy(payload or {}), profile)
        data = expand_env(merged)

        models_raw = data.get("models") or {}
        default_raw = models_raw.get("default") or {}
        default_endpoint = ModelEndpoint.from_dict(default_raw)
        models: dict[str, ModelEndpoint] = {"default": default_endpoint}
        for role in ROLE_NAMES:
            models[role] = default_endpoint.merged_with(models_raw.get(role) or {})

        return cls(
            pipeline=_build(PipelineSettings, data.get("pipeline")),
            agent=_build(AgentSettings, data.get("agent")),
            tool_switches=_build(ToolSwitches, data.get("tool_switches")),
            recording=_build(RecordingSettings, data.get("recording")),
            batch=_build(BatchSettings, data.get("batch")),
            decoration=_build(DecorationSettings, data.get("decoration")),
            models=models,
            profile=resolved_profile,
            raw=data,
        )

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        overrides: dict[str, Any] | None = None,
        profile: str | None = None,
    ) -> "LongVideoConfig":
        """Load the YAML config, apply ``overrides``, and resolve it.

        Args:
            path: Config file path. Defaults to the packaged reference config, or
                ``$LONGVIDEO_POLAR_CONFIG`` when set.
            overrides: Deep-merged on top of the file contents (CLI flags). Applied
                after the profile overlay, so an explicit flag always wins.
            profile: Model-mix profile name (e.g. ``all_qwen``, ``critic_gemini``).
        """
        config_path = Path(
            path or os.environ.get("LONGVIDEO_POLAR_CONFIG") or DEFAULT_CONFIG_PATH
        ).expanduser()
        if not config_path.is_file():
            raise FileNotFoundError(f"config not found: {config_path}")
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config root must be a mapping: {config_path}")
        loaded, resolved_profile = resolve_profile(loaded, profile)
        if overrides:
            loaded = deep_merge(loaded, overrides)
        config = cls.from_dict(loaded, profile=resolved_profile or None)
        config.raw["_config_path"] = str(config_path)
        return config


def _build(cls: type, payload: Any) -> Any:
    """Instantiate a settings dataclass from a mapping, ignoring unknown keys."""
    if not isinstance(payload, dict):
        return cls()
    fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{key: value for key, value in payload.items() if key in fields})
