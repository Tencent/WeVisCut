"""Asset provider registry: pluggable footage / BGM retrieval back ends.

WeVisCut needs two kinds of external assets: **video footage** for each shot
and **BGM tracks** for the finished cut. Everything else in the pipeline is
self-contained, so these two are the only places a deployment has to be wired
into whatever media source it happens to own.

A provider is just a function::

    from pipeline.providers import VideoQuery, register_video_provider

    @register_video_provider("my_footage")
    def search(query: VideoQuery) -> list[dict]:
        return [{"clip_path": "/data/a.mp4", "score": 0.9, "duration_sec": 12.0}]

Register it, point `services.retrieval_provider` (or `services.bgm_provider`)
at the same name in `studio_config.yaml`, and the pipeline will use it. Returned
dicts are passed through `normalize_video_clip` / `normalize_bgm_track`, so a
provider only has to supply the fields it actually knows — common aliases such
as ``url`` / ``path`` / ``title`` / ``duration`` are mapped automatically.

Providers living outside this repo are loaded from
`services.providers.plugins`, which accepts dotted module names
(``mypkg.providers``) and plain file paths (``/opt/weviscut/my_provider.py``).

See `docs/providers.md` for the full contract.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("aicut_pipeline")

__all__ = [
    "VideoQuery",
    "BgmQuery",
    "VideoProvider",
    "BgmProvider",
    "register_video_provider",
    "register_bgm_provider",
    "get_video_provider",
    "get_bgm_provider",
    "list_video_providers",
    "list_bgm_providers",
    "normalize_video_clip",
    "normalize_bgm_track",
    "provider_options",
    "load_provider_plugins",
    "run_video_provider",
    "run_bgm_provider",
    "BUILTIN_VIDEO_BACKENDS",
    "BUILTIN_BGM_BACKENDS",
]

# Backend names handled by the built-in branches inside
# `pipeline.tools.retrieve_visual` / `pipeline.tools.music_search`. Registry
# providers must not shadow these.
BUILTIN_VIDEO_BACKENDS = frozenset({
    "mmfinder_video_cli",
    "mmfinder_subprocess",
    "mmfinder_embedding_recall",
    "asset_retrieval_tool",
    "local_scan",
    "mmfinder",
})
BUILTIN_BGM_BACKENDS = frozenset({
    "mmfinder_music_cli",
    "mmfinder_music_svrkit",
    "asset_retrieval_tool",
    "local_scan",
    "mmfinder",
})


# ---------------------------------------------------------------------------
# query objects
# ---------------------------------------------------------------------------
@dataclass
class VideoQuery:
    """One footage search request for a single shot."""

    query: str
    top_k: int = 12
    #: TTS-derived length the clip has to cover. 0.0 means "unconstrained".
    target_duration_sec: float = 0.0
    #: Shortest acceptable source clip, already derived from `min_duration_ratio`.
    min_duration_sec: float = 0.0
    #: Longest useful source clip, derived from `max_loop_count`.
    max_duration_sec: float = 0.0
    #: Where a provider may materialise downloads.
    output_dir: str = "/tmp/weviscut_assets"
    #: The provider's own block from `services.providers.<name>`.
    options: dict[str, Any] = field(default_factory=dict)
    #: Full projected pipeline config, for providers that need more context.
    pipeline_config: dict[str, Any] = field(default_factory=dict)
    #: The shot dict that triggered this search (may be empty).
    shot: dict[str, Any] = field(default_factory=dict)


@dataclass
class BgmQuery:
    """One BGM search request for a whole cut."""

    query: str
    top_k: int = 10
    #: Total video duration the playlist should cover.
    target_duration_sec: float = 0.0
    output_dir: str = "/tmp/weviscut_assets"
    options: dict[str, Any] = field(default_factory=dict)
    pipeline_config: dict[str, Any] = field(default_factory=dict)


VideoProvider = Callable[[VideoQuery], "list[dict[str, Any]]"]
BgmProvider = Callable[[BgmQuery], "list[dict[str, Any]]"]


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
_VIDEO_PROVIDERS: dict[str, VideoProvider] = {}
_BGM_PROVIDERS: dict[str, BgmProvider] = {}
_LOADED_PLUGINS: set[str] = set()


def _register(
    table: dict[str, Any],
    kind: str,
    builtins: frozenset[str],
    name: str,
    fn: Any | None,
) -> Any:
    key = str(name or "").strip()
    if not key:
        raise ValueError(f"{kind} provider name must be non-empty")
    if key in builtins:
        raise ValueError(
            f"{kind} provider name {key!r} is reserved by a built-in backend; pick another name"
        )

    def _apply(func: Any) -> Any:
        if not callable(func):
            raise TypeError(f"{kind} provider {key!r} must be callable")
        if key in table and table[key] is not func:
            logger.warning("[providers] %s provider %r re-registered", kind, key)
        table[key] = func
        return func

    return _apply if fn is None else _apply(fn)


def register_video_provider(name: str, fn: VideoProvider | None = None) -> Any:
    """Register a footage provider. Usable as a decorator or a direct call."""
    return _register(_VIDEO_PROVIDERS, "video", BUILTIN_VIDEO_BACKENDS, name, fn)


def register_bgm_provider(name: str, fn: BgmProvider | None = None) -> Any:
    """Register a BGM provider. Usable as a decorator or a direct call."""
    return _register(_BGM_PROVIDERS, "bgm", BUILTIN_BGM_BACKENDS, name, fn)


def get_video_provider(name: str) -> VideoProvider | None:
    return _VIDEO_PROVIDERS.get(str(name or "").strip())


def get_bgm_provider(name: str) -> BgmProvider | None:
    return _BGM_PROVIDERS.get(str(name or "").strip())


def list_video_providers() -> list[str]:
    return sorted(_VIDEO_PROVIDERS)


def list_bgm_providers() -> list[str]:
    return sorted(_BGM_PROVIDERS)


def provider_options(pipeline_config: dict[str, Any] | None, name: str) -> dict[str, Any]:
    """Read `services.providers.<name>` out of the projected pipeline config."""
    providers = (pipeline_config or {}).get("providers")
    if not isinstance(providers, dict):
        return {}
    block = providers.get(str(name or "").strip())
    return dict(block) if isinstance(block, dict) else {}


# ---------------------------------------------------------------------------
# plugin loading
# ---------------------------------------------------------------------------
def load_provider_plugins(specs: Any) -> list[str]:
    """Import every plugin in `specs` so its providers register themselves.

    Each spec is either a dotted module path or a filesystem path to a `.py`
    file. Failures are logged and skipped: a broken third-party plugin must not
    take the whole service down.
    """
    if isinstance(specs, str):
        specs = [specs]
    if not isinstance(specs, (list, tuple)):
        return []

    loaded: list[str] = []
    for raw in specs:
        spec = str(raw or "").strip()
        if not spec or spec in _LOADED_PLUGINS:
            continue
        try:
            if spec.endswith(".py") or "/" in spec or "\\" in spec:
                _load_plugin_file(Path(spec).expanduser())
            else:
                importlib.import_module(spec)
        except Exception as exc:  # noqa: BLE001 - a bad plugin must not be fatal
            logger.warning("[providers] failed to load plugin %r: %s", spec, exc)
            continue
        _LOADED_PLUGINS.add(spec)
        loaded.append(spec)
        logger.info("[providers] loaded plugin %s", spec)
    return loaded


def _load_plugin_file(path: Path) -> None:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(str(resolved))
    mod_name = f"weviscut_plugin_{resolved.stem}_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(mod_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)


# ---------------------------------------------------------------------------
# result normalisation
# ---------------------------------------------------------------------------
_VIDEO_PATH_ALIASES = (
    "clip_path", "local_video_path", "local_path", "path", "file", "filepath",
    "video_path", "video_url", "url", "source_video", "src",
)
_VIDEO_TEXT_ALIASES = (
    "description", "caption", "caption_zh", "title", "text", "summary", "name",
)
_DURATION_ALIASES = ("duration_sec", "duration", "length_sec", "length", "seconds")
_BGM_PATH_ALIASES = (
    "music_path", "local_path", "audio_path", "path", "file", "filepath",
    "download_url", "audio_url", "url",
)


def _first_str(raw: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = raw.get(key)
        if value in (None, ""):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _first_float(raw: dict[str, Any], keys: tuple[str, ...], default: float = 0.0) -> float:
    for key in keys:
        if key not in raw:
            continue
        try:
            return float(raw[key])
        except (TypeError, ValueError):
            continue
    return default


def _as_tag_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def normalize_video_clip(
    raw: dict[str, Any],
    index: int = 0,
    id_prefix: str = "clip",
) -> dict[str, Any]:
    """Coerce a provider result into the clip-candidate schema the pipeline uses.

    Required downstream keys are always present. Any extra keys the provider
    supplied are preserved so nothing is lost on the way to the frontend.
    """
    if not isinstance(raw, dict):
        raise TypeError(f"video provider result #{index} is {type(raw).__name__}, expected dict")

    clip_path = _first_str(raw, _VIDEO_PATH_ALIASES)
    clip_id = _first_str(raw, ("clip_id", "id", "video_id", "asset_id"))
    if not clip_id:
        clip_id = f"{id_prefix}_{index:03d}"

    out = dict(raw)
    out.update({
        "clip_id": clip_id,
        "clip_path": clip_path,
        "score": _first_float(raw, ("score", "relevance", "similarity"), max(0.0, 1.0 - index * 0.02)),
        "duration_sec": _first_float(raw, _DURATION_ALIASES, 0.0),
        "description": _first_str(raw, _VIDEO_TEXT_ALIASES)[:300],
        "tags": _as_tag_list(raw.get("tags")),
    })
    return out


def normalize_bgm_track(
    raw: dict[str, Any],
    index: int = 0,
    id_prefix: str = "bgm",
) -> dict[str, Any]:
    """Coerce a provider result into the BGM-track schema the pipeline uses."""
    if not isinstance(raw, dict):
        raise TypeError(f"bgm provider result #{index} is {type(raw).__name__}, expected dict")

    music_path = _first_str(raw, _BGM_PATH_ALIASES)
    track_id = _first_str(raw, ("track_id", "id", "music_id", "asset_id"))
    if not track_id:
        track_id = f"{id_prefix}_{index:03d}"

    out = dict(raw)
    out.update({
        "track_id": track_id,
        "music_path": music_path,
        "score": _first_float(raw, ("score", "relevance", "similarity"), max(0.0, 1.0 - index * 0.05)),
        "duration_sec": _first_float(raw, _DURATION_ALIASES, 0.0),
        "name": _first_str(raw, ("name", "title", "track", "filename")) or track_id,
        "caption": _first_str(raw, ("caption", "artist", "description", "album")),
        "cdn_path": _first_str(raw, ("cdn_path", "download_url", "url")),
        "rank": int(_first_float(raw, ("rank",), float(index + 1))),
    })
    return out


# ---------------------------------------------------------------------------
# invocation
# ---------------------------------------------------------------------------
def run_video_provider(name: str, query: VideoQuery) -> list[dict[str, Any]]:
    """Call a registered footage provider and normalise its results.

    Returns `[]` when the provider is unknown, raises, or yields nothing —
    callers treat that as "this backend found no candidates" and move on.
    """
    provider = get_video_provider(name)
    if provider is None:
        logger.warning(
            "[providers] unknown video provider %r (registered: %s)",
            name, ", ".join(list_video_providers()) or "none",
        )
        return []
    try:
        results = provider(query) or []
    except Exception as exc:  # noqa: BLE001 - provider errors are data, not crashes
        logger.warning("[providers] video provider %r failed: %s", name, exc)
        return []
    return _normalize_many(results, name, normalize_video_clip, id_prefix=name)


def run_bgm_provider(name: str, query: BgmQuery) -> list[dict[str, Any]]:
    """Call a registered BGM provider and normalise its results."""
    provider = get_bgm_provider(name)
    if provider is None:
        logger.warning(
            "[providers] unknown bgm provider %r (registered: %s)",
            name, ", ".join(list_bgm_providers()) or "none",
        )
        return []
    try:
        results = provider(query) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("[providers] bgm provider %r failed: %s", name, exc)
        return []
    return _normalize_many(results, name, normalize_bgm_track, id_prefix=name)


def _normalize_many(
    results: Any,
    name: str,
    normalizer: Callable[..., dict[str, Any]],
    id_prefix: str,
) -> list[dict[str, Any]]:
    if isinstance(results, dict):
        results = results.get("results") or results.get("items") or []
    if not isinstance(results, (list, tuple)):
        logger.warning("[providers] provider %r returned %s, expected list", name, type(results).__name__)
        return []
    out: list[dict[str, Any]] = []
    for i, item in enumerate(results):
        try:
            out.append(normalizer(item, i, id_prefix=id_prefix))
        except (TypeError, ValueError) as exc:
            logger.warning("[providers] provider %r result #%d dropped: %s", name, i, exc)
    return out
