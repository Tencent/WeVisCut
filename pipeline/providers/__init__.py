"""Pluggable footage / BGM providers.

Importing this package registers the bundled providers (`http_video`,
`http_bgm`, `local_folder_video`, `local_folder_bgm`). Third-party providers
are loaded on demand from `services.providers.plugins`.

See `docs/providers.md` for the contract.
"""

from __future__ import annotations

from .registry import (
    BUILTIN_BGM_BACKENDS,
    BUILTIN_VIDEO_BACKENDS,
    BgmProvider,
    BgmQuery,
    VideoProvider,
    VideoQuery,
    get_bgm_provider,
    get_video_provider,
    list_bgm_providers,
    list_video_providers,
    load_provider_plugins,
    normalize_bgm_track,
    normalize_video_clip,
    provider_options,
    register_bgm_provider,
    register_video_provider,
    run_bgm_provider,
    run_video_provider,
)

# Bundled providers self-register on import.
from . import http_api as _http_api  # noqa: F401,E402
from . import local_folder as _local_folder  # noqa: F401,E402

__all__ = [
    "BUILTIN_BGM_BACKENDS",
    "BUILTIN_VIDEO_BACKENDS",
    "BgmProvider",
    "BgmQuery",
    "VideoProvider",
    "VideoQuery",
    "get_bgm_provider",
    "get_video_provider",
    "list_bgm_providers",
    "list_video_providers",
    "load_provider_plugins",
    "normalize_bgm_track",
    "normalize_video_clip",
    "provider_options",
    "register_bgm_provider",
    "register_video_provider",
    "run_bgm_provider",
    "run_video_provider",
]
