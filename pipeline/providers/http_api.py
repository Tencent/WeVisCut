"""Config-only HTTP providers: `http_video` and `http_bgm`.

These let a deployment plug its own search service into WevisionCut without
writing any Python. Point them at an endpoint that takes a text query and
returns JSON, and describe the response shape in `studio_config.yaml`::

    services:
      retrieval_provider: http_video
      bgm_provider: http_bgm
      providers:
        http_video:
          url: https://footage.example.com/search
          method: POST
          headers:
            Authorization: "Bearer ${FOOTAGE_TOKEN}"
          query_field: query
          top_k_field: top_k
          results_path: data.items
          field_map:
            clip_path: playback_url
            description: title
            duration_sec: seconds
        http_bgm:
          url: https://music.example.com/search
          results_path: tracks

Only the standard library is used, so no extra dependency is introduced.
Everything not covered by `field_map` still goes through the alias matching in
`normalize_video_clip` / `normalize_bgm_track`, so a response that already uses
names like `url` / `title` / `duration` needs no mapping at all.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .registry import (
    BgmQuery,
    VideoQuery,
    register_bgm_provider,
    register_video_provider,
)

logger = logging.getLogger("aicut_pipeline")

_DEFAULT_TIMEOUT_SEC = 30.0


def _expand(value: Any) -> Any:
    """Expand `${VAR}` in strings so tokens can stay in the environment."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _dig(payload: Any, path: str) -> Any:
    """Walk a dotted path into a decoded JSON payload."""
    node = payload
    for part in str(path or "").split("."):
        key = part.strip()
        if not key:
            continue
        if isinstance(node, dict):
            node = node.get(key)
        elif isinstance(node, list) and key.isdigit():
            idx = int(key)
            node = node[idx] if 0 <= idx < len(node) else None
        else:
            return None
    return node


def _extract_results(payload: Any, results_path: str) -> list[dict[str, Any]]:
    node = _dig(payload, results_path) if results_path else payload
    if node is None and not results_path:
        return []
    if isinstance(node, dict):
        for key in ("results", "items", "data", "candidates", "list"):
            if isinstance(node.get(key), list):
                node = node[key]
                break
    if not isinstance(node, list):
        return []
    return [item for item in node if isinstance(item, dict)]


def _apply_field_map(item: dict[str, Any], field_map: dict[str, Any]) -> dict[str, Any]:
    """Copy `item[source]` to `item[canonical]` for each canonical: source pair."""
    if not field_map:
        return item
    out = dict(item)
    for canonical, source in field_map.items():
        value = _dig(item, str(source)) if "." in str(source) else item.get(str(source))
        if value not in (None, ""):
            out[str(canonical)] = value
    return out


def _request(options: dict[str, Any], payload: dict[str, Any]) -> Any:
    url = str(options.get("url", "") or "").strip()
    if not url:
        raise ValueError("providers.<name>.url is not set")

    method = str(options.get("method", "POST") or "POST").upper()
    timeout = float(options.get("timeout_sec", _DEFAULT_TIMEOUT_SEC) or _DEFAULT_TIMEOUT_SEC)
    headers = {str(k): str(v) for k, v in _expand(options.get("headers") or {}).items()}
    extra = _expand(options.get("extra_params") or {})
    if isinstance(extra, dict):
        payload = {**payload, **extra}

    if method == "GET":
        sep = "&" if urllib.parse.urlparse(url).query else "?"
        full = f"{url}{sep}{urllib.parse.urlencode(payload)}"
        request = urllib.request.Request(full, headers=headers, method="GET")
    else:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")
        request = urllib.request.Request(url, data=body, headers=headers, method=method)

    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        charset = response.headers.get_content_charset() or "utf-8"
        return json.loads(response.read().decode(charset, errors="replace"))


def _search(options: dict[str, Any], payload: dict[str, Any], label: str) -> list[dict[str, Any]]:
    try:
        decoded = _request(options, payload)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        logger.warning("[providers] %s request failed: %s", label, exc)
        return []
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("[providers] %s returned unusable payload: %s", label, exc)
        return []

    items = _extract_results(decoded, str(options.get("results_path", "") or ""))
    field_map = options.get("field_map") or {}
    if not isinstance(field_map, dict):
        field_map = {}
    return [_apply_field_map(item, field_map) for item in items]


@register_video_provider("http_video")
def http_video_provider(query: VideoQuery) -> list[dict[str, Any]]:
    """Footage search against a user-supplied HTTP JSON endpoint."""
    options = query.options
    payload: dict[str, Any] = {
        str(options.get("query_field", "query") or "query"): query.query,
        str(options.get("top_k_field", "top_k") or "top_k"): query.top_k,
    }
    if query.target_duration_sec > 0:
        payload[str(options.get("duration_field", "target_duration_sec") or "target_duration_sec")] = round(
            query.target_duration_sec, 3
        )
    if query.min_duration_sec > 0:
        payload.setdefault("min_duration_sec", round(query.min_duration_sec, 3))
    if query.max_duration_sec > 0:
        payload.setdefault("max_duration_sec", round(query.max_duration_sec, 3))
    return _search(options, payload, "http_video")


@register_bgm_provider("http_bgm")
def http_bgm_provider(query: BgmQuery) -> list[dict[str, Any]]:
    """BGM search against a user-supplied HTTP JSON endpoint."""
    options = query.options
    payload: dict[str, Any] = {
        str(options.get("query_field", "query") or "query"): query.query,
        str(options.get("top_k_field", "top_k") or "top_k"): query.top_k,
    }
    if query.target_duration_sec > 0:
        payload[str(options.get("duration_field", "target_duration_sec") or "target_duration_sec")] = round(
            query.target_duration_sec, 3
        )
    return _search(options, payload, "http_bgm")
