"""Offline providers backed by folders on disk: `local_folder_video` / `local_folder_bgm`.

The fastest way to run WevisionCut end to end without any external service: drop
your own footage and music into a directory tree and describe it in
`studio_config.yaml`::

    services:
      retrieval_provider: local_folder_video
      bgm_provider: local_folder_bgm
      providers:
        local_folder_video:
          roots:
            - /data/my_footage
        local_folder_bgm:
          roots:
            - /data/my_music

Matching is lexical: the query is tokenised and scored against the file name,
its parent directory names, and an optional sidecar caption. A sidecar is a
file next to the media with the same stem and a `.txt` / `.json` suffix; JSON
sidecars may carry `caption` / `description` / `tags` / `duration_sec`.

This module is also the shortest complete example of the provider contract —
copy it as the starting point for a real integration.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from .registry import (
    BgmQuery,
    VideoQuery,
    register_bgm_provider,
    register_video_provider,
)

logger = logging.getLogger("aicut_pipeline")

_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
_TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")
_MAX_SCAN_FILES = 20000


def _tokenize(text: str) -> list[str]:
    """Split a query into lexical tokens; CJK runs also yield 2-grams."""
    tokens: list[str] = []
    for chunk in _TOKEN_RE.findall(str(text or "").lower()):
        tokens.append(chunk)
        if len(chunk) > 1 and any("\u4e00" <= ch <= "\u9fff" for ch in chunk):
            tokens.extend(chunk[i:i + 2] for i in range(len(chunk) - 1))
    return list(dict.fromkeys(t for t in tokens if len(t) >= 2 or not t.isascii()))


def _roots(options: dict[str, Any]) -> list[Path]:
    raw = options.get("roots") or options.get("root") or []
    if isinstance(raw, str):
        raw = [raw]
    out: list[Path] = []
    for entry in raw:
        path = Path(str(entry).strip()).expanduser()
        if str(path) and path not in out:
            out.append(path)
    return out


def _sidecar(media: Path) -> tuple[str, list[str], float]:
    """Return (caption, tags, duration_sec) from a sidecar file, if present."""
    json_path = media.with_suffix(".json")
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
        if isinstance(data, dict):
            caption = str(
                data.get("caption") or data.get("description") or data.get("title") or ""
            ).strip()
            tags_raw = data.get("tags") or []
            tags = (
                [str(t).strip() for t in tags_raw if str(t).strip()]
                if isinstance(tags_raw, (list, tuple))
                else []
            )
            try:
                duration = float(data.get("duration_sec") or data.get("duration") or 0.0)
            except (TypeError, ValueError):
                duration = 0.0
            return caption, tags, duration

    txt_path = media.with_suffix(".txt")
    if txt_path.exists():
        try:
            return txt_path.read_text(encoding="utf-8").strip(), [], 0.0
        except OSError:
            pass
    return "", [], 0.0


def _iter_media(roots: list[Path], exts: set[str]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.is_dir():
            logger.warning("[providers] local_folder root is not a directory: %s", root)
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() in exts and path.is_file():
                files.append(path)
                if len(files) >= _MAX_SCAN_FILES:
                    logger.warning(
                        "[providers] local_folder stopped at %d files; narrow `roots`",
                        _MAX_SCAN_FILES,
                    )
                    return files
    return files


def _score(tokens: list[str], haystack: str) -> float:
    if not tokens:
        return 0.5
    hits = sum(1 for token in tokens if token in haystack)
    return round(hits / len(tokens), 4)


def _rank(
    media_files: list[Path],
    query_text: str,
    top_k: int,
    require_match: bool,
) -> list[tuple[Path, float, str, list[str], float]]:
    tokens = _tokenize(query_text)
    scored: list[tuple[Path, float, str, list[str], float]] = []
    for path in media_files:
        caption, tags, duration = _sidecar(path)
        haystack = " ".join([
            path.stem.lower().replace("_", " ").replace("-", " "),
            " ".join(part.lower() for part in path.parent.parts[-2:]),
            caption.lower(),
            " ".join(tags).lower(),
        ])
        score = _score(tokens, haystack)
        if require_match and score <= 0.0:
            continue
        scored.append((path, score, caption, tags, duration))
    scored.sort(key=lambda row: (-row[1], row[0].name))
    return scored[:max(1, top_k)]


@register_video_provider("local_folder_video")
def local_folder_video_provider(query: VideoQuery) -> list[dict[str, Any]]:
    """Footage search over local directories."""
    options = query.options
    roots = _roots(options)
    if not roots:
        logger.warning("[providers] local_folder_video: providers.local_folder_video.roots is empty")
        return []

    require_match = bool(options.get("require_match", False))
    exts = {str(e).lower() for e in options.get("extensions", [])} or _VIDEO_EXTS
    ranked = _rank(_iter_media(roots, exts), query.query, query.top_k, require_match)

    return [
        {
            "clip_id": f"local_{index:03d}_{path.stem[:40]}",
            "clip_path": str(path),
            "score": max(score, 0.01),
            "duration_sec": duration,
            "description": caption or path.stem.replace("_", " "),
            "tags": tags,
        }
        for index, (path, score, caption, tags, duration) in enumerate(ranked)
    ]


@register_bgm_provider("local_folder_bgm")
def local_folder_bgm_provider(query: BgmQuery) -> list[dict[str, Any]]:
    """BGM search over local directories."""
    options = query.options
    roots = _roots(options)
    if not roots:
        logger.warning("[providers] local_folder_bgm: providers.local_folder_bgm.roots is empty")
        return []

    exts = {str(e).lower() for e in options.get("extensions", [])} or _AUDIO_EXTS
    ranked = _rank(_iter_media(roots, exts), query.query, query.top_k, require_match=False)

    return [
        {
            "track_id": f"local_bgm_{index:03d}_{path.stem[:40]}",
            "music_path": str(path),
            "score": max(score, 0.01),
            "duration_sec": duration,
            "name": path.stem,
            "caption": caption,
            "rank": index + 1,
        }
        for index, (path, score, caption, _tags, duration) in enumerate(ranked)
    ]
