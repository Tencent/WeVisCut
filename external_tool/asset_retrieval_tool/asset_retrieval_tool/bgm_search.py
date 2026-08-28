"""Free BGM search via Internet Archive netlabels (no API key required).

过滤机制（双重保障纯音乐）：
  1. Item 级：subject 含 spoken-word 关键词的整个 item 跳过
  2. File 级：文件名含 spoken-word 关键词的单个文件跳过
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

logger = logging.getLogger(__name__)

_TIMEOUT = 12
_IA_BASE = "https://archive.org"

# ── 纯音乐过滤关键词 ──────────────────────────────────────────────────────────

# Item 级：subject 含以下词则整个 item 跳过
_SPOKEN_WORD_SUBJECTS: frozenset[str] = frozenset({
    "poetry", "reading", "spoken word", "interview", "lecture", "podcast",
    "speech", "talk", "comedy", "radio", "audiobook", "dictionary",
    "conceptual writing", "cut-up", "news", "drama", "theatre", "theater",
    "field recording", "nature sounds", "sound art", "soundscape",
    "radio show", "broadcast", "narration", "storytelling",
})

# File 级：文件名（小写）含以下词则跳过该文件
_SPOKEN_WORD_FILENAME_TOKENS: frozenset[str] = frozenset({
    "interview", "podcast", "speech", "spoken", "reading", "lecture",
    "radio", "talk", "narration", "commentary", "intro_speech",
    "field_recording", "field recording", "nature", "ambient_noise",
})


def _is_music_item(subject: Any) -> bool:
    """Return True if the IA item's subject does NOT indicate spoken word."""
    if isinstance(subject, list):
        subj_str = " ".join(str(s) for s in subject).lower()
    else:
        subj_str = str(subject or "").lower()
    return not any(kw in subj_str for kw in _SPOKEN_WORD_SUBJECTS)


def _is_music_file(filename: str) -> bool:
    """Return True if the filename does NOT indicate spoken word content."""
    name_lower = filename.lower()
    return not any(kw in name_lower for kw in _SPOKEN_WORD_FILENAME_TOKENS)


# ── Internet Archive helpers ──────────────────────────────────────────────────

def _ia_query_docs(q: str, rows: int) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode({
        "q": q,
        "fl[]": "identifier,title,creator,subject,downloads",
        "rows": rows,
        "page": 1,
        "output": "json",
        "sort[]": "downloads desc",
    })
    url = f"{_IA_BASE}/advancedsearch.php?{params}"
    with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", {}).get("docs", [])


def _ia_fetch_tracks(identifier: str, limit: int = 5) -> list[dict[str, Any]]:
    """Fetch MP3 tracks from one IA item, applying file-level music filter."""
    try:
        url = f"{_IA_BASE}/metadata/{identifier}/files"
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        mp3s = [
            f for f in data.get("result", [])
            if str(f.get("name", "")).lower().endswith(".mp3")
            and f.get("source", "") != "derivative"
            and _is_music_file(str(f.get("name", "")))
        ]
        results = []
        for f in mp3s[:limit]:
            name = str(f.get("name", ""))
            duration_sec = _parse_duration(f.get("length"))
            download_url = f"{_IA_BASE}/download/{identifier}/{urllib.parse.quote(name)}"
            title = name.replace("_", " ").replace("-", " ").rsplit(".", 1)[0].strip()
            results.append({
                "source": "internet_archive",
                "id": f"{identifier}/{name}",
                "title": title,
                "artist": str(f.get("creator", identifier)),
                "duration_sec": duration_sec,
                "genre": "netlabel / CC",
                "url": f"{_IA_BASE}/details/{identifier}",
                "download_url": download_url,
                "license": "CC",
            })
        return results
    except Exception as exc:
        logger.debug("IA fetch tracks failed for %s: %s", identifier, exc)
        return []


def _ia_search(query: str, limit: int) -> list[dict[str, Any]]:
    """Search IA netlabels for CC music, with progressive query fallback.

    Item-level spoken-word filter applied before fetching file lists.
    """
    rows = min(max(limit, 8), 20)
    keywords = " ".join(query.split()[:2])
    queries = [
        f'collection:netlabels AND mediatype:audio AND ({query})',
        f'collection:netlabels AND mediatype:audio AND ({keywords})',
        f'collection:netlabels AND mediatype:audio',
    ]
    docs: list[dict[str, Any]] = []
    for q in queries:
        try:
            candidates = _ia_query_docs(q, rows)
            # Apply item-level music filter
            docs = [d for d in candidates if _is_music_item(d.get("subject", ""))]
        except Exception as exc:
            logger.debug("IA query failed (%s): %s", q[:60], exc)
            continue
        if docs:
            break

    results: list[dict[str, Any]] = []
    per_item = max(2, limit // max(1, len(docs))) if docs else 2
    for doc in docs:
        ident = str(doc.get("identifier", ""))
        if not ident:
            continue
        tracks = _ia_fetch_tracks(ident, limit=per_item)
        results.extend(tracks)
        if len(results) >= limit:
            break
    return results[:limit]


def _parse_duration(value: Any) -> float:
    """Parse 'mm:ss', 'ss.ms', or numeric seconds to float."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if ":" in s:
        parts = s.split(":")
        try:
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except Exception:
            pass
    try:
        return float(s)
    except Exception:
        return 0.0


# ── 公开接口 ──────────────────────────────────────────────────────────────────

def search_bgm(
    query: str,
    *,
    top_k: int = 10,
    min_duration: float = 0.0,
    max_duration: float = 600.0,
    sources: list[str] | None = None,
    max_workers: int = 4,
) -> list[dict[str, Any]]:
    """搜索免费 CC 授权纯音乐 BGM（Internet Archive netlabels）。

    过滤机制：
      - Item 级：subject 含 spoken-word/radio/field-recording 等关键词的 item 整体跳过
      - File 级：文件名含 interview/podcast/speech 等关键词的文件跳过
      - 时长过滤：min_duration / max_duration 参数控制

    Args:
        query: 风格或情绪描述，如 "upbeat corporate background" 或 "ambient chill"。
        top_k: 返回结果数量上限。
        min_duration: 最短时长（秒）。
        max_duration: 最长时长（秒）。
        sources: 保留参数，目前仅支持 ["internet_archive"]。
        max_workers: 并发请求线程数。

    Returns:
        list of dict，每项包含：rank, source, id, title, artist, duration_sec,
        genre, url, download_url, license。
    """
    enabled = [s.lower() for s in (sources or ["internet_archive"])]
    per_source = max(top_k * 3, 30)

    _source_fns: dict[str, Any] = {
        "internet_archive": lambda: _ia_search(query, per_source),
    }

    all_results: list[dict[str, Any]] = []
    fns = {name: fn for name, fn in _source_fns.items() if name in enabled}

    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(fns)))) as pool:
        futures = {pool.submit(fn): name for name, fn in fns.items()}
        for fut in as_completed(futures):
            try:
                all_results.extend(fut.result())
            except Exception as exc:
                logger.warning("BGM source %s error: %s", futures[fut], exc)

    filtered = [
        r for r in all_results
        if min_duration <= r.get("duration_sec", 0) <= max_duration
    ]

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for r in filtered:
        key = r.get("download_url") or r.get("url") or r.get("id")
        if key and key not in seen:
            seen.add(key)
            deduped.append(r)

    top = deduped[: max(1, top_k)]
    return [{"rank": i + 1, **item} for i, item in enumerate(top)]
