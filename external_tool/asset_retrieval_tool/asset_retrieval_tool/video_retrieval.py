"""Online video search via YouTube Data API v3 and Bilibili public API."""
from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .semantic import similarity_score

logger = logging.getLogger(__name__)

_TIMEOUT = 12
_YT_BASE = "https://www.googleapis.com/youtube/v3"
_BILI_BASE = "https://api.bilibili.com"


def _parse_iso8601_duration(s: str) -> float:
    """Parse YouTube ISO 8601 duration string to seconds."""
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s or "")
    if not m:
        return 0.0
    h, mn, sec = (int(x or 0) for x in m.groups())
    return h * 3600 + mn * 60 + sec


def _parse_bili_duration(s: Any) -> float:
    """Parse Bilibili duration (MM:SS string or numeric seconds) to float."""
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
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


def _yt_search(query: str, limit: int, api_key: str) -> list[dict[str, Any]]:
    """Search YouTube via Data API v3. Requires a valid API key."""
    search_params = urllib.parse.urlencode({
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": min(limit, 50),
        "key": api_key,
    })
    with urllib.request.urlopen(f"{_YT_BASE}/search?{search_params}", timeout=_TIMEOUT) as r:
        items = json.loads(r.read().decode()).get("items", [])

    video_ids = [it["id"]["videoId"] for it in items if it.get("id", {}).get("videoId")]
    if not video_ids:
        return []

    dur_params = urllib.parse.urlencode({
        "part": "contentDetails",
        "id": ",".join(video_ids),
        "key": api_key,
    })
    with urllib.request.urlopen(f"{_YT_BASE}/videos?{dur_params}", timeout=_TIMEOUT) as r:
        dur_items = json.loads(r.read().decode()).get("items", [])
    durations = {
        it["id"]: _parse_iso8601_duration(it["contentDetails"]["duration"])
        for it in dur_items
    }

    results = []
    for it in items:
        vid = it.get("id", {}).get("videoId", "")
        if not vid:
            continue
        sn = it.get("snippet", {})
        title = sn.get("title", "")
        desc = sn.get("description", "")
        dur = durations.get(vid, 0.0)
        results.append({
            "platform": "youtube",
            "video_id": vid,
            "source_video": f"https://www.youtube.com/watch?v={vid}",
            "title": title,
            "description": desc,
            "channel": sn.get("channelTitle", ""),
            "thumbnail": sn.get("thumbnails", {}).get("high", {}).get("url", ""),
            "start": 0.0,
            "end": dur,
            "duration": dur,
            "semantic_text": f"{title} {desc}",
        })
    return results


def _normalize_url(url: str) -> str:
    """B站接口返回的是协议相对 URL（//i0.hdslb.com/...），补上 https。

    协议相对 URL 在 file:// 页面下会被解析成 file://i0.hdslb.com/... 而失效，
    在源头补齐可以让所有下游消费方（HTML 报告、前端、下载器）都拿到可用地址。
    """
    url = str(url or "").strip()
    if url.startswith("//"):
        return f"https:{url}"
    return url


def _bili_search(query: str, limit: int, cookie: str = "") -> list[dict[str, Any]]:
    """Search Bilibili via public search API (no API key required)."""
    import gzip as _gzip
    params = urllib.parse.urlencode({
        "search_type": "video",
        "keyword": query,
        "page": 1,
        "page_size": min(limit, 50),
    })
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Referer": f"https://search.bilibili.com/all?keyword={urllib.parse.quote(query)}",
        "Origin": "https://www.bilibili.com",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    }
    if cookie:
        headers["Cookie"] = cookie
    req = urllib.request.Request(
        f"{_BILI_BASE}/x/web-interface/search/type?{params}",
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        raw = r.read()
    body = _gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    data = json.loads(body)

    results = []
    for it in data.get("data", {}).get("result", []):
        bvid = str(it.get("bvid", ""))
        aid = str(it.get("aid", ""))
        vid = bvid or aid
        if not vid:
            continue
        title = re.sub(r"<[^>]+>", "", str(it.get("title", "")))
        desc = str(it.get("description", ""))
        dur = _parse_bili_duration(it.get("duration", 0))
        url = (
            f"https://www.bilibili.com/video/{bvid}"
            if bvid
            else f"https://www.bilibili.com/video/av{aid}"
        )
        results.append({
            "platform": "bilibili",
            "video_id": vid,
            "source_video": url,
            "title": title,
            "description": desc,
            "channel": str(it.get("author", "")),
            "thumbnail": _normalize_url(it.get("pic", "")),
            "start": 0.0,
            "end": dur,
            "duration": dur,
            "semantic_text": f"{title} {desc}",
        })
    return results


def search_video_segments(
    query: str,
    *,
    top_k: int = 12,
    min_duration: float = 3.0,
    max_duration: float = 600.0,
    sources: list[str] | None = None,
    youtube_api_key: str | None = None,
    bilibili_cookie: str | None = None,
    max_workers: int = 4,
) -> list[dict[str, Any]]:
    """Search online video platforms for clips matching query.

    Args:
        query: Semantic description of desired video content.
        top_k: Max results to return.
        min_duration: Minimum video duration in seconds.
        max_duration: Maximum video duration in seconds.
        sources: Platforms to search. Defaults to ["youtube", "bilibili"] when
                 youtube_api_key is provided, else ["bilibili"].
        youtube_api_key: YouTube Data API v3 key. Required for YouTube search.
        bilibili_cookie: Optional Bilibili session cookie string to bypass 412.
        max_workers: Concurrent search threads.

    Returns:
        list of dicts with: rank, score, platform, video_id, source_video,
        title, description, channel, thumbnail, start, end, duration, semantic_text.
    """
    if sources is None:
        sources = ["youtube", "bilibili"] if youtube_api_key else ["bilibili"]

    enabled = [s.lower() for s in sources]
    per_source = max(top_k * 2, 20)

    fns: dict[str, Any] = {}
    if "youtube" in enabled:
        if not youtube_api_key:
            logger.warning("YouTube search requested but youtube_api_key not provided; skipping.")
        else:
            fns["youtube"] = lambda: _yt_search(query, per_source, youtube_api_key)
    if "bilibili" in enabled:
        cookie = bilibili_cookie or ""
        fns["bilibili"] = lambda: _bili_search(query, per_source, cookie)

    all_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(fns)))) as pool:
        futures = {pool.submit(fn): name for name, fn in fns.items()}
        for fut in as_completed(futures):
            try:
                all_results.extend(fut.result())
            except Exception as exc:
                logger.warning("Video source %s error: %s", futures[fut], exc)

    scored = []
    for item in all_results:
        dur = item.get("duration", 0.0)
        if not (min_duration <= dur <= max_duration):
            continue
        score = similarity_score(query, item.get("semantic_text", ""), dur)
        if score > 0:
            scored.append({**item, "score": round(score, 4)})

    scored.sort(key=lambda x: x["score"], reverse=True)

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for r in scored:
        key = r.get("source_video", "")
        if key and key not in seen:
            seen.add(key)
            deduped.append(r)

    top = deduped[: max(1, top_k)]
    return [{"rank": i + 1, **item} for i, item in enumerate(top)]
