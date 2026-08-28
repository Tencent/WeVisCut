"""Tool_RetrieveVisual: search and select the best-matching clip for a shot.

Retrieval is pluggable. Which backend runs is decided by
`services.retrieval_provider` in `studio_config.yaml`:

  * ``asset_retrieval_tool`` — bundled open-web search (YouTube / Bilibili).
  * ``mmfinder``             — an internal embedding-recall service, reached
                               through the subprocess CLIs under
                               ``external_tool/mmfinder``. Not publicly
                               reachable; kept for the original deployment.
  * anything registered in ``pipeline.providers`` — your own source, either a
    config-only HTTP endpoint or a Python plugin. See ``docs/providers.md``.

Whatever the backend, candidates are filtered against the shot's target
duration and orientation, then one is selected (by the policy LLM or by score).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from ..llm import LLMClient
from ..providers import (
    VideoQuery,
    get_video_provider,
    list_video_providers,
    load_provider_plugins,
    provider_options,
    run_video_provider,
)

logger = logging.getLogger("aicut_pipeline")

# The mmfinder client lives outside the repo (it is large and internal-only).
# Install it under external_tool/mmfinder/ or point MMFINDER_ROOT elsewhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MMFINDER = _REPO_ROOT / "external_tool" / "mmfinder" / "WeAgent_WeStoryCut"
MMFINDER_ROOT = Path(os.environ.get(
    "MMFINDER_ROOT",
    str(_DEFAULT_MMFINDER),
))
# Some storage mounts ship their own ffmpeg-compatible libs here; prepended to
# LD_LIBRARY_PATH when present, ignored otherwise.
FFMPEG_COMPAT_LIB_DIR = Path(
    os.environ.get("AICUT_FFMPEG_COMPAT_LIB_DIR", "/usr/lib64/dop-fuse")
)
_DIAGNOSTICS_LOCAL = threading.local()
_PROXY_KEYS = {
    "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
    "all_proxy", "ALL_PROXY",
}

# Bundled open-web retrieval package (external_tool/asset_retrieval_tool/).
_ASSET_RETRIEVAL_PACKAGE = _REPO_ROOT / "external_tool" / "asset_retrieval_tool"


def _strip_proxy(env: dict[str, str]) -> dict[str, str]:
    """Remove proxy env vars so subprocesses reach their service directly."""
    for key in _PROXY_KEYS:
        env.pop(key, None)
    return env


def _python_bin() -> str:
    return os.environ.get("AICUT_MMFINDER_PYTHON", "").strip() or sys.executable


def _media_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    _strip_proxy(env)
    if FFMPEG_COMPAT_LIB_DIR.exists():
        current = env.get("LD_LIBRARY_PATH", "")
        parts = [part for part in current.split(":") if part]
        compat = str(FFMPEG_COMPAT_LIB_DIR)
        if compat not in parts:
            env["LD_LIBRARY_PATH"] = ":".join([compat, *parts])
    return env


def _reset_diagnostics() -> None:
    _DIAGNOSTICS_LOCAL.items = []


def _diag(backend: str, status: str, **extra: Any) -> None:
    items = getattr(_DIAGNOSTICS_LOCAL, "items", None)
    if items is None:
        items = []
        _DIAGNOSTICS_LOCAL.items = items
    items.append({"backend": backend, "status": status, **extra})


def get_last_diagnostics() -> list[dict[str, Any]]:
    return list(getattr(_DIAGNOSTICS_LOCAL, "items", []))

SYSTEM_PROMPT = """\
你是视频素材检索专家。根据视觉检索关键词在素材库中找到最匹配的视频片段。"""

SELECT_TEMPLATE = """\
根据以下镜头的视觉需求，从候选素材中选择最佳匹配:

镜头信息:
- shot_id: {shot_id}
- visual_query: {visual_query}
- visual_direction: {visual_direction}
- vocal_script: {vocal_script}
- target_duration_sec: {duration_sec}
- minimum_source_duration_sec: {minimum_duration_sec}

候选素材:
{candidates_json}

硬性要求:
1. 只能从候选素材中选择，禁止返回候选列表外的 clip_id
2. 候选素材已通过当前配置的时长/循环次数限制；优先保证视觉语义匹配，
   在语义接近时再选择 duration_sec 更长、loop_count 更小的素材
3. 候选素材已通过当前配置的横屏与宽高比限制，后续会统一归一化到 1920x1080
4. 不得编造候选中不存在的画面内容

请仔细评估每个候选素材与镜头需求的匹配度，输出选择结果(严格JSON格式):
{{
  "selected_clip_id": "clip_xxx",
  "clip_path": "素材路径",
  "match_score": 0.0-1.0,
  "selection_rationale": "选择理由(具体说明为何该素材最匹配)"
}}"""


def _search_via_subprocess(
    visual_query: str,
    save_dir: str,
    num: int = 5,
    target_duration: float = 8.0,
    timeout_sec: int | None = None,
) -> list[dict[str, Any]]:
    """Call GetFinderMaterials.search_and_prepare_videos via subprocess."""
    if timeout_sec is None:
        timeout_sec = int(os.environ.get("AICUT_FINDER_SEARCH_TIMEOUT", "120"))
    script_path = MMFINDER_ROOT / "scripts" / "get-finder-search" / "GetFinderMaterials.py"
    if not script_path.exists():
        _diag("mmfinder_subprocess", "missing_script", script_path=str(script_path))
        return []

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        _python_bin(),
        str(script_path),
        "--query", visual_query,
        "--num", str(num),
        "--save-dir", save_dir,
        "--target-duration-sec", str(target_duration),
    ]

    env = _strip_proxy(os.environ.copy())
    src_root = str(MMFINDER_ROOT / "src")
    pb_root = str(MMFINDER_ROOT / "src" / "pb")
    script_root = str(script_path.parent)
    prep_root = str(MMFINDER_ROOT / "scripts" / "video-material-prep")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(
        [src_root, pb_root, script_root, prep_root] +
        ([existing] if existing else [])
    )

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_sec, env=env,
        )
        result_json = Path(save_dir) / "search_result.json"
        if result_json.exists():
            data = json.loads(result_json.read_text(encoding="utf-8"))
            candidates = _normalize_finder_results(data)
            if candidates:
                _diag("mmfinder_subprocess", "success", count=len(candidates), returncode=proc.returncode)
                return candidates

        for line in reversed(proc.stdout.strip().split("\n")):
            try:
                data = json.loads(line)
                if isinstance(data, dict) and "videos" in data:
                    candidates = _normalize_finder_results(data)
                    if candidates:
                        _diag("mmfinder_subprocess", "success", count=len(candidates), returncode=proc.returncode)
                        return candidates
            except json.JSONDecodeError:
                continue
        if proc.returncode != 0:
            _diag("mmfinder_subprocess", "error", returncode=proc.returncode, stderr=proc.stderr[-500:])
            return _parse_fallback_error(proc.stderr, visual_query)
    except subprocess.TimeoutExpired:
        _diag("mmfinder_subprocess", "timeout", timeout_sec=timeout_sec)
        pass
    except FileNotFoundError:
        _diag("mmfinder_subprocess", "python_not_found", python=_python_bin())
        pass
    return []


def _recall_videos_via_cli(query: str, size: int = 100, timeout_sec: int | None = None) -> list[dict[str, Any]]:
    """Call mmfinder text→video recall CLI without the expensive download/crop step."""
    if timeout_sec is None:
        timeout_sec = int(os.environ.get("AICUT_MMFINDER_CLI_TIMEOUT", "60"))
    script_dir = MMFINDER_ROOT / "scripts" / "get-video-music-recall"
    script_path = script_dir / "GetVideoMusicRecall.py"
    if not script_path.exists():
        _diag("mmfinder_video_cli", "missing_script", script_path=str(script_path))
        return []

    src_root = str(MMFINDER_ROOT / "src")
    pb_root = str(MMFINDER_ROOT / "src" / "pb")
    env = _strip_proxy(os.environ.copy())
    env["PYTHONPATH"] = ":".join([
        src_root, pb_root, str(script_dir), env.get("PYTHONPATH", ""),
    ])
    cmd = [
        _python_bin(),
        str(script_path),
        "--text", query,
        "--mode", "video",
        "--size", str(size),
        "--embed-timeout", str(int(os.environ.get("AICUT_MMFINDER_EMBED_TIMEOUT", "45"))),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, env=env)
        if proc.returncode != 0 or not proc.stdout.strip():
            _diag("mmfinder_video_cli", "error", returncode=proc.returncode, stderr=proc.stderr[-500:])
            return []
        data = _parse_json_from_output(proc.stdout)
        raw_items = data.get("data_list", []) if isinstance(data, dict) else []
        candidates = [_build_recall_clip(raw, i + 1) for i, raw in enumerate(raw_items[:size]) if isinstance(raw, dict)]
        _diag("mmfinder_video_cli", "success" if candidates else "empty", count=len(candidates))
        return candidates
    except (subprocess.TimeoutExpired, FileNotFoundError):
        _diag("mmfinder_video_cli", "timeout_or_not_found", timeout_sec=timeout_sec, python=_python_bin())
        return []


def _search_via_embedding(
    visual_query: str,
    base_url: str = "",
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Direct embedding service → recall pipeline."""
    embedding_client_path = MMFINDER_ROOT / "src" / "pipeline" / "tools" / "_internal" / "embedding_client.py"
    if not embedding_client_path.exists():
        _diag("mmfinder_embedding", "missing_script", script_path=str(embedding_client_path))
        return []

    script = (
        "import json, sys\n"
        "sys.path.insert(0, '{src_root}')\n"
        "sys.path.insert(0, '{pb_root}')\n"
        "from pipeline.tools._internal.embedding_client import embed_inputs\n"
        "query = sys.argv[1]\n"
        "base_url = sys.argv[2] or None\n"
        "embeddings = embed_inputs([{{'text': query}}], base_url=base_url)\n"
        "print(json.dumps({{'embedding': embeddings[0], 'dim': len(embeddings[0])}}))\n"
    ).format(
        src_root=str(MMFINDER_ROOT / "src"),
        pb_root=str(MMFINDER_ROOT / "src" / "pb"),
    )

    env = _strip_proxy(os.environ.copy())
    env["PYTHONPATH"] = f"{MMFINDER_ROOT / 'src'}:{MMFINDER_ROOT / 'src' / 'pb'}:{env.get('PYTHONPATH', '')}"

    try:
        proc = subprocess.run(
            [_python_bin(), "-c", script, visual_query, base_url or ""],
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("AICUT_MMFINDER_EMBED_TIMEOUT", "45")),
            env=env,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            data = json.loads(proc.stdout.strip().split("\n")[-1])
            embedding = data.get("embedding", [])
            if embedding:
                _diag("mmfinder_embedding", "success", dim=len(embedding))
                return _recall_with_embedding(embedding, top_k, env)
        _diag("mmfinder_embedding", "error", returncode=proc.returncode, stderr=proc.stderr[-500:])
    except subprocess.TimeoutExpired:
        _diag("mmfinder_embedding", "timeout")
    except (json.JSONDecodeError, FileNotFoundError) as exc:
        _diag("mmfinder_embedding", "error", error=str(exc))
        pass
    return []


def _recall_with_embedding(
    embedding: list[float],
    top_k: int,
    env: dict[str, str],
) -> list[dict[str, Any]]:
    """Call recall_client.recall_videos with a pre-computed embedding."""
    recall_script = (
        "import asyncio, json, sys\n"
        "sys.path.insert(0, '{src_root}')\n"
        "sys.path.insert(0, '{pb_root}')\n"
        "from client.mmfindertext2videoagentbroker.recall_client import recall_videos\n"
        "embedding = json.loads(sys.argv[1])\n"
        "size = int(sys.argv[2])\n"
        "results = asyncio.run(recall_videos(embedding, size=size))\n"
        "print(json.dumps(results, ensure_ascii=False))\n"
    ).format(
        src_root=str(MMFINDER_ROOT / "src"),
        pb_root=str(MMFINDER_ROOT / "src" / "pb"),
    )

    try:
        proc = subprocess.run(
            [_python_bin(), "-c", recall_script, json.dumps(embedding), str(top_k)],
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("AICUT_MMFINDER_RECALL_TIMEOUT", "30")),
            env=env,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            results = json.loads(proc.stdout.strip().split("\n")[-1])
            if isinstance(results, list):
                candidates = [
                    {
                        "clip_id": str(r.get("id_str", r.get("id", f"recall_{i}"))),
                        "clip_path": str(r.get("src_path", r.get("cdn_path", ""))),
                        "score": float(r.get("score", 0.0)),
                        "duration_sec": 0.0,
                        "description": str(r.get("ext_info", ""))[:200],
                        "tags": [],
                    }
                    for i, r in enumerate(results[:top_k])
                ]
                _diag("mmfinder_embedding_recall", "success" if candidates else "empty", count=len(candidates))
                return candidates
        _diag("mmfinder_embedding_recall", "error", returncode=proc.returncode, stderr=proc.stderr[-500:])
    except subprocess.TimeoutExpired:
        _diag("mmfinder_embedding_recall", "timeout")
    except (json.JSONDecodeError, FileNotFoundError) as exc:
        _diag("mmfinder_embedding_recall", "error", error=str(exc))
        pass
    return []


def _parse_json_from_output(output: str) -> Any:
    """Parse the last JSON object/array from noisy subprocess output."""
    text = output.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return {}


def _build_recall_clip(raw: dict[str, Any], rank: int) -> dict[str, Any]:
    """Normalize mmfinder recall output into the AICut clip candidate schema."""
    ext_info: dict[str, Any] = {}
    if raw.get("ext_info"):
        try:
            parsed = json.loads(raw["ext_info"]) if isinstance(raw["ext_info"], str) else raw["ext_info"]
            if isinstance(parsed, dict):
                ext_info = parsed
        except (json.JSONDecodeError, TypeError):
            ext_info = {}
    anno = ext_info.get("anno") if isinstance(ext_info.get("anno"), dict) else {}
    clip_path = (
        str(ext_info.get("video_clip_path") or "")
        or str(raw.get("src_path") or "")
        or str(raw.get("cdn_path") or "")
    )
    duration = 0.0
    try:
        duration = float(anno.get("duration") or 0)
    except (TypeError, ValueError):
        pass
    return {
        "clip_id": str(raw.get("id_str", raw.get("id", f"recall_{rank:03d}"))),
        "clip_path": clip_path,
        "score": float(raw.get("score", 0.0) or 0.0),
        "duration_sec": duration,
        "description": str(ext_info.get("title") or ext_info.get("description") or raw.get("ext_info") or "")[:300],
        "tags": [],
        "cdn_path": str(raw.get("cdn_path", "")),
        "rank": rank,
    }


def _normalize_finder_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize GetFinderMaterials output to standard candidate format."""
    videos = data.get("videos", [])
    candidates = []
    for i, v in enumerate(videos):
        candidates.append({
            "clip_id": v.get("id", f"finder_{i:03d}"),
            "clip_path": v.get("clean_path", v.get("raw_path", v.get("path", ""))),
            "score": float(v.get("score", 1.0 - i * 0.05)),
            "duration_sec": float(v.get("duration", v.get("duration_sec", 0))),
            "description": v.get("title", v.get("description", "")),
            "tags": v.get("tags", []),
        })
    return candidates


def _parse_fallback_error(stderr: str, query: str) -> list[dict[str, Any]]:
    """When subprocess fails, return empty list (upstream should handle gracefully)."""
    return []


def _search_via_asset_retrieval_tool(
    visual_query: str,
    top_k: int = 12,
    asset_retrieval_cfg: dict[str, Any] | None = None,
    output_dir: str = "/tmp/aicut_retrieval",
) -> list[dict[str, Any]]:
    """Search video clips via asset_retrieval_tool (YouTube/Bilibili online search).

    Integrates the external_tool/asset_retrieval_tool package which performs
    online semantic video search without requiring internal mmfinder services.
    """
    cfg = asset_retrieval_cfg or {}
    sources = cfg.get("video_sources", ["bilibili"])
    if not isinstance(sources, list):
        sources = ["bilibili"]
    youtube_api_key = (
        os.environ.get("YOUTUBE_API_KEY", "").strip()
        or str(cfg.get("youtube_api_key", "") or "").strip()
    )
    bilibili_cookie = (
        os.environ.get("BILIBILI_COOKIE", "").strip()
        or str(cfg.get("bilibili_cookie", "") or "").strip()
    )
    auto_download = bool(cfg.get("auto_download", True))
    max_download_height = int(cfg.get("max_download_height", 720) or 720)
    download_workers = int(cfg.get("download_workers", 3) or 3)

    # Import the package (add to sys.path if needed)
    pkg_dir = str(_ASSET_RETRIEVAL_PACKAGE)
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

    try:
        from asset_retrieval_tool import search_video_segments, download_video_results
    except ImportError as exc:
        _diag("asset_retrieval_tool", "import_error", error=str(exc))
        return []

    try:
        results = search_video_segments(
            visual_query,
            top_k=top_k,
            sources=sources,
            youtube_api_key=youtube_api_key or None,
            bilibili_cookie=bilibili_cookie or None,
        )
    except Exception as exc:
        _diag("asset_retrieval_tool", "search_error", error=str(exc))
        return []

    if not results:
        _diag("asset_retrieval_tool", "empty")
        return []

    # Download videos to get local paths
    if auto_download:
        try:
            dl_dir = str(Path(output_dir) / "asset_retrieval")
            results = download_video_results(
                results,
                dl_dir,
                max_workers=download_workers,
                cookie=bilibili_cookie or None,
                max_height=max_download_height,
            )
        except Exception as exc:
            _diag("asset_retrieval_tool", "download_error", error=str(exc))

    # Normalize to AICut clip candidate schema
    candidates: list[dict[str, Any]] = []
    for i, item in enumerate(results):
        local_path = str(item.get("local_video_path", "") or "")
        source_url = str(item.get("source_video", "") or "")
        clip_path = local_path if local_path and Path(local_path).exists() else source_url
        candidates.append({
            "clip_id": f"art_{i:03d}_{item.get('platform', 'online')}_{item.get('video_id', '')}",
            "clip_path": clip_path,
            "score": float(item.get("score", 0.8 - i * 0.03) or 0.0),
            "duration_sec": float(item.get("duration", 0.0) or 0.0),
            "description": str(item.get("title", "") or "")[:300],
            "tags": [],
            "source_url": source_url,
            "platform": str(item.get("platform", "") or ""),
            "thumbnail": str(item.get("thumbnail", "") or ""),
        })

    _diag("asset_retrieval_tool", "success", count=len(candidates))
    return candidates


def _search_via_registered_provider(
    provider_name: str,
    visual_query: str,
    top_k: int,
    target_duration: float,
    min_duration_ratio: float,
    max_loop_count: float,
    output_dir: str,
    pipeline_config: dict[str, Any] | None = None,
    shot: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Bridge a `pipeline.providers` provider into the clip-candidate schema."""
    if get_video_provider(provider_name) is None:
        _diag(provider_name, "unknown_provider", registered=list_video_providers())
        return []

    query = VideoQuery(
        query=visual_query,
        top_k=top_k,
        target_duration_sec=target_duration,
        min_duration_sec=max(0.0, target_duration * min_duration_ratio),
        max_duration_sec=max(0.0, target_duration * max_loop_count),
        output_dir=str(Path(output_dir) / "providers" / provider_name),
        options=provider_options(pipeline_config, provider_name),
        pipeline_config=pipeline_config or {},
        shot=shot or {},
    )
    candidates = run_video_provider(provider_name, query)
    _diag(
        provider_name,
        "success" if candidates else "empty",
        count=len(candidates),
    )
    return candidates


def search_clips(
    visual_query: str,
    asset_library: str = "",
    top_k: int = 10,
    output_dir: str = "/tmp/aicut_retrieval",
    target_duration: float = 0.0,
    min_duration_ratio: float = 0.5,
    max_loop_count: float = 2.0,
) -> list[dict[str, Any]]:
    """Search video clips by visual query."""
    return search_clips_with_backend(
        visual_query=visual_query,
        asset_library=asset_library,
        top_k=top_k,
        output_dir=output_dir,
        target_duration=target_duration,
        min_duration_ratio=min_duration_ratio,
        max_loop_count=max_loop_count,
    )[0]


def search_clips_with_backend(
    visual_query: str,
    asset_library: str = "",
    top_k: int = 10,
    output_dir: str = "/tmp/aicut_retrieval",
    backend_order: list[str] | None = None,
    video_cli_size: int | None = None,
    target_duration: float = 0.0,
    min_duration_ratio: float = 0.5,
    max_loop_count: float = 2.0,
    duration_tolerance_sec: float = 0.75,
    allow_short_fallback: bool = False,
    require_horizontal: bool = True,
    require_duration_coverage: bool = False,
    min_aspect_ratio: float = 16.0 / 9.0,
    local_scan_exclude_prefixes: list[str] | None = None,
    local_scan_exclude_keywords: list[str] | None = None,
    asset_retrieval_cfg: dict[str, Any] | None = None,
    pipeline_config: dict[str, Any] | None = None,
    shot: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Search video clips by visual query.

    `backend_order` is tried left to right until one yields candidates that
    survive the duration/orientation filter, e.g.:
      ["mmfinder_video_cli", "mmfinder_subprocess", "mmfinder_embedding_recall", "local_scan"]
      ["asset_retrieval_tool"]           # bundled open-web search
      ["http_video", "local_folder_video"]  # providers from pipeline.providers
    """
    _reset_diagnostics()
    save_dir = str(Path(output_dir) / "clips" / visual_query[:20].replace("/", "_"))
    order = _normalize_backend_order(backend_order)
    recall_size = max(top_k, int(video_cli_size or 20))
    short_fallback_pool: list[dict[str, Any]] = []

    for backend in order:
        candidates: list[dict[str, Any]] = []
        if backend == "mmfinder_subprocess":
            candidates = _search_via_subprocess(
                visual_query,
                save_dir,
                num=top_k,
                target_duration=max(1.0, target_duration),
            )
        elif backend == "mmfinder_video_cli":
            candidates = _recall_videos_via_cli(visual_query, size=recall_size)
        elif backend == "mmfinder_embedding_recall":
            candidates = _search_via_embedding(visual_query, top_k=recall_size)
        elif backend == "asset_retrieval_tool":
            candidates = _search_via_asset_retrieval_tool(
                visual_query,
                top_k=recall_size,
                asset_retrieval_cfg=asset_retrieval_cfg,
                output_dir=output_dir,
            )
        elif backend == "local_scan":
            candidates = _scan_local_library(
                visual_query,
                asset_library,
                recall_size,
                exclude_prefixes=local_scan_exclude_prefixes or [],
                exclude_keywords=local_scan_exclude_keywords or [],
            )
        else:
            candidates = _search_via_registered_provider(
                backend,
                visual_query,
                top_k=recall_size,
                target_duration=target_duration,
                min_duration_ratio=min_duration_ratio,
                max_loop_count=max_loop_count,
                output_dir=output_dir,
                pipeline_config=pipeline_config,
                shot=shot,
            )
        if candidates:
            eligible, rejected = _filter_candidates_by_duration(
                candidates,
                target_duration=target_duration,
                min_duration_ratio=min_duration_ratio,
                max_loop_count=max_loop_count,
                duration_tolerance_sec=duration_tolerance_sec,
                top_k=top_k,
                require_horizontal=require_horizontal,
                require_duration_coverage=require_duration_coverage,
                min_aspect_ratio=min_aspect_ratio,
            )
            if eligible:
                _diag(
                    backend,
                    "duration_constrained",
                    target_duration_sec=round(target_duration, 3),
                    eligible_count=len(eligible),
                    rejected_count=len(rejected),
                )
                return eligible, backend
            short_fallback_pool.extend(rejected)
            _diag(
                backend,
                "duration_rejected_all",
                target_duration_sec=round(target_duration, 3),
                rejected_count=len(rejected),
            )

    if allow_short_fallback and short_fallback_pool:
        fallback = sorted(
            short_fallback_pool,
            key=lambda item: (
                float(item.get("duration_sec", 0.0) or 0.0),
                float(item.get("score", 0.0) or 0.0),
            ),
            reverse=True,
        )[:top_k]
        _diag(
            "duration_fallback",
            "short_candidates",
            target_duration_sec=round(target_duration, 3),
            count=len(fallback),
        )
        return fallback, "duration_fallback"

    _diag("none", "empty")
    return [], "none"


def _probe_clip_resolution(path: str, timeout_sec: int = 15) -> tuple[int, int]:
    """Return (width, height) for a video clip, (0, 0) if unknown."""
    candidate = Path(path).expanduser()
    if not candidate.exists() or candidate.stat().st_size <= 0:
        return (0, 0)
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                str(candidate),
            ],
            capture_output=True, text=True, timeout=timeout_sec,
            env=_media_subprocess_env(),
        )
        if proc.returncode == 0 and proc.stdout.strip():
            parts = proc.stdout.strip().split(",")
            return (int(parts[0] or 0), int(parts[1] or 0))
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError):
        pass
    return (0, 0)


def _filter_candidates_by_duration(
    candidates: list[dict[str, Any]],
    *,
    target_duration: float,
    min_duration_ratio: float,
    max_loop_count: float,
    duration_tolerance_sec: float,
    top_k: int,
    require_horizontal: bool = True,
    require_duration_coverage: bool = False,
    min_aspect_ratio: float = 16.0 / 9.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter candidates by duration, orientation, and aspect ratio.

    *require_duration_coverage* — when True the source clip must be at
    least ``target_duration - tolerance`` seconds long.  This ensures the
    retrieved material already covers the TTS narration so no downstream
    looping/trimming is needed.

    *require_horizontal* — reject portrait clips.

    *min_aspect_ratio* — reject clips narrower than 16:9 so they fit the
    HyperFrames 1920x1080 template without heavy pillarboxing.
    """
    target = max(0.0, float(target_duration or 0.0))
    ratio = min(max(float(min_duration_ratio or 0.0), 0.0), 1.0)
    max_loops = max(1.0, float(max_loop_count or 1.0))
    tolerance = max(0.0, float(duration_tolerance_sec or 0.0))
    minimum_duration = max(0.0, target * ratio, target / max_loops - tolerance)
    # When duration coverage is required, the clip must be >= target - tolerance.
    if require_duration_coverage and target > 0:
        minimum_duration = max(minimum_duration, target - tolerance)

    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for raw in candidates:
        item = dict(raw)
        duration = _candidate_duration(item)
        item["duration_sec"] = round(duration, 3)
        item["target_duration_sec"] = round(target, 3)
        item["minimum_source_duration_sec"] = round(minimum_duration, 3)
        item["duration_shortfall_sec"] = round(max(0.0, target - duration), 3)
        item["loop_count"] = round(target / duration, 3) if target > 0 and duration > 0 else 1.0

        # --- Orientation & aspect-ratio gate ---
        if require_horizontal or min_aspect_ratio > 0:
            clip_path = str(item.get("clip_path") or "")
            width = int(item.get("width") or 0)
            height = int(item.get("height") or 0)
            if width <= 0 or height <= 0:
                width, height = _probe_clip_resolution(clip_path)
            if width > 0 and height > 0:
                aspect = width / height
                item["resolution"] = f"{width}x{height}"
                item["aspect_ratio"] = round(aspect, 3)
                item["orientation"] = "landscape" if width >= height else "portrait"
                if require_horizontal and height > width:
                    item["duration_constraint_status"] = "rejected_portrait"
                    rejected.append(item)
                    continue
                if min_aspect_ratio > 0 and aspect < min_aspect_ratio - 0.01:
                    item["duration_constraint_status"] = (
                        f"rejected_aspect_ratio:{aspect:.2f}<{min_aspect_ratio:.2f}"
                    )
                    rejected.append(item)
                    continue
            else:
                item["orientation"] = "unknown"

        if target <= 0 or duration + tolerance >= minimum_duration:
            eligible.append(item)
        else:
            item["duration_constraint_status"] = "rejected_too_short"
            rejected.append(item)

    eligible.sort(
        key=lambda item: (
            float(item.get("score", 0.0) or 0.0),
            -float(item.get("loop_count", 1.0) or 1.0),
            float(item.get("duration_sec", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return eligible[:top_k], rejected


def _candidate_duration(candidate: dict[str, Any]) -> float:
    try:
        duration = float(candidate.get("duration_sec") or candidate.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration > 0:
        return duration
    clip_path = str(candidate.get("clip_path") or "")
    if not clip_path:
        return 0.0
    return _probe_media_duration(clip_path)


def _probe_media_duration(path: str, timeout_sec: int = 20) -> float:
    candidate = Path(path).expanduser()
    if not candidate.exists() or candidate.stat().st_size <= 0:
        return 0.0
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(candidate),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=_media_subprocess_env(),
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return max(0.0, float(proc.stdout.strip()))
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return 0.0


def _scan_local_library(
    visual_query: str,
    asset_library: str,
    top_k: int,
    exclude_prefixes: list[str] | None = None,
    exclude_keywords: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not asset_library or not Path(asset_library).exists():
        return []
    clips_dir = Path(asset_library) / "clips"
    if not clips_dir.exists():
        clips_dir = Path(asset_library)

    prefixes = [str(Path(p).expanduser().resolve()) for p in (exclude_prefixes or []) if str(p).strip()]
    keywords = [str(k).strip().lower() for k in (exclude_keywords or []) if str(k).strip()]

    def _is_excluded(path_obj: Path) -> bool:
        full = str(path_obj.resolve())
        lower_full = full.lower()
        if any(lower_full.startswith(pref.lower()) for pref in prefixes):
            return True
        if any(k in lower_full for k in keywords):
            return True
        return False

    all_mp4s = sorted(clips_dir.glob("**/*.mp4"))
    available = [p for p in all_mp4s if not _is_excluded(p)]
    query_tokens = _local_query_tokens(visual_query)

    def _semantic_score(path_obj: Path) -> float:
        haystack = f"{path_obj.parent.name} {path_obj.stem}".lower().replace("-", " ").replace("_", " ")
        return float(sum(1 for token in query_tokens if token and token in haystack))

    mp4s = sorted(
        available,
        key=lambda path_obj: (_semantic_score(path_obj), path_obj.stat().st_size),
        reverse=True,
    )[:top_k]
    if not mp4s:
        _diag(
            "local_scan",
            "empty",
            directory=str(clips_dir),
            excluded_prefixes=prefixes,
            excluded_keywords=keywords,
            total_candidates=len(all_mp4s),
        )
        return []
    _diag(
        "local_scan",
        "success",
        count=len(mp4s),
        directory=str(clips_dir),
        excluded_prefixes=prefixes,
        excluded_keywords=keywords,
    )
    max_semantic = max((_semantic_score(path_obj) for path_obj in mp4s), default=0.0)
    return [
        {
            "clip_id": f"local_{i:03d}",
            "clip_path": str(p),
            "score": round(
                0.35
                + (0.45 * _semantic_score(p) / max(1.0, max_semantic))
                + max(0.0, 0.15 - i * 0.01),
                3,
            ),
            "duration_sec": 0.0,
            "description": f"{p.parent.name} / {p.stem}",
            "tags": [token for token in query_tokens if token in p.stem.lower()][:6],
        }
        for i, p in enumerate(mp4s)
    ]


def _local_query_tokens(query: str) -> list[str]:
    text = str(query or "").lower()
    tokens = [token for token in re.split(r"[\s，。；、/]+", text) if len(token) >= 2]
    aliases = {
        "足球": ["soccer", "football", "world cap", "world cup"],
        "体育场": ["soccer", "stadium", "world cap", "world cup"],
        "门将": ["soccer", "goalkeeper", "world cap", "world cup"],
        "世界杯": ["world cap", "world cup", "soccer"],
        "学习": ["study", "school"],
        "手机": ["phone", "mobile"],
        "新闻": ["news"],
    }
    for keyword, values in aliases.items():
        if keyword in text:
            tokens.extend(values)
    return list(dict.fromkeys(tokens))


def _normalize_backend_order(backend_order: list[str] | None) -> list[str]:
    default = [
        "mmfinder_video_cli",
        "mmfinder_subprocess",
        "mmfinder_embedding_recall",
    ]
    # Names registered in `pipeline.providers` are accepted alongside the
    # built-in backends, so a plugin can be listed directly in backend_order.
    allowed = (
        set(default)
        | {"local_scan", "asset_retrieval_tool"}
        | set(list_video_providers())
    )
    if not backend_order:
        return default
    cleaned = [str(x).strip() for x in backend_order if str(x).strip() in allowed]
    if not cleaned:
        return default
    # 严格按照 config 指定的顺序，不强制追加未声明的 backend。
    # config 中省略 local_scan 意味着不希望扫描本地素材库（避免误命中原始视频）。
    return cleaned


def select_clip(
    llm: LLMClient,
    shot: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Use LLM to select best clip from candidates."""
    if not candidates:
        return {"selected_clip_id": "", "match_score": 0.0, "status": "no_candidates"}

    prompt = SELECT_TEMPLATE.format(
        shot_id=shot.get("shot_id", ""),
        visual_query=shot.get("visual_query", ""),
        visual_direction=shot.get("visual_direction", ""),
        vocal_script=shot.get("vocal_script", ""),
        duration_sec=shot.get("duration_sec", 8),
        minimum_duration_sec=min(
            (float(item.get("minimum_source_duration_sec", 0.0) or 0.0) for item in candidates),
            default=0.0,
        ),
        candidates_json=json.dumps(candidates, ensure_ascii=False, indent=2),
    )
    result = _as_dict(llm.chat_json(SYSTEM_PROMPT, prompt, temperature=0.1))
    candidate_by_id = {str(item.get("clip_id", "")): item for item in candidates}
    selected = candidate_by_id.get(str(result.get("selected_clip_id", "")))
    if selected is None:
        selected = candidates[0]
        result["selected_clip_id"] = selected["clip_id"]
    result["clip_path"] = selected.get("clip_path", "")
    result["match_score"] = selected.get("score", 0.0)
    result["duration_sec"] = selected.get("duration_sec", 0.0)
    result["duration_shortfall_sec"] = selected.get("duration_shortfall_sec", 0.0)
    result["loop_count"] = selected.get("loop_count", 1.0)
    result["orientation"] = selected.get("orientation", "unknown")
    result["resolution"] = selected.get("resolution", "")
    result["aspect_ratio"] = selected.get("aspect_ratio", 0.0)
    result["caption"] = str(
        selected.get("caption")
        or selected.get("caption_zh")
        or selected.get("description")
        or selected.get("title")
        or ""
    )
    return result


def select_clip_fast(
    shot: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select the highest-scoring candidate without an LLM call."""
    if not candidates:
        return {"selected_clip_id": "", "match_score": 0.0, "status": "no_candidates"}
    selected = max(
        candidates,
        key=lambda item: (
            float(item.get("score", 0.0) or 0.0),
            float(item.get("duration_sec", 0.0) or 0.0),
            -float(item.get("loop_count", 1.0) or 1.0),
        ),
    )
    return {
        "selected_clip_id": selected.get("clip_id", ""),
        "clip_path": selected.get("clip_path", ""),
        "match_score": selected.get("score", 0.0),
        "selection_rationale": "fast_score_select: use highest retrieval score to avoid generator bottleneck",
        "duration_sec": selected.get("duration_sec", 0.0),
        "duration_shortfall_sec": selected.get("duration_shortfall_sec", 0.0),
        "loop_count": selected.get("loop_count", 1.0),
        "orientation": selected.get("orientation", "unknown"),
        "resolution": selected.get("resolution", ""),
        "aspect_ratio": selected.get("aspect_ratio", 0.0),
        "caption": str(
            selected.get("caption")
            or selected.get("caption_zh")
            or selected.get("description")
            or selected.get("title")
            or ""
        ),
    }


def run(
    llm: LLMClient,
    shot: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Full retrieval pipeline: search → LLM select."""
    asset_library = config.get("paths", {}).get("asset_library", "")
    retrieval_cfg = config.get("retrieval", {})
    top_k = retrieval_cfg.get("top_k", 10)
    output_dir = config.get("paths", {}).get("output_root", "/tmp/aicut_retrieval")
    backend_order_raw = retrieval_cfg.get("backend_order", [])
    backend_order = backend_order_raw if isinstance(backend_order_raw, list) else []
    video_cli_size = retrieval_cfg.get("video_cli_size", 20)
    target_duration = _safe_float(
        shot.get("tts_duration_sec", shot.get("duration_sec", 0.0)),
        0.0,
    )
    min_duration_ratio = _safe_float(retrieval_cfg.get("min_duration_ratio", 1.0), 1.0)
    max_loop_count = _safe_float(retrieval_cfg.get("max_loop_count", 1.0), 1.0)
    duration_tolerance_sec = _safe_float(retrieval_cfg.get("duration_tolerance_sec", 1.0), 1.0)
    allow_short_fallback = bool(retrieval_cfg.get("allow_short_candidate_fallback", False))
    require_horizontal = bool(retrieval_cfg.get("require_horizontal", True))
    require_duration_coverage = bool(retrieval_cfg.get("require_duration_coverage", False))
    min_aspect_ratio = _safe_float(retrieval_cfg.get("min_aspect_ratio", 16.0 / 9.0), 16.0 / 9.0)
    return_candidates = max(1, int(retrieval_cfg.get("return_candidates", top_k) or top_k))
    select_with_llm = bool(retrieval_cfg.get("select_with_llm", False))

    paths_cfg = config.get("paths", {}) if isinstance(config.get("paths"), dict) else {}
    local_scan_exclude_prefixes = retrieval_cfg.get("local_scan_exclude_prefixes", [])
    if not isinstance(local_scan_exclude_prefixes, list):
        local_scan_exclude_prefixes = []
    raw_videos_dir = str(paths_cfg.get("raw_videos_dir", "") or "").strip()
    if raw_videos_dir:
        local_scan_exclude_prefixes.append(raw_videos_dir)

    local_scan_exclude_keywords = retrieval_cfg.get("local_scan_exclude_keywords", ["/test_data/", "/stage1_parse/"])
    if not isinstance(local_scan_exclude_keywords, list):
        local_scan_exclude_keywords = ["/test_data/", "/stage1_parse/"]

    # Which retrieval backend runs. `mmfinder` keeps the configured
    # backend_order; every other provider name replaces it outright.
    providers_cfg = config.get("providers", {})
    if isinstance(providers_cfg, dict):
        load_provider_plugins(providers_cfg.get("plugins"))

    retrieval_provider = str(config.get("retrieval_provider", "mmfinder") or "mmfinder")
    asset_retrieval_cfg = config.get("asset_retrieval", {})
    if not isinstance(asset_retrieval_cfg, dict):
        asset_retrieval_cfg = {}
    if retrieval_provider == "asset_retrieval_tool":
        backend_order = ["asset_retrieval_tool"]
    elif retrieval_provider not in ("", "mmfinder"):
        if get_video_provider(retrieval_provider) is not None:
            backend_order = [retrieval_provider]
        else:
            logger.warning(
                "[retrieve_visual] retrieval_provider=%r is not registered "
                "(available: %s); falling back to backend_order=%s",
                retrieval_provider,
                ", ".join(list_video_providers()) or "none",
                backend_order or "default",
            )

    candidates, backend = search_clips_with_backend(
        visual_query=shot.get("visual_query", ""),
        asset_library=asset_library,
        top_k=top_k,
        output_dir=output_dir,
        backend_order=backend_order,
        video_cli_size=video_cli_size,
        target_duration=target_duration,
        min_duration_ratio=min_duration_ratio,
        max_loop_count=max_loop_count,
        duration_tolerance_sec=duration_tolerance_sec,
        allow_short_fallback=allow_short_fallback,
        require_horizontal=require_horizontal,
        require_duration_coverage=require_duration_coverage,
        min_aspect_ratio=min_aspect_ratio,
        local_scan_exclude_prefixes=local_scan_exclude_prefixes,
        local_scan_exclude_keywords=local_scan_exclude_keywords,
        asset_retrieval_cfg=asset_retrieval_cfg,
        pipeline_config=config,
        shot=shot,
    )
    selection = select_clip(llm, shot, candidates) if select_with_llm else select_clip_fast(shot, candidates)
    return {
        "shot_id": shot.get("shot_id", ""),
        "candidates_count": len(candidates),
        "candidates": candidates[:return_candidates],
        "selection": selection,
        "clip_path": selection.get("clip_path", ""),
        "backend": backend,
        "retrieval_diagnostics": get_last_diagnostics(),
        "clip_meta": {
            "clip_id": selection.get("selected_clip_id", ""),
            "match_score": selection.get("match_score", 0.0),
            "rationale": selection.get("selection_rationale", ""),
            "caption": selection.get("caption", ""),
            "duration_sec": selection.get("duration_sec", 0.0),
            "target_duration_sec": round(target_duration, 3),
            "duration_shortfall_sec": selection.get("duration_shortfall_sec", 0.0),
            "loop_count": selection.get("loop_count", 1.0),
            "duration_constraint_met": bool(selection.get("selected_clip_id")),
            "orientation": selection.get("orientation", "unknown"),
            "resolution": selection.get("resolution", ""),
            "aspect_ratio": selection.get("aspect_ratio", 0.0),
            "require_horizontal": require_horizontal,
            "require_duration_coverage": require_duration_coverage,
            "min_aspect_ratio": min_aspect_ratio,
        },
        "status": "success" if selection.get("selected_clip_id") else "no_match",
    }


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, tuple):
        for item in value:
            if isinstance(item, dict):
                return item
            if isinstance(item, list) and item and isinstance(item[0], dict):
                return item[0]
        return {}
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return {}


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
