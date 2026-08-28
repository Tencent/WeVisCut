"""Tool_MusicSearch: Search BGM tracks from music library using music_style description.

Like footage retrieval, the BGM source is pluggable through
`services.bgm_provider` (which defaults to `services.retrieval_provider`):

  * ``asset_retrieval_tool`` — bundled search over Internet Archive netlabels
                               (free, CC-licensed instrumental music).
  * ``mmfinder``             — internal embedding-recall service, reached
                               through the CLIs under ``external_tool/mmfinder``.
  * anything registered in ``pipeline.providers`` — your own music library.

Whatever the source, a scan of `music.local_library_dirs` is the last resort,
so a deployment with just a folder of mp3s always has something to fall back on.

If the selected tracks are shorter than the video, the search is repeated with a
varied query until the timeline is covered.
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

from ..providers import (
    BgmQuery,
    get_bgm_provider,
    list_bgm_providers,
    load_provider_plugins,
    provider_options,
    run_bgm_provider,
)

logger = logging.getLogger("aicut_pipeline")

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Internal-only retrieval client; install it under external_tool/mmfinder/ or
# point MMFINDER_ROOT at your own checkout.
MMFINDER_ROOT = Path(os.environ.get(
    "MMFINDER_ROOT",
    str(_REPO_ROOT / "external_tool" / "mmfinder" / "WeAgent_WeStoryCut"),
))

# Optional pre-existing music library. Prefer `music.local_library_dirs` in
# studio_config.yaml; this env var is kept for backwards compatibility.
MUSIC_HUB_DIR = os.environ.get("WESTORYCUT_MUSIC_HUB_DIR", "").strip()

MUSIC_EMBED_INSTRUCTION = "Represent the music caption for retrieval."
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac"}
_LAST_DIAGNOSTICS: list[dict[str, Any]] = []
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


def _reset_diagnostics() -> None:
    _LAST_DIAGNOSTICS.clear()


def _diag(backend: str, status: str, **extra: Any) -> None:
    _LAST_DIAGNOSTICS.append({"backend": backend, "status": status, **extra})


def get_last_diagnostics() -> list[dict[str, Any]]:
    return list(_LAST_DIAGNOSTICS)


def _split_path_list(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = str(raw).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(os.pathsep) if part.strip()]


def _candidate_music_dirs(extra_dirs: list[str] | None = None) -> list[Path]:
    raw_dirs = (
        _split_path_list(os.environ.get("AICUT_MUSIC_LIBRARY_DIRS"))
        + _split_path_list(MUSIC_HUB_DIR)
        + (extra_dirs or [])
        + [str(MMFINDER_ROOT / "resource" / "bgms")]
    )
    seen: set[str] = set()
    dirs: list[Path] = []
    for raw in raw_dirs:
        text = str(raw or "").strip()
        if not text or text == ".":
            continue
        path = Path(text).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        dirs.append(path)
    return dirs


def _get_audio_duration(path: str) -> float:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return round(float(proc.stdout.strip()), 2)
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return 0.0


def _embed_music_query(query: str, timeout_sec: int | None = None) -> list[float]:
    """Embed a music description query via mmfinder embedding service."""
    if timeout_sec is None:
        timeout_sec = int(os.environ.get("AICUT_MMFINDER_EMBED_TIMEOUT", "45"))
    src_root = str(MMFINDER_ROOT / "src")
    pb_root = str(MMFINDER_ROOT / "src" / "pb")
    script = (
        "import json, sys\n"
        f"sys.path.insert(0, '{src_root}')\n"
        f"sys.path.insert(0, '{pb_root}')\n"
        "from pipeline.tools._internal.embedding_client import embed_inputs\n"
        "query = sys.argv[1]\n"
        "instruction = sys.argv[2]\n"
        "vectors = embed_inputs([{'text': query}], instruction=instruction, normalize=True)\n"
        "print(json.dumps(vectors[0]))\n"
    )
    env = _strip_proxy(os.environ.copy())
    env["PYTHONPATH"] = f"{src_root}:{pb_root}:{env.get('PYTHONPATH', '')}"

    try:
        proc = subprocess.run(
            [_python_bin(), "-c", script, query, MUSIC_EMBED_INSTRUCTION],
            capture_output=True, text=True, timeout=timeout_sec, env=env,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            embedding = json.loads(proc.stdout.strip().split("\n")[-1])
            _diag("mmfinder_music_embedding", "success", dim=len(embedding))
            return embedding
        _diag("mmfinder_music_embedding", "error", returncode=proc.returncode, stderr=proc.stderr[-500:])
    except subprocess.TimeoutExpired:
        _diag("mmfinder_music_embedding", "timeout", timeout_sec=timeout_sec)
    except (json.JSONDecodeError, FileNotFoundError) as exc:
        _diag("mmfinder_music_embedding", "error", error=str(exc))
    return []


def _recall_music_svrkit(embedding: list[float], size: int = 100) -> list[dict[str, Any]]:
    """Call the mmfinder music recall service through its subprocess client."""
    src_root = str(MMFINDER_ROOT / "src")
    pb_root = str(MMFINDER_ROOT / "src" / "pb")
    script = (
        "import asyncio, json, sys\n"
        f"sys.path.insert(0, '{src_root}')\n"
        f"sys.path.insert(0, '{pb_root}')\n"
        "from client.mmfindertext2videoagentbroker.recall_client import recall_music\n"
        "embedding = json.loads(sys.argv[1])\n"
        "size = int(sys.argv[2])\n"
        "results = asyncio.run(recall_music(embedding, size=size))\n"
        "print(json.dumps(results, ensure_ascii=False))\n"
    )
    env = _strip_proxy(os.environ.copy())
    env["PYTHONPATH"] = f"{src_root}:{pb_root}:{env.get('PYTHONPATH', '')}"

    try:
        proc = subprocess.run(
            [_python_bin(), "-c", script, json.dumps(embedding), str(size)],
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("AICUT_MMFINDER_RECALL_TIMEOUT", "30")),
            env=env,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            results = json.loads(proc.stdout.strip().split("\n")[-1])
            if isinstance(results, list):
                _diag("mmfinder_music_svrkit", "success" if results else "empty", count=len(results))
                return results
        _diag("mmfinder_music_svrkit", "error", returncode=proc.returncode, stderr=proc.stderr[-500:])
    except subprocess.TimeoutExpired:
        _diag("mmfinder_music_svrkit", "timeout", timeout_sec=int(os.environ.get("AICUT_MMFINDER_RECALL_TIMEOUT", "30")))
    except (json.JSONDecodeError, FileNotFoundError) as exc:
        _diag("mmfinder_music_svrkit", "error", error=str(exc))
    return []


def _recall_music_via_cli(query: str, size: int = 100) -> list[dict[str, Any]]:
    """Call GetVideoMusicRecall CLI as subprocess fallback."""
    script_dir = MMFINDER_ROOT / "scripts" / "get-video-music-recall"
    script_path = script_dir / "GetVideoMusicRecall.py"
    if not script_path.exists():
        _diag("mmfinder_music_cli", "missing_script", script_path=str(script_path))
        return []

    src_root = str(MMFINDER_ROOT / "src")
    pb_root = str(MMFINDER_ROOT / "src" / "pb")
    env = _strip_proxy(os.environ.copy())
    env["PYTHONPATH"] = ":".join([
        src_root, pb_root, str(script_dir), env.get("PYTHONPATH", ""),
    ])

    cmd = [
        _python_bin(), str(script_path),
        "--text", query,
        "--mode", "music",
        "--size", str(size),
        "--embed-timeout", str(int(os.environ.get("AICUT_MMFINDER_EMBED_TIMEOUT", "45"))),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("AICUT_MMFINDER_CLI_TIMEOUT", "60")),
            env=env,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            result = _parse_json_from_output(proc.stdout)
            if isinstance(result, dict):
                data_list = result.get("data_list", [])
                _diag("mmfinder_music_cli", "success" if data_list else "empty", count=len(data_list))
                return data_list
        _diag("mmfinder_music_cli", "error", returncode=proc.returncode, stderr=proc.stderr[-500:])
    except subprocess.TimeoutExpired:
        _diag("mmfinder_music_cli", "timeout", timeout_sec=int(os.environ.get("AICUT_MMFINDER_CLI_TIMEOUT", "60")))
    except (json.JSONDecodeError, FileNotFoundError) as exc:
        _diag("mmfinder_music_cli", "error", error=str(exc))
    return []


def _parse_json_from_output(output: str) -> Any:
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


def _build_track(raw: dict[str, Any], rank: int, local_library_dirs: list[str] | None = None) -> dict[str, Any]:
    """Normalize a recall result into a BGM track dict."""
    src_name = Path(str(raw.get("src_path", ""))).name
    ext_info = {}
    if raw.get("ext_info"):
        try:
            ext_info = json.loads(raw["ext_info"]) if isinstance(raw["ext_info"], str) else raw["ext_info"]
        except (json.JSONDecodeError, TypeError):
            pass

    src_path = str(raw.get("src_path", "") or "")
    cdn_path = str(raw.get("cdn_path", "") or "")
    music_path = src_path or cdn_path
    if src_path and Path(src_path).exists():
        music_path = src_path
    elif src_name:
        for directory in _candidate_music_dirs(local_library_dirs):
            candidate = directory / src_name
            if candidate.exists():
                music_path = str(candidate)
                break

    duration = 0.0
    if music_path and Path(music_path).exists():
        duration = _get_audio_duration(music_path)

    return {
        "track_id": str(raw.get("id_str", raw.get("id", f"bgm_{rank:03d}"))),
        "music_path": music_path,
        "score": float(raw.get("score", 0.0) or 0.0),
        "duration_sec": duration,
        "name": Path(music_path).stem if music_path else f"bgm_{rank:03d}",
        "caption": str(ext_info.get("description", ext_info.get("caption", "")))[:200],
        "cdn_path": cdn_path,
        "rank": rank,
    }


def _scan_local_music(directory: str, limit: int = 20) -> list[dict[str, Any]]:
    """Scan a local directory for audio files as fallback."""
    files: list[Path] = []
    d = Path(directory)
    if not d.exists():
        return []
    cap = max(limit * 8, limit)
    for root, dirs, names in os.walk(d):
        dirs[:] = [
            name for name in dirs
            if name not in {"__MACOSX", ".git", "node_modules", "__pycache__"}
            and not name.startswith(".")
        ]
        for name in sorted(names):
            if name.startswith("._"):
                continue
            f = Path(root) / name
            if f.suffix.lower() in _AUDIO_EXTS and f.is_file() and f.stat().st_size > 0:
                files.append(f)
                if len(files) >= cap:
                    break
        if len(files) >= cap:
            break
    if not files:
        return []
    files.sort(key=_local_music_rank_key)

    tracks = []
    for f in files[:limit]:
        duration = _get_audio_duration(str(f))
        tracks.append({
            "track_id": f"local_{len(tracks):03d}",
            "music_path": str(f),
            "score": round(1.0 - len(tracks) * 0.02, 3),
            "duration_sec": duration,
            "name": f.stem,
            "caption": "",
            "cdn_path": "",
            "rank": len(tracks) + 1,
        })
    return tracks


def _local_music_rank_key(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    boost = 0
    if any(token in name for token in ("bgm", "music", "r&b", "纯音乐")):
        boost -= 10
    if any(token in name for token in ("shutter", "click", "sfx", "whoosh", "tick", "chime", "pop", "胶卷", "快门")):
        boost += 10
    return boost, str(path)


def _search_bgm_via_asset_retrieval_tool(
    music_style: str,
    top_k: int = 10,
    asset_retrieval_cfg: dict[str, Any] | None = None,
    output_dir: str = "/tmp/aicut_bgm",
) -> list[dict[str, Any]]:
    """Search BGM via asset_retrieval_tool (Internet Archive CC music).

    Uses the external_tool/asset_retrieval_tool package which searches
    Internet Archive netlabels for free CC-licensed pure music.
    """
    cfg = asset_retrieval_cfg or {}
    bgm_sources = cfg.get("bgm_sources", ["internet_archive"])
    if not isinstance(bgm_sources, list):
        bgm_sources = ["internet_archive"]
    auto_download = bool(cfg.get("auto_download", True))
    download_workers = int(cfg.get("download_workers", 3) or 3)

    pkg_dir = str(_ASSET_RETRIEVAL_PACKAGE)
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

    try:
        from asset_retrieval_tool import search_bgm, download_bgm_results
    except ImportError as exc:
        _diag("asset_retrieval_tool_bgm", "import_error", error=str(exc))
        return []

    try:
        results = search_bgm(
            music_style,
            top_k=top_k,
            sources=bgm_sources,
        )
    except Exception as exc:
        _diag("asset_retrieval_tool_bgm", "search_error", error=str(exc))
        return []

    if not results:
        _diag("asset_retrieval_tool_bgm", "empty")
        return []

    # Download BGM files to get local paths
    if auto_download:
        try:
            dl_dir = str(Path(output_dir) / "asset_retrieval_bgm")
            results = download_bgm_results(
                results,
                dl_dir,
                max_workers=download_workers,
            )
        except Exception as exc:
            _diag("asset_retrieval_tool_bgm", "download_error", error=str(exc))

    # Normalize to AICut BGM track schema
    tracks: list[dict[str, Any]] = []
    for i, item in enumerate(results):
        local_path = str(item.get("local_path", "") or "")
        download_url = str(item.get("download_url", "") or "")
        music_path = local_path if local_path and Path(local_path).exists() else download_url

        duration = float(item.get("duration_sec", 0.0) or 0.0)
        if duration <= 0 and music_path and Path(music_path).exists():
            duration = _get_audio_duration(music_path)

        tracks.append({
            "track_id": f"art_bgm_{i:03d}_{item.get('id', '')}",
            "music_path": music_path,
            "score": round(1.0 - i * 0.05, 3),
            "duration_sec": duration,
            "name": str(item.get("title", "") or f"bgm_{i:03d}"),
            "caption": str(item.get("artist", "") or ""),
            "cdn_path": download_url,
            "rank": i + 1,
            "license": str(item.get("license", "CC") or "CC"),
            "source": "internet_archive",
        })

    _diag("asset_retrieval_tool_bgm", "success", count=len(tracks))
    return tracks


def _pure_music_style(value: str) -> str:
    """Keep retrieval text focused on musical attributes only."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return "中速现代纪录片纯音乐，钢琴、弦乐与轻电子节拍，克制清晰"
    text = text.replace("背景纯音乐", "纯音乐").replace("背景音乐", "纯音乐").replace("配乐", "纯音乐")
    clauses = [
        clause.strip(" ，,。；;")
        for clause in re.split(r"[，,。；;]", text)
        if clause.strip(" ，,。；;")
    ]
    banned = ("适合", "贴合", "配合", "用于", "服务", "视频", "画面", "叙事", "观众", "主题", "内容", "背景")
    kept = [clause for clause in clauses if not any(word in clause for word in banned)]
    cleaned = "，".join(kept[:4])
    cleaned = re.sub(r"(?:纯音乐){2,}", "纯音乐", cleaned).strip("，")
    if not cleaned:
        cleaned = "中速现代纪录片纯音乐，钢琴、弦乐与轻电子节拍，克制清晰"
    elif "纯音乐" not in cleaned:
        cleaned = f"{cleaned}，纯音乐"
    return cleaned[:72]


def search_music(
    music_style: str,
    target_duration: float = 60.0,
    top_k: int = 20,
    local_library_dirs: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Search music library for BGM tracks matching the style description."""
    return search_music_with_backend(
        music_style=music_style,
        target_duration=target_duration,
        top_k=top_k,
        local_library_dirs=local_library_dirs,
    )[0]


def _search_bgm_via_registered_provider(
    provider_name: str,
    music_style: str,
    top_k: int,
    target_duration: float,
    output_dir: str,
    pipeline_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Bridge a `pipeline.providers` BGM provider into the track schema."""
    if get_bgm_provider(provider_name) is None:
        _diag(provider_name, "unknown_provider", registered=list_bgm_providers())
        return []

    query = BgmQuery(
        query=music_style,
        top_k=top_k,
        target_duration_sec=target_duration,
        output_dir=str(Path(output_dir) / "providers" / provider_name),
        options=provider_options(pipeline_config, provider_name),
        pipeline_config=pipeline_config or {},
    )
    tracks = run_bgm_provider(provider_name, query)
    _diag(provider_name, "success" if tracks else "empty", count=len(tracks))
    return tracks


def search_music_with_backend(
    music_style: str,
    target_duration: float = 60.0,
    top_k: int = 20,
    local_library_dirs: list[str] | None = None,
    retrieval_provider: str = "mmfinder",
    asset_retrieval_cfg: dict[str, Any] | None = None,
    output_dir: str = "/tmp/aicut_bgm",
    pipeline_config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Search for BGM tracks matching the style description.

    `retrieval_provider` selects the source: ``asset_retrieval_tool``,
    ``mmfinder``, or the name of anything registered in `pipeline.providers`.
    A local-directory scan is the shared fallback when the chosen source comes
    back empty.
    """
    music_style = _pure_music_style(music_style)

    _reset_diagnostics()
    all_tracks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    # A registered provider takes precedence over the built-in backends.
    custom_provider = (
        retrieval_provider
        if retrieval_provider not in ("", "mmfinder", "asset_retrieval_tool")
        and get_bgm_provider(retrieval_provider) is not None
        else ""
    )
    if custom_provider:
        tracks = _search_bgm_via_registered_provider(
            custom_provider,
            music_style,
            top_k=top_k,
            target_duration=target_duration,
            output_dir=output_dir,
            pipeline_config=pipeline_config,
        )
        for track in tracks:
            tid = track["track_id"]
            if tid not in seen_ids:
                seen_ids.add(tid)
                all_tracks.append(track)
        if all_tracks:
            return all_tracks[:top_k], custom_provider
        # Fall through to the local scan below.

    # asset_retrieval_tool provider: Internet Archive CC music
    if not custom_provider and retrieval_provider == "asset_retrieval_tool":
        tracks = _search_bgm_via_asset_retrieval_tool(
            music_style,
            top_k=top_k,
            asset_retrieval_cfg=asset_retrieval_cfg,
            output_dir=output_dir,
        )
        for track in tracks:
            tid = track["track_id"]
            if tid not in seen_ids:
                seen_ids.add(tid)
                all_tracks.append(track)
        if all_tracks:
            return all_tracks[:top_k], "asset_retrieval_tool"
        # Fall through to local scan if asset_retrieval_tool returned nothing

    # mmfinder provider (default)
    if not custom_provider and retrieval_provider != "asset_retrieval_tool":
        # Primary: mmfinder CLI recall (text embedding + service-side recall).
        raw_results = _recall_music_via_cli(music_style, size=min(top_k * 3, 200))
        for i, raw in enumerate(raw_results):
            track = _build_track(raw, i + 1, local_library_dirs=local_library_dirs)
            tid = track["track_id"]
            if tid not in seen_ids and track["music_path"]:
                seen_ids.add(tid)
                all_tracks.append(track)
        if all_tracks:
            accessible = [t for t in all_tracks if Path(t["music_path"]).exists()]
            return (accessible or all_tracks[:top_k])[:top_k], "mmfinder_music_cli"

        # Secondary: direct embedding + svrkit recall.
        embedding = _embed_music_query(music_style)
        if embedding:
            raw_results = _recall_music_svrkit(embedding, size=min(top_k * 3, 200))
            for i, raw in enumerate(raw_results):
                track = _build_track(raw, i + 1, local_library_dirs=local_library_dirs)
                tid = track["track_id"]
                if tid not in seen_ids and track["music_path"]:
                    seen_ids.add(tid)
                    all_tracks.append(track)
            if all_tracks:
                accessible = [t for t in all_tracks if Path(t["music_path"]).exists()]
                return (accessible or all_tracks[:top_k])[:top_k], "mmfinder_music_svrkit"

    # Fallback: local scan from configured music library directories.
    if not all_tracks:
        for directory in _candidate_music_dirs(local_library_dirs):
            found = _scan_local_music(str(directory), limit=max(1, top_k - len(all_tracks)))
            if found:
                _diag("local_scan", "success", count=len(found), directory=str(directory))
            all_tracks.extend(found)
            if len(all_tracks) >= top_k:
                break
        if all_tracks:
            return all_tracks[:top_k], "local_scan"

    # Filter to tracks with accessible files
    accessible = [t for t in all_tracks if Path(t["music_path"]).exists()]
    if not accessible:
        accessible = all_tracks[:top_k]

    _diag("none", "empty")
    return accessible[:top_k], "none"


def select_bgm_tracks(
    candidates: list[dict[str, Any]],
    target_duration: float,
    min_tracks: int = 1,
) -> list[dict[str, Any]]:
    """Select enough BGM tracks to cover the target video duration.

    Picks tracks greedily by score until total duration >= target_duration,
    while keeping enough variety for long-form videos.
    """
    if not candidates:
        return []

    min_tracks = max(1, min(int(min_tracks), len(candidates)))
    sorted_candidates = sorted(candidates, key=lambda t: t.get("score", 0), reverse=True)
    selected: list[dict[str, Any]] = []
    total_dur = 0.0

    for source_track in sorted_candidates:
        track = dict(source_track)
        dur = track.get("duration_sec", 0.0)
        if dur <= 0:
            dur = 180.0  # assume ~3 min if unknown
            track["duration_sec"] = dur
            track["duration_estimated"] = True
        selected.append(track)
        total_dur += dur
        if total_dur >= target_duration and len(selected) >= min_tracks:
            break

    # Assign playback offsets
    offset = 0.0
    for track in selected:
        track["offset_sec"] = round(offset, 2)
        track["end_sec"] = round(offset + track["duration_sec"], 2)
        offset += track["duration_sec"]

    return selected


def _minimum_bgm_track_count(target_duration: float, music_cfg: dict[str, Any]) -> int:
    """Choose a varied playlist for long-form videos."""
    long_threshold = float(music_cfg.get("long_video_threshold_sec", 90.0) or 90.0)
    very_long_threshold = float(music_cfg.get("very_long_video_threshold_sec", 300.0) or 300.0)
    if target_duration >= very_long_threshold:
        return max(3, int(music_cfg.get("min_tracks_very_long_video", 3) or 3))
    if target_duration >= long_threshold:
        return max(2, int(music_cfg.get("min_tracks_long_video", 2) or 2))
    return max(1, int(music_cfg.get("min_tracks_short_video", 1) or 1))


def run(
    music_style: str,
    total_video_duration: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Full music search pipeline: search -> select enough tracks to cover duration."""
    music_style = _pure_music_style(music_style)
    # --- fake mode: skip real BGM retrieval ---
    exec_mode = str(config.get("execution", {}).get("mode", "real")).lower()
    skip_bgm = bool(
        config.get("execution", {}).get("fake_mode", {}).get("skip_bgm_search", True)
    )
    if exec_mode == "fake" and skip_bgm:
        from . import fake_utils
        logger.info("[music_search] fake 模式: 跳过真实 BGM 检索")
        return fake_utils.fake_music_search(music_style, total_video_duration)

    music_cfg = config.get("music", {})
    top_k = music_cfg.get("top_k", config.get("retrieval", {}).get("top_k", 20))
    min_tracks = _minimum_bgm_track_count(total_video_duration, music_cfg)
    local_library_dirs = _split_path_list(music_cfg.get("local_library_dirs", []))
    if music_cfg.get("music_hub_dir"):
        local_library_dirs.append(str(music_cfg["music_hub_dir"]))

    logger.info("[music_search] query=%s target_dur=%.1fs", music_style[:80], total_video_duration)

    providers_cfg = config.get("providers", {})
    if isinstance(providers_cfg, dict):
        load_provider_plugins(providers_cfg.get("plugins"))

    # `bgm_provider` lets BGM use a different source than footage; it falls back
    # to `retrieval_provider` so the single-knob config keeps working.
    retrieval_provider = str(
        config.get("bgm_provider")
        or config.get("retrieval_provider")
        or "mmfinder"
    )
    asset_retrieval_cfg = config.get("asset_retrieval", {})
    if not isinstance(asset_retrieval_cfg, dict):
        asset_retrieval_cfg = {}
    output_dir = config.get("paths", {}).get("output_root", "/tmp/aicut_bgm")

    candidates, backend = search_music_with_backend(
        music_style=music_style,
        target_duration=total_video_duration,
        top_k=top_k,
        local_library_dirs=local_library_dirs,
        retrieval_provider=retrieval_provider,
        asset_retrieval_cfg=asset_retrieval_cfg,
        output_dir=output_dir,
        pipeline_config=config,
    )
    logger.info("[music_search] found %d candidates", len(candidates))

    selected = select_bgm_tracks(candidates, total_video_duration, min_tracks=min_tracks)
    selected_duration = sum(t.get("duration_sec", 0) for t in selected)

    # If still not enough duration, try a second search with varied query
    if selected_duration < total_video_duration and len(candidates) < 3:
        fallback_query = f"中长篇纪录片纯音乐，{music_style[:48]}"
        extra, extra_backend = search_music_with_backend(
            fallback_query,
            target_duration=total_video_duration,
            top_k=10,
            local_library_dirs=local_library_dirs,
            retrieval_provider=retrieval_provider,
            asset_retrieval_cfg=asset_retrieval_cfg,
            output_dir=output_dir,
            pipeline_config=config,
        )
        if backend == "none" and extra_backend != "none":
            backend = extra_backend
        existing_ids = {t["track_id"] for t in selected}
        for t in extra:
            if t["track_id"] not in existing_ids:
                candidates.append(t)
        selected = select_bgm_tracks(candidates, total_video_duration, min_tracks=min_tracks)
        selected_duration = sum(t.get("duration_sec", 0) for t in selected)

    logger.info(
        "[music_search] selected %d tracks, total_dur=%.1fs (target=%.1fs)",
        len(selected), selected_duration, total_video_duration,
    )

    return {
        "music_style": music_style,
        "candidates_count": len(candidates),
        "candidates": candidates[:10],
        "selected_tracks": selected,
        "selected_count": len(selected),
        "total_bgm_duration_sec": round(selected_duration, 2),
        "target_duration_sec": round(total_video_duration, 2),
        "duration_covered": selected_duration >= total_video_duration,
        "minimum_track_count": min_tracks,
        "playlist_policy_version": 2,
        "backend": backend,
        "retrieval_diagnostics": get_last_diagnostics(),
        "local_library_dirs_checked": [str(p) for p in _candidate_music_dirs(local_library_dirs)],
        "status": "success" if selected else "no_tracks_found",
    }
