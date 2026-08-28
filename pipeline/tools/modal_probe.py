"""Reusable preflight probe for the multimodal tools.

This module backs both the standalone smoke-test CLI and the formal pipeline
preflight node. It exercises the three external multimodal capabilities that
the later graph stages depend on: visual retrieval, TTS, and BGM retrieval.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from . import generate_audio, music_search, retrieve_visual


DEFAULT_VIDEO_QUERY = "足球比赛精彩进攻画面，球员射门和门将扑救"
DEFAULT_TTS_TEXT = "这是一段 AICutPipeline 多模态工具预检旁白，用于验证文本转语音功能是否可用。"
DEFAULT_MUSIC_STYLE = "热血体育集锦背景音乐，节奏中快，鼓点明显，适合足球比赛复盘"

STRICT_VIDEO_BACKENDS = {"mmfinder_subprocess", "mmfinder_video_cli", "mmfinder_embedding_recall"}
STRICT_MUSIC_BACKENDS = {"mmfinder_music_svrkit", "mmfinder_music_cli"}


def enabled_for_mode(config: dict[str, Any], mode: str) -> bool:
    """Return whether modal preflight should run for this pipeline mode."""
    cfg = _probe_cfg(config)
    if not bool(cfg.get("enabled", True)):
        return False
    modes = cfg.get("run_modes", ["inference", "data_pipeline"])
    if isinstance(modes, str):
        modes = [part.strip() for part in modes.split(",") if part.strip()]
    return not modes or mode in set(modes)


def fail_on_error(config: dict[str, Any]) -> bool:
    return bool(_probe_cfg(config).get("fail_on_error", True))


def run_probe(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    mode: str = "inference",
    require_strict: bool | None = None,
    allow_fallback: bool | None = None,
    video_query: str = "",
    tts_text: str = "",
    music_style: str = "",
    top_k: int | None = None,
    music_target_duration: float | None = None,
    force_clear: bool = False,
) -> dict[str, Any]:
    """Run visual retrieval, TTS, and music retrieval probes."""
    cfg = _probe_cfg(config)
    if require_strict is None:
        require_strict = bool(cfg.get("require_strict", True))
    if allow_fallback is None:
        allow_fallback = bool(cfg.get("allow_fallback", not require_strict))

    out_dir = _fresh_output_dir(Path(output_dir).expanduser().resolve(), force=force_clear)
    top_k = int(top_k if top_k is not None else cfg.get("top_k", 5))
    target_duration = float(
        music_target_duration
        if music_target_duration is not None
        else cfg.get("music_target_duration", 60.0)
    )

    summary: dict[str, Any] = {
        "output_dir": str(out_dir),
        "mode": mode,
        "started_at": datetime.now().isoformat(),
        "require_strict": bool(require_strict),
        "allow_fallback": bool(allow_fallback),
        "tests": {},
    }

    test_plan = [
        (
            "video_retrieval",
            lambda: run_video_retrieval_probe(
                config,
                video_query or str(cfg.get("video_query", "") or DEFAULT_VIDEO_QUERY),
                out_dir,
                top_k,
                bool(allow_fallback),
            ),
        ),
        (
            "tts",
            lambda: run_tts_probe(
                config,
                tts_text or str(cfg.get("tts_text", "") or DEFAULT_TTS_TEXT),
                out_dir,
                bool(allow_fallback),
            ),
        ),
        (
            "music_retrieval",
            lambda: run_music_retrieval_probe(
                config,
                music_style or str(cfg.get("music_style", "") or DEFAULT_MUSIC_STYLE),
                out_dir,
                target_duration,
                bool(allow_fallback),
            ),
        ),
    ]

    for name, fn in test_plan:
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - probes should report all failures.
            result = {
                "passed": False,
                "strict_pass": False,
                "error": str(exc),
                "exception_type": type(exc).__name__,
            }
        summary["tests"][name] = result

    summary["finished_at"] = datetime.now().isoformat()
    summary["strict_passed"] = all(item.get("strict_pass", False) for item in summary["tests"].values())
    summary["passed"] = all(item.get("passed", False) for item in summary["tests"].values())
    if require_strict:
        summary["passed"] = bool(summary["passed"] and summary["strict_passed"])
    summary["status"] = "success" if summary["passed"] else "failed"
    summary["failure_reasons"] = failure_reasons(summary)
    _json_dump(out_dir / "summary.json", summary)
    return summary


def run_video_retrieval_probe(
    config: dict[str, Any],
    query: str,
    out_dir: Path,
    top_k: int,
    allow_fallback: bool,
) -> dict[str, Any]:
    video_dir = out_dir / "video_retrieval"
    candidates, backend = retrieve_visual.search_clips_with_backend(
        visual_query=query,
        asset_library=config.get("paths", {}).get("asset_library", ""),
        top_k=top_k,
        output_dir=str(video_dir),
    )
    candidates_with_links = _link_existing_files(candidates, "clip_path", video_dir / "candidate_links")
    strict_pass = backend in STRICT_VIDEO_BACKENDS and len(candidates) > 0
    result = {
        "query": query,
        "top_k": top_k,
        "backend": backend,
        "candidate_count": len(candidates),
        "passed": strict_pass or (allow_fallback and len(candidates) > 0),
        "strict_pass": strict_pass,
        "fallback_used": backend in {"local_scan", "none"},
        "retrieval_diagnostics": retrieve_visual.get_last_diagnostics(),
        "candidates": candidates_with_links,
    }
    _json_dump(video_dir / "video_retrieval_result.json", result)
    return result


def run_tts_probe(
    config: dict[str, Any],
    text: str,
    out_dir: Path,
    allow_fallback: bool,
) -> dict[str, Any]:
    tts_dir = out_dir / "tts"
    tts_cfg = config.get("tts", {})
    audio_path = tts_dir / "narration.wav"
    result = generate_audio.run_tts(
        text=text,
        output_path=str(audio_path),
        voice=tts_cfg.get("voice", "default"),
        speed=tts_cfg.get("speed", 1.0),
        backend=tts_cfg.get("backend", "auto"),
        service_url=tts_cfg.get("service_url", ""),
        language=tts_cfg.get("language", "Chinese"),
        instruct=tts_cfg.get("instruct", ""),
    )
    output_file = Path(str(result.get("audio_path", audio_path)))
    output_size = output_file.stat().st_size if output_file.exists() else 0
    enriched = {
        **result,
        "input_text": text,
        "output_exists": output_file.exists(),
        "output_size_bytes": output_size,
        "strict_pass": result.get("status") == "success" and output_size > 0,
        "fallback_used": result.get("status") == "fallback" or result.get("backend") == "estimated",
    }
    enriched["passed"] = enriched["strict_pass"] or (
        allow_fallback
        and output_size > 0
        and result.get("status") in {"success", "fallback"}
    )
    _json_dump(tts_dir / "tts_result.json", enriched)
    return enriched


def run_music_retrieval_probe(
    config: dict[str, Any],
    music_style: str,
    out_dir: Path,
    target_duration: float,
    allow_fallback: bool,
) -> dict[str, Any]:
    music_dir = out_dir / "music_retrieval"
    result = music_search.run(
        music_style=music_style,
        total_video_duration=target_duration,
        config=config,
    )
    candidates = result.get("candidates", [])
    selected = result.get("selected_tracks", [])
    result = {
        **result,
        "candidates": _link_existing_files(candidates, "music_path", music_dir / "candidate_links"),
        "selected_tracks": _link_existing_files(selected, "music_path", music_dir / "selected_links"),
        "strict_pass": result.get("backend") in STRICT_MUSIC_BACKENDS and bool(selected or candidates),
        "fallback_used": result.get("backend") in {"local_scan", "none"},
    }
    result["passed"] = result["strict_pass"] or (allow_fallback and bool(selected or candidates))
    _json_dump(music_dir / "music_retrieval_result.json", result)
    return result


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Compact a probe summary for trajectory/state logs."""
    tests = summary.get("tests", {})
    compact_tests: dict[str, Any] = {}
    for name, result in tests.items():
        compact_tests[name] = {
            "passed": bool(result.get("passed", False)),
            "strict_pass": bool(result.get("strict_pass", False)),
            "backend": result.get("backend", ""),
            "fallback_used": bool(result.get("fallback_used", False)),
            "candidate_count": result.get("candidate_count", result.get("candidates_count", 0)),
            "selected_count": result.get("selected_count", 0),
            "output_size_bytes": result.get("output_size_bytes", 0),
            "message": result.get("message", ""),
            "error": result.get("error", ""),
        }
    return {
        "status": summary.get("status", ""),
        "passed": bool(summary.get("passed", False)),
        "strict_passed": bool(summary.get("strict_passed", False)),
        "require_strict": bool(summary.get("require_strict", False)),
        "output_dir": summary.get("output_dir", ""),
        "summary_path": str(Path(str(summary.get("output_dir", ""))) / "summary.json") if summary.get("output_dir") else "",
        "failure_reasons": summary.get("failure_reasons", []),
        "tests": compact_tests,
    }


def failure_reasons(summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for name, result in summary.get("tests", {}).items():
        if result.get("passed") and (not summary.get("require_strict") or result.get("strict_pass")):
            continue
        detail = result.get("message") or result.get("error") or ""
        backend = result.get("backend") or ""
        if detail:
            reasons.append(f"{name}: {backend} {detail}".strip())
        else:
            reasons.append(f"{name}: backend={backend or 'unknown'} did not pass")
    return reasons


def _probe_cfg(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("modal_tools_preflight")
    if isinstance(cfg, dict):
        return cfg
    cfg = config.get("modal_probe")
    return cfg if isinstance(cfg, dict) else {}


def _fresh_output_dir(path: Path, force: bool = False) -> Path:
    if path.exists():
        if any(path.iterdir()):
            if not force:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = path.with_name(f"{path.name}_{stamp}")
            else:
                shutil.rmtree(path)
        elif not path.is_dir():
            raise RuntimeError(f"Output path exists and is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _link_existing_files(items: list[dict[str, Any]], key: str, dest_dir: Path) -> list[dict[str, Any]]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    linked: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        src = Path(str(item.get(key, "")))
        if not src.exists():
            linked.append({**item, "local_link": ""})
            continue
        suffix = src.suffix or ".bin"
        link_path = dest_dir / f"{idx + 1:03d}_{src.stem[:80]}{suffix}"
        try:
            if not link_path.exists():
                link_path.symlink_to(src)
            local_link = str(link_path)
        except OSError:
            shutil.copy2(src, link_path)
            local_link = str(link_path)
        linked.append({**item, "local_link": local_link})
    return linked
