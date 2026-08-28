"""Transparent animation-layer render + compose (inference acceleration).

Splits the single-pass HyperFrames render (background clip + animation together)
into three cacheable phases:

  A. base video   — ffmpeg loops footage to shot duration + muxes TTS. Cacheable.
  B. overlay layer — assemble.py standalone (no --bg-video) → webm with alpha.
  C. compose       — ffmpeg overlays alpha layer onto base + burns subtitles.

Non-invasive: install() monkeypatches timeline_compose._render_shot_hyperframes
only when render_split.enabled is set. On ANY failure, falls back to the original
full renderer (correctness never regresses).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from pipeline.tools import timeline_compose as tc

logger = logging.getLogger("aicut_infer")

_ORIGINAL_RENDERER: Callable[..., dict[str, Any]] | None = None


def install(config: dict[str, Any] | None = None) -> bool:
    """Patch timeline_compose to use overlay-only rendering. Idempotent."""
    global _ORIGINAL_RENDERER
    rs = ((config or {}).get("render_split") or {})
    if not rs.get("enabled"):
        return False
    if _ORIGINAL_RENDERER is not None:
        return True
    _ORIGINAL_RENDERER = tc._render_shot_hyperframes

    def _patched(**kwargs: Any) -> dict[str, Any]:
        try:
            result = render_shot_overlay_split(orig=_ORIGINAL_RENDERER, **kwargs)
            if result is not None:
                return result
        except Exception as exc:
            logger.warning("[render_split] split render failed, falling back: %s", exc)
        return _ORIGINAL_RENDERER(**kwargs)  # type: ignore[misc]

    tc._render_shot_hyperframes = _patched  # type: ignore[assignment]
    logger.info("[render_split] overlay-only rendering enabled")
    return True


def uninstall() -> None:
    global _ORIGINAL_RENDERER
    if _ORIGINAL_RENDERER is not None:
        tc._render_shot_hyperframes = _ORIGINAL_RENDERER  # type: ignore[assignment]
        _ORIGINAL_RENDERER = None


def render_shot_overlay_split(
    *,
    shot: dict[str, Any],
    assets: dict[str, Any],
    timeline: dict[str, Any],
    output_dir: str,
    output_path: str,
    duration: float,
    config: dict[str, Any],
    orig: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Three-phase render. Returns None to signal fallback to original."""
    clip_path = str(assets.get("clip_path", "") or "")
    if not tc._existing_file(clip_path):
        return None

    shot_id = str(shot.get("shot_id", "unknown"))
    work_dir = Path(output_dir).expanduser().resolve() / "shots" / shot_id / "render_split"
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(
        str((config.get("render_split", {}) or {}).get("cache_dir") or "/tmp/aicut_render_split")
    ).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    audio_path = str(assets.get("audio_path", "") or "")
    fps = int(tc._cfg_int(config, ("services", "hyperframes", "accelerated_render", "fps"), 24))

    # Phase A: base video (cacheable)
    base_key = _hash({"clip": clip_path, "audio": audio_path, "dur": round(duration, 3), "fps": fps})
    base_video = cache_dir / f"base_{base_key}.mp4"
    if not tc._existing_file(str(base_video)):
        if not _build_base_video(clip_path, audio_path, duration, fps, base_video):
            return None
    else:
        logger.info("[render_split] base video cache hit: %s", base_video.name)

    # Phase B: transparent overlay
    overlay = _render_overlay_layer(
        shot=shot, assets=assets, output_dir=output_dir, duration=duration, config=config, cache_dir=cache_dir
    )
    if overlay is None or not tc._existing_file(overlay.get("overlay_path", "")):
        return None
    if not overlay.get("has_alpha"):
        logger.info("[render_split] overlay lacks alpha; falling back")
        return None

    # Phase C: compose + subtitle burn
    vocal_script = str(shot.get("vocal_script", "") or assets.get("vocal_script", "") or "")
    audio_dur = tc._probe_media_duration(audio_path) if tc._existing_file(audio_path) else duration
    subtitles = tc.build_subtitle_track(vocal_script, audio_dur if audio_dur > 0 else duration, config=config)
    if not _compose(
        base_video=base_video, overlay_path=Path(overlay["overlay_path"]),
        output_path=Path(output_path), duration=duration,
        subtitles=subtitles, fps=fps, config=config,
    ):
        return None

    result = {
        "status": "hyperframes_rendered",
        "output_path": str(Path(output_path).resolve()),
        "render_backend": "hyperframes_overlay_split",
        "work_dir": str(work_dir),
        "structure_id": overlay.get("structure_id", ""),
        "motion_id": overlay.get("motion_id", ""),
        "visual_id": overlay.get("visual_id", ""),
        "base_video": str(base_video),
        "overlay_layer": overlay.get("overlay_path", ""),
        "subtitle_count": len(subtitles),
        "audio_policy": "tts_only" if tc._existing_file(audio_path) else "silent",
        "original_audio_removed": True,
        "text_overlay_rendered": True,
        "subtitle_overlay_applied": bool(subtitles),
    }
    tc._write_hyperframes_report(work_dir, result)
    return result


# ─── Phase A ───

def _build_base_video(clip_path: str, audio_path: str, duration: float, fps: int, out_path: Path) -> bool:
    w, h = 1920, 1080
    vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1,fps={fps}"
    cmd = [tc._ffmpeg_binary(), "-y", "-stream_loop", "-1", "-i", clip_path]
    has_audio = tc._existing_file(audio_path)
    if has_audio:
        cmd += ["-i", audio_path]
    cmd += ["-t", f"{duration:.3f}", "-vf", vf, "-c:v", "mpeg4", "-q:v", "4", "-pix_fmt", "yuv420p", "-map", "0:v:0"]
    if has_audio:
        cmd += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "128k", "-shortest"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", str(out_path)]
    return _run(cmd) == 0 and tc._existing_file(str(out_path))


# ─── Phase B ───

def _render_overlay_layer(
    *, shot: dict[str, Any], assets: dict[str, Any],
    output_dir: str, duration: float, config: dict[str, Any], cache_dir: Path,
) -> dict[str, Any] | None:
    hf_bin, _ = tc._resolve_hyperframes_binary(config)
    if not hf_bin:
        return None
    gold_root = Path(str((config.get("paths", {}) or {}).get("gold_template_root", "") or "")).expanduser().resolve()
    assembler = gold_root / "assembler" / "assemble.py"
    if not assembler.exists():
        return None

    decoration = assets.get("decoration", {}) if isinstance(assets.get("decoration"), dict) else {}
    structure_id = tc._resolve_structure_id(decoration, gold_root)
    motion_id = tc._resolve_motion_id(decoration, gold_root)
    visual_id = tc._resolve_visual_id(decoration, gold_root, config)
    slots = decoration.get("slots", {}) if isinstance(decoration.get("slots"), dict) else {}
    if not slots:
        slots = tc._default_slots_from_shot(shot)

    timing = decoration.get("timing", {}) if isinstance(decoration.get("timing"), dict) else {}
    decor_start = tc._as_float(timing.get("decor_start_sec"), 0.3)
    decor_duration = tc._as_float(timing.get("decor_duration_sec"), min(4.0, duration))
    overlay_format = str((config.get("render_split", {}) or {}).get("overlay_format", "webm"))

    key = _hash({
        "s": structure_id, "m": motion_id, "v": visual_id,
        "slots": slots, "dur": round(duration, 3),
        "ds": round(decor_start, 3), "dd": round(decor_duration, 3), "fmt": overlay_format,
    })
    overlay_out = cache_dir / f"overlay_{key}.{overlay_format}"
    meta: dict[str, Any] = {
        "overlay_path": str(overlay_out),
        "structure_id": structure_id, "motion_id": motion_id, "visual_id": visual_id,
    }
    if tc._existing_file(str(overlay_out)):
        meta["has_alpha"] = _has_alpha(overlay_out)
        logger.info("[render_split] overlay cache hit: %s", overlay_out.name)
        return meta

    shot_id = str(shot.get("shot_id", "unknown"))
    stage_dir = Path(output_dir).expanduser().resolve() / "shots" / shot_id / "overlay_stage"
    stage_dir.mkdir(parents=True, exist_ok=True)
    tc._ensure_project_assets(gold_root, stage_dir)
    slots_path = stage_dir / "slots.json"
    slots_path.write_text(json.dumps(slots, ensure_ascii=False, indent=2), encoding="utf-8")

    import sys
    assemble_cmd = [
        sys.executable, str(assembler),
        "--structure", structure_id, "--motion", motion_id, "--visual", visual_id,
        "--slots", str(slots_path.resolve()), "--duration", f"{duration:.3f}",
        "--decor-start", f"{max(0.0, min(decor_start, duration)):.3f}",
        "--decor-duration", f"{max(0.1, min(decor_duration, duration)):.3f}",
        "--out", str((stage_dir / "index.html").resolve()),
    ]
    assembled = tc._run_command(assemble_cmd, cwd=str(gold_root), timeout_sec=120, env=os.environ.copy())
    if assembled.get("returncode") != 0:
        logger.warning("[render_split] assemble failed: %s", assembled.get("stderr", "")[-300:])
        return None

    tc._inline_font_family_variables(stage_dir / "index.html")
    tc._prepare_composition_fonts(index_path=stage_dir / "index.html", gold_root=gold_root, work_dir=stage_dir)
    _force_transparent_stage(stage_dir / "index.html")

    env = tc._build_hyperframes_env(config)
    render_cmd = tc._hf_command(hf_bin, [
        "render", str(stage_dir.resolve()),
        "--fps", str(tc._cfg_int(config, ("services", "hyperframes", "accelerated_render", "fps"), 24)),
        "--quality", str(tc._cfg_get(config, ("services", "hyperframes", "accelerated_render", "quality"), "draft")),
        "--workers", str(tc._cfg_get(config, ("services", "hyperframes", "accelerated_render", "workers"), "auto")),
        "--format", overlay_format,
        "--output", str(overlay_out),
    ])
    report = tc._run_command(
        render_cmd, cwd=str(stage_dir),
        timeout_sec=tc._cfg_int(config, ("services", "hyperframes", "accelerated_render", "timeout_seconds"),
                                tc._cfg_int(config, ("services", "hyperframes", "timeout"), 600)),
        env=env,
    )
    if report.get("returncode") != 0 or not tc._existing_file(str(overlay_out)):
        logger.warning("[render_split] hf render failed: %s", str(report.get("stderr", ""))[-300:])
        return None
    meta["has_alpha"] = _has_alpha(overlay_out)
    return meta


# ─── Phase C ───

def _compose(
    *, base_video: Path, overlay_path: Path, output_path: Path,
    duration: float, subtitles: list[dict[str, Any]], fps: int, config: dict[str, Any],
) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_chain = "[1:v]scale=1920:1080[ovl];[0:v][ovl]overlay=0:0:format=auto:shortest=1[comp]"
    last_label = "comp"
    subtitle_filters = tc._build_subtitle_filters(subtitles, config=config) if subtitles else []
    if subtitle_filters:
        overlay_chain += f";[{last_label}]" + ",".join(subtitle_filters) + "[outv]"
        last_label = "outv"
    cmd = [
        tc._text_ffmpeg_binary() if subtitle_filters else tc._ffmpeg_binary(),
        "-y", "-i", str(base_video), "-stream_loop", "-1", "-i", str(overlay_path),
        "-filter_complex", overlay_chain, "-map", f"[{last_label}]",
    ]
    if _has_audio_stream(base_video):
        cmd += ["-map", "0:a:0", "-c:a", "copy"]
    else:
        cmd += ["-an"]
    cmd += ["-t", f"{duration:.3f}", "-r", str(fps), "-c:v", "mpeg4", "-q:v", "4",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)]
    return _run(cmd) == 0 and tc._existing_file(str(output_path))


# ─── Helpers ───

def _hash(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def _run(cmd: list[str], timeout_sec: int = 300) -> int:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        if proc.returncode != 0:
            logger.warning("[render_split] cmd failed rc=%s: %s", proc.returncode, (proc.stderr or "")[-300:])
        return proc.returncode
    except subprocess.TimeoutExpired:
        return 124
    except FileNotFoundError:
        return 127


def _ffprobe_bin() -> str:
    return shutil.which("ffprobe") or "ffprobe"


def _has_alpha(path: Path) -> bool:
    try:
        proc = subprocess.run(
            [_ffprobe_bin(), "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=pix_fmt", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=20,
        )
        pix_fmt = (proc.stdout or "").strip()
        return any(tok in pix_fmt for tok in ("yuva", "rgba", "argb", "bgra"))
    except Exception:
        return False


def _has_audio_stream(path: Path) -> bool:
    try:
        proc = subprocess.run(
            [_ffprobe_bin(), "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=20,
        )
        return bool((proc.stdout or "").strip())
    except Exception:
        return False


def _force_transparent_stage(index_path: Path) -> None:
    """Inject CSS so standalone stage renders on transparent background."""
    try:
        if not index_path.exists():
            return
        html = index_path.read_text(encoding="utf-8")
        if "aicut-transparent-stage" in html:
            return
        css = (
            '<style id="aicut-transparent-stage">'
            "html,body{background:transparent!important;}"
            ".stage,.stage-panel,#stage,.composition,.bg,.background"
            "{background:transparent!important;background-color:transparent!important;}"
            "</style>"
        )
        html = html.replace("</head>", css + "</head>", 1) if "</head>" in html else css + html
        index_path.write_text(html, encoding="utf-8")
    except Exception as exc:
        logger.debug("[render_split] transparent CSS inject skipped: %s", exc)
