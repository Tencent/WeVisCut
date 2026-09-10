"""Tool_TimelineCompose: Align video, audio, decoration, and subtitles on a timeline.

Produces a timeline JSON descriptor and attempts real ffmpeg render when
all asset files exist on disk.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from studio.newvis0907 import footage as footage_mod
from studio.newvis0907 import pacing as pacing_mod
from studio.newvis0907.subtitles import build_subtitle_cues

logger = logging.getLogger("aicut_pipeline")
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
FFMPEG_COMPAT_LIB_DIR = Path("/usr/lib64/dop-fuse")
STATIC_FFMPEG_PATH = Path(
    "/usr/local/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)
_HYPERFRAMES_RUNTIME_LOCK = threading.Lock()
_FONT_SOURCE_CACHE_DIR = Path("/tmp/aicut_hyperframes_fonts")


def assemble_shot_hyperframes(
    *,
    shot: dict[str, Any],
    assets: dict[str, Any],
    output_dir: str,
    duration: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    """只搭建 hyperframes 工程目录, 不渲染动画 (不起 Chrome).

    给 VLM-judge 的按镜头抽帧用: 判分只需要每个镜头几帧, 没必要先把整段动画
    渲染成 mp4。返回 dict 里的 ``work_dir`` 可以直接喂给
    ``hyperframes snapshot <work_dir> --at t1,t2,...``。

    成功时 status == "hyperframes_assembled"。所有失败分支都沿用完整渲染路径
    的状态码 (no_clip / hyperframes_runtime_error / hyperframes_assemble_error
    等), 调用方按 status 判断即可。
    """
    assets = _crop_footage_for_assemble(assets, duration)
    return _render_shot_hyperframes(
        shot=shot,
        assets=assets,
        timeline={},
        output_dir=output_dir,
        output_path="",
        duration=duration,
        config=config,
        assemble_only=True,
    )


def _crop_footage_for_assemble(assets: dict[str, Any], duration: float) -> dict[str, Any]:
    """把 clip 先本地裁到镜头时长, 再交给 assemble。

    ⚠️ 不裁会挂。``_link_or_copy`` 默认是 symlink, 于是 footage/source.mp4 指向
    Ceph 上的完整素材; 而 hyperframes 的静态服务器不支持 HTTP Range, 它是
    readFileSync 整个文件进内存再走 loopback。实测检索命中的素材有 2758s (46 分钟)
    的长片, snapshot 直接 `Navigation timeout of 10000 ms exceeded` —— 而且是在
    并发下才暴露, 单跑短素材时看不出来。

    裁剪走 stream copy (-c copy), 约 50-200ms, 落到本地盘。裁完长度变了, 所以
    clip_meta 里的 duration 也要一起改 —— 它驱动 assemble.py 的背景循环链
    (bg_clip_len), 不改的话背景会按原始 2758s 排布, 镜头这几秒等于是静止的。
    """
    src = str(assets.get("clip_path", "") or "")
    if not _existing_file(src) or duration <= 0:
        return assets
    src_dur = _probe_media_duration(src)
    # 已经足够短就别多此一举: 多一次转码就多一份磁盘和时延。
    if src_dur <= 0 or src_dur <= duration * 1.5 + 1.0:
        return assets

    keep = min(src_dur, duration * 1.5 + 1.0)   # 留点余量给 loop 边界
    dst_dir = Path(tempfile.mkdtemp(prefix="hf_footage_"))
    dst = dst_dir / f"cropped{Path(src).suffix or '.mp4'}"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", "0", "-t", f"{keep:.3f}",
           "-i", src, "-c", "copy", "-avoid_negative_ts", "make_zero", str(dst)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                              env=_media_subprocess_env())
    except Exception:
        return assets
    if proc.returncode != 0 or not _existing_file(str(dst)):
        # stream copy 在某些封装上会失败 (关键帧对不齐)。不值得为此再转一次码,
        # 直接用原素材 —— 顶多是慢, 不是错。
        return assets

    new_dur = _probe_media_duration(str(dst)) or keep
    meta = dict(assets.get("clip_meta") or {})
    for k in ("duration", "duration_sec"):
        if k in meta:
            meta[k] = new_dur
    return {**assets, "clip_path": str(dst), "clip_meta": meta}


def _media_subprocess_env() -> dict[str, str]:
    """Make the host FFmpeg usable on images that keep OpenSSL 1.0 compat libs isolated."""
    env = os.environ.copy()
    if FFMPEG_COMPAT_LIB_DIR.exists():
        current = env.get("LD_LIBRARY_PATH", "")
        parts = [part for part in current.split(":") if part]
        compat = str(FFMPEG_COMPAT_LIB_DIR)
        if compat not in parts:
            env["LD_LIBRARY_PATH"] = ":".join([compat, *parts])
    return env


def _ffmpeg_binary() -> str:
    configured = str(os.environ.get("AICUT_FFMPEG_PATH") or "").strip()
    if configured and Path(configured).exists():
        return configured
    if STATIC_FFMPEG_PATH.exists():
        return str(STATIC_FFMPEG_PATH)
    return shutil.which("ffmpeg") or "ffmpeg"


@lru_cache(maxsize=16)
def _ffmpeg_has(binary: str, kind: str, name: str) -> bool:
    """Probe whether *binary* exposes a given filter/encoder. Cached per process.

    The two ffmpeg builds available here have disjoint capabilities: the host
    build carries `drawtext` but no `libx264`, while the bundled imageio build
    carries `libx264` but no `drawtext`.  Anything that hardcodes one binary or
    one encoder breaks on the other, so callers must ask instead of assume.
    """
    flag = "-filters" if kind == "filter" else "-encoders"
    try:
        proc = subprocess.run(
            [binary, "-hide_banner", flag],
            capture_output=True,
            text=True,
            timeout=120,
            env=_media_subprocess_env(),
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
    if proc.returncode != 0:
        return False
    return any(
        name in line.split() for line in (proc.stdout or "").splitlines()
    )


def _text_ffmpeg_binary() -> str:
    """Pick an ffmpeg that can actually burn text (`drawtext`).

    Subtitle burn-in and decoration overlays are useless without `drawtext`;
    when it is missing every text render fails and HyperFrames shots collapse
    to `subtitle_overlay_failed`.  The host build is the usual owner of that
    filter, so it wins whenever it really provides it.
    """
    host = Path("/usr/bin/ffmpeg")
    if host.exists() and _ffmpeg_has(str(host), "filter", "drawtext"):
        return str(host)
    bundled = _ffmpeg_binary()
    if _ffmpeg_has(bundled, "filter", "drawtext"):
        return bundled
    # Neither build has drawtext: fall back to the host binary so behaviour
    # matches the historical default rather than silently changing encoders.
    return str(host) if host.exists() else bundled


def _text_video_encoder_args(binary: str) -> list[str]:
    """H.264 when the text binary supports it, else MPEG-4 Part 2.

    Browsers cannot decode mpeg4/mp4v, but the final stitch transcodes the
    concatenated result with the bundled libx264 build, so shots encoded here
    as mpeg4 still end up as playable H.264 in the delivered video.
    """
    if _ffmpeg_has(binary, "encoder", "libx264"):
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    return ["-c:v", "mpeg4", "-q:v", "4"]


def _subtitle_cfg(config: dict[str, Any] | None) -> dict[str, Any]:
    timeline_cfg = (config or {}).get("timeline", {}) if isinstance((config or {}).get("timeline"), dict) else {}
    cfg = timeline_cfg.get("subtitles", {}) if isinstance(timeline_cfg.get("subtitles"), dict) else {}
    return cfg


def _use_0907_subtitles(config: dict[str, Any] | None) -> bool:
    config = config or {}
    if str(config.get("policy_version", "") or "").strip() == "0907":
        return True
    newvis = config.get("newvis0907")
    return isinstance(newvis, dict) and bool(newvis.get("enabled"))


def _split_subtitle_clauses(text: str) -> list[str]:
    text = " ".join(str(text or "").split())
    text = re.sub(r"\s*([，。！？；：、])\s*", r"\1", text)
    if not text:
        return []
    clauses: list[str] = []
    current = ""
    for ch in text:
        current += ch
        if ch in "，。！？；：\n" and current.strip():
            clauses.append(current.strip())
            current = ""
    if current.strip():
        clauses.append(current.strip())
    return clauses or [text]


def _pack_subtitle_chunks(clauses: list[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        while len(clause) > max_chars:
            head = clause[:max_chars].strip()
            if head:
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.append(head)
            clause = clause[max_chars:].strip()
        if not clause:
            continue
        if current and len(current) + len(clause) > max_chars:
            chunks.append(current.strip())
            current = clause
        else:
            current += clause
    if current.strip():
        chunks.append(current.strip())
    return chunks


def build_subtitle_track(
    vocal_script: str,
    duration: float,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Split vocal script into readable subtitle blocks aligned to TTS duration."""
    if not vocal_script:
        return []
    cfg = _subtitle_cfg(config)
    if _use_0907_subtitles(config):
        max_chars_per_line = max(8, int(cfg.get("max_chars_per_line", 14) or 14))
        max_lines = max(1, int(cfg.get("max_lines", 2) or 2))
        return build_subtitle_cues(
            vocal_script,
            duration,
            max_chars_per_line=max_chars_per_line,
            max_lines=max_lines,
        )["cues"]
    max_chars_per_line = max(8, int(cfg.get("max_chars_per_line", 18) or 18))
    max_lines = max(1, int(cfg.get("max_lines", 2) or 2))
    max_chars = max(10, max_chars_per_line * max_lines)
    min_segment_duration = max(0.2, float(cfg.get("min_segment_duration_sec", 0.45) or 0.45))

    chunks = _pack_subtitle_chunks(_split_subtitle_clauses(vocal_script), max_chars)
    if not chunks:
        return [{"start": 0.0, "end": max(0.0, duration), "text": str(vocal_script)}]

    total_weight = max(1, sum(max(1, len(re.sub(r"\s+", "", chunk))) for chunk in chunks))
    available = max(float(duration or 0.0), min_segment_duration if chunks else 0.0)
    subtitles = []
    cursor = 0.0
    for i, text in enumerate(chunks):
        if i == len(chunks) - 1:
            end = available
        else:
            weight = max(1, len(re.sub(r"\s+", "", text)))
            seg_duration = max(0.05, available * weight / total_weight)
            end = min(available, cursor + seg_duration)
        subtitles.append({
            "start": round(cursor, 2),
            "end": round(max(end, cursor + 0.2), 2),
            "text": text,
        })
        cursor = subtitles[-1]["end"]
    return subtitles


def _render_shot_ffmpeg(
    clip_path: str,
    audio_path: str,
    subtitles: list[dict[str, Any]],
    output_path: str,
    duration: float,
    timeout_sec: int = 300,
    decoration: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    allow_stream_loop: bool = True,
) -> dict[str, Any]:
    """Render one shot by compositing video + audio plus lightweight visible text.

    ``allow_stream_loop=False`` (the 0907 path) forbids the ``-stream_loop``
    fallback: the visual has already been planned to exactly cover ``duration``
    via slow motion, so looping/holding the last frame is never acceptable.
    """
    clip_exists = clip_path and Path(clip_path).exists() and Path(clip_path).stat().st_size > 0
    audio_exists = audio_path and Path(audio_path).exists() and Path(audio_path).stat().st_size > 0

    if not clip_exists:
        return {"status": "no_clip", "output_path": ""}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    text_binary = _text_ffmpeg_binary()
    video_args = _text_video_encoder_args(text_binary)
    cmd = [text_binary, "-y"]

    # Loop short retrieval clips so the visual stream can never end before TTS.
    # 0907 disables this: the playlist render already spans the full duration.
    if allow_stream_loop:
        cmd += ["-stream_loop", "-1"]
    cmd += ["-i", clip_path]
    text_filters = _build_text_filters(subtitles, decoration or {}, config=config)

    if audio_exists:
        cmd += ["-i", audio_path]
        cmd += [
            "-map", "0:v:0",
            "-map", "1:a:0",
        ]
        if text_filters:
            cmd += ["-vf", ",".join(text_filters)]
        cmd += [
            *video_args,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-t", str(duration),
            "-movflags", "+faststart",
            output_path,
        ]
    else:
        cmd += [
            "-map", "0:v:0",
        ]
        if text_filters:
            cmd += ["-vf", ",".join(text_filters)]
        cmd += [
            *video_args,
            "-pix_fmt", "yuv420p",
            "-an",
            "-t", str(duration),
            "-movflags", "+faststart",
            output_path,
        ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=_media_subprocess_env(),
        )
        if proc.returncode == 0 and Path(output_path).exists():
            return {
                "status": "rendered",
                "output_path": output_path,
                "audio_policy": "tts_only" if audio_exists else "silent",
                "original_audio_removed": True,
                "text_overlay_rendered": bool(text_filters),
            }
        return {"status": "ffmpeg_error", "output_path": "", "stderr": proc.stderr[-500:]}
    except subprocess.TimeoutExpired:
        return {"status": "render_timeout", "output_path": ""}
    except FileNotFoundError:
        return {"status": "ffmpeg_not_found", "output_path": ""}


def _build_visual_playlist(
    *,
    primary_clip_path: str,
    clip_meta: dict[str, Any] | None,
    candidates: list[dict[str, Any]] | None,
    target_duration: float,
) -> dict[str, Any]:
    """Plan retrieved visual clips in order until they cover the TTS duration."""
    clip_meta = clip_meta or {}
    target = max(0.0, _as_float(target_duration, 0.0))
    raw_items: list[dict[str, Any]] = []
    primary_id = str(clip_meta.get("clip_id") or primary_clip_path or "primary")
    raw_items.append({
        "clip_id": primary_id,
        "clip_path": primary_clip_path,
        "duration_sec": _as_float(clip_meta.get("duration_sec", clip_meta.get("duration")), 0.0),
        "score": _as_float(clip_meta.get("match_score", clip_meta.get("score")), 0.0),
    })
    raw_items.extend(candidates or [])

    available: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_items:
        path = str(raw.get("clip_path") or "").strip()
        if not path:
            continue
        key = str(raw.get("clip_id") or path)
        if key in seen:
            continue
        seen.add(key)
        duration = _as_float(raw.get("duration_sec", raw.get("duration")), 0.0)
        if duration <= 0:
            duration = _probe_media_duration(path)
        if duration <= 0:
            continue
        available.append({
            **raw,
            "clip_id": key,
            "clip_path": path,
            "duration_sec": round(duration, 3),
        })

    clips: list[dict[str, Any]] = []
    source_total = 0.0
    idx = 0
    max_segments = max(1, min(24, len(available) * 3 if available else 1))
    while available and source_total < target and len(clips) < max_segments:
        item = available[idx % len(available)]
        source_duration = _as_float(item.get("duration_sec"), 0.0)
        remaining = max(0.0, target - source_total)
        used_duration = source_duration if remaining <= 0 else min(source_duration, remaining)
        clips.append({
            "clip_id": item.get("clip_id", ""),
            "clip_path": item.get("clip_path", ""),
            "duration_sec": round(source_duration, 3),
            "used_duration_sec": round(used_duration, 3),
            "offset_sec": round(source_total, 3),
            "score": item.get("score", item.get("match_score", 0.0)),
        })
        source_total += source_duration
        idx += 1

    if not clips and primary_clip_path:
        clips.append({
            "clip_id": primary_id,
            "clip_path": primary_clip_path,
            "duration_sec": 0.0,
            "used_duration_sec": round(target, 3),
            "offset_sec": 0.0,
            "score": 0.0,
        })

    return {
        "clips": clips,
        "source_duration_sec": round(source_total, 3),
        "render_duration_sec": round(target, 3),
        "uses_multiple_clips": len({item.get("clip_path") for item in clips}) > 1,
        "coverage_status": "covered" if source_total >= target or target <= 0 else "short",
    }


def _prepare_visual_playlist_for_render(
    *,
    primary_clip_path: str,
    clip_meta: dict[str, Any],
    candidates: list[dict[str, Any]],
    shot_dir: Path,
    duration: float,
    timeout_sec: int = 240,
) -> dict[str, Any]:
    existing_candidates: list[dict[str, Any]] = []
    for raw in candidates:
        path = Path(str(raw.get("clip_path") or "")).expanduser()
        if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
            continue
        existing_candidates.append({
            **raw,
            "clip_path": str(path),
            "duration_sec": _probe_media_duration(path),
        })

    playlist = _build_visual_playlist(
        primary_clip_path=primary_clip_path,
        clip_meta=clip_meta,
        candidates=existing_candidates,
        target_duration=duration,
    )
    unique_paths = [item["clip_path"] for item in playlist["clips"] if item.get("clip_path")]
    if len(dict.fromkeys(unique_paths)) <= 1:
        return {"status": "single_clip", "playlist": playlist, "clip_path": primary_clip_path}

    shot_dir.mkdir(parents=True, exist_ok=True)
    output_path = shot_dir / "source_1920x1080_playlist.mp4"
    vf = (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
    )
    cmd = [_ffmpeg_binary(), "-y"]
    render_clips = playlist["clips"]
    for item in render_clips:
        cmd += ["-i", str(item["clip_path"])]
    filters = []
    labels = []
    for idx, item in enumerate(render_clips):
        used = max(0.05, _as_float(item.get("used_duration_sec"), item.get("duration_sec", 0.0)))
        label = f"v{idx}"
        filters.append(
            f"[{idx}:v:0]{vf},fps=24,trim=duration={used:.3f},setpts=PTS-STARTPTS[{label}]"
        )
        labels.append(f"[{label}]")
    filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0[vout]")
    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[vout]",
        "-an",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-t", str(duration),
        str(output_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=_media_subprocess_env(),
        )
        if proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return {
                "status": "playlist_rendered",
                "playlist": playlist,
                "clip_path": str(output_path),
                "duration_sec": round(_probe_media_duration(output_path), 3),
            }
        return {
            "status": "playlist_render_failed",
            "playlist": playlist,
            "clip_path": primary_clip_path,
            "stderr": (proc.stderr or "")[-500:],
        }
    except subprocess.TimeoutExpired:
        return {"status": "playlist_render_timeout", "playlist": playlist, "clip_path": primary_clip_path}
    except FileNotFoundError:
        return {"status": "ffmpeg_not_found", "playlist": playlist, "clip_path": primary_clip_path}


def _preserved_visual_playlist_0907(
    clip_meta: dict[str, Any],
    candidates: list[dict[str, Any]],
    duration: float,
    shot_id: str,
) -> dict[str, Any] | None:
    """Keep an explicitly edited playlist authoritative during re-render."""
    raw_playlist = clip_meta.get("visual_playlist")
    if not isinstance(raw_playlist, dict) or not isinstance(raw_playlist.get("clips"), list):
        return None
    raw_clips = [dict(c) for c in raw_playlist["clips"] if isinstance(c, dict)]
    if not raw_clips or not any(c.get("user_override") or c.get("replaced") for c in raw_clips):
        return None

    used_paths = {
        str(c.get("clip_path") or c.get("path") or "")
        for c in raw_clips if c.get("clip_path") or c.get("path")
    }
    available = [dict(c) for c in candidates if isinstance(c, dict)]
    clips: list[dict[str, Any]] = []
    cursor = 0.0
    for idx, clip in enumerate(raw_clips, start=1):
        path = str(clip.get("clip_path") or clip.get("path") or "").strip()
        if clip.get("needs_retrieval") or not _existing_file(path):
            query = str(clip.get("query") or "").strip().casefold()
            replacement = next(
                (
                    c for c in available
                    if _existing_file(str(c.get("clip_path") or c.get("path") or ""))
                    and str(c.get("clip_path") or c.get("path") or "") not in used_paths
                    and (not query or query in str(c.get("query") or c.get("visual_query") or "").casefold())
                ),
                None,
            )
            if replacement is None:
                return None
            path = str(replacement.get("clip_path") or replacement.get("path") or "")
            used_paths.add(path)
            clip.update({
                "clip_id": replacement.get("clip_id") or path,
                "clip_path": path,
                "path": path,
                "fingerprint": replacement.get("fingerprint", ""),
                "score": replacement.get("score", 0.0),
                "needs_retrieval": False,
            })

        final = _as_float(
            clip.get("final_duration_sec", clip.get("used_duration_sec", clip.get("duration_sec"))),
            0.0,
        )
        source_used = _as_float(
            clip.get("source_used_duration_sec", clip.get("source_duration_sec", final)),
            final,
        )
        rate = _as_float(clip.get("playback_rate"), source_used / final if final > 0 else 1.0)
        if not _existing_file(path) or final < 2.0 - 1e-6 or rate < (1.0 / 3.0) - 1e-6:
            return None
        clip.update({
            "subclip_id": clip.get("subclip_id") or f"{shot_id}::subclip::{idx}",
            "clip_id": clip.get("clip_id") or path,
            "clip_path": path,
            "path": path,
            "source_used_duration_sec": round(source_used, 4),
            "final_duration_sec": round(final, 4),
            "playback_rate": round(rate, 6),
            "offset_sec": round(cursor, 4),
            "start_sec": round(cursor, 4),
            "end_sec": round(cursor + final, 4),
        })
        clips.append(clip)
        cursor += final

    if abs(cursor - duration) > 0.05:
        return None
    return {
        "clips": clips,
        "target_duration_sec": round(duration, 4),
        "covered_duration_sec": round(cursor, 4),
        "coverage_status": "covered",
        "deficit_sec": 0.0,
        "needs_retrieval": False,
        "needs_split": False,
        "uses_multiple_clips": len(clips) > 1,
        "quality_warning": False,
        "warnings": [],
    }


def _prepare_visual_playlist_0907(
    *,
    primary_clip_path: str,
    clip_meta: dict[str, Any],
    candidates: list[dict[str, Any]],
    shot_dir: Path,
    duration: float,
    config: dict[str, Any],
    shot_id: str,
    shot_query: str,
) -> dict[str, Any]:
    """0907 opt-in visual playlist: pacing-planner + slow-motion render.

    Unlike the legacy path this never loops/holds a frame; it slows sources with
    ``setpts`` (down to the 1/3 floor) and returns ``coverage_status`` honestly
    so ``run`` can fail closed instead of shipping a static/looped shot.
    """
    occupancy = None
    runtime = config.get("newvis0907_runtime") if isinstance(config, dict) else None
    if isinstance(runtime, dict):
        occupancy = runtime.get("occupancy")

    normalized_query = " ".join(str(shot_query or clip_meta.get("query") or "").split())
    primary: dict[str, Any] | None = None
    if primary_clip_path and _existing_file(primary_clip_path):
        pdur = _as_float(clip_meta.get("duration_sec", clip_meta.get("duration")), 0.0)
        if pdur <= 0:
            pdur = _probe_media_duration(primary_clip_path)
        primary = {
            "clip_id": clip_meta.get("clip_id") or primary_clip_path,
            "clip_path": primary_clip_path,
            "source_duration_sec": pdur,
            "score": clip_meta.get("match_score", clip_meta.get("score", 0.0)),
            "query": normalized_query or clip_meta.get("query", ""),
            "fingerprint": clip_meta.get("fingerprint", ""),
        }

    src_candidates: list[dict[str, Any]] = []
    for raw in candidates or []:
        path = str(raw.get("clip_path") or raw.get("path") or "").strip()
        if not _existing_file(path):
            continue
        raw_query = " ".join(str(raw.get("query") or raw.get("visual_query") or "").split())
        if normalized_query and raw_query and raw_query.casefold() != normalized_query.casefold():
            # Keep one-query-per-model-shot contract for 0907 visual units.
            continue
        dur = _as_float(raw.get("source_duration_sec", raw.get("duration_sec")), 0.0)
        if dur <= 0:
            dur = _probe_media_duration(path)
        src_candidates.append({
            **raw,
            "clip_path": path,
            "source_duration_sec": dur,
            "query": normalized_query or raw_query,
        })

    playlist = _preserved_visual_playlist_0907(
        clip_meta, src_candidates, duration, shot_id,
    )
    if playlist is None:
        playlist = pacing_mod.plan_visual_playlist(
            primary,
            src_candidates,
            duration,
            occupied=occupancy,
            shot_id=shot_id,
            prefer_retrieval_order=True,
        )
    coverage = playlist.get("coverage_status", "short")
    if coverage in ("short", "invalid"):
        # Fail closed *before* spending any ffmpeg render: a short/invalid
        # playlist must never be padded, frozen or looped, so we skip the render
        # subprocess entirely and report the deficit for Stage4 to act on.
        return {
            "status": "coverage_short",
            "playlist": playlist,
            "clip_path": "",
            "coverage_status": coverage,
            "deficit_sec": playlist.get("deficit_sec", 0.0),
            "needs_retrieval": playlist.get("needs_retrieval", False),
            "needs_split": playlist.get("needs_split", False),
        }
    render = _render_visual_playlist_0907(
        playlist=playlist, shot_dir=shot_dir, duration=duration,
    )
    return {
        "status": render.get("status", "unknown"),
        "playlist": playlist,
        "clip_path": render.get("clip_path", ""),
        "coverage_status": coverage,
        "deficit_sec": playlist.get("deficit_sec", 0.0),
        "needs_retrieval": playlist.get("needs_retrieval", False),
        "needs_split": playlist.get("needs_split", False),
    }


def _render_visual_playlist_0907(
    *,
    playlist: dict[str, Any],
    shot_dir: Path,
    duration: float,
    timeout_sec: int = 240,
) -> dict[str, Any]:
    """Render a 0907 playlist: trim each source by ``source_used_duration`` and
    ``setpts``-stretch to ``final_duration`` (slow motion), then concat. No
    ``stream_loop`` — every segment's length equals ``final_duration_sec``."""
    clips = [c for c in playlist.get("clips", []) if _existing_file(str(c.get("clip_path") or c.get("path") or ""))]
    if not clips:
        return {"status": "playlist_no_clips", "clip_path": ""}

    shot_dir.mkdir(parents=True, exist_ok=True)
    output_path = shot_dir / "source_1920x1080_playlist0907.mp4"
    vf = (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
    )
    cmd = [_ffmpeg_binary(), "-y"]
    for item in clips:
        cmd += ["-i", str(item.get("clip_path") or item.get("path"))]
    filters: list[str] = []
    labels: list[str] = []
    for idx, item in enumerate(clips):
        used = max(0.05, _as_float(item.get("source_used_duration_sec"),
                                   _as_float(item.get("used_duration_sec"), 0.0)))
        final = max(0.05, _as_float(item.get("final_duration_sec"), used))
        source_in = max(0.0, _as_float(item.get("source_in_sec"), 0.0))
        # setpts factor stretches the trimmed segment to its final duration:
        # factor = final/used = 1/playback_rate. factor==1 is natural speed.
        factor = final / used if used > 0 else 1.0
        label = f"v{idx}"
        filters.append(
            f"[{idx}:v:0]{vf},fps=24,"
            f"trim=start={source_in:.3f}:duration={used:.3f},"
            f"setpts=PTS-STARTPTS,setpts={factor:.6f}*PTS[{label}]"
        )
        labels.append(f"[{label}]")
    filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0[vout]")
    total = sum(max(0.05, _as_float(c.get("final_duration_sec"), 0.0)) for c in clips)
    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[vout]",
        "-an",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-t", f"{max(duration, total):.3f}",
        str(output_path),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_sec,
            env=_media_subprocess_env(),
        )
        if proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return {"status": "playlist0907_rendered", "clip_path": str(output_path)}
        return {"status": "playlist0907_render_failed", "clip_path": "",
                "stderr": (proc.stderr or "")[-500:]}
    except subprocess.TimeoutExpired:
        return {"status": "playlist0907_render_timeout", "clip_path": ""}
    except FileNotFoundError:
        return {"status": "ffmpeg_not_found", "clip_path": ""}


def run(
    shot: dict[str, Any],
    assets: dict[str, Any],
    output_dir: str,
    config: dict[str, Any] | None = None,
    render_backend: str | None = None,
) -> dict[str, Any]:
    """Compose timeline for one shot from its multi-modal assets."""
    config = config or {}
    shot_id = shot.get("shot_id", "unknown")
    audio_meta_src = assets.get("audio_meta", {}) if isinstance(assets.get("audio_meta"), dict) else {}
    vocal_script = str(
        shot.get("vocal_script")
        or shot.get("vocal_query")
        or shot.get("tts_text")
        or shot.get("audio_text")
        or shot.get("narration")
        or shot.get("script")
        or shot.get("voiceover")
        or shot.get("subtitle")
        or audio_meta_src.get("text")
        or audio_meta_src.get("tts_text")
        or ""
    )
    work_shot_dir = (Path(output_dir).expanduser().resolve() / "shots" / shot_id)

    audio_duration = _as_float(audio_meta_src.get("duration_sec", 0.0), 0.0)
    planned_duration = _as_float(shot.get("duration_sec", 8.0), 8.0)
    tail_padding = max(0.0, _as_float(_cfg_get(config, ("timeline", "audio_tail_padding_sec"), 0.15), 0.15))
    duration = max(1.0, audio_duration + tail_padding) if audio_duration > 0 else max(1.0, planned_duration)

    clip_meta = assets.get("clip_meta", {}) if isinstance(assets.get("clip_meta"), dict) else {}
    candidates = assets.get("retrieval_candidates", [])
    if not isinstance(candidates, list):
        candidates = []
    raw_clip_path = str(assets.get("clip_path", "") or "")
    if raw_clip_path and not _existing_file(raw_clip_path):
        logger.warning(
            "[TimelineCompose] shot %s: clip_path set but file missing: %s (exists=%s, size=%s)",
            shot_id, raw_clip_path,
            Path(raw_clip_path).exists() if raw_clip_path else False,
            Path(raw_clip_path).stat().st_size if raw_clip_path and Path(raw_clip_path).exists() else "N/A",
        )
    use_0907 = _use_0907_subtitles(config)
    allow_stream_loop = True
    coverage_short_0907 = False
    playlist_payload_0907: dict[str, Any] = {}
    if use_0907:
        # 0907 opt-in: pacing planner + slow-motion (setpts) render, never loop.
        visual_playlist = _prepare_visual_playlist_0907(
            primary_clip_path=raw_clip_path,
            clip_meta=clip_meta,
            candidates=candidates,
            shot_dir=work_shot_dir,
            duration=round(duration, 2),
            config=config,
            shot_id=str(shot_id),
            shot_query=str(shot.get("visual_query", "") or ""),
        )
        allow_stream_loop = False
        # Persist the planned playlist so hierarchy / UI can read every subclip.
        playlist_payload = visual_playlist.get("playlist", {})
        playlist_payload_0907 = playlist_payload
        if isinstance(assets, dict):
            base_meta = assets.get("clip_meta") if isinstance(assets.get("clip_meta"), dict) else {}
            merged_meta = {**base_meta, "visual_playlist": playlist_payload}
            assets["clip_meta"] = merged_meta
            assets["visual_playlist"] = playlist_payload
        coverage_short_0907 = visual_playlist.get("coverage_status") in ("short", "invalid")
        # When coverage is short we still build subtitles / timeline below, but
        # fail the render closed (no static/looped shot is ever shipped).
        source_clip_path = str(visual_playlist.get("clip_path") or "")
        if not coverage_short_0907 and not source_clip_path:
            source_clip_path = str(raw_clip_path or "")
        # The 0907 playlist render is already 1920x1080 and exactly ``duration``
        # long, so skip the stream_loop normalize path entirely.
        clip_path_for_render = source_clip_path
        clip_meta_for_render = {
            **clip_meta,
            "render_clip_path": clip_path_for_render,
            "render_target_resolution": f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
            "render_clip_status": "playlist0907",
            "visual_playlist_status": visual_playlist.get("status", "unknown"),
            "visual_playlist": playlist_payload,
        }
    else:
        visual_playlist = _prepare_visual_playlist_for_render(
            primary_clip_path=raw_clip_path,
            clip_meta=clip_meta,
            candidates=candidates,
            shot_dir=work_shot_dir,
            duration=round(duration, 2),
        )
        source_clip_path = str(visual_playlist.get("clip_path") or raw_clip_path or "")
        normalized_clip = _normalize_clip_for_render(
            clip_path=source_clip_path,
            shot_dir=work_shot_dir,
            duration=round(duration, 2),
        )
        clip_path_for_render = normalized_clip.get("clip_path", raw_clip_path)
        clip_meta_for_render = {
            **clip_meta,
            "render_clip_path": clip_path_for_render,
            "render_target_resolution": f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
            "render_clip_status": normalized_clip.get("status", "unknown"),
            "visual_playlist_status": visual_playlist.get("status", "unknown"),
            "visual_playlist": visual_playlist.get("playlist", {}),
        }
        if normalized_clip.get("normalized"):
            clip_meta_for_render["original_clip_path"] = str(assets.get("clip_path", "") or "")
            clip_meta_for_render["original_resolution"] = normalized_clip.get("original_resolution", "")
            clip_meta_for_render["normalized_resolution"] = f"{TARGET_WIDTH}x{TARGET_HEIGHT}"

    subtitle_duration = audio_duration if audio_duration > 0 else duration
    # `subtitle_text` is written by the editor's `set_subtitle`, which exists so
    # on-screen text can be fixed without paying for a TTS re-run. It must
    # therefore override the spoken script here — the narration audio keeps
    # using `vocal_script`.
    subtitle_source = str(shot.get("subtitle_text") or "").strip() or vocal_script
    if _use_0907_subtitles(config):
        max_chars_per_line = max(8, int(_subtitle_cfg(config).get("max_chars_per_line", 14) or 14))
        max_lines = max(1, int(_subtitle_cfg(config).get("max_lines", 2) or 2))
        subtitle_payload = build_subtitle_cues(
            subtitle_source,
            subtitle_duration,
            max_chars_per_line=max_chars_per_line,
            max_lines=max_lines,
        )
        subtitles = subtitle_payload["cues"]
        # 0907 treats subtitle punctuation/layout as a derived display artifact:
        # keep the original subtitle/narration source unchanged, but write back
        # the formatted display text and cue plan so hierarchy/UI/render all see
        # the same on-screen text without introducing subtitle drift.
        shot["subtitle_display_text"] = subtitle_payload["display_text"]
        shot["subtitle_cues"] = [
            {
                **cue,
                "lines": list(cue.get("lines", [])),
            }
            for cue in subtitles
        ]
    else:
        subtitles = build_subtitle_track(subtitle_source, subtitle_duration, config=config)

    # 0907 semantic animation: read the shot's primary anchor and keep the
    # decoration timing in sync with the spoken phrase interval (user overrides
    # win), then expose the whole semantic plan on the timeline tracks so the
    # hierarchy / renderer can drive per-phrase playback.
    semantic_track = _sync_semantic_decoration(shot, assets, duration)

    timeline = {
        "shot_id": shot_id,
        "duration_sec": round(duration, 2),
        "tracks": {
            "visual": {
                "clip_path": clip_path_for_render,
                "clip_meta": clip_meta_for_render,
                "start": 0.0,
                "end": duration,
            },
            "audio": {
                "audio_path": assets.get("audio_path", ""),
                "audio_meta": assets.get("audio_meta", {}),
                "start": 0.0,
                "end": audio_duration,
            },
            "subtitles": subtitles,
            "decoration": assets.get("decoration", {}),
            "semantic_animations": semantic_track,
        },
        "transition": {
            "type": "crossfade",
            "duration_ms": 500,
        },
    }

    out_path = work_shot_dir / "timeline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")

    # Attempt real render. Spotlight shots can use the HyperFrames accelerated
    # path; ordinary shots keep the lightweight ffmpeg path.
    rendered_path = str(work_shot_dir / "shot.mp4")
    if use_0907 and coverage_short_0907:
        # Fail closed BEFORE any ffmpeg render subprocess: an insufficient visual
        # playlist must never be padded/looped, and we must not waste a render.
        # Timeline + subtitles are already built above so Stage4 can still repair.
        render_result = {
            "status": "visual_coverage_short",
            "coverage_status": visual_playlist.get("coverage_status", "short"),
            "deficit_sec": visual_playlist.get("deficit_sec", 0.0),
            "needs_retrieval": visual_playlist.get("needs_retrieval", False),
            "needs_split": visual_playlist.get("needs_split", False),
            "visual_playlist": playlist_payload_0907,
            "output_path": "",
        }
        return {
            "shot_id": shot_id,
            "timeline": timeline,
            "timeline_path": str(out_path),
            "duration_sec": round(duration, 2),
            "has_decoration": bool(assets.get("decoration")),
            "subtitle_count": len(subtitles),
            "rendered_path": "",
            "render_status": "visual_coverage_short",
            "render_success": False,
            "render_backend": "none",
            "render_report": render_result,
            "audio_policy": "tts_only" if assets.get("audio_path") else "silent",
            "original_audio_removed": False,
            "text_overlay_rendered": False,
            "subtitle_overlay": {"status": "skipped", "applied": False},
            "compose_status": "success",
            "status": "render_incomplete",
        }
    selected_backend = _select_render_backend(shot, assets, config, render_backend)
    render_assets = dict(assets)
    render_assets["clip_path"] = clip_path_for_render
    render_assets["clip_meta"] = clip_meta_for_render
    if selected_backend == "hyperframes":
        render_result = _render_shot_hyperframes(
            shot=shot,
            assets=render_assets,
            timeline=timeline,
            output_dir=output_dir,
            output_path=rendered_path,
            duration=round(duration, 2),
            config=config,
        )
        render_success = render_result.get("status") == "hyperframes_rendered"
        if not render_success:
            # Keep spotlight shots renderable when HyperFrames runtime is missing
            # or transiently fails; Stage4 can still judge/repair using ffmpeg render.
            fallback_result = _render_shot_ffmpeg(
                clip_path=clip_path_for_render,
                audio_path=assets.get("audio_path", ""),
                subtitles=subtitles,
                output_path=rendered_path,
                duration=round(duration, 2),
                decoration=assets.get("decoration", {}),
                config=config,
                allow_stream_loop=allow_stream_loop,
            )
            if fallback_result.get("status") == "rendered":
                render_result = {
                    **fallback_result,
                    "fallback_from": "hyperframes",
                    "hyperframes_error": render_result,
                    "render_backend": "ffmpeg",
                }
                render_success = True
                selected_backend = "ffmpeg"
    else:
        render_result = _render_shot_ffmpeg(
            clip_path=clip_path_for_render,
            audio_path=assets.get("audio_path", ""),
            subtitles=subtitles,
            output_path=rendered_path,
            duration=round(duration, 2),
            decoration=assets.get("decoration", {}),
            config=config,
            allow_stream_loop=allow_stream_loop,
        )
        render_success = render_result.get("status") == "rendered"

    rendered_duration = _probe_media_duration(render_result.get("output_path", rendered_path))
    duration_tolerance = max(
        0.05,
        _as_float(_cfg_get(config, ("timeline", "rendered_duration_tolerance_sec"), 0.35), 0.35),
    )
    minimum_required_duration = max(audio_duration, duration - tail_padding)
    if render_success and (
        rendered_duration <= 0
        or rendered_duration + duration_tolerance < minimum_required_duration
    ):
        render_result = {
            **render_result,
            "status": "rendered_duration_too_short",
            "output_path": "",
            "rendered_duration_sec": round(rendered_duration, 3),
            "required_duration_sec": round(minimum_required_duration, 3),
            "duration_tolerance_sec": round(duration_tolerance, 3),
        }
        render_success = False
    else:
        render_result["rendered_duration_sec"] = round(rendered_duration, 3)
        render_result["required_duration_sec"] = round(minimum_required_duration, 3)

    subtitle_overlay: dict[str, Any] = {"status": "skipped", "applied": False}
    needs_unified_subtitle_burn = (
        selected_backend == "hyperframes"
        or (not bool(render_result.get("text_overlay_rendered", False)) and bool(subtitles))
    )
    if render_success and needs_unified_subtitle_burn:
        subtitle_overlay = _burn_subtitles_on_rendered_shot(
            video_path=render_result.get("output_path", rendered_path),
            subtitles=subtitles,
            duration=round(duration, 2),
            config=config,
            allow_stream_loop=allow_stream_loop,
        )
        if subtitle_overlay.get("status") == "ok":
            render_result["output_path"] = subtitle_overlay.get("output_path", rendered_path)
            render_result["subtitle_overlay_applied"] = True
        elif selected_backend == "hyperframes" and subtitles:
            # HyperFrames path must have subtitle burn-in; fail closed so stage4 can repair.
            render_result = {
                **render_result,
                "status": "subtitle_overlay_failed",
                "output_path": "",
                "subtitle_overlay": subtitle_overlay,
            }
            render_success = False

    if render_success:
        final_rendered_duration = _probe_media_duration(render_result.get("output_path", rendered_path))
        if final_rendered_duration + duration_tolerance < minimum_required_duration:
            render_result = {
                **render_result,
                "status": "subtitle_burn_duration_too_short",
                "output_path": "",
                "rendered_duration_sec": round(final_rendered_duration, 3),
                "required_duration_sec": round(minimum_required_duration, 3),
            }
            render_success = False
        else:
            render_result["rendered_duration_sec"] = round(final_rendered_duration, 3)

    # 0907 fails closed on insufficient visual coverage: never ship a static or
    # looped shot. The render (if any) is discarded in favour of an explicit
    # coverage-short report so Stage4 retrieves more footage or splits the shot.
    if use_0907 and coverage_short_0907:
        render_success = False
        render_result = {
            **render_result,
            "status": "visual_coverage_short",
            "coverage_status": "short",
            "deficit_sec": visual_playlist.get("deficit_sec", 0.0),
            "needs_retrieval": visual_playlist.get("needs_retrieval", False),
            "needs_split": visual_playlist.get("needs_split", False),
            "visual_playlist": playlist_payload_0907,
            "output_path": "",
        }

    return {
        "shot_id": shot_id,
        "timeline": timeline,
        "timeline_path": str(out_path),
        "duration_sec": round(duration, 2),
        "has_decoration": bool(assets.get("decoration")),
        "subtitle_count": len(subtitles),
        "rendered_path": render_result.get("output_path", ""),
        "render_status": render_result.get("status", "not_attempted"),
        "render_success": render_success,
        "render_backend": selected_backend,
        "render_report": render_result,
        "audio_policy": render_result.get("audio_policy", "tts_only" if assets.get("audio_path") else "silent"),
        "original_audio_removed": bool(render_result.get("original_audio_removed", render_success)),
        "text_overlay_rendered": bool(render_result.get("text_overlay_rendered", False)),
        "subtitle_overlay": subtitle_overlay,
        "compose_status": "success",
        "status": "success" if render_success else "render_incomplete",
    }


def _sync_semantic_decoration(
    shot: dict[str, Any],
    assets: dict[str, Any],
    duration: float,
) -> dict[str, Any]:
    """Return the shot's semantic-animation plan, syncing decoration timing.

    The primary anchor's ``start_sec``/``end_sec`` (from the subtitle cues) drive
    ``decoration.timing`` so the ffmpeg ``drawtext`` overlay and HyperFrames
    assemble step both play the decoration during the spoken phrase — but a
    user-authored emphasis (``source == "user"``) keeps whatever timing the user
    settled on, so a manual edit is never silently retimed.
    """
    semantic = assets.get("semantic_animations")
    if not isinstance(semantic, dict) or not semantic.get("primary"):
        semantic = shot.get("semantic_animations")
    if not isinstance(semantic, dict):
        return {}
    primary = semantic.get("primary")
    if not isinstance(primary, dict) or not primary.get("phrase"):
        return semantic

    decoration = assets.get("decoration")
    if not isinstance(decoration, dict):
        return semantic
    is_user = str(primary.get("source") or "") == "user"
    start = primary.get("start_sec")
    end = primary.get("end_sec")
    if not is_user and start is not None and end is not None:
        try:
            start_f = max(0.0, float(start))
            end_f = float(end)
        except (TypeError, ValueError):
            return semantic
        if duration and duration > 0:
            start_f = min(start_f, duration)
            end_f = min(end_f, duration)
        if end_f > start_f:
            timing = dict(decoration.get("timing") or {})
            timing["decor_start_sec"] = round(start_f, 3)
            timing["decor_duration_sec"] = round(end_f - start_f, 3)
            decoration = {**decoration, "timing": timing}
            assets["decoration"] = decoration
    return semantic


def _select_render_backend(
    shot: dict[str, Any],
    assets: dict[str, Any],
    config: dict[str, Any],
    requested: str | None,
) -> str:
    if requested:
        return requested
    if shot.get("is_spotlight") and assets.get("decoration"):
        enabled = _cfg_bool(config, ("services", "hyperframes", "accelerated_render", "enabled"), False)
        if enabled:
            return "hyperframes"
    return "ffmpeg"


def _render_shot_hyperframes(
    *,
    shot: dict[str, Any],
    assets: dict[str, Any],
    timeline: dict[str, Any],
    output_dir: str,
    output_path: str,
    duration: float,
    config: dict[str, Any],
    assemble_only: bool = False,
) -> dict[str, Any]:
    """Render one spotlight shot through Gold_Template + HyperFrames CLI.

    ``assemble_only=True`` stops right after the project directory is built
    (index.html + footage + fonts + slots.json) and never launches Chrome.
    That is the cheap half: assembling is a few hundred ms, while rendering
    every frame of the animation is tens of seconds per shot. The VLM judge
    only needs a handful of frames per shot, so it assembles here and then
    calls ``hyperframes snapshot --at t1,t2,...`` on the returned ``work_dir``
    to render just those timestamps.
    """
    clip_path = str(assets.get("clip_path", "") or "")
    if not _existing_file(clip_path):
        return {"status": "no_clip", "output_path": "", "render_backend": "hyperframes"}

    hf_cfg = _hyperframes_cfg(config)
    hf_bin, runtime_report = _resolve_hyperframes_binary(config)
    if not hf_bin:
        return {
            "status": "hyperframes_runtime_error",
            "output_path": "",
            "render_backend": "hyperframes",
            "runtime": runtime_report,
        }
    gold_root = Path(
        str((config.get("paths", {}) or {}).get("gold_template_root", "") or "")
    ).expanduser().resolve()
    assembler = gold_root / "assembler" / "assemble.py"
    if not assembler.exists():
        return {
            "status": "hyperframes_assembler_missing",
            "output_path": "",
            "render_backend": "hyperframes",
            "assembler": str(assembler),
        }

    shot_id = str(shot.get("shot_id", "unknown"))
    work_dir = (Path(output_dir).expanduser().resolve() / "shots" / shot_id / "hyperframes")
    output_path = str(Path(output_path).expanduser().resolve())
    footage_dir = work_dir / "footage"
    audio_dir = work_dir / "audio"
    work_dir.mkdir(parents=True, exist_ok=True)
    footage_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    _ensure_project_assets(gold_root, work_dir)

    clip_dest = footage_dir / f"source{Path(clip_path).suffix or '.mp4'}"
    _link_or_copy(Path(clip_path), clip_dest)

    audio_path = str(assets.get("audio_path", "") or "")
    audio_dest = None
    if _existing_file(audio_path):
        audio_dest = audio_dir / f"narration{Path(audio_path).suffix or '.wav'}"
        _link_or_copy(Path(audio_path), audio_dest)

    decoration = assets.get("decoration", {}) if isinstance(assets.get("decoration"), dict) else {}
    structure_id = _resolve_structure_id(decoration, gold_root)
    motion_id = _resolve_motion_id(decoration, gold_root)
    visual_id = _resolve_visual_id(decoration, gold_root, config)
    slots = decoration.get("slots", {}) if isinstance(decoration.get("slots"), dict) else {}
    if not slots:
        slots = _default_slots_from_shot(shot)
    slots_path = work_dir / "slots.json"
    slots_path.write_text(json.dumps(slots, ensure_ascii=False, indent=2), encoding="utf-8")

    timing = decoration.get("timing", {}) if isinstance(decoration.get("timing"), dict) else {}
    decor_start = _as_float(timing.get("decor_start_sec"), 0.3)
    decor_duration = _as_float(timing.get("decor_duration_sec"), min(4.0, duration))
    clip_meta = assets.get("clip_meta", {}) if isinstance(assets.get("clip_meta"), dict) else {}
    bg_clip_len = _probe_media_duration(clip_path)
    if bg_clip_len <= 0:
        bg_clip_len = _as_float(clip_meta.get("duration", clip_meta.get("duration_sec")), duration)

    assemble_cmd = [
        sys.executable,
        str(assembler),
        "--structure",
        structure_id,
        "--motion",
        motion_id,
        "--visual",
        visual_id,
        "--slots",
        str(slots_path.resolve()),
        "--duration",
        str(duration),
        "--bg-video",
        f"footage/{clip_dest.name}",
        "--bg-clip-len",
        str(bg_clip_len),
        "--decor-start",
        str(max(0.0, min(decor_start, duration))),
        "--decor-duration",
        str(max(0.1, min(decor_duration, duration))),
        "--out",
        str((work_dir / "index.html").resolve()),
    ]
    assembled = _run_command(assemble_cmd, cwd=str(gold_root), timeout_sec=120, env=os.environ.copy())
    if assembled.get("returncode") != 0:
        result = {
            "status": "hyperframes_assemble_error",
            "output_path": "",
            "render_backend": "hyperframes",
            "work_dir": str(work_dir),
            "cmd": assemble_cmd,
            "stderr": assembled.get("stderr", ""),
        }
        _write_hyperframes_report(work_dir, result)
        return result

    _inline_font_family_variables(work_dir / "index.html")
    _localize_gsap_cdn(work_dir / "index.html", config)
    font_report = _prepare_composition_fonts(
        index_path=work_dir / "index.html",
        gold_root=gold_root,
        work_dir=work_dir,
    )

    if audio_dest is not None:
        _inject_narration_audio(
            work_dir / "index.html",
            f"audio/{audio_dest.name}",
            min(duration, _as_float(assets.get("audio_meta", {}).get("duration_sec"), duration)),
        )

    env = _build_hyperframes_env(config)
    lint_report: dict[str, Any] | None = None

    if assemble_only:
        # 到这里工程目录已经可以被 `hyperframes snapshot` 直接吃了。
        # 故意跳过 lint: lint 会再起一次 CLI (Node 冷启动数秒), 而 config.yaml 里
        # force_hyperframes_lint 本来就是 false; 抽帧失败会在 snapshot 那层降级。
        result = {
            "status": "hyperframes_assembled",
            "output_path": "",
            "render_backend": "hyperframes",
            "work_dir": str(work_dir),
            "structure_id": structure_id,
            "motion_id": motion_id,
            "visual_id": visual_id,
            "duration_sec": duration,
            "bg_clip_len": bg_clip_len,
            "fonts": font_report,
            "runtime": runtime_report,
        }
        _write_hyperframes_report(work_dir, result)
        return result

    if _cfg_bool(config, ("stage3_trajectory", "spotlight_judge", "force_hyperframes_lint"), True):
        lint_cmd = _hf_command(hf_bin, ["lint", str(work_dir.resolve())])
        lint_report = _run_command(
            lint_cmd,
            cwd=str(work_dir),
            timeout_sec=max(
                120,
                _cfg_int(config, ("services", "hyperframes", "check_timeout"), 30000) // 1000,
            ),
            env=env,
        )
        if lint_report.get("returncode") != 0:
            result = {
                "status": "hyperframes_lint_error",
                "output_path": "",
                "render_backend": "hyperframes",
                "work_dir": str(work_dir),
                "lint": lint_report,
                "runtime": runtime_report,
                "fonts": font_report,
            }
            _write_hyperframes_report(work_dir, result)
            return result

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    local_output_root = Path(
        str(hf_cfg.get("render_cache_dir") or "/tmp/aicut_hyperframes_outputs")
    ).expanduser()
    output_key = hashlib.sha1(output_path.encode("utf-8")).hexdigest()[:16]
    local_output_dir = local_output_root / output_key
    local_output_dir.mkdir(parents=True, exist_ok=True)
    local_output_path = local_output_dir / "shot.mp4"
    if local_output_path.exists():
        local_output_path.unlink()
    render_cmd = _hf_command(hf_bin, [
        "render",
        str(work_dir.resolve()),
        "--fps",
        str(_cfg_int(config, ("services", "hyperframes", "accelerated_render", "fps"), 24)),
        "--quality",
        str(_cfg_get(config, ("services", "hyperframes", "accelerated_render", "quality"), "draft")),
        "--workers",
        str(_cfg_get(config, ("services", "hyperframes", "accelerated_render", "workers"), "auto")),
        "--format",
        str(_cfg_get(config, ("services", "hyperframes", "accelerated_render", "format"), "mp4")),
        "--output",
        str(local_output_path),
    ])
    extra_args = _cfg_get(config, ("services", "hyperframes", "accelerated_render", "extra_args"), [])
    if isinstance(extra_args, str):
        render_cmd.extend(extra_args.split())
    elif isinstance(extra_args, list):
        render_cmd.extend(str(part) for part in extra_args)

    render_report = _run_command(
        render_cmd,
        cwd=str(work_dir),
        timeout_sec=_cfg_int(
            config,
            ("services", "hyperframes", "accelerated_render", "timeout_seconds"),
            _cfg_int(config, ("services", "hyperframes", "timeout"), 600),
        ),
        env=env,
    )
    if render_report.get("returncode") == 0 and _existing_file(str(local_output_path)):
        shutil.copy2(local_output_path, output_path)
        try:
            local_output_path.unlink()
        except OSError:
            pass
        result = {
            "status": "hyperframes_rendered",
            "output_path": output_path,
            "render_backend": "hyperframes",
            "work_dir": str(work_dir),
            "structure_id": structure_id,
            "motion_id": motion_id,
            "visual_id": visual_id,
            "lint": lint_report,
            "render": render_report,
            "runtime": runtime_report,
            "fonts": font_report,
            "audio_policy": "tts_only" if audio_dest else "silent",
            "original_audio_removed": True,
            "text_overlay_rendered": True,
        }
        _write_hyperframes_report(work_dir, result)
        return result
    result = {
        "status": "hyperframes_render_error",
        "output_path": "",
        "render_backend": "hyperframes",
        "work_dir": str(work_dir),
        "structure_id": structure_id,
        "motion_id": motion_id,
        "visual_id": visual_id,
        "lint": lint_report,
        "render": render_report,
        "runtime": runtime_report,
        "fonts": font_report,
    }
    _write_hyperframes_report(work_dir, result)
    return result


def _write_hyperframes_report(work_dir: Path, report: dict[str, Any]) -> None:
    """Persist full CLI stdout/stderr beside the generated composition."""
    try:
        (work_dir / "render_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Failed to persist HyperFrames render report: %s", exc)


def _hyperframes_cfg(config: dict[str, Any]) -> dict[str, Any]:
    services = config.get("services", {}) if isinstance(config.get("services"), dict) else {}
    return services.get("hyperframes", {}) if isinstance(services.get("hyperframes"), dict) else {}


def _hf_command(hf_bin: str, args: list[str]) -> list[str]:
    parts = hf_bin.split() if " " in hf_bin.strip() else [hf_bin]
    return parts + args


def _resolve_hyperframes_binary(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Use a local-disk HyperFrames runtime instead of loading Node modules from Ceph.

    HyperFrames imports many small dependency files. Running it directly from a
    network filesystem can make even ``--version`` take minutes. The runtime is
    installed once under /tmp and reused by every spotlight render.
    """
    hf_cfg = _hyperframes_cfg(config)
    configured_bin = str(hf_cfg.get("hf_bin") or "").strip() or "npx hyperframes"
    runtime_dir_raw = str(hf_cfg.get("runtime_cache_dir") or "").strip()
    if not runtime_dir_raw:
        return configured_bin, {"status": "configured_binary", "binary": configured_bin}

    runtime_dir = Path(runtime_dir_raw).expanduser()
    cached_bin = runtime_dir / "node_modules" / ".bin" / "hyperframes"
    package_version = str(hf_cfg.get("package_version") or "0.6.112").strip()
    node_bin_dir = Path(str(hf_cfg.get("node_bin_dir") or "").strip()).expanduser()
    node_bin = node_bin_dir / "node"
    npm_bin = node_bin_dir / "npm"

    if not node_bin.exists() or not npm_bin.exists():
        # Fall back to configured_bin instead of failing outright.
        # Verify configured_bin itself is reachable; if also missing, return empty
        # so callers properly handle it as hyperframes_runtime_error rather than
        # getting a confusing FileNotFoundError from subprocess.
        configured_path = Path(configured_bin.split()[0]) if " " in configured_bin else Path(configured_bin)
        configured_exists = configured_bin.startswith("npx") or configured_path.exists()
        if not configured_exists:
            return "", {
                "status": "node_runtime_missing_no_fallback",
                "binary": configured_bin,
                "node_bin_dir": str(node_bin_dir),
                "reason": "both node runtime and configured_bin are missing",
            }
        return configured_bin, {
            "status": "node_runtime_missing_fallback",
            "binary": configured_bin,
            "node_bin_dir": str(node_bin_dir),
        }

    try:
        version_proc = subprocess.run(
            [str(node_bin), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        node_version = version_proc.stdout.strip()
        major = int(node_version.lstrip("v").split(".", 1)[0])
    except (OSError, ValueError, subprocess.TimeoutExpired):
        node_version = ""
        major = 0
    if major < 22:
        # Fall back to configured_bin — the wrapper script may use a
        # different node runtime from the one specified in node_bin_dir.
        return configured_bin, {
            "status": "node_version_unsupported_fallback",
            "binary": configured_bin,
            "required": "Node.js >=22",
            "actual": node_version or "unknown",
            "node": str(node_bin),
        }

    if cached_bin.exists():
        return str(cached_bin), {
            "status": "ready",
            "binary": str(cached_bin),
            "node_version": node_version,
        }

    with _HYPERFRAMES_RUNTIME_LOCK:
        if cached_bin.exists():
            return str(cached_bin), {
                "status": "ready",
                "binary": str(cached_bin),
                "node_version": node_version,
            }

        # Try extracting from a pre-built tar.gz archive before npm install.
        archive_path_raw = str(hf_cfg.get("runtime_archive") or "").strip()
        if archive_path_raw:
            archive_path = Path(archive_path_raw).expanduser()
            if archive_path.exists():
                try:
                    import tarfile
                    if runtime_dir.exists():
                        shutil.rmtree(runtime_dir, ignore_errors=True)
                    runtime_dir.mkdir(parents=True, exist_ok=True)
                    with tarfile.open(str(archive_path), "r:gz") as tf:
                        tf.extractall(str(runtime_dir.parent))
                    extracted_name = archive_path.stem.replace(".tar", "")
                    extracted_dir = runtime_dir.parent / extracted_name
                    if extracted_dir != runtime_dir and extracted_dir.exists():
                        if runtime_dir.exists():
                            shutil.rmtree(runtime_dir, ignore_errors=True)
                        extracted_dir.rename(runtime_dir)
                    if cached_bin.exists():
                        logger.info("[HyperFrames] extracted runtime from %s", archive_path)
                        return str(cached_bin), {
                            "status": "extracted_from_archive",
                            "binary": str(cached_bin),
                            "archive": str(archive_path),
                            "node_version": node_version,
                        }
                except Exception as exc:
                    logger.warning("[HyperFrames] archive extraction failed: %s", exc)

        runtime_dir.mkdir(parents=True, exist_ok=True)
        env = _media_subprocess_env()
        env["PATH"] = f"{node_bin_dir}:{env.get('PATH', '')}"
        install_cmd = [
            str(npm_bin),
            "install",
            "--prefix",
            str(runtime_dir),
            "--omit=dev",
            "--no-audit",
            "--no-fund",
            f"hyperframes@{package_version}",
        ]
        try:
            proc = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                timeout=max(30, int(hf_cfg.get("runtime_install_timeout_sec", 180))),
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            return configured_bin, {
                "status": "runtime_install_timeout_fallback",
                "binary": configured_bin,
                "stderr": str(exc.stderr or "")[-500:],
            }
        if proc.returncode != 0 or not cached_bin.exists():
            return configured_bin, {
                "status": "runtime_install_error_fallback",
                "binary": configured_bin,
                "returncode": proc.returncode,
                "stderr": (proc.stderr or "")[-500:],
            }
        return str(cached_bin), {
            "status": "installed",
            "binary": str(cached_bin),
            "node_version": node_version,
            "package_version": package_version,
        }


_GSAP_CDN_PATTERN = re.compile(
    r'<script\s+src="https://cdn\.jsdelivr\.net/npm/gsap@\d+(?:\.\d+)*/dist/gsap\.min\.js"\s*>\s*</script>',
    re.IGNORECASE,
)


def _localize_gsap_cdn(index_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Replace the GSAP CDN <script> tag with a locally vendored copy."""
    if not index_path.exists():
        return {"status": "skipped", "reason": "index.html missing"}
    html = index_path.read_text(encoding="utf-8")
    match = _GSAP_CDN_PATTERN.search(html)
    if not match:
        return {"status": "skipped", "reason": "no gsap cdn tag found"}

    local_gsap = str(
        _cfg_get(config, ("services", "hyperframes", "gsap_local_path"), "")
        or "/mnt/shang2cephfs/mm-base-vision-ascend/chenyumu/DATA/Online_test_hyperframes/my-video6/vendor/gsap.min.js"
    )
    if not Path(local_gsap).exists():
        return {"status": "skipped", "reason": f"local gsap not found: {local_gsap}"}

    vendor_dir = index_path.parent / "vendor"
    vendor_dir.mkdir(parents=True, exist_ok=True)
    dest = vendor_dir / "gsap.min.js"
    shutil.copy2(local_gsap, dest)

    replacement = '<script src="vendor/gsap.min.js"></script>'
    html = html[: match.start()] + replacement + html[match.end() :]
    index_path.write_text(html, encoding="utf-8")
    return {"status": "localized", "source": local_gsap, "dest": str(dest)}


def _build_hyperframes_env(config: dict[str, Any]) -> dict[str, str]:
    hf_cfg = _hyperframes_cfg(config)
    acc = hf_cfg.get("accelerated_render", {}) if isinstance(hf_cfg.get("accelerated_render"), dict) else {}
    env = _media_subprocess_env()
    env["CI"] = "1"
    env["HYPERFRAMES_NO_TELEMETRY"] = "1"
    env["HYPERFRAMES_NO_UPDATE_CHECK"] = "1"
    env["HYPERFRAMES_SKIP_LINT"] = "0"
    env["HYPERFRAMES_FFMPEG_PATH"] = str(
        hf_cfg.get("ffmpeg_path") or _ffmpeg_binary()
    )
    env["HYPERFRAMES_FFPROBE_PATH"] = str(
        hf_cfg.get("ffprobe_path") or shutil.which("ffprobe") or _ffmpeg_binary()
    )

    node_bin_dir = str(hf_cfg.get("node_bin_dir") or "").strip()
    if node_bin_dir and Path(node_bin_dir).exists():
        path_parts = [part for part in env.get("PATH", "").split(":") if part]
        if node_bin_dir not in path_parts:
            env["PATH"] = ":".join([node_bin_dir, *path_parts])

    # Prefer the wrapper script (which adds --no-sandbox, --enable-gpu,
    # --run-all-compositor-stages-before-draw, etc.) over the raw cached
    # binary, so Chrome receives essential launch flags in this container.
    wrapper_path = str(acc.get("browser_path") or "").strip()
    cached_path = _resolve_local_browser_binary(config)
    if wrapper_path and Path(wrapper_path).exists():
        browser_path = wrapper_path
    elif cached_path and Path(cached_path).exists():
        browser_path = cached_path
    else:
        browser_path = str(hf_cfg.get("browser_path") or "").strip()
    if browser_path and Path(browser_path).exists():
        env["HYPERFRAMES_BROWSER_PATH"] = browser_path
        env["PRODUCER_HEADLESS_SHELL_PATH"] = browser_path
        env["PUPPETEER_EXECUTABLE_PATH"] = browser_path

    if acc.get("enable_browser_pool") is not None:
        env["PRODUCER_ENABLE_BROWSER_POOL"] = "true" if acc.get("enable_browser_pool") else "false"

    # NOTE: proxy_url is deliberately NOT injected into the HyperFrames
    # subprocess env.  Setting all_proxy/http_proxy would cause Chrome's
    # internal DevTools Protocol websocket connections (ws://127.0.0.1:…)
    # to be routed through the corporate proxy, breaking rendering.

    nav_timeout_ms = acc.get("nav_timeout_ms")
    if nav_timeout_ms is not None:
        env["HYPERFRAMES_NAV_TIMEOUT_MS"] = str(int(nav_timeout_ms))

    # ── HyperFrames manifest resolution ──────────────────────────────
    # When the CLI binary sits under node_modules/.bin/ (either via
    # runtime_cache_dir or the configured hf_bin), it cannot find
    # hyperframe.manifest.json because its import.meta.url-based path
    # arithmetic resolves to the wrong directory.  We pre-resolve the
    # manifest ourselves and inject it via PRODUCER_HYPERFRAME_MANIFEST_PATH.
    manifest_candidates = []
    runtime_cache = str(hf_cfg.get("runtime_cache_dir") or "").strip()
    if runtime_cache:
        manifest_candidates.append(
            Path(runtime_cache).expanduser() / "node_modules" / "hyperframes" / "dist" / "hyperframe.manifest.json"
        )
    configured_bin = str(hf_cfg.get("hf_bin") or "").strip()
    if configured_bin:
        bin_path = Path(configured_bin).expanduser()
        # configured_bin lives at <prefix>/node_modules/.bin/hyperframes
        manifest_candidates.append(
            bin_path.parent.parent / "hyperframes" / "dist" / "hyperframe.manifest.json"
        )
    for candidate in manifest_candidates:
        if candidate.exists():
            env["PRODUCER_HYPERFRAME_MANIFEST_PATH"] = str(candidate)
            break

    return env


def _resolve_local_browser_binary(config: dict[str, Any]) -> str:
    """Cache Chrome on local disk so each worker does not launch it from Ceph."""
    hf_cfg = _hyperframes_cfg(config)
    acc = hf_cfg.get("accelerated_render", {}) if isinstance(hf_cfg.get("accelerated_render"), dict) else {}
    source_raw = str(acc.get("browser_source_path") or "").strip()
    cache_raw = str(acc.get("browser_cache_dir") or "").strip()
    configured = str(acc.get("browser_path") or hf_cfg.get("browser_path") or "").strip()
    if not source_raw or not cache_raw:
        return configured

    source = Path(source_raw).expanduser()
    cache_dir = Path(cache_raw).expanduser()
    cached = cache_dir / source.name
    if cached.exists() and cached.stat().st_size > 0:
        return str(cached)
    if not source.exists():
        return configured

    with _HYPERFRAMES_RUNTIME_LOCK:
        if cached.exists() and cached.stat().st_size > 0:
            return str(cached)
        try:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            shutil.copytree(source.parent, cache_dir)
        except OSError as exc:
            logger.warning("HyperFrames browser local cache failed: %s", exc)
            return configured
    return str(cached) if cached.exists() else configured


def _run_command(
    cmd: list[str],
    *,
    cwd: str,
    timeout_sec: int,
    env: dict[str, str],
) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "cmd": cmd,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "cmd": cmd,
            "timeout_sec": timeout_sec,
            "error": "timeout",
        }
    except FileNotFoundError as exc:
        return {"returncode": 127, "stdout": "", "stderr": str(exc), "cmd": cmd, "error": "not_found"}


def _ensure_project_assets(gold_root: Path, work_dir: Path) -> None:
    (work_dir / "hyperframes.json").write_text(
        (gold_root / "hyperframes.json").read_text(encoding="utf-8")
        if (gold_root / "hyperframes.json").exists()
        else json.dumps({"paths": {"blocks": "compositions", "assets": "assets"}}, indent=2),
        encoding="utf-8",
    )
    sfx_src = gold_root / "viewer" / "sfx"
    sfx_dst = work_dir / "sfx"
    if sfx_src.exists() and not sfx_dst.exists():
        _link_or_copy(sfx_src, sfx_dst)


def _link_or_copy(src: Path, dst: Path) -> None:
    src = src.expanduser().resolve()
    if dst.is_symlink():
        try:
            if dst.resolve(strict=True) == src:
                return
        except FileNotFoundError:
            pass
        dst.unlink()
    elif dst.exists():
        try:
            if dst.samefile(src):
                return
        except OSError:
            pass
        if src.is_file() and dst.is_file():
            shutil.copy2(src, dst)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(str(src), str(dst), target_is_directory=src.is_dir())
        return
    except OSError:
        pass
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def _inject_narration_audio(index_path: Path, src: str, duration: float) -> None:
    html = index_path.read_text(encoding="utf-8")
    tag = (
        f'      <audio id="narration" data-start="0" data-duration="{duration}" '
        f'data-track-index="80" src="{src}" data-volume="1"></audio>\n'
    )
    if 'id="narration"' in html:
        return
    marker = "    <!-- ===== Layer 2:"
    marker_idx = html.find(marker)
    search_area = html[:marker_idx] if marker_idx >= 0 else html
    insert_pos = search_area.rfind("    </div>")
    if insert_pos < 0:
        insert_pos = html.rfind("</body>")
    if insert_pos < 0:
        return
    html = html[:insert_pos] + tag + html[insert_pos:]
    index_path.write_text(html, encoding="utf-8")


def _inline_font_family_variables(index_path: Path) -> None:
    """Resolve CSS font-family custom properties so HyperFrames lint sees real families."""
    html = index_path.read_text(encoding="utf-8")
    variables = {
        match.group("name"): match.group("value").strip()
        for match in re.finditer(
            r"--(?P<name>font-(?:display|body|caption))\s*:\s*(?P<value>[^;]+);",
            html,
        )
    }
    for name, value in variables.items():
        html = re.sub(
            rf"(font-family\s*:\s*)var\(--{re.escape(name)}\)",
            lambda match: f"{match.group(1)}{value}",
            html,
        )
    index_path.write_text(html, encoding="utf-8")


def _prepare_composition_fonts(
    *,
    index_path: Path,
    gold_root: Path,
    work_dir: Path,
) -> dict[str, Any]:
    """Subset only the two CJK font weights used by the current composition."""
    html = index_path.read_text(encoding="utf-8")
    fonts_dir = work_dir / "fonts"
    if fonts_dir.is_symlink():
        fonts_dir.unlink()
    fonts_dir.mkdir(parents=True, exist_ok=True)

    source_fonts = [
        ("NotoSansSC-Black.ttf", "display.ttf", "900"),
        ("NotoSansSC-Medium.ttf", "body.ttf", "500"),
    ]
    text = "".join(sorted(set(html + "".join(chr(code) for code in range(32, 127)))))
    subset_used = False
    outputs: list[dict[str, Any]] = []
    try:
        from fontTools import subset  # type: ignore
    except ImportError:
        subset = None

    for source_name, output_name, weight in source_fonts:
        src = _cache_font_source(gold_root / "fonts" / source_name)
        dst = fonts_dir / output_name
        if not src.exists():
            continue
        if subset is not None:
            options = subset.Options()
            options.layout_features = ["*"]
            options.name_IDs = ["*"]
            options.name_legacy = True
            options.name_languages = ["*"]
            font = subset.load_font(str(src), options)
            subsetter = subset.Subsetter(options=options)
            subsetter.populate(text=text)
            subsetter.subset(font)
            subset.save_font(font, str(dst), options)
            subset_used = True
        else:
            shutil.copy2(src, dst)
        outputs.append({
            "weight": weight,
            "path": str(dst),
            "bytes": dst.stat().st_size if dst.exists() else 0,
        })

    html = re.sub(r"@font-face\s*\{[^}]*\}", "", html, flags=re.IGNORECASE)
    font_css = (
        '@font-face{font-family:"Noto Sans SC";font-weight:900;font-style:normal;'
        'font-display:block;src:url("fonts/display.ttf") format("truetype");}\n'
        '@font-face{font-family:"Noto Sans SC";font-weight:500;font-style:normal;'
        'font-display:block;src:url("fonts/body.ttf") format("truetype");}\n'
    )
    html = html.replace("<style>", f"<style>\n      {font_css}", 1)
    index_path.write_text(html, encoding="utf-8")
    return {
        "status": "subset" if subset_used else "reduced_copy",
        "font_count": len(outputs),
        "fonts": outputs,
        "total_bytes": sum(int(item["bytes"]) for item in outputs),
    }


def _cache_font_source(source: Path) -> Path:
    """Copy large source fonts to local disk once before per-shot subsetting."""
    if not source.exists():
        return source
    cached = _FONT_SOURCE_CACHE_DIR / source.name
    if cached.exists() and cached.stat().st_size == source.stat().st_size:
        return cached
    with _HYPERFRAMES_RUNTIME_LOCK:
        if cached.exists() and cached.stat().st_size == source.stat().st_size:
            return cached
        _FONT_SOURCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, cached)
    return cached


def _resolve_structure_id(decoration: dict[str, Any], gold_root: Path) -> str:
    requested = str(decoration.get("template_id") or decoration.get("structure_id") or "").strip()
    if requested and (gold_root / "generated" / "structures" / requested / "structure.html").exists():
        return requested
    for candidate in (
        "key_takeaway",
        "definition_card",
        "thesis_statement",
        "question_hook",
        "timeline_step",
    ):
        if (gold_root / "generated" / "structures" / candidate / "structure.html").exists():
            return candidate
    return requested or "key_takeaway"


def _resolve_motion_id(decoration: dict[str, Any], gold_root: Path) -> str:
    # Honour an explicitly edited motion (``swap_template`` writes ``motion_id``)
    # the same way ``_resolve_visual_id`` honours ``visual_id``. Without this the
    # editor's "换动画" is silently ignored because the old code only inferred
    # motion from the decoration's ``animation.enter``.
    requested = str(decoration.get("motion_id") or "").strip()
    if requested and (gold_root / "generated" / "motions" / requested / "motion.js").exists():
        return requested

    animation = decoration.get("animation", {}) if isinstance(decoration.get("animation"), dict) else {}
    enter = str(animation.get("enter") or "").lower()
    candidates = []
    if "slide" in enter:
        candidates.append("slide_in_hold")
    if "mask" in enter or "reveal" in enter or "wipe" in enter:
        candidates.append("mask_reveal")
    if "stack" in enter or "card" in enter:
        candidates.append("card_stack_sequence")
    candidates.extend(["scale_pop_hold", "slide_in_hold", "mask_reveal"])
    for candidate in candidates:
        if (gold_root / "generated" / "motions" / candidate / "motion.js").exists():
            return candidate
    return "scale_pop_hold"


def _resolve_visual_id(decoration: dict[str, Any], gold_root: Path, config: dict[str, Any]) -> str:
    requested = str(
        decoration.get("visual_id")
        or _cfg_get(config, ("stage3_trajectory", "spotlight_judge", "visual_id"), "")
        or ""
    ).strip()
    if requested and (gold_root / "generated" / "visuals" / requested / "tokens.css").exists():
        return requested
    for candidate in ("dark_tech", "clean_saas", "documentary_caption", "warm_editorial"):
        if (gold_root / "generated" / "visuals" / candidate / "tokens.css").exists():
            return candidate
    return requested or "dark_tech"


def _default_slots_from_shot(shot: dict[str, Any]) -> dict[str, str]:
    text = str(shot.get("vocal_script") or shot.get("chapter_title") or "").strip()
    title = _clip_text(text.split("，")[0].split("。")[0], 22) if text else str(shot.get("chapter_title", ""))[:22]
    return {
        "headline": title or "重点镜头",
        "title": title or "重点镜头",
        "subtitle": _clip_text(text, 30),
        "eyebrow": str(shot.get("chapter_title") or "SPOTLIGHT")[:14],
        "takeaway1": _clip_text(text, 18),
        "takeaway2": str(shot.get("visual_query") or "")[:18],
        "takeaway3": str(shot.get("chapter_title") or "")[:18],
    }


def _build_text_filters(
    subtitles: list[dict[str, Any]],
    decoration: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> list[str]:
    filters: list[str] = []
    filters.extend(_build_subtitle_filters(subtitles, config=config))
    if not _cfg_bool(config or {}, ("timeline", "decorations", "burn_in_text_overlay"), False):
        return filters
    decor_lines = _decoration_lines(decoration)
    if decor_lines:
        timing = decoration.get("timing", {}) if isinstance(decoration.get("timing"), dict) else {}
        start = _as_float(timing.get("decor_start_sec"), 0.3)
        end = start + max(1.0, _as_float(timing.get("decor_duration_sec"), 3.8))
        x, y0 = _decoration_position(decoration)
        for idx, line in enumerate(decor_lines[:4]):
            filters.append(_drawtext_filter(
                text=line,
                x=x,
                y=f"{y0}+{idx}*54",
                fontsize=34 if idx else 40,
                boxcolor="black@0.62",
                start=start,
                end=end,
            ))
    return filters


def _subtitle_fontfile(config: dict[str, Any] | None) -> str:
    cfg = _subtitle_cfg(config)
    configured = str(cfg.get("fontfile") or cfg.get("font_file") or "").strip()
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    gold_root = Path(str(((config or {}).get("paths", {}) or {}).get("gold_template_root", "") or "")).expanduser()
    if gold_root:
        candidates.extend([
            gold_root / "fonts" / "NotoSerifSC-Bold.otf",
            gold_root / "fonts" / "NotoSansSC-Bold.ttf",
            gold_root / "fonts" / "NotoSansSC-Medium.ttf",
        ])
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return str(candidate)
    return ""


def _wrap_subtitle_lines(text: str, *, max_chars_per_line: int, max_lines: int) -> list[str]:
    compact = re.sub(r"\s+", "", str(text or "").strip())
    if not compact:
        return []
    lines: list[str] = []
    cursor = 0
    while cursor < len(compact) and len(lines) < max_lines:
        lines.append(compact[cursor:cursor + max_chars_per_line])
        cursor += max_chars_per_line
    return [line for line in lines if line]


def _build_subtitle_filters(
    subtitles: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> list[str]:
    filters: list[str] = []
    cfg = _subtitle_cfg(config)
    font_size = int(cfg.get("font_size", 56) or 56)
    line_spacing = int(cfg.get("line_spacing", max(62, font_size + 10)) or max(62, font_size + 10))
    bottom_margin = int(cfg.get("bottom_margin", 132) or 132)
    boxcolor = str(cfg.get("boxcolor", "black@0.52") or "black@0.52")
    max_chars_per_line = max(8, int(cfg.get("max_chars_per_line", 18) or 18))
    max_lines = max(1, int(cfg.get("max_lines", 2) or 2))
    max_filters = max(1, int(cfg.get("max_filters", 240) or 240))
    fontfile = _subtitle_fontfile(config)
    for subtitle in subtitles[:max_filters]:
        text = str(subtitle.get("text", "")).strip()
        if not text:
            continue
        start = _as_float(subtitle.get("start"), 0.0)
        end = max(start + 0.2, _as_float(subtitle.get("end"), start + 2.0))
        lines = _wrap_subtitle_lines(text, max_chars_per_line=max_chars_per_line, max_lines=max_lines)
        line_count = len(lines)
        y0 = f"h-{bottom_margin + max(0, line_count - 1) * line_spacing}"
        for line_idx, line in enumerate(lines):
            filters.append(_drawtext_filter(
                text=line,
                x="(w-text_w)/2",
                y=f"{y0}+{line_idx}*{line_spacing}",
                fontsize=font_size,
                boxcolor=boxcolor,
                start=start,
                end=end,
                fontfile=fontfile,
                boxborderw=int(cfg.get("boxborderw", 18) or 18),
            ))
    return filters


def _burn_subtitles_on_rendered_shot(
    video_path: str,
    subtitles: list[dict[str, Any]],
    duration: float,
    timeout_sec: int = 180,
    config: dict[str, Any] | None = None,
    allow_stream_loop: bool = True,
) -> dict[str, Any]:
    """Burn subtitles on rendered shot to keep all backends consistent."""
    if not subtitles:
        return {"status": "no_subtitles", "applied": False, "output_path": video_path}
    src = Path(str(video_path or "")).expanduser()
    if not src.exists() or src.stat().st_size <= 0:
        return {"status": "missing_rendered_video", "applied": False, "output_path": ""}

    subtitle_filters = _build_subtitle_filters(subtitles, config=config)
    if not subtitle_filters:
        return {"status": "empty_subtitle_filters", "applied": False, "output_path": str(src)}

    tmp_path = src.with_suffix(".subtitle_tmp.mp4")
    has_audio = _video_has_audio_stream(src)
    text_binary = _text_ffmpeg_binary()
    cmd = [text_binary, "-y"]
    if allow_stream_loop:
        cmd += ["-stream_loop", "-1"]
    cmd += [
        "-i",
        str(src),
        "-vf",
        ",".join(subtitle_filters),
        "-map",
        "0:v:0",
        *_text_video_encoder_args(text_binary),
        "-pix_fmt",
        "yuv420p",
        "-t",
        str(duration),
        "-movflags",
        "+faststart",
    ]
    if has_audio:
        cmd += ["-map", "0:a:0", "-c:a", "copy"]
    else:
        cmd += ["-an"]
    cmd += [str(tmp_path)]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=_media_subprocess_env(),
        )
        if proc.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 0:
            tmp_path.replace(src)
            return {"status": "ok", "applied": True, "output_path": str(src)}
        return {
            "status": "ffmpeg_error",
            "applied": False,
            "output_path": str(src),
            "stderr": (proc.stderr or "")[-500:],
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "applied": False, "output_path": str(src)}
    except FileNotFoundError:
        return {"status": "ffmpeg_not_found", "applied": False, "output_path": str(src)}
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _normalize_clip_for_render(
    clip_path: str,
    shot_dir: Path,
    duration: float,
    timeout_sec: int = 180,
) -> dict[str, Any]:
    """Normalize retrieval clip to 1920x1080 to stabilize HyperFrames rendering."""
    raw_path = str(clip_path or "").strip()
    normalized = shot_dir / "source_1920x1080.mp4"
    if not raw_path:
        if _existing_file(str(normalized)):
            return {
                "status": "reused_cached_normalized",
                "normalized": True,
                "clip_path": str(normalized),
                "duration_sec": round(_probe_media_duration(normalized), 3),
            }
        return {"status": "missing_clip", "normalized": False, "clip_path": ""}
    src = Path(raw_path).expanduser()
    if not src.exists() or not src.is_file() or src.stat().st_size <= 0:
        if _existing_file(str(normalized)):
            return {
                "status": "reused_cached_normalized",
                "normalized": True,
                "clip_path": str(normalized),
                "duration_sec": round(_probe_media_duration(normalized), 3),
            }
        return {"status": "missing_clip", "normalized": False, "clip_path": clip_path}

    probed = _probe_video_resolution(src)
    source_duration = _probe_media_duration(src)
    duration_tolerance = 0.1
    if (
        probed.get("width") == TARGET_WIDTH
        and probed.get("height") == TARGET_HEIGHT
        and source_duration + duration_tolerance >= duration
    ):
        return {
            "status": "already_target",
            "normalized": False,
            "clip_path": str(src),
            "original_resolution": f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
            "duration_sec": round(source_duration, 3),
        }

    shot_dir.mkdir(parents=True, exist_ok=True)
    if src.resolve() == normalized.resolve():
        normalized = shot_dir / "source_1920x1080_looped.mp4"
    vf = (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
    )
    cmd = [
        _ffmpeg_binary(),
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        str(src),
        "-vf",
        vf,
        "-r",
        "24",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-t",
        str(duration),
        str(normalized),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=_media_subprocess_env(),
        )
        if proc.returncode == 0 and normalized.exists() and normalized.stat().st_size > 0:
            return {
                "status": "normalized",
                "normalized": True,
                "clip_path": str(normalized),
                "original_resolution": _format_resolution(probed),
                "original_duration_sec": round(source_duration, 3),
                "duration_sec": round(_probe_media_duration(normalized), 3),
            }
        return {
            "status": "normalize_failed",
            "normalized": False,
            "clip_path": str(src),
            "original_resolution": _format_resolution(probed),
            "stderr": (proc.stderr or "")[-500:],
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "normalize_timeout",
            "normalized": False,
            "clip_path": str(src),
            "original_resolution": _format_resolution(probed),
        }
    except FileNotFoundError:
        return {
            "status": "ffmpeg_not_found",
            "normalized": False,
            "clip_path": str(src),
            "original_resolution": _format_resolution(probed),
        }


def _probe_video_resolution(path: Path, timeout_sec: int = 20) -> dict[str, int]:
    """Parse width/height from ``ffmpeg -i`` stderr."""
    try:
        proc = subprocess.run(
            [_ffmpeg_binary(), "-i", str(path), "-hide_banner"],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=_media_subprocess_env(),
        )
        import re as _re
        m = _re.search(r"(\d{2,5})x(\d{2,5})", proc.stderr or "")
        if m:
            w, h = int(m.group(1)), int(m.group(2))
            if w > 0 and h > 0:
                return {"width": w, "height": h}
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return {}
    return {}


def _video_has_audio_stream(path: Path, timeout_sec: int = 20) -> bool:
    try:
        proc = subprocess.run(
            [_ffmpeg_binary(), "-i", str(path), "-hide_banner"],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=_media_subprocess_env(),
        )
        return bool(proc.stderr and "Audio:" in proc.stderr)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _probe_media_duration(path: str | Path, timeout_sec: int = 20) -> float:
    src = Path(str(path or "")).expanduser()
    if not src.exists() or src.stat().st_size <= 0:
        return 0.0
    try:
        proc = subprocess.run(
            [_ffmpeg_binary(), "-i", str(src), "-hide_banner"],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=_media_subprocess_env(),
        )
        import re as _re
        m = _re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr or "")
        if m:
            h, mi, s = float(m.group(1)), float(m.group(2)), float(m.group(3))
            return round(h * 3600 + mi * 60 + s, 2)
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return 0.0
    return 0.0


def _format_resolution(meta: dict[str, Any]) -> str:
    width = int(meta.get("width") or 0) if isinstance(meta, dict) else 0
    height = int(meta.get("height") or 0) if isinstance(meta, dict) else 0
    if width > 0 and height > 0:
        return f"{width}x{height}"
    return ""


def _decoration_lines(decoration: dict[str, Any]) -> list[str]:
    if not isinstance(decoration, dict) or not decoration:
        return []
    slots = decoration.get("slots", {}) if isinstance(decoration.get("slots"), dict) else {}
    raw_lines: list[str] = []
    headline = slots.get("headline") or slots.get("title") or slots.get("chapter_title")
    if headline:
        raw_lines.append(str(headline))
    for key, value in slots.items():
        if key in {"headline", "title", "chapter_title"}:
            continue
        raw_lines.extend(_slot_texts(value))
        if len(raw_lines) >= 4:
            break
    return [_clip_text(line, 24) for line in raw_lines if str(line).strip()]


def _slot_texts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("label") or item.get("title") or item.get("text") or item.get("id")
                if text:
                    out.append(str(text))
            elif item:
                out.append(str(item))
        return out
    if isinstance(value, dict):
        text = value.get("label") or value.get("title") or value.get("text") or value.get("id")
        return [str(text)] if text else []
    return []


def _decoration_position(decoration: dict[str, Any]) -> tuple[str, str]:
    overlay = decoration.get("overlay_config", {}) if isinstance(decoration.get("overlay_config"), dict) else {}
    position = str(overlay.get("position", "bottom-left")).lower()
    if "right" in position:
        x = "w-text_w-54"
    elif "center" in position:
        x = "(w-text_w)/2"
    else:
        x = "54"
    if "top" in position:
        y = "64"
    elif "center" in position:
        y = "(h-text_h)/2"
    else:
        y = "h-330"
    return x, y


def _drawtext_filter(
    *,
    text: str,
    x: str,
    y: str,
    fontsize: int,
    boxcolor: str,
    start: float,
    end: float,
    fontfile: str = "",
    boxborderw: int = 16,
) -> str:
    safe_text = _escape_drawtext(text)
    font_part = f"fontfile='{_escape_drawtext(fontfile)}':" if fontfile else "font='sans':"
    return (
        "drawtext="
        f"{font_part}"
        f"text='{safe_text}':"
        "fontcolor=white:"
        f"fontsize={fontsize}:"
        "box=1:"
        f"boxcolor={boxcolor}:"
        f"boxborderw={boxborderw}:"
        f"x={x}:"
        f"y={y}:"
        f"enable='between(t\\,{start:.2f}\\,{end:.2f})':"
        "expansion=none"
    )


def _escape_drawtext(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("%", "\\%")
        .replace(",", "\\,")
        .replace("\n", " ")
    )


def _clip_text(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max(1, max_chars - 1)] + "…"


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _cfg_get(config: dict[str, Any], path: tuple[str, ...], default: Any) -> Any:
    cur: Any = config
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _cfg_int(config: dict[str, Any], path: tuple[str, ...], default: int) -> int:
    try:
        return int(_cfg_get(config, path, default))
    except (TypeError, ValueError):
        return default


def _cfg_bool(config: dict[str, Any], path: tuple[str, ...], default: bool) -> bool:
    value = _cfg_get(config, path, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"1", "true", "yes", "on"}:
            return True
        if norm in {"0", "false", "no", "off", ""}:
            return False
    return default


def _existing_file(path: str) -> bool:
    return bool(path) and Path(path).exists() and Path(path).stat().st_size > 0
