"""Tool_StitchVideo: Concatenate all approved shot videos into final output.

Uses ffmpeg for real video concatenation when rendered shots exist,
otherwise creates a stitch manifest for downstream rendering.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _output_stamp() -> str:
    """Timestamp used to version each stitch output (local time, sortable)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _mirror_to_stable(src: str, stable_path: str) -> None:
    """Copy this run's output onto the fixed `final_video*.mp4` name.

    A copy rather than a symlink: `/api/media` resolves symlinks before its
    containment check, and some players/copy tools treat links inconsistently.
    Failures are non-fatal — the versioned file is the real deliverable.
    """
    if not src or not Path(src).exists() or str(src) == str(stable_path):
        return
    try:
        shutil.copy2(src, stable_path)
    except OSError as exc:
        logger.warning("[stitch] could not refresh %s: %s", stable_path, exc)


STATIC_FFMPEG_PATH = Path(
    "/usr/local/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)


def _ffmpeg_binary() -> str:
    return str(STATIC_FFMPEG_PATH) if STATIC_FFMPEG_PATH.exists() else "ffmpeg"

FFMPEG_COMPAT_LIB_DIR = Path("/usr/lib64/dop-fuse")


def _media_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    if FFMPEG_COMPAT_LIB_DIR.exists():
        current = env.get("LD_LIBRARY_PATH", "")
        parts = [part for part in current.split(":") if part]
        compat = str(FFMPEG_COMPAT_LIB_DIR)
        if compat not in parts:
            env["LD_LIBRARY_PATH"] = ":".join([compat, *parts])
    return env


def _ffmpeg_concat(
    shot_paths: list[str],
    output_path: str,
    transition_ms: int = 500,
    timeout_sec: int = 600,
    fast_copy: bool = False,
) -> dict[str, Any]:
    """Concatenate multiple video files via ffmpeg concat demuxer.

    Stream copy (``fast_copy``) is only attempted when every input already
    shares a browser-playable codec and identical geometry/frame rate; shot
    files can come from mixed render backends (HyperFrames/ffmpeg) with
    different fps and time_base, and ``-c copy`` then reports success while
    producing a file whose video track no browser can decode.  Whenever the
    inputs are not uniform, or the copied result fails verification, we fall
    back to the unified transcode which is slower but always playable.
    """
    existing: list[str] = []
    missing: list[str] = []
    for p in shot_paths:
        candidate = Path(p).expanduser()
        if candidate.exists() and candidate.stat().st_size > 0:
            existing.append(str(candidate.resolve()))
        else:
            missing.append(str(candidate))
    if missing:
        return {
            "status": "missing_rendered_shots",
            "output_path": "",
            "missing_paths": missing,
            "expected_count": len(shot_paths),
            "existing_count": len(existing),
        }
    if not existing:
        return {"status": "no_rendered_shots", "output_path": ""}

    concat_list = Path(output_path).parent / "concat_list.txt"
    concat_list.parent.mkdir(parents=True, exist_ok=True)
    with open(concat_list, "w", encoding="utf-8") as f:
        for p in existing:
            escaped = p.replace("'", r"'\''")
            f.write(f"file '{escaped}'\n")

    copy_safe = False
    copy_skip_reason = "" if fast_copy else "fast_copy_disabled"
    if fast_copy:
        copy_safe, copy_skip_reason = _concat_copy_is_safe(existing)
        if not copy_safe:
            logger.info(
                "[stitch] stream-copy concat disabled (%s); using unified transcode",
                copy_skip_reason,
            )
    if fast_copy and copy_safe:
        copy_cmd = [
            _ffmpeg_binary(), "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            "-movflags", "+faststart",
            output_path,
        ]
        try:
            copy_proc = subprocess.run(
                copy_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=_media_subprocess_env(),
            )
            if copy_proc.returncode == 0 and Path(output_path).exists():
                duration = _get_video_duration(output_path)
                out_params = _probe_video_params(output_path)
                out_codec = str(out_params.get("codec", ""))
                if duration > 0 and out_codec in _BROWSER_SAFE_VIDEO_CODECS:
                    return {
                        "status": "success",
                        "output_path": output_path,
                        "duration_sec": duration,
                        "concat_count": len(existing),
                        "concat_mode": "copy_fast",
                        "video_codec": out_codec,
                    }
                logger.warning(
                    "[stitch] stream-copy output rejected (duration=%.2f codec=%r); "
                    "falling back to unified transcode",
                    duration, out_codec,
                )
                copy_skip_reason = f"output_rejected:{out_codec or 'no_video'}"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            copy_skip_reason = "copy_failed"

    transcode_cmd = [
        _ffmpeg_binary(), "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-vsync", "cfr",
        "-r", "24",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high", "-level", "4.1",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-dn",
        "-movflags", "+faststart",
        output_path,
    ]

    try:
        transcode_proc = subprocess.run(
            transcode_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=_media_subprocess_env(),
        )
        if transcode_proc.returncode == 0 and Path(output_path).exists():
            duration = _get_video_duration(output_path)
            return {
                "status": "success",
                "output_path": output_path,
                "duration_sec": duration,
                "concat_count": len(existing),
                "concat_mode": "transcode_unified",
                "video_codec": "h264",
                "copy_skip_reason": copy_skip_reason,
            }
        return {
            "status": "ffmpeg_error",
            "output_path": "",
            "stderr": transcode_proc.stderr[-1200:] if transcode_proc.stderr else "",
        }
    except subprocess.TimeoutExpired:
        # Some environments still flush a usable output file before timeout.
        # If we can read a positive duration, treat it as recovered success and
        # let downstream duration validation decide whether it is complete.
        if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
            duration = _get_video_duration(output_path)
            if duration > 0:
                return {
                    "status": "success",
                    "output_path": output_path,
                    "duration_sec": duration,
                    "concat_count": len(existing),
                    "concat_mode": "timeout_recovered",
                    "timeout_recovered": True,
                }
        return {"status": "ffmpeg_timeout", "output_path": ""}
    except FileNotFoundError:
        return {"status": "ffmpeg_not_found", "output_path": ""}


def _get_video_duration(path: str) -> float:
    """Get video duration via ffmpeg -i (stderr parsing)."""
    try:
        proc = subprocess.run(
            [_ffmpeg_binary(), "-i", path, "-hide_banner"],
            capture_output=True,
            text=True,
            timeout=10,
            env=_media_subprocess_env(),
        )
        import re
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr or "")
        if m:
            h, mi, s = float(m.group(1)), float(m.group(2)), float(m.group(3))
            return round(h * 3600 + mi * 60 + s, 2)
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return 0.0


def _verify_has_video_stream(path: str) -> bool:
    """Check that the file contains at least one video stream via ffmpeg -i."""
    return bool(_probe_video_params(path).get("codec"))


# Codecs every mainstream browser can decode inside an MP4 container.
# `mpeg4` (MPEG-4 Part 2 / mp4v) is NOT among them: Chrome/Safari/Firefox will
# play the audio track and render a black frame, which is exactly the failure
# mode we must never ship.
_BROWSER_SAFE_VIDEO_CODECS = {"h264", "av1", "vp9", "vp8"}


def _probe_video_params(path: str) -> dict[str, Any]:
    """Extract codec / resolution / fps of the first video stream.

    Parses ``ffmpeg -i`` stderr because the bundled imageio distribution ships
    ffmpeg only (no ffprobe), and the host ffprobe is unusable on images that
    isolate the OpenSSL 1.0 compat libraries.
    """
    import re

    try:
        proc = subprocess.run(
            [_ffmpeg_binary(), "-i", path, "-hide_banner"],
            capture_output=True,
            text=True,
            timeout=15,
            env=_media_subprocess_env(),
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}

    line = ""
    for candidate in (proc.stderr or "").splitlines():
        if "Video:" in candidate:
            line = candidate
            break
    if not line:
        return {}

    out: dict[str, Any] = {}
    m = re.search(r"Video:\s*([A-Za-z0-9_]+)", line)
    if m:
        out["codec"] = m.group(1).lower()
    m = re.search(r"(?<![\d.])(\d{2,5})x(\d{2,5})(?![\d.])", line)
    if m:
        out["width"] = int(m.group(1))
        out["height"] = int(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", line)
    if m:
        # Round to 2 decimals so 23.98 vs 23.976 do not look different.
        out["fps"] = round(float(m.group(1)), 2)
    return out


def _concat_copy_is_safe(paths: list[str]) -> tuple[bool, str]:
    """Decide whether ``-c copy`` concat can be trusted for these inputs.

    Stream copy only rewrites the container, so every input must already share
    a browser-playable codec and identical geometry/frame rate.  Mixing e.g.
    14fps and 24fps clips yields a file with broken timestamps that reports a
    nonsense average frame rate and refuses to decode in browsers.
    """
    if not paths:
        return False, "no_inputs"

    reference: dict[str, Any] | None = None
    for path in paths:
        params = _probe_video_params(path)
        codec = params.get("codec")
        if not codec:
            return False, f"probe_failed:{Path(path).name}"
        if codec not in _BROWSER_SAFE_VIDEO_CODECS:
            return False, f"codec_not_browser_safe:{codec}"
        signature = (params.get("codec"), params.get("width"), params.get("height"), params.get("fps"))
        if reference is None:
            reference = {"path": path, "signature": signature}
        elif signature != reference["signature"]:
            return False, (
                f"mismatch:{reference['signature']} vs {signature}"
            )
    return True, "uniform"


def _sanitize_shot_for_stitch(
    shot_id: str,
    rendered_path: str,
    audio_path: str,
    duration_sec: float,
    output_dir: Path,
    timeout_sec: int = 240,
) -> dict[str, Any]:
    """Normalize each shot to a stable A/V format before final concat.

    This strips potentially corrupted internal audio from rendered shots and
    remuxes with trusted narration wav (if present).

    Every output gets exactly one video + one audio stream with identical
    parameters (h264 / 24fps / yuv420p / aac 48kHz stereo).  Uniformity is not
    cosmetic: the rendered shots arrive from different backends (ffmpeg shots at
    25fps with 24kHz mono narration, HyperFrames shots at 14-24fps with 48kHz
    stereo), and the concat demuxer cannot reconcile that — it emits
    non-monotonic DTS, then feeds NaN samples to the AAC encoder which aborts
    the whole concat.  Shots without narration get silence rather than no audio
    track at all, because concat also requires an identical stream layout.
    """
    src = Path(rendered_path).expanduser()
    if not src.exists() or src.stat().st_size <= 0:
        return {"status": "missing_rendered", "output_path": "", "shot_id": shot_id}

    norm_dir = output_dir / "_stitch_normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)
    out_path = norm_dir / f"{shot_id}.mp4"

    cmd = [_ffmpeg_binary(), "-y", "-i", str(src)]
    has_audio = bool(audio_path) and Path(audio_path).expanduser().exists() and Path(audio_path).expanduser().stat().st_size > 0
    if has_audio:
        cmd += ["-i", str(Path(audio_path).expanduser())]
        cmd += ["-map", "0:v:0", "-map", "1:a:0"]
    else:
        # Silent filler track so this shot still has the v+a layout concat needs.
        cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-shortest"]

    # 用源文件真实 duration 做 -t 截断，避免 timeline 理论 duration 与实际渲染
    # 时长不一致导致时间戳断裂、concat 静默停止的问题。
    real_duration = _get_video_duration(str(src))
    if real_duration <= 0:
        real_duration = float(duration_sec or 0.0)
    truncate_sec = max(0.1, real_duration)

    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", "24",
        "-profile:v", "high", "-level", "4.1",
    ]
    # Always AAC 48k stereo — including the silent filler — so every normalized
    # shot is byte-compatible for stream-copy concat.
    cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]

    cmd += ["-t", str(truncate_sec), "-movflags", "+faststart", str(out_path)]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=_media_subprocess_env(),
        )
        if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            return {"status": "success", "output_path": str(out_path), "shot_id": shot_id, "has_audio": has_audio}
        return {"status": "ffmpeg_error", "output_path": "", "shot_id": shot_id, "stderr": (proc.stderr or "")[-800:]}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "output_path": "", "shot_id": shot_id}
    except FileNotFoundError:
        return {"status": "ffmpeg_not_found", "output_path": "", "shot_id": shot_id}


def _fast_bgm_remux(
    new_video_path: str,
    old_bgm_video_path: str,
    video_duration: float,
    timeout_sec: int = 60,
    duration_tolerance: float = 0.35,
) -> dict[str, Any]:
    """Replace only the video stream in the old BGM-mixed file with the new one.

    Uses ``-map 0:v -map 1:a -c copy`` so nothing is re-encoded — typically
    completes in seconds even for long videos. The result is written to a temp
    file and then atomically replaces the old BGM video.

    ⚠️ The reused audio is `_mix_bgm`'s output, i.e. **narration mixed with
    BGM** — the narration cannot be separated back out.  So this is only safe
    when the narration did not change.  We detect that by comparing the old
    audio's length against the new video: a re-run TTS shifts every shot after
    the edited one, and reusing the old track then desynced the whole cut
    (observed: shot 1 TTS 20.96s -> 22.08s left 19 shots 1.1s out of sync while
    the tool still reported success).  On mismatch we refuse and let the caller
    fall back to a full re-mix.

    The tolerance absorbs AAC frame padding (measured 0.15-0.18s here) while
    staying far below any real narration change, which shifts things by a second
    or more.  Erring tight is deliberate: a false refusal only costs a full
    re-mix (~5s slower), a false accept ships a silently desynced cut.
    """
    import tempfile

    if not Path(old_bgm_video_path).exists():
        return {"status": "old_bgm_missing"}

    old_audio_duration = _audio_stream_duration(old_bgm_video_path)
    if old_audio_duration <= 0:
        return {"status": "old_bgm_audio_unreadable"}
    drift = abs(old_audio_duration - video_duration)
    if drift > duration_tolerance:
        # Narration (or shot count) changed — the old mix no longer lines up.
        return {
            "status": "old_bgm_duration_mismatch",
            "old_audio_duration_sec": round(old_audio_duration, 3),
            "new_video_duration_sec": round(video_duration, 3),
            "drift_sec": round(drift, 3),
        }

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=str(Path(old_bgm_video_path).parent))
    os.close(tmp_fd)
    cmd = [
        _ffmpeg_binary(), "-y",
        "-i", new_video_path,
        "-i", old_bgm_video_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c", "copy",
        "-t", str(video_duration),
        "-movflags", "+faststart",
        tmp_path,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_sec, env=_media_subprocess_env(),
        )
        if proc.returncode == 0 and Path(tmp_path).exists() and Path(tmp_path).stat().st_size > 0:
            shutil.move(tmp_path, old_bgm_video_path)
            return {"status": "bgm_remuxed", "output_path": old_bgm_video_path}
        Path(tmp_path).unlink(missing_ok=True)
        return {"status": "bgm_remux_error", "stderr": (proc.stderr or "")[-500:]}
    except subprocess.TimeoutExpired:
        Path(tmp_path).unlink(missing_ok=True)
        return {"status": "bgm_remux_timeout"}
    except FileNotFoundError:
        Path(tmp_path).unlink(missing_ok=True)
        return {"status": "ffmpeg_not_found"}


def _audio_stream_duration(path: str) -> float:
    """Duration of the first audio stream, decoded rather than read from a header.

    The container `Duration:` reflects the longest stream, so a video that is
    longer than its audio would report the video length and hide exactly the
    drift we are trying to catch here.
    """
    try:
        proc = subprocess.run(
            [_ffmpeg_binary(), "-i", path, "-map", "0:a:0", "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=120,
            env=_media_subprocess_env(),
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return 0.0
    # ffmpeg reports progress as `time=HH:MM:SS.ms`; the final one is the length.
    best = 0.0
    for m in re.finditer(r"time=\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr or ""):
        h, mi, s = float(m.group(1)), float(m.group(2)), float(m.group(3))
        best = max(best, h * 3600 + mi * 60 + s)
    return round(best, 3)

    cmd = [
        _ffmpeg_binary(), "-y",
        "-i", new_video_path,
        "-i", old_bgm_video_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c", "copy",
        "-t", str(video_duration),
        "-movflags", "+faststart",
        tmp_path,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_sec, env=_media_subprocess_env(),
        )
        if proc.returncode == 0 and Path(tmp_path).exists() and Path(tmp_path).stat().st_size > 0:
            shutil.move(tmp_path, old_bgm_video_path)
            return {"status": "bgm_remuxed", "output_path": old_bgm_video_path}
        Path(tmp_path).unlink(missing_ok=True)
        return {"status": "bgm_remux_error", "stderr": (proc.stderr or "")[-500:]}
    except subprocess.TimeoutExpired:
        Path(tmp_path).unlink(missing_ok=True)
        return {"status": "bgm_remux_timeout"}
    except FileNotFoundError:
        Path(tmp_path).unlink(missing_ok=True)
        return {"status": "ffmpeg_not_found"}


def _mix_bgm(
    video_path: str,
    bgm_tracks: list[dict[str, Any]],
    output_path: str,
    video_duration: float,
    bgm_volume: float = 0.5,
    duck_under_voice: bool = True,
    timeout_sec: int = 600,
) -> dict[str, Any]:
    """Concatenate a BGM playlist, trim it to the video, then mix under voice."""
    existing_bgm = [
        dict(t) for t in bgm_tracks
        if t.get("music_path") and Path(t["music_path"]).exists()
        and Path(t["music_path"]).stat().st_size > 0
    ]
    if not existing_bgm or not Path(video_path).exists():
        return {"status": "skipped", "output_path": video_path}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    playlist = _build_bgm_playlist(existing_bgm, video_duration)
    if not playlist:
        return {"status": "bgm_playlist_empty", "output_path": video_path}

    cmd = [_ffmpeg_binary(), "-y", "-i", video_path]
    for item in playlist:
        cmd += ["-i", str(item["music_path"])]
    cmd += [
        "-filter_complex",
        _build_bgm_mix_filter(
            playlist=playlist,
            video_duration=video_duration,
            bgm_volume=bgm_volume,
            duck_under_voice=duck_under_voice,
            has_voice=_has_audio_stream(video_path),
        ),
        "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-ar", "48000", "-ac", "2",
        "-t", str(video_duration),
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
                "status": "bgm_mixed",
                "output_path": output_path,
                "bgm_track_count": len(existing_bgm),
                "bgm_segment_count": len(playlist),
                "playlist_duration_sec": round(sum(item["duration_sec"] for item in playlist), 2),
                "trimmed_to_video_sec": round(video_duration, 2),
                "bgm_volume": bgm_volume,
                "duck_under_voice": duck_under_voice,
            }
        return {
            "status": "bgm_mix_error",
            "output_path": video_path,
            "stderr": proc.stderr[-500:] if proc.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"status": "bgm_mix_timeout", "output_path": video_path}
    except FileNotFoundError:
        return {"status": "ffmpeg_not_found", "output_path": video_path}


def _build_bgm_playlist(
    tracks: list[dict[str, Any]],
    target_duration: float,
    max_segments: int = 64,
) -> list[dict[str, Any]]:
    """Repeat the selected playlist only when needed to cover the full video."""
    if not tracks or target_duration <= 0:
        return []
    normalized: list[dict[str, Any]] = []
    for track in tracks:
        duration = float(track.get("duration_sec", 0) or 0)
        if duration <= 0:
            duration = _get_video_duration(str(track.get("music_path", "")))
        if duration > 0:
            normalized.append({**track, "duration_sec": duration})
    if not normalized:
        return []

    playlist: list[dict[str, Any]] = []
    total = 0.0
    index = 0
    while total < target_duration and len(playlist) < max_segments:
        source = normalized[index % len(normalized)]
        playlist.append({
            **source,
            "playlist_index": len(playlist),
            "offset_sec": round(total, 3),
        })
        total += float(source["duration_sec"])
        index += 1
    return playlist


def _build_bgm_mix_filter(
    *,
    playlist: list[dict[str, Any]],
    video_duration: float,
    bgm_volume: float,
    duck_under_voice: bool,
    has_voice: bool,
) -> str:
    """Normalize heterogeneous audio files, concatenate, trim, and mix."""
    bgm_volume = max(0.0, min(float(bgm_volume), 1.0))
    effective_bgm = bgm_volume * 0.35 if duck_under_voice else bgm_volume
    parts: list[str] = []
    labels: list[str] = []
    for index, item in enumerate(playlist):
        label = f"bgm{index}"
        labels.append(f"[{label}]")
        parts.append(
            f"[{index + 1}:a:0]"
            "aresample=48000,"
            "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
            f"atrim=duration={float(item['duration_sec']):.3f},"
            f"asetpts=PTS-STARTPTS[{label}]"
        )
    parts.append(
        "".join(labels)
        + f"concat=n={len(labels)}:v=0:a=1,"
        + f"atrim=duration={float(video_duration):.3f},"
        + f"asetpts=PTS-STARTPTS,volume={effective_bgm:.4f}[bgm]"
    )
    if has_voice:
        parts.append(
            "[0:a:0]aresample=48000,"
            "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
            "volume=1.0[voice]"
        )
        parts.append("[voice][bgm]amix=inputs=2:duration=first:dropout_transition=3[aout]")
    else:
        parts.append("[bgm]anull[aout]")
    return ";".join(parts)


def _has_audio_stream(path: str) -> bool:
    try:
        proc = subprocess.run(
            [_ffmpeg_binary(), "-i", path, "-hide_banner"],
            capture_output=True,
            text=True,
            timeout=10,
            env=_media_subprocess_env(),
        )
        return bool(proc.stderr and "Audio:" in proc.stderr)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _attempt_inline_recompose(
    shot: dict[str, Any],
    assets: dict[str, Any],
    output_dir: str,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Best-effort render recovery for stitch step.

    When Stage4 leaves a shot without rendered_path, run one direct timeline compose
    so stitching can still produce a preview video.
    """
    try:
        from . import timeline_compose
    except Exception as exc:
        return dict(assets), {}, {"status": "import_failed", "error": str(exc)}

    safe_assets = dict(assets)
    try:
        compose_result = timeline_compose.run(
            shot=shot,
            assets=safe_assets,
            output_dir=output_dir,
            config=config or {},
        )
    except Exception as exc:
        return safe_assets, {}, {"status": "recompose_exception", "error": str(exc)}

    safe_assets["timeline"] = compose_result.get("timeline", safe_assets.get("timeline", {}))
    safe_assets["rendered_path"] = compose_result.get("rendered_path", safe_assets.get("rendered_path", ""))
    safe_assets["compose_result"] = compose_result
    return safe_assets, compose_result, {"status": "ok"}


def run(
    shot_plan: list[dict[str, Any]],
    shot_assets: dict[str, Any],
    output_dir: str,
    bgm_tracks: list[dict[str, Any]] | None = None,
    bgm_volume: float = 0.5,
    duck_under_voice: bool = True,
    config: dict[str, Any] | None = None,
    progress_callback: Any | None = None,
    skip_bgm_remix: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Stitch all approved shots into a final video.

    Attempts real ffmpeg concat first; always produces a manifest.

    With *dry_run* the shots are validated and the manifest is written, but the
    three expensive steps — per-shot sanitize transcode, ffmpeg concat and the
    BGM mix — are skipped. The returned ``playlist`` lets a player run the
    per-shot clips back to back, which is what the studio shows while editing;
    the real file is produced later on explicit export.

    When *skip_bgm_remix* is True and a previous ``final_video_with_bgm.mp4``
    exists, the BGM audio track is copied onto the new concat video via stream
    copy instead of re-reading/re-mixing the original BGM files.  This is much
    faster for chat-triggered re-stitches where only the video changed.

    Strict time constraint: each rendered shot must be at least as long as its
    TTS narration audio.  The final concatenated video must be at least as long
    as the sum of all TTS durations.  Violations cause validation failure so the
    pipeline never ships a video where speech is truncated.
    """
    ordered_shots = []
    total_duration = 0.0
    total_tts_duration = 0.0
    missing = []
    validation_errors: list[dict[str, Any]] = []
    rendered_paths = []
    ready_items: list[dict[str, Any]] = []
    output_dir_path = Path(output_dir).expanduser().resolve()
    # Tight tolerance: rendered shot may be at most 0.35s shorter than TTS audio.
    duration_tolerance = 0.35

    config = config or {}
    stitch_cfg = config.get("stitch", {}) if isinstance(config.get("stitch"), dict) else {}
    recompose_missing = bool(stitch_cfg.get("recompose_missing_shots", False))
    # Sanitize transcodes every clip, which is the single most expensive part of
    # a stitch; a dry run exists precisely to avoid it.
    sanitize_shots = bool(stitch_cfg.get("sanitize_shots", True)) and not dry_run
    fast_concat_copy = bool(stitch_cfg.get("fast_concat_copy", True))
    concat_timeout_sec = max(60, int(_as_float(stitch_cfg.get("concat_timeout_sec", 600), 600)))
    sanitize_timeout_sec = max(30, int(_as_float(stitch_cfg.get("sanitize_timeout_sec", 240), 240)))

    def emit(stage: str, current: int, total: int, label: str, **metadata: Any) -> None:
        if progress_callback:
            try:
                progress_callback({
                    "stage": stage,
                    "current": current,
                    "total": total,
                    "label": label,
                    "metadata": metadata,
                })
            except Exception:
                pass

    total_shots = len(shot_plan)
    for idx, shot in enumerate(shot_plan, 1):
        sid = shot.get("shot_id", "")
        emit("validate", idx, total_shots, f"校验镜头 {sid}")
        assets = dict(shot_assets.get(sid, {}))
        timeline = assets.get("timeline", {})
        duration = _as_float(timeline.get("duration_sec", shot.get("duration_sec", 0)), 0.0)

        if not assets:
            missing.append(sid)
            validation_errors.append({
                "shot_id": sid,
                "reason": "missing_assets",
            })
            continue

        rendered_path = str(assets.get("rendered_path", "") or "")
        compose_result = assets.get("compose_result") if isinstance(assets.get("compose_result"), dict) else {}
        if recompose_missing:
            needs_recompose = (
                not rendered_path
                or not (Path(rendered_path).expanduser().exists() if rendered_path else False)
                or not bool(compose_result.get("render_success", False))
            )
            if needs_recompose:
                assets, recovered_compose, recover_meta = _attempt_inline_recompose(
                    shot=shot,
                    assets=assets,
                    output_dir=output_dir,
                    config=config,
                )
                if recovered_compose:
                    shot_assets[sid] = assets
                    compose_result = assets.get("compose_result") if isinstance(assets.get("compose_result"), dict) else {}
                    rendered_path = str(assets.get("rendered_path", "") or "")
                if recover_meta.get("status") not in {"ok", ""}:
                    validation_errors.append({
                        "shot_id": sid,
                        "reason": f"inline_recompose_{recover_meta.get('status')}",
                        "detail": str(recover_meta.get("error", "") or "")[:300],
                    })

        render_backend = str(compose_result.get("render_backend", "") or "")
        render_success = bool(compose_result.get("render_success", False))
        rendered_exists = bool(rendered_path) and Path(rendered_path).expanduser().exists()
        actual_duration = _get_video_duration(rendered_path) if rendered_exists else 0.0
        if actual_duration <= 0:
            render_report = compose_result.get("render_report") if isinstance(compose_result.get("render_report"), dict) else {}
            actual_duration = _as_float(render_report.get("rendered_duration_sec"), 0.0)
        audio_meta = assets.get("audio_meta") if isinstance(assets.get("audio_meta"), dict) else {}
        audio_duration = _as_float(audio_meta.get("duration_sec"), 0.0)
        # The minimum acceptable rendered duration is the TTS audio duration —
        # the video must not end before the narration finishes.
        required_duration = audio_duration if audio_duration > 0 else duration
        duration_ok = actual_duration > 0 and actual_duration + duration_tolerance >= required_duration
        is_spotlight = bool(shot.get("is_spotlight"))
        has_decoration = bool(assets.get("decoration"))

        reasons: list[str] = []
        if not render_success:
            reasons.append(f"render_not_success:{compose_result.get('render_status', 'unknown')}")
        if not rendered_exists:
            reasons.append("rendered_path_missing")
        if not duration_ok:
            reasons.append(
                f"rendered_duration_short:{actual_duration:.3f}<{required_duration:.3f}"
            )
        if reasons:
            missing.append(sid)
            validation_errors.append({
                "shot_id": sid,
                "rendered_path": rendered_path,
                "render_backend": render_backend,
                "actual_duration_sec": round(actual_duration, 3),
                "required_duration_sec": round(required_duration, 3),
                "audio_duration_sec": round(audio_duration, 3),
                "reasons": reasons,
            })
        else:
            if sanitize_shots:
                emit("sanitize", idx, total_shots, f"标准化镜头 {sid}")
                sanitized = _sanitize_shot_for_stitch(
                    shot_id=sid,
                    rendered_path=rendered_path,
                    audio_path=str(assets.get("audio_path", "") or ""),
                    duration_sec=duration,
                    output_dir=output_dir_path,
                    timeout_sec=sanitize_timeout_sec,
                )
            else:
                sanitized = {"status": "success", "output_path": rendered_path, "shot_id": sid, "has_audio": True}
            if sanitized.get("status") == "success":
                rendered_paths.append(str(sanitized.get("output_path", "")))
                ready_items.append({
                    "shot_id": sid,
                    "rendered_path": rendered_path,
                    "audio_path": str(assets.get("audio_path", "") or ""),
                    "duration_sec": duration,
                })
            else:
                missing.append(sid)
                validation_errors.append({
                    "shot_id": sid,
                    "rendered_path": rendered_path,
                    "render_backend": render_backend,
                    "audio_duration_sec": round(audio_duration, 3),
                    "reasons": [f"sanitize_failed:{sanitized.get('status', 'unknown')}"] ,
                    "sanitize_error": str(sanitized.get("stderr", "") or "")[:500],
                })

        ordered_shots.append({
            "shot_id": sid,
            "rendered_path": rendered_path,
            "timeline_path": str(Path(output_dir) / "shots" / sid / "timeline.json"),
            "duration_sec": duration,
            "offset_sec": round(total_duration, 2),
            "has_decoration": has_decoration,
            "is_spotlight": is_spotlight,
            "approved": assets.get("approved", False),
            "render_backend": render_backend,
            "render_success": render_success,
            "actual_duration_sec": round(actual_duration, 3),
            "audio_duration_sec": round(audio_duration, 3),
            "duration_ok": duration_ok,
            "ready_for_stitch": not reasons,
        })
        total_duration += duration
        total_tts_duration += audio_duration

    # Versioned outputs: each stitch writes its own timestamped file so an edit
    # can never overwrite the cut you are still comparing against, and the
    # player URL changes on every render (no stale-cache guessing).  The stable
    # `final_video*.mp4` names are kept as copies of the newest run for any
    # external script or notebook that reaches for them by name.
    stamp = _output_stamp()
    final_path = str(output_dir_path / f"final_video_{stamp}.mp4")
    final_stable_path = str(output_dir_path / "final_video.mp4")

    manifest = {
        "shots": ordered_shots,
        "total_duration_sec": round(total_duration, 2),
        "total_tts_duration_sec": round(total_tts_duration, 2),
        "shot_count": len(ordered_shots),
        "missing_shots": missing,
        "validation_errors": validation_errors,
        "ready_shot_count": len(rendered_paths),
        "output_path": final_path,
        "bgm_volume": bgm_volume,
        "duck_under_voice": duck_under_voice,
        "recompose_missing_shots": recompose_missing,
    }

    manifest_path = output_dir_path / "stitch_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if dry_run:
        emit("deferred", len(rendered_paths), len(shot_plan), "校验通过，实际合成延后到导出")
        return {
            "status": "deferred" if rendered_paths else "validation_failed",
            "deferred": True,
            # Empty on purpose: nothing has been written, and anything reading
            # this key treats a non-empty value as "there is a file to play".
            "final_video_path": "",
            "planned_output_path": final_path,
            # `path` mirrors what /api/project hands the player, so both
            # producers of a playlist agree on the key.
            "playlist": [
                {"shot_id": it["shot_id"], "path": it["rendered_path"],
                 "duration_sec": it["duration_sec"]}
                for it in ready_items
            ],
            "manifest": manifest,
            "manifest_path": str(manifest_path),
            "total_duration_sec": manifest["total_duration_sec"],
            "total_tts_duration_sec": manifest["total_tts_duration_sec"],
            "validation_errors": validation_errors,
            "ready_shot_count": len(rendered_paths),
            "shot_count": len(shot_plan),
            "missing_shots": missing,
            "concat_status": "deferred",
        }

    min_ready_ratio = max(0.5, _as_float(stitch_cfg.get("min_ready_ratio", 0.8), 0.8))
    ready_ratio = len(rendered_paths) / max(1, len(shot_plan))
    if len(rendered_paths) == 0:
        concat_result = {
            "status": "validation_failed",
            "output_path": "",
            "validation_errors": validation_errors,
            "expected_count": len(shot_plan),
            "ready_count": 0,
        }
    elif validation_errors and ready_ratio < min_ready_ratio:
        concat_result = {
            "status": "validation_failed",
            "output_path": "",
            "validation_errors": validation_errors,
            "expected_count": len(shot_plan),
            "ready_count": len(rendered_paths),
        }
    else:
        emit("concat", 1, 1, f"开始拼接 {len(rendered_paths)} 个镜头")
        # `rendered_paths` are the normalized clips when sanitize_shots is on, so
        # they are already uniform and `-c copy` succeeds in ~1s for a 3min cut.
        # _ffmpeg_concat re-probes them anyway and transcodes if anything differs,
        # so a stale/foreign clip degrades to slow-but-correct instead of failing.
        concat_result = _ffmpeg_concat(
            rendered_paths,
            final_path,
            timeout_sec=concat_timeout_sec,
            fast_copy=fast_concat_copy or sanitize_shots,
        )
        if concat_result.get("status") == "ffmpeg_error" and not sanitize_shots and ready_items:
            # Only reachable with sanitize_shots off: normalize per-shot A/V, then
            # retry concat once.  With sanitize on, normalization already happened
            # during validation and there is nothing left to heal.
            retry_paths: list[str] = []
            sanitize_failures: list[dict[str, Any]] = []
            for idx, item in enumerate(ready_items, 1):
                sid = str(item.get("shot_id", ""))
                emit("sanitize_retry", idx, len(ready_items), f"拼接失败重试: 标准化镜头 {sid}")
                retry = _sanitize_shot_for_stitch(
                    shot_id=sid,
                    rendered_path=str(item.get("rendered_path", "") or ""),
                    audio_path=str(item.get("audio_path", "") or ""),
                    duration_sec=_as_float(item.get("duration_sec"), 0.0),
                    output_dir=output_dir_path,
                    timeout_sec=sanitize_timeout_sec,
                )
                if retry.get("status") == "success":
                    retry_paths.append(str(retry.get("output_path", "")))
                else:
                    sanitize_failures.append({
                        "shot_id": sid,
                        "status": retry.get("status", "unknown"),
                        "stderr": str(retry.get("stderr", "") or "")[:500],
                    })
            if not sanitize_failures and len(retry_paths) == len(ready_items):
                emit("concat", 1, 1, f"拼接失败重试: 使用标准化镜头重新拼接 {len(retry_paths)} 个镜头")
                retry_result = _ffmpeg_concat(
                    retry_paths,
                    final_path,
                    timeout_sec=concat_timeout_sec,
                    fast_copy=False,
                )
                retry_result["concat_mode"] = (
                    f"{retry_result.get('concat_mode', 'transcode_unified')}_after_sanitize_retry"
                )
                concat_result = retry_result
            else:
                concat_result = {
                    **concat_result,
                    "status": "sanitize_retry_failed",
                    "sanitize_failures": sanitize_failures,
                }
        # Post-concat validation: final video must cover the TTS of stitched shots.
        concat_duration = _as_float(concat_result.get("duration_sec"), 0.0)
        stitched_tts = sum(
            _as_float(ri.get("duration_sec"), 0.0) for ri in ready_items
        )
        if concat_result.get("status") == "success" and stitched_tts > 0:
            if concat_duration + duration_tolerance < stitched_tts:
                logger.warning(
                    "[stitch] duration short: concat=%.1f stitched_tts=%.1f (total_tts=%.1f)",
                    concat_duration, stitched_tts, total_tts_duration,
                )
                concat_result = {
                    **concat_result,
                    "status": "duration_mismatch",
                    "output_path": "",
                    "concat_duration_sec": round(concat_duration, 3),
                    "total_tts_duration_sec": round(stitched_tts, 3),
                    "shortfall_sec": round(stitched_tts - concat_duration, 3),
                }

    # Mix BGM if available
    bgm_result = {"status": "no_bgm"}
    effective_path = concat_result.get("output_path", "") if concat_result.get("status") == "success" else ""
    if effective_path and Path(effective_path).exists():
        bgm_output = str(output_dir_path / f"final_video_with_bgm_{stamp}.mp4")
        bgm_stable_path = str(output_dir_path / "final_video_with_bgm.mp4")
        # The fast path needs the *previous* mix; with versioned names that is
        # the stable copy left behind by the last run, not our own output path.
        prev_bgm = bgm_stable_path
        old_bgm_exists = Path(prev_bgm).exists() and Path(prev_bgm).stat().st_size > 0

        if skip_bgm_remix and old_bgm_exists:
            # Fast path: copy the BGM audio from the old mix onto the new video.
            # Refuses when the old audio no longer matches the new video length,
            # because that track carries the old narration baked in.
            emit("bgm_mix", 1, 1, "快速复用旧配乐音轨（仅替换视频流）...")
            shutil.copy2(prev_bgm, bgm_output)
            bgm_result = _fast_bgm_remux(
                effective_path, bgm_output,
                video_duration=concat_result.get("duration_sec", 0) or total_duration,
            )
            if bgm_result.get("status") == "bgm_remuxed":
                effective_path = bgm_result["output_path"]
                emit("bgm_mix", 1, 1, "✓ 配乐快速复用完成")
            else:
                Path(bgm_output).unlink(missing_ok=True)
                if bgm_result.get("status") == "old_bgm_duration_mismatch":
                    emit("bgm_mix", 1, 1, (
                        f"口播已变化（旧音轨 {bgm_result.get('old_audio_duration_sec')}s "
                        f"vs 新视频 {bgm_result.get('new_video_duration_sec')}s，"
                        f"差 {bgm_result.get('drift_sec')}s），不复用旧音轨，改为完整混音"
                    ))
                else:
                    emit("bgm_mix", 1, 1, f"快速复用失败 ({bgm_result.get('status')}), 回退完整混音")
                bgm_result = {"status": "no_bgm"}

        if bgm_result.get("status") == "no_bgm" and bgm_tracks:
            emit("bgm_mix", 1, 1, "混合配乐到最终视频...")
            bgm_result = _mix_bgm(
                effective_path,
                bgm_tracks,
                bgm_output,
                video_duration=concat_result.get("duration_sec", 0) or total_duration,
                bgm_volume=bgm_volume,
                duck_under_voice=duck_under_voice,
            )
            if bgm_result.get("status") == "bgm_mixed":
                effective_path = bgm_result["output_path"]
                emit("bgm_mix", 1, 1, "✓ 配乐混合完成")
            else:
                emit("bgm_mix", 1, 1, f"配乐混合失败: {bgm_result.get('status', 'unknown')}, 使用无配乐版本")

        # Keep the fixed names pointing at this run's result.
        _mirror_to_stable(concat_result.get("output_path", ""), final_stable_path)
        if effective_path != concat_result.get("output_path", ""):
            _mirror_to_stable(effective_path, bgm_stable_path)

    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "final_video_path": effective_path,
        "total_duration_sec": concat_result.get("duration_sec", 0) or manifest["total_duration_sec"],
        "total_tts_duration_sec": round(total_tts_duration, 2),
        "shot_count": len(ordered_shots),
        "missing_count": len(missing),
        "validation_errors": validation_errors,
        "ready_shot_count": len(rendered_paths),
        "concat_status": concat_result.get("status", "not_attempted"),
        "concat_error": concat_result.get("stderr", "") or json.dumps(
            concat_result.get("validation_errors", []),
            ensure_ascii=False,
        ),
        "bgm_status": bgm_result.get("status", "no_bgm"),
        "bgm_track_count": bgm_result.get("bgm_track_count", 0),
        "bgm_volume": bgm_result.get("bgm_volume", bgm_volume),
        "duck_under_voice": bgm_result.get("duck_under_voice", duck_under_voice),
        "ducking_fallback_used": bgm_result.get("ducking_fallback_used", False),
        "status": "success" if concat_result.get("status") == "success" else (
            "manifest_only" if not missing else "partial"
        ),
    }


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
