"""Tool_GenerateAudio: turn a shot's vocal_script into narration audio.

TTS backends are tried in order:
  1. The HTTP service at `tts.service_url` (the reference deployment runs
     Qwen3-TTS there; any compatible server works).
  2. A cloud TTS vendor reached through the optional mmfinder client. Requires
     TTS_APP_ID plus that client under external_tool/mmfinder; skipped when
     either is absent.
  3. edge-tts — free, no API key, good enough for a smoke test.
  4. Duration estimation only, for offline runs where audio is not needed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
import urllib.parse
import concurrent.futures
from pathlib import Path
from typing import Any
import logging

MMFINDER_ROOT = Path(os.environ.get(
    "MMFINDER_ROOT",
    str(Path(__file__).resolve().parents[2] / "external_tool" / "mmfinder" / "WeAgent_WeStoryCut"),
))


def _is_local_service_url(url: str) -> bool:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "0.0.0.0"}


def _urlopen_no_proxy_for_local(req: urllib.request.Request, timeout: int | float) -> Any:
    target = req.full_url
    if _is_local_service_url(target):
        # Bypass environment proxy for localhost to avoid corporate proxy
        # returning 403 "forbidden host" on 127.0.0.1 requests.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


def _python_bin() -> str:
    return os.environ.get("AICUT_MMFINDER_PYTHON", "").strip() or sys.executable


def estimate_duration(text: str, speed: float = 1.0) -> float:
    """Estimate audio duration from text length (Chinese ~4.5 chars/sec)."""
    return max(1.0, round(len(text) / 4.5 / speed, 2))


def _shot_tts_text(shot: dict[str, Any]) -> str:
    """Return the narration text, tolerating model-emitted field aliases."""
    for key in (
        "vocal_script",
        "vocal_query",
        "tts_text",
        "audio_text",
        "narration",
        "script",
        "voiceover",
        "subtitle",
    ):
        value = str(shot.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _tts_via_edge_tts(
    text: str,
    output_path: str,
    voice: str = "zh-CN-XiaoxiaoNeural",
    speed: float = 1.0,
    timeout_sec: int | None = None,
) -> dict[str, Any]:
    """Generate TTS via edge-tts (Microsoft free TTS)."""
    if timeout_sec is None:
        timeout_sec = int(os.environ.get("AICUT_EDGE_TTS_TIMEOUT", "120"))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    rate_str = f"+{int((speed - 1) * 100)}%" if speed >= 1 else f"{int((speed - 1) * 100)}%"
    script = (
        "import asyncio, edge_tts, sys\n"
        "async def main():\n"
        "    communicate = edge_tts.Communicate(sys.argv[1], sys.argv[2], rate=sys.argv[3])\n"
        "    await communicate.save(sys.argv[4])\n"
        "asyncio.run(main())\n"
    )
    cmd = [sys.executable, "-c", script, text, voice, rate_str, output_path]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        if proc.returncode == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 0:
            duration = _get_audio_duration(output_path)
            return {
                "audio_path": output_path,
                "duration_sec": duration,
                "backend": "edge_tts",
                "voice": voice,
                "status": "success",
            }
        return {"status": "error", "backend": "edge_tts", "stderr": proc.stderr[-500:]}
    except subprocess.TimeoutExpired:
        return {"status": "error", "backend": "edge_tts", "message": "TTS timeout"}
    except FileNotFoundError:
        return {"status": "error", "backend": "edge_tts", "message": "edge-tts not installed"}


def _tts_via_qwen3_service(
    text: str,
    output_path: str,
    speaker: str = "Serena",
    language: str = "Chinese",
    instruct: str = "",
    speed: float = 1.0,
    service_url: str = "",
    timeout_sec: int | None = None,
) -> dict[str, Any]:
    """Generate TTS via the local Qwen3-TTS service."""
    if timeout_sec is None:
        timeout_sec = int(os.environ.get("AICUT_QWEN3_TTS_TIMEOUT", "600"))
    base_url = (
        service_url.strip()
        or os.environ.get("AICUT_QWEN3_TTS_URL", "").strip()
        or "http://127.0.0.1:7865"
    ).rstrip("/")
    payload = json.dumps(
        {
            "text": text,
            "output_path": output_path,
            "speaker": speaker,
            "language": language,
            "instruct": instruct,
            "speed": speed,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/tts",
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with _urlopen_no_proxy_for_local(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body or "{}")
        audio_path = str(data.get("audio_path") or output_path)
        if Path(audio_path).exists() and Path(audio_path).stat().st_size > 0:
            duration = float(data.get("duration_sec", 0.0) or 0.0)
            if duration <= 0:
                duration = _get_audio_duration(audio_path)
            return {
                "audio_path": audio_path,
                "duration_sec": duration,
                "backend": "qwen3_service",
                "voice": speaker,
                "status": "success",
                "service_url": base_url,
                "model_id": data.get("model_id", ""),
                "device_map": data.get("device_map", ""),
                "sample_rate": data.get("sample_rate", 0),
                "language": data.get("language", language),
            }
        return {"status": "error", "backend": "qwen3_service", "message": "service returned no audio file"}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[-500:]
        return {"status": "error", "backend": "qwen3_service", "message": f"HTTP {exc.code}: {detail}"}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {"status": "error", "backend": "qwen3_service", "message": str(exc)}


def _tts_via_volcano(
    text: str,
    output_path: str,
    voice_type: str = "BV437_streaming",
    speed: float = 1.0,
    timeout_sec: int | None = None,
) -> dict[str, Any]:
    """Generate TTS through the optional cloud-vendor client under MMFINDER_ROOT.

    Needs both that client and a TTS_APP_ID; without either it reports an error
    so the caller falls through to edge-tts.
    """
    if timeout_sec is None:
        timeout_sec = int(os.environ.get("AICUT_TTS_TIMEOUT", "30"))
    get_tts_dir = MMFINDER_ROOT / "scripts" / "get-tts-api"
    if not get_tts_dir.exists():
        return {"status": "error", "backend": "volcano", "message": "TTS script dir not found"}

    app_id = (
        os.environ.get("TTS_APP_ID", "").strip()
        or os.environ.get("VOLCANO_TTS_APPID", "").strip()
    )
    if not app_id:
        return {
            "status": "error",
            "backend": "volcano",
            "message": "TTS_APP_ID is not set; skipping the cloud TTS vendor",
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    src_root = str(MMFINDER_ROOT / "src")
    pb_root = str(MMFINDER_ROOT / "src" / "pb")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(
        [src_root, pb_root, str(get_tts_dir)] +
        ([existing] if existing else [])
    )

    speed = max(0.2, min(float(speed or 1.0), 3.0))
    script = (
        "import asyncio, json, os, sys\n"
        "sys.path.insert(0, '{src_root}')\n"
        "sys.path.insert(0, '{pb_root}')\n"
        "sys.path.insert(0, '{tts_dir}')\n"
        "from GetTTS import gen_volcano_tts_ext_full\n"
        "async def main():\n"
        "    result = await gen_volcano_tts_ext_full(\n"
        "        sys.argv[1],\n"
        "        app_id=os.environ.get('TTS_APP_ID', '').strip() or os.environ.get('VOLCANO_TTS_APPID', '').strip(),\n"
        "        voice_type=sys.argv[2],\n"
        "        speed_ratio=float(sys.argv[3]),\n"
        "        cluster=os.environ.get('TTS_CLUSTER', 'volcano_tts'),\n"
        "        proxy_url=os.environ.get('TTS_PROXY_URL', '') or None,\n"
        "    )\n"
        "    print(json.dumps(result, ensure_ascii=False))\n"
        "asyncio.run(main())\n"
    ).format(
        src_root=src_root,
        pb_root=pb_root,
        tts_dir=str(get_tts_dir),
    )
    cmd = [_python_bin(), "-c", script, text, voice_type, str(speed)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, env=env)
        if proc.returncode == 0:
            for line in reversed(proc.stdout.strip().split("\n")):
                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and data.get("audio_url"):
                        _download_audio_url(str(data["audio_url"]), output_path)
                        if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                            duration = _get_audio_duration(output_path)
                            return {
                                "audio_path": output_path,
                                "duration_sec": duration,
                                "backend": "volcano",
                                "voice": voice_type,
                                "audio_url": data.get("audio_url", ""),
                                "status": "success",
                            }
                except json.JSONDecodeError:
                    continue
        return {
            "status": "error",
            "backend": "volcano",
            "stderr": proc.stderr[-800:],
            "stdout": proc.stdout[-800:],
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "backend": "volcano", "message": "Volcano TTS timeout"}
    except FileNotFoundError:
        return {"status": "error", "backend": "volcano", "message": "Python not found"}
    except (OSError, urllib.error.URLError) as exc:
        return {"status": "error", "backend": "volcano", "message": f"audio download failed: {exc}"}


def _download_audio_url(audio_url: str, output_path: str, timeout_sec: int = 90) -> None:
    """Download a TTS CDN URL to the requested local path."""
    if not audio_url:
        raise ValueError("empty audio_url")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(audio_url, headers={"User-Agent": "WevisionCut/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        data = resp.read()
    if not data:
        raise RuntimeError("downloaded audio is empty")
    Path(output_path).write_bytes(data)


def _get_audio_duration(path: str) -> float:
    """Get audio duration via ffprobe."""
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
    return estimate_duration("", 1.0)


def run_tts(
    text: str,
    output_path: str,
    voice: str = "default",
    speed: float = 1.0,
    backend: str = "auto",
    service_url: str = "",
    language: str = "Chinese",
    instruct: str = "",
    qwen3_timeout_sec: int | None = None,
) -> dict[str, Any]:
    """Generate TTS audio, trying multiple backends in order."""
    attempts: list[dict[str, Any]] = []
    if not text.strip():
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"")
        return {
            "audio_path": output_path,
            "duration_sec": 0.0,
            "text": "",
            "voice": voice,
            "backend": "none",
            "status": "empty_text",
        }

    # Try local Qwen3-TTS service first when requested.
    if backend in ("auto", "qwen3_service", "qwen3", "hf_service", "huggingface", "local_hf"):
        speaker = voice if voice and voice != "default" else os.environ.get("AICUT_QWEN3_TTS_SPEAKER", "Serena")
        result = _tts_via_qwen3_service(
            text,
            output_path,
            speaker=speaker,
            language=language,
            instruct=instruct,
            speed=speed,
            service_url=service_url,
            timeout_sec=qwen3_timeout_sec,
        )
        attempts.append(_attempt_summary(result))
        if result.get("status") == "success":
            result["text"] = text
            result["speed"] = speed
            result["attempts"] = attempts
            return result
        if backend in ("qwen3_service", "qwen3", "hf_service", "huggingface", "local_hf"):
            result["text"] = text
            result["speed"] = speed
            result["attempts"] = attempts
            result.setdefault("audio_path", output_path)
            result.setdefault("duration_sec", 0.0)
            result.setdefault("voice", voice)
            return result

    # Try Volcano TTS first
    if backend in ("auto", "volcano", "cosyvoice"):
        volcano_voice = voice if voice != "default" else "BV437_streaming"
        result = _tts_via_volcano(text, output_path, volcano_voice, speed)
        attempts.append(_attempt_summary(result))
        if result.get("status") == "success":
            result["text"] = text
            result["speed"] = speed
            result["attempts"] = attempts
            return result

    # Try edge-tts
    if backend in ("auto", "edge_tts", "edge-tts"):
        edge_voice = voice if voice not in ("default", "BV437_streaming") else "zh-CN-XiaoxiaoNeural"
        result = _tts_via_edge_tts(text, output_path, edge_voice, speed)
        attempts.append(_attempt_summary(result))
        if result.get("status") == "success":
            result["text"] = text
            result["speed"] = speed
            result["attempts"] = attempts
            return result

    # Fallback: estimate duration and create empty file for trajectory tracking
    duration = estimate_duration(text, speed)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(output_path).exists():
        Path(output_path).write_bytes(b"")
    return {
        "audio_path": output_path,
        "duration_sec": duration,
        "text": text,
        "voice": voice,
        "speed": speed,
        "backend": "estimated",
        "status": "fallback",
        "attempts": attempts,
    }


def _attempt_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "backend": result.get("backend", ""),
        "status": result.get("status", ""),
        "message": result.get("message", ""),
        "stderr": str(result.get("stderr", ""))[-300:],
    }


def _tts_via_qwen3_service_batch(
    requests: list[dict[str, Any]],
    service_url: str = "",
    timeout_sec: int = 1200,
) -> dict[str, dict[str, Any]]:
    """一次 HTTP 请求处理全部 TTS，消除线程争用，GPU 保持热状态。"""
    base_url = (
        service_url.strip()
        or os.environ.get("AICUT_QWEN3_TTS_URL", "").strip()
        or "http://127.0.0.1:7865"
    ).rstrip("/")
    payload = json.dumps({"requests": requests}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/tts_batch",
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with _urlopen_no_proxy_for_local(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[-500:]
        return {"_batch_error": {"status": "error", "message": f"HTTP {exc.code}: {detail}"}}
    except Exception as exc:
        return {"_batch_error": {"status": "error", "message": str(exc)}}

    results: dict[str, dict[str, Any]] = {}
    request_by_output_path = {
        str(req.get("output_path", "")): req for req in requests if isinstance(req, dict)
    }
    all_requests = [req for req in requests if isinstance(req, dict)]
    for idx, item in enumerate(data.get("results", [])):
        raw_path = str(item.get("audio_path", ""))
        output_path = raw_path
        # output_path 由请求参数中的 output_path 决定，服务端返回的实际路径作为 fallback
        sid = ""
        matched_req = request_by_output_path.get(raw_path)
        if matched_req is None and idx < len(all_requests):
            # 部分服务版本不会回传完整的 output_path，退化为按请求顺序对齐。
            matched_req = all_requests[idx]
        if matched_req is not None:
            sid = str(matched_req.get("_shot_id", ""))
            output_path = str(matched_req.get("output_path", raw_path))
        if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
            duration = float(item.get("duration_sec", 0.0) or 0.0)
            if duration <= 0:
                duration = _get_audio_duration(output_path)
            if sid:
                results[sid] = {
                    "audio_path": output_path,
                    "duration_sec": duration,
                    "backend": "qwen3_service_batch",
                    "voice": item.get("speaker", ""),
                    "status": "success",
                }
    return results


def _qwen3_service_health(service_url: str, timeout_sec: int = 3) -> dict[str, Any]:
    base_url = (
        service_url.strip()
        or os.environ.get("AICUT_QWEN3_TTS_URL", "").strip()
        or "http://127.0.0.1:7865"
    ).rstrip("/")
    req = urllib.request.Request(f"{base_url}/health", method="GET")
    try:
        with _urlopen_no_proxy_for_local(req, timeout=max(1, int(timeout_sec))) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body or "{}")
        if isinstance(data, dict):
            return data
        probe_error = f"/health 返回了非 JSON 对象: {type(data).__name__}"
    except Exception as exc:
        probe_error = f"{type(exc).__name__}: {exc}"
    # Callers need to tell "service is down" apart from "service is old", since
    # the two have completely different fixes.
    return {"_probe_ok": False, "_probe_error": probe_error, "_probe_url": base_url}


def batch_run(
    shots: list[dict[str, Any]],
    output_dir: str,
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Batch Qwen3-TTS using the service's native list inference."""
    tts_cfg = config.get("tts", {})
    backend = tts_cfg.get("backend", "auto")
    lang = tts_cfg.get("language", "Chinese")
    voice = tts_cfg.get("voice", "default")
    instruct = tts_cfg.get("instruct", "")
    speed = tts_cfg.get("speed", 1.0)
    service_url = tts_cfg.get("service_url", "")
    batch_size = max(1, int(tts_cfg.get("batch_size", 0) or 0) or 20)
    batch_concurrency = max(1, int(tts_cfg.get("batch_concurrency", 0) or 0) or 3)
    timeout_sec = int(tts_cfg.get("timeout_sec", 0) or 0) or 1200
    dispatch_mode = str(tts_cfg.get("dispatch_mode", "pool")).strip().lower()
    dispatch_workers = int(tts_cfg.get("dispatch_workers", 0) or 0)
    health_timeout_sec = int(tts_cfg.get("health_probe_timeout_sec", 0) or 0) or 3
    require_true_batch = bool(tts_cfg.get("require_true_batch", True))
    batch_retry_max = max(0, int(tts_cfg.get("batch_retry_max", 1) or 0))
    length_bucket = bool(tts_cfg.get("length_bucket", True))
    logging.getLogger("aicut_pipeline").info(
        "[generate_audio] batch TTS: %d shots, batch_size=%d, concurrency=%d",
        len(shots), batch_size, batch_concurrency,
    )

    # 只在 qwen3_service 时走批量
    if backend not in ("auto", "qwen3_service", "qwen3"):
        results = {}
        for shot in shots:
            result = run(shot, output_dir, config)
            results[shot.get("shot_id", "")] = result
        return results

    health = _qwen3_service_health(service_url=service_url, timeout_sec=health_timeout_sec)
    if dispatch_mode == "batch" and require_true_batch and not bool(health.get("true_batch_inference", False)):
        if health.get("_probe_ok") is False:
            raise RuntimeError(
                f"Qwen3-TTS 服务不可达 ({health.get('_probe_url')}/health "
                f"探测失败: {health.get('_probe_error')})。"
                "请先启动 run_qwen3_tts_service.sh，或在配置里把 tts.service_url "
                "指向可用实例。"
            )
        raise RuntimeError(
            "Qwen3-TTS 服务已启动但未声明 true_batch_inference（版本过旧）。"
            "请用更新后的 run_qwen3_tts_service.sh 重启，"
            "或把 tts.dispatch_mode 改为 pool。"
        )
    service_max_batch_size = int(health.get("max_batch_size", 0) or 0)
    if service_max_batch_size > 0 and batch_size > service_max_batch_size:
        logging.getLogger("aicut_pipeline").warning(
            "[generate_audio] clamp TTS batch_size from %d to service max_batch_size=%d",
            batch_size,
            service_max_batch_size,
        )
        batch_size = service_max_batch_size

    # 模型池调度模式：按镜头并发提交 /tts，空闲模型自动接下一条。
    # 这比固定大批次 /tts_batch 更容易把多副本 GPU 吃满。
    if dispatch_mode in {"pool", "model_pool", "per_shot"}:
        if dispatch_workers <= 0:
            replicas = int(health.get("model_replicas", 0) or 0)
            # 预留少量排队，避免只有等量并发时产生 idle 缝隙。
            if replicas > 0:
                dispatch_workers = min(len(shots), max(1, replicas * 2))
            else:
                dispatch_workers = min(len(shots), max(1, batch_concurrency * 2))
        logger = logging.getLogger("aicut_pipeline")
        logger.info(
            "[generate_audio] pool dispatch: shots=%d workers=%d mode=%s",
            len(shots), dispatch_workers, dispatch_mode,
        )
        results: dict[str, dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, dispatch_workers)) as ex:
            futures = {ex.submit(run, shot, output_dir, config): shot for shot in shots}
            for future in concurrent.futures.as_completed(futures):
                shot = futures[future]
                sid = str(shot.get("shot_id", ""))
                try:
                    results[sid] = future.result()
                except Exception as exc:  # noqa: BLE001
                    text = _shot_tts_text(shot)
                    results[sid] = {
                        "shot_id": sid,
                        "audio_path": "",
                        "audio_meta": {
                            "duration_sec": max(1.0, len(text) / 4.5),
                            "voice": voice,
                            "backend": "qwen3_service",
                            "status": "error",
                            "message": str(exc),
                            "attempts": [],
                            "text": text,
                            "text_length": len(text),
                        },
                        "status": "error",
                    }
        return results

    # 构造全部请求 payload
    all_payloads = []
    for shot in shots:
        sid = str(shot.get("shot_id", "unknown"))
        text = _shot_tts_text(shot)
        if not text.strip():
            continue
        out_path = str(Path(output_dir) / "shots" / sid / "narration.wav")
        all_payloads.append({
            "text": text,
            "output_path": out_path,
            "speaker": voice,
            "language": lang,
            "instruct": instruct,
            "speed": speed,
            "_shot_id": sid,
        })

    if not all_payloads:
        return {}

    # Autoregressive batches wait for the longest sequence. Group similar text
    # lengths together to reduce padding/tail idle while keeping output mapping by shot_id.
    if length_bucket:
        all_payloads.sort(key=lambda item: len(str(item.get("text", ""))))

    # 按 batch_size 拆分批
    batches = [all_payloads[i:i + batch_size] for i in range(0, len(all_payloads), batch_size)]
    all_results: dict[str, dict[str, Any]] = {}
    def _send_batch(batch_payloads: list[dict]) -> dict[str, dict]:
        return _tts_via_qwen3_service_batch(
            batch_payloads, service_url=service_url, timeout_sec=timeout_sec,
        )

    pending_batches = batches
    logger = logging.getLogger("aicut_pipeline")
    for attempt in range(batch_retry_max + 1):
        errors: list[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_concurrency) as ex:
            futures = {ex.submit(_send_batch, batch): idx for idx, batch in enumerate(pending_batches)}
            for future in concurrent.futures.as_completed(futures):
                try:
                    chunk_results = future.result()
                except Exception as exc:
                    errors.append(str(exc))
                    continue
                if "_batch_error" in chunk_results:
                    errors.append(chunk_results["_batch_error"].get("message", "unknown"))
                    continue
                all_results.update(chunk_results)

        missing_payloads = [
            payload
            for payload in all_payloads
            if str(payload.get("_shot_id", "")) not in all_results
        ]
        if not missing_payloads:
            break
        if attempt >= batch_retry_max:
            logger.error(
                "[generate_audio] true batch TTS incomplete after %d attempts: missing=%s errors=%s",
                attempt + 1,
                [item.get("_shot_id") for item in missing_payloads[:10]],
                errors[:3],
            )
            break
        retry_size = max(1, min(batch_size, max(2, batch_size // 2)))
        pending_batches = [
            missing_payloads[i:i + retry_size]
            for i in range(0, len(missing_payloads), retry_size)
        ]
        logger.warning(
            "[generate_audio] retrying %d missing TTS items in %d true batches (attempt %d/%d)",
            len(missing_payloads),
            len(pending_batches),
            attempt + 2,
            batch_retry_max + 1,
        )

    # 包装为标准返回格式
    results: dict[str, dict[str, Any]] = {}
    for shot in shots:
        sid = str(shot.get("shot_id", "unknown"))
        text = _shot_tts_text(shot)
        out_path = str(Path(output_dir) / "shots" / sid / "narration.wav")
        if sid in all_results:
            r = all_results[sid]
            results[sid] = {
                "shot_id": sid,
                "audio_path": r.get("audio_path", out_path),
                "audio_meta": {
                    "duration_sec": r.get("duration_sec", 0.0),
                    "voice": voice,
                    "backend": r.get("backend", ""),
                    "status": r.get("status", "unknown"),
                    "message": "",
                    "attempts": [],
                    "text": text,
                    "text_length": len(text),
                },
                "status": r.get("status", "unknown"),
            }
        else:
            results[sid] = {
                "shot_id": sid,
                "audio_path": "",
                "audio_meta": {
                    "duration_sec": max(1.0, len(text) / 4.5),
                    "voice": voice,
                    "backend": "qwen3_service_batch",
                    "status": "error",
                    "message": "batch result missing",
                    "attempts": [],
                    "text": text,
                    "text_length": len(text),
                },
                "status": "error",
            }
    return results


def run(
    shot: dict[str, Any],
    output_dir: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Full TTS pipeline for one shot."""
    shot_id = shot.get("shot_id", "unknown")
    text = _shot_tts_text(shot)
    tts_cfg = config.get("tts", {})

    out_path = str(Path(output_dir) / "shots" / shot_id / "narration.wav")
    result = run_tts(
        text=text,
        output_path=out_path,
        voice=tts_cfg.get("voice", "default"),
        speed=tts_cfg.get("speed", 1.0),
        backend=tts_cfg.get("backend", "auto"),
        service_url=tts_cfg.get("service_url", ""),
        language=tts_cfg.get("language", "Chinese"),
        instruct=tts_cfg.get("instruct", ""),
        qwen3_timeout_sec=int(tts_cfg.get("timeout_sec", 0) or 0) or None,
    )
    return {
        "shot_id": shot_id,
        "audio_path": result.get("audio_path", out_path),
        "audio_meta": {
            "duration_sec": result.get("duration_sec", 0.0),
            "voice": result.get("voice", ""),
            "backend": result.get("backend", ""),
            "status": result.get("status", "unknown"),
            "message": result.get("message", ""),
            "attempts": result.get("attempts", []),
            "text": text,
            "text_length": len(text),
        },
        "status": result.get("status", "unknown"),
    }
