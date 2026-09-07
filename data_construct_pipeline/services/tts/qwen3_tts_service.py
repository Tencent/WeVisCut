#!/usr/bin/env python3
"""Local Qwen3-TTS service for AICutPipeline.

Default model:
  Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice

Run:
  python server/qwen3_tts_service.py --host 127.0.0.1 --port 7865

Pre-download/load check:
  python server/qwen3_tts_service.py --download-only
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_SPEAKER = "Serena"
DEFAULT_LANGUAGE = "Chinese"
logger = logging.getLogger("aicut_qwen3_tts")

app = FastAPI(title="AICut Local Qwen3-TTS", version="1.0")
_MODEL_POOL: list[Any] = []
_MODEL_ID = ""
_DEVICE_MAP = ""
_DTYPE = ""
_ATTN = ""
_MODEL_REPLICAS = 0
_POOL_SEM: threading.BoundedSemaphore | None = None
_POOL_LOCK = threading.Lock()
_STATS_LOCK = threading.Lock()
_ACTIVE_GENERATIONS = 0
_TOTAL_REQUESTS = 0
_TOTAL_BATCHES = 0
_TOTAL_ITEMS = 0
_LAST_BATCH_SIZE = 0
_LAST_GENERATION_SEC = 0.0


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    output_path: str = Field(..., min_length=1)
    speaker: str = DEFAULT_SPEAKER
    language: str = DEFAULT_LANGUAGE
    instruct: str = ""
    speed: float = 1.0


class TTSBatchRequest(BaseModel):
    requests: list[TTSRequest] = Field(..., min_length=1)


class TTSResponse(BaseModel):
    status: str
    audio_path: str
    duration_sec: float
    backend: str
    model_id: str
    sample_rate: int
    speaker: str
    language: str
    device_map: str
    dtype: str


class TTSBatchResponse(BaseModel):
    status: str
    results: list[TTSResponse]


def _dtype_from_env(device_map: str) -> Any:
    global _DTYPE
    raw = os.environ.get("AICUT_QWEN3_TTS_DTYPE", "bfloat16").strip().lower()
    if raw == "auto":
        raw = "bfloat16" if str(device_map).startswith("cuda") else "float32"
    _DTYPE = raw
    if raw in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if raw in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def _device_map_from_env() -> str:
    raw = os.environ.get("AICUT_QWEN3_TTS_DEVICE_MAP", "cuda:0").strip()
    if raw == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return raw


def _model_replicas_from_env() -> int:
    """模型副本数：由环境变量控制，GPU 默认按 VRAM 自动估算。"""
    raw = os.environ.get("AICUT_QWEN3_TTS_MODEL_REPLICAS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            pass
    device_map = _DEVICE_MAP or _device_map_from_env()
    if not str(device_map).startswith("cuda"):
        return 1
    # 默认: 每个 replica 约 3.5GB，按 VRAM 80% 估算
    try:
        import torch
        vram_bytes = torch.cuda.get_device_properties(device_map).total_memory
        vram_gb = vram_bytes / (1024**3)
        return max(1, min(20, int(vram_gb * 0.8 / 3.5)))
    except Exception:
        return 4


def _tts_queue_timeout_sec() -> float:
    raw = os.environ.get("AICUT_QWEN3_TTS_QUEUE_TIMEOUT_SEC", "240").strip()
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return 240.0


def _max_batch_size() -> int:
    raw = os.environ.get("AICUT_QWEN3_TTS_MAX_BATCH_SIZE", "64").strip()
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 64


def _empty_cache_after_generation() -> bool:
    raw = os.environ.get("AICUT_QWEN3_TTS_EMPTY_CACHE_AFTER_BATCH", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _maybe_empty_cuda_cache() -> None:
    if _empty_cache_after_generation() and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_pool_semaphore() -> threading.BoundedSemaphore:
    global _POOL_SEM, _MODEL_REPLICAS
    if _POOL_SEM is None:
        _POOL_SEM = threading.BoundedSemaphore(_MODEL_REPLICAS)
    return _POOL_SEM


def _take_model() -> Any:
    with _POOL_LOCK:
        if not _MODEL_POOL:
            raise RuntimeError("Qwen3-TTS model pool is empty")
        return _MODEL_POOL.pop()


def _return_model(model: Any) -> None:
    with _POOL_LOCK:
        _MODEL_POOL.append(model)


def _load_model_pool() -> list[Any]:
    """加载 N 个模型副本到显存，使多个请求可以真正并行推理。"""
    global _MODEL_POOL, _MODEL_ID, _DEVICE_MAP, _ATTN, _MODEL_REPLICAS
    if _MODEL_POOL:
        return _MODEL_POOL

    try:
        from qwen_tts import Qwen3TTSModel
    except Exception as exc:
        raise RuntimeError(
            "Missing Qwen3-TTS runtime in the service Python environment. "
            f"python={sys.executable}. Install with: {sys.executable} -m pip install -U qwen-tts"
        ) from exc

    model_id = os.environ.get("AICUT_QWEN3_TTS_MODEL", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
    device_map = _device_map_from_env()
    if str(device_map).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Qwen3-TTS is configured for {device_map}, but torch.cuda.is_available() is false. "
            "Use a GPU Python environment or set AICUT_QWEN3_TTS_DEVICE_MAP=auto/cpu explicitly."
        )
    dtype = _dtype_from_env(device_map)
    attn = os.environ.get("AICUT_QWEN3_TTS_ATTN", "auto").strip()
    if attn == "auto":
        try:
            import flash_attn  # noqa: F401
            attn = "flash_attention_2"
        except Exception:
            attn = "sdpa" if str(device_map).startswith("cuda") else ""

    kwargs: dict[str, Any] = {
        "device_map": device_map,
        "dtype": dtype,
    }
    if attn:
        kwargs["attn_implementation"] = attn

    replicas = _model_replicas_from_env()
    _MODEL_REPLICAS = replicas
    logger.info(
        "Loading %d Qwen3-TTS replicas model=%s device_map=%s dtype=%s attn=%s",
        replicas, model_id, device_map, _DTYPE, attn or "default",
    )

    for i in range(replicas):
        try:
            model = Qwen3TTSModel.from_pretrained(model_id, **kwargs)
        except Exception:
            if not attn:
                raise
            require_flash_attn = os.environ.get(
                "AICUT_QWEN3_TTS_REQUIRE_FLASH_ATTN", "0"
            ).strip().lower() not in {"0", "false", "no", "off"}
            if require_flash_attn:
                raise RuntimeError(
                    "Qwen3-TTS failed to load with flash_attention_2. "
                    "Install a flash-attn build compatible with this CUDA/PyTorch environment, "
                    "or explicitly set AICUT_QWEN3_TTS_REQUIRE_FLASH_ATTN=0."
                )
            if i == 0:
                logger.warning("Qwen3-TTS load failed with attn=%s; retrying with PyTorch SDPA", attn)
                kwargs["attn_implementation"] = "sdpa"
                attn = "sdpa"
                model = Qwen3TTSModel.from_pretrained(model_id, **kwargs)
            else:
                raise
        # ── warmup: 触发 code_predictor 初始化，避免首次请求时才加载 ──
        warmup_text = "测试。"
        try:
            with torch.inference_mode():
                _ = model.generate_custom_voice(
                    text=warmup_text, language="Chinese", speaker=DEFAULT_SPEAKER, instruct="",
                )
        except Exception:
            pass
        _MODEL_POOL.append(model)
        logger.info("  replica %d/%d loaded + warmed up", i + 1, replicas)

    _MODEL_ID = model_id
    _DEVICE_MAP = device_map
    _ATTN = attn
    logger.info("Loaded %d Qwen3-TTS replicas model=%s dtype=%s", replicas, _MODEL_ID, _DTYPE)
    return _MODEL_POOL


def _build_atempo_filter(speed: float) -> str:
    """Build ffmpeg atempo chain. Single atempo only supports [0.5, 2.0]."""
    speed = max(0.5, min(float(speed or 1.0), 2.0))
    return f"atempo={speed:.4f}"


def _apply_speed_to_audio_file(audio_path: Path, speed: float, sample_rate: int) -> None:
    """Apply speaking speed after synthesis to ensure req.speed always takes effect."""
    if abs(float(speed or 1.0) - 1.0) < 1e-3:
        return
    tmp_path = audio_path.with_suffix(".speed_tmp.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-filter:a",
        _build_atempo_filter(speed),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(tmp_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0 or not tmp_path.exists() or tmp_path.stat().st_size <= 0:
        detail = (proc.stderr or "")[-500:]
        raise RuntimeError(f"speed post-process failed: {detail}")
    tmp_path.replace(audio_path)


def _generate(req: TTSRequest, model: Any) -> tuple[np.ndarray, int]:
    model_id = _MODEL_ID or os.environ.get("AICUT_QWEN3_TTS_MODEL", DEFAULT_MODEL_ID)
    language = req.language or DEFAULT_LANGUAGE

    if "VoiceDesign" in model_id:
        with torch.inference_mode():
            wavs, sr = model.generate_voice_design(
                text=req.text,
                language=language,
                instruct=req.instruct or "自然清晰的中文视频解说声，语速适中，情绪稳定。",
            )
    elif "CustomVoice" in model_id:
        with torch.inference_mode():
            wavs, sr = model.generate_custom_voice(
                text=req.text,
                language=language,
                speaker=req.speaker or DEFAULT_SPEAKER,
                instruct=req.instruct or "",
            )
    else:
        raise RuntimeError(
            "Qwen3-TTS Base models require reference audio/text. "
            "Use a CustomVoice or VoiceDesign checkpoint for this service."
        )

    if not wavs:
        raise RuntimeError("Qwen3-TTS returned no waveform")
    audio = np.asarray(wavs[0], dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("Qwen3-TTS returned empty waveform")
    peak = float(np.max(np.abs(audio)))
    if peak > 0.98:
        audio = audio / peak * 0.98
    return audio, int(sr)


def _generate_batch(
    requests: list[TTSRequest],
    model: Any,
) -> tuple[list[np.ndarray], int]:
    """Run one real batched autoregressive generation call on one model."""
    if not requests:
        return [], 0
    model_id = _MODEL_ID or os.environ.get("AICUT_QWEN3_TTS_MODEL", DEFAULT_MODEL_ID)
    texts = [req.text.strip() for req in requests]
    languages = [req.language or DEFAULT_LANGUAGE for req in requests]
    instructs = [req.instruct or "" for req in requests]

    if "VoiceDesign" in model_id:
        with torch.inference_mode():
            wavs, sample_rate = model.generate_voice_design(
                text=texts,
                language=languages,
                instruct=instructs,
            )
    elif "CustomVoice" in model_id:
        with torch.inference_mode():
            wavs, sample_rate = model.generate_custom_voice(
                text=texts,
                language=languages,
                speaker=[req.speaker or DEFAULT_SPEAKER for req in requests],
                instruct=instructs,
            )
    else:
        raise RuntimeError(
            "Qwen3-TTS Base models require reference audio/text. "
            "Use a CustomVoice or VoiceDesign checkpoint for this service."
        )

    if len(wavs) != len(requests):
        raise RuntimeError(f"Qwen3-TTS batch size mismatch: expected {len(requests)}, got {len(wavs)}")
    normalized: list[np.ndarray] = []
    for wav in wavs:
        audio = np.asarray(wav, dtype=np.float32)
        if audio.size == 0:
            raise RuntimeError("Qwen3-TTS returned empty waveform in batch")
        peak = float(np.max(np.abs(audio)))
        if peak > 0.98:
            audio = audio / peak * 0.98
        normalized.append(audio)
    return normalized, int(sample_rate)


@app.get("/health")
def health() -> dict[str, Any]:
    with _STATS_LOCK:
        stats = {
            "active_generations": _ACTIVE_GENERATIONS,
            "total_requests": _TOTAL_REQUESTS,
            "total_batches": _TOTAL_BATCHES,
            "total_items": _TOTAL_ITEMS,
            "last_batch_size": _LAST_BATCH_SIZE,
            "last_generation_sec": round(_LAST_GENERATION_SEC, 3),
        }
    with _POOL_LOCK:
        free_models = len(_MODEL_POOL)
    return {
        "status": "ok",
        "loaded": len(_MODEL_POOL) > 0,
        "model_id": _MODEL_ID or os.environ.get("AICUT_QWEN3_TTS_MODEL", DEFAULT_MODEL_ID),
        "device_map": _DEVICE_MAP or _device_map_from_env(),
        "dtype": _DTYPE or os.environ.get("AICUT_QWEN3_TTS_DTYPE", "bfloat16"),
        "attn": _ATTN,
        "model_replicas": _MODEL_REPLICAS,
        "free_model_replicas": free_models,
        "max_batch_size": _max_batch_size(),
        "empty_cache_after_generation": _empty_cache_after_generation(),
        "true_batch_inference": True,
        "queue_timeout_sec": _tts_queue_timeout_sec(),
        **stats,
    }


def _finalize_audio(req: TTSRequest, audio: np.ndarray, sample_rate: int) -> TTSResponse:
    """Write/post-process audio after the GPU model has been returned to the pool."""
    out_path = Path(req.output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, sample_rate, subtype="PCM_16")
    _apply_speed_to_audio_file(out_path, req.speed, sample_rate)
    final_duration = round(float(len(audio)) / float(sample_rate) / max(float(req.speed or 1.0), 1e-6), 3)
    return TTSResponse(
        status="success",
        audio_path=str(out_path),
        duration_sec=final_duration,
        backend="qwen3_service",
        model_id=_MODEL_ID,
        sample_rate=sample_rate,
        speaker=req.speaker or DEFAULT_SPEAKER,
        language=req.language or DEFAULT_LANGUAGE,
        device_map=_DEVICE_MAP,
        dtype=_DTYPE,
    )


def _finalize_audio_safe(req: TTSRequest, audio: np.ndarray, sample_rate: int) -> TTSResponse:
    try:
        return _finalize_audio(req, audio, sample_rate)
    except Exception:
        logger.error("Qwen3-TTS audio finalize failed:\n%s", traceback.format_exc())
        return TTSResponse(
            status="error",
            audio_path="",
            duration_sec=0.0,
            backend="qwen3_service",
            model_id=_MODEL_ID,
            sample_rate=sample_rate,
            speaker=req.speaker or DEFAULT_SPEAKER,
            language=req.language or DEFAULT_LANGUAGE,
            device_map=_DEVICE_MAP,
            dtype=_DTYPE,
        )


def _mark_generation_start(*, batch_size: int) -> float:
    global _ACTIVE_GENERATIONS, _TOTAL_REQUESTS, _TOTAL_BATCHES, _TOTAL_ITEMS, _LAST_BATCH_SIZE
    with _STATS_LOCK:
        _ACTIVE_GENERATIONS += 1
        _TOTAL_REQUESTS += 1
        if batch_size > 1:
            _TOTAL_BATCHES += 1
        _TOTAL_ITEMS += batch_size
        _LAST_BATCH_SIZE = batch_size
    return time.perf_counter()


def _mark_generation_end(started: float) -> None:
    global _ACTIVE_GENERATIONS, _LAST_GENERATION_SEC
    elapsed = time.perf_counter() - started
    with _STATS_LOCK:
        _ACTIVE_GENERATIONS = max(0, _ACTIVE_GENERATIONS - 1)
        _LAST_GENERATION_SEC = elapsed


@app.post("/tts", response_model=TTSResponse)
def tts(req: TTSRequest) -> TTSResponse:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    sem = _get_pool_semaphore()
    acquired = sem.acquire(timeout=_tts_queue_timeout_sec())
    if not acquired:
        raise HTTPException(status_code=503, detail="qwen3_tts service busy, queue timeout")
    model = _take_model()
    started = _mark_generation_start(batch_size=1)
    try:
        audio, sample_rate = _generate(req, model)
    except Exception as exc:  # noqa: BLE001
        logger.error("Qwen3-TTS request failed:\n%s", traceback.format_exc())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _mark_generation_end(started)
        _return_model(model)
        _maybe_empty_cuda_cache()
        sem.release()
    return _finalize_audio(req, audio, sample_rate)


@app.post("/tts_batch", response_model=TTSBatchResponse)
def tts_batch(batch: TTSBatchRequest) -> TTSBatchResponse:
    """True batched TTS: one model.generate call handles the full text list."""
    if not batch.requests:
        raise HTTPException(status_code=400, detail="requests is required")
    max_batch_size = _max_batch_size()
    if len(batch.requests) > max_batch_size:
        raise HTTPException(
            status_code=413,
            detail=f"batch too large: {len(batch.requests)} > max_batch_size={max_batch_size}",
        )
    sem = _get_pool_semaphore()
    acquired = sem.acquire(timeout=_tts_queue_timeout_sec())
    if not acquired:
        raise HTTPException(status_code=503, detail="qwen3_tts service busy, queue timeout")
    model = _take_model()
    started = _mark_generation_start(batch_size=len(batch.requests))
    try:
        wavs, sample_rate = _generate_batch(batch.requests, model)
    except Exception as exc:  # noqa: BLE001
        logger.error("Qwen3-TTS batch request failed:\n%s", traceback.format_exc())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _mark_generation_end(started)
        _return_model(model)
        _maybe_empty_cuda_cache()
        sem.release()

    io_workers = max(
        1,
        min(
            len(batch.requests),
            int(os.environ.get("AICUT_QWEN3_TTS_IO_WORKERS", "8") or 8),
        ),
    )
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        results = list(pool.map(
            lambda pair: _finalize_audio_safe(pair[0], pair[1], sample_rate),
            zip(batch.requests, wavs),
        ))

    success_count = sum(result.status == "success" for result in results)
    logger.info(
        "Qwen3-TTS true batch: %d/%d success, generation=%.3fs, io_workers=%d, pool=%d/%d free",
        success_count,
        len(results),
        _LAST_GENERATION_SEC,
        io_workers,
        len(_MODEL_POOL),
        _MODEL_REPLICAS,
    )
    return TTSBatchResponse(
        status="success" if success_count == len(results) else "partial_failure",
        results=results,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AICut local Qwen3-TTS service.")
    parser.add_argument("--host", default=os.environ.get("AICUT_QWEN3_TTS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AICUT_QWEN3_TTS_PORT", "7865")))
    parser.add_argument("--model-id", default=os.environ.get("AICUT_QWEN3_TTS_MODEL", DEFAULT_MODEL_ID))
    parser.add_argument("--speaker", default=os.environ.get("AICUT_QWEN3_TTS_SPEAKER", DEFAULT_SPEAKER))
    parser.add_argument("--device-map", default=os.environ.get("AICUT_QWEN3_TTS_DEVICE_MAP", "cuda:0"))
    parser.add_argument("--dtype", default=os.environ.get("AICUT_QWEN3_TTS_DTYPE", "bfloat16"))
    parser.add_argument("--attn", default=os.environ.get("AICUT_QWEN3_TTS_ATTN", "auto"))
    parser.add_argument(
        "--no-eager-load",
        action="store_true",
        default=os.environ.get("AICUT_QWEN3_TTS_EAGER_LOAD", "1").lower() in {"0", "false", "no"},
        help="Start the HTTP server without loading the model first.",
    )
    parser.add_argument("--download-only", action="store_true", help="Download/load model and exit.")
    args = parser.parse_args()

    logging.basicConfig(level=os.environ.get("AICUT_QWEN3_TTS_LOG_LEVEL", "INFO"))
    os.environ["AICUT_QWEN3_TTS_MODEL"] = args.model_id
    os.environ["AICUT_QWEN3_TTS_SPEAKER"] = args.speaker
    os.environ["AICUT_QWEN3_TTS_DEVICE_MAP"] = args.device_map
    os.environ["AICUT_QWEN3_TTS_DTYPE"] = args.dtype
    os.environ["AICUT_QWEN3_TTS_ATTN"] = args.attn

    if args.download_only or not args.no_eager_load:
        _load_model_pool()
        print(f"Loaded {len(_MODEL_POOL)} replicas of {_MODEL_ID} on {_DEVICE_MAP}, dtype={_DTYPE}, attn={_ATTN or 'default'}")
    if args.download_only:
        return

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
