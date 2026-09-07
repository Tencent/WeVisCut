#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# 推荐把 TTS 服务放到非 ASR GPU，例如:
#   AICUT_QWEN3_TTS_CUDA_VISIBLE_DEVICES=4 ./run_qwen3_tts_service.sh
# 此时服务内部仍使用 cuda:0，但实际是可见设备列表中的第 1 张物理卡。
if [[ -n "${AICUT_QWEN3_TTS_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$AICUT_QWEN3_TTS_CUDA_VISIBLE_DEVICES"
fi

export AICUT_QWEN3_TTS_MODEL="${AICUT_QWEN3_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice}"
export AICUT_QWEN3_TTS_DEVICE_MAP="${AICUT_QWEN3_TTS_DEVICE_MAP:-cuda:0}"
export AICUT_QWEN3_TTS_DTYPE="${AICUT_QWEN3_TTS_DTYPE:-bfloat16}"
export AICUT_QWEN3_TTS_ATTN="${AICUT_QWEN3_TTS_ATTN:-auto}"
export AICUT_QWEN3_TTS_REQUIRE_FLASH_ATTN="${AICUT_QWEN3_TTS_REQUIRE_FLASH_ATTN:-0}"
export AICUT_QWEN3_TTS_SPEAKER="${AICUT_QWEN3_TTS_SPEAKER:-Serena}"
export AICUT_QWEN3_TTS_EAGER_LOAD="${AICUT_QWEN3_TTS_EAGER_LOAD:-1}"
# 单 GPU 使用一个模型做原生 list batch；多副本会复制权重并争用同一 GPU。
export AICUT_QWEN3_TTS_MODEL_REPLICAS="${AICUT_QWEN3_TTS_MODEL_REPLICAS:-1}"
export AICUT_QWEN3_TTS_IO_WORKERS="${AICUT_QWEN3_TTS_IO_WORKERS:-8}"
# 等待队列超时拉长，避免批量请求在服务端排队时过早失败。
export AICUT_QWEN3_TTS_QUEUE_TIMEOUT_SEC="${AICUT_QWEN3_TTS_QUEUE_TIMEOUT_SEC:-900}"
# 批量生成后的 CUDA cache 默认释放，避免大 batch 后服务常驻占满显存。
export AICUT_QWEN3_TTS_EMPTY_CACHE_AFTER_BATCH="${AICUT_QWEN3_TTS_EMPTY_CACHE_AFTER_BATCH:-1}"
# 服务端保护阈值；需要冲极限时可手动提高，但 ASR/TTS 共卡时不要设太大。
export AICUT_QWEN3_TTS_MAX_BATCH_SIZE="${AICUT_QWEN3_TTS_MAX_BATCH_SIZE:-64}"
PYTHON_BIN="${AICUT_QWEN3_TTS_PYTHON:-python}"

"$PYTHON_BIN" - <<'PY'
import sys
import torch

print(f"[qwen3-tts] python={sys.executable}", flush=True)
print(
    f"[qwen3-tts] torch={torch.__version__} "
    f"cuda_available={torch.cuda.is_available()} cuda_devices={torch.cuda.device_count()}",
    flush=True,
)
print(
    f"[qwen3-tts] CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES', '') or '(all)'}",
    flush=True,
)
if not torch.cuda.is_available():
    print(
        "[qwen3-tts] ERROR: this Python environment has no CUDA-enabled PyTorch. "
        "Set AICUT_QWEN3_TTS_PYTHON to the GPU environment's python.",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(1)
try:
    import qwen_tts  # noqa: F401
except Exception as exc:
    print(
        "[qwen3-tts] ERROR: current Python cannot import qwen_tts.\n"
        f"[qwen3-tts] detail: {exc}\n"
        f"[qwen3-tts] fix: {sys.executable} -m pip install -U qwen-tts",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(1)
print("[qwen3-tts] qwen_tts import ok", flush=True)
try:
    import flash_attn  # noqa: F401
except Exception as exc:
    print(
        "[qwen3-tts] WARNING: flash-attn is unavailable or incompatible; service will use PyTorch SDPA.\n"
        f"[qwen3-tts] detail: {exc}\n"
        "[qwen3-tts] install a matching flash-attn build only after validating it on your GPU.",
        file=sys.stderr,
        flush=True,
    )
else:
    print("[qwen3-tts] flash-attn import ok", flush=True)
PY

"$PYTHON_BIN" qwen3_tts_service.py \
  --host "${AICUT_QWEN3_TTS_HOST:-0.0.0.0}" \
  --port "${AICUT_QWEN3_TTS_PORT:-8080}" \
  --model-id "$AICUT_QWEN3_TTS_MODEL" \
  --speaker "$AICUT_QWEN3_TTS_SPEAKER" \
  --device-map "$AICUT_QWEN3_TTS_DEVICE_MAP" \
  --dtype "$AICUT_QWEN3_TTS_DTYPE" \
  --attn "$AICUT_QWEN3_TTS_ATTN"
