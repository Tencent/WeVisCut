#!/bin/bash
# WeVisCut Studio — autonomous long-form video generation + post editing
#
# Usage:
#   bash run_studio.sh                        # use studio_config.yaml
#   bash run_studio.sh my_config.yaml         # use another config
#   bash run_studio.sh --port 8096            # override the port
#   bash run_studio.sh --check                # validate config only (no GPU)
#   bash run_studio.sh --no-model             # serve without loading weights
#
# Every external interface (asset library, TTS, BGM, HyperFrames, model and
# template paths) lives in studio_config.yaml. This script hardcodes none.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="studio_config.yaml"
PORT=""
CHECK_ONLY=0
SKIP_MODEL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)     PORT="$2"; shift 2 ;;
    --check)    CHECK_ONLY=1; shift ;;
    --no-model) SKIP_MODEL=1; shift ;;
    --help|-h)  sed -n '2,13p' "$0"; exit 0 ;;
    -*)         echo "unknown flag: $1" >&2; exit 2 ;;
    *)          CONFIG="$1"; shift ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: no usable python found; set PYTHON_BIN" >&2
  exit 2
fi

if [[ ! "$CONFIG" = /* ]]; then
  CONFIG="$SCRIPT_DIR/$CONFIG"
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG" >&2
  exit 2
fi

# Self-contained layout: server/ (inference engine) + pipeline/ (tools) +
# studio/ (backend) + aicut_agent/ (deepagents integration) all live here, so
# putting the repo root on PYTHONPATH is enough.
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
export AICUT_STUDIO_CONFIG="$CONFIG"

if [[ -z "$PORT" ]]; then
  PORT=$("$PYTHON_BIN" -c "
import yaml
try:
    print((yaml.safe_load(open('$CONFIG')) or {}).get('server', {}).get('port', 8081))
except Exception:
    print(8081)
" 2>/dev/null)
fi
PORT="${PORT:-8081}"
export AICUT_STUDIO_PORT="$PORT"

# --check: validate the config without loading weights.
if [[ "$CHECK_ONLY" == "1" ]]; then
  exec "$PYTHON_BIN" -m studio.config --check --config "$CONFIG"
fi

if [[ "$SKIP_MODEL" == "1" ]]; then
  export AICUT_STUDIO_SKIP_MODEL=1
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":${PORT}\\b"; then
  echo "ERROR: port ${PORT} is in use; try --port $((PORT + 1))" >&2
  exit 2
fi

# The pipeline talks to loopback services (TTS, the in-process /v1 endpoint) and
# to retrieval back ends that may sit on a private network. An HTTP proxy in the
# environment breaks both, so drop the proxy variables for this process tree.
# Set WEVISCUT_KEEP_PROXY=1 if your services are only reachable via a proxy.
if [[ "${WEVISCUT_KEEP_PROXY:-0}" != "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
fi

if [[ "$SKIP_MODEL" != "1" ]]; then
  if ! "$PYTHON_BIN" - <<'PY'
from importlib.util import find_spec
required = ("torch", "swift", "uvicorn", "fastapi", "deepagents", "langchain", "langgraph")
missing = [m for m in required if find_spec(m) is None]
if missing:
    raise SystemExit(
        f"missing modules: {missing}\n"
        "Install them, or pass --no-model to run the UI without the policy model."
    )
print("python environment preflight OK")
PY
  then
    echo "ERROR: python environment preflight failed ($PYTHON_BIN)" >&2
    exit 2
  fi
fi

# Print the resolved interfaces, so "I edited the config but nothing changed"
# is visible immediately instead of three tool calls into a run.
"$PYTHON_BIN" -m studio.config --summary --config "$CONFIG" || true

echo "============================================"
echo "  WeVisCut Studio"
echo "============================================"
echo "  Config : $CONFIG"
echo "  Port   : $PORT"
echo "  GPU    : ${CUDA_VISIBLE_DEVICES:-<skipped>}"
echo "  URL    : http://0.0.0.0:$PORT/"
echo "============================================"
if [[ "$SKIP_MODEL" == "1" ]]; then
  echo "Weights skipped (--no-model): UI works, generation and re-render do not."
else
  echo "Loading weights in the background (~30-60s); the UI opens immediately."
fi
echo ""

exec "$PYTHON_BIN" -m uvicorn studio.app:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --workers 1 \
  --ws-ping-interval 30 \
  --ws-ping-timeout 60 \
  --log-level info
