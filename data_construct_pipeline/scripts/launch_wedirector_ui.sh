#!/usr/bin/env bash
#
# Start the campaign console for batch trajectory annotation.
#
#   bash data_construct_pipeline/scripts/launch_wedirector_ui.sh
#
# First run builds the template slot-schema cache and registers the machine
# pool; later runs skip that. Open http://<host>:<port> to add workers,
# pick a mode, and start.
#
# Environment:
#   PORT=8099
#   HOST=0.0.0.0
#   ENDPOINTS="host-a:8000,host-b:8000"   initial machine pool (optional)
#   DATASET_ROOT=...                      directory with scripts/ or manifest.jsonl
#   OUTPUT_ROOT=...                       where episodes land
#   AICUT_PIPELINE_ROOT=...               WeDirector / environment checkout
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$PKG_ROOT/.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/workspace/scripts}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/workspace/data_construct}"
CONFIG="${CONFIG:-$PKG_ROOT/config/wedirector_260821.yaml}"
GOLD_ROOT="${GOLD_ROOT:-${WEDIRECTOR_GOLD_TEMPLATE:-$REPO_ROOT/external_tool/gold_template}}"
PIPELINE_ROOT="${AICUT_PIPELINE_ROOT:-${WEDIRECTOR_ROOT:-$REPO_ROOT}}"
ENDPOINTS="${ENDPOINTS:-}"
PORT="${PORT:-8099}"
HOST="${HOST:-0.0.0.0}"

export AICUT_PIPELINE_ROOT="$PIPELINE_ROOT"
export WEDIRECTOR_ROOT="${WEDIRECTOR_ROOT:-$REPO_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PKG_ROOT/src"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${TMPDIR:-/tmp}/longvideo_polar_pycache}"
mkdir -p "$PYTHONPYCACHEPREFIX" 2>/dev/null || true

if [[ -n "$ENDPOINTS" ]]; then
  IFS=',' read -ra _EPS <<< "$ENDPOINTS"
  _NOPROXY="${no_proxy:-127.0.0.1,localhost,0.0.0.0}"
  for ep in "${_EPS[@]}"; do
    host="${ep%%:*}"
    [[ -n "$host" ]] && _NOPROXY="$_NOPROXY,$host"
  done
  export no_proxy="$_NOPROXY"
  export NO_PROXY="$_NOPROXY"
fi

PY="${PYTHON:-python3}"

echo "==> Checking dependencies"
if ! "$PY" -c "import longvideo_polar" 2>/dev/null; then
  echo "    Installing longvideo_polar in editable mode…"
  (cd "$PKG_ROOT" && "$PY" -m pip install -q -e ".[web]")
fi
if ! "$PY" -c "import fastapi, uvicorn" 2>/dev/null; then
  echo "    Installing web extras (fastapi + uvicorn)…"
  "$PY" -m pip install -q fastapi uvicorn
fi

if [[ ! -f "$OUTPUT_ROOT/_state/slot_schema_cache.json" ]]; then
  echo "==> First-time prepare: template slot cache + machine registry"
  ENDPOINT_ARGS=()
  if [[ -n "$ENDPOINTS" ]]; then
    IFS=',' read -ra _EPS <<< "$ENDPOINTS"
    for ep in "${_EPS[@]}"; do
      [[ -n "$ep" ]] && ENDPOINT_ARGS+=(--endpoint "$ep")
    done
  fi
  "$PY" -m longvideo_polar.cli prepare \
    --config "$CONFIG" \
    --output-root "$OUTPUT_ROOT" \
    --gold-template-root "$GOLD_ROOT" \
    "${ENDPOINT_ARGS[@]}"
else
  echo "==> Template slot cache already present, skipping prepare"
fi

echo
echo "==> Starting campaign console"
echo "    dataset : $DATASET_ROOT"
echo "    output  : $OUTPUT_ROOT"
echo "    config  : $CONFIG"
echo "    url     : http://${HOST}:${PORT}"
echo

ENDPOINT_ARGS=()
if [[ -n "$ENDPOINTS" ]]; then
  IFS=',' read -ra _EPS <<< "$ENDPOINTS"
  for ep in "${_EPS[@]}"; do
    [[ -n "$ep" ]] && ENDPOINT_ARGS+=(--endpoint "$ep")
  done
fi

exec "$PY" -m longvideo_polar.cli ui \
  --config "$CONFIG" \
  --dataset-root "$DATASET_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --host "$HOST" \
  --port "$PORT" \
  "${ENDPOINT_ARGS[@]}"
