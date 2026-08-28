#!/usr/bin/env bash
# ==========================================================================
#  Headless-Chrome wrapper for GPU-accelerated HyperFrames rendering.
#
#  Point services.hyperframes.browser_executable at this script (rather than
#  at Chrome itself) to render template layers on the GPU. It resolves the
#  browser binary, sets up the runtime directory Chrome needs when running as
#  root in a container, hands Chrome the NVIDIA EGL/Vulkan drivers if they are
#  installed, and forces the GPU flags on.
#
#  Environment:
#    WEDIRECTOR_CHROME       browser binary (default: the one the setup
#                            script installs under chrome/)
#    WEDIRECTOR_CHROME_LOG   debug log path
#    WEDIRECTOR_CHROME_DEBUG 1 to log every invocation
#
#  On machines without a usable GPU, do not use this wrapper: set
#  services.hyperframes.accelerated_render.enabled to false instead.
# ==========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOG=${WEDIRECTOR_CHROME_LOG:-/tmp/wedirector-chrome-wrapper.log}
DEBUG=${WEDIRECTOR_CHROME_DEBUG:-0}

CHROME_BIN="${WEDIRECTOR_CHROME:-}"
if [[ -z "$CHROME_BIN" ]]; then
  # Whatever @puppeteer/browsers dropped under chrome/ — the directory name
  # carries the version, so glob rather than hardcode it.
  for candidate in "$HERE"/chrome/*/chrome-headless-shell; do
    [[ -x "$candidate" ]] && CHROME_BIN="$candidate" && break
  done
fi
if [[ -z "$CHROME_BIN" ]]; then
  CHROME_BIN="$(command -v chrome-headless-shell || command -v google-chrome || true)"
fi
if [[ -z "$CHROME_BIN" || ! -x "$CHROME_BIN" ]]; then
  echo "browser-gpu-wrapper: no headless Chrome found." >&2
  echo "  Run: bash scripts/setup_external_tools.sh chrome" >&2
  echo "  Or:  export WEDIRECTOR_CHROME=/path/to/chrome-headless-shell" >&2
  exit 127
fi

# Chrome needs a writable XDG runtime dir; containers often do not set one.
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/wedirector-chrome-runtime}
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

if [ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}
fi

if [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}
fi

GPU_FLAGS=(
  --no-sandbox
  --ignore-gpu-blocklist
  --enable-gpu
  --enable-webgl
  --enable-accelerated-2d-canvas
  --run-all-compositor-stages-before-draw
)

if [[ "$DEBUG" == "1" ]]; then
  {
    echo "========== $(date) pid=$$ =========="
    echo "chrome_bin: $CHROME_BIN"
    "$CHROME_BIN" --version || true
    echo "args:"
    printf '%q ' "$@" "${GPU_FLAGS[@]}"
    echo
  } >> "$LOG"
fi

exec "$CHROME_BIN" "$@" "${GPU_FLAGS[@]}"
