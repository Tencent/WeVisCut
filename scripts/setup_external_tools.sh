#!/bin/bash
# Fetch the third-party runtimes WeDirector needs but does not vendor.
#
# Usage:
#   bash scripts/setup_external_tools.sh              # everything
#   bash scripts/setup_external_tools.sh ffmpeg node  # only these
#   bash scripts/setup_external_tools.sh --list
#
# Components: ffmpeg, node, hyperframes, chrome, gsap, fonts
#
# Already-installed components are skipped. Anything that fails is reported at
# the end rather than aborting the run, so one unreachable mirror does not block
# the rest. Every component has manual instructions in the matching
# external_tool/*/README.md.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXT="$REPO_ROOT/external_tool"

NODE_VERSION="${NODE_VERSION:-22.14.0}"
NODE_PLATFORM="${NODE_PLATFORM:-linux-x64}"
FFMPEG_URL="${FFMPEG_URL:-https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz}"
GSAP_URL="${GSAP_URL:-https://cdn.jsdelivr.net/npm/gsap@3/dist/gsap.min.js}"
FONT_URL="${FONT_URL:-https://github.com/notofonts/noto-cjk/raw/main/Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Bold.otf}"

ALL_COMPONENTS=(ffmpeg node hyperframes chrome gsap fonts)
FAILED=()
SKIPPED=()
INSTALLED=()

log()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
fail() { printf '    \033[31mFAILED: %s\033[0m\n' "$*"; FAILED+=("$1"); }

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    info "missing required command: $1"
    return 1
  fi
}

fetch() {
  # fetch <url> <dest>
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --connect-timeout 20 -o "$2" "$1"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$2" "$1"
  else
    info "neither curl nor wget is available"
    return 1
  fi
}

# ---------------------------------------------------------------------------
install_ffmpeg() {
  if [[ -x "$EXT/ffmpeg/ffmpeg" && -x "$EXT/ffmpeg/ffprobe" ]]; then
    SKIPPED+=(ffmpeg); info "already present"; return 0
  fi
  mkdir -p "$EXT/ffmpeg"
  local tmp; tmp="$(mktemp -d)"
  fetch "$FFMPEG_URL" "$tmp/ffmpeg.tar.xz" || { rm -rf "$tmp"; return 1; }
  tar xf "$tmp/ffmpeg.tar.xz" -C "$tmp" || { rm -rf "$tmp"; return 1; }
  local src; src="$(find "$tmp" -maxdepth 2 -name ffmpeg -type f | head -1)"
  [[ -n "$src" ]] || { info "no ffmpeg binary inside the archive"; rm -rf "$tmp"; return 1; }
  cp "$src" "$EXT/ffmpeg/ffmpeg"
  cp "$(dirname "$src")/ffprobe" "$EXT/ffmpeg/ffprobe"
  chmod +x "$EXT/ffmpeg/ffmpeg" "$EXT/ffmpeg/ffprobe"
  rm -rf "$tmp"
  "$EXT/ffmpeg/ffmpeg" -version | head -1 | sed 's/^/    /'
}

install_node() {
  if [[ -x "$EXT/node/bin/node" ]]; then
    SKIPPED+=(node); info "already present ($("$EXT/node/bin/node" --version))"; return 0
  fi
  # A system Node 22+ is just as good; do not download 650 MB for nothing.
  if command -v node >/dev/null 2>&1; then
    local major; major="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
    if [[ "$major" -ge 22 ]]; then
      SKIPPED+=(node)
      info "system node $(node --version) is new enough"
      info "set services.hyperframes.node_bin_dir to $(dirname "$(command -v node)")"
      info "or export WEDIRECTOR_NODE_BIN=$(dirname "$(command -v node)")"
      return 0
    fi
  fi
  mkdir -p "$EXT/node"
  local url="https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-${NODE_PLATFORM}.tar.xz"
  local tmp; tmp="$(mktemp -d)"
  info "downloading $url"
  fetch "$url" "$tmp/node.tar.xz" || { rm -rf "$tmp"; return 1; }
  tar xf "$tmp/node.tar.xz" -C "$EXT/node" --strip-components=1 || { rm -rf "$tmp"; return 1; }
  rm -rf "$tmp"
  "$EXT/node/bin/node" --version | sed 's/^/    node /'
}

node_bin_dir() {
  if [[ -x "$EXT/node/bin/node" ]]; then echo "$EXT/node/bin"; else dirname "$(command -v node 2>/dev/null || echo /usr/bin)"; fi
}

install_hyperframes() {
  if [[ -f "$EXT/hyperframes/cli/hyperframes/dist/cli.js" ]]; then
    SKIPPED+=(hyperframes); info "already present"; return 0
  fi
  local nbin; nbin="$(node_bin_dir)"
  export PATH="$nbin:$PATH"
  need npm || return 1
  mkdir -p "$EXT/hyperframes/cli"
  ( cd "$EXT/hyperframes/cli" \
    && [[ -f package.json ]] || echo '{"name":"wedirector-hyperframes-host","private":true}' > package.json ) || return 1
  ( cd "$EXT/hyperframes/cli" && npm install --no-audit --no-fund hyperframes ) || return 1
  ln -sfn node_modules/hyperframes "$EXT/hyperframes/cli/hyperframes"
  [[ -f "$EXT/hyperframes/cli/hyperframes/dist/cli.js" ]] || {
    info "installed, but dist/cli.js is missing - check the package layout"; return 1; }
  info "installed at external_tool/hyperframes/cli/hyperframes"
  info "optional: build the fast-path runtime tarball with"
  info "  (cd $EXT/hyperframes/cli && mkdir -p ../runtime && tar czf ../runtime/hyperframes-runtime.tar.gz node_modules)"
}

install_chrome() {
  if compgen -G "$EXT/hyperframes/chrome/*/chrome-headless-shell" > /dev/null; then
    SKIPPED+=(chrome); info "already present"; return 0
  fi
  local nbin; nbin="$(node_bin_dir)"
  export PATH="$nbin:$PATH"
  need npx || return 1
  mkdir -p "$EXT/hyperframes/chrome"
  npx --yes @puppeteer/browsers install chrome-headless-shell@stable \
      --path "$EXT/hyperframes/chrome" || return 1
  local bin; bin="$(find "$EXT/hyperframes/chrome" -name chrome-headless-shell -type f | head -1)"
  [[ -n "$bin" ]] || { info "binary not found after install"; return 1; }
  chmod +x "$bin"
  info "installed: $bin"
  if [[ "$bin" != "$EXT/hyperframes/chrome/chrome-headless-shell-linux64/chrome-headless-shell" ]]; then
    info "this differs from the config default; export WEDIRECTOR_CHROME=$bin"
  fi
}

install_gsap() {
  local dest="$EXT/hyperframes/vendor/gsap.min.js"
  if [[ -s "$dest" ]]; then SKIPPED+=(gsap); info "already present"; return 0; fi
  mkdir -p "$(dirname "$dest")"
  fetch "$GSAP_URL" "$dest" || return 1
  info "installed ($(wc -c < "$dest") bytes) - review the GSAP licence before redistributing"
}

install_fonts() {
  local dest="$EXT/gold_template/fonts/NotoSerifSC-Bold.otf"
  if [[ -s "$dest" ]]; then SKIPPED+=(fonts); info "already present"; return 0; fi
  mkdir -p "$(dirname "$dest")"
  fetch "$FONT_URL" "$dest" || return 1
  # A failed download can leave an HTML error page behind; an OTF is far bigger.
  if [[ "$(wc -c < "$dest")" -lt 100000 ]]; then
    info "downloaded file is too small to be a font; removing"
    rm -f "$dest"; return 1
  fi
  info "installed $(basename "$dest") ($(wc -c < "$dest") bytes)"
}

# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--list" ]]; then
  printf '%s\n' "${ALL_COMPONENTS[@]}"
  exit 0
fi

COMPONENTS=("$@")
[[ ${#COMPONENTS[@]} -eq 0 ]] && COMPONENTS=("${ALL_COMPONENTS[@]}")

for component in "${COMPONENTS[@]}"; do
  log "$component"
  if ! declare -F "install_$component" >/dev/null; then
    fail "$component (unknown component)"
    continue
  fi
  if "install_$component"; then
    # Skips already recorded themselves.
    [[ " ${SKIPPED[*]:-} " == *" $component "* ]] || INSTALLED+=("$component")
  else
    fail "$component"
  fi
done

log "summary"
[[ ${#INSTALLED[@]} -gt 0 ]] && info "installed: ${INSTALLED[*]}"
[[ ${#SKIPPED[@]}   -gt 0 ]] && info "skipped:   ${SKIPPED[*]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  info "failed:    ${FAILED[*]}"
  info "see external_tool/<component>/README.md for manual instructions"
fi

log "next"
info "bash run_studio.sh --check     # verify every resolved path"
info "Model weights and the mmfinder client are separate; see docs/external_tools.md"

[[ ${#FAILED[@]} -eq 0 ]] || exit 1
