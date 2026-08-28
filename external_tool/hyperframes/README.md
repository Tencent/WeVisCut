# HyperFrames renderer

HyperFrames renders the animated template layer for spotlight shots: it loads a
Gold Template HTML composition in headless Chrome, steps it frame by frame, and
writes a video (or a transparent WebM overlay when render splitting is on).

Four pieces are needed. `bash scripts/setup_external_tools.sh` installs all of
them; the manual steps are below.

```
external_tool/hyperframes/
├── cli/hyperframes/dist/cli.js     # the CLI entry point
├── runtime/hyperframes-runtime.tar.gz   # self-contained runtime (optional, fast path)
├── chrome/chrome-headless-shell-linux64/chrome-headless-shell
├── vendor/gsap.min.js
└── browser-gpu-wrapper.sh          # in the repo already
```

## 1. CLI package

```bash
cd external_tool/hyperframes
mkdir -p cli && cd cli
npm install --prefix . hyperframes
# leaves node_modules/hyperframes; expose it where the config expects it
ln -sfn node_modules/hyperframes hyperframes
node hyperframes/dist/cli.js --help
```

Config key: `services.hyperframes.hf_bin`.

## 2. Runtime tarball (optional but recommended)

A gzipped tar of a fully installed HyperFrames tree, including every
`node_modules` dependency. On first render it is unpacked into
`runtime_cache_dir` (default `/tmp/wedirector_hyperframes_runtime`) and reused
from there.

This exists for a specific reason: rendering reads thousands of small files, and
doing that from a network filesystem on every shot dominates the wall clock. One
local unpack removes it.

Build one from a working install:

```bash
cd external_tool/hyperframes/cli
tar czf ../runtime/hyperframes-runtime.tar.gz node_modules
```

Config key: `services.hyperframes.runtime_archive`. Leave the path pointing at a
missing file and the renderer falls back to `hf_bin`.

## 3. Headless Chrome

```bash
cd external_tool/hyperframes
npx --yes @puppeteer/browsers install chrome-headless-shell@stable --path "$PWD/chrome"
```

Point `services.hyperframes.browser_path` at the resulting binary, or export
`WEDIRECTOR_CHROME`. A system Chrome or Chromium also works.

`browser-gpu-wrapper.sh` wraps that binary with the flags needed for GPU-backed
rendering and is used by `accelerated_render.browser_path`. On a machine without
a usable GPU, set `services.hyperframes.accelerated_render.enabled: false`.

## 4. GSAP

Template HTML animates with GSAP. Vendoring it locally keeps rendering
deterministic and offline:

```bash
mkdir -p external_tool/hyperframes/vendor
curl -L -o external_tool/hyperframes/vendor/gsap.min.js \
  https://cdn.jsdelivr.net/npm/gsap@3/dist/gsap.min.js
```

Config key: `services.hyperframes.gsap_local_path`. GSAP ships under its own
licence — review it before redistributing the file.

## Turning rendering off

Template rendering is the heaviest dependency in the stack. To evaluate the
agent's planning and retrieval behaviour without it:

```yaml
services:
  hyperframes:
    enabled: false
```

Shots then render through the ffmpeg path only: footage, narration, subtitles and
BGM still work, but there is no animated decoration layer.
