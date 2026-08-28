# external_tool — third-party runtimes and asset back ends

Everything WeDirector needs that is not Python source lives here. The directory
ships **code and manifests only**; the large binaries and media are downloaded
at setup time, because they are third-party artifacts with their own licences
and together weigh about 1.6 GB.

Run this once after cloning:

```bash
bash scripts/setup_external_tools.sh
```

Then verify with `bash run_studio.sh --check`. Each subdirectory below has its
own README with manual instructions if the script cannot reach the internet.

## What is in the repository

| Path | Size | Contents |
|---|---|---|
| `asset_retrieval_tool/` | ~200 KB | Open-web footage and BGM search (stdlib only). The default provider. |
| `gold_template/` | ~6 MB | Template library: registry, prompts, assembler, viewer, sample footage |
| `hyperframes/browser-gpu-wrapper.sh` | 1 KB | Wrapper that gives headless Chrome a GPU |
| `configs/` | — | Optional decoration-hint YAMLs (see that README) |

## What you download

| Path | Size | What | Needed for |
|---|---|---|---|
| `node/` | ~650 MB | Node.js >= 22 runtime | HyperFrames CLI |
| `hyperframes/cli/` | ~12 MB | `hyperframes` npm package | Template rendering |
| `hyperframes/runtime/` | ~320 MB | Self-contained runtime tarball | Fast rendering |
| `hyperframes/chrome/` | ~260 MB | `chrome-headless-shell` | Template rendering |
| `hyperframes/vendor/gsap.min.js` | ~70 KB | GSAP animation library | Template rendering |
| `ffmpeg/` | ~80 MB | Static `ffmpeg` + `ffprobe` | Everything audio/video |
| `gold_template/fonts/` | ~52 MB | CJK subtitle fonts | Subtitle burn-in |
| `mmfinder/` | ~720 MB | Internal retrieval client | Only the `mmfinder` provider |

Model weights are separate; see `docs/external_tools.md`.

## Configuration mapping

Each item is referenced from exactly one place in `studio_config.yaml`, and all
of those defaults are `${REPO_ROOT}/external_tool/...`. Put a tool somewhere
else and you only change that one line (or export the matching variable):

| Path | Config key | Env override |
|---|---|---|
| `node/bin` | `services.hyperframes.node_bin_dir` | `WEDIRECTOR_NODE_BIN` |
| `hyperframes/cli/hyperframes/dist/cli.js` | `services.hyperframes.hf_bin` | — |
| `hyperframes/runtime/hyperframes-runtime.tar.gz` | `services.hyperframes.runtime_archive` | — |
| `hyperframes/chrome/.../chrome-headless-shell` | `services.hyperframes.browser_path` | `WEDIRECTOR_CHROME` |
| `hyperframes/vendor/gsap.min.js` | `services.hyperframes.gsap_local_path` | — |
| `ffmpeg/ffmpeg` | `services.ffmpeg.ffmpeg_path` | `WEDIRECTOR_FFMPEG` |
| `ffmpeg/ffprobe` | `services.ffmpeg.ffprobe_path` | `WEDIRECTOR_FFPROBE` |
| `gold_template/` | `paths.gold_template_root` | `WEDIRECTOR_GOLD_TEMPLATE` |
| `gold_template/fonts/NotoSerifSC-Bold.otf` | `render.subtitles.fontfile` | `WEDIRECTOR_SUBTITLE_FONT` |
| `mmfinder/WeAgent_WeStoryCut` | — | `MMFINDER_ROOT` |

## Notes on individual components

**HyperFrames** renders the animated template layer for spotlight shots. The
runtime tarball is the primary mechanism: it is unpacked once into
`runtime_cache_dir` (default `/tmp`) and reused, which avoids reading thousands
of small `node_modules` files off a network filesystem on every render. The CLI
package under `cli/` is the fallback when no tarball is present.

**ffprobe** is often dynamically linked against libraries a container may not
have. Prefer a static build — `setup_external_tools.sh` fetches one — or point
`services.ffmpeg.ffprobe_path` at your distribution's `/usr/bin/ffprobe`.

**mmfinder** is a client for an internal embedding-recall service and is not
reachable from the public internet. It is documented for completeness; the
default configuration uses `asset_retrieval_tool` instead, and
`docs/providers.md` explains how to plug in your own library.
