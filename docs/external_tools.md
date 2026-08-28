# External tools, model weights, and large files

The repository is about 12 MB. Everything heavy is downloaded, because it is
either third-party with its own licence or too large to version sensibly.

Fast path:

```bash
bash scripts/setup_external_tools.sh          # ~1.1 GB of runtimes
hf download <org>/WeDirector-9B --local-dir checkpoints/wedirector-9b
bash run_studio.sh --check                    # verify every resolved path
```

The rest of this document is the manual version, plus what to do when you want a
component somewhere else or not at all.

---

## 1. Inventory

| Component | Size | Needed for | Config key |
|---|---|---|---|
| Policy weights | ~18 GB | Generation | `model.profiles.<name>.base_model` |
| Node.js >= 22 | ~650 MB | Template rendering | `services.hyperframes.node_bin_dir` |
| HyperFrames runtime tarball | ~320 MB | Fast template rendering | `services.hyperframes.runtime_archive` |
| HyperFrames CLI | ~12 MB | Template rendering | `services.hyperframes.hf_bin` |
| chrome-headless-shell | ~260 MB | Template rendering | `services.hyperframes.browser_path` |
| ffmpeg + ffprobe (static) | ~80 MB | Everything audio/video | `services.ffmpeg.*` |
| CJK subtitle font | ~15 MB | Subtitle burn-in | `render.subtitles.fontfile` |
| GSAP | ~70 KB | Template rendering | `services.hyperframes.gsap_local_path` |
| ms-swift fork (`qwen3_5`) | ~200 MB | Loading the weights | `model.ms_swift_path` |
| mmfinder client | ~720 MB | Only the `mmfinder` provider | env `MMFINDER_ROOT` |

Every default is `${REPO_ROOT}/external_tool/...`, so the layout below works with
no configuration at all:

```
WeDirector/
├── checkpoints/wedirector-9b/
└── external_tool/
    ├── ffmpeg/{ffmpeg,ffprobe}
    ├── node/bin/node
    ├── gold_template/fonts/NotoSerifSC-Bold.otf
    ├── hyperframes/
    │   ├── cli/hyperframes/dist/cli.js
    │   ├── runtime/hyperframes-runtime.tar.gz
    │   ├── chrome/chrome-headless-shell-linux64/chrome-headless-shell
    │   └── vendor/gsap.min.js
    └── mmfinder/WeAgent_WeStoryCut/      (optional)
```

---

## 2. Model weights

```bash
pip install -U "huggingface_hub[cli]"
hf download <org>/WeDirector-9B --local-dir checkpoints/wedirector-9b
```

Somewhere else? Either export a variable or edit the profile:

```bash
export WEDIRECTOR_MODEL_PATH=/data/models/wedirector-9b
```

```yaml
model:
  active_profile: wedirector_9b
  profiles:
    wedirector_9b:
      base_model: /data/models/wedirector-9b
      adapter: ""
      infer_backend: vllm       # or transformers
```

Multiple profiles can coexist; switch with `model.active_profile` or
`AICUT_MODEL_PROFILE=<name>` without editing the file.

### The `qwen3_5` model type

The policy is a custom architecture. Loading it needs a `ms-swift` build that
registers `qwen3_5` in `swift/model/constant.py`, and a `transformers` that knows
`Qwen3_5ForConditionalGeneration`. If `pip install ms-swift` gives you one that
does not:

```bash
export AICUT_MS_SWIFT_PATH=/path/to/ms-swift
```

or set `model.ms_swift_path`. The path is prepended to `sys.path`, so a plain
checkout is enough — no install step.

### Training your own

DAPO emits sharded verl FSDP checkpoints (`global_step_N/actor/`), which vLLM
cannot read. Merge them to HF format first, then point a profile at the merged
directory. The released checkpoint is already merged.

---

## 3. Runtimes

Each of these has its own README with more detail:
[ffmpeg](../external_tool/ffmpeg/README.md) ·
[node](../external_tool/node/README.md) ·
[hyperframes](../external_tool/hyperframes/README.md) ·
[mmfinder](../external_tool/mmfinder/README.md).

```bash
# One component at a time
bash scripts/setup_external_tools.sh ffmpeg
bash scripts/setup_external_tools.sh --list
```

Overrides, if you already have these installed:

| Variable | Points at |
|---|---|
| `WEDIRECTOR_FFMPEG` / `WEDIRECTOR_FFPROBE` | ffmpeg / ffprobe binaries |
| `WEDIRECTOR_NODE_BIN` | Directory containing `node` |
| `WEDIRECTOR_CHROME` | `chrome-headless-shell` or a Chrome/Chromium binary |
| `WEDIRECTOR_SUBTITLE_FONT` | Subtitle `.otf` / `.ttf` |
| `WEDIRECTOR_GOLD_TEMPLATE` | Template library root |

### Subtitle font

`render.subtitles.fontfile` must exist — ffmpeg's `drawtext` filter fails hard
without it, mid-render. Any CJK font with a bold weight works; the reference
deployment uses Noto Serif SC Bold. The setup script fetches it from the
[noto-cjk](https://github.com/notofonts/noto-cjk) repository (SIL Open Font
License).

---

## 4. Templates: `external_tool/gold_template/`

The Gold Template library composes decorations from three independent layers
(structure x motion x visual), which is what lets a small registry cover a wide
range of shots. `CONTRACT.md` in that directory is the authoritative spec.

The repository ships the registry, prompts, assembler, viewer and sample footage
(~6 MB). Two directories are excluded:

- `fonts/` (~52 MB) — the full font set. `setup_external_tools.sh` installs just
  the subtitle font, which is all the pipeline requires. Templates that name other
  families fall back to whatever the browser resolves.
- `renders/` (~150 MB) — pre-rendered previews. Nothing at runtime reads them.

To use your own template library, point `paths.gold_template_root` at it. It must
satisfy `CONTRACT.md`.

---

## 5. Services you run yourself

**TTS.** `services.tts.service_url`, default `http://127.0.0.1:7865`. The
reference deployment runs Qwen3-TTS; any server with a compatible HTTP interface
works. Without it, `Tool_GenerateAudio` falls back to `edge-tts`
(`pip install edge-tts`, free, no key), and failing that to duration estimation
only — enough to exercise planning, not enough to produce a watchable cut.

**Footage and BGM.** Nothing to run: the default provider searches the open web.
See [providers.md](providers.md) to attach your own library.

**Internal LLM calls.** Some tools make auxiliary LLM calls (clip selection,
severity ranking, slot filling). At startup the studio rewrites `config.llm` to
its own `/v1` endpoint, so the loaded policy model serves those too and there is
no second model to deploy. Running the pipeline standalone, set
`WEDIRECTOR_LLM_BASE_URL` / `WEDIRECTOR_LLM_MODEL`.

---

## 6. Running with less

| Skip | How | You lose |
|---|---|---|
| GPU and weights | `bash run_studio.sh --no-model` | Generation and re-render. UI, APIs and editing panels work |
| Template rendering | `services.hyperframes.enabled: false` | The animated decoration layer. Footage, narration, subtitles and BGM still render |
| GPU-accelerated rendering | `services.hyperframes.accelerated_render.enabled: false` | Speed only |
| BGM | `services.bgm.enabled: false` | Music track |
| Conversational editing | `chat_edit.enabled: false` | The chat tab; manual editing remains |

`bash run_studio.sh --check` reports missing paths as warnings rather than
errors, so a partial installation still starts and tells you what is absent.
