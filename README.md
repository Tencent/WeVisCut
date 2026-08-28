# WeDirector

**Long-horizon tool agents for end-to-end video editing.**

WeDirector turns a plain narration script into a finished, multi-minute edited
video. A single 9B policy model, trained with SFT then DAPO reinforcement
learning, drives the whole thing: it segments the script, plans shots,
synthesises narration, retrieves footage, chooses and fills animated templates,
picks background music, reviews its own output, repairs what it does not like,
and stitches the final cut — one tool call at a time, with no scripted pipeline
telling it what to do next.

A typical run is around 12 tool calls over several minutes of video. There is no
hand-written state machine: the trajectory is the model's.

[English](README.md) · [中文](README_zh.md)

---

## What is in this repository

| Piece | Where |
|---|---|
| The 12-tool video-editing environment | `pipeline/` |
| Training-aligned inference engine (state serialisation, tool schemas, parsing) | `server/` |
| Agent harness on [deepagents](https://github.com/langchain-ai/deepagents) / LangGraph | `aicut_agent/` |
| FastAPI backend, editing and conversational-edit APIs | `studio/` |
| Browser UI: live timeline, manual editing, chat editing | `static/` |
| **Pluggable footage / BGM providers** | `pipeline/providers/` |
| Open-web footage and CC-music retrieval | `external_tool/asset_retrieval_tool/` |
| Gold Template library (structure x motion x visual) | `external_tool/gold_template/` |

Model weights and the large third-party runtimes are downloaded separately —
see [Installation](#installation).

### Documentation

| Guide | Covers |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | How the agent loop, runtime and guardrails fit together, and why |
| [`docs/configuration.md`](docs/configuration.md) | Every `studio_config.yaml` field and environment variable |
| [`docs/providers.md`](docs/providers.md) | Connecting your own footage and music sources |
| [`docs/external_tools.md`](docs/external_tools.md) | The large downloads: what, where, how big |
| [`docs/huggingface.md`](docs/huggingface.md) | Consuming and publishing the checkpoint |

---

## How it works

```
narration script
  → POST /api/runs
  → one AICutRuntime per run (owns the authoritative editing state)
  → create_deep_agent(model=AICutChatModel, tools=12, system_prompt=training prompt,
                      middleware=[AICutGuardrailMiddleware])
  → LangGraph ReAct loop:
       AICutChatModel   rebuilds a single-turn state JSON from the authoritative
                        runtime state, calls the policy, parses one tool call
       Guardrail        preconditions / repetition / budget (rejects or halts;
                        never rewrites the model's decision)
       Tool             executes against the real environment
       Tool_StitchVideo succeeds → Command(goto=END)
  → events streamed to the browser over WebSocket
  → finished cut written to workspace/projects/<project_id>/
```

Two design choices are worth calling out, because they are what makes the agent
behave the same in deployment as it did in training.

**The runtime is the state authority.** On every step the chat model *discards*
the harness's accumulated message history and rebuilds a single-turn user
message from the runtime's authoritative state. Tool schemas are regenerated from
the same source of truth the training data generator used. What the model sees at
inference is therefore byte-identical to what it saw during training — no drift
from a growing conversation, no divergence between two schema definitions.

**Guardrails observe, they do not correct.** The middleware can reject a call
whose preconditions are unmet, or stop a run that has exhausted its budget, but
it never substitutes a different tool call. Illegal calls are fed back as
observations and the model corrects itself. Every "helpful" server-side override
is off by default (`agent.autonomy.*` in the config), because a policy that has
been trained to recover is measured on whether it actually does.

### The 12 tools

| Tool | What it does |
|---|---|
| `Tool_SegmentArticle` | Split the script into chapters |
| `Tool_PlanShots` | Plan shots for one chapter: narration, visual query, template intent |
| `Tool_GenerateAudio` | TTS narration; establishes each shot's real duration |
| `Tool_RetrieveVisual` | Find footage matching the shot's visual query |
| `Tool_DecorateShot` | Pick a template triple, fill slots, verify, repair |
| `Tool_TimelineCompose` | Render one shot: footage + narration + subtitles + template layer |
| `Tool_MusicSearch` | Build a BGM playlist covering the timeline |
| `Tool_TextQuickJudge` | Cheap text-only quality gate |
| `Tool_VLMRenderJudge` | Vision review of rendered shots |
| `Tool_SelfReflect` | Decide what to repair next |
| `Tool_ModalProbe` | Preflight the external services |
| `Tool_StitchVideo` | Normalise and concatenate every shot into the final cut |

---

## Installation

### Requirements

- Python **>= 3.11**
- For generation: an NVIDIA GPU with enough memory for a 9B model in bf16
  (~24 GB). Frontend and API development need no GPU (`--no-model`).
- Node **>= 22** and ffmpeg, both installable via the setup script below.

### 1. Python dependencies

```bash
git clone https://git.woa.com/chenyumu/WeDirector.git
cd WeDirector

pip install -r requirements.txt

# Inference only (GPU machines). Install torch matching your CUDA version first:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu.txt
```

`ms-swift` must be a build that registers the `qwen3_5` model type. If yours does
not, point `model.ms_swift_path` at a checkout that does, or export
`AICUT_MS_SWIFT_PATH`.

### 2. External runtimes

```bash
bash scripts/setup_external_tools.sh
```

This fetches ffmpeg, Node, the HyperFrames renderer, headless Chrome, GSAP and
the subtitle font into `external_tool/`. Roughly 1.1 GB. Components already
present are skipped, and each one has manual instructions in the matching
`external_tool/*/README.md` if a download fails.

### 3. Model weights

```bash
pip install -U "huggingface_hub[cli]"
hf download <org>/WeDirector-9B --local-dir checkpoints/wedirector-9b
```

Then either leave the default (`checkpoints/wedirector-9b`) or point at your own
copy:

```bash
export WEDIRECTOR_MODEL_PATH=/data/models/wedirector-9b
```

Full details, including sizes and how to skip individual components, are in
[`docs/external_tools.md`](docs/external_tools.md).

### 4. Check and run

```bash
bash run_studio.sh --check     # validate every resolved path; no GPU needed
bash run_studio.sh             # start on the configured port
bash run_studio.sh --no-model  # UI + API only, no weights, no GPU
```

Open `http://<host>:8081/`. Weights load in the background over 30-60 seconds;
the UI opens immediately and reports progress.

---

## Configuration

`studio_config.yaml` is the only file you need to edit. Nothing else reads a
hardcoded path. Values support `${VAR}` and `${VAR:-fallback}`, and
`${REPO_ROOT}` is injected automatically, so a fresh clone runs without
exporting anything.

| Section | Controls |
|---|---|
| `model` | Weight profiles, decoding, thinking mode, the training system prompt |
| `agent` | Step and cost ceilings, and the autonomy switches |
| `services.retrieval_provider` | **Where footage comes from** |
| `services.bgm_provider` | **Where music comes from** |
| `services.providers` | Provider settings and Python plugins |
| `services.tts` | Narration service |
| `services.hyperframes` | Renderer, browser, GPU acceleration |
| `services.ffmpeg` | ffmpeg / ffprobe binaries |
| `paths` | Asset library, template library, output roots |
| `render` | Subtitles, layer-split acceleration, stitching |
| `editor`, `chat_edit` | Post-production and conversational editing |

Inspect what actually took effect:

```bash
python3 -m studio.config --summary     # resolved interfaces + registered providers
python3 -m studio.config --dump pipeline
```

See [`docs/configuration.md`](docs/configuration.md) for the full reference and
the environment-variable list.

---

## Plugging in your own footage and music

Retrieval is the one part of the system that has to be wired into whatever media
you own, so it is a first-class extension point. Three options, in increasing
order of effort.

### Out of the box: open-web search

The default. Searches Bilibili / YouTube for footage and Internet Archive
netlabels for CC-licensed instrumental BGM. No extra service, no API key for the
Bilibili path.

```yaml
services:
  retrieval_provider: asset_retrieval_tool
```

### Config only: your own HTTP endpoint

If you already have a search service, describe it in YAML — no code:

```yaml
services:
  retrieval_provider: http_video
  bgm_provider: http_bgm
  providers:
    http_video:
      url: https://footage.internal/search
      headers:
        Authorization: "Bearer ${FOOTAGE_TOKEN}"
      results_path: data.items
      field_map:
        clip_path: playback_url
        duration_sec: seconds
```

Response fields already named `url`, `path`, `title` or `duration` are matched
automatically; `field_map` is only for the ones that are not.

### A folder of files

```yaml
services:
  retrieval_provider: local_folder_video
  bgm_provider: local_folder_bgm
  providers:
    local_folder_video:
      roots: [/data/my_footage]
    local_folder_bgm:
      roots: [/data/my_music]
```

Clips are matched on file name, parent directory names, and optional
`<name>.json` sidecars carrying `caption` / `tags` / `duration_sec`.

### Python: anything else

```python
# my_providers.py
from pipeline.providers import VideoQuery, register_video_provider

@register_video_provider("my_footage")
def search(query: VideoQuery) -> list[dict]:
    hits = my_index.search(query.query, k=query.top_k)
    return [
        {"clip_path": h.path, "score": h.score,
         "duration_sec": h.seconds, "description": h.caption}
        for h in hits
    ]
```

```yaml
services:
  retrieval_provider: my_footage
  providers:
    plugins: [my_providers]          # module name or /path/to/my_providers.py
```

Return whatever fields you have; the registry normalises them and fills in the
rest. A provider that raises or returns nothing is treated as "no candidates" and
the next backend runs, so a flaky source degrades instead of failing the run.

[`docs/providers.md`](docs/providers.md) has the full contract: the exact schemas,
what each `VideoQuery` / `BgmQuery` field means, how the duration and orientation
filters treat your results, and how to test a provider offline.

---

## Using the studio

1. **Generate.** Paste a narration script, press start. Eight live tracks
   (segments, shots, narration, subtitles, footage, spotlight, BGM, render) fill
   in as the agent works, alongside its thinking, tool calls, and raw output.
   Pause, resume, or cancel at any point.
2. **Timeline.** Multi-track view of every shot: position, duration, footage,
   template. Click to select.
3. **Manual editing.** On a selected shot: rewrite narration (re-runs TTS),
   rewrite subtitles (does not), swap footage or template, delete, reorder,
   adjust BGM. Re-render touches only the shots you changed and re-stitches.
4. **Conversational editing.** "Change the opening line to ...", "replace the
   visual at 0:15". The model locates the target and proposes edits; destructive
   ones need confirmation.
5. **History.** Past runs and finished projects, with version rollback.

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/status` | Model state, active runs |
| GET | `/api/config` | Config the frontend needs (tracks, features, editor) |
| POST | `/api/runs` | Start a run: `{input_text}` → `{run_id}` |
| GET | `/api/runs` · `/api/runs/{id}` | List / detail |
| POST | `/api/runs/{id}/pause` · `resume` · `cancel` | Control |
| DELETE | `/api/runs/{id}` | Delete a run record |
| WS | `/ws/{run_id}` | Live event stream |
| GET | `/api/projects` · `/api/project?dir=` | Projects |
| POST | `/api/project/ops` · `regenerate` | Apply edits · re-render |
| GET | `/api/project/versions?dir=` · POST `/api/project/restore` | Versions |
| POST | `/api/project/chat` | Conversational editing |
| GET | `/api/media?path=` | Serve media (allow-listed roots only) |
| POST | `/v1/chat/completions` | OpenAI-compatible; the pipeline's internal LLM calls reuse the same policy model |

WebSocket events: `run_started`, `stage_update`, `thinking`, `tool_call`,
`tool_result`, `progress`, `preview_update`, `console_log`, `error`,
`pipeline_complete`, `run_complete`, `run_paused`, `run_resumed`.

---

## Tests

Everything here runs offline, without a GPU:

```bash
python3 tests/test_studio_offline.py           # config, tracks, editor, chat parsing, guardrails
python3 -m pytest tests/test_aicut_agent.py    # deepagents integration (mock loader + dispatch)
python3 -m pytest tests/test_retrieval_provider_switch.py   # provider routing
python3 -m pytest tests/test_providers.py      # provider registry and bundled providers
python3 -m pytest tests/test_thinking_parse.py # think/tool_call parsing
node tests/test_frontend.js                    # frontend (needs node)
```

`test_aicut_agent.py` asserts the properties that keep inference aligned with
training: the model sees exactly the 12 trained tools (deepagents' built-ins are
removed), format conversion round-trips, preconditions block correctly, a
successful stitch terminates the graph, and plain-text conversation still works.

---

## Troubleshooting

| Symptom | Where to look |
|---|---|
| Startup reports missing modules | `pip install -r requirements.txt`; GPU machines also need `requirements-gpu.txt` |
| Model stuck on "loading" | Check the log; confirm `model.profiles.<active>.base_model` exists and the GPU is visible |
| A tool fails repeatedly | The frontend's terminal panel has the real error. Usually an external service (TTS / retrieval / renderer) is down |
| `retrieval_provider ... is not registered` | Typo, or the plugin failed to import. `python3 -m studio.config --summary` lists what did register |
| Empty thinking panel | `model.enable_thinking` must be `true`, otherwise the chat template pre-closes `<think>` |
| Render times out | Set `services.hyperframes.accelerated_render.enabled: false` on machines without a usable GPU |
| Port in use | `bash run_studio.sh --port <free port>` |

---

## Notes for readers of the paper

`AICutPipeline` was this system's internal codename and still appears in the
training system prompt and in the state JSON the model receives. Those strings
are part of the training distribution and are deliberately left byte-identical —
renaming them would move inference off-distribution. The public name is
WeDirector.

`configs/training/` holds the data-preparation and SFT configs behind the
released checkpoint. They are not read at inference time.

## License

Apache License 2.0 — see [LICENSE](LICENSE). Bundled third-party components
(HyperFrames, GSAP, ffmpeg, Node, Noto fonts, deepagents) keep their own
licences; retrieved footage and music keep theirs, and it is the deployer's
responsibility to respect them.
