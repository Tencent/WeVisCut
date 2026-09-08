# Configuration reference

`studio_config.yaml` is the only file you edit. No module in the codebase reads a
hardcoded path, so retargeting a deployment is a single-file change.

```bash
bash run_studio.sh --check              # validate; exits non-zero on errors
python3 -m studio.config --summary      # what actually took effect
python3 -m studio.config --dump pipeline
python3 -m studio.config --dump infer
```

`--summary` is worth running after every edit. The most common configuration
failure is not a wrong value; it is editing one file while the process reads
another.

---

## Interpolation

Any string value supports `${VAR}` and `${VAR:-fallback}`, resolved against the
process environment.

```yaml
paths:
  asset_library: ${WEVISIONCUT_ASSET_LIBRARY:-${REPO_ROOT}/workspace/asset_library}
```

`${REPO_ROOT}` is injected automatically and equals the directory containing the
config file, so the shipped defaults work from a fresh clone with nothing
exported. Expansion repeats until stable, so a fallback may itself contain
`${REPO_ROOT}`.

An unset variable with no fallback expands to an empty string. That is
deliberate: the validator then reports a missing setting, instead of a literal
`${FOO}` reaching a filesystem call and producing a confusing error later.

## Environment variables

| Variable | Effect |
|---|---|
| `AICUT_STUDIO_CONFIG` | Use a different config file |
| `AICUT_MODEL_PROFILE` | Override `model.active_profile` |
| `AICUT_STUDIO_SKIP_MODEL` | Serve without loading weights (`--no-model`) |
| `AICUT_MS_SWIFT_PATH` | ms-swift checkout registering `qwen3_5` |
| `WEVISIONCUT_WORKSPACE` | Output root |
| `WEVISIONCUT_MODEL_PATH` / `WEVISIONCUT_LORA_PATH` | Weights / adapter |
| `WEVISIONCUT_ASSET_LIBRARY` / `WEVISIONCUT_MUSIC_LIBRARY` | Media roots |
| `WEVISIONCUT_FFMPEG` / `WEVISIONCUT_FFPROBE` | Binaries |
| `WEVISIONCUT_NODE_BIN` / `WEVISIONCUT_CHROME` | Renderer dependencies |
| `WEVISIONCUT_GOLD_TEMPLATE` / `WEVISIONCUT_SUBTITLE_FONT` | Templates / font |
| `WEVISIONCUT_VIDEO_PROVIDER` / `WEVISIONCUT_BGM_PROVIDER` | Provider names |
| `WEVISIONCUT_VIDEO_API_URL` / `_TOKEN`, `WEVISIONCUT_BGM_API_URL` / `_TOKEN` | HTTP provider endpoints |
| `WEVISIONCUT_TTS_URL` | TTS service |
| `WEVISIONCUT_LLM_BASE_URL` / `_MODEL` / `_API_KEY` | Auxiliary LLM when running the pipeline standalone |
| `BILIBILI_COOKIE` / `YOUTUBE_API_KEY` | Open-web retrieval credentials |
| `TTS_APP_ID` / `TTS_CLUSTER` / `TTS_PROXY_URL` | Optional cloud TTS vendor |
| `MMFINDER_ROOT` | Internal retrieval client |
| `WEVISIONCUT_KEEP_PROXY=1` | Keep `*_proxy` set (default: cleared, see below) |

`run_studio.sh` clears `http_proxy` and friends by default. The pipeline talks to
loopback services and possibly to a private network, and a proxy in the
environment breaks both — usually as a `403 forbidden host` on `127.0.0.1`. Set
`WEVISIONCUT_KEEP_PROXY=1` if your services genuinely require one.

---

## Sections

### `studio`

| Key | Meaning |
|---|---|
| `name`, `locale` | UI title and language (`zh` / `en`) |
| `workspace_root` | Output root; one subdirectory per run |

### `server`

| Key | Meaning |
|---|---|
| `host`, `port` | Bind address |
| `media_roots` | Directory allow-list for `/api/media` |

`media_roots` is a security boundary, not a convenience list. Only files beneath
these directories are ever served, which is what prevents path traversal through
the `?path=` parameter. The workspace root is always included.

### `model`

| Key | Meaning |
|---|---|
| `active_profile` | Which entry under `profiles` is live |
| `profiles.<name>.base_model` | Weights directory |
| `profiles.<name>.adapter` | Optional LoRA (transformers backend only) |
| `profiles.<name>.infer_backend` | `vllm` or `transformers` |
| `model_type`, `template_type` | `qwen3_5` — needs the matching ms-swift |
| `ms_swift_path` | ms-swift checkout; empty uses the installed package |
| `enable_thinking` | Must be `true` — see below |
| `temperature`, `top_p`, `max_new_tokens` | Decoding |
| `stop_words` | `["</tool_call>"]`; one tool call per step |
| `vllm_*` | vLLM engine settings |
| `system_prompt_file`, `stage_config_file` | Training-aligned prompt and stage config |

Two settings are correctness-critical rather than preference.

`enable_thinking` must stay `true`. Qwen3.5 is a hybrid-thinking model: with
thinking off, the chat template prepends an already-closed
`<think>\n\n</think>\n\n` to the assistant turn, so the model physically cannot
emit a thought. Both SFT and DAPO trained think-then-`tool_call`, so turning this
off is an off-distribution change that also empties the frontend's thinking panel.

`system_prompt_file` must be byte-identical to the training prompt. It is not
documentation; it is part of the input distribution.

Decoding defaults to greedy (`temperature: 0.0`) so a given script reproduces the
same cut.

### `agent`

| Key | Meaning |
|---|---|
| `mode` | `long_horizon` |
| `max_steps` | Hard step ceiling |
| `max_same_tool_repeats` | Backstop against a stuck loop |

`agent.autonomy` decides how much the server is allowed to intervene. With an
RL-trained policy, every correction is off:

| Key | Meaning | Default |
|---|---|---|
| `fully_autonomous` | The model picks every tool and decides when to finish | `true` |
| `force_valid_tool` | Rewrite an illegal call server-side | `false` |
| `force_terminal_stitch` | Inject `Tool_StitchVideo` when the model will not finish | `false` |
| `force_stage_order` | Advance stages along the expert path | `false` |
| `report_invalid_as_observation` | Feed illegal calls back as observations | `true` |
| `max_consecutive_parse_failures` | Unparseable outputs before halting | `4` |
| `max_consecutive_blocked` | Blocked calls before halting | `6` |
| `max_identical_tool_failures` | Same failure this often = broken environment | `3` |
| `max_identical_tool_calls` | Byte-identical repeats = livelock | `3` |
| `max_stage4_repairs` / `max_stage4_attempts` | Repair budget | `2` / `6` |
| `max_tool_calls` | Executed calls per run (a clean run is ~12) | `30` |

The two identical-call limits look redundant but are not.
`max_identical_tool_failures` watches for the same *failure*;
`max_identical_tool_calls` ignores the return status, because a Stage-4 repair
returns `completed_with_warnings` rather than `error`, which would reset a
failure counter and let the same call repeat indefinitely.

Turning the `force_*` switches on is useful for measuring how much a weaker
policy depends on scaffolding. It is not how the released checkpoint is evaluated.

### `services`

Provider selection — `retrieval_provider`, `bgm_provider`, `providers` — is
documented in [providers.md](providers.md).

**`services.tts`**

| Key | Meaning |
|---|---|
| `service_url` | TTS server; falls back to `edge-tts` then duration estimation |
| `voice`, `language`, `instruct`, `speed` | Narration style |
| `dispatch_mode` | `batch` sends all shots in one request |
| `batch_size`, `batch_concurrency` | Batching |
| `timeout_sec`, `health_probe_timeout_sec` | Timeouts |

**`services.visual_retrieval`** — provider-independent footage constraints
(`top_k`, `require_horizontal`, `min_duration_ratio`, `max_loop_count`,
`select_with_llm`, ...). See [providers.md §5](providers.md).

**`services.bgm`** — playlist policy: `top_k`, `bgm_volume`, `duck_under_voice`,
and the `min_tracks_*` thresholds that scale variety with video length.

**`services.hyperframes`**

| Key | Meaning |
|---|---|
| `enabled` | Off = no animated decoration layer |
| `hf_bin`, `node_bin_dir` | CLI and Node runtime |
| `runtime_archive`, `runtime_cache_dir` | Self-contained runtime, unpacked once and reused |
| `browser_path` | `chrome-headless-shell` |
| `gsap_local_path` | Local GSAP, so templates never reach a CDN |
| `render_quality`, `timeout` | Quality and per-shot timeout |
| `accelerated_render.*` | GPU rendering: fps, workers, browser wrapper |

The runtime tarball exists for a measured reason: rendering reads thousands of
small `node_modules` files, and doing that over a network filesystem on every
shot dominates the wall clock. One local unpack removes it.

**`services.ffmpeg`** — `ffmpeg_path`, `ffprobe_path`. Use static builds; a
dynamically linked binary copied between machines fails on the first missing
`.so`, typically deep inside a render step.

### `paths`

| Key | Meaning |
|---|---|
| `asset_library` | Your video library |
| `gold_template_root` | Template library (must satisfy its `CONTRACT.md`) |
| `output_root`, `storage_root` | Intermediates and artifacts |
| `gold_template_policies_file`, `instructions_file` | Optional decoration hints; missing files degrade silently |

### `render`

`render.split` turns on transparent-layer acceleration: a cacheable base video, a
WebM alpha animation layer, and an ffmpeg overlay. Any step failing falls back to
rendering the whole scene, so output quality never degrades — only speed.

`render.subtitles` — `fontfile` (must exist), `font_size`, `line_spacing`,
`bottom_margin`, `max_chars_per_line`, `max_lines`, `boxcolor`, `boxborderw`.

`render.stitch` — keep both `sanitize_shots` and `fast_concat_copy` on. Rendered
shots are not uniform: the ffmpeg path emits 25 fps + 24 kHz mono narration while
HyperFrames emits 14-24 fps + 48 kHz stereo. `concat` cannot reconcile that — it
produces non-monotonic DTS and feeds NaN samples to the AAC encoder, failing the
whole stitch. With sanitising on, each shot is first normalised to h264/24fps +
aac 48k stereo, after which a stream copy finishes in about a second.
`fast_concat_copy` additionally probes every shot and transcodes on any mismatch,
which is what stops the output from being audio over a black screen.

### `quality`

`quality.severity_rank` scores shots for likely defects and sends only the worst
`top_ratio` through VLM review, instead of paying for a judgement on every shot.
Keep `allow_heuristic_fallback: true` — when model scoring fails, the ranking
otherwise degenerates to all-zero and Stage 4 reviews nothing.

### `progress`

`tracks` (order = display order), `push_throttle_ms`, `raw_output_buffer`,
`max_events_per_run`. Valid tracks: `segments`, `shots`, `narration`, `subtitle`,
`footage`, `spotlight`, `bgm`, `render`.

### `editor`

`manual_ops` toggles each operation individually (`edit_subtitle`,
`edit_narration`, `replace_footage`, `edit_spotlight_slots`, `swap_template`,
`reorder_shots`, `delete_shot`, `adjust_bgm`).

`rerender.scope: only_dirty` re-renders just the edited shots and reuses the
cache elsewhere. `keep_version_history` snapshots each stitch for rollback, capped
at `max_versions`.

### `chat_edit`

`backend: inproc` reuses the already-loaded policy model; `openai` calls an
external `/v1` endpoint.

`context` manages the token budget: over `max_history_tokens`, turns older than
`keep_recent_turns` are summarised to `summary_target_tokens`. When a project has
more than `project_digest_max_shots` shots, only the `relevant_shot_topk` most
relevant are injected.

`safety.require_confirm_for` lists operations needing explicit UI confirmation —
`delete_shot`, `swap_template`, `reorder_shots` by default.

---

## Validation

`--check` distinguishes two levels:

- **error** — the studio cannot work: no active profile, unreadable system
  prompt, invalid port, empty `media_roots`, an unregistered provider name.
- **warn** — something is missing but degrades gracefully: absent weights (the
  studio still boots and serves the UI), a missing renderer binary, a missing
  font.

This is why a partially installed machine still starts and tells you what it is
missing, rather than refusing to run.
