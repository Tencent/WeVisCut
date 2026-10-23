# WeVisCut data-construct pipeline

[English](README.md) · [中文](README_zh.md) · [back to WeVisCut](../README.md)

This package records **training trajectories** for WeVisCut: one narration
script in, one finished cut plus three role-split data streams out.

It is the data-construction side of the paper, not the deployed studio. The
studio (`aicut_agent/`, `studio/`) serves a trained policy. This package
**produces** the trajectories that policy was trained on.

```
narration script
  → LongVideoRunner
  → InferenceOrchestrator.run_agentic_pipeline   (server/infer_graph.py)
       each step: orchestrator observes state → emits one tool call
       dispatch:  generator / critic skill  or  external media tool
  → TrajectoryRecorder splits the run into three streams
```

---

## What kind of loop is this?

**The orchestrator chooses the next tool.** Every turn it sees the full action
space (12 tools), the current flags, recent history, last observation, and the
tool contract, then emits exactly one `tool_call`. There is no hardcoded
`STAGE_SEQUENCE` driving a successful run — that older path exists only as a
legacy fallback when `agent.agent_mode` is not `long_horizon`.

**The runtime is not a free-form agent.** Preconditions, repeat limits, parse
fallbacks and a terminal `Tool_StitchVideo` gate keep a rollout from stalling
or skipping the quality arc. Those interventions are written into the
trajectory; they are part of the training signal, not hidden rewrites after
the fact.

**Content, reflection and media are decoupled.** The same long-horizon episode
is stored as three causally linked streams:

| Role | Chinese | What it produces | On-disk shape |
|---|---|---|---|
| `orchestrator` | 主流程 | Next-tool decisions | one multi-turn tool-calling episode |
| `generator` | 内容生成器 | Segmentation, shot plans, decoration, patches | one single-turn sample per call |
| `critic` | 反思修复 | Plan self-check, dimension judge, severity rank | one single-turn sample per call |

`interactions.jsonl` records the edges (`step N → role → result`) so the three
files stay one causal trajectory, not three independent logs.

This is why the model can be trained — and later deployed — as a tool-calling
policy whose content skills and judge skills are separately replaceable.

---

## Relation to the rest of this repository

| This package | Released WeVisCut studio |
|---|---|
| Training-time loop in `server/infer_graph.py` | `aicut_agent/` deepagents / LangGraph harness |
| Runtime may redirect illegal / repeated calls so annotation finishes | Guardrails observe; they do not substitute a different tool |
| Writes orchestrator / generator / critic SFT+RL dumps | Serves one trained 9B policy in the browser |

`pipeline.root` defaults to **this WeVisCut checkout** (`pipeline/` +
`server/`). You do not need a second internal repository to run it.

---

## Install

From the WeVisCut repo root:

```bash
pip install -e data_construct_pipeline            # core (httpx + pyyaml)
pip install -e "data_construct_pipeline[dev]"     # pytest
pip install -e "data_construct_pipeline[web]"     # campaign UI
```

A real rollout also needs:

- this WeVisCut tree (`pipeline/`, `server/`, `configs/`, `external_tool/`)
- an OpenAI-compatible served model (vLLM / SGLang)
- TTS: `bash data_construct_pipeline/services/tts/run_qwen3_tts_service.sh`
- `ffmpeg` / `ffprobe` (see the parent [external tools guide](../docs/external_tools.md))

---

## Quick start

```bash
# 0. Paths, system prompt, stage table, three role endpoints
longvideo-polar check

# 0.5 Offline wiring test — no GPU, no key, no TTS
longvideo-polar smoke --pipeline-root ..

# 1. One script → cut + three trajectories
longvideo-polar run --script-file /path/to/script.txt --output ./rollouts/demo

# 2. Batch (process-isolated, resumable, auto-aggregate)
longvideo-polar batch --input-dir /path/to/scripts --workers 4

# 3. Stronger critic (Gemini / Opus) for distillation
longvideo-polar batch --input-dir /path/to/scripts --profile critic_gemini
```

Python:

```python
from longvideo_polar import LongVideoConfig, LongVideoRunner

config = LongVideoConfig.load("data_construct_pipeline/config/longvideo_polar.yaml")
result = await LongVideoRunner(config).run(script=script_text, episode_id="ep_001")
print(result.summary())
```

`agent.agent_mode` **must** be `long_horizon`. Anything else selects the old
fixed-stage workflow and the builder refuses to start.

---

## Model mix

Three endpoints, independently configurable. Profiles deep-merge onto `models`.

| Profile | Orchestrator | Generator | Critic | Use |
|---|---|---|---|---|
| `all_qwen` (default) | local Qwen | local Qwen | local Qwen | offline / RL |
| `critic_gemini` | Qwen | Qwen | Gemini | higher-quality judge labels |
| `critic_opus` | Qwen | Qwen | Claude Opus | same, different teacher |
| `skills_gemini` | Qwen | Gemini | Gemini | train orchestrator only |

The policy you train should be the model that *acted*. Teacher traces
(`metadata.provider`) must stay separable from self-generated ones.

---

## Artifacts

```
<output>/<episode_id>/
├── episode.json                 # manifest, rewards, errors, timing
├── <project_id>/                # pipeline products (plan, assets, cut)
│   └── agent_trace.jsonl
└── trajectories/
    ├── orchestrator.jsonl       # multi-turn tool-calling episode
    ├── generator.jsonl          # single-turn skill samples
    ├── critic.jsonl
    ├── interactions.jsonl       # cross-role edges
    ├── llm_calls.jsonl          # raw I/O audit
    ├── sft/{orchestrator,generator,critic}_messages.jsonl
    └── rl/reward_scores.jsonl
```

A batch writes `aggregate/orchestrator_sft.jsonl`, `generator_sft.jsonl`,
`critic_sft.jsonl`, `interactions.jsonl`, `rewards.jsonl`, `index.json`.

---

## Configuration

[`config/longvideo_polar.yaml`](config/longvideo_polar.yaml) is the control
plane. [`config/pipeline_runtime.yaml`](config/pipeline_runtime.yaml) is what
the orchestrator reads (execution, TTS, templates) — not `studio_config.yaml`.

| Section | Controls |
|---|---|
| `pipeline` | Environment root, runtime YAML, system prompt, TTS |
| `profiles` / `models` | Per-role endpoints and sampling |
| `agent` | `long_horizon`, `max_steps`, repeat limit |
| `tool_switches` | TTS / retrieval / render / stitch |
| `recording` | Output root, prompt retention |
| `batch` | Input glob, workers, resume |

Useful environment variables: `WEVISCUT_ROOT`, `AICUT_PIPELINE_ROOT`,
`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `LVP_MODEL`, `LVP_PROFILE`,
`LVP_OUTPUT_ROOT`, `LVP_INPUT_DIR`.

---

## Layout

```
data_construct_pipeline/
├── config/                 # master YAML + runtime YAML + campaign example
├── src/longvideo_polar/    # runner, recorder, role routing, bridge
├── services/tts/           # optional Qwen3-TTS helper
├── scripts/                # campaign UI launcher
└── tests/                  # offline unit tests (no GPU)
```

```bash
cd data_construct_pipeline && PYTHONPATH=src python -m pytest -q
```

---

## Troubleshooting

| Symptom | What to check |
|---|---|
| `pipeline.root does not exist` | Point `pipeline.root` at this WeVisCut repo, or export `WEVISCUT_ROOT` |
| `missing server/infer_graph.py` | Checkout is incomplete; the loop is `server/infer_graph.py` |
| `agent_mode=... does not select the long-horizon loop` | Set `agent.agent_mode: long_horizon` |
| every step `parse_ok: false` | Served model has no native tool calling → `native_tool_calls: false` |
| trajectories but no cut | `episode.json` errors; usually render (`Tool_BatchTimelineCompose`) or stitch |
| want a faster debug cycle | turn off `tool_switches.hyperframes_render` / `tts` (no real video then) |
