# Model weights on Hugging Face

How to consume the released WeVisCut checkpoint, and how to publish your own.

---

## Downloading

The checkpoint is a `qwen3_5` architecture, roughly 18 GB in bf16.

```bash
pip install -U "huggingface_hub[cli]"
hf download <org>/WeVisCut-9B --local-dir checkpoints/weviscut-9b
```

`checkpoints/weviscut-9b` is the path `studio_config.yaml` looks in by default,
so nothing else is needed. To keep weights elsewhere — a shared volume, a
different disk — point the config at them:

```bash
export WEVISCUT_MODEL_PATH=/data/models/weviscut-9b
```

Behind a mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

To keep the Hub cache off your home partition:

```bash
export HF_HOME=/data/hf_home
```

Then confirm it resolves before starting the model:

```bash
bash run_studio.sh --check
```

### LoRA adapters

To serve a base model plus an adapter, use the `transformers` profile — vLLM in
this setup loads merged weights only:

```bash
export WEVISCUT_MODEL_PROFILE=weviscut_9b_hf
export WEVISCUT_MODEL_PATH=/data/models/base
export WEVISCUT_LORA_PATH=/data/models/my_adapter
```

`transformers` is markedly slower than vLLM. For anything beyond a quick
comparison, merge the adapter and serve the merged weights with vLLM.

---

## `ms-swift` and the `qwen3_5` model type

The one non-obvious dependency. Weights load through `ms-swift`'s `VllmEngine`,
and the policy's architecture must be registered in swift's model registry
(`swift/model/constant.py`). If your installed `ms-swift` does not know
`qwen3_5`, loading fails with an unknown-model-type error.

Point the config at a checkout that does register it:

```yaml
model:
  ms_swift_path: /path/to/ms-swift
```

or:

```bash
export AICUT_MS_SWIFT_PATH=/path/to/ms-swift
```

`ms_swift_path` is prepended to `sys.path`, so that checkout wins over any
pip-installed copy.

---

## Publishing your own checkpoint

### 1. Prepare the directory

A servable directory holds the weight shards, `config.json`, the tokenizer
files, and `chat_template.jinja`. The chat template is not optional: it defines
the `<think>` and `<tool_call>` framing the model was trained with, and the
studio's parsing assumes it. Verify locally first:

```bash
export WEVISCUT_MODEL_PATH=/path/to/your/checkpoint
bash run_studio.sh --check
bash run_studio.sh          # generate one video end to end
```

If `enable_thinking` produces an empty thinking panel, the template is closing
`<think>` before generation — that is a template problem, not a config one.

### 2. Upload

```bash
hf auth login
hf repo create <org>/WeVisCut-9B --type model
hf upload <org>/WeVisCut-9B /path/to/your/checkpoint . --repo-type model
```

Large uploads resume, so an interrupted transfer can simply be re-run.

### 3. Write the model card

`README.md` at the repo root is the model card. Include the YAML front matter —
it drives search and the license banner:

```yaml
---
license: apache-2.0
language: [zh, en]
pipeline_tag: any-to-any
tags: [video-editing, agent, tool-use, long-horizon, reinforcement-learning]
library_name: transformers
---
```

Then cover, at minimum:

- **What it is.** A 9B policy trained to drive a 12-tool video-editing
  environment. It emits one tool call per step against a serialised state JSON;
  it is not a general chat model and it is not a text-to-video model.
- **How to run it.** Link back to this repository. The weights are not useful
  without the tool environment: the model outputs tool calls, and something has
  to execute them.
- **Input format.** Single-turn `[system, user]`, where the user message is the
  state JSON, with tool schemas passed separately. Point at
  [`architecture.md`](architecture.md#the-alignment-problem) — this is the part
  people get wrong when they try to use the weights directly.
- **Training.** SFT then DAPO reinforcement learning; data configs in
  `configs/training/`.
- **Evaluation.** Whatever numbers you are reporting, with the protocol.
- **Limitations.** Output quality depends on the retrieval back end and the
  template library, so results are not comparable across deployments with
  different media sources.
- **License and citation.**

### 4. Optional: a Space

The studio is a FastAPI app with a static frontend, so a Docker Space works:

```dockerfile
FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN bash scripts/setup_external_tools.sh
ENV WEVISCUT_WORKSPACE=/data/workspace
CMD ["bash", "run_studio.sh", "--port", "7860"]
```

Two constraints to plan around. Generation needs a GPU with roughly 24 GB, so a
free-tier Space can only run `--no-model` — enough to show the UI, the timeline
and the API, not to generate. And rendering needs headless Chrome, which the
setup script installs but which requires the shared libraries listed in
[`external_tools.md`](external_tools.md); on a minimal base image you will need
to add them.

For a demo without a GPU, serving pre-generated projects from
`workspace/projects/` through the existing project APIs shows the output without
loading weights.

---

## Datasets

If you release training data, keep it in a separate dataset repo and reference it
from the model card. `configs/training/` documents the expected layout and the
environment variables (`WEVISCUT_DATA_ROOT` and friends) that point the
training configs at a local copy.
