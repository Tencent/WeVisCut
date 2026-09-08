# Training-time configs

Nothing in this directory is read by the studio at inference time. These are the
data-preparation and SFT configs from the training pipeline, kept here so the
recipe behind the released checkpoint is documented rather than implied.

| File | Purpose |
|---|---|
| `paths.yaml` | Roots for the raw-video batch pipeline and processed-data output |
| `sample_mix.yaml` | Sampling ratios when mixing SFT sources |
| `tool_registry.yaml` | Tool inventory used to generate training tool schemas |
| `content_sft_v2.yaml` | Content-generation SFT stage |
| `agent_sft_v3.yaml` | Verifiable agentic-control SFT stage |
| `long_horizon_agent.yaml` | Long-horizon trajectory SFT (the v5/v6 lineage) |

Paths have been replaced with `${WEVISIONCUT_DATA_ROOT}`,
`${WEVISIONCUT_RESULT_ROOT}` and `${WEVISIONCUT_TRAINING_REPO}` placeholders.
Export those before reusing a file.

The code that rolls out scripts and writes the three role streams is
[`data_construct_pipeline/`](../../data_construct_pipeline/README.md) in this
repository. These YAML files are the SFT-mix / path recipes that sit on top of
those dumps; they are not a second hidden pipeline.

The matching system prompts live one level up in `configs/`. Only
`unified_v5_system_prompt.txt` is loaded at inference — it must stay
byte-identical to the prompt the policy was trained with, so do not reformat it.
