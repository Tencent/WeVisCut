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

Paths have been replaced with `${WEDIRECTOR_DATA_ROOT}`,
`${WEDIRECTOR_RESULT_ROOT}` and `${WEDIRECTOR_TRAINING_REPO}` placeholders.
Export those before reusing a file, and expect to adapt the rest: the data
pipeline that consumes them is a separate codebase.

The matching system prompts live one level up in `configs/`. Only
`unified_v5_system_prompt.txt` is loaded at inference — it must stay
byte-identical to the prompt the policy was trained with, so do not reformat it.
