<div align="center">

<img src="assets/weviscut_logo.png" width="420" alt="WeVisCut">

# WeVisCut

### A Long-Horizon Tool Agent for Turning Knowledge Articles into Edited Videos

<p>
  <a href="https://tencent.github.io/WeVisCut/"><img src="https://img.shields.io/badge/Project-Page-4285F4?logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://huggingface.co/tencent/WeVisCut"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-WeVisCut--9B-FBBC04" alt="WeVisCut-9B"></a>
  <a href="https://github.com/Tencent/WeVisCut"><img src="https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white" alt="Code"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="License"></a>
</p>

**English** | [简体中文](README_zh.md)

</div>

> **WeVisCut** is a long-horizon tool agent that turns a knowledge article or narration script into a multi-minute edited video. At every step, a single 9B policy reads the current editing state and emits **one** tool call — segment the article, plan shots, synthesize narration, retrieve footage, fill animated templates, select BGM, judge, reflect, repair, stitch — for roughly twelve calls per rollout. There is no hardcoded stage machine: the trajectory is decided by the policy itself, and guardrails only *reject* illegal calls instead of substituting a decision of their own. The released runtime is **training-aligned** — the agent sees the same 12 tools and the same state JSON as during training — and ships with the Studio, a live timeline that stays editable after the cut, and with `data_construct_pipeline/`, the trajectory-collection pipeline that produced the training data.

![Overview of WeVisCut](assets/overview.png)

## 📋 At a Glance

| | |
| :--- | :--- |
| **Task** | long-form article / narration script → multi-minute edited video |
| **Policy** | a single 9B model, one tool call per step (~12 calls per rollout) |
| **Observation** | structured editing state (chapters, shots, narration, footage, templates, BGM, render) |
| **Action space** | 12 editing tools (segment, plan, TTS, retrieval, template, compose, BGM, judge, reflect, probe, stitch) |
| **Runtime** | `studio/` — live timeline, pause / resume, manual & chat revision, per-shot re-render, version rollback |
| **Data engine** | `data_construct_pipeline/` — long-horizon trajectory annotation (orchestrator / generator / critic) |
| **Requirements** | Python ≥ 3.11; ~24 GB NVIDIA GPU for bf16 9B (UI-only mode needs no GPU) |
| **License** | Apache-2.0 ([LICENSE](LICENSE)) |

## ✨ Highlights

- 🎬 **End-to-end long-form editing.** Knowledge-style long copy in, a multi-minute cut out — typically ~12 tool calls, no hand-written stage machine in between.
- 🕹️ **Policy-driven, not scripted.** The policy decides the next action from the editing state; guardrails reject illegal calls but never replace them.
- 🎚️ **Editable Studio.** Eight synchronized tracks (chapters, shots, narration, subtitles, footage, templates, BGM, render) with pause / resume, manual editing, chat-based revision, per-shot re-rendering and version rollback.
- 🔁 **Training-aligned runtime.** The same 12 tools and the same state JSON as in training, so the served policy runs in its training distribution.
- 🧾 **Open data engine.** `data_construct_pipeline/` records the long-horizon trajectories behind the policy (orchestrator / generator / critic).

## 📢 News

- **[2026-10]** 🎉 We release **WeVisCut**: the technical report, the 9B policy weights, the Studio runtime and the trajectory-construction pipeline.
- **[2026-10]** 🌐 The project page is live at [WeVisCut Project Link](https://tencent.github.io/WeVisCut/).

## 🏗️ How It Works

**1. State-conditioned tool loop.** At each step the policy observes the structured editing state and emits exactly **one** tool call. The tool result updates the state, and the loop continues until `Tool_StitchVideo` assembles the cut. Reflections and judges are themselves tool calls: the agent reviews its own intermediate renders and repairs whatever fails the gates.

**2. Action space.** Twelve editing tools form the whole action space:

| Tool | Role |
| :--- | :--- |
| `Tool_SegmentArticle` | Split the article into chapters |
| `Tool_PlanShots` | Plan the shots of one chapter |
| `Tool_GenerateAudio` | Synthesize narration; locks the shot duration |
| `Tool_RetrieveVisual` | Retrieve footage for a shot |
| `Tool_DecorateShot` | Bind an animated template and fill its slots |
| `Tool_TimelineCompose` | Render one shot |
| `Tool_MusicSearch` | Build the BGM playlist |
| `Tool_TextQuickJudge` / `Tool_VLMRenderJudge` | Quality gates (text / rendered video) |
| `Tool_SelfReflect` | Decide what to repair next |
| `Tool_ModalProbe` | Preflight external services |
| `Tool_StitchVideo` | Concatenate the final cut |

**3. Training-aligned runtime.** The Studio serves the policy with the identical tool schema and state JSON seen in training, so behavior at inference matches the distribution the policy was optimized on. `AICutPipeline` — the name that appears inside prompts and state JSON — is the training-time identifier and is kept unchanged on purpose.

**4. Data engine.** [`data_construct_pipeline/`](data_construct_pipeline/README.md) is how the trajectories are collected: one script in → one cut plus three role streams (orchestrator, generator, critic). The Studio *serves* the trained policy; this package *records* the data it was trained on.

Further notes: [`docs/architecture.md`](docs/architecture.md).

## 🚀 Quick Start

### Installation

**Requirements:** Python ≥ 3.11; an NVIDIA GPU with ~24 GB for the bf16 9B policy (UI-only mode: `--no-model`). Node and ffmpeg are fetched by the setup script.

```bash
git clone https://github.com/Tencent/WeVisCut.git && cd WeVisCut

pip install -r requirements.txt
# GPU machines: install the torch build matching your CUDA first, then
pip install -r requirements-gpu.txt

bash scripts/setup_external_tools.sh          # ffmpeg, Node, renderer (~1.1 GB)

pip install -U "huggingface_hub[cli]"
hf download tencent/WeVisCut --local-dir checkpoints/weviscut-9b
```

### Launch the Studio

```bash
bash run_studio.sh --check                    # path check, no GPU needed
bash run_studio.sh                            # default: http://<host>:8081/
# UI only, no weights loaded:
# bash run_studio.sh --no-model
```

Weights load in the background (30–60 s) while the UI is already usable. To keep the checkpoint elsewhere: `export WEVISCUT_MODEL_PATH=/path/to/WeVisCut-9B`.

> [!TIP]
> If `github.com` is unreachable, replace the git pin in `requirements.txt` with `pip install deepagents==0.7.7`. Loading the weights needs an `ms-swift` that registers `qwen3_5` (set `AICUT_MS_SWIFT_PATH` if yours does not).

### Use the Studio

1. Open `http://<host>:8081/` (trustworthy-cut UI: `/0907_newvis`).
2. Paste a narration / knowledge script and start. The eight tracks fill as the agent works.
3. After a cut: edit any shot (narration, footage, template, BGM), or revise it by chat — “change the opening line”, “replace the visual at 0:15”.
4. Re-render only the shots you changed; the history supports version rollback.

Default footage / BGM comes from open-web search (`asset_retrieval_tool`). Point `studio_config.yaml` at your own HTTP API, a local folder, or a Python plugin — see [`docs/providers.md`](docs/providers.md); it is the only file you normally need to edit.

```bash
python3 -m studio.config --summary
```

<details>
<summary><b>Troubleshooting</b></summary>
<br>

| Symptom | Check |
| :--- | :--- |
| Missing modules | `requirements.txt` / `requirements-gpu.txt` |
| Model stuck on “loading” | Log; `checkpoints/weviscut-9b` (or `WEVISCUT_MODEL_PATH`); GPU visible |
| A tool keeps failing | Frontend terminal — usually TTS / retrieval / renderer |
| Empty thinking panel | `model.enable_thinking: true` |
| Port already in use | `bash run_studio.sh --port <free>` |

</details>

## 🧾 Data-Construct Pipeline

```bash
pip install -e data_construct_pipeline
longvideo-polar smoke --pipeline-root .     # offline wiring check, no GPU
longvideo-polar run --script-file /path/to/script.txt --output ./rollouts/demo
```

A full rollout additionally needs a served OpenAI-compatible model, TTS and ffmpeg. Batch jobs and the three-role dumps are documented in the pipeline [README](data_construct_pipeline/README.md).

## 📚 Documentation

| Document | Covers |
| :--- | :--- |
| [`docs/architecture.md`](docs/architecture.md) | Runtime architecture, tool loop, state JSON |
| [`docs/configuration.md`](docs/configuration.md) | `studio_config.yaml` reference |
| [`docs/providers.md`](docs/providers.md) | Custom footage / BGM providers |
| [`docs/huggingface.md`](docs/huggingface.md) | Weights download and model profiles |
| [`docs/external_tools.md`](docs/external_tools.md) | Third-party runtimes fetched by the setup script |

## 📖 Citation

If WeVisCut is helpful to your work, please consider giving us a ⭐ and citing:

```bibtex
@misc{mu2026weviscuteditingroomagent,
      title={WeVisCut: The Whole Editing Room in One Agent From Narration to Long-Form Video}, 
      author={Chenyu Mu and Wenming Weng and Qinrui Wang and Xingsong Ye and Hao Yu and Jun He and Chong Sun and Chen Li and Jing Lyu and Erkun Yang and Cheng Deng},
      year={2026},
      eprint={2610.38352},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.38352}, 
}
```

## 📜 License

Tencent-authored code in this repository is released under the [Apache License 2.0](LICENSE). Third-party runtimes and retrieved media keep their own licenses.
