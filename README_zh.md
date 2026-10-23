<div align="center">

<img src="assets/weviscut_logo.png" alt="WeVisCut logo" width="420" />

# WeVisCut

### *面向知识类长视频剪辑的长程工具智能体*

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg?logo=python&logoColor=white)](requirements.txt)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-WeVisCut--9B-FFD21E.svg)](https://huggingface.co/tencent/WeVisCut)
[![Tools](https://img.shields.io/badge/Tools-12%20editing%20tools-2EA44F.svg)](#工作原理简述)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20NVIDIA%20GPU-76B900.svg)](requirements-gpu.txt)

[English](README.md) · [中文](README_zh.md) · [模型](https://huggingface.co/tencent/WeVisCut) · [代码](https://github.com/Tencent/WeVisCut) · [文档](docs/architecture.md)

</div>

---

## 摘要

**WeVisCut** 是一个长程工具智能体，把知识类长文 / 口播稿直接做成数分钟成片。一个 9B 策略模型（[`tencent/WeVisCut`](https://huggingface.co/tencent/WeVisCut)）在每一步读取当前剪辑状态，**一次只发出一个工具调用**——切分文章、规划分镜、合成口播、检索素材、填充动画模板、挑选配乐、质量判别、自我反思、修复、拼接成片，一次完整运行约 12 次调用。系统里没有硬编码的流程状态机：轨迹由策略自行决定，护栏只负责**拒绝**非法调用，而不会替模型重写动作。本次公开的推理运行时与训练**严格对齐**——工具集与状态 JSON 与训练时一致——并同时提供 Studio 交互界面和用于采集训练轨迹的数据管线。

![WeVisCut 模型总览](assets/overview.png)

[下载总览图 PDF](assets/overview.pdf)

---

## 速览

| | |
|---|---|
| **任务** | 长文口播稿 → 数分钟成片 |
| **策略** | 单个 9B 模型，每步一次工具调用（约 12 次 / 条轨迹） |
| **动作空间** | 12 个剪辑工具（切分、分镜、TTS、检索、模板、合成、配乐、判别、反思、预检、拼接） |
| **推理运行时** | `studio/` — 实时时间轴、暂停 / 继续、手动剪辑与对话改片、按镜头重渲染、版本回滚 |
| **数据** | `data_construct_pipeline/` — 长程轨迹标注（主流程 / 内容生成 / 反思三路） |
| **环境** | Python ≥ 3.11；bf16 9B 需约 24 GB NVIDIA 显存（仅界面模式无需 GPU） |
| **许可** | Apache-2.0（[LICENSE](LICENSE)） |

---

## 目录

- [快速启用](#快速启用)
- [使用 Studio](#使用-studio)
- [数据构建管线](#数据构建管线)
- [工作原理简述](#工作原理简述)
- [测试（离线，不需要 GPU）](#测试离线不需要-gpu)
- [引用](#引用)
- [许可](#许可)

---

## 能做什么

- **长文视频化。** 知识类长口播进，数分钟成片出（一次典型运行约 12 次工具调用）。
- **Studio。** 实时时间轴（章节、分镜、口播、字幕、素材、模板、BGM、渲染），可暂停 / 继续，成片后支持**手动剪辑**和**对话改片**。
- **与训练对齐的推理。** 模型看到的 12 个工具和状态 JSON 与训练时一致。
- **数据管线。** `data_construct_pipeline/` 负责采集训练轨迹（主流程 / 内容生成 / 反思三路）。

---

## 快速启用

**环境：** Python ≥ 3.11；生成需要约 24 GB 显存的 NVIDIA GPU（只看界面用 `--no-model`）。Node 和 ffmpeg 由安装脚本拉取。

```bash
git clone https://github.com/Tencent/WeVisCut.git
cd WeVisCut

pip install -r requirements.txt
# GPU 机器：先装匹配 CUDA 的 torch，再
pip install -r requirements-gpu.txt

bash scripts/setup_external_tools.sh          # ffmpeg、Node、渲染器，约 1.1 GB

pip install -U "huggingface_hub[cli]"
hf download tencent/WeVisCut --local-dir checkpoints/weviscut-9b

bash run_studio.sh --check                   # 校验路径，不需要 GPU
bash run_studio.sh                          # 默认 http://<host>:8081/
```

权重在后台加载（约 30–60 秒），界面会立刻打开。权重不在默认目录时：`export WEVISCUT_MODEL_PATH=/path/to/WeVisCut`。

访问不了 `github.com` 时，把 `requirements.txt` 里的 git pin 换成 `pip install deepagents==0.7.7`。加载权重需要能识别 `qwen3_5` 的 `ms-swift`（否则设 `AICUT_MS_SWIFT_PATH`）。

只起前端、不加载模型：`bash run_studio.sh --no-model`。

---

## 使用 Studio

1. 打开 `http://<host>:8081/`（trustworthy-cut 界面：`/0907_newvis`）。
2. 粘贴口播 / 知识长文，开始生成。八条轨道随智能体推进填充。
3. 成片后：改某一镜的口播、素材、模板、BGM，或用对话（「把开头第一句改掉」「把 0:15 的画面换掉」）。
4. 重渲染只处理改过的镜头。历史支持版本回滚。

默认素材 / BGM 走公开网络检索（`asset_retrieval_tool`）。在 `studio_config.yaml` 里可改成你自己的 HTTP 接口、本地文件夹或 Python 插件，见 [`docs/providers.md`](docs/providers.md)。日常只改这一份配置即可。

```bash
python3 -m studio.config --summary
```

---

## 数据轨迹标注管线

[`data_construct_pipeline/`](data_construct_pipeline/README_zh.md) 用来**标注长程轨迹**：一篇口播稿进 → 一条成片 + 三路角色数据（主流程、内容生成、反思修复）。Studio 负责*部署*训练好的策略；这个包负责*采出*训练数据。

```bash
pip install -e data_construct_pipeline
longvideo-polar smoke --pipeline-root .     # 离线接线检查，不需要 GPU
longvideo-polar run --script-file /path/to/script.txt --output ./rollouts/demo
```

完整 rollout 还需要已启动的 OpenAI 兼容模型、TTS 和 ffmpeg。批量任务与三路落盘格式见管线 [README](data_construct_pipeline/README_zh.md)。

---

## 工作方式（简要）

每一步策略看到当前剪辑状态，从 12 个工具里发出**一次**调用。护栏可以拒绝非法调用，但不会替模型另选工具。

| 工具 | 作用 |
|---|---|
| `Tool_SegmentArticle` | 把长文切成章节 |
| `Tool_PlanShots` | 为章节规划分镜 |
| `Tool_GenerateAudio` | TTS，锁定镜头时长 |
| `Tool_RetrieveVisual` | 检索画面素材 |
| `Tool_DecorateShot` | 模板与填槽 |
| `Tool_TimelineCompose` | 渲染单镜 |
| `Tool_MusicSearch` | 配乐 |
| `Tool_TextQuickJudge` / `Tool_VLMRenderJudge` | 质量门 |
| `Tool_SelfReflect` | 下一步修什么 |
| `Tool_ModalProbe` | 外部服务预检 |
| `Tool_StitchVideo` | 拼接成片 |

更多说明：[`docs/architecture.md`](docs/architecture.md)。配置：[`docs/configuration.md`](docs/configuration.md)。大文件下载：[`docs/external_tools.md`](docs/external_tools.md)。

提示词和状态 JSON 里的 `AICutPipeline` 是训练时的内部名，为保持分布一致未改。

---

## 测试（离线，不需要 GPU）

```bash
python3 tests/test_studio_offline.py
python3 -m pytest tests/test_aicut_agent.py tests/test_providers.py tests/test_thinking_parse.py
node tests/test_frontend.js
```

| 现象 | 查看 |
|---|---|
| 缺模块 | `requirements.txt` / `requirements-gpu.txt` |
| 模型一直 loading | 日志；`checkpoints/weviscut-9b`（或 `WEVISCUT_MODEL_PATH`）；GPU 是否可见 |
| 某个工具反复失败 | 前端终端，多半是 TTS / 检索 / 渲染 |
| thinking 面板空白 | `model.enable_thinking: true` |
| 端口占用 | `bash run_studio.sh --port <空闲端口>` |

## 引用

如果 WeVisCut 对你的研究有帮助，请引用：

```bibtex
@misc{weviscut2026,
  title        = {WeVisCut: A Long-Horizon Tool Agent for Turning Knowledge Articles into Edited Videos},
  author       = {WeVisCut Team},
  organization = {Tencent},
  year         = {2026},
  howpublished = {\url{https://github.com/Tencent/WeVisCut}},
  note         = {Technical report and code release}
}
```

## 许可

Apache License 2.0，见 [LICENSE](LICENSE)。第三方运行时与检索到的素材保留各自授权。
