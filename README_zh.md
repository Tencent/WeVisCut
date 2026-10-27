<div align="center">

<img src="assets/weviscut_logo.png" width="420" alt="WeVisCut">

# WeVisCut

### 面向知识类长文成片的长程工具智能体

<p>
  <a href="https://tencent.github.io/WeVisCut/"><img src="https://img.shields.io/badge/Project-Page-4285F4?logo=googlechrome&logoColor=white" alt="项目主页"></a>
  <a href="https://huggingface.co/tencent/WeVisCut"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-WeVisCut--9B-FBBC04" alt="WeVisCut-9B"></a>
  <a href="https://github.com/Tencent/WeVisCut"><img src="https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white" alt="代码"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="许可"></a>
</p>

[English](README.md) | **简体中文**

</div>

> **WeVisCut** 是一个把知识类长文 / 口播稿直接做成数分钟成片的长程工具智能体。每一步，一个 9B 策略模型读取当前剪辑状态，只发出**一次**工具调用——切分文章、规划分镜、合成口播、检索素材、填充动画模板、挑选配乐、质量判别、自我反思、修复、拼接成片，一次完整运行约 12 次调用。系统里没有硬编码的流程状态机：轨迹由策略自行决定，护栏只负责**拒绝**非法调用，而不会替模型重写动作。本次公开的推理运行时与训练**严格对齐**——智能体看到的 12 个工具与状态 JSON 与训练时完全一致——并同时提供 Studio（成片后仍可编辑的实时时间轴）与 `data_construct_pipeline/`（产出训练数据的轨迹采集管线）。

![WeVisCut 总览](assets/overview.png)

## 📋 速览

| | |
| :--- | :--- |
| **任务** | 知识类长文 / 口播稿 → 数分钟成片 |
| **策略** | 单个 9B 模型，每步一次工具调用（约 12 次 / 条轨迹） |
| **观测** | 结构化剪辑状态（章节、分镜、口播、素材、模板、BGM、渲染） |
| **动作空间** | 12 个剪辑工具（切分、分镜、TTS、检索、模板、合成、配乐、判别、反思、预检、拼接） |
| **推理运行时** | `studio/` — 实时时间轴、暂停 / 继续、手动剪辑与对话改片、按镜头重渲染、版本回滚 |
| **数据引擎** | `data_construct_pipeline/` — 长程轨迹标注（主流程 / 内容生成 / 反思三路） |
| **环境** | Python ≥ 3.11；bf16 9B 需约 24 GB NVIDIA 显存（仅界面模式无需 GPU） |
| **许可** | Apache-2.0（[LICENSE](LICENSE)） |

## ✨ 亮点

- 🎬 **端到端长文成片。** 知识类长口播进，数分钟成片出，一次典型运行约 12 次工具调用，中间没有手写状态机。
- 🕹️ **策略驱动而非脚本驱动。** 策略依据剪辑状态自行决定下一步；护栏只拒绝非法调用，不替它决策。
- 🎚️ **可编辑的 Studio。** 八条同步轨道（章节、分镜、口播、字幕、素材、模板、BGM、渲染），支持暂停 / 继续、手动改镜、对话改片、按镜头重渲染与版本回滚。
- 🔁 **与训练对齐的推理。** 工具集与状态 JSON 与训练时一致，部署时策略仍处在训练分布内。
- 🧾 **开源数据引擎。** `data_construct_pipeline/` 记录策略背后的长程轨迹（主流程 / 内容生成 / 反思修复）。

## 📢 新闻

- **[2026-10]** 🎉 我们发布 **WeVisCut**：技术报告、9B 策略权重、Studio 推理运行时与轨迹构建管线。
- **[2026-10]** 🌐 项目主页上线：[WeVisCut项目链接](https://tencent.github.io/WeVisCut/)。

## 🏗️ 工作原理

**1. 状态驱动的工具循环。** 每一步策略观测结构化剪辑状态，只发出**一次**工具调用；工具结果写回状态，循环继续，直到 `Tool_StitchVideo` 拼出成片。判别与反思本身也是工具调用：智能体会审视自己的中间渲染结果，并修复未通过质量门的镜头。

**2. 动作空间。** 12 个剪辑工具构成完整动作空间：

| 工具 | 作用 |
| :--- | :--- |
| `Tool_SegmentArticle` | 把长文切成章节 |
| `Tool_PlanShots` | 为单个章节规划分镜 |
| `Tool_GenerateAudio` | 合成口播，锁定镜头时长 |
| `Tool_RetrieveVisual` | 为镜头检索画面素材 |
| `Tool_DecorateShot` | 绑定动画模板并填槽 |
| `Tool_TimelineCompose` | 渲染单个镜头 |
| `Tool_MusicSearch` | 生成配乐列表 |
| `Tool_TextQuickJudge` / `Tool_VLMRenderJudge` | 质量门（文本 / 渲染视频） |
| `Tool_SelfReflect` | 决定下一步修什么 |
| `Tool_ModalProbe` | 外部服务预检 |
| `Tool_StitchVideo` | 拼接最终成片 |

**3. 与训练对齐的运行时。** Studio 以与训练完全一致的工具 schema 和状态 JSON 部署策略，使推理行为落在策略被优化的分布内。提示词与状态 JSON 中的 `AICutPipeline` 是训练期的内部标识，为保持分布一致刻意保留。

**4. 数据引擎。** [`data_construct_pipeline/`](data_construct_pipeline/README_zh.md) 负责采集轨迹：一篇口播稿进 → 一条成片 + 三路角色数据（主流程、内容生成、反思修复）。Studio 负责*部署*训练好的策略，这个包负责*采出*训练它的数据。

更多说明：[`docs/architecture.md`](docs/architecture.md)。

## 🚀 快速开始

### 安装

**环境要求：** Python ≥ 3.11；bf16 9B 策略需要约 24 GB 显存的 NVIDIA GPU（仅看界面用 `--no-model`）。Node 与 ffmpeg 由安装脚本自动拉取。

```bash
git clone https://github.com/Tencent/WeVisCut.git && cd WeVisCut

pip install -r requirements.txt
# GPU 机器：先安装与 CUDA 匹配的 torch，再
pip install -r requirements-gpu.txt

bash scripts/setup_external_tools.sh          # ffmpeg、Node、渲染器（约 1.1 GB）

pip install -U "huggingface_hub[cli]"
hf download tencent/WeVisCut --local-dir checkpoints/weviscut-9b
```

### 启动 Studio

```bash
bash run_studio.sh --check                    # 路径校验，不需要 GPU
bash run_studio.sh                            # 默认 http://<host>:8081/
# 仅界面、不加载权重：
# bash run_studio.sh --no-model
```

权重在后台加载（约 30–60 秒），界面立即可用。权重放在别处时：`export WEVISCUT_MODEL_PATH=/path/to/WeVisCut-9B`。

> [!TIP]
> 访问不了 `github.com` 时，把 `requirements.txt` 里的 git pin 换成 `pip install deepagents==0.7.7`。加载权重需要能识别 `qwen3_5` 的 `ms-swift`（否则设 `AICUT_MS_SWIFT_PATH`）。

### 使用 Studio

1. 打开 `http://<host>:8081/`（trustworthy-cut 界面：`/0907_newvis`）。
2. 粘贴口播 / 知识长文并开始生成，八条轨道随智能体推进填充。
3. 成片后：改某一镜的口播、素材、模板、BGM，或用对话改片（「把开头第一句改掉」「把 0:15 的画面换掉」）。
4. 重渲染只处理改过的镜头；历史支持版本回滚。

默认素材 / BGM 走公开网络检索（`asset_retrieval_tool`）。在 `studio_config.yaml` 里可换成你自己的 HTTP 接口、本地文件夹或 Python 插件，见 [`docs/providers.md`](docs/providers.md)；日常只改这一份配置即可。

```bash
python3 -m studio.config --summary
```

<details>
<summary><b>常见问题排查</b></summary>
<br>

| 现象 | 查看 |
| :--- | :--- |
| 缺模块 | `requirements.txt` / `requirements-gpu.txt` |
| 模型一直 loading | 日志；`checkpoints/weviscut-9b`（或 `WEVISCUT_MODEL_PATH`）；GPU 是否可见 |
| 某个工具反复失败 | 前端终端，多半是 TTS / 检索 / 渲染 |
| thinking 面板空白 | `model.enable_thinking: true` |
| 端口占用 | `bash run_studio.sh --port <空闲端口>` |

</details>

## 🧾 轨迹构建管线

```bash
pip install -e data_construct_pipeline
longvideo-polar smoke --pipeline-root .     # 离线接线检查，不需要 GPU
longvideo-polar run --script-file /path/to/script.txt --output ./rollouts/demo
```

完整 rollout 还需要已启动的 OpenAI 兼容模型、TTS 和 ffmpeg。批量任务与三路落盘格式见管线 [README](data_construct_pipeline/README_zh.md)。

## 📚 文档

| 文档 | 内容 |
| :--- | :--- |
| [`docs/architecture.md`](docs/architecture.md) | 运行时架构、工具循环、状态 JSON |
| [`docs/configuration.md`](docs/configuration.md) | `studio_config.yaml` 配置参考 |
| [`docs/providers.md`](docs/providers.md) | 自定义素材 / 配乐来源 |
| [`docs/huggingface.md`](docs/huggingface.md) | 权重下载与模型 profile |
| [`docs/external_tools.md`](docs/external_tools.md) | 安装脚本拉取的第三方运行时 |

## 📖 引用

如果 WeVisCut 对你的研究有帮助，欢迎给我们一个 ⭐ 并引用：

```bibtex
@misc{weviscut2026,
  title        = {WeVisCut: A Long-Horizon Tool Agent for Turning Knowledge Articles into Edited Videos},
  author       = {WeVisCut Team},
  organization = {Tencent},
  year         = {2026},
  howpublished = {\url{https://tencent.github.io/WeVisCut/}},
  note         = {Technical report and code release}
}
```

## 📜 许可

本仓库中由腾讯编写的代码以 [Apache License 2.0](LICENSE) 发布。第三方运行时与检索到的素材保留各自的授权。
