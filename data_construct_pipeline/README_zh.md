# WeVisCut 数据构造管线

[English](README.md) · [中文](README_zh.md) · [返回 WeVisCut](../README_zh.md)

这个包负责 **WeVisCut 的训练轨迹落盘**：一份口播稿进去，一条成片加上三份按角色切开的数据流出来。

它是论文的数据构造侧，不是上线用的 Studio。Studio（`aicut_agent/`、`studio/`）跑的是已经训好的策略；这个包 **生产** 那条策略当初吃进去的轨迹。

```
口播稿
  → LongVideoRunner
  → InferenceOrchestrator.run_agentic_pipeline   (server/infer_graph.py)
       每一步：主流程观察状态 → 发出一个 tool call
       分发：generator / critic 技能，或外部媒体工具
  → TrajectoryRecorder 把整轮切成三份流
```

---

## 这是自主选工具，还是固定 workflow？

**主流程自己选下一步工具。** 每一轮它看到完整的 12 工具 action space、当前 flags、近期历史、上一步 observation 和工具契约，然后只发一个 `tool_call`。成功跑通并不依赖写死的 `STAGE_SEQUENCE`——那条旧路径只在 `agent.agent_mode` 不是 `long_horizon` 时才会走。

**运行时也不是完全放飞的 agent。** 前置条件、重复上限、解析失败兜底，以及终点工具 `Tool_StitchVideo` 的门控，用来避免一轮标注卡死或跳过质检弧。这些干预会写进轨迹，本身就是训练信号，不是事后偷偷改决策。

**内容、反思、媒体执行是解耦的。** 同一条长程 episode 落成三份有因果关系的流：

| 角色 | 职责 | 落盘形态 |
|---|---|---|
| `orchestrator` 主流程 | 决定下一步调哪个工具 | 一条多轮 tool-calling episode |
| `generator` 内容生成器 | 分段、分镜、装饰、补丁修复 | 每次调用一条单轮样本 |
| `critic` 反思修复 | 计划自检、多维质检、严重度排序 | 每次调用一条单轮样本 |

`interactions.jsonl` 记下 `第 N 步 → 交给谁 → 返回什么`，三份文件是一条因果轨迹，不是三份互不相关的日志。

所以模型可以按「长程工具决策」来训，内容技能和质检技能也可以单独换、单独蒸。

---

## 和仓库其余部分的关系

| 本包 | 发布用的 WeVisCut Studio |
|---|---|
| 训练时循环，在 `server/infer_graph.py` | `aicut_agent/` 的 deepagents / LangGraph harness |
| 运行时可以重定向非法 / 重复调用，保证标注跑完 | 护栏只观察，不替换成另一个工具 |
| 写出 orchestrator / generator / critic 的 SFT+RL 数据 | 浏览器里跑一个训好的 9B 策略 |

`pipeline.root` 默认就是 **本仓库**（`pipeline/` + `server/`），不需要再指一份内部 AICutPipeline。

---

## 安装

在 WeVisCut 仓库根目录：

```bash
pip install -e data_construct_pipeline            # 核心（httpx + pyyaml）
pip install -e "data_construct_pipeline[dev]"     # pytest
pip install -e "data_construct_pipeline[web]"     # 批量标注控制台
```

真出片还需要：

- 本仓库的 `pipeline/`、`server/`、`configs/`、`external_tool/`
- OpenAI 兼容的 served model（vLLM / SGLang）
- TTS：`bash data_construct_pipeline/services/tts/run_qwen3_tts_service.sh`
- `ffmpeg` / `ffprobe`（见上层 [外部工具说明](../docs/external_tools.md)）

---

## 快速开始

```bash
# 0. 路径、system prompt、stage 表、三个角色端点
longvideo-polar check

# 0.5 离线接线自测 —— 不需要 GPU、key、TTS
longvideo-polar smoke --pipeline-root ..

# 1. 单条：口播稿 → 成片 + 三份轨迹
longvideo-polar run --script-file /path/to/口播稿.txt --output ./rollouts/demo

# 2. 批量（进程隔离、可续跑、自动汇总）
longvideo-polar batch --input-dir /path/to/scripts --workers 4

# 3. 反思修复换更强的教师模型，做蒸馏
longvideo-polar batch --input-dir /path/to/scripts --profile critic_gemini
```

Python：

```python
from longvideo_polar import LongVideoConfig, LongVideoRunner

config = LongVideoConfig.load("data_construct_pipeline/config/longvideo_polar.yaml")
result = await LongVideoRunner(config).run(script=script_text, episode_id="ep_001")
print(result.summary())
```

`agent.agent_mode` **必须** 是 `long_horizon`。写成别的会走旧的固定阶段模式，构建时直接报错，不会静默降级。

---

## 模型组合

三个端点互相独立。`profiles` 深合并到 `models` 上。

| profile | 主流程 | 内容生成器 | 反思修复 | 适用 |
|---|---|---|---|---|
| `all_qwen`（默认） | 本地 Qwen | 本地 Qwen | 本地 Qwen | 离线 / RL |
| `critic_gemini` | Qwen | Qwen | Gemini | 更高质量的质检标签 |
| `critic_opus` | Qwen | Qwen | Claude Opus | 同上，换教师 |
| `skills_gemini` | Qwen | Gemini | Gemini | 只训主流程 |

被训练的对象必须是当时真正在做决策 / 生成的那个模型。教师轨迹用 `metadata.provider` 标清，合并分片前不要和自生成样本混在一起。

---

## 落盘产物

```
<output>/<episode_id>/
├── episode.json                 # 清单、reward、错误、耗时
├── <project_id>/                # 管线产物（分镜、素材、成片）
│   └── agent_trace.jsonl
└── trajectories/
    ├── orchestrator.jsonl       # 多轮 tool-calling episode
    ├── generator.jsonl          # 单轮技能样本
    ├── critic.jsonl
    ├── interactions.jsonl       # 跨角色边
    ├── llm_calls.jsonl          # 原始 I/O 审计
    ├── sft/{orchestrator,generator,critic}_messages.jsonl
    └── rl/reward_scores.jsonl
```

批次结束会在 `aggregate/` 写出合并后的 `orchestrator_sft.jsonl`、`generator_sft.jsonl`、`critic_sft.jsonl`、`interactions.jsonl`、`rewards.jsonl`、`index.json`。

---

## 配置

控制面是 [`config/longvideo_polar.yaml`](config/longvideo_polar.yaml)。编排器读的是 [`config/pipeline_runtime.yaml`](config/pipeline_runtime.yaml)（执行、TTS、模版），**不是** `studio_config.yaml`。

| 小节 | 控制什么 |
|---|---|
| `pipeline` | 环境根目录、运行时 YAML、system prompt、TTS |
| `profiles` / `models` | 三个角色的端点与采样 |
| `agent` | `long_horizon`、步数上限、重复上限 |
| `tool_switches` | TTS / 检索 / 渲染 / 拼接 |
| `recording` | 落盘位置、是否保留完整 prompt |
| `batch` | 输入 glob、并发、续跑 |

常用环境变量：`WEVISCUT_ROOT`、`AICUT_PIPELINE_ROOT`、`OPENAI_BASE_URL`、`OPENAI_API_KEY`、`LVP_MODEL`、`LVP_PROFILE`、`LVP_OUTPUT_ROOT`、`LVP_INPUT_DIR`。

---

## 目录

```
data_construct_pipeline/
├── config/                 # 总配置 + 运行时 YAML + 批量标注示例
├── src/longvideo_polar/    # runner、recorder、角色路由、bridge
├── services/tts/           # 可选的 Qwen3-TTS 辅助脚本
├── scripts/                # 批量标注控制台启动脚本
└── tests/                  # 离线单测（不需要 GPU）
```

```bash
cd data_construct_pipeline && PYTHONPATH=src python -m pytest -q
```

---

## 排查

| 现象 | 从哪查 |
|---|---|
| `pipeline.root does not exist` | 把 `pipeline.root` 指到本仓库，或导出 `WEVISCUT_ROOT` |
| `missing server/infer_graph.py` | checkout 不完整，循环在 `server/infer_graph.py` |
| `agent_mode=... does not select the long-horizon loop` | 写成 `long_horizon` |
| 每步都 `parse_ok: false` | served model 不支持原生 tool calling → `native_tool_calls: false` |
| 有轨迹没成片 | 看 `episode.json` 的 errors，多半卡在渲染或拼接 |
| 调试太慢 | 关掉 `tool_switches.hyperframes_render` / `tts`（此时没有真成片） |
