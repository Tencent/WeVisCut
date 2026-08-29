# WeDirector

**面向端到端视频剪辑的长程工具智能体。**

WeDirector 把一段纯文本口播稿变成一条剪好的、时长可达数分钟的成片。整个过程由
一个 9B 策略模型驱动（先 SFT、再用 DAPO 强化学习训练）：它切分脚本、规划分镜、
合成口播、检索素材、挑选并填充动画模板、配背景音乐、审查自己的产出、修掉不满意
的部分，最后拼接成片——一次一个工具调用，没有任何脚本化流水线告诉它下一步该做
什么。

一次典型运行是 12 次左右的工具调用，产出数分钟的视频。这里没有手写状态机：轨迹
是模型自己走出来的。

[English](README.md) · [中文](README_zh.md)

---

## 仓库里有什么

| 部分 | 位置 |
|---|---|
| 12 个工具构成的视频剪辑环境 | `pipeline/` |
| 与训练对齐的推理引擎（状态序列化、工具 schema、解析） | `server/` |
| 基于 [deepagents](https://github.com/langchain-ai/deepagents) / LangGraph 的智能体框架 | `aicut_agent/` |
| FastAPI 后端、剪辑与对话式改片接口 | `studio/` |
| 浏览器前端：实时时间轴、手动剪辑、对话改片 | `static/` |
| **可插拔的素材 / BGM 检索接口** | `pipeline/providers/` |
| 公开网络素材与 CC 音乐检索 | `external_tool/asset_retrieval_tool/` |
| Gold Template 模板库（结构 × 运动 × 视觉） | `external_tool/gold_template/` |

模型权重和体积较大的第三方运行时需要单独下载，见 [安装](#安装)。

### 文档

| 文档 | 内容 |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | 智能体循环、运行时、护栏如何组合，以及为什么这样设计 |
| [`docs/configuration.md`](docs/configuration.md) | `studio_config.yaml` 全部字段与环境变量 |
| [`docs/providers.md`](docs/providers.md) | 接入你自己的视频素材与音乐源 |
| [`docs/external_tools.md`](docs/external_tools.md) | 需要下载的大文件：是什么、放哪里、多大 |
| [`docs/huggingface.md`](docs/huggingface.md) | 使用与发布模型权重 |

---

## 工作方式

```
口播稿
  → POST /api/runs
  → 每次运行一个 AICutRuntime（持有权威剪辑状态）
  → create_deep_agent(model=AICutChatModel, tools=12, system_prompt=训练用提示词,
                      middleware=[AICutGuardrailMiddleware])
  → LangGraph ReAct 循环：
       AICutChatModel   从运行时的权威状态重建单轮状态 JSON，调用策略模型，
                        解析出一次工具调用
       Guardrail        前置条件 / 重复 / 预算检查（可以拒绝或终止，
                        但绝不改写模型的决策）
       Tool             在真实环境中执行
       Tool_StitchVideo 成功 → Command(goto=END)
  → 事件通过 WebSocket 推给浏览器
  → 成片写入 workspace/projects/<project_id>/
```

有两个设计选择值得单独说明，因为正是它们让智能体在部署时的行为与训练时保持一致。

**运行时是状态的唯一权威。** 每一步，chat model 都会*丢弃*框架累积的消息历史，
从运行时的权威状态重新构造一条单轮 user 消息；工具 schema 也由训练数据生成器所
用的同一份定义重新生成。因此模型在推理时看到的内容与训练时逐字节一致——不会因为
对话越来越长而漂移，也不会出现两份 schema 定义互相打架。

**护栏只观察，不代劳。** 中间件可以拒绝前置条件不满足的调用，也可以在预算耗尽时
终止运行，但它绝不会换成另一个工具调用。非法调用会作为 observation 反馈给模型，
由模型自己纠正。所有"帮模型一把"的服务端兜底默认关闭（配置里的
`agent.autonomy.*`）——一个被训练成能自我恢复的策略，就应该按它是否真的做到来
衡量。

### 12 个工具

| 工具 | 作用 |
|---|---|
| `Tool_SegmentArticle` | 把脚本切成章节 |
| `Tool_PlanShots` | 为一个章节规划分镜：口播、画面检索词、模板意图 |
| `Tool_GenerateAudio` | TTS 口播；确定每个镜头的真实时长 |
| `Tool_RetrieveVisual` | 按镜头的画面检索词找素材 |
| `Tool_DecorateShot` | 挑选模板三元组、填槽、校验、修复 |
| `Tool_TimelineCompose` | 渲染单个镜头：素材 + 口播 + 字幕 + 模板层 |
| `Tool_MusicSearch` | 生成覆盖整条时间轴的 BGM 播放列表 |
| `Tool_TextQuickJudge` | 低成本的纯文本质量门 |
| `Tool_VLMRenderJudge` | 对渲染结果做视觉审查 |
| `Tool_SelfReflect` | 决定下一步修什么 |
| `Tool_ModalProbe` | 外部服务预检 |
| `Tool_StitchVideo` | 归一化并拼接所有镜头，产出成片 |

---

## 安装

### 环境要求

- Python **>= 3.11**
- 生成需要一块显存足够跑 bf16 9B 模型的 NVIDIA GPU（约 24 GB）。只做前端和接口
  开发不需要 GPU（用 `--no-model`）。
- Node **>= 22** 和 ffmpeg，都可以用下面的脚本装。

### 1. Python 依赖

```bash
git clone https://git.woa.com/chenyumu/WeDirector.git
cd WeDirector

pip install -r requirements.txt

# 仅推理机（GPU）需要。先装匹配你 CUDA 版本的 torch：
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu.txt
```

有两处 pin 是关键的。

**harness。** `deepagents` 从上游 `0.7.7` release tag 安装，而不是从 PyPI 装，这样
代码树与本策略开发时所用的完全一致。这条路径需要 `git` 和访问 `github.com` 的能力；
在无法访问的环境下，发布包是等价的（该 tag 上真正打包的源码与之完全相同），改用
`pip install deepagents==0.7.7` 即可。两种写法都不要放宽版本约束：
`aicut_agent/agent.py` 依赖 `HarnessProfile` 和 `register_harness_profile` 来剔除
内置工具集，如果 harness 版本改动了这些 API，结果是直接 import 失败，而不是悄悄
降级。

**`ms-swift`。** 必须是注册了 `qwen3_5` 模型类型的版本。如果你装的那个没有，把
`model.ms_swift_path` 指向一个包含该注册的 checkout，或者导出
`AICUT_MS_SWIFT_PATH`。

### 2. 外部运行时

```bash
bash scripts/setup_external_tools.sh
```

脚本会把 ffmpeg、Node、HyperFrames 渲染器、headless Chrome、GSAP 和字幕字体装到
`external_tool/` 下，约 1.1 GB。已存在的组件会跳过；如果某项下载失败，对应的
`external_tool/*/README.md` 里有手动安装步骤。

### 3. 模型权重

```bash
pip install -U "huggingface_hub[cli]"
hf download <org>/WeDirector-9B --local-dir checkpoints/wedirector-9b
```

之后要么用默认路径（`checkpoints/wedirector-9b`），要么指向你自己的副本：

```bash
export WEDIRECTOR_MODEL_PATH=/data/models/wedirector-9b
```

完整说明（含各组件体积、如何跳过某一项）见
[`docs/external_tools.md`](docs/external_tools.md)。

### 4. 自检并启动

```bash
bash run_studio.sh --check     # 校验所有解析后的路径，不需要 GPU
bash run_studio.sh             # 用配置里的端口启动
bash run_studio.sh --no-model  # 只起前端 + 接口，不加载权重、不需要 GPU
```

浏览器打开 `http://<host>:8081/`。权重在后台加载，约 30–60 秒；界面会立刻可用并
显示加载进度。

---

## 配置

`studio_config.yaml` 是唯一需要改的文件，代码里没有别的地方读死路径。配置值支持
`${VAR}` 和 `${VAR:-默认值}`，并且会自动注入 `${REPO_ROOT}`，所以一份新克隆的仓库
不导出任何环境变量也能跑。

| 配置段 | 控制什么 |
|---|---|
| `model` | 权重 profile、解码参数、thinking 模式、训练用系统提示词 |
| `agent` | 步数与成本上限、自主性开关 |
| `services.retrieval_provider` | **视频素材从哪来** |
| `services.bgm_provider` | **音乐从哪来** |
| `services.providers` | 各 provider 的参数与 Python 插件 |
| `services.tts` | 口播服务 |
| `services.hyperframes` | 渲染器、浏览器、GPU 加速 |
| `services.ffmpeg` | ffmpeg / ffprobe 可执行文件 |
| `paths` | 素材库、模板库、输出根目录 |
| `render` | 字幕、分层渲染加速、拼接 |
| `editor`、`chat_edit` | 后期编辑与对话式改片 |

查看实际生效的配置：

```bash
python3 -m studio.config --summary     # 解析后的接口 + 已注册的 provider
python3 -m studio.config --dump pipeline
```

完整字段说明和环境变量清单见
[`docs/configuration.md`](docs/configuration.md)。

---

## 接入你自己的视频素材和音乐

检索是整个系统里唯一必须和你自己的媒体资源对接的部分，所以它被设计成一等扩展点。
三种方式，按工作量从小到大。

### 开箱即用：公开网络检索

默认方式。视频走 Bilibili / YouTube 搜索，BGM 走 Internet Archive 上的
netlabel（CC 授权的纯音乐）。不需要额外部署服务，Bilibili 那条路也不需要 API key。

```yaml
services:
  retrieval_provider: asset_retrieval_tool
```

### 只改配置：接你自己的 HTTP 接口

如果你已经有一个检索服务，用 YAML 描述它即可，不用写代码：

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

返回字段里已经叫 `url`、`path`、`title`、`duration` 的会被自动识别；`field_map`
只用来映射那些名字对不上的字段。

### 一个本地文件夹

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

匹配依据是文件名、上层目录名，以及可选的 `<文件名>.json` 旁挂文件（可写
`caption` / `tags` / `duration_sec`）。

### 写 Python：任何其他情况

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
    plugins: [my_providers]          # 模块名，或 /path/to/my_providers.py
```

你有什么字段就返回什么，注册表会做归一化并补齐其余字段。抛异常或返回空的
provider 会被当作"没有候选"，接着走下一个后端——所以一个不稳定的源只会降级，不会
让整次运行失败。

[`docs/providers.md`](docs/providers.md) 里有完整契约：精确的 schema、
`VideoQuery` / `BgmQuery` 每个字段的含义、时长与画幅过滤会怎么处理你的结果，以及
如何离线测一个 provider。

---

## 使用 Studio

1. **生成。** 粘贴口播稿，点开始。八条实时轨道（章节、分镜、口播、字幕、素材、
   高亮、BGM、渲染）随智能体推进逐步填充，旁边同时显示它的思考、工具调用和原始
   输出。随时可以暂停、继续、取消。
2. **时间轴。** 多轨查看每个镜头的位置、时长、素材、模板，点击即选中。
3. **手动剪辑。** 对选中的镜头：改口播（会重跑 TTS）、改字幕（不重跑）、换素材或
   换模板、删除、调序、调整 BGM。重渲染只处理你改过的镜头，然后重新拼接。
4. **对话改片。** 例如"把开头第一句改成……""把 0:15 的画面换掉"。模型定位目标并
   给出修改建议；破坏性操作需要确认。
5. **历史。** 查看过往运行与已完成项目，支持版本回滚。

---

## 接口

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/status` | 模型状态、进行中的运行 |
| GET | `/api/config` | 前端需要的配置（轨道、特性开关、编辑器） |
| POST | `/api/runs` | 启动一次运行：`{input_text}` → `{run_id}` |
| GET | `/api/runs` · `/api/runs/{id}` | 列表 / 详情 |
| POST | `/api/runs/{id}/pause` · `resume` · `cancel` | 控制 |
| DELETE | `/api/runs/{id}` | 删除运行记录 |
| WS | `/ws/{run_id}` | 实时事件流 |
| GET | `/api/projects` · `/api/project?dir=` | 项目 |
| POST | `/api/project/ops` · `regenerate` | 应用编辑 · 重渲染 |
| GET | `/api/project/versions?dir=` · POST `/api/project/restore` | 版本 |
| POST | `/api/project/chat` | 对话式改片 |
| GET | `/api/media?path=` | 提供媒体文件（仅白名单目录） |
| POST | `/v1/chat/completions` | OpenAI 兼容接口；流水线内部的 LLM 调用复用同一个策略模型 |

WebSocket 事件：`run_started`、`stage_update`、`thinking`、`tool_call`、
`tool_result`、`progress`、`preview_update`、`console_log`、`error`、
`pipeline_complete`、`run_complete`、`run_paused`、`run_resumed`。

---

## 测试

下面这些全部离线运行，不需要 GPU：

```bash
python3 tests/test_studio_offline.py           # 配置、轨道、编辑器、对话解析、护栏
python3 -m pytest tests/test_aicut_agent.py    # deepagents 集成（mock loader + 分发）
python3 -m pytest tests/test_retrieval_provider_switch.py   # provider 路由
python3 -m pytest tests/test_providers.py      # provider 注册表与内置 provider
python3 -m pytest tests/test_thinking_parse.py # think / tool_call 解析
node tests/test_frontend.js                    # 前端（需要 node）
```

`test_aicut_agent.py` 断言的正是那些让推理与训练保持对齐的性质：模型看到的恰好
是训练时那 12 个工具（deepagents 自带的内置工具被移除）、格式转换可往返、前置
条件能正确拦截、拼接成功会终止图、纯文本对话仍然可用。

---

## 排查

| 现象 | 从哪查 |
|---|---|
| 启动报缺模块 | `pip install -r requirements.txt`；GPU 机器还需要 `requirements-gpu.txt` |
| 模型一直停在 "loading" | 看日志；确认 `model.profiles.<active>.base_model` 存在、GPU 可见 |
| 某个工具反复失败 | 前端的终端面板里有真实报错，通常是某个外部服务（TTS / 检索 / 渲染）挂了 |
| `retrieval_provider ... is not registered` | 名字写错，或插件导入失败。`python3 -m studio.config --summary` 会列出真正注册成功的 provider |
| thinking 面板空白 | `model.enable_thinking` 必须为 `true`，否则 chat template 会提前闭合 `<think>` |
| 渲染超时 | 没有可用 GPU 的机器上把 `services.hyperframes.accelerated_render.enabled` 设为 `false` |
| 端口被占用 | `bash run_studio.sh --port <空闲端口>` |

---

## 给论文读者的说明

`AICutPipeline` 是这个系统的内部代号，它仍然出现在训练用系统提示词和模型收到的
状态 JSON 里。这些字符串属于训练分布的一部分，被刻意保持逐字节一致——改名会让推理
偏离分布。对外名称是 WeDirector。

`configs/training/` 放的是发布 checkpoint 背后的数据准备与 SFT 配置，推理时不会
读取。

## 许可

Apache License 2.0，见 [LICENSE](LICENSE)。随仓库分发或下载的第三方组件
（HyperFrames、GSAP、ffmpeg、Node、Noto 字体、deepagents）各自遵循其原有许可；
检索到的视频素材与音乐同样保留其原有授权，遵守这些授权是部署方的责任。
