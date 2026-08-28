# asset_retrieval_tool

WeDirector 的默认素材检索后端：轻量的在线视频检索 + 免费 BGM 检索库。

不依赖任何内部服务，公开网络可达即可工作，因此适合作为开箱即用的默认 provider。

## 功能

| 接口 | 说明 |
|------|------|
| `search_video_segments` | 从 YouTube / Bilibili 在线检索匹配的视频素材 |
| `search_bgm` | 从 Internet Archive netlabels 检索免费 CC 授权纯音乐 BGM |
| `download_bgm_results` | 将 BGM 检索结果下载为 MP3 文件到本地目录 |
| `download_video_results` | 将视频检索结果下载为真实 mp4 文件（yt-dlp 抽流 + FFmpeg 合并） |

- **检索**（`search_video_segments` / `search_bgm`）+ **BGM 落盘**：纯 Python 标准库，零第三方依赖。
- **视频落盘**（`download_video_results`）：需要 `yt-dlp` + 系统 `ffmpeg`，
  因为平台页面 URL 不是媒体直链。详见[下载落盘](#下载落盘)。

---

## 安装

```bash
cd asset_retrieval_tool
pip install -e .
# 或直接把父目录加入 PYTHONPATH
```

---

## 快速上手

### 视频检索（YouTube + Bilibili）

```python
from asset_retrieval_tool import search_video_segments

# 仅搜索 Bilibili（无需 API Key）
results = search_video_segments(
    query="晨光下校园航拍，学生步行，氛围积极",
    top_k=10,
    min_duration=30.0,
    max_duration=600.0,
    sources=["bilibili"],
)

# 同时搜索 YouTube + Bilibili（需要 YouTube Data API v3 Key）
results = search_video_segments(
    query="campus morning aerial shot students walking",
    top_k=10,
    min_duration=30.0,
    max_duration=600.0,
    sources=["youtube", "bilibili"],
    youtube_api_key="YOUR_YOUTUBE_API_KEY",
)

for r in results:
    print(r["rank"], r["score"], r["platform"], r["title"], r["source_video"])
```

每条结果字段：

```python
{
    "rank": 1,
    "score": 0.72,
    "platform": "bilibili",          # "youtube" 或 "bilibili"
    "video_id": "BV1xx411c7mD",
    "source_video": "https://www.bilibili.com/video/BV1xx411c7mD",
    "title": "校园晨光航拍 4K",
    "description": "...",
    "channel": "UP主名称",
    "thumbnail": "https://...",
    "start": 0.0,
    "end": 180.0,
    "duration": 180.0,
    "semantic_text": "校园晨光航拍 4K ...",
}
```

#### 获取 YouTube Data API v3 Key

1. 前往 [Google Cloud Console](https://console.cloud.google.com/)
2. 创建项目 → 启用 **YouTube Data API v3**
3. 创建 API 凭据 → API Key
4. 免费配额：每日 10,000 单位（每次搜索约消耗 100 单位）

---

### BGM 检索

```python
from asset_retrieval_tool import search_bgm

tracks = search_bgm(
    query="upbeat corporate background music",
    top_k=5,
    min_duration=60.0,
    max_duration=300.0,
)

for t in tracks:
    print(t["rank"], t["title"], t["artist"], t["duration_sec"], t["download_url"])
```

每条结果字段：

```python
{
    "rank": 1,
    "source": "internet_archive",
    "id": "DWK123/ProleteR_-_01_-_April_Showers.mp3",
    "title": "ProleteR   01   April Showers",
    "artist": "ProleteR",
    "duration_sec": 269.0,
    "genre": "netlabel / CC",
    "url": "https://archive.org/details/DWK123",
    "download_url": "https://archive.org/download/DWK123/ProleteR_-_01_-_April_Showers.mp3",
    "license": "CC",
}
```

---

### 下载落盘

```python
from asset_retrieval_tool import search_bgm, search_video_segments
from asset_retrieval_tool import download_bgm_results, download_video_results

# 下载 BGM 到 ./output/bgm/
bgm_results = search_bgm("ambient chill", top_k=5)
downloaded = download_bgm_results(bgm_results, output_dir="./output", max_workers=4)
for r in downloaded:
    print(r.get("local_path") or r.get("download_error"))

# 下载视频到 ./output/video/
video_results = search_video_segments("校园晨光", sources=["bilibili"], top_k=5)
downloaded = download_video_results(
    video_results,
    output_dir="./output",
    max_workers=3,
    cookie=BILIBILI_COOKIE,   # B站登录态内容需要
    max_height=480,           # 画质上限，控制文件大小
    clip_seconds=15.0,        # 只截前 15 秒，短素材场景
)
for r in downloaded:
    print(r.get("local_video_path"), r.get("local_video_bytes"))
```

`download_video_results` 按顺序尝试三条路径：

| 情况 | 处理方式 |
|------|----------|
| 本地文件路径 | 直接 `shutil.copy2` 复制 |
| 直链媒体 URL（`.mp4` / `.webm` …） | `urllib` 流式下载 |
| 平台页面 URL（B站 / YouTube …） | **yt-dlp** 抽取 DASH 流 + FFmpeg 合并 |

> **为什么必须要 yt-dlp**：`search_video_segments` 返回的 `source_video` 是
> `https://www.bilibili.com/video/BVxxx` 这样的**网页地址**，不是媒体直链。
> 直接 `urlretrieve` 拿到的只是 HTML。B站的音视频还是 DASH 分离流，需要
> FFmpeg 合并成单个 mp4。

安装：

```bash
pip install yt-dlp
brew install ffmpeg        # macOS；Ubuntu 用 apt install ffmpeg
```

未安装 yt-dlp 时，平台页面 URL 会在结果里返回 `download_error`，
其余两条路径不受影响（核心检索仍是纯标准库）。

---

### Bilibili Cookie 配置

B站搜索接口和 playurl 接口都有反爬风控，**匿名请求会返回
`HTTP 412 Precondition Failed`**。传入自己浏览器的 Cookie 即可正常使用。

获取方式：浏览器登录 bilibili.com → F12 → Network → 任意请求 →
Request Headers → 复制 `Cookie:` 整行。

推荐放到项目根目录的 `config.json`：

```json
{
  "bilibili_cookie": "buvid3=...; SESSDATA=...; bili_jct=...",
  "youtube_api_key": ""
}
```

读取（环境变量优先，便于 CI）：

```python
import json, os
from pathlib import Path

cookie = os.environ.get("BILIBILI_COOKIE", "")
cfg = Path("config.json")
if not cookie and cfg.exists():
    cookie = json.loads(cfg.read_text(encoding="utf-8")).get("bilibili_cookie", "")
```

> ⚠️ **Cookie 等同于账号凭证**（`SESSDATA` / `bili_jct` 可代表你操作账号）。
> 务必把 `config.json` 加进 `.gitignore`，不要提交、不要写进日志。
> Cookie 会过期，失效后重新抓一次即可。

内部实现要点：yt-dlp 必须走 **Netscape 格式 cookiefile**，直接塞 `Cookie:`
请求头会绕过它内部的 cookiejar，仍然被风控拦成 412。
`download_video_results` 已自动做这个转换（写入临时文件、用完即删）。

---

## 纯音乐过滤机制

`search_bgm` 采用双重过滤保障返回结果 100% 为纯音乐：

**Item 级过滤（subject 关键词黑名单）**

整个 IA item 的 subject 字段含以下词时，整个 item 跳过：

```
poetry, reading, spoken word, interview, lecture, podcast, speech, talk,
comedy, radio, audiobook, dictionary, conceptual writing, cut-up, news,
drama, theatre, theater, field recording, nature sounds, sound art,
soundscape, radio show, broadcast, narration, storytelling
```

**File 级过滤（文件名关键词黑名单）**

单个 MP3 文件名含以下词时，该文件跳过：

```
interview, podcast, speech, spoken, reading, lecture, radio, talk,
narration, commentary, intro_speech, field_recording, nature, ambient_noise
```

**数据源选择**

使用 [Internet Archive netlabels 合集](https://archive.org/details/netlabels)，该合集专门收录 CC 授权电子音乐、器乐、实验音乐，是目前无需 API Key 且可稳定访问的最纯净免费音乐源。

---

## 接入 AICutPipeline

在 `pipeline/tools/retrieve_visual.py` 的 `search_clips_with_backend()` 里加一个 `online_semantic` backend：

```python
from asset_retrieval_tool import search_video_segments as _online_search

def _search_via_online(visual_query, top_k, target_duration,
                       min_duration_ratio, max_loop_count,
                       youtube_api_key=None):
    min_dur = max(10.0, target_duration * min_duration_ratio) if target_duration > 0 else 30.0
    max_dur = target_duration * max_loop_count if target_duration > 0 else 600.0
    raw = _online_search(
        query=visual_query,
        top_k=top_k,
        min_duration=min_dur,
        max_duration=max_dur,
        youtube_api_key=youtube_api_key,
    )
    return [
        {
            "clip_id": f"online_{r['platform']}_{r['rank']:03d}",
            "clip_path": r["source_video"],
            "score": r["score"],
            "duration_sec": r["duration"],
            "description": r["title"],
            "tags": [],
            "start": r["start"],
            "end": r["end"],
        }
        for r in raw
    ]
```

然后在 `config.yaml` 的 `backend_order` 里加上：

```yaml
retrieval:
  backend_order:
    - online_semantic
    - mmfinder_video_cli
    - local_scan
```

---

## 运行测试

### 单元测试

```bash
cd asset_retrieval_tool
pip install pytest pytest-cov
pytest asset_retrieval_tool/tests/ -v --tb=short
```

带覆盖率：

```bash
pytest asset_retrieval_tool/tests/ --cov=asset_retrieval_tool --cov-report=term-missing
```

### 端到端集成测试（真实网络）

```bash
# 仅 Bilibili（无需配置）
python test_results/run_integration_test.py

# 同时测试 YouTube（需要 API Key）
YOUTUBE_API_KEY=your_key python test_results/run_integration_test.py
```

测试结果保存到：

```
test_results/bgm_results.json        # BGM 检索结果（含纯音乐验证）
test_results/video_results.json      # 视频检索结果
test_results/downloaded/bgm/         # 下载的 BGM MP3 文件
```

---

## 并发支持

两个检索接口均线程安全，支持高并发：

- `search_video_segments` 内部使用 `ThreadPoolExecutor` 并发请求多个平台。
- `search_bgm` 并发发起多个 IA 请求，无全局可变状态。
- 可通过 `asyncio.to_thread` 在异步 worker 中调用。

---

## 依赖

运行时：Python 3.9+ 标准库（无第三方包）。

测试：`pytest`、`pytest-cov`。

网络请求使用 `urllib.request`（标准库），无需 `requests`、`httpx` 或 `aiohttp`。
