"""从 bgm_results.json + video_results.json 生成 report.html。"""
from __future__ import annotations

import html
import json
import os
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent
BGM_JSON = RESULTS_DIR / "bgm_results.json"
VIDEO_JSON = RESULTS_DIR / "video_results.json"
OUT_HTML = RESULTS_DIR / "report.html"

_CSS = """
:root {
  --paper:      #fbfaf7;
  --surface:    #ffffff;
  --ink:        #16181d;
  --ink-2:      #565c66;
  --ink-3:      #949aa3;
  --rule:       #e8e5df;
  --rule-firm:  #d2cec6;
  --accent:     #1f3f9e;
  --ok:         #0f7549;
  --ok-soft:    #eef6f1;
  --warn:       #a3540b;
  --warn-soft:  #fdf4e7;

  --text-xs:    0.72rem;
  --text-sm:    0.82rem;
  --text-base:  0.94rem;
  --text-lg:    1.15rem;
  --text-kpi:   clamp(1.6rem, 1.2rem + 1.4vw, 2.35rem);
  --text-title: clamp(1.35rem, 1.1rem + 1vw, 1.9rem);

  --space-sec:  clamp(2.5rem, 2rem + 2vw, 4rem);
  --radius:     8px;
  --dur:        160ms;
  --ease:       cubic-bezier(0.16, 1, 0.3, 1);
}

* { box-sizing: border-box; margin: 0; padding: 0 }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue',
               'PingFang SC', 'Hiragino Sans GB', sans-serif;
  background: var(--paper);
  color: var(--ink);
  font-size: var(--text-base);
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}

.wrap { max-width: 1180px; margin: 0 auto; padding: clamp(2rem, 4vw, 3.5rem) 1.5rem 4rem }

/* ── 页头 ─────────────────────────────────────────────────────────── */
.masthead { border-bottom: 2px solid var(--ink); padding-bottom: 1.1rem }
.eyebrow {
  font-size: var(--text-xs); letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--ink-3); margin-bottom: 0.5rem;
}
h1 { font-size: var(--text-title); font-weight: 600; letter-spacing: -0.02em }
.stamp { font-size: var(--text-sm); color: var(--ink-3); margin-top: 0.35rem }

/* ── 顶部 KPI 带 ──────────────────────────────────────────────────── */
.kpis {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  border-bottom: 1px solid var(--rule-firm); margin-bottom: var(--space-sec);
}
.kpi { padding: 1.4rem 1.5rem 1.5rem; border-right: 1px solid var(--rule) }
.kpi:first-child { padding-left: 0 }
.kpi:last-child { border-right: 0 }
.kpi-num {
  font-size: var(--text-kpi); font-weight: 300; letter-spacing: -0.03em;
  line-height: 1.05; font-variant-numeric: tabular-nums;
}
.kpi-num .unit { font-size: 0.44em; font-weight: 400; color: var(--ink-3); margin-left: 0.15em }
.kpi-num.is-ok { color: var(--ok) }
.kpi-label {
  font-size: var(--text-xs); letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--ink-3); margin-top: 0.55rem;
}

/* ── 区块 ─────────────────────────────────────────────────────────── */
section { margin-bottom: var(--space-sec) }
.sec-head {
  display: flex; align-items: baseline; gap: 0.8rem;
  border-bottom: 1px solid var(--ink); padding-bottom: 0.55rem; margin-bottom: 0.3rem;
}
.sec-head h2 { font-size: var(--text-lg); font-weight: 600; letter-spacing: -0.01em }
.sec-note { font-size: var(--text-xs); color: var(--ink-3); margin-left: auto }

/* 区块内的耗时小指标条 */
.metrics { display: flex; flex-wrap: wrap; gap: 0 2.2rem; padding: 0.9rem 0 1.6rem }
.metric { font-size: var(--text-sm) }
.metric b {
  display: block; font-size: var(--text-base); font-weight: 500;
  font-variant-numeric: tabular-nums;
}
.metric span { color: var(--ink-3); font-size: var(--text-xs) }

/* ── query 分组 ───────────────────────────────────────────────────── */
.group { padding: 1.1rem 0 1.5rem; border-top: 1px solid var(--rule) }
.group-head { display: flex; align-items: baseline; gap: 0.7rem; margin-bottom: 0.9rem }
.q { font-weight: 600; font-size: var(--text-base) }
.q::before, .q::after { content: '"'; color: var(--ink-3); font-weight: 400 }
.group-meta {
  margin-left: auto; display: flex; gap: 1rem; font-size: var(--text-sm);
  color: var(--ink-2); font-variant-numeric: tabular-nums;
}

.pill {
  font-size: var(--text-xs); padding: 0.1rem 0.45rem; border-radius: 3px;
  border: 1px solid var(--rule-firm); color: var(--ink-2); background: var(--surface);
}
.pill.ok { color: var(--ok); background: var(--ok-soft); border-color: #c8e2d5 }
.pill.warn { color: var(--warn); background: var(--warn-soft); border-color: #eddcc0 }

/* ── 视频卡片 ─────────────────────────────────────────────────────── */
.vgrid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(212px, 1fr)); gap: 1rem;
}
.vcard {
  background: var(--surface); border: 1px solid var(--rule); border-radius: var(--radius);
  overflow: hidden; transition: border-color var(--dur) var(--ease),
                                transform var(--dur) var(--ease);
}
.vcard:hover { border-color: var(--rule-firm); transform: translateY(-2px) }
.vcard video, .vcard img, .vthumb-none {
  width: 100%; aspect-ratio: 16 / 9; display: block; object-fit: cover;
  background: #f0eee9; border-bottom: 1px solid var(--rule);
}
.vthumb-none { display: grid; place-items: center; color: var(--ink-3); font-size: var(--text-sm) }
.vbody { padding: 0.6rem 0.7rem 0.7rem }
/* 截断到 2 行，避免个别超长标题把同一行卡片撑高、破坏栅格节奏 */
.vtitle {
  font-size: var(--text-sm); font-weight: 500; line-height: 1.35;
  margin-bottom: 0.3rem; min-height: 2.7em;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
  overflow: hidden;
}
.vtitle a { color: inherit; text-decoration: none }
.vtitle a:hover { color: var(--accent); text-decoration: underline; text-underline-offset: 2px }
.vsub {
  display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center;
  font-size: var(--text-xs); color: var(--ink-3); font-variant-numeric: tabular-nums;
}
.vsub .score { color: var(--ink-2); font-weight: 500 }

/* ── BGM 列表 ─────────────────────────────────────────────────────── */
.tracks { display: flex; flex-direction: column }
.track {
  display: grid; grid-template-columns: 1.8rem minmax(0, 1fr) auto 3.2rem 230px;
  gap: 0.9rem; align-items: center; padding: 0.5rem 0;
  border-bottom: 1px solid var(--rule); font-size: var(--text-sm);
}
.track:last-child { border-bottom: 0 }
.track .r { color: var(--ink-3); font-variant-numeric: tabular-nums; font-size: var(--text-xs) }
.track .t {
  font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.track .t a { color: inherit; text-decoration: none }
.track .t a:hover { color: var(--accent); text-decoration: underline; text-underline-offset: 2px }
.track .a { color: var(--ink-2); font-size: var(--text-xs) }
.track .d { color: var(--ink-3); text-align: right; font-variant-numeric: tabular-nums;
            font-size: var(--text-xs) }
audio { width: 100%; height: 30px }
.nofile { color: var(--ink-3); font-size: var(--text-xs); font-style: italic }

footer {
  border-top: 1px solid var(--rule-firm); padding-top: 1rem;
  font-size: var(--text-xs); color: var(--ink-3); line-height: 1.7;
}

@media (max-width: 760px) {
  .track { grid-template-columns: 1.6rem minmax(0, 1fr) 3rem; row-gap: 0.35rem }
  .track .a { grid-column: 2 / 4 }
  .track audio, .track .nofile { grid-column: 1 / 4 }
  .kpi:first-child { padding-left: 1.5rem }
}
"""


def _esc(text: object) -> str:
    """HTML 转义，防止标题/频道名里的特殊字符破坏结构。"""
    return html.escape(str(text), quote=True)


def _rel(path: str) -> str:
    """把绝对路径转成相对于 test_results/ 的路径，供 HTML 引用。"""
    try:
        return os.path.relpath(path, RESULTS_DIR)
    except Exception:
        return path


def _kpi(num: str, label: str, unit: str = "", ok: bool = False) -> str:
    unit_html = f'<span class="unit">{_esc(unit)}</span>' if unit else ""
    return (
        f'<div class="kpi"><div class="kpi-num{" is-ok" if ok else ""}">'
        f"{_esc(num)}{unit_html}</div>"
        f'<div class="kpi-label">{_esc(label)}</div></div>'
    )


def _metrics(stats: dict) -> str:
    labels = [
        ("wall_seconds", "并发总耗时"),
        ("avg_seconds", "平均"),
        ("max_seconds", "最慢"),
        ("min_seconds", "最快"),
    ]
    cells = "".join(
        f'<div class="metric"><b>{stats[k]}s</b><span>{label}</span></div>'
        for k, label in labels
        if k in stats
    )
    return f'<div class="metrics">{cells}</div>'


def _overview(bgm: dict, video: dict) -> str:
    v_stats = video.get("stats", {})
    v_dl = [d for c in video.get("cases", []) for d in c.get("downloaded", [])]
    v_ok = sum(1 for d in v_dl if d.get("local_video_path"))
    v_mb = sum(d.get("local_video_bytes", 0) for d in v_dl) / 1024 / 1024

    b_dl = bgm.get("downloaded", [])
    b_ok = sum(1 for d in b_dl if d.get("local_path"))
    flagged = sum(len(c.get("pure_music_flagged", [])) for c in bgm.get("cases", []))

    return (
        '<div class="kpis">'
        + _kpi(f"{v_ok}/{len(v_dl)}", "视频落盘", ok=v_ok == len(v_dl) and v_dl != [])
        + _kpi(f"{v_stats.get('avg_seconds', '—')}", "视频端到端均耗时", unit="s")
        + _kpi(f"{v_mb:.1f}", "视频体积", unit="MB")
        + _kpi(f"{b_ok}/{len(b_dl)}", "BGM 落盘", ok=b_ok == len(b_dl) and b_dl != [])
        + _kpi(f"{flagged}", "纯音乐误报", ok=flagged == 0)
        + "</div>"
    )


def _video_section(data: dict) -> str:
    cases = data.get("cases", [])
    if not cases:
        return ""

    groups = ""
    for case in cases:
        local_map = {
            d.get("source_video", ""): d
            for d in case.get("downloaded", [])
            if d.get("local_video_path")
        }

        cards = ""
        for r in case.get("results", []):
            src = _esc(r.get("source_video", ""))
            title = _esc(r.get("title", "")[:80])
            dl = local_map.get(r.get("source_video", ""))

            if dl:
                mb = dl.get("local_video_bytes", 0) / 1024 / 1024
                media = (
                    f'<video controls preload="metadata" '
                    f'poster="{_esc(r.get("thumbnail", ""))}" '
                    f'src="{_esc(_rel(dl["local_video_path"]))}"></video>'
                )
                state = f'<span class="pill ok">{mb:.1f} MB</span>'
            else:
                thumb = r.get("thumbnail", "")
                inner = (
                    f'<img src="{_esc(thumb)}" alt="" loading="lazy">'
                    if thumb
                    else '<div class="vthumb-none">无预览</div>'
                )
                media = f'<a href="{src}" target="_blank" rel="noopener">{inner}</a>'
                state = '<span class="pill">未落盘</span>'

            cards += f"""
              <article class="vcard">
                {media}
                <div class="vbody">
                  <div class="vtitle"><a href="{src}" target="_blank" rel="noopener">{title}</a></div>
                  <div class="vsub">
                    <span class="score">{r.get('score', 0):.3f}</span>
                    <span>{_esc(r.get('channel', '')[:18])}</span>
                    <span>{r.get('duration', 0):.0f}s</span>
                    {state}
                  </div>
                </div>
              </article>"""

        groups += f"""
          <div class="group">
            <div class="group-head">
              <span class="q">{_esc(case.get('query', ''))}</span>
              <span class="pill">{_esc(", ".join(case.get('sources', [])))}</span>
              <div class="group-meta">
                <span>{case.get('returned', 0)} 条</span>
                <span>{case.get('elapsed_seconds', '')}s</span>
              </div>
            </div>
            <div class="vgrid">{cards}</div>
          </div>"""

    return f"""
    <section>
      <div class="sec-head">
        <h2>视频检索</h2>
        <span class="sec-note">耗时含搜索 + yt-dlp 抽流 + FFmpeg 合并落盘全程</span>
      </div>
      {_metrics(data.get("stats", {}))}
      {groups}
    </section>"""


def _bgm_section(data: dict) -> str:
    cases = data.get("cases", [])
    if not cases:
        return ""

    local = {
        r.get("download_url", ""): r.get("local_path", "")
        for r in data.get("downloaded", [])
        if r.get("local_path")
    }

    groups = ""
    for case in cases:
        flagged = case.get("pure_music_flagged", [])
        badge = (
            '<span class="pill ok">纯音乐 ✓</span>'
            if not flagged
            else f'<span class="pill warn">疑似含人声: {_esc(", ".join(flagged))}</span>'
        )

        rows = ""
        for r in case.get("results", []):
            path = local.get(r.get("download_url", ""), "")
            player = (
                f'<audio controls preload="none" src="{_esc(_rel(path))}"></audio>'
                if path
                else '<span class="nofile">未下载</span>'
            )
            rows += f"""
              <div class="track">
                <span class="r">{r.get('rank', '')}</span>
                <span class="t"><a href="{_esc(r.get('url', ''))}" target="_blank"
                   rel="noopener">{_esc(r.get('title', '')[:70])}</a></span>
                <span class="a">{_esc(r.get('artist', '')[:24])}</span>
                <span class="d">{r.get('duration_sec', 0):.0f}s</span>
                {player}
              </div>"""

        groups += f"""
          <div class="group">
            <div class="group-head">
              <span class="q">{_esc(case.get('query', ''))}</span>
              {badge}
              <div class="group-meta">
                <span>{case.get('returned', 0)} 条</span>
                <span>{case.get('elapsed_seconds', '')}s</span>
              </div>
            </div>
            <div class="tracks">{rows}</div>
          </div>"""

    return f"""
    <section>
      <div class="sec-head">
        <h2>BGM 检索</h2>
        <span class="sec-note">Internet Archive netlabels · CC 授权 · 双层纯音乐过滤</span>
      </div>
      {_metrics(data.get("stats", {}))}
      {groups}
    </section>"""


def generate() -> None:
    bgm = json.loads(BGM_JSON.read_text(encoding="utf-8")) if BGM_JSON.exists() else {}
    video = json.loads(VIDEO_JSON.read_text(encoding="utf-8")) if VIDEO_JSON.exists() else {}

    html_out = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>asset_retrieval_tool 测试报告</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <header class="masthead">
    <div class="eyebrow">AICutPipeline · 素材检索</div>
    <h1>asset_retrieval_tool 集成测试报告</h1>
    <div class="stamp">生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')} · 高并发实测</div>
  </header>
  {_overview(bgm, video)}
  <main>
    {_video_section(video)}
    {_bgm_section(bgm)}
  </main>
  <footer>
    视频经 yt-dlp 抽取 DASH 流、FFmpeg 合并为 mp4，已限制画质与截取时长（短素材场景）。<br>
    媒体文件位于 <code>downloaded/</code>，本页直接内嵌播放。
  </footer>
</div>
</body>
</html>"""

    OUT_HTML.write_text(html_out, encoding="utf-8")
    print(f"报告已生成: {OUT_HTML}")


if __name__ == "__main__":
    generate()
