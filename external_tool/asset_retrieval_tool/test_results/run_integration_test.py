"""
端到端集成测试：高并发 BGM + 在线视频检索，统计每条耗时
结果保存到 test_results/，媒体文件下载到 test_results/downloaded/
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from asset_retrieval_tool import search_bgm, search_video_segments
from asset_retrieval_tool import download_bgm_results, download_video_results

RESULTS_DIR = Path(__file__).parent.parent / "test_results"
RESULTS_DIR.mkdir(exist_ok=True)
DOWNLOAD_DIR = RESULTS_DIR / "downloaded"

YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")
BILIBILI_COOKIE = os.environ.get("BILIBILI_COOKIE", "")

# Load from config.json if not set via env
_CONFIG_FILE = Path(__file__).parent.parent / "config.json"
if _CONFIG_FILE.exists():
    try:
        _cfg = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
        if not YOUTUBE_API_KEY:
            YOUTUBE_API_KEY = _cfg.get("youtube_api_key", "")
        if not BILIBILI_COOKIE:
            BILIBILI_COOKIE = _cfg.get("bilibili_cookie", "")
    except Exception:
        pass

BGM_CASES = [
    ("upbeat background music",       5, 60.0),
    ("ambient chill electronic",      3, 90.0),
    ("cinematic orchestral dramatic", 5, 60.0),
    ("jazz lofi beats",               3, 30.0),
]

VIDEO_CASES = [
    ("晨光校园学生",  3),
    ("自然风光山水",  5),
    ("城市夜景",      2),
    ("学生学习专注",  3),
]

# 短素材场景：只截前 N 秒、限制画质，避免整片下载
VIDEO_CLIP_SECONDS = 15.0
VIDEO_MAX_HEIGHT = 480


def _run_bgm(query: str, top_k: int, min_duration: float) -> dict:
    t0 = time.perf_counter()
    results = search_bgm(query, top_k=top_k, min_duration=min_duration, max_duration=600.0)
    elapsed = round(time.perf_counter() - t0, 2)
    from asset_retrieval_tool.bgm_search import _SPOKEN_WORD_FILENAME_TOKENS
    flagged = [
        r["title"] for r in results
        if any(kw in r["title"].lower() or kw in r["download_url"].lower()
               for kw in _SPOKEN_WORD_FILENAME_TOKENS)
    ]
    return {"query": query, "top_k": top_k, "min_duration": min_duration,
            "elapsed_seconds": elapsed, "returned": len(results),
            "pure_music_flagged": flagged, "results": results}


def _run_video(query: str, top_k: int) -> dict:
    sources = ["youtube", "bilibili"] if YOUTUBE_API_KEY else ["bilibili"]
    print(f"\n[视频] query='{query}' top_k={top_k} sources={sources}")
    t0 = time.perf_counter()
    results = search_video_segments(
        query, top_k=top_k, min_duration=30.0, max_duration=1800.0,
        sources=sources, youtube_api_key=YOUTUBE_API_KEY or None,
        bilibili_cookie=BILIBILI_COOKIE or None,
    )
    downloaded = download_video_results(
        results, DOWNLOAD_DIR,
        max_workers=3, cookie=BILIBILI_COOKIE or None,
        max_height=VIDEO_MAX_HEIGHT, clip_seconds=VIDEO_CLIP_SECONDS,
    )
    elapsed = round(time.perf_counter() - t0, 2)
    return {"query": query, "top_k": top_k, "sources": sources,
            "elapsed_seconds": elapsed, "returned": len(results),
            "results": results, "downloaded": downloaded}


def main() -> None:
    print("=" * 60)
    print("asset_retrieval_tool 高并发集成测试")
    print(f"YouTube: {'已配置' if YOUTUBE_API_KEY else '未配置（仅 Bilibili）'}")
    print("=" * 60)

    # ── 并发跑所有 BGM query ──────────────────────────────────────────────────
    print(f"\n[BGM] 并发执行 {len(BGM_CASES)} 个 query ...")
    wall_t0 = time.perf_counter()
    bgm_all: list[dict] = [{}] * len(BGM_CASES)
    with ThreadPoolExecutor(max_workers=len(BGM_CASES)) as pool:
        futures = {pool.submit(_run_bgm, q, k, d): i for i, (q, k, d) in enumerate(BGM_CASES)}
        for fut in as_completed(futures):
            idx = futures[fut]
            r = fut.result()
            bgm_all[idx] = r
            print(f"  ✓ '{r['query']}' → {r['returned']} 条  {r['elapsed_seconds']}s")
    bgm_wall = round(time.perf_counter() - wall_t0, 2)
    bgm_elapsed = [r["elapsed_seconds"] for r in bgm_all]
    bgm_stats = {
        "wall_seconds": bgm_wall,
        "avg_seconds": round(sum(bgm_elapsed) / len(bgm_elapsed), 2),
        "max_seconds": max(bgm_elapsed),
        "min_seconds": min(bgm_elapsed),
    }
    print(f"  并发总耗时: {bgm_wall}s  平均: {bgm_stats['avg_seconds']}s")

    # ── 下载 BGM ──────────────────────────────────────────────────────────────
    all_bgm_results = [r for case in bgm_all for r in case["results"]]
    print(f"\n[下载] BGM {len(all_bgm_results)} 条 → {DOWNLOAD_DIR}/bgm/ ...")
    downloaded_bgm = download_bgm_results(all_bgm_results, DOWNLOAD_DIR, max_workers=6)
    ok = sum(1 for r in downloaded_bgm if "local_path" in r)
    fail = sum(1 for r in downloaded_bgm if "download_error" in r)
    print(f"  ✅ 成功: {ok}  ❌ 失败: {fail}")
    for r in downloaded_bgm:
        tag = "→" if "local_path" in r else "✗"
        val = r.get("local_path") or r.get("download_error", "")
        print(f"  {tag} {val}")

    # ── 并发跑所有视频 query（搜索 + 落盘计入同一计时）────────────────────────
    print(f"\n[视频] 并发执行 {len(VIDEO_CASES)} 个 query（含下载落盘）...")
    wall_t0 = time.perf_counter()
    video_all: list[dict] = [{}] * len(VIDEO_CASES)
    with ThreadPoolExecutor(max_workers=len(VIDEO_CASES)) as pool:
        futures = {pool.submit(_run_video, q, k): i for i, (q, k) in enumerate(VIDEO_CASES)}
        for fut in as_completed(futures):
            idx = futures[fut]
            r = fut.result()
            video_all[idx] = r
            v_ok = sum(1 for d in r["downloaded"] if "local_video_path" in d)
            v_fail = sum(1 for d in r["downloaded"] if "download_error" in d)
            print(f"  ✓ '{r['query']}' → {r['returned']} 条  "
                  f"下载 {v_ok}✅{v_fail}❌  {r['elapsed_seconds']}s（搜索+落盘）")
    video_wall = round(time.perf_counter() - wall_t0, 2)
    video_elapsed = [r["elapsed_seconds"] for r in video_all]
    video_stats = {
        "wall_seconds": video_wall,
        "avg_seconds": round(sum(video_elapsed) / len(video_elapsed), 2),
        "max_seconds": max(video_elapsed),
        "min_seconds": min(video_elapsed),
        "note": "耗时含搜索 + yt-dlp 落盘全程",
    }
    print(f"  并发总耗时: {video_wall}s  平均: {video_stats['avg_seconds']}s（搜索+落盘）")

    # ── 汇总下载结果 ──────────────────────────────────────────────────────────
    downloaded_video = [d for case in video_all for d in case["downloaded"]]
    total_ok = sum(1 for d in downloaded_video if "local_video_path" in d)
    total_fail = sum(1 for d in downloaded_video if "download_error" in d)
    total_bytes = sum(d.get("local_video_bytes", 0) for d in downloaded_video)
    print(f"\n[视频落盘汇总] ✅ {total_ok}  ❌ {total_fail}  "
          f"总大小: {total_bytes / 1024 / 1024:.1f} MB")
    for d in downloaded_video:
        if "local_video_path" in d:
            mb = d.get("local_video_bytes", 0) / 1024 / 1024
            print(f"  → {Path(d['local_video_path']).name}  {mb:.1f} MB")
        else:
            print(f"  ✗ {d.get('title', '')[:30]}: {d.get('download_error', '')[:80]}")
    video_stats["download_ok"] = total_ok
    video_stats["download_failed"] = total_fail
    video_stats["download_total_mb"] = round(total_bytes / 1024 / 1024, 2)

    # ── 保存 JSON ─────────────────────────────────────────────────────────────
    bgm_out = {
        "stats": bgm_stats,
        "cases": bgm_all,
        "downloaded": downloaded_bgm,
    }
    video_out = {
        "stats": video_stats,
        "cases": video_all,
        "downloaded": downloaded_video,
    }
    (RESULTS_DIR / "bgm_results.json").write_text(
        json.dumps(bgm_out, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS_DIR / "video_results.json").write_text(
        json.dumps(video_out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("✅ 测试完成")
    print(f"  {RESULTS_DIR}/bgm_results.json")
    print(f"  {RESULTS_DIR}/video_results.json")
    print(f"  {RESULTS_DIR}/downloaded/bgm/")
    print(f"  {RESULTS_DIR}/downloaded/video/")
    print("=" * 60)


if __name__ == "__main__":
    main()

