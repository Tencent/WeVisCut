"""下载检索到的素材（BGM 音频 + 视频片段）到本地目录。"""
from __future__ import annotations

import logging
import shutil
import tempfile
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TIMEOUT = 30
_CHUNK = 1024 * 256  # 256 KB


def _safe_filename(name: str, max_len: int = 120) -> str:
    """把任意字符串变成安全文件名。"""
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    cleaned = "".join(c if c in keep else "_" for c in name)
    cleaned = cleaned.strip("._")
    return cleaned[:max_len] or "track"


def _download_file(url: str, dest: Path) -> Path:
    """下载单个文件到 dest，返回落盘路径。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "asset_retrieval_tool/1.0"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp, dest.open("wb") as f:
        shutil.copyfileobj(resp, f, _CHUNK)
    return dest


def download_bgm_results(
    results: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    max_workers: int = 4,
    skip_existing: bool = True,
) -> list[dict[str, Any]]:
    """将 search_bgm 返回的结果下载到 output_dir/bgm/。

    Args:
        results: search_bgm 的返回值。
        output_dir: 下载根目录。
        max_workers: 并发下载线程数。
        skip_existing: 文件已存在时跳过。

    Returns:
        每条结果追加 local_path（成功）或 download_error（失败）字段。
    """
    out = Path(output_dir) / "bgm"
    out.mkdir(parents=True, exist_ok=True)

    def _dl(item: dict[str, Any]) -> dict[str, Any]:
        item = dict(item)
        url = item.get("download_url", "")
        if not url:
            item["download_error"] = "no download_url"
            return item
        ext = Path(urllib.parse.urlparse(url).path).suffix or ".mp3"
        fname = _safe_filename(f"{item.get('artist', 'unknown')}_{item.get('title', 'track')}") + ext
        dest = out / fname
        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            item["local_path"] = str(dest)
            return item
        try:
            _download_file(url, dest)
            item["local_path"] = str(dest)
            logger.info("BGM 下载完成: %s", dest.name)
        except Exception as exc:
            item["download_error"] = str(exc)
            logger.warning("BGM 下载失败 %s: %s", url, exc)
        return item

    updated: list[dict[str, Any]] = [{}] * len(results)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_dl, item): i for i, item in enumerate(results)}
        for fut in as_completed(futures):
            idx = futures[fut]
            updated[idx] = fut.result()
    return updated


_DIRECT_MEDIA_EXT = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".flv", ".avi"}

_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_COOKIE_EXPIRY = "1900000000"


def _ytdlp_available() -> bool:
    """yt-dlp 是否可用（可选依赖，缺失时降级为直链下载）。"""
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        return False
    return True


def _write_netscape_cookiefile(
    cookie_header: str, dest: Path, domain: str = ".bilibili.com"
) -> Path:
    """把 "k=v; k2=v2" 形式的 Cookie 头转成 Netscape cookiefile。

    yt-dlp 必须走 cookiefile 而非 raw header：直接塞 Cookie 头会绕过它内部的
    cookiejar，导致 B 站风控返回 412 Precondition Failed。
    """
    lines = ["# Netscape HTTP Cookie File", ""]
    for part in cookie_header.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, value = part.split("=", 1)
        lines.append(
            "\t".join(
                [domain, "TRUE", "/", "FALSE", _COOKIE_EXPIRY, name.strip(), value.strip()]
            )
        )
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def _ytdlp_download(
    url: str,
    out_dir: Path,
    stem: str,
    *,
    cookiefile: Path | None = None,
    max_height: int = 720,
    clip_seconds: float | None = None,
) -> Path:
    """用 yt-dlp 抽取平台页面里的真实媒体流并下载合并，返回落盘路径。

    支持 YouTube / Bilibili 等 1800+ 站点。Bilibili 的 DASH 音视频分离流
    由 yt-dlp 调用 FFmpeg 自动合并成单个 mp4。

    Args:
        url: 平台视频页面 URL。
        out_dir: 输出目录。
        stem: 输出文件名（不含扩展名）。
        cookiefile: Netscape 格式 cookie 文件路径，登录态内容需要。
        max_height: 画质上限，避免拉取超大文件。
        clip_seconds: 仅下载前 N 秒（短素材场景）。None 表示整段。

    Returns:
        实际落盘的文件路径。
    """
    import yt_dlp
    from yt_dlp.utils import download_range_func

    opts: dict[str, Any] = {
        "format": (
            f"bv*[height<=?{max_height}]+ba/"
            f"b[height<=?{max_height}]/bv*+ba/b"
        ),
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / f"{stem}.%(ext)s"),
        "http_headers": {"User-Agent": _BROWSER_UA, "Referer": "https://www.bilibili.com/"},
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "retries": 3,
        "socket_timeout": 30,
        "concurrent_fragment_downloads": 4,
        "ignoreerrors": False,
    }
    if cookiefile is not None:
        opts["cookiefile"] = str(cookiefile)
    if clip_seconds and clip_seconds > 0:
        opts["download_ranges"] = download_range_func(None, [(0, clip_seconds)])
        opts["force_keyframes_at_cuts"] = True

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    produced = sorted(
        (p for p in out_dir.glob(f"{stem}.*") if p.is_file() and p.stat().st_size > 0),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    if not produced:
        raise RuntimeError("yt-dlp 未产出任何文件")
    return produced[0]


def download_video_results(
    results: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    max_workers: int = 3,
    skip_existing: bool = True,
    cookie: str | None = None,
    cookie_domain: str = ".bilibili.com",
    max_height: int = 720,
    clip_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """将 search_video_segments 的结果下载为真实视频文件到 output_dir/video/。

    三条路径按顺序尝试：
      1. 本地路径 → 直接复制；
      2. 直链媒体 URL（.mp4/.webm 等）→ 流式下载；
      3. 平台页面 URL（B站/YouTube 等）→ 交给 yt-dlp 抽流 + FFmpeg 合并。

    Args:
        results: search_video_segments 的返回值。
        output_dir: 下载根目录。
        max_workers: 并发线程数（过高易触发平台限流）。
        skip_existing: 同名文件已存在时跳过。
        cookie: 可选 Cookie 头字符串，B 站登录态内容需要。
        cookie_domain: cookie 归属域名。
        max_height: 画质上限（像素高度）。
        clip_seconds: 仅截取前 N 秒，适合短素材场景。

    Returns:
        每条结果追加 local_video_path + local_video_bytes（成功）
        或 download_error（失败）字段。
    """
    out = Path(output_dir) / "video"
    out.mkdir(parents=True, exist_ok=True)
    has_ytdlp = _ytdlp_available()

    cookiefile: Path | None = None
    tmpdir: tempfile.TemporaryDirectory[str] | None = None
    if cookie:
        tmpdir = tempfile.TemporaryDirectory(prefix="art_cookies_")
        cookiefile = _write_netscape_cookiefile(
            cookie, Path(tmpdir.name) / "cookies.txt", cookie_domain
        )

    def _dl(item: dict[str, Any]) -> dict[str, Any]:
        item = dict(item)
        src = item.get("source_video", "")
        if not src:
            item["download_error"] = "no source_video"
            return item

        stem = _safe_filename(
            f"{item.get('platform', 'video')}_{item.get('video_id', '')}"
            f"_{item.get('title', '')}"
        )
        existing = [
            p for p in out.glob(f"{stem}.*") if p.is_file() and p.stat().st_size > 0
        ]
        if skip_existing and existing:
            item["local_video_path"] = str(existing[0])
            item["local_video_bytes"] = existing[0].stat().st_size
            return item

        try:
            src_path = Path(src)
            is_url = src.startswith("http://") or src.startswith("https://")
            suffix = Path(urllib.parse.urlparse(src).path).suffix.lower()

            if not is_url and src_path.exists():
                dest = out / (stem + (src_path.suffix or ".mp4"))
                shutil.copy2(src_path, dest)
            elif is_url and suffix in _DIRECT_MEDIA_EXT:
                dest = out / (stem + suffix)
                _download_file(src, dest)
            elif is_url and has_ytdlp:
                dest = _ytdlp_download(
                    src, out, stem,
                    cookiefile=cookiefile,
                    max_height=max_height,
                    clip_seconds=clip_seconds,
                )
            elif is_url:
                item["download_error"] = (
                    "平台页面 URL 需要 yt-dlp 抽流，请先 pip install yt-dlp"
                )
                return item
            else:
                item["download_error"] = f"source not found: {src}"
                return item

            item["local_video_path"] = str(dest)
            item["local_video_bytes"] = dest.stat().st_size
            logger.info("视频下载完成: %s (%d bytes)", dest.name, dest.stat().st_size)
        except Exception as exc:
            item["download_error"] = str(exc)
            logger.warning("视频下载失败 %s: %s", src, exc)
        return item

    try:
        updated: list[dict[str, Any]] = [{}] * len(results)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_dl, item): i for i, item in enumerate(results)}
            for fut in as_completed(futures):
                idx = futures[fut]
                updated[idx] = fut.result()
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()
    return updated
