"""Tests for asset_retrieval_tool — online video search and BGM search."""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
import pytest


# ── semantic.py ───────────────────────────────────────────────────────────────

class TestNormalizeText:
    def test_lowercases(self):
        from asset_retrieval_tool.semantic import normalize_text
        assert normalize_text("Hello World") == "hello world"

    def test_strips_markdown(self):
        from asset_retrieval_tool.semantic import normalize_text
        assert normalize_text("**bold** _italic_") == "bold italic"

    def test_empty(self):
        from asset_retrieval_tool.semantic import normalize_text
        assert normalize_text("") == ""


class TestTokenize:
    def test_basic(self):
        from asset_retrieval_tool.semantic import tokenize
        tokens = tokenize("晨光下校园航拍")
        assert "晨" in tokens or len(tokens) > 0

    def test_stop_words_removed(self):
        from asset_retrieval_tool.semantic import tokenize
        tokens = tokenize("的了和是在")
        assert tokens == set()

    def test_english(self):
        from asset_retrieval_tool.semantic import tokenize
        tokens = tokenize("sunny beach waves")
        assert "sunny" in tokens
        assert "the" not in tokens


class TestSimilarityScore:
    def test_exact_match(self):
        from asset_retrieval_tool.semantic import similarity_score
        score = similarity_score("晨光校园", "晨光下校园航拍学生步行", 8.0)
        assert score > 0.3

    def test_no_overlap(self):
        from asset_retrieval_tool.semantic import similarity_score
        score = similarity_score("海浪沙滩", "城市夜景霓虹灯", 8.0)
        assert score == 0.0

    def test_duration_bonus_applied(self):
        from asset_retrieval_tool.semantic import similarity_score
        s_good = similarity_score("校园", "校园晨光", 8.0)
        s_bad = similarity_score("校园", "校园晨光", 60.0)
        assert s_good >= s_bad

    def test_contains_bonus(self):
        from asset_retrieval_tool.semantic import similarity_score
        s = similarity_score("校园晨光", "校园晨光航拍", 8.0)
        assert s > 0.5


class TestExtractSegmentsFromText:
    def test_parses_time_ranges(self):
        from asset_retrieval_tool.semantic import extract_segments_from_text
        text = "t=0s-t=5s 晨光校园\nt=5s-t=12s 学生步行"
        segs = extract_segments_from_text(text)
        assert len(segs) == 2
        assert segs[0]["start"] == 0.0
        assert segs[0]["end"] == 5.0

    def test_empty_text(self):
        from asset_retrieval_tool.semantic import extract_segments_from_text
        assert extract_segments_from_text("") == []

    def test_deduplication(self):
        from asset_retrieval_tool.semantic import extract_segments_from_text
        text = "t=0s-t=5s 晨光\nt=0s-t=5s 晨光"
        segs = extract_segments_from_text(text)
        assert len(segs) == 1


class TestEnsureSemanticIndex:
    def test_writes_back_when_missing(self, tmp_path):
        from asset_retrieval_tool.semantic import ensure_semantic_index
        fp = tmp_path / "v_analysis.json"
        payload = {
            "source_video": "v.mp4",
            "analysis_text": "t=0s-t=5s 晨光校园",
            "semantic_segments": [],
        }
        fp.write_text(json.dumps(payload), encoding="utf-8")
        segs = ensure_semantic_index(fp, payload)
        assert len(segs) > 0
        saved = json.loads(fp.read_text(encoding="utf-8"))
        assert len(saved["semantic_segments"]) > 0

    def test_no_write_when_already_present(self, tmp_path):
        from asset_retrieval_tool.semantic import ensure_semantic_index
        segs = [{"start": 0.0, "end": 5.0, "duration": 5.0, "semantic_text": "晨光校园"}]
        fp = tmp_path / "v_analysis.json"
        payload = {"source_video": "v.mp4", "analysis_text": "", "semantic_segments": segs}
        fp.write_text(json.dumps(payload), encoding="utf-8")
        mtime_before = fp.stat().st_mtime
        time.sleep(0.01)
        ensure_semantic_index(fp, payload)
        assert fp.stat().st_mtime == mtime_before


class TestIterAnalysisFiles:
    def test_finds_files(self, tmp_path):
        from asset_retrieval_tool.semantic import iter_analysis_files
        (tmp_path / "a_analysis.json").write_text("{}", encoding="utf-8")
        (tmp_path / "b_analysis.json").write_text("{}", encoding="utf-8")
        (tmp_path / "c.json").write_text("{}", encoding="utf-8")
        files = iter_analysis_files([tmp_path])
        names = {f.name for f in files}
        assert "a_analysis.json" in names
        assert "b_analysis.json" in names
        assert "c.json" not in names

    def test_missing_dir(self):
        from asset_retrieval_tool.semantic import iter_analysis_files
        files = iter_analysis_files([Path("/nonexistent/path")])
        assert files == []


# ── video_retrieval.py ────────────────────────────────────────────────────────

class TestParseIso8601Duration:
    def test_minutes_seconds(self):
        from asset_retrieval_tool.video_retrieval import _parse_iso8601_duration
        assert _parse_iso8601_duration("PT3M45S") == 225.0

    def test_hours(self):
        from asset_retrieval_tool.video_retrieval import _parse_iso8601_duration
        assert _parse_iso8601_duration("PT1H2M3S") == 3723.0

    def test_seconds_only(self):
        from asset_retrieval_tool.video_retrieval import _parse_iso8601_duration
        assert _parse_iso8601_duration("PT30S") == 30.0

    def test_empty(self):
        from asset_retrieval_tool.video_retrieval import _parse_iso8601_duration
        assert _parse_iso8601_duration("") == 0.0


class TestParseBiliDuration:
    def test_numeric(self):
        from asset_retrieval_tool.video_retrieval import _parse_bili_duration
        assert _parse_bili_duration(120) == 120.0

    def test_mm_ss(self):
        from asset_retrieval_tool.video_retrieval import _parse_bili_duration
        assert _parse_bili_duration("3:45") == 225.0

    def test_string_seconds(self):
        from asset_retrieval_tool.video_retrieval import _parse_bili_duration
        assert _parse_bili_duration("90.5") == 90.5


class TestSearchVideoSegments:
    def _fake_bili(self, query, limit, cookie=""):
        return [
            {"platform": "bilibili", "video_id": "BV1", "source_video": "https://www.bilibili.com/video/BV1",
             "title": "晨光校园航拍学生步行", "description": "校园早晨", "channel": "UP主A",
             "thumbnail": "", "start": 0.0, "end": 60.0, "duration": 60.0,
             "semantic_text": "晨光校园航拍学生步行校园早晨"},
            {"platform": "bilibili", "video_id": "BV2", "source_video": "https://www.bilibili.com/video/BV2",
             "title": "城市夜景霓虹灯", "description": "夜晚城市", "channel": "UP主B",
             "thumbnail": "", "start": 0.0, "end": 120.0, "duration": 120.0,
             "semantic_text": "城市夜景霓虹灯夜晚城市"},
        ]

    def test_basic_retrieval(self):
        from asset_retrieval_tool import video_retrieval
        with patch.object(video_retrieval, "_bili_search", self._fake_bili):
            results = video_retrieval.search_video_segments("晨光校园", sources=["bilibili"], top_k=5)
        assert len(results) >= 1
        assert results[0]["rank"] == 1
        assert results[0]["score"] > 0
        assert "source_video" in results[0]

    def test_no_match_returns_empty(self):
        from asset_retrieval_tool import video_retrieval
        with patch.object(video_retrieval, "_bili_search", self._fake_bili):
            results = video_retrieval.search_video_segments("海底珊瑚礁深海", sources=["bilibili"], top_k=5)
        assert results == []

    def test_duration_filter(self):
        from asset_retrieval_tool import video_retrieval
        with patch.object(video_retrieval, "_bili_search", self._fake_bili):
            results = video_retrieval.search_video_segments(
                "晨光校园", sources=["bilibili"], min_duration=90.0, max_duration=600.0
            )
        assert all(r["duration"] >= 90.0 for r in results)

    def test_top_k_respected(self):
        from asset_retrieval_tool import video_retrieval

        def _many(q, l):
            return [
                {"platform": "bilibili", "video_id": f"BV{i}",
                 "source_video": f"https://www.bilibili.com/video/BV{i}",
                 "title": f"晨光校园场景{i}", "description": "", "channel": "",
                 "thumbnail": "", "start": 0.0, "end": 60.0, "duration": 60.0,
                 "semantic_text": f"晨光校园场景{i}"}
                for i in range(20)
            ]

        with patch.object(video_retrieval, "_bili_search", _many):
            results = video_retrieval.search_video_segments("晨光校园", sources=["bilibili"], top_k=3)
        assert len(results) <= 3

    def test_ranks_sequential(self):
        from asset_retrieval_tool import video_retrieval
        with patch.object(video_retrieval, "_bili_search", self._fake_bili):
            results = video_retrieval.search_video_segments("晨光校园", sources=["bilibili"], top_k=5)
        for i, r in enumerate(results, start=1):
            assert r["rank"] == i

    def test_scores_descending(self):
        from asset_retrieval_tool import video_retrieval
        with patch.object(video_retrieval, "_bili_search", self._fake_bili):
            results = video_retrieval.search_video_segments("晨光校园", sources=["bilibili"], top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_deduplication(self):
        from asset_retrieval_tool import video_retrieval

        def _dup(q, l, cookie=""):
            item = {"platform": "bilibili", "video_id": "BV1",
                    "source_video": "https://www.bilibili.com/video/BV1",
                    "title": "晨光校园", "description": "", "channel": "",
                    "thumbnail": "", "start": 0.0, "end": 60.0, "duration": 60.0,
                    "semantic_text": "晨光校园"}
            return [item, item]

        with patch.object(video_retrieval, "_bili_search", _dup):
            results = video_retrieval.search_video_segments("晨光校园", sources=["bilibili"])
        assert len(results) == 1

    def test_source_failure_graceful(self):
        from asset_retrieval_tool import video_retrieval

        def _raise(q, l):
            raise RuntimeError("network error")

        with patch.object(video_retrieval, "_bili_search", _raise):
            results = video_retrieval.search_video_segments("晨光校园", sources=["bilibili"])
        assert isinstance(results, list)

    def test_youtube_skipped_without_key(self):
        from asset_retrieval_tool import video_retrieval
        with patch.object(video_retrieval, "_bili_search", self._fake_bili):
            results = video_retrieval.search_video_segments(
                "晨光校园", sources=["youtube", "bilibili"], youtube_api_key=None
            )
        assert all(r["platform"] == "bilibili" for r in results)

    def test_concurrent_calls(self):
        from asset_retrieval_tool import video_retrieval
        errors: list[Exception] = []
        results_list: list[list] = []
        lock = threading.Lock()

        def _call():
            try:
                with patch.object(video_retrieval, "_bili_search", self._fake_bili):
                    r = video_retrieval.search_video_segments("晨光校园", sources=["bilibili"], top_k=5)
                with lock:
                    results_list.append(r)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = [pool.submit(_call) for _ in range(16)]
            for f in futs:
                f.result()

        assert errors == []
        assert all(isinstance(r, list) for r in results_list)

    def test_required_fields_present(self):
        from asset_retrieval_tool import video_retrieval
        with patch.object(video_retrieval, "_bili_search", self._fake_bili):
            results = video_retrieval.search_video_segments("晨光校园", sources=["bilibili"])
        required = {"rank", "score", "platform", "video_id", "source_video",
                    "title", "description", "channel", "thumbnail",
                    "start", "end", "duration", "semantic_text"}
        for r in results:
            assert required.issubset(r.keys())


# ── bgm_search.py ─────────────────────────────────────────────────────────────

class TestParseDuration:
    def test_numeric(self):
        from asset_retrieval_tool.bgm_search import _parse_duration
        assert _parse_duration(120) == 120.0
        assert _parse_duration("90.5") == 90.5

    def test_mm_ss(self):
        from asset_retrieval_tool.bgm_search import _parse_duration
        assert _parse_duration("3:45") == 225.0

    def test_hh_mm_ss(self):
        from asset_retrieval_tool.bgm_search import _parse_duration
        assert _parse_duration("1:02:03") == 3723.0

    def test_none(self):
        from asset_retrieval_tool.bgm_search import _parse_duration
        assert _parse_duration(None) == 0.0


class TestSearchBgm:
    def _fake_ia(self, query, limit):
        return [
            {"source": "internet_archive", "id": "item1/track1.mp3", "title": "Upbeat Morning",
             "artist": "Artist A", "duration_sec": 180.0, "genre": "netlabel / CC",
             "url": "https://archive.org/details/item1",
             "download_url": "https://archive.org/download/item1/track1.mp3",
             "license": "CC"},
            {"source": "internet_archive", "id": "item2/track2.mp3", "title": "Chill Vibes",
             "artist": "Artist B", "duration_sec": 240.0, "genre": "netlabel / CC",
             "url": "https://archive.org/details/item2",
             "download_url": "https://archive.org/download/item2/track2.mp3",
             "license": "CC"},
        ]

    def test_returns_ranked_results(self):
        from asset_retrieval_tool import bgm_search
        with patch.object(bgm_search, "_ia_search", self._fake_ia):
            results = bgm_search.search_bgm("upbeat corporate", top_k=5)
        assert len(results) == 2
        assert results[0]["rank"] == 1
        assert results[1]["rank"] == 2

    def test_duration_filter(self):
        from asset_retrieval_tool import bgm_search
        with patch.object(bgm_search, "_ia_search", self._fake_ia):
            results = bgm_search.search_bgm("upbeat", min_duration=200.0, max_duration=600.0)
        assert all(r["duration_sec"] >= 200.0 for r in results)

    def test_source_selection(self):
        from asset_retrieval_tool import bgm_search
        with patch.object(bgm_search, "_ia_search", self._fake_ia):
            results = bgm_search.search_bgm("upbeat", sources=["internet_archive"])
        assert all(r["source"] == "internet_archive" for r in results)

    def test_deduplication(self):
        from asset_retrieval_tool import bgm_search
        dup = {"source": "internet_archive", "id": "x/t.mp3", "title": "T", "artist": "A",
               "duration_sec": 120.0, "genre": "", "url": "u",
               "download_url": "https://archive.org/download/x/t.mp3", "license": "CC"}

        def _dup_ia(q, l):
            return [dup, dup]

        with patch.object(bgm_search, "_ia_search", _dup_ia):
            results = bgm_search.search_bgm("test")
        assert len(results) == 1

    def test_top_k_respected(self):
        from asset_retrieval_tool import bgm_search

        def _many(q, l):
            return [
                {"source": "internet_archive", "id": f"item/{i}.mp3", "title": f"T{i}",
                 "artist": "A", "duration_sec": 120.0, "genre": "",
                 "url": f"https://archive.org/details/item{i}",
                 "download_url": f"https://archive.org/download/item/{i}.mp3", "license": "CC"}
                for i in range(20)
            ]

        with patch.object(bgm_search, "_ia_search", _many):
            results = bgm_search.search_bgm("test", top_k=3)
        assert len(results) == 3

    def test_concurrent_calls(self):
        """Multiple threads calling search_bgm simultaneously."""
        from asset_retrieval_tool import bgm_search
        with patch.object(bgm_search, "_ia_search", self._fake_ia):
            errors: list[Exception] = []
            results_list: list[list] = []
            lock = threading.Lock()

            def _call():
                try:
                    r = bgm_search.search_bgm("upbeat", top_k=5)
                    with lock:
                        results_list.append(r)
                except Exception as exc:
                    with lock:
                        errors.append(exc)

            with ThreadPoolExecutor(max_workers=8) as pool:
                futs = [pool.submit(_call) for _ in range(16)]
                for f in futs:
                    f.result()

        assert errors == []
        assert all(isinstance(r, list) for r in results_list)

    def test_source_failure_graceful(self):
        """Source raising an exception should not crash the whole call."""
        from asset_retrieval_tool import bgm_search

        def _raise(q, l):
            raise RuntimeError("network error")

        with patch.object(bgm_search, "_ia_search", _raise):
            results = bgm_search.search_bgm("upbeat")
        assert isinstance(results, list)

    def test_required_fields_present(self):
        from asset_retrieval_tool import bgm_search
        with patch.object(bgm_search, "_ia_search", self._fake_ia):
            results = bgm_search.search_bgm("upbeat")
        required = {"rank", "source", "id", "title", "artist", "duration_sec",
                    "genre", "url", "download_url", "license"}
        for r in results:
            assert required.issubset(r.keys())


# ── downloader.py ─────────────────────────────────────────────────────────────

class TestWriteNetscapeCookiefile:
    """yt-dlp 必须走 cookiefile；直接塞 Cookie 头会被 B站风控拦成 412。"""

    def test_converts_cookie_header(self, tmp_path):
        from asset_retrieval_tool.downloader import _write_netscape_cookiefile

        dest = _write_netscape_cookiefile(
            "SESSDATA=abc123; bili_jct=def456", tmp_path / "c.txt"
        )
        lines = dest.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "# Netscape HTTP Cookie File"
        entries = [l.split("\t") for l in lines if l and not l.startswith("#")]
        assert len(entries) == 2
        assert entries[0][0] == ".bilibili.com"
        assert entries[0][5:7] == ["SESSDATA", "abc123"]
        assert entries[1][5:7] == ["bili_jct", "def456"]

    def test_skips_malformed_parts(self, tmp_path):
        from asset_retrieval_tool.downloader import _write_netscape_cookiefile

        dest = _write_netscape_cookiefile("a=1; ; garbage; b=2", tmp_path / "c.txt")
        entries = [
            l for l in dest.read_text(encoding="utf-8").splitlines()
            if l and not l.startswith("#")
        ]
        assert len(entries) == 2

    def test_value_containing_equals_preserved(self, tmp_path):
        """SESSDATA 里含 % 和 = ，只能按第一个 = 切分。"""
        from asset_retrieval_tool.downloader import _write_netscape_cookiefile

        dest = _write_netscape_cookiefile("SESSDATA=a%2Cb%2A71Cj==", tmp_path / "c.txt")
        entry = [
            l.split("\t") for l in dest.read_text(encoding="utf-8").splitlines()
            if l and not l.startswith("#")
        ][0]
        assert entry[6] == "a%2Cb%2A71Cj=="

    def test_custom_domain(self, tmp_path):
        from asset_retrieval_tool.downloader import _write_netscape_cookiefile

        dest = _write_netscape_cookiefile(
            "k=v", tmp_path / "c.txt", domain=".youtube.com"
        )
        entry = [
            l.split("\t") for l in dest.read_text(encoding="utf-8").splitlines()
            if l and not l.startswith("#")
        ][0]
        assert entry[0] == ".youtube.com"


class TestDownloadVideoResults:
    def test_no_source_video(self, tmp_path):
        from asset_retrieval_tool import download_video_results

        out = download_video_results([{"title": "x"}], tmp_path)
        assert "download_error" in out[0]
        assert "no source_video" in out[0]["download_error"]

    def test_local_file_copied(self, tmp_path):
        from asset_retrieval_tool import download_video_results

        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake video data")
        out = download_video_results(
            [{"source_video": str(src), "platform": "local", "video_id": "v1",
              "title": "clip"}],
            tmp_path / "dl",
        )
        assert "local_video_path" in out[0], out[0].get("download_error")
        assert Path(out[0]["local_video_path"]).read_bytes() == b"fake video data"
        assert out[0]["local_video_bytes"] == 15

    def test_missing_local_source(self, tmp_path):
        from asset_retrieval_tool import download_video_results

        out = download_video_results(
            [{"source_video": "/nope/missing.mp4", "video_id": "v1"}], tmp_path
        )
        assert "source not found" in out[0]["download_error"]

    def test_order_preserved_under_concurrency(self, tmp_path):
        """并发下载后结果顺序必须与输入一致。"""
        from asset_retrieval_tool import download_video_results

        items = []
        for i in range(8):
            f = tmp_path / f"s{i}.mp4"
            f.write_bytes(bytes([i]) * (i + 1))
            items.append({"source_video": str(f), "platform": "local",
                          "video_id": f"v{i}", "title": f"t{i}"})
        out = download_video_results(items, tmp_path / "dl", max_workers=4)
        assert [r["video_id"] for r in out] == [f"v{i}" for i in range(8)]
        assert [r["local_video_bytes"] for r in out] == list(range(1, 9))

    def test_skip_existing(self, tmp_path):
        from asset_retrieval_tool import download_video_results

        src = tmp_path / "src.mp4"
        src.write_bytes(b"original")
        item = {"source_video": str(src), "platform": "local",
                "video_id": "v1", "title": "clip"}
        first = download_video_results([item], tmp_path / "dl")
        dest = Path(first[0]["local_video_path"])
        dest.write_bytes(b"modified-in-place")

        second = download_video_results([item], tmp_path / "dl", skip_existing=True)
        assert dest.read_bytes() == b"modified-in-place"
        assert second[0]["local_video_path"] == str(dest)

    def test_input_not_mutated(self, tmp_path):
        """不得修改调用方传入的 dict。"""
        from asset_retrieval_tool import download_video_results

        src = tmp_path / "src.mp4"
        src.write_bytes(b"data")
        item = {"source_video": str(src), "platform": "local", "video_id": "v1"}
        download_video_results([item], tmp_path / "dl")
        assert "local_video_path" not in item


class TestNormalizeUrl:
    """B站返回协议相对 URL，file:// 页面下会失效。"""

    def test_protocol_relative_gets_https(self):
        from asset_retrieval_tool.video_retrieval import _normalize_url
        assert _normalize_url("//i0.hdslb.com/x.jpg") == "https://i0.hdslb.com/x.jpg"

    def test_absolute_url_untouched(self):
        from asset_retrieval_tool.video_retrieval import _normalize_url
        assert _normalize_url("https://a.com/x.jpg") == "https://a.com/x.jpg"
        assert _normalize_url("http://a.com/x.jpg") == "http://a.com/x.jpg"

    def test_empty_and_none(self):
        from asset_retrieval_tool.video_retrieval import _normalize_url
        assert _normalize_url("") == ""
        assert _normalize_url(None) == ""
