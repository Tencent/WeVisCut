"""Offline smoke-test: verify retrieval_provider switch routes correctly.

Runs without network — all external calls are mocked.
Usage:
    cd aicut_online
    python -m pytest tests/test_retrieval_provider_switch.py -v
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

# Ensure the project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Also add asset_retrieval_tool package path
_ART_PKG = _PROJECT_ROOT.parent / "external_tool" / "asset_retrieval_tool"
if str(_ART_PKG) not in sys.path:
    sys.path.insert(0, str(_ART_PKG))


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_art_video_results():
    """Simulated asset_retrieval_tool.search_video_segments output."""
    return [
        {
            "rank": 1,
            "platform": "bilibili",
            "video_id": "BV1test123",
            "source_video": "https://www.bilibili.com/video/BV1test123",
            "title": "城市夜景航拍",
            "description": "无人机俯瞰城市霓虹灯",
            "channel": "TestChannel",
            "thumbnail": "https://example.com/thumb.jpg",
            "start": 0.0,
            "end": 45.0,
            "duration": 45.0,
            "score": 0.85,
            "semantic_text": "城市夜景航拍 无人机俯瞰城市霓虹灯",
        },
        {
            "rank": 2,
            "platform": "bilibili",
            "video_id": "BV2test456",
            "source_video": "https://www.bilibili.com/video/BV2test456",
            "title": "城市延时摄影",
            "description": "日落到夜晚的城市变化",
            "channel": "TestChannel2",
            "thumbnail": "https://example.com/thumb2.jpg",
            "start": 0.0,
            "end": 30.0,
            "duration": 30.0,
            "score": 0.72,
            "semantic_text": "城市延时摄影 日落到夜晚的城市变化",
        },
    ]


@pytest.fixture
def mock_art_bgm_results():
    """Simulated asset_retrieval_tool.search_bgm output."""
    return [
        {
            "rank": 1,
            "source": "internet_archive",
            "id": "test_album/track01.mp3",
            "title": "Ambient Journey",
            "artist": "TestArtist",
            "duration_sec": 195.0,
            "genre": "netlabel / CC",
            "url": "https://archive.org/details/test_album",
            "download_url": "https://archive.org/download/test_album/track01.mp3",
            "license": "CC",
        },
        {
            "rank": 2,
            "source": "internet_archive",
            "id": "test_album/track02.mp3",
            "title": "Chill Piano",
            "artist": "TestArtist",
            "duration_sec": 240.0,
            "genre": "netlabel / CC",
            "url": "https://archive.org/details/test_album",
            "download_url": "https://archive.org/download/test_album/track02.mp3",
            "license": "CC",
        },
    ]


@pytest.fixture
def base_pipeline_config():
    """Minimal pipeline config dict for testing."""
    return {
        "paths": {
            "asset_library": "/tmp/test_asset_lib",
            "output_root": "/tmp/test_output",
        },
        "retrieval": {
            "top_k": 10,
            "backend_order": ["mmfinder_video_cli"],
            "video_cli_size": 20,
            "min_duration_ratio": 0.55,
            "max_loop_count": 2.0,
            "duration_tolerance_sec": 1.5,
            "allow_short_candidate_fallback": False,
            "require_horizontal": True,
            "min_aspect_ratio": 1.33,
            "return_candidates": 10,
            "select_with_llm": False,
        },
        "music": {
            "enabled": True,
            "top_k": 10,
        },
        "execution": {"mode": "real"},
    }


# ─── Tests: Config loading ────────────────────────────────────────────────────

class TestConfigSwitch:
    """Verify studio_config.yaml loads retrieval_provider correctly."""

    def test_config_load_default_provider(self):
        """Default config ships the open-web provider, which needs no service."""
        from studio.config import load
        cfg = load()
        provider = cfg.pipeline_config.get("retrieval_provider", "")
        assert provider == "asset_retrieval_tool"

    def test_config_provider_env_override(self, monkeypatch):
        """WEVISIONCUT_VIDEO_PROVIDER selects a provider without editing YAML."""
        from studio.config import load
        monkeypatch.setenv("WEVISIONCUT_VIDEO_PROVIDER", "local_folder_video")
        cfg = load()
        assert cfg.pipeline_config.get("retrieval_provider") == "local_folder_video"

    def test_config_has_asset_retrieval_section(self):
        """Config should contain asset_retrieval settings."""
        from studio.config import load
        cfg = load()
        art_cfg = cfg.pipeline_config.get("asset_retrieval", {})
        assert isinstance(art_cfg, dict)
        assert "video_sources" in art_cfg or art_cfg == {}


# ─── Tests: Video retrieval routing ───────────────────────────────────────────

class TestVideoRetrievalRouting:
    """Verify video retrieval routes to correct backend based on provider."""

    def test_mmfinder_backend_order_unchanged(self, base_pipeline_config):
        """When provider=mmfinder, backend_order uses mmfinder backends."""
        from pipeline.tools.retrieve_visual import _normalize_backend_order
        order = _normalize_backend_order(["mmfinder_video_cli", "mmfinder_subprocess"])
        assert "mmfinder_video_cli" in order
        assert "asset_retrieval_tool" not in order

    def test_asset_retrieval_tool_backend_allowed(self):
        """asset_retrieval_tool should be in the allowed backends set."""
        from pipeline.tools.retrieve_visual import _normalize_backend_order
        order = _normalize_backend_order(["asset_retrieval_tool"])
        assert order == ["asset_retrieval_tool"]

    def test_run_with_asset_retrieval_provider(
        self, base_pipeline_config, mock_art_video_results
    ):
        """run() with provider=asset_retrieval_tool should call asset_retrieval_tool."""
        config = {
            **base_pipeline_config,
            "retrieval_provider": "asset_retrieval_tool",
            "asset_retrieval": {
                "video_sources": ["bilibili"],
                "bilibili_cookie": "",
                "youtube_api_key": "",
                "auto_download": False,
            },
        }

        mock_search = MagicMock(return_value=mock_art_video_results)
        mock_download = MagicMock(return_value=mock_art_video_results)

        with patch.dict(sys.modules, {
            "asset_retrieval_tool": types.ModuleType("asset_retrieval_tool"),
        }):
            sys.modules["asset_retrieval_tool"].search_video_segments = mock_search
            sys.modules["asset_retrieval_tool"].download_video_results = mock_download

            from pipeline.tools.retrieve_visual import run

            shot = {
                "shot_id": "shot_001",
                "visual_query": "城市夜景航拍",
                "visual_direction": "俯瞰都市霓虹",
                "vocal_script": "在这座不夜城中...",
                "duration_sec": 8.0,
                "tts_duration_sec": 7.5,
            }

            mock_llm = MagicMock()
            result = run(mock_llm, shot, config)

            assert result["status"] == "success"
            assert result["backend"] == "asset_retrieval_tool"
            assert result["candidates_count"] > 0
            mock_search.assert_called_once()

    def test_run_with_mmfinder_provider_does_not_call_art(
        self, base_pipeline_config
    ):
        """run() with provider=mmfinder should NOT call asset_retrieval_tool."""
        config = {
            **base_pipeline_config,
            "retrieval_provider": "mmfinder",
        }

        shot = {
            "shot_id": "shot_002",
            "visual_query": "足球比赛精彩进球",
            "visual_direction": "特写慢动作",
            "vocal_script": "进球了！",
            "duration_sec": 6.0,
        }

        mock_llm = MagicMock()

        with patch(
            "pipeline.tools.retrieve_visual._recall_videos_via_cli",
            return_value=[],
        ), patch(
            "pipeline.tools.retrieve_visual._search_via_subprocess",
            return_value=[],
        ), patch(
            "pipeline.tools.retrieve_visual._search_via_embedding",
            return_value=[],
        ), patch(
            "pipeline.tools.retrieve_visual._search_via_asset_retrieval_tool",
        ) as mock_art:
            from pipeline.tools.retrieve_visual import run
            result = run(mock_llm, shot, config)
            mock_art.assert_not_called()
            assert result["backend"] == "none"


# ─── Tests: BGM retrieval routing ─────────────────────────────────────────────

class TestBGMRetrievalRouting:
    """Verify BGM search routes to correct backend based on provider."""

    def test_search_music_with_asset_retrieval_provider(self, mock_art_bgm_results):
        """search_music_with_backend with provider=asset_retrieval_tool uses IA."""
        mock_search = MagicMock(return_value=mock_art_bgm_results)
        mock_download = MagicMock(return_value=[
            {**r, "local_path": f"/tmp/fake/{r['title']}.mp3"}
            for r in mock_art_bgm_results
        ])

        with patch.dict(sys.modules, {
            "asset_retrieval_tool": types.ModuleType("asset_retrieval_tool"),
        }):
            sys.modules["asset_retrieval_tool"].search_bgm = mock_search
            sys.modules["asset_retrieval_tool"].download_bgm_results = mock_download

            from pipeline.tools.music_search import search_music_with_backend

            tracks, backend = search_music_with_backend(
                music_style="轻快钢琴纯音乐",
                target_duration=60.0,
                top_k=5,
                retrieval_provider="asset_retrieval_tool",
                asset_retrieval_cfg={
                    "bgm_sources": ["internet_archive"],
                    "auto_download": True,
                },
            )

            assert backend == "asset_retrieval_tool"
            assert len(tracks) > 0
            assert tracks[0]["track_id"].startswith("art_bgm_")
            mock_search.assert_called_once()

    def test_search_music_with_mmfinder_provider_skips_art(self):
        """search_music_with_backend with provider=mmfinder skips asset_retrieval_tool."""
        with patch(
            "pipeline.tools.music_search._recall_music_via_cli",
            return_value=[],
        ), patch(
            "pipeline.tools.music_search._embed_music_query",
            return_value=[],
        ), patch(
            "pipeline.tools.music_search._scan_local_music",
            return_value=[],
        ), patch(
            "pipeline.tools.music_search._search_bgm_via_asset_retrieval_tool",
        ) as mock_art:
            from pipeline.tools.music_search import search_music_with_backend

            tracks, backend = search_music_with_backend(
                music_style="电子节拍纯音乐",
                target_duration=60.0,
                top_k=5,
                retrieval_provider="mmfinder",
            )

            mock_art.assert_not_called()

    def test_run_with_asset_retrieval_provider(
        self, base_pipeline_config, mock_art_bgm_results
    ):
        """Full run() with provider=asset_retrieval_tool."""
        config = {
            **base_pipeline_config,
            "retrieval_provider": "asset_retrieval_tool",
            "asset_retrieval": {
                "bgm_sources": ["internet_archive"],
                "auto_download": False,
            },
        }

        mock_search = MagicMock(return_value=mock_art_bgm_results)
        mock_download = MagicMock(return_value=mock_art_bgm_results)

        with patch.dict(sys.modules, {
            "asset_retrieval_tool": types.ModuleType("asset_retrieval_tool"),
        }):
            sys.modules["asset_retrieval_tool"].search_bgm = mock_search
            sys.modules["asset_retrieval_tool"].download_bgm_results = mock_download

            from pipeline.tools.music_search import run

            result = run(
                music_style="安静舒缓的钢琴纯音乐",
                total_video_duration=120.0,
                config=config,
            )

            assert result["status"] == "success"
            assert result["backend"] == "asset_retrieval_tool"
            assert result["selected_count"] > 0
            mock_search.assert_called()


# ─── Tests: Integration — config → pipeline_config projection ─────────────────

class TestConfigProjection:
    """Verify the full config → pipeline_config → tool route."""

    def test_pipeline_config_carries_provider(self):
        """pipeline_config must carry retrieval_provider and asset_retrieval."""
        from studio.config import load
        cfg = load()
        pc = cfg.pipeline_config
        assert "retrieval_provider" in pc
        assert pc["retrieval_provider"] in ("mmfinder", "asset_retrieval_tool")
        assert "asset_retrieval" in pc
        assert isinstance(pc["asset_retrieval"], dict)

    def test_switch_to_asset_retrieval_in_config(self, tmp_path):
        """Overriding retrieval_provider in YAML should propagate."""
        import yaml
        from studio.config import load, DEFAULT_CONFIG_PATH

        # Read current config and patch provider
        original = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
        patched = original.replace(
            "retrieval_provider: mmfinder",
            "retrieval_provider: asset_retrieval_tool",
        )
        tmp_cfg = tmp_path / "test_config.yaml"
        tmp_cfg.write_text(patched, encoding="utf-8")

        cfg = load(str(tmp_cfg))
        assert cfg.pipeline_config["retrieval_provider"] == "asset_retrieval_tool"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
