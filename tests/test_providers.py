"""Offline tests for the pluggable footage / BGM provider layer.

No network, no GPU, no external services.

    python3 -m pytest tests/test_providers.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.providers import (  # noqa: E402
    BgmQuery,
    VideoQuery,
    get_bgm_provider,
    get_video_provider,
    list_bgm_providers,
    list_video_providers,
    load_provider_plugins,
    normalize_bgm_track,
    normalize_video_clip,
    provider_options,
    register_bgm_provider,
    register_video_provider,
    run_bgm_provider,
    run_video_provider,
)


# ─── Registry ─────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_bundled_providers_are_registered(self):
        assert "http_video" in list_video_providers()
        assert "local_folder_video" in list_video_providers()
        assert "http_bgm" in list_bgm_providers()
        assert "local_folder_bgm" in list_bgm_providers()

    def test_register_and_resolve(self):
        @register_video_provider("test_echo_video")
        def _echo(query: VideoQuery) -> list[dict]:
            return [{"clip_path": f"/tmp/{query.query}.mp4"}]

        assert get_video_provider("test_echo_video") is _echo
        assert "test_echo_video" in list_video_providers()

    def test_unknown_provider_resolves_to_none(self):
        assert get_video_provider("nope_not_here") is None
        assert get_bgm_provider("nope_not_here") is None

    def test_builtin_names_are_reserved(self):
        # Shadowing a built-in backend name would make dispatch ambiguous.
        with pytest.raises(ValueError):
            register_video_provider("mmfinder_video_cli", lambda q: [])
        with pytest.raises(ValueError):
            register_bgm_provider("asset_retrieval_tool", lambda q: [])

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError):
            register_video_provider("  ", lambda q: [])

    def test_provider_options_reads_config_block(self):
        cfg = {"providers": {"mine": {"url": "https://x", "k": 3}}}
        assert provider_options(cfg, "mine") == {"url": "https://x", "k": 3}
        assert provider_options(cfg, "absent") == {}
        assert provider_options(None, "mine") == {}


# ─── Normalisation ────────────────────────────────────────────────────────────

class TestVideoNormalisation:
    def test_canonical_fields_always_present(self):
        out = normalize_video_clip({}, index=0)
        for key in ("clip_id", "clip_path", "score", "duration_sec", "description", "tags"):
            assert key in out

    def test_aliases_are_mapped(self):
        out = normalize_video_clip(
            {"url": "https://cdn/a.mp4", "title": "City at night", "duration": "42.5"},
            index=1,
        )
        assert out["clip_path"] == "https://cdn/a.mp4"
        assert out["description"] == "City at night"
        assert out["duration_sec"] == 42.5

    def test_explicit_fields_win_over_aliases(self):
        out = normalize_video_clip(
            {"clip_path": "/data/real.mp4", "url": "https://cdn/other.mp4"},
        )
        assert out["clip_path"] == "/data/real.mp4"

    def test_extra_keys_are_preserved(self):
        out = normalize_video_clip({"path": "/a.mp4", "platform": "bilibili"})
        assert out["platform"] == "bilibili"

    def test_generated_id_is_stable_and_prefixed(self):
        out = normalize_video_clip({"path": "/a.mp4"}, index=7, id_prefix="mysrc")
        assert out["clip_id"] == "mysrc_007"

    def test_scalar_tag_is_wrapped(self):
        assert normalize_video_clip({"tags": "aerial"})["tags"] == ["aerial"]

    def test_unparseable_duration_falls_back_to_zero(self):
        assert normalize_video_clip({"duration": "n/a"})["duration_sec"] == 0.0

    def test_non_dict_result_rejected(self):
        with pytest.raises(TypeError):
            normalize_video_clip(["not", "a", "dict"])  # type: ignore[arg-type]


class TestBgmNormalisation:
    def test_canonical_fields_always_present(self):
        out = normalize_bgm_track({}, index=0)
        for key in ("track_id", "music_path", "score", "duration_sec", "name", "caption", "rank"):
            assert key in out

    def test_aliases_are_mapped(self):
        out = normalize_bgm_track(
            {"download_url": "https://ia/x.mp3", "title": "April Showers",
             "artist": "ProleteR", "duration": 269},
            index=0,
        )
        assert out["music_path"] == "https://ia/x.mp3"
        assert out["name"] == "April Showers"
        assert out["caption"] == "ProleteR"
        assert out["duration_sec"] == 269.0
        assert out["rank"] == 1


# ─── Invocation wrapper ───────────────────────────────────────────────────────

class TestRunProvider:
    def test_results_are_normalised(self):
        register_video_provider(
            "test_loose_video",
            lambda q: [{"url": "/a.mp4", "title": "A"}, {"url": "/b.mp4", "title": "B"}],
        )
        out = run_video_provider("test_loose_video", VideoQuery(query="x"))
        assert [c["clip_path"] for c in out] == ["/a.mp4", "/b.mp4"]
        assert all(c["clip_id"].startswith("test_loose_video_") for c in out)

    def test_raising_provider_yields_empty_list(self):
        def _boom(query: VideoQuery) -> list[dict]:
            raise RuntimeError("index unreachable")

        register_video_provider("test_boom_video", _boom)
        assert run_video_provider("test_boom_video", VideoQuery(query="x")) == []

    def test_unknown_provider_yields_empty_list(self):
        assert run_video_provider("test_absent", VideoQuery(query="x")) == []

    def test_dict_response_is_unwrapped(self):
        register_video_provider(
            "test_dict_video",
            lambda q: {"results": [{"path": "/a.mp4"}]},  # type: ignore[return-value]
        )
        out = run_video_provider("test_dict_video", VideoQuery(query="x"))
        assert len(out) == 1

    def test_malformed_entries_are_dropped_not_fatal(self):
        register_video_provider(
            "test_mixed_video",
            lambda q: [{"path": "/good.mp4"}, "garbage", {"path": "/also_good.mp4"}],
        )
        out = run_video_provider("test_mixed_video", VideoQuery(query="x"))
        assert [c["clip_path"] for c in out] == ["/good.mp4", "/also_good.mp4"]

    def test_bgm_provider_runs(self):
        register_bgm_provider(
            "test_bgm",
            lambda q: [{"path": "/m.mp3", "title": "T", "duration": 100}],
        )
        out = run_bgm_provider("test_bgm", BgmQuery(query="calm piano"))
        assert out[0]["music_path"] == "/m.mp3"
        assert out[0]["duration_sec"] == 100.0

    def test_query_carries_options_and_context(self):
        seen: dict = {}

        def _capture(query: VideoQuery) -> list[dict]:
            seen["options"] = query.options
            seen["target"] = query.target_duration_sec
            seen["shot"] = query.shot
            return []

        register_video_provider("test_capture_video", _capture)
        run_video_provider("test_capture_video", VideoQuery(
            query="x",
            target_duration_sec=9.5,
            options={"index": "prod"},
            shot={"shot_id": "shot_003"},
        ))
        assert seen["options"] == {"index": "prod"}
        assert seen["target"] == 9.5
        assert seen["shot"]["shot_id"] == "shot_003"


# ─── local_folder providers ───────────────────────────────────────────────────

class TestLocalFolderProviders:
    @pytest.fixture
    def library(self, tmp_path):
        root = tmp_path / "footage"
        (root / "nested").mkdir(parents=True)
        for name in ("city_night_aerial.mp4", "forest_morning_walk.mp4"):
            (root / name).write_bytes(b"\x00")
        (root / "nested" / "ocean_sunset.mp4").write_bytes(b"\x00")
        (root / "city_night_aerial.json").write_text(
            json.dumps({"caption": "drone over neon skyline",
                        "tags": ["drone", "city"], "duration_sec": 42.0}),
            encoding="utf-8",
        )
        (root / "forest_morning_walk.txt").write_text("hiker in a misty forest", encoding="utf-8")
        return root

    def test_finds_and_ranks_by_query(self, library):
        out = get_video_provider("local_folder_video")(VideoQuery(
            query="city night aerial drone",
            top_k=5,
            options={"roots": [str(library)]},
        ))
        assert out
        assert "city_night_aerial.mp4" in out[0]["clip_path"]

    def test_json_sidecar_supplies_caption_tags_duration(self, library):
        out = get_video_provider("local_folder_video")(VideoQuery(
            query="drone city",
            top_k=5,
            options={"roots": [str(library)]},
        ))
        top = out[0]
        assert top["description"] == "drone over neon skyline"
        assert "drone" in top["tags"]
        assert top["duration_sec"] == 42.0

    def test_txt_sidecar_supplies_caption(self, library):
        out = get_video_provider("local_folder_video")(VideoQuery(
            query="misty forest hiker",
            top_k=5,
            options={"roots": [str(library)]},
        ))
        assert out[0]["description"] == "hiker in a misty forest"

    def test_recurses_into_subdirectories(self, library):
        out = get_video_provider("local_folder_video")(VideoQuery(
            query="ocean sunset",
            top_k=5,
            options={"roots": [str(library)]},
        ))
        assert any("ocean_sunset.mp4" in c["clip_path"] for c in out)

    def test_require_match_drops_unrelated_files(self, library):
        out = get_video_provider("local_folder_video")(VideoQuery(
            query="zzz totally unrelated query",
            top_k=5,
            options={"roots": [str(library)], "require_match": True},
        ))
        assert out == []

    def test_missing_roots_returns_empty(self):
        assert get_video_provider("local_folder_video")(VideoQuery(query="x")) == []

    def test_nonexistent_root_is_tolerated(self):
        out = get_video_provider("local_folder_video")(VideoQuery(
            query="x", options={"roots": ["/definitely/not/here"]},
        ))
        assert out == []

    def test_bgm_scans_audio_files(self, tmp_path):
        root = tmp_path / "music"
        root.mkdir()
        (root / "calm_piano_loop.mp3").write_bytes(b"\x00")
        (root / "upbeat_electronic.mp3").write_bytes(b"\x00")
        out = get_bgm_provider("local_folder_bgm")(BgmQuery(
            query="calm piano",
            top_k=5,
            options={"roots": [str(root)]},
        ))
        assert out
        assert "calm_piano_loop.mp3" in out[0]["music_path"]


# ─── http providers ───────────────────────────────────────────────────────────

class TestHttpProviders:
    def test_missing_url_is_not_fatal(self):
        out = run_video_provider("http_video", VideoQuery(query="x", options={}))
        assert out == []

    def test_response_is_parsed_and_mapped(self, monkeypatch):
        import pipeline.providers.http_api as http_api

        captured: dict = {}

        def fake_request(options, payload):
            captured["options"] = options
            captured["payload"] = payload
            return {"data": {"items": [
                {"playback_url": "https://cdn/a.mp4", "name": "A", "seconds": 30},
                {"playback_url": "https://cdn/b.mp4", "name": "B", "seconds": 12},
            ]}}

        monkeypatch.setattr(http_api, "_request", fake_request)

        out = run_video_provider("http_video", VideoQuery(
            query="city night",
            top_k=4,
            target_duration_sec=8.0,
            options={
                "url": "https://footage.test/search",
                "results_path": "data.items",
                "field_map": {"clip_path": "playback_url", "duration_sec": "seconds"},
            },
        ))
        assert [c["clip_path"] for c in out] == ["https://cdn/a.mp4", "https://cdn/b.mp4"]
        assert out[0]["duration_sec"] == 30.0
        assert captured["payload"]["query"] == "city night"
        assert captured["payload"]["top_k"] == 4
        assert captured["payload"]["target_duration_sec"] == 8.0

    def test_custom_request_field_names(self, monkeypatch):
        import pipeline.providers.http_api as http_api

        captured: dict = {}

        def fake_request(options, payload):
            captured.update(payload)
            return []

        monkeypatch.setattr(http_api, "_request", fake_request)
        run_video_provider("http_video", VideoQuery(
            query="x", top_k=3,
            options={"url": "https://t/s", "query_field": "q", "top_k_field": "n"},
        ))
        assert captured["q"] == "x"
        assert captured["n"] == 3

    def test_bare_array_response(self, monkeypatch):
        import pipeline.providers.http_api as http_api
        monkeypatch.setattr(
            http_api, "_request",
            lambda options, payload: [{"url": "https://cdn/a.mp4", "title": "A"}],
        )
        out = run_video_provider("http_video", VideoQuery(
            query="x", options={"url": "https://t/s"},
        ))
        assert len(out) == 1

    def test_env_expansion_in_headers(self, monkeypatch):
        import pipeline.providers.http_api as http_api
        monkeypatch.setenv("TEST_PROVIDER_TOKEN", "s3cret")
        assert http_api._expand({"Authorization": "Bearer ${TEST_PROVIDER_TOKEN}"}) == {
            "Authorization": "Bearer s3cret"
        }

    def test_bgm_endpoint(self, monkeypatch):
        import pipeline.providers.http_api as http_api
        monkeypatch.setattr(
            http_api, "_request",
            lambda options, payload: {"tracks": [
                {"url": "https://cdn/x.mp3", "title": "X", "duration": 180},
            ]},
        )
        out = run_bgm_provider("http_bgm", BgmQuery(
            query="ambient", options={"url": "https://t/s", "results_path": "tracks"},
        ))
        assert out[0]["music_path"] == "https://cdn/x.mp3"


# ─── plugin loading ───────────────────────────────────────────────────────────

class TestPluginLoading:
    def test_load_from_file_path(self, tmp_path):
        plugin = tmp_path / "my_plugin.py"
        plugin.write_text(
            "from pipeline.providers import register_video_provider\n"
            "@register_video_provider('test_from_file')\n"
            "def s(query):\n"
            "    return [{'path': '/from_file.mp4'}]\n",
            encoding="utf-8",
        )
        assert load_provider_plugins([str(plugin)]) == [str(plugin)]
        assert get_video_provider("test_from_file") is not None

    def test_broken_plugin_is_skipped_not_fatal(self):
        assert load_provider_plugins(["definitely.not.a.real.module"]) == []

    def test_empty_and_bad_input_tolerated(self):
        assert load_provider_plugins(None) == []
        assert load_provider_plugins([]) == []
        assert load_provider_plugins("") == []


# ─── config integration ──────────────────────────────────────────────────────

class TestConfigIntegration:
    def test_pipeline_config_exposes_provider_knobs(self):
        from studio.config import load
        pc = load().pipeline_config
        assert "retrieval_provider" in pc
        assert "bgm_provider" in pc
        assert isinstance(pc["providers"], dict)

    def test_bgm_provider_defaults_to_retrieval_provider(self, tmp_path):
        from studio.config import DEFAULT_CONFIG_PATH, load
        text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(text, encoding="utf-8")
        pc = load(str(cfg_path)).pipeline_config
        # The shipped config leaves bgm_provider empty on purpose.
        assert pc["bgm_provider"] == pc["retrieval_provider"]

    def test_env_interpolation(self, monkeypatch):
        from studio.config import interpolate
        monkeypatch.setenv("WEVISIONCUT_TEST_VAR", "/data/x")
        assert interpolate("${WEVISIONCUT_TEST_VAR}/y", {}) == "/data/x/y"
        assert interpolate("${WEVISIONCUT_UNSET_VAR:-/fallback}", {}) == "/fallback"
        assert interpolate("${REPO_ROOT}/z", {"REPO_ROOT": "/repo"}) == "/repo/z"
        # A fallback may itself reference an injected variable.
        assert interpolate("${NOPE:-${REPO_ROOT}/z}", {"REPO_ROOT": "/repo"}) == "/repo/z"
        # Nested structures are walked.
        assert interpolate(
            {"a": ["${REPO_ROOT}/1", {"b": "${REPO_ROOT}/2"}]}, {"REPO_ROOT": "/r"},
        ) == {"a": ["/r/1", {"b": "/r/2"}]}

    def test_registered_provider_accepted_in_backend_order(self):
        register_video_provider("test_order_video", lambda q: [])
        from pipeline.tools.retrieve_visual import _normalize_backend_order
        assert _normalize_backend_order(["test_order_video"]) == ["test_order_video"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))
