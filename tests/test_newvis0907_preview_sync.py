"""Focused regressions for preview BGM and segment timeline layout.

Run: python3 tests/test_newvis0907_preview_sync.py
"""

from __future__ import annotations

import sys
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studio import app as app_mod  # noqa: E402
from studio import tracks  # noqa: E402


class PreviewBgmCompatTest(unittest.TestCase):
    def test_preview_bgm_accepts_path_key(self) -> None:
        tmp = Path("/tmp/aicut_preview_sync")
        tmp.mkdir(parents=True, exist_ok=True)
        inside = tmp / "bgm.mp3"
        inside.write_bytes(b"ID3")

        out = app_mod._preview_bgm(
            {"bgm_tracks": [{"path": str(inside), "name": "bgm_a"}], "bgm_volume": 0.4},
            [tmp.resolve()],
        )
        self.assertEqual(len(out["tracks"]), 1, out)
        self.assertEqual(out["tracks"][0]["path"], str(inside.resolve()), out)

    def test_music_library_is_playable_so_preview_bgm_is_not_silent(self) -> None:
        # The music hub is not under media_roots, so every track used to be
        # dropped here and the shot-by-shot preview played with no music at all
        # — while export sounded fine, because ffmpeg reads the file directly.
        library = Path("/tmp/aicut_preview_music_hub")
        library.mkdir(parents=True, exist_ok=True)
        track = library / "hopeful.mp3"
        track.write_bytes(b"ID3")

        app_mod._music_library_roots.cache_clear()
        stub_studio = SimpleNamespace(config=SimpleNamespace(media_roots=[]))
        with mock.patch.object(app_mod, "studio", stub_studio), \
                mock.patch.object(app_mod, "_music_library_roots",
                                  return_value=(library.resolve(),)):
            roots = app_mod._playable_roots()
            self.assertIn(library.resolve(), roots)
            out = app_mod._preview_bgm(
                {"bgm_tracks": [{"music_path": str(track)}], "bgm_volume": 0.5},
                roots,
            )
        app_mod._music_library_roots.cache_clear()
        self.assertEqual(len(out["tracks"]), 1, out)
        self.assertEqual(out["tracks"][0]["path"], str(track.resolve()), out)

    def test_0907_policy_copy_keeps_configured_stitch_deferral(self) -> None:
        # Forcing a real concat during generation aborts the run whenever a shot
        # has no rendered file yet, so the policy copy must not touch it.
        base = {"render": {"defer_stitch": True}, "music": {"bgm_volume": 0.5}}
        out = app_mod._pipeline_config_for_policy(base, "0907")
        self.assertIsNot(out, base)
        self.assertEqual(out["policy_version"], "0907", out)
        self.assertTrue(out["render"]["defer_stitch"], out)
        self.assertNotIn("policy_version", base, base)

    def test_bgm_mix_accepts_path_and_audio_path_keys(self) -> None:
        from pipeline.tools import stitch_video

        tmp = Path("/tmp/aicut_bgm_mix_keys")
        tmp.mkdir(parents=True, exist_ok=True)
        video = tmp / "video.mp4"
        path_track = tmp / "path_track.mp3"
        audio_track = tmp / "audio_track.mp3"
        video.write_bytes(b"video")
        path_track.write_bytes(b"ID3path")
        audio_track.write_bytes(b"ID3audio")
        out = tmp / "mixed.mp4"

        def fake_run(cmd, **_kwargs):
            Path(cmd[-1]).write_bytes(b"mixed")
            return SimpleNamespace(returncode=0, stderr="", stdout="")

        with mock.patch.object(stitch_video, "_get_video_duration", return_value=12.0), \
                mock.patch.object(stitch_video, "_has_audio_stream", return_value=True), \
                mock.patch.object(stitch_video.subprocess, "run", side_effect=fake_run):
            result = stitch_video._mix_bgm(
                str(video),
                [{"path": str(path_track)}, {"audio_path": str(audio_track)}],
                str(out),
                video_duration=10.0,
            )

        self.assertEqual(result["status"], "bgm_mixed", result)
        self.assertEqual(result["bgm_track_count"], 2, result)


class SegmentTimelineLayoutTest(unittest.TestCase):
    def test_segments_are_non_overlapping_even_with_duplicate_raw_segments(self) -> None:
        state = {
            "segments": [
                {"segment_id": "seg_a", "title": "A"},
                {"segment_id": "seg_b", "title": "B"},
                # Duplicate entry that previously produced an overlapping segment tile.
                {"segment_id": "seg_b", "title": "B-dup"},
            ],
            "shot_plan": [
                {"shot_id": "S01", "segment_id": "seg_a", "duration_sec": 4.0, "vocal_script": "a"},
                {"shot_id": "S02", "segment_id": "seg_b", "duration_sec": 5.0, "vocal_script": "b1"},
                {"shot_id": "S03", "segment_id": "seg_b", "duration_sec": 6.0, "vocal_script": "b2"},
            ],
            "shot_assets": {},
        }
        segments = tracks.build_snapshot(state)["clips"]["segments"]
        self.assertEqual(len(segments), 2, segments)
        for i in range(1, len(segments)):
            self.assertGreaterEqual(
                segments[i]["start"],
                segments[i - 1]["end"],
                segments,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
