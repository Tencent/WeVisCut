"""Offline tests for newvis0907 hierarchy projection.

Run: python3 tests/test_newvis0907_hierarchy.py
"""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studio.newvis0907.hierarchy import build_hierarchy  # noqa: E402


def make_state() -> dict:
    return {
        "segments": [
            {"segment_id": "seg_a", "title": "A", "summary": "sum A"},
            {"segment_id": "seg_b", "title": "B", "summary": "sum B"},
        ],
        "shot_plan": [
            {
                "shot_id": "S1",
                "segment_id": "seg_a",
                "duration_sec": 10,
                "vocal_script": "vocal1",
                "subtitle_text": "sub1",
                "subtitle_display_text": "disp1",
                "semantic_animations": {"primary": {"type": "zoom"}},
                "quality_status": "ok",
            },
            {
                "shot_id": "S2",
                "segment_id": "seg_unknown",
                "duration_sec": 5,
                "vocal_script": "vocal2",
                "subtitle_text": "sub2",
            },
            {
                "shot_id": "S3",
                "segment_id": "seg_b",
                "vocal_script": "vocal3",
            },
        ],
        "shot_assets": {
            "S1": {
                "audio_meta": {"duration_sec": 12},
                "clip_meta": {
                    "visual_playlist": {
                        "clips": [
                            {
                                "clip_id": "n1",
                                "path": "/n1.mp4",
                                "offset_sec": 0,
                                "used_duration_sec": 2,
                                "source_in_sec": 1,
                                "source_out_sec": 5,
                                "score": 0.9,
                                "query": "q1",
                                "fingerprint": "fp-n1",
                            },
                            {
                                "path": "/n2.mp4",
                                "used_duration_sec": 3,
                                "source_in_sec": 0,
                                "source_out_sec": 3,
                            },
                            {
                                "path": "/n3.mp4",
                                "offset_sec": 11,
                                "used_duration_sec": 5,
                                "replaced": True,
                                "user_override": True,
                            },
                        ]
                    }
                },
            },
            "S2": {
                "visual_playlist": {
                    "clips": [
                        {
                            "clip_id": "d1",
                            "path": "/d1.mp4",
                            "offset_sec": 1.5,
                            "used_duration_sec": 2.0,
                            "source_in_sec": 1.0,
                            "source_out_sec": 5.0,
                        }
                    ]
                },
                "decoration": {"structure_id": "decor-2"},
            },
            "S3": {
                "clip_path": "/s3.mp4",
                "clip_meta": {"source_duration_sec": 9},
            },
        },
    }


class BuildHierarchyTest(unittest.TestCase):
    def test_nested_and_direct_playlist_and_fallback_subclips(self) -> None:
        payload = build_hierarchy(make_state())
        shots = {s["shot_id"]: s for s in payload["model_shots"]}

        s1_sub = shots["S1"]["subclips"]
        self.assertEqual(len(s1_sub), 3)
        self.assertEqual(s1_sub[0]["clip_id"], "n1")
        self.assertEqual(s1_sub[0]["display_number"], "1.1")
        self.assertAlmostEqual(s1_sub[0]["start"], 0.0, places=4)
        self.assertAlmostEqual(s1_sub[0]["end"], 2.0, places=4)
        self.assertAlmostEqual(s1_sub[0]["playback_rate"], 2.0, places=4)

        self.assertEqual(s1_sub[1]["display_number"], "1.2")
        self.assertTrue(s1_sub[1]["subclip_id"].startswith("S1::subclip::"))
        self.assertAlmostEqual(s1_sub[1]["start"], 2.0, places=4)
        self.assertAlmostEqual(s1_sub[1]["end"], 5.0, places=4)

        self.assertEqual(s1_sub[2]["display_number"], "1.3")
        self.assertAlmostEqual(s1_sub[2]["start"], 11.0, places=4)
        self.assertAlmostEqual(s1_sub[2]["end"], 12.0, places=4)
        self.assertTrue(s1_sub[2]["replaced"])
        self.assertTrue(s1_sub[2]["user_override"])

        s2_sub = shots["S2"]["subclips"]
        self.assertEqual(len(s2_sub), 1)
        self.assertEqual(s2_sub[0]["clip_id"], "d1")
        self.assertEqual(s2_sub[0]["display_number"], "2.1")

        s3_sub = shots["S3"]["subclips"]
        self.assertEqual(len(s3_sub), 1)
        self.assertEqual(s3_sub[0]["path"], "/s3.mp4")
        self.assertEqual(s3_sub[0]["display_number"], "3.1")

    def test_chapter_range_and_unknown_segment_bucket(self) -> None:
        payload = build_hierarchy(make_state())
        chapters = payload["chapters"]
        by_id = {c["segment_id"]: c for c in chapters}
        unknown = next(c for c in chapters if c["title"] == "未分章")

        self.assertEqual(chapters[0]["segment_id"], "seg_a")
        self.assertEqual(chapters[1]["segment_id"], "seg_b")
        self.assertEqual(by_id["seg_a"]["model_shot_ids"], ["S1"])
        self.assertEqual(by_id["seg_b"]["model_shot_ids"], ["S3"])
        self.assertEqual(unknown["model_shot_ids"], ["S2"])

        self.assertAlmostEqual(by_id["seg_a"]["start"], 0.0, places=4)
        self.assertAlmostEqual(by_id["seg_a"]["end"], 12.0, places=4)
        self.assertAlmostEqual(by_id["seg_b"]["start"], 17.0, places=4)
        self.assertAlmostEqual(by_id["seg_b"]["end"], 25.0, places=4)
        self.assertAlmostEqual(unknown["start"], 12.0, places=4)
        self.assertAlmostEqual(unknown["end"], 17.0, places=4)

    def test_shot_fields_and_priority_rules(self) -> None:
        payload = build_hierarchy(make_state())
        shots = payload["model_shots"]

        self.assertEqual([s["display_number"] for s in shots], ["01", "02", "03"])

        self.assertEqual(shots[0]["display_subtitle"], "disp1")
        self.assertEqual(shots[1]["display_subtitle"], "sub2")
        self.assertEqual(shots[2]["display_subtitle"], "vocal3")

        self.assertEqual(shots[0]["animation_type"], "zoom")
        self.assertEqual(shots[1]["animation_type"], "decor-2")
        self.assertEqual(shots[2]["animation_type"], "")

        self.assertAlmostEqual(shots[0]["start"], 0.0, places=4)
        self.assertAlmostEqual(shots[0]["end"], 12.0, places=4)
        self.assertAlmostEqual(shots[1]["start"], 12.0, places=4)
        self.assertAlmostEqual(shots[1]["end"], 17.0, places=4)
        self.assertAlmostEqual(shots[2]["start"], 17.0, places=4)
        self.assertAlmostEqual(shots[2]["end"], 25.0, places=4)

    def test_missing_fields_are_tolerated(self) -> None:
        state = {"shot_plan": [{}], "segments": [], "shot_assets": {}}
        payload = build_hierarchy(state)
        self.assertEqual(len(payload["model_shots"]), 1)
        shot = payload["model_shots"][0]
        self.assertEqual(shot["shot_id"], "shot_01")
        self.assertEqual(shot["display_number"], "01")
        self.assertAlmostEqual(shot["duration"], 8.0, places=4)
        self.assertEqual(shot["subclips"], [])

    def test_payload_is_json_serializable(self) -> None:
        payload = build_hierarchy(make_state())
        blob = json.dumps(payload, ensure_ascii=False)
        self.assertIn('"chapters"', blob)
        self.assertIn('"model_shots"', blob)

    def test_reorder_and_delete_keeps_continuous_numbering(self) -> None:
        state = make_state()
        state["shot_plan"] = [state["shot_plan"][2], state["shot_plan"][0]]
        payload = build_hierarchy(state)
        shots = payload["model_shots"]
        self.assertEqual([s["shot_id"] for s in shots], ["S3", "S1"])
        self.assertEqual([s["display_number"] for s in shots], ["01", "02"])
        self.assertEqual(
            [sc["display_number"] for sc in shots[1]["subclips"]],
            ["2.1", "2.2", "2.3"],
        )

    def test_input_state_not_modified(self) -> None:
        state = make_state()
        before = copy.deepcopy(state)
        _ = build_hierarchy(state)
        self.assertEqual(state, before)

    def test_pick_playlist_supports_direct_list_shape(self) -> None:
        state = {
            "segments": [{"segment_id": "seg_x", "title": "X"}],
            "shot_plan": [{"shot_id": "S10", "segment_id": "seg_x", "duration_sec": 5}],
            "shot_assets": {
                "S10": {
                    "visual_playlist": [
                        {"clip_id": "l1", "path": "/l1.mp4", "used_duration_sec": 2},
                        {"clip_id": "l2", "path": "/l2.mp4", "used_duration_sec": 3},
                    ]
                }
            },
        }
        payload = build_hierarchy(state)
        sub = payload["model_shots"][0]["subclips"]
        self.assertEqual([c["clip_id"] for c in sub], ["l1", "l2"])
        self.assertEqual([c["display_number"] for c in sub], ["1.1", "1.2"])

    def test_subclip_start_priority_and_clamp(self) -> None:
        state = {
            "segments": [{"segment_id": "seg_x", "title": "X"}],
            "shot_plan": [{"shot_id": "S11", "segment_id": "seg_x", "duration_sec": 6}],
            "shot_assets": {
                "S11": {
                    "clip_meta": {
                        "visual_playlist": {
                            "clips": [
                                {
                                    "clip_id": "p1",
                                    "path": "/p1.mp4",
                                    "start": 0.5,
                                    "start_sec": 2.0,
                                    "offset_sec": 4.0,
                                    "used_duration_sec": 4.0,
                                },
                                {
                                    "clip_id": "p2",
                                    "path": "/p2.mp4",
                                    "start": 1.0,
                                    "start_sec": 2.5,
                                    "used_duration_sec": 1.0,
                                },
                                {
                                    "clip_id": "p3",
                                    "path": "/p3.mp4",
                                    "start": 5.0,
                                    "used_duration_sec": 5.0,
                                },
                            ]
                        }
                    }
                }
            },
        }
        payload = build_hierarchy(state)
        sub = payload["model_shots"][0]["subclips"]
        # offset_sec > start_sec > start
        self.assertAlmostEqual(sub[0]["start"], 4.0, places=4)
        self.assertAlmostEqual(sub[0]["end"], 6.0, places=4)  # clamped
        self.assertAlmostEqual(sub[1]["start"], 2.5, places=4)
        self.assertAlmostEqual(sub[2]["start"], 5.0, places=4)
        self.assertAlmostEqual(sub[2]["end"], 6.0, places=4)  # clamped

    def test_unique_stable_shot_ids_and_id_reuse_in_chapters_subclips(self) -> None:
        state = {
            "segments": [
                {"segment_id": "seg_u", "title": "U"},
                {"segment_id": "seg_v", "title": "V"},
            ],
            "shot_plan": [
                {"shot_id": "", "segment_id": "seg_u", "duration_sec": 4},
                {"shot_id": "A", "segment_id": "seg_u", "duration_sec": 4},
                {"shot_id": "A", "segment_id": "seg_v", "duration_sec": 4},
                {"segment_id": "seg_v", "duration_sec": 4},
                {"shot_id": "A__2", "segment_id": "seg_v", "duration_sec": 4},
                {"shot_id": "A", "segment_id": "seg_v", "duration_sec": 4},
            ],
            "shot_assets": {
                "A": {"clip_path": "/a.mp4"},
                "A__2": {"clip_path": "/a2.mp4"},
            },
        }
        before = copy.deepcopy(state)
        payload = build_hierarchy(state)
        shots = payload["model_shots"]
        ids = [s["shot_id"] for s in shots]
        self.assertEqual(ids[0], "shot_01")
        self.assertEqual(ids[1], "A")
        self.assertEqual(ids[2], "A__2")
        self.assertEqual(ids[3], "shot_04")
        self.assertEqual(ids[4], "A__2__2")
        self.assertEqual(ids[5], "A__3")
        self.assertEqual(len(ids), len(set(ids)))

        chapters = {c["segment_id"]: c for c in payload["chapters"]}
        self.assertEqual(chapters["seg_u"]["model_shot_ids"], ["shot_01", "A"])
        self.assertEqual(
            chapters["seg_v"]["model_shot_ids"],
            ["A__2", "shot_04", "A__2__2", "A__3"],
        )

        for shot in shots:
            for idx, sub in enumerate(shot["subclips"], start=1):
                self.assertTrue(sub["subclip_id"].startswith(f'{shot["shot_id"]}::subclip::'))
                self.assertEqual(sub["clip_id"], f'{shot["shot_id"]}::subclip::{idx}')

        # must not mutate original input
        self.assertEqual(state, before)

        # still JSON serializable with generated IDs
        json.dumps(payload, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
