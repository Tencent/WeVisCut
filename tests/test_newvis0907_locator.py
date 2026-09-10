"""Offline tests for the newvis0907 locator and rule-based planner.

Run: python3 tests/test_newvis0907_locator.py
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studio.newvis0907.chat_planner import (  # noqa: E402
    SUPPORTED_OPS,
    plan_edit,
    validate_model_plan,
)
from studio.newvis0907.locator import resolve_targets  # noqa: E402


def make_state() -> dict:
    return {
        "segments": [
            {"segment_id": "seg_intro", "title": "序章"},
            {"segment_id": "seg_future", "title": "软硬结合的未来图景"},
            {"segment_id": "seg_outro", "title": "结尾复盘"},
        ],
        "shot_plan": [
            {
                "shot_id": "S01",
                "segment_id": "seg_intro",
                "duration_sec": 4,
                "vocal_script": "先看一个简单背景。",
                "subtitle_text": "先看一个简单背景。",
            },
            {
                "shot_id": "S02",
                "segment_id": "seg_intro",
                "duration_sec": 4,
                "vocal_script": "这里先铺垫一个常见误区。",
                "subtitle_text": "这里先铺垫一个常见误区。",
            },
            {
                "shot_id": "S03",
                "segment_id": "seg_future",
                "duration_sec": 6,
                "vocal_script": "真正的效率不是更快地点击，而是让工具主动理解你。",
                "subtitle_text": "真正的效率不是更快地点击，而是让工具主动理解你。",
                "semantic_animations": {
                    "primary": {
                        "type": "spotlight",
                        "phrase": "真正的效率",
                    }
                },
            },
            {
                "shot_id": "S04",
                "segment_id": "seg_future",
                "duration_sec": 5,
                "vocal_script": "关键不是追热点，而是怎样让 AI 成为自己的工具。",
                "subtitle_text": "关键不是追热点，而是怎样让 AI 成为自己的工具。",
            },
            {
                "shot_id": "S05",
                "segment_id": "seg_future",
                "duration_sec": 4,
                "vocal_script": "让硬件和软件协同，才能把反馈回路做短。",
                "subtitle_text": "让硬件和软件协同，才能把反馈回路做短。",
            },
            {
                "shot_id": "S06",
                "segment_id": "seg_outro",
                "duration_sec": 5,
                "vocal_script": "怎样让 AI 成为自己的工具，还需要长期训练团队。",
                "subtitle_text": "怎样让 AI 成为自己的工具，还需要长期训练团队。",
            },
            {
                "shot_id": "S07",
                "segment_id": "seg_outro",
                "duration_sec": 6,
                "vocal_script": "最后看素材调度和节奏。",
                "subtitle_text": "最后看素材调度和节奏。",
            },
            {
                "shot_id": "S08",
                "segment_id": "seg_outro",
                "duration_sec": 4,
                "vocal_script": "收尾总结。",
                "subtitle_text": "收尾总结。",
            },
        ],
        "shot_assets": {
            "S03": {
                "clip_meta": {
                    "visual_playlist": {
                        "clips": [
                            {
                                "clip_id": "future-1",
                                "path": "/future-1.mp4",
                                "used_duration_sec": 6,
                            }
                        ]
                    }
                },
                "decoration": {
                    "slots": {
                        "headline": "真正的效率",
                        "subline": "工具主动理解你",
                    }
                },
            },
            "S04": {
                "clip_meta": {
                    "visual_playlist": {
                        "clips": [
                            {
                                "clip_id": "future-2",
                                "path": "/future-2.mp4",
                                "used_duration_sec": 5,
                            }
                        ]
                    }
                }
            },
            "S06": {
                "clip_meta": {
                    "visual_playlist": {
                        "clips": [
                            {
                                "clip_id": "outro-1",
                                "path": "/outro-1.mp4",
                                "used_duration_sec": 5,
                            }
                        ]
                    }
                }
            },
            "S07": {
                "clip_meta": {
                    "visual_playlist": {
                        "clips": [
                            {
                                "subclip_id": "S07::subclip::1",
                                "clip_id": "tail-1",
                                "path": "/tail-1.mp4",
                                "used_duration_sec": 2,
                            },
                            {
                                "subclip_id": "S07::subclip::2",
                                "clip_id": "tail-2",
                                "path": "/tail-2.mp4",
                                "used_duration_sec": 2,
                            },
                            {
                                "subclip_id": "S07::subclip::3",
                                "clip_id": "tail-3",
                                "path": "/tail-3.mp4",
                                "used_duration_sec": 2,
                            },
                        ]
                    }
                }
            },
        },
    }


class LocatorAndPlannerTest(unittest.TestCase):
    def test_supported_ops_include_legacy_and_0907_additions(self) -> None:
        for name in [
            "set_narration",
            "set_subtitle",
            "set_spotlight_slots",
            "swap_template",
            "replace_footage",
            "reorder_shots",
            "delete_shot",
            "set_bgm",
            "remove_animation",
            "set_emphasis",
            "replace_subclip",
            "set_subclip_timing",
            "set_shot_duration",
            "optimize_cut",
        ]:
            self.assertIn(name, SUPPORTED_OPS)

    def test_original_case_yields_two_ops_with_real_ids(self) -> None:
        state = make_state()
        before = copy.deepcopy(state)
        payload = plan_edit(
            state,
            "《软硬结合的未来图景》这章的开头动画“真正的效率……”可以删掉，然后做一个动画强调“怎样让 AI 成为自己的工具”",
        )
        self.assertEqual(payload["intent"], "edit")
        self.assertEqual([op["op"] for op in payload["ops"]], ["remove_animation", "set_emphasis"])
        self.assertEqual(payload["ops"][0]["shot_id"], "S03")
        self.assertEqual(payload["ops"][0]["phrase"], "真正的效率")
        self.assertEqual(payload["ops"][1]["shot_id"], "S04")
        self.assertEqual(payload["ops"][1]["phrase"], "怎样让 AI 成为自己的工具")
        self.assertEqual(payload["ops"][1]["emphasis_type"], "key_point")
        self.assertEqual(payload["needs_confirm"], [])
        self.assertNotIn("S08", [op.get("shot_id") for op in payload["ops"]])
        self.assertEqual(state, before)

    def test_scoped_phrase_beats_same_phrase_in_other_chapter(self) -> None:
        result = resolve_targets(
            make_state(),
            "《软硬结合的未来图景》这章强调“怎样让 AI 成为自己的工具”",
        )
        self.assertEqual(result["status"], "unique")
        self.assertEqual(result["targets"][0]["shot_id"], "S04")
        self.assertEqual(result["scope"]["title"], "软硬结合的未来图景")

    def test_chapter_start_returns_earliest_shot_in_scope(self) -> None:
        result = resolve_targets(make_state(), "《软硬结合的未来图景》这章开头")
        self.assertEqual(result["status"], "unique")
        self.assertEqual(result["targets"][0]["shot_id"], "S03")
        self.assertEqual(result["targets"][0]["reason"], "chapter_start")

    def test_locator_handles_ordinal_time_and_subclip(self) -> None:
        by_chapter = resolve_targets(make_state(), "第2章开头")
        self.assertEqual(by_chapter["targets"][0]["shot_id"], "S03")

        by_shot = resolve_targets(make_state(), "第4个镜头")
        self.assertEqual(by_shot["targets"][0]["shot_id"], "S04")

        by_time = resolve_targets(make_state(), "15秒左右")
        self.assertEqual(by_time["targets"][0]["shot_id"], "S04")

        by_subclip = resolve_targets(make_state(), "素材子镜头7.2")
        self.assertEqual(by_subclip["status"], "unique")
        self.assertEqual(by_subclip["targets"][0]["shot_id"], "S07")
        self.assertEqual(by_subclip["targets"][0]["subclip_id"], "S07::subclip::2")
        self.assertEqual(by_subclip["targets"][0]["subclip_display_number"], "7.2")

    def test_all_keyword_returns_multiple_but_plain_query_needs_clarify(self) -> None:
        ambiguous = resolve_targets(make_state(), "强调“怎样让 AI 成为自己的工具”")
        self.assertEqual(ambiguous["status"], "multiple")
        self.assertFalse(ambiguous["targets"])
        self.assertEqual([c["shot_id"] for c in ambiguous["candidates"][:2]], ["S04", "S06"])

        all_hits = resolve_targets(make_state(), "把所有“怎样让 AI 成为自己的工具”都强调一下")
        self.assertEqual(all_hits["status"], "multiple")
        self.assertEqual([t["shot_id"] for t in all_hits["targets"]], ["S04", "S06"])

    def test_none_returns_nearest_candidates(self) -> None:
        result = resolve_targets(make_state(), "强调“工具箱革命”")
        self.assertEqual(result["status"], "none")
        self.assertEqual(result["targets"], [])
        self.assertGreaterEqual(len(result["candidates"]), 1)

    def test_previous_target_pronoun_is_supported(self) -> None:
        result = resolve_targets(
            make_state(),
            "刚才那个",
            previous_targets=[{"shot_id": "S07"}],
        )
        self.assertEqual(result["status"], "unique")
        self.assertEqual(result["targets"][0]["shot_id"], "S07")
        self.assertEqual(result["targets"][0]["reason"], "previous_target")

    def test_validate_model_plan_rejects_fake_ids_unknown_ops_and_extra_fields(self) -> None:
        good = validate_model_plan(
            """
            {
              "intent": "edit",
              "reply": "准备这样改",
              "ops": [
                {
                  "op": "set_emphasis",
                  "shot_id": "S04",
                  "phrase": "怎样让 AI 成为自己的工具",
                  "emphasis_type": "key_point"
                }
              ]
            }
            """,
            allowed_target_ids={"S04"},
        )
        self.assertTrue(good["ok"], good)
        self.assertEqual(good["ops"][0]["shot_id"], "S04")

        fake = validate_model_plan(
            {"ops": [{"op": "set_emphasis", "shot_id": "FAKE", "phrase": "x", "emphasis_type": "key_point"}]},
            allowed_target_ids={"S04"},
        )
        self.assertFalse(fake["ok"])

        unknown = validate_model_plan(
            {"ops": [{"op": "hack_everything", "shot_id": "S04"}]},
            allowed_target_ids={"S04"},
        )
        self.assertFalse(unknown["ok"])

        extra = validate_model_plan(
            {
                "ops": [
                    {
                        "op": "set_emphasis",
                        "shot_id": "S04",
                        "phrase": "怎样让 AI 成为自己的工具",
                        "emphasis_type": "key_point",
                        "status": "success",
                    }
                ]
            },
            allowed_target_ids={"S04"},
        )
        self.assertFalse(extra["ok"])

    def test_remove_animation_accepts_optional_anchor_id(self) -> None:
        # anchor_id is an allowed optional field; the shot_id must still be a
        # known target and unknown fields are still rejected.
        ok = validate_model_plan(
            {"ops": [{"op": "remove_animation", "shot_id": "S04", "anchor_id": "S04::anchor::1"}]},
            allowed_target_ids={"S04"},
        )
        self.assertTrue(ok["ok"], ok)
        self.assertEqual(ok["ops"][0]["anchor_id"], "S04::anchor::1")

        both = validate_model_plan(
            {"ops": [{"op": "remove_animation", "shot_id": "S04",
                      "phrase": "真正的效率", "anchor_id": "S04::anchor::1"}]},
            allowed_target_ids={"S04"},
        )
        self.assertTrue(both["ok"], both)

        unknown_target = validate_model_plan(
            {"ops": [{"op": "remove_animation", "shot_id": "FAKE", "anchor_id": "a"}]},
            allowed_target_ids={"S04"},
        )
        self.assertFalse(unknown_target["ok"])

        extra_field = validate_model_plan(
            {"ops": [{"op": "remove_animation", "shot_id": "S04", "bogus": "x"}]},
            allowed_target_ids={"S04"},
        )
        self.assertFalse(extra_field["ok"])

    def test_plan_edit_handles_json_model_plan_and_rule_based_ops(self) -> None:
        state = make_state()

        replace_payload = plan_edit(
            state,
            "把素材子镜头7.2换成实验室里操作设备的近景",
            model_plan="""
            {
              "reply": "计划替换为实验室近景",
              "ops": [
                {
                  "op": "replace_subclip",
                  "subclip_id": "S07::subclip::2",
                  "query": "实验室里操作设备的近景"
                }
              ]
            }
            """,
        )
        self.assertEqual(replace_payload["intent"], "edit")
        self.assertEqual(replace_payload["ops"][0]["op"], "replace_subclip")
        self.assertEqual(replace_payload["ops"][0]["subclip_id"], "S07::subclip::2")

        timing_payload = plan_edit(state, "把素材子镜头7.2调到1.5秒")
        self.assertEqual(timing_payload["ops"][0]["op"], "set_subclip_timing")
        self.assertAlmostEqual(timing_payload["ops"][0]["duration_sec"], 1.5, places=4)

        duration_payload = plan_edit(state, "第4个镜头改到8秒")
        self.assertEqual(duration_payload["ops"][0]["op"], "set_shot_duration")
        self.assertEqual(duration_payload["needs_confirm"], ["set_shot_duration"])

        optimize_payload = plan_edit(state, "整体优化一下")
        self.assertEqual(optimize_payload["ops"][0]["op"], "optimize_cut")

    def test_at_subclip_mention_resolves_to_subclip(self) -> None:
        result = resolve_targets(make_state(), "@子镜头7.2 换成实验室近景")
        self.assertEqual(result["status"], "unique", result)
        self.assertEqual(result["targets"][0]["subclip_id"], "S07::subclip::2", result)

    def test_at_subclip_mention_plans_replace_subclip(self) -> None:
        payload = plan_edit(make_state(), "@子镜头7.2 换成实验室近景")
        self.assertEqual(payload["intent"], "edit", payload)
        self.assertEqual(payload["ops"][0]["op"], "replace_subclip", payload)
        self.assertEqual(payload["ops"][0]["subclip_id"], "S07::subclip::2", payload)
        self.assertEqual(payload["ops"][0]["query"], "实验室近景", payload)

    def test_model_plan_must_not_be_empty(self) -> None:
        blank = validate_model_plan({"reply": "", "ops": []}, allowed_target_ids={"S04"})
        self.assertFalse(blank["ok"])
        talk = validate_model_plan({"reply": "可以换素材或改口播。", "ops": []}, allowed_target_ids={"S04"})
        self.assertTrue(talk["ok"], talk)
        self.assertEqual(talk["ops"], [])
        self.assertEqual(talk["intent"], "question")

    def test_locator_and_planner_do_not_mutate_inputs(self) -> None:
        state = make_state()
        previous = [{"shot_id": "S07"}]
        before_state = copy.deepcopy(state)
        before_prev = copy.deepcopy(previous)

        _ = resolve_targets(state, "刚才那个", previous_targets=previous)
        _ = plan_edit(state, "《软硬结合的未来图景》这章开头")

        self.assertEqual(state, before_state)
        self.assertEqual(previous, before_prev)


if __name__ == "__main__":
    unittest.main(verbosity=2)
