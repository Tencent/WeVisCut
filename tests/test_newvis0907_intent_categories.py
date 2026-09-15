"""The assistant must act on the *kind* of change the user asked for.

The regression: 「换一个更加醒目的动画装饰」was answered with a stock-footage
search for a clip literally named「更加醒目的动画装饰」. The model reads intent;
this layer only refuses an edit that belongs to a different category than the
one the user unmistakably named.

Run: python3 tests/test_newvis0907_intent_categories.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from test_newvis0907_conversation import make_state  # noqa: E402

from studio.newvis0907.chat_planner import plan_edit, reconcile_ops  # noqa: E402


def _ops(payload: dict) -> list[dict]:
    return list(payload.get("ops") or [])


class CategoryGuard(unittest.TestCase):
    #: The exact wrong plan from the bug report.
    FOOTAGE_FOR_ANIMATION = {
        "intent": "edit",
        "reply": "好的，我来处理：\n· 镜头 01：换成「更加醒目的动画装饰」的画面素材",
        "ops": [{"op": "replace_footage", "shot_id": "S01", "query": "更加醒目的动画装饰"}],
    }

    def test_an_animation_request_is_never_answered_with_a_footage_search(self) -> None:
        payload = plan_edit(
            make_state(),
            "@镜头01 第一个镜头的hyperframes动画你换一个更加醒目的动画装饰",
            model_plan=self.FOOTAGE_FOR_ANIMATION,
        )
        self.assertEqual(payload["intent"], "edit", payload)
        self.assertEqual([op["op"] for op in _ops(payload)], ["set_emphasis"], payload)

    def test_the_reply_describes_what_will_actually_happen(self) -> None:
        payload = plan_edit(
            make_state(),
            "@镜头01 第一个镜头的hyperframes动画你换一个更加醒目的动画装饰",
            model_plan=self.FOOTAGE_FOR_ANIMATION,
        )
        self.assertNotIn("画面素材", payload["reply"], payload)

    def test_a_footage_query_naming_an_animation_is_rejected_on_its_own(self) -> None:
        kept, dropped = reconcile_ops("随便改改", [
            {"op": "replace_footage", "shot_id": "S01", "query": "醒目的动画装饰"},
        ])
        self.assertEqual(kept, [])
        self.assertEqual(dropped, ["replace_footage"])

    def test_a_genuine_footage_request_still_goes_through(self) -> None:
        kept, dropped = reconcile_ops("@镜头02 画面换成海边落日", [
            {"op": "replace_footage", "shot_id": "S02", "query": "海边落日"},
        ])
        self.assertEqual(dropped, [])
        self.assertEqual(kept[0]["op"], "replace_footage")

    def test_an_unqualified_request_is_not_second_guessed(self) -> None:
        # No category word at all: the model is on its own, and that is fine.
        kept, dropped = reconcile_ops("@镜头02 换成海边落日", [
            {"op": "replace_footage", "shot_id": "S02", "query": "海边落日"},
        ])
        self.assertEqual(dropped, [])
        self.assertEqual(len(kept), 1)

    def test_bgm_and_timing_requests_keep_their_own_ops(self) -> None:
        kept, _ = reconcile_ops("配乐换成轻快电子乐", [{"op": "set_bgm", "query": "轻快电子乐"}])
        self.assertEqual(kept[0]["op"], "set_bgm")
        kept, _ = reconcile_ops("@镜头02 时长调到8秒",
                                [{"op": "set_shot_duration", "shot_id": "S02", "duration_sec": 8}])
        self.assertEqual(kept[0]["op"], "set_shot_duration")


class SubtitleEditsKeepAudioInSync(unittest.TestCase):
    def test_changing_the_caption_also_re_records_the_line(self) -> None:
        payload = plan_edit(make_state(), "@镜头02 把字幕改成今晚早点睡")
        self.assertEqual([op["op"] for op in _ops(payload)], ["set_narration"], payload)
        self.assertEqual(_ops(payload)[0]["text"], "今晚早点睡", payload)

    def test_display_only_is_honoured_when_asked_for(self) -> None:
        payload = plan_edit(
            make_state(), "@镜头02 只改字幕，改成今晚早点睡，不重新配音")
        self.assertEqual([op["op"] for op in _ops(payload)], ["set_subtitle"], payload)

    def test_the_how_to_apply_clause_does_not_land_on_screen(self) -> None:
        payload = plan_edit(
            make_state(), "@镜头02 只改字幕，改成今晚早点睡，不重新配音")
        self.assertEqual(_ops(payload)[0]["text"], "今晚早点睡", payload)

    def test_a_model_caption_op_is_upgraded_too(self) -> None:
        kept, _ = reconcile_ops("@镜头02 把字幕改成今晚早点睡", [
            {"op": "set_subtitle", "shot_id": "S02", "text": "今晚早点睡"},
        ])
        self.assertEqual(kept[0]["op"], "set_narration", kept)


class AnimationSwapPicksADifferentTemplate(unittest.TestCase):
    """「换一个动画」has to look different, or nothing visibly happened."""

    def test_select_template_avoids_the_current_structure(self) -> None:
        from studio.newvis0907 import animations

        registry = {
            "structures": ["key_takeaway", "key_takeaway_center", "thesis_statement"],
            "motions": ["scale_pop_hold"],
            "visuals": ["dark_tech"],
        }
        first = animations.select_template({"type": "key_point"}, registry, current={})
        second = animations.select_template(
            {"type": "key_point"}, registry,
            current={"structure_id": first["structure_id"]},
        )
        self.assertTrue(first["ok"] and second["ok"], (first, second))
        self.assertNotEqual(first["structure_id"], second["structure_id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
