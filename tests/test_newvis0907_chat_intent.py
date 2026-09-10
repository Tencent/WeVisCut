"""Conversational targeting tests for newvis0907 chat editing.

Run: python3 tests/test_newvis0907_chat_intent.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json  # noqa: E402

from studio.newvis0907.chat_planner import (  # noqa: E402
    build_model_plan_prompt,
    plan_edit,
)
from studio.newvis0907.locator import resolve_targets  # noqa: E402


def make_state() -> dict:
    return {
        "segments": [
            {"segment_id": "seg_a", "title": "物价上涨与购买力流逝"},
            {"segment_id": "seg_b", "title": "跑步机上的原地踏步"},
        ],
        "shot_plan": [
            {
                "shot_id": "S01",
                "segment_id": "seg_a",
                "duration_sec": 5,
                "vocal_script": "十年前那碗面只要十块钱。",
                "subtitle_text": "十年前那碗面只要十块钱。",
            },
            {
                "shot_id": "S02",
                "segment_id": "seg_a",
                "duration_sec": 5,
                "vocal_script": "现在同样的面要二十块。",
                "subtitle_text": "现在同样的面要二十块。",
            },
            {
                "shot_id": "S03",
                "segment_id": "seg_b",
                "duration_sec": 6,
                "vocal_script": "货币总量增加，购买力自然下降。",
                "subtitle_text": "货币总量增加，购买力自然下降。",
            },
            {
                "shot_id": "S04",
                "segment_id": "seg_b",
                "duration_sec": 6,
                "vocal_script": "收银台里的钞票变厚了。",
                "subtitle_text": "收银台里的钞票变厚了。",
            },
        ],
        "shot_assets": {},
    }


class ChapterOrdinalTargeting(unittest.TestCase):
    def test_chinese_ordinal_inside_named_chapter_resolves_uniquely(self) -> None:
        result = resolve_targets(
            make_state(), "跑步机上的原地踏步这一章节第一个镜头换一个大海的视频")
        self.assertEqual(result["status"], "unique", result)
        self.assertEqual(result["targets"][0]["shot_id"], "S03", result)

    def test_chapter_scope_matches_zhang_jie_wording(self) -> None:
        result = resolve_targets(make_state(), "跑步机上的原地踏步这一章节第二个镜头")
        self.assertEqual(result["status"], "unique", result)
        self.assertEqual(result["targets"][0]["shot_id"], "S04", result)


class ShotFootageReplacement(unittest.TestCase):
    def test_replacing_shot_footage_by_ordinal_produces_edit_op(self) -> None:
        payload = plan_edit(make_state(), "我希望你把第3个镜头视频素材换成摇滚乐队")
        self.assertEqual(payload["intent"], "edit", payload)
        self.assertEqual(len(payload["ops"]), 1, payload)
        op = payload["ops"][0]
        self.assertEqual(op["op"], "replace_footage", payload)
        self.assertEqual(op["shot_id"], "S03", payload)
        self.assertEqual(op["query"], "摇滚乐队", payload)

    def test_chapter_scoped_footage_replacement_targets_first_shot(self) -> None:
        payload = plan_edit(
            make_state(), "跑步机上的原地踏步这一章节第一个镜头换一个大海的视频")
        self.assertEqual(payload["intent"], "edit", payload)
        op = payload["ops"][0]
        self.assertEqual(op["op"], "replace_footage", payload)
        self.assertEqual(op["shot_id"], "S03", payload)
        self.assertEqual(op["query"], "大海", payload)

    def test_model_plan_can_resolve_more_conversational_edit(self) -> None:
        payload = plan_edit(
            make_state(),
            "把后面那段开头画面换成大海",
            model_plan={
                "reply": "好的，我来处理：\n· 镜头 03：换成「大海」的画面素材",
                "ops": [{"op": "replace_footage", "shot_id": "S03", "query": "大海"}],
            },
        )
        self.assertEqual(payload["intent"], "edit", payload)
        self.assertEqual(payload["ops"][0]["shot_id"], "S03", payload)
        self.assertIn("大海", payload["reply"], payload)

    def test_model_plan_rejects_unknown_target(self) -> None:
        payload = plan_edit(
            make_state(),
            "把后面那段开头画面换成大海",
            model_plan={
                "reply": "好的，我来处理。",
                "ops": [{"op": "replace_footage", "shot_id": "NOPE", "query": "大海"}],
            },
        )
        self.assertEqual(payload["intent"], "clarify", payload)
        self.assertEqual(payload["ops"], [], payload)


class ReplyQuality(unittest.TestCase):
    def test_edit_reply_states_target_and_change(self) -> None:
        payload = plan_edit(make_state(), "我希望你把第3个镜头视频素材换成摇滚乐队")
        reply = payload["reply"]
        self.assertIn("03", reply, payload)
        self.assertIn("摇滚乐队", reply, payload)

    def test_unresolved_reply_asks_a_concrete_question(self) -> None:
        payload = plan_edit(make_state(), "把那个蓝色的画面改一下")
        self.assertNotEqual(payload["intent"], "edit", payload)
        self.assertEqual(payload["ops"], [], payload)
        self.assertNotIn("没找到唯一目标", payload["reply"], payload)


class SmallTalkAndCapability(unittest.TestCase):
    def test_greeting_gets_a_greeting_not_a_candidate_dump(self) -> None:
        payload = plan_edit(make_state(), "你好")
        self.assertEqual(payload["intent"], "question", payload)
        self.assertEqual(payload["candidates"], [], payload)
        self.assertNotIn("镜头 01", payload["reply"], payload)

    def test_capability_question_lists_supported_edits(self) -> None:
        payload = plan_edit(make_state(), "可以改什么")
        self.assertEqual(payload["intent"], "question", payload)
        self.assertIn("素材", payload["reply"], payload)
        self.assertIn("时长", payload["reply"], payload)
        self.assertEqual(payload["ops"], [], payload)

    def test_confusion_followup_does_not_dump_candidates(self) -> None:
        payload = plan_edit(make_state(), "什么意思")
        self.assertEqual(payload["intent"], "question", payload)
        self.assertNotIn("没找到唯一目标", payload["reply"], payload)
        self.assertEqual(payload["ops"], [], payload)


class PromptContract(unittest.TestCase):
    """The model, not a regex, is what reads intent — so the prompt must carry
    the op catalog, the targeting rules and the conversation playbook."""

    def _prompt(self, message: str = "第3个镜头换成海边落日", **kwargs):
        return build_model_plan_prompt(make_state(), message, **kwargs)

    def test_system_prompt_declares_json_only_intent_contract(self) -> None:
        system, _ = self._prompt()
        for token in ('"intent"', '"reply"', '"ops"', "edit", "question", "clarify"):
            self.assertIn(token, system, token)

    def test_system_prompt_lists_every_supported_op_with_fields(self) -> None:
        system, _ = self._prompt()
        for op in (
            "replace_footage", "replace_subclip", "set_shot_duration",
            "set_subclip_timing", "set_narration", "set_subtitle", "set_bgm",
            "set_emphasis", "remove_animation", "delete_shot", "reorder_shots",
            "optimize_cut",
        ):
            self.assertIn(op, system, op)
        self.assertIn("shot_id", system)
        self.assertIn("subclip_id", system)
        self.assertIn("duration_sec", system)

    def test_system_prompt_teaches_targeting_and_forbids_fake_ids(self) -> None:
        system, _ = self._prompt()
        for hint in ("章节", "台词", "子镜头", "@镜头", "previous_targets", "素材"):
            self.assertIn(hint, system, hint)
        self.assertIn("S03", system)

    def test_system_prompt_covers_small_talk_and_reply_style(self) -> None:
        system, _ = self._prompt()
        for scene in ("你好", "取消", "谢谢", "什么意思", "可以干啥"):
            self.assertIn(scene, system, scene)
        self.assertIn("没找到唯一目标", system)
        self.assertIn("已修改成功", system)

    def test_user_payload_carries_structure_history_and_previous_targets(self) -> None:
        _, user = self._prompt(
            "那个也换成大海",
            previous_targets=[{"shot_id": "S02", "subclip_id": ""}],
            history=[
                {"role": "user", "content": "第2个镜头换成超市货架"},
                {"role": "assistant", "content": "好的，我来处理：· 镜头 02"},
            ],
        )
        payload = json.loads(user)
        self.assertEqual(payload["用户这句话"], "那个也换成大海")
        self.assertEqual(len(payload["最近对话"]), 2, payload)
        self.assertEqual(payload["上次改过的目标"][0]["shot_id"], "S02", payload)
        numbers = [row["display_number"] for row in payload["片子镜头"]]
        ids = [row["shot_id"] for row in payload["片子镜头"]]
        self.assertEqual(ids, ["S01", "S02", "S03", "S04"], payload)
        self.assertEqual(numbers[2], "03", payload)
        self.assertIn("跑步机上的原地踏步", json.dumps(payload["片子章节"], ensure_ascii=False))
        self.assertEqual(payload["片子镜头"][2]["chapter"], "跑步机上的原地踏步", payload)

    def test_prompt_never_leaks_a_success_claim(self) -> None:
        system, user = self._prompt()
        self.assertNotIn("修改成功了", system)
        self.assertNotIn("已生效", user)


class CasualPhrasing(unittest.TestCase):
    def test_capability_question_in_slang(self) -> None:
        for msg in ("你可以干啥", "你能干嘛", "你会啥", "你能帮我做什么"):
            payload = plan_edit(make_state(), msg)
            self.assertEqual(payload["intent"], "question", msg)
            self.assertIn("素材", payload["reply"], msg)
            self.assertEqual(payload["candidates"], [], msg)

    def test_bare_question_mark_offers_help(self) -> None:
        for msg in ("？", "?", "??"):
            payload = plan_edit(make_state(), msg)
            self.assertEqual(payload["intent"], "question", msg)
            self.assertEqual(payload["ops"], [], msg)

    def test_replace_material_with_wei_marker(self) -> None:
        payload = plan_edit(make_state(), "你帮我修改第3个素材为海边落日")
        self.assertEqual(payload["intent"], "edit", payload)
        op = payload["ops"][0]
        self.assertEqual(op["op"], "replace_footage", payload)
        self.assertEqual(op["shot_id"], "S03", payload)
        self.assertEqual(op["query"], "海边落日", payload)

    def test_material_word_counts_as_shot_ordinal(self) -> None:
        for msg, query in (
            ("第2个画面换成图书馆", "图书馆"),
            ("第2个视频换成图书馆", "图书馆"),
            ("把第2个素材改为图书馆", "图书馆"),
        ):
            payload = plan_edit(make_state(), msg)
            self.assertEqual(payload["intent"], "edit", msg)
            self.assertEqual(payload["ops"][0]["shot_id"], "S02", msg)
            self.assertEqual(payload["ops"][0]["query"], query, msg)


class MentionTargeting(unittest.TestCase):
    def test_at_shot_mention_resolves_uniquely(self) -> None:
        result = resolve_targets(make_state(), "@镜头03 换成海边落日")
        self.assertEqual(result["status"], "unique", result)
        self.assertEqual(result["targets"][0]["shot_id"], "S03", result)

    def test_at_shot_id_mention_resolves(self) -> None:
        result = resolve_targets(make_state(), "@S02 换成图书馆")
        self.assertEqual(result["status"], "unique", result)
        self.assertEqual(result["targets"][0]["shot_id"], "S02", result)

    def test_at_mention_drives_footage_replacement(self) -> None:
        payload = plan_edit(make_state(), "@镜头03 换成海边落日")
        self.assertEqual(payload["intent"], "edit", payload)
        op = payload["ops"][0]
        self.assertEqual(op["op"], "replace_footage", payload)
        self.assertEqual(op["shot_id"], "S03", payload)
        self.assertEqual(op["query"], "海边落日", payload)

    def test_at_mention_wins_over_other_ordinals_in_text(self) -> None:
        payload = plan_edit(make_state(), "@镜头04 我之前说的第1个镜头先别管，这个换成大海")
        self.assertEqual(payload["intent"], "edit", payload)
        self.assertEqual(payload["ops"][0]["shot_id"], "S04", payload)

    def test_at_chapter_mention_scopes_ordinal(self) -> None:
        payload = plan_edit(make_state(), "@章节02 第一个镜头换成大海")
        self.assertEqual(payload["intent"], "edit", payload)
        self.assertEqual(payload["ops"][0]["shot_id"], "S03", payload)


class ModelFirstPlanning(unittest.TestCase):
    def test_model_conversational_reply_allowed_without_ops(self) -> None:
        payload = plan_edit(
            make_state(),
            "你好，今天这支片子能怎么改",
            model_plan={
                "intent": "question",
                "reply": "可以换素材、改口播字幕、调时长或换配乐。例如「第 3 个镜头换成海边落日」。",
                "ops": [],
            },
        )
        self.assertEqual(payload["intent"], "question", payload)
        self.assertEqual(payload["ops"], [], payload)
        self.assertIn("海边落日", payload["reply"], payload)
        self.assertNotIn("没找到唯一目标", payload["reply"], payload)

    def test_model_plan_wins_when_rules_cannot_locate(self) -> None:
        payload = plan_edit(
            make_state(),
            "后面跑步机那章开头换成大海",
            model_plan={
                "intent": "edit",
                "reply": "好的，我来处理：\n· 镜头 03：换成「大海」的画面素材",
                "ops": [{"op": "replace_footage", "shot_id": "S03", "query": "大海"}],
            },
        )
        self.assertEqual(payload["intent"], "edit", payload)
        self.assertEqual(payload["ops"][0]["shot_id"], "S03", payload)
        self.assertEqual(payload["ops"][0]["query"], "大海", payload)

    def test_model_display_number_is_mapped_to_shot_id(self) -> None:
        payload = plan_edit(
            make_state(),
            "第三个镜头换成摇滚乐队",
            model_plan={
                "reply": "好的，我来处理：\n· 镜头 03：换成「摇滚乐队」的画面素材",
                "ops": [{"op": "replace_footage", "shot_id": "03", "query": "摇滚乐队"}],
            },
        )
        self.assertEqual(payload["intent"], "edit", payload)
        self.assertEqual(payload["ops"][0]["shot_id"], "S03", payload)

    def test_clear_rule_edit_rescues_empty_model_ops(self) -> None:
        payload = plan_edit(
            make_state(),
            "我希望你把第3个镜头视频素材换成摇滚乐队",
            model_plan={"intent": "question", "reply": "你好，需要我改哪个镜头？", "ops": []},
        )
        self.assertEqual(payload["intent"], "edit", payload)
        self.assertEqual(payload["ops"][0]["shot_id"], "S03", payload)
        self.assertEqual(payload["ops"][0]["query"], "摇滚乐队", payload)

    def test_followup_uses_previous_target_when_model_says_so(self) -> None:
        payload = plan_edit(
            make_state(),
            "那个也换成海边落日",
            previous_targets=[{"shot_id": "S02"}],
            model_plan={
                "reply": "好的，我来处理：\n· 镜头 02：换成「海边落日」的画面素材",
                "ops": [{"op": "replace_footage", "shot_id": "S02", "query": "海边落日"}],
            },
        )
        self.assertEqual(payload["intent"], "edit", payload)
        self.assertEqual(payload["ops"][0]["shot_id"], "S02", payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
