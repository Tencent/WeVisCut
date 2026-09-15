"""Multi-turn editing conversation: memory, follow-ups and loop avoidance.

The regression these cover is a real transcript in which every turn came back
with「我还不能确定你指的是哪个镜头」— including the turns where the user had
already answered that exact question.

Run: python3 tests/test_newvis0907_conversation.py
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studio.newvis0907 import conversation  # noqa: E402
from studio.newvis0907.chat_planner import plan_edit  # noqa: E402


def make_state() -> dict:
    """A film shaped like the one in the bug report: 11 shots, 5 chapters."""
    scripts = [
        "睡眠是大脑清洁时间，你知道吗，人脑在深睡时其实正在做一次彻底的清理。",
        "白天累积的代谢废物，需要在夜里被冲走。",
        "睡眠是维护模式，大脑会在这段时间重新整理白天的记忆。",
        "研究发现，睡眠不足的人，认知功能会明显下降。",
        "睡眠是维护模式，研究发现只睡四小时的人反应速度慢了百分之四十。",
        "长期熬夜的代价，远比你以为的大。",
        "免疫系统同样依赖夜里的这段修复时间。",
        "情绪的稳定，也建立在充足的睡眠之上。",
        "今晚早睡一小时，请相信，就是给大脑最好的礼物。",
        "不需要一次改变太多，从今晚开始就好。",
        "愿你今晚睡个好觉。",
    ]
    chapters = ["清洁时间", "代谢废物", "维护模式", "熬夜代价", "今晚早睡"]
    shot_plan = []
    for index, script in enumerate(scripts, start=1):
        shot_plan.append({
            "shot_id": f"S{index:02d}",
            "segment_id": f"seg_{min((index - 1) // 3, 4)}",
            "duration_sec": 6,
            "vocal_script": script,
            "subtitle_text": script,
        })
    return {
        "segments": [
            {"segment_id": f"seg_{i}", "title": title}
            for i, title in enumerate(chapters)
        ],
        "shot_plan": shot_plan,
        "shot_assets": {},
    }


class MemoryFollowUps(unittest.TestCase):
    """A short reply is an answer to the last question, not a new request."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _memory(self) -> conversation.Memory:
        return conversation.Memory.load(self.dir)

    def test_yes_answers_a_single_candidate_question(self) -> None:
        memory = self._memory()
        memory.pending = conversation.Pending(
            kind="target", instruction="改一下动画",
            candidates=[{"shot_id": "S03", "display_number": "03"}],
            turn=memory.turn,
        )
        follow = memory.interpret("是的")
        self.assertIn("@镜头03", follow.message, follow.message)
        self.assertIn("动画", follow.message, follow.message)

    def test_yes_after_an_edit_never_restarts_the_shot_search(self) -> None:
        memory = self._memory()
        turn = memory.interpret("@镜头03 这个镜头改一下动画")
        payload = plan_edit(make_state(), turn.message, memory=memory)
        memory.record(turn, payload)

        follow = memory.interpret("是的")
        reply = plan_edit(make_state(), follow.message, memory=memory)["reply"]
        self.assertNotIn("哪个镜头", reply, reply)
        self.assertIn("镜头 03", reply, reply)

    def test_picking_by_ordinal_binds_to_the_listed_candidate(self) -> None:
        memory = self._memory()
        memory.pending = conversation.Pending(
            kind="target",
            instruction="换成海边落日",
            candidates=[
                {"shot_id": "S01", "display_number": "01", "chapter_title": "清洁时间"},
                {"shot_id": "S05", "display_number": "05", "chapter_title": "维护模式"},
            ],
            turn=memory.turn,
        )
        turn = memory.interpret("第2个")
        self.assertIn("@镜头05", turn.message, turn.message)
        self.assertIn("换成海边落日", turn.message, turn.message)

    def test_pasting_the_option_line_back_selects_that_shot(self) -> None:
        memory = self._memory()
        candidate = {"shot_id": "S05", "display_number": "05",
                     "chapter_title": "维护模式", "matched_text": "研究发现"}
        memory.pending = conversation.Pending(
            kind="target", instruction="加一个动画",
            candidates=[{"shot_id": "S01", "display_number": "01"}, candidate],
            turn=memory.turn,
        )
        turn = memory.interpret("· " + conversation.candidate_label(candidate))
        self.assertIn("@镜头05", turn.message, turn.message)

    def test_a_restated_instruction_replaces_the_pending_one(self) -> None:
        memory = self._memory()
        memory.pending = conversation.Pending(
            kind="target", instruction="加一个动画",
            candidates=[{"shot_id": "S05", "display_number": "05"}],
            turn=memory.turn,
        )
        turn = memory.interpret("@镜头05 换成海边落日")
        self.assertIn("换成海边落日", turn.message, turn.message)
        self.assertNotIn("加一个动画", turn.message, turn.message)

    def test_a_bare_instruction_reuses_the_locked_shot(self) -> None:
        memory = self._memory()
        memory.pending = conversation.Pending(
            kind="action", instruction="改一下动画",
            target={"shot_id": "S03", "display_number": "03"},
            turn=memory.turn,
        )
        turn = memory.interpret("换成海边落日")
        self.assertIn("@镜头03", turn.message, turn.message)
        self.assertIn("换成海边落日", turn.message, turn.message)

    def test_naming_another_shot_overrides_the_locked_one(self) -> None:
        memory = self._memory()
        memory.pending = conversation.Pending(
            kind="action", instruction="改一下动画",
            target={"shot_id": "S03", "display_number": "03"},
            turn=memory.turn,
        )
        turn = memory.interpret("@镜头07 换成海边落日")
        self.assertNotIn("@镜头03", turn.message, turn.message)

    def test_memory_survives_a_reload(self) -> None:
        memory = self._memory()
        memory.interpret("@镜头03 改一下动画")
        memory.focus = {"shot_id": "S03", "display_number": "03"}
        memory.pending = conversation.Pending(
            kind="action", instruction="改一下动画",
            target=memory.focus, turn=memory.turn)
        memory.save()

        reloaded = self._memory()
        self.assertIsNotNone(reloaded.pending)
        self.assertEqual(reloaded.pending.target["shot_id"], "S03")


class AnimationRequests(unittest.TestCase):
    """「加个动画」names a shot but no phrase — that is enough to act on."""

    def test_mentioned_shot_with_vague_animation_request_produces_an_op(self) -> None:
        payload = plan_edit(make_state(), "@镜头03 这个镜头改一下动画")
        self.assertEqual(payload["intent"], "edit", payload)
        self.assertEqual(payload["ops"][0]["op"], "set_emphasis", payload)
        self.assertEqual(payload["ops"][0]["shot_id"], "S03", payload)

    def test_the_auto_chosen_phrase_comes_from_that_shots_narration(self) -> None:
        payload = plan_edit(make_state(), "@镜头03 加一个hyperframes动画")
        phrase = payload["ops"][0]["phrase"]
        self.assertIn(phrase, make_state()["shot_plan"][2]["vocal_script"], payload)

    def test_removing_animation_clears_the_shot(self) -> None:
        payload = plan_edit(make_state(), "@镜头03 把动画去掉")
        self.assertEqual(payload["intent"], "edit", payload)
        self.assertEqual(payload["ops"][0]["op"], "remove_animation", payload)


class NeverAskTheSameQuestionTwice(unittest.TestCase):
    def test_a_located_shot_is_never_asked_about_again(self) -> None:
        # A mention resolves the shot; anything unclear afterwards must be about
        # the *change*, never about which shot.
        payload = plan_edit(make_state(), "@镜头03 弄好看一点")
        if payload["intent"] == "clarify":
            self.assertNotIn("哪个镜头", payload["reply"], payload)
            self.assertEqual(payload["located"]["shot_id"], "S03", payload)

    def test_the_third_ask_commits_instead_of_asking_again(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = conversation.Memory.load(Path(tmp))
            memory.pending = conversation.Pending(
                kind="target",
                instruction="换成海边落日",
                candidates=[{"shot_id": "S05", "display_number": "05",
                             "chapter_title": "维护模式"}],
                asked=conversation.MAX_CLARIFY_STREAK,
                turn=memory.turn,
            )
            turn = memory.interpret("那个啊")
            payload = plan_edit(make_state(), turn.message, memory=memory)
            self.assertEqual(payload["intent"], "edit", payload)
            self.assertEqual(payload["ops"][0]["shot_id"], "S05", payload)
            self.assertIn("如果不是这个镜头", payload["reply"], payload)


class ModelCannotReopenTheLoop(unittest.TestCase):
    """Production runs with the LoRA loaded, so its replies need the same guard.

    The model has no idea how many times it has already asked a question, so a
    ``clarify`` it produces must go through the same anti-loop path as a
    rule-based one rather than being echoed to the user.
    """

    WHICH_SHOT = {
        "intent": "clarify",
        "reply": "我还不能确定你指的是哪个镜头，最接近的是这几个：\n· 镜头 01\n· 镜头 05",
        "ops": [],
    }

    def test_a_model_clarify_never_overrides_a_resolved_target(self) -> None:
        payload = plan_edit(make_state(), "@镜头03 这个镜头改一下动画",
                            model_plan=self.WHICH_SHOT)
        self.assertEqual(payload["intent"], "edit", payload)
        self.assertNotIn("哪个镜头", payload["reply"], payload)

    def test_a_model_clarify_is_replaced_not_echoed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = conversation.Memory.load(Path(tmp))
            memory.focus = {"shot_id": "S03", "display_number": "03"}
            payload = plan_edit(make_state(), "是的",
                                model_plan=self.WHICH_SHOT, memory=memory)
            self.assertNotIn("哪个镜头", payload["reply"], payload)

    def test_a_model_question_still_reaches_the_user(self) -> None:
        payload = plan_edit(make_state(), "你能做什么", model_plan={
            "intent": "question", "reply": "我可以换素材、调时长、改字幕。", "ops": []})
        self.assertEqual(payload["intent"], "question", payload)
        self.assertIn("换素材", payload["reply"], payload)

    def test_the_whole_transcript_survives_a_uselessly_clarifying_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = conversation.Memory.load(Path(tmp))
            state = make_state()
            replies = []
            for message in ("@镜头03 这个镜头改一下动画", "是的",
                            "@镜头03 加一个hyperframes动画",
                            "· 镜头 03 · 睡眠是维护模式 · @子镜头3.1"):
                turn = memory.interpret(message)
                payload = plan_edit(state, turn.message,
                                    model_plan=self.WHICH_SHOT, memory=memory)
                memory.record(turn, payload)
                replies.append(payload)
            self.assertEqual(
                [item["reply"] for item in replies if "哪个镜头" in item["reply"]], [])
            self.assertTrue(any(item["intent"] == "edit" for item in replies), replies)


class PendingInstructionSurvivesRetargeting(unittest.TestCase):
    def test_pasting_a_candidate_line_applies_the_waiting_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = conversation.Memory.load(Path(tmp))
            memory.pending = conversation.Pending(
                kind="action", instruction="加一个hyperframes动画",
                target={"shot_id": "S01", "display_number": "01"},
                turn=memory.turn,
            )
            turn = memory.interpret("· 镜头 03 · 睡眠是维护模式 · @镜头03")
            self.assertIn("加一个hyperframes动画", turn.message, turn.message)
            payload = plan_edit(make_state(), turn.message, memory=memory)
            self.assertEqual(payload["intent"], "edit", payload)
            self.assertEqual(payload["ops"][0]["shot_id"], "S03", payload)


class ContextCompaction(unittest.TestCase):
    def test_old_episodes_fold_into_a_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = conversation.Memory.load(Path(tmp))
            for index in range(conversation.EPISODE_WINDOW + 5):
                turn = memory.interpret(f"第{index + 1}个镜头换成海边落日")
                memory.record(turn, {
                    "intent": "edit",
                    "ops": [{"op": "replace_footage", "shot_id": "S01"}],
                    "targets": ["S01"],
                })
            self.assertLessEqual(len(memory.episodes), conversation.EPISODE_WINDOW)
            self.assertTrue(memory.summary, "older turns must survive as a summary")

    def test_reset_keeps_preferences_but_drops_the_thread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = conversation.Memory.load(Path(tmp))
            memory.preferences["animation_engine"] = "hyperframes"
            memory.pending = conversation.Pending(kind="target", instruction="x")
            memory.reset()
            self.assertIsNone(memory.pending)
            self.assertEqual(memory.preferences["animation_engine"], "hyperframes")


class ReportedTranscript(unittest.TestCase):
    """The exact exchange from the bug report must not loop."""

    def test_four_turns_reach_an_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = conversation.Memory.load(Path(tmp))
            state = make_state()
            replies = []
            for message in ("@镜头03 这个镜头改一下动画", "是的",
                            "@镜头03 加一个hyperframes动画"):
                turn = memory.interpret(message)
                payload = plan_edit(state, turn.message, memory=memory)
                memory.record(turn, payload)
                replies.append(payload)

            self.assertTrue(
                any(item["intent"] == "edit" for item in replies),
                [item["reply"] for item in replies],
            )
            asked_which_shot = [
                item["reply"] for item in replies if "哪个镜头" in item["reply"]
            ]
            self.assertEqual(asked_which_shot, [], asked_which_shot)


class ChatEndpointCarriesContext(unittest.TestCase):
    """Memory has to survive between HTTP requests, not just within a call."""

    def _client(self, tmp: Path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from studio import project as project_mod
        from studio.newvis0907 import router as router_mod

        state = make_state()
        state["project_id"] = "proj_convo"
        state["input_text"] = "demo"
        state["final_video_path"] = ""
        (tmp / "final_state.json").write_text(
            __import__("json").dumps(state, ensure_ascii=False), encoding="utf-8")
        project = project_mod.load(tmp)
        router_mod.configure_router(router_mod.RouterDeps(
            load_project=lambda d: project,
            project_payload=lambda p: {"project_id": p.project_id,
                                       "project_dir": str(p.project_dir)},
        ))
        app = FastAPI()
        app.include_router(router_mod.router)
        return TestClient(app), str(tmp)

    def test_an_answer_to_a_clarification_is_not_re_asked(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, pdir = self._client(Path(d))
            first = client.post("/api/0907/project/chat",
                                json={"project_dir": pdir, "message": "那个画面不好看"}).json()
            self.assertEqual(first["intent"], "clarify", first)
            self.assertTrue(first["choices"], first)

            token = first["choices"][0]["token"]
            second = client.post("/api/0907/project/chat",
                                 json={"project_dir": pdir, "message": token}).json()
            self.assertNotIn("哪个镜头", second["reply"], second)

    def test_clarifications_offer_clickable_choices(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, pdir = self._client(Path(d))
            body = client.post("/api/0907/project/chat",
                               json={"project_dir": pdir, "message": "那个画面不好看"}).json()
            for choice in body["choices"]:
                self.assertTrue(choice["token"].startswith("@"), choice)
                self.assertTrue(choice["label"], choice)

    def test_a_mentioned_shot_edits_straight_away(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, pdir = self._client(Path(d))
            body = client.post(
                "/api/0907/project/chat",
                json={"project_dir": pdir, "message": "@镜头03 这个镜头改一下动画"}).json()
            self.assertEqual(body["intent"], "edit", body)
            self.assertEqual(body["ops"][0]["shot_id"], "S03", body)

    def test_reset_clears_the_pending_question(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, pdir = self._client(Path(d))
            client.post("/api/0907/project/chat",
                        json={"project_dir": pdir, "message": "那个画面不好看"})
            reset = client.post("/api/0907/project/chat/reset",
                                json={"project_dir": pdir})
            self.assertEqual(reset.status_code, 200, reset.text)
            memory = conversation.Memory.load(Path(d))
            self.assertIsNone(memory.pending)


if __name__ == "__main__":
    unittest.main(verbosity=2)
