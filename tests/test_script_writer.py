"""Offline tests for the brief → 口播稿 path (no GPU, no model).

Covers the two pieces that decide whether mode 2 produces a usable script:
length resolution from a free-text brief, and stripping the scaffolding models
wrap around narration (headings, 分镜 labels, bullets, fences, quotes) — any of
which would otherwise be read aloud by TTS and burned into the subtitles.

Run:  python3 tests/test_script_writer.py
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from studio import script_writer as sw  # noqa: E402


class ParseBriefTest(unittest.TestCase):
    def test_seconds(self):
        t = sw.parse_brief("剪一支 60 秒的城市清晨短片")
        self.assertEqual(t["duration_sec"], 60)
        self.assertTrue(t["explicit_duration"])

    def test_minutes(self):
        self.assertEqual(sw.parse_brief("做一个两分钟的产品回顾")["duration_sec"], 120)
        self.assertEqual(sw.parse_brief("三十秒的公益广告")["duration_sec"], 30)

    def test_mixed_and_half_minutes(self):
        self.assertEqual(sw.parse_brief("1分30秒的纪录片开场")["duration_sec"], 90)
        self.assertEqual(sw.parse_brief("一分半的旅行 vlog")["duration_sec"], 90)
        self.assertEqual(sw.parse_brief("半分钟的快节奏预告")["duration_sec"], 30)

    def test_english(self):
        self.assertEqual(sw.parse_brief("a 45 second teaser")["duration_sec"], 45)

    def test_word_count_drives_duration(self):
        t = sw.parse_brief("写个 300 字的口播")
        self.assertEqual(t["word_count"], 300)
        self.assertTrue(t["explicit_word_count"])
        self.assertFalse(t["explicit_duration"])
        self.assertEqual(t["duration_sec"], round(300 / sw.CHARS_PER_SECOND))

    def test_both_stated_are_both_kept(self):
        t = sw.parse_brief("90秒、400字的科普短片")
        self.assertEqual((t["duration_sec"], t["word_count"]), (90, 400))

    def test_default_when_unstated(self):
        t = sw.parse_brief("做一支介绍我们团队的短片")
        self.assertEqual(t["duration_sec"], sw.DEFAULT_DURATION_SEC)
        self.assertFalse(t["explicit_duration"])
        self.assertGreaterEqual(t["word_count"], sw.MIN_WORDS)

    def test_absurd_durations_are_clamped(self):
        self.assertLessEqual(sw.parse_brief("做一支 10 小时的片子")["duration_sec"],
                             sw.MAX_DURATION_SEC)
        self.assertGreaterEqual(sw.parse_brief("1 秒的片子")["duration_sec"],
                                sw.MIN_DURATION_SEC)


class CleanDraftTest(unittest.TestCase):
    def test_strips_thinking_fences_headings_labels_bullets(self):
        raw = (
            "<think>先想结构</think>\n"
            "```markdown\n"
            "# 城市清晨\n"
            "【开头】\n"
            "1. 五点半，城市还没睁眼。\n"
            "- 第一班地铁已经出发。\n"
            "\n"
            "旁白：这就是清晨。\n"
            "```"
        )
        out = sw.clean_draft(raw)
        for junk in ("<think>", "```", "#", "【开头】", "旁白：", "1.", "- "):
            self.assertNotIn(junk, out)
        self.assertIn("五点半，城市还没睁眼。", out)
        self.assertIn("这就是清晨。", out)
        self.assertNotIn("城市清晨第", out)  # the heading is dropped, not inlined

    def test_heading_only_draft_is_not_emptied(self):
        # The lenient second pass must catch what the strict one rejected.
        self.assertEqual(sw.clean_draft("# 只有一个标题"), "只有一个标题")

    def test_unwraps_whole_script_quotes(self):
        self.assertEqual(sw.clean_draft("“整段被引号包住。”"), "整段被引号包住。")

    def test_output_is_always_one_paragraph(self):
        out = sw.clean_draft("五点半，城市还没睁眼。\n\n第一班地铁已经出发。\n\n城市醒了。")
        self.assertNotIn("\n", out)
        self.assertEqual(out, "五点半，城市还没睁眼。第一班地铁已经出发。城市醒了。")

    def test_repeated_paragraphs_are_deduped(self):
        para = "五点半，城市还没睁眼。第一班地铁已经出发。"
        self.assertEqual(sw.clean_draft(para + "\n" + para + "\n" + para), para)

    def test_strips_inline_character_tallies(self):
        """The model narrates its own word count: "…发现密室。(29)墙缝里…"."""
        raw = ("道士王圆箓清理积沙，意外发现密室。(29)"
               "墙缝里漏出的泥丸，成了改变历史的线索。（18字）"
               "起初他以为藏了佛像。(共19)")
        out = sw.clean_draft(raw)
        for junk in ("(", ")", "（", "）", "29", "18", "19"):
            self.assertNotIn(junk, out)
        self.assertIn("意外发现密室。", out)
        self.assertIn("成了改变历史的线索。", out)

    def test_keeps_years_in_parentheses(self):
        # Four digits is a year, not a tally, and must survive.
        self.assertIn("1900", sw.clean_draft("那是（1900）年发生的事，谁都没想到。"))

    def test_empty_input(self):
        self.assertEqual(sw.clean_draft(""), "")
        self.assertEqual(sw.clean_draft(None), "")


class SalvageTest(unittest.TestCase):
    """A draft cut off inside <think> has no closing tag for anyone to strip."""

    TRACE = (
        "<think>\n"
        "**Analyze the Request:**\n"
        "**Duration:** Approximately 300 seconds.\n"
        "Topic: Financial Tax Education (财务税务).\n"
        "*Idea:* Taxes are often misunderstood by most people.\n"
        "*Draft:* 区分成本与费用是基础，合规申报才能享受优惠。\n"
        "\n"
        "*Final Expansion:*\n"
        "        很多人以为交税只是数字，其实它关乎你每一分收入的落袋。\n"
        "\n"
        "        发票管理是财务风控的第一道防线，进项没有，利润表就会失真。\n"
        "\n"
        "        真正的财务自由，不是不交税，而是懂得在规则内保留财富。\n"
        "\n"
        "        很多人以为交税只是数字，其实它关乎你每一分收入的落袋。\n"
        "        发票管理是财务风控的第一道防线，进项没有，利润表就会失真。\n"
        "\n"
        "        (Count check: ~1100 characters. Still short.)\n"
        "*Continuing:*\n"
        "        老板们要特别注意资金流向，大额"
    )

    def setUp(self):
        self.out = sw.clean_draft(self.TRACE)

    def test_drops_the_english_reasoning(self):
        for junk in ("<think>", "Analyze", "Duration", "Idea", "Count check",
                     "Financial", "Approximately", "**", "*Draft"):
            self.assertNotIn(junk, self.out)

    def test_recovers_the_longest_draft_not_the_fragments(self):
        self.assertIn("很多人以为交税只是数字", self.out)
        self.assertIn("真正的财务自由", self.out)
        # An early throwaway one-liner from a different block must not lead.
        self.assertNotIn("区分成本与费用是基础", self.out)

    def test_result_is_one_deduped_paragraph(self):
        self.assertNotIn("\n", self.out)
        self.assertEqual(self.out.count("发票管理是财务风控的第一道防线"), 1)

    def test_trailing_fragment_is_trimmed(self):
        trimmed = sw.trim_to_sentence(self.out)
        self.assertFalse(sw.ends_mid_sentence(trimmed))
        self.assertNotIn("大额", trimmed)


class LengthControlTest(unittest.TestCase):
    def test_trim_to_sentence_drops_partial_tail(self):
        self.assertEqual(sw.trim_to_sentence("甲很长的一句话。乙没说完"), "甲很长的一句话。")

    def test_trim_to_sentence_keeps_complete_text(self):
        self.assertEqual(sw.trim_to_sentence("甲。乙。"), "甲。乙。")

    def test_trim_to_length_cuts_on_a_sentence_boundary(self):
        text = "第一句话在这里。第二句话在这里。第三句话在这里。"
        out = sw.trim_to_length(text, 16)
        self.assertTrue(out.endswith("。"))
        self.assertLessEqual(sw.visible_length(out), 16)

    def test_trim_to_length_keeps_the_closing_sentence(self):
        """Cutting the tail would drop the 总分总 payoff, so the body goes first."""
        text = "开头钩子在这里。中间第一层。中间第二层。中间第三层。结尾的收束句在这里。"
        out = sw.trim_to_length(text, 20)
        self.assertTrue(out.startswith("开头钩子在这里。"))
        self.assertTrue(out.endswith("结尾的收束句在这里。"))
        self.assertNotIn("中间第三层", out)

    def test_trim_to_length_is_a_noop_when_short_enough(self):
        self.assertEqual(sw.trim_to_length("甲。乙。", 999), "甲。乙。")

    def test_token_budget_covers_thinking_and_script(self):
        # Still leave breathing room above the bare script length, but keep
        # latency bounded for interactive drafting.
        self.assertGreater(sw.token_budget(1380), 4300)
        self.assertLessEqual(sw.token_budget(99999), sw.MAX_OUTPUT_TOKENS)
        self.assertGreaterEqual(sw.token_budget(30), 2048)

    def test_the_budget_leaves_room_to_overshoot_the_target(self):
        """A Chinese character costs about a token, so clear 2x the target.

        Overshooting is trimmed back by ``trim_to_length``; stopping short is
        what the user actually sees, so the budget errs high. Unused budget is
        free — generation stops at EOS.
        """
        for target in (280, 550, 1380):
            self.assertGreaterEqual(sw.token_budget(target), target * 2, f"{target}")


class VisibleLengthTest(unittest.TestCase):
    def test_ignores_punctuation_and_whitespace(self):
        self.assertEqual(sw.visible_length("五点半，城市还没睁眼。"), 9)
        self.assertEqual(sw.visible_length("  甲。\n\n乙！  "), 2)


class _FakeModel:
    """Stands in for ``ModelService``; ``replies`` is consumed one per attempt."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def compose(self, system, user, *, max_tokens, temperature):
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return {"text": reply, "raw": reply, "truncated": False, "completion_tokens": 0}


class _GenerateOnlyModel:
    """A backend without ``compose`` must still work through ``generate``."""

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def generate(self, system, user, *, max_tokens, temperature):
        return self.reply


def _script_of(chars: int) -> str:
    """A draft of roughly ``chars`` visible characters, all distinct sentences."""
    sentences = []
    while sw.visible_length("".join(sentences)) < chars:
        i = len(sentences) + 1
        sentences.append(f"这是第{i}句话，写的是清晨城市里的一个很具体的画面。")
    return "".join(sentences)


class DraftTest(unittest.TestCase):
    # Comfortably over the 280-char target so the expansion loop stays out.
    LONG = _script_of(320)

    def test_happy_path(self):
        model = _FakeModel("<think>x</think>\n# 标题\n" + self.LONG)
        out = asyncio.run(sw.draft(model, "剪一支 60 秒的城市清晨短片"))

        self.assertNotIn("标题", out["script"])
        self.assertNotIn("\n", out["script"])
        self.assertEqual(out["target"]["duration_sec"], 60)
        self.assertEqual(out["char_count"], sw.visible_length(out["script"]))
        self.assertEqual(out["brief"], "剪一支 60 秒的城市清晨短片")
        self.assertFalse(out["short"])
        self.assertEqual(len(model.calls), 1)   # no retry, no expansion

        self.assertIn("60 秒", model.calls[0]["user"])
        self.assertIn("280 字", model.calls[0]["user"])
        self.assertEqual(model.calls[0]["max_tokens"], sw.token_budget(sw.upper_bound(280)))

    def test_the_prompt_asks_for_a_range_that_starts_at_the_target(self):
        """Naming one number gets ~80% of it back, so name a range above it."""
        model = _FakeModel(_script_of(560))
        asyncio.run(sw.draft(model, "做一支 120 秒、约 550 字的历史科普短片"))

        prompt = model.calls[0]["user"]
        self.assertEqual(sw.upper_bound(550), 690)
        self.assertIn("550-690 字", prompt)
        self.assertIn("少于 550 字算不合格", prompt)

    def test_the_prompt_states_the_length_as_a_sentence_count(self):
        """A character count alone is not something a model can track."""
        model = _FakeModel(_script_of(560))
        asyncio.run(sw.draft(model, "做一支 120 秒、约 550 字的历史科普短片"))

        self.assertEqual(sw.sentence_target(550), 20)
        self.assertIn(
            f"{sw.sentence_target(550)}-{sw.sentence_target(690)} 句",
            model.calls[0]["user"],
        )

    def test_retries_once_when_first_draft_is_a_fragment(self):
        model = _FakeModel("太短了。", "<think>ok</think>\n" + self.LONG)
        out = asyncio.run(sw.draft(model, "剪一支 60 秒的城市清晨短片"))

        self.assertGreaterEqual(len(model.calls), 2)
        self.assertIn(sw.EXPAND_SUFFIX.strip().splitlines()[0], model.calls[1]["system"])
        self.assertGreater(out["char_count"], 100)
        self.assertFalse(out["short"])

    def test_keeps_the_best_of_two_bad_attempts(self):
        model = _FakeModel("短。", "稍微长一点点的一句话在这里。")
        out = asyncio.run(sw.draft(model, "剪一支 60 秒的城市清晨短片"))
        self.assertTrue(out["short"])
        self.assertEqual(out["script"], "稍微长一点点的一句话在这里。")

    def test_overlong_draft_is_capped_at_a_sentence_boundary(self):
        model = _FakeModel("这是一个很完整的句子。" * 200)
        out = asyncio.run(sw.draft(model, "剪一支 60 秒的城市清晨短片"))
        # 280-char target, +15% headroom; dedupe already collapses the repeats.
        self.assertLessEqual(out["char_count"], int(280 * sw.MAX_KEEP_RATIO))
        self.assertFalse(sw.ends_mid_sentence(out["script"]))

    def test_falls_back_to_generate_without_compose(self):
        out = asyncio.run(sw.draft(_GenerateOnlyModel(self.LONG), "60 秒的城市清晨"))
        self.assertGreater(out["char_count"], 100)

    def test_empty_brief_rejected(self):
        with self.assertRaises(ValueError):
            asyncio.run(sw.draft(_FakeModel("whatever"), "   "))

    def test_expands_a_short_draft_up_to_target(self):
        """The 300s case: a 42%-of-target draft has to be grown, not accepted."""
        model = _FakeModel(_script_of(320), _script_of(420), _script_of(560))
        out = asyncio.run(sw.draft(model, "做一支 120 秒、约 550 字的历史科普短片"))

        self.assertEqual(out["target"]["word_count"], 550)
        self.assertGreaterEqual(out["char_count"], int(550 * sw.MIN_ACCEPT_RATIO))
        self.assertLessEqual(out["char_count"], int(550 * sw.MAX_KEEP_RATIO))
        self.assertFalse(out["short"])
        self.assertTrue(any(a.get("expand") for a in out["attempts"]))
        self.assertIn("还差约", model.calls[-1]["user"])

    def test_expansion_gives_up_when_it_stops_growing(self):
        model = _FakeModel(_script_of(200))     # same reply every call
        out = asyncio.run(sw.draft(model, "做一支 120 秒、约 550 字的历史科普短片"))

        self.assertTrue(out["short"])
        self.assertLessEqual(len(model.calls), 2 + sw.MAX_EXPAND_ROUNDS)
        self.assertGreater(out["char_count"], 100)   # best effort is still returned

    def test_expansion_keeps_going_while_it_is_still_gaining(self):
        """A draft half the target needs more than one round to close the gap."""
        model = _FakeModel(_script_of(280), _script_of(400), _script_of(540))
        out = asyncio.run(sw.draft(model, "做一支 120 秒、约 550 字的历史科普短片"))

        self.assertEqual(len(model.calls), 3)
        self.assertGreaterEqual(out["char_count"], int(550 * sw.MIN_ACCEPT_RATIO))
        self.assertFalse(out["short"])

    def test_expansion_rounds_are_not_given_a_smaller_budget(self):
        """Halving the budget mid-expansion is self-defeating: length is the bug."""
        model = _FakeModel(_script_of(280), _script_of(540))
        asyncio.run(sw.draft(model, "做一支 120 秒、约 550 字的历史科普短片"))

        self.assertEqual(model.calls[1]["max_tokens"], model.calls[0]["max_tokens"])

    def test_expansion_aims_past_the_target_not_at_it(self):
        """Aiming at the target means landing under it, the same as round one."""
        model = _FakeModel(_script_of(280), _script_of(540))
        asyncio.run(sw.draft(model, "做一支 120 秒、约 550 字的历史科普短片"))

        self.assertIn("要写到 690 字", model.calls[1]["user"])

    def test_a_near_target_draft_is_accepted_without_another_call(self):
        """Speed still matters: don't spend a round chasing the last few 字."""
        model = _FakeModel(_script_of(530))
        out = asyncio.run(sw.draft(model, "做一支 120 秒、约 550 字的历史科普短片"))

        self.assertEqual(len(model.calls), 1)
        self.assertFalse(out["short"])

    def test_empty_draft_rejected(self):
        with self.assertRaises(RuntimeError):
            asyncio.run(sw.draft(_FakeModel("<think>only thinking</think>"), "随便拍点什么"))


class ComposeTest(unittest.TestCase):
    """``compose`` is the call ``draft`` makes; it must not reason first.

    Reasoning about a narration draft costs seconds per call and spends the
    token budget the prose needs — which is how a 550-字 request came back at
    288 字, cut off while the model was still deliberating.
    """

    def _service(self):
        from studio.model_client import ModelService

        calls: list[dict] = []

        class _Loader:
            def predict(self, system, user, **kw):
                calls.append(kw)
                from types import SimpleNamespace
                return SimpleNamespace(
                    cleaned_output="正文。", raw_output="正文。",
                    thinking_summary="", completion_tokens=3,
                )

        service = object.__new__(ModelService)
        service._require_loader = lambda: _Loader()   # type: ignore[method-assign]
        return service, calls

    def test_compose_asks_for_a_direct_completion(self):
        service, calls = self._service()
        service.compose("sys", "user", max_tokens=2454, temperature=0.7)

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["enable_thinking"], False)
        self.assertEqual(calls[0]["role"], "writer")
        self.assertEqual(calls[0]["max_tokens"], 2454)


if __name__ == "__main__":
    unittest.main(verbosity=2)
