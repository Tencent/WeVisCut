"""Tests for the opt-in 0907 subtitle builder.

Run: python3 tests/test_newvis0907_subtitles.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studio.newvis0907.subtitles import (  # noqa: E402
    build_subtitle_cues,
    equivalent_to_narration,
    normalize_lexical,
    punctuate_chinese,
)


class NormalizeLexicalTest(unittest.TestCase):
    def test_drops_punctuation_and_whitespace_only(self) -> None:
        text = " AI，ChatGPT（2026年） 达到 95% 了吗？ "
        self.assertEqual(normalize_lexical(text), "AIChatGPT2026年达到95%了吗")

    def test_preserves_character_sequence(self) -> None:
        original = "今天，我们看 AI 与 2026 年。"
        normalized = normalize_lexical(original)
        self.assertEqual(normalized, "今天我们看AI与2026年")
        self.assertEqual(normalized, normalize_lexical("今天我们看AI与2026年"))


class EquivalenceTest(unittest.TestCase):
    def test_treats_punctuation_only_differences_as_equal(self) -> None:
        self.assertTrue(
            equivalent_to_narration(
                "为什么 AI 在 2026 年增长这么快？",
                "为什么AI在2026年增长这么快",
            )
        )

    def test_detects_real_character_drift(self) -> None:
        self.assertFalse(equivalent_to_narration("为什么增长", "为什么减少"))


class PunctuateChineseTest(unittest.TestCase):
    def test_preserves_existing_punctuation(self) -> None:
        result = punctuate_chinese("今天很好。我们继续？")
        self.assertTrue(result["valid"])
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["text"], "今天很好。我们继续？")

    def test_adds_question_mark_for_question_words(self) -> None:
        result = punctuate_chinese("为什么AI在2026年增长这么快")
        self.assertTrue(result["valid"])
        self.assertEqual(result["text"], "为什么AI在2026年增长这么快？")
        self.assertEqual(
            normalize_lexical(result["text"]),
            normalize_lexical("为什么AI在2026年增长这么快"),
        )

    def test_adds_period_for_plain_statement(self) -> None:
        result = punctuate_chinese("这是一个没有标点的陈述句")
        self.assertTrue(result["valid"])
        self.assertEqual(result["text"], "这是一个没有标点的陈述句。")

    def test_ellipsis_is_a_terminal_punctuation(self) -> None:
        self.assertEqual(punctuate_chinese("他突然停住了…")["text"], "他突然停住了…")
        self.assertEqual(punctuate_chinese("他突然停住了......")["text"], "他突然停住了......")

    def test_soft_tail_is_converged_instead_of_appended(self) -> None:
        self.assertEqual(punctuate_chinese("你好，")["text"], "你好。")
        self.assertEqual(punctuate_chinese("为什么是这样，")["text"], "为什么是这样？")

    def test_dash_tail_converges_to_sentence_end(self) -> None:
        self.assertEqual(punctuate_chinese("接下来是重点——")["text"], "接下来是重点。")
        self.assertEqual(punctuate_chinese("为什么会这样——")["text"], "为什么会这样？")

    def test_falls_back_when_punctuator_changes_characters(self) -> None:
        result = punctuate_chinese(
            "我们一起看AI的发展",
            punctuator=lambda text: "我们一起看看AI的发展。",
        )
        self.assertTrue(result["valid"])
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["text"], "我们一起看AI的发展。")

    def test_falls_back_when_punctuator_drops_or_duplicates_characters(self) -> None:
        dropped = punctuate_chinese(
            "这里展示ChatGPT的能力",
            punctuator=lambda text: "这里展示ChatGPT能力。",
        )
        duplicated = punctuate_chinese(
            "这里展示ChatGPT的能力",
            punctuator=lambda text: "这里展示ChatGPT的的能力。",
        )
        self.assertTrue(dropped["fallback_used"])
        self.assertTrue(duplicated["fallback_used"])
        self.assertEqual(normalize_lexical(dropped["text"]), normalize_lexical("这里展示ChatGPT的能力"))
        self.assertEqual(normalize_lexical(duplicated["text"]), normalize_lexical("这里展示ChatGPT的能力"))


class BuildSubtitleCuesTest(unittest.TestCase):
    def test_keeps_existing_punctuation_and_reports_validation(self) -> None:
        result = build_subtitle_cues("第一句。第二句？", duration=4.0)
        self.assertEqual(result["display_text"], "第一句。第二句？")
        self.assertEqual(len(result["cues"]), 2)
        self.assertEqual(result["cues"][0]["text"], "第一句。")
        self.assertEqual(result["cues"][1]["text"], "第二句？")
        self.assertTrue(result["validation"]["equivalent"])
        self.assertFalse(result["validation"]["fallback_used"])

    def test_builds_question_and_statement_display_text(self) -> None:
        question = build_subtitle_cues("如何让AI真正帮助团队", duration=3.0)
        statement = build_subtitle_cues("这是一条普通陈述", duration=3.0)
        self.assertEqual(question["display_text"], "如何让AI真正帮助团队？")
        self.assertEqual(statement["display_text"], "这是一条普通陈述。")

    def test_wraps_long_parallel_sentence_without_breaking_ascii_tokens(self) -> None:
        text = "我们先看AI工具如何接入团队流程然后看ChatGPT在2026年的协作效率提升百分之35再看复盘机制是否稳定"
        result = build_subtitle_cues(text, duration=12.0, max_chars_per_line=14, max_lines=2)
        self.assertTrue(result["validation"]["equivalent"])
        self.assertGreaterEqual(len(result["cues"]), 2)
        for cue in result["cues"]:
            self.assertLessEqual(len(cue["lines"]), 2)
            for line in cue["lines"]:
                self.assertLessEqual(len(line), 14, line)
            if "ChatGPT" in cue["text"]:
                self.assertTrue(any("ChatGPT" in line for line in cue["lines"]), cue["lines"])
        self.assertEqual(result["cues"][-1]["end"], 12.0)

    def test_keeps_quotes_and_parentheses_balanced_inside_a_cue(self) -> None:
        text = '他说“AI（尤其是ChatGPT）会改变流程”然后我们继续分析'
        result = build_subtitle_cues(text, duration=6.0, max_chars_per_line=14, max_lines=2)
        joined = "".join(cue["text"] for cue in result["cues"])
        self.assertEqual(normalize_lexical(joined), normalize_lexical(result["display_text"]))
        for cue in result["cues"]:
            cue_text = cue["text"]
            self.assertEqual(cue_text.count("“"), cue_text.count("”"))
            self.assertEqual(cue_text.count("（"), cue_text.count("）"))

    def test_handles_ascii_terms_numbers_and_units_as_protected_tokens(self) -> None:
        text = "AI和ChatGPT在2026年把转化率提升到95%"
        result = build_subtitle_cues(text, duration=5.0, max_chars_per_line=10, max_lines=2)
        self.assertTrue(result["validation"]["equivalent"])
        flattened = [line for cue in result["cues"] for line in cue["lines"]]
        self.assertTrue(any("AI" in line for line in flattened))
        self.assertTrue(any("ChatGPT" in line for line in flattened))
        self.assertTrue(any("2026年" in line for line in flattened))
        self.assertTrue(any("95%" in line for line in flattened))

    def test_duration_weights_are_monotonic_and_end_exactly_at_duration(self) -> None:
        result = build_subtitle_cues("第一句。第二句更长一点。第三句最长最长。", duration=9.0)
        previous_end = 0.0
        for cue in result["cues"]:
            self.assertGreaterEqual(cue["start"], 0.0)
            self.assertGreaterEqual(cue["end"], cue["start"])
            self.assertGreaterEqual(cue["start"], previous_end)
            previous_end = cue["end"]
        self.assertEqual(result["cues"][-1]["end"], 9.0)

    def test_zero_or_short_duration_never_goes_negative(self) -> None:
        zero = build_subtitle_cues("只有一句", duration=0.0)
        tiny = build_subtitle_cues("第一句。第二句。", duration=0.1)
        self.assertEqual(zero["cues"][0]["start"], 0.0)
        self.assertEqual(zero["cues"][-1]["end"], 0.0)
        for cue in tiny["cues"]:
            self.assertGreaterEqual(cue["start"], 0.0)
            self.assertGreaterEqual(cue["end"], cue["start"])

    def test_validation_reports_character_conservation(self) -> None:
        result = build_subtitle_cues("我们保持字序完全一致", duration=2.0)
        self.assertTrue(result["validation"]["equivalent"])
        self.assertEqual(
            result["validation"]["normalized_source"],
            result["validation"]["normalized_display"],
        )

    def test_long_chinese_text_splits_at_cue_level_without_losing_characters(self) -> None:
        text = "这是一个特别长的中文句子需要继续延展以确保超过两行容量同时不能丢失任何一个字并且还要稳定切分"
        result = build_subtitle_cues(text, duration=8.0, max_chars_per_line=14, max_lines=2)
        self.assertGreaterEqual(len(result["cues"]), 2)
        self.assertEqual("".join(cue["text"] for cue in result["cues"]), result["display_text"])
        for cue in result["cues"]:
            self.assertLessEqual(len(cue["lines"]), 2)
            self.assertEqual("".join(cue["lines"]), cue["text"])
            for line in cue["lines"]:
                self.assertLessEqual(len(line), 14)

    def test_long_protected_ascii_token_is_not_truncated(self) -> None:
        token = "HyperSuperProtectedToken2026AlphaBeta"
        text = f"我们测试{token}在字幕里绝不能丢字"
        result = build_subtitle_cues(text, duration=6.0, max_chars_per_line=14, max_lines=2)
        self.assertEqual(normalize_lexical(result["display_text"]), normalize_lexical(text))
        self.assertEqual("".join(cue["text"] for cue in result["cues"]), result["display_text"])
        self.assertTrue(any(token in cue["text"] for cue in result["cues"]))
        for cue in result["cues"]:
            self.assertLessEqual(len(cue["lines"]), 2)
            self.assertEqual("".join(cue["lines"]), cue["text"])
            for line in cue["lines"]:
                if token in line:
                    self.assertGreater(len(line), 14)
                else:
                    self.assertLessEqual(len(line), 14)


class CueBoundariesTest(unittest.TestCase):
    """A cue is a complete clause, not the first half of one.

    Splitting on length first used to strand「睡眠不足的人」in one cue and
    「认知功能会明显下降」in the next, so the viewer read half a thought at a time.
    """

    #: Punctuation a readable unit is allowed to end on.
    CLAUSE_END = "。！？!?…，；：,;:"

    def _cues(self, text: str, duration: float = 8.0) -> list[dict]:
        return build_subtitle_cues(
            text, duration=duration, max_chars_per_line=14, max_lines=2)["cues"]

    def test_every_cue_but_the_last_ends_on_punctuation(self) -> None:
        text = "睡眠是大脑清洁时间，你知道吗，人脑在深睡时其实正在做一次彻底的清理。"
        cues = self._cues(text)
        self.assertGreater(len(cues), 1, cues)
        for cue in cues[:-1]:
            self.assertIn(cue["text"][-1], self.CLAUSE_END, cue["text"])

    def test_a_clause_is_never_spread_over_two_cues(self) -> None:
        text = "研究发现，睡眠不足的人，认知功能会明显下降，反应速度慢了百分之四十。"
        cues = self._cues(text)
        for cue in cues:
            # Each cue is whole clauses, so stripping the clause marks off the
            # end leaves no dangling fragment start.
            self.assertIn(cue["text"][-1], self.CLAUSE_END, cue["text"])

    def test_a_short_sentence_stays_in_one_cue(self) -> None:
        cues = self._cues("请相信，今晚早睡一小时，就是给大脑最好的礼物。")
        self.assertEqual(len(cues), 1, [c["text"] for c in cues])

    def test_clause_packing_still_loses_no_characters(self) -> None:
        text = "第一点，睡够七小时；第二点，别在睡前刷手机；第三点，固定起床时间，周末也一样。"
        result = build_subtitle_cues(text, duration=12.0, max_chars_per_line=14, max_lines=2)
        self.assertEqual("".join(c["text"] for c in result["cues"]), result["display_text"])
        self.assertEqual(normalize_lexical(result["display_text"]), normalize_lexical(text))


class BurnedInLinesMatchThePlanTest(unittest.TestCase):
    """What ffmpeg draws has to be what the builder planned and validated."""

    def test_render_uses_the_planned_line_breaks(self) -> None:
        from pipeline.tools.timeline_compose import _wrap_subtitle_lines

        cue = build_subtitle_cues(
            "研究发现，睡眠不足的人，认知功能会明显下降。",
            duration=6.0, max_chars_per_line=14, max_lines=2)["cues"][0]
        drawn = _wrap_subtitle_lines(
            cue["text"], max_chars_per_line=14, max_lines=2, planned=cue["lines"])
        self.assertEqual(drawn, cue["lines"])

    def test_render_never_drops_the_tail_of_a_cue(self) -> None:
        from pipeline.tools.timeline_compose import _wrap_subtitle_lines

        text = "这一整句必须完整出现在屏幕上一个字都不能少掉"
        drawn = _wrap_subtitle_lines(text, max_chars_per_line=14, max_lines=2)
        self.assertEqual("".join(drawn), text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
