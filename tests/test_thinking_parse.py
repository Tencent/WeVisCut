"""Unit tests for think extraction / empty-think handling.

Run:  python3 tests/test_thinking_parse.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

STUDIO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDIO_DIR))

from server.model_loader import (  # noqa: E402
    _build_prediction_result,
    _extract_thinking,
    _with_thinking_extra_body,
)


class ThinkingParseTests(unittest.TestCase):
    def test_extracts_sft_style_think(self):
        raw = (
            "<think>\n观察：未分段。\n判断：先切章节。\n下一步：调用 Tool_SegmentArticle。\n"
            "</think>\n<tool_call>\n{\"name\":\"Tool_SegmentArticle\",\"arguments\":{}}\n</tool_call>"
        )
        self.assertIn("观察：未分段", _extract_thinking(raw))

    def test_empty_think_prefix_is_ignored(self):
        raw = "<think>\n\n</think>\n\n<tool_call>\n{\"name\":\"Tool_PlanShots\"}\n</tool_call>"
        self.assertEqual(_extract_thinking(raw), "")

    def test_reasoning_content_fallback(self):
        self.assertEqual(
            _extract_thinking("<tool_call>{}</tool_call>", "先规划镜头再装饰"),
            "先规划镜头再装饰",
        )

    def test_prediction_reads_reasoning_content(self):
        msg = SimpleNamespace(
            content="<tool_call>\n{\"name\":\"Tool_StitchVideo\",\"arguments\":{}}\n</tool_call>",
            reasoning_content="质检已过，收尾拼接。",
            tool_calls=None,
        )
        resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)
        result = _build_prediction_result([resp])
        self.assertEqual(result.thinking_summary, "质检已过，收尾拼接。")
        self.assertIn("<think>", result.raw_output)
        self.assertIn("Tool_StitchVideo", result.cleaned_output)

    def test_openai_extra_body_gets_enable_thinking(self):
        merged = _with_thinking_extra_body({"top_k": 1}, True)
        self.assertEqual(merged["chat_template_kwargs"]["enable_thinking"], True)
        self.assertEqual(merged["top_k"], 1)


if __name__ == "__main__":
    unittest.main()
