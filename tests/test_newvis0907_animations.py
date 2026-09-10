"""Offline tests for the newvis0907 semantic animation planner.

Run: python3 tests/test_newvis0907_animations.py

These tests are strict TDD for :mod:`studio.newvis0907.animations` and for the
propagation of semantic anchors through operations / verification. They never
touch the network filesystem, model, ffmpeg or HyperFrames — a tiny on-disk
Gold Template registry (only ``generated/<layer>/<id>/<marker>`` marker files)
stands in for the real library so ``select_template`` can only ever return real
ids, never invented ones.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studio import project as project_mod  # noqa: E402
from studio.newvis0907 import animations, operations, verification  # noqa: E402


# ---------------------------------------------------------------------------
# tiny real (temp) Gold Template registry
# ---------------------------------------------------------------------------
_STRUCTURES = (
    "quote_card",
    "key_takeaway",
    "thesis_statement",
    "three_points",
    "recap_checklist",
    "definition_card",
    "stat_highlight",
    "metric_cards",
    "bar_chart",
    "line_chart",
    "timeline_step",
    "historical_moment",
)
_MOTIONS = ("scale_pop_hold", "slide_in_hold", "mask_reveal", "card_stack_sequence", "parallax_pan")
_VISUALS = ("dark_tech", "clean_saas", "warm_editorial")

_MARKER = {"structures": "structure.html", "motions": "motion.js", "visuals": "tokens.css"}


def make_gold_root(tmp: Path) -> Path:
    root = tmp / "gold_template"
    for layer, ids in (("structures", _STRUCTURES), ("motions", _MOTIONS), ("visuals", _VISUALS)):
        for sid in ids:
            d = root / "generated" / layer / sid
            d.mkdir(parents=True, exist_ok=True)
            (d / _MARKER[layer]).write_text("x", encoding="utf-8")
    return root


def registry_lists() -> dict:
    """A plain list-based registry (no disk) as the task allows for mocks."""
    return {
        "structure_id": list(_STRUCTURES),
        "motion_id": list(_MOTIONS),
        "visual_id": list(_VISUALS),
    }


def make_gold_root_with_schema(tmp: Path, structure_id: str, schema: dict) -> Path:
    """Gold root plus a real ``slot_schema.json`` for one structure (F3)."""
    root = make_gold_root(tmp)
    struct_dir = root / "generated" / "structures" / structure_id
    struct_dir.mkdir(parents=True, exist_ok=True)
    (struct_dir / "structure.html").write_text("x", encoding="utf-8")
    (struct_dir / "slot_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False), encoding="utf-8"
    )
    return root


# ---------------------------------------------------------------------------
# classify_semantics — 7 types, positive & negative
# ---------------------------------------------------------------------------
class ClassifyTest(unittest.TestCase):
    def _types(self, text: str) -> set[str]:
        return {item["type"] for item in animations.classify_semantics(text)}

    def _items(self, text: str, kind: str) -> list[dict]:
        return [i for i in animations.classify_semantics(text) if i["type"] == kind]

    def test_every_item_has_contract_fields(self) -> None:
        for item in animations.classify_semantics('他说“效率就是理解”，关键是必须行动。'):
            self.assertIn("phrase", item)
            self.assertIn("type", item)
            self.assertIn("reason", item)
            self.assertIn("priority", item)
            self.assertTrue(item["phrase"])
            self.assertIsInstance(item["priority"], (int, float))

    def test_quote_positive_quotes_and_said(self) -> None:
        self.assertIn("quote", self._types('乔布斯说“求知若饥”。'))
        self.assertIn("quote", self._types('他表示，"慢就是快"。'))
        # exact phrase is the quoted content, not the whole sentence
        item = self._items('乔布斯说“求知若饥”。', "quote")[0]
        self.assertIn("求知若饥", item["phrase"])

    def test_quote_negative_no_marks(self) -> None:
        self.assertNotIn("quote", self._types("今天我们聊聊效率这件事。"))

    def test_key_point_positive_keywords(self) -> None:
        self.assertIn("key_point", self._types("真正的效率不是更快地点击。"))
        self.assertIn("key_point", self._types("这意味着必须重新理解工具。"))

    def test_key_point_negative(self) -> None:
        self.assertNotIn("key_point", self._types("我昨天买了一杯咖啡。"))

    def test_parallel_list_positive_two_items(self) -> None:
        self.assertIn("parallel_list", self._types("我们需要速度、质量和成本三方面平衡。"))
        self.assertIn("parallel_list", self._types("第一是数据，第二是算法，第三是算力。"))

    def test_parallel_list_negative_single_item(self) -> None:
        self.assertNotIn("parallel_list", self._types("我们需要速度。"))

    def test_proper_noun_positive_ascii_and_title(self) -> None:
        self.assertIn("proper_noun", self._types("OpenAI 发布了 GPT-4 模型。"))
        item = self._items("我们使用 GPT-4 完成任务。", "proper_noun")[0]
        self.assertIn("GPT-4", item["phrase"])

    def test_proper_noun_negative(self) -> None:
        self.assertNotIn("proper_noun", self._types("我们今天很开心。"))

    def test_statistic_positive_number_unit(self) -> None:
        self.assertIn("statistic", self._types("效率提升了80%。"))
        self.assertIn("statistic", self._types("速度快了3倍。"))

    def test_statistic_negative_bare_number(self) -> None:
        # a lone ordinal with no unit is not a statistic
        self.assertNotIn("statistic", self._types("我们来看第一个例子。"))

    def test_chart_gate_needs_two_comparable_values(self) -> None:
        self.assertIn("chart", self._types("A 方案 80% 而 B 方案只有 45%。"))
        # a single quantitative value is a statistic, never a chart
        self.assertNotIn("chart", self._types("效率提升了80%。"))
        # non-quantitative text is never a chart
        self.assertNotIn("chart", self._types("我们对比一下两种思路。"))

    def test_timeline_gate_needs_two_ordered_nodes(self) -> None:
        self.assertIn("timeline", self._types("首先采集数据，其次训练模型，最后上线。"))
        self.assertIn("timeline", self._types("从2019年到2023年，行业彻底改变。"))
        # a single time node is not a timeline
        self.assertNotIn("timeline", self._types("首先我们打个招呼。"))
        # non-temporal comparison is not a timeline
        self.assertNotIn("timeline", self._types("我们对比一下两种思路。"))


# ---------------------------------------------------------------------------
# locate_phrase_time — cue based, intra-cue ratio, cross-cue, honest failure
# ---------------------------------------------------------------------------
class LocateTest(unittest.TestCase):
    def _cues(self):
        return [
            {"start": 0.0, "end": 4.0, "text": "先看一个简单的背景"},
            {"start": 5.2, "end": 8.8, "text": "怎样让AI成为自己的工具"},
        ]

    def test_locate_within_cue_uses_start_of_matching_cue(self) -> None:
        out = animations.locate_phrase_time("怎样让AI成为自己的工具", self._cues(), 8.8)
        self.assertTrue(out["matched"])
        self.assertAlmostEqual(out["start"], 5.2, places=2)
        self.assertAlmostEqual(out["end"], 8.8, places=2)

    def test_locate_refines_by_depunctuated_char_ratio_inside_cue(self) -> None:
        cues = [{"start": 0.0, "end": 10.0, "text": "前半段内容，真正的效率在这里"}]
        out = animations.locate_phrase_time("真正的效率", cues, 10.0)
        self.assertTrue(out["matched"])
        # "真正的效率" starts after ~6 of 13 depunctuated chars -> well past 0
        self.assertGreater(out["start"], 3.5)
        self.assertLess(out["end"], 10.0 + 1e-6)
        self.assertGreater(out["end"], out["start"])

    def test_locate_spans_across_cues(self) -> None:
        cues = [
            {"start": 0.0, "end": 3.0, "text": "真正的"},
            {"start": 3.0, "end": 6.0, "text": "效率不是点击"},
        ]
        out = animations.locate_phrase_time("真正的效率", cues, 6.0)
        self.assertTrue(out["matched"])
        self.assertAlmostEqual(out["start"], 0.0, places=2)
        self.assertLess(out["end"], 6.0)
        self.assertGreater(out["end"], 3.0)

    def test_locate_matches_ignoring_punctuation(self) -> None:
        cues = [{"start": 1.0, "end": 5.0, "text": "真正的、效率！"}]
        out = animations.locate_phrase_time("真正的效率", cues, 5.0)
        self.assertTrue(out["matched"])

    def test_locate_missing_phrase_fails_without_defaulting_to_zero(self) -> None:
        out = animations.locate_phrase_time("不存在的短语", self._cues(), 8.8)
        self.assertFalse(out["matched"])
        self.assertIsNone(out["start"])
        self.assertIsNone(out["end"])

    def test_locate_clamps_to_duration(self) -> None:
        cues = [{"start": 4.0, "end": 20.0, "text": "尾巴内容"}]
        out = animations.locate_phrase_time("尾巴内容", cues, 6.0)
        self.assertTrue(out["matched"])
        self.assertLessEqual(out["end"], 6.0 + 1e-6)


# ---------------------------------------------------------------------------
# select_template — real ids only, type preference, structured failure
# ---------------------------------------------------------------------------
class SelectTemplateTest(unittest.TestCase):
    def test_type_preference_returns_real_ids(self) -> None:
        reg = registry_lists()
        cases = {
            "quote": "quote_card",
            "key_point": "key_takeaway",
            "statistic": "stat_highlight",
            "chart": "bar_chart",
            "timeline": "timeline_step",
            "proper_noun": "definition_card",
            "parallel_list": "three_points",
        }
        for kind, expected in cases.items():
            out = animations.select_template({"type": kind}, reg)
            self.assertTrue(out.get("ok"), f"{kind}: {out}")
            self.assertEqual(out["structure_id"], expected, kind)
            self.assertIn(out["structure_id"], _STRUCTURES)
            self.assertIn(out["motion_id"], _MOTIONS)
            self.assertIn(out["visual_id"], _VISUALS)

    def test_never_invents_ids_and_fails_on_empty_registry(self) -> None:
        out = animations.select_template({"type": "quote"}, {})
        self.assertFalse(out.get("ok"))
        self.assertIn("reason", out)
        self.assertNotIn("structure_id", {k: v for k, v in out.items() if v})

    def test_registry_missing_a_layer_is_structured_failure(self) -> None:
        reg = {"structure_id": list(_STRUCTURES), "motion_id": [], "visual_id": list(_VISUALS)}
        out = animations.select_template({"type": "quote"}, reg)
        self.assertFalse(out.get("ok"))

    def test_current_visual_is_kept_when_valid(self) -> None:
        reg = registry_lists()
        out = animations.select_template(
            {"type": "quote"}, reg, current={"visual_id": "warm_editorial"}
        )
        self.assertTrue(out["ok"])
        self.assertEqual(out["visual_id"], "warm_editorial")

    def test_gold_root_registry_from_disk(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = make_gold_root(Path(d))
            reg = animations.registry_from_gold_root(str(root))
            out = animations.select_template({"type": "statistic"}, reg)
            self.assertTrue(out["ok"])
            self.assertEqual(out["structure_id"], "stat_highlight")


# ---------------------------------------------------------------------------
# build_semantic_animation — one primary, anchors, user override
# ---------------------------------------------------------------------------
class BuildAnimationTest(unittest.TestCase):
    def _shot(self):
        return {
            "shot_id": "S1",
            "vocal_script": "真正的效率不是更快地点击，而是让工具主动理解你。",
            "duration_sec": 6.0,
        }

    def test_builds_primary_and_anchors_with_real_ids(self) -> None:
        reg = registry_lists()
        out = animations.build_semantic_animation(self._shot(), {}, reg)
        self.assertIn("primary", out)
        self.assertIn("anchors", out)
        self.assertIn("quality", out)
        primary = out["primary"]
        self.assertTrue(primary)
        self.assertIn(primary["structure_id"], _STRUCTURES)
        # anchor contract
        for key in ("phrase", "start_sec", "end_sec", "type", "structure_id",
                    "motion_id", "visual_id", "slots", "source"):
            self.assertIn(key, primary, key)
        self.assertEqual(primary["source"], "auto")
        # slots carry the exact phrase
        self.assertIn(primary["phrase"], json.dumps(primary["slots"], ensure_ascii=False))
        # time comes from cues, not shot start 0
        self.assertGreater(primary["end_sec"], 0.0)

    def test_at_most_one_primary(self) -> None:
        reg = registry_lists()
        out = animations.build_semantic_animation(self._shot(), {}, reg)
        self.assertIsInstance(out["primary"], dict)
        # anchors may be many, primary is a single dict (or empty)
        self.assertTrue(out["primary"] == {} or isinstance(out["primary"].get("phrase"), str))

    def test_user_emphasis_overrides_and_is_source_user(self) -> None:
        reg = registry_lists()
        out = animations.build_semantic_animation(
            self._shot(), {}, reg,
            user_emphasis={"phrase": "让工具主动理解你", "type": "key_point"},
        )
        self.assertEqual(out["primary"]["phrase"], "让工具主动理解你")
        self.assertEqual(out["primary"]["source"], "user")
        self.assertEqual(out["primary"]["type"], "key_point")

    def test_uses_provided_cues_for_timing(self) -> None:
        reg = registry_lists()
        assets = {"subtitle_cues": [
            {"start": 0.0, "end": 5.2, "text": "真正的效率不是更快地点击"},
            {"start": 5.2, "end": 8.8, "text": "而是让工具主动理解你"},
        ]}
        out = animations.build_semantic_animation(
            {"shot_id": "S1", "vocal_script": "真正的效率不是更快地点击，而是让工具主动理解你。", "duration_sec": 8.8},
            assets, reg,
            user_emphasis={"phrase": "让工具主动理解你", "type": "key_point"},
        )
        self.assertGreaterEqual(out["primary"]["start_sec"], 5.0)


# ---------------------------------------------------------------------------
# operations propagation: set_emphasis / remove_animation
# ---------------------------------------------------------------------------
def _state_payload(gold_root: str = "") -> dict:
    return {
        "project_id": "proj_anim",
        "config": {"paths": {"gold_template_root": gold_root}} if gold_root else {},
        "final_video_path": "/tmp/none/final.mp4",
        "segments": [{"segment_id": "seg1", "title": "章节一"}],
        "shot_plan": [
            {
                "shot_id": "S01",
                "segment_id": "seg1",
                "duration_sec": 6.0,
                "vocal_script": "真正的效率不是更快地点击。",
                "subtitle_text": "真正的效率不是更快地点击。",
                "is_spotlight": True,
                "semantic_animations": {
                    "primary": {"type": "key_point", "phrase": "真正的效率"},
                    "anchors": [{"phrase": "真正的效率", "type": "key_point"}],
                },
            },
            {
                "shot_id": "S02",
                "segment_id": "seg1",
                "duration_sec": 8.8,
                "vocal_script": "怎样让AI成为自己的工具，是我们要回答的问题。",
                "subtitle_text": "怎样让AI成为自己的工具，是我们要回答的问题。",
                "subtitle_cues": [
                    {"start": 0.0, "end": 5.2, "text": "先做一个铺垫说明"},
                    {"start": 5.2, "end": 8.8, "text": "怎样让AI成为自己的工具"},
                ],
            },
        ],
        "shot_assets": {
            "S01": {
                "decoration": {"structure_id": "quote_card", "slots": {"headline": "真正的效率"}},
                "clip_meta": {"visual_playlist": {"clips": [
                    {"subclip_id": "S01::subclip::1", "path": "/clips/a.mp4", "used_duration_sec": 6.0},
                ]}},
            },
            "S02": {
                "clip_meta": {"visual_playlist": {"clips": [
                    {"subclip_id": "S02::subclip::1", "path": "/clips/b.mp4", "used_duration_sec": 8.8},
                ]}},
            },
        },
    }


def write_project(tmp: Path, gold_root: str = "") -> project_mod.Project:
    (tmp / "final_state.json").write_text(
        json.dumps(_state_payload(gold_root), ensure_ascii=False), encoding="utf-8"
    )
    pipeline_config = {"paths": {"gold_template_root": gold_root}} if gold_root else None
    return project_mod.load(tmp, pipeline_config=pipeline_config)


class SetEmphasisPropagationTest(unittest.TestCase):
    def test_set_emphasis_writes_semantic_decoration_and_timing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = make_gold_root(Path(d))
            proj = write_project(Path(d), str(root))
            operations.apply_operations(proj, [{
                "op": "set_emphasis", "shot_id": "S02",
                "phrase": "怎样让AI成为自己的工具", "emphasis_type": "key_point",
            }])
            shot = proj.shot("S02")
            sem = shot["semantic_animations"]
            self.assertEqual(sem["primary"]["phrase"], "怎样让AI成为自己的工具")
            self.assertEqual(sem["primary"]["source"], "user")
            self.assertIn(sem["primary"]["structure_id"], _STRUCTURES)
            assets = proj.assets_for("S02")
            deco = assets["decoration"]
            self.assertIn(deco["structure_id"], _STRUCTURES)
            self.assertIn(deco["motion_id"], _MOTIONS)
            self.assertIn(deco["visual_id"], _VISUALS)
            # decor timing follows the phrase cue (~5.2s), never a shot-start 0
            self.assertGreaterEqual(deco["timing"]["decor_start_sec"], 5.0)
            self.assertGreater(deco["timing"]["decor_duration_sec"], 0.0)
            # slots carry the exact phrase
            self.assertIn("怎样让AI成为自己的工具",
                          json.dumps(deco["slots"], ensure_ascii=False))
            self.assertTrue(shot.get("is_spotlight"))
            self.assertIn("render", proj.dirty.get("S02", set()))
            # semantic animations mirrored to assets (for preservation)
            self.assertTrue(assets.get("semantic_animations"))

    def test_set_emphasis_rejects_phrase_absent_everywhere(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = make_gold_root(Path(d))
            proj = write_project(Path(d), str(root))
            before = copy.deepcopy(proj.state)
            with self.assertRaises(operations.OperationError):
                operations.apply_operations(proj, [{
                    "op": "set_emphasis", "shot_id": "S02", "phrase": "根本不在旁白里的话",
                }])
            self.assertEqual(proj.state, before)  # atomic: nothing changed

    def test_set_emphasis_without_registry_still_sets_primary(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))  # no gold root -> empty registry
            operations.apply_operations(proj, [{
                "op": "set_emphasis", "shot_id": "S02",
                "phrase": "怎样让AI成为自己的工具", "emphasis_type": "key_point",
            }])
            sem = proj.shot("S02")["semantic_animations"]
            self.assertEqual(sem["primary"]["phrase"], "怎样让AI成为自己的工具")


class RemoveAnimationPropagationTest(unittest.TestCase):
    def test_remove_by_phrase_clears_primary_and_deco_and_spotlight(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))
            operations.apply_operations(proj, [{
                "op": "remove_animation", "shot_id": "S01", "phrase": "真正的效率",
            }])
            shot = proj.shot("S01")
            sem = shot.get("semantic_animations", {})
            self.assertFalse(sem.get("primary", {}).get("phrase"))
            self.assertFalse(any(a.get("phrase") == "真正的效率" for a in sem.get("anchors", [])))
            # legacy decoration slot matching the phrase is removed / decoration cleared
            deco = proj.assets_for("S01").get("decoration", {})
            self.assertNotIn("真正的效率", json.dumps(deco, ensure_ascii=False))
            # no anchors remain -> spotlight off
            self.assertFalse(shot.get("is_spotlight"))
            # shot / narration / subtitle / footage untouched
            self.assertIsNotNone(proj.shot("S01"))
            self.assertEqual(shot["vocal_script"], "真正的效率不是更快地点击。")
            self.assertTrue(proj.assets_for("S01").get("clip_meta"))

    def test_remove_punctuation_equivalent_phrase(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))
            operations.apply_operations(proj, [{
                "op": "remove_animation", "shot_id": "S01", "phrase": "真正的、效率！",
            }])
            sem = proj.shot("S01").get("semantic_animations", {})
            self.assertFalse(sem.get("primary", {}).get("phrase"))


class FixtureCaseTest(unittest.TestCase):
    """The user's original case: two ops, one atomic transaction."""

    def test_remove_old_and_add_new_key_point_in_one_batch(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = make_gold_root(Path(d))
            proj = write_project(Path(d), str(root))
            before_ids = proj.shot_ids()
            narration_s02 = proj.shot("S02")["vocal_script"]
            footage_s02 = copy.deepcopy(proj.assets_for("S02")["clip_meta"])

            operations.apply_operations(proj, [
                {"op": "remove_animation", "shot_id": "S01", "phrase": "真正的效率"},
                {"op": "set_emphasis", "shot_id": "S02",
                 "phrase": "怎样让AI成为自己的工具", "emphasis_type": "key_point"},
            ])

            # old animation gone
            sem1 = proj.shot("S01").get("semantic_animations", {})
            self.assertFalse(sem1.get("primary", {}).get("phrase"))
            # new key_point present with real ids and decor_start ~5.2 (not 0)
            deco2 = proj.assets_for("S02")["decoration"]
            self.assertIn(deco2["structure_id"], _STRUCTURES)
            self.assertGreaterEqual(deco2["timing"]["decor_start_sec"], 5.0)
            # narration / subtitle / footage completely unchanged
            self.assertEqual(proj.shot("S02")["vocal_script"], narration_s02)
            self.assertEqual(proj.assets_for("S02")["clip_meta"], footage_s02)
            # no new shot (never an S08 style phantom)
            self.assertEqual(proj.shot_ids(), before_ids)

    def test_batch_is_atomic_when_second_op_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = make_gold_root(Path(d))
            proj = write_project(Path(d), str(root))
            before = copy.deepcopy(proj.state)
            with self.assertRaises(operations.OperationError):
                operations.apply_operations(proj, [
                    {"op": "remove_animation", "shot_id": "S01", "phrase": "真正的效率"},
                    {"op": "set_emphasis", "shot_id": "S02", "phrase": "不存在的短语"},
                ])
            self.assertEqual(proj.state, before)


# ---------------------------------------------------------------------------
# verification: strict post-conditions for set/remove
# ---------------------------------------------------------------------------
class VerificationStrictTest(unittest.TestCase):
    def test_set_emphasis_postcondition_requires_time_and_phrase(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = make_gold_root(Path(d))
            proj = write_project(Path(d), str(root))
            ops = [{"op": "set_emphasis", "shot_id": "S02",
                    "phrase": "怎样让AI成为自己的工具", "emphasis_type": "key_point"}]
            operations.apply_operations(proj, ops)
            checks = operations.verify_postconditions(proj, ops)
            emp = [c for c in checks if c["name"].endswith("set_emphasis")]
            self.assertTrue(emp and emp[0]["ok"], str(checks))

    def test_remove_postcondition_requires_absent_anchor_and_deco(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))
            ops = [{"op": "remove_animation", "shot_id": "S01", "phrase": "真正的效率"}]
            operations.apply_operations(proj, ops)
            checks = operations.verify_postconditions(proj, ops)
            rem = [c for c in checks if c["name"].endswith("remove_animation")]
            self.assertTrue(rem and rem[0]["ok"], str(checks))

    def test_full_verify_transaction_pure_visual_set_emphasis(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = make_gold_root(Path(d))
            proj = write_project(Path(d), str(root))
            before = operations.compute_digest(proj)
            ops = [{"op": "set_emphasis", "shot_id": "S02",
                    "phrase": "怎样让AI成为自己的工具", "emphasis_type": "key_point"}]
            operations.apply_operations(proj, ops)
            rendered = Path(d) / "r.mp4"
            final = Path(d) / "f.mp4"
            rendered.write_bytes(b"x" * 64)
            final.write_bytes(b"y" * 64)
            proj.assets_for("S02")["rendered_path"] = str(rendered)
            proj.state["final_video_path"] = str(final)
            result = verification.verify_transaction(
                proj, before, ops,
                render_result={"outputs": [str(rendered)], "final_video_path": str(final)},
            )
            self.assertTrue(result["complete"], str(result["checks"]))


# ---------------------------------------------------------------------------
# F3: slots filled from a real structure slot_schema.json
# ---------------------------------------------------------------------------
_SCHEMA = {
    "slots": {
        "headline": {"type": "text", "max_chars": 24, "label": "主标题", "required": True},
        "points": {"type": "json", "label": "要点", "default": ["要点一", "要点二"]},
    },
    "data_slots": ["points"],
}


class SlotSchemaTest(unittest.TestCase):
    def test_registry_from_gold_root_loads_slot_schema(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = make_gold_root_with_schema(Path(d), "key_takeaway", _SCHEMA)
            reg = animations.registry_from_gold_root(str(root))
            self.assertIn("schemas", reg)
            self.assertIn("key_takeaway", reg["schemas"])
            self.assertEqual(
                set(reg["schemas"]["key_takeaway"].get("slots", {}).keys()),
                {"headline", "points"},
            )

    def test_build_semantic_animation_fills_slots_per_schema(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = make_gold_root_with_schema(Path(d), "key_takeaway", _SCHEMA)
            reg = animations.registry_from_gold_root(str(root))
            shot = {
                "shot_id": "S1",
                "vocal_script": "真正的效率不是更快地点击，而是让工具主动理解你。",
                "duration_sec": 6.0,
            }
            out = animations.build_semantic_animation(shot, {}, reg)
            primary = out["primary"]
            self.assertEqual(primary["structure_id"], "key_takeaway")
            slots = primary["slots"]
            # keys match the schema exactly
            self.assertEqual(set(slots.keys()), {"headline", "points"})
            # text slot is a string carrying the exact phrase; list slot is a list
            self.assertIsInstance(slots["headline"], str)
            self.assertIn(primary["phrase"], json.dumps(slots, ensure_ascii=False))
            self.assertIsInstance(slots["points"], list)
            self.assertTrue(slots["points"])
            # max_chars respected on the text slot
            self.assertLessEqual(len(slots["headline"]), 24)

    def test_set_emphasis_writes_schema_slots(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = make_gold_root_with_schema(Path(d), "key_takeaway", _SCHEMA)
            proj = write_project(Path(d), str(root))
            operations.apply_operations(proj, [{
                "op": "set_emphasis", "shot_id": "S02",
                "phrase": "怎样让AI成为自己的工具", "emphasis_type": "key_point",
            }])
            deco = proj.assets_for("S02")["decoration"]
            self.assertEqual(deco["structure_id"], "key_takeaway")
            self.assertEqual(set(deco["slots"].keys()), {"headline", "points"})
            self.assertIsInstance(deco["slots"]["points"], list)
            self.assertIn("怎样让AI成为自己的工具",
                          json.dumps(deco["slots"], ensure_ascii=False))


# ---------------------------------------------------------------------------
# F2: verification template_ok is real (fail-closed when registry unreachable)
# ---------------------------------------------------------------------------
class TemplateOkVerificationTest(unittest.TestCase):
    def test_template_ok_false_for_id_not_in_registry(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = make_gold_root(Path(d))
            proj = write_project(Path(d), str(root))
            ops = [{"op": "set_emphasis", "shot_id": "S02",
                    "phrase": "怎样让AI成为自己的工具", "emphasis_type": "key_point"}]
            operations.apply_operations(proj, ops)
            # tamper the recorded structure id with one the registry does not have
            proj.shot("S02")["semantic_animations"]["primary"]["structure_id"] = "ghost_structure"
            checks = operations.verify_postconditions(proj, ops)
            emp = [c for c in checks if c["name"].endswith("set_emphasis")]
            self.assertTrue(emp and not emp[0]["ok"], str(checks))

    def test_template_ok_fail_closed_when_registry_unreachable(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))  # no gold root -> registry unreachable
            # craft a primary that *claims* real template ids
            proj.shot("S02")["semantic_animations"] = {
                "primary": {
                    "phrase": "怎样让AI成为自己的工具",
                    "type": "key_point",
                    "start_sec": 5.2,
                    "end_sec": 8.8,
                    "structure_id": "key_takeaway",
                    "motion_id": "slide_in_hold",
                    "visual_id": "dark_tech",
                },
                "anchors": [],
            }
            ops = [{"op": "set_emphasis", "shot_id": "S02",
                    "phrase": "怎样让AI成为自己的工具", "emphasis_type": "key_point"}]
            checks = operations.verify_postconditions(proj, ops)
            emp = [c for c in checks if c["name"].endswith("set_emphasis")]
            self.assertTrue(emp and not emp[0]["ok"], str(checks))
            self.assertIn("registry", emp[0]["detail"].lower())


# ---------------------------------------------------------------------------
# F1: auto-wiring — plan-level autodecorate + promotion + graph node behavior
# ---------------------------------------------------------------------------
def _cued_shot(sid: str, vocal: str, cue_text: str, *, spotlight: bool, dur: float = 8.0):
    shot = {"shot_id": sid, "segment_id": "seg1", "vocal_script": vocal,
            "subtitle_text": vocal, "duration_sec": dur}
    if spotlight:
        shot["is_spotlight"] = True
    return shot


def _cued_assets(cue_text: str, dur: float = 8.0):
    return {
        "audio_meta": {"duration_sec": dur},
        "subtitle_cues": [
            {"start": 0.0, "end": dur / 2, "text": "先做一个铺垫说明"},
            {"start": dur / 2, "end": dur, "text": cue_text},
        ],
    }


class AutoDecoratePlanTest(unittest.TestCase):
    """decorate_shot.autodecorate_shot_plan_0907 over the whole plan (F1/req.2)."""

    def _run(self, root):
        from pipeline.tools import decorate_shot
        user_primary = {
            "phrase": "用户锁定的短语", "type": "key_point", "source": "user",
            "start_sec": 1.0, "end_sec": 3.0, "structure_id": "quote_card",
            "motion_id": "mask_reveal", "visual_id": "dark_tech",
            "slots": {"headline": "用户锁定的短语"},
        }
        shot_plan = [
            _cued_shot("S1", "真正的效率不是更快地点击。", "真正的效率不是更快地点击", spotlight=True),
            _cued_shot("S2", "关键是必须主动使用工具。", "关键是必须主动使用工具", spotlight=False),
            _cued_shot("S3", "我们使用 GPT-4 完成任务。", "我们使用 GPT-4 完成任务", spotlight=False),
            _cued_shot("S4", "用户锁定的短语在这里。", "用户锁定的短语在这里", spotlight=True),
        ]
        shot_assets = {
            "S1": _cued_assets("真正的效率不是更快地点击"),
            "S2": _cued_assets("关键是必须主动使用工具"),
            "S3": _cued_assets("我们使用 GPT-4 完成任务"),
            "S4": {**_cued_assets("用户锁定的短语在这里"),
                   "semantic_animations": {"primary": copy.deepcopy(user_primary), "anchors": [copy.deepcopy(user_primary)]}},
        }
        config = {"policy_version": "0907", "paths": {"gold_template_root": str(root)}}
        frozen_user = copy.deepcopy(shot_assets["S4"]["semantic_animations"])
        res = decorate_shot.autodecorate_shot_plan_0907(shot_plan, shot_assets, config)
        return shot_plan, shot_assets, res, frozen_user

    def test_autodecorate_promotes_strong_preserves_user_skips_weak(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = make_gold_root(Path(d))
            shot_plan, shot_assets, res, frozen_user = self._run(root)
            self.assertTrue(res.get("enabled"))
            # spotlight shot decorated with a real id + non-zero timing
            s1 = shot_assets["S1"]["semantic_animations"]["primary"]
            self.assertIn(s1["structure_id"], _STRUCTURES)
            self.assertGreater(float(s1["end_sec"]), 0.0)
            # strong non-spotlight shot promoted + decorated
            self.assertTrue(shot_plan[1].get("is_spotlight"))
            self.assertIn("semantic_animations", shot_assets["S2"])
            # weak non-spotlight shot (proper noun only) not promoted / not decorated
            self.assertFalse(shot_plan[2].get("is_spotlight"))
            self.assertNotIn("semantic_animations", shot_assets["S3"])
            # user override preserved verbatim
            self.assertEqual(shot_assets["S4"]["semantic_animations"], frozen_user)

    def test_non_0907_is_a_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = make_gold_root(Path(d))
            from pipeline.tools import decorate_shot
            shot_plan = [_cued_shot("S1", "真正的效率不是更快地点击。", "真正的效率不是更快地点击", spotlight=True)]
            shot_assets = {"S1": _cued_assets("真正的效率不是更快地点击")}
            config = {"paths": {"gold_template_root": str(root)}}  # no 0907 opt-in
            res = decorate_shot.autodecorate_shot_plan_0907(shot_plan, shot_assets, config)
            self.assertFalse(res.get("enabled"))
            self.assertNotIn("semantic_animations", shot_assets["S1"])


class GraphDecorateNodeAutoWireTest(unittest.TestCase):
    """The real node_decorate_spotlights must auto-wire 0907 semantic anchors."""

    def _state(self, root, tmp, *, opt_in: bool):
        user_primary = {
            "phrase": "用户锁定的短语", "type": "key_point", "source": "user",
            "start_sec": 1.0, "end_sec": 3.0, "structure_id": "quote_card",
            "motion_id": "mask_reveal", "visual_id": "dark_tech",
            "slots": {"headline": "用户锁定的短语"},
        }
        config = {
            "paths": {"gold_template_root": str(root)},
            "execution": {"mode": "fake", "fake_mode": {"skip_hyperframes_render": True}},
        }
        if opt_in:
            config["policy_version"] = "0907"
        shot_plan = [
            _cued_shot("S1", "真正的效率不是更快地点击。", "真正的效率不是更快地点击", spotlight=True),
            _cued_shot("S2", "关键是必须主动使用工具。", "关键是必须主动使用工具", spotlight=False),
            _cued_shot("S4", "用户锁定的短语在这里。", "用户锁定的短语在这里", spotlight=True),
        ]
        shot_assets = {
            "S1": _cued_assets("真正的效率不是更快地点击"),
            "S2": _cued_assets("关键是必须主动使用工具"),
            "S4": {**_cued_assets("用户锁定的短语在这里"),
                   "semantic_animations": {"primary": copy.deepcopy(user_primary),
                                           "anchors": [copy.deepcopy(user_primary)]}},
        }
        return {
            "config": config,
            "shot_plan": shot_plan,
            "shot_assets": shot_assets,
            "output_dir": str(tmp / "out"),
        }, copy.deepcopy(shot_assets["S4"]["semantic_animations"])

    def _run_node(self, state):
        import unittest.mock as mock
        from pipeline import graph

        class _LLM:
            def close(self):
                pass

        with mock.patch.object(graph, "_get_llm", lambda *a, **k: _LLM()), \
             mock.patch.object(graph.decorate_shot, "batch_run",
                               lambda *a, **k: {"results": {}, "summary": "ok", "status": "success"}), \
             mock.patch.object(graph.fake_utils, "fake_render_success",
                               lambda *a, **k: {"timeline": {}, "rendered_path": "",
                                                "render_status": "ok", "render_success": True}), \
             mock.patch.object(graph, "_save_storage_root_decoration", lambda *a, **k: None), \
             mock.patch.object(graph, "_save_json_artifact", lambda *a, **k: ""), \
             mock.patch.object(graph, "_emit_live_trajectory", lambda *a, **k: None):
            return graph.node_decorate_spotlights(state)

    def test_0907_node_auto_wires_and_preserves_user(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            root = make_gold_root(tmp)
            (tmp / "out").mkdir(parents=True, exist_ok=True)
            state, frozen_user = self._state(root, tmp, opt_in=True)
            result = self._run_node(state)
            assets = result["shot_assets"]
            # auto anchor exists on the spotlight shot, timing not uniformly 0
            s1 = assets["S1"]["semantic_animations"]["primary"]
            self.assertIn(s1["structure_id"], _STRUCTURES)
            self.assertGreater(float(s1["end_sec"]), 0.0)
            # strong non-spotlight shot promoted through the graph node
            plan = {s["shot_id"]: s for s in result.get("shot_plan", state["shot_plan"])}
            self.assertTrue(plan["S2"].get("is_spotlight"))
            self.assertIn("semantic_animations", assets["S2"])
            # user anchor untouched
            self.assertEqual(assets["S4"]["semantic_animations"], frozen_user)

    def test_non_0907_node_does_not_touch_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            root = make_gold_root(tmp)
            (tmp / "out").mkdir(parents=True, exist_ok=True)
            state, _ = self._state(root, tmp, opt_in=False)
            result = self._run_node(state)
            assets = result["shot_assets"]
            self.assertNotIn("semantic_animations", assets["S1"])
            self.assertNotIn("semantic_animations", assets["S2"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
