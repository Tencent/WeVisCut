"""End-to-end acceptance test for the newvis0907 editing surface (Task 10).

Run: python3 tests/test_newvis0907_acceptance.py

This is the user's *original* acceptance case, driven through the real 0907
router + transaction manager + verification stack (no model, no ffmpeg, no
HyperFrames — a tiny on-disk Gold Template registry stands in for the real
library and the regeneration runner is a fake that writes *fresh* shot / final
files):

    在精确的章 scope 内，一次事务里同时
      * remove 掉旧的“真正的效率”动画（S01），
      * 在“怎样让 AI 成为自己的工具”的真实 cue 时间给 S02 加一个 key_point，
    并且只有在 render -> verify succeeded 之后才算“修改成功”。

Asserted invariants:
  * 单事务（一个 transaction 覆盖两个 op）;
  * succeeded 只在 render 写了新文件 *且* verify 通过后出现;
  * 旧动画消失、新 key_point 的 decor_start 非 0（来自真实 cue）;
  * 旁白 / 字幕 / footage 完全不变、不产生任何幻影 S08;
  * render 没有真正产出新文件（no-change）时事务落在 verification_failed，
    而不是 succeeded。
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from studio import project as project_mod  # noqa: E402
from studio.newvis0907 import operations  # noqa: E402
from studio.newvis0907 import router as router_mod  # noqa: E402


# --------------------------------------------------------------------------- #
# tiny real (temp) Gold Template registry so set_emphasis resolves real ids
# --------------------------------------------------------------------------- #
_STRUCTURES = ("quote_card", "key_takeaway", "thesis_statement", "three_points",
               "stat_highlight", "bar_chart", "timeline_step", "definition_card")
_MOTIONS = ("scale_pop_hold", "slide_in_hold", "mask_reveal")
_VISUALS = ("dark_tech", "clean_saas", "warm_editorial")
_MARKER = {"structures": "structure.html", "motions": "motion.js", "visuals": "tokens.css"}


def _make_gold_root(tmp: Path) -> Path:
    root = tmp / "gold_template"
    for layer, ids in (("structures", _STRUCTURES), ("motions", _MOTIONS),
                       ("visuals", _VISUALS)):
        for sid in ids:
            d = root / "generated" / layer / sid
            d.mkdir(parents=True, exist_ok=True)
            (d / _MARKER[layer]).write_text("x", encoding="utf-8")
    return root


# --------------------------------------------------------------------------- #
# fixture project: chapter 2 (seg_future) holds S01/S02 in the edit scope
# --------------------------------------------------------------------------- #
def _state_payload() -> dict:
    return {
        "project_id": "proj_accept",
        "input_text": "原文……",
        "final_video_path": "/tmp/does-not-exist/final.mp4",
        "segments": [
            {"segment_id": "seg_intro", "title": "序章"},
            {"segment_id": "seg_future", "title": "未来图景"},
        ],
        "shot_plan": [
            {
                "shot_id": "S01",
                "segment_id": "seg_future",
                "duration_sec": 6.0,
                "vocal_script": "真正的效率不是更快地点击。",
                "subtitle_text": "真正的效率不是更快地点击。",
                "subtitle_display_text": "真正的效率不是更快地点击。",
                "is_spotlight": True,
                "semantic_animations": {
                    "primary": {"type": "key_point", "phrase": "真正的效率"},
                    "anchors": [{"phrase": "真正的效率", "type": "key_point"}],
                },
            },
            {
                "shot_id": "S02",
                "segment_id": "seg_future",
                "duration_sec": 8.8,
                "vocal_script": "怎样让AI成为自己的工具，是我们要回答的问题。",
                "subtitle_text": "怎样让AI成为自己的工具，是我们要回答的问题。",
                "subtitle_display_text": "怎样让AI成为自己的工具，是我们要回答的问题。",
                "subtitle_cues": [
                    {"start": 0.0, "end": 5.2, "text": "先做一个铺垫说明"},
                    {"start": 5.2, "end": 8.8, "text": "怎样让AI成为自己的工具"},
                ],
            },
        ],
        "shot_assets": {
            "S01": {
                "decoration": {"structure_id": "quote_card",
                               "slots": {"headline": "真正的效率"}},
                "clip_meta": {"visual_playlist": {"clips": [
                    {"subclip_id": "S01::subclip::1", "clip_id": "c1a",
                     "path": "/clips/c1a.mp4", "used_duration_sec": 6.0, "query": "原始A"},
                ]}},
            },
            "S02": {
                "clip_meta": {"visual_playlist": {"clips": [
                    {"subclip_id": "S02::subclip::1", "clip_id": "c2a",
                     "path": "/clips/c2a.mp4", "used_duration_sec": 8.8, "query": "原始B"},
                ]}},
            },
        },
    }


def _make_project(tmp: Path, gold_root: str) -> project_mod.Project:
    (tmp / "final_state.json").write_text(
        json.dumps(_state_payload(), ensure_ascii=False), encoding="utf-8"
    )
    pipeline_config = {"paths": {"gold_template_root": gold_root}}
    return project_mod.load(tmp, pipeline_config=pipeline_config)


def _payload_stub(project) -> dict:
    return {
        "project_id": project.project_id,
        "project_dir": str(project.project_dir),
        "final_video_path": project.final_video_path,
        "dirty": {sid: sorted(r) for sid, r in project.dirty.items()},
    }


def _fresh_render_runner(outdir: Path):
    """A runner that actually writes new per-shot + final files (real change)."""

    def _exec(project, tx) -> dict:
        outputs: list[str] = []
        for sid in sorted(operations.target_shot_ids(project, tx.ops)):
            out = outdir / f"regen_{tx.id}_{sid}.mp4"
            out.write_bytes(os.urandom(64))
            project.shot_assets.setdefault(sid, {})["rendered_path"] = str(out)
            outputs.append(str(out))
        final = outdir / f"final_{tx.id}.mp4"
        final.write_bytes(os.urandom(96))
        project.state["final_video_path"] = str(final)
        project.clear_dirty()
        return {"outputs": outputs, "final_video_path": str(final)}

    def _runner(mgr, tx) -> None:
        mgr.execute(tx, _exec)

    return _runner


def _nochange_runner(mgr, tx) -> None:
    """Reports render success but never writes a fresh file (no real change)."""

    def _exec(project, _tx) -> dict:
        project.clear_dirty()
        return {"outputs": [], "final_video_path": project.final_video_path}

    mgr.execute(tx, _exec)


def _build(tmp: Path, gold_root: str, runner):
    project = _make_project(tmp, gold_root)
    deps = router_mod.RouterDeps(
        load_project=lambda d: project,
        project_payload=_payload_stub,
        discover_projects=lambda: [{"project_dir": str(tmp), "project_id": "proj_accept"}],
    )
    # NB: plan_edit is deliberately the *real* rule-based planner (RouterDeps
    # default), so this acceptance goes locator -> plan -> ops for real.
    deps.schedule_runner = runner
    app = FastAPI()
    router_mod.configure_router(deps)
    app.include_router(router_mod.router)
    return TestClient(app), project, str(tmp)


# The user's *natural language* request (chapter-scoped): remove the old
# "真正的效率" animation and emphasise "怎样让AI成为自己的工具". The planner turns
# this into exactly two ops in ONE transaction.
_NL_MESSAGE = "在《未来图景》这章，把动画“真正的效率”删掉，并强调“怎样让AI成为自己的工具”"


class AcceptanceChapterScopeTest(unittest.TestCase):
    def test_chat_nl_success_only_after_render_and_verification(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            root = _make_gold_root(tmp)
            outdir = tmp / "out"
            outdir.mkdir()
            client, proj, pdir = _build(tmp, str(root), _fresh_render_runner(outdir))

            before_ids = proj.shot_ids()
            narration_s02 = proj.shot("S02")["vocal_script"]
            subtitle_s02 = proj.shot("S02")["subtitle_text"]
            footage_s02 = copy.deepcopy(proj.assets_for("S02")["clip_meta"])

            # drive the REAL chat endpoint with natural language.
            r = client.post("/api/0907/project/chat",
                            json={"project_dir": pdir, "message": _NL_MESSAGE})
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()

            # planner produced an edit intent with exactly the two scoped ops.
            self.assertEqual(body["intent"], "edit", body)
            ops = body["ops"]
            self.assertEqual([o["op"] for o in ops],
                             ["remove_animation", "set_emphasis"], ops)
            self.assertEqual(ops[0]["shot_id"], "S01", ops)
            self.assertEqual(ops[1]["shot_id"], "S02", ops)
            # scope really narrowed to the requested chapter.
            self.assertEqual(body.get("scope", {}).get("segment_id"), "seg_future", body.get("scope"))

            # single transaction covering both ops.
            tx = body["transaction"]
            self.assertEqual(len(tx["ops"]), 2, tx)

            # succeeded ONLY after render wrote fresh files and verify passed.
            self.assertEqual(tx["state"], "succeeded", json.dumps(tx, ensure_ascii=False))
            self.assertTrue(tx["verification"]["complete"], tx["verification"])

            # old "真正的效率" animation is gone on S01.
            sem1 = proj.shot("S01").get("semantic_animations", {})
            self.assertFalse(sem1.get("primary", {}).get("phrase"))
            self.assertFalse(any(a.get("phrase") == "真正的效率"
                                 for a in sem1.get("anchors", [])))

            # new key_point on S02 with real ids and a non-zero cue-based start.
            deco2 = proj.assets_for("S02")["decoration"]
            self.assertIn(deco2["structure_id"], _STRUCTURES)
            self.assertGreaterEqual(deco2["timing"]["decor_start_sec"], 5.0)

            # narration / subtitle / footage completely unchanged on S02.
            self.assertEqual(proj.shot("S02")["vocal_script"], narration_s02)
            self.assertEqual(proj.shot("S02")["subtitle_text"], subtitle_s02)
            self.assertEqual(proj.assets_for("S02")["clip_meta"], footage_s02)

            # no phantom shot ever appears (never an S08).
            self.assertEqual(proj.shot_ids(), before_ids)
            self.assertNotIn("S08", proj.shot_ids())

    def test_chat_nl_render_no_change_does_not_succeed(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            root = _make_gold_root(tmp)
            client, proj, pdir = _build(tmp, str(root), _nochange_runner)

            r = client.post("/api/0907/project/chat",
                            json={"project_dir": pdir, "message": _NL_MESSAGE})
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            self.assertEqual(body["intent"], "edit", body)
            tx = body["transaction"]

            # a runner that produced no fresh render must NOT be reported succeeded.
            self.assertNotEqual(tx["state"], "succeeded", tx)
            self.assertEqual(tx["state"], "verification_failed", tx)
            self.assertFalse(tx["verification"]["complete"], tx["verification"])
            failed = [c for c in tx["verification"]["checks"] if not c["ok"]]
            self.assertTrue(
                any("render_written" in c["name"] or "final_written" in c["name"]
                    for c in failed),
                failed,
            )

    def test_chat_unlocatable_phrase_clarifies_without_mutation(self) -> None:
        # A natural-language request whose phrase is not in any narration cannot
        # be located; the turn must clarify (no ops, no transaction) and leave
        # the project completely untouched — never an optimistic success.
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            root = _make_gold_root(tmp)
            outdir = tmp / "out"
            outdir.mkdir()
            client, proj, pdir = _build(tmp, str(root), _fresh_render_runner(outdir))
            before = copy.deepcopy(proj.state)

            msg = "强调“这句话不存在于任何一条旁白当中哦”"
            r = client.post("/api/0907/project/chat",
                            json={"project_dir": pdir, "message": msg})
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            self.assertNotEqual(body.get("intent"), "edit", body)
            self.assertFalse(body.get("applied"), body)
            self.assertIsNone(body.get("transaction"), body)
            self.assertNotIn("succeeded", r.text)
            self.assertEqual(proj.state, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
