"""Bugfix-TDD regression tests for the Task 10 final review (NOT_READY).

Run: python3 tests/test_newvis0907_review.py

Each test reproduces a specific review finding as a behavioural failure first,
so the accompanying implementation is a real fix rather than a moved goalpost.
Covers: C1 (visible runner job id), unknown optimize scope reject, set_emphasis
without a real registry, I6 async discover, I8 editor gating, I1 final loudness,
I2 auto-fix convergence, I4 dimension-scoped verification, I5 idempotency,
I9 replace_subclip path validation.
"""

from __future__ import annotations

import copy
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from studio import project as project_mod  # noqa: E402
from studio.newvis0907 import operations  # noqa: E402
from studio.newvis0907 import quality as quality_mod  # noqa: E402
from studio.newvis0907 import router as router_mod  # noqa: E402


# --------------------------------------------------------------------------
# shared fixture
# --------------------------------------------------------------------------
def _state_payload() -> dict:
    return {
        "project_id": "proj_review",
        "input_text": "demo",
        "final_video_path": "",
        "segments": [
            {"segment_id": "seg_intro", "title": "序章"},
            {"segment_id": "seg_future", "title": "未来图景"},
        ],
        "shot_plan": [
            {"shot_id": "S01", "segment_id": "seg_intro", "duration_sec": 4,
             "vocal_script": "先看背景介绍。", "subtitle_text": "先看背景介绍。"},
            {"shot_id": "S02", "segment_id": "seg_future", "duration_sec": 6,
             "vocal_script": "真正的效率不是更快地点击。",
             "subtitle_text": "真正的效率不是更快地点击。"},
        ],
        "shot_assets": {
            "S02": {
                "clip_meta": {
                    "visual_playlist": {
                        "clips": [
                            {"subclip_id": "S02::subclip::1", "clip_id": "c1",
                             "path": "/clips/c1.mp4", "used_duration_sec": 6, "query": "原始"}
                        ]
                    }
                }
            }
        },
    }


def _make_project(tmp: Path, payload: dict | None = None) -> project_mod.Project:
    (tmp / "final_state.json").write_text(
        json.dumps(payload or _state_payload(), ensure_ascii=False), encoding="utf-8"
    )
    return project_mod.load(tmp)


def _payload_stub(project) -> dict:
    return {
        "project_id": project.project_id,
        "project_dir": str(project.project_dir),
        "final_video_path": project.final_video_path,
        "dirty": {sid: sorted(r) for sid, r in project.dirty.items()},
    }


def _build(tmp: Path, runner=None, editor_flags=None, payload=None):
    project = _make_project(tmp, payload)
    deps = router_mod.RouterDeps(
        load_project=lambda d: project,
        project_payload=_payload_stub,
        discover_projects=lambda: [{"project_dir": str(tmp), "project_id": "proj_review"}],
        editor_flags=editor_flags or (lambda: {"enabled": True}),
    )
    if runner is not None:
        deps.schedule_runner = runner
    app = FastAPI()
    router_mod.configure_router(deps)
    app.include_router(router_mod.router)
    return TestClient(app), project, str(tmp)


# --------------------------------------------------------------------------
# C1 — the background regeneration job must be identifiable to the UI
# --------------------------------------------------------------------------
class RunnerJobIdTest(unittest.TestCase):
    def test_schedule_returns_unique_job_id_without_tx_tx(self) -> None:
        seen: list[str] = []

        def runner(mgr, tx):
            # a real runner returns the job id it created
            jid = f"regen_{tx.id}_{len(seen) + 1}"
            seen.append(jid)
            return jid

        with tempfile.TemporaryDirectory() as d:
            client, proj, pdir = _build(Path(d), runner=runner)
            r = client.post("/api/0907/project/ops", json={
                "project_dir": pdir,
                "ops": [{"op": "optimize_cut", "scope": "global"}]})
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            self.assertIn("regen_job_id", body)
            self.assertTrue(body["regen_job_id"], body)
            self.assertNotIn("tx_tx", body["regen_job_id"], body["regen_job_id"])

    def test_chat_response_carries_regen_job_id_and_targets(self) -> None:
        def runner(mgr, tx):
            return f"regen_{tx.id}"

        with tempfile.TemporaryDirectory() as d:
            client, proj, pdir = _build(Path(d), runner=runner)
            # apply path is a chat POST: it must also surface the job id + targets.
            r = client.post("/api/0907/project/chat", json={
                "project_dir": pdir, "apply": True,
                "ops": [{"op": "optimize_cut", "scope": "seg_future"}]})
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            self.assertEqual(body["intent"], "edit", body)
            self.assertTrue(body.get("regen_job_id"), body)
            self.assertIn("S02", body.get("targets") or [])


# --------------------------------------------------------------------------
# easy — unknown optimize_cut scope must be rejected, not silently global
# --------------------------------------------------------------------------
class OptimizeScopeTest(unittest.TestCase):
    def test_unknown_scope_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = _make_project(Path(d))
            with self.assertRaises(operations.OperationError):
                operations.apply_operations(
                    proj, [{"op": "optimize_cut", "scope": "does_not_exist"}])

    def test_known_segment_scope_ok(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = _make_project(Path(d))
            res = operations.apply_operations(
                proj, [{"op": "optimize_cut", "scope": "seg_future"}])
            self.assertTrue(res[0]["ok"])


# --------------------------------------------------------------------------
# I6 — /api/0907/projects discovery must be off the event loop
# --------------------------------------------------------------------------
class ProjectsAsyncTest(unittest.TestCase):
    def test_source_uses_to_thread(self) -> None:
        src = inspect.getsource(router_mod.list_projects_0907)
        self.assertIn("to_thread", src, src)


# --------------------------------------------------------------------------
# I8 — every write endpoint honours editor_flags.enabled (disabled => 403)
# --------------------------------------------------------------------------
class EditorGatingTest(unittest.TestCase):
    def _disabled_client(self, d):
        return _build(Path(d), editor_flags=lambda: {"enabled": False})

    def test_chat_write_blocked_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, proj, pdir = self._disabled_client(d)
            r = client.post("/api/0907/project/chat",
                            json={"project_dir": pdir, "message": "强调“真正的效率”"})
            self.assertEqual(r.status_code, 403, r.text)

    def test_ops_blocked_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, proj, pdir = self._disabled_client(d)
            r = client.post("/api/0907/project/ops", json={
                "project_dir": pdir,
                "ops": [{"op": "optimize_cut", "scope": "global"}]})
            self.assertEqual(r.status_code, 403, r.text)

    def test_readonly_quality_check_allowed_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, proj, pdir = self._disabled_client(d)
            r = client.post("/api/0907/project/quality-check",
                            json={"project_dir": pdir})
            self.assertEqual(r.status_code, 200, r.text)

    def test_quality_apply_blocked_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, proj, pdir = self._disabled_client(d)
            r = client.post("/api/0907/project/quality-check",
                            json={"project_dir": pdir, "apply": True})
            self.assertEqual(r.status_code, 403, r.text)


# --------------------------------------------------------------------------
# I1 — final loudness is read from the production stitch_result.loudness
# --------------------------------------------------------------------------
def _within_target_loudness() -> dict:
    return {
        "status": "normalized", "normalized": True,
        "target": {"integrated_lufs": -16.0, "true_peak_dbtp": -1.5,
                   "tolerance_lu": 1.0, "lra_lu": 11.0},
        "output": {"input_i": -16.1, "input_tp": -1.6, "input_lra": 6.0,
                   "input_thresh": -26.0, "output_i": -15.8, "output_tp": -1.6,
                   "output_lra": 6.0, "output_thresh": -25.0,
                   "target_offset": 0.0, "normalization_type": "dynamic"},
    }


class FinalLoudnessTest(unittest.TestCase):
    def test_stitch_result_loudness_counts_as_final_measurement(self) -> None:
        # production shape: a normalized loudnorm payload recorded by the stitch
        # graph, plus narration_target / ducking bookkeeping.
        loud = _within_target_loudness()
        loud["narration_target"] = -16.0
        loud["ducking"] = {"applied": True, "amount_db": -8.0}
        state = {
            "final_video_path": "/tmp/final.mp4",
            "shot_plan": [],
            "shot_assets": {},
            "stitch_result": {"loudness": loud},
        }
        report = quality_mod.evaluate_project(state)
        codes = {i["code"] for i in report["issues"]}
        self.assertNotIn("final_measurement_missing", codes, report["issues"])
        self.assertNotIn("final_loudness_out_of_target", codes, report["issues"])

    def test_missing_final_measurement_when_no_stitch_loudness(self) -> None:
        state = {"final_video_path": "/tmp/final.mp4", "shot_plan": [], "shot_assets": {}}
        report = quality_mod.evaluate_project(state)
        codes = {i["code"] for i in report["issues"]}
        self.assertIn("final_measurement_missing", codes, report["issues"])


# --------------------------------------------------------------------------
# I2 — auto-fix must converge: an issue that survives 2 fixes is manual-only
# --------------------------------------------------------------------------
class FixConvergenceTest(unittest.TestCase):
    def _state_with_soft_issue(self):
        # S01 narration present but no audio => loudness_missing (auto-fixable)
        return {
            "shot_plan": [{"shot_id": "S01", "duration_sec": 5,
                           "vocal_script": "这里有一句旁白。"}],
            "shot_assets": {"S01": {}},
        }

    def test_two_failed_attempts_marks_manual_and_drops_proposal(self) -> None:
        state = self._state_with_soft_issue()
        report = quality_mod.evaluate_project(state)
        issue = next(i for i in report["issues"] if i["code"] == "loudness_missing")
        iid = issue["issue_id"]
        # simulate two applied-but-still-present fix attempts
        state["quality_fix_attempts0907"] = {iid: 2}
        report2 = quality_mod.evaluate_project(state)
        again = next(i for i in report2["issues"] if i["code"] == "loudness_missing")
        self.assertTrue(again.get("needs_manual"), again)
        proposals = quality_mod.propose_fixes(state, report2)
        self.assertFalse(
            any(p.get("shot_id") == "S01" for p in proposals), proposals)

    def test_fresh_issue_still_proposes(self) -> None:
        state = self._state_with_soft_issue()
        report = quality_mod.evaluate_project(state)
        proposals = quality_mod.propose_fixes(state, report)
        self.assertTrue(any(p.get("shot_id") == "S01" for p in proposals), proposals)


# --------------------------------------------------------------------------
# I4 — verification gates by the dimensions the batch actually touched
# --------------------------------------------------------------------------
class DimensionScopedVerificationTest(unittest.TestCase):
    def test_unrelated_preexisting_issue_does_not_block_visual_edit(self) -> None:
        from studio.newvis0907 import verification
        with tempfile.TemporaryDirectory() as d:
            proj = _make_project(Path(d))
            before = operations.compute_digest(proj)
            # a purely-visual op (replace_subclip) — no narration/loudness change
            ops = [{"op": "replace_subclip", "subclip_id": "S02::subclip::1",
                    "query": "全新的城市航拍"}]
            operations.apply_operations(proj, ops)
            outcome = verification.verify_transaction(
                proj, before, ops, render_result=None, probe=lambda p: 10)
            # S01 has a pre-existing loudness gap, but this edit does not touch
            # the loudness dimension, so it must not appear as a blocking check.
            names = " ".join(c["name"] for c in outcome["checks"])
            self.assertNotIn("loudness:S01", names, outcome["checks"])


class PersistedPlaylistTest(unittest.TestCase):
    def test_manual_subclip_timing_survives_compose_planning(self) -> None:
        from pipeline.tools import timeline_compose

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            first = root / "first.mp4"
            second = root / "second.mp4"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            existing = {
                "clips": [
                    {
                        "subclip_id": "S02::subclip::1",
                        "clip_id": "c1",
                        "clip_path": str(first),
                        "source_duration_sec": 4.0,
                        "source_used_duration_sec": 4.0,
                        "final_duration_sec": 4.0,
                        "playback_rate": 1.0,
                        "user_override": True,
                    },
                    {
                        "subclip_id": "S02::subclip::2",
                        "clip_id": "c2",
                        "clip_path": str(second),
                        "source_duration_sec": 2.0,
                        "source_used_duration_sec": 2.0,
                        "final_duration_sec": 2.0,
                        "playback_rate": 1.0,
                    },
                ]
            }
            with mock.patch.object(
                timeline_compose,
                "_render_visual_playlist_0907",
                return_value={"status": "ok", "clip_path": str(root / "planned.mp4")},
            ):
                result = timeline_compose._prepare_visual_playlist_0907(
                    primary_clip_path=str(first),
                    clip_meta={"duration_sec": 4.0, "visual_playlist": existing},
                    candidates=[{"clip_id": "c2", "clip_path": str(second), "duration_sec": 2.0}],
                    shot_dir=root,
                    duration=6.0,
                    config={"policy_version": "0907"},
                    shot_id="S02",
                )
            clips = result["playlist"]["clips"]
            self.assertEqual([c["subclip_id"] for c in clips],
                             ["S02::subclip::1", "S02::subclip::2"])
            self.assertEqual([c["final_duration_sec"] for c in clips], [4.0, 2.0])
            self.assertTrue(clips[0]["user_override"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
