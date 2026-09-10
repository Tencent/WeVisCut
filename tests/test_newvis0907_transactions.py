"""Offline tests for the newvis0907 operations / transaction / verification core.

Run: python3 tests/test_newvis0907_transactions.py

These tests use throwaway on-disk projects (a written ``final_state.json`` loaded
through :mod:`studio.project`) and synchronous fake executors, so they never
touch the network filesystem scan or start a model.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studio import project as project_mod  # noqa: E402
from studio.newvis0907 import operations, verification  # noqa: E402
from studio.newvis0907.transactions import (  # noqa: E402
    TransactionError,
    TransactionManager,
)


def _state_payload() -> dict:
    return {
        "project_id": "proj_demo",
        "input_text": "demo",
        "final_video_path": "/tmp/does-not-exist/old_final.mp4",
        "segments": [
            {"segment_id": "seg_intro", "title": "序章"},
            {"segment_id": "seg_future", "title": "未来图景"},
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
                "segment_id": "seg_future",
                "duration_sec": 6,
                "vocal_script": "真正的效率不是更快地点击。",
                "subtitle_text": "真正的效率不是更快地点击。",
                "semantic_animations": {
                    "primary": {"type": "spotlight", "phrase": "真正的效率"},
                    "anchors": [{"phrase": "真正的效率", "type": "spotlight"}],
                },
            },
            {
                "shot_id": "S03",
                "segment_id": "seg_future",
                "duration_sec": 5,
                "vocal_script": "让工具主动理解你。",
                "subtitle_text": "让工具主动理解你。",
            },
        ],
        "shot_assets": {
            "S02": {
                "decoration": {
                    "structure_id": "struct_a",
                    "slots": {"headline": "真正的效率"},
                },
                "clip_meta": {
                    "visual_playlist": {
                        "clips": [
                            {
                                "subclip_id": "S02::subclip::1",
                                "clip_id": "c1",
                                "path": "/clips/c1.mp4",
                                "used_duration_sec": 6,
                                "query": "原始查询",
                            }
                        ]
                    }
                },
            },
            "S03": {
                "clip_meta": {
                    "visual_playlist": {
                        "clips": [
                            {
                                "subclip_id": "S03::subclip::1",
                                "clip_id": "c2",
                                "path": "/clips/c2.mp4",
                                "used_duration_sec": 5,
                                "query": "近景",
                            }
                        ]
                    }
                }
            },
        },
    }


def write_project(tmp: Path) -> project_mod.Project:
    (tmp / "final_state.json").write_text(
        json.dumps(_state_payload(), ensure_ascii=False), encoding="utf-8"
    )
    return project_mod.load(tmp)


def success_executor_factory(outdir: Path, fingerprint: str = "fp-new"):
    """Return an executor that writes fresh per-shot renders and a new final.

    It sets each targeted shot's ``rendered_path`` to a newly-written file and
    points ``final_video_path`` at a new final, so the real file-signal
    verification sees genuinely fresh outputs.
    """

    def _exec(project: project_mod.Project, tx) -> dict:
        outputs: list[str] = []
        for sid in sorted(operations.target_shot_ids(project, tx.ops)):
            out = outdir / f"regen_{tx.id}_{sid}.mp4"
            out.write_bytes(b"x" * 32)
            assets = project.shot_assets.setdefault(sid, {})
            assets["rendered_path"] = str(out)
            assets["audio_meta"] = {
                "duration_sec": float(project.shot(sid).get("duration_sec", 0) or 0.0),
                "loudness": {
                    "status": "normalized",
                    "normalized": True,
                    "target": {
                        "integrated_lufs": -16.0,
                        "true_peak_dbtp": -1.5,
                        "tolerance_lu": 1.0,
                        "lra_lu": 11.0,
                    },
                    "output": {"output_i": -16.1, "output_tp": -1.6},
                },
            }
            outputs.append(str(out))
        final = outdir / f"final_{tx.id}.mp4"
        final.write_bytes(b"y" * 64)
        project.state["final_video_path"] = str(final)
        project.clear_dirty()
        return {
            "outputs": outputs,
            "final_video_path": str(final),
            "fingerprint": fingerprint,
            "version": "v2",
        }

    return _exec


def boom_executor(project: project_mod.Project, tx) -> dict:
    raise RuntimeError("render boom")


def restitch_executor_factory(outdir: Path):
    """A restitch-only executor: writes a fresh final video, no per-shot render.

    This mirrors what actually happens for structural ops (reorder/delete) and
    bgm changes — the concat/mix is rebuilt but no shot clip is re-rendered.
    """

    def _exec(project: project_mod.Project, tx) -> dict:
        final = outdir / f"final_{tx.id}.mp4"
        final.write_bytes(b"z" * 128)
        project.state["final_video_path"] = str(final)
        project.clear_dirty()
        return {"outputs": [], "final_video_path": str(final)}

    return _exec


class OperationsTest(unittest.TestCase):
    def test_set_shot_duration_mutates_and_marks_render(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))
            results = operations.apply_operations(
                proj, [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 8.0}]
            )
            self.assertTrue(all(r["ok"] for r in results))
            self.assertEqual(proj.shot("S01")["duration_sec"], 8.0)
            self.assertIn("render", proj.dirty.get("S01", set()))

    def test_set_emphasis_writes_primary_and_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))
            operations.apply_operations(
                proj,
                [{
                    "op": "set_emphasis",
                    "shot_id": "S03",
                    "phrase": "让工具主动理解你",
                    "emphasis_type": "key_point",
                }],
            )
            sem = proj.shot("S03")["semantic_animations"]
            self.assertEqual(sem["primary"]["phrase"], "让工具主动理解你")
            self.assertEqual(sem["primary"]["type"], "key_point")
            self.assertTrue(any(a.get("phrase") == "让工具主动理解你" for a in sem["anchors"]))
            self.assertIn("render", proj.dirty.get("S03", set()))

    def test_remove_animation_clears_matching_phrase_only(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))
            before_shots = len(proj.shot_plan)
            operations.apply_operations(
                proj,
                [{"op": "remove_animation", "shot_id": "S02", "phrase": "真正的效率"}],
            )
            sem = proj.shot("S02").get("semantic_animations", {})
            # primary removed / emptied, anchors no longer contain the phrase
            self.assertFalse(sem.get("primary", {}).get("phrase"))
            self.assertFalse(any(a.get("phrase") == "真正的效率" for a in sem.get("anchors", [])))
            # shot itself is not deleted
            self.assertEqual(len(proj.shot_plan), before_shots)
            self.assertIsNotNone(proj.shot("S02"))

    def test_replace_and_timing_edit_subclip(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))
            operations.apply_operations(
                proj,
                [{"op": "replace_subclip", "subclip_id": "S02::subclip::1", "query": "新查询"}],
            )
            clip = proj.shot_assets["S02"]["clip_meta"]["visual_playlist"]["clips"][0]
            self.assertEqual(clip["query"], "新查询")
            self.assertTrue(clip.get("replaced"))
            self.assertIn("render", proj.dirty.get("S02", set()))

            operations.apply_operations(
                proj,
                [{"op": "set_subclip_timing", "subclip_id": "S03::subclip::1", "duration_sec": 2.5}],
            )
            clip3 = proj.shot_assets["S03"]["clip_meta"]["visual_playlist"]["clips"][0]
            self.assertAlmostEqual(float(clip3["used_duration_sec"]), 2.5, places=4)

            operations.apply_operations(
                proj,
                [{"op": "set_subclip_timing", "subclip_id": "S03::subclip::1", "mode": "auto"}],
            )
            self.assertNotIn("used_duration_sec", clip3)
            self.assertNotIn("final_duration_sec", clip3)
            self.assertFalse(bool(clip3.get("user_override")))

    def test_replace_footage_query_resets_stale_playlist_units(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))
            # Mimic a previously edited pinned subclip that must not leak after
            # a shot-level replace_footage query.
            clips = proj.shot_assets["S02"]["clip_meta"]["visual_playlist"]["clips"]
            clips[0]["replaced"] = True
            clips[0]["user_override"] = True
            proj.shot_assets["S02"]["retrieval_candidates"] = [{"clip_id": "old", "clip_path": "/clips/old.mp4"}]
            operations.apply_operations(
                proj,
                [{"op": "replace_footage", "shot_id": "S02", "query": "海边落日"}],
            )
            self.assertEqual(proj.shot("S02")["visual_query"], "海边落日")
            self.assertEqual(proj.shot_assets["S02"].get("retrieval_candidates"), [])
            self.assertNotIn("visual_playlist", proj.shot_assets["S02"]["clip_meta"])
            self.assertNotIn("visual_playlist", proj.shot_assets["S02"])
            self.assertIn("footage", proj.dirty.get("S02", set()))

    def test_optimize_cut_records_request_and_marks_render(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))
            operations.apply_operations(proj, [{"op": "optimize_cut", "scope": "global"}])
            requests = proj.state.get("quality_requests0907") or []
            self.assertTrue(requests)
            self.assertEqual(requests[-1]["scope"], "global")
            self.assertTrue(proj.dirty)  # some shot marked render dirty

    def test_batch_is_atomic_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))
            before_state = copy.deepcopy(proj.state)
            before_dirty = {k: set(v) for k, v in proj.dirty.items()}
            before_hist = copy.deepcopy(proj.history)
            with self.assertRaises(operations.OperationError):
                operations.apply_operations(
                    proj,
                    [
                        {"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 9.0},
                        {"op": "set_shot_duration", "shot_id": "NOPE", "duration_sec": 3.0},
                    ],
                )
            self.assertEqual(proj.state, before_state)
            self.assertEqual({k: set(v) for k, v in proj.dirty.items()}, before_dirty)
            self.assertEqual(proj.history, before_hist)


class TransactionManagerTest(unittest.TestCase):
    def test_illegal_transition_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))
            mgr = TransactionManager(proj)
            tx = mgr.open_planned(
                "k-illegal", [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 8.0}]
            )
            # persisted -> succeeded is not a legal direct jump
            with self.assertRaises(TransactionError):
                mgr.transition(tx, "succeeded")

    def test_idempotency_key_returns_same_transaction_without_reapply(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))
            apply_calls = {"n": 0}
            orig_apply = operations.apply_operations

            def counting_apply(p, ops):
                apply_calls["n"] += 1
                return orig_apply(p, ops)

            mgr = TransactionManager(proj, apply_ops=counting_apply)
            ops = [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 8.0}]
            tx1 = mgr.open_planned("dup-key", ops)
            tx2 = mgr.open_planned("dup-key", ops)
            self.assertEqual(tx1.id, tx2.id)
            self.assertEqual(apply_calls["n"], 1)
            self.assertEqual(proj.shot("S01")["duration_sec"], 8.0)
            self.assertEqual(len(mgr.all()), 1)

    def test_confirmation_required_defers_apply_until_confirm(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))
            mgr = TransactionManager(proj)
            ops = [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 20.0}]
            tx = mgr.open_planned("k-conf", ops, needs_confirm=["set_shot_duration"])
            self.assertEqual(tx.state, "confirmation_required")
            self.assertEqual(proj.shot("S01")["duration_sec"], 4)  # unchanged
            mgr.confirm(tx.id)
            self.assertEqual(mgr.get(tx.id).state, "persisted")
            self.assertEqual(proj.shot("S01")["duration_sec"], 20.0)

    def test_render_failure_keeps_dirty_and_marks_failed(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))
            mgr = TransactionManager(proj)
            tx = mgr.open_planned(
                "k-fail", [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 8.0}]
            )
            self.assertTrue(proj.is_dirty())
            mgr.execute(tx, boom_executor)
            self.assertEqual(mgr.get(tx.id).state, "render_failed")
            self.assertTrue(proj.is_dirty())  # dirty preserved on failure

    def test_retry_continues_from_failure_without_reapply(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            outdir = Path(d) / "out"
            outdir.mkdir()
            proj = write_project(Path(d))
            apply_calls = {"n": 0}
            orig_apply = operations.apply_operations

            def counting_apply(p, ops):
                apply_calls["n"] += 1
                return orig_apply(p, ops)

            mgr = TransactionManager(proj, apply_ops=counting_apply)
            tx = mgr.open_planned(
                "k-retry", [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 8.0}]
            )
            mgr.execute(tx, boom_executor)
            self.assertEqual(mgr.get(tx.id).state, "render_failed")
            mgr.execute(tx, success_executor_factory(outdir))
            self.assertEqual(mgr.get(tx.id).state, "succeeded")
            self.assertEqual(apply_calls["n"], 1)  # apply not repeated on retry

    def test_success_path_produces_nonempty_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            outdir = Path(d) / "out"
            outdir.mkdir()
            proj = write_project(Path(d))
            mgr = TransactionManager(proj)
            tx = mgr.open_planned(
                "k-ok", [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 8.0}]
            )
            mgr.execute(tx, success_executor_factory(outdir))
            tx = mgr.get(tx.id)
            self.assertEqual(tx.state, "succeeded")
            self.assertTrue(tx.verification["complete"])
            for path in tx.render_result["outputs"]:
                self.assertTrue(Path(path).is_file())
                self.assertGreater(Path(path).stat().st_size, 0)

    def test_verification_fails_when_nothing_changed(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            outdir = Path(d) / "out"
            outdir.mkdir()
            proj = write_project(Path(d))
            mgr = TransactionManager(proj)
            # set_shot_duration to the *same* current value => no real change
            tx = mgr.open_planned(
                "k-nochange",
                [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 4.0}],
            )
            mgr.execute(tx, success_executor_factory(outdir))
            self.assertEqual(mgr.get(tx.id).state, "verification_failed")
            self.assertFalse(mgr.get(tx.id).verification["complete"])

    def test_rollback_restores_pre_edit_state(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))
            mgr = TransactionManager(proj)
            tx = mgr.open_planned(
                "k-rb", [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 15.0}]
            )
            self.assertEqual(proj.shot("S01")["duration_sec"], 15.0)
            mgr.rollback(tx.id)
            self.assertEqual(mgr.get(tx.id).state, "rolled_back")
            self.assertEqual(proj.shot("S01")["duration_sec"], 4)

    def test_transactions_persist_to_disk_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))
            mgr = TransactionManager(proj)
            tx = mgr.open_planned(
                "k-persist", [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 8.0}]
            )
            store = Path(d) / "edit_transactions0907.json"
            self.assertTrue(store.is_file())
            # No fixed leftover temp file
            self.assertFalse((Path(d) / "edit_transactions0907.json.tmp").exists())
            # A fresh manager on the same dir must see the transaction.
            proj2 = project_mod.load(Path(d))
            mgr2 = TransactionManager(proj2)
            self.assertIsNotNone(mgr2.get(tx.id))
            self.assertEqual(mgr2.get(tx.id).state, "persisted")


class CrossInstanceTransactionTest(unittest.TestCase):
    def test_idempotency_key_across_manager_instances(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            m1 = TransactionManager(write_project(Path(d)))
            m2 = TransactionManager(write_project(Path(d)))  # constructed before the write
            ops = [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 8.0}]
            tx1 = m1.open_planned("SHARED", ops)
            tx2 = m2.open_planned("SHARED", ops)
            self.assertEqual(tx1.id, tx2.id)
            m3 = TransactionManager(write_project(Path(d)))
            self.assertEqual(len(m3.all()), 1)

    def test_two_managers_do_not_lose_transactions(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            m1 = TransactionManager(write_project(Path(d)))
            m2 = TransactionManager(write_project(Path(d)))
            tx1 = m1.open_planned("k1", [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 8.0}])
            tx2 = m2.open_planned("k2", [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 9.0}])
            m3 = TransactionManager(write_project(Path(d)))
            ids = {t.id for t in m3.all()}
            self.assertIn(tx1.id, ids)
            self.assertIn(tx2.id, ids)
            self.assertEqual(len(ids), 2)

    def test_concurrent_adds_are_not_lost(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            managers = [TransactionManager(write_project(Path(d))) for _ in range(4)]
            errors: list[BaseException] = []

            def worker(mgr: TransactionManager, base: int) -> None:
                for i in range(10):
                    try:
                        mgr.open_planned(
                            f"key-{base}-{i}",
                            [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 5.0}],
                        )
                    except BaseException as exc:  # noqa: BLE001
                        errors.append(exc)

            threads = [threading.Thread(target=worker, args=(m, idx)) for idx, m in enumerate(managers)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [], f"concurrent open raised: {errors[:3]}")
            final = TransactionManager(write_project(Path(d)))
            self.assertEqual(len(final.all()), 40)


class RollbackExtraStateTest(unittest.TestCase):
    def test_rollback_restores_quality_requests_extra_state(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))
            mgr = TransactionManager(proj)
            tx = mgr.open_planned("qopt", [{"op": "optimize_cut", "scope": "global"}])
            self.assertTrue(proj.state.get("quality_requests0907"))
            mgr.rollback(tx.id)
            self.assertEqual(mgr.get(tx.id).state, "rolled_back")
            self.assertFalse(proj.state.get("quality_requests0907"))


class StructuralOpVerificationTest(unittest.TestCase):
    def test_reorder_shots_succeeds_via_restitch_only(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            outdir = Path(d) / "out"
            outdir.mkdir()
            proj = write_project(Path(d))
            mgr = TransactionManager(proj)
            tx = mgr.open_planned(
                "k-reorder", [{"op": "reorder_shots", "shot_ids": ["S03", "S01", "S02"]}]
            )
            mgr.execute(tx, restitch_executor_factory(outdir))
            tx = mgr.get(tx.id)
            self.assertEqual(tx.state, "succeeded")
            self.assertTrue(tx.verification["complete"])
            self.assertEqual(proj.shot_ids(), ["S03", "S01", "S02"])

    def test_reorder_verification_fails_if_order_not_applied(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            outdir = Path(d) / "out"
            outdir.mkdir()
            proj = write_project(Path(d))
            before = operations.compute_digest(proj)
            (outdir / "f.mp4").write_bytes(b"z" * 64)
            proj.state["final_video_path"] = str(outdir / "f.mp4")
            # order deliberately NOT applied -> reorder postcondition must fail
            ops = [{"op": "reorder_shots", "shot_ids": ["S03", "S01", "S02"]}]
            result = verification.verify_transaction(
                proj, before, ops, render_result={"outputs": [], "final_video_path": str(outdir / "f.mp4")}
            )
            self.assertFalse(result["complete"])

    def test_delete_shot_succeeds_via_restitch_only(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            outdir = Path(d) / "out"
            outdir.mkdir()
            proj = write_project(Path(d))
            mgr = TransactionManager(proj)
            tx = mgr.open_planned("k-del", [{"op": "delete_shot", "shot_id": "S02"}])
            mgr.execute(tx, restitch_executor_factory(outdir))
            tx = mgr.get(tx.id)
            self.assertEqual(tx.state, "succeeded")
            self.assertTrue(tx.verification["complete"])
            self.assertNotIn("S02", proj.shot_ids())
            self.assertIn("S01", proj.shot_ids())
            self.assertIn("S03", proj.shot_ids())

    def test_delete_verification_fails_if_target_still_present(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            outdir = Path(d) / "out"
            outdir.mkdir()
            proj = write_project(Path(d))
            before = operations.compute_digest(proj)
            (outdir / "f.mp4").write_bytes(b"z" * 64)
            proj.state["final_video_path"] = str(outdir / "f.mp4")
            # never actually delete -> delete postcondition must fail
            ops = [{"op": "delete_shot", "shot_id": "S02"}]
            result = verification.verify_transaction(
                proj, before, ops, render_result={"outputs": [], "final_video_path": str(outdir / "f.mp4")}
            )
            self.assertFalse(result["complete"])


class VerificationTest(unittest.TestCase):
    def test_non_target_shot_change_fails_verification(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))
            ops = [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 8.0}]
            before = operations.compute_digest(proj)
            operations.apply_operations(proj, ops)
            # tamper an unrelated shot to simulate collateral damage
            proj.shot("S03")["vocal_script"] = "被意外改动"
            result = verification.verify_transaction(proj, before, ops, render_result=None)
            self.assertFalse(result["complete"])

    def test_tts_edit_requires_in_target_loudness(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))
            before = operations.compute_digest(proj)
            ops = [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 8.0}]
            operations.apply_operations(proj, ops)

            rendered = Path(d) / "rendered_new.mp4"
            final = Path(d) / "final_new.mp4"
            rendered.write_bytes(b"x" * 64)
            final.write_bytes(b"y" * 128)
            proj.shot_assets.setdefault("S01", {})["rendered_path"] = str(rendered)
            proj.shot_assets["S01"]["audio_meta"] = {
                "duration_sec": 8.0,
                "loudness": {
                    "status": "normalized",
                    "normalized": True,
                    "target": {
                        "integrated_lufs": -16.0,
                        "true_peak_dbtp": -1.5,
                        "tolerance_lu": 1.0,
                        "lra_lu": 11.0,
                    },
                    "output": {"output_i": -18.1, "output_tp": -1.4},
                },
            }
            proj.state["final_video_path"] = str(final)

            result = verification.verify_transaction(
                proj,
                before,
                ops,
                render_result={"outputs": [str(rendered)], "final_video_path": str(final)},
            )
            self.assertFalse(result["complete"])
            self.assertTrue(any(c["name"].startswith("loudness:") and not c["ok"] for c in result["checks"]))

    def test_tts_edit_rejects_predicted_output_when_input_measurement_is_bad(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))
            before = operations.compute_digest(proj)
            ops = [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 8.0}]
            operations.apply_operations(proj, ops)

            rendered = Path(d) / "rendered_predicted.mp4"
            final = Path(d) / "final_predicted.mp4"
            rendered.write_bytes(b"x" * 64)
            final.write_bytes(b"y" * 128)
            proj.shot_assets.setdefault("S01", {})["rendered_path"] = str(rendered)
            proj.shot_assets["S01"]["audio_meta"] = {
                "duration_sec": 8.0,
                "loudness": {
                    "status": "normalized",
                    "normalized": True,
                    "target": {
                        "integrated_lufs": -16.0,
                        "true_peak_dbtp": -1.5,
                        "tolerance_lu": 1.0,
                        "lra_lu": 11.0,
                    },
                    "input": {
                        "input_i": -21.0, "input_tp": -3.0,
                        "input_lra": 6.0, "input_thresh": -30.0,
                        "output_i": -16.0, "output_tp": -1.6,
                        "output_lra": 6.0, "output_thresh": -26.0,
                        "target_offset": 0.0, "normalization_type": "dynamic",
                    },
                    "output": {
                        "input_i": -18.0, "input_tp": -0.8,
                        "input_lra": 6.0, "input_thresh": -28.0,
                        "output_i": -16.0, "output_tp": -1.6,
                        "output_lra": 6.0, "output_thresh": -26.0,
                        "target_offset": 0.0, "normalization_type": "dynamic",
                    },
                },
            }
            proj.state["final_video_path"] = str(final)

            result = verification.verify_transaction(
                proj,
                before,
                ops,
                render_result={"outputs": [str(rendered)], "final_video_path": str(final)},
            )
            self.assertFalse(result["complete"])
            self.assertTrue(any(c["name"].startswith("loudness:") and not c["ok"] for c in result["checks"]))

    def test_tts_edit_accepts_normalized_top_level_output_shape(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))
            before = operations.compute_digest(proj)
            ops = [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 8.0}]
            operations.apply_operations(proj, ops)

            rendered = Path(d) / "rendered_ok.mp4"
            final = Path(d) / "final_ok.mp4"
            rendered.write_bytes(b"x" * 64)
            final.write_bytes(b"y" * 128)
            proj.shot_assets.setdefault("S01", {})["rendered_path"] = str(rendered)
            proj.shot_assets["S01"]["audio_meta"] = {
                "duration_sec": 8.0,
                "loudness": {
                    "status": "normalized",
                    "normalized": True,
                    "target": {
                        "integrated_lufs": -16.0,
                        "true_peak_dbtp": -1.5,
                        "tolerance_lu": 1.0,
                        "lra_lu": 11.0,
                    },
                    "input": {
                        "input_i": -21.0, "input_tp": -3.0,
                        "input_lra": 6.0, "input_thresh": -30.0,
                        "output_i": -16.0, "output_tp": -1.6,
                        "output_lra": 6.0, "output_thresh": -26.0,
                        "target_offset": 0.0, "normalization_type": "dynamic",
                    },
                    "output": {
                        "input_i": -16.1, "input_tp": -1.6,
                        "input_lra": 6.0, "input_thresh": -26.0,
                        "output_i": -15.2, "output_tp": -1.0,
                        "output_lra": 6.0, "output_thresh": -25.0,
                        "target_offset": 0.0, "normalization_type": "dynamic",
                    },
                },
            }
            proj.state["final_video_path"] = str(final)

            result = verification.verify_transaction(
                proj,
                before,
                ops,
                render_result={"outputs": [str(rendered)], "final_video_path": str(final)},
            )
            self.assertTrue(result["complete"], str(result["checks"]))

    def test_tts_edit_rejects_failed_payload_even_if_output_looks_good(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))
            before = operations.compute_digest(proj)
            ops = [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 8.0}]
            operations.apply_operations(proj, ops)

            rendered = Path(d) / "rendered_failed.mp4"
            final = Path(d) / "final_failed.mp4"
            rendered.write_bytes(b"x" * 64)
            final.write_bytes(b"y" * 128)
            proj.shot_assets.setdefault("S01", {})["rendered_path"] = str(rendered)
            proj.shot_assets["S01"]["audio_meta"] = {
                "duration_sec": 8.0,
                "loudness": {
                    "status": "failed",
                    "normalized": False,
                    "target": {
                        "integrated_lufs": -16.0,
                        "true_peak_dbtp": -1.5,
                        "tolerance_lu": 1.0,
                        "lra_lu": 11.0,
                    },
                    "input": {
                        "input_i": -21.0, "input_tp": -3.0,
                        "input_lra": 6.0, "input_thresh": -30.0,
                        "output_i": -16.0, "output_tp": -1.6,
                        "output_lra": 6.0, "output_thresh": -26.0,
                        "target_offset": 0.0, "normalization_type": "dynamic",
                    },
                    "output": {
                        "input_i": -16.1, "input_tp": -1.6,
                        "input_lra": 6.0, "input_thresh": -26.0,
                        "output_i": -15.2, "output_tp": -1.0,
                        "output_lra": 6.0, "output_thresh": -25.0,
                        "target_offset": 0.0, "normalization_type": "dynamic",
                    },
                    "error": {"code": "ffmpeg_failed", "message": "mock fail"},
                },
            }
            proj.state["final_video_path"] = str(final)

            result = verification.verify_transaction(
                proj,
                before,
                ops,
                render_result={"outputs": [str(rendered)], "final_video_path": str(final)},
            )
            self.assertFalse(result["complete"])
            self.assertTrue(any(c["name"].startswith("loudness:") and not c["ok"] for c in result["checks"]))

    def test_pure_visual_edit_does_not_require_loudness_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proj = write_project(Path(d))
            before = operations.compute_digest(proj)
            ops = [{"op": "set_emphasis", "shot_id": "S03", "phrase": "让工具主动理解你", "emphasis_type": "key_point"}]
            operations.apply_operations(proj, ops)

            rendered = Path(d) / "rendered_new.mp4"
            final = Path(d) / "final_new.mp4"
            rendered.write_bytes(b"x" * 64)
            final.write_bytes(b"y" * 128)
            proj.shot_assets.setdefault("S03", {})["rendered_path"] = str(rendered)
            proj.state["final_video_path"] = str(final)

            result = verification.verify_transaction(
                proj,
                before,
                ops,
                render_result={"outputs": [str(rendered)], "final_video_path": str(final)},
            )
            self.assertTrue(result["complete"], str(result["checks"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
