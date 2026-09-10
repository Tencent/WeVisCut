"""Production-runner tests for the newvis0907 background regeneration.

Run: python3 tests/test_newvis0907_prod_runner.py

These do NOT use the transaction-level fake executor: they exercise the real
``studio.app._execute_regeneration`` (with ``dispatch`` patched) to prove the
strict abort path, and the real ``studio.app._run_0907_regeneration`` (with
``_execute_regeneration`` patched) to prove fingerprint-based verification and
dirty-snapshot restore. Importing ``studio.app`` is heavy but intentional.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import studio.app as app_mod  # noqa: E402
from studio.app import studio  # noqa: E402
from studio import orchestrator as orch_mod  # noqa: E402
from studio import project as project_mod  # noqa: E402
from studio.newvis0907 import operations, schemas  # noqa: E402
from studio.newvis0907.transactions import TransactionManager  # noqa: E402
from studio.runs import RunStore  # noqa: E402


def _bootstrap_studio() -> None:
    studio.runs = RunStore()  # journal_root=None => no disk journal
    studio.cfg = SimpleNamespace(
        editor={"rerender": {"timeout_sec": 5, "scope": "only_dirty",
                             "keep_version_history": False, "max_versions": 5}},
        pipeline_config=None,
        stage_config=None,
        system_prompt="",
        infer_config=None,
        progress={"tracks": []},
    )
    studio.model = SimpleNamespace(loader=object(), is_loaded=True)
    studio.regen_locks.clear()


def _state_payload() -> dict:
    return {
        "project_id": "proj_prod",
        "input_text": "demo",
        "final_video_path": "",
        "shot_plan": [
            {"shot_id": "S01", "segment_id": "seg1", "duration_sec": 4, "vocal_script": "a"},
            {"shot_id": "S02", "segment_id": "seg1", "duration_sec": 4, "vocal_script": "b"},
            {"shot_id": "S03", "segment_id": "seg2", "duration_sec": 4, "vocal_script": "c"},
        ],
        "shot_assets": {},
    }


def _make_project(tmp: Path) -> project_mod.Project:
    (tmp / "final_state.json").write_text(
        json.dumps(_state_payload(), ensure_ascii=False), encoding="utf-8"
    )
    return project_mod.load(tmp)


def _make_project_with_render(tmp: Path) -> tuple[project_mod.Project, Path, Path]:
    r1 = tmp / "r_s01_old.mp4"
    r1.write_bytes(b"old-render" * 500)
    f1 = tmp / "final_old.mp4"
    f1.write_bytes(b"old-final" * 500)
    payload = _state_payload()
    payload["final_video_path"] = str(f1)
    payload["shot_assets"] = {"S01": {"rendered_path": str(r1)}}
    (tmp / "final_state.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return project_mod.load(tmp), r1, f1


class _DummyOrchestrator:
    def __init__(self, *a, **k) -> None:
        pass

    def update_state_from_respond(self, tool, output) -> None:
        pass

    def _apply_tool_output(self, tool, output, state) -> None:
        pass

    def _make_agent_observation(self, tool, output):
        return {}


class StrictRegenerationTest(unittest.TestCase):
    def setUp(self) -> None:
        _bootstrap_studio()

    def _run_with_timeout_dispatch(self, strict: bool, job_id: str):
        async def fake_dispatch(orchestrator, tool, args, state, run_id, **kw):
            return {"status": "timeout", "error": "boom timed out"}

        with tempfile.TemporaryDirectory() as d:
            proj = _make_project(Path(d))
            proj.mark_dirty("S01", "render")
            job = studio.store.create("t", run_id=job_id)
            job.project_dir = str(proj.project_dir)
            steps = [{"tool": "Tool_BatchTimelineCompose",
                      "args": {"shot_ids": ["S01"]}, "label": "compose"}]
            with mock.patch.object(orch_mod, "AutonomousOrchestrator", _DummyOrchestrator), \
                 mock.patch.object(app_mod, "_dispatch_scoped", fake_dispatch), \
                 mock.patch.object(app_mod, "_tool_result_event", lambda *a, **k: {}), \
                 mock.patch.object(app_mod.tracks, "build_snapshot", lambda *a, **k: {}):
                asyncio.run(app_mod._execute_regeneration(
                    job_id, proj, steps, restitch=False, strict=strict))
            return proj, studio.store.get(job_id)

    def test_strict_timeout_errors_job_and_keeps_dirty(self) -> None:
        proj, job = self._run_with_timeout_dispatch(strict=True, job_id="jstrict")
        self.assertEqual(job.status, "error")
        self.assertTrue(proj.is_dirty(), "strict run must NOT clear dirty on timeout")

    def test_default_nonstrict_swallows_timeout_and_completes(self) -> None:
        # Guards that the old (non-0907) behaviour is unchanged by the new flag.
        proj, job = self._run_with_timeout_dispatch(strict=False, job_id="jsoft")
        self.assertEqual(job.status, "completed")
        self.assertFalse(proj.is_dirty(), "non-strict run clears dirty as before")

    def test_policy_version_uses_config_copy_without_mutating_global(self) -> None:
        captured: dict[str, object] = {}

        class _CapturingOrchestrator(_DummyOrchestrator):
            def __init__(self, _loader, pipeline_config, *_args, **_kwargs) -> None:
                captured["pipeline_config"] = pipeline_config

        with tempfile.TemporaryDirectory() as d:
            proj = _make_project(Path(d))
            studio.config.pipeline_config = {
                "services": {"tts": {"enabled": True}},
                "newvis0907": {"enabled": False},
            }
            job = studio.store.create("t", run_id="jcopy")
            job.project_dir = str(proj.project_dir)
            with mock.patch.object(orch_mod, "AutonomousOrchestrator", _CapturingOrchestrator), \
                 mock.patch.object(app_mod.tracks, "build_snapshot", lambda *a, **k: {}):
                asyncio.run(app_mod._execute_regeneration(
                    "jcopy", proj, [], restitch=False, strict=True, policy_version="0907"))

        passed = captured["pipeline_config"]
        self.assertIsInstance(passed, dict)
        self.assertEqual(passed.get("policy_version"), "0907")
        self.assertIsNot(passed, studio.config.pipeline_config)
        self.assertNotIn("policy_version", studio.config.pipeline_config)
        self.assertEqual(
            studio.config.pipeline_config["newvis0907"],
            {"enabled": False},
        )


class ProdRunnerVerificationTest(unittest.TestCase):
    def setUp(self) -> None:
        _bootstrap_studio()

    def _run(self, manager, tx):
        # The job is now created synchronously by the scheduler (so the API can
        # return its id); mirror that here and pass the id into the runner.
        job_id = f"regen_{tx.id}_{uuid.uuid4().hex[:6]}"
        job = studio.store.create("t", run_id=job_id)
        job.project_dir = str(manager.project.project_dir)
        with mock.patch.object(app_mod.editor, "plan_regeneration", lambda p, s: []):
            asyncio.run(app_mod._run_0907_regeneration(manager, tx, job_id))

    def test_success_when_targets_and_final_freshly_written(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            proj, _r1, _f1 = _make_project_with_render(tmp)
            manager = TransactionManager(proj)
            tx = manager.open_planned(
                "k-ok", [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 8.0}])
            captured = {}

            async def fake_exec(job_id, project, steps, restitch, strict=False, policy_version=None):
                captured["strict"] = strict
                captured["policy_version"] = policy_version
                for sid in list(project.dirty.keys()):
                    newf = tmp / f"r_{sid}_new.mp4"
                    newf.write_bytes(os.urandom(400))
                    assets = project.shot_assets.setdefault(sid, {})
                    assets["rendered_path"] = str(newf)
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
                newfinal = tmp / "final_new.mp4"
                newfinal.write_bytes(os.urandom(600))
                project.state["final_video_path"] = str(newfinal)
                project.clear_dirty()
                studio.store.get(job_id).status = "completed"

            with mock.patch.object(app_mod, "_execute_regeneration", fake_exec):
                self._run(manager, tx)

            self.assertTrue(captured.get("strict"), "0907 wrapper must call strict=True")
            self.assertEqual(captured.get("policy_version"), "0907")
            self.assertEqual(manager.get(tx.id).state, "succeeded")
            self.assertTrue(manager.get(tx.id).verification["complete"])

    def test_verification_failed_when_old_files_unchanged_restores_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            proj, r1, f1 = _make_project_with_render(tmp)
            manager = TransactionManager(proj)
            tx = manager.open_planned(
                "k-vf", [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 8.0}])

            async def fake_exec(job_id, project, steps, restitch, strict=False, policy_version=None):
                # "completed" but reuses the OLD, unchanged files.
                for sid in list(project.dirty.keys()):
                    project.shot_assets.setdefault(sid, {})["rendered_path"] = str(r1)
                project.state["final_video_path"] = str(f1)
                project.clear_dirty()
                studio.store.get(job_id).status = "completed"

            with mock.patch.object(app_mod, "_execute_regeneration", fake_exec):
                self._run(manager, tx)

            self.assertEqual(manager.get(tx.id).state, "verification_failed")
            self.assertTrue(proj.is_dirty(), "verification failure must restore dirty snapshot")

    def test_render_failure_status_marks_failed_and_keeps_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            proj, _r1, _f1 = _make_project_with_render(tmp)
            manager = TransactionManager(proj)
            tx = manager.open_planned(
                "k-rf", [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 8.0}])

            async def fake_exec(job_id, project, steps, restitch, strict=False, policy_version=None):
                # strict path: job ends in error, dirty left intact.
                studio.store.get(job_id).status = "error"

            with mock.patch.object(app_mod, "_execute_regeneration", fake_exec):
                self._run(manager, tx)

            self.assertEqual(manager.get(tx.id).state, "render_failed")
            self.assertTrue(proj.is_dirty(), "render failure must keep/restore dirty")
            self.assertNotEqual(manager.get(tx.id).state, "succeeded")

    def test_loudness_verification_failure_restores_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            proj, _r1, _f1 = _make_project_with_render(tmp)
            manager = TransactionManager(proj)
            tx = manager.open_planned(
                "k-loud", [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 8.0}])

            async def fake_exec(job_id, project, steps, restitch, strict=False, policy_version=None):
                for sid in list(project.dirty.keys()):
                    newf = tmp / f"r_{sid}_new.mp4"
                    newf.write_bytes(os.urandom(400))
                    assets = project.shot_assets.setdefault(sid, {})
                    assets["rendered_path"] = str(newf)
                    assets["audio_meta"] = {
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
                            "output": {"output_i": -18.2, "output_tp": -1.4},
                        },
                    }
                newfinal = tmp / "final_loud.mp4"
                newfinal.write_bytes(os.urandom(600))
                project.state["final_video_path"] = str(newfinal)
                project.clear_dirty()
                studio.store.get(job_id).status = "completed"

            with mock.patch.object(app_mod, "_execute_regeneration", fake_exec):
                self._run(manager, tx)

            self.assertEqual(manager.get(tx.id).state, "verification_failed")
            self.assertTrue(proj.is_dirty(), "verification failure must restore dirty snapshot")

    def test_predicted_output_loudness_still_fails_and_restores_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            proj, _r1, _f1 = _make_project_with_render(tmp)
            manager = TransactionManager(proj)
            tx = manager.open_planned(
                "k-loud-pred", [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 8.0}])

            async def fake_exec(job_id, project, steps, restitch, strict=False, policy_version=None):
                for sid in list(project.dirty.keys()):
                    newf = tmp / f"r_{sid}_pred.mp4"
                    newf.write_bytes(os.urandom(400))
                    assets = project.shot_assets.setdefault(sid, {})
                    assets["rendered_path"] = str(newf)
                    assets["audio_meta"] = {
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
                newfinal = tmp / "final_loud_pred.mp4"
                newfinal.write_bytes(os.urandom(600))
                project.state["final_video_path"] = str(newfinal)
                project.clear_dirty()
                studio.store.get(job_id).status = "completed"

            with mock.patch.object(app_mod, "_execute_regeneration", fake_exec):
                self._run(manager, tx)

            self.assertEqual(manager.get(tx.id).state, "verification_failed")
            self.assertTrue(proj.is_dirty(), "predicted-only loudness must restore dirty snapshot")

    def test_reorder_restitch_only_succeeds_via_runner(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            proj = _make_project(tmp)
            manager = TransactionManager(proj)
            tx = manager.open_planned(
                "k-ro", [{"op": "reorder_shots", "shot_ids": ["S03", "S01", "S02"]}])

            async def fake_exec(job_id, project, steps, restitch, strict=False, policy_version=None):
                # restitch-only: rewrite the final video, no per-shot render.
                newfinal = tmp / "final_reorder.mp4"
                newfinal.write_bytes(os.urandom(600))
                project.state["final_video_path"] = str(newfinal)
                project.clear_dirty()
                studio.store.get(job_id).status = "completed"

            with mock.patch.object(app_mod, "_execute_regeneration", fake_exec):
                self._run(manager, tx)

            self.assertEqual(manager.get(tx.id).state, "succeeded")
            self.assertEqual(proj.shot_ids(), ["S03", "S01", "S02"])

    def test_retry_regenerates_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            proj, _r1, _f1 = _make_project_with_render(tmp)
            manager = TransactionManager(proj)
            tx = manager.open_planned(
                "k-retry", [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 8.0}])

            async def fail_exec(job_id, project, steps, restitch, strict=False, policy_version=None):
                studio.store.get(job_id).status = "error"

            async def ok_exec(job_id, project, steps, restitch, strict=False, policy_version=None):
                for sid in list(project.dirty.keys()):
                    newf = tmp / f"r_{sid}_retry.mp4"
                    newf.write_bytes(os.urandom(400))
                    assets = project.shot_assets.setdefault(sid, {})
                    assets["rendered_path"] = str(newf)
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
                newfinal = tmp / "final_retry.mp4"
                newfinal.write_bytes(os.urandom(600))
                project.state["final_video_path"] = str(newfinal)
                project.clear_dirty()
                studio.store.get(job_id).status = "completed"

            with mock.patch.object(app_mod, "_execute_regeneration", fail_exec):
                self._run(manager, tx)
            self.assertEqual(manager.get(tx.id).state, "render_failed")
            self.assertTrue(proj.dirty, "steps must be available for retry")

            manager.retry(tx.id)
            with mock.patch.object(app_mod, "_execute_regeneration", ok_exec):
                self._run(manager, tx)
            self.assertEqual(manager.get(tx.id).state, "succeeded")


if __name__ == "__main__":
    unittest.main(verbosity=2)
