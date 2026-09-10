"""Tests for the dedicated 0907 generation endpoint + pipeline marker wiring.

Run: python3 tests/test_newvis0907_runs.py

Covers Task 4/5/10:
* ``/api/0907/runs`` exists, binds ``policy_version="0907"`` and shares the run
  store, without mutating the global studio config;
* ``_execute_run`` hands the runtime a *deep-copied* pipeline config carrying the
  0907 marker (legacy runs are byte-for-byte unchanged);
* the marker actually reaches every implemented pipeline branch
  (plan_shots / generate_audio / decorate_shot / timeline_compose / stitch_video);
* a completed 0907 run does not leave ``policy_version`` on the shared config
  (no leak into a subsequent legacy run).
"""

from __future__ import annotations

import asyncio
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import studio.app as app_mod  # noqa: E402
from studio.app import studio  # noqa: E402
from studio import project as project_mod  # noqa: E402
from studio.runs import RunStore  # noqa: E402


def _bootstrap_studio() -> None:
    studio.runs = RunStore()
    studio.cfg = SimpleNamespace(
        editor={"rerender": {"timeout_sec": 5}},
        pipeline_config={"services": {"tts": {"enabled": True}}, "newvis0907": {"enabled": False}},
        stage_config=None,
        system_prompt="",
        infer_config={"agent": {"max_steps": 3}},
        progress={"tracks": []},
        active_profile="test",
        workspace_root=Path(tempfile.gettempdir()) / "aicut_runs_test",
    )
    studio.model = SimpleNamespace(loader=object(), is_loaded=True, load_error="")


class ExecuteRunPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        _bootstrap_studio()

    def _run(self, policy_version):
        captured: dict = {}

        class _Runtime:
            def __init__(self):
                self.orchestrator = SimpleNamespace(project_dir="")

        def fake_create_runtime(loader, *, pipeline_config, **kwargs):
            captured["pipeline_config"] = pipeline_config
            return _Runtime()

        async def fake_generation(runtime, loader):
            return {"output_dir": "", "errors": []}

        run = studio.store.create("hello world")
        with mock.patch("aicut_agent.runner.create_runtime", fake_create_runtime), \
             mock.patch("aicut_agent.runner.run_aicut_generation", fake_generation), \
             mock.patch.object(app_mod.tracks, "build_snapshot", lambda *a, **k: {}):
            asyncio.run(app_mod._execute_run(run.id, "hello world", policy_version=policy_version))
        return captured, run

    def test_0907_run_uses_config_copy_with_marker(self) -> None:
        before = copy.deepcopy(studio.config.pipeline_config)
        captured, _run = self._run("0907")
        passed = captured["pipeline_config"]
        self.assertEqual(passed.get("policy_version"), "0907")
        self.assertIsNot(passed, studio.config.pipeline_config)
        # global config untouched — no marker leaks into the shared config.
        self.assertNotIn("policy_version", studio.config.pipeline_config)
        self.assertEqual(studio.config.pipeline_config, before)

    def test_legacy_run_passes_config_unchanged(self) -> None:
        captured, _run = self._run(None)
        # legacy path forwards the shared config object itself (no copy, no marker).
        self.assertIs(captured["pipeline_config"], studio.config.pipeline_config)
        self.assertNotIn("policy_version", captured["pipeline_config"])


class RunsEndpointTest(unittest.TestCase):
    def test_endpoint_exists_and_binds_policy(self) -> None:
        _bootstrap_studio()
        started: dict = {}

        async def fake_exec(run_id, text, policy_version=None):
            started["run_id"] = run_id
            started["policy_version"] = policy_version

        # call the endpoint coroutine directly with the shared run store.
        with mock.patch.object(app_mod, "_execute_run", fake_exec), \
             mock.patch.object(app_mod, "require_user",
                               lambda: SimpleNamespace(username="tester")):
            req = app_mod.StartRunRequest(input_text="生成一个视频")
            body = asyncio.run(app_mod.start_run_0907(req))
        self.assertEqual(body["policy_version"], "0907")
        self.assertIn("run_id", body)
        run = studio.store.get(body["run_id"])
        self.assertIsNotNone(run)
        self.assertEqual(run.policy_version, "0907")

    def test_source_registers_route_and_reuses_ws(self) -> None:
        src = (ROOT / "studio" / "app.py").read_text(encoding="utf-8")
        self.assertIn('@app.post("/api/0907/runs")', src)
        # shares the legacy websocket + run store (no new socket path invented).
        self.assertIn('async def run_socket(websocket: WebSocket, run_id: str)', src)
        # legacy endpoint stays intact.
        self.assertIn('@app.post("/api/runs")', src)


class PipelineMarkerReachTest(unittest.TestCase):
    """The 0907 marker must flip every implemented pipeline branch on."""

    def test_marker_enables_all_stage_branches(self) -> None:
        from pipeline.tools import plan_shots, generate_audio, decorate_shot
        from pipeline.tools import timeline_compose, stitch_video
        from pipeline import graph

        cfg = {"policy_version": "0907"}
        self.assertTrue(plan_shots._is_0907(cfg), "plan_shots")
        self.assertTrue(generate_audio._use_0907(cfg), "generate_audio")
        self.assertTrue(decorate_shot._semantic_autogen_enabled(cfg), "decorate_shot")
        self.assertTrue(timeline_compose._use_0907_subtitles(cfg), "timeline_compose")
        self.assertTrue(stitch_video._use_0907(cfg), "stitch_video")
        self.assertTrue(graph._is_0907_compose(cfg), "graph compose")

        legacy = {"services": {}}
        self.assertFalse(plan_shots._is_0907(legacy))
        self.assertFalse(generate_audio._use_0907(legacy))
        self.assertFalse(decorate_shot._semantic_autogen_enabled(legacy))
        self.assertFalse(timeline_compose._use_0907_subtitles(legacy))
        self.assertFalse(stitch_video._use_0907(legacy))
        self.assertFalse(graph._is_0907_compose(legacy))


class RunPolicyPersistenceTest(unittest.TestCase):
    """M3: policy_version survives summary/save_meta/rehydrate, legacy-safe."""

    def test_summary_and_rehydrate_keep_policy_version(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            store = RunStore(journal_root=d)
            run = store.create("hello world")
            run.policy_version = "0907"
            run.status = "completed"
            store.save_meta(run)
            self.assertEqual(run.summary()["policy_version"], "0907")

            store2 = RunStore(journal_root=d)
            store2.rehydrate()
            restored = store2.get(run.id)
            self.assertIsNotNone(restored)
            self.assertEqual(restored.policy_version, "0907")
            self.assertEqual(restored.summary()["policy_version"], "0907")

    def test_legacy_meta_without_policy_defaults_empty(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            rid = "run_legacy_meta"
            jd = Path(d) / rid
            jd.mkdir(parents=True)
            (jd / "meta.json").write_text(json.dumps({
                "id": rid, "status": "completed", "input_text": "x",
                "started_at": "", "finished_at": "", "progress": {},
                "result": {}, "errors": [],
            }), encoding="utf-8")
            store = RunStore(journal_root=d)
            store.rehydrate()
            restored = store.get(rid)
            self.assertIsNotNone(restored)
            self.assertEqual(restored.policy_version, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
