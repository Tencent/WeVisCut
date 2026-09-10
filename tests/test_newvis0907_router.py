"""Offline tests for the dedicated newvis0907 API router.

Run: python3 tests/test_newvis0907_router.py

The router is mounted on a bare FastAPI app and configured with fake callbacks
and a throwaway on-disk project, so nothing here scans the network filesystem or
loads a model.
"""

from __future__ import annotations

import json
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


def _state_payload() -> dict:
    return {
        "project_id": "proj_router",
        "input_text": "demo",
        "final_video_path": "",
        "segments": [
            {"segment_id": "seg_intro", "title": "序章"},
            {"segment_id": "seg_future", "title": "未来图景"},
        ],
        "shot_plan": [
            {"shot_id": "S01", "segment_id": "seg_intro", "duration_sec": 4,
             "vocal_script": "先看背景。", "subtitle_text": "先看背景。"},
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


def _make_project(tmp: Path) -> project_mod.Project:
    (tmp / "final_state.json").write_text(
        json.dumps(_state_payload(), ensure_ascii=False), encoding="utf-8"
    )
    return project_mod.load(tmp)


def _payload_stub(project) -> dict:
    return {
        "project_id": project.project_id,
        "project_dir": str(project.project_dir),
        "digest": project.digest(),
        "final_video_path": project.final_video_path,
        "dirty": {sid: sorted(r) for sid, r in project.dirty.items()},
    }


def _success_runner(outdir: Path):
    def _exec(project, tx) -> dict:
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
            outputs.append(str(out))
        final = outdir / f"final_{tx.id}.mp4"
        final.write_bytes(b"y" * 64)
        project.state["final_video_path"] = str(final)
        project.clear_dirty()
        return {"outputs": outputs, "final_video_path": str(final),
                "fingerprint": "fp-new", "version": "v2"}

    def _runner(mgr, tx) -> None:
        mgr.execute(tx, _exec)

    return _runner


def _build(tmp: Path, runner=None):
    project = _make_project(tmp)
    deps = router_mod.RouterDeps(
        load_project=lambda d: project,
        project_payload=_payload_stub,
        discover_projects=lambda: [{"project_dir": str(tmp), "project_id": "proj_router"}],
        delete_project=lambda d: {"deleted": True, "project_dir": d},
        export_project=lambda p: {"status": "started", "job_id": "export_test"},
    )
    if runner is not None:
        deps.schedule_runner = runner
    app = FastAPI()
    router_mod.configure_router(deps)
    app.include_router(router_mod.router)
    return TestClient(app), project, str(tmp)


def _build_with_model_planner(tmp: Path, planner):
    project = _make_project(tmp)
    deps = router_mod.RouterDeps(
        load_project=lambda d: project,
        project_payload=_payload_stub,
        discover_projects=lambda: [{"project_dir": str(tmp), "project_id": "proj_router"}],
        delete_project=lambda d: {"deleted": True, "project_dir": d},
        export_project=lambda p: {"status": "started", "job_id": "export_test"},
        model_plan=planner,
    )
    app = FastAPI()
    router_mod.configure_router(deps)
    app.include_router(router_mod.router)
    return TestClient(app), project, str(tmp)


class RouterTest(unittest.TestCase):
    def test_project_payload_has_hierarchy_and_policy_version(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, _proj, pdir = _build(Path(d))
            r = client.get("/api/0907/project", params={"dir": pdir})
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            self.assertEqual(body["project_id"], "proj_router")
            self.assertIn("hierarchy", body)
            self.assertIn("quality", body)
            self.assertIn("chapters", body["hierarchy"])
            self.assertIn("model_shots", body["hierarchy"])
            self.assertIn("active_transactions", body)
            self.assertEqual(body["policy_version"], "0907")

    def test_projects_library_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, _proj, _pdir = _build(Path(d))
            r = client.get("/api/0907/projects")
            self.assertEqual(r.status_code, 200, r.text)
            self.assertIn("projects", r.json())

    def test_chat_clarify_does_not_apply(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, proj, pdir = _build(Path(d))
            r = client.post("/api/0907/project/chat",
                            json={"project_dir": pdir, "message": "强调“工具箱革命”"})
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            self.assertEqual(body["intent"], "clarify")
            self.assertFalse(body.get("applied"))
            self.assertIsNone(body.get("transaction"))
            self.assertNotIn("succeeded", r.text)
            self.assertNotIn("修改成功", r.text)

    def test_chat_uses_model_plan_when_rules_need_clarification(self) -> None:
        calls = []

        def planner(state, message, previous_targets=None, history=None):
            calls.append({"message": message, "history": history})
            return {
                "reply": "好的，我来处理：\n· 镜头 02：换成「大海」的画面素材",
                "ops": [{"op": "replace_footage", "shot_id": "S02", "query": "大海"}],
            }

        with tempfile.TemporaryDirectory() as d:
            client, proj, pdir = _build_with_model_planner(Path(d), planner)
            r = client.post("/api/0907/project/chat",
                            json={"project_dir": pdir, "message": "把后面那个画面换成大海"})
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            self.assertEqual(body["intent"], "edit", body)
            self.assertTrue(calls, body)
            self.assertEqual(calls[0]["message"], "把后面那个画面换成大海")
            self.assertEqual(body["ops"][0]["shot_id"], "S02", body)
            self.assertNotIn("修改成功", body["reply"])

    def test_chat_always_asks_model_even_for_clear_commands(self) -> None:
        calls = []

        def planner(state, message, previous_targets=None, history=None):
            calls.append(message)
            return {
                "intent": "edit",
                "reply": "好的，我来处理：\n· 镜头 01：换成「摇滚乐队」的画面素材",
                "ops": [{"op": "replace_footage", "shot_id": "S01", "query": "摇滚乐队"}],
            }

        with tempfile.TemporaryDirectory() as d:
            client, _proj, pdir = _build_with_model_planner(Path(d), planner)
            r = client.post("/api/0907/project/chat",
                            json={"project_dir": pdir, "message": "第1个镜头换成摇滚乐队"})
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            self.assertEqual(calls, ["第1个镜头换成摇滚乐队"], body)
            self.assertEqual(body["intent"], "edit", body)
            self.assertEqual(body["ops"][0]["query"], "摇滚乐队", body)

    def test_chat_rejects_invalid_model_plan(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, _proj, pdir = _build_with_model_planner(
                Path(d),
                lambda state, message, previous_targets=None: {
                    "reply": "已经修改成功。",
                    "ops": [{"op": "replace_footage", "shot_id": "NOPE", "query": "大海"}],
                },
            )
            r = client.post("/api/0907/project/chat",
                            json={"project_dir": pdir, "message": "把后面那个画面换成大海"})
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            self.assertEqual(body["intent"], "clarify", body)
            self.assertEqual(body["ops"], [], body)
            self.assertNotIn("修改成功", r.text)

    def test_chat_needs_confirm_returns_confirmation_required(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, proj, pdir = _build(Path(d))
            r = client.post("/api/0907/project/chat",
                            json={"project_dir": pdir, "message": "第1个镜头改到20秒"})
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            self.assertEqual(body["intent"], "edit")
            self.assertIn("set_shot_duration", body.get("needs_confirm", []))
            self.assertFalse(body.get("applied"))
            self.assertEqual(body["transaction"]["state"], "confirmation_required")
            # not applied yet
            self.assertEqual(proj.shot("S01")["duration_sec"], 4)

    def test_chat_low_risk_persists_without_optimistic_success(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            # no-op runner: transaction stays persisted, never reaches succeeded
            client, proj, pdir = _build(Path(d), runner=lambda mgr, tx: None)
            r = client.post("/api/0907/project/chat",
                            json={"project_dir": pdir,
                                  "message": "把素材子镜头2.1换成实验室近景"})
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            self.assertEqual(body["intent"], "edit")
            self.assertTrue(body.get("applied"))
            self.assertEqual(body["transaction"]["state"], "persisted")
            self.assertNotIn("succeeded", r.text)
            self.assertNotIn("修改成功", r.text)

    def test_confirm_then_success_only_after_verification(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            outdir = Path(d) / "out"
            outdir.mkdir()
            client, proj, pdir = _build(Path(d), runner=_success_runner(outdir))
            r = client.post("/api/0907/project/chat",
                            json={"project_dir": pdir, "message": "第1个镜头改到20秒"})
            txid = r.json()["transaction"]["transaction_id"]
            self.assertEqual(proj.shot("S01")["duration_sec"], 4)  # deferred
            c = client.post(f"/api/0907/project/transactions/{txid}/confirm",
                            json={"project_dir": pdir})
            self.assertEqual(c.status_code, 200, c.text)
            self.assertEqual(proj.shot("S01")["duration_sec"], 20.0)
            self.assertEqual(c.json()["transaction"]["state"], "succeeded")

    def test_rollback_endpoint_restores_state(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, proj, pdir = _build(Path(d), runner=lambda mgr, tx: None)
            r = client.post("/api/0907/project/ops",
                            json={"project_dir": pdir,
                                  "ops": [{"op": "set_shot_duration",
                                           "shot_id": "S01", "duration_sec": 12.0}]})
            self.assertEqual(r.status_code, 200, r.text)
            txid = r.json()["transaction"]["transaction_id"]
            self.assertEqual(proj.shot("S01")["duration_sec"], 12.0)
            rb = client.post(f"/api/0907/project/transactions/{txid}/rollback",
                             json={"project_dir": pdir})
            self.assertEqual(rb.status_code, 200, rb.text)
            self.assertEqual(rb.json()["transaction"]["state"], "rolled_back")
            self.assertEqual(proj.shot("S01")["duration_sec"], 4)

    def test_retry_endpoint_exists(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, proj, pdir = _build(Path(d), runner=lambda mgr, tx: None)
            r = client.post("/api/0907/project/ops",
                            json={"project_dir": pdir,
                                  "ops": [{"op": "set_shot_duration",
                                           "shot_id": "S01", "duration_sec": 8.0}]})
            txid = r.json()["transaction"]["transaction_id"]
            retry = client.post(f"/api/0907/project/transactions/{txid}/retry",
                                json={"project_dir": pdir})
            self.assertNotEqual(retry.status_code, 404, retry.text)

    def test_regenerate_endpoint_exists(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, proj, pdir = _build(Path(d), runner=lambda mgr, tx: None)
            r = client.post("/api/0907/project/ops",
                            json={"project_dir": pdir,
                                  "ops": [{"op": "set_shot_duration",
                                           "shot_id": "S01", "duration_sec": 8.0}]})
            txid = r.json()["transaction"]["transaction_id"]
            rg = client.post("/api/0907/project/regenerate",
                             json={"project_dir": pdir, "transaction_id": txid})
            self.assertEqual(rg.status_code, 200, rg.text)
            self.assertIn("transaction", rg.json())

    def test_ops_endpoint_applies(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, proj, pdir = _build(Path(d), runner=lambda mgr, tx: None)
            r = client.post("/api/0907/project/ops",
                            json={"project_dir": pdir,
                                  "ops": [{"op": "set_shot_duration",
                                           "shot_id": "S01", "duration_sec": 7.0}]})
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            self.assertTrue(body.get("applied"))
            self.assertEqual(proj.shot("S01")["duration_sec"], 7.0)
            self.assertIn("transaction", body)

    def test_chat_history_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, proj, pdir = _build(Path(d), runner=lambda mgr, tx: None)
            client.post("/api/0907/project/chat",
                        json={"project_dir": pdir, "message": "把素材子镜头2.1换成夜景"})
            h = client.get("/api/0907/project/chat/history", params={"dir": pdir})
            self.assertEqual(h.status_code, 200, h.text)
            msgs = h.json().get("messages", [])
            self.assertTrue(any(m.get("role") == "user" for m in msgs))

    def test_quality_check_returns_real_report(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, proj, pdir = _build(Path(d))
            r = client.post("/api/0907/project/quality-check",
                            json={"project_dir": pdir})
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            # no more not_implemented — a real, computed report is returned.
            self.assertNotEqual(body.get("status"), "not_implemented")
            self.assertIn(body.get("status"), ("pass", "warning", "fail"))
            self.assertIsInstance(body.get("score"), int)
            self.assertIn("summary", body)
            self.assertIn("proposals", body)
            self.assertEqual(body.get("policy_version"), "0907")
            self.assertNotIn("not_implemented", r.text)

    def test_quality_check_scope_and_shot_ids(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, proj, pdir = _build(Path(d))
            r = client.post("/api/0907/project/quality-check",
                            json={"project_dir": pdir, "shot_ids": ["S02"]})
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            self.assertEqual(body.get("shot_ids"), ["S02"])
            for issue in body.get("issues", []):
                sid = issue.get("shot_id") or ""
                self.assertIn(sid, ("", "S02"), issue)

    def test_quality_check_apply_opens_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, proj, pdir = _build(Path(d), runner=lambda mgr, tx: None)
            r = client.post("/api/0907/project/quality-check",
                            json={"project_dir": pdir, "apply": True})
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            # only when there is something safe to fix do we open a tx.
            if body.get("proposals"):
                self.assertIn("transaction", body)
                self.assertTrue(body.get("applied"))
                self.assertTrue(any(o.get("op") == "optimize_cut" for o in body.get("ops", [])))

    def test_project_state_persists_quality_report0907(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, proj, pdir = _build(Path(d))
            client.post("/api/0907/project/quality-check", json={"project_dir": pdir})
            self.assertIn("quality_report0907", proj.state)
            # a reload keeps the namespaced field (legacy readers ignore it).
            import studio.project as project_mod2
            reloaded = project_mod2.load(pdir)
            self.assertIn("quality_report0907", reloaded.state)

    def test_project_quality_accepts_normalized_top_level_output_shape(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, proj, pdir = _build(Path(d))
            proj.shot_assets.setdefault("S01", {})["audio_meta"] = {
                "duration_sec": 4.0,
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
            r = client.get("/api/0907/project", params={"dir": pdir})
            self.assertEqual(r.status_code, 200, r.text)
            quality = r.json()["quality"]
            # a within-target normalized loudness must not raise any loudness issue
            codes = {i.get("code") for i in quality.get("issues", [])}
            self.assertNotIn("loudness_out_of_target", codes, quality)
            self.assertNotIn("loudness_not_normalized", codes, quality)
            self.assertIn(quality.get("status"), ("pass", "warning", "fail"))
            self.assertIsInstance(quality.get("score"), int)

    def test_quality_overrides_persist_and_exclude_from_proposals(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, proj, pdir = _build(Path(d))
            # ignore a soft, auto-fixable issue (S02 has no audio => loudness_missing)
            r = client.post("/api/0907/project/quality-check", json={
                "project_dir": pdir,
                "overrides": [{"code": "loudness_missing", "shot_id": "S02", "action": "ignore"}],
            })
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            issue = next((i for i in body["issues"]
                          if i["code"] == "loudness_missing" and i["shot_id"] == "S02"), None)
            self.assertIsNotNone(issue, body)
            self.assertTrue(issue["user_override"])
            self.assertFalse(issue["auto_fixable"])
            self.assertFalse(
                any(f.get("shot_id") == "S02" and f.get("action") == "normalize_audio"
                    for f in body.get("proposals", [])),
                body.get("proposals"),
            )
            # persisted + survives reload (whitelisted key)
            self.assertIn("quality_overrides0907", proj.state)
            import studio.project as pm
            reloaded = pm.load(pdir)
            self.assertIn("quality_overrides0907", reloaded.state)
            from studio.newvis0907 import quality as q
            rep = q.evaluate_project(reloaded.state)
            again = next((i for i in rep["issues"]
                          if i["code"] == "loudness_missing" and i["shot_id"] == "S02"), None)
            self.assertIsNotNone(again)
            self.assertTrue(again["user_override"])

    def test_quality_override_restore_auto_readmits_fix(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, proj, pdir = _build(Path(d))
            client.post("/api/0907/project/quality-check", json={
                "project_dir": pdir,
                "overrides": [{"code": "loudness_missing", "shot_id": "S02", "action": "ignore"}],
            })
            r = client.post("/api/0907/project/quality-check", json={
                "project_dir": pdir,
                "overrides": [{"code": "loudness_missing", "shot_id": "S02", "action": "auto"}],
            })
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            issue = next((i for i in body["issues"]
                          if i["code"] == "loudness_missing" and i["shot_id"] == "S02"), None)
            self.assertIsNotNone(issue)
            self.assertFalse(issue["user_override"])
            self.assertTrue(
                any(f.get("shot_id") == "S02" and f.get("action") == "normalize_audio"
                    for f in body.get("proposals", [])),
                body.get("proposals"),
            )
            # the mapping is gone after restore
            self.assertNotIn("loudness_missing|S02|",
                             set(proj.state.get("quality_overrides0907", {}).keys()))

    def test_rollback_refreshes_persisted_quality_report(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, proj, pdir = _build(Path(d), runner=lambda mgr, tx: None)
            r = client.post("/api/0907/project/ops", json={
                "project_dir": pdir,
                "ops": [{"op": "set_shot_duration", "shot_id": "S01", "duration_sec": 12.0}]})
            txid = r.json()["transaction"]["transaction_id"]
            # store a report that reflects the *edited* state
            client.post("/api/0907/project/quality-check", json={"project_dir": pdir})
            rb = client.post(f"/api/0907/project/transactions/{txid}/rollback",
                             json={"project_dir": pdir})
            self.assertEqual(rb.status_code, 200, rb.text)
            self.assertEqual(proj.shot("S01")["duration_sec"], 4)
            # the persisted report must match a fresh evaluate of the restored
            # state — never stale.
            from studio.newvis0907 import quality as q
            fresh = q.evaluate_project(proj.state)
            stored = proj.state.get("quality_report0907", {})
            self.assertEqual(stored.get("score"), fresh["score"], stored)
            self.assertEqual(stored.get("status"), fresh["status"])

    def test_to_payload_whitelists_0907_keys_and_caps_requests(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, proj, pdir = _build(Path(d))
            import studio.project as pm
            proj.state["quality_report0907"] = {"status": "pass", "score": 100}
            proj.state["quality_overrides0907"] = {"code|S1|": {"action": "ignore"}}
            proj.state["quality_requests0907"] = [{"i": i} for i in range(150)]
            proj.state["stray0907"] = {"leak": True}  # NOT whitelisted
            proj.save()
            reloaded = pm.load(pdir)
            self.assertIn("quality_report0907", reloaded.state)
            self.assertIn("quality_overrides0907", reloaded.state)
            # a non-whitelisted 0907 key is never persisted (no endswith sweep).
            self.assertNotIn("stray0907", reloaded.state)
            # append-only request log capped to the most recent 100.
            self.assertEqual(len(reloaded.state["quality_requests0907"]), 100)
            self.assertEqual(reloaded.state["quality_requests0907"][0], {"i": 50})

    def test_delete_and_export_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            client, proj, pdir = _build(Path(d))
            de = client.request("DELETE", "/api/0907/project", params={"dir": pdir})
            self.assertIn(de.status_code, (200, 409), de.text)
            de2 = client.post("/api/0907/project/delete", params={"dir": pdir})
            self.assertIn(de2.status_code, (200, 409), de2.text)
            ex = client.post("/api/0907/project/export", json={"project_dir": pdir})
            self.assertEqual(ex.status_code, 200, ex.text)
            self.assertEqual(ex.json().get("status"), "started")


if __name__ == "__main__":
    unittest.main(verbosity=2)
