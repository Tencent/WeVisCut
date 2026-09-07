"""Offline tests for run history durability (no GPU, no model).

The studio used to keep the whole execution narrative in ``RunStore`` only, so
a restart — or simply coming back the next day — left the conversation column
empty for a project that had obviously been produced. These tests cover the
journal that fixes it: events are appended to disk as they are emitted, and a
fresh store rebuilds the run list and replays the trajectory from those files.

Run:  python3 tests/test_run_journal.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from studio.runs import EVENTS_FILENAME, META_FILENAME, RunStore  # noqa: E402


class _FakeSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


def _emit_a_run(store: RunStore, run_id: str = "run_test_0001") -> None:
    run = store.create("一支 60 秒的城市清晨短片", run_id=run_id)
    run.project_dir = "/tmp/projects/aicut_demo"
    asyncio.run(_emit_all(store, run_id))
    run.status = "completed"
    run.result = {"final_video_path": "", "project_dir": run.project_dir}
    store.save_meta(run)


async def _emit_all(store: RunStore, run_id: str) -> None:
    await store.emit(run_id, {"type": "run_started", "run_id": run_id})
    await store.emit(run_id, {"type": "thinking", "step": 1, "thinking": "先分段"})
    await store.emit(run_id, {"type": "tool_call", "tool_name": "Tool_SegmentArticle"})
    await store.emit(run_id, {
        "type": "tool_result", "tool_name": "Tool_SegmentArticle", "status": "success",
    })
    await store.emit(run_id, {"type": "progress", "progress": {"overall_pct": 40}})


class JournalWriteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="journal_"))
        self.store = RunStore(journal_root=self.tmp)

    def test_events_land_on_disk(self):
        _emit_a_run(self.store)
        lines = (self.tmp / "run_test_0001" / EVENTS_FILENAME).read_text().splitlines()
        types = [json.loads(l)["type"] for l in lines]

        self.assertIn("thinking", types)
        self.assertIn("tool_call", types)
        self.assertIn("tool_result", types)
        # Progress is a snapshot, not history: journalling every one of them
        # would dwarf the narrative it is meant to accompany.
        self.assertNotIn("progress", types)

    def test_meta_records_the_outcome(self):
        _emit_a_run(self.store)
        meta = json.loads((self.tmp / "run_test_0001" / META_FILENAME).read_text())

        self.assertEqual(meta["id"], "run_test_0001")
        self.assertEqual(meta["status"], "completed")
        self.assertEqual(meta["project_dir"], "/tmp/projects/aicut_demo")
        self.assertIn("城市清晨", meta["input_text"])
        self.assertEqual(meta["progress"]["overall_pct"], 40)

    def test_a_broken_journal_dir_does_not_break_the_run(self):
        store = RunStore(journal_root=self.tmp / "nested" / "deep")
        _emit_a_run(store, "run_ok")
        self.assertEqual(store.get("run_ok").status, "completed")

    def test_delete_removes_the_journal(self):
        _emit_a_run(self.store)
        self.assertTrue((self.tmp / "run_test_0001").is_dir())
        self.assertTrue(self.store.delete("run_test_0001"))
        self.assertFalse((self.tmp / "run_test_0001").exists())


class RehydrateTest(unittest.TestCase):
    """A new process must see what the previous one recorded."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="journal_"))
        _emit_a_run(RunStore(journal_root=self.tmp))
        self.fresh = RunStore(journal_root=self.tmp)

    def test_run_list_is_empty_before_rehydrate(self):
        self.assertEqual(self.fresh.list(), [])

    def test_rehydrate_restores_the_summary(self):
        self.assertEqual(self.fresh.rehydrate(), 1)
        runs = self.fresh.list()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["status"], "completed")
        self.assertEqual(runs[0]["project_dir"], "/tmp/projects/aicut_demo")
        self.assertIn("城市清晨", runs[0]["title"])

    def test_rehydrate_is_idempotent(self):
        self.assertEqual(self.fresh.rehydrate(), 1)
        self.assertEqual(self.fresh.rehydrate(), 0)
        self.assertEqual(len(self.fresh.list()), 1)

    def test_replay_reads_the_journal_for_a_restored_run(self):
        self.fresh.rehydrate()
        sock = _FakeSocket()
        asyncio.run(self.fresh.replay("run_test_0001", sock))

        types = [e["type"] for e in sock.sent]
        self.assertEqual(types[0], "replay_start")
        self.assertIn("thinking", types)
        self.assertIn("tool_result", types)
        self.assertIn("replay_end", types)
        # A finished run must also announce that it is finished, or the UI
        # would stay locked waiting for a run that ended yesterday.
        self.assertEqual(types[-1], "run_complete")

    def test_read_journal_for_unknown_run_is_empty(self):
        self.assertEqual(self.fresh.read_journal("nope"), [])

    def test_a_run_interrupted_mid_flight_is_reported_as_cancelled(self):
        # meta.json says "running" because the process died before finishing.
        meta = self.tmp / "run_test_0001" / META_FILENAME
        payload = json.loads(meta.read_text())
        payload["status"] = "running"
        meta.write_text(json.dumps(payload))

        store = RunStore(journal_root=self.tmp)
        store.rehydrate()
        run = store.get("run_test_0001")
        self.assertEqual(run.status, "cancelled")
        self.assertFalse(run.is_active)      # nothing is driving it any more

    def test_restored_elapsed_is_frozen_not_a_live_clock(self):
        self.fresh.rehydrate()
        run = self.fresh.get("run_test_0001")
        first = run.elapsed
        self.assertEqual(first, run.elapsed)


class NoJournalTest(unittest.TestCase):
    """Without a journal root the store must behave exactly as before."""

    def test_in_memory_only(self):
        store = RunStore()
        _emit_a_run(store)
        run = store.get("run_test_0001")
        self.assertIsNone(run.journal_dir)
        self.assertEqual(len(store.list()), 1)
        self.assertEqual(store.rehydrate(), 0)
        self.assertTrue(store.delete("run_test_0001"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
