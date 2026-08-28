"""Dump real backend payloads for the frontend test to consume.

The frontend test must not hand-write its own idea of what the API returns —
that would pass while the two sides drift apart. These fixtures come out of the
actual FastAPI app and the actual track projection, so a backend field rename
breaks the frontend test.

Run:  python tests/dump_fixtures.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

STUDIO_DIR = Path(__file__).resolve().parents[1]
if str(STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIO_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("AICUT_STUDIO_SKIP_MODEL", "1")

FIXTURE_DIR = STUDIO_DIR / "tests" / "fixtures"


def main() -> int:
    from fastapi.testclient import TestClient

    from studio import config as studio_config, tracks
    from studio.app import app
    from test_studio_offline import make_state

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, object] = {}

    cfg = studio_config.load()
    pdir = cfg.workspace_root / "projects" / "aicut_fixture_0001"
    pdir.mkdir(parents=True, exist_ok=True)
    state = make_state()
    state["project_id"] = "aicut_fixture_0001"
    state["input_text"] = "为什么大模型的上下文窗口越来越长？这期讲清楚三件事。"
    (pdir / "final_state.json").write_text(
        json.dumps(state, ensure_ascii=False), encoding="utf-8")

    try:
        with TestClient(app) as client:
            out["config"] = client.get("/api/config").json()
            out["status"] = client.get("/api/status").json()
            out["projects"] = client.get("/api/projects").json()

            r = client.get("/api/project", params={"dir": str(pdir)})
            r.raise_for_status()
            out["project"] = r.json()

            r = client.post("/api/project/ops", json={
                "project_dir": str(pdir),
                "ops": [{"op": "set_narration", "shot_id": "S01",
                         "text": "改写后的第一句，冲击力更强。"}],
            })
            r.raise_for_status()
            out["project_after_edit"] = r.json()

            out["runs"] = client.get("/api/runs").json()

        # a mid-run snapshot: shots planned, nothing rendered yet
        partial = json.loads(json.dumps(state, ensure_ascii=False))
        partial["final_video_path"] = ""
        partial.pop("stitch_result", None)
        for entry in partial["shot_assets"].values():
            entry.pop("rendered_path", None)
            entry.pop("clip_path", None)
        order = [t["key"] for t in out["config"]["tracks"]]
        out["progress_partial"] = tracks.build_snapshot(partial, order)
        out["progress_final"] = tracks.build_snapshot(state, order)
        out["project_dir"] = str(pdir)
    finally:
        shutil.rmtree(pdir, ignore_errors=True)

    target = FIXTURE_DIR / "api_payloads.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {target}")
    for key, value in out.items():
        size = len(json.dumps(value, ensure_ascii=False))
        print(f"  {key:22s} {size:>8d} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
