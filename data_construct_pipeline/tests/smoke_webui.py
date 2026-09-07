"""Drive the control-plane HTTP API in-process, without a model endpoint.

Covers the sequence the operator actually performs in the browser: add a
machine, preview a selection, commit the worklist, read status, and reach the
export endpoint. Uses FastAPI's TestClient so no port is bound.

Run::

    python3 tests/smoke_webui.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.append(str(PKG_SRC))

CONFIG = Path(__file__).resolve().parents[1] / "config" / "campaign.example.yaml"
DATASET_ROOT = Path(os.environ.get("LVP_SMOKE_DATASET", ""))

_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        _failures.append(label)


def main() -> int:
    from fastapi.testclient import TestClient

    from longvideo_polar.webui.server import create_app

    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(
            config_path=CONFIG,
            dataset_root=DATASET_ROOT,
            output_root=Path(tmp),
            bootstrap_endpoints=[],
        )
        client = TestClient(app)

        print("=== static assets ===")
        page = client.get("/")
        check("index page is served", page.status_code == 200, f"HTTP {page.status_code}")
        check("page mounts the app script", "app.js" in page.text)
        for asset in ("/static/app.js", "/static/app.css"):
            response = client.get(asset)
            check(f"{asset} is served", response.status_code == 200)

        print("\n=== machine pool ===")
        added = client.post("/api/endpoints", json={"address": "10.0.0.1:8080"})
        check("a machine can be added as bare ip:port", added.status_code == 200,
              f"HTTP {added.status_code} {added.text[:120]}")
        listed = client.get("/api/endpoints").json()
        pool = listed.get("endpoints", listed)
        check("the machine shows up in the pool", len(pool) == 1, str(pool)[:200])
        check(
            "an unreachable machine is reported, not crashed on",
            pool[0].get("healthy") is False and bool(pool[0].get("last_error")),
            str(pool[0].get("last_error"))[:120],
        )
        client.post("/api/endpoints", json={"address": "10.0.0.2:8080"})
        check("a second machine can be added live", len(client.get("/api/endpoints").json()["endpoints"]) == 2)
        removed = client.delete("/api/endpoints", params={"address": "10.0.0.2:8080"})
        check("a machine can be removed", removed.status_code == 200,
              f"HTTP {removed.status_code}")

        print("\n=== selection ===")
        preview = client.post(
            "/api/selection/preview",
            json={"min_chars": 800, "max_chars": 8000, "languages": ["zh"], "limit": 50},
        )
        check("preview answers", preview.status_code == 200, f"HTTP {preview.status_code} {preview.text[:200]}")
        body = preview.json()
        check(
            "preview reports the kept count",
            (body.get("stats") or {}).get("count") == 50,
            str(body.get("stats"))[:200],
        )
        check("preview reports why rows were dropped", bool(body.get("rejected")), str(body.get("rejected")))
        check(
            "preview shows sample folder names",
            bool(body.get("preview")) and all(row.get("episode_id") for row in body["preview"]),
            ", ".join(row["episode_id"] for row in (body.get("preview") or [])[:3]),
        )

        commit = client.post(
            "/api/selection/commit",
            json={"min_chars": 800, "max_chars": 8000, "languages": ["zh"], "limit": 50},
        )
        check("worklist commits", commit.status_code == 200, f"HTTP {commit.status_code} {commit.text[:200]}")
        check(
            "worklist lands on disk",
            (Path(tmp) / "_state" / "worklist.jsonl").is_file()
            or any(Path(tmp).rglob("worklist.jsonl")),
        )

        print("\n=== campaign status ===")
        status = client.get("/api/status")
        check("status answers before any run", status.status_code == 200, f"HTTP {status.status_code}")
        payload = status.json()
        summary = payload.get("summary") or {}
        check(
            "status carries progress counters",
            {"phase", "total", "settled", "progress_pct"} <= set(summary),
            str(summary)[:200],
        )
        check("status carries the machine pool", "pool" in payload)
        check("status carries the episode table", isinstance(payload.get("episodes"), list))

        worklist = client.get("/api/worklist").json()
        check("committed worklist is readable back", worklist.get("count") == 50, str(worklist)[:160])

        print("\n=== controls with no campaign running ===")
        for path in ("/api/campaign/pause", "/api/campaign/resume", "/api/campaign/stop"):
            response = client.post(path)
            check(f"{path} refuses cleanly", response.status_code == 400, f"HTTP {response.status_code}")
        modes = client.get("/api/modes").json()
        check("modes are advertised to the UI", bool(modes.get("modes")),
              ", ".join(m.get("id", "?") for m in modes.get("modes", [])))

        print("\n=== export ===")
        export = client.post("/api/export", json={})
        check(
            "export endpoint responds on an empty corpus",
            export.status_code in (200, 400, 409),
            f"HTTP {export.status_code} {export.text[:160]}",
        )

    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed:")
        for name in _failures:
            print(f"  - {name}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
