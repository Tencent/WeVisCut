"""Isolation checks for the dedicated 0907 newvis surface.

Run: python3 tests/test_newvis0907_isolation.py
"""

from __future__ import annotations

import hashlib
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_FILE = ROOT / "studio" / "app.py"
NEWVIS0907_DIR = ROOT / "static" / "newvis0907"

BASELINE_0901_HASHES = {
    "index.html": "c1e87fb1a79cdbd5d2b82dba92aa0d483059b207caf1ce5bec255c1e728f6ffe",
    "newvis.css": "cd97f7072f2706e27124d8d06a310bad5bfa0541c88bb0442e0d5ac62f4f3a5e",
    "newvis.js": "cc2d8b5959393eb7fd020e56bc38099e6db4b9395904954c196f6d4569507144",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Newvis0907IsolationTest(unittest.TestCase):
    def test_route_contract_from_source(self) -> None:
        src = APP_FILE.read_text(encoding="utf-8")
        self.assertIn('@app.get("/0907_newvis")', src)
        self.assertIn(
            'return FileResponse(str(STATIC_DIR / "newvis0907" / "index.html"))',
            src,
        )
        # Existing 0901 route must remain unchanged.
        self.assertIn('@app.get("/0901_newvis")', src)
        self.assertIn(
            'return FileResponse(str(STATIC_DIR / "newvis" / "index.html"))',
            src,
        )

    def test_newvis0907_static_files_exist(self) -> None:
        expected = {
            "index.html",
            "newvis.css",
            "newvis.js",
            "navigator.css",
            "shot_navigator.js",
        }
        actual = {p.name for p in NEWVIS0907_DIR.glob("*") if p.is_file()}
        self.assertTrue(expected.issubset(actual), f"missing={sorted(expected - actual)}")

    def test_0901_baseline_files_unchanged(self) -> None:
        base = ROOT / "static" / "newvis"
        for name, digest in BASELINE_0901_HASHES.items():
            self.assertEqual(_sha256(base / name), digest, name)

    def test_storage_keys_are_isolated(self) -> None:
        js = (NEWVIS0907_DIR / "newvis.js").read_text(encoding="utf-8")
        self.assertIn("aicut.newvis0907.inputMode", js)
        self.assertIn("aicut.newvis0907.lastSession", js)

    def test_project_endpoints_use_0907_prefix(self) -> None:
        js = (NEWVIS0907_DIR / "newvis.js").read_text(encoding="utf-8")
        self.assertIn("/api/0907/project", js)
        self.assertIn("/api/0907/projects", js)
        self.assertIn("/api/0907/project/chat", js)
        self.assertIn("/api/0907/project/export", js)
        self.assertIn("/api/0907/project/delete", js)
        self.assertNotIn('"/api/project', js)
        self.assertNotIn("'/api/project", js)
        self.assertNotIn('"/api/projects', js)
        self.assertNotIn("'/api/projects", js)

    def test_newvis0907_does_not_depend_on_0901_static_js_css(self) -> None:
        js = (NEWVIS0907_DIR / "newvis.js").read_text(encoding="utf-8")
        css = (NEWVIS0907_DIR / "newvis.css").read_text(encoding="utf-8")
        self.assertNotIn("/static/newvis/newvis.js", js)
        self.assertNotIn("loadLegacyScript", js)
        self.assertNotIn("patchLocalStorage", js)
        self.assertNotIn("patchFetch", js)
        self.assertNotIn("/static/newvis/newvis.css", css)
        self.assertNotIn("@import", css)

    def test_script_dom_ids_match_index(self) -> None:
        html = (NEWVIS0907_DIR / "index.html").read_text(encoding="utf-8")
        ids = set(re.findall(r'id="([^"]+)"', html))
        scripts = [
            NEWVIS0907_DIR / "newvis.js",
            NEWVIS0907_DIR / "shot_navigator.js",
        ]
        needed: set[str] = set()
        for script in scripts:
            code = script.read_text(encoding="utf-8")
            needed.update(re.findall(r"getElementById\('([^']+)'\)", code))
        missing = sorted(k for k in needed if k not in ids)
        self.assertFalse(missing, f"missing ids: {missing}")

    def test_shot_navigator_skeleton_exists(self) -> None:
        js = (NEWVIS0907_DIR / "shot_navigator.js").read_text(encoding="utf-8")
        self.assertIn("class AICutShotNavigator", js)
        self.assertIn("constructor(root, options)", js)
        self.assertIn("update(", js)
        self.assertIn("selectShot(", js)
        self.assertIn("selectSubclip(", js)
        self.assertIn("syncTime(", js)
        self.assertIn("clear()", js)


if __name__ == "__main__":
    unittest.main(verbosity=2)
