"""Replay real recorded completions through both JSON parsers.

Qwen3.5's chat template primes ``<think>``, so completions carry only the
closing tag. Verifies that reasoning is now separated from the answer, and —
the part that actually mattered in production — that the pipeline's own
``_parse_skill_json`` recovers the requested keys from what we hand it.

Run::

    python3 tests/smoke_reasoning_split.py [episode_dir ...]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

from longvideo_polar.model_client import _split_reasoning  # noqa: E402

DEFAULT_EPISODES = [
    p for p in os.environ.get("LVP_SMOKE_EPISODES", "").split(os.pathsep) if p
]

# The key each skill's caller reads off the parsed object; an answer that parses
# but lacks it is exactly the "empty decoration" / "no shots" failure mode.
REQUIRED_KEY = {
    "template_fill": "slots",
    "plan_shots": "shots",
    "segment_article": "segments",
    "template_retrieve": "template_id",
}


def pipeline_parse(text: str) -> dict:
    """The pipeline's parser, transcribed from ``infer_graph._parse_skill_json``."""
    import re

    text = str(text or "").strip()
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    text = re.sub(r"</?tool_call>", "", text).strip()
    text = re.sub(r"```(?:json)?\s*\n?", "", text).strip()
    text = re.sub(r"```\s*$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, TypeError):
        pass
    start = text.find("{")
    if start < 0:
        return {}
    depth, in_str, esc, best_end = 0, False, False, -1
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                best_end = i
                break
    if best_end > start:
        try:
            obj = json.loads(text[start:best_end + 1])
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def check_shapes() -> list[str]:
    """Shapes the splitter has to survive besides the primed-think one."""
    cases = [
        ("primed open tag", "reasoning {draft}\n</think>\n{\"slots\": {\"a\": 1}}", "{\"slots\": {\"a\": 1}}"),
        ("well-formed pair", "<think>reasoning</think>{\"slots\": {}}", "{\"slots\": {}}"),
        ("no reasoning at all", "{\"slots\": {}}", "{\"slots\": {}}"),
        ("fenced answer", "think\n</think>\n```json\n{\"a\": 1}\n```", "```json\n{\"a\": 1}\n```"),
        ("reasoning only, no answer", "still thinking</think>", ""),
    ]
    failures = []
    print("=== splitter shapes ===")
    for label, raw, expected in cases:
        got = _split_reasoning(raw)[1]
        ok = got == expected
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            print(f"        expected {expected!r}\n        got      {got!r}")
            failures.append(label)
    return failures


def main(argv: list[str]) -> int:
    shape_failures = check_shapes()
    episodes = argv[1:] or DEFAULT_EPISODES
    before_ok = after_ok = total = 0
    regressions: list[str] = []

    for episode in episodes:
        shard = Path(episode) / "trajectories" / "generator.jsonl"
        if not shard.is_file():
            print(f"skip (no generator shard): {episode}")
            continue
        document = json.loads(shard.read_text(encoding="utf-8"))
        print(f"\n=== {Path(episode).name} ===")
        per_task: dict[str, list[int]] = {}

        for sample in document.get("samples") or []:
            meta = sample.get("metadata") or {}
            task = str(meta.get("stage_task") or "")
            need = REQUIRED_KEY.get(task)
            if not need:
                continue
            raw = sample["messages"][-1]["content"]

            old = need in pipeline_parse(raw)
            new = need in pipeline_parse(_split_reasoning(raw)[1])
            total += 1
            before_ok += int(old)
            after_ok += int(new)
            counts = per_task.setdefault(task, [0, 0, 0])
            counts[0] += 1
            counts[1] += int(old)
            counts[2] += int(new)
            if old and not new:
                regressions.append(f"{task} #{meta.get('llm_call_index')}")

        for task, (n, old_n, new_n) in sorted(per_task.items()):
            flag = "  <-- recovered" if new_n > old_n else ""
            print(f"  {task:<18} {n:>3} samples   before {old_n}/{n}   after {new_n}/{n}{flag}")

    print(f"\ntotal: before {before_ok}/{total}, after {after_ok}/{total}")
    if shape_failures:
        print("SHAPE FAILURES: " + ", ".join(shape_failures))
        return 1
    if regressions:
        print("REGRESSIONS (parsed before, not after): " + ", ".join(regressions))
        return 1
    if not total:
        print("no samples found — nothing verified")
        return 1
    print("no regressions" + (f"; recovered {after_ok - before_ok} sample(s)" if after_ok > before_ok else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
