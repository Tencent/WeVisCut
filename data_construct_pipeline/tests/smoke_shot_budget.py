"""Replay recorded runs through the shot budget to confirm plans stop collapsing.

``Tool_PlanShots`` is called once per segment and carries that segment's id, so
its ``num_shots_target`` counts shots for that segment. Read as a whole-video
total it inverted: a 5-segment script asking for 5 shots each was clamped to 5
shots overall, one per segment, and a one-shot segment leaves nothing to mark as
a spotlight — which is why 8 of 20 episodes produced no decoration samples.

Feeds each finished episode's real segments and real orchestrator arguments to
the live ``_segment_shot_budgets`` and compares against the full-media baseline
for the same script.

Run::

    AICUT_PIPELINE_ROOT=/path/to/AICutPipeline python3 tests/smoke_shot_budget.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.append(str(PKG_SRC))

PIPELINE_ROOT = os.environ.get("AICUT_PIPELINE_ROOT") or os.environ.get("WEVISCUT_ROOT", "")
CORPUS = Path(os.environ.get("LVP_SMOKE_CORPUS", ""))
# Same script as episode 000001, planned with every media stage on.
BASELINE_SHOTS = 18

_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        _failures.append(label)


def planshots_args(episode: Path) -> dict:
    """The orchestrator's first Tool_PlanShots arguments for this episode."""
    calls = episode / "trajectories" / "llm_calls.jsonl"
    if not calls.is_file():
        return {}
    for line in calls.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("tool_name") == "Tool_PlanShots":
            return row.get("arguments") or {}
    return {}


def main() -> int:
    from longvideo_polar.pipeline_bridge import bootstrap_pipeline_paths

    if not PIPELINE_ROOT:
        print("skip: set AICUT_PIPELINE_ROOT or WEVISCUT_ROOT to a checkout with server/infer_graph.py")
        return 0
    bootstrap_pipeline_paths(PIPELINE_ROOT)
    from server.infer_graph import InferenceOrchestrator  # type: ignore[import-not-found]

    orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
    orchestrator.pipeline_config = {}
    budgets_of = InferenceOrchestrator._segment_shot_budgets

    print("=== semantics ===")
    segments = [
        {"segment_id": f"seg_{i:02d}", "title": "t", "summary": "s" * 200,
         "key_points": ["a", "b", "c"], "narration_script": "字" * 400}
        for i in range(1, 6)
    ]
    as_total = budgets_of(orchestrator, segments, requested_total=5)
    as_per_seg = budgets_of(orchestrator, segments, requested_per_segment=5)
    check(
        "read as a whole-video total, 5 segments collapse to one shot each",
        sum(as_total.values()) == 5,
        f"total={sum(as_total.values())} {as_total}",
    )
    check(
        "read per segment, the same request plans five shots per segment",
        sum(as_per_seg.values()) == 25,
        f"total={sum(as_per_seg.values())} {as_per_seg}",
    )
    check(
        "no segment is left with a single shot",
        min(as_per_seg.values()) > 1,
        f"min={min(as_per_seg.values())}",
    )
    natural = budgets_of(orchestrator, segments)
    check(
        "with no request at all the char-weight default still applies",
        sum(natural.values()) > len(segments),
        f"total={sum(natural.values())}",
    )

    print("\n=== replayed against the recorded batch ===")
    print(f"  {'episode':<34} {'segs':>4} {'was':>4} {'now':>4}")
    replayed = 0
    still_flat = []
    ep_000001 = 0
    for episode in sorted(CORPUS.iterdir()) if CORPUS.is_dir() else []:
        states = glob.glob(str(episode / "aicut_*" / "final_state.json"))
        if not states:
            continue
        state = json.loads(Path(states[0]).read_text(encoding="utf-8"))
        segs = state.get("segments") or []
        args = planshots_args(episode)
        if not segs or not args:
            continue
        try:
            asked = int(float(str(args.get("num_shots_target") or 0)))
        except (TypeError, ValueError):
            asked = 0
        scoped = bool(str(args.get("segment_id") or "").strip())
        now = budgets_of(
            orchestrator, segs,
            requested_total=0 if scoped else max(0, asked),
            requested_per_segment=max(0, asked) if scoped else 0,
        )
        was = len(state.get("shot_plan") or [])
        total = sum(now.values())
        replayed += 1
        if total <= len(segs):
            still_flat.append(episode.name)
        if episode.name.startswith("000001"):
            ep_000001 = total
        print(f"  {episode.name[:34]:<34} {len(segs):>4} {was:>4} {total:>4}")

    print()
    check("episodes were replayed", replayed > 0, f"replayed={replayed}")
    check(
        "no episode is left at one shot per segment",
        not still_flat,
        ", ".join(still_flat) if still_flat else "",
    )
    check(
        "episode 000001 lands near its full-media baseline of 18 shots",
        abs(ep_000001 - BASELINE_SHOTS) <= 6,
        f"now={ep_000001}, baseline={BASELINE_SHOTS}",
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
