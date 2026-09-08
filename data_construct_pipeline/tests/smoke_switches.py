"""Check that a text-only run walks the whole arc and still finishes.

Drives the real ``InferenceOrchestrator`` gating logic through the sequence an
episode actually takes. Two properties matter and they pull against each other:

- the loop must be able to reach ``Tool_StitchVideo`` and break, or every
  episode burns its step budget and fails
- it must not reach it early, or the run skips severity ranking, dimension
  judging and repair — which is what produced seven-step episodes with zero
  ``quality_judge`` samples in the critic stream

Run::

    AICUT_PIPELINE_ROOT=/path/to/AICutPipeline python3 tests/smoke_switches.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.append(str(PKG_SRC))

PIPELINE_ROOT = os.environ.get("AICUT_PIPELINE_ROOT") or os.environ.get("WEVISIONCUT_ROOT", "")

_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        _failures.append(label)


def make_state() -> dict[str, object]:
    return {
        "segments": [{"segment_id": "seg_01", "title": "t", "summary": "s"}],
        "shot_plan": [
            {"shot_id": "1", "segment_id": "seg_01", "is_spotlight": True, "vocal_script": "a"},
            {"shot_id": "2", "segment_id": "seg_01", "is_spotlight": False, "vocal_script": "b"},
        ],
        "shot_assets": {"1": {"decoration": {"template_id": "agenda_roadmap", "slots": {}}}},
        "plan_approved": True,
    }


def fresh(cls: type) -> object:
    """An orchestrator positioned just after decoration, before quality judging."""
    orchestrator = cls.__new__(cls)
    orchestrator.visual_audio_done = True
    orchestrator.decorated_done = True
    orchestrator.bgm = {}
    orchestrator.quality_checked = False
    orchestrator.last_judge = {}
    orchestrator.repaired_done = False
    orchestrator.timeline_done = False
    orchestrator.stitched_done = False
    orchestrator.open_issues = []
    orchestrator.patch_memory = {}
    orchestrator._reflect_generation = 0
    orchestrator._last_reflect_generation = 0
    return orchestrator


def main() -> int:
    from longvideo_polar.config import ToolSwitches
    from longvideo_polar.pipeline_bridge import bootstrap_pipeline_paths
    from longvideo_polar.switch_policy import (
        expected_skip_error,
        install_switch_aware_preconditions,
        substituted_flags,
        waived_preconditions,
    )

    if not PIPELINE_ROOT:
        print("skip: set AICUT_PIPELINE_ROOT or WEVISIONCUT_ROOT to a checkout with server/infer_graph.py")
        return 0
    bootstrap_pipeline_paths(PIPELINE_ROOT)
    from server.infer_graph import InferenceOrchestrator  # type: ignore[import-not-found]

    text_only = ToolSwitches(
        tts=False, visual_retrieval=False, bgm_search=False,
        timeline_compose=False, hyperframes_render=False, final_stitch=False,
    )
    full = ToolSwitches()

    print("=== policy scope ===")
    check("full media changes nothing", not substituted_flags(full) and not waived_preconditions(full))
    check(
        "text-only substitutes flags by invocation",
        substituted_flags(text_only).get("timeline_done") == "Tool_BatchTimelineCompose",
        str(substituted_flags(text_only)),
    )
    check(
        "only the artifact check is waived outright",
        waived_preconditions(text_only) == {"rendered_path_ready"},
        str(waived_preconditions(text_only)),
    )

    print("\n=== before the patch: the loop cannot end ===")
    orchestrator = fresh(InferenceOrchestrator)
    state = make_state()
    before = InferenceOrchestrator._agent_precondition_missing(orchestrator, "Tool_StitchVideo", state)
    check("stitching is blocked", "timeline_render" in before, f"missing={before}")

    install_switch_aware_preconditions(InferenceOrchestrator, text_only)

    print("\n=== after the patch: the arc is still enforced ===")
    agent = fresh(InferenceOrchestrator)
    missing = InferenceOrchestrator._agent_precondition_missing
    state = make_state()

    check(
        "stitching is still blocked right after decoration",
        "timeline_render" in missing(agent, "Tool_StitchVideo", state),
        f"missing={missing(agent, 'Tool_StitchVideo', state)}",
    )
    check(
        "composing first requires the dimension judge to have run",
        "dimension_quality_check" in missing(agent, "Tool_BatchTimelineCompose", state),
        f"missing={missing(agent, 'Tool_BatchTimelineCompose', state)}",
    )

    # The judge runs and records a verdict.
    agent.last_judge = {"passed": False, "shot_id": "1"}
    check(
        "composing unlocks once quality has been judged",
        missing(agent, "Tool_BatchTimelineCompose", state) == [],
        f"missing={missing(agent, 'Tool_BatchTimelineCompose', state)}",
    )
    check(
        "stitching is still blocked before compose is called",
        "timeline_render" in missing(agent, "Tool_StitchVideo", state),
    )

    # Music search is a real prerequisite of stitching and must be reached too.
    check("bgm is still required", "bgm" in missing(agent, "Tool_StitchVideo", state))
    InferenceOrchestrator._apply_tool_output(
        agent, "Tool_MusicSearch", {"status": "skipped", "tracks": []}, state
    )
    check("calling music search satisfies bgm", "bgm" not in missing(agent, "Tool_StitchVideo", state))

    # Compose is invoked and reports skipped.
    InferenceOrchestrator._apply_tool_output(
        agent, "Tool_BatchTimelineCompose", {"status": "skipped", "message": "disabled"}, state
    )
    check(
        "stitching unlocks only after compose was actually called",
        missing(agent, "Tool_StitchVideo", state) == [],
        f"missing={missing(agent, 'Tool_StitchVideo', state)}",
    )

    print("\n=== ordering constraints that must survive ===")
    empty = fresh(InferenceOrchestrator)
    empty.visual_audio_done = False
    empty.decorated_done = False
    bare = {"segments": [], "shot_plan": [], "shot_assets": {}}
    check(
        "planning is still required before stitching",
        "shot_plan" in missing(empty, "Tool_StitchVideo", bare),
        str(missing(empty, "Tool_StitchVideo", bare)),
    )
    check(
        "segmentation is still required before planning shots",
        "segments" in missing(empty, "Tool_PlanShots", bare),
    )
    # A spotlight shot with no decoration yet — with no spotlights at all the
    # flag is legitimately satisfied, since there is nothing to decorate.
    undecorated = make_state()
    undecorated["shot_assets"] = {}
    check(
        "decoration is still required before judging",
        "spotlight_decoration" in missing(empty, "Tool_Stage4DimensionJudge", undecorated),
        str(missing(empty, "Tool_Stage4DimensionJudge", undecorated)),
    )
    check(
        "an invalid tool name is still rejected",
        missing(empty, "Tool_NotARealTool", bare) == ["valid_tool_name"],
    )

    print("\n=== episode status classification ===")
    check(
        "a stitch-disabled notice is not a failure in text-only",
        expected_skip_error("[final_stitch] 拼接验证失败 (concat_status=)", text_only),
    )
    check(
        "the same notice IS a failure in full media",
        not expected_skip_error("拼接验证失败 (concat_status=)", full),
    )
    check(
        "a genuine error still fails a text-only run",
        not expected_skip_error("TimeoutError: generator call timed out", text_only),
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
