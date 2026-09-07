"""Let a media-free run walk the whole arc and still terminate.

``InferenceOrchestrator`` gates each tool on flags its predecessors set. Two of
those flags can never be set when the media stages are switched off, because the
disabled branches return a bare ``{"status": "skipped"}``::

    if tool_name == "Tool_BatchTimelineCompose":
        render_ok = int(output.get("render_success_count") or 0)
        self.timeline_done = bool(render_ok > 0)      # False when skipped

``Tool_StitchVideo`` — the only tool that breaks the loop — requires
``timeline_render``, so without help the agent is told it cannot stitch until it
runs out of steps.

The tempting fix, waiving that precondition outright, is wrong, and measurably
so: ``timeline_render`` is what transitively forces the quality arc to happen,
since ``Tool_BatchTimelineCompose`` itself requires ``dimension_quality_check``.
Waiving it let two real episodes jump from decoration straight to stitching in
seven steps, skipping severity ranking, dimension judging and repair, and
leaving the critic stream with zero ``quality_judge`` samples.

So the flag is satisfied by *invocation* instead. Calling a switched-off stage
still counts as reaching it — the orchestrator's trajectory records the tool call,
which is the behaviour we want to teach — but the ordering constraints in front
of it survive untouched. Only ``rendered_path_ready``, which inspects rendered
files on disk and no amount of tool calling can satisfy, is genuinely waived.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_FLAGS_PATCH = "_lvp_switch_aware_flags"
_PRECOND_PATCH = "_lvp_switch_aware_preconditions"
_APPLY_PATCH = "_lvp_switch_aware_apply"
_INVOKED_ATTR = "_lvp_invoked_tools"

# Flag -> (tool whose invocation stands in for it, switches that must be off for
# the substitution to apply). With the switch on, the tool sets the flag itself
# from real work and nothing here interferes.
_INVOCATION_SATISFIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "timeline_done": ("Tool_BatchTimelineCompose", ("timeline_compose",)),
    "visual_audio_done": ("Tool_BatchVisualAudio", ("tts", "visual_retrieval")),
    "bgm_done": ("Tool_MusicSearch", ("bgm_search",)),
}

# Preconditions that inspect produced files rather than progress, so no tool call
# can satisfy them once rendering is off.
_ARTIFACT_LABELS: dict[str, tuple[str, ...]] = {
    "rendered_path_ready": ("timeline_compose",),
}


def _disabled(switches: Any, names: tuple[str, ...]) -> bool:
    return not all(bool(getattr(switches, name, True)) for name in names)


def substituted_flags(switches: Any) -> dict[str, str]:
    """Flags that an invocation stands in for, keyed by flag name."""
    return {
        flag: tool
        for flag, (tool, required) in _INVOCATION_SATISFIES.items()
        if _disabled(switches, required)
    }


def waived_preconditions(switches: Any) -> set[str]:
    """Artifact checks that are unreachable under the current switches."""
    return {
        label
        for label, required in _ARTIFACT_LABELS.items()
        if _disabled(switches, required)
    }


def install_switch_aware_preconditions(orchestrator_cls: type, switches: Any) -> dict[str, Any]:
    """Make switched-off stages advance the run instead of deadlocking it.

    Args:
        orchestrator_cls: ``InferenceOrchestrator``.
        switches: The run's :class:`~longvideo_polar.config.ToolSwitches`.

    Returns:
        ``{"substituted": {flag: tool}, "waived": {label}}``, both empty when
        every subsystem is on.
    """
    substituted = substituted_flags(switches)
    waived = waived_preconditions(switches)
    if not substituted and not waived:
        return {"substituted": {}, "waived": set()}

    # Re-installing in a live process must rebuild from the pristine methods,
    # or the wrappers stack and each layer widens the previous one's allowances.
    original_flags = getattr(orchestrator_cls, _FLAGS_PATCH, None) or orchestrator_cls._agent_flags
    original_precond = (
        getattr(orchestrator_cls, _PRECOND_PATCH, None)
        or orchestrator_cls._agent_precondition_missing
    )
    original_apply = (
        getattr(orchestrator_cls, _APPLY_PATCH, None) or orchestrator_cls._apply_tool_output
    )

    def _patched_apply(self: Any, tool_name: str, output: dict[str, Any], state: dict[str, Any]) -> None:
        original_apply(self, tool_name, output, state)
        invoked = getattr(self, _INVOKED_ATTR, None)
        if invoked is None:
            invoked = set()
            setattr(self, _INVOKED_ATTR, invoked)
        invoked.add(str(tool_name))

    def _patched_flags(self: Any, state: dict[str, Any]) -> dict[str, Any]:
        flags = original_flags(self, state)
        invoked = getattr(self, _INVOKED_ATTR, None) or set()
        for flag, tool in substituted.items():
            if not flags.get(flag) and tool in invoked:
                flags[flag] = True
        return flags

    def _patched_precond(self: Any, tool_name: str, state: dict[str, Any]) -> list[str]:
        missing = original_precond(self, tool_name, state)
        if not missing or not waived:
            return missing
        # Labels can carry a progress suffix, e.g. ``rendered_path_ready:0/18``.
        return [item for item in missing if str(item).split(":", 1)[0] not in waived]

    orchestrator_cls._apply_tool_output = _patched_apply  # noqa: SLF001
    orchestrator_cls._agent_flags = _patched_flags  # noqa: SLF001
    orchestrator_cls._agent_precondition_missing = _patched_precond  # noqa: SLF001
    setattr(orchestrator_cls, _APPLY_PATCH, original_apply)
    setattr(orchestrator_cls, _FLAGS_PATCH, original_flags)
    setattr(orchestrator_cls, _PRECOND_PATCH, original_precond)

    logger.info(
        "switch policy: %s satisfied by invocation, waiving %s",
        sorted(substituted), sorted(waived) or "nothing",
    )
    return {"substituted": substituted, "waived": waived}


def expected_skip_error(message: str, switches: Any) -> bool:
    """Whether an episode error is just a switched-off stage reporting itself.

    With ``final_stitch`` off the run still ends by calling ``Tool_StitchVideo``,
    which is what the orchestrator trajectory should show. The tool returns
    ``skipped`` and writes no file, so the loop logs a stitch-validation failure
    on its way out. That line describes the configuration, not a defect.
    """
    if bool(getattr(switches, "final_stitch", True)):
        return False
    text = str(message or "")
    return any(
        marker in text
        for marker in ("拼接验证失败", "最终视频未生成", "Stitch disabled", "未完成 StitchVideo")
    )
