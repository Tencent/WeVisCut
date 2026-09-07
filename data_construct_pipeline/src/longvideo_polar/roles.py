"""The three-part role split for the long-horizon 口播稿 → 成片 pipeline.

This module is the single source of truth for *which role owns which tool*, and
therefore for how a rollout gets split into three trajectory streams:

``orchestrator`` (主流程)
    Drives overall progress. Each turn it observes runtime state and emits exactly
    one tool call with light control parameters only. It never writes content.
    Every generator/critic invocation is downstream of an orchestrator decision,
    which is what makes the three streams causally linked rather than independent.

``generator`` (内容生成器)
    Produces content: article segmentation, shot planning, spotlight decoration,
    plan patches and quality repairs.

``critic`` (反思修复)
    Reflects and judges: plan self-reflection, per-shot dimension judging and
    severity ranking. Its findings are what trigger generator repairs.

The tool→role mapping is loaded from AICutPipeline's own
``Infer_Vis_new/server/tool_schema.py`` whenever that tree is reachable, so the
split can never silently drift from the pipeline the rollout actually executes.
A vendored snapshot is used as a fallback so this package stays importable and
testable on its own (e.g. on a Polar control plane that has no AICutPipeline
checkout).
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Final

logger = logging.getLogger(__name__)


class Role(str, Enum):
    """A trajectory stream owner."""

    ORCHESTRATOR = "orchestrator"
    GENERATOR = "generator"
    CRITIC = "critic"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


ALL_ROLES: Final[tuple[Role, ...]] = (Role.ORCHESTRATOR, Role.GENERATOR, Role.CRITIC)

ROLE_LABELS_ZH: Final[dict[Role, str]] = {
    Role.ORCHESTRATOR: "主流程",
    Role.GENERATOR: "内容生成器",
    Role.CRITIC: "反思修复",
}


# ---------------------------------------------------------------------------
# Vendored fallback snapshot of AICutPipeline's v5 tool schema.
#
# Keep in sync with Infer_Vis_new/server/tool_schema.py. `load_tool_schema()`
# prefers the live module and only falls back to these values.
# ---------------------------------------------------------------------------

_FALLBACK_ACTION_SPACE: Final[tuple[str, ...]] = (
    "Tool_SegmentArticle",
    "Tool_PlanShots",
    "Tool_SelfReflect",
    "Tool_SelfReflect_Patch",
    "Tool_BatchVisualAudio",
    "Tool_BatchDecorateSpotlights",
    "Tool_BatchTimelineCompose",
    "Tool_MusicSearch",
    "Tool_SeverityRank",
    "Tool_Stage4DimensionJudge",
    "Tool_Stage4DimensionalRepair",
    "Tool_StitchVideo",
)

_FALLBACK_GENERATOR_TOOLS: Final[frozenset[str]] = frozenset({
    "Tool_SegmentArticle",
    "Tool_PlanShots",
    "Tool_SelfReflect_Patch",
    "Tool_BatchDecorateSpotlights",
    "Tool_Stage4DimensionalRepair",
})

_FALLBACK_CRITIC_TOOLS: Final[frozenset[str]] = frozenset({
    "Tool_SelfReflect",
    "Tool_Stage4DimensionJudge",
    "Tool_SeverityRank",
})

# Executed by pipeline code (TTS, retrieval, ffmpeg, hyperframes), not by a model.
# They still belong to the orchestrator stream: the orchestrator chose them and
# observes their result.
_FALLBACK_EXTERNAL_TOOLS: Final[frozenset[str]] = frozenset({
    "Tool_BatchVisualAudio",
    "Tool_BatchTimelineCompose",
    "Tool_MusicSearch",
    "Tool_StitchVideo",
})


class ToolSchema:
    """Resolved tool metadata plus the role routing derived from it."""

    def __init__(
        self,
        *,
        action_space: tuple[str, ...],
        generator_tools: frozenset[str],
        critic_tools: frozenset[str],
        external_tools: frozenset[str],
        source: str,
    ) -> None:
        self.action_space = action_space
        self.generator_tools = generator_tools
        self.critic_tools = critic_tools
        self.external_tools = external_tools
        self.source = source

    def role_for_tool(self, tool_name: str) -> Role:
        """Return the role that *produces* the content for ``tool_name``.

        External and unknown tools map to the orchestrator, matching
        ``UnifiedModelLoader.engine_for_tool``.
        """
        if tool_name in self.generator_tools:
            return Role.GENERATOR
        if tool_name in self.critic_tools:
            return Role.CRITIC
        return Role.ORCHESTRATOR

    def is_external(self, tool_name: str) -> bool:
        """Whether the tool runs as pipeline code rather than a model call."""
        return tool_name in self.external_tools

    def tools_for_role(self, role: Role) -> tuple[str, ...]:
        """Return the action-space subset owned by ``role``."""
        return tuple(name for name in self.action_space if self.role_for_tool(name) is role)

    def describe(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot for episode metadata."""
        return {
            "source": self.source,
            "action_space": list(self.action_space),
            "orchestrator_tools": list(self.tools_for_role(Role.ORCHESTRATOR)),
            "generator_tools": sorted(self.generator_tools),
            "critic_tools": sorted(self.critic_tools),
            "external_tools": sorted(self.external_tools),
        }


def _fallback_schema() -> ToolSchema:
    return ToolSchema(
        action_space=_FALLBACK_ACTION_SPACE,
        generator_tools=_FALLBACK_GENERATOR_TOOLS,
        critic_tools=_FALLBACK_CRITIC_TOOLS,
        external_tools=_FALLBACK_EXTERNAL_TOOLS,
        source="vendored_fallback",
    )


def load_tool_schema(*, strict: bool = False) -> ToolSchema:
    """Load the tool schema, preferring AICutPipeline's live definition.

    Args:
        strict: When true, raise if AICutPipeline's ``server.tool_schema`` cannot
            be imported instead of falling back to the vendored snapshot. Use this
            for real rollouts, where a drifted schema would silently mislabel
            trajectories.

    Returns:
        The resolved :class:`ToolSchema`.

    Raises:
        RuntimeError: If ``strict`` and the live schema is unavailable.
    """
    try:
        from server.tool_schema import (  # type: ignore[import-not-found]
            CRITIC_TOOLS,
            EXTERNAL_TOOLS,
            GENERATOR_TOOLS,
        )
        from server.tool_schema import TOOL_CARDS  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - any import failure means "not reachable"
        if strict:
            raise RuntimeError(
                "AICutPipeline tool schema (server.tool_schema) is not importable; "
                "call bootstrap_pipeline_paths() first or set "
                "roles.strict_tool_schema=false in the config"
            ) from exc
        logger.debug("Falling back to vendored tool schema: %s", exc)
        return _fallback_schema()

    # The live action space is the orchestrator's; keep the vendored ordering for
    # any tool present in both so prompts stay stable across runs.
    live_names = set(TOOL_CARDS)
    ordered = [name for name in _FALLBACK_ACTION_SPACE if name in live_names]
    ordered.extend(sorted(live_names - set(ordered)))
    return ToolSchema(
        action_space=tuple(ordered),
        generator_tools=frozenset(GENERATOR_TOOLS),
        critic_tools=frozenset(CRITIC_TOOLS),
        external_tools=frozenset(EXTERNAL_TOOLS),
        source="aicutpipeline.server.tool_schema",
    )


# ---------------------------------------------------------------------------
# Generator / critic stage tasks
#
# AICutPipeline names each single-turn skill call via a ``stage_task`` field in
# the user payload. Recovering that label lets us bucket the generator/critic
# streams by sub-skill, which is what downstream SFT builders key on.
# ---------------------------------------------------------------------------

GENERATOR_STAGE_TASKS: Final[frozenset[str]] = frozenset({
    "segment_article",
    "plan_shots",
    "repair_plan",
    "quality_repair",
    "template_retrieve",
    "template_fill",
    "music_style",
})

CRITIC_STAGE_TASKS: Final[frozenset[str]] = frozenset({
    "self_reflect_plan",
    "quality_judge",
})

STAGE_TASKS_BY_ROLE: Final[dict[Role, frozenset[str]]] = {
    Role.ORCHESTRATOR: frozenset({"agent_loop"}),
    Role.GENERATOR: GENERATOR_STAGE_TASKS,
    Role.CRITIC: CRITIC_STAGE_TASKS,
}


def normalize_role(value: object) -> Role:
    """Coerce an arbitrary role label into a :class:`Role`.

    Unrecognized values fall back to the orchestrator so recording never drops
    an interaction just because a label changed upstream.
    """
    if isinstance(value, Role):
        return value
    text = str(value or "").strip().lower()
    for role in ALL_ROLES:
        if text == role.value:
            return role
    if text in {"director", "main", "main_flow", "主流程"}:
        return Role.ORCHESTRATOR
    if text in {"content", "content_generator", "内容生成器"}:
        return Role.GENERATOR
    if text in {"judge", "reflect", "reflection", "反思修复"}:
        return Role.CRITIC
    return Role.ORCHESTRATOR
