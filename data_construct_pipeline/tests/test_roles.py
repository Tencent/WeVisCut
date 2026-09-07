"""Role split and tool routing."""

from __future__ import annotations

from longvideo_polar.roles import (
    ALL_ROLES,
    ROLE_LABELS_ZH,
    Role,
    load_tool_schema,
    normalize_role,
)


def test_three_roles_are_labeled():
    assert [role.value for role in ALL_ROLES] == ["orchestrator", "generator", "critic"]
    assert ROLE_LABELS_ZH[Role.ORCHESTRATOR] == "主流程"
    assert ROLE_LABELS_ZH[Role.GENERATOR] == "内容生成器"
    assert ROLE_LABELS_ZH[Role.CRITIC] == "反思修复"


def test_generator_owns_planning_and_segmentation():
    schema = load_tool_schema(strict=False)
    assert schema.role_for_tool("Tool_SegmentArticle") is Role.GENERATOR
    assert schema.role_for_tool("Tool_PlanShots") is Role.GENERATOR
    assert schema.role_for_tool("Tool_BatchDecorateSpotlights") is Role.GENERATOR


def test_critic_owns_reflection_and_judging():
    schema = load_tool_schema(strict=False)
    assert schema.role_for_tool("Tool_SelfReflect") is Role.CRITIC
    assert schema.role_for_tool("Tool_Stage4DimensionJudge") is Role.CRITIC
    assert schema.role_for_tool("Tool_SeverityRank") is Role.CRITIC


def test_repair_is_generator_work_triggered_by_critic():
    schema = load_tool_schema(strict=False)
    # The critic finds problems; the generator writes the fix.
    assert schema.role_for_tool("Tool_SelfReflect_Patch") is Role.GENERATOR
    assert schema.role_for_tool("Tool_Stage4DimensionalRepair") is Role.GENERATOR


def test_external_media_tools_stay_with_orchestrator():
    schema = load_tool_schema(strict=False)
    for tool in ("Tool_BatchVisualAudio", "Tool_MusicSearch", "Tool_StitchVideo"):
        assert schema.is_external(tool)
        assert schema.role_for_tool(tool) is Role.ORCHESTRATOR


def test_unknown_tool_defaults_to_orchestrator():
    schema = load_tool_schema(strict=False)
    assert schema.role_for_tool("Tool_DoesNotExist") is Role.ORCHESTRATOR
    assert not schema.is_external("Tool_DoesNotExist")


def test_action_space_partitions_across_roles():
    schema = load_tool_schema(strict=False)
    partitioned = [tool for role in ALL_ROLES for tool in schema.tools_for_role(role)]
    assert sorted(partitioned) == sorted(schema.action_space)
    assert len(partitioned) == len(set(partitioned))


def test_describe_is_json_serializable_snapshot():
    described = load_tool_schema(strict=False).describe()
    assert described["source"]
    assert "Tool_StitchVideo" in described["action_space"]
    assert "Tool_SegmentArticle" in described["generator_tools"]
    assert "Tool_SelfReflect" in described["critic_tools"]


def test_normalize_role_accepts_aliases_and_falls_back():
    assert normalize_role("generator") is Role.GENERATOR
    assert normalize_role("内容生成器") is Role.GENERATOR
    assert normalize_role("反思修复") is Role.CRITIC
    assert normalize_role(Role.CRITIC) is Role.CRITIC
    assert normalize_role("nonsense") is Role.ORCHESTRATOR
    assert normalize_role(None) is Role.ORCHESTRATOR
