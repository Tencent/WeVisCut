"""Single source of truth for AICut v5 tool definitions.

This module defines the tool cards, preconditions, parameter hints, and
role-to-tool mappings used by BOTH the inference agent loop AND the training
data build. Any mismatch between training and inference tool schemas causes
the model to hallucinate incorrect tool calls or fail to route properly.

If you change tools here, you MUST also re-run the training data build to stay
aligned. Use ``verify_consistency()`` to assert parity with the training script.
"""

from __future__ import annotations

# ─── Tool Cards (short description shown to the model) ───

TOOL_CARDS = {
    "Tool_SegmentArticle": "章节划分工具：根据口播稿和轻量结构约束生成 segments 与 music_style。",
    "Tool_PlanShots": "镜头规划工具：根据 segment_id、chunk_index、num_shots_target 和 planning_policy 生成 shots。",
    "Tool_SelfReflect": "规划反思工具：检查分段和镜头覆盖、节奏、重复、spotlight 比例等问题。",
    "Tool_SelfReflect_Patch": "规划修复工具：根据 issue_ids/target_shot_ids 对章节或镜头做最小 patch。",
    "Tool_BatchVisualAudio": "外部工具：为 shot_ids 批量生成 TTS 并检索视觉素材。",
    "Tool_BatchDecorateSpotlights": "装饰工具：为 spotlight shot_ids 选择模板并填写 slots。",
    "Tool_BatchTimelineCompose": "外部工具：用镜头、TTS、素材、装饰合成 timeline/render clips。",
    "Tool_MusicSearch": "外部工具：根据 music_style 检索 BGM。",
    "Tool_SeverityRank": "质检抽样工具：根据 timeline/visual_audio 选择优先质检的镜头。",
    "Tool_Stage4DimensionJudge": "质检工具：按多维度检查渲染、音频、视觉、装饰和节奏。",
    "Tool_Stage4DimensionalRepair": "渲染修复工具：按 failed_dimensions 对失败镜头做最小修复。",
    "Tool_StitchVideo": "外部工具：最终拼接完整视频。",
}

# ─── Tool Preconditions (shown to model + used by state machine) ───

TOOL_PRECONDITIONS = {
    "Tool_SegmentArticle": "需要 input_script；通常是第一步。",
    "Tool_PlanShots": "需要 segments 和 music_style。",
    "Tool_SelfReflect": "需要已生成的 shot_plan。",
    "Tool_SelfReflect_Patch": "需要 SelfReflect issues。",
    "Tool_BatchVisualAudio": "需要可执行 shots。",
    "Tool_BatchDecorateSpotlights": "需要 spotlight shots；最好已有 visual/audio observation。",
    "Tool_BatchTimelineCompose": "需要 TTS/素材；spotlight 镜头需要 decoration。",
    "Tool_MusicSearch": "需要 music_style；必须在 StitchVideo 前完成。",
    "Tool_SeverityRank": "需要 timeline 或 visual/audio observation。",
    "Tool_Stage4DimensionJudge": "需要 timeline/render observation。",
    "Tool_Stage4DimensionalRepair": "需要 failed_dimensions 或 failed_shot_ids。",
    "Tool_StitchVideo": "需要 shots、TTS/素材、BGM、timeline、质检和必要修复均完成。",
}

# ─── Light Tool Schemas (INPUT parameters, must match training exactly) ───
# These are the orchestrator's light control params that the model was trained to
# generate in tool_call arguments. They must be identical to LIGHT_TOOL_SCHEMAS
# in build_orchestrator_sft_v4.py.

LIGHT_TOOL_SCHEMAS: dict[str, dict[str, dict[str, str]]] = {
    "Tool_SegmentArticle": {
        "num_segments_target": {"type": "integer"},
        "segment_granularity": {"type": "string"},
        "preserve_narration_order": {"type": "boolean"},
        "cover_full_script": {"type": "boolean"},
        "need_music_style": {"type": "boolean"},
    },
    "Tool_PlanShots": {
        "mode": {"type": "string"},
        "segment_id": {"type": "string"},
        "chunk_index": {"type": "integer"},
        "num_chunks_for_segment": {"type": "integer"},
        "num_shots_target": {"type": "integer"},
        "planning_policy": {"type": "object"},
    },
    "Tool_SelfReflect": {
        "scope": {"type": "string"},
        "num_segments": {"type": "integer"},
        "num_shots": {"type": "integer"},
        "focus_dimensions": {"type": "array"},
        "round": {"type": "integer"},
        "mode": {"type": "string"},
        "augmentation_strategy": {"type": "string"},
    },
    "Tool_SelfReflect_Patch": {
        "issue_ids": {"type": "array"},
        "target_shot_ids": {"type": "array"},
        "minimal_repair": {"type": "boolean"},
        "requires_downstream_propagation_hint": {"type": "boolean"},
        "round": {"type": "integer"},
        "augmentation_strategy": {"type": "string"},
    },
    "Tool_BatchVisualAudio": {
        "shot_ids": {"type": "array"},
        "mode": {"type": "string"},
    },
    "Tool_BatchDecorateSpotlights": {
        "shot_ids": {"type": "array"},
        "template_policy": {"type": "object"},
        "mode": {"type": "string"},
        "style": {"type": "string"},
    },
    "Tool_BatchTimelineCompose": {
        "shot_ids": {"type": "array"},
        "mode": {"type": "string"},
        "require_tts_visual_audio": {"type": "boolean"},
        "require_spotlight_decoration": {"type": "boolean"},
    },
    "Tool_MusicSearch": {
        "query": {"type": "string"},
        "music_style": {"type": "string"},
        "duration_sec": {"type": "number"},
        "target_duration_sec": {"type": "number"},
        "must_finish_before_stitch": {"type": "boolean"},
    },
    "Tool_SeverityRank": {
        "shot_count": {"type": "integer"},
        "top_ratio": {"type": "number"},
        "min_shots": {"type": "integer"},
    },
    "Tool_Stage4DimensionJudge": {
        "shot_ids": {"type": "array"},
        "dimensions": {"type": "array"},
        "threshold": {"type": "number"},
        "focus_decoration_slots": {"type": "boolean"},
        "judge_round": {"type": "integer"},
    },
    "Tool_Stage4DimensionalRepair": {
        "failed_shot_ids": {"type": "array"},
        "target_dimensions": {"type": "array"},
        "minimal_repair": {"type": "boolean"},
        "rejudge_after_repair": {"type": "boolean"},
        "repair_round": {"type": "integer"},
    },
    "Tool_StitchVideo": {
        "shot_count": {"type": "integer"},
        "approved_count": {"type": "integer"},
        "final_gate_required": {"type": "boolean"},
        "bgm_volume": {"type": "number"},
        "duck_under_voice": {"type": "boolean"},
    },
}

REQUIRED_TOOL_ARGS: dict[str, list[str]] = {
    "Tool_SegmentArticle": [
        "num_segments_target", "segment_granularity",
        "preserve_narration_order", "cover_full_script", "need_music_style",
    ],
    "Tool_PlanShots": [
        "mode", "segment_id", "chunk_index",
        "num_chunks_for_segment", "num_shots_target", "planning_policy",
    ],
    "Tool_SelfReflect": ["scope", "num_segments", "num_shots", "focus_dimensions"],
    "Tool_SelfReflect_Patch": [
        "issue_ids", "target_shot_ids", "minimal_repair",
        "requires_downstream_propagation_hint",
    ],
    "Tool_BatchVisualAudio": ["shot_ids"],
    "Tool_BatchDecorateSpotlights": ["shot_ids", "template_policy"],
    "Tool_BatchTimelineCompose": ["shot_ids", "require_tts_visual_audio", "require_spotlight_decoration"],
    "Tool_MusicSearch": ["query", "music_style", "must_finish_before_stitch"],
    "Tool_SeverityRank": ["shot_count", "top_ratio", "min_shots"],
    "Tool_Stage4DimensionJudge": ["shot_ids", "dimensions", "threshold", "focus_decoration_slots"],
    "Tool_Stage4DimensionalRepair": [
        "failed_shot_ids", "target_dimensions", "minimal_repair", "rejudge_after_repair",
    ],
    "Tool_StitchVideo": ["shot_count", "approved_count", "final_gate_required"],
}

# ─── Tool Parameter Hints (OUTPUT descriptions, kept for backward compat) ───

TOOL_PARAMETER_HINTS: dict[str, dict[str, dict[str, str]]] = {
    "Tool_SegmentArticle": {
        "intent": {"type": "string", "description": "视频整体意图摘要。"},
        "music_style": {"type": "string", "description": "适合整条长视频的 BGM 风格。"},
        "segments": {"type": "array", "description": "文章分段结果，每项包含 segment_id、summary、script 等字段。"},
        "num_segments": {"type": "integer", "description": "segments 数量。"},
    },
    "Tool_PlanShots": {
        "segment_id": {"type": "string", "description": "本次规划所属 segment。"},
        "shots": {"type": "array", "description": "镜头列表。"},
    },
    "Tool_SelfReflect": {
        "passed": {"type": "boolean", "description": "是否通过。"},
        "issues": {"type": "array", "description": "问题列表。"},
    },
    "Tool_SelfReflect_Patch": {
        "patched_shot_plan": {"type": "object", "description": "局部修改。"},
    },
    "Tool_BatchVisualAudio": {
        "shots": {"type": "array", "description": "镜头列表。"},
    },
    "Tool_BatchDecorateSpotlights": {
        "decorations": {"type": "object", "description": "装饰结果。"},
    },
    "Tool_BatchTimelineCompose": {
        "shots": {"type": "array", "description": "镜头列表。"},
    },
    "Tool_MusicSearch": {
        "music_style": {"type": "string", "description": "BGM 风格。"},
    },
    "Tool_SeverityRank": {
        "top_shots": {"type": "array", "description": "优先质检的镜头。"},
    },
    "Tool_Stage4DimensionJudge": {
        "shots": {"type": "object", "description": "质检结果。"},
    },
    "Tool_Stage4DimensionalRepair": {
        "shots": {"type": "object", "description": "修复结果。"},
    },
    "Tool_StitchVideo": {
        "project_id": {"type": "string", "description": "项目 ID。"},
    },
}

# ─── Allowed tool args (light-args whitelist, must match training) ───

ALLOWED_TOOL_ARGS = {
    "Tool_SegmentArticle": {
        "num_segments_target", "segment_granularity",
        "preserve_narration_order", "cover_full_script", "need_music_style",
    },
    "Tool_PlanShots": {
        "mode", "segment_id", "chunk_index", "num_chunks_for_segment",
        "num_shots_target", "planning_policy",
    },
    "Tool_SelfReflect": {
        "scope", "num_segments", "num_shots", "focus_dimensions",
        "round", "mode", "augmentation_strategy",
    },
    "Tool_SelfReflect_Patch": {
        "issue_ids", "target_shot_ids", "minimal_repair",
        "requires_downstream_propagation_hint", "round", "augmentation_strategy",
    },
    "Tool_BatchVisualAudio": {"shot_ids", "mode"},
    "Tool_BatchDecorateSpotlights": {"shot_ids", "template_policy", "mode", "style"},
    "Tool_BatchTimelineCompose": {
        "shot_ids", "mode", "require_tts_visual_audio", "require_spotlight_decoration",
    },
    "Tool_MusicSearch": {
        "query", "music_style", "duration_sec", "target_duration_sec",
        "must_finish_before_stitch",
    },
    "Tool_SeverityRank": {"shot_count", "top_ratio", "min_shots"},
    "Tool_Stage4DimensionJudge": {
        "shot_ids", "dimensions", "threshold", "focus_decoration_slots", "judge_round",
    },
    "Tool_Stage4DimensionalRepair": {
        "failed_shot_ids", "target_dimensions", "minimal_repair",
        "rejudge_after_repair", "repair_round",
    },
    "Tool_StitchVideo": {
        "shot_count", "approved_count", "final_gate_required", "bgm_volume", "duck_under_voice",
    },
}

# ─── Heavy keys (stripped from tool_call args before sending to model) ───

HEAVY_KEYS = {
    "segments", "shots", "decorations", "patched_shot_plan",
    "results", "visual", "audio", "timeline",
}

# ─── Role-to-tool mapping ───

GENERATOR_TOOLS = {
    "Tool_SegmentArticle",
    "Tool_PlanShots",
    "Tool_SelfReflect_Patch",
    "Tool_BatchDecorateSpotlights",
    "Tool_Stage4DimensionalRepair",
}

CRITIC_TOOLS = {
    "Tool_SelfReflect",
    "Tool_Stage4DimensionJudge",
    "Tool_SeverityRank",
}

# Tools that are dispatched externally (no LLM call, run pipeline code directly)
EXTERNAL_TOOLS = {
    "Tool_BatchVisualAudio",
    "Tool_BatchTimelineCompose",
    "Tool_MusicSearch",
    "Tool_StitchVideo",
}

ALL_TOOLS = sorted(TOOL_CARDS.keys())

# ─── Phase labels ───

PHASE_LABELS = {
    "plan": "理解稿件与镜头规划",
    "reflect_repair_plan": "自检与局部修复",
    "asset_and_compose": "素材/TTS/装饰/配乐/合成",
    "quality_repair": "质量检查与修复",
    "final_stitch": "最终拼接",
}

# Tool aliases for robustness
TOOL_ALIASES = {
    "Tool_DecorateShot": "Tool_BatchDecorateSpotlights",
    "Tool_Decorate": "Tool_BatchDecorateSpotlights",
    "Tool_DecorateSpotlight": "Tool_BatchDecorateSpotlights",
    "Tool_DecorateSpotlights": "Tool_BatchDecorateSpotlights",
}


def verify_consistency(training_tools_json: list[dict] | None = None) -> dict[str, bool]:
    """Compare this schema against the training build's tool definitions.

    If `training_tools_json` is None, attempts to load from the training script.
    Returns a dict of check_name -> passed.
    """
    import sys
    from pathlib import Path

    results: dict[str, bool] = {}

    if training_tools_json is None:
        train_src = Path(__file__).resolve().parents[2] / "preprocess_ori_data" / "src"
        if str(train_src) not in sys.path:
            sys.path.insert(0, str(train_src))
        try:
            import build_orchestrator_sft_v4 as o4
            results["import_training_module"] = True
        except ImportError:
            results["import_training_module"] = False
            return results

        results["heavy_keys_match"] = (HEAVY_KEYS == o4.HEAVY_KEYS)
        results["allowed_tool_args_match"] = (ALLOWED_TOOL_ARGS == {
            k: set(v) for k, v in o4.ALLOWED_TOOL_ARGS.items()
        })
        results["tool_names_match"] = (set(TOOL_CARDS.keys()) == set(o4.TOOL_CARDS.keys()))
    else:
        infer_names = set(TOOL_CARDS.keys())
        train_names = {t.get("function", {}).get("name", "") for t in training_tools_json}
        results["tool_names_match"] = (infer_names == train_names)

    return results
