"""LangGraph state machine definition for AICut inference pipeline.

Carries all data through the 5-stage video generation workflow:
  Stage 1&2: Director Planning (segment → plan → self-reflect)
  Stage 3:   Shot Execution (visual + audio → spotlight decoration with check & repair)
  Stage 4:   Shot Composition (timeline → VLM judge with auto-repair)
  Stage 5:   Final Assembly (stitch all shots into preview)
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from langgraph.graph import MessagesState


# ---------------------------------------------------------------------------
# Trajectory entry (SFT-format)
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryEntry:
    role: Literal["user", "thought", "call", "respond", "human_feedback", "stage_marker"]
    content: str = ""
    tool: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_output: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool:
            d["tool"] = self.tool
        if self.tool_input:
            d["tool_input"] = self.tool_input
        if self.tool_output:
            d["tool_output"] = self.tool_output
        if self.metadata:
            d["metadata"] = self.metadata
        return d


# ---------------------------------------------------------------------------
# Shot-level asset bundle
# ---------------------------------------------------------------------------
@dataclass
class ShotAssets:
    shot_id: str = ""
    clip_path: str = ""
    clip_meta: dict[str, Any] = field(default_factory=dict)
    audio_path: str = ""
    audio_meta: dict[str, Any] = field(default_factory=dict)
    decoration: dict[str, Any] = field(default_factory=dict)
    timeline: dict[str, Any] = field(default_factory=dict)
    rendered_path: str = ""
    judge_result: dict[str, Any] = field(default_factory=dict)
    approved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "clip_path": self.clip_path,
            "clip_meta": self.clip_meta,
            "audio_path": self.audio_path,
            "audio_meta": self.audio_meta,
            "decoration": self.decoration,
            "timeline": self.timeline,
            "rendered_path": self.rendered_path,
            "judge_result": self.judge_result,
            "approved": self.approved,
        }


# ---------------------------------------------------------------------------
# HITL decision context
# ---------------------------------------------------------------------------
@dataclass
class HITLContext:
    node_name: str = ""
    question: str = ""
    current_data: dict[str, Any] = field(default_factory=dict)
    options: list[dict[str, Any]] = field(default_factory=list)
    selected_option: int = -1
    custom_input: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_name": self.node_name,
            "question": self.question,
            "current_data": self.current_data,
            "options": self.options,
            "selected_option": self.selected_option,
            "custom_input": self.custom_input,
        }


def _merge_trajectory(
    existing: list[dict[str, Any]], new: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Append-only merge for trajectory logs across graph steps."""
    return existing + new


# ---------------------------------------------------------------------------
# Main pipeline state — used as LangGraph TypedDict-style state
# ---------------------------------------------------------------------------
class PipelineState(MessagesState):
    # --- Execution mode ---
    mode: str                                # "inference" | "data_pipeline"
    project_id: str
    output_dir: str
    config: dict[str, Any]

    # --- Input ---
    input_text: str                          # article text (inference mode)
    input_video_path: str                    # source video (data_pipeline mode)

    # --- Stage 1: Article parsing (data_pipeline only) ---
    asr_transcript: str
    asr_segments: list[dict[str, Any]]       # corrected ASR sentence windows [{text, start, end}, ...]
    video_shots: list[dict[str, Any]]        # sentence-level shot windows; may include covered TransNetV2 ids
    shot_captions: list[dict[str, Any]]      # legacy VLM captions; empty in ASR-sentence mode
    video_summary: str
    narration_script: str                    # oral script from stage1 summary

    # --- Stage 2: Director planning ---
    segments: list[dict[str, Any]]           # article segments
    music_style: str                         # BGM style description from article segmentation
    shot_plan: list[dict[str, Any]]          # shots: chapter_title, vocal_script, visual_query, is_spotlight
    plan_issues: list[str]
    plan_approved: bool
    reflect_round: int

    # --- Stage 3: Shot execution ---
    current_shot_idx: int
    modal_probe_result: dict[str, Any]        # preflight result for visual/TTS/music dependencies
    shot_assets: dict[str, Any]              # shot_id -> ShotAssets.to_dict()

    # --- Stage 3c: Severity ranking & dual-track quality ---
    severity_scores: dict[str, Any]              # shot_id -> {score, rank, in_top_ratio}
    severity_ranked_shots: list[str]             # shot_ids sorted by severity desc

    # --- Stage 4: Composition ---
    compose_shot_idx: int
    judge_round: int

    # --- Stage 3b: BGM search ---
    bgm_tracks: list[dict[str, Any]]         # selected BGM tracks with paths + durations
    bgm_search_result: dict[str, Any]        # full music search result

    # --- Stage 5: Final assembly (preview) ---
    final_video_path: str

    # --- Trajectory logging (append-only) ---
    trajectory: Annotated[list[dict[str, Any]], _merge_trajectory]

    # --- HITL ---
    hitl_context: dict[str, Any]
    hitl_pending: bool

    # --- Error tracking ---
    errors: list[str]
    current_stage: str
