"""Modified LangGraph pipeline that uses the local LoRA model as orchestrator.

Instead of the original LLM API-based routing, this graph:
  1. Builds SFT-format state JSON from PipelineState (same as training data)
  2. Calls model_loader.predict() to get a tool call decision
  3. Dispatches the real pipeline tool
  4. Broadcasts progress via a callback
  5. Updates PipelineState and loops through stages
"""

from __future__ import annotations

import asyncio
import bisect
import copy
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import requests
import yaml

logger = logging.getLogger("aicut_infer")
_GRAPH_NODE_PATCH_LOCK = threading.Lock()

TASK_GOAL = "用户输入一段口播稿，主控 Agent 编排工具链生成可验收长视频。"

# A segment whose budget exceeds this is planned over several chunks, matching
# the training data where long chapters were split across 2-4 generator calls.
_PLAN_SHOTS_MAX_PER_CHUNK = 12

# Stage ordering for inference pipeline
STAGE_SEQUENCE = [
    "stage2_planning",
    "stage2_self_reflect",
    "stage3_visual_audio",
    "stage3_decoration",
    "stage3_post_decoration_reflect",
    "stage3_bgm_search",
    "stage4_severity_rank",
    "stage4_compose",
    "stage4_dimension_judge",
    "stage5_final_render",
    "stage5_stitch",
]

# Tool -> handler function name mapping (for dispatching)
TOOL_DISPATCH = {
    "Tool_SegmentArticle": "dispatch_segment_article",
    "Tool_PlanShots": "dispatch_plan_shots",
    "Tool_SelfReflect": "dispatch_self_reflect",
    "Tool_SelfReflect_RepairShot": "dispatch_self_reflect_repair",
    "Tool_SelfReflect_Patch": "dispatch_self_reflect_patch",
    "Tool_BatchVisualAudio": "dispatch_batch_visual_audio",
    "Tool_BatchDecorateSpotlights": "dispatch_batch_decorate",
    "Tool_BatchTimelineCompose": "dispatch_batch_timeline_compose",
    "Tool_MusicSearch": "dispatch_music_search",
    "Tool_SeverityRank": "dispatch_severity_rank",
    "Tool_Stage4DimensionJudge": "dispatch_dimension_judge",
    "Tool_TextQuickJudge": "dispatch_text_quick_judge",
    "Tool_Stage4DimensionalRepair": "dispatch_dimension_repair",
    "Tool_FinalRender": "dispatch_final_render",
    "Tool_StitchVideo": "dispatch_stitch_video",
}

LONG_HORIZON_ACTION_SPACE = [
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
]

LONG_HORIZON_SEMANTIC_TOOLS = {
    "Tool_SegmentArticle",
    "Tool_PlanShots",
    "Tool_SelfReflect",
    "Tool_SelfReflect_Patch",
    "Tool_BatchDecorateSpotlights",
    "Tool_SeverityRank",
    "Tool_Stage4DimensionJudge",
    "Tool_Stage4DimensionalRepair",
}

LONG_HORIZON_TOOL_SUBTYPES = {
    "Tool_SegmentArticle": "segment_article",
    "Tool_PlanShots": "plan_shots_chunk",
    "Tool_SelfReflect": "self_reflect",
    "Tool_SelfReflect_Patch": "self_reflect_patch",
    "Tool_BatchVisualAudio": "batch_visual_audio",
    "Tool_BatchDecorateSpotlights": "batch_decorate_spotlights",
    "Tool_BatchTimelineCompose": "batch_timeline_compose",
    "Tool_MusicSearch": "music_search",
    "Tool_SeverityRank": "severity_rank",
    "Tool_Stage4DimensionJudge": "dimension_judge",
    "Tool_Stage4DimensionalRepair": "dimension_repair",
    "Tool_StitchVideo": "stitch_video",
}

LONG_HORIZON_TOOL_NAME_ALIASES = {
    "Tool_DecorateShot": "Tool_BatchDecorateSpotlights",
    "Tool_Decorate": "Tool_BatchDecorateSpotlights",
    "Tool_DecorateSpotlight": "Tool_BatchDecorateSpotlights",
    "Tool_DecorateSpotlights": "Tool_BatchDecorateSpotlights",
    "Tool_SelfReflect_RepairShot": "Tool_SelfReflect_Patch",
    "Tool_SelfReflectPatch": "Tool_SelfReflect_Patch",
    "Tool_Repair": "Tool_Stage4DimensionalRepair",
    "Tool_DimensionRepair": "Tool_Stage4DimensionalRepair",
    "Tool_DimensionalRepair": "Tool_Stage4DimensionalRepair",
    "Tool_DimensionJudge": "Tool_Stage4DimensionJudge",
    "Tool_Judge": "Tool_Stage4DimensionJudge",
    "Tool_TextQuickJudge": "Tool_Stage4DimensionJudge",
    "Tool_FinalRender": "Tool_BatchTimelineCompose",
    "Tool_TimelineCompose": "Tool_BatchTimelineCompose",
    "Tool_Timeline": "Tool_BatchTimelineCompose",
    "Tool_VisualAudio": "Tool_BatchVisualAudio",
    "Tool_TTS": "Tool_BatchVisualAudio",
    "Tool_Stitch": "Tool_StitchVideo",
    "Tool_BGM": "Tool_MusicSearch",
    "Tool_Segment": "Tool_SegmentArticle",
    "Tool_PlanShot": "Tool_PlanShots",
}

from server.tool_schema import (
    TOOL_CARDS as LONG_HORIZON_TOOL_CARDS,
    TOOL_PRECONDITIONS as LONG_HORIZON_TOOL_PRECONDITIONS,
    TOOL_PARAMETER_HINTS as LONG_HORIZON_TOOL_PARAMETER_HINTS,
    ALLOWED_TOOL_ARGS as _ALLOWED_TOOL_ARGS,
    HEAVY_KEYS as _HEAVY_KEYS,
    PHASE_LABELS as LONG_HORIZON_PHASE_LABELS,
    GENERATOR_TOOLS,
    CRITIC_TOOLS,
)


_TOOL_PRECONDITIONS = dict(LONG_HORIZON_TOOL_PRECONDITIONS)


def _trunc(text: Any, n: int) -> str:
    s = str(text or "")
    return s if len(s) <= n else s[:n] + "..."


def _round_val(v: Any) -> Any:
    try:
        return round(float(v), 3)
    except (TypeError, ValueError):
        return v


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
                elif item.get("type"):
                    parts.append(f"[{item.get('type')}]")
            elif item is not None:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _unwrap_agent_json(parsed: Any) -> Any:
    if (
        isinstance(parsed, dict)
        and isinstance(parsed.get("tool_call"), dict)
        and isinstance(parsed["tool_call"].get("arguments"), dict)
    ):
        return parsed["tool_call"]["arguments"]
    if (
        isinstance(parsed, dict)
        and isinstance(parsed.get("action_input"), dict)
        and (parsed.get("trace_type") == "decision" or parsed.get("action"))
    ):
        return parsed["action_input"]
    return parsed


class LocalModelLLM:
    """Small LLMClient-compatible adapter backed by the already-loaded model.

    In multi-model mode, routes LLM calls to the appropriate skill model
    (generator or critic) based on the active_role hint.
    """

    def __init__(
        self,
        model_loader,
        config: dict[str, Any],
        *,
        service: str | None = None,
        is_cancelled: Callable[[], bool] | None = None,
        active_role: str = "orchestrator",
    ) -> None:
        llm_cfg = dict(config.get("llm", {}))
        if service:
            service_cfg = (config.get("services", {}) or {}).get(service, {})
            if isinstance(service_cfg, dict) and service_cfg:
                llm_cfg = dict(service_cfg)

        self.model_loader = model_loader
        self.service_name = service or "local"
        self.base_url = "local://model_loader"
        self.api_key = "local"
        self.model = llm_cfg.get("model", "local-qwen3.5-9b-lora")
        self.temperature = llm_cfg.get("temperature", 0.2)
        self.max_tokens = llm_cfg.get("max_tokens", 2048)
        self.timeout = llm_cfg.get("timeout", 300)
        self.endpoint = "local"
        self.offline = False
        self.max_prompt_chars = int(llm_cfg.get("max_prompt_chars", 262144) or 262144)
        self.max_frames_per_call = int(
            llm_cfg.get("max_frames_per_call", llm_cfg.get("max_frames", 8)) or 8
        )
        self._is_cancelled = is_cancelled or (lambda: False)
        self._active_role = active_role
        self._is_multi = hasattr(model_loader, "predict_orchestrator")

    def _check_cancelled(self) -> None:
        if self._is_cancelled():
            raise asyncio.CancelledError("Pipeline cancelled by user")

    def chat_messages(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self._check_cancelled()
        system_parts: list[str] = []
        user_parts: list[str] = []
        for msg in messages:
            role = str(msg.get("role", "user"))
            text = _content_to_text(msg.get("content", ""))
            if not text:
                continue
            if role == "system":
                system_parts.append(text)
            else:
                prefix = "" if role == "user" else f"{role}: "
                user_parts.append(prefix + text)

        effective_temp = temperature if temperature is not None else self.temperature
        effective_max = max_tokens or self.max_tokens

        if self._is_multi:
            sys_text = "\n\n".join(system_parts) if system_parts else None
            if self._active_role == "generator":
                prediction = self.model_loader.predict_generator(
                    "\n\n".join(user_parts),
                    system_prompt=sys_text,
                    max_tokens=effective_max,
                    temperature=effective_temp,
                )
            elif self._active_role == "critic":
                prediction = self.model_loader.predict_critic(
                    "\n\n".join(user_parts),
                    system_prompt=sys_text,
                    max_tokens=effective_max,
                    temperature=effective_temp,
                )
            else:
                prediction = self.model_loader.predict_orchestrator(
                    "\n\n".join(system_parts) or self.model_loader.orchestrator_prompt,
                    "\n\n".join(user_parts),
                    max_tokens=effective_max,
                    temperature=effective_temp,
                )
        else:
            prediction = self.model_loader.predict(
                "\n\n".join(system_parts),
                "\n\n".join(user_parts),
                max_tokens=effective_max,
                temperature=effective_temp,
            )
        self._check_cancelled()
        return prediction.cleaned_output

    def chat(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return self.chat_messages(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        from pipeline.llm import _extract_json

        parsed = _extract_json(self.chat(system, user, temperature=temperature, max_tokens=max_tokens))
        parsed = _unwrap_agent_json(parsed)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"items": parsed}
        if isinstance(parsed, tuple):
            for item in parsed:
                item = _unwrap_agent_json(item)
                if isinstance(item, dict):
                    return item
        return {}

    def chat_json_multimodal(
        self,
        system: str,
        user_parts: list[dict[str, Any]],
        temperature: float | None = None,
    ) -> dict[str, Any]:
        return self.chat_json(system, _content_to_text(user_parts), temperature=temperature)

    def generate_options(
        self,
        system: str,
        user: str,
        num_options: int = 4,
    ) -> list[dict[str, Any]]:
        from pipeline.llm import _extract_json

        prompt = (
            f"{user}\n\n"
            f"请生成 {num_options} 个不同的解决方案选项，每个方案用 JSON 对象表示，"
            f"包含 'option_id'(int), 'label'(str简短标题), 'description'(str详细说明), "
            f"'changes'(dict具体修改内容)。返回一个 JSON 数组。"
        )
        parsed = _extract_json(self.chat(system, prompt, temperature=0.7))
        if isinstance(parsed, list):
            return parsed[:num_options]
        if isinstance(parsed, dict) and isinstance(parsed.get("options"), list):
            return parsed["options"][:num_options]
        return [parsed] if isinstance(parsed, dict) and parsed else []

    def close(self) -> None:
        return None


class InferenceOrchestrator:
    """Runs the AICutPipeline with the local fine-tuned model as the decision-maker.

    Supports two deployment modes:
      - single_model: one ModelLoader handles all predictions
      - multi_model: MultiModelLoader routes orchestrator/generator/critic
    """

    def __init__(
        self,
        model_loader,
        pipeline_config: dict[str, Any],
        stage_config: dict[str, Any],
        system_prompt: str,
        broadcast_fn: Callable | None = None,
        infer_config: dict[str, Any] | None = None,
    ):
        self.model = model_loader
        self.pipeline_config = pipeline_config
        self.stage_config = stage_config
        self.system_prompt = system_prompt
        self.broadcast = broadcast_fn or (lambda *a, **kw: None)
        self.infer_config = infer_config or {}
        self.agent_mode = str(self.infer_config.get("agent_mode") or "tool_call")
        self.agent_loop_cfg = self.infer_config.get("agent", {}) if isinstance(self.infer_config.get("agent"), dict) else {}
        self.max_agent_steps = int(self.agent_loop_cfg.get("max_steps", 36) or 36)
        self.max_same_tool = int(self.agent_loop_cfg.get("max_same_tool_repeats", 4) or 4)

        self.allowed_map = stage_config.get("allowed_tools", {})
        self.stage_order = stage_config.get("stage_order", {})

        self._is_multi_model = hasattr(model_loader, "predict_orchestrator")

        # Pause / cancel control
        self._paused = asyncio.Event()
        self._paused.set()  # starts unpaused
        self._cancelled = False

        # Tool switches — independent of the annotation pipeline config
        switches = self.infer_config.get("tool_switches", {})
        self.sw_tts = bool(switches.get("tts", True))
        self.sw_visual = bool(switches.get("visual_retrieval", True))
        self.sw_bgm = bool(switches.get("bgm_search", True))
        self.sw_compose = bool(switches.get("timeline_compose", True))
        self.sw_hyperframes = bool(switches.get("hyperframes_render", True))
        self.sw_stitch = bool(switches.get("final_stitch", True))
        self.tts_service_url = str(
            self.infer_config.get("tts_service_url", "http://127.0.0.1:7865")
        )

        # Runtime state accumulator (mirrors training data format)
        self.num_segments = 0
        self.segments: list[dict[str, Any]] = []
        self.shots: dict[str, dict[str, Any]] = {}
        self.bgm: dict[str, Any] = {}
        self.last_judge: dict[str, Any] = {}
        self.last_observation: dict[str, Any] = {}
        self.repair_used: dict[str, int] = {}
        self.agent_history: list[dict[str, Any]] = []
        self.open_issues: list[dict[str, Any]] = []
        self.patch_memory: dict[str, Any] = {}
        self.timeline_done = False
        self.visual_audio_done = False
        self.decorated_done = False
        self.quality_checked = False
        self.repaired_done = False
        self.stitched_done = False
        self._reflect_generation = 0
        self._last_reflect_generation = -1

    def pause(self) -> None:
        self._paused.clear()

    def resume(self) -> None:
        self._paused.set()

    def cancel(self) -> None:
        self._cancelled = True
        self._paused.set()  # unblock if paused so loop can exit

    @property
    def is_paused(self) -> bool:
        return not self._paused.is_set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    async def _wait_if_paused(self, run_id: str) -> None:
        """Block until resumed; raise if cancelled."""
        if self._cancelled:
            raise asyncio.CancelledError("Pipeline cancelled by user")
        if not self._paused.is_set():
            await self.broadcast(run_id, {
                "type": "console_log",
                "text": "⏸ 管线已暂停，等待继续...",
                "level": "warn",
            })
            await self._paused.wait()
            if self._cancelled:
                raise asyncio.CancelledError("Pipeline cancelled by user")
            await self.broadcast(run_id, {
                "type": "console_log",
                "text": "▶ 管线继续执行",
                "level": "system",
            })

    def build_state_json(self, stage: str, input_script: str) -> str:
        """Build the user state JSON in the same format as SFT training data."""
        allowed_tools = list(self.allowed_map.get(stage, []))
        max_shots = 40
        shot_list = list(self.shots.values())[:max_shots]
        spotlight = sum(1 for s in self.shots.values() if s.get("is_spotlight"))

        state_summary: dict[str, Any] = {
            "input_script_summary": _trunc(input_script, 1200),
        }
        if self.num_segments:
            state_summary["num_segments"] = self.num_segments
        if shot_list:
            state_summary["num_shots"] = len(self.shots)
            state_summary["spotlight_count"] = spotlight
            state_summary["shots"] = shot_list
        if self.last_judge:
            state_summary["judge"] = self.last_judge
        if self.bgm:
            state_summary["bgm"] = self.bgm

        budget_left = {}
        for tool, budget in self.stage_config.get("repair_budget", {}).items():
            budget_left[tool] = max(0, int(budget) - self.repair_used.get(tool, 0))

        failure_flags: list[str] = sorted({
            str(dim)
            for s in self.shots.values()
            for dim in (s.get("failed_dimensions") or [])
        })

        if self.agent_mode == "agent_v3":
            user_state = {
                "task": TASK_GOAL,
                "agent_mode": "decision",
                "stage": stage,
                "allowed_tools": allowed_tools,
                "current_action_candidate": self._preferred_tool_for_stage(stage, allowed_tools),
                "input_script": _trunc(input_script, 10000),
                "memory": {
                    "segments": self.segments[:12],
                    "shots": shot_list[:24],
                    "music_style": self.bgm.get("music_style", "") if isinstance(self.bgm, dict) else "",
                    "judge": self.last_judge,
                },
                "previous_observation": self.last_observation,
                "budget_left": budget_left,
                "constraints": {},
                "output_schema": {
                    "trace_type": "decision",
                    "thought": "string",
                    "reflection": "string",
                    "action": "one of action_enum",
                    "action_enum": allowed_tools,
                    "action_input": "object with concrete content",
                    "verification_plan": "string",
                    "expected_observation": "object",
                    "next_action": "string",
                },
            }
            return json.dumps(user_state, ensure_ascii=False)

        user_state = {
            "task": TASK_GOAL,
            "stage": stage,
            "allowed_tools": allowed_tools,
            "input_script": input_script,
            "state_summary": state_summary,
            "previous_observation": self.last_observation,
            "budget_left": budget_left,
            "failure_flags": failure_flags,
        }
        return json.dumps(user_state, ensure_ascii=False)

    def is_long_horizon_agent(self) -> bool:
        return self.agent_mode in {"long_horizon", "agent_long_horizon", "long_horizon_agent"}

    def build_agentic_state_json(self, input_script: str, state: dict[str, Any]) -> str:
        """Build the long-horizon SFT user payload used by the new agent model.

        Must match the format produced by build_orchestrator_sft_v4.py
        make_user_payload() exactly so the model sees identical structure
        at inference as during training.
        """
        flags = self._agent_flags(state)
        segmented = flags.get("segmented", False)
        user_state = {
            "task": "script_to_long_video",
            "goal": TASK_GOAL,
            "input_script": "" if segmented else _trunc(input_script, 12000),
            "script_digest": self._script_digest(input_script, state),
            "current_focus": self._current_focus(state),
            "global_state": self._agent_memory_snapshot(state),
            "recent_history": self.agent_history[-12:],
            "last_observation": self.last_observation,
            "tool_preconditions": _TOOL_PRECONDITIONS,
            "allowed_tool_args": {tool: sorted(args) for tool, args in _ALLOWED_TOOL_ARGS.items()},
            "agent_contract": {
                "one_tool_call_per_turn": True,
                "light_arguments_only": True,
                "forbidden_argument_keys": sorted(_HEAVY_KEYS),
                "whitelist_arguments_only": True,
                "runtime_is_state_authority": True,
                "wait_for_tool_response": True,
            },
        }
        return json.dumps(user_state, ensure_ascii=False)

    def _agent_memory_snapshot(self, state: dict[str, Any]) -> dict[str, Any]:
        """Must match the global_state format from build_long_horizon_agent_sft.py."""
        shot_plan = [s for s in state.get("shot_plan", []) if isinstance(s, dict)]
        shot_assets = state.get("shot_assets", {}) if isinstance(state.get("shot_assets"), dict) else {}
        spotlight = [s for s in shot_plan if s.get("is_spotlight")]
        decorated = [
            sid for sid, assets in shot_assets.items()
            if isinstance(assets, dict) and assets.get("decoration")
        ]
        failed = self._failed_judge_ids(state)
        return {
            "flags": self._agent_flags(state),
            "counts": {
                "segments": len(state.get("segments", []) or []),
                "shots": len(shot_plan),
                "spotlights": len(spotlight),
                "issues": len(self.open_issues),
                "decorated_spotlights": len(decorated),
                "judge_failed": len(failed),
            },
            "music_style": state.get("music_style", ""),
            "segments": self._compact_segments(state.get("segments", [])[:10]),
            "shots": self._compact_shots(shot_plan[:48]),
            "spotlight_shots": self._compact_shots(spotlight[:18]),
            "open_issues": self.open_issues[:12],
            "visual_audio": self._visual_audio_summary(state),
            "decorated_shot_ids": decorated[:30],
            "bgm": {
                "status": self.bgm.get("status", "") if isinstance(self.bgm, dict) else "",
                "track_count": len(state.get("bgm_tracks", []) or []),
            },
            "judge_failed_shot_ids": failed[:30],
        }

    def _all_segments_planned(self, state: dict[str, Any]) -> bool:
        """Check if shot_plan has been produced (at least one shot exists)."""
        shot_plan = state.get("shot_plan") or []
        return bool(shot_plan)

    def _agent_flags(self, state: dict[str, Any]) -> dict[str, Any]:
        ready_count, total_count = self._render_ready_counts(state)
        return {
            "segmented": bool(state.get("segments")),
            "planned": self._all_segments_planned(state),
            "self_checked": not self._needs_reflect() and bool(
                state.get("plan_approved") or state.get("plan_issues") or self.open_issues
            ),
            "patched": bool(self.patch_memory),
            "visual_audio_done": bool(self.visual_audio_done or self._has_visual_audio(state)),
            "decorated": bool(self.decorated_done or self._has_required_decoration(state)),
            "timeline_done": bool(self.timeline_done or (total_count > 0 and ready_count >= total_count)),
            "bgm_done": bool(state.get("bgm_tracks") or self.bgm),
            "severity_ranked": bool(state.get("severity_scores") or state.get("severity_ranked_shots")),
            "quality_checked": bool(self.quality_checked or self.last_judge),
            "repaired": bool(self.repaired_done),
            "stitched": bool(self.stitched_done or state.get("final_video_path")),
        }

    def _script_digest(self, input_script: str, state: dict[str, Any]) -> dict[str, Any]:
        """Matches build_orchestrator_sft_v4.py script_digest()."""
        segments = state.get("segments") or []
        if segments:
            return {
                "mode": "post_segmentation",
                "num_segments": len(segments),
                "segment_outline": [
                    {
                        "segment_id": seg.get("segment_id", ""),
                        "title": _trunc(seg.get("title", ""), 48),
                        "summary": _trunc(seg.get("summary", ""), 120),
                    }
                    for seg in segments[:10]
                    if isinstance(seg, dict)
                ],
            }
        return {
            "mode": "initial_script",
            "script_head": _trunc(input_script, 900),
        }

    def _current_focus(self, state: dict[str, Any]) -> dict[str, Any]:
        """Matches build_orchestrator_sft_v4.py current_focus()."""
        segments = state.get("segments") or []
        shot_plan = state.get("shot_plan") or []
        planned_segments = {
            str(s.get("segment_id"))
            for s in shot_plan
            if isinstance(s, dict) and s.get("segment_id")
        }
        next_segment: dict[str, Any] = {}
        for seg in segments:
            sid = str(seg.get("segment_id") or "")
            if sid and sid not in planned_segments:
                next_segment = {
                    "next_unplanned_segment_id": sid,
                    "title": _trunc(seg.get("title", ""), 48),
                    "summary": _trunc(seg.get("summary", ""), 120),
                }
                break
        return {
            "next_unplanned_segment": next_segment,
            "failed_judge_shot_ids": self._failed_judge_ids(state)[:20],
            "open_issue_count": len(self.open_issues),
        }

    @staticmethod
    def _compact_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            out.append({
                "segment_id": seg.get("segment_id", ""),
                "structure_label": seg.get("structure_label", ""),
                "title": _trunc(seg.get("title", ""), 60),
                "summary": _trunc(seg.get("summary", ""), 180),
                "key_points": (seg.get("key_points") or [])[:4] if isinstance(seg.get("key_points"), list) else [],
                "planner_strategy": _trunc(seg.get("planner_strategy", ""), 120),
            })
        return out

    @staticmethod
    def _compact_shots(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for shot in shots:
            if not isinstance(shot, dict):
                continue
            out.append({
                "shot_id": str(shot.get("shot_id", "")),
                "segment_id": shot.get("segment_id", ""),
                "chapter_title": _trunc(shot.get("chapter_title", ""), 80),
                "vocal_script": _trunc(shot.get("vocal_script", ""), 260),
                "visual_query": _trunc(shot.get("visual_query", ""), 180),
                "intent": _trunc(shot.get("intent", ""), 120),
                "is_spotlight": bool(shot.get("is_spotlight")),
            })
        return out

    @staticmethod
    def _has_visual_audio(state: dict[str, Any]) -> bool:
        assets = state.get("shot_assets", {}) if isinstance(state.get("shot_assets"), dict) else {}
        return any(
            isinstance(v, dict) and (v.get("clip_path") or v.get("audio_path"))
            for v in assets.values()
        )

    @staticmethod
    def _has_timeline(state: dict[str, Any]) -> bool:
        assets = state.get("shot_assets", {}) if isinstance(state.get("shot_assets"), dict) else {}
        return any(
            isinstance(v, dict) and v.get("rendered_path")
            for v in assets.values()
        )

    @staticmethod
    def _render_ready_counts(state: dict[str, Any]) -> tuple[int, int]:
        shots = [s for s in (state.get("shot_plan") or []) if isinstance(s, dict)]
        shot_assets = state.get("shot_assets", {}) if isinstance(state.get("shot_assets"), dict) else {}
        total = len(shots)
        ready = 0
        for shot in shots:
            sid = str(shot.get("shot_id", ""))
            assets = shot_assets.get(sid, {}) if isinstance(shot_assets.get(sid), dict) else {}
            if str(assets.get("rendered_path", "") or ""):
                ready += 1
        return ready, total

    @staticmethod
    def _has_required_decoration(state: dict[str, Any]) -> bool:
        shots = state.get("shot_plan", []) or []
        spotlight_ids = {str(s.get("shot_id", "")) for s in shots if isinstance(s, dict) and s.get("is_spotlight")}
        if not spotlight_ids:
            return True
        assets = state.get("shot_assets", {}) if isinstance(state.get("shot_assets"), dict) else {}
        decorated = {
            str(sid) for sid, val in assets.items()
            if isinstance(val, dict) and val.get("decoration")
        }
        return bool(decorated & spotlight_ids)

    def _visual_audio_summary(self, state: dict[str, Any]) -> dict[str, Any]:
        assets = state.get("shot_assets", {}) if isinstance(state.get("shot_assets"), dict) else {}
        visual = sum(1 for val in assets.values() if isinstance(val, dict) and val.get("clip_path"))
        audio = sum(1 for val in assets.values() if isinstance(val, dict) and val.get("audio_path"))
        return {"visual_count": visual, "audio_count": audio, "status": "success" if visual or audio else ""}

    def _failed_judge_ids(self, state: dict[str, Any]) -> list[str]:
        assets = state.get("shot_assets", {}) if isinstance(state.get("shot_assets"), dict) else {}
        failed = []
        for sid, val in assets.items():
            if isinstance(val, dict) and val.get("approved") is False:
                failed.append(str(sid))
        return failed

    def _preferred_tool_for_stage(self, stage: str, allowed_tools: list[str]) -> str:
        if not allowed_tools:
            return ""
        if stage == "stage2_planning":
            if not self.segments and "Tool_SegmentArticle" in allowed_tools:
                return "Tool_SegmentArticle"
            if self.segments and not self.shots and "Tool_PlanShots" in allowed_tools:
                return "Tool_PlanShots"
        if stage == "stage3_decoration":
            has_spotlight = any(bool(s.get("is_spotlight")) for s in self.shots.values())
            has_decoration = any(bool(s.get("decoration")) for s in self.shots.values())
            if has_spotlight and not has_decoration and "Tool_BatchDecorateSpotlights" in allowed_tools:
                return "Tool_BatchDecorateSpotlights"
        return allowed_tools[0]

    def _predict_for_tool(self, tool_name: str, system_prompt: str, user_content: str, **kwargs):
        """Route prediction to the correct model engine based on tool name."""
        if self._is_multi_model:
            role = self.model.engine_for_tool(tool_name)
            if role == "generator":
                return self.model.predict_generator(user_content, **kwargs)
            elif role == "critic":
                return self.model.predict_critic(user_content, **kwargs)
            else:
                return self.model.predict_orchestrator(system_prompt, user_content, **kwargs)
        return self.model.predict(system_prompt, user_content, **kwargs)

    def _role_for_tool(self, tool_name: str) -> str:
        """Return the model role (orchestrator/generator/critic) for a tool."""
        if self._is_multi_model:
            return self.model.engine_for_tool(tool_name)
        return "orchestrator"

    def _make_local_llm(
        self,
        config: dict[str, Any],
        service: str | None = None,
        active_role: str = "orchestrator",
    ) -> LocalModelLLM:
        return LocalModelLLM(
            self.model,
            config,
            service=service,
            is_cancelled=lambda: self._cancelled,
            active_role=active_role,
        )

    @staticmethod
    def _thinking_event(
        *,
        stage: str,
        prediction: Any,
        step: int | None = None,
    ) -> dict[str, Any]:
        event = {
            "type": "thinking",
            "stage": stage,
            "summary": prediction.thinking_summary,
            "thinking": prediction.thinking_summary,
            "tool_name": prediction.tool_name,
            "latency_ms": prediction.latency_ms,
            "completion_tokens": prediction.completion_tokens,
            "tokens_per_sec": prediction.tokens_per_sec,
            "parse_ok": prediction.parse_ok,
            "agent_trace": prediction.agent_trace,
            "raw_output": prediction.raw_output,
            "cleaned_output": prediction.cleaned_output,
        }
        if step is not None:
            event["step"] = step
        return event

    def _run_graph_node_sync(
        self,
        node_name: str,
        state: dict[str, Any],
        active_role: str = "orchestrator",
    ) -> dict[str, Any]:
        """Run a pipeline.graph node while routing its LLM calls to the local model."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

        import pipeline.graph as graph_mod

        original_get_llm = graph_mod._get_llm

        def _local_get_llm(node_state: dict[str, Any], service: str | None = None):
            return self._make_local_llm(
                node_state.get("config", {}), service=service, active_role=active_role,
            )

        with _GRAPH_NODE_PATCH_LOCK:
            graph_mod._get_llm = _local_get_llm
            try:
                node_fn = getattr(graph_mod, node_name)
                return node_fn(state)
            finally:
                graph_mod._get_llm = original_get_llm

    @staticmethod
    def _merge_state_patch(state: dict[str, Any], patch: dict[str, Any]) -> None:
        for key, value in (patch or {}).items():
            if key in {"trajectory", "current_stage"}:
                continue
            if key == "errors" and isinstance(value, list):
                state.setdefault("errors", []).extend(value)
            else:
                state[key] = value

    def _normalize_shot_plan(
        self,
        segments: list[dict[str, Any]],
        shots: list[dict[str, Any]],
        *,
        target_count: int = 0,
        input_text: str = "",
    ) -> list[dict[str, Any]]:
        """Light normalization: assign IDs, durations, segments, spotlights.

        Does NOT split or redistribute text — lets the model control granularity
        by calling PlanShots multiple times for each chunk/segment.
        """
        if not shots:
            return shots

        planner_cfg = (
            (self.pipeline_config.get("stage2_segment", {}) or {}).get("shot_planner", {})
            if isinstance(self.pipeline_config.get("stage2_segment", {}), dict)
            else {}
        )
        max_ratio = float(planner_cfg.get("max_spotlight_ratio", 0.50) or 0.50)
        max_ratio = max(0.0, min(1.0, max_ratio))

        normalized = [copy.deepcopy(s) for s in shots if isinstance(s, dict)]

        for idx, shot in enumerate(normalized):
            sid = f"shot_{idx + 1:03d}"
            shot["shot_id"] = sid
            if not shot.get("segment_id") and segments:
                seg = segments[min(len(segments) - 1, idx * len(segments) // max(1, len(normalized)))]
                shot["segment_id"] = seg.get("segment_id", f"seg_{idx + 1:03d}")
                shot.setdefault("chapter_title", seg.get("title", ""))
            text = str(shot.get("vocal_script", "") or "")
            if not text:
                text = str(
                    shot.get("vocal_query")
                    or shot.get("tts_text")
                    or shot.get("audio_text")
                    or shot.get("narration")
                    or shot.get("script")
                    or shot.get("voiceover")
                    or shot.get("subtitle")
                    or ""
                )
                if text:
                    shot["vocal_script"] = text
            if not shot.get("duration_sec"):
                shot["duration_sec"] = max(3.0, round(len(text) / 4.5, 2))

        cap = int(len(normalized) * max_ratio + 0.999)
        cap = max(1 if normalized else 0, min(cap, len(normalized)))
        first_of_segment: set[int] = set()
        seen_seg: set[str] = set()
        for idx, shot in enumerate(normalized):
            seg_id = str(shot.get("segment_id", ""))
            if seg_id and seg_id not in seen_seg:
                first_of_segment.add(idx)
                seen_seg.add(seg_id)
        ranked = sorted(
            range(len(normalized)),
            key=lambda i: (
                i in first_of_segment,
                len(str(normalized[i].get("vocal_script", ""))),
            ),
            reverse=True,
        )
        spotlight_ids = set(ranked[:cap])
        for idx, shot in enumerate(normalized):
            shot["is_spotlight"] = idx in spotlight_ids
            shot["is_segment_first"] = idx in first_of_segment
        return normalized

    @staticmethod
    def _segment_narration_text(segment: dict[str, Any]) -> str:
        return str(
            segment.get("narration_script")
            or segment.get("raw_span")
            or segment.get("summary")
            or ""
        ).strip()

    @staticmethod
    def _segment_coverage_units(segment: dict[str, Any]) -> list[dict[str, Any]]:
        text = InferenceOrchestrator._segment_narration_text(segment)
        import re
        sents = [s.strip() for s in re.split(r"(?<=[。！？!?])\s*", text) if s.strip()]
        if not sents and text:
            sents = [text]
        units = []
        pos = 0
        for i, sent in enumerate(sents):
            start = text.find(sent, pos)
            if start == -1:
                start = pos
            end = start + len(sent)
            units.append({
                "unit_id": f"u{i + 1:02d}",
                "char_start": start,
                "char_end": end,
                "text": sent,
            })
            pos = end
        return units

    def _segment_shot_budgets(
        self,
        segments: list[dict[str, Any]],
        requested_total: int = 0,
    ) -> dict[str, int]:
        """每个 segment 分几个镜头。

        ``requested_total``: 策略在 ``Tool_PlanShots`` 里给的 ``num_shots_target``。
        给了就以它为准, 只在超出 ``段数 × max_shots_per_segment`` 时才 clamp。

        为什么加这个参数: 之前这里完全按字数权重自算总数, 调用方再把算出来的值
        **覆盖**掉策略给的 num_shots_target。后果是 (a) 长稿被压成十几个镜头,
        每个镜头要扛几百字口播, 生成端排不下就把稿子尾巴丢了; (b) num_shots_target
        是 RL 策略最主要的决策 token, 线上却完全无效, sim 与真实环境契约不一致。
        """
        planner_cfg = (
            (self.pipeline_config.get("stage2_segment", {}) or {}).get("shot_planner", {})
            if isinstance(self.pipeline_config.get("stage2_segment", {}), dict)
            else {}
        )
        min_per_segment = max(1, int(planner_cfg.get("min_shots_per_segment", 1) or 1))
        max_per_segment = max(min_per_segment, int(planner_cfg.get("max_shots_per_segment", 5) or 5))
        target_chars_per_shot = max(60, int(planner_cfg.get("target_chars_per_shot", 120) or 120))
        min_total = int(planner_cfg.get("min_total_shots", 0) or 0)
        max_total = int(planner_cfg.get("max_total_shots", 0) or 0)

        rows: list[dict[str, Any]] = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            sid = str(segment.get("segment_id") or "")
            if not sid:
                continue
            text = self._segment_narration_text(segment)
            units = self._segment_coverage_units(segment)
            key_points = segment.get("key_points", [])
            key_count = len(key_points) if isinstance(key_points, list) else 0
            structure = str(segment.get("structure_label", "") or "")
            weight = max(1.0, len(text) / target_chars_per_shot)
            weight = max(weight, 1.0 + max(0, len(units) - 1) * 0.25)
            weight = max(weight, key_count * 0.8)
            if any(token in structure for token in ("引子", "总结", "升华", "结尾")):
                weight *= 0.85
            rows.append({"segment_id": sid, "weight": weight, "text_len": len(text)})

        if not rows:
            return {}

        min_chars_per_shot = max(25, int(planner_cfg.get("min_chars_per_shot", 30) or 30))

        budgets = {
            row["segment_id"]: max(
                min_per_segment,
                min(max_per_segment, int(round(row["weight"]))),
            )
            for row in rows
        }
        target_total = sum(budgets.values())
        if requested_total > 0:
            target_total = requested_total
        if min_total > 0:
            target_total = min(max(target_total, min_total), len(rows) * max_per_segment)
        if max_total > 0:
            target_total = max(len(rows) * min_per_segment, min(target_total, max_total))
        # 每段 min/max 能表达的区间是硬边界, 超出的目标无论来自哪里都要 clamp,
        # 否则下面的增/减循环会空转到 progressed=False 才退出, 实际总数与目标不符。
        target_total = max(len(rows) * min_per_segment,
                           min(target_total, len(rows) * max_per_segment))

        ranked = sorted(rows, key=lambda row: row["weight"], reverse=True)
        while sum(budgets.values()) < target_total:
            progressed = False
            for row in ranked:
                sid = row["segment_id"]
                if budgets[sid] >= max_per_segment:
                    continue
                if row["text_len"] / (budgets[sid] + 1) < min_chars_per_shot:
                    continue
                budgets[sid] += 1
                progressed = True
                if sum(budgets.values()) >= target_total:
                    break
            if not progressed:
                break
        while sum(budgets.values()) > target_total:
            progressed = False
            for row in reversed(ranked):
                sid = row["segment_id"]
                if budgets[sid] <= min_per_segment:
                    continue
                budgets[sid] -= 1
                progressed = True
                if sum(budgets.values()) <= target_total:
                    break
            if not progressed:
                break
        return budgets

    @staticmethod
    def _infer_covered_unit_ids(shot: dict[str, Any], coverage_units: list[dict[str, Any]]) -> list[str]:
        existing = shot.get("covered_unit_ids")
        if isinstance(existing, list) and existing:
            return [str(item) for item in existing if str(item)]
        vocal = str(shot.get("vocal_script", "") or "")
        if not vocal:
            return []
        covered = []
        for unit in coverage_units:
            uid = str(unit.get("unit_id", "") or "")
            text = str(unit.get("text", "") or "")
            if not uid or not text:
                continue
            if text in vocal or vocal in text:
                covered.append(uid)
                continue
            overlap = set(text) & set(vocal)
            denom = max(1, min(len(set(text)), len(set(vocal))))
            if len(overlap) / denom >= 0.55:
                covered.append(uid)
        return covered

    @staticmethod
    def _anchor_shots_to_units(
        shots: list[dict[str, Any]],
        global_units: list[dict[str, Any]],
        input_text: str,
    ) -> int:
        """Attach covered_unit_ids by locating each shot's narration in the script.

        Shots come back from the per-segment planner carrying verbatim slices of
        the article but no unit ids, and the fill step needs one anchor per shot
        to order them. `_infer_covered_unit_ids` matches on character overlap,
        which is far too loose to establish order, so match by position and let
        the caller fall back for anything that cannot be located.

        Returns the number of shots that were anchored.
        """
        text = str(input_text or "")
        units = [u for u in global_units if isinstance(u, dict)]
        if not text or not units:
            return 0

        # Whitespace differs between the script and what the model echoes back,
        # so search in a normalized copy and map offsets back.
        norm_chars: list[str] = []
        norm_to_orig: list[int] = []
        for i, ch in enumerate(text):
            if ch.isspace():
                continue
            norm_chars.append(ch)
            norm_to_orig.append(i)
        norm_text = "".join(norm_chars)

        starts = [int(u.get("char_start") or 0) for u in units]
        cursor = 0
        anchored = 0
        for shot in shots:
            if not isinstance(shot, dict):
                continue
            vocal = "".join(str(shot.get("vocal_script", "") or "").split())
            if not vocal:
                continue
            pos = norm_text.find(vocal, cursor)
            if pos < 0:
                # Retry from the top: a repaired shot may sit before the cursor.
                pos = norm_text.find(vocal, 0)
            if pos < 0:
                continue
            end = min(pos + len(vocal), len(norm_to_orig)) - 1
            orig_start = norm_to_orig[pos]
            orig_end = norm_to_orig[end] if end >= pos else orig_start
            covered = [
                str(u.get("unit_id") or "")
                for u, s in zip(units, starts)
                if orig_start <= s <= orig_end
            ]
            if not covered:
                # Shorter than one unit: fall back to the unit containing it.
                idx = max(0, bisect.bisect_right(starts, orig_start) - 1)
                covered = [str(units[idx].get("unit_id") or "")]
            shot["covered_unit_ids"] = [uid for uid in covered if uid]
            cursor = pos + len(vocal)
            anchored += 1
        return anchored

    @staticmethod
    def _coverage_report_for_segment(
        shots: list[dict[str, Any]], coverage_units: list[dict[str, Any]],
    ) -> dict[str, Any]:
        expected = [str(unit.get("unit_id", "") or "") for unit in coverage_units if unit.get("unit_id")]
        counts: dict[str, int] = {uid: 0 for uid in expected}
        for shot in shots:
            covered = InferenceOrchestrator._infer_covered_unit_ids(shot, coverage_units)
            shot["covered_unit_ids"] = covered
            for uid in covered:
                if uid in counts:
                    counts[uid] += 1
        missing = [uid for uid in expected if counts.get(uid, 0) == 0]
        duplicate = [uid for uid, count in counts.items() if count > 1]
        return {
            "expected_unit_ids": expected,
            "covered_unit_ids": [uid for uid, count in counts.items() if count > 0],
            "missing_unit_ids": missing,
            "duplicate_unit_ids": duplicate,
            "coverage_rate": round((len(expected) - len(missing)) / max(1, len(expected)), 3),
        }

    @staticmethod
    def _script_coverage_units(input_text: str) -> list[dict[str, Any]]:
        """Cut the script into coverage units (roughly: sentences).

        The unit count is a hard ceiling on how many shots the script can be
        cut into (`_fill_shots_with_full_script` drops any shot beyond
        ``len(units)``), so this is not merely cosmetic. Splitting only on
        。！？ collapses a script written with line breaks or long
        comma-joined runs into one or two units, which then caps a
        multi-minute narration at a two-shot video. Line breaks and long runs
        are therefore split as well; a normally punctuated script is unaffected
        because its sentences are already under the length ceiling.
        """
        text = str(input_text or "").strip()
        if not text:
            return []

        max_unit_chars = 60
        min_secondary_chars = max_unit_chars // 2

        # 1. Sentence-final marks, semicolons and line breaks all end a unit.
        spans: list[tuple[int, int]] = []
        seg_start = 0
        for i, ch in enumerate(text):
            if ch in "。！？!?；;\n":
                spans.append((seg_start, i + 1))
                seg_start = i + 1
        if seg_start < len(text):
            spans.append((seg_start, len(text)))

        # 2. A run that is still too long gets broken at commas, but only where
        #    that leaves a piece worth calling a unit.
        split_on_commas: list[tuple[int, int]] = []
        for start, end in spans:
            if end - start <= max_unit_chars:
                split_on_commas.append((start, end))
                continue
            cut = start
            for i in range(start, end):
                if text[i] in "，,、" and (i + 1 - cut) >= min_secondary_chars:
                    split_on_commas.append((cut, i + 1))
                    cut = i + 1
            if cut < end:
                split_on_commas.append((cut, end))

        # 3. Text with no punctuation at all still has to be divisible.
        pieces: list[tuple[int, int]] = []
        for start, end in split_on_commas:
            while end - start > max_unit_chars * 2:
                pieces.append((start, start + max_unit_chars))
                start += max_unit_chars
            pieces.append((start, end))

        units: list[dict[str, Any]] = []
        for start, end in pieces:
            fragment = text[start:end]
            stripped = fragment.strip()
            if not stripped:
                continue
            offset = start + (len(fragment) - len(fragment.lstrip()))
            units.append({
                "unit_id": f"u{len(units) + 1:02d}",
                "char_start": offset,
                "char_end": offset + len(stripped),
                "text": stripped,
            })
        if not units:
            units = [{
                "unit_id": "u01", "char_start": 0,
                "char_end": len(text), "text": text,
            }]
        return units

    @staticmethod
    def _segment_anchor_unit_ids(
        segments: list[dict[str, Any]],
        global_units: list[dict[str, Any]],
    ) -> dict[str, list[str]]:
        units = [u for u in global_units if isinstance(u, dict)]
        if not segments:
            return {}
        anchors: dict[str, list[str]] = {}
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            sid = str(segment.get("segment_id") or "")
            if not sid:
                continue
            seg_text = InferenceOrchestrator._segment_narration_text(segment)
            hit_ids: list[str] = []
            for unit in units:
                uid = str(unit.get("unit_id", "") or "")
                ut = str(unit.get("text", "") or "")
                if not uid or not ut:
                    continue
                if ut in seg_text or seg_text in ut:
                    hit_ids.append(uid)
            anchors[sid] = hit_ids
        # Fallback: evenly partition coverage units across segments when matching fails
        if units and any(not anchors.get(str(seg.get("segment_id", "") or "")) for seg in segments if isinstance(seg, dict)):
            seg_ids = [str(seg.get("segment_id", "") or "") for seg in segments if isinstance(seg, dict) and seg.get("segment_id")]
            if seg_ids:
                for idx, unit in enumerate(units):
                    sid = seg_ids[min(len(seg_ids) - 1, idx * len(seg_ids) // max(1, len(units)))]
                    anchors.setdefault(sid, [])
                    uid = str(unit.get("unit_id", "") or "")
                    if uid and uid not in anchors[sid]:
                        anchors[sid].append(uid)
        return anchors

    @staticmethod
    def _enforce_total_shot_budget(shots: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
        if target_count <= 0:
            return shots
        if len(shots) <= target_count:
            return shots
        groups: list[list[dict[str, Any]]] = [[] for _ in range(target_count)]
        for idx, shot in enumerate(shots):
            gidx = min(target_count - 1, idx * target_count // max(1, len(shots)))
            groups[gidx].append(shot)
        merged: list[dict[str, Any]] = []
        for group in groups:
            if not group:
                continue
            base = copy.deepcopy(group[0])
            vocal_parts = [
                str(s.get("vocal_script", "") or "").strip()
                for s in group
                if str(s.get("vocal_script", "") or "").strip()
            ]
            if vocal_parts:
                base["vocal_script"] = " ".join(dict.fromkeys(vocal_parts))
            visual_parts = [
                str(s.get("visual_query", "") or "").strip()
                for s in group
                if str(s.get("visual_query", "") or "").strip()
            ]
            if visual_parts:
                base["visual_query"] = "；".join(dict.fromkeys(visual_parts[:2]))
            covered: list[str] = []
            for s in group:
                for uid in s.get("covered_unit_ids", []) if isinstance(s.get("covered_unit_ids"), list) else []:
                    suid = str(uid)
                    if suid and suid not in covered:
                        covered.append(suid)
            if covered:
                base["covered_unit_ids"] = covered
            base["duration_sec"] = max(3.0, round(len(str(base.get("vocal_script", "") or "")) / 4.5, 2))
            merged.append(base)
        return merged

    @staticmethod
    def _normalize_vocal_text(text: str) -> str:
        import re
        s = str(text or "").strip().lower()
        s = re.sub(r"[\s\u3000]+", "", s)
        s = re.sub(r"[，。！？、；：,.!?;:\"'“”‘’（）()\\-—…]", "", s)
        return s

    @staticmethod
    def _is_duplicate_vocal_text(text: str, previous_norms: list[str]) -> bool:
        from difflib import SequenceMatcher
        cur = InferenceOrchestrator._normalize_vocal_text(text)
        if not cur:
            return True
        for prev in previous_norms:
            if not prev:
                continue
            if cur == prev:
                return True
            if len(cur) >= 20 and (cur in prev or prev in cur):
                return True
            if SequenceMatcher(None, cur, prev).ratio() >= 0.90:
                return True
        return False

    @staticmethod
    def _fill_shots_with_full_script(
        shots: list[dict[str, Any]],
        global_units: list[dict[str, Any]],
        input_text: str,
    ) -> dict[str, Any]:
        """用模型给的镜头边界, 把**原文逐字**切给各个镜头 —— 全文一个字不丢。

        分工: 模型决定切几个镜头、每个镜头从哪一句开始; 系统负责把完整口播稿按这些
        边界铺满。单个镜头的文字**没有上限** —— 宁可某个镜头长, 也不丢内容。

        为什么不是"把 missing_unit_ids 补回去": _infer_covered_unit_ids 是模糊匹配
        (字符重叠 >= 55% 就算覆盖), 所以一个句元可以被判成"已覆盖"而镜头里其实只有
        它的一半。基于那份报告做增量补齐, 保证不了完整性。这里改成从原文构造:
        切点取自模型, 内容取自原文, 相邻镜头的字符区间首尾相接且铺满 [0, len),
        于是"各镜头 vocal_script 拼起来 == 原文"是构造上成立的, 不依赖任何匹配。
        """
        text = str(input_text or "")
        units = [u for u in global_units if isinstance(u, dict)]
        real_shots = [s for s in shots if isinstance(s, dict)]
        if not text or not units or not real_shots:
            return {"filled": 0, "dropped_shots": 0, "complete": False}

        n_units, n_shots = len(units), len(real_shots)
        # 镜头比句子还多时无法让每个镜头都拿到互不重复的内容, 合并掉多出来的尾巴。
        dropped = 0
        if n_shots > n_units:
            dropped = n_shots - n_units
            real_shots = real_shots[:n_units]
            n_shots = n_units

        uid_index = {str(u.get("unit_id") or ""): i for i, u in enumerate(units)}

        # 1. 锚点 = 该镜头覆盖到的第一个句元下标; 模型没给就先留空。
        anchors: list[int | None] = []
        for shot in real_shots:
            covered = shot.get("covered_unit_ids")
            idxs = [
                uid_index[str(uid)]
                for uid in (covered if isinstance(covered, list) else [])
                if str(uid) in uid_index
            ]
            anchors.append(min(idxs) if idxs else None)

        # 2. 空锚点按位置等比插值, 再夹成严格递增 —— 保证每个镜头至少分到一句,
        #    且不越界: 第 i 个镜头的起点必须 >= i, 且 <= n_units-(n_shots-i)。
        resolved: list[int] = []
        for i, a in enumerate(anchors):
            guess = a if a is not None else round(i * n_units / max(1, n_shots))
            low = i if not resolved else max(i, resolved[-1] + 1)
            high = n_units - (n_shots - i)
            resolved.append(max(low, min(int(guess), high)))
        resolved[0] = 0  # 开头的句子永远归第一个镜头, 否则前面会漏

        # 3. 按字符区间切原文。首尾相接, 铺满全文。
        starts = [0]
        for i in range(1, n_shots):
            starts.append(int(units[resolved[i]].get("char_start") or 0))
        bounds = starts + [len(text)]

        filled = 0
        for i, shot in enumerate(real_shots):
            piece = text[bounds[i]:bounds[i + 1]].strip()
            if not piece:
                continue
            shot["vocal_script"] = piece
            shot["covered_unit_ids"] = [
                str(u.get("unit_id") or "")
                for u in units[resolved[i]:(resolved[i + 1] if i + 1 < n_shots else n_units)]
                if str(u.get("unit_id") or "")
            ]
            shot["duration_sec"] = max(3.0, round(len(piece) / 4.5, 2))
            filled += 1

        if dropped:
            keep = {id(s) for s in real_shots}
            shots[:] = [s for s in shots if not isinstance(s, dict) or id(s) in keep]

        rejoined = "".join(
            str(s.get("vocal_script") or "") for s in real_shots
        )
        complete = len(rejoined) >= len("".join(text.split()))
        return {"filled": filled, "dropped_shots": dropped, "complete": complete}

    @staticmethod
    def _dedupe_shot_vocal_scripts(
        shots: list[dict[str, Any]],
        global_units: list[dict[str, Any]],
    ) -> dict[str, int]:
        unit_map = {
            str(u.get("unit_id", "") or ""): str(u.get("text", "") or "").strip()
            for u in global_units if isinstance(u, dict)
        }
        ordered_unit_ids = [uid for uid in unit_map.keys() if uid]
        used_unit_ids: set[str] = set()
        previous_norms: list[str] = []
        rewritten = 0

        for idx, shot in enumerate(shots):
            if not isinstance(shot, dict):
                continue
            original = str(shot.get("vocal_script", "") or "").strip()
            covered = [
                str(uid) for uid in shot.get("covered_unit_ids", [])
                if str(uid)
            ] if isinstance(shot.get("covered_unit_ids"), list) else []
            need_rewrite = InferenceOrchestrator._is_duplicate_vocal_text(original, previous_norms)
            if not need_rewrite:
                previous_norms.append(InferenceOrchestrator._normalize_vocal_text(original))
                used_unit_ids.update(covered)
                continue

            chosen_ids: list[str] = []
            for uid in covered:
                if uid in unit_map and uid not in used_unit_ids:
                    chosen_ids.append(uid)
            if not chosen_ids:
                for uid in covered:
                    if uid in unit_map:
                        chosen_ids.append(uid)
                        break
            if not chosen_ids:
                for uid in ordered_unit_ids:
                    if uid not in used_unit_ids:
                        chosen_ids.append(uid)
                        break
            if not chosen_ids and ordered_unit_ids:
                chosen_ids.append(ordered_unit_ids[min(len(ordered_unit_ids) - 1, idx)])

            rebuilt_parts = [unit_map.get(uid, "") for uid in chosen_ids if unit_map.get(uid, "")]
            rebuilt = " ".join(rebuilt_parts).strip()
            if not rebuilt:
                rebuilt = original
            if InferenceOrchestrator._is_duplicate_vocal_text(rebuilt, previous_norms):
                for uid in ordered_unit_ids:
                    extra = unit_map.get(uid, "")
                    if not extra:
                        continue
                    candidate = (rebuilt + " " + extra).strip() if rebuilt else extra
                    if not InferenceOrchestrator._is_duplicate_vocal_text(candidate, previous_norms):
                        rebuilt = candidate
                        if uid not in chosen_ids:
                            chosen_ids.append(uid)
                        break
            if InferenceOrchestrator._is_duplicate_vocal_text(rebuilt, previous_norms):
                rebuilt = f"{rebuilt}（镜头{idx + 1}）".strip()

            shot["vocal_script"] = rebuilt
            if chosen_ids:
                shot["covered_unit_ids"] = chosen_ids
            previous_norms.append(InferenceOrchestrator._normalize_vocal_text(rebuilt))
            used_unit_ids.update(chosen_ids)
            rewritten += 1

        return {"rewritten_count": rewritten}

    @staticmethod
    def _rebuild_segments_from_shots(
        segments: list[dict[str, Any]],
        shots: list[dict[str, Any]],
        anchor_units: dict[str, list[str]],
    ) -> list[dict[str, Any]]:
        if not segments:
            return segments
        ordered_segments = [copy.deepcopy(s) for s in segments if isinstance(s, dict)]
        seg_index = {str(s.get("segment_id", "")): s for s in ordered_segments}
        ordered_ids = [str(s.get("segment_id", "") or "") for s in ordered_segments if s.get("segment_id")]
        if not ordered_ids:
            return segments

        # Assign each shot to the best matching anchor segment by covered_unit_ids.
        for shot_idx, shot in enumerate(shots):
            if not isinstance(shot, dict):
                continue
            covered = [str(uid) for uid in shot.get("covered_unit_ids", [])] if isinstance(shot.get("covered_unit_ids"), list) else []
            best_sid = ""
            best_score = -1
            for sid in ordered_ids:
                units = anchor_units.get(sid, [])
                if not units:
                    continue
                score = len(set(covered) & set(units))
                if score > best_score:
                    best_score = score
                    best_sid = sid
            if not best_sid:
                best_sid = ordered_ids[min(len(ordered_ids) - 1, shot_idx * len(ordered_ids) // max(1, len(shots)))]
            seg_meta = seg_index.get(best_sid, {})
            shot["segment_id"] = best_sid
            if seg_meta.get("title"):
                shot["chapter_title"] = str(seg_meta.get("title", ""))
            if seg_meta.get("structure_label"):
                shot["structure_label"] = str(seg_meta.get("structure_label", ""))

        # Ensure every segment has at least one shot (avoid planned=false dead loop).
        bucket: dict[str, list[dict[str, Any]]] = {}
        for shot in shots:
            if isinstance(shot, dict) and shot.get("segment_id"):
                sid = str(shot.get("segment_id"))
                bucket.setdefault(sid, []).append(shot)
        if shots:
            for i, sid in enumerate(ordered_ids):
                if bucket.get(sid):
                    continue
                donor = shots[min(len(shots) - 1, i * len(shots) // max(1, len(ordered_ids)))]
                donor["segment_id"] = sid
                meta = seg_index.get(sid, {})
                donor["chapter_title"] = str(meta.get("title", donor.get("chapter_title", "")) or "")
                donor["structure_label"] = str(meta.get("structure_label", donor.get("structure_label", "")) or "")
                bucket.setdefault(sid, []).append(donor)

        # Backfill segment summaries based on rebuilt shot buckets.
        for sid in ordered_ids:
            seg = seg_index.get(sid, {})
            seg_shots = bucket.get(sid, [])
            if not seg_shots:
                continue
            seg["num_shots"] = len(seg_shots)
            merged_text = " ".join(str(s.get("vocal_script", "") or "") for s in seg_shots).strip()
            if merged_text:
                # 不再截断到 1200: 这个字段是章节口播的回填, 截断会让章节稿与镜头稿
                # 对不上, 长稿尤其明显。镜头文字没有上限, 章节汇总同理。
                seg["narration_script"] = merged_text
        return ordered_segments

    @staticmethod
    def _build_minimal_segment_shot(
        segment: dict[str, Any],
        shot_id: str,
        coverage_units: list[dict[str, Any]],
    ) -> dict[str, Any]:
        text = InferenceOrchestrator._segment_narration_text(segment)
        primary = ""
        if coverage_units:
            primary = str(coverage_units[0].get("text", "") or "").strip()
        if not primary:
            primary = text
        if not primary:
            primary = str(segment.get("summary", "") or segment.get("title", "") or "本章节重点内容").strip()
        title = str(segment.get("title", "") or "")
        structure = str(segment.get("structure_label", "") or "")
        seg_id = str(segment.get("segment_id", "") or "")
        all_unit_ids = [
            str(item.get("unit_id", "") or "")
            for item in coverage_units
            if str(item.get("unit_id", "") or "")
        ]
        return {
            "shot_id": str(shot_id),
            "segment_id": seg_id,
            "structure_label": structure,
            "chapter_title": title,
            "vocal_script": primary,
            "visual_query": f"{title} 主题相关纪实画面，突出关键事实与信息点",
            "visual_direction": str(segment.get("planner_strategy", "") or ""),
            "intent": f"补齐 {title} 章节覆盖，避免漏章",
            "design_rationale": "模型未稳定产出本章镜头时，使用最小可执行镜头保证流程可继续。",
            "covered_unit_ids": all_unit_ids or ["u01"],
            "coverage_rationale": "兜底镜头覆盖本章节主要句元",
            "is_segment_first": True,
            "is_spotlight": False,
            "duration_sec": max(3.0, round(len(primary) / 4.5, 2)),
        }

    @staticmethod
    def _enforce_segment_shot_budget(
        shots: list[dict[str, Any]], budget: int,
    ) -> list[dict[str, Any]]:
        if budget <= 0 or len(shots) <= budget:
            return shots
        groups: list[list[dict[str, Any]]] = [[] for _ in range(budget)]
        for idx, shot in enumerate(shots):
            group_idx = min(budget - 1, idx * budget // max(1, len(shots)))
            groups[group_idx].append(shot)

        merged: list[dict[str, Any]] = []
        for group in groups:
            if not group:
                continue
            base = copy.deepcopy(group[0])
            vocal_parts = [
                str(item.get("vocal_script", "") or "").strip()
                for item in group
                if str(item.get("vocal_script", "") or "").strip()
            ]
            if vocal_parts:
                base["vocal_script"] = " ".join(dict.fromkeys(vocal_parts))
                base["duration_sec"] = max(3.0, round(len(base["vocal_script"]) / 4.5, 2))
            visual_parts = [
                str(item.get("visual_query", "") or "").strip()
                for item in group
                if str(item.get("visual_query", "") or "").strip()
            ]
            if visual_parts:
                base["visual_query"] = "；".join(dict.fromkeys(visual_parts[:2]))
            intent_parts = [
                str(item.get("intent", "") or "").strip()
                for item in group
                if str(item.get("intent", "") or "").strip()
            ]
            if intent_parts:
                base["intent"] = "；".join(dict.fromkeys(intent_parts[:2]))
            covered: list[str] = []
            for item in group:
                for uid in item.get("covered_unit_ids", []) if isinstance(item.get("covered_unit_ids"), list) else []:
                    if str(uid) and str(uid) not in covered:
                        covered.append(str(uid))
            if covered:
                base["covered_unit_ids"] = covered
            base["is_spotlight"] = any(bool(item.get("is_spotlight")) for item in group)
            base["budget_merge_count"] = len(group)
            merged.append(base)
        return merged

    @staticmethod
    def _heuristic_severity_scores(
        state: dict[str, Any],
        tool_args: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Rank shots by problem severity using real asset metadata.

        Used when the judge LLM fails to produce a complete score set. Reuses
        ``pipeline.graph._heuristic_severity`` so the signal (audio/video
        duration mismatch, missing caption, decoration verify score) matches
        what the graph node itself would fall back to, then selects the top
        slice honouring the ``top_ratio``/``min_shots`` the model asked for.
        """
        from pipeline.graph import _heuristic_severity

        tool_args = tool_args or {}
        shot_assets = state.get("shot_assets", {}) or {}
        shot_plan = state.get("shot_plan", []) or []

        def _num(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        scores: dict[str, dict[str, Any]] = {}
        for shot in shot_plan:
            sid = str(shot.get("shot_id", ""))
            if not sid:
                continue
            assets = shot_assets.get(sid, {})
            assets = assets if isinstance(assets, dict) else {}
            score = _heuristic_severity(shot, assets)
            # Missing footage/audio outranks every quality nit.
            if not assets.get("clip_path") or not assets.get("audio_path"):
                score = 1.0
            scores[sid] = {"score": score, "rank": 0, "in_top_ratio": False}

        if not scores:
            return scores

        top_ratio = max(0.0, min(1.0, _num(tool_args.get("top_ratio"), 0.25)))
        min_shots = int(_num(tool_args.get("min_shots"), 4))
        ordered = sorted(scores, key=lambda s: scores[s]["score"], reverse=True)
        num_top = max(min_shots, int(len(ordered) * top_ratio + 0.5))
        num_top = max(1, min(num_top, len(ordered)))
        for rank, sid in enumerate(ordered):
            scores[sid]["rank"] = rank + 1
            scores[sid]["in_top_ratio"] = rank < num_top
        return scores

    @staticmethod
    def _concat_clips_to_tts_duration(
        shot_id: str,
        primary_clip: str,
        candidates: list[dict[str, Any]],
        target_duration: float,
        output_dir: str,
    ) -> dict[str, Any]:
        """Concatenate candidate clips until total >= target_duration, then trim.

        Mirrors the data-annotation pipeline: pick clips in match-score order,
        concatenate with ffmpeg, and trim the last clip to align exactly to
        the TTS duration.
        """
        import subprocess

        project_dir = Path(output_dir) / "concat_clips"
        project_dir.mkdir(parents=True, exist_ok=True)

        clip_paths = [primary_clip]
        total_dur = 0.0

        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", primary_clip],
                capture_output=True, text=True, timeout=10,
            )
            total_dur = float(probe.stdout.strip()) if probe.returncode == 0 else 0.0
        except Exception:
            total_dur = 0.0

        if total_dur >= target_duration:
            out_path = str(project_dir / f"{shot_id}_trimmed.mp4")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", primary_clip,
                     "-t", f"{target_duration:.3f}",
                     "-c", "copy", "-avoid_negative_ts", "1", out_path],
                    capture_output=True, timeout=60,
                )
                return {
                    "success": True,
                    "output_path": out_path if Path(out_path).exists() else primary_clip,
                    "final_duration": target_duration,
                    "clip_count": 1,
                    "trimmed": True,
                }
            except Exception:
                return {"success": False}

        for cand in candidates:
            if total_dur >= target_duration:
                break
            cand_path = str(cand.get("clip_path", "") or cand.get("path", "") or "")
            if not cand_path or cand_path == primary_clip or not Path(cand_path).exists():
                continue
            try:
                probe = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                     "-of", "csv=p=0", cand_path],
                    capture_output=True, text=True, timeout=10,
                )
                cand_dur = float(probe.stdout.strip()) if probe.returncode == 0 else 0.0
            except Exception:
                continue
            if cand_dur <= 0.5:
                continue
            clip_paths.append(cand_path)
            total_dur += cand_dur

        if len(clip_paths) <= 1 and total_dur < target_duration:
            return {"success": False, "reason": "insufficient candidates"}

        concat_list = project_dir / f"{shot_id}_concat.txt"
        concat_list.write_text(
            "\n".join(f"file '{p}'" for p in clip_paths),
            encoding="utf-8",
        )
        raw_concat = str(project_dir / f"{shot_id}_raw.mp4")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", str(concat_list), "-c", "copy", raw_concat],
                capture_output=True, timeout=120,
            )
        except Exception:
            return {"success": False, "reason": "ffmpeg concat failed"}

        if not Path(raw_concat).exists():
            return {"success": False, "reason": "concat output missing"}

        final_path = str(project_dir / f"{shot_id}_aligned.mp4")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", raw_concat,
                 "-t", f"{target_duration:.3f}",
                 "-c", "copy", "-avoid_negative_ts", "1", final_path],
                capture_output=True, timeout=60,
            )
        except Exception:
            final_path = raw_concat

        out = final_path if Path(final_path).exists() else raw_concat
        return {
            "success": Path(out).exists(),
            "output_path": out,
            "final_duration": round(min(total_dur, target_duration), 3),
            "clip_count": len(clip_paths),
            "trimmed": total_dur > target_duration,
        }

    def _force_tool_for_stage(self, stage: str, tool_name: str, state: dict[str, Any]) -> str:
        """Conservative runtime ordering guardrails for inference."""
        allowed = set(self.allowed_map.get(stage, []))
        if stage == "stage2_planning":
            if not state.get("segments") and "Tool_SegmentArticle" in allowed:
                return "Tool_SegmentArticle"
            if state.get("segments") and not state.get("shot_plan") and "Tool_PlanShots" in allowed:
                return "Tool_PlanShots"
        if stage == "stage3_decoration":
            has_spotlight = any(s.get("is_spotlight") for s in state.get("shot_plan", []))
            has_decoration = any(
                isinstance(v, dict) and v.get("decoration")
                for v in (state.get("shot_assets", {}) or {}).values()
            )
            if has_spotlight and not has_decoration and "Tool_BatchDecorateSpotlights" in allowed:
                return "Tool_BatchDecorateSpotlights"
        if stage == "stage3_post_decoration_reflect":
            if self._needs_reflect() and "Tool_SelfReflect" in allowed:
                return "Tool_SelfReflect"
            if self.open_issues and "Tool_SelfReflect_Patch" in allowed:
                return "Tool_SelfReflect_Patch"
        if stage == "stage4_severity_rank" and "Tool_SeverityRank" in allowed:
            return "Tool_SeverityRank"
        if stage == "stage5_final_render" and "Tool_FinalRender" in allowed:
            return "Tool_FinalRender"
        if stage == "stage5_stitch" and "Tool_StitchVideo" in allowed:
            return "Tool_StitchVideo"
        return tool_name

    def update_state_from_respond(self, tool_name: str, output: dict[str, Any]) -> None:
        """Update internal state accumulator from tool output (mirrors _StateAccumulator)."""
        if tool_name == "Tool_SegmentArticle":
            segs = output.get("segments")
            if isinstance(segs, list):
                self.segments = [s for s in segs if isinstance(s, dict)]
                self.num_segments = len(segs)

        if tool_name == "Tool_PlanShots":
            shots = output.get("shots")
            if isinstance(shots, list):
                for s in shots:
                    if isinstance(s, dict) and s.get("shot_id") is not None:
                        sid = str(s["shot_id"])
                        self.shots.setdefault(sid, {})
                        self.shots[sid].update({
                            "shot_id": sid,
                            "segment_id": s.get("segment_id", ""),
                            "vocal_script": _trunc(s.get("vocal_script", ""), 300),
                            "visual_query": _trunc(s.get("visual_query", ""), 300),
                            "is_spotlight": bool(s.get("is_spotlight")),
                            "chapter_title": _trunc(s.get("chapter_title", ""), 60),
                        })

        if tool_name == "Tool_BatchVisualAudio":
            self.visual_audio_done = True
            for track_key, dur_key, score_key in [
                ("visual", "asset_duration", "match_score"),
                ("audio", "tts_duration", None),
            ]:
                track = output.get(track_key)
                if isinstance(track, dict):
                    for sid, info in track.items():
                        if not isinstance(info, dict):
                            continue
                        sid = str(sid)
                        self.shots.setdefault(sid, {"shot_id": sid})
                        if "duration_sec" in info:
                            self.shots[sid][dur_key] = _round_val(info["duration_sec"])
                        if score_key and score_key in info:
                            self.shots[sid][score_key] = _round_val(info[score_key])

        if tool_name in ("Tool_Stage4DimensionJudge", "Tool_TextQuickJudge"):
            self.quality_checked = True
            shots = output.get("shots")
            if isinstance(shots, dict):
                for sid, info in shots.items():
                    if not isinstance(info, dict):
                        continue
                    sid = str(sid)
                    self.shots.setdefault(sid, {"shot_id": sid})
                    passed = bool(info.get("passed"))
                    self.shots[sid].update({
                        "judge_status": "passed" if passed else "failed",
                        "score": _round_val(info.get("overall_score")),
                        "failed_dimensions": info.get("failed_dimensions") or [],
                    })
            self.last_judge = {
                "passed_count": output.get("passed_count"),
                "failed_count": output.get("failed_count"),
            }

        if tool_name == "Tool_MusicSearch":
            self.bgm = {"status": output.get("status")}

        if tool_name == "Tool_BatchDecorateSpotlights":
            self.decorated_done = True

        if tool_name == "Tool_BatchTimelineCompose":
            render_ok = int(output.get("render_success_count") or 0)
            self.timeline_done = bool(render_ok > 0)

        if tool_name == "Tool_Stage4DimensionalRepair":
            self.repaired_done = True

        if tool_name == "Tool_StitchVideo":
            self.stitched_done = bool(
                output.get("final_video_path")
                or output.get("video_path")
                or output.get("output_path")
            )

        # Summarize observation
        keep_keys = (
            "status", "passed", "failed_count", "passed_count", "shot_count",
            "num_shots", "num_spotlight", "overall_score", "summary",
            "tts_success_count", "visual_success_count", "fallback_used",
        )
        obs: dict[str, Any] = {"tool": tool_name}
        for k in keep_keys:
            if k in output:
                v = output[k]
                if isinstance(v, str):
                    v = v[:120]
                obs[k] = v
        self.last_observation = obs

    def _agent_precondition_missing(self, tool_name: str, state: dict[str, Any]) -> list[str]:
        flags = self._agent_flags(state)
        missing: list[str] = []
        if tool_name not in LONG_HORIZON_ACTION_SPACE:
            return ["valid_tool_name"]

        if flags.get("quality_checked") and tool_name in {
            "Tool_SegmentArticle", "Tool_PlanShots",
        }:
            missing.append("phase_regression_blocked")
            return missing

        if tool_name == "Tool_PlanShots" and not flags["segmented"]:
            missing.append("segments")
        elif tool_name == "Tool_SelfReflect":
            # SelfReflect can be triggered flexibly at multiple points:
            # 1. After planning (self_reflect_plan): needs shot_plan
            # 2. After decoration (quality_judge): needs decorated flag
            # Only block if there's truly nothing to reflect on
            if not flags["planned"]:
                missing.append("shot_plan")
        elif tool_name == "Tool_SelfReflect_Patch" and not self.open_issues:
            missing.append("self_reflect_issues")
        elif tool_name == "Tool_BatchVisualAudio" and not flags["planned"]:
            missing.append("shots")
        elif tool_name == "Tool_BatchDecorateSpotlights":
            has_spotlight = any(
                isinstance(s, dict) and s.get("is_spotlight")
                for s in state.get("shot_plan", []) or []
            )
            if not has_spotlight:
                missing.append("spotlight_shots")
        elif tool_name == "Tool_BatchTimelineCompose":
            if not flags["visual_audio_done"]:
                missing.append("visual_audio_observation")
            if not flags["decorated"]:
                missing.append("spotlight_decoration")
            if not flags["quality_checked"]:
                missing.append("dimension_quality_check")
        elif tool_name == "Tool_MusicSearch":
            if not state.get("music_style") and not flags["visual_audio_done"]:
                missing.append("music_style")
        elif tool_name == "Tool_SeverityRank" and not flags["visual_audio_done"]:
            missing.append("visual_audio_observation")
        elif tool_name == "Tool_Stage4DimensionJudge" and not flags["decorated"]:
            missing.append("spotlight_decoration")
        elif tool_name == "Tool_Stage4DimensionalRepair" and not (self._failed_judge_ids(state) or self.last_judge):
            missing.append("failed_dimensions")
        elif tool_name == "Tool_StitchVideo":
            for flag, label in [
                ("planned", "shot_plan"),
                ("visual_audio_done", "visual_audio"),
                ("timeline_done", "timeline_render"),
                ("bgm_done", "bgm"),
            ]:
                if not flags[flag]:
                    missing.append(label)
            if not flags["decorated"]:
                missing.append("spotlight_decoration")
            ready_count, total_count = self._render_ready_counts(state)
            if ready_count < total_count:
                if not flags["timeline_done"] or ready_count < max(1, total_count // 2):
                    missing.append(f"rendered_path_ready:{ready_count}/{total_count}")
        return missing

    def _agent_tool_already_done(self, tool_name: str, state: dict[str, Any]) -> bool:
        """Detect repeated actions that cannot add useful state in this run."""
        flags = self._agent_flags(state)
        if tool_name == "Tool_SegmentArticle":
            return flags["segmented"]
        if tool_name == "Tool_PlanShots":
            return flags["planned"]
        if tool_name == "Tool_SelfReflect":
            # Never mark as done if there's a pending reflect trigger
            if self._needs_reflect():
                return False
            return self._last_reflect_generation >= self._reflect_generation
        if tool_name == "Tool_SelfReflect_Patch":
            return self._last_reflect_generation >= self._reflect_generation and not self.open_issues
        if tool_name == "Tool_BatchVisualAudio":
            return flags["visual_audio_done"]
        if tool_name == "Tool_BatchDecorateSpotlights":
            return flags["decorated"]
        if tool_name == "Tool_BatchTimelineCompose":
            return flags["timeline_done"]
        if tool_name == "Tool_MusicSearch":
            return flags["bgm_done"]
        if tool_name == "Tool_SeverityRank":
            return bool(state.get("severity_scores") or state.get("severity_ranked_shots"))
        if tool_name == "Tool_Stage4DimensionJudge":
            return flags["quality_checked"]
        if tool_name == "Tool_Stage4DimensionalRepair":
            return bool(flags["quality_checked"] and (flags["repaired"] or not self._failed_judge_ids(state)))
        if tool_name == "Tool_StitchVideo":
            return flags["stitched"]
        return False

    def _needs_reflect(self) -> bool:
        """True when state has changed since the last SelfReflect (new generation)."""
        return self._last_reflect_generation < self._reflect_generation

    def _next_missing_agent_tool(self, state: dict[str, Any]) -> str:
        """Deterministically advance the long-horizon loop when the model repeats."""
        flags = self._agent_flags(state)
        if not flags["segmented"]:
            return "Tool_SegmentArticle"
        if not flags["planned"]:
            return "Tool_PlanShots"
        # SelfReflect after planning (self_reflect_plan mode)
        if self._needs_reflect() and flags["planned"] and not flags["visual_audio_done"]:
            return "Tool_SelfReflect"
        if self.open_issues:
            return "Tool_SelfReflect_Patch"
        if not flags["visual_audio_done"]:
            return "Tool_BatchVisualAudio"
        if not flags["decorated"]:
            return "Tool_BatchDecorateSpotlights"
        # SelfReflect after decoration (quality_judge mode) - HIGH PRIORITY
        if self._needs_reflect() and flags["decorated"]:
            return "Tool_SelfReflect"
        if self.open_issues:
            return "Tool_SelfReflect_Patch"
        # Quality checks BEFORE rendering
        if not state.get("severity_scores") and not state.get("severity_ranked_shots"):
            return "Tool_SeverityRank"
        if not flags["quality_checked"]:
            return "Tool_Stage4DimensionJudge"
        if self._failed_judge_ids(state) and not flags["repaired"]:
            return "Tool_Stage4DimensionalRepair"
        # Render ONCE after all quality checks pass
        if not flags["timeline_done"]:
            return "Tool_BatchTimelineCompose"
        if not flags["bgm_done"]:
            return "Tool_MusicSearch"
        return "Tool_StitchVideo"

    def _phase_for_tool(self, tool_name: str) -> str:
        if tool_name in {"Tool_SegmentArticle", "Tool_PlanShots"}:
            return "plan"
        if tool_name in {"Tool_SelfReflect", "Tool_SelfReflect_Patch"}:
            return "reflect_repair_plan"
        if tool_name in {
            "Tool_BatchVisualAudio", "Tool_BatchDecorateSpotlights",
            "Tool_BatchTimelineCompose", "Tool_MusicSearch",
        }:
            return "asset_and_compose"
        if tool_name in {"Tool_SeverityRank", "Tool_Stage4DimensionJudge", "Tool_Stage4DimensionalRepair"}:
            return "quality_repair"
        if tool_name == "Tool_StitchVideo":
            return "final_stitch"
        return "agent_loop"

    def _record_agent_history(self, tool_name: str, output: dict[str, Any]) -> None:
        obs = self._make_agent_observation(tool_name, output)
        self.last_observation = obs
        subtype = LONG_HORIZON_TOOL_SUBTYPES.get(tool_name, tool_name.lower().replace("tool_", ""))
        self.agent_history.append({
            "tool": tool_name,
            "subtype": subtype,
            "status": obs.get("status", ""),
            "summary": obs.get("summary", {}),
        })
        if len(self.agent_history) > 80:
            self.agent_history = self.agent_history[-80:]

    def _make_agent_observation(self, tool_name: str, output: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(output, dict):
            status = "success"
            return {"event": "tool_observation", "tool": tool_name, "status": status, "summary": {}}
        summary: dict[str, Any] = {}
        for key in (
            "num_segments", "num_shots", "num_spotlight", "shot_count",
            "passed_count", "failed_count", "tts_success_count",
            "visual_success_count", "render_success_count", "approved_count",
            "selected_count", "total_duration_sec", "has_final_video",
            "missing", "message",
        ):
            if key in output:
                value = output[key]
                summary[key] = _trunc(value, 240) if isinstance(value, str) else value
        if tool_name == "Tool_SegmentArticle":
            summary["num_segments"] = output.get("num_segments", len(output.get("segments") or []))
            summary["music_style"] = str(output.get("music_style") or "")
        if tool_name == "Tool_PlanShots":
            summary["num_shots_added"] = output.get("num_shots", len(output.get("shots") or []))
            summary["total_shots"] = len(self.shots) if hasattr(self, "shots") else 0
            summary["segments_planned_this_call"] = output.get("_segments_planned", 1)
            summary["all_segments_planned"] = True
        if tool_name == "Tool_SelfReflect":
            summary["num_issues"] = len(output.get("issues") or [])
        if tool_name == "Tool_SelfReflect_Patch":
            summary["patched_shot_ids"] = list((output.get("patched_shot_plan") or output.get("patch") or {}).keys())[:20]
        if tool_name == "Tool_BatchDecorateSpotlights":
            summary["decorated_shot_ids"] = list((output.get("results") or {}).keys())[:20]
        if tool_name == "Tool_BatchVisualAudio":
            summary.setdefault("tts_success_count", output.get("tts_success_count", 0))
            summary.setdefault("visual_success_count", output.get("visual_success_count", 0))
        if tool_name == "Tool_MusicSearch":
            summary["tracks"] = (output.get("tracks") or [])[:3]
        if tool_name == "Tool_SeverityRank":
            summary["top_shot_ids"] = (output.get("top_shot_ids") or [])[:20]
            summary["ranked_count"] = output.get("ranked_count", 0)
        if tool_name == "Tool_Stage4DimensionJudge":
            summary["failed_shot_ids"] = self._failed_judge_ids_from_output(output)
        if tool_name == "Tool_Stage4DimensionalRepair":
            summary["repaired_shot_ids"] = list((output.get("shots") or output.get("repairs") or {}).keys())[:20]
        if tool_name == "Tool_StitchVideo":
            summary["has_final_video"] = bool(
                output.get("final_video_path")
                or output.get("video_path")
                or output.get("output_path")
            )
            # With `render.defer_stitch` the call validates the shots and stops
            # there, so "no final video" is the expected outcome, not a
            # failure. Without these keys nothing downstream can tell the two
            # apart and the run reads as broken.
            if output.get("deferred"):
                summary["deferred"] = True
                summary["ready_shot_count"] = output.get("ready_shot_count", 0)
        raw_status = str(output.get("status") or output.get("concat_status") or "")
        status = raw_status if raw_status else "success"
        return {
            "event": "tool_observation",
            "tool": tool_name,
            "status": status,
            "summary": summary,
        }

    @staticmethod
    def _failed_judge_ids_from_output(output: dict[str, Any]) -> list[str]:
        shots = output.get("shots")
        if not isinstance(shots, dict):
            return []
        return [
            str(sid) for sid, info in shots.items()
            if isinstance(info, dict) and not info.get("passed", True)
        ][:20]

    @staticmethod
    def _parse_jsonish_agent_arg(value: Any) -> Any:
        """Parse model-emitted JSON arguments without inventing fallback content."""
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return value
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        in_string = False
        escape = False
        stack: list[str] = []
        for ch in text:
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in ("[", "{"):
                stack.append("]" if ch == "[" else "}")
            elif ch in ("]", "}") and stack:
                stack.pop()

        if in_string:
            text += '"'
        repaired = text + "".join(reversed(stack))
        try:
            return json.loads(repaired)
        except (json.JSONDecodeError, TypeError):
            salvaged = InferenceOrchestrator._salvage_json_objects_from_text(text)
            return salvaged if salvaged else value

    @staticmethod
    def _salvage_json_objects_from_text(text: str) -> list[dict[str, Any]]:
        """Extract complete JSON objects from a partially generated list."""
        objects: list[dict[str, Any]] = []
        depth = 0
        start = -1
        in_string = False
        escape = False
        for idx, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif ch == "}" and depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        obj = json.loads(text[start:idx + 1])
                    except (json.JSONDecodeError, TypeError):
                        obj = None
                    if isinstance(obj, dict):
                        objects.append(obj)
                    start = -1
        return objects

    @staticmethod
    def _salvage_json_mapping(text: str) -> dict[str, Any]:
        """Recover complete "key": value pairs from a truncated JSON object."""
        text = str(text or "").strip()
        start = text.find("{")
        if start < 0:
            return {}
        i = start + 1
        n = len(text)
        result: dict[str, Any] = {}

        def skip_ws(j: int) -> int:
            while j < n and text[j] in " \t\r\n,":
                j += 1
            return j

        while i < n:
            i = skip_ws(i)
            if i >= n or text[i] == "}":
                break
            if text[i] != '"':
                break
            key_start = i + 1
            j = key_start
            esc = False
            while j < n:
                c = text[j]
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    break
                j += 1
            if j >= n:
                break
            key = text[key_start:j]
            i = skip_ws(j + 1)
            if i >= n or text[i] != ":":
                break
            i = skip_ws(i + 1)
            if i >= n:
                break
            val_start = i
            depth = 0
            in_str = False
            esc = False
            val_end = -1
            while i < n:
                c = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                elif c == '"':
                    in_str = True
                elif c in "{[":
                    depth += 1
                elif c in "}]":
                    depth -= 1
                    if depth == 0:
                        val_end = i + 1
                        break
                    if depth < 0:
                        break
                elif depth == 0 and c == ",":
                    val_end = i
                    break
                i += 1
            if val_end < 0:
                break
            raw_val = text[val_start:val_end].strip().rstrip(",").strip()
            try:
                result[key] = json.loads(raw_val)
            except (json.JSONDecodeError, TypeError):
                break
            i = val_end
        return result

    @classmethod
    def _coerce_jsonish_mapping(cls, value: Any) -> dict[str, Any]:
        """Coerce a model-emitted mapping arg (possibly truncated string) to dict."""
        if isinstance(value, dict):
            return value
        parsed = cls._parse_jsonish_agent_arg(value)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(value, str):
            salvaged = cls._salvage_json_mapping(value)
            if salvaged:
                return salvaged
        return {}

    @staticmethod
    def _normalize_agent_segments(raw_segments: Any) -> list[dict[str, Any]]:
        """Normalize model-emitted segment schema while preserving model content."""
        if isinstance(raw_segments, dict) and isinstance(raw_segments.get("segments"), list):
            raw_segments = raw_segments.get("segments", [])
        if not isinstance(raw_segments, list):
            return []

        segments: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_segments, start=1):
            if not isinstance(item, dict):
                continue
            seg = copy.deepcopy(item)
            raw_id = (
                seg.get("segment_id")
                or seg.get("segment_no")
                or seg.get("segment_index")
                or seg.get("id")
                or idx
            )
            raw_id = str(raw_id)
            if raw_id.isdigit():
                raw_id = f"seg_{int(raw_id):02d}"
            seg["segment_id"] = raw_id
            if not seg.get("summary") and seg.get("overview"):
                seg["summary"] = seg.get("overview")
            if not isinstance(seg.get("key_points"), list):
                seg["key_points"] = []
            segments.append(seg)
        return segments

    @staticmethod
    def _normalize_agent_shots(raw_shots: Any) -> list[dict[str, Any]]:
        """Normalize model-emitted shot schema while preserving model content."""
        if isinstance(raw_shots, dict):
            for key in ("shots", "shot_plan", "video_shots"):
                if isinstance(raw_shots.get(key), list):
                    raw_shots = raw_shots.get(key)
                    break
        if isinstance(raw_shots, dict):
            if all(isinstance(v, dict) for v in raw_shots.values()):
                raw_shots = list(raw_shots.values())
            else:
                return []
        if not isinstance(raw_shots, list):
            return []

        shots: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_shots, start=1):
            if not isinstance(item, dict):
                continue
            shot = copy.deepcopy(item)
            shot["shot_id"] = str(
                shot.get("shot_id")
                or shot.get("shot_no")
                or shot.get("shot_index")
                or f"shot_{idx:03d}"
            )
            if shot["shot_id"].isdigit():
                shot["shot_id"] = f"shot_{int(shot['shot_id']):03d}"
            if not shot.get("vocal_script"):
                shot["vocal_script"] = (
                    shot.get("vocal_query")
                    or shot.get("tts_text")
                    or shot.get("audio_text")
                    or shot.get("narration")
                    or shot.get("script")
                    or shot.get("voiceover")
                    or shot.get("subtitle")
                    or ""
                )
            if not shot.get("visual_query"):
                shot["visual_query"] = (
                    shot.get("visual_prompt")
                    or shot.get("visual_description")
                    or shot.get("scene_description")
                    or shot.get("image_prompt")
                    or ""
                )
            if not shot.get("chapter_title"):
                shot["chapter_title"] = shot.get("title", "")
            shots.append(shot)
        return shots

    def _dispatch_semantic_agent_action(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Apply long-horizon semantic actions directly from model arguments."""
        if not self.is_long_horizon_agent() or not isinstance(tool_args, dict):
            return None

        if tool_name == "Tool_SegmentArticle":
            raw_segments = (
                tool_args.get("segments")
                or tool_args.get("segment_plan")
                or tool_args.get("segmentations")
                or []
            )
            raw_segments = self._parse_jsonish_agent_arg(raw_segments)
            segments = self._normalize_agent_segments(raw_segments)
            if segments:
                return {
                    "status": "success",
                    "intent": tool_args.get("intent", ""),
                    "segments": segments,
                    "music_style": (
                        tool_args.get("music_style")
                        or tool_args.get("bgm_style")
                        or tool_args.get("style")
                        or ""
                    ),
                    "num_segments": len(segments),
                }

        if tool_name == "Tool_PlanShots" and isinstance(tool_args.get("shots"), list):
            existing = [
                copy.deepcopy(s)
                for s in state.get("shot_plan", [])
                if isinstance(s, dict)
            ]
            incoming = self._normalize_agent_shots(tool_args.get("shots", []))
            merged = self._merge_shot_chunks(existing, incoming)
            requested = int(
                tool_args.get("num_video_shots")
                or tool_args.get("shot_count")
                or tool_args.get("num_shots")
                or tool_args.get("num_shots_added")
                or 0
            )
            input_text = str(state.get("input_text", "") or "")
            if input_text:
                requested = max(requested, int(len(input_text) / 120))
            normalized = self._normalize_shot_plan(
                state.get("segments", []),
                merged,
                target_count=requested,
                input_text=input_text,
            )
            return {
                "status": "success",
                "shots": normalized,
                "num_shots": len(normalized),
                "num_spotlight": sum(1 for s in normalized if s.get("is_spotlight")),
            }

        if tool_name == "Tool_PlanShots" and not isinstance(tool_args.get("shots"), list):
            raw_shots = (
                tool_args.get("shots")
                or tool_args.get("shot_plan")
                or tool_args.get("video_shots")
                or []
            )
            raw_shots = self._parse_jsonish_agent_arg(raw_shots)
            raw_shots = self._normalize_agent_shots(raw_shots)
            if raw_shots:
                existing = [
                    copy.deepcopy(s)
                    for s in state.get("shot_plan", [])
                    if isinstance(s, dict)
                ]
                merged = self._merge_shot_chunks(existing, raw_shots)
                requested = int(tool_args.get("num_video_shots") or tool_args.get("num_shots") or 0)
                input_text = str(state.get("input_text", "") or "")
                if input_text:
                    requested = max(requested, int(len(input_text) / 120))
                normalized = self._normalize_shot_plan(
                    state.get("segments", []),
                    merged,
                    target_count=requested,
                    input_text=input_text,
                )
                return {
                    "status": "success",
                    "shots": normalized,
                    "num_shots": len(normalized),
                    "num_spotlight": sum(1 for s in normalized if s.get("is_spotlight")),
                }

        if tool_name == "Tool_SelfReflect" and ("passed" in tool_args or "issues" in tool_args):
            return {
                "status": "approved" if tool_args.get("passed") else "needs_fix",
                "passed": bool(tool_args.get("passed")),
                "overall_score": tool_args.get("overall_score"),
                "issues": tool_args.get("issues") if isinstance(tool_args.get("issues"), list) else [],
                "summary": tool_args.get("summary", ""),
            }

        if tool_name == "Tool_SelfReflect_Patch" and isinstance(tool_args.get("patched_shot_plan"), dict):
            patch = tool_args.get("patched_shot_plan") or {}
            patched = self._apply_plan_patch(state.get("shot_plan", []), patch)
            return {
                "status": "patched",
                "repair_applied": True,
                "repair_summary": tool_args.get("repair_summary", ""),
                "requires_downstream_propagation": bool(tool_args.get("requires_downstream_propagation")),
                "patched_shot_plan": patch,
                "shots": patched,
            }

        if tool_name == "Tool_BatchDecorateSpotlights":
            decorations = self._coerce_jsonish_mapping(
                tool_args.get("decorations")
                or tool_args.get("spotlight_decorations")
                or tool_args.get("decoration")
            )
            if decorations:
                shot_assets = state.setdefault("shot_assets", {})
                for sid, item in decorations.items():
                    sid = str(sid)
                    if not isinstance(item, dict):
                        continue
                    assets = shot_assets.setdefault(sid, {})
                    assets["decoration"] = item.get("decoration", item)
                    assets["decoration_verify"] = item.get("verify_result", {})
                return {
                    "status": "success",
                    "decorated_count": len(decorations),
                    "results": decorations,
                }

        if tool_name == "Tool_SeverityRank":
            top_shots = self._parse_jsonish_agent_arg(
                tool_args.get("top_shots") or tool_args.get("ranked_shots") or []
            )
            if not isinstance(top_shots, list):
                top_shots = []
            scores = {}
            for idx, item in enumerate(top_shots):
                sid = str(item.get("shot_id") if isinstance(item, dict) else item)
                if sid:
                    scores[sid] = {"score": max(0.0, 1.0 - idx * 0.05), "in_top_ratio": True}
            return {
                "status": "success",
                "ranked_count": len(scores),
                "top_shots": top_shots,
                "severity_scores": scores,
            }

        if tool_name == "Tool_Stage4DimensionJudge":
            judged = self._coerce_jsonish_mapping(
                tool_args.get("shots") or tool_args.get("results") or tool_args.get("judgements")
            )
            if judged:
                shot_assets = state.setdefault("shot_assets", {})
                failed = []
                for sid, info in judged.items():
                    sid = str(sid)
                    if not isinstance(info, dict):
                        continue
                    passed = bool(info.get("passed"))
                    if not passed:
                        failed.append(sid)
                    assets = shot_assets.setdefault(sid, {})
                    assets["approved"] = passed
                    assets["judge_result"] = info
                return {
                    "status": "success" if not failed else "completed_with_warnings",
                    "shots": judged,
                    "shot_count": len(judged),
                    "passed_count": len(judged) - len(failed),
                    "failed_count": len(failed),
                    "failed_shot_ids": failed,
                }
            if tool_args.get("overall_passed") is not None or tool_args.get("summary") or tool_args.get("passed"):
                self.quality_checked = True
                self.last_judge = {
                    "overall_passed": bool(tool_args.get("overall_passed", tool_args.get("passed", True))),
                    "summary": str(tool_args.get("summary") or ""),
                }
                return {
                    "status": "success",
                    "shots": {},
                    "shot_count": 0,
                    "passed_count": 0,
                    "failed_count": 0,
                    "failed_shot_ids": [],
                }

        if tool_name == "Tool_Stage4DimensionalRepair":
            repaired = self._coerce_jsonish_mapping(
                tool_args.get("shots") or tool_args.get("results") or tool_args.get("repairs")
            )
            if repaired:
                shot_assets = state.setdefault("shot_assets", {})
                for sid, info in repaired.items():
                    sid = str(sid)
                    if not isinstance(info, dict):
                        continue
                    assets = shot_assets.setdefault(sid, {})
                    assets["repair_result"] = info
                    if info.get("render_success", True):
                        assets["approved"] = True
                return {
                    "status": "success",
                    "shots": repaired,
                    "shot_count": len(repaired),
                    "render_success_count": sum(
                        1 for v in repaired.values()
                        if isinstance(v, dict) and v.get("render_success", True)
                    ),
                    "failed_count": 0,
                }
            target_ids = tool_args.get("target_shot_ids") or tool_args.get("shot_ids") or []
            if isinstance(target_ids, str):
                target_ids = self._parse_jsonish_agent_arg(target_ids)
            if isinstance(target_ids, list) or tool_args.get("repair_strategy") or tool_args.get("summary"):
                self.repaired_done = True
                return {
                    "status": "success",
                    "repair_strategy": tool_args.get("repair_strategy", "generic"),
                    "target_shot_ids": target_ids if isinstance(target_ids, list) else [],
                    "render_success_count": int(tool_args.get("render_success_count") or len(target_ids) if isinstance(target_ids, list) else 0),
                    "failed_count": int(tool_args.get("failed_count") or 0),
                    "summary": str(tool_args.get("summary") or tool_args.get("repair_details") or ""),
                }

        # v4 light-action orchestrator: model only outputs control params,
        # NOT full content. Return None to fall through to pipeline tools
        # which call Generator/Critic models for actual content generation.
        return None

    @staticmethod
    def _merge_shot_chunks(existing: list[dict[str, Any]], new_shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_id: "dict[str, dict[str, Any]]" = OrderedDict()
        for shot in existing + [s for s in new_shots if isinstance(s, dict)]:
            sid = str(shot.get("shot_id") or f"shot_{len(by_id) + 1:03d}")
            merged = dict(by_id.get(sid, {}))
            merged.update(copy.deepcopy(shot))
            merged["shot_id"] = sid
            by_id[sid] = merged
        return list(by_id.values())

    def _apply_plan_patch(self, shot_plan: list[dict[str, Any]], patch: dict[str, Any]) -> list[dict[str, Any]]:
        patched = [copy.deepcopy(s) for s in shot_plan if isinstance(s, dict)]
        by_id = {str(s.get("shot_id")): s for s in patched}
        for sid, changes in patch.items():
            if str(sid) in by_id and isinstance(changes, dict):
                by_id[str(sid)].update({k: v for k, v in changes.items() if not str(k).startswith("_")})
        return patched

    async def run_agentic_pipeline(self, input_script: str, run_id: str, output_dir: str) -> dict[str, Any]:
        """Execute the long-horizon autonomous agent loop."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from pipeline.graph import _make_project_id

        project_id = _make_project_id()
        project_dir = Path(output_dir) / project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        state: dict[str, Any] = {
            "mode": "agent_long_horizon",
            "project_id": project_id,
            "output_dir": str(project_dir),
            "config": self.pipeline_config,
            "input_text": input_script,
            "segments": [],
            "shot_plan": [],
            "shot_assets": {},
            "bgm_tracks": [],
            "severity_scores": {},
            "severity_ranked_shots": [],
            "current_stage": "agent_loop",
            "plan_approved": False,
            "reflect_round": 0,
            "errors": [],
            "trajectory": [],
        }

        results: dict[str, Any] = {
            "project_id": project_id,
            "output_dir": str(project_dir),
            "stages_completed": [],
            "final_video_path": "",
            "errors": [],
            "agent_mode": self.agent_mode,
        }

        trace_path = project_dir / "agent_trace.jsonl"

        await self.broadcast(run_id, {
            "type": "stage_update",
            "stage": "agent_loop",
            "status": "started",
            "message": f"Agent 自主循环启动: {project_id}",
            "allowed_tools": LONG_HORIZON_ACTION_SPACE,
            "progress_pct": 0,
        })

        same_tool_count = 0
        last_tool = ""
        parse_fail_count = 0

        for step_idx in range(1, self.max_agent_steps + 1):
            await self._wait_if_paused(run_id)
            user_state_json = self.build_agentic_state_json(input_script, state)
            if self._is_multi_model:
                prediction = await asyncio.to_thread(
                    self.model.predict_orchestrator,
                    self.system_prompt,
                    user_state_json,
                    stage="agent_loop",
                    allowed_tools=LONG_HORIZON_ACTION_SPACE,
                )
            else:
                prediction = await asyncio.to_thread(
                    self.model.predict,
                    self.system_prompt,
                    user_state_json,
                    stage="agent_loop",
                    allowed_tools=LONG_HORIZON_ACTION_SPACE,
                )
            await self._wait_if_paused(run_id)

            tool_name = prediction.tool_name
            tool_name = LONG_HORIZON_TOOL_NAME_ALIASES.get(tool_name, tool_name)
            tool_args = prediction.arguments if isinstance(prediction.arguments, dict) else {}

            # Validate tool name against known action space
            if tool_name and tool_name not in LONG_HORIZON_ACTION_SPACE:
                logger.warning(
                    "[agent_loop] Invalid tool name '%s' not in action space, treating as parse failure",
                    tool_name,
                )
                prediction.parse_ok = False

            phase = self._phase_for_tool(tool_name)
            agent_state = prediction.agent_trace.get("agent_state", {}) if isinstance(prediction.agent_trace, dict) else {}
            if isinstance(agent_state, dict) and agent_state.get("phase"):
                phase = str(agent_state.get("phase"))

            await self.broadcast(run_id, {
                "type": "stage_update",
                "stage": phase,
                "status": "started",
                "message": LONG_HORIZON_PHASE_LABELS.get(phase, phase),
                "allowed_tools": LONG_HORIZON_ACTION_SPACE,
                "progress_pct": self._agent_progress_pct(state, phase),
            })
            thinking_event = self._thinking_event(stage=phase, prediction=prediction, step=step_idx)
            thinking_event["tool_name"] = tool_name
            await self.broadcast(run_id, thinking_event)

            if not prediction.parse_ok:
                parse_fail_count += 1
                if parse_fail_count <= 3:
                    logger.warning(
                        "[agent_loop] parse fail step=%d (attempt %d). raw=%s",
                        step_idx, parse_fail_count, prediction.raw_output[:300],
                    )
                    # Use _next_missing_agent_tool as an intelligent fallback
                    fallback_tool = self._next_missing_agent_tool(state)
                    if fallback_tool:
                        logger.info("[agent_loop] Using fallback tool: %s", fallback_tool)
                        tool_name = fallback_tool
                        tool_args = {}
                        phase = self._phase_for_tool(tool_name)
                        await self.broadcast(run_id, {
                            "type": "console_log",
                            "text": f"解析失败 step={step_idx}，自动推进到: {tool_name} ({parse_fail_count}/3)",
                            "level": "warn",
                        })
                        # Don't continue — fall through to tool execution below
                    else:
                        await self.broadcast(run_id, {
                            "type": "console_log",
                            "text": f"解析失败 step={step_idx}，重试中 ({parse_fail_count}/3)",
                            "level": "warn",
                        })
                        continue
                else:
                    err = f"模型输出解析失败 step={step_idx}: {prediction.cleaned_output[:240]}"
                    logger.error(err)
                    results["errors"].append(err)
                    await self.broadcast(run_id, {"type": "error", "stage": phase, "message": err})
                    break

            if tool_name == last_tool:
                same_tool_count += 1
            else:
                same_tool_count = 1
                last_tool = tool_name

            _REPAIR_TOOLS_MAX_1 = {
                "Tool_Stage4DimensionalRepair",
                "Tool_Stage4DimensionJudge",
                "Tool_SelfReflect_Patch",
            }
            if same_tool_count > 1 and tool_name in _REPAIR_TOOLS_MAX_1:
                self.repaired_done = True
                self.quality_checked = True
                await self.broadcast(run_id, {
                    "type": "console_log",
                    "text": f"{tool_name} 已重试 {same_tool_count} 次，跳过修复直接进入最终合成",
                    "level": "warn",
                })
                tool_name = "Tool_StitchVideo"
                same_tool_count = 1
                last_tool = tool_name

            # Tool_PlanShots iterates ALL segments internally in one call.
            # If model repeats it, skip entirely (no dispatch, no trace).
            _SINGLE_DISPATCH_ITERATING_TOOLS = {"Tool_PlanShots"}
            if same_tool_count > 1 and tool_name in _SINGLE_DISPATCH_ITERATING_TOOLS:
                next_tool = self._next_missing_agent_tool(state)
                if next_tool != tool_name:
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": f"{tool_name} 内部已完成所有章节迭代，跳过重复调用，推进到 {next_tool}",
                        "level": "stdout",
                    })
                    tool_name = next_tool
                    tool_args = {}
                    phase = self._phase_for_tool(tool_name)
                    same_tool_count = 1
                    last_tool = tool_name
                else:
                    if self._all_segments_planned(state):
                        await self.broadcast(run_id, {
                            "type": "console_log",
                            "text": f"{tool_name} 所有章节已规划完毕，跳过",
                            "level": "stdout",
                        })
                        continue
                    # Still not fully planned: keep dispatching PlanShots so remaining
                    # segments can be completed instead of stalling in a skip loop.
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": (
                            f"{tool_name} 仍有未规划章节，继续执行本轮规划"
                        ),
                        "level": "warn",
                    })
                    same_tool_count = 1
                    last_tool = tool_name

            # Tool_SelfReflect can be legitimately called twice (post-plan + post-decoration)
            # but not more than that.
            _MAX_2_TOOLS = {"Tool_SelfReflect"}
            if same_tool_count > 2 and tool_name in _MAX_2_TOOLS:
                next_tool = self._next_missing_agent_tool(state)
                if next_tool and next_tool != tool_name:
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": f"{tool_name} 已执行 {same_tool_count} 次（规划+装饰后），推进到 {next_tool}",
                        "level": "stdout",
                    })
                    tool_name = next_tool
                    tool_args = {}
                    phase = self._phase_for_tool(tool_name)
                    same_tool_count = 1
                    last_tool = tool_name

            already_completed = (
                self._agent_tool_already_done(tool_name, state)
                and tool_name != "Tool_StitchVideo"
            )
            already_completed_blocked = False
            if already_completed and same_tool_count > 2:
                next_tool = self._next_missing_agent_tool(state)
                if next_tool and next_tool != tool_name:
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": f"重复动作重定向: {tool_name} 已完成，推进到 {next_tool}",
                        "level": "warn",
                    })
                    tool_name = next_tool
                    tool_args = {}
                    phase = self._phase_for_tool(tool_name)
                    same_tool_count = 1
                    last_tool = tool_name
                    await self.broadcast(run_id, {
                        "type": "stage_update",
                        "stage": phase,
                        "status": "started",
                        "message": LONG_HORIZON_PHASE_LABELS.get(phase, phase),
                        "allowed_tools": LONG_HORIZON_ACTION_SPACE,
                        "progress_pct": self._agent_progress_pct(state, phase),
                    })
                    already_completed = False
            elif already_completed:
                already_completed_blocked = True

            if same_tool_count > self.max_same_tool:
                err = f"模型连续 {same_tool_count} 次选择 {tool_name}，疑似循环，停止执行"
                results["errors"].append(err)
                await self.broadcast(run_id, {"type": "error", "stage": phase, "message": err})
                break

            stitch_missing = self._agent_precondition_missing("Tool_StitchVideo", state)
            if not stitch_missing and tool_name != "Tool_StitchVideo" and tool_name not in {
                "Tool_Stage4DimensionJudge", "Tool_Stage4DimensionalRepair",
            }:
                await self.broadcast(run_id, {
                    "type": "console_log",
                    "text": f"所有前置条件已满足，强制重定向 {tool_name} → Tool_StitchVideo",
                    "level": "warn",
                })
                tool_name = "Tool_StitchVideo"
                tool_args = {
                    "project_id": state.get("project_id", project_id),
                    "timeline_path": state.get("timeline_path", ""),
                    "bgm": state.get("bgm_result", {}),
                }
                phase = "final_stitch"

            missing = (
                ["already_completed"]
                if already_completed_blocked
                else self._agent_precondition_missing(tool_name, state)
            )
            await self.broadcast(run_id, {
                "type": "tool_call",
                "stage": phase,
                "step": step_idx,
                "tool_name": tool_name,
                "arguments": tool_args,
                "precondition_missing": missing,
            })

            if missing:
                next_tool = self._next_missing_agent_tool(state)
                if next_tool and next_tool != tool_name:
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": f"前置条件阻止: {tool_name} (缺 {', '.join(missing)})，自动重定向到 {next_tool}",
                        "level": "warn",
                    })
                    tool_name = next_tool
                    tool_args = {}
                    phase = self._phase_for_tool(tool_name)
                    same_tool_count = 1
                    last_tool = tool_name
                    missing = self._agent_precondition_missing(tool_name, state)

                if missing:
                    tool_output = {
                        "status": "blocked_precondition",
                        "missing": missing,
                        "message": f"{tool_name} 前置条件不足: {', '.join(missing)}",
                    }
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": f"前置条件阻止: {tool_name} (缺 {', '.join(missing)})",
                        "level": "warn",
                    })
                else:
                    await self._wait_if_paused(run_id)
                    tool_output = await self._dispatch_tool(tool_name, tool_args, state, run_id)
                    if isinstance(tool_output, dict):
                        self.update_state_from_respond(tool_name, tool_output)
                        self._apply_tool_output(tool_name, tool_output, state)
                    else:
                        tool_output = {"status": "done"}
            else:
                await self._wait_if_paused(run_id)
                tool_output = await self._dispatch_tool(tool_name, tool_args, state, run_id)
                if isinstance(tool_output, dict):
                    self.update_state_from_respond(tool_name, tool_output)
                    self._apply_tool_output(tool_name, tool_output, state)
                else:
                    tool_output = {"status": "done"}

            self._record_agent_history(tool_name, tool_output)
            results["stages_completed"].append(f"{step_idx:02d}:{tool_name}")

            self._append_trace(trace_path, {
                "step": step_idx,
                "tool_name": tool_name,
                "arguments": tool_args,
                "output": tool_output if isinstance(tool_output, dict) else {"status": "done"},
                "raw_prediction": prediction.raw_output[:2000] if hasattr(prediction, "raw_output") else "",
                "completion_tokens": getattr(prediction, "completion_tokens", 0),
                "timestamp": datetime.now().isoformat(),
            })

            await self.broadcast(run_id, {
                "type": "tool_result",
                "stage": phase,
                "step": step_idx,
                "tool_name": tool_name,
                "status": tool_output.get("status", "unknown") if isinstance(tool_output, dict) else "done",
                "observation": self.last_observation,
            })

            await self.broadcast(run_id, {
                "type": "preview_update",
                "stage": phase,
                "preview": self._build_preview(state),
            })
            await self.broadcast(run_id, {
                "type": "stage_update",
                "stage": phase,
                "status": "completed",
                "message": f"{LONG_HORIZON_PHASE_LABELS.get(phase, phase)} 完成一步",
                "progress_pct": self._agent_progress_pct(state, phase),
            })

            if tool_name == "Tool_StitchVideo" and not missing:
                final_path = state.get("final_video_path", "")
                if final_path:
                    self.stitched_done = True
                else:
                    stitch_status = tool_output.get("concat_status", "") if isinstance(tool_output, dict) else ""
                    stitch_errors = tool_output.get("validation_errors", []) if isinstance(tool_output, dict) else []
                    err_detail = f"拼接验证失败 (concat_status={stitch_status})"
                    if stitch_errors:
                        reasons = [
                            f"{e.get('shot_id','?')}: {','.join(e.get('reasons',[]))}"
                            for e in stitch_errors[:5] if isinstance(e, dict)
                        ]
                        err_detail += f" 失败镜头: {'; '.join(reasons)}"
                    results["errors"].append(err_detail)
                    await self.broadcast(run_id, {
                        "type": "error",
                        "stage": "final_stitch",
                        "message": err_detail,
                    })
                break

        else:
            err = f"达到 max_agent_steps={self.max_agent_steps}，Agent 未完成 StitchVideo"
            results["errors"].append(err)
            await self.broadcast(run_id, {"type": "error", "stage": "agent_loop", "message": err})

        results["final_video_path"] = state.get("final_video_path", "")
        if self.sw_stitch and not results["final_video_path"] and not results["errors"]:
            stitch = state.get("stitch_result", {}) if isinstance(state.get("stitch_result"), dict) else {}
            results["errors"].append(
                "最终视频未生成"
                f" (concat_status={stitch.get('concat_status', 'unknown')}, "
                f"ready={stitch.get('ready_shot_count', 0)}/{stitch.get('shot_count', 0)})"
            )

        # 落盘最终状态快照 —— 批量评测 / 事后审计用。
        #
        # agent_trace.jsonl 记的是"每一步做了什么", 但 segments / shot_plan /
        # shot_assets / stitch_result 这些**最终产物**原来只存在于内存里的 state,
        # 返回的 results 只带了 final_video_path。任何离线打分 (Table2 的
        # SF/SEG/PTF 都要读 shot_plan 和 timeline) 都拿不到, 只能去反解 trace。
        # 这里额外写一个文件, 纯增量, 不影响任何既有流程; 写失败也只记 warning,
        # 绝不能因为落盘问题把一次成功的推理判成失败。
        try:
            (project_dir / "final_state.json").write_text(
                json.dumps({
                    "project_id": project_id,
                    "input_text": input_script,
                    "segments": state.get("segments", []),
                    "shot_plan": state.get("shot_plan", []),
                    "shot_assets": state.get("shot_assets", {}),
                    "stitch_result": state.get("stitch_result", {}),
                    "bgm_tracks": state.get("bgm_tracks", []),
                    "final_video_path": state.get("final_video_path", ""),
                    "stages_completed": results.get("stages_completed", []),
                    "errors": results.get("errors", []),
                    "agent_mode": self.agent_mode,
                    "saved_at": datetime.now().isoformat(),
                }, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("final_state.json 落盘失败 (不影响推理结果): %s", exc)

        await self.broadcast(run_id, {
            "type": "pipeline_complete",
            "status": "completed" if not results["errors"] else "completed_with_errors",
            "results": results,
            "preview": self._build_preview(state),
        })
        return results

    def _agent_progress_pct(self, state: dict[str, Any], phase: str = "") -> int:
        flags = self._agent_flags(state)
        keys = ["segmented", "planned", "visual_audio_done", "decorated", "timeline_done", "bgm_done", "quality_checked", "stitched"]
        done = sum(1 for key in keys if flags.get(key))
        return min(100, int(round(done * 100 / len(keys))))

    async def run_pipeline(self, input_script: str, run_id: str, output_dir: str) -> dict[str, Any]:
        """Execute the full inference pipeline."""
        if self.is_long_horizon_agent():
            return await self.run_agentic_pipeline(input_script, run_id, output_dir)

        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

        from pipeline.graph import (
            _get_llm, _cfg_get, _cfg_int, _cfg_float, _cfg_bool,
            _make_project_id, _is_fake_mode,
        )
        from pipeline.tools import (
            segment_article, plan_shots, self_reflect,
            retrieve_visual, generate_audio, decorate_shot,
            music_search, timeline_compose, text_quick_judge,
            vlm_render_judge, stitch_video,
        )
        from pipeline.llm import LLMClient

        project_id = _make_project_id()
        project_dir = Path(output_dir) / project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        # Initialize pipeline state dict
        state: dict[str, Any] = {
            "mode": "inference",
            "project_id": project_id,
            "output_dir": str(project_dir),
            "config": self.pipeline_config,
            "input_text": input_script,
            "segments": [],
            "shot_plan": [],
            "shot_assets": {},
            "bgm_tracks": [],
            "severity_scores": {},
            "severity_ranked_shots": [],
            "current_stage": "init",
            "plan_approved": False,
            "reflect_round": 0,
            "errors": [],
            "trajectory": [],
        }

        await self.broadcast(run_id, {
            "type": "stage_update",
            "stage": "init",
            "status": "started",
            "message": f"项目初始化: {project_id}",
        })

        results: dict[str, Any] = {
            "project_id": project_id,
            "output_dir": str(project_dir),
            "stages_completed": [],
            "final_video_path": "",
            "errors": [],
        }

        # Execute through stages
        for stage in STAGE_SEQUENCE:
            allowed_tools = list(self.allowed_map.get(stage, []))
            if not allowed_tools:
                continue

            state["current_stage"] = stage

            await self.broadcast(run_id, {
                "type": "stage_update",
                "stage": stage,
                "status": "started",
                "message": f"进入阶段: {stage}",
                "allowed_tools": allowed_tools,
            })

            try:
                # Pause / cancel check
                await self._wait_if_paused(run_id)

                user_state_json = self.build_state_json(stage, input_script)
                if self._is_multi_model:
                    prediction = await asyncio.to_thread(
                        self.model.predict_orchestrator,
                        self.system_prompt,
                        user_state_json,
                        stage=stage,
                        allowed_tools=allowed_tools,
                    )
                else:
                    prediction = await asyncio.to_thread(
                        self.model.predict,
                        self.system_prompt,
                        user_state_json,
                        stage=stage,
                        allowed_tools=allowed_tools,
                    )
                await self._wait_if_paused(run_id)

                await self.broadcast(run_id, self._thinking_event(stage=stage, prediction=prediction))

                if not prediction.parse_ok:
                    err = f"模型输出解析失败 at {stage}: {prediction.cleaned_output[:200]}"
                    logger.error(err)
                    results["errors"].append(err)
                    await self.broadcast(run_id, {
                        "type": "error",
                        "stage": stage,
                        "message": err,
                    })
                    continue

                tool_name = prediction.tool_name
                tool_name = LONG_HORIZON_TOOL_NAME_ALIASES.get(tool_name, tool_name)
                if tool_name and tool_name not in LONG_HORIZON_ACTION_SPACE:
                    logger.warning("[stage_loop] Invalid tool '%s', skipping", tool_name)
                    results["errors"].append(f"无效工具名: {tool_name}")
                    continue
                tool_args = prediction.arguments
                forced_tool = self._force_tool_for_stage(stage, tool_name, state)
                if forced_tool != tool_name:
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": f"路由校正: {stage} 将 {tool_name} 调整为 {forced_tool}",
                        "level": "warn",
                    })
                    tool_name = forced_tool
                    tool_args = {}

                await self.broadcast(run_id, {
                    "type": "tool_call",
                    "stage": stage,
                    "tool_name": tool_name,
                    "arguments": tool_args,
                })

                # Pause / cancel check before dispatch
                await self._wait_if_paused(run_id)

                # Dispatch to real tool
                tool_output = await self._dispatch_tool(
                    tool_name, tool_args, state, run_id,
                )

                # Update state accumulator
                if isinstance(tool_output, dict):
                    self.update_state_from_respond(tool_name, tool_output)

                    # Update pipeline state from tool output
                    self._apply_tool_output(tool_name, tool_output, state)

                await self.broadcast(run_id, {
                    "type": "tool_result",
                    "stage": stage,
                    "tool_name": tool_name,
                    "status": tool_output.get("status", "unknown") if isinstance(tool_output, dict) else "done",
                })

                if (
                    tool_name == "Tool_SelfReflect"
                    and isinstance(tool_output, dict)
                    and tool_output.get("status") in ("repair_exhausted", "needs_fix")
                ):
                    warn_msg = (
                        f"自检未完全通过 ({tool_output.get('status')}): "
                        f"{tool_output.get('error') or '仍有遗留问题，但继续执行后续阶段'}"
                    )
                    results["errors"].append(warn_msg)
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "stage": stage,
                        "text": f"⚠ {warn_msg}",
                        "level": "warn",
                    })

                results["stages_completed"].append(stage)

                # Handle multi-tool stages: after segmentation, plan shots for ALL segments
                if stage == "stage2_planning" and tool_name == "Tool_SegmentArticle":
                    segments = state.get("segments") or []
                    for seg_iter in segments:
                        seg_iter_id = seg_iter.get("segment_id", "") if isinstance(seg_iter, dict) else ""
                        if not seg_iter_id:
                            continue
                        plan_args = {
                            "segment_id": seg_iter_id,
                            "num_shots_target": 3,
                        }
                        await self.broadcast(run_id, {
                            "type": "console_log",
                            "text": f"为章节 {seg_iter_id} 规划镜头...",
                            "level": "stdout",
                        })
                        await self.broadcast(run_id, {
                            "type": "tool_call",
                            "stage": stage,
                            "tool_name": "Tool_PlanShots",
                            "arguments": plan_args,
                        })
                        await self._wait_if_paused(run_id)
                        output2 = await self._dispatch_tool(
                            "Tool_PlanShots", plan_args, state, run_id,
                        )
                        if isinstance(output2, dict):
                            self.update_state_from_respond("Tool_PlanShots", output2)
                            self._apply_tool_output("Tool_PlanShots", output2, state)
                        await self.broadcast(run_id, {
                            "type": "tool_result",
                            "stage": stage,
                            "tool_name": "Tool_PlanShots",
                            "status": output2.get("status", "unknown") if isinstance(output2, dict) else "done",
                        })

                # Send preview update after each stage
                await self.broadcast(run_id, {
                    "type": "preview_update",
                    "stage": stage,
                    "preview": self._build_preview(state),
                })

                await self.broadcast(run_id, {
                    "type": "stage_update",
                    "stage": stage,
                    "status": "completed",
                    "message": f"阶段完成: {stage}",
                })

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                err_msg = f"[{stage}] {type(exc).__name__}: {str(exc)[:300]}"
                logger.exception("Pipeline error at %s", stage)
                results["errors"].append(err_msg)
                await self.broadcast(run_id, {
                    "type": "error",
                    "stage": stage,
                    "message": err_msg,
                })

        results["final_video_path"] = state.get("final_video_path", "")
        if self.sw_stitch and not results["final_video_path"] and not results["errors"]:
            stitch = state.get("stitch_result", {}) if isinstance(state.get("stitch_result"), dict) else {}
            results["errors"].append(
                "最终视频未生成"
                f" (concat_status={stitch.get('concat_status', 'unknown')}, "
                f"ready={stitch.get('ready_shot_count', 0)}/{stitch.get('shot_count', 0)})"
            )

        await self.broadcast(run_id, {
            "type": "pipeline_complete",
            "status": "completed" if not results["errors"] else "completed_with_errors",
            "results": results,
            "preview": self._build_preview(state),
        })

        return results

    # ─── Agent trace disk logging ──────────────────────────────────────────

    @staticmethod
    def _append_trace(trace_path: Path, record: dict[str, Any]) -> None:
        """Append one JSON record to the agent_trace.jsonl file."""
        try:
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to write trace: %s", exc)

    # ─── Skill JSON parser (robust to <tool_call> wrapping / think blocks) ──

    # Slot names in registry/structures.json are bare strings, so the only
    # signal about what belongs in a slot is its name. Filling every slot as
    # generic 50-char text makes the model truncate a narration sentence into
    # `year` or `emoji_icon`; these buckets give it the semantic contract the
    # template actually expects.
    #
    # Matching is on the name with any trailing index stripped (`pro1` -> `pro`)
    # rather than on substrings, because `con` would otherwise swallow
    # `concept_desc` and label a paragraph slot as a 12-char chip.
    _SLOT_LIST_BASES = frozenset({
        "nodes", "items", "branches", "cases", "columns", "edges",
        "metrics", "modules", "rows",
    })
    _SLOT_SHORT_BASES: dict[str, tuple[int, str]] = {
        "year": (12, "年份或时间点，如「1969」「上世纪 80 年代」"),
        "person_name": (12, "人名或主体名称"),
        "chapter_no": (6, "章节序号，如「01」「第三章」"),
        "ratio_text": (12, "比例或数值，如「73%」「3 : 1」"),
        "unit": (8, "计量单位，如「亿元」「km/h」"),
        "source": (16, "信息来源"),
        "speaker": (12, "说话人"),
    }
    _SLOT_LABEL_BASES = frozenset({
        "eyebrow", "era_tag", "series", "term", "tag", "keyword", "layer",
        "item", "pro", "con", "concept_name", "step",
    })
    _SLOT_HEADLINE_BASES = frozenset({"headline", "subhead"})
    _SLOT_NOTE_BASES = frozenset({
        "takeaway", "answer_hint", "source_note", "chapter_sub",
    })

    @staticmethod
    def _slot_spec_from_name(name: str) -> dict[str, Any]:
        """Infer a slot's content contract from its name."""
        import re as _re

        key = str(name or "").strip().lower()
        base = _re.sub(r"\d+$", "", key)

        def spec(stype: str, limit: int, hint: str) -> dict[str, Any]:
            return {"name": name, "type": stype, "max_chars": limit, "hint": hint}

        if key in InferenceOrchestrator._SLOT_LIST_BASES or base in InferenceOrchestrator._SLOT_LIST_BASES:
            return {
                "name": name, "type": "list", "max_items": 4,
                "max_chars_per_item": 12, "hint": "若干并列短词条",
            }
        if "emoji" in key:
            return spec("emoji", 2, "单个 emoji，不要文字")
        short = InferenceOrchestrator._SLOT_SHORT_BASES.get(base)
        if short is not None:
            return spec("short_text", short[0], short[1])
        if key.endswith("_value"):
            return spec("short_text", 12, "数值或短结论")
        if key.endswith("_label") or base in InferenceOrchestrator._SLOT_LABEL_BASES:
            return spec("label", 12, "短词组标签，不要成句")
        if key.endswith("_title") or base in InferenceOrchestrator._SLOT_HEADLINE_BASES:
            return spec("headline", 16, "主标题，一句话讲清重点")
        if base in InferenceOrchestrator._SLOT_NOTE_BASES:
            return spec("text", 24, "一行补充说明")
        # Everything left (*_text, *_desc, definition, ...) is body copy.
        return spec("text", 40, "正文短句，来自口播稿，不要整句照搬")

    @staticmethod
    def _load_template_registry(config: dict[str, Any]) -> dict[str, Any]:
        """Load gold template registry for two-step decoration.

        Returns {"candidates": [...], "slot_schemas": {template_id: {...}}}.
        """
        result: dict[str, Any] = {"candidates": [], "slot_schemas": {}}
        try:
            gold_root = Path(
                str((config.get("paths", {}) or {}).get("gold_template_root", ""))
            ).expanduser().resolve()
            structures_path = gold_root / "registry" / "structures.json"
            if not structures_path.exists():
                return result
            data = json.loads(structures_path.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else data.get("structures", data.get("templates", []))
            candidates = []
            slot_schemas: dict[str, dict[str, Any]] = {}
            for item in items:
                if not isinstance(item, dict):
                    continue
                tid = str(item.get("template_id", item.get("id", "")))
                if not tid:
                    continue
                # The registry carries far more selection signal than the four
                # fields that used to be forwarded. Without `use_for` and
                # `family` the model was picking from 90 near-identical looking
                # ids, which is why whole families never got chosen.
                candidates.append({
                    "template_id": tid,
                    "name": item.get("display_name", item.get("name", "")),
                    "category": item.get("category", ""),
                    "family": item.get("family", ""),
                    "use_for": item.get("use_for", []),
                    "keywords": item.get("keywords", item.get("tags", [])),
                    "spotlight_level": item.get("spotlight_level", ""),
                    "duration_range": item.get("duration_range", []),
                })

                # Prefer the generated slot_schema.json: it carries the real
                # labels, defaults and max_chars per slot, whereas the registry
                # only lists slot names and the fallback has to guess.
                schema = InferenceOrchestrator._load_slot_schema_file(gold_root, tid)
                if schema:
                    slot_schemas[tid] = schema
                    continue
                slots_raw = item.get("slots", item.get("slot_schema", {}))
                if isinstance(slots_raw, dict):
                    slot_schemas[tid] = slots_raw
                elif isinstance(slots_raw, list):
                    parsed: dict[str, Any] = {}
                    for i, s in enumerate(slots_raw):
                        if isinstance(s, dict):
                            parsed[s.get("name", f"slot_{i}")] = s
                        elif isinstance(s, str):
                            parsed[s] = InferenceOrchestrator._slot_spec_from_name(s)
                    slot_schemas[tid] = parsed
            result["candidates"] = candidates
            result["slot_schemas"] = slot_schemas
        except Exception as exc:
            logger.warning("[_load_template_registry] failed: %s", exc)
        return result

    @staticmethod
    def _nearest_template_id(
        wanted: str, valid: set[str], already_used: list[str]
    ) -> str:
        """Best real template for an id the model invented.

        Fuzzy match first — a hallucination is usually a near-miss of a real id
        such as ``stat_highlight_card`` for ``stat_highlight`` — and only fall
        back to an unused generic when there is nothing close.
        """
        import difflib

        close = difflib.get_close_matches(wanted, sorted(valid), n=3, cutoff=0.6)
        for cand in close:
            if cand not in already_used:
                return cand
        if close:
            return close[0]
        for generic in ("key_takeaway", "definition_card", "thesis_statement",
                        "question_hook", "timeline_step"):
            if generic in valid and generic not in already_used:
                return generic
        unused = sorted(valid - set(already_used))
        return unused[0] if unused else (sorted(valid)[0] if valid else wanted)

    @staticmethod
    def _load_slot_schema_file(gold_root: Path, template_id: str) -> dict[str, Any]:
        """Per-slot spec from ``generated/structures/<id>/slot_schema.json``."""
        path = gold_root / "generated" / "structures" / template_id / "slot_schema.json"
        try:
            if not path.is_file():
                return {}
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        slots = payload.get("slots") if isinstance(payload, dict) else None
        return slots if isinstance(slots, dict) else {}

    @staticmethod
    def _clamp_slot_text(slots: dict[str, Any], slot_schema: dict[str, Any]) -> dict[str, Any]:
        """Enforce each slot's ``max_chars``.

        Overlong slot text does not wrap in the templates — it overflows its box
        or gets clipped mid-character at render time, so it is cut here where
        the limit is actually known.
        """
        out: dict[str, Any] = {}
        for name, value in (slots or {}).items():
            text = value if isinstance(value, str) else value
            spec = slot_schema.get(name) if isinstance(slot_schema, dict) else None
            limit = 0
            if isinstance(spec, dict):
                try:
                    limit = int(spec.get("max_chars", 0) or 0)
                except (TypeError, ValueError):
                    limit = 0
            if isinstance(text, str) and limit > 0 and len(text) > limit:
                text = text[:limit].rstrip("，。、；：,;: ")
            out[name] = text
        return out

    @staticmethod
    def _parse_skill_json(text: str) -> dict[str, Any]:
        """Parse Generator/Critic output, stripping any <think> or <tool_call> wrappers."""
        import re as _re
        text = str(text or "").strip()
        text = _re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
        text = _re.sub(r"</?tool_call>", "", text).strip()
        text = _re.sub(r"```(?:json)?\s*\n?", "", text).strip()
        text = _re.sub(r"```\s*$", "", text).strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError):
            pass
        start = text.find("{")
        if start < 0:
            return {}
        depth = 0
        in_str = False
        esc = False
        best_end = -1
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

    # ─── Training-format payload builders for Generator/Critic ────────────

    @staticmethod
    def _build_skill_payload_segment_article(
        input_text: str, tool_args: dict[str, Any], max_segments: int,
    ) -> dict[str, Any]:
        """Build AICutSkill user payload for segment_article (matches training exactly)."""
        import re
        text = str(input_text or "")
        sents = [s.strip() for s in re.split(r"(?<=[。！？!?])\s*", text) if s.strip()]
        if not sents:
            sents = [text]
        units = []
        pos = 0
        for i, sent in enumerate(sents):
            start = text.find(sent, pos)
            if start == -1:
                start = pos
            end = start + len(sent)
            units.append({"unit_id": f"u{i+1:02d}", "char_start": start, "char_end": end, "text": sent})
            pos = end

        return {
            "task": "将用户输入的一段口播稿，通过 AICutPipeline 工具链做成完整长视频。",
            "stage_task": "segment_article",
            "input_prior_version": "custom_tool_priors_v2",
            "instruction": "单次处理完整口播稿，按语义连续性拆成结构清晰、互不重叠且覆盖全文的章节/分段，并给出全局意图和配乐风格。",
            "input_script": text,
            "memory": {
                "article_digest": {
                    "char_count": len(text),
                    "sentence_unit_count": len(units),
                    "script_head": text[:200],
                    "script_tail": text[-200:] if len(text) > 200 else text,
                    "source_units": units,
                }
            },
            "constraints": {
                "article_text_length": len(text),
                "max_segments": 10,
                "target_num_segments": max_segments,
                "cost_policy": {
                    "model_call_granularity": "single_call_for_full_article",
                    "do_not_call_per_segment": True,
                },
                "segmentation_policy": {
                    "cover_full_script": True,
                    "preserve_original_order": True,
                    "no_overlap_between_segments": True,
                    "no_missing_major_topic": True,
                    "segment_count_should_match_target": True,
                },
                "boundary_policy": {
                    "each_segment_raw_span_should_be_contiguous": True,
                    "boundary_should_follow_topic_shift_or_rhetorical_turn": True,
                    "avoid_cutting_inside_one_example_or_argument": True,
                },
                "segment_role_prior": [
                    "开场/问题引入", "背景或概念铺垫", "核心机制/方法展开",
                    "案例/步骤/证据", "总结/行动建议/升华",
                ],
                "music_style_policy": {
                    "derive_from_topic_pace_and_emotion": True,
                    "return_searchable_style_phrase": True,
                },
            },
            "output_schema": {
                "stage_task": "segment_article",
                "intent": "string",
                "music_style": "string",
                "segments": [{
                    "segment_id": "seg_01",
                    "structure_label": "引子/核心章节/总结升华",
                    "title": "string",
                    "summary": "string",
                    "key_points": ["string"],
                    "raw_span": "string",
                    "planner_strategy": "string",
                    "narration_script": "string",
                }],
                "num_segments": "integer",
            },
        }

    @staticmethod
    def _plan_shots_tail_shot(shot: dict[str, Any]) -> dict[str, Any]:
        """Trim a planned shot down to the fields the training payload carries."""
        return {
            "shot_id": str(shot.get("shot_id", "")),
            "segment_id": str(shot.get("segment_id", "")),
            "chapter_title": str(shot.get("chapter_title", "")),
            "vocal_script": str(shot.get("vocal_script", "")),
            "visual_query": str(shot.get("visual_query", "")),
            "intent": str(shot.get("intent", "")),
            "is_spotlight": bool(shot.get("is_spotlight")),
        }

    @staticmethod
    def _build_skill_payload_plan_shots(
        state: dict[str, Any],
        target_seg: dict[str, Any],
        *,
        segments: list[dict[str, Any]],
        planned_shots: list[dict[str, Any]],
        chunk_index: int = 1,
        num_chunks: int = 1,
        expected_shots: int = 1,
        total_segment_shots: int = 1,
        total_target_shots: int = 0,
    ) -> dict[str, Any]:
        """Generator payload for one segment/chunk of plan_shots.

        Field-for-field mirror of the unified v6 training rows. The model was
        trained to plan a single segment per call ("只为当前 target_segment/chunk
        生成连续镜头" in its own system prompt), so handing it the whole article
        in one payload puts it off-distribution: it rambles until it runs out of
        tokens and never closes the JSON.
        """
        input_text = str(state.get("input_text", "") or "")
        seg_id = str(target_seg.get("segment_id", "seg_01"))
        seg_idx = next(
            (i for i, s in enumerate(segments)
             if isinstance(s, dict) and s.get("segment_id") == seg_id),
            0,
        )
        prev_seg = segments[seg_idx - 1] if seg_idx > 0 else {}
        next_seg = segments[seg_idx + 1] if seg_idx < len(segments) - 1 else {}
        coverage_units = InferenceOrchestrator._segment_coverage_units(target_seg)

        planned = [s for s in (planned_shots or []) if isinstance(s, dict)]
        same_segment = [s for s in planned if str(s.get("segment_id", "")) == seg_id]
        last_shot_id = str(planned[-1].get("shot_id", "")) if planned else ""
        covered_tail = str(planned[-1].get("vocal_script", "")) if planned else ""

        prefix_summary: dict[str, Any] = {"planned_shot_count": len(planned)}
        if planned:
            prefix_summary["first_shot_id"] = str(planned[0].get("shot_id", ""))
        prefix_summary.update({
            "covered_vocal_tail": covered_tail,
            "last_shot_id": last_shot_id,
            "spotlight_count": sum(1 for s in planned if s.get("is_spotlight")),
        })

        first_id = len(planned) + 1
        tail = InferenceOrchestrator._plan_shots_tail_shot
        return {
            "role": "generator",
            "task": "将用户输入的一段口播稿，通过 AICutPipeline 工具链做成完整长视频。",
            "stage_task": "plan_shots",
            "input_prior_version": "custom_tool_priors_v2",
            "instruction": "只为当前 target_segment/chunk 批量生成多个连续镜头；保持原文顺序、shot_id 连续、visual_query 不重复，并衔接前序 chunk。",
            "input_script": input_text,
            "memory": {
                "intent": state.get("intent", ""),
                "music_style": state.get("music_style", ""),
                "segments": [
                    {k: s.get(k, "") for k in (
                        "segment_id", "structure_label", "title",
                        "summary", "key_points", "planner_strategy",
                    )}
                    for s in segments if isinstance(s, dict)
                ],
                "target_segment": target_seg,
                "prev_segment": prev_seg or {},
                "next_segment": next_seg or {},
                "coverage_units": coverage_units,
                "planned_prefix_summary": prefix_summary,
                "previous_chunk_tail_shots": (
                    [tail(s) for s in same_segment[-2:]] if chunk_index > 1 else []
                ),
                "previous_segment_tail_shots": [
                    tail(s) for s in planned[-2:]
                    if str(s.get("segment_id", "")) != seg_id
                ],
                "visual_query_history": [
                    str(s.get("visual_query", ""))
                    for s in planned if s.get("visual_query")
                ][-10:],
            },
            "constraints": {
                "num_segments": len(segments),
                "num_video_shots": total_target_shots or total_segment_shots,
                "mode": "video_shots",
                "segment_id": seg_id,
                "chunk_index": chunk_index,
                "num_chunks_for_segment": num_chunks,
                "expected_num_shots": expected_shots,
                "total_segment_shots": total_segment_shots,
                "model_call_granularity": "one_call_per_segment_or_chunk",
                "note": "只输出当前 target_segment/chunk 的 shots，不输出其他章节，不逐镜头调用模型。",
                "shot_id_policy": {
                    "first_shot_id": str(first_id),
                    "last_shot_id": str(first_id + max(0, expected_shots - 1)),
                    "preserve_continuous_order": True,
                    "do_not_renumber_previous_shots": True,
                },
                "coverage_policy": {
                    "coverage_source": "memory.target_segment.raw_span 或 narration_script",
                    "cover_units_in_order": True,
                    "each_shot_should_map_to_one_or_more_adjacent_units": True,
                    "avoid_skipping_key_points": True,
                    "avoid_repeating_same_sentence_unit": True,
                },
                "continuity_policy": {
                    "continue_from_previous_chunk_tail": True,
                    "connect_to_prev_and_next_segment_summary": True,
                    "avoid_topic_jump": True,
                    "keep_vocal_script_natural_and_consecutive": True,
                },
                "visual_query_policy": {
                    "query_should_be_concrete_and_searchable": True,
                    "avoid_repeated_visual_query_from_history": True,
                    "match_vocal_script_semantics": True,
                },
                "spotlight_policy": {
                    "use_spotlight_for_key_concept_turning_point_or_summary": True,
                    "avoid_all_shots_becoming_spotlight": True,
                },
                "alignment_policy": {
                    "shots_tile_segment_narration_exactly": True,
                    "last_chunk_last_shot_is_segment_end": True,
                    "last_segment_last_shot_is_article_end": True,
                    "do_not_leave_uncovered_tail": True,
                },
            },
            "output_schema": {
                "stage_task": "plan_shots",
                "segment_id": "seg_01",
                "chapter_title": "string",
                "chunk_index": "integer",
                "num_chunks_for_segment": "integer",
                "shots": [{
                    "shot_id": "string",
                    "segment_id": "seg_01",
                    "chapter_title": "string",
                    "vocal_script": "string",
                    "visual_query": "string",
                    "visual_direction": "string",
                    "intent": "string",
                    "design_rationale": "string",
                    "is_spotlight": "boolean",
                }],
                "num_shots": "integer",
                "total_segment_shots": "integer",
            },
        }

    @staticmethod
    def _build_skill_payload_plan_shots_global(
        state: dict[str, Any],
        tool_args: dict[str, Any],
        segments: list[dict[str, Any]],
        global_units: list[dict[str, Any]],
    ) -> dict[str, Any]:
        input_text = str(state.get("input_text", "") or "")
        existing_shots = state.get("shot_plan", [])
        total_target_shots = int(tool_args.get("total_target_shots") or max(1, int(tool_args.get("num_shots_target") or 6)))
        last_shot_id = int(existing_shots[-1].get("shot_id", "0").replace("shot_", "")) if existing_shots else 0
        cfg = state.get("config", {}) if isinstance(state.get("config"), dict) else {}
        planner_cfg = (
            (cfg.get("stage2_segment", {}) or {}).get("shot_planner", {})
            if isinstance(cfg.get("stage2_segment", {}), dict)
            else {}
        )
        max_ratio = float(planner_cfg.get("max_spotlight_ratio", 0.50) or 0.50)
        segment_anchors = [
            {k: s.get(k, "") for k in ("segment_id", "structure_label", "title", "summary", "key_points", "planner_strategy")}
            for s in segments if isinstance(s, dict)
        ]
        return {
            "task": "将用户输入的一段口播稿，通过 AICutPipeline 工具链做成完整长视频。",
            "stage_task": "plan_shots",
            "input_prior_version": "custom_tool_priors_v2",
            "instruction": "先进行全文连续切镜头，再根据 segment_anchors 回填章节；不要按单章单次输出，优先保证全文 coverage_units 覆盖与镜头节奏自然。",
            "input_script": input_text,
            "memory": {
                "intent": state.get("intent", ""),
                "music_style": state.get("music_style", ""),
                "segments": segment_anchors,
                "segment_anchors": segment_anchors,
                "coverage_units": global_units,
                "planned_prefix_summary": {
                    "planned_shot_count": len(existing_shots),
                    "last_shot_id": str(last_shot_id).zfill(3),
                    "spotlight_count": sum(1 for s in existing_shots if isinstance(s, dict) and s.get("is_spotlight")),
                },
                "visual_query_history": [
                    str(s.get("visual_query", ""))
                    for s in existing_shots if isinstance(s, dict) and s.get("visual_query")
                ][-12:],
            },
            "constraints": {
                "mode": "global_shot_first",
                "num_segments": len(segment_anchors),
                "num_video_shots": total_target_shots,
                "expected_num_shots": total_target_shots,
                "model_call_granularity": "single_call_for_full_script",
                "note": "先全文切镜头，再回填章节；不要复述；不要跳过关键句元。",
                "shot_id_policy": {
                    "first_shot_id": str(last_shot_id + 1),
                    "last_shot_id": str(last_shot_id + total_target_shots),
                    "preserve_continuous_order": True,
                    "do_not_renumber_previous_shots": True,
                },
                "coverage_policy": {
                    "coverage_source": "memory.coverage_units",
                    "cover_units_in_order": True,
                    "must_cover_all_unit_ids_once": True,
                    "each_shot_must_output_covered_unit_ids": True,
                    "avoid_repeating_same_sentence_unit": True,
                    "forbid_copying_previous_segment_content": False,
                },
                "structure_policy": {
                    "allow_missing_segment_id_in_initial_output": True,
                    "segment_id_will_be_backfilled_by_runtime": True,
                    "chapter_title_will_be_backfilled_by_runtime": True,
                },
                "visual_query_policy": {
                    "query_should_be_concrete_and_searchable": True,
                    "avoid_repeated_visual_query_from_history": True,
                    "match_vocal_script_semantics": True,
                },
                "spotlight_policy": {
                    "use_spotlight_for_key_concept_turning_point_or_summary": True,
                    "target_spotlight_ratio": max(0.35, min(0.60, max_ratio)),
                    "prefer_more_spotlights_for_template_decoration": True,
                    "avoid_all_shots_becoming_spotlight": True,
                },
            },
            "output_schema": {
                "stage_task": "plan_shots",
                "shots": [{
                    "shot_id": "string",
                    "segment_id": "seg_optional",
                    "chapter_title": "string_optional",
                    "vocal_script": "string",
                    "visual_query": "string",
                    "visual_direction": "string",
                    "intent": "string",
                    "design_rationale": "string",
                    "covered_unit_ids": ["u01"],
                    "coverage_rationale": "string",
                    "is_spotlight": "boolean",
                }],
                "num_shots": "integer",
                "coverage_report": {
                    "covered_unit_ids": ["u01"],
                    "missing_unit_ids": [],
                    "duplicate_unit_ids": [],
                },
            },
        }

    @staticmethod
    def _build_skill_payload_self_reflect(
        state: dict[str, Any], current_plan: list, round_idx: int,
    ) -> dict[str, Any]:
        """Build AICutSkill user payload for self_reflect_plan (Critic evaluates shot plan)."""
        segments = state.get("segments", [])
        return {
            "task": "将用户输入的一段口播稿，通过 AICutPipeline 工具链做成完整长视频。",
            "stage_task": "self_reflect_plan",
            "input_prior_version": "custom_tool_priors_v2",
            "instruction": "质检分段和镜头规划，发现语义覆盖、镜头节奏、视觉检索、装饰表达等问题。",
            "input_script": str(state.get("input_text", "") or ""),
            "memory": {
                "segments": [
                    {k: s.get(k, "") for k in ("segment_id", "title", "summary", "narration_script")}
                    for s in segments if isinstance(s, dict)
                ],
                "shots": current_plan,
            },
            "constraints": {
                "num_segments": len(segments),
                "num_shots": len(current_plan),
                "round": round_idx,
                "mode": "agentic",
            },
            "output_schema": {
                "stage_task": "self_reflect_plan",
                "passed": "boolean",
                "overall_score": "number",
                "issues": [{
                    "issue_id": "string",
                    "severity": "low/medium/high/critical",
                    "shot_id": "string|global",
                    "category": "string",
                    "description": "string",
                    "suggested_fix": "string",
                }],
                "summary": "string",
            },
        }

    @staticmethod
    def _build_skill_payload_quality_judge(
        state: dict[str, Any], shot: dict[str, Any], round_idx: int,
    ) -> dict[str, Any]:
        """Build AICutSkill user payload for quality_judge (Critic evaluates decorated shot)."""
        shot_id = str(shot.get("shot_id", ""))
        shot_assets = state.get("shot_assets", {}) if isinstance(state.get("shot_assets"), dict) else {}
        assets = shot_assets.get(shot_id, {}) if isinstance(shot_assets.get(shot_id), dict) else {}
        all_shot_ids = [
            str(s.get("shot_id", ""))
            for s in state.get("shot_plan", [])
            if isinstance(s, dict) and s.get("shot_id")
        ]

        decoration_info = assets.get("decoration", {})
        verify_info = assets.get("decoration_verify", {})
        decoration_mem = {}
        if decoration_info or verify_info:
            decoration_mem = {
                "status": "success" if decoration_info else "none",
                "shot_id": shot_id,
                "decoration": decoration_info,
                "verify_result": verify_info,
                "passed": bool(verify_info.get("passed", True) if verify_info else True),
            }

        return {
            "task": "将用户输入的一段口播稿，通过 AICutPipeline 工具链做成完整长视频。",
            "stage_task": "quality_judge",
            "input_prior_version": "custom_tool_priors_v2",
            "instruction": "按视觉匹配、音频同步、装饰质量、时间对齐等维度质检一个已合成镜头。",
            "input_script": "",
            "memory": {
                "shot": {
                    "shot_id": shot_id,
                    "segment_id": shot.get("segment_id", ""),
                    "chapter_title": shot.get("chapter_title", ""),
                    "vocal_script": shot.get("vocal_script", ""),
                    "visual_query": shot.get("visual_query", ""),
                    "intent": shot.get("intent", ""),
                    "is_spotlight": bool(shot.get("is_spotlight")),
                },
                "visual_audio": {},
                "decoration": decoration_mem,
                "compose": {},
            },
            "constraints": {
                "shot_ids": all_shot_ids,
                "judge_round": round_idx,
                "dimensions": [
                    "visual_match",
                    "audio_sync",
                    "decoration_quality",
                    "timing_alignment",
                ],
            },
            "output_schema": {
                "stage_task": "quality_judge",
                "shot_id": "string",
                "passed": "boolean",
                "overall_score": "number",
                "dimension_scores": {
                    "visual_match": "number",
                    "audio_sync": "number",
                    "decoration_quality": "number",
                    "timing_alignment": "number",
                },
                "failed_dimensions": ["string"],
                "repair_strategy": "string",
            },
        }

    @staticmethod
    def _build_skill_payload_repair_plan(
        state: dict[str, Any], current_plan: list,
        issues: list, reflect_result: dict,
    ) -> dict[str, Any]:
        """Build AICutSkill user payload for repair_plan (Generator patches shot plan)."""
        segments = state.get("segments", [])
        affected_shot_ids = {
            str(iss.get("shot_id", ""))
            for iss in issues if isinstance(iss, dict) and iss.get("shot_id")
        }
        affected_shots = [
            s for s in current_plan
            if isinstance(s, dict) and str(s.get("shot_id", "")) in affected_shot_ids
        ] or current_plan[:5]

        return {
            "task": "将用户输入的一段口播稿，通过 AICutPipeline 工具链做成完整长视频。",
            "stage_task": "repair_plan",
            "input_prior_version": "custom_tool_priors_v2",
            "instruction": "根据 self_reflect issues 对 shot_plan 做最小必要修复，输出需要传播到下游的 patch。",
            "input_script": str(state.get("input_text", "") or ""),
            "memory": {
                "segments": [
                    {k: s.get(k, "") for k in ("segment_id", "structure_label", "title", "summary", "key_points", "planner_strategy")}
                    for s in segments if isinstance(s, dict)
                ],
                "affected_shots": [
                    {k: s.get(k, "") for k in ("shot_id", "segment_id", "chapter_title", "vocal_script", "visual_query", "intent", "is_spotlight")}
                    for s in affected_shots if isinstance(s, dict)
                ],
                "self_reflect": {
                    "passed": reflect_result.get("passed", False),
                    "overall_score": reflect_result.get("overall_score", 0),
                    "issues": issues,
                    "summary": reflect_result.get("summary", ""),
                },
            },
            "constraints": {
                "requires_downstream_propagation": True,
            },
            "output_schema": {
                "stage_task": "repair_plan",
                "repair_applied": "boolean",
                "repair_summary": "string",
                "requires_downstream_propagation": "boolean",
                "patched_shot_plan": {
                    "shot_id": {
                        "field": "patched value",
                    },
                },
            },
        }

    @staticmethod
    def _build_skill_payload_template_retrieve(
        shot: dict[str, Any],
        segment: dict[str, Any],
        tool_args: dict[str, Any],
        *,
        candidate_templates: list[dict[str, Any]] | None = None,
        already_used: list[str] | None = None,
    ) -> dict[str, Any]:
        """Step 1: Retrieve the most suitable template for this spotlight shot.

        Payload structure matches build_unified_sft_v5.py build_template_rows() exactly.
        """
        shot_ctx = {
            "shot_id": str(shot.get("shot_id", "")),
            "segment_id": str(shot.get("segment_id", "")),
            "chapter_title": str(shot.get("chapter_title", "")),
            "vocal_script": str(shot.get("vocal_script", ""))[:420],
            "design_rationale": str(shot.get("design_rationale", "")),
            "visual_query": str(shot.get("visual_query", "")),
            "intent": str(shot.get("intent", "")),
            "is_spotlight": True,
        }
        segment_ctx = {
            "segment_id": segment.get("segment_id", ""),
            "structure_label": segment.get("structure_label", ""),
            "title": segment.get("title", ""),
            "summary": segment.get("summary", ""),
            "key_points": segment.get("key_points", []),
            "planner_strategy": segment.get("planner_strategy", ""),
        } if segment else {}
        return {
            "task": "用户输入一段口播稿，同一个模型分别以主控/生成/质检角色协作生成可验收长视频。",
            "role": "generator",
            "stage_task": "template_retrieve",
            "input_prior_version": "custom_tool_priors_v2",
            "instruction": (
                "根据口播稿与章节语义，从 candidates 中检索唯一最合适的 spotlight 模版，"
                "输出其 template_id、template_type 和简短理由。选择规则：\n"
                "1. 先判断这一镜在讲什么类型的内容——开场钩子、概念解释、因果关系、"
                "数据对比、叙事过渡还是总结收束，再从对应 category 里挑；\n"
                "2. 参考每个候选的 use_for 和 family："
                "narrative 适合讲述，diagram 适合结构与关系，dataviz 适合数字与对比；\n"
                "3. 有具体数字、比例、对比时优先选 dataviz；"
                "有步骤、层级、因果时优先选 diagram；\n"
                "4. 避开 already_used 里已经用过的模版，同一支片子里模版要有变化；\n"
                "5. template_id 必须是 candidates 里出现过的原文，不能自己造。"
            ),
            "memory": {"shot": shot_ctx, "segment": segment_ctx},
            "candidates": candidate_templates or [],
            "already_used": list(already_used or []),
            "constraints": {
                "select_exactly_one": True,
                "match_spoken_semantics": True,
                "prefer_high_spotlight_level": True,
                "template_id_must_be_from_candidates": True,
                "prefer_unused_template": True,
                "match_category_to_content_type": True,
            },
            "output_schema": {
                "stage_task": "template_retrieve",
                "template_id": "string",
                "template_type": "string",
                "reason": "string",
            },
        }

    @staticmethod
    def _build_skill_payload_template_fill(
        shot: dict[str, Any],
        segment: dict[str, Any],
        tool_args: dict[str, Any],
        *,
        template_id: str,
        slot_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Step 2: Fill all slots of the chosen template with relevant content.

        Payload structure matches build_unified_sft_v5.py build_template_rows() exactly.
        """
        vocal = str(shot.get("vocal_script", ""))[:420]
        shot_dur = float(shot.get("duration_sec", 0) or 0)
        shot_ctx = {
            "shot_id": str(shot.get("shot_id", "")),
            "segment_id": str(shot.get("segment_id", "")),
            "chapter_title": str(shot.get("chapter_title", "")),
            "vocal_script": vocal,
            "design_rationale": str(shot.get("design_rationale", "")),
            "visual_query": str(shot.get("visual_query", "")),
            "intent": str(shot.get("intent", "")),
            "is_spotlight": True,
            "duration_sec": round(shot_dur, 2),
            "spotlight_reason": str(shot.get("spotlight_reason", "")),
        }
        seg_narration = str(segment.get("narration_script", ""))[:600] if segment else ""
        segment_ctx = {
            "segment_id": segment.get("segment_id", ""),
            "structure_label": segment.get("structure_label", ""),
            "title": segment.get("title", ""),
            "summary": segment.get("summary", ""),
            "key_points": segment.get("key_points", []),
            "planner_strategy": segment.get("planner_strategy", ""),
            "narration_excerpt": seg_narration,
        } if segment else {}
        return {
            "task": "用户输入一段口播稿，同一个模型分别以主控/生成/质检角色协作生成可验收长视频。",
            "role": "generator",
            "stage_task": "template_fill",
            "input_prior_version": "custom_tool_priors_v2",
            "instruction": (
                "根据选中模版的 slot_schema，把全部槽位填满。\n"
                "核心要求：槽位文字是给观众看的**概念提炼**，不是口播稿的字幕。"
                "字幕已经单独烧在画面上了，槽位再复述一遍就是重复信息。\n"
                "规则：\n"
                "1. 从这一镜在讲的事情里抽出上位概念、判断或结论，"
                "用观众扫一眼就能记住的短语表达，不要照抄或缩写口播原句；\n"
                "2. 优先写名词短语和判断句，例如「退相干」「成本被谁承担」"
                "「三个前提」，而不是「我们先从个税起征点说起」这种口语流水句；\n"
                "3. 有数字就把数字单独拎出来放进对应槽位，不要把数字埋在句子里；\n"
                "4. 每个槽位的文案必须不同，严禁多个槽位填相同或高度重叠的文字；\n"
                "5. 同一槽位不要出现完整句号结尾的长句，短到能当标题；\n"
                "6. 如果口播稿很短（不足30字），从 segment.summary 和 key_points 提炼；\n"
                "7. 严格遵守各槽位 max_chars 限制，宁可更短；\n"
                "8. timing.decor_start_sec 在 [0.2, 0.8] 之间，"
                f"timing.decor_duration_sec 不超过 {min(4.0, max(2.0, shot_dur * 0.6)):.1f}s"
                f"（本镜头时长 {shot_dur:.1f}s）。"
            ),
            "memory": {"shot": shot_ctx, "segment": segment_ctx},
            "template": slot_schema or {},
            "constraints": {
                "fill_all_slots": True,
                "respect_max_chars": True,
                # Was `content_from_vocal_script_or_segment_context`, which is
                # what told the model to lift text out of the narration.
                "abstract_concept_not_narration_transcript": True,
                "no_verbatim_from_vocal_script": True,
                "prefer_noun_phrase_or_claim": True,
                "avoid_placeholder_or_pipeline_jargon": True,
                "no_duplicate_slot_values": True,
            },
            "output_schema": {
                "stage_task": "template_fill",
                "shot_id": "string",
                "slots": {"name": "text"},
                "animation": {},
                "overlay_config": {},
                "timing": {
                    "decor_start_sec": "float, 0.2~0.8",
                    "decor_duration_sec": f"float, max {min(4.0, max(2.0, shot_dur * 0.6)):.1f}",
                },
            },
        }

    async def _dispatch_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        state: dict[str, Any],
        run_id: str,
    ) -> dict[str, Any]:
        """Dispatch a tool call to the real pipeline implementation."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

        config = state.get("config", {})
        role = self._role_for_tool(tool_name)
        llm = self._make_local_llm(config, active_role=role)

        await self.broadcast(run_id, {
            "type": "console_log",
            "text": f"执行工具: {tool_name}...",
            "level": "stdout",
        })

        t0 = time.monotonic()
        result: dict[str, Any] = {}

        try:
            if tool_name == "Tool_SegmentArticle":
                input_text = state.get("input_text", "")
                max_segments = int(tool_args.get("num_segments_target") or 6)
                user_payload = self._build_skill_payload_segment_article(
                    input_text, tool_args, max_segments,
                )
                _payload_str = json.dumps(user_payload, ensure_ascii=False)
                prediction = await asyncio.to_thread(
                    lambda: self.model.predict_generator(
                        _payload_str, max_tokens=16384, temperature=0.3,
                    )
                )
                raw_out = prediction.cleaned_output or ""
                comp_tokens = getattr(prediction, "completion_tokens", 0)
                logger.info(
                    "[Tool_SegmentArticle] completion_tokens=%d, raw_len=%d, first_200=%s",
                    comp_tokens, len(raw_out), raw_out[:200],
                )
                out_dir = state.get("output_dir", "")
                if out_dir:
                    self._append_trace(
                        Path(out_dir) / "agent_trace.jsonl",
                        {
                            "event": "skill_raw_output",
                            "tool": "Tool_SegmentArticle",
                            "completion_tokens": comp_tokens,
                            "raw_output": raw_out,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
                parsed = self._parse_skill_json(raw_out)
                num_segs = len(parsed.get("segments", []))
                if num_segs < max_segments:
                    logger.warning(
                        "[Tool_SegmentArticle] Expected %d segments but got %d. "
                        "parsed_keys=%s, raw_tail=%.300s",
                        max_segments, num_segs,
                        list(parsed.keys()),
                        raw_out[-300:] if raw_out else "",
                    )
                result = {
                    "status": "success",
                    "intent": parsed.get("intent", ""),
                    "segments": parsed.get("segments", []),
                    "music_style": parsed.get("music_style", ""),
                    "num_segments": num_segs,
                    "source": "generator_skill",
                }

            elif tool_name == "Tool_PlanShots":
                segments = state.get("segments", [])
                if not segments:
                    result = {
                        "status": "error",
                        "shots": [],
                        "num_shots": 0,
                        "num_spotlight": 0,
                        "message": "缺少章节锚点，无法执行全文切镜头",
                        "source": "precondition_failed",
                    }
                    return result
                if state.get("shot_plan"):
                    result = {
                        "status": "success",
                        "shots": [],
                        "num_shots": 0,
                        "num_spotlight": 0,
                        "message": "镜头已生成，跳过重复规划",
                        "source": "no_op",
                    }
                    return result

                # 策略给的 num_shots_target 优先 (它可能是字符串 "20")。
                try:
                    requested_shots = int(float(str(tool_args.get("num_shots_target") or 0)))
                except (TypeError, ValueError):
                    requested_shots = 0
                shot_budgets = self._segment_shot_budgets(
                    segments, requested_total=max(0, requested_shots),
                )
                total_target_shots = max(1, sum(shot_budgets.values()))
                input_text = str(state.get("input_text", "") or "")
                global_units = self._script_coverage_units(input_text)
                anchor_units = self._segment_anchor_unit_ids(segments, global_units)
                if requested_shots > 0 and total_target_shots != requested_shots:
                    budget_note = (
                        f"策略要 {requested_shots} 个, 受每段上限约束调整为 {total_target_shots}"
                    )
                else:
                    budget_note = f"目标 {total_target_shots} 镜头"
                await self.broadcast(run_id, {
                    "type": "console_log",
                    "text": (
                        f"  逐章节切镜头: {len(segments)} 个章节, {budget_note}"
                    ),
                    "level": "stdout" if total_target_shots == (requested_shots or total_target_shots) else "warn",
                })
                # One generator call per segment/chunk, exactly as trained. The
                # orchestrator still calls Tool_PlanShots once with
                # mode=plan_shots_all; fanning that out is the runtime's job.
                shots: list[dict[str, Any]] = []
                coverage_report: dict[str, Any] = {}
                failed_segments: list[str] = []
                for seg in segments:
                    if not isinstance(seg, dict):
                        continue
                    seg_id = str(seg.get("segment_id", "") or "")
                    seg_total = max(1, int(shot_budgets.get(seg_id, 1) or 1))
                    num_chunks = 1 + (seg_total - 1) // _PLAN_SHOTS_MAX_PER_CHUNK
                    seg_shots: list[dict[str, Any]] = []
                    for chunk_index in range(1, num_chunks + 1):
                        lo = (chunk_index - 1) * seg_total // num_chunks
                        hi = chunk_index * seg_total // num_chunks
                        expected = max(1, hi - lo)
                        payload = self._build_skill_payload_plan_shots(
                            state, seg,
                            segments=segments,
                            planned_shots=shots + seg_shots,
                            chunk_index=chunk_index,
                            num_chunks=num_chunks,
                            expected_shots=expected,
                            total_segment_shots=seg_total,
                            total_target_shots=total_target_shots,
                        )
                        _payload_str = json.dumps(payload, ensure_ascii=False)
                        prediction = await asyncio.to_thread(
                            lambda s=_payload_str: self.model.predict_generator(
                                s, max_tokens=8192, temperature=0.3,
                            )
                        )
                        parsed = self._parse_skill_json(prediction.cleaned_output)
                        chunk_shots = [
                            s for s in parsed.get("shots", []) if isinstance(s, dict)
                        ]
                        if not chunk_shots:
                            logger.warning(
                                "[Tool_PlanShots] segment %s chunk %d/%d produced no shots, raw(200): %.200s",
                                seg_id, chunk_index, num_chunks,
                                prediction.cleaned_output or "",
                            )
                            continue
                        seg_shots.extend(chunk_shots)

                    if not seg_shots:
                        failed_segments.append(seg_id)
                        cov_units = self._segment_coverage_units(seg)
                        budget = min(seg_total, max(1, len(cov_units)))
                        for slot in range(budget):
                            lo = slot * len(cov_units) // budget
                            hi = (slot + 1) * len(cov_units) // budget
                            seg_shots.append(self._build_minimal_segment_shot(
                                seg, "0", cov_units[lo:hi] or cov_units,
                            ))

                    for offset, shot in enumerate(seg_shots):
                        shot["shot_id"] = str(len(shots) + 1)
                        shot["segment_id"] = seg_id
                        shot.setdefault("chapter_title", seg.get("title", ""))
                        shot.setdefault("structure_label", seg.get("structure_label", ""))
                        shot["is_segment_first"] = offset == 0
                        shot["is_segment_last"] = offset == len(seg_shots) - 1
                        shots.append(shot)

                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": (
                            f"  章节 {seg_id} 切出 {len(seg_shots)} 个镜头"
                            f"（预算 {seg_total}，{num_chunks} 块）"
                            + ("  ⚠ 模型未产出，已用兜底镜头" if seg_id in failed_segments else "")
                        ),
                        "level": "warn" if seg_id in failed_segments else "stdout",
                    })

                for shot in shots:
                    shot["is_article_last"] = shot is shots[-1]
                # Must run before the coverage report, which would otherwise fill
                # covered_unit_ids in by fuzzy match and hide the real anchors.
                anchored = self._anchor_shots_to_units(shots, global_units, input_text)
                if shots and anchored < len(shots):
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": (
                            f"  ⚠ {len(shots) - anchored}/{len(shots)} 个镜头的口播稿"
                            f"在原文中定位失败，将按位置均分回填"
                        ),
                        "level": "warn",
                    })
                coverage_report = self._coverage_report_for_segment(shots, global_units)
                if failed_segments:
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": (
                            f"  ⚠ {len(failed_segments)}/{len(segments)} 个章节模型未产出有效镜头，"
                            f"已注入兜底: {', '.join(failed_segments)}"
                        ),
                        "level": "warn",
                    })
                if not shots:
                    # Fallback: honour each segment's shot budget rather than
                    # emitting one shot per segment. `_fill_shots_with_full_script`
                    # spreads the script over however many shots exist, so one
                    # shot per segment would hand a 5-chapter script a 5-shot cut
                    # with minutes of narration per shot.
                    for seg in segments:
                        cov_units = self._segment_coverage_units(seg)
                        seg_id = str(seg.get("segment_id", "") or "")
                        budget = max(1, int(shot_budgets.get(seg_id, 1) or 1))
                        budget = min(budget, max(1, len(cov_units)))
                        for slot in range(budget):
                            # Give each shot its own slice of the chapter so the
                            # anchors come out distinct and ordered.
                            lo = slot * len(cov_units) // budget
                            hi = (slot + 1) * len(cov_units) // budget
                            shot = self._build_minimal_segment_shot(
                                seg, str(len(shots) + 1), cov_units[lo:hi] or cov_units,
                            )
                            shot["is_segment_first"] = slot == 0
                            shots.append(shot)
                    coverage_report = self._coverage_report_for_segment(shots, global_units)
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": (
                            f"  ⚠ 全文切镜头未产出有效结果，已按章节预算注入 "
                            f"{len(shots)} 个兜底镜头（目标 {total_target_shots}）"
                        ),
                        "level": "warn",
                    })

                dedupe_info = self._dedupe_shot_vocal_scripts(shots, global_units)
                # 最后一道: 用模型给的切点把原文逐字铺满所有镜头。放在 dedupe 之后,
                # 因为它对 vocal_script 有最终裁定权 —— 切出来的区间互不重叠, 本来
                # 也不可能重复。有了它, 上面的两次重试只影响切点质量, 不再影响完整性。
                fill_info = self._fill_shots_with_full_script(shots, global_units, input_text)
                coverage_report = self._coverage_report_for_segment(shots, global_units)
                if int(fill_info.get("dropped_shots", 0) or 0) > 0:
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": (
                            f"  ⚠ 镜头数超过全文句子数，已合并末尾 "
                            f"{int(fill_info.get('dropped_shots', 0) or 0)} 个空镜头"
                        ),
                        "level": "warn",
                    })
                if int(dedupe_info.get("rewritten_count", 0) or 0) > 0:
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": (
                            f"  ⚠ 检测到重复口播，已强制去重重写 "
                            f"{int(dedupe_info.get('rewritten_count', 0) or 0)} 个镜头"
                        ),
                        "level": "warn",
                    })

                # A cut this far below target means the script never got divided
                # properly; every later stage inherits it, so say so instead of
                # quietly building a video out of a handful of huge shots.
                degenerate = len(shots) * 2 < total_target_shots
                if degenerate:
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": (
                            f"  ⚠ 镜头数 {len(shots)} 远低于目标 {total_target_shots}"
                            f"（全文切出 {len(global_units)} 个句元）。"
                            f"稿件可能缺少句末标点，或模型未产出有效切点，"
                            f"成片节奏会明显偏慢。"
                        ),
                        "level": "warn",
                    })

                rebuilt_segments = self._rebuild_segments_from_shots(segments, shots, anchor_units)
                coverage_rate = coverage_report.get("coverage_rate", 0)
                script_complete = bool(fill_info.get("complete"))
                await self.broadcast(run_id, {
                    "type": "console_log",
                    "text": (
                        f"  ✓ 全文切镜头完成: {len(shots)} 个镜头, "
                        f"覆盖率 {coverage_rate:.0%}, 口播稿"
                        f"{'完整' if script_complete else '疑似不完整'}, 已回填章节"
                    ),
                    "level": "stdout" if script_complete and not degenerate else "warn",
                })
                result = {
                    "status": "success",
                    "shots": shots,
                    "segments": rebuilt_segments,
                    "num_shots": len(shots),
                    "num_spotlight": sum(1 for s in shots if isinstance(s, dict) and s.get("is_spotlight")),
                    "coverage_report": coverage_report,
                    "source": "generator_skill",
                    "script_units": len(global_units),
                    "target_shots": total_target_shots,
                    "degenerate_plan": degenerate,
                    "_segments_planned": len(rebuilt_segments),
                }

            elif tool_name == "Tool_SelfReflect":
                is_post_decoration = self.decorated_done
                base_max_rounds = int(
                    (config.get("stage2_segment", {}) or {}).get("max_repair_rounds", 3)
                    if isinstance(config.get("stage2_segment", {}), dict)
                    else 3
                )
                # Post-decoration quality_judge gets higher repair budget
                if is_post_decoration:
                    max_rounds = max(2, min(base_max_rounds + 2, 6))
                else:
                    max_rounds = max(1, min(base_max_rounds, 5))
                issues: list[dict[str, Any]] = []
                history: list[dict[str, Any]] = []
                current_plan = list(state.get("shot_plan", []))
                final_reflect: dict[str, Any] = {}

                mode_label = "quality_judge (装饰后质检，高优先修复)" if is_post_decoration else "self_reflect_plan (规划质检)"
                await self.broadcast(run_id, {
                    "type": "console_log",
                    "text": f"  开始自检修复循环 [{mode_label}]: 最多 {max_rounds} 轮修复, 当前 {len(current_plan)} 个镜头",
                    "level": "stdout",
                })

                if is_post_decoration:
                    spotlight_shots = [
                        s for s in current_plan
                        if isinstance(s, dict) and s.get("is_spotlight")
                    ]
                    judge_targets = spotlight_shots if spotlight_shots else current_plan[:5]
                    all_issues: list[dict[str, Any]] = []

                    for shot in judge_targets:
                        shot_id = str(shot.get("shot_id", ""))
                        await self.broadcast(run_id, {
                            "type": "console_log",
                            "text": f"  ── quality_judge: 镜头 {shot_id} ──",
                            "level": "stdout",
                        })
                        judge_payload = self._build_skill_payload_quality_judge(
                            state, shot, 0,
                        )
                        _judge_str = json.dumps(judge_payload, ensure_ascii=False)
                        judge_pred = await asyncio.to_thread(
                            lambda s=_judge_str: self.model.predict_critic(
                                s, max_tokens=4096, temperature=0.2,
                            )
                        )
                        judge_result = self._parse_skill_json(judge_pred.cleaned_output)
                        passed = bool(judge_result.get("passed"))
                        score = judge_result.get("overall_score", 0.0)
                        failed_dims = judge_result.get("failed_dimensions", [])

                        if not passed and failed_dims:
                            for dim in failed_dims:
                                all_issues.append({
                                    "issue_id": f"qj_{shot_id}_{dim}",
                                    "severity": "medium",
                                    "shot_id": shot_id,
                                    "category": dim,
                                    "description": judge_result.get("repair_strategy", f"{dim} 质量不达标"),
                                    "suggested_fix": judge_result.get("repair_strategy", ""),
                                })
                            await self.broadcast(run_id, {
                                "type": "console_log",
                                "text": f"  ✗ 镜头 {shot_id} 未通过 (score={score:.2f}), 问题维度: {failed_dims}",
                                "level": "warn",
                            })
                        else:
                            await self.broadcast(run_id, {
                                "type": "console_log",
                                "text": f"  ✓ 镜头 {shot_id} 通过 (score={score:.2f})",
                                "level": "stdout",
                            })

                    issues = all_issues
                    overall_passed = len(all_issues) == 0
                    result = {
                        "stage_task": "quality_judge",
                        "status": "approved" if overall_passed else "needs_fix",
                        "passed": overall_passed,
                        "overall_score": 1.0 if overall_passed else 0.7,
                        "issues": all_issues,
                        "summary": "装饰质检通过" if overall_passed else f"发现 {len(all_issues)} 个装饰质量问题",
                        "repair_history": [{"round": 0, "passed": overall_passed, "issue_count": len(all_issues)}],
                        "repair_rounds": 0,
                        "shot_plan": current_plan,
                    }

                    if all_issues and max_rounds > 0:
                        await self.broadcast(run_id, {
                            "type": "console_log",
                            "text": f"  → 执行装饰后修复补丁...",
                            "level": "stdout",
                        })
                        repair_payload = self._build_skill_payload_repair_plan(
                            state, current_plan, all_issues,
                            {"summary": "装饰后质检发现问题", "fix_options": []},
                        )
                        _repair_str = json.dumps(repair_payload, ensure_ascii=False)
                        repair_pred = await asyncio.to_thread(
                            lambda s=_repair_str: self.model.predict_generator(
                                s, max_tokens=8192, temperature=0.3,
                            )
                        )
                        patched = self._parse_skill_json(repair_pred.cleaned_output)
                        patched_shots = patched.get("shots", current_plan)
                        if isinstance(patched_shots, list) and patched_shots:
                            current_plan = patched_shots
                        result["shot_plan"] = current_plan
                        result["repair_rounds"] = 1
                        await self.broadcast(run_id, {
                            "type": "console_log",
                            "text": f"  ← 装饰后修复完成",
                            "level": "stdout",
                        })

                else:
                    for round_idx in range(max_rounds + 1):
                        await self._wait_if_paused(run_id)
                        await self.broadcast(run_id, {
                            "type": "console_log",
                            "text": f"  ── 自检第 {round_idx + 1}/{max_rounds + 1} 轮 ──",
                            "level": "stdout",
                        })
                        reflect_payload = self._build_skill_payload_self_reflect(
                            state, current_plan, round_idx,
                        )
                        _reflect_str = json.dumps(reflect_payload, ensure_ascii=False)
                        reflect_pred = await asyncio.to_thread(
                            lambda s=_reflect_str: self.model.predict_critic(
                                s, max_tokens=4096, temperature=0.2,
                            )
                        )
                        final_reflect = self._parse_skill_json(reflect_pred.cleaned_output)
                        issues = [
                            item for item in final_reflect.get("issues", [])
                            if isinstance(item, dict)
                        ]
                        passed = bool(final_reflect.get("passed")) and not issues
                        score = final_reflect.get("overall_score", 0.0)
                        history.append({
                            "round": round_idx, "passed": passed,
                            "issue_count": len(issues), "overall_score": score,
                        })

                        if passed:
                            await self.broadcast(run_id, {
                                "type": "console_log",
                                "text": f"  ✓ 第 {round_idx + 1} 轮自检通过 (score={score:.2f})",
                                "level": "stdout",
                            })
                            result = {
                                **final_reflect, "status": "approved", "passed": True,
                                "repair_history": history, "repair_rounds": round_idx,
                                "shot_plan": current_plan,
                            }
                            break

                        issue_summary = "; ".join(
                            f"{item.get('shot_id','?')}: {item.get('description','')[:60]}"
                            for item in issues[:3]
                        )
                        await self.broadcast(run_id, {
                            "type": "console_log",
                            "text": f"  ✗ 第 {round_idx + 1} 轮发现 {len(issues)} 个问题 (score={score:.2f}): {issue_summary}",
                            "level": "warn",
                        })

                        if round_idx >= max_rounds:
                            await self.broadcast(run_id, {
                                "type": "console_log",
                                "text": f"  修复预算耗尽 ({max_rounds} 轮), 使用当前最优方案继续",
                                "level": "warn",
                            })
                            result = {
                                **final_reflect, "status": "repair_exhausted", "passed": False,
                                "repair_history": history, "repair_rounds": round_idx,
                                "shot_plan": current_plan,
                            }
                            break

                        await self.broadcast(run_id, {
                            "type": "console_log",
                            "text": f"  → 执行修复补丁 (第 {round_idx + 1} 轮)...",
                            "level": "stdout",
                        })
                        repair_payload = self._build_skill_payload_repair_plan(
                            state, current_plan, issues, final_reflect,
                        )
                        _repair_str = json.dumps(repair_payload, ensure_ascii=False)
                        repair_pred = await asyncio.to_thread(
                            lambda s=_repair_str: self.model.predict_generator(
                                s, max_tokens=8192, temperature=0.3,
                            )
                        )
                        patched = self._parse_skill_json(repair_pred.cleaned_output)
                        patched_shots = patched.get("shots", current_plan)
                        old_count = len(current_plan)
                        if isinstance(patched_shots, list) and patched_shots:
                            current_plan = patched_shots
                        await self.broadcast(run_id, {
                            "type": "console_log",
                            "text": f"  ← 修复完成: {old_count}→{len(current_plan)} 个镜头, 进入下一轮验证...",
                            "level": "stdout",
                        })

                state["shot_plan"] = result.get("shot_plan", current_plan)
                state["plan_approved"] = bool(result.get("passed"))
                state["plan_issues"] = issues

            elif tool_name == "Tool_SelfReflect_Patch":
                current_plan = list(state.get("shot_plan", []))
                repair_payload = self._build_skill_payload_repair_plan(
                    state, current_plan, self.open_issues or [],
                    {"summary": tool_args.get("repair_summary", ""), "fix_options": []},
                )
                _repair_str = json.dumps(repair_payload, ensure_ascii=False)
                repair_pred = await asyncio.to_thread(
                    lambda s=_repair_str: self.model.predict_generator(
                        s, max_tokens=8192, temperature=0.3,
                    )
                )
                patched = self._parse_skill_json(repair_pred.cleaned_output)
                patched_shots = patched.get("shots", current_plan)
                if isinstance(patched_shots, list) and patched_shots:
                    state["shot_plan"] = patched_shots
                result = {
                    "status": "patched",
                    "repair_applied": True,
                    "shots": state["shot_plan"],
                    "source": "generator_skill",
                }

            elif tool_name == "Tool_BatchVisualAudio":
                if not (self.sw_tts and self.sw_visual):
                    result = {
                        "status": "skipped",
                        "visual": {},
                        "audio": {},
                        "shot_count": len(state.get("shot_plan", [])),
                        "message": "TTS or visual retrieval disabled in infer_config",
                    }
                else:
                    num_shots = len(state.get("shot_plan", []))
                    _va_timeout = max(600.0, num_shots * 40.0)
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": (
                            f"  批量生成 TTS ({num_shots} 个镜头)，"
                            f"随后并发检索视觉素材 (超时={_va_timeout:.0f}s)..."
                        ),
                        "level": "stdout",
                    })
                    try:
                        patch = await asyncio.wait_for(
                            asyncio.to_thread(
                                self._run_graph_node_sync,
                                "node_visual_audio_batch",
                                state,
                                "generator",
                            ),
                            timeout=_va_timeout,
                        )
                    except (asyncio.TimeoutError, TimeoutError):
                        logger.error("[Tool_BatchVisualAudio] 全局超时 %.0fs，部分镜头可能未完成", _va_timeout)
                        await self.broadcast(run_id, {
                            "type": "console_log",
                            "text": f"  ⚠ 素材批量处理超时({_va_timeout:.0f}s)，继续使用已完成的结果",
                            "level": "warn",
                        })
                        patch = {}
                    self._merge_state_patch(state, patch)
                    shot_assets = state.get("shot_assets", {})

                    visual_result = {}
                    audio_result = {}
                    for shot in state.get("shot_plan", []):
                        sid = str(shot.get("shot_id", ""))
                        assets = shot_assets.get(sid, {}) if isinstance(shot_assets.get(sid), dict) else {}
                        clip_meta = assets.get("clip_meta", {}) if isinstance(assets.get("clip_meta"), dict) else {}
                        audio_meta = assets.get("audio_meta", {}) if isinstance(assets.get("audio_meta"), dict) else {}
                        visual_result[sid] = {
                            "status": "success" if assets.get("clip_path") else "error",
                            "clip_path": assets.get("clip_path", ""),
                            "duration_sec": clip_meta.get("duration_sec", 0),
                            "match_score": clip_meta.get("match_score", 0),
                            "candidate_count": len(assets.get("retrieval_candidates", []) or []),
                            "playlist_ready": bool(assets.get("retrieval_candidates")),
                        }
                        # The TTS node does not always stamp a status onto
                        # audio_meta, and an empty string counted as a failure —
                        # which is how a re-dub that produced a real wav still
                        # reported "0 条配音". Fall back to the artefact.
                        audio_status = str(
                            audio_meta.get("status")
                            or assets.get("status")
                            or ("success" if assets.get("audio_path") else "error")
                        )
                        audio_result[sid] = {
                            "status": audio_status,
                            "audio_path": assets.get("audio_path", ""),
                            "duration_sec": audio_meta.get("duration_sec", 0),
                            "backend": audio_meta.get("backend", ""),
                        }
                        await self.broadcast(run_id, {
                            "type": "console_log",
                            "text": (
                                f"  素材+TTS完成: {sid} | "
                                f"TTS {audio_result[sid]['duration_sec']:.1f}s | "
                                f"视频 {visual_result[sid]['duration_sec']:.1f}s | "
                                f"候选 {visual_result[sid]['candidate_count']} 个 | "
                                f"{_trunc(shot.get('vocal_query', shot.get('visual_query', '')), 80)}"
                            ),
                            "level": "stdout",
                        })
                    state["shot_assets"] = shot_assets
                    visual_ok = sum(1 for v in visual_result.values() if v.get("status") == "success")
                    audio_ok = sum(1 for v in audio_result.values() if v.get("status") and "error" not in v.get("status", ""))
                    total = len(state.get("shot_plan", []))
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": (
                            f"  素材批量处理完成: "
                            f"视频检索 {visual_ok}/{total} 成功, "
                            f"TTS {audio_ok}/{total} 成功"
                        ),
                        "level": "stdout" if visual_ok == total else "warn",
                    })
                    result = {
                        "status": "success",
                        "visual": visual_result,
                        "audio": audio_result,
                        "shot_count": total,
                        "visual_success_count": visual_ok,
                        "tts_success_count": audio_ok,
                    }

            elif tool_name == "Tool_BatchDecorateSpotlights":
                await self._wait_if_paused(run_id)
                shot_plan = state.get("shot_plan", [])
                segments = state.get("segments", [])
                spotlight_shots = [s for s in shot_plan if isinstance(s, dict) and s.get("is_spotlight")]
                shot_assets = state.get("shot_assets", {})
                if not isinstance(shot_assets, dict):
                    shot_assets = {}

                total_spotlights = len(spotlight_shots)
                segment_shot_ids_by_segment: dict[str, list[str]] = {}
                for planned_shot in shot_plan:
                    if not isinstance(planned_shot, dict):
                        continue
                    seg_key = str(planned_shot.get("segment_id", ""))
                    shot_id = str(planned_shot.get("shot_id", ""))
                    if seg_key and shot_id:
                        segment_shot_ids_by_segment.setdefault(seg_key, []).append(shot_id)

                _HTTP_TIMEOUT = 90.0
                _MAX_RETRIES = 3
                _CONCURRENCY = 1

                # Health check: verify generator engine is responsive before batch
                _gen_endpoint = getattr(self.model, '_get_endpoint', None)
                if _gen_endpoint:
                    try:
                        _health_resp = requests.get(
                            f"{_gen_endpoint('generator')}/health", timeout=10
                        )
                        if _health_resp.status_code != 200:
                            logger.warning("[BatchDecorate] generator health check status: %s", _health_resp.status_code)
                    except Exception as _hc_err:
                        logger.error("[BatchDecorate] generator unreachable: %s", _hc_err)
                        await self.broadcast(run_id, {
                            "type": "console_log",
                            "text": f"  ✗ Generator 引擎不可达: {_hc_err}",
                            "level": "error",
                        })
                        return {"error": f"generator unreachable: {_hc_err}", "decorated_count": 0}

                await self.broadcast(run_id, {
                    "type": "console_log",
                    "text": (
                        f"  批量装饰 {total_spotlights} 个 Spotlight 镜头 "
                        f"(串行调用 generator, 超时={_HTTP_TIMEOUT:.0f}s, 最多重试={_MAX_RETRIES})..."
                    ),
                    "level": "stdout",
                })
                progress_counter = {"done": 0}
                sem = asyncio.Semaphore(_CONCURRENCY)

                # Load template registry for two-step flow
                _template_registry = self._load_template_registry(state.get("config", {}))
                _valid_template_ids = {
                    str(c.get("template_id", ""))
                    for c in _template_registry.get("candidates", [])
                    if isinstance(c, dict) and c.get("template_id")
                }
                # Shared across shots so one film does not decorate every
                # spotlight with the same structure.
                _used_templates: list[str] = []

                async def _decorate_one(shot: dict) -> tuple[str, bool]:
                    """Two-step template decoration: retrieve → fill."""
                    sid = str(shot.get("shot_id", ""))
                    seg_id = str(shot.get("segment_id", ""))
                    seg = next(
                        (s for s in segments if isinstance(s, dict) and s.get("segment_id") == seg_id),
                        {},
                    )

                    async with sem:
                        for attempt in range(_MAX_RETRIES + 1):
                            try:
                                # Step 1: template_retrieve
                                retrieve_payload = self._build_skill_payload_template_retrieve(
                                    shot, seg, tool_args,
                                    candidate_templates=_template_registry.get("candidates", []),
                                    already_used=_used_templates,
                                )
                                _ret_str = json.dumps(retrieve_payload, ensure_ascii=False)
                                ret_pred = await asyncio.to_thread(
                                    lambda s=_ret_str: self.model.predict_generator(
                                        s, max_tokens=4096, temperature=0.1,
                                        timeout=_HTTP_TIMEOUT,
                                    )
                                )
                                ret_out = ret_pred.cleaned_output or ""
                                ret_result = self._parse_skill_json(ret_out)
                                chosen_id = str(
                                    ret_result.get("template_id")
                                    or ret_result.get("chosen_template_id")
                                    or ""
                                )
                                if not chosen_id:
                                    raise RuntimeError(f"template_retrieve returned empty template_id")
                                # A hallucinated id used to sail through and only
                                # surface at render time as the hardcoded
                                # key_takeaway fallback, silently flattening the
                                # whole film onto one template.
                                if _valid_template_ids and chosen_id not in _valid_template_ids:
                                    picked = self._nearest_template_id(
                                        chosen_id, _valid_template_ids, _used_templates)
                                    await self.broadcast(run_id, {
                                        "type": "console_log", "level": "warn",
                                        "text": (f"  ! 镜头 {sid} 选了不存在的模版 "
                                                 f"{chosen_id}，改用 {picked}"),
                                    })
                                    chosen_id = picked

                                # Look up slot_schema for the chosen template
                                slot_schema = _template_registry.get("slot_schemas", {}).get(chosen_id, {})
                                if chosen_id not in _used_templates:
                                    _used_templates.append(chosen_id)

                                # Step 2: template_fill
                                fill_payload = self._build_skill_payload_template_fill(
                                    shot, seg, tool_args,
                                    template_id=chosen_id,
                                    slot_schema=slot_schema,
                                )
                                _fill_str = json.dumps(fill_payload, ensure_ascii=False)
                                fill_pred = await asyncio.to_thread(
                                    lambda s=_fill_str: self.model.predict_generator(
                                        s, max_tokens=4096, temperature=0.1,
                                        timeout=_HTTP_TIMEOUT,
                                    )
                                )
                                fill_out = fill_pred.cleaned_output or ""
                                fill_result = self._parse_skill_json(fill_out)

                                decoration = fill_result.get("decoration") or {}
                                if not decoration and fill_result.get("slots"):
                                    decoration = {
                                        "template_id": chosen_id,
                                        "template_type": str(ret_result.get("template_type", "")),
                                        "slots": fill_result.get("slots", {}),
                                        "animation": fill_result.get("animation", {}),
                                        "overlay_config": fill_result.get("overlay_config", {}),
                                        "timing": fill_result.get("timing", {}),
                                    }

                                if decoration:
                                    # ── Post-process: fix timing ──
                                    shot_dur = float(shot.get("duration_sec", 0) or 0)
                                    timing = decoration.get("timing") or {}
                                    if not isinstance(timing, dict):
                                        timing = {}
                                    decor_start = max(0.2, min(0.8, float(timing.get("decor_start_sec", 0.3) or 0.3)))
                                    max_decor_dur = max(2.0, min(4.0, shot_dur * 0.6)) if shot_dur > 0 else 3.5
                                    decor_dur = min(max_decor_dur, float(timing.get("decor_duration_sec", max_decor_dur) or max_decor_dur))
                                    if shot_dur > 0 and decor_start + decor_dur > shot_dur - 0.2:
                                        decor_dur = max(1.5, shot_dur - decor_start - 0.2)
                                    decoration["timing"] = {
                                        "decor_start_sec": round(decor_start, 2),
                                        "decor_duration_sec": round(decor_dur, 2),
                                    }

                                    # ── Post-process: de-duplicate slot values ──
                                    slots = decoration.get("slots")
                                    if isinstance(slots, dict) and len(slots) >= 2:
                                        vals = list(slots.values())
                                        str_vals = [str(v).strip() for v in vals]
                                        unique_vals = set(str_vals)
                                        if len(unique_vals) == 1 and len(str_vals) > 1:
                                            seg_kps = seg.get("key_points", []) if isinstance(seg, dict) else []
                                            seg_title = str(seg.get("title", "")) if isinstance(seg, dict) else ""
                                            fallbacks = [str(kp) for kp in seg_kps if isinstance(kp, str)] if isinstance(seg_kps, list) else []
                                            if seg_title and seg_title not in fallbacks:
                                                fallbacks.insert(0, seg_title)
                                            slot_keys = list(slots.keys())
                                            for ki, sk in enumerate(slot_keys[1:], 1):
                                                if ki - 1 < len(fallbacks):
                                                    slots[sk] = fallbacks[ki - 1][:40]
                                            decoration["slots"] = slots

                                    # ── Post-process: honour per-slot max_chars ──
                                    if isinstance(decoration.get("slots"), dict) and slot_schema:
                                        decoration["slots"] = self._clamp_slot_text(
                                            decoration["slots"], slot_schema)

                                    if sid not in shot_assets:
                                        shot_assets[sid] = {}
                                    shot_assets[sid]["decoration"] = decoration
                                    shot_assets[sid]["decoration_verify"] = fill_result.get("verify_result", {})
                                    shot_assets[sid]["template_retrieve"] = {
                                        "chosen_template_id": chosen_id,
                                        "match_reason": str(ret_result.get("reason", "")),
                                    }
                                    progress_counter["done"] += 1
                                    await self.broadcast(run_id, {
                                        "type": "console_log",
                                        "text": f"  ✓ [{progress_counter['done']}/{total_spotlights}] 镜头 {sid} 模版={chosen_id} 装饰完成",
                                        "level": "stdout",
                                    })
                                    return sid, True
                                else:
                                    logger.warning("[BatchDecorate] shot %s empty decoration after fill, raw(200): %s", sid, fill_out[:200])
                                    raise RuntimeError("template_fill returned empty decoration")
                            except Exception as exc:
                                exc_str = str(exc)
                                if attempt < _MAX_RETRIES:
                                    logger.warning("[BatchDecorate] shot %s retry %d/%d: %s", sid, attempt + 1, _MAX_RETRIES, exc_str[:100])
                                    await asyncio.sleep(10 + attempt * 10)
                                else:
                                    logger.warning("[BatchDecorate] shot %s failed after %d retries: %s", sid, _MAX_RETRIES, exc_str[:150])
                                    progress_counter["done"] += 1
                                    await self.broadcast(run_id, {
                                        "type": "console_log",
                                        "text": f"  ✗ [{progress_counter['done']}/{total_spotlights}] 镜头 {sid} 装饰失败",
                                        "level": "warn",
                                    })
                                    return sid, False
                        progress_counter["done"] += 1
                        return sid, False

                raw_results = await asyncio.gather(
                    *[_decorate_one(shot) for shot in spotlight_shots],
                    return_exceptions=True,
                )
                results_list: list[tuple[str, bool]] = []
                for r in raw_results:
                    if isinstance(r, Exception):
                        results_list.append(("unknown", False))
                    else:
                        results_list.append(r)

                decorated_ids = []
                failed_ids = []
                for sid, ok in results_list:
                    if ok:
                        decorated_ids.append(sid)
                    else:
                        failed_ids.append(sid)

                await self.broadcast(run_id, {
                    "type": "console_log",
                    "text": (
                        f"  装饰完成: {len(decorated_ids)}/{total_spotlights} 成功, "
                        f"{len(failed_ids)} 失败"
                    ),
                    "level": "stdout" if not failed_ids else "warn",
                })

                state["shot_assets"] = shot_assets
                result = {
                    "status": "success" if decorated_ids else "skipped_no_assets",
                    "decorated_count": len(decorated_ids),
                    "failed_count": len(failed_ids),
                    "verified_count": len([
                        sid for sid in decorated_ids
                        if (shot_assets.get(sid, {}).get("decoration_verify", {}) or {}).get("passed")
                    ]),
                    "spotlight_count": total_spotlights,
                    "results": {
                        sid: {
                            "decoration": shot_assets.get(sid, {}).get("decoration", {}),
                            "verify_result": shot_assets.get(sid, {}).get("decoration_verify", {}),
                        }
                        for sid in decorated_ids
                    },
                }

            elif tool_name == "Tool_MusicSearch":
                if self.sw_bgm:
                    await self._wait_if_paused(run_id)
                    patch = await asyncio.to_thread(
                        self._run_graph_node_sync,
                        "node_music_search",
                        state,
                        "generator",
                    )
                    self._merge_state_patch(state, patch)
                    bgm_tracks = state.get("bgm_tracks", [])
                    result = {
                        "status": "success" if bgm_tracks else "no_match",
                        "tracks": bgm_tracks,
                        "selected_count": len(bgm_tracks),
                    }
                else:
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": "[跳过] 配乐检索已关闭",
                        "level": "warn",
                    })
                    result = {
                        "status": "skipped",
                        "tracks": [],
                        "message": "BGM search disabled in infer_config",
                    }

            elif tool_name == "Tool_BatchTimelineCompose":
                if self.sw_compose:
                    from pipeline.tools import timeline_compose
                    config.setdefault("services", {}).setdefault("hyperframes", {}).setdefault(
                        "accelerated_render", {}
                    )["enabled"] = self.sw_hyperframes
                    if self.sw_hyperframes:
                        # Override fake_mode to enable real rendering
                        config.setdefault("execution", {})["mode"] = "real"
                        config.setdefault("fake_mode", {})["skip_hyperframes_render"] = False
                        config.setdefault("fake_mode", {})["final_hyperframes_render"] = True
                    else:
                        config.setdefault("fake_mode", {})["skip_hyperframes_render"] = True
                        config.setdefault("fake_mode", {})["final_hyperframes_render"] = False
                        await self.broadcast(run_id, {
                            "type": "console_log",
                            "text": "HyperFrames 渲染已关闭，Spotlight 镜头将使用 ffmpeg 渲染",
                            "level": "warn",
                        })
                    shot_plan = state.get("shot_plan", [])
                    shot_assets = state.get("shot_assets", {})
                    output_dir = state.get("output_dir", ".")

                    # ── Pre-check: validate all clip files exist ──
                    missing_clips = []
                    for shot in shot_plan:
                        sid = str(shot.get("shot_id", ""))
                        assets = shot_assets.get(sid, {}) if isinstance(shot_assets.get(sid), dict) else {}
                        clip_path = str(assets.get("clip_path", "") or "")
                        if not clip_path or not Path(clip_path).exists():
                            missing_clips.append(sid)
                    if missing_clips:
                        await self.broadcast(run_id, {
                            "type": "console_log",
                            "text": f"  ⚠ {len(missing_clips)} 个镜头素材文件缺失 ({', '.join(missing_clips[:5])}), 重新检索...",
                            "level": "warn",
                        })
                        try:
                            saved_decorations: dict[str, Any] = {}
                            for sid, sa in shot_assets.items():
                                if isinstance(sa, dict) and sa.get("decoration"):
                                    saved_decorations[sid] = {
                                        k: v for k, v in sa.items()
                                        if k in ("decoration", "decoration_verify", "template_retrieve")
                                    }

                            patch = await asyncio.wait_for(
                                asyncio.to_thread(
                                    self._run_graph_node_sync,
                                    "node_visual_audio_batch",
                                    state,
                                    "generator",
                                ),
                                timeout=max(300.0, len(missing_clips) * 40.0),
                            )
                            self._merge_state_patch(state, patch)
                            shot_assets = state.get("shot_assets", {})

                            for sid, deco_data in saved_decorations.items():
                                if sid in shot_assets and isinstance(shot_assets[sid], dict):
                                    shot_assets[sid].update(deco_data)

                            recovered = sum(
                                1 for sid in missing_clips
                                if Path(str((shot_assets.get(sid) or {}).get("clip_path", "") or "")).exists()
                            )
                            await self.broadcast(run_id, {
                                "type": "console_log",
                                "text": f"  重新检索完成: {recovered}/{len(missing_clips)} 个镜头素材恢复",
                                "level": "stdout" if recovered == len(missing_clips) else "warn",
                            })
                        except Exception as exc:
                            logger.warning("[BatchTimelineCompose] re-retrieval failed: %s", exc)

                    # ── Spotlight clip concat ──
                    concat_count = 0
                    for shot in shot_plan:
                        sid = str(shot.get("shot_id", ""))
                        assets = shot_assets.get(sid, {}) if isinstance(shot_assets.get(sid), dict) else {}
                        if not shot.get("is_spotlight"):
                            continue
                        primary_clip = str(assets.get("clip_path", "") or "")
                        candidates = list(assets.get("retrieval_candidates") or [])
                        audio_dur = float(
                            (assets.get("audio_meta") or {}).get("duration_sec", 0)
                            or shot.get("duration_sec", 0) or 0
                        )
                        if not primary_clip or not Path(primary_clip).exists() or audio_dur <= 0:
                            reasons = []
                            if not primary_clip:
                                reasons.append("clip_path 为空")
                            elif not Path(primary_clip).exists():
                                reasons.append(f"文件不存在: {primary_clip}")
                            if audio_dur <= 0:
                                reasons.append(f"audio_dur={audio_dur}")
                            await self.broadcast(run_id, {
                                "type": "console_log",
                                "text": f"  ⚠ Spotlight {sid} 跳过拼接: {', '.join(reasons)}",
                                "level": "warn",
                            })
                            continue
                        concat_result = await asyncio.to_thread(
                            self._concat_clips_to_tts_duration,
                            sid, primary_clip, candidates, audio_dur, output_dir,
                        )
                        if concat_result.get("success") and concat_result.get("output_path"):
                            assets["clip_path_original"] = primary_clip
                            assets["clip_path"] = concat_result["output_path"]
                            assets["concat_result"] = concat_result
                            shot_assets[sid] = assets
                            concat_count += 1
                            await self.broadcast(run_id, {
                                "type": "console_log",
                                "text": (
                                    f"  Spotlight {sid}: 拼接 {concat_result.get('clip_count', 1)} 段视频 → "
                                    f"{concat_result.get('final_duration', 0):.1f}s（用于 HyperFrames 整镜头渲染）"
                                ),
                                "level": "stdout",
                            })

                    if concat_count > 0:
                        await self.broadcast(run_id, {
                            "type": "console_log",
                            "text": f"  已拼接 {concat_count} 个 Spotlight 镜头的检索视频，准备 HyperFrames 渲染...",
                            "level": "stdout",
                        })

                    # ── Parallel render with retry ──
                    total_shots = len(shot_plan)
                    # Each slot runs its own headless Chrome plus ffmpeg, so the
                    # ceiling is CPU and disk bandwidth rather than the model.
                    # Tunable because the right number depends on the box.
                    try:
                        _want_conc = int(
                            (config.get("render", {}) or {}).get("shot_concurrency", 4) or 4)
                    except (TypeError, ValueError):
                        _want_conc = 4
                    _RENDER_CONCURRENCY = max(1, min(_want_conc, total_shots))
                    _RENDER_MAX_RETRIES = 2
                    render_sem = asyncio.Semaphore(_RENDER_CONCURRENCY)
                    compose_results: list[dict[str, Any]] = [{}] * total_shots
                    render_counter = {"done": 0}
                    hf_enabled = bool(
                        (config.get("services", {}) or {})
                        .get("hyperframes", {})
                        .get("accelerated_render", {})
                        .get("enabled", False)
                    )
                    spots_with_deco = sum(
                        1 for s in shot_plan
                        if s.get("is_spotlight") and (shot_assets.get(str(s.get("shot_id", "")), {}) or {}).get("decoration")
                    )
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": (
                            f"  开始并行渲染 {total_shots} 个镜头 (并发={_RENDER_CONCURRENCY}, 重试={_RENDER_MAX_RETRIES})"
                            f" HyperFrames={'ON' if hf_enabled else 'OFF'}, {spots_with_deco} 个 Spotlight 有装饰..."
                        ),
                        "level": "stdout",
                    })

                    async def _render_one(shot_idx: int, shot: dict) -> None:
                        sid = shot.get("shot_id", "")
                        assets = shot_assets.get(sid, {})

                        # Reuse is opt-in: the repair loop re-renders a shot after
                        # fixing its assets without clearing rendered_path, so
                        # skipping by default would silently discard those fixes.
                        existing = str(assets.get("rendered_path", "") or "")
                        if tool_args.get("reuse_rendered") and existing and Path(existing).exists():
                            compose_results[shot_idx] = {
                                "render_success": True,
                                "render_status": "cached",
                                "rendered_path": existing,
                            }
                            render_counter["done"] += 1
                            return

                        async with render_sem:
                            cr = None
                            for attempt in range(_RENDER_MAX_RETRIES + 1):
                                try:
                                    cr = await asyncio.to_thread(
                                        timeline_compose.run,
                                        shot, assets, output_dir, config,
                                        render_backend=None,
                                    )
                                    if cr.get("render_success"):
                                        break
                                    if cr.get("render_backend") == "hyperframes":
                                        hf_status = cr.get("render_status", "unknown")
                                        hf_report = cr.get("render_report", {}) if isinstance(cr.get("render_report"), dict) else {}
                                        diag = hf_status
                                        if hf_report.get("stderr"):
                                            diag += f" | {str(hf_report['stderr'])[-120:]}"
                                        await self.broadcast(run_id, {
                                            "type": "console_log",
                                            "text": f"  ⚠ 镜头 {sid} HyperFrames 失败 ({diag}), 降级 ffmpeg",
                                            "level": "warn",
                                        })
                                        cr = await asyncio.to_thread(
                                            timeline_compose.run,
                                            shot, assets, output_dir, config,
                                            render_backend="ffmpeg",
                                        )
                                        if cr.get("render_success"):
                                            break
                                    if attempt < _RENDER_MAX_RETRIES:
                                        await asyncio.sleep(2 + attempt * 3)
                                except Exception as exc:
                                    logger.warning("[TimelineCompose] shot %s attempt %d error: %s", sid, attempt, exc)
                                    if attempt >= _RENDER_MAX_RETRIES:
                                        cr = {"render_success": False, "render_status": f"exception:{exc}", "rendered_path": ""}

                            cr = cr or {"render_success": False, "render_status": "no_result", "rendered_path": ""}
                            compose_results[shot_idx] = cr
                            shot_assets.setdefault(sid, {}).update({
                                "timeline": cr.get("timeline", {}),
                                "rendered_path": cr.get("rendered_path", ""),
                                "compose_result": cr,
                                "approved": bool(cr.get("render_success", False)),
                            })
                            render_counter["done"] += 1
                            done = render_counter["done"]
                            render_ok = cr.get("render_success")
                            backend = cr.get("render_backend", "unknown")
                            status = "✓" if render_ok else "✗"
                            detail = "" if render_ok else f" [{cr.get('render_status', 'unknown')}]"
                            await self.broadcast(run_id, {
                                "type": "console_log",
                                "text": f"  {status} [{done}/{total_shots}] 镜头 {sid} 渲染完成 ({backend}){detail}",
                                "level": "stdout" if render_ok else "warn",
                            })

                    await asyncio.gather(
                        *[_render_one(idx, shot) for idx, shot in enumerate(shot_plan)],
                        return_exceptions=True,
                    )
                    state["shot_assets"] = shot_assets
                    _render_ok = sum(1 for r in compose_results if r.get("render_success"))
                    _render_cached = sum(1 for r in compose_results if r.get("render_status") == "cached")
                    _render_fail = len(compose_results) - _render_ok
                    if _render_cached > 0:
                        await self.broadcast(run_id, {
                            "type": "console_log",
                            "text": (
                                f"  渲染统计: {_render_ok} 成功 "
                                f"(其中 {_render_cached} 个复用已有渲染), "
                                f"{_render_fail} 失败"
                            ),
                            "level": "stdout",
                        })
                    result = {
                        "status": "success",
                        "composed": len(compose_results),
                        "render_success_count": _render_ok,
                        "failed_count": _render_fail,
                        "concat_count": concat_count,
                    }
                else:
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": "[跳过] 时间轴合成已关闭",
                        "level": "warn",
                    })
                    result = {"status": "skipped", "message": "Timeline compose disabled"}

            elif tool_name == "Tool_SeverityRank":
                await self._wait_if_paused(run_id)
                rank_error = ""
                try:
                    patch = await asyncio.to_thread(
                        self._run_graph_node_sync,
                        "node_severity_rank",
                        state,
                        "critic",
                    )
                    self._merge_state_patch(state, patch)
                except Exception as exc:
                    rank_error = str(exc)
                    logger.warning("node_severity_rank failed: %s", exc)

                severity_scores = state.get("severity_scores", {})
                scoring_source = "llm_strict"
                if not severity_scores:
                    # The graph node raised (typically because the judge LLM did
                    # not return a complete score set). Rank on real asset
                    # metadata instead of the old presence-only rule, which gave
                    # every complete shot 0.0 and left the top list empty — that
                    # silently disabled Stage4 entirely.
                    scoring_source = "heuristic_metadata"
                    severity_scores = self._heuristic_severity_scores(state, tool_args)
                    state["severity_scores"] = severity_scores
                    state["severity_ranked_shots"] = sorted(
                        severity_scores,
                        key=lambda s: severity_scores[s]["score"],
                        reverse=True,
                    )
                    top_ids = [
                        sid for sid, row in severity_scores.items() if row.get("in_top_ratio")
                    ]
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": (
                            f"  ⚠ 模型严重度打分不可用（{rank_error or '未返回完整分数'}），"
                            f"改用素材元数据启发式排序: {len(severity_scores)} 个镜头已评分，"
                            f"前 {len(top_ids)} 个进入质检"
                        ),
                        "level": "warn",
                    })

                top_shots = [
                    sid for sid, row in severity_scores.items()
                    if isinstance(row, dict) and row.get("in_top_ratio")
                ]
                result = {
                    "status": "success",
                    "ranked_count": len(severity_scores),
                    "top_shots": top_shots,
                    "severity_scores": severity_scores,
                    "scoring_source": scoring_source,
                }
                if rank_error:
                    result["rank_error"] = rank_error[:300]

            elif tool_name in {
                "Tool_Stage4DimensionJudge",
                "Tool_TextQuickJudge",
                "Tool_Stage4DimensionalRepair",
            }:
                await self._wait_if_paused(run_id)
                # Skip compose: rendering happens later in Tool_BatchTimelineCompose
                state["_skip_compose"] = True
                try:
                    patch = await asyncio.to_thread(
                        self._run_graph_node_sync,
                        "node_compose_and_judge",
                        state,
                        "critic",
                    )
                    self._merge_state_patch(state, patch)
                except Exception as exc:
                    logger.warning("node_compose_and_judge failed: %s", exc)
                finally:
                    state.pop("_skip_compose", None)

                shot_assets = state.get("shot_assets", {})
                # Drop assets left over from an earlier, longer plan before
                # judging. A cache hit on Tool_BatchVisualAudio can carry shot
                # ids this run never planned; they own no render and are never
                # approved, so they would sit in `failed_shot_ids` forever while
                # repair has no shot to act on. The model then sees an identical
                # observation every round and the loop is killed as a deadlock.
                planned_ids = {
                    str(s.get("shot_id", "")) for s in state.get("shot_plan", [])
                }
                orphans = sorted(
                    sid for sid in list(shot_assets) if str(sid) not in planned_ids
                )
                if orphans:
                    for sid in orphans:
                        shot_assets.pop(sid, None)
                    state["shot_assets"] = shot_assets
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": (
                            f"  清理 {len(orphans)} 个不属于本次分镜计划的残留镜头资产: "
                            f"{', '.join(orphans)}"
                        ),
                        "level": "warn",
                    })
                any_judged = any(
                    isinstance(a, dict) and ("approved" in a or "judge_result" in a)
                    for a in shot_assets.values()
                )
                if not any_judged:
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": "  质检节点未产生结果，执行素材完整性兜底质检...",
                        "level": "warn",
                    })
                    issues_found = 0
                    for shot in state.get("shot_plan", []):
                        sid = str(shot.get("shot_id", ""))
                        assets = shot_assets.get(sid, {}) if isinstance(shot_assets.get(sid), dict) else {}
                        has_clip = bool(assets.get("clip_path"))
                        has_audio = bool(assets.get("audio_path"))
                        has_script = bool(str(shot.get("vocal_script", "")).strip())
                        check_passed = has_clip and has_audio and has_script
                        if not check_passed:
                            issues_found += 1
                        missing = []
                        if not has_clip:
                            missing.append("视频素材")
                        if not has_audio:
                            missing.append("TTS音频")
                        if not has_script:
                            missing.append("旁白文本")
                        assets["approved"] = check_passed
                        assets["judge_result"] = {
                            "passed": check_passed,
                            "source": "fallback_completeness_check",
                            "missing": missing,
                        }
                        shot_assets[sid] = assets
                    state["shot_assets"] = shot_assets
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": f"  兜底质检完成: {len(shot_assets) - issues_found}/{len(shot_assets)} 通过",
                        "level": "stdout" if issues_found == 0 else "warn",
                    })

                passed_count = sum(
                    1 for assets in shot_assets.values()
                    if isinstance(assets, dict) and assets.get("approved")
                )
                failed = [
                    sid for sid, assets in shot_assets.items()
                    if isinstance(assets, dict) and not assets.get("approved")
                ]
                result = {
                    "status": "success" if not failed else "completed_with_warnings",
                    "passed_count": passed_count,
                    "failed_count": len(failed),
                    "failed_shot_ids": failed[:10],
                    "shot_count": len(state.get("shot_plan", [])),
                }

            elif tool_name == "Tool_FinalRender":
                await self._wait_if_paused(run_id)
                try:
                    patch = await asyncio.to_thread(
                        self._run_graph_node_sync,
                        "node_final_render",
                        state,
                        "generator",
                    )
                    self._merge_state_patch(state, patch)
                except Exception as exc:
                    logger.warning("node_final_render failed: %s", exc)

                shot_assets = state.get("shot_assets", {})
                rendered = [
                    sid for sid, assets in shot_assets.items()
                    if isinstance(assets, dict) and assets.get("rendered_path")
                ]
                result = {
                    "status": "success" if rendered else "render_incomplete",
                    "rendered_count": len(rendered),
                    "shot_count": len(state.get("shot_plan", [])),
                }

            elif tool_name == "Tool_StitchVideo":
                if self.sw_stitch:
                    from pipeline.tools import stitch_video as sv_tool
                    await self._wait_if_paused(run_id)
                    loop = asyncio.get_running_loop()

                    def _stitch_progress(event: dict[str, Any]) -> None:
                        current = int(event.get("current", 0) or 0)
                        total = int(event.get("total", 0) or 0)
                        label = str(event.get("label", "") or "")
                        stage = str(event.get("stage", "stitch") or "stitch")
                        payload = {
                            "type": "console_log",
                            "text": f"  拼接进度 [{current}/{total}] {label}",
                            "level": "stdout",
                        }
                        loop.call_soon_threadsafe(
                            lambda p=payload: asyncio.create_task(self.broadcast(run_id, p))
                        )
                        progress_payload = {
                            "type": "progress",
                            "tool": "Tool_StitchVideo",
                            "sub_tool": stage,
                            "current": current,
                            "total": total,
                            "progress_pct": round(current * 100 / max(1, total), 1),
                            "label": label,
                        }
                        loop.call_soon_threadsafe(
                            lambda p=progress_payload: asyncio.create_task(self.broadcast(run_id, p))
                        )

                    shot_assets = state.get("shot_assets", {})
                    ready_count = 0
                    for shot in state.get("shot_plan", []):
                        sid = str(shot.get("shot_id", ""))
                        assets = shot_assets.get(sid, {}) if isinstance(shot_assets.get(sid), dict) else {}
                        rendered_path = str(assets.get("rendered_path", "") or "")
                        if rendered_path:
                            ready_count += 1
                    state["shot_assets"] = shot_assets
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": f"  准备拼接: {ready_count}/{len(state.get('shot_plan', []))} 个镜头就绪",
                        "level": "stdout",
                    })
                    skip_bgm = bool(tool_args.get("skip_bgm_remix", False))
                    # `bgm_volume` / `duck_under_voice` are declared in this
                    # tool's arg schema and the editor sends them after a
                    # "调音量" edit. Dropping them here left `stitch_video.run`
                    # on its own defaults, so the mix never changed.
                    music_cfg = config.get("music", {}) if isinstance(config.get("music"), dict) else {}
                    try:
                        bgm_volume = float(
                            tool_args.get("bgm_volume",
                                          state.get("bgm_volume",
                                                    music_cfg.get("bgm_volume", 0.5)))
                        )
                    except (TypeError, ValueError):
                        bgm_volume = 0.5
                    bgm_volume = max(0.0, min(1.0, bgm_volume))
                    duck_under_voice = bool(
                        tool_args.get("duck_under_voice",
                                      music_cfg.get("duck_under_voice", True))
                    )
                    # Transcode + concat + BGM mix cost minutes and get thrown
                    # away by the next edit, so generation stops at a validated
                    # manifest and the real file is built on export instead.
                    defer_stitch = bool(
                        (config.get("render", {}) or {}).get("defer_stitch", True))
                    try:
                        result = await asyncio.to_thread(
                            sv_tool.run,
                            state.get("shot_plan", []),
                            shot_assets,
                            state.get("output_dir", "."),
                            bgm_tracks=state.get("bgm_tracks", []),
                            bgm_volume=bgm_volume,
                            duck_under_voice=duck_under_voice,
                            config=config,
                            progress_callback=_stitch_progress,
                            skip_bgm_remix=skip_bgm,
                            dry_run=defer_stitch,
                        )
                    except Exception as exc:
                        logger.warning("stitch_video.run failed: %s", exc)
                        result = {"status": "error", "error": str(exc)}
                    state["stitch_result"] = result
                    state["final_video_path"] = (
                        result.get("final_video_path")
                        or result.get("video_path")
                        or result.get("output_path")
                        or ""
                    )
                    # The loop's terminal condition is "stitch produced
                    # something"; with deferral that something is a manifest,
                    # not a file, so it needs its own flag.
                    state["stitch_pending"] = bool(
                        result.get("deferred") and result.get("ready_shot_count"))
                    if state["stitch_pending"]:
                        await self.broadcast(run_id, {
                            "type": "console_log",
                            "text": (
                                f"  ✓ 校验通过 {result.get('ready_shot_count', 0)}/"
                                f"{result.get('shot_count', 0)} 个镜头，"
                                f"总时长 {result.get('total_duration_sec', 0)}s；"
                                "实际合成已延后到「导出成片」"
                            ),
                            "level": "stdout",
                        })
                    elif state["final_video_path"]:
                        await self.broadcast(run_id, {
                            "type": "console_log",
                            "text": f"  ✓ 最终视频: {state['final_video_path']}",
                            "level": "stdout",
                        })
                    else:
                        concat_status = result.get("concat_status", "unknown")
                        val_errors = result.get("validation_errors", [])
                        await self.broadcast(run_id, {
                            "type": "console_log",
                            "text": f"  ✗ 拼接失败: concat_status={concat_status}, 验证失败={len(val_errors)}个镜头, ready={result.get('ready_shot_count',0)}/{result.get('shot_count',0)}",
                            "level": "error",
                        })
                        if val_errors:
                            for ve in val_errors[:3]:
                                if isinstance(ve, dict):
                                    await self.broadcast(run_id, {
                                        "type": "console_log",
                                        "text": f"    - {ve.get('shot_id','?')}: {', '.join(ve.get('reasons', [ve.get('reason','unknown')]))}",
                                        "level": "error",
                                    })
                else:
                    await self.broadcast(run_id, {
                        "type": "console_log",
                        "text": "[跳过] 视频拼接已关闭",
                        "level": "warn",
                    })
                    result = {"status": "skipped", "video_path": "", "message": "Stitch disabled"}

            else:
                result = {
                    "status": "skipped",
                    "message": f"Tool {tool_name} not fully integrated yet",
                }

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            result = {
                "status": "error",
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            }
            logger.exception("Tool dispatch error for %s", tool_name)
        finally:
            llm.close()

        elapsed = time.monotonic() - t0
        result["_elapsed_sec"] = round(elapsed, 2)

        await self.broadcast(run_id, {
            "type": "console_log",
            "text": f"工具 {tool_name} 完成 ({elapsed:.1f}s)",
            "level": "stdout",
        })

        return result

    def _apply_tool_output(
        self, tool_name: str, output: dict[str, Any], state: dict[str, Any],
    ) -> None:
        """Apply tool output to pipeline state dict."""
        for key in (
            "shot_assets", "severity_scores", "severity_ranked_shots",
            "compose_shot_idx", "judge_round", "bgm_search_result", "stitch_result",
        ):
            if key in output:
                state[key] = output[key]
        if output.get("final_video_path"):
            state["final_video_path"] = output.get("final_video_path", "")

        if tool_name == "Tool_SegmentArticle":
            state["segments"] = output.get("segments", [])
            state["music_style"] = output.get("music_style", "")

        elif tool_name == "Tool_PlanShots":
            new_shots = output.get("shots", [])
            existing = state.get("shot_plan") or []
            existing.extend(new_shots)
            state["shot_plan"] = existing
            if isinstance(output.get("segments"), list) and output.get("segments"):
                state["segments"] = output.get("segments", [])
            # After all segments are planned, trigger reflect
            if self._all_segments_planned(state):
                self._reflect_generation += 1

        elif tool_name == "Tool_SelfReflect":
            state["plan_approved"] = bool(output.get("passed"))
            state["plan_issues"] = output.get("issues", [])
            self.open_issues = [
                item for item in output.get("issues", [])
                if isinstance(item, dict)
            ]
            if output.get("shot_plan"):
                state["shot_plan"] = output.get("shot_plan", [])
            self._last_reflect_generation = self._reflect_generation

        elif tool_name == "Tool_SelfReflect_Patch":
            patch = output.get("patched_shot_plan") if isinstance(output.get("patched_shot_plan"), dict) else {}
            self.patch_memory.update(patch)
            if isinstance(output.get("shots"), list):
                state["shot_plan"] = output.get("shots", [])
            self.open_issues = []

        elif tool_name == "Tool_BatchVisualAudio":
            self.visual_audio_done = True

        elif tool_name == "Tool_BatchDecorateSpotlights":
            self.decorated_done = True
            self._reflect_generation += 1
            if isinstance(output.get("results"), dict):
                shot_assets = state.setdefault("shot_assets", {})
                for sid, item in output["results"].items():
                    if not isinstance(item, dict):
                        continue
                    assets = shot_assets.setdefault(str(sid), {})
                    if item.get("decoration"):
                        assets["decoration"] = item.get("decoration")
                    if item.get("verify_result"):
                        assets["decoration_verify"] = item.get("verify_result")

        elif tool_name == "Tool_BatchTimelineCompose":
            self.timeline_done = True

        elif tool_name == "Tool_SeverityRank":
            if isinstance(output.get("severity_scores"), dict):
                state["severity_scores"] = output["severity_scores"]

        elif tool_name == "Tool_MusicSearch":
            state["bgm_tracks"] = output.get("tracks", output.get("selected_tracks", []))
            self.bgm = {"status": output.get("status", ""), "selected_count": len(state["bgm_tracks"])}

        elif tool_name in ("Tool_Stage4DimensionJudge", "Tool_TextQuickJudge"):
            self.quality_checked = True
            self.last_judge = {
                "passed_count": output.get("passed_count"),
                "failed_count": output.get("failed_count"),
                "failed_shot_ids": output.get("failed_shot_ids", []),
            }

        elif tool_name == "Tool_Stage4DimensionalRepair":
            self.repaired_done = True

        elif tool_name == "Tool_StitchVideo":
            state["stitch_result"] = output
            state["final_video_path"] = (
                output.get("final_video_path")
                or output.get("video_path")
                or output.get("output_path")
                or state.get("final_video_path", "")
            )
            self.stitched_done = bool(state["final_video_path"])

    def _build_preview(self, state: dict[str, Any]) -> dict[str, Any]:
        """Build 5-track timeline preview from current state."""
        shot_plan = state.get("shot_plan", [])
        shot_assets = state.get("shot_assets", {})
        bgm_tracks = state.get("bgm_tracks", [])
        segments = state.get("segments", [])
        stitch = state.get("stitch_result", {}) if isinstance(state.get("stitch_result"), dict) else {}

        segment_tracks = []
        video_tracks = []
        narration_tracks = []
        subtitle_tracks = []
        render_tracks = []
        offset = 0.0
        shot_offsets: dict[str, tuple[float, float]] = {}

        for shot in shot_plan:
            sid = shot.get("shot_id", "")
            assets = shot_assets.get(sid, {})
            audio_dur = float(
                (assets.get("audio_meta") or {}).get("duration_sec", 0) or 0
            )
            duration = max(audio_dur, float(shot.get("duration_sec", 8.0) or 8.0))

            video_tracks.append({
                "shot_id": sid,
                "title": shot.get("chapter_title", sid),
                "start": round(offset, 2),
                "end": round(offset + duration, 2),
                "duration": round(duration, 2),
                "segment_id": shot.get("segment_id", ""),
                "clip_path": assets.get("clip_path", ""),
                "has_clip": bool(assets.get("clip_path")),
                "is_spotlight": bool(shot.get("is_spotlight")),
            })

            narration_tracks.append({
                "shot_id": sid,
                "start": round(offset, 2),
                "end": round(offset + audio_dur, 2) if audio_dur else round(offset + duration, 2),
                "duration": round(audio_dur or duration, 2),
                "audio_path": assets.get("audio_path", ""),
                "has_audio": bool(assets.get("audio_path")),
                "text": shot.get("vocal_script", ""),
            })

            subtitle_tracks.append({
                "shot_id": sid,
                "start": round(offset, 2),
                "end": round(offset + duration, 2),
                "text": shot.get("vocal_script", ""),
            })

            if shot.get("is_spotlight"):
                render_tracks.append({
                    "shot_id": sid,
                    "start": round(offset, 2),
                    "end": round(offset + duration, 2),
                    "rendered_path": assets.get("rendered_path", ""),
                    "has_render": bool(assets.get("rendered_path")),
                    "decoration": assets.get("decoration", {}),
                })

            shot_offsets[sid] = (offset, offset + duration)
            offset += duration

        segment_by_id = {
            str(seg.get("segment_id", "") or f"seg_{idx + 1:02d}"): seg
            for idx, seg in enumerate(segments)
            if isinstance(seg, dict)
        }
        segment_order = [
            str(seg.get("segment_id", "") or f"seg_{idx + 1:02d}")
            for idx, seg in enumerate(segments)
            if isinstance(seg, dict)
        ]
        grouped: dict[str, list[tuple[float, float, dict[str, Any]]]] = {}
        for shot in shot_plan:
            sid = str(shot.get("shot_id", ""))
            seg_id = str(shot.get("segment_id", "") or "")
            if not seg_id:
                seg_id = f"seg_{min(len(segment_order), len(grouped) + 1):02d}" if segment_order else "segment"
            if sid in shot_offsets:
                grouped.setdefault(seg_id, []).append((shot_offsets[sid][0], shot_offsets[sid][1], shot))

        seen_segments: set[str] = set()
        for seg_id in segment_order + [sid for sid in grouped.keys() if sid not in segment_order]:
            rows = grouped.get(seg_id, [])
            if not rows or seg_id in seen_segments:
                continue
            seen_segments.add(seg_id)
            seg = segment_by_id.get(seg_id, {})
            start = min(row[0] for row in rows)
            end = max(row[1] for row in rows)
            segment_tracks.append({
                "segment_id": seg_id,
                "title": seg.get("title") or seg.get("structure_label") or seg_id,
                "summary": seg.get("summary", ""),
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(end - start, 2),
                "shot_count": len(rows),
                "shot_ids": [str(row[2].get("shot_id", "")) for row in rows],
            })

        bgm_track_list = []
        for t in bgm_tracks:
            bgm_track_list.append({
                "track_id": t.get("track_id", ""),
                "name": t.get("name", ""),
                "start": float(t.get("offset_sec", 0) or 0),
                "end": float(
                    t.get("end_sec", 0)
                    or (float(t.get("offset_sec", 0) or 0) + float(t.get("duration_sec", 0) or 0))
                ),
                "duration": float(t.get("duration_sec", 0) or 0),
            })

        return {
            "total_duration": round(offset, 2),
            "segment_tracks": segment_tracks,
            "video_tracks": video_tracks,
            "narration_tracks": narration_tracks,
            "subtitle_tracks": subtitle_tracks,
            "bgm_tracks": bgm_track_list,
            "render_tracks": render_tracks,
            "music_style": state.get("music_style", ""),
            "final_video_path": state.get("final_video_path", ""),
            "stitch_status": stitch.get("status", ""),
            "stitch_concat_status": stitch.get("concat_status", ""),
            "stitch_manifest_path": stitch.get("manifest_path", ""),
            "stitch_validation_errors": stitch.get("validation_errors", []),
            "stitch_ready_shot_count": stitch.get("ready_shot_count", 0),
            "stitch_shot_count": stitch.get("shot_count", 0),
        }
