"""LangGraph state machine for the AICut 5-stage video generation pipeline.

Graph structure:
  init → [data_stage1] → segment_article → plan_shots → self_reflect (HITL interrupt)
    → visual_audio_batch → decorate_spotlights (with internal check & repair)
    → music_search → severity_rank (pre-sort by problem severity, top 30%)
    → compose_and_judge (dual-track: spotlight render loop / non-spotlight text judge)
    → [final_render in fake preview mode] → stitch_video → END

Key design decisions:
  - Stage 3 visual+audio processed in batch FIRST, then spotlights decorated separately
  - Decoration includes internal check & self-repair loop (no HITL needed)
  - severity_rank pre-sorts shots by problem severity; only top 30% enter repair
  - compose_and_judge implements dual-track routing:
    * Spotlight shots → full render loop (render → VLM-Judge → HITL → repair → re-render)
    * Non-spotlight shots → text quick judge + single HITL repair (no render loop)
    * Low-severity shots (bottom 70%) → approved=true, skip quality check
  - HITL interrupt at self_reflect (shot plan review) AND at repair decisions
  - All steps logged to trajectory for SFT data generation
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .llm import LLMClient
from .state import PipelineState
from .tools import (
    decorate_shot,
    generate_audio,
    music_search,
    modal_probe,
    plan_shots,
    retrieve_visual,
    segment_article,
    self_reflect,
    stitch_video,
    text_quick_judge,
    timeline_compose,
    vlm_render_judge,
)
from .tools import fake_utils
from .trajectory import TrajectoryLogger, compact_payload, extract_reward_scores_from_output

logger = logging.getLogger("aicut_pipeline")
LIVE_TRAJ_PREFIX = "__AICUT_TRAJ__ "
CACHE_SCHEMA_VERSION = "aicut-v3-asr-sentence-stage2"


def _make_project_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"aicut_{ts}_{uuid.uuid4().hex[:6]}"


def _get_llm(state: PipelineState, service: str | None = None) -> LLMClient:
    return LLMClient(state.get("config", {}), service=service)


def _llm_routing_split(config: dict[str, Any]) -> bool:
    return str(config.get("llm_routing", "all_qwen")).strip().lower() == "qwen_gemini_split"


def _normalize_llm_service(value: Any) -> str | None:
    service = str(value or "").strip().lower()
    if not service or service in {"auto", "default", "none", "null"}:
        return None
    return service


def _configured_llm_service(config: dict[str, Any], paths: tuple[tuple[str, ...], ...]) -> str | None:
    for path in paths:
        service = _normalize_llm_service(_cfg_get(config, path, None))
        if service:
            return service
    return None


def _resolve_stage1_polish_service(config: dict[str, Any]) -> str | None:
    explicit = _configured_llm_service(
        config,
        (
            ("stage1_parse", "asr_polish", "service"),
            ("stage1_parse", "asr_polish", "backend"),
            ("stage1_parse", "sentence_split", "service"),
            ("stage1_parse", "sentence_split", "backend"),
        ),
    )
    if explicit:
        return explicit
    if _llm_routing_split(config):
        return "gemini"
    return None


def _resolve_self_reflect_service(config: dict[str, Any]) -> str | None:
    explicit = _configured_llm_service(
        config,
        (
            ("stage2_segment", "self_reflect", "service"),
            ("stage2_segment", "self_reflect", "backend"),
        ),
    )
    if explicit:
        return explicit
    if _llm_routing_split(config):
        return "gemini"
    return None


def _resolve_judge_service(config: dict[str, Any]) -> str | None:
    explicit = _configured_llm_service(
        config,
        (
            ("stage3_trajectory", "judge", "service"),
            ("stage3_trajectory", "judge", "backend"),
        ),
    )
    if explicit:
        return explicit
    if _llm_routing_split(config):
        return "gemini"
    return None


def _get_logger(state: PipelineState) -> TrajectoryLogger:
    return TrajectoryLogger(
        project_id=state.get("project_id", "unknown"),
        output_dir=state.get("output_dir", "./outputs/trajectories"),
    )


def _cfg_get(config: dict[str, Any], path: tuple[str, ...], default: Any) -> Any:
    cur: Any = config
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _cfg_int(config: dict[str, Any], path: tuple[str, ...], default: int) -> int:
    try:
        return int(_cfg_get(config, path, default))
    except (TypeError, ValueError):
        return default


def _cfg_float(config: dict[str, Any], path: tuple[str, ...], default: float) -> float:
    try:
        return float(_cfg_get(config, path, default))
    except (TypeError, ValueError):
        return default


def _log_progress(current: int, total: int, label: str) -> None:
    """Output progress bar to stderr (doesn't interfere with stdout JSON output)."""
    import sys as _sys
    pct = current / max(total, 1) * 100
    bar_len = 30
    filled = int(bar_len * current / max(total, 1))
    bar = "█" * filled + "░" * (bar_len - filled)
    _sys.stderr.write(f"\r  [{bar}] {pct:5.1f}%  {label}  ({current}/{total})")
    _sys.stderr.flush()
    if current >= total:
        _sys.stderr.write("\n")


def _emit_progress(
    tool: str,
    sub_tool: str,
    current: int,
    total: int,
    label: str,
    stage: str = "",
) -> None:
    """Emit a live progress event to frontend (NOT saved to trajectory, only broadcast).

    Frontend renders this as a progress bar showing which sub-tool is running.
    """
    pct = current / max(total, 1) * 100
    entry = {
        "role": "progress",
        "tool": tool,
        "sub_tool": sub_tool,
        "current": current,
        "total": total,
        "progress_pct": round(pct, 1),
        "label": label,
        "metadata": {"stage": stage, "ts": datetime.now().isoformat()},
    }
    _emit_live_trajectory(entry)
    # Also print to stderr for CLI mode
    _log_progress(current, total, f"{tool} → {sub_tool}: {label}")


def _add_stage_marker(
    all_entries: list[dict[str, Any]],
    stage: str,
    label: str,
    marker_type: str = "start",
    summary: str = "",
) -> None:
    """Add a stage boundary marker to trajectory (saved, helps divide stages).

    marker_type: "start" or "end"
    """
    entry = {
        "role": "stage_marker",
        "stage": stage,
        "label": label,
        "marker_type": marker_type,
        "summary": summary,
        "metadata": {
            "stage": stage,
            "marker_type": marker_type,
            "ts": datetime.now().isoformat(),
            # The same entry is emitted live and later returned in the node update.
            # A stable uid lets the dashboard deduplicate those two delivery paths.
            "live_uid": f"stage:{stage}:{marker_type}:{uuid.uuid4().hex[:10]}",
        },
    }
    all_entries.append(entry)
    _emit_live_trajectory(entry)


def _cfg_bool(config: dict[str, Any], path: tuple[str, ...], default: bool) -> bool:
    value = _cfg_get(config, path, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"1", "true", "yes", "on"}:
            return True
        if norm in {"0", "false", "no", "off", ""}:
            return False
    return default


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, tuple):
        for item in value:
            mapped = _as_mapping(item)
            if mapped:
                return mapped
    return {}


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, tuple):
        out: list[dict[str, Any]] = []
        for item in value:
            out.extend(_dict_list(item))
        return out
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _hitl_enabled(state: PipelineState) -> bool:
    return bool(_cfg_get(state.get("config", {}), ("hitl", "enabled"), True))


def _is_fake_mode(config: dict[str, Any]) -> bool:
    """Check if execution.mode is 'fake' (annotation/labeling mode)."""
    return str(config.get("execution", {}).get("mode", "real")).lower() == "fake"


def _bulk_text_only_enabled(config: dict[str, Any]) -> bool:
    """Fast bulk-labeling mode: keep tool traces, simulate expensive media tools."""
    return _is_fake_mode(config) and _cfg_bool(
        config,
        ("execution", "fake_mode", "bulk_text_only", "enabled"),
        False,
    )


def _simulate_visual_audio_enabled(config: dict[str, Any]) -> bool:
    return _bulk_text_only_enabled(config) and _cfg_bool(
        config,
        ("execution", "fake_mode", "bulk_text_only", "simulate_visual_audio"),
        True,
    )


def _simulate_stage4_upstream_enabled(config: dict[str, Any]) -> bool:
    if _bulk_text_only_enabled(config) and _cfg_bool(
        config,
        ("execution", "fake_mode", "bulk_text_only", "simulate_stage4_upstream_tools"),
        True,
    ):
        return True
    return _is_fake_mode(config) and not _cfg_bool(
        config,
        ("stage3_trajectory", "repair", "enable_upstream_execution"),
        True,
    )


def _stage4_focus_decoration_slots(config: dict[str, Any]) -> bool:
    return _cfg_bool(config, ("stage3_trajectory", "text_judge", "focus_decoration_slots"), False) or _cfg_bool(
        config,
        ("execution", "fake_mode", "bulk_text_only", "focus_decoration_slots"),
        False,
    )


def _stage4_ignored_dimensions(config: dict[str, Any]) -> set[str]:
    ignored: set[str] = set()
    for value in (
        _cfg_get(config, ("stage3_trajectory", "text_judge", "ignored_dimensions"), []),
        _cfg_get(config, ("execution", "fake_mode", "bulk_text_only", "ignored_stage4_dimensions"), []),
    ):
        if isinstance(value, list):
            ignored.update(str(item).strip() for item in value if str(item).strip())
    return ignored


def _hitl_auto_mode(config: dict[str, Any]) -> bool:
    """Check if HITL should auto-select (vs interrupting for manual choice)."""
    explicit_mode = str(_cfg_get(config, ("hitl", "mode"), "") or "").strip().lower()
    if explicit_mode in {"auto", "automatic", "off", "disabled"}:
        return True
    if explicit_mode in {"interactive", "manual"}:
        return False
    if not _is_fake_mode(config):
        return False
    return str(
        config.get("execution", {}).get("fake_mode", {}).get("hitl_mode", "interactive")
    ).lower() == "auto"


def _should_interrupt_hitl(state: PipelineState) -> bool:
    """Interactive mode waits for the configured timeout; auto mode returns immediately."""
    if not _hitl_enabled(state):
        return False
    return not _hitl_auto_mode(state.get("config", {}))


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _log_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_output: dict[str, Any],
    stage: str,
    thought: str = "",
) -> list[dict[str, Any]]:
    """Create compact, outcome-aware trajectory entries for a tool call."""
    reward_scores = extract_reward_scores_from_output(
        tool_output,
        entry_index=-1,
        stage=stage,
        tool=tool_name,
    )
    compact_input = compact_payload(tool_input)
    compact_output = _trajectory_tool_output(tool_name, tool_output)
    response_summary = _tool_result_summary(tool_name, tool_output)
    smart_thought = _tool_thought_summary(
        tool_name,
        tool_input,
        tool_output,
        thought,
    )
    entries = []
    if smart_thought:
        entries.append({
            "role": "thought",
            "content": smart_thought,
            "tool": tool_name,
            "metadata": {"tool": tool_name, "stage": stage, "ts": datetime.now().isoformat()},
        })
    entries.append({
        "role": "call",
        "tool": tool_name,
        "tool_input": compact_input,
        "metadata": {"stage": stage, "ts": datetime.now().isoformat()},
    })
    respond_metadata = {"stage": stage, "ts": datetime.now().isoformat()}
    if reward_scores:
        respond_metadata["reward_scores"] = reward_scores
    entries.append({
        "role": "respond",
        "content": response_summary,
        "tool": tool_name,
        "tool_output": compact_output,
        "metadata": respond_metadata,
    })
    return entries


def _trajectory_tool_output(tool_name: str, tool_output: dict[str, Any]) -> dict[str, Any]:
    compact = compact_payload(tool_output)
    if tool_name != "DataPipeline_Stage1_Reverse" or not isinstance(compact, dict):
        return compact
    # Stage1 text is the source article/oral script used for training. Keep the
    # long transcript fields exact in the primary trajectory while still
    # compacting bulky segment/frame lists.
    keep_full = (
        "narration_script",
        "video_summary_text",
        "asr_transcript",
        "raw_asr_transcript",
    )
    for key in keep_full:
        if key in tool_output:
            compact[key] = tool_output[key]
    return compact


def _tool_result_summary(tool_name: str, output: dict[str, Any]) -> str:
    status = str(output.get("status") or "").strip()
    total = (
        output.get("shot_count")
        or output.get("num_shots")
        or output.get("total")
        or output.get("total_count")
    )
    success = (
        output.get("render_success_count")
        or output.get("success_count")
        or output.get("passed_count")
    )
    failed = (
        output.get("render_failed_count")
        or output.get("failed_count")
        or output.get("error_count")
    )
    parts = [tool_name]
    if status:
        parts.append(status)
    if total is not None:
        parts.append(f"总数 {total}")
    if success is not None:
        parts.append(f"成功 {success}")
    if failed is not None:
        parts.append(f"失败 {failed}")
    if len(parts) == 1:
        parts.append("已返回结果")
    return "；".join(str(part) for part in parts)


def _stage1_llm_timing_summary(result: dict[str, Any]) -> str:
    step_results = result.get("step_results", {})
    step = step_results.get("asr_polish_sentence_split", {}) if isinstance(step_results, dict) else {}
    if not isinstance(step, dict):
        return ""
    meta = step.get("meta", {}) if isinstance(step.get("meta"), dict) else {}
    seg_meta = step.get("sentence_segmentation_meta", {})
    if not isinstance(seg_meta, dict):
        seg_meta = meta.get("sentence_segmentation_meta", {}) if isinstance(meta.get("sentence_segmentation_meta"), dict) else {}
    combined_meta = meta.get("combined_attempt_meta", {}) if isinstance(meta.get("combined_attempt_meta"), dict) else {}

    service = (
        step.get("llm_service")
        or meta.get("llm_service")
        or combined_meta.get("llm_service")
        or "llm"
    )
    service_label = str(service).strip() or "llm"
    parts: list[str] = []
    if combined_meta:
        elapsed = combined_meta.get("llm_elapsed_sec", combined_meta.get("qwen_elapsed_sec"))
        status = combined_meta.get("status", "")
        if elapsed is not None:
            parts.append(f"合并纠错+断句 {elapsed}s({status})")
    elapsed = meta.get("llm_elapsed_sec", meta.get("qwen_elapsed_sec"))
    if elapsed is not None:
        parts.append(f"全文纠错 {elapsed}s({meta.get('status', '')})")
    if seg_meta:
        elapsed = seg_meta.get("llm_elapsed_sec", seg_meta.get("qwen_elapsed_sec"))
        if elapsed is not None:
            parts.append(f"完整断句 {elapsed}s({seg_meta.get('status', '')})")
    if not parts:
        return ""
    sentence_count = step.get("sentence_count", result.get("scene_count", 0))
    return f"Stage1 {service_label}耗时: {'；'.join(parts)}；句子分镜={sentence_count}"


def _tool_thought_summary(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_output: dict[str, Any],
    supplied: str,
) -> str:
    """Keep the rationale short and prevent optimistic text from contradicting failures."""
    status = str(tool_output.get("status") or "").lower()
    failed = int(
        tool_output.get("render_failed_count")
        or tool_output.get("failed_count")
        or tool_output.get("error_count")
        or 0
    )
    success = int(
        tool_output.get("render_success_count")
        or tool_output.get("success_count")
        or tool_output.get("passed_count")
        or 0
    )
    total = int(
        tool_input.get("shot_count")
        or tool_output.get("shot_count")
        or tool_output.get("num_shots")
        or success + failed
        or 0
    )
    if failed > 0 or status in {"error", "failed", "partial_failure", "validation_failed"}:
        scope = f"{total} 项中成功 {success}、失败 {failed}" if total else f"检测到 {failed} 项失败"
        return f"{tool_name} 返回部分异常：{scope}。先保留失败项和原因，未满足门禁前不进入下一阶段。"
    cleaned = " ".join(str(supplied or "").split())
    if cleaned:
        return cleaned[:220]
    if total:
        return f"{tool_name} 将批量处理 {total} 项；完成后根据状态与失败计数决定是否继续。"
    return f"{tool_name} 处理当前阶段输入，并用返回状态决定下一步。"


def _emit_live_trajectory(entry: dict[str, Any]) -> None:
    """Emit a live trajectory marker for dashboard real-time rendering."""
    try:
        logger.info("%s%s", LIVE_TRAJ_PREFIX, json.dumps(entry, ensure_ascii=False))
    except Exception:
        logger.exception("failed to emit live trajectory marker")


def _save_json_artifact(
    state: PipelineState,
    stage_dir: str,
    filename: str,
    data: Any,
) -> str:
    """Persist intermediate stage artifacts under output_dir and storage_root cache."""
    output_dir = Path(state.get("output_dir", "") or ".")
    artifact_dir = output_dir / stage_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / filename
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    cache_path = _cache_artifact_path(state, stage_dir, filename)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _cache_scope_key(state: PipelineState) -> str:
    mode = str(state.get("mode", "inference"))
    config = state.get("config", {}) or {}
    stage2_cfg = config.get("stage2_segment", {}) or {}
    if mode == "data_pipeline":
        video_path = str(state.get("input_video_path", "") or "")
        if video_path:
            path = Path(video_path).expanduser()
            try:
                stat = path.stat()
                source_version = {
                    "path": str(path.resolve()),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            except OSError:
                source_version = {"path": str(path)}
            payload = {
                "schema": CACHE_SCHEMA_VERSION,
                "source": source_version,
                "stage1": config.get("stage1_parse", {}) or {},
                "stage2": stage2_cfg,
            }
            digest = hashlib.sha1(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:12]
            return f"video_{path.stem}_{digest}"
    input_text = str(state.get("input_text", "") or "")
    if not input_text:
        input_text = f"{mode}_default"
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "input_text": input_text,
        "stage2": stage2_cfg,
    }
    digest = hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    return f"{mode}_{digest}"


def _cache_root_dir(state: PipelineState) -> Path | None:
    config = state.get("config", {}) or {}
    storage_root = str((config.get("paths", {}) or {}).get("storage_root", "") or "").strip()
    if not storage_root:
        return None
    return Path(storage_root) / "aicut_pipeline_cache" / _cache_scope_key(state)


def _cache_artifact_path(state: PipelineState, stage_dir: str, filename: str) -> Path | None:
    root = _cache_root_dir(state)
    if root is None:
        return None
    return root / stage_dir / filename


def _load_cached_json(state: PipelineState, stage_dir: str, filename: str) -> dict[str, Any] | None:
    cache_path = _cache_artifact_path(state, stage_dir, filename)
    if cache_path is None or not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        payload["_cache_path"] = str(cache_path)
        return payload
    return None


# ---------------------------------------------------------------------------
# Node: init
# ---------------------------------------------------------------------------
def node_init(state: PipelineState) -> dict[str, Any]:
    """Initialize project, set up output directories."""
    project_id = _make_project_id()
    config = state.get("config", {})
    output_root = config.get("paths", {}).get("output_root", "./outputs")
    output_dir = str(Path(output_root) / project_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    mode = state.get("mode", "inference")
    task_desc = (
        f"将长文章转化为完整长视频 (project: {project_id})"
        if mode == "inference"
        else f"从视频反向构建训练数据 (project: {project_id})"
    )

    trajectory_entries = [{
        "role": "user",
        "content": task_desc,
        "metadata": {"stage": "init", "mode": mode, "ts": datetime.now().isoformat()},
    }]

    return {
        "project_id": project_id,
        "output_dir": output_dir,
        "current_stage": "init",
        "trajectory": trajectory_entries,
        "modal_probe_result": {},
        "shot_assets": {},
        "errors": [],
        "plan_approved": False,
        "reflect_round": 0,
        "current_shot_idx": 0,
        "compose_shot_idx": 0,
        "judge_round": 0,
        "hitl_pending": False,
        "severity_scores": {},
        "severity_ranked_shots": [],
    }


# ---------------------------------------------------------------------------
# Node: modal_tools_preflight
# ---------------------------------------------------------------------------
def node_modal_tools_preflight(state: PipelineState) -> dict[str, Any]:
    """Run visual/TTS/music probes before the expensive data/inference stages."""
    config = state.get("config", {})
    mode = state.get("mode", "inference")
    if not modal_probe.enabled_for_mode(config, mode):
        result = {
            "status": "skipped",
            "passed": True,
            "strict_passed": False,
            "mode": mode,
            "reason": "modal_tools_preflight disabled for this mode",
        }
        entries = _log_tool(
            "Tool_ModalToolsPreflight",
            {"mode": mode, "enabled": False},
            result,
            "preflight_modal_tools",
            thought="多模态工具预检未启用，跳过视频检索、TTS和音乐检索探针",
        )
        return {
            "modal_probe_result": result,
            "current_stage": "modal_tools_preflight_skipped",
            "trajectory": entries,
        }

    output_dir = Path(state.get("output_dir", "")) / "modal_tools_preflight"
    logger.info("[modal_tools_preflight] 开始: mode=%s output=%s", mode, output_dir)
    summary = modal_probe.run_probe(
        config,
        output_dir,
        mode=mode,
        force_clear=True,
    )
    compact = modal_probe.compact_summary(summary)
    logger.info(
        "[modal_tools_preflight] 完成: status=%s strict=%s summary=%s",
        compact.get("status"),
        compact.get("strict_passed"),
        compact.get("summary_path"),
    )

    entries = _log_tool(
        "Tool_ModalToolsPreflight",
        {
            "mode": mode,
            "require_strict": compact.get("require_strict", False),
        },
        compact,
        "preflight_modal_tools",
        thought="在进入数据/推理主流程前验证视频检索、Qwen3-TTS和音乐检索服务均可真实调用",
    )

    if modal_probe.fail_on_error(config) and not summary.get("passed", False):
        reasons = "; ".join(summary.get("failure_reasons", []))
        raise RuntimeError(
            "多模态工具预检失败，请先修复检索/TTS/音乐服务。"
            f" summary={compact.get('summary_path')}, reasons={reasons[:800]}"
        )

    return {
        "modal_probe_result": compact,
        "current_stage": "modal_tools_preflight_complete",
        "trajectory": entries,
    }


# ---------------------------------------------------------------------------
# Node: data_stage1 (data_pipeline mode only)
# ---------------------------------------------------------------------------
def node_data_stage1(state: PipelineState) -> dict[str, Any]:
    """Reverse Stage 1: Video → narration script (data pipeline mode only)."""
    from data_pipeline.stage1_reverse import run as run_stage1

    video_path = state.get("input_video_path", "")
    config = state.get("config", {})
    stage1_service = _resolve_stage1_polish_service(config)
    stage1_service_label = stage1_service or "default"
    logger.info("[data_stage1] 开始: 视频路径=%s", video_path)
    logger.info(
        "[data_stage1] 调用 Stage1 (ASR→切镜→%s纠错→按句分镜，无VLM-caption图片输入)...",
        stage1_service_label,
    )

    llm = _get_llm(state, service=stage1_service)
    try:
        result = run_stage1(
            video_path=video_path,
            llm=llm,
            config=config,
            output_dir=state.get("output_dir", ""),
        )
    except Exception as e:
        logger.error("[data_stage1] 执行异常: %s", e)
        raise
    finally:
        llm.close()

    logger.info("[data_stage1] 完成: narration_script长度=%d, 场景数=%d",
                len(result.get("narration_script", "")), result.get("scene_count", 0))

    entries = _log_tool(
        "DataPipeline_Stage1_Reverse",
        {"video_path": state.get("input_video_path", ""), "llm_service": stage1_service_label},
        result,
        "data_stage1",
        thought=f"数据管线模式：先提取ASR并用 {stage1_service_label} 只纠错不改写，再按纠错后的每句话生成分镜时间窗作为后续导演规划输入",
    )
    stage1_timing = _stage1_llm_timing_summary(result)
    if stage1_timing:
        timing_entry = {
            "role": "thought",
            "content": stage1_timing,
            "tool": "DataPipeline_Stage1_Reverse",
            "metadata": {
                "tool": "DataPipeline_Stage1_Reverse",
                "stage": "data_stage1",
                "ts": datetime.now().isoformat(),
            },
        }
        entries.append(timing_entry)
        _emit_live_trajectory(timing_entry)

    return {
        "input_text": result.get("narration_script", ""),
        "asr_transcript": result.get("asr_transcript", ""),
        "asr_segments": result.get("asr_segments", []),
        "video_shots": result.get("video_shots", []),
        "shot_captions": result.get("shot_captions", []),
        "video_summary": result.get("video_summary", ""),
        "video_summary_text": result.get("video_summary_text", ""),
        "narration_script": result.get("narration_script", ""),
        "current_stage": "data_stage1_complete",
        "trajectory": entries,
    }


# ---------------------------------------------------------------------------
# Node: segment_article
# ---------------------------------------------------------------------------
def node_segment_article(state: PipelineState) -> dict[str, Any]:
    """Tool_SegmentArticle: Split input text into structured segments."""
    input_text = state.get("input_text", "")
    logger.info("[segment_article] 开始: 文章长度=%d 字符", len(input_text))
    cached = _load_cached_json(state, "stage2_segment", "article_segmentation.json")
    if cached and isinstance(cached.get("segments"), list) and cached.get("segments"):
        logger.info("[segment_article] 命中缓存: %s", cached.get("_cache_path", ""))
        entries = _log_tool(
            "Tool_SegmentArticle",
            {"article_text_length": len(input_text), "cache_hit": True},
            {
                "status": "success",
                "cache_hit": True,
                "artifact_path": cached.get("_cache_path", ""),
                "num_segments": len(cached.get("segments", [])),
            },
            "stage2_planning",
            thought="命中文章分段缓存，跳过LLM分段步骤",
        )
        return {
            "segments": cached.get("segments", []),
            "music_style": str(cached.get("music_style", "")),
            "current_stage": "stage2_planning",
            "trajectory": entries,
        }

    logger.info("[segment_article] 调用 LLM 进行文章分段...")

    config = state.get("config", {})
    max_segments = _cfg_int(config, ("stage2_segment", "article_segmenter", "max_segments"), 8)
    llm = _get_llm(state)
    try:
        result = segment_article.run(
            llm=llm,
            article_text=input_text,
            max_segments=max_segments,
            video_summary=state.get("video_summary", {}),
            video_shots=state.get("video_shots", []),
            asr_segments=state.get("asr_segments", []),
        )
    except Exception as e:
        logger.error("[segment_article] LLM调用异常: %s", e)
        raise
    finally:
        llm.close()

    logger.info("[segment_article] 完成: 生成 %d 个段落", len(result.get("segments", [])))
    artifact_path = _save_json_artifact(
        state,
        "stage2_segment",
        "article_segmentation.json",
        {
            "intent": result.get("intent", ""),
            "music_style": result.get("music_style", ""),
            "segments": result.get("segments", []),
            "source": result.get("source", ""),
            "fallback_used": bool(result.get("fallback_used", False)),
        },
    )
    result = {**result, "artifact_path": artifact_path}

    entries = _log_tool(
        "Tool_SegmentArticle",
        {"article_text_length": len(state.get("input_text", "")), "max_segments": max_segments},
        result,
        "stage2_planning",
        thought="将长文章拆分为结构化的逻辑段落，每段有明确的叙事重点和口播脚本",
    )

    segments = result.get("segments", [])

    return {
        "segments": segments,
        "music_style": result.get("music_style", ""),
        "current_stage": "stage2_planning",
        "trajectory": entries,
    }


# ---------------------------------------------------------------------------
# Node: plan_shots
# ---------------------------------------------------------------------------
def _estimate_plan_total_duration(state: PipelineState, shots: list[Any]) -> float:
    """Best-effort total video duration at plan time (no TTS has run yet).

    ``postprocess_shot_plan_0907`` needs a *reliable* duration so the 20+ model
    shot floor fires for 4-5min scripts. At plan time there is no audio, so we
    take ``max(sum of planned shot durations, TTS estimate of the full source
    narration)`` — the sum guards short-but-many plans, the text estimate guards
    plans whose per-shot durations are unset/placeholder.
    """
    shot_sum = 0.0
    for s in shots:
        if isinstance(s, dict):
            try:
                shot_sum += float(s.get("duration_sec") or 0.0)
            except (TypeError, ValueError):
                pass

    text = str(state.get("input_text", "") or "")
    if not text:
        parts: list[str] = []
        for seg in state.get("segments", []) or []:
            if isinstance(seg, dict):
                parts.append(str(seg.get("text") or seg.get("content")
                                 or seg.get("narration") or seg.get("vocal_script") or ""))
        text = "".join(parts)
    text_estimate = generate_audio.estimate_duration(text) if text.strip() else 0.0
    return max(shot_sum, text_estimate)


def node_plan_shots(state: PipelineState) -> dict[str, Any]:
    """Tool_PlanShots: Generate shot plan from segments (+ video shots if available)."""
    segments = state.get("segments", [])
    video_shots = state.get("video_shots", [])
    asr_segments = state.get("asr_segments", [])
    shot_captions = state.get("shot_captions", [])
    cached = _load_cached_json(state, "stage2_segment", "shot_plan.json")
    if cached and isinstance(cached.get("shots"), list) and cached.get("shots"):
        logger.info("[plan_shots] 命中缓存: %s", cached.get("_cache_path", ""))
        entries = _log_tool(
            "Tool_PlanShots",
            {
                "num_segments": len(segments),
                "num_video_shots": len(video_shots),
                "cache_hit": True,
            },
            {
                "status": "success",
                "cache_hit": True,
                "artifact_path": cached.get("_cache_path", ""),
                "num_shots": len(cached.get("shots", [])),
                "mode": cached.get("mode", ""),
            },
            "stage2_planning",
            thought="命中镜头规划缓存，跳过LLM镜头规划步骤",
        )
        return {
            "shot_plan": cached.get("shots", []),
            "current_stage": "stage2_planning",
            "trajectory": entries,
        }

    if video_shots:
        logger.info("[plan_shots] 开始: %d 个段落 + %d 个ASR句子分镜 → 镜头标注",
                     len(segments), len(video_shots))
    else:
        logger.info("[plan_shots] 开始: %d 个段落 → 镜头规划", len(segments))

    llm = _get_llm(state)
    try:
        result = plan_shots.run(
            llm=llm,
            segments=segments,
            video_shots=video_shots if video_shots else None,
            asr_segments=asr_segments if asr_segments else None,
            shot_captions=shot_captions if shot_captions else None,
            config=state.get("config", {}),
        )
    except Exception as e:
        logger.error("[plan_shots] 异常: %s", e)
        raise
    finally:
        llm.close()

    if not isinstance(result, dict):
        result = {"shots": [], "status": "failed", "raw_result_type": type(result).__name__}

    shots = result.get("shots", [])
    if not isinstance(shots, list):
        shots = []
    if not shots and segments:
        raise RuntimeError("plan_shots strict validation failed: no usable shots returned")

    plan_total_duration = _estimate_plan_total_duration(state, shots)
    result = plan_shots.postprocess_shot_plan_0907(
        result, config=state.get("config", {}), total_duration=plan_total_duration
    )
    shots = result.get("shots", shots)

    mode = result.get("mode", "segments")
    logger.info("[plan_shots] 完成: 生成 %d 个镜头 (mode=%s)", len(shots), mode)
    artifact_path = _save_json_artifact(
        state,
        "stage2_segment",
        "shot_plan.json",
        {
            "shots": shots,
            "mode": mode,
            "fallback_used": bool(result.get("fallback_used", False)),
            "script_polish_applied": bool(result.get("script_polish_applied", False)),
        },
    )
    result = {**result, "artifact_path": artifact_path}

    entries = _log_tool(
        "Tool_PlanShots",
        {
            "num_segments": len(segments),
            "num_video_shots": len(video_shots),
            "mode": mode,
        },
        result,
        "stage2_planning",
        thought=(
            f"基于 {len(video_shots)} 个ASR句子分镜为每句话生成visual_query并分配段落标签"
            if video_shots
            else "根据分段内容生成镜头剧本，为每个镜头确定旁白、视觉查询和spotlight标记"
        ),
    )

    node_out: dict[str, Any] = {
        "shot_plan": shots,
        "current_stage": "stage2_planning",
        "trajectory": entries,
    }
    if result.get("pacing_postprocessed"):
        node_out["minimum_model_shots"] = result.get("minimum_model_shots", 0)
        node_out["shot_pacing"] = result.get("pacing_analysis", {})
    return node_out


# ---------------------------------------------------------------------------
# Node: self_reflect (HITL interrupt)
# ---------------------------------------------------------------------------
def node_self_reflect(state: PipelineState) -> dict[str, Any]:
    """Tool_SelfReflect: Verify shot plan quality, HITL if issues found."""
    logger.info("[self_reflect] 开始: 自反思第 %d 轮", state.get("reflect_round", 0) + 1)
    config = state.get("config", {})
    self_reflect_service = _resolve_self_reflect_service(config)
    num_options = _cfg_int(config, ("hitl", "num_options"), 4)
    segments = _dict_list(state.get("segments", []))
    shot_plan = _dict_list(state.get("shot_plan", []))

    # ── fake mode: 轻量 agentic 自反思（vocal_script 质检 + O→R→R→V 循环） ──
    if _is_fake_mode(config) and _cfg_bool(
        config, ("execution", "fake_mode", "skip_self_reflect_deep_verify"), True
    ):
        max_rounds = _cfg_int(config, ("stage2_segment", "max_repair_rounds"), 3)
        max_rounds = min(max_rounds, 2)  # 自反思 repair 最大 2 轮
        min_similarity = _cfg_float(config, ("execution", "text_similarity", "min_similarity"), 0.75)
        max_compression = _cfg_float(config, ("execution", "text_similarity", "max_compression_ratio"), 0.10)
        diagnose_batch_size = max(
            1,
            _cfg_int(config, ("execution", "text_similarity", "diagnose_batch_size"), 6),
        )
        diagnose_concurrency = max(
            1,
            _cfg_int(config, ("execution", "text_similarity", "diagnose_concurrency"), 3),
        )

        llm = _get_llm(state, service=self_reflect_service)
        try:
            agentic_result = self_reflect.agentic_run(
                llm,
                segments,
                shot_plan,
                max_rounds=max_rounds,
                min_similarity=min_similarity,
                max_compression_ratio=max_compression,
                diagnose_batch_size=diagnose_batch_size,
                diagnose_concurrency=diagnose_concurrency,
                progress_callback=lambda sub_tool, current, total, label: _emit_progress(
                    "Tool_SelfReflect",
                    sub_tool,
                    current,
                    total,
                    label,
                    "stage2_self_reflect",
                ),
            )
        except Exception as e:
            logger.warning("[self_reflect] agentic 模式异常，降级到本地自检: %s", e)
            local_result = self_reflect._local_verify(segments, shot_plan)
            agentic_result = {
                "passed": local_result.get("passed", False),
                "overall_score": local_result.get("overall_score", 0.0),
                "issues": local_result.get("issues", []),
                "summary": str(local_result.get("summary", "")),
                "fix_options": [],
                "status": "approved" if local_result.get("passed") else "needs_fix",
                "shot_plan": shot_plan,
                "trajectory": [],
            }
        finally:
            llm.close()

        agentic_trajectory = agentic_result.get("trajectory", [])
        new_segments = _dict_list(agentic_result.get("segments", segments))
        new_shot_plan = agentic_result.get("shot_plan", shot_plan)

        entries = _log_tool(
            "Tool_SelfReflect",
            {
                "num_segments": len(new_segments),
                "num_shots": len(new_shot_plan),
                "round": state.get("reflect_round", 0),
                "mode": "agentic",
            },
            {
                "passed": agentic_result.get("passed", False),
                "overall_score": agentic_result.get("overall_score", 0.0),
                "issues": agentic_result.get("issues", []),
                "summary": agentic_result.get("summary", ""),
                "status": agentic_result.get("status", ""),
                "mode": "agentic_lightweight",
                "segmentation_reflected": bool(new_segments),
            },
            "stage2_self_reflect",
            thought=(
                f"第 {state.get('reflect_round', 0)+1} 轮自检先审 Stage 2a 分段，再审 shot_plan；"
                "如果分段边界被修正，就级联更新镜头章节归属后再检查旁白覆盖、压缩率和 spotlight 分布"
            ),
        )
        all_entries = entries + agentic_trajectory

        verify_artifact_path = _save_json_artifact(
            state,
            "stage2_segment",
            f"verify_report_round_{state.get('reflect_round', 0) + 1:02d}_agentic.json",
            agentic_result,
        )

        if agentic_result.get("passed"):
            return {
                "segments": new_segments,
                "shot_plan": new_shot_plan,
                "plan_approved": True,
                "plan_issues": [],
                "reflect_round": state.get("reflect_round", 0) + 1,
                "current_stage": "stage2_approved",
                "trajectory": all_entries,
                "hitl_pending": False,
            }
        all_entries.append({
            "role": "thought",
            "content": (
                "自反思已经修了两轮了，剩下的问题不堵在这里了——后面的素材匹配和 Stage 4 质检还能兜底，"
                "先把 flow 继续走下去。"
            ),
            "metadata": {
                "stage": "stage2_self_reflect",
                "repair_exhausted": True,
                "ts": datetime.now().isoformat(),
            },
        })
        return {
            "segments": new_segments,
            "shot_plan": new_shot_plan,
            "plan_approved": True,
            "plan_issues": agentic_result.get("issues", []),
            "reflect_round": state.get("reflect_round", 0) + 1,
            "current_stage": "stage2_approved_with_warnings",
            "trajectory": all_entries,
            "hitl_pending": False,
        }

    # ── real mode: 原始 LLM verify + HITL interrupt 流程 ──
    llm = _get_llm(state, service=self_reflect_service)
    try:
        try:
            result = self_reflect.run(
                llm=llm,
                segments=segments,
                shot_plan=shot_plan,
                num_options=num_options,
            )
        except Exception as e:
            logger.warning("[self_reflect] LLM调用异常，切换本地自检: %s", e)
            result = self_reflect._local_verify(segments, shot_plan)
            result = {
                **result,
                "fix_options": [],
                "status": "approved" if result.get("passed", False) else "needs_fix",
                "fallback_used": True,
                "fallback_reason": str(e),
            }
    finally:
        llm.close()

    # Normalize result to always be a dict
    if not isinstance(result, dict):
        if isinstance(result, tuple):
            result = next((item for item in result if isinstance(item, dict)), {})
        else:
            result = {}

    if not result:
        result = {
            "passed": False,
            "overall_score": 0.0,
            "issues": [],
            "summary": "self_reflect returned empty result",
            "fix_options": [],
            "status": "needs_fix",
            "fallback_used": True,
        }

    result["issues"] = _dict_list(result.get("issues", []))
    result["fix_options"] = _dict_list(result.get("fix_options", []))
    result["overall_score"] = _cfg_float({"score": result.get("overall_score", 0.0)}, ("score",), 0.0)
    if not result["issues"] and not bool(result.get("passed", False)):
        # Never expose an empty self-reflect HITL question.
        fallback_verify = self_reflect._local_verify(segments, shot_plan)
        result["issues"] = _dict_list(fallback_verify.get("issues", []))
        result["summary"] = str(fallback_verify.get("summary", "")) or "反思结果为空，已切换本地自检"
        result["overall_score"] = _cfg_float({"score": fallback_verify.get("overall_score", 0.0)}, ("score",), 0.0)
        if not result["fix_options"]:
            result["fix_options"] = self_reflect._fallback_fix_options(
                result["issues"], shot_plan, num_options=num_options
            )
    result["passed"] = bool(result.get("passed", False)) and not result["issues"]
    result["status"] = "approved" if result["passed"] else "needs_fix"
    result["summary"] = str(result.get("summary", ""))
    verify_artifact_path = _save_json_artifact(
        state,
        "stage2_segment",
        f"verify_report_round_{state.get('reflect_round', 0) + 1:02d}.json",
        result,
    )
    result["artifact_path"] = verify_artifact_path

    entries = _log_tool(
        "Tool_SelfReflect",
        {"num_shots": len(shot_plan), "round": state.get("reflect_round", 0)},
        result,
        "stage2_self_reflect",
        thought=f"自反思第 {state.get('reflect_round', 0)+1} 轮: 验证镜头规划的逻辑完整性和质量",
    )

    if result["passed"]:
        return {
            "plan_approved": True,
            "plan_issues": [],
            "reflect_round": state.get("reflect_round", 0) + 1,
            "current_stage": "stage2_approved",
            "trajectory": entries,
            "hitl_pending": False,
        }

    # Not passed → prepare HITL context and interrupt
    options = result["fix_options"]
    score = result["overall_score"]
    issues = result["issues"]
    recommended_option = next(
        (
            idx
            for idx, option in enumerate(options)
            if isinstance(option, dict) and bool(
                option.get("recommended")
                or option.get("is_recommended")
                or option.get("model_recommended")
            )
        ),
        0 if options else -1,
    )
    if 0 <= recommended_option < len(options):
        options[recommended_option]["recommended"] = True
    timeout_sec = max(0, _cfg_int(config, ("hitl", "decision_timeout_sec"), 20))
    hitl_ctx = {
        "node_name": "self_reflect",
        "question": f"镜头规划自检未通过 (score={score:.2f})，请选择修复方案或自定义修改",
        "current_data": {
            "issues": issues,
            "summary": result["summary"],
            "score": score,
        },
        "options": options,
        "recommended_option": recommended_option,
        "timeout_sec": timeout_sec,
        "selected_option": -1,
    }

    entries_with_issues = entries + [{
        "role": "thought",
        "content": (
            f"自反思发现 {len(issues)} 个问题，"
            f"整体评分 {score:.2f}，需要人工审核决策"
        ),
        "metadata": {"stage": "stage2_self_reflect", "ts": datetime.now().isoformat()},
    }]

    if _should_interrupt_hitl(state):
        human_decision = interrupt({
            "type": "self_reflect_review",
            "context": hitl_ctx,
            "message": f"镜头规划自检未通过 (score={score:.2f})",
            "timeout_sec": timeout_sec,
        })
    else:
        human_decision = {
            "selected_option": recommended_option,
            "custom_input": "[auto_mode] 立即采用模型推荐方案" if _hitl_auto_mode(config) else "",
        }

    # Process human decision after resume — normalize to dict
    if not isinstance(human_decision, dict):
        human_decision = _as_mapping(human_decision)
    selected = _cfg_int(human_decision, ("selected_option",), -1)
    custom_input = str(human_decision.get("custom_input", "") or "")

    # Log HITL as trajectory
    chosen_label = ""
    if 0 <= selected < len(options):
        opt = options[selected]
        chosen_label = opt.get("label", "") if isinstance(opt, dict) else str(opt)
    else:
        chosen_label = custom_input

    hitl_entry = {
        "role": "human_feedback",
        "content": (
            f"[自反思] 在 self_reflect 节点，系统提出了 {len(options)} 个方案。\n"
            f"经过人工审查，选择方案 {selected}: {chosen_label}"
        ),
        "metadata": {
            "node_name": "self_reflect",
            "options": options,
            "selected_option": selected,
            "custom_input": custom_input,
            "feedback_type": "self_reflect_demo",
            "ts": datetime.now().isoformat(),
        },
    }

    # Apply fix
    if 0 <= selected < len(options):
        opt = options[selected]
        fix_changes = opt.get("changes", {}) if isinstance(opt, dict) else {}
    elif custom_input:
        fix_changes = {"custom_fix": custom_input}
    else:
        fix_changes = {}

    llm2 = _get_llm(state, service=self_reflect_service)
    try:
        patched_raw = self_reflect.patch(
            llm2,
            {"issues": issues, "fix_changes": fix_changes},
            shot_plan,
        )
        patched = _as_mapping(patched_raw) if not isinstance(patched_raw, dict) else patched_raw
    except Exception as e:
        logger.warning("[self_reflect] patch异常: %s", e)
        patched = {"shots": shot_plan}
    finally:
        llm2.close()

    new_shots = _dict_list(patched.get("shots", shot_plan)) or shot_plan
    patched_path = _save_json_artifact(
        state,
        "stage2_segment",
        f"shot_plan_patch_round_{state.get('reflect_round', 0) + 1:02d}.json",
        {"shots": new_shots, "fix_changes": fix_changes, "issues": issues},
    )
    patch_entries = _log_tool(
        "Tool_SelfReflect_Patch",
        {"fix_changes": fix_changes},
        {"new_shot_count": len(new_shots), "artifact_path": patched_path},
        "stage2_self_reflect",
        thought="根据人工选择的修复方案，重新生成修正后的镜头规划",
    )

    return {
        "shot_plan": new_shots,
        "plan_issues": issues,
        "reflect_round": state.get("reflect_round", 0) + 1,
        "current_stage": "stage2_self_reflect",
        "trajectory": entries_with_issues + [hitl_entry] + patch_entries,
        "hitl_pending": False,
    }


# ---------------------------------------------------------------------------
# Node: reflect_failed
# ---------------------------------------------------------------------------
def node_reflect_failed(state: PipelineState) -> dict[str, Any]:
    """Stop when planning cannot pass after configured repair rounds."""
    max_rounds = _cfg_int(
        state.get("config", {}),
        ("stage2_segment", "max_repair_rounds"),
        3,
    )
    issues = state.get("plan_issues", [])
    entry = {
        "role": "thought",
        "content": (
            f"镜头规划已完成 {state.get('reflect_round', 0)} 轮 verify-reflect-patch-retry，"
            f"仍未通过自检，达到最大轮数 {max_rounds}，终止本次流程并保存轨迹"
        ),
        "metadata": {
            "stage": "stage2_reflect_failed",
            "issues": issues,
            "reflect_round": state.get("reflect_round", 0),
            "max_repair_rounds": max_rounds,
            "ts": datetime.now().isoformat(),
        },
    }
    return {
        "current_stage": "stage2_reflect_failed",
        "errors": [f"self_reflect failed after {max_rounds} repair rounds"],
        "trajectory": [entry],
    }


# ---------------------------------------------------------------------------
# Node: visual_audio_batch
# ---------------------------------------------------------------------------
def _audio_cache_tts_speed_matches(
    shot: Any, cached_asset: Any, config: dict[str, Any] | None
) -> bool:
    """Whether a cached stage3 audio row is still valid w.r.t. TTS speed.

    0907 lets ``set_shot_duration`` write a per-shot ``tts_speed``; reusing audio
    rendered at a different speed would ship the wrong narration timing, so the
    cache must miss on a speed change. Legacy has no per-shot ``tts_speed`` field,
    so this always returns ``True`` and legacy cache behaviour is unchanged.
    """
    if not _is_0907_compose(config):
        return True
    cached = cached_asset if isinstance(cached_asset, dict) else {}
    audio_meta = cached.get("audio_meta") if isinstance(cached.get("audio_meta"), dict) else {}
    planned = generate_audio._effective_tts_speed(shot if isinstance(shot, dict) else {}, config)
    try:
        cached_speed = float(audio_meta.get("speed", 1.0) if audio_meta.get("speed") is not None else 1.0)
    except (TypeError, ValueError):
        cached_speed = 1.0
    return abs(cached_speed - float(planned)) <= 1e-6


def _audio_cache_loudness_matches(
    shot: Any, cached_asset: Any, config: dict[str, Any] | None
) -> bool:
    """Whether a cached stage3 audio row is valid w.r.t. 0907 loudness target.

    Legacy has no loudness normalization requirement, so its cache behaviour is
    unchanged. For 0907 we only reuse a cached narration when it carries
    loudness metadata for the current target and the measured output is still in
    range; otherwise an older pre-normalized audio file could be silently reused.
    """
    if not _is_0907_compose(config):
        return True
    cached = cached_asset if isinstance(cached_asset, dict) else {}
    audio_meta = cached.get("audio_meta") if isinstance(cached.get("audio_meta"), dict) else {}
    loudness = audio_meta.get("loudness") if isinstance(audio_meta.get("loudness"), dict) else {}
    if not loudness:
        return False
    try:
        from studio.newvis0907 import loudness as loudness_mod
    except Exception:
        return False
    if not loudness_mod.target_matches(loudness.get("target")):
        return False
    if str(loudness.get("status", "")).lower() not in {"normalized", "skipped"}:
        return False
    return loudness_mod.payload_is_within_target(loudness)


def node_visual_audio_batch(state: PipelineState) -> dict[str, Any]:
    """Process all shots: retrieve visual clips + generate TTS audio."""
    shot_plan = state.get("shot_plan", [])
    logger.info("[visual_audio_batch] 开始: 为 %d 个镜头批量检索素材+生成音频", len(shot_plan))
    all_entries: list[dict[str, Any]] = []
    _add_stage_marker(all_entries, "stage3_visual_audio", "Stage 3a: 视觉+音频批处理", "start",
                      summary=f"为 {len(shot_plan)} 个镜头并发检索素材 + 生成TTS")
    cached = _load_cached_json(state, "stage3_visual_audio", "shot_assets.json")
    if cached and isinstance(cached.get("shot_assets"), dict):
        cached_assets = cached.get("shot_assets", {})
        plan_ids = {str(s.get("shot_id", "")) for s in shot_plan if isinstance(s, dict)}
        cached_ids = {str(k) for k in cached_assets.keys()}
        shot_by_id = {
            str(s.get("shot_id", "")): s for s in shot_plan if isinstance(s, dict)
        }
        # Do not reuse broken cache rows (e.g. empty clip_path/clip_id), otherwise
        # stage4/5 will repeatedly see missing caption/render failures.
        def _visual_asset_cache_usable(asset: Any) -> bool:
            if not isinstance(asset, dict):
                return False
            backend = str(asset.get("backend", "") or "").strip().lower()
            if backend == "placeholder_visual":
                # Placeholder clips are only for keeping one run alive; never
                # cache-hit them in later runs.
                return False
            clip_path = str(asset.get("clip_path", "") or "").strip()
            if not clip_path:
                return False
            from pathlib import Path as _P
            if not _P(clip_path).exists():
                logger.warning("[visual_audio_batch] 缓存 clip_path 文件不存在: %s", clip_path)
                return False
            clip_meta = asset.get("clip_meta", {}) if isinstance(asset.get("clip_meta"), dict) else {}
            clip_id = str(clip_meta.get("clip_id", "") or "").strip()
            duration_ok = bool(clip_meta.get("duration_constraint_met", False))
            return bool(clip_id) or duration_ok

        def _script_matches(sid: str) -> bool:
            """Reject a cached row whose narration or visual query changed.

            The cache scope key is the article text, not the shot cut, so the
            same article re-planned into a different number of shots hits the
            same cache and rows are reused by shot_id. Shot 6 then inherits the
            audio and footage generated for what is now shot 4, and the whole
            voice track slides out of sync with the subtitles.
            """
            asset = cached_assets.get(sid, {})
            if not isinstance(asset, dict):
                return False
            audio_meta = asset.get("audio_meta", {}) if isinstance(asset.get("audio_meta"), dict) else {}
            cached_text = " ".join(str(audio_meta.get("text", "") or "").split())
            planned_text = " ".join(
                str((shot_by_id.get(sid) or {}).get("vocal_script", "") or "").split()
            )
            cached_query = " ".join(str(asset.get("visual_query", "") or "").split())
            planned_query = " ".join(
                str((shot_by_id.get(sid) or {}).get("visual_query", "") or "").split()
            )
            if not planned_text or not cached_text:
                return False
            if cached_text != planned_text:
                logger.warning(
                    "[visual_audio_batch] 缓存镜头 %s 口播稿已变化，放弃复用缓存", sid
                )
                return False
            if cached_query != planned_query:
                logger.warning(
                    "[visual_audio_batch] 缓存镜头 %s 视觉检索词已变化，放弃复用缓存", sid
                )
                return False
            if not _audio_cache_tts_speed_matches(
                shot_by_id.get(sid), asset, state.get("config", {})
            ):
                logger.warning(
                    "[visual_audio_batch] 缓存镜头 %s TTS 语速已变化，放弃复用缓存", sid
                )
                return False
            if not _audio_cache_loudness_matches(
                shot_by_id.get(sid), asset, state.get("config", {})
            ):
                logger.warning(
                    "[visual_audio_batch] 缓存镜头 %s loudness 未达 0907 目标，放弃复用缓存", sid
                )
                return False
            return True

        cache_assets_usable = all(
            _visual_asset_cache_usable(cached_assets.get(sid, {})) and _script_matches(sid)
            for sid in plan_ids
        )
        if plan_ids and plan_ids.issubset(cached_ids) and cache_assets_usable:
            logger.info("[visual_audio_batch] 命中缓存: %s", cached.get("_cache_path", ""))
            all_entries.extend(_log_tool(
                "Tool_BatchVisualAudio",
                {"num_shots": len(shot_plan), "cache_hit": True},
                {"status": "success", "cache_hit": True, "artifact_path": cached.get("_cache_path", "")},
                "stage3_visual_audio",
                thought="命中视觉+音频批处理缓存，跳过检索和TTS",
            ))
            _add_stage_marker(
                all_entries,
                "stage3_visual_audio",
                "Stage 3a: 视觉+音频批处理",
                "end",
                summary=f"缓存命中，复用 {len(shot_plan)} 个镜头资产",
            )
            return {
                "shot_assets": cached_assets,
                "current_stage": "stage3_visual_audio_done",
                "trajectory": all_entries,
            }
        if plan_ids and plan_ids.issubset(cached_ids) and not cache_assets_usable:
            logger.warning("[visual_audio_batch] 检测到坏缓存(缺少真实 clip_path/clip_id 或占位素材)，将重新执行检索与TTS")
    config = state.get("config", {})
    output_dir = state.get("output_dir", "")
    all_assets = dict(state.get("shot_assets", {}))
    workers = max(
        1,
        _cfg_int(config, ("stage3_trajectory", "visual_audio_parallel_workers"), 6),
    )
    visual_workers = max(
        1,
        _cfg_int(config, ("stage3_trajectory", "visual_parallel_workers"), workers),
    )
    retry_max = max(1, _cfg_int(config, ("stage3_trajectory", "visual_audio_retry_max"), 2))
    retry_backoff_sec = max(0.0, _cfg_float(config, ("stage3_trajectory", "visual_audio_retry_backoff_sec"), 1.0))
    fail_fast = _cfg_bool(config, ("stage3_trajectory", "visual_audio_fail_fast"), False)
    tts_parallel_limit = max(1, _cfg_int(config, ("stage3_trajectory", "tts_parallel_limit"), 4))
    require_all_success = _cfg_bool(
        config,
        ("stage3_trajectory", "require_all_visual_audio_success"),
        True,
    )
    visual_placeholder_on_failure = _cfg_bool(
        config,
        ("stage3_trajectory", "visual_placeholder_on_failure"),
        False,
    )
    visual_relaxed_retry = _cfg_bool(
        config,
        ("stage3_trajectory", "visual_relaxed_retry_on_no_match"),
        True,
    )

    def _run_with_retry(op_name: str, sid: str, fn: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(1, retry_max + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= retry_max:
                    break
                delay = retry_backoff_sec * (2 ** (attempt - 1))
                logger.warning(
                    "[visual_audio_batch] %s 失败 shot=%s attempt=%d/%d: %s; %.1fs 后重试",
                    op_name,
                    sid,
                    attempt,
                    retry_max,
                    exc,
                    delay,
                )
                if delay > 0:
                    time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def _process_visual(
        i: int,
        shot: dict[str, Any],
        audio_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sid = shot.get("shot_id", "")
        logger.info("[visual_audio_batch] 处理镜头 %d/%d: %s", i + 1, len(shot_plan), sid)
        logger.info("[visual_audio_batch]   检索视觉素材: %s", shot.get("visual_query", "")[:50])
        retrieval_shot = dict(shot)
        audio_meta = (
            audio_result.get("audio_meta", {})
            if isinstance(audio_result, dict) and isinstance(audio_result.get("audio_meta"), dict)
            else {}
        )
        tts_duration = _safe_float(audio_meta.get("duration_sec"), 0.0)
        if tts_duration > 0:
            retrieval_shot["tts_duration_sec"] = tts_duration
            retrieval_shot["duration_sec"] = tts_duration

        def _do_visual() -> dict[str, Any]:
            llm_local = _get_llm(state)
            try:
                return retrieve_visual.run(llm_local, retrieval_shot, config)
            finally:
                llm_local.close()
        result = _run_with_retry("检索视觉素材", sid, _do_visual)

        def _build_visual_placeholder(reason: str) -> dict[str, Any]:
            """Create a fallback clip so Stage 3 can continue when retrieval is empty."""
            sid_local = str(shot.get("shot_id", "") or "unknown")
            shots_dir = Path(output_dir).expanduser().resolve() / "shots" / sid_local
            shots_dir.mkdir(parents=True, exist_ok=True)
            clip_path = shots_dir / "placeholder_visual.mp4"
            target_duration = max(
                1.0,
                _safe_float(
                    (audio_result or {}).get("audio_meta", {}).get("duration_sec", 0.0)
                    if isinstance((audio_result or {}).get("audio_meta"), dict)
                    else 0.0,
                    0.0,
                ),
                _safe_float(shot.get("duration_sec", 0.0), 0.0),
            )
            ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
            cmd = [
                ffmpeg_bin, "-y",
                "-f", "lavfi",
                "-i", "color=c=black:s=1280x720:r=25",
                "-t", str(round(target_duration, 3)),
                "-pix_fmt", "yuv420p",
                str(clip_path),
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=90,
                    env=os.environ.copy(),
                )
                if proc.returncode == 0 and clip_path.exists() and clip_path.stat().st_size > 0:
                    logger.warning(
                        "[visual_audio_batch] 检索无候选，镜头 %s 使用占位视频兜底: %s (%s)",
                        sid_local,
                        clip_path,
                        reason,
                    )
                    return {
                        "shot_id": sid_local,
                        "clip_path": str(clip_path),
                        "backend": "placeholder_visual",
                        "retrieval_diagnostics": [{
                            "backend": "placeholder_visual",
                            "status": "fallback",
                            "reason": reason,
                        }],
                        "clip_meta": {
                            "clip_id": f"placeholder_{sid_local}",
                            "match_score": 0.0,
                            "caption": f"placeholder visual for {sid_local}",
                            "duration_sec": round(target_duration, 3),
                            "target_duration_sec": round(target_duration, 3),
                            "duration_shortfall_sec": 0.0,
                            "loop_count": 1.0,
                            "duration_constraint_met": True,
                            "orientation": "landscape",
                            "resolution": "1280x720",
                            "aspect_ratio": 1.778,
                        },
                        "status": "success",
                    }
            except Exception as exc:  # noqa: BLE001
                logger.warning("[visual_audio_batch] 占位视频生成失败 shot=%s: %s", sid_local, exc)
            return result
        if visual_relaxed_retry and str(result.get("status", "")) != "success":
            logger.warning(
                "[visual_audio_batch] 首轮召回无候选，切换扩展检索 shot=%s",
                sid,
            )
            relaxed_config = copy.deepcopy(config)
            retrieval_cfg = (
                relaxed_config.setdefault("retrieval", {})
                if isinstance(relaxed_config, dict)
                else {}
            )
            if isinstance(retrieval_cfg, dict):
                backend_order = retrieval_cfg.get("backend_order", [])
                if not isinstance(backend_order, list):
                    backend_order = []
                allow_local_scan = bool(retrieval_cfg.get("allow_local_scan_fallback", False))
                if allow_local_scan and "local_scan" not in backend_order:
                    backend_order = [*backend_order, "local_scan"]
                retrieval_cfg["backend_order"] = backend_order
                retrieval_cfg["allow_short_candidate_fallback"] = True
                retrieval_cfg["require_duration_coverage"] = False
                retrieval_cfg["min_duration_ratio"] = min(
                    _safe_float(retrieval_cfg.get("min_duration_ratio", 1.0), 1.0),
                    0.6,
                )
                retrieval_cfg["max_loop_count"] = max(
                    _safe_float(retrieval_cfg.get("max_loop_count", 1.0), 1.0),
                    2.0,
                )
                retrieval_cfg["duration_tolerance_sec"] = max(
                    _safe_float(retrieval_cfg.get("duration_tolerance_sec", 1.0), 1.0),
                    2.0,
                )
                retrieval_cfg["min_aspect_ratio"] = min(
                    _safe_float(retrieval_cfg.get("min_aspect_ratio", 1.33), 1.33),
                    1.0,
                )

            def _do_visual_relaxed() -> dict[str, Any]:
                llm_local = _get_llm(state)
                try:
                    return retrieve_visual.run(llm_local, retrieval_shot, relaxed_config)
                finally:
                    llm_local.close()

            relaxed_result = _run_with_retry("检索视觉素材(兜底)", sid, _do_visual_relaxed)
            if str(relaxed_result.get("status", "")) == "success":
                return relaxed_result
            result = relaxed_result

        if str(result.get("status", "")) != "success":
            detail = str(result.get("status", "") or "no_match")
            diag = result.get("retrieval_diagnostics") or []
            backend = result.get("backend") or "none"
            if visual_placeholder_on_failure:
                logger.warning(
                    "[visual_audio_batch] 检索失败 shot=%s backend=%s status=%s; "
                    "启用占位视频兜底 (config: stage3_trajectory.visual_placeholder_on_failure=true)",
                    sid, backend, detail,
                )
                return _build_visual_placeholder(detail)
            raise RuntimeError(
                f"视觉检索失败 shot={sid} backend={backend} status={detail} "
                f"diagnostics={diag} "
                f"— 在线后端均未返回候选。请检查 embedding 服务可达性与 proxy 配置；"
                f"若需临时兜底，可在 config.yaml 设 stage3_trajectory.visual_placeholder_on_failure=true"
            )
        return result

    def _process_audio(i: int, shot: dict[str, Any]) -> dict[str, Any]:
        sid = shot.get("shot_id", "")
        logger.info("[visual_audio_batch]   生成TTS音频: %s", shot.get("vocal_script", "")[:30])
        tts_backend = str(config.get("tts", {}).get("backend", "auto")).lower()
        strict_tts = tts_backend in {"qwen3_service", "qwen3", "hf_service", "huggingface", "local_hf"}

        def _do_audio() -> dict[str, Any]:
            audio_result_local = generate_audio.run(shot, output_dir, config)
            audio_path = Path(str(audio_result_local.get("audio_path", "")))
            if strict_tts and (
                audio_result_local.get("status") != "success"
                or not audio_path.exists()
                or audio_path.stat().st_size <= 0
            ):
                audio_meta = audio_result_local.get("audio_meta", {})
                attempts = audio_meta.get("attempts", [])
                detail = audio_meta.get("message") or (attempts[-1].get("message", "") if attempts else "")
                raise RuntimeError(f"Qwen3-TTS 生成失败: shot_id={sid}, backend={tts_backend}, detail={detail}")
            return audio_result_local

        return _run_with_retry("生成TTS音频", sid, _do_audio)

    if _simulate_visual_audio_enabled(config):
        logger.info("[visual_audio_batch] bulk_text_only: simulate visual/audio for %d shots", len(shot_plan))
        results = []
        for i, shot in enumerate(shot_plan):
            sid = str(shot.get("shot_id", ""))
            audio_result = _fake_generate_audio_result(shot, reason="bulk_text_only_stage3")
            vis_result = _fake_retrieve_visual_result(shot, audio_result, reason="bulk_text_only_stage3")
            results.append({"idx": i, "shot": shot, "sid": sid, "vis_result": vis_result, "audio_result": audio_result})
        _emit_progress(
            "Tool_BatchVisualAudio",
            "模拟素材准备",
            1,
            1,
            f"bulk_text_only 已模拟 {len(results)} 个镜头的视觉检索与TTS",
            "stage3_visual_audio",
        )
    elif len(shot_plan) <= 1:
        results = []
        for i, shot in enumerate(shot_plan):
            sid = shot.get("shot_id", "")
            try:
                audio_result = _process_audio(i, shot)
            except Exception as exc:  # noqa: BLE001
                logger.exception("[visual_audio_batch] TTS 失败 shot=%s: %s", sid, exc)
                if fail_fast:
                    raise
                audio_result = {
                    "shot_id": sid,
                    "audio_path": "",
                    "audio_meta": {"status": "error", "message": str(exc), "duration_sec": 0.0},
                    "status": "error",
                }
            try:
                vis_result = _process_visual(i, shot, audio_result)
            except Exception as exc:  # noqa: BLE001
                logger.exception("[visual_audio_batch] 视觉检索失败 shot=%s: %s", sid, exc)
                if fail_fast:
                    raise
                vis_result = {
                    "shot_id": sid,
                    "clip_path": "",
                    "clip_meta": {"status": "error", "message": str(exc)},
                    "status": "error",
                }
            results.append({"idx": i, "shot": shot, "sid": sid, "vis_result": vis_result, "audio_result": audio_result})
    else:
        logger.info(
            "[visual_audio_batch] 批量执行启用: visual_workers=%d, shots=%d",
            visual_workers,
            len(shot_plan),
        )
        # ── TTS 批量：一次 HTTP 请求处理全部镜头，GPU 利用率从 16% 拉到 ~100% ──
        batch_audio = generate_audio.batch_run(shot_plan, output_dir, config)
        _emit_progress("Tool_BatchVisualAudio", "TTS批量生成", 1, 1,
                       f"TTS 批量完成: {len(batch_audio)} 个镜头", "stage3_visual_audio")

        # ── 视觉检索并发 ──
        rows: list[dict[str, Any]] = []
        visual_futures: dict[Any, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=visual_workers) as visual_ex:
            for i, shot in enumerate(shot_plan):
                sid = shot.get("shot_id", "")
                audio_result = batch_audio.get(sid, {
                    "shot_id": sid,
                    "audio_path": "",
                    "audio_meta": {"status": "error", "message": "batch result missing", "duration_sec": 0.0},
                    "status": "error",
                })
                row = {"idx": i, "shot": shot, "sid": sid, "audio_result": audio_result}
                rows.append(row)
                visual_future = visual_ex.submit(
                    _process_visual, i, shot, audio_result,
                )
                row["visual_future"] = visual_future
                visual_futures[visual_future] = row

            vis_done = 0
            total_shots = len(rows)
            for future in as_completed(visual_futures):
                row = visual_futures[future]
                sid = row["sid"]
                vis_done += 1
                _emit_progress("Tool_BatchVisualAudio", "视觉检索", vis_done, total_shots,
                               f"视觉检索完成: 镜头 {sid}", "stage3_visual_audio")
                try:
                    row["vis_result"] = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.exception("[visual_audio_batch] 视觉检索失败 shot=%s: %s", sid, exc)
                    if fail_fast:
                        raise
                    row["vis_result"] = {
                        "shot_id": sid,
                        "clip_path": "",
                        "clip_meta": {"status": "error", "message": str(exc)},
                        "status": "error",
                    }

        results = sorted(
            [{"idx": row["idx"], "shot": row["shot"], "sid": row["sid"],
              "vis_result": row["vis_result"], "audio_result": row["audio_result"]}
             for row in rows],
            key=lambda x: x["idx"],
        )

    # ── 汇总所有结果到 shot_assets + 批量汇总 JSON（压缩轨迹） ──
    batch_visual_summary = {}
    batch_audio_summary = {}
    for item in results:
        shot = item["shot"]
        sid = item["sid"]
        vis_result = item["vis_result"]
        audio_result = item["audio_result"]

        # 每个镜头的结果压缩到汇总 dict（不逐镜头展开 trajectory）
        batch_visual_summary[sid] = {
            "clip_path": vis_result.get("clip_path", ""),
            "match_score": (
                vis_result.get("clip_meta", {}).get("match_score", 0)
                if isinstance(vis_result.get("clip_meta"), dict) else 0
            ),
            "duration_sec": (
                vis_result.get("clip_meta", {}).get("duration_sec", 0)
                if isinstance(vis_result.get("clip_meta"), dict) else 0
            ),
            "caption": (
                str(vis_result.get("clip_meta", {}).get("caption", ""))[:240]
                if isinstance(vis_result.get("clip_meta"), dict) else ""
            ),
            "backend": vis_result.get("backend", ""),
            "status": vis_result.get("status", ""),
        }
        batch_audio_summary[sid] = {
            "duration_sec": (
                audio_result.get("audio_meta", {}).get("duration_sec", 0)
                if isinstance(audio_result.get("audio_meta"), dict) else 0
            ),
            "backend": (
                audio_result.get("audio_meta", {}).get("backend", "")
                if isinstance(audio_result.get("audio_meta"), dict) else ""
            ),
            "status": audio_result.get("status", ""),
        }

        all_assets[sid] = {
            "shot_id": sid,
            "visual_query": str(shot.get("visual_query", "") or ""),
            "clip_path": vis_result.get("clip_path", ""),
            "clip_meta": vis_result.get("clip_meta", {}),
            "retrieval_candidates": (
                vis_result.get("candidates", [])
                if isinstance(vis_result.get("candidates"), list)
                else []
            ),
            "retrieval_backend": vis_result.get("backend", ""),
            "retrieval_diagnostics": vis_result.get("retrieval_diagnostics", []),
            "audio_path": audio_result.get("audio_path", ""),
            "audio_meta": audio_result.get("audio_meta", {}),
            "decoration": {},
            "timeline": {},
            "rendered_path": "",
            "judge_result": {},
            "approved": False,
        }

    # ── 轨迹压缩：只记录一条汇总 tool call ──
    tts_success = sum(1 for v in batch_audio_summary.values() if v.get("status") == "success")
    vis_success = sum(1 for v in batch_visual_summary.values() if v.get("status") == "success")
    failed_tts_ids = [sid for sid, v in batch_audio_summary.items() if v.get("status") != "success"]
    failed_visual_ids = [sid for sid, v in batch_visual_summary.items() if v.get("status") != "success"]
    all_entries.extend(_log_tool(
        "Tool_BatchVisualAudio",
        {"shot_ids": list(batch_visual_summary.keys()), "shot_count": len(results)},
        {
            "visual": batch_visual_summary,
            "audio": batch_audio_summary,
            "tts_success_count": tts_success,
            "visual_success_count": vis_success,
            "status": (
                "success"
                if tts_success == len(results) and vis_success == len(results)
                else "partial_failure"
            ),
        },
        "stage3_visual_audio",
        thought=f"这批 {len(results)} 个镜头的素材准备完了——TTS 旁白生成 {tts_success}/{len(results)}，视觉检索命中 {vis_success}/{len(results)}，可以进入下一步装饰了",
    ))
    assets_artifact_path = _save_json_artifact(
        state,
        "stage3_visual_audio",
        "shot_assets.json",
        {
            "shot_assets": all_assets,
            "num_shots": len(all_assets),
            "status": "success",
        },
    )
    all_entries.append({
        "role": "thought",
        "content": f"视觉+音频批处理产物已落盘: {assets_artifact_path}",
        "metadata": {"stage": "stage3_visual_audio", "ts": datetime.now().isoformat()},
    })
    _add_stage_marker(all_entries, "stage3_visual_audio", "Stage 3a: 视觉+音频批处理", "end",
                      summary=f"TTS {tts_success}/{len(results)} 成功, 视觉 {vis_success}/{len(results)} 成功")

    if require_all_success and (failed_tts_ids or failed_visual_ids):
        msg = (
            "Stage 3a 要求全量成功但存在失败: "
            f"TTS失败={failed_tts_ids[:8]}{'...' if len(failed_tts_ids) > 8 else ''}, "
            f"视觉失败={failed_visual_ids[:8]}{'...' if len(failed_visual_ids) > 8 else ''}"
        )
        logger.error("[visual_audio_batch] %s", msg)
        raise RuntimeError(msg)

    return {
        "shot_assets": all_assets,
        "current_stage": "stage3_visual_audio_done",
        "trajectory": all_entries,
    }


# ---------------------------------------------------------------------------
# Node: decorate_spotlights
# ---------------------------------------------------------------------------
def node_decorate_spotlights(state: PipelineState) -> dict[str, Any]:
    """Process spotlight shots: template selection → slot filling → check & self-repair.

    Key improvements:
      - Passes already_used_templates for diversity across shots
      - Passes config for loading gold_template_policies + instructions
      - decorate_shot.run() now includes internal check & repair loop
      - Check status correctly reflects passed/failed (not always "success")
      - Persists results to storage_root in reference format
    """
    shot_plan = state.get("shot_plan", [])
    spotlight_count = sum(1 for s in shot_plan if s.get("is_spotlight"))
    logger.info("[decorate_spotlights] 开始: %d 个 spotlight 镜头", spotlight_count)
    config = state.get("config", {})
    max_check_repair = _cfg_int(
        config,
        ("stage3_trajectory", "repair", "max_check_repair_iterations"),
        2,
    )
    llm = _get_llm(state)
    all_assets = dict(state.get("shot_assets", {}))
    all_entries: list[dict[str, Any]] = []
    live_stage = "stage3_decoration"
    _add_stage_marker(all_entries, "stage3_decoration", "Stage 3b: 模板装饰", "start",
                      summary=f"{spotlight_count} 个 spotlight 镜头")
    output_dir = state.get("output_dir", "")
    already_used_templates: list[str] = []

    try:
        spotlight_shots = [s for s in shot_plan if s.get("is_spotlight")]

        # ── fake mode: 批量 1 次全局 LLM plan + 跳过 HyperFrames 渲染 ──
        if _is_fake_mode(config):
            logger.info("[decorate_spotlights] fake 模式: 批量装饰 %d 个 spotlight 镜头", len(spotlight_shots))
            batch_result = decorate_shot.batch_run(
                llm,
                spotlight_shots,
                config=config,
                max_repair_rounds=max_check_repair,
                progress_callback=lambda sub_tool, current, total, label: _emit_progress(
                    "Tool_BatchDecorateSpotlights",
                    sub_tool,
                    current,
                    total,
                    label,
                    live_stage,
                ),
            )

            batch_summary = batch_result.get("summary", "")
            entry = _log_tool(
                "Tool_BatchDecorateSpotlights",
                {
                    "shot_ids": [s.get("shot_id", "") for s in spotlight_shots],
                    "spotlight_count": len(spotlight_shots),
                },
                {
                    "results": batch_result.get("results", {}),
                    "summary": batch_summary,
                    "status": batch_result.get("status", "success"),
                },
                "stage3_decoration",
                thought=f"一口气搞定了 {len(spotlight_shots)} 个 spotlight 的模板装饰——{batch_summary}",
            )
            for i, e in enumerate(entry):
                e.setdefault("metadata", {})
                e["metadata"]["node"] = "decorate_spotlights"
                e["metadata"]["stage"] = live_stage
                e["metadata"]["live_uid"] = f"decorate:batch:{i}"
                all_entries.append(e)
                _emit_live_trajectory(e)

            # ── 写入 per-shot assets + 假渲染（批量汇总 trajectory） ──
            skip_render = _cfg_bool(config, ("execution", "fake_mode", "skip_hyperframes_render"), True)
            render_summary: dict[str, dict] = {}

            for shot in spotlight_shots:
                sid = shot.get("shot_id", "")
                dec_info = batch_result.get("results", {}).get(sid, {})
                decoration = dec_info.get("decoration", {}) if isinstance(dec_info.get("decoration"), dict) else {}
                if sid in all_assets:
                    all_assets[sid]["decoration"] = decoration
                    all_assets[sid]["decoration_verify"] = dec_info.get("verify_result", {})

                # fake render: 不调 HyperFrames，数据写入 assets
                if skip_render:
                    fake_result = fake_utils.fake_render_success(shot, all_assets.get(sid, {}), output_dir)
                    all_assets[sid]["timeline"] = fake_result.get("timeline", {})
                    all_assets[sid]["rendered_path"] = fake_result.get("rendered_path", "")
                    all_assets[sid]["compose_result"] = fake_result
                    # 只收集到汇总 dict，不逐镜头展开 trajectory
                    render_summary[sid] = {
                        "render_status": fake_result.get("render_status", ""),
                        "render_success": fake_result.get("render_success", False),
                    }

                # Persist decoration report
                dec_path = _save_json_artifact(
                    state, f"stage3_decoration/{sid}", "decoration_report.json", dec_info,
                )
                _save_storage_root_decoration(state, sid, {**dec_info, "artifact_path": dec_path})

            # ── 轨迹压缩：假渲染只记录一条汇总 tool call ──
            if skip_render and render_summary:
                render_success_count = sum(1 for v in render_summary.values() if v.get("render_success"))
                render_entries = _log_tool(
                    "Tool_BatchTimelineCompose",
                    {"shot_ids": list(render_summary.keys()), "shot_count": len(render_summary), "render_backend": "fake"},
                    {
                        "shots": render_summary,
                        "render_success_count": render_success_count,
                        "status": "success",
                    },
                    "stage3_decoration_render",
                    thought=f"批量假渲染 {len(render_summary)} 个 spotlight 镜头: {render_success_count}/{len(render_summary)} 成功（标注模式跳过 HyperFrames）",
                )
                for i2, e2 in enumerate(render_entries):
                    e2.setdefault("metadata", {})
                    e2["metadata"]["node"] = "decorate_spotlights"
                    e2["metadata"]["stage"] = live_stage
                    e2["metadata"]["live_uid"] = f"decorate:batch_render:{i2}"
                    all_entries.append(e2)
                    _emit_live_trajectory(e2)

        # ── real mode: 原始 per-shot 装饰流程（轨迹压缩为批量汇总）──
        else:
            batch_results: dict[str, dict] = {}
            for shot in spotlight_shots:
                sid = shot.get("shot_id", "")
                logger.info("[decorate_spotlights] 镜头 %s: 开始模板装饰 (已用模板: %s)",
                            sid, already_used_templates)

                dec_result = decorate_shot.run(
                    llm,
                    shot,
                    already_used_templates=already_used_templates,
                    config=config,
                    max_repair_rounds=max_check_repair,
                )
                decoration = dec_result.get("decoration", {}) if isinstance(dec_result.get("decoration"), dict) else {}
                verify_result = dec_result.get("verify_result", {}) if isinstance(dec_result.get("verify_result"), dict) else {}
                passed = bool(dec_result.get("passed", False))
                repair_rounds = dec_result.get("repair_rounds", 0)

                # Track used templates for diversity
                template_id = decoration.get("template_id", "")
                if template_id and template_id not in already_used_templates:
                    already_used_templates.append(template_id)

                # 收集到批量汇总 dict，不逐镜头展开轨迹
                batch_results[sid] = {
                    "template_id": template_id,
                    "template_type": decoration.get("template_type", ""),
                    "passed": passed,
                    "repair_rounds": repair_rounds,
                    "fallback_used": bool(dec_result.get("fallback_used", False)),
                }

                # Persist decoration report
                dec_path = _save_json_artifact(
                    state,
                    f"stage3_decoration/{sid}",
                    "decoration_report.json",
                    dec_result,
                )
                dec_result = {**dec_result, "artifact_path": dec_path}
                _save_storage_root_decoration(state, sid, dec_result)

                if sid in all_assets:
                    all_assets[sid]["decoration"] = decoration
                    all_assets[sid]["decoration_verify"] = verify_result
                    if _cfg_bool(config, ("stage3_trajectory", "spotlight_judge", "render_in_decoration_stage"), True):
                        compose_result = timeline_compose.run(
                            shot,
                            all_assets[sid],
                            output_dir,
                            config=config,
                            render_backend="hyperframes",
                        )
                        all_assets[sid]["timeline"] = compose_result.get("timeline", {})
                        all_assets[sid]["rendered_path"] = compose_result.get("rendered_path", "")
                        all_assets[sid]["compose_result"] = compose_result

            # ── 轨迹压缩：整个 real-mode 装饰只产出一条批量汇总 ──
            passed_count = sum(1 for v in batch_results.values() if v.get("passed"))
            templates_used = sorted(set(v.get("template_id", "") for v in batch_results.values() if v.get("template_id")))
            batch_entries = _log_tool(
                "Tool_DecorateShots",
                {
                    "shot_ids": list(batch_results.keys()),
                    "spotlight_count": len(spotlight_shots),
                },
                {
                    "shots": batch_results,
                    "passed_count": passed_count,
                    "failed_count": len(spotlight_shots) - passed_count,
                    "templates_used": templates_used,
                    "status": "success",
                },
                "stage3_decoration",
                thought=(
                    f"为 {len(spotlight_shots)} 个 spotlight 镜头完成模板装饰: "
                    f"{passed_count} 个通过自检，使用了 {len(templates_used)} 种模板 {templates_used}，"
                    f"模板多样性保持良好"
                ),
            )
            for i, entry in enumerate(batch_entries):
                entry.setdefault("metadata", {})
                entry["metadata"]["node"] = "decorate_spotlights"
                entry["metadata"]["stage"] = live_stage
                all_entries.append(entry)
    finally:
        llm.close()

    if not any(s.get("is_spotlight") for s in shot_plan):
        all_entries.append({
            "role": "thought",
            "content": "没有 spotlight 镜头，跳过模板装饰阶段",
            "metadata": {"stage": "stage3_decoration", "ts": datetime.now().isoformat()},
        })
    passed_count = sum(
        1
        for shot in shot_plan
        if shot.get("is_spotlight")
        and bool(
            (all_assets.get(str(shot.get("shot_id", "")), {}).get("decoration_verify", {}) or {}).get("passed", False)
        )
    )
    _add_stage_marker(
        all_entries,
        "stage3_decoration",
        "Stage 3b: 模板装饰",
        "end",
        summary=f"完成 {spotlight_count} 个 spotlight 镜头，校验通过 {passed_count} 个",
    )

    try:
        auto_summary = decorate_shot.autodecorate_shot_plan_0907(shot_plan, all_assets, config)
    except Exception as exc:
        logger.warning("[decorate_spotlights] 0907 语义自动接线失败: %s", exc)
        auto_summary = {"enabled": False, "results": {}}
    if auto_summary.get("enabled"):
        decorated = [sid for sid, r in auto_summary.get("results", {}).items()
                     if r.get("status") == "decorated"]
        promoted = [sid for sid, r in auto_summary.get("results", {}).items()
                    if r.get("status") == "decorated" and r.get("promoted")]
        logger.info("[decorate_spotlights] 0907 语义自动装饰 %d 个镜头（新增 spotlight %d 个）",
                    len(decorated), len(promoted))
        all_entries.append({
            "role": "thought",
            "content": (
                f"0907 语义锚点自动装饰 {len(decorated)} 个镜头"
                + (f"，其中 {len(promoted)} 个由强语义提升为 spotlight" if promoted else "")
            ),
            "metadata": {"stage": "stage3_decoration", "ts": datetime.now().isoformat()},
        })

    return {
        "shot_assets": all_assets,
        "shot_plan": shot_plan,
        "current_stage": "stage3_decoration_done",
        "trajectory": all_entries,
    }


# ---------------------------------------------------------------------------
# Node: music_search (BGM retrieval)
# ---------------------------------------------------------------------------
def node_music_search(state: PipelineState) -> dict[str, Any]:
    """Tool_MusicSearch: Search BGM tracks from music library."""
    stage_entries: list[dict[str, Any]] = []
    _add_stage_marker(
        stage_entries,
        "stage3_bgm_search",
        "Stage 3c: BGM 配乐检索",
        "start",
        summary="按全片风格检索或生成占位 BGM 元数据",
    )
    # ``bgm_query`` is the post-production override written by the editor's
    # ``set_bgm``; it wins over the style inferred at segmentation time.
    requested_style = str(
        state.get("bgm_query") or state.get("music_style") or ""
    ).strip()

    cached = _load_cached_json(state, "stage3_bgm", "music_search.json")
    if (
        cached
        and isinstance(cached, dict)
        and cached.get("selected_tracks")
        and cached.get("playlist_policy_version") == 2
        # The cache scope key is derived from the article text, so every BGM
        # search for one project maps to the same file. Reusing it whenever it
        # exists means a user asking for a different style keeps getting the
        # tracks picked on the first run. Only a cache produced by the *same*
        # request may be reused.
        and str(cached.get("requested_style", "") or "").strip() == requested_style
    ):
        logger.info("[music_search] 命中缓存: %s", cached.get("_cache_path", ""))
        entries = _log_tool(
            "Tool_MusicSearch",
            {"cache_hit": True},
            {
                "status": "success",
                "cache_hit": True,
                "artifact_path": cached.get("_cache_path", ""),
                "selected_count": len(cached.get("selected_tracks", [])),
            },
            "stage3_bgm_search",
            thought="命中BGM检索缓存，跳过音乐检索",
        )
        _add_stage_marker(
            entries,
            "stage3_bgm_search",
            "Stage 3c: BGM 配乐检索",
            "end",
            summary=f"缓存命中，复用 {len(cached.get('selected_tracks', []))} 条 BGM",
        )
        return {
            "bgm_tracks": cached.get("selected_tracks", []),
            "bgm_search_result": cached,
            "current_stage": "stage3_bgm_done",
            "trajectory": stage_entries + entries,
        }

    music_style = requested_style
    if not music_style:
        music_style = "中速现代纪录片纯音乐，钢琴、弦乐与轻电子节拍，克制清晰"

    # Estimate total video duration from shot assets
    shot_assets = state.get("shot_assets", {})
    total_duration = 0.0
    for sid, assets in shot_assets.items():
        audio_dur = float(assets.get("audio_meta", {}).get("duration_sec", 0) or 0)
        total_duration += max(audio_dur, 8.0)
    if total_duration <= 0:
        total_duration = len(state.get("shot_plan", [])) * 10.0

    logger.info(
        "[music_search] 开始: style=%s, target_duration=%.1fs",
        music_style[:60], total_duration,
    )

    config = state.get("config", {})
    result = music_search.run(
        music_style=music_style,
        total_video_duration=total_duration,
        config=config,
    )
    # Stamp the request this artifact answers. `music_search.run` normalises the
    # style it was given, so its own `music_style` field cannot be compared back
    # against the raw request when deciding whether the cache is still valid.
    result["requested_style"] = requested_style

    logger.info(
        "[music_search] 完成: %d tracks selected, total_bgm=%.1fs",
        result.get("selected_count", 0),
        result.get("total_bgm_duration_sec", 0),
    )

    entries = _log_tool(
        "Tool_MusicSearch",
        {"music_style": music_style, "target_duration_sec": round(total_duration, 2)},
        {
            "selected_count": result.get("selected_count", 0),
            "total_bgm_duration_sec": result.get("total_bgm_duration_sec", 0),
            "duration_covered": result.get("duration_covered", False),
            "tracks": [
                {"name": t.get("name", ""), "duration_sec": t.get("duration_sec", 0), "score": t.get("score", 0)}
                for t in result.get("selected_tracks", [])
            ],
            "status": result.get("status", ""),
        },
        "stage3_bgm_search",
        thought=f"根据全片的音乐风格描述 '{music_style[:40]}' 从音乐库里扒了几首 BGM，凑够了 {total_duration:.0f}s 的时长",
    )
    bgm_artifact_path = _save_json_artifact(
        state,
        "stage3_bgm",
        "music_search.json",
        result,
    )
    entries.append({
        "role": "thought",
        "content": f"BGM检索产物已落盘: {bgm_artifact_path}",
        "metadata": {"stage": "stage3_bgm_search", "ts": datetime.now().isoformat()},
    })
    _add_stage_marker(
        entries,
        "stage3_bgm_search",
        "Stage 3c: BGM 配乐检索",
        "end",
        summary=f"选中 {result.get('selected_count', 0)} 条 BGM",
    )

    return {
        "bgm_tracks": result.get("selected_tracks", []),
        "bgm_search_result": result,
        "current_stage": "stage3_bgm_done",
        "trajectory": stage_entries + entries,
    }


# ---------------------------------------------------------------------------
# Node: severity_rank (pre-sort shots by problem severity, select top 30%)
# ---------------------------------------------------------------------------
SEVERITY_RANK_SYSTEM = """\
你是视频质量分析师。你需要对镜头进行问题严重程度预排序，纯文本分析，不需要看视频。

硬性输出规则:
1. 只能输出一个合法 JSON object。
2. scores 必须是 JSON array，每一项是 JSON object。
3. 必须覆盖输入中的每一个 shot_id，不能缺项，不能额外添加 shot_id。
4. severity_score 必须是 0.0-1.0 的数字；除非证据完全一致，不要给所有镜头相同分数。
5. 不要输出推理过程，只输出 JSON。"""

SEVERITY_RANK_TEMPLATE = """\
请对以下镜头做文本层面的问题严重程度打分，0-1，分越高问题越严重。

判断维度:
1. shot_caption 与 visual_query/vocal_script 的语义不匹配程度
2. video_duration_sec 是否短于 audio_duration_sec；长素材可裁剪，不算问题
3. visual_query 是否模糊、过宽或不能检索到所需画面
4. spotlight 镜头的装饰复杂度和失败代价

镜头列表:
{shots_json}

输出(严格JSON):
{{
  "scores": [
    {{
      "shot_id": "镜头ID",
      "severity_score": 0.0-1.0,
      "rank_reason": "不超过20字"
    }}
  ]
}}"""


def node_severity_rank(state: PipelineState) -> dict[str, Any]:
    """Pre-sort all shots by problem severity; only top_ratio enter repair."""
    config = state.get("config", {})
    shot_plan = _dict_list(state.get("shot_plan", []))
    all_assets = dict(state.get("shot_assets", {}))
    all_entries: list[dict[str, Any]] = []
    _add_stage_marker(
        all_entries,
        "stage4_severity_rank",
        "Stage 4a: 问题严重度路由",
        "start",
        summary=f"对 {len(shot_plan)} 个镜头做元数据预排序",
    )

    enable = _cfg_bool(config, ("stage3_trajectory", "severity_rank", "enable"), True)
    top_ratio = _cfg_float(config, ("stage3_trajectory", "severity_rank", "top_ratio"), 0.30)
    min_shots = _cfg_int(config, ("stage3_trajectory", "severity_rank", "min_shots_to_process"), 3)
    temperature = _cfg_float(config, ("stage3_trajectory", "severity_rank", "llm_temperature"), 0.1)
    max_retries = _cfg_int(config, ("stage3_trajectory", "severity_rank", "max_retries"), 2)
    allow_heuristic_fallback = _cfg_bool(
        config,
        ("stage3_trajectory", "severity_rank", "allow_heuristic_fallback"),
        False,
    )

    if not enable or not shot_plan:
        _emit_progress("Tool_SeverityRank", "严重度路由", 1, 1, "严重度预排序已跳过", "stage4_severity_rank")
        _add_stage_marker(
            all_entries,
            "stage4_severity_rank",
            "Stage 4a: 问题严重度路由",
            "end",
            summary="已跳过严重度预排序",
        )
        return {
            "severity_scores": {},
            "severity_ranked_shots": [str(s.get("shot_id", "")) for s in shot_plan],
            "current_stage": "severity_rank_skipped",
            "trajectory": all_entries,
        }

    _emit_progress("Tool_SeverityRank", "严重度路由", 0, 2, f"准备分析 {len(shot_plan)} 个镜头", "stage4_severity_rank")
    compact_shots: list[dict[str, Any]] = []
    for shot in shot_plan:
        sid = str(shot.get("shot_id", ""))
        assets = all_assets.get(sid, {}) if isinstance(all_assets.get(sid), dict) else {}
        clip_meta = assets.get("clip_meta", {}) if isinstance(assets.get("clip_meta"), dict) else {}
        audio_meta = assets.get("audio_meta", {}) if isinstance(assets.get("audio_meta"), dict) else {}
        compact_shots.append({
            "shot_id": sid,
            "vocal_script": str(shot.get("vocal_script", ""))[:200],
            "visual_query": str(shot.get("visual_query", "")),
            "shot_caption": str(clip_meta.get("caption", clip_meta.get("caption_zh", "")))[:200],
            "audio_duration_sec": _safe_float(audio_meta.get("duration_sec"), 0.0),
            "video_duration_sec": _safe_float(clip_meta.get("duration", clip_meta.get("duration_sec")), 0.0),
            "match_score": _safe_float(clip_meta.get("match_score", clip_meta.get("score", clip_meta.get("similarity"))), 0.0),
            "is_spotlight": bool(shot.get("is_spotlight", False)),
        })

    raw: dict[str, Any] = {}
    validation_error = ""
    scoring_source = "llm_strict"
    if _is_fake_mode(config):
        # 标注加速模式直接复用真实 clip/audio 元数据 + decoration verify，
        # 不再为严重度预排序额外发起一次 LLM 请求。
        scoring_source = "metadata_rules"
        raw = {
            "scores": [
                {
                    "shot_id": str(shot.get("shot_id", "")),
                    "severity_score": _heuristic_severity(
                        shot,
                        all_assets.get(str(shot.get("shot_id", "")), {}),
                    ),
                    "rank_reason": "真实素材/TTS/装饰校验元数据",
                }
                for shot in shot_plan
            ]
        }
    else:
        llm = _get_llm(state, service=_resolve_judge_service(config))
        try:
            for attempt in range(max_retries + 1):
                prompt = SEVERITY_RANK_TEMPLATE.format(
                    shots_json=json.dumps(compact_shots, ensure_ascii=False, separators=(",", ":"))
                )
                try:
                    candidate = llm.chat_json(SEVERITY_RANK_SYSTEM, prompt, temperature=temperature)
                except Exception as exc:
                    candidate = {}
                    validation_error = f"llm_exception: {exc}"
                rows, validation_error = _validate_severity_rank_output(candidate, [s["shot_id"] for s in compact_shots])
                if rows:
                    raw = {"scores": rows}
                    break
                logger.warning("[severity_rank] LLM输出无效 attempt=%d/%d: %s", attempt + 1, max_retries + 1, validation_error)
        finally:
            llm.close()

    raw = raw if isinstance(raw, dict) else {}
    score_rows = raw.get("scores", [])
    if not isinstance(score_rows, list):
        score_rows = []

    by_sid: dict[str, float] = {}
    for row in score_rows:
        if isinstance(row, dict) and row.get("shot_id"):
            by_sid[str(row["shot_id"])] = round(max(0.0, min(1.0, _safe_float(row.get("severity_score"), 0.0))), 3)

    if len(by_sid) != len(compact_shots):
        if not allow_heuristic_fallback:
            raise RuntimeError(f"SeverityRank 模型未返回完整有效分数，已停止避免统一兜底分数: {validation_error}")
        logger.warning("[severity_rank] 使用启发式兜底排序: %s", validation_error)

    severity_scores: dict[str, Any] = {}
    for shot in shot_plan:
        sid = str(shot.get("shot_id", ""))
        if sid in by_sid:
            score = by_sid[sid]
        elif allow_heuristic_fallback:
            score = _heuristic_severity(shot, all_assets.get(sid, {}))
        else:
            raise RuntimeError(f"SeverityRank 缺少 shot_id={sid} 的模型分数")
        severity_scores[sid] = {"score": score, "rank": 0, "in_top_ratio": False}

    sorted_sids = sorted(severity_scores.keys(), key=lambda s: severity_scores[s]["score"], reverse=True)
    num_to_process = max(min_shots, int(len(sorted_sids) * top_ratio + 0.5))
    num_to_process = min(num_to_process, len(sorted_sids))

    for rank, sid in enumerate(sorted_sids):
        severity_scores[sid]["rank"] = rank + 1
        severity_scores[sid]["in_top_ratio"] = rank < num_to_process

    _emit_progress(
        "Tool_SeverityRank",
        "严重度路由",
        2,
        2,
        f"前 {num_to_process}/{len(sorted_sids)} 个镜头进入质检修复",
        "stage4_severity_rank",
    )

    all_entries.extend(_log_tool(
        "Tool_SeverityRank",
        {"shot_count": len(shot_plan), "top_ratio": top_ratio, "min_shots": min_shots},
        {
            "ranked_count": len(sorted_sids),
            "process_count": num_to_process,
            "skip_count": len(sorted_sids) - num_to_process,
            "top_shots": sorted_sids[:num_to_process],
            "severity_scores": severity_scores,
            "scoring_source": scoring_source if len(by_sid) == len(compact_shots) else "heuristic_fallback",
            "status": "success",
        },
        "stage4_severity_rank",
        thought=f"对 {len(shot_plan)} 个镜头按问题严重程度预排序，取前 {num_to_process} 个进入修复流程",
    ))

    _save_json_artifact(
        state,
        "stage4_severity_rank",
        "severity_scores.json",
        severity_scores,
    )
    _add_stage_marker(
        all_entries,
        "stage4_severity_rank",
        "Stage 4a: 问题严重度路由",
        "end",
        summary=f"前 {num_to_process}/{len(sorted_sids)} 个镜头进入质检修复",
    )

    return {
        "severity_scores": severity_scores,
        "severity_ranked_shots": sorted_sids,
        "current_stage": "severity_rank_done",
        "trajectory": all_entries,
    }


def _heuristic_severity(shot: dict[str, Any], assets: Any) -> float:
    """Fallback heuristic when LLM scoring is unavailable."""
    score = 0.3
    assets = assets if isinstance(assets, dict) else {}
    clip_meta = assets.get("clip_meta", {}) if isinstance(assets.get("clip_meta"), dict) else {}
    audio_meta = assets.get("audio_meta", {}) if isinstance(assets.get("audio_meta"), dict) else {}

    audio_dur = _safe_float(audio_meta.get("duration_sec"), 0.0)
    video_dur = _safe_float(clip_meta.get("duration", clip_meta.get("duration_sec")), 0.0)
    if audio_dur > 0 and video_dur > 0:
        ratio = max(0.0, audio_dur - video_dur) / max(audio_dur, 1.0)
        if ratio > 0.5:
            score += 0.25
        elif ratio > 0.3:
            score += 0.1

    if not shot.get("visual_query"):
        score += 0.2
    if not clip_meta.get("caption") and not clip_meta.get("caption_zh"):
        score += 0.15
    decoration_verify = assets.get("decoration_verify", {}) if isinstance(assets.get("decoration_verify"), dict) else {}
    if decoration_verify:
        verify_score = _safe_float(decoration_verify.get("score"), 1.0)
        score += max(0.0, 1.0 - verify_score) * 0.35
        if not decoration_verify.get("passed", False):
            score += 0.1

    return round(min(score, 1.0), 3)


def _validate_severity_rank_output(raw: Any, expected_ids: list[str]) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(raw, dict):
        return [], "output_not_object"
    rows = raw.get("scores", [])
    if not isinstance(rows, list):
        return [], "scores_not_array"
    expected = [str(sid) for sid in expected_ids]
    expected_set = set(expected)
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            return [], "score_row_not_object"
        sid = str(row.get("shot_id", "")).strip()
        if sid not in expected_set:
            return [], f"unexpected_shot_id:{sid}"
        if sid in seen:
            return [], f"duplicate_shot_id:{sid}"
        value = row.get("severity_score")
        if not isinstance(value, (int, float)):
            return [], f"non_numeric_score:{sid}"
        score = float(value)
        if score < 0.0 or score > 1.0:
            return [], f"score_out_of_range:{sid}:{score}"
        seen.add(sid)
        normalized.append({
            "shot_id": sid,
            "severity_score": round(score, 3),
            "rank_reason": str(row.get("rank_reason") or row.get("top_issues") or "")[:80],
        })
    missing = [sid for sid in expected if sid not in seen]
    if missing:
        return [], f"missing_shot_ids:{','.join(missing[:5])}"
    if len(normalized) > 1:
        scores = [row["severity_score"] for row in normalized]
        if max(scores) - min(scores) < 0.001:
            return [], "score_collapse_all_equal"
    return normalized, ""


# ---------------------------------------------------------------------------
# Node: compose_and_judge (layered batch judge + bounded auto-repair)
# ---------------------------------------------------------------------------
def node_compose_and_judge(state: PipelineState) -> dict[str, Any]:
    """Stage 4: pure-text severity routing + batch observation/reflection/repair."""
    shot_plan = state.get("shot_plan", [])
    idx = state.get("compose_shot_idx", 0)
    all_assets = dict(state.get("shot_assets", {}))
    output_dir = state.get("output_dir", "")
    config = state.get("config", {})
    all_entries: list[dict[str, Any]] = []
    severity_scores = state.get("severity_scores", {}) or {}
    batch_size = max(1, _cfg_int(config, ("stage3_trajectory", "judge", "batch_size"), 6))

    if idx >= len(shot_plan):
        return {
            "current_stage": "stage4_all_done",
            "trajectory": [],
        }

    text_judge_threshold = _cfg_float(
        config,
        ("stage3_trajectory", "text_judge", "pass_threshold"),
        0.70,
    )
    text_judge_enabled = _cfg_bool(config, ("stage3_trajectory", "text_judge", "enable"), True)
    max_repair_rounds = _cfg_int(
        config,
        ("stage3_trajectory", "repair", "max_repair_iterations"),
        2,
    )
    if not _cfg_get(config, ("stage3_trajectory", "repair", "enable"), True):
        max_repair_rounds = 0
    max_repair_rounds = min(
        max_repair_rounds,
        max(0, _cfg_int(config, ("stage3_trajectory", "judge", "max_repair_rounds_per_shot"), 1)),
    )
    judge_enabled = bool(_cfg_get(config, ("stage3_trajectory", "judge", "enable"), True))
    severity_rank_enabled = _cfg_bool(config, ("stage3_trajectory", "severity_rank", "enable"), True)

    # Stage 4 is intentionally trajectory-compact: process the remaining shots
    # in one node invocation and expose progress through live progress events.
    batch_end = len(shot_plan)
    batch = shot_plan[idx:batch_end]
    _add_stage_marker(
        all_entries,
        "stage4_compose_judge",
        f"Stage 4b: 合成质检 ({idx + 1}-{batch_end})",
        "start",
        summary="纯文本严重度路由 + Observation/Reflection/Repair(batch)",
    )
    new_shot_plan = list(shot_plan)
    logger.info(
        "[compose_and_judge] 维度化批处理开始: shots=%d-%d/%d judge_batch_size=%d",
        idx + 1,
        batch_end,
        len(shot_plan),
        batch_size,
    )

    batch_items: list[dict[str, Any]] = []
    for offset, shot in enumerate(batch):
        absolute_idx = idx + offset
        sid = str(shot.get("shot_id", "") or f"shot_{absolute_idx + 1}")
        assets = dict(all_assets.get(sid, {"shot_id": sid}))

        in_top_ratio = True
        if severity_rank_enabled and severity_scores:
            shot_severity = severity_scores.get(sid, {})
            in_top_ratio = bool(shot_severity.get("in_top_ratio", True)) if isinstance(shot_severity, dict) else True

        batch_items.append({
            "absolute_idx": absolute_idx,
            "shot": shot,
            "sid": sid,
            "assets": assets,
            "needs_repair_route": in_top_ratio,
        })

    _process_text_routed_batch(
        state=state,
        items=batch_items,
        all_assets=all_assets,
        new_shot_plan=new_shot_plan,
        all_entries=all_entries,
        output_dir=output_dir,
        config=config,
        text_judge_threshold=text_judge_threshold,
        text_judge_enabled=text_judge_enabled,
        judge_enabled=judge_enabled,
        max_repair_rounds=max_repair_rounds,
    )

    approved_count = sum(
        1
        for shot in batch
        if all_assets.get(str(shot.get("shot_id", "")), {}).get("approved", False)
    )
    summary_entry = {
        "role": "thought",
        "content": (
            f"这批镜头 {idx+1}-{batch_end} 已完成纯文本路由质检——"
            f"需要修复路由 {sum(1 for it in batch_items if it.get('needs_repair_route'))} 个，"
            f"直接通过路由 {sum(1 for it in batch_items if not it.get('needs_repair_route'))} 个，"
            f"最终通过 {approved_count}/{len(batch)}"
        ),
        "metadata": {
            "stage": "stage4_text_routed_batch",
            "batch_start": idx,
            "batch_end": batch_end,
            "ts": datetime.now().isoformat(),
        },
    }
    all_entries.append(summary_entry)
    _add_stage_marker(all_entries, "stage4_compose_judge", f"Stage 4: 合成质检 (批次 {idx+1}-{batch_end})", "end",
                      summary=f"通过 {approved_count}/{len(batch)}")

    return {
        "shot_assets": all_assets,
        "shot_plan": new_shot_plan,
        "compose_shot_idx": batch_end,
        "judge_round": 0,
        "current_stage": f"stage4_batch_{idx + 1}_{batch_end}_done",
        "trajectory": all_entries,
    }


def _is_0907_compose(config: dict[str, Any] | None) -> bool:
    config = config or {}
    if str(config.get("policy_version", "") or "").strip() == "0907":
        return True
    newvis = config.get("newvis0907")
    return isinstance(newvis, dict) and bool(newvis.get("enabled"))


def _compose_config_with_occupancy(
    config: dict[str, Any],
    sid: str,
    all_assets: dict[str, Any],
) -> dict[str, Any]:
    """For 0907, hand the composer a config *copy* carrying this run's private
    global occupancy (all other shots' chosen footage) so playlists never reuse
    a source across the film / adjacent shots. Runtime occupancy is never written
    into the global studio config. Legacy composes are returned unchanged."""
    if not _is_0907_compose(config):
        return config
    try:
        from studio.newvis0907 import footage as _footage
        occ = _footage.build_occupancy(
            {"shot_plan": [], "shot_assets": all_assets}, exclude_shot_ids=[sid]
        )
        runtime = config.get("newvis0907_runtime")
        base = runtime.get("base_occupancy") if isinstance(runtime, dict) else None
        if isinstance(base, _footage.Occupancy):
            occ = _footage.Occupancy(list(base.identities) + list(occ.identities))
    except Exception:
        return config
    return {**config, "newvis0907_runtime": {"occupancy": occ}}


def _process_text_routed_batch(
    state: PipelineState,
    items: list[dict[str, Any]],
    all_assets: dict[str, Any],
    new_shot_plan: list[dict[str, Any]],
    all_entries: list[dict[str, Any]],
    output_dir: str,
    config: dict[str, Any],
    text_judge_threshold: float,
    text_judge_enabled: bool,
    judge_enabled: bool,
    max_repair_rounds: int,
) -> None:
    """Stage4 compact loop:
    compose(all) -> per-dimension judge -> repair failed dimensions -> rejudge.
    """
    if not items:
        return
    is_fake = _is_fake_mode(config)

    skip_compose = bool(state.get("_skip_compose", False))

    compose_summary: dict[str, dict[str, Any]] = {}
    for i, item in enumerate(items):
        sid = item["sid"]
        shot = item["shot"]
        assets = item["assets"]

        existing_rendered = str(assets.get("rendered_path", "") or "").strip()
        if skip_compose or existing_rendered:
            compose_result = assets.get("compose_result", {}) or {
                "render_success": True,
                "rendered_path": existing_rendered,
                "render_status": "skipped" if skip_compose else "already_rendered",
                "timeline": assets.get("timeline", {}),
            }
        elif is_fake:
            _log_progress(i + 1, len(items), "stage4 compose")
            _emit_progress("Tool_BatchTimelineCompose", "模拟合成", i + 1, len(items), f"镜头合成: {sid}", "stage4_compose")
            compose_result = fake_utils.fake_render_success(shot, assets, output_dir)
            assets["timeline"] = compose_result.get("timeline", {})
            assets["rendered_path"] = compose_result.get("rendered_path", "")
        else:
            _log_progress(i + 1, len(items), "stage4 compose")
            _emit_progress("Tool_BatchTimelineCompose", "合成渲染", i + 1, len(items), f"镜头合成: {sid}", "stage4_compose")
            compose_result = timeline_compose.run(
                shot, assets, output_dir,
                config=_compose_config_with_occupancy(config, sid, all_assets),
            )
            assets["timeline"] = compose_result.get("timeline", {})
            assets["rendered_path"] = compose_result.get("rendered_path", "")

        assets["compose_result"] = compose_result
        item["assets"] = assets
        item["compose_result"] = compose_result
        all_assets[sid] = assets
        compose_summary[sid] = {
            "render_success": compose_result.get("render_success", True) if skip_compose else compose_result.get("render_success", False),
            "render_status": compose_result.get("render_status", "skipped" if skip_compose else ""),
            "needs_repair_route": bool(item.get("needs_repair_route", True)),
        }

    compose_ok = sum(1 for value in compose_summary.values() if value.get("render_success"))
    all_entries.extend(_log_tool(
        "Tool_BatchTimelineCompose",
        {"shot_ids": list(compose_summary.keys()), "shot_count": len(items), "render_backend": "fake" if is_fake else "real"},
        {"shots": compose_summary, "render_success_count": compose_ok, "status": "success"},
        "stage4_compose",
        thought=f"批量合成 {len(items)} 个镜头: {compose_ok}/{len(items)} 成功",
    ))

    repair_route_items = [item for item in items if bool(item.get("needs_repair_route", True))]
    bypass_items = [item for item in items if not bool(item.get("needs_repair_route", True))]

    for item in bypass_items:
        sid = item["sid"]
        assets = item["assets"]
        compose_result = item.get("compose_result", {})
        approved = bool(compose_result.get("render_success", False))
        judge_result = {
            "shot_id": sid,
            "passed": approved,
            "overall_score": 1.0 if approved else 0.0,
            "status": "severity_route_bypass" if approved else "render_failed",
            "summary": "严重度路由未命中修复阈值，直接通过",
            "judge_type": "severity_skip",
            "observation_source": "severity_routing",
        }
        judge_result["dimension_observation"] = _stage4_dimension_observation(
            judge_result,
            pass_threshold=text_judge_threshold,
            assume_passed=approved,
            config=config,
        )
        assets["approved"] = approved
        assets["judge_result"] = judge_result
        all_assets[sid] = assets

    if not repair_route_items:
        return

    if not (text_judge_enabled and judge_enabled):
        for item in repair_route_items:
            sid = item["sid"]
            assets = item["assets"]
            compose_result = item.get("compose_result", {})
            approved = bool(compose_result.get("render_success", False))
            assets["approved"] = approved
            assets["judge_result"] = {
                "shot_id": sid,
                "passed": approved,
                "overall_score": 1.0 if approved else 0.0,
                "status": "text_judge_disabled" if approved else "render_failed",
                "summary": "文本质检关闭，按渲染结果通过",
                "judge_type": "text_quick",
                "observation_source": "text_judge_disabled",
            }
            assets["judge_result"]["dimension_observation"] = _stage4_dimension_observation(
                assets["judge_result"],
                pass_threshold=text_judge_threshold,
                assume_passed=approved,
                config=config,
            )
            all_assets[sid] = assets
        return

    items_to_judge = repair_route_items
    previous_failed_dimensions: dict[str, list[str]] = {}
    round_num = 0
    last_failed_count = 0
    while items_to_judge:
        _emit_progress(
            "Tool_Stage4DimensionJudge",
            "维度质检",
            min(round_num + 1, max_repair_rounds + 1),
            max_repair_rounds + 1,
            f"第 {round_num + 1} 轮批量 judge: {len(items_to_judge)} 个镜头",
            "stage4_compose_judge",
        )
        llm = _get_llm(state, service=_resolve_judge_service(config))
        try:
            text_judge_results = text_quick_judge.batch_run(
                llm,
                [{"shot": item["shot"], "sid": item["sid"], "assets": item["assets"]} for item in items_to_judge],
                pass_threshold=text_judge_threshold,
                focus_decoration_slots=_stage4_focus_decoration_slots(config),
            )
        except Exception as exc:
            logger.warning("[stage4_text_routed] text_quick_judge batch failed: %s", exc)
            text_judge_results = {}
        finally:
            llm.close()

        failed_items: list[dict[str, Any]] = []
        judge_rows: dict[str, dict[str, Any]] = {}
        passed_count = 0
        for item in items_to_judge:
            sid = item["sid"]
            shot = item["shot"]
            assets = item["assets"]
            judge_result = text_judge_results.get(sid)
            if judge_result is None:
                judge_result = text_quick_judge._fallback_judge(shot, assets)
                judge_result["fallback_used"] = True
                judge_result = text_quick_judge._finalize_result(
                    judge_result, shot, assets, text_judge_threshold, "batch_result_missing"
                )
            dim_obs_full = _stage4_dimension_observation(
                judge_result,
                pass_threshold=text_judge_threshold,
                config=config,
            )
            target_recheck_dims = previous_failed_dimensions.get(sid) or []
            if target_recheck_dims:
                previous_obs = item.get("judge_result", {}).get("dimension_observation", {})
                dim_obs = {
                    dim: (
                        dim_obs_full.get(dim, {})
                        if dim in target_recheck_dims
                        else previous_obs.get(dim, dim_obs_full.get(dim, {}))
                    )
                    for dim in STAGE4_DIMENSIONS
                }
            else:
                dim_obs = dim_obs_full
            failed_dims = _stage4_failed_dimensions(dim_obs)
            judge_result = {
                **judge_result,
                "dimension_observation": dim_obs,
                "failed_dimensions": failed_dims,
                "rechecked_dimensions": target_recheck_dims or list(dim_obs.keys()),
                "repair_round": round_num,
            }
            judge_artifact = _save_stage4_judge_round(state, sid, round_num, judge_result)
            assets["judge_result"] = {**judge_result, "artifact_path": judge_artifact}
            assets["approved"] = (
                bool(item.get("compose_result", {}).get("render_success", False))
                and not failed_dims
                and not bool(judge_result.get("fallback_used", False))
            )
            all_assets[sid] = assets
            if assets["approved"]:
                passed_count += 1
            else:
                failed_items.append({**item, "assets": assets, "judge_result": judge_result, "failed_dimensions": failed_dims})
            judge_rows[sid] = {
                "passed": assets["approved"],
                "overall_score": judge_result.get("overall_score", 0.0),
                "failed_dimensions": failed_dims,
                "dimension_scores": {
                    dim: data.get("score", 0.0)
                    for dim, data in dim_obs.items()
                    if isinstance(data, dict)
                },
            }

        all_entries.extend(_log_tool(
            "Tool_Stage4DimensionJudge",
            {
                "shot_ids": [item["sid"] for item in items_to_judge],
                "threshold": text_judge_threshold,
                "judge_round": round_num,
                "dimensions": STAGE4_DIMENSIONS,
                "ignored_dimensions": sorted(_stage4_ignored_dimensions(config)),
                "focus_decoration_slots": _stage4_focus_decoration_slots(config),
            },
            {
                "shot_count": len(items_to_judge),
                "passed_count": passed_count,
                "failed_count": len(failed_items),
                "shots": judge_rows,
                "status": "success",
            },
            "stage4_dimension_judge",
            thought=(
                f"第 {round_num + 1} 轮按 visual_match、audio_sync、decoration_quality、timing_alignment "
                f"四个维度批量质检 {len(items_to_judge)} 个镜头，后续只修失败维度"
            ),
        ))

        last_failed_count = len(failed_items)
        if not failed_items:
            return
        if round_num >= max_repair_rounds:
            for item in failed_items:
                _mark_stage4_repair_exhausted(
                    all_assets,
                    item["sid"],
                    item["assets"],
                    item["judge_result"],
                    repair_round=round_num,
                    repairable_threshold=0.45,
                    all_entries=all_entries,
                )
            return

        repair_ratio = _safe_float(
            _cfg_get(config, ("stage4_dimension_repair", "max_repair_ratio"), 0.10),
            0.10,
        )
        repair_ratio = max(0.0, min(1.0, repair_ratio))
        total_shots = max(1, len(state.get("shot_plan", []) or items_to_judge))
        max_repair_count = max(1, math.ceil(total_shots * repair_ratio)) if repair_ratio > 0 else 0
        decoration_failed_items = [
            item for item in failed_items
            if "decoration_quality" in (item.get("failed_dimensions") or [])
        ]
        skipped_repair_count = len(failed_items) - len(decoration_failed_items)
        failed_items = decoration_failed_items[:max_repair_count]
        skipped_repair_count += max(0, len(decoration_failed_items) - len(failed_items))
        if skipped_repair_count:
            all_entries.append({
                "role": "thought",
                "content": (
                    f"[REPAIR_SCOPE] 仅修复 decoration_quality（模板/槽位）问题，"
                    f"本轮最多抽样 {max_repair_count} 个镜头；跳过 {skipped_repair_count} 个非抽样或非装饰问题。"
                ),
                "metadata": {
                    "stage": "stage4_dimension_repair",
                    "phase": "repair_scope",
                    "round": round_num,
                    "max_repair_ratio": repair_ratio,
                    "max_repair_count": max_repair_count,
                    "skipped_repair_count": skipped_repair_count,
                    "ts": datetime.now().isoformat(),
                },
            })
        if not failed_items:
            return

        observation = _stage4_batch_failure_summary(failed_items)
        repair_plans: dict[str, dict[str, Any]] = {}
        reflection_parts: list[str] = []
        for item in failed_items:
            sid = item["sid"]
            plan = {"strategy": "redecorate", "strategies": ["redecorate"]}
            repair_plans[sid] = plan
            reflection_parts.append(
                f"{sid}: {','.join(item.get('failed_dimensions', []))} -> {','.join(plan.get('strategies', [plan.get('strategy', 'retime_timeline')]))}"
            )
        all_entries.append({
            "role": "thought",
            "content": (
                f"[OBSERVATION] 第 {round_num + 1} 轮发现 {len(failed_items)} 个镜头仍有失败维度：{observation}。"
                f"[REFLECTION] " + " | ".join(reflection_parts[:8]) + (" ..." if len(reflection_parts) > 8 else "")
            ),
            "metadata": {
                "stage": "stage4_dimension_repair",
                "phase": "observation_reflection",
                "round": round_num,
                "failed_count": len(failed_items),
                "ts": datetime.now().isoformat(),
            },
        })

        repaired_items: list[dict[str, Any]] = []
        repair_rows: dict[str, dict[str, Any]] = {}
        for i, item in enumerate(failed_items):
            sid = item["sid"]
            absolute_idx = item["absolute_idx"]
            plan = repair_plans.get(sid, {"strategy": "retime_timeline", "strategies": ["retime_timeline"]})
            _emit_progress(
                "Tool_Stage4DimensionalRepair",
                "最小化修复",
                i + 1,
                len(failed_items),
                f"镜头 {sid}: {','.join(plan.get('strategies', [plan.get('strategy', 'retime_timeline')]))}",
                "stage4_dimension_repair",
            )
            repair_payload = _apply_stage4_auto_patch(
                state,
                item["shot"],
                item["assets"],
                item.get("compose_result", {}),
                item.get("judge_result", {}),
                repair_round=round_num,
                max_repair_rounds=max_repair_rounds,
                fix_changes=plan,
            )
            updated_shot = repair_payload["shot"]
            updated_assets = repair_payload["assets"]
            new_shot_plan[absolute_idx] = updated_shot

            if state.get("_skip_compose", False):
                compose_result = {
                    **(updated_assets.get("compose_result", {}) if isinstance(updated_assets.get("compose_result"), dict) else {}),
                    "render_success": True,
                    "rendered_path": updated_assets.get("rendered_path", ""),
                    "render_status": "skipped",
                    "timeline": updated_assets.get("timeline", {}),
                    "partial_reuse": {
                        "changed_layers": ["decoration"],
                        "reused_layers": ["visual", "audio", "timing"],
                        "strategies": plan.get("strategies", [plan.get("strategy", "redecorate")]),
                        "note": "Stage4 修复只更新模板/槽位；最终渲染统一留给 Tool_BatchTimelineCompose。",
                    },
                }
            else:
                compose_result = _stage4_compose_after_repair(
                    updated_shot,
                    updated_assets,
                    output_dir,
                    config,
                    strategies=plan.get("strategies", [plan.get("strategy", "retime_timeline")]),
                    is_fake=is_fake,
                )
            updated_assets["timeline"] = compose_result.get("timeline", {})
            updated_assets["rendered_path"] = compose_result.get("rendered_path", "")
            updated_assets["compose_result"] = compose_result
            updated_assets["approved"] = False
            all_assets[sid] = updated_assets
            failed_dimensions = item.get("failed_dimensions", [])
            previous_failed_dimensions[sid] = failed_dimensions
            repaired_items.append({
                **item,
                "shot": updated_shot,
                "assets": updated_assets,
                "compose_result": compose_result,
            })
            repair_rows[sid] = {
                "target_dimensions": failed_dimensions,
                "strategies": plan.get("strategies", [plan.get("strategy", "retime_timeline")]),
                "changed_layers": _stage4_changed_layers(plan.get("strategies", [plan.get("strategy", "retime_timeline")])),
                "reused_layers": compose_result.get("partial_reuse", {}).get("reused_layers", []),
                "render_success": bool(compose_result.get("render_success", False)),
                "reran_tools": repair_payload.get("reran_tools", []),
            }

        repaired_render_success = sum(1 for row in repair_rows.values() if row.get("render_success"))
        all_entries.extend(_log_tool(
            "Tool_Stage4DimensionalRepair",
            {
                "shot_ids": list(repair_rows.keys()),
                "repair_round": round_num + 1,
                "minimal_repair": True,
            },
            {
                "shot_count": len(repair_rows),
                "render_success_count": repaired_render_success,
                "failed_count": len(repair_rows) - repaired_render_success,
                "shots": repair_rows,
                "status": "success" if repaired_render_success == len(repair_rows) else "partial_failure",
            },
            "stage4_dimension_repair",
            thought=f"只重跑失败维度对应的上游工具，并在合成时复用未变化的视觉、音频或装饰层",
        ))
        items_to_judge = repaired_items
        round_num += 1

    if last_failed_count:
        all_entries.append({
            "role": "thought",
            "content": f"[VERIFY] Stage 4 维度化修复循环结束，仍有 {last_failed_count} 个镜头未完全通过。",
            "metadata": {"stage": "stage4_dimension_repair", "phase": "verify", "ts": datetime.now().isoformat()},
        })


def _build_text_batch_repair_plan(
    shot: dict[str, Any],
    assets: dict[str, Any],
    judge_result: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic text-only repair plan (fast and stable, no extra model call)."""
    fix_changes = _auto_select_repair_strategy(judge_result)
    strategy = str(fix_changes.get("strategy", "retime_timeline") or "retime_timeline")
    shot_patch: dict[str, Any] = {}
    actions: list[str] = []

    if strategy in {"reretrieve_visual", "regenerate_shot_execution"}:
        query = str(shot.get("visual_query", "") or "").strip()
        chapter = str(shot.get("chapter_title", "") or "").strip()
        if not query and chapter:
            shot_patch["visual_query"] = chapter[:64]
            actions.append("补全visual_query")

    if strategy == "retime_timeline":
        audio_meta = assets.get("audio_meta", {}) if isinstance(assets.get("audio_meta"), dict) else {}
        audio_duration = _safe_float(audio_meta.get("duration_sec"), 0.0)
        duration_sec = _safe_float(shot.get("duration_sec"), 0.0)
        if audio_duration > 0.0 and audio_duration > duration_sec:
            shot_patch["duration_sec"] = round(audio_duration, 2)
            actions.append("对齐duration_sec到audio_duration")

    if strategy == "repair_subtitle_and_decoration":
        vocal = str(shot.get("vocal_script", "") or "").strip()
        if not vocal:
            chapter = str(shot.get("chapter_title", "") or "").strip()
            if chapter:
                shot_patch["vocal_script"] = chapter
                actions.append("用chapter_title补全vocal_script")

    if not actions:
        actions.append("保留文本并记录修复元数据")
    return {
        "strategy": strategy,
        "actions": actions,
        "shot_patch": shot_patch,
        "issue_count": len(_dict_list(judge_result.get("issues", []))),
    }


STAGE4_DIMENSIONS = ("visual_match", "audio_sync", "decoration_quality", "timing_alignment")


def _stage4_issue_dimension(issue: dict[str, Any]) -> str:
    raw_dim = str(issue.get("dimension", "") or "").strip().lower()
    text = " ".join([
        raw_dim,
        str(issue.get("description", "") or "").lower(),
        str(issue.get("suggested_fix", "") or "").lower(),
    ])
    if raw_dim in {"visual_match", "query_accuracy", "素材匹配", "画面匹配"}:
        return "visual_match"
    if raw_dim in {"audio_sync", "audio_quality", "tts_audio", "音频同步", "音频质量"}:
        return "audio_sync"
    if raw_dim in {"timing_alignment", "timeline_compose", "时序对齐", "时间线"}:
        if any(token in text for token in ("audio_path", "tts", "音频路径", "重新生成音频", "旁白音频")):
            return "audio_sync"
        return "timing_alignment"
    if raw_dim in {
        "decoration_quality",
        "subtitle_accuracy",
        "effect_quality",
        "content_coherence",
        "slot_fill_accuracy",
        "slot_accuracy",
        "layout_quality",
        "text_layout",
        "template_slot",
        "字幕正确",
        "动效合理",
        "装饰质量",
        "槽位准确",
        "排版质量",
    }:
        return "decoration_quality"
    if any(token in text for token in ("clip", "visual", "素材", "画面", "检索词")):
        return "visual_match"
    if any(token in text for token in ("tts", "音频路径", "旁白音频", "audio_path")):
        return "audio_sync"
    if any(token in text for token in ("时长", "时间线", "同步", "duration", "timing")):
        return "timing_alignment"
    if any(token in text for token in ("字幕", "装饰", "模板", "槽位", "动效", "title", "slot")):
        return "decoration_quality"
    return "decoration_quality"


def _stage4_dimension_scores(judge_result: dict[str, Any]) -> dict[str, float]:
    dims = judge_result.get("dimensions", {}) if isinstance(judge_result.get("dimensions"), dict) else {}
    overall = _safe_float(judge_result.get("overall_score"), 1.0)

    def score(*keys: str, default: float = overall) -> float:
        values = [_safe_float(dims.get(key), -1.0) for key in keys if key in dims]
        values = [value for value in values if value >= 0.0]
        if not values:
            return default
        return round(sum(values) / len(values), 3)

    decoration_keys = [
        key
        for key in (
            "decoration_quality",
            "subtitle_accuracy",
            "effect_quality",
            "content_coherence",
            "slot_fill_accuracy",
            "slot_accuracy",
            "layout_quality",
            "text_layout",
        )
        if key in dims
    ]
    decoration_score = score(*decoration_keys, default=1.0 if not decoration_keys else overall)
    return {
        "visual_match": score("visual_match", "query_accuracy", default=overall),
        "audio_sync": score("audio_sync", "audio_quality", default=1.0),
        "decoration_quality": decoration_score,
        "timing_alignment": score("timing_alignment", default=overall),
    }


def _stage4_dimension_observation(
    judge_result: dict[str, Any],
    *,
    pass_threshold: float,
    assume_passed: bool | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    config = config or {}
    ignored_dimensions = _stage4_ignored_dimensions(config) if config else set()
    threshold_cfg = _cfg_get(config, ("stage3_trajectory", "text_judge", "dimension_thresholds"), {}) if config else {}
    thresholds = threshold_cfg if isinstance(threshold_cfg, dict) else {}
    issues_by_dim: dict[str, list[dict[str, Any]]] = {dim: [] for dim in STAGE4_DIMENSIONS}
    for issue in _dict_list(judge_result.get("issues", [])):
        dim = _stage4_issue_dimension(issue)
        issues_by_dim.setdefault(dim, []).append(issue)

    scores = _stage4_dimension_scores(judge_result)
    observation: dict[str, dict[str, Any]] = {}
    for dim in STAGE4_DIMENSIONS:
        issues = issues_by_dim.get(dim, [])
        blocking = [
            issue for issue in issues
            if str(issue.get("severity", "")).lower() in {"critical", "warning"}
        ]
        score = max(0.0, min(1.0, _safe_float(scores.get(dim), 1.0)))
        threshold = _safe_float(
            thresholds.get(dim),
            pass_threshold if dim in {"visual_match", "timing_alignment"} else min(pass_threshold, 0.70),
        )
        passed = (score >= threshold and not blocking)
        ignored = dim in ignored_dimensions
        if ignored:
            passed = True
            blocking = []
            score = max(score, threshold)
        if assume_passed is not None:
            passed = bool(assume_passed) and not blocking
            if assume_passed and not issues:
                score = max(score, 1.0)
        observation[dim] = {
            "passed": passed,
            "score": round(score, 3),
            "issues": issues,
            "ignored": ignored,
        }
    return observation


def _stage4_failed_dimensions(observation: dict[str, Any]) -> list[str]:
    return [
        dim for dim in STAGE4_DIMENSIONS
        if isinstance(observation.get(dim), dict) and not bool(observation[dim].get("passed", False))
    ]


def _stage4_batch_failure_summary(items: list[dict[str, Any]]) -> str:
    counts = {dim: 0 for dim in STAGE4_DIMENSIONS}
    for item in items:
        for dim in item.get("failed_dimensions", []) or []:
            if dim in counts:
                counts[dim] += 1
    return "，".join(f"{dim}={count}" for dim, count in counts.items() if count) or "无显式维度"


def _stage4_strategies_for_dimensions(dimensions: list[str]) -> list[str]:
    mapping = {
        "visual_match": "reretrieve_visual",
        "audio_sync": "regenerate_audio",
        "decoration_quality": "redecorate",
        "timing_alignment": "retime_timeline",
    }
    strategies: list[str] = []
    for dim in dimensions:
        strategy = mapping.get(dim)
        if strategy and strategy not in strategies:
            strategies.append(strategy)
    return strategies or ["retime_timeline"]


def _stage4_changed_layers(strategies: list[str]) -> list[str]:
    layers: list[str] = []
    for strategy in strategies:
        if strategy == "reretrieve_visual" and "visual" not in layers:
            layers.append("visual")
        elif strategy == "regenerate_audio" and "audio" not in layers:
            layers.append("audio")
        elif strategy in {"redecorate", "repair_subtitle_and_decoration"} and "decoration" not in layers:
            layers.append("decoration")
        elif strategy == "retime_timeline" and "timing" not in layers:
            layers.append("timing")
        elif strategy == "regenerate_shot_execution":
            for layer in ("visual", "audio", "decoration", "timing"):
                if layer not in layers:
                    layers.append(layer)
    return layers


def _stage4_compose_after_repair(
    shot: dict[str, Any],
    assets: dict[str, Any],
    output_dir: str,
    config: dict[str, Any],
    *,
    strategies: list[str],
    is_fake: bool,
) -> dict[str, Any]:
    changed_layers = _stage4_changed_layers(strategies)
    all_layers = ["visual", "audio", "decoration", "timing"]
    reused_layers = [layer for layer in all_layers if layer not in changed_layers]
    if is_fake:
        compose_result = fake_utils.fake_render_success(shot, assets, output_dir)
    else:
        compose_result = timeline_compose.run(
                shot, assets, output_dir,
                config=_compose_config_with_occupancy(config, sid, all_assets),
            )
    return {
        **compose_result,
        "partial_reuse": {
            "changed_layers": changed_layers,
            "reused_layers": reused_layers,
            "strategies": strategies,
            "note": "上游只重跑失败维度对应工具；合成复用未变化的视觉/音频/装饰配置。",
        },
    }


def _compose_spotlight_with_reuse(
    shot: dict[str, Any],
    assets: dict[str, Any],
    output_dir: str,
    config: dict[str, Any],
    repair_round: int,
) -> dict[str, Any]:
    if repair_round == 0:
        existing = assets.get("compose_result") if isinstance(assets.get("compose_result"), dict) else {}
        rendered_path = str(existing.get("rendered_path") or assets.get("rendered_path") or "")
        if (
            existing.get("render_success")
            and existing.get("render_backend") == "hyperframes"
            and _existing_file(rendered_path)
        ):
            return {
                **existing,
                "rendered_path": rendered_path,
                "reused_render": True,
                "status": "success",
            }
    return timeline_compose.run(
        shot,
        assets,
        output_dir,
        config=config,
        render_backend="hyperframes",
    )


def _process_spotlight_track(
    state: PipelineState,
    items: list[dict[str, Any]],
    all_assets: dict[str, Any],
    new_shot_plan: list[dict[str, Any]],
    all_entries: list[dict[str, Any]],
    output_dir: str,
    config: dict[str, Any],
    pass_threshold: float,
    cheap_accept_threshold: float,
    repairable_threshold: float,
    max_repair_rounds: int,
    judge_enabled: bool,
) -> None:
    """Full render loop for spotlight shots: render → VLM-Judge → HITL → repair → re-render."""
    if not items:
        return

    # ── fake mode: 跳过 HyperFrames 渲染，用 metadata-based fake VLM 分析 ──
    if _is_fake_mode(config):
        _process_spotlight_track_fake(
            state=state,
            items=items,
            all_assets=all_assets,
            new_shot_plan=new_shot_plan,
            all_entries=all_entries,
            output_dir=output_dir,
            config=config,
            pass_threshold=pass_threshold,
            max_repair_rounds=max_repair_rounds,
            judge_enabled=judge_enabled,
        )
        return

    num_options = _cfg_int(config, ("hitl", "num_options"), 4)
    timeout_sec = max(0, _cfg_int(config, ("hitl", "decision_timeout_sec"), 20))

    vlm_items: list[dict[str, Any]] = []
    failed_items: list[dict[str, Any]] = []

    for item in items:
        sid = item["sid"]
        shot = item["shot"]
        assets = item["assets"]
        absolute_idx = item["absolute_idx"]

        compose_result = _compose_spotlight_with_reuse(
            shot,
            assets,
            output_dir,
            config,
            repair_round=0,
        )
        all_entries.extend(_log_tool(
            "Tool_TimelineCompose",
            {"shot_id": sid, "batch_index": absolute_idx, "repair_round": 0},
            compose_result,
            "stage4_compose",
            thought=f"Spotlight 镜头 {sid}: 时间线对齐合成并渲染",
        ))

        assets["timeline"] = compose_result.get("timeline", {})
        assets["rendered_path"] = compose_result.get("rendered_path", "")
        assets["compose_result"] = compose_result
        cheap_check = _cheap_render_check(
            shot, assets, compose_result,
            accept_threshold=cheap_accept_threshold,
            repair_round=0,
        )
        needs_vlm = judge_enabled and _needs_vlm_judge(shot, cheap_check, config, repair_round=0)
        cheap_check["needs_vlm"] = needs_vlm
        cheap_check["decision"] = "vlm_judge" if needs_vlm else ("accept" if cheap_check.get("passed") else "repair")
        cheap_artifact = _save_stage4_render_check(state, sid, 0, cheap_check)
        all_entries.extend(_log_tool(
            "Tool_RenderCheapCheck",
            {"shot_id": sid, "repair_round": 0},
            {**cheap_check, "artifact_path": cheap_artifact},
            "stage4_render_check",
            thought=f"Spotlight 镜头 {sid}: 本地快检渲染结构完整性",
        ))

        if needs_vlm:
            vlm_items.append({
                "absolute_idx": absolute_idx,
                "shot": shot,
                "sid": sid,
                "assets": assets,
                "compose_result": compose_result,
                "timeline": compose_result.get("timeline", {}),
                "clip_meta": assets.get("clip_meta", {}),
                "cheap_check": cheap_check,
            })
        else:
            judge_result = _judge_from_cheap_check(sid, cheap_check, source="cheap_check")
            judge_artifact = _save_stage4_judge_round(state, sid, 0, judge_result)
            assets["judge_result"] = {**judge_result, "artifact_path": judge_artifact}
            assets["approved"] = bool(compose_result.get("render_success", False)) and bool(
                judge_result.get("passed", False)
            )
            all_assets[sid] = assets
            if not assets["approved"]:
                failed_items.append({
                    "absolute_idx": absolute_idx,
                    "shot": shot,
                    "sid": sid,
                    "assets": assets,
                    "compose_result": compose_result,
                    "judge_result": judge_result,
                    "source": "cheap_check",
                })

    if vlm_items:
        batch_results = _run_vlm_judge_batch(
            state, vlm_items,
            pass_threshold=pass_threshold,
            judge_round=0,
            all_entries=all_entries,
            batch_label=f"spotlight_vlm",
        )
        for item in vlm_items:
            sid = item["sid"]
            assets = item["assets"]
            judge_result = batch_results.get(sid, _judge_from_cheap_check(sid, item["cheap_check"], source="missing_vlm"))
            judge_artifact = _save_stage4_judge_round(state, sid, 0, judge_result)
            assets["judge_result"] = {**judge_result, "artifact_path": judge_artifact}
            assets["approved"] = bool(item["compose_result"].get("render_success", False)) and bool(
                judge_result.get("passed", False)
            )
            all_assets[sid] = assets
            if not assets["approved"]:
                failed_items.append({
                    "absolute_idx": item["absolute_idx"],
                    "shot": item["shot"],
                    "sid": sid,
                    "assets": assets,
                    "compose_result": item["compose_result"],
                    "judge_result": judge_result,
                    "source": "vlm_batch",
                })

    # Spotlight repair loop with HITL
    rejudge_items: list[dict[str, Any]] = []
    if failed_items:
        failed_descs = []
        for item in failed_items:
            jr = item.get("judge_result", {})
            issue_dims = [
                str(issue.get("dimension", ""))
                for issue in _dict_list(jr.get("issues", []))[:3]
            ]
            failed_descs.append(
                f"{item['sid']}: score={float(jr.get('overall_score', 0.0) or 0.0):.2f}, "
                f"issues={issue_dims}"
            )
        all_entries.append({
            "role": "thought",
            "content": (
                f"[OBSERVATION] Spotlight 本批失败 {len(failed_items)} 个: "
                + " | ".join(failed_descs[:5])
                + (" ..." if len(failed_descs) > 5 else "")
            ),
            "metadata": {
                "stage": "stage4_spotlight_repair",
                "phase": "observation",
                "round": 0,
                "failed_count": len(failed_items),
            },
        })
    if max_repair_rounds > 0 and failed_items:
        for item in failed_items:
            sid = item["sid"]
            shot = item["shot"]
            assets = item["assets"]
            judge_result = item["judge_result"]
            absolute_idx = item["absolute_idx"]

            # Reflection → generate fix options → HITL decision
            fix_changes = _spotlight_hitl_repair(
                state, shot, assets, judge_result,
                all_entries=all_entries,
                num_options=num_options,
                timeout_sec=timeout_sec,
            )

            repair_payload = _apply_stage4_auto_patch(
                state, shot, assets, item["compose_result"],
                judge_result, repair_round=0,
                max_repair_rounds=max_repair_rounds,
                fix_changes=fix_changes,
            )
            all_entries.extend(repair_payload["entries"])
            updated_shot = repair_payload["shot"]
            updated_assets = repair_payload["assets"]
            new_shot_plan[absolute_idx] = updated_shot

            # Re-render after repair
            compose_result = _compose_spotlight_with_reuse(
                updated_shot,
                updated_assets,
                output_dir,
                config,
                repair_round=1,
            )
            all_entries.extend(_log_tool(
                "Tool_TimelineCompose",
                {"shot_id": sid, "batch_index": absolute_idx, "repair_round": 1},
                compose_result,
                "stage4_compose_repair",
                thought=f"Spotlight 镜头 {sid}: 修复后重新渲染",
            ))
            updated_assets["timeline"] = compose_result.get("timeline", {})
            updated_assets["rendered_path"] = compose_result.get("rendered_path", "")
            updated_assets["compose_result"] = compose_result
            cheap_check = _cheap_render_check(
                updated_shot, updated_assets, compose_result,
                accept_threshold=cheap_accept_threshold,
                repair_round=1,
            )
            needs_rejudge = judge_enabled and _needs_vlm_judge(updated_shot, cheap_check, config, repair_round=1)
            cheap_check["needs_vlm"] = needs_rejudge
            cheap_check["decision"] = "vlm_rejudge" if needs_rejudge else ("accept" if cheap_check.get("passed") else "repair_exhausted")
            cheap_artifact = _save_stage4_render_check(state, sid, 1, cheap_check)
            all_entries.extend(_log_tool(
                "Tool_RenderCheapCheck",
                {"shot_id": sid, "repair_round": 1},
                {**cheap_check, "artifact_path": cheap_artifact},
                "stage4_render_check_repair",
                thought=f"Spotlight 镜头 {sid}: 修复后本地快检",
            ))

            if needs_rejudge:
                rejudge_items.append({
                    "absolute_idx": absolute_idx,
                    "shot": updated_shot,
                    "sid": sid,
                    "assets": updated_assets,
                    "compose_result": compose_result,
                    "timeline": compose_result.get("timeline", {}),
                    "clip_meta": updated_assets.get("clip_meta", {}),
                    "cheap_check": cheap_check,
                })
                all_assets[sid] = updated_assets
                continue

            judge_result = _judge_from_cheap_check(sid, cheap_check, source="repaired_by_rule")
            if not judge_result.get("passed", False):
                judge_result = {
                    **judge_result,
                    "repair_exhausted": True,
                    "repair_round": 1,
                    "status": "repair_exhausted",
                }
            judge_artifact = _save_stage4_judge_round(state, sid, 1, judge_result)
            updated_assets["judge_result"] = {**judge_result, "artifact_path": judge_artifact}
            updated_assets["approved"] = bool(judge_result.get("passed", False))
            all_assets[sid] = updated_assets

    elif failed_items:
        for item in failed_items:
            _mark_stage4_repair_exhausted(
                all_assets, item["sid"], item["assets"], item["judge_result"],
                repair_round=0,
                repairable_threshold=repairable_threshold,
                all_entries=all_entries,
            )

    if rejudge_items:
        rejudge_results = _run_vlm_judge_batch(
            state, rejudge_items,
            pass_threshold=pass_threshold,
            judge_round=1,
            all_entries=all_entries,
            batch_label="spotlight_repair_vlm",
        )
        for item in rejudge_items:
            sid = item["sid"]
            assets = item["assets"]
            judge_result = rejudge_results.get(sid, _judge_from_cheap_check(sid, item["cheap_check"], source="missing_vlm_rejudge"))
            if not judge_result.get("passed", False):
                judge_result = {
                    **judge_result,
                    "repair_exhausted": True,
                    "repair_round": 1,
                    "status": "repair_exhausted" if not judge_result.get("fallback_used", False) else "needs_review",
                }
            judge_artifact = _save_stage4_judge_round(state, sid, 1, judge_result)
            assets["judge_result"] = {**judge_result, "artifact_path": judge_artifact}
            assets["approved"] = bool(item["compose_result"].get("render_success", False)) and (
                bool(judge_result.get("passed", False))
                or (
                    float(judge_result.get("overall_score", 0.0) or 0.0) >= repairable_threshold
                    and not judge_result.get("fallback_used", False)
                )
            )
            all_assets[sid] = assets

    if failed_items:
        repaired_pass = sum(
            1 for item in failed_items
            if bool(all_assets.get(item["sid"], {}).get("approved", False))
        )
        remaining_fail = max(0, len(failed_items) - repaired_pass)
        all_entries.append({
            "role": "thought",
            "content": (
                f"[VERIFY] Spotlight 批量修复完成: 通过 {repaired_pass}/{len(failed_items)}"
                + (f"，剩余未通过 {remaining_fail}" if remaining_fail else "，全部通过")
            ),
            "metadata": {
                "stage": "stage4_spotlight_repair",
                "phase": "verify",
                "round": 1 if max_repair_rounds > 0 else 0,
                "failed_count": len(failed_items),
                "passed_after_repair": repaired_pass,
            },
        })


def _process_spotlight_track_fake(
    state: PipelineState,
    items: list[dict[str, Any]],
    all_assets: dict[str, Any],
    new_shot_plan: list[dict[str, Any]],
    all_entries: list[dict[str, Any]],
    output_dir: str,
    config: dict[str, Any],
    pass_threshold: float,
    max_repair_rounds: int,
    judge_enabled: bool,
) -> None:
    """Fake mode spotlight track: skip HyperFrames render, use metadata-based VLM analysis.

    轨迹压缩：compose + judge 不逐镜头展开，内部循环处理所有镜头，
    最后只输出一条汇总 tool call entry。
    """
    if not items:
        return

    num_options = _cfg_int(config, ("hitl", "num_options"), 4)
    timeout_sec = max(0, _cfg_int(config, ("hitl", "decision_timeout_sec"), 15))

    failed_items: list[dict[str, Any]] = []
    compose_summary: dict[str, dict] = {}
    judge_summary: dict[str, dict] = {}

    # ── 批量处理：内部循环所有 spotlight 镜头，不逐镜头展开 trajectory ──
    for i, item in enumerate(items):
        sid = item["sid"]
        shot = item["shot"]
        assets = item["assets"]
        absolute_idx = item["absolute_idx"]

        _log_progress(i + 1, len(items), f"spotlight fake compose+judge")
        _emit_progress("Tool_BatchVLMJudge", "合成+质检", i + 1, len(items),
                       f"Spotlight 镜头 {sid}: fake 合成 + VLM 分析", "stage4_compose")

        # Fake compose: 不调 HyperFrames，用 fake_utils
        compose_result = fake_utils.fake_render_success(shot, assets, output_dir)
        assets["timeline"] = compose_result.get("timeline", {})
        assets["rendered_path"] = compose_result.get("rendered_path", "")
        assets["compose_result"] = compose_result
        compose_summary[sid] = {
            "render_success": compose_result.get("render_success", False),
            "render_status": compose_result.get("render_status", ""),
            "duration_sec": compose_result.get("duration_sec", 0),
        }

        # Fake VLM analysis: 基于 metadata 生成"假装分析"
        judge_result = fake_utils.fake_vlm_analysis(shot, assets, compose_result)
        judge_artifact = _save_stage4_judge_round(state, sid, 0, judge_result)
        assets["judge_result"] = {**judge_result, "artifact_path": judge_artifact}
        assets["approved"] = bool(judge_result.get("passed", False))
        all_assets[sid] = assets
        judge_summary[sid] = {
            "passed": judge_result.get("passed", False),
            "overall_score": judge_result.get("overall_score", 0),
            "verdict": judge_result.get("verdict", ""),
            "issues_count": len(judge_result.get("issues", [])),
        }

        if not assets["approved"]:
            failed_items.append({
                "absolute_idx": absolute_idx,
                "shot": shot,
                "sid": sid,
                "assets": assets,
                "compose_result": compose_result,
                "judge_result": judge_result,
                "source": "fake_vlm",
            })

    # ── 轨迹压缩：只记录一条汇总 compose + 一条汇总 judge ──
    compose_ok = sum(1 for v in compose_summary.values() if v.get("render_success"))
    all_entries.extend(_log_tool(
        "Tool_BatchTimelineCompose",
        {"shot_ids": list(compose_summary.keys()), "shot_count": len(items), "render_backend": "fake"},
        {"shots": compose_summary, "render_success_count": compose_ok, "status": "success"},
        "stage4_compose",
        thought=f"批量 fake 渲染 {len(items)} 个 spotlight 镜头: {compose_ok}/{len(items)} 成功（标注模式跳过 HyperFrames）",
    ))

    judge_ok = sum(1 for v in judge_summary.values() if v.get("passed"))
    all_entries.extend(_log_tool(
        "Tool_BatchVLMJudge",
        {"shot_ids": list(judge_summary.keys()), "judge_round": 0, "mode": "fake_metadata"},
        {"shots": judge_summary, "passed_count": judge_ok, "failed_count": len(items) - judge_ok, "status": "success"},
        "stage4_judge",
        thought=f"批量 fake VLM 分析 {len(items)} 个 spotlight 镜头: {judge_ok}/{len(items)} 通过（基于 metadata，不抽帧）",
    ))

    # ── Spotlight repair: OBSERVATION → REFLECTION → REPAIR → VERIFY (agentic batch) ──
    if max_repair_rounds > 0 and failed_items:
        # ── OBSERVATION + REFLECTION (一条 thought 汇总全部失败镜头) ──
        failed_descs = []
        for item in failed_items:
            jr = item["judge_result"]
            failed_descs.append(
                f"{item['sid']}: score={jr.get('overall_score', 0)}, "
                f"issues={[i.get('dimension', '') for i in jr.get('issues', [])[:3]]}"
            )
        all_entries.append({
            "role": "thought",
            "content": (
                f"[OBSERVATION] {len(failed_items)}/{len(items)} 个 spotlight 质检不通过: "
                + " | ".join(failed_descs[:5]) + ("..." if len(failed_descs) > 5 else "")
            ),
            "metadata": {"stage": "stage4_repair", "round": 0},
        })

        repair_render_summary: dict[str, dict] = {}
        repair_judge_summary: dict[str, dict] = {}
        repair_actions: list[str] = []

        for item in failed_items:
            sid = item["sid"]
            shot = item["shot"]
            assets = item["assets"]
            judge_result = item["judge_result"]
            absolute_idx = item["absolute_idx"]

            # Reflection → generate fix options → HITL decision
            fix_changes = _spotlight_hitl_repair(
                state, shot, assets, judge_result,
                all_entries=all_entries,
                num_options=num_options,
                timeout_sec=timeout_sec,
            )

            repair_payload = _apply_stage4_auto_patch(
                state, shot, assets, item["compose_result"],
                judge_result, repair_round=0,
                max_repair_rounds=max_repair_rounds,
                fix_changes=fix_changes,
            )
            all_entries.extend(repair_payload["entries"])
            updated_shot = repair_payload["shot"]
            updated_assets = repair_payload["assets"]
            new_shot_plan[absolute_idx] = updated_shot

            # Fake re-render after repair
            compose_result = fake_utils.fake_render_success(updated_shot, updated_assets, output_dir)
            updated_assets["timeline"] = compose_result.get("timeline", {})
            updated_assets["rendered_path"] = compose_result.get("rendered_path", "")
            updated_assets["compose_result"] = compose_result
            repair_render_summary[sid] = {
                "render_success": compose_result.get("render_success", False),
                "repair_round": 1,
            }

            # Fake re-judge
            rejudge_result = fake_utils.fake_vlm_analysis(updated_shot, updated_assets, compose_result)
            if not rejudge_result.get("passed", False):
                rejudge_result = {
                    **rejudge_result,
                    "repair_exhausted": True,
                    "repair_round": 1,
                    "status": "repair_exhausted",
                }
            judge_artifact = _save_stage4_judge_round(state, sid, 1, rejudge_result)
            updated_assets["judge_result"] = {**rejudge_result, "artifact_path": judge_artifact}
            updated_assets["approved"] = bool(rejudge_result.get("passed", False))
            all_assets[sid] = updated_assets
            repair_judge_summary[sid] = {
                "passed": rejudge_result.get("passed", False),
                "overall_score": rejudge_result.get("overall_score", 0),
                "repair_exhausted": rejudge_result.get("repair_exhausted", False),
            }

        # 修复后的 re-render + re-judge 压缩为一条汇总
        if repair_render_summary:
            repair_ok = sum(1 for v in repair_render_summary.values() if v.get("render_success"))
            all_entries.extend(_log_tool(
                "Tool_BatchTimelineCompose_Repair",
                {"shot_ids": list(repair_render_summary.keys()), "repair_round": 1, "render_backend": "fake"},
                {"shots": repair_render_summary, "render_success_count": repair_ok, "status": "success"},
                "stage4_compose_repair",
                thought=f"批量 fake 修复后重新渲染 {len(repair_render_summary)} 个 spotlight 镜头",
            ))
        if repair_judge_summary:
            rejudge_ok = sum(1 for v in repair_judge_summary.values() if v.get("passed"))
            all_entries.extend(_log_tool(
                "Tool_BatchVLMJudge_Repair",
                {"shot_ids": list(repair_judge_summary.keys()), "judge_round": 1, "mode": "fake_metadata"},
                {"shots": repair_judge_summary, "passed_count": rejudge_ok, "status": "success"},
                "stage4_judge_repair",
                thought=f"批量 fake 修复后重新质检 {len(repair_judge_summary)} 个 spotlight 镜头: {rejudge_ok} 通过",
            ))

        # ── VERIFY: 汇总修复结果，一条 thought ──
        final_fail = sum(1 for v in repair_judge_summary.values() if not v.get("passed"))
        all_entries.append({
            "role": "thought",
            "content": (
                f"[VERIFY] 修复完成——{len(failed_items)} 个 spotlight 中 "
                f"{len(failed_items) - final_fail} 个修复后通过质检"
                + (f"，{final_fail} 个仍不达标（repair_exhausted）" if final_fail else "，全部通过")
            ),
            "metadata": {"stage": "stage4_repair", "round": 1},
        })

    elif failed_items:
        for item in failed_items:
            _mark_stage4_repair_exhausted(
                all_assets, item["sid"], item["assets"], item["judge_result"],
                repair_round=0,
                repairable_threshold=0.45,
                all_entries=all_entries,
            )


def _prepare_issue_grounded_hitl_options(
    judge_result: dict[str, Any],
    generated_options: list[dict[str, Any]],
    num_options: int,
) -> tuple[list[dict[str, Any]], int]:
    """Return exactly N executable options, each explicitly grounded in observed issues."""
    issues = _dict_list(judge_result.get("issues", []))
    issue_summary = "；".join(
        str(issue.get("description", "")).strip()
        for issue in issues[:3]
        if str(issue.get("description", "")).strip()
    ) or str(judge_result.get("summary", "")).strip() or "当前镜头质检未通过"
    fallback = vlm_render_judge._fallback_options(judge_result, num_options)
    issue_dims = {
        str(issue.get("dimension", "")).strip().lower()
        for issue in issues
        if str(issue.get("dimension", "")).strip()
    }
    relevant_strategies = {"regenerate_shot_execution"}
    if issue_dims & {"visual_match", "素材匹配", "query_accuracy"}:
        relevant_strategies.add("reretrieve_visual")
    if issue_dims & {"audio_sync", "audio_quality", "tts_audio", "音频同步", "音频质量"}:
        relevant_strategies.add("regenerate_audio")
    if issue_dims & {"timing_alignment", "时序对齐", "timeline_compose"}:
        relevant_strategies.add("retime_timeline")
    if issue_dims & {"subtitle_accuracy", "字幕正确", "effect_quality", "动效合理", "content_coherence"}:
        relevant_strategies.add("repair_subtitle_and_decoration")
        relevant_strategies.add("redecorate")
    if len(relevant_strategies) == 1:
        relevant_strategies.update({
            "reretrieve_visual",
            "regenerate_audio",
            "retime_timeline",
            "repair_subtitle_and_decoration",
            "redecorate",
        })

    recommended_strategy = str(_auto_select_repair_strategy(judge_result).get("strategy", ""))
    if recommended_strategy not in relevant_strategies:
        recommended_strategy = next(
            (s for s in relevant_strategies if s != "regenerate_shot_execution"),
            "regenerate_shot_execution",
        )

    candidate_options = [
        dict(option)
        for option in generated_options + fallback
        if isinstance(option, dict)
    ]
    options: list[dict[str, Any]] = []
    for candidate in candidate_options:
        option = dict(candidate)
        changes = dict(option.get("changes", {})) if isinstance(option.get("changes"), dict) else {}
        strategy = str(changes.get("strategy", "") or "").strip()
        if strategy and strategy not in relevant_strategies:
            continue
        if not strategy:
            strategy = recommended_strategy
        changes["strategy"] = strategy
        changes["judge_issues"] = issues
        option["changes"] = changes
        option["problem_basis"] = issue_summary
        description = str(option.get("description", "")).strip() or "修复当前已检出的镜头问题。"
        option["description"] = f"{description} 当前依据：{issue_summary[:180]}"
        options.append(option)
        if len(options) >= num_options:
            break

    fill_variants = [
        ("按主要问题定点修复", "只修改当前主要问题直接对应的字段，保持其他结果不变。", recommended_strategy, "targeted"),
        ("最小改动保守修复", "仅采用 judge suggested_fix 中明确要求的最小字段改动。", recommended_strategy, "minimal"),
        ("同维度关联修复", "修复当前问题及同一质量维度下直接关联的字段。", recommended_strategy, "dimension_wide"),
        ("重新生成当前镜头", "基于当前全部真实 issues 重做该镜头的检索、音频、装饰与时间线。", "regenerate_shot_execution", "regenerate"),
    ]
    while len(options) < num_options:
        label, description, strategy, repair_policy = fill_variants[len(options) % len(fill_variants)]
        options.append({
            "label": label,
            "description": f"{description} 当前依据：{issue_summary[:180]}",
            "changes": {
                "strategy": strategy,
                "judge_issues": issues,
                "repair_scope": "shot",
                "repair_policy": repair_policy,
            },
            "problem_basis": issue_summary,
        })

    recommended_option = next(
        (
            idx
            for idx, option in enumerate(options)
            if str((option.get("changes") or {}).get("strategy", "")) == recommended_strategy
        ),
        0 if options else -1,
    )
    for idx, option in enumerate(options):
        option["option_id"] = idx
        option["recommended"] = idx == recommended_option
    return options, recommended_option


def _spotlight_hitl_repair(
    state: PipelineState,
    shot: dict[str, Any],
    assets: dict[str, Any],
    judge_result: dict[str, Any],
    all_entries: list[dict[str, Any]],
    num_options: int = 4,
    timeout_sec: int = 15,
    auto_select: bool = False,
) -> dict[str, Any]:
    """Generate fix options and run HITL decision for spotlight repair."""
    config = state.get("config", {})
    sid = str(shot.get("shot_id", ""))
    timeline = assets.get("timeline", {})

    llm = _get_llm(state, service=_resolve_judge_service(config))
    try:
        options = vlm_render_judge.generate_fix_options(
            llm, judge_result, shot, timeline, num_options=num_options,
        )
    except Exception:
        options = vlm_render_judge._fallback_options(judge_result, num_options)
    finally:
        llm.close()

    options, recommended_option = _prepare_issue_grounded_hitl_options(
        judge_result,
        options,
        num_options,
    )

    hitl_ctx = {
        "node_name": "spotlight_repair",
        "question": f"Spotlight 镜头 {sid} VLM质检未通过 (score={judge_result.get('overall_score', 0):.2f})，请选择修复方案",
        "current_data": {
            "issues": judge_result.get("issues", []),
            "summary": judge_result.get("summary", ""),
            "score": judge_result.get("overall_score", 0),
        },
        "options": options,
        "recommended_option": recommended_option,
        "timeout_sec": timeout_sec,
    }

    # interactive: 15s 倒计时由 server/client 执行，超时采用推荐项；
    # auto: 无人值守场景立即采用推荐项。
    if _should_interrupt_hitl(state):
        human_decision = interrupt({
            "type": "spotlight_repair_review",
            "context": hitl_ctx,
            "message": f"Spotlight 镜头 {sid} VLM质检未通过",
            "timeout_sec": timeout_sec,
        })
    else:
        human_decision = {
            "selected_option": recommended_option,
            "custom_input": "[auto_mode] 立即采用模型推荐方案" if _hitl_auto_mode(config) else "",
        }

    if not isinstance(human_decision, dict):
        human_decision = _as_mapping(human_decision)
    selected = _cfg_int(human_decision, ("selected_option",), 0)
    custom_input = str(human_decision.get("custom_input", "") or "")

    chosen_label = ""
    if 0 <= selected < len(options):
        opt = options[selected]
        chosen_label = opt.get("label", "") if isinstance(opt, dict) else str(opt)
        fix_changes = opt.get("changes", {}) if isinstance(opt, dict) else {}
    elif custom_input:
        chosen_label = custom_input
        fix_changes = {"custom_fix": custom_input}
    else:
        fix_changes = _auto_select_repair_strategy(judge_result)
        chosen_label = fix_changes.get("strategy", "auto")

    hitl_entry = {
        "role": "human_feedback",
        "content": (
            f"[Spotlight修复] 镜头 {sid} VLM-Judge 未通过，系统提出 {len(options)} 个方案。\n"
            f"经过 HIL 审查 (timeout={timeout_sec}s)，选择方案 {selected}: {chosen_label}\n"
            f"标记: HIL checked"
        ),
        "metadata": {
            "node_name": "spotlight_repair",
            "shot_id": sid,
            "options": options,
            "selected_option": selected,
            "custom_input": custom_input,
            "feedback_type": "spotlight_repair_hitl",
            "hitl_checked": True,
            "ts": datetime.now().isoformat(),
        },
    }
    all_entries.append(hitl_entry)
    return fix_changes


def _process_non_spotlight_track(
    state: PipelineState,
    items: list[dict[str, Any]],
    all_assets: dict[str, Any],
    new_shot_plan: list[dict[str, Any]],
    all_entries: list[dict[str, Any]],
    output_dir: str,
    config: dict[str, Any],
    text_judge_threshold: float,
    text_judge_enabled: bool,
    cheap_accept_threshold: float,
    repairable_threshold: float,
) -> None:
    """Text quick judge + single HITL repair for non-spotlight shots.

    轨迹压缩：compose 不逐镜头展开，内部循环处理所有镜头，
    最后只输出一条汇总 tool call entry。
    """
    if not items:
        return

    num_options = _cfg_int(config, ("hitl", "num_options"), 4)
    timeout_sec = max(0, _cfg_int(config, ("hitl", "decision_timeout_sec"), 20))
    is_fake = _is_fake_mode(config)

    # ── 批量 compose：内部循环所有非 spotlight 镜头，不逐镜头展开 trajectory ──
    compose_summary: dict[str, dict] = {}
    for i, item in enumerate(items):
        sid = item["sid"]
        shot = item["shot"]
        assets = item["assets"]
        absolute_idx = item["absolute_idx"]

        _log_progress(i + 1, len(items), f"non-spotlight compose")
        if is_fake:
            _emit_progress("Tool_BatchTimelineCompose", "模拟合成", i + 1, len(items),
                       f"非Spotlight 镜头 {sid}: 时间线合成", "stage4_compose")
            compose_result = fake_utils.fake_render_success(shot, assets, output_dir)
        else:
            _emit_progress("Tool_BatchTimelineCompose", "合成渲染", i + 1, len(items),
                       f"非Spotlight 镜头 {sid}: 时间线合成", "stage4_compose")
            compose_result = timeline_compose.run(
                shot, assets, output_dir,
                config=_compose_config_with_occupancy(config, sid, all_assets),
            )
        assets["timeline"] = compose_result.get("timeline", {})
        assets["rendered_path"] = compose_result.get("rendered_path", "")
        assets["compose_result"] = compose_result
        item["compose_result"] = compose_result
        all_assets[sid] = assets
        compose_summary[sid] = {
            "render_success": compose_result.get("render_success", False),
            "render_status": compose_result.get("render_status", ""),
        }

    # 轨迹压缩：只记录一条汇总 compose entry
    compose_ok = sum(1 for v in compose_summary.values() if v.get("render_success"))
    all_entries.extend(_log_tool(
        "Tool_BatchTimelineCompose",
        {"shot_ids": list(compose_summary.keys()), "shot_count": len(items), "render_backend": "fake" if is_fake else "real"},
        {"shots": compose_summary, "render_success_count": compose_ok, "status": "success"},
        "stage4_compose",
        thought=f"批量合成 {len(items)} 个非Spotlight 镜头: {compose_ok}/{len(items)} 成功",
    ))

    if not text_judge_enabled:
        for item in items:
            assets = item["assets"]
            assets["approved"] = bool(item.get("compose_result", {}).get("render_success", False))
            assets["judge_result"] = {
                "shot_id": item["sid"],
                "passed": assets["approved"],
                "overall_score": 1.0 if assets["approved"] else 0.0,
                "status": "text_judge_disabled" if assets["approved"] else "render_failed",
                "judge_type": "text_quick",
            }
            all_assets[item["sid"]] = assets
        return

    # Batch text quick judge
    judge_items = [
        {"shot": item["shot"], "sid": item["sid"], "assets": item["assets"]}
        for item in items
    ]
    llm = _get_llm(state, service=_resolve_judge_service(config))
    try:
        text_judge_results = text_quick_judge.batch_run(
            llm,
            judge_items,
            pass_threshold=text_judge_threshold,
            focus_decoration_slots=_stage4_focus_decoration_slots(config),
        )
    except Exception as exc:
        logger.warning("[non_spotlight_track] text_quick_judge batch failed: %s", exc)
        text_judge_results = {}
    finally:
        llm.close()

    all_entries.extend(_log_tool(
        "Tool_TextQuickJudge_Batch",
        {"shot_ids": [item["sid"] for item in items], "threshold": text_judge_threshold},
        {
            "shot_count": len(items),
            "passed_count": sum(1 for r in text_judge_results.values() if r.get("passed", False)),
            "failed_count": sum(1 for r in text_judge_results.values() if not r.get("passed", False)),
            "status": "success",
        },
        "stage4_text_judge",
        thought=f"对 {len(items)} 个非Spotlight镜头执行文本快速质检",
    ))

    # Process results: passed → approved; failed → single HITL repair
    failed_items: list[dict[str, Any]] = []
    for item in items:
        sid = item["sid"]
        assets = item["assets"]
        judge_result = text_judge_results.get(sid)
        if judge_result is None:
            judge_result = text_quick_judge._fallback_judge(item["shot"], assets)
            judge_result["fallback_used"] = True
            judge_result = text_quick_judge._finalize_result(
                judge_result, item["shot"], assets, text_judge_threshold, ""
            )

        judge_artifact = _save_stage4_judge_round(state, sid, 0, judge_result)
        assets["judge_result"] = {**judge_result, "artifact_path": judge_artifact}
        assets["approved"] = bool(judge_result.get("passed", False)) and bool(
            item.get("compose_result", {}).get("render_success", False)
        )
        all_assets[sid] = assets

        if not assets["approved"]:
            failed_items.append({
                "absolute_idx": item["absolute_idx"],
                "shot": item["shot"],
                "sid": sid,
                "assets": assets,
                "compose_result": item.get("compose_result", {}),
                "judge_result": judge_result,
            })

    if failed_items:
        failed_descs = []
        for item in failed_items:
            jr = item.get("judge_result", {})
            issue_dims = [
                str(issue.get("dimension", ""))
                for issue in _dict_list(jr.get("issues", []))[:3]
            ]
            failed_descs.append(
                f"{item['sid']}: score={float(jr.get('overall_score', 0.0) or 0.0):.2f}, "
                f"issues={issue_dims}"
            )
        all_entries.append({
            "role": "thought",
            "content": (
                f"[OBSERVATION] 非Spotlight 本批文本质检失败 {len(failed_items)} 个: "
                + " | ".join(failed_descs[:6])
                + (" ..." if len(failed_descs) > 6 else "")
            ),
            "metadata": {
                "stage": "stage4_non_spotlight_repair",
                "phase": "observation",
                "round": 0,
                "failed_count": len(failed_items),
            },
        })

    # ── HITL repair for text-judge-failed non-spotlight shots ──
    # HITL 是 per-shot 的（每个镜头需要独立决策），但 re-compose 压缩为批量汇总
    repair_compose_summary: dict[str, dict] = {}
    for item in failed_items:
        sid = item["sid"]
        shot = item["shot"]
        assets = item["assets"]
        judge_result = item["judge_result"]
        absolute_idx = item["absolute_idx"]

        # Generate fix options + HITL (per-shot)
        fix_changes = _non_spotlight_hitl_repair(
            state, shot, assets, judge_result,
            all_entries=all_entries,
            num_options=num_options,
            timeout_sec=timeout_sec,
        )

        repair_payload = _apply_stage4_auto_patch(
            state, shot, assets, item["compose_result"],
            judge_result, repair_round=0,
            max_repair_rounds=1,
            fix_changes=fix_changes,
        )
        all_entries.extend(repair_payload["entries"])
        updated_shot = repair_payload["shot"]
        updated_assets = repair_payload["assets"]
        new_shot_plan[absolute_idx] = updated_shot

        # Re-compose after repair (no re-judge for non-spotlight)
        if is_fake:
            compose_result = fake_utils.fake_render_success(updated_shot, updated_assets, output_dir)
        else:
            compose_result = timeline_compose.run(updated_shot, updated_assets, output_dir, config=config)
        updated_assets["timeline"] = compose_result.get("timeline", {})
        updated_assets["rendered_path"] = compose_result.get("rendered_path", "")
        updated_assets["compose_result"] = compose_result
        updated_assets["approved"] = bool(compose_result.get("render_success", False))
        updated_assets["judge_result"] = {
            **judge_result,
            "repair_round": 1,
            "status": "text_repaired" if updated_assets["approved"] else "render_failed",
            "judge_type": "text_quick",
        }
        all_assets[sid] = updated_assets
        repair_compose_summary[sid] = {
            "render_success": compose_result.get("render_success", False),
            "approved": updated_assets["approved"],
        }

    # 修复后 re-compose 压缩为一条汇总
    if repair_compose_summary:
        repair_ok = sum(1 for v in repair_compose_summary.values() if v.get("render_success"))
        all_entries.extend(_log_tool(
            "Tool_BatchTimelineCompose_Repair",
            {"shot_ids": list(repair_compose_summary.keys()), "repair_round": 1, "render_backend": "fake" if is_fake else "real"},
            {"shots": repair_compose_summary, "render_success_count": repair_ok, "status": "success"},
            "stage4_compose_repair",
            thought=f"批量修复后重新合成 {len(repair_compose_summary)} 个非Spotlight 镜头: {repair_ok} 成功（不再二次质检）",
        ))

    if failed_items:
        repaired_pass = sum(
            1 for item in failed_items
            if bool(all_assets.get(item["sid"], {}).get("approved", False))
        )
        remaining_fail = max(0, len(failed_items) - repaired_pass)
        all_entries.append({
            "role": "thought",
            "content": (
                f"[VERIFY] 非Spotlight 批量修复完成: 通过 {repaired_pass}/{len(failed_items)}"
                + (f"，剩余未通过 {remaining_fail}" if remaining_fail else "，全部通过")
            ),
            "metadata": {
                "stage": "stage4_non_spotlight_repair",
                "phase": "verify",
                "round": 1,
                "failed_count": len(failed_items),
                "passed_after_repair": repaired_pass,
            },
        })


def _non_spotlight_hitl_repair(
    state: PipelineState,
    shot: dict[str, Any],
    assets: dict[str, Any],
    judge_result: dict[str, Any],
    all_entries: list[dict[str, Any]],
    num_options: int = 4,
    timeout_sec: int = 20,
) -> dict[str, Any]:
    """Generate fix options and run HITL decision for non-spotlight single repair."""
    sid = str(shot.get("shot_id", ""))
    timeline = assets.get("timeline", {})

    llm = _get_llm(state, service=_resolve_judge_service(config))
    try:
        options = vlm_render_judge.generate_fix_options(
            llm, judge_result, shot, timeline, num_options=num_options,
        )
    except Exception:
        options = vlm_render_judge._fallback_options(judge_result, num_options)
    finally:
        llm.close()

    options, recommended_option = _prepare_issue_grounded_hitl_options(
        judge_result,
        options,
        num_options,
    )

    hitl_ctx = {
        "node_name": "text_repair",
        "question": f"非Spotlight 镜头 {sid} 文本质检未通过 (score={judge_result.get('overall_score', 0):.2f})，请选择修复方案",
        "current_data": {
            "issues": judge_result.get("issues", []),
            "summary": judge_result.get("summary", ""),
            "score": judge_result.get("overall_score", 0),
        },
        "options": options,
        "recommended_option": recommended_option,
        "timeout_sec": timeout_sec,
    }

    if _should_interrupt_hitl(state):
        human_decision = interrupt({
            "type": "text_repair_review",
            "context": hitl_ctx,
            "message": f"非Spotlight 镜头 {sid} 文本质检未通过",
            "timeout_sec": timeout_sec,
        })
    else:
        human_decision = {
            "selected_option": recommended_option,
            "custom_input": "[auto_mode] 立即采用模型推荐方案" if _hitl_auto_mode(state.get("config", {})) else "",
        }

    if not isinstance(human_decision, dict):
        human_decision = _as_mapping(human_decision)
    selected = _cfg_int(human_decision, ("selected_option",), 0)
    custom_input = str(human_decision.get("custom_input", "") or "")

    chosen_label = ""
    if 0 <= selected < len(options):
        opt = options[selected]
        chosen_label = opt.get("label", "") if isinstance(opt, dict) else str(opt)
        fix_changes = opt.get("changes", {}) if isinstance(opt, dict) else {}
    elif custom_input:
        chosen_label = custom_input
        fix_changes = {"custom_fix": custom_input}
    else:
        fix_changes = _auto_select_repair_strategy(judge_result)
        chosen_label = fix_changes.get("strategy", "auto")

    hitl_entry = {
        "role": "human_feedback",
        "content": (
            f"[非Spotlight修复] 镜头 {sid} 文本Judge未通过，系统提出 {len(options)} 个方案。\n"
            f"经过 HIL 审查 (timeout={timeout_sec}s)，选择方案 {selected}: {chosen_label}\n"
            f"标记: HIL checked（单次修复，不进入渲染闭环）"
        ),
        "metadata": {
            "node_name": "text_repair",
            "shot_id": sid,
            "options": options,
            "selected_option": selected,
            "custom_input": custom_input,
            "feedback_type": "text_repair_hitl",
            "hitl_checked": True,
            "ts": datetime.now().isoformat(),
        },
    }
    all_entries.append(hitl_entry)
    return fix_changes


def _save_stage4_judge_round(
    state: PipelineState,
    shot_id: str,
    judge_round: int,
    judge_result: dict[str, Any],
) -> str:
    artifact = _save_json_artifact(
        state,
        f"stage4_judge/{shot_id}",
        f"judge_round_{judge_round:02d}.json",
        judge_result,
    )
    _save_storage_root_judge(state, shot_id, judge_round, judge_result)
    return artifact


def _save_stage4_render_check(
    state: PipelineState,
    shot_id: str,
    judge_round: int,
    cheap_check: dict[str, Any],
) -> str:
    artifact = _save_json_artifact(
        state,
        f"stage4_judge/{shot_id}",
        f"cheap_check_round_{judge_round:02d}.json",
        cheap_check,
    )
    _save_storage_root_render_check(state, shot_id, judge_round, cheap_check)
    return artifact


def _cheap_render_check(
    shot: dict[str, Any],
    assets: dict[str, Any],
    compose_result: dict[str, Any],
    accept_threshold: float,
    repair_round: int,
) -> dict[str, Any]:
    timeline = compose_result.get("timeline") if isinstance(compose_result.get("timeline"), dict) else {}
    tracks = timeline.get("tracks", {}) if isinstance(timeline.get("tracks"), dict) else {}
    visual_track = tracks.get("visual", {}) if isinstance(tracks.get("visual"), dict) else {}
    audio_track = tracks.get("audio", {}) if isinstance(tracks.get("audio"), dict) else {}
    subtitles = tracks.get("subtitles", []) if isinstance(tracks.get("subtitles"), list) else []
    decoration = tracks.get("decoration", {}) if isinstance(tracks.get("decoration"), dict) else {}

    issues: list[dict[str, Any]] = []

    def add_issue(dimension: str, severity: str, description: str, suggested_fix: str) -> None:
        issues.append({
            "dimension": dimension,
            "severity": severity,
            "description": description,
            "suggested_fix": suggested_fix,
        })

    sid = str(shot.get("shot_id", "") or compose_result.get("shot_id", ""))
    render_success = bool(compose_result.get("render_success", False))
    render_status = str(compose_result.get("render_status", "") or "")
    compose_status = str(compose_result.get("compose_status", "") or "")
    rendered_path = str(compose_result.get("rendered_path", "") or assets.get("rendered_path", "") or "")
    clip_path = str(assets.get("clip_path", "") or visual_track.get("clip_path", "") or "")
    audio_path = str(assets.get("audio_path", "") or audio_track.get("audio_path", "") or "")
    clip_meta = assets.get("clip_meta", {}) if isinstance(assets.get("clip_meta"), dict) else {}
    audio_meta = assets.get("audio_meta", {}) if isinstance(assets.get("audio_meta"), dict) else {}

    if compose_status != "success":
        add_issue("timeline_compose", "critical", "时间线合成状态异常，无法进入可靠质检", "重新执行 Tool_TimelineCompose")
    if not render_success:
        severity = "critical" if render_status in {"no_clip", "render_timeout"} or not clip_path else "warning"
        add_issue(
            "render_output",
            severity,
            f"镜头渲染未完成或不可预览: render_status={render_status or 'unknown'}",
            "优先检查 clip_path、ffmpeg 环境和输出路径，然后重新合成",
        )
    elif rendered_path and not _existing_file(rendered_path):
        add_issue("render_output", "warning", "render_success 为真但未在磁盘找到 rendered_path", "重新确认输出文件路径")

    if not clip_path:
        severity = "critical" if not clip_meta.get("clip_id") else "warning"
        add_issue("visual_match", severity, "缺少可用于最终渲染的视频素材路径 clip_path", "重新检索当前镜头视觉素材")
    if not clip_meta.get("clip_id") and not clip_path:
        add_issue("visual_match", "warning", "缺少 clip_id，后续很难追踪素材选择是否正确", "保留素材检索元数据")

    vocal_script = str(shot.get("vocal_script", "") or "").strip()
    subtitle_count = _safe_int(compose_result.get("subtitle_count"), len(subtitles))
    if vocal_script and subtitle_count <= 0:
        add_issue("subtitle_accuracy", "critical", "旁白不为空但没有生成字幕分段", "重新根据 vocal_script 构建字幕轨")
    for sub in subtitles:
        text = str(sub.get("text", "") if isinstance(sub, dict) else "")
        if _contains_internal_placeholder(text):
            add_issue("subtitle_accuracy", "critical", f"字幕暴露内部占位符或流程词: {text[:40]}", "重写字幕文本为观众可见表达")
            break

    duration = _safe_float(compose_result.get("duration_sec") or timeline.get("duration_sec") or shot.get("duration_sec"), 0.0)
    audio_duration = _safe_float(audio_meta.get("duration_sec") or audio_track.get("end"), 0.0)
    if audio_path and not _existing_file(audio_path):
        add_issue("timing_alignment", "warning", "音频路径不存在，无法确认旁白对齐", "重新生成或校正当前镜头音频")
    if duration > 0 and audio_duration > 0 and abs(duration - audio_duration) > max(1.5, audio_duration * 0.35):
        add_issue(
            "timing_alignment",
            "warning",
            f"时间线时长 {duration:.2f}s 与音频时长 {audio_duration:.2f}s 偏差较大",
            "按音频时长重新对齐 duration_sec 和字幕分段",
        )

    if shot.get("is_spotlight") and not decoration:
        add_issue("effect_quality", "warning", "Spotlight 镜头缺少装饰模板", "重新执行 Tool_DecorateShot 或选择更合适模板")
    if decoration:
        template_id = str(decoration.get("template_id", "") or "")
        if "fallback" in template_id.lower():
            add_issue("effect_quality", "warning", f"模板仍为 fallback: {template_id}", "更换为 Gold Template 中匹配镜头语义的结构")
        slots = decoration.get("slots", {}) if isinstance(decoration.get("slots"), dict) else {}
        if shot.get("is_spotlight") and not slots:
            add_issue("subtitle_accuracy", "warning", "Spotlight 装饰槽位为空", "根据 vocal_script 填充 title/subtitle 等可见槽位")
        for slot_id, value in slots.items():
            text = str(value or "")
            if _contains_internal_placeholder(text):
                add_issue(
                    "subtitle_accuracy",
                    "critical",
                    f"装饰槽位 {slot_id} 暴露内部占位符或流程词: {text[:40]}",
                    "用 vocal_script/chapter_title 提炼观众可见文案",
                )
                break
        overlay = decoration.get("overlay_config", {}) if isinstance(decoration.get("overlay_config"), dict) else {}
        opacity = _safe_float(overlay.get("opacity"), 0.0)
        if opacity > 0.78:
            add_issue("effect_quality", "warning", f"装饰遮罩透明度过高 opacity={opacity:.2f}", "将 opacity 降至 0.6-0.7")

    severity_counts = {
        "critical": sum(1 for issue in issues if issue.get("severity") == "critical"),
        "warning": sum(1 for issue in issues if issue.get("severity") == "warning"),
        "info": sum(1 for issue in issues if issue.get("severity") == "info"),
    }
    score = max(
        0.0,
        min(1.0, 1.0 - severity_counts["critical"] * 0.30 - severity_counts["warning"] * 0.12 - severity_counts["info"] * 0.03),
    )
    dimensions = _cheap_dimensions_from_issues(score, issues)
    blocking = severity_counts["critical"] > 0 or severity_counts["warning"] > 0
    passed = not blocking and score >= accept_threshold
    high_risk_reasons = [
        str(issue.get("description", ""))
        for issue in issues
        if issue.get("severity") in {"critical", "warning"}
    ]
    return {
        "shot_id": sid,
        "repair_round": repair_round,
        "passed": passed,
        "score": round(score, 3),
        "overall_score": round(score, 3),
        "dimensions": dimensions,
        "issues": issues,
        "severity_counts": severity_counts,
        "high_risk": bool(high_risk_reasons),
        "high_risk_reasons": high_risk_reasons,
        "render_success": render_success,
        "render_status": render_status,
        "rendered_path": rendered_path,
        "status": "accepted_by_rule" if passed else "needs_fix",
    }


def _needs_vlm_judge(
    shot: dict[str, Any],
    cheap_check: dict[str, Any],
    config: dict[str, Any],
    repair_round: int,
) -> bool:
    if repair_round > 0:
        policy = str(_cfg_get(config, ("stage3_trajectory", "judge", "rejudge_after_patch"), "high_risk_only") or "").lower()
        if policy == "never":
            return False
        if policy == "always":
            return True
        return bool(cheap_check.get("high_risk", False)) or not bool(cheap_check.get("passed", False))

    if not _cfg_bool(config, ("stage3_trajectory", "judge", "vlm_only_high_risk"), True):
        return True
    if cheap_check.get("high_risk", False) or not cheap_check.get("passed", False):
        return True
    if _cfg_bool(config, ("stage3_trajectory", "judge", "judge_spotlight_shots"), True) and shot.get("is_spotlight"):
        return True
    return False


def _judge_from_cheap_check(shot_id: str, cheap_check: dict[str, Any], source: str) -> dict[str, Any]:
    passed = bool(cheap_check.get("passed", False))
    score = _safe_float(cheap_check.get("overall_score", cheap_check.get("score")), 0.0)
    return {
        "shot_id": shot_id,
        "passed": passed,
        "overall_score": round(score, 3),
        "dimensions": cheap_check.get("dimensions", {}),
        "issues": cheap_check.get("issues", []),
        "verdict": "pass" if passed else "adjust",
        "summary": "本地快检通过，低风险镜头跳过 VLM" if passed else "本地快检发现问题，需要修复或复审",
        "status": "approved" if passed else "needs_fix",
        "fallback_used": False,
        "batch_judge": False,
        "source": source,
        "cheap_check": cheap_check,
    }


def _run_vlm_judge_batch(
    state: PipelineState,
    items: list[dict[str, Any]],
    pass_threshold: float,
    judge_round: int,
    all_entries: list[dict[str, Any]],
    batch_label: str,
) -> dict[str, dict[str, Any]]:
    config = state.get("config", {})
    output_dir = state.get("output_dir", "")
    num_frames = _cfg_int(config, ("stage3_trajectory", "spotlight_judge", "num_observation_frames"), 6)
    extract_timeout = _cfg_int(config, ("stage3_trajectory", "spotlight_judge", "frame_extract_timeout_sec"), 90)
    for item in items:
        sid = item["sid"]
        rendered_path = str(
            item.get("rendered_path")
            or (item.get("assets") or {}).get("rendered_path")
            or (item.get("compose_result") or {}).get("rendered_path")
            or ""
        )
        obs_dir = Path(output_dir) / "stage4_judge" / sid / f"render_observation_round_{judge_round:02d}"
        observation = vlm_render_judge.extract_render_observation(
            rendered_path,
            str(obs_dir),
            num_frames=num_frames,
            timeout_sec=extract_timeout,
        )
        obs_artifact = _save_json_artifact(
            state,
            f"stage4_judge/{sid}",
            f"render_observation_round_{judge_round:02d}.json",
            observation,
        )
        observation["artifact_path"] = obs_artifact
        item["render_observation"] = observation
        item["require_render_observation"] = True
        assets = item.get("assets")
        if isinstance(assets, dict):
            assets["render_observation"] = observation
        all_entries.extend(_log_tool(
            "Tool_RenderObservation",
            {"shot_id": sid, "judge_round": judge_round, "rendered_path": rendered_path, "num_frames": num_frames},
            {
                "status": observation.get("status", ""),
                "frame_count": observation.get("frame_count", 0),
                "artifact_path": obs_artifact,
            },
            "stage4_render_observation",
            thought=f"镜头 {sid}: 从 spotlight 渲染 MP4 抽取 {observation.get('frame_count', 0)} 帧作为 VLM observation",
        ))
    llm = _get_llm(state, service=_resolve_judge_service(config))
    try:
        results = vlm_render_judge.batch_run(
            llm,
            items,
            pass_threshold=pass_threshold,
        )
    finally:
        llm.close()

    shot_ids = [item["sid"] for item in items]
    summaries = {
        sid: {
            "passed": result.get("passed", False),
            "score": result.get("overall_score", 0.0),
            "status": result.get("status", ""),
            "fallback_used": result.get("fallback_used", False),
            "issue_count": len(result.get("issues", [])) if isinstance(result.get("issues"), list) else 0,
        }
        for sid, result in results.items()
    }
    artifact = _save_json_artifact(
        state,
        "stage4_judge",
        f"vlm_batch_{_safe_filename(batch_label)}_round_{judge_round:02d}.json",
        {
            "judge_round": judge_round,
            "shot_ids": shot_ids,
            "results": results,
            "summaries": summaries,
        },
    )
    passed_count = sum(1 for result in results.values() if result.get("passed", False))
    fallback_count = sum(1 for result in results.values() if result.get("fallback_used", False))
    all_entries.extend(_log_tool(
        "Tool_VLMRenderJudge_Batch",
        {"shot_ids": shot_ids, "judge_round": judge_round, "batch_label": batch_label},
        {
            "artifact_path": artifact,
            "shot_count": len(shot_ids),
            "passed_count": passed_count,
            "failed_count": len(shot_ids) - passed_count,
            "fallback_count": fallback_count,
            "summaries": summaries,
            "status": "success",
        },
        "stage4_judge",
        thought=f"对 {len(shot_ids)} 个高风险镜头进行一次批量 VLM 质量审查",
    ))
    return results


def _apply_stage4_auto_patch(
    state: PipelineState,
    shot: dict[str, Any],
    assets: dict[str, Any],
    compose_result: dict[str, Any],
    judge_result: dict[str, Any],
    repair_round: int,
    max_repair_rounds: int,
    fix_changes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = state.get("config", {})
    output_dir = state.get("output_dir", "")
    sid = str(shot.get("shot_id", "") or judge_result.get("shot_id", ""))
    score = _safe_float(judge_result.get("overall_score"), 0.0)
    fix_changes = fix_changes if isinstance(fix_changes, dict) and fix_changes else _auto_select_repair_strategy(judge_result)
    raw_strategies = fix_changes.get("strategies")
    if isinstance(raw_strategies, list):
        strategies = [str(item) for item in raw_strategies if str(item).strip()]
    else:
        strategies = [str(fix_changes.get("strategy") or "retime_timeline")]
    strategy_alias = {
        "repair_subtitle_and_decoration": "redecorate",
        "subtitle_accuracy": "redecorate",
        "effect_quality": "redecorate",
    }
    strategies = [strategy_alias.get(strategy, strategy) for strategy in strategies]
    if not strategies:
        strategies = ["retime_timeline"]
    fix_changes = {
        **fix_changes,
        "strategy": strategies[0],
        "strategies": strategies,
        "changed_layers": _stage4_changed_layers(strategies),
    }
    simulate_upstream = _simulate_stage4_upstream_enabled(config)
    skip_patch_llm = simulate_upstream and _cfg_bool(
        config,
        ("execution", "fake_mode", "bulk_text_only", "skip_stage4_patch_llm"),
        True,
    )

    if skip_patch_llm:
        deterministic_plan = _build_text_batch_repair_plan(shot, assets, judge_result)
        shot_patch = deterministic_plan.get("shot_patch", {}) if isinstance(deterministic_plan.get("shot_patch"), dict) else {}
        asset_patch: dict[str, Any] = {}
        timeline_patch: dict[str, Any] = {}
        patch_result = {
            "shot_patch": shot_patch,
            "asset_patch": asset_patch,
            "timeline_patch": timeline_patch,
            "repair_summary": "bulk_text_only: deterministic patch plan, skipped patch LLM",
            "deterministic_plan": deterministic_plan,
            "simulated": True,
        }
    else:
        llm_repair = _get_llm(state, service=_resolve_judge_service(config))
        try:
            patch_result = vlm_render_judge.patch(
                llm_repair,
                shot,
                compose_result.get("timeline", {}),
                judge_result,
                fix_changes,
            )
        finally:
            llm_repair.close()

    shot_patch = patch_result.get("shot_patch", {}) if isinstance(patch_result.get("shot_patch"), dict) else {}
    asset_patch = patch_result.get("asset_patch", {}) if isinstance(patch_result.get("asset_patch"), dict) else {}
    timeline_patch = patch_result.get("timeline_patch", {}) if isinstance(patch_result.get("timeline_patch"), dict) else {}
    updated_shot = _deep_merge(shot, shot_patch)
    updated_assets = _deep_merge(dict(assets), asset_patch)
    if timeline_patch:
        updated_assets["timeline"] = _deep_merge(updated_assets.get("timeline", {}), timeline_patch)

    repair_execution_entries: list[dict[str, Any]] = []
    strategy = str(fix_changes.get("strategy", "") or "")
    if any(item in {"reretrieve_visual", "regenerate_shot_execution"} for item in strategies):
        if simulate_upstream:
            vis_result = _fake_retrieve_visual_result(
                updated_shot,
                {"audio_meta": updated_assets.get("audio_meta", {})}
                if isinstance(updated_assets.get("audio_meta"), dict)
                else None,
                reason="bulk_text_only_stage4_repair",
            )
        else:
            llm_visual = _get_llm(state)
            try:
                vis_result = retrieve_visual.run(llm_visual, updated_shot, config)
            finally:
                llm_visual.close()
        if vis_result.get("clip_path"):
            updated_assets["clip_path"] = vis_result.get("clip_path", "")
            updated_assets["clip_meta"] = vis_result.get("clip_meta", {})
        elif isinstance(vis_result.get("clip_meta"), dict):
            updated_assets["clip_meta"] = vis_result.get("clip_meta", {})
        repair_execution_entries.extend(_log_tool(
            "Tool_RetrieveVisual",
            {"shot_id": sid, "visual_query": updated_shot.get("visual_query", ""), "repair_round": repair_round},
            vis_result,
            "stage4_judge_auto_repair",
            thought=(
                f"模拟对镜头 {sid} 重新检索视觉素材，保留工具调用轨迹"
                if simulate_upstream
                else f"仅对镜头 {sid} 重新检索视觉素材，供修复后合成使用"
            ),
        ))

    if (
        any(item in {"regenerate_audio", "regenerate_shot_execution"} for item in strategies)
        or str(updated_shot.get("vocal_script", "")) != str(shot.get("vocal_script", ""))
    ):
        audio_result = (
            _fake_generate_audio_result(updated_shot, reason="bulk_text_only_stage4_repair")
            if simulate_upstream
            else generate_audio.run(updated_shot, output_dir, config)
        )
        if audio_result.get("audio_path"):
            updated_assets["audio_path"] = audio_result.get("audio_path", "")
            updated_assets["audio_meta"] = audio_result.get("audio_meta", {})
        elif isinstance(audio_result.get("audio_meta"), dict):
            updated_assets["audio_meta"] = audio_result.get("audio_meta", {})
        repair_execution_entries.extend(_log_tool(
            "Tool_GenerateAudio",
            {"shot_id": sid, "repair_round": repair_round, "vocal_script_length": len(updated_shot.get("vocal_script", ""))},
            audio_result,
            "stage4_judge_auto_repair",
            thought=(
                f"模拟对镜头 {sid} 重新生成旁白音频，保留工具调用轨迹"
                if simulate_upstream
                else f"仅对镜头 {sid} 重新生成旁白音频，供修复后合成使用"
            ),
        ))

    if any(item in {"redecorate", "regenerate_shot_execution"} for item in strategies):
        if simulate_upstream:
            dec_result = _fake_decoration_repair_result(updated_shot, updated_assets)
            decoration = dec_result.get("decoration", {}) if isinstance(dec_result.get("decoration"), dict) else {}
            if decoration:
                updated_assets["decoration"] = decoration
            if isinstance(dec_result.get("verify_result"), dict):
                updated_assets["decoration_verify"] = dec_result.get("verify_result", {})
            repair_execution_entries.extend(_log_tool(
                "Tool_DecorateShot",
                {"shot_id": sid, "repair_round": repair_round, "strategy": "redecorate", "simulated": True},
                dec_result,
                "stage4_judge_auto_repair",
                thought=f"模拟对镜头 {sid} 修复模板槽位，保留装饰工具调用轨迹",
            ))
        elif updated_shot.get("is_spotlight"):
            llm_decoration = _get_llm(state)
            try:
                dec_result = decorate_shot.run(
                    llm_decoration,
                    updated_shot,
                    config=config,
                    max_repair_rounds=1,
                )
            finally:
                llm_decoration.close()
            decoration = dec_result.get("decoration", {}) if isinstance(dec_result.get("decoration"), dict) else {}
            if decoration:
                updated_assets["decoration"] = decoration
            if isinstance(dec_result.get("verify_result"), dict):
                updated_assets["decoration_verify"] = dec_result.get("verify_result", {})
            repair_execution_entries.extend(_log_tool(
                "Tool_DecorateShot",
                {"shot_id": sid, "repair_round": repair_round, "strategy": "redecorate"},
                dec_result,
                "stage4_judge_auto_repair",
                thought=f"仅对镜头 {sid} 重新生成装饰配置，复用现有视觉素材与旁白音频",
            ))
        else:
            decoration = updated_assets.get("decoration", {}) if isinstance(updated_assets.get("decoration"), dict) else {}
            if decoration:
                slots = _coerce_slots_mapping(decoration.get("slots"))
                slots["title"] = str(updated_shot.get("chapter_title") or updated_shot.get("vocal_script") or "")[:24]
                slots["subtitle"] = str(updated_shot.get("vocal_script") or "")[:36]
                decoration["slots"] = slots
                updated_assets["decoration"] = decoration

    if any(item in {"retime_timeline", "regenerate_shot_execution"} for item in strategies):
        audio_meta = updated_assets.get("audio_meta", {}) if isinstance(updated_assets.get("audio_meta"), dict) else {}
        audio_duration = _safe_float(audio_meta.get("duration_sec"), 0.0)
        current_duration = _safe_float(updated_shot.get("duration_sec"), 0.0)
        if audio_duration > 0 and audio_duration > current_duration:
            updated_shot["duration_sec"] = round(audio_duration, 2)
            updated_assets["timeline"] = _deep_merge(
                updated_assets.get("timeline", {}) if isinstance(updated_assets.get("timeline"), dict) else {},
                {"duration_sec": round(audio_duration, 2)},
            )

    updated_assets["judge_result"] = {
        **judge_result,
        "repair_round": repair_round,
        "fix_changes": fix_changes,
        "patch_result": patch_result,
    }
    updated_assets["approved"] = False
    repair_entry = {
        "role": "thought",
        "content": (
            f"镜头 {sid} 质检未通过 (score={score:.2f})，"
            f"自动执行第 {repair_round + 1}/{max_repair_rounds} 轮修复: "
            f"{fix_changes.get('strategy', 'auto')}"
        ),
        "metadata": {
            "stage": "stage4_judge_auto_repair",
            "shot_id": sid,
            "repair_round": repair_round,
            "strategy": fix_changes.get("strategy", "auto"),
            "ts": datetime.now().isoformat(),
        },
    }
    patch_payload = {
        "shot_id": sid,
        "judge_round": repair_round,
        "fix_changes": fix_changes,
        "patch_result": patch_result,
        "auto_repair": True,
        "reran_tools": [
            entry.get("tool")
            for entry in repair_execution_entries
            if entry.get("role") == "respond" and entry.get("tool")
        ],
        "strategies": strategies,
        "changed_layers": _stage4_changed_layers(strategies),
    }
    patch_artifact = _save_json_artifact(
        state,
        f"stage4_judge/{sid}",
        f"patch_round_{repair_round:02d}.json",
        patch_payload,
    )
    _save_storage_root_judge_patch(state, sid, repair_round, patch_payload)
    patch_entries = _log_tool(
        "Tool_VLMRenderJudge_Patch",
        {
            "shot_id": sid,
            "repair_round": repair_round,
            "fix_changes": fix_changes,
            "auto_repair": True,
        },
        {
            "shot_patch_keys": sorted(shot_patch.keys()),
            "asset_patch_keys": sorted(asset_patch.keys()),
            "timeline_patch_keys": sorted(timeline_patch.keys()),
            "repair_summary": patch_result.get("repair_summary", ""),
            "artifact_path": patch_artifact,
            "status": "auto_patched",
        },
        "stage4_judge_auto_repair",
        thought=(
            f"自动修复镜头 {sid} 第 {repair_round + 1} 轮，"
            f"策略: {fix_changes.get('strategy', 'auto')}"
        ),
    )
    return {
        "shot": updated_shot,
        "assets": updated_assets,
        "patch_result": patch_result,
        "reran_tools": patch_payload["reran_tools"],
        "entries": [repair_entry] + patch_entries + repair_execution_entries,
    }


def _mark_stage4_repair_exhausted(
    all_assets: dict[str, Any],
    shot_id: str,
    assets: dict[str, Any],
    judge_result: dict[str, Any],
    repair_round: int,
    repairable_threshold: float,
    all_entries: list[dict[str, Any]],
) -> None:
    score = _safe_float(judge_result.get("overall_score"), 0.0)
    final_judge = {
        **judge_result,
        "repair_exhausted": True,
        "repair_round": repair_round,
        "status": "repair_exhausted" if not judge_result.get("fallback_used", False) else "needs_review",
    }
    assets["judge_result"] = final_judge
    compose_result = assets.get("compose_result") if isinstance(assets.get("compose_result"), dict) else {}
    assets["approved"] = (
        bool(compose_result.get("render_success", False))
        and score >= repairable_threshold
        and not judge_result.get("fallback_used", False)
    )
    all_assets[shot_id] = assets
    all_entries.append({
        "role": "thought",
        "content": (
            f"镜头 {shot_id} 未进入自动修复或修复预算为 0，"
            f"最终评分 {score:.2f}，"
            f"{'标记为可用' if assets['approved'] else '标记为未通过'}"
        ),
        "metadata": {
            "stage": "stage4_judge",
            "shot_id": shot_id,
            "repair_round": repair_round,
            "final_score": score,
            "approved": assets["approved"],
            "ts": datetime.now().isoformat(),
        },
    })


def _cheap_dimensions_from_issues(score: float, issues: list[dict[str, Any]]) -> dict[str, float]:
    dims = {
        "timing_alignment": 1.0,
        "subtitle_accuracy": 1.0,
        "visual_match": 1.0,
        "effect_quality": 1.0,
        "overall_impression": score,
    }
    dimension_map = {
        "timeline_compose": "timing_alignment",
        "render_output": "overall_impression",
    }
    for issue in issues:
        raw_dim = str(issue.get("dimension", "overall_impression"))
        dim = dimension_map.get(raw_dim, raw_dim)
        if dim not in dims:
            dim = "overall_impression"
        penalty = 0.40 if issue.get("severity") == "critical" else 0.18
        dims[dim] = max(0.0, dims[dim] - penalty)
    dims["overall_impression"] = score
    return {key: round(value, 3) for key, value in dims.items()}


def _contains_internal_placeholder(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r"(段落\s*\d+|fallback|placeholder|todo|待填写|槽位|slot_|internal)", text, re.IGNORECASE))


def _estimate_audio_duration_from_text(text: str) -> float:
    stripped = re.sub(r"\s+", "", str(text or ""))
    return round(max(1.0, len(stripped) / 4.5), 2)


def _fake_generate_audio_result(shot: dict[str, Any], *, reason: str = "bulk_text_only") -> dict[str, Any]:
    sid = str(shot.get("shot_id", ""))
    text = str(shot.get("vocal_script", "") or "")
    duration = _estimate_audio_duration_from_text(text)
    return {
        "shot_id": sid,
        "audio_path": "",
        "audio_meta": {
            "duration_sec": duration,
            "backend": "simulated_annotation",
            "voice": "simulated",
            "status": "success",
            "message": reason,
            "text_length": len(text),
            "attempts": [{"backend": "simulated", "status": "success", "reason": reason}],
        },
        "status": "success",
        "simulated": True,
    }


def _fake_retrieve_visual_result(
    shot: dict[str, Any],
    audio_result: dict[str, Any] | None = None,
    *,
    reason: str = "bulk_text_only",
) -> dict[str, Any]:
    sid = str(shot.get("shot_id", ""))
    audio_meta = audio_result.get("audio_meta", {}) if isinstance(audio_result, dict) else {}
    target_duration = _safe_float(
        audio_meta.get("duration_sec"),
        _safe_float(shot.get("duration_sec"), _estimate_audio_duration_from_text(str(shot.get("vocal_script", "")))),
    )
    caption = str(shot.get("visual_query") or shot.get("chapter_title") or shot.get("vocal_script") or "")[:240]
    return {
        "shot_id": sid,
        "candidates_count": 1,
        "candidates": [],
        "selection": {
            "selected_clip_id": f"simulated_{sid}",
            "clip_path": "",
            "match_score": 0.92,
            "reason": reason,
        },
        "clip_path": "",
        "backend": "simulated_annotation",
        "retrieval_diagnostics": [{"backend": "simulated", "status": "success", "reason": reason}],
        "clip_meta": {
            "clip_id": f"simulated_{sid}",
            "caption": caption,
            "caption_zh": caption,
            "match_score": 0.92,
            "duration_sec": target_duration,
            "duration": target_duration,
            "target_duration_sec": target_duration,
            "duration_shortfall_sec": 0.0,
            "duration_constraint_met": True,
            "loop_count": 1.0,
            "orientation": "landscape",
            "resolution": "1920x1080",
            "aspect_ratio": 1.778,
            "simulated": True,
        },
        "status": "success",
        "simulated": True,
    }


def _fake_decoration_repair_result(shot: dict[str, Any], assets: dict[str, Any]) -> dict[str, Any]:
    sid = str(shot.get("shot_id", ""))
    decoration = copy.deepcopy(assets.get("decoration", {})) if isinstance(assets.get("decoration"), dict) else {}
    slots = _coerce_slots_mapping(decoration.get("slots"))
    title = str(shot.get("chapter_title") or shot.get("vocal_script") or "")[:24]
    subtitle = str(shot.get("vocal_script") or "")[:36]
    if title:
        slots.setdefault("title", title)
    if subtitle:
        slots.setdefault("subtitle", subtitle)
    decoration["slots"] = {
        key: str(value).strip()[:48]
        for key, value in slots.items()
        if str(value).strip()
    }
    decoration.setdefault("template_id", "simulated_existing_template")
    decoration.setdefault("template_type", "simulated")
    return {
        "shot_id": sid,
        "decoration": decoration,
        "verify_result": {
            "score": 0.82,
            "passed": True,
            "issues": [],
            "backend": "simulated_slot_repair",
        },
        "passed": True,
        "status": "success",
        "simulated": True,
    }


def _existing_file(path: str) -> bool:
    return bool(path) and Path(path).exists() and Path(path).stat().st_size > 0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_slots_mapping(raw: Any) -> dict[str, Any]:
    """Normalize arbitrary slot payloads into a dict without raising."""
    if isinstance(raw, dict):
        return dict(raw)
    out: dict[str, Any] = {}
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, dict):
                for key, value in item.items():
                    key_text = str(key).strip()
                    if key_text:
                        out[key_text] = value
                continue
            if isinstance(item, (list, tuple)) and len(item) == 2:
                key_text = str(item[0]).strip()
                if key_text:
                    out[key_text] = item[1]
                continue
            if isinstance(item, str):
                text = item.strip()
                if ":" in text:
                    key_text, value_text = text.split(":", 1)
                    key_text = key_text.strip()
                    if key_text:
                        out[key_text] = value_text.strip()
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_") or "batch"


# ---------------------------------------------------------------------------
# Node: stitch_video
# ---------------------------------------------------------------------------
def node_stitch_video(state: PipelineState) -> dict[str, Any]:
    """Tool_StitchVideo: Concatenate ALL rendered shots into final preview video.

    This is the final assembly step. It concatenates all shot videos (both approved
    and repair_exhausted) into one continuous video, which serves as the preview.
    BGM is mixed under the voice track if available.
    """
    shot_plan = state.get("shot_plan", [])
    shot_assets = state.get("shot_assets", {})
    total = len(shot_plan)
    approved = sum(1 for s in shot_plan
                   if shot_assets.get(s.get("shot_id", ""), {}).get("approved", False))
    logger.info("[stitch_video] 开始: 拼接 %d 个镜头 (%d 通过质检)", total, approved)

    config = state.get("config", {})
    _stitch_stage_entries: list[dict[str, Any]] = []
    _add_stage_marker(_stitch_stage_entries, "stage5_stitch", "Stage 5: 成片拼接", "start",
                      summary=f"{total} 个镜头, {approved} 通过质检")

    # ── fake 模式: 检查 skip_final_stitch 开关 ──
    if _is_fake_mode(config) and _cfg_bool(
        config, ("execution", "fake_mode", "skip_final_stitch"), False
    ):
        logger.info("[stitch_video] fake 模式: skip_final_stitch=true，跳过成片拼接")
        result = fake_utils.fake_stitch_result(shot_plan, shot_assets, state.get("output_dir", ""))
    else:
        result = stitch_video.run(
            shot_plan=shot_plan,
            shot_assets=shot_assets,
            output_dir=state.get("output_dir", ""),
            bgm_tracks=state.get("bgm_tracks", []),
            bgm_volume=_cfg_float(config, ("music", "bgm_volume"), 0.5),
            duck_under_voice=bool(_cfg_get(config, ("music", "duck_under_voice"), True)),
            config=config,
        )

    final_path = result.get("final_video_path", "")
    duration = result.get("total_duration_sec", 0)
    entries = _log_tool(
        "Tool_StitchVideo",
        {
            "shot_count": total,
            "approved_count": approved,
            "bgm_volume": _cfg_float(config, ("music", "bgm_volume"), 0.5),
            "duck_under_voice": bool(_cfg_get(config, ("music", "duck_under_voice"), True)),
        },
        result,
        "stage5_stitch",
        thought=(
            f"将全部 {total} 个镜头按顺序拼接为完整成片预览 "
            f"(时长 {duration:.1f}s, 路径: {final_path})"
        ),
    )
    stitch_artifact_path = _save_json_artifact(
        state,
        "stage5_stitch",
        "stitch_result.json",
        result,
    )

    # Also save to storage_root
    _save_storage_root_stitch(state, result)

    concat_status = str(result.get("concat_status", "") or "")
    if concat_status == "success" and final_path:
        stitch_msg = (
            f"成片预览已生成: {final_path}\n"
            f"拼接清单已落盘: {stitch_artifact_path}\n"
            f"总时长: {duration:.1f}s, 镜头数: {total}, 通过: {approved}"
        )
    else:
        stitch_msg = (
            "成片预览未生成（仅落盘拼接清单）\n"
            f"concat_status: {concat_status or 'unknown'}\n"
            f"拼接清单: {stitch_artifact_path}\n"
            f"错误信息: {str(result.get('concat_error', '') or '')[:300]}"
        )

    entries.append({
        "role": "thought",
        "content": stitch_msg,
        "metadata": {"stage": "stage5_stitch", "ts": datetime.now().isoformat()},
    })
    _add_stage_marker(entries, "stage5_stitch", "Stage 5: 成片拼接", "end",
                      summary=f"总时长 {duration:.1f}s, 通过 {approved}/{total}")

    return {
        "final_video_path": final_path,
        "current_stage": "stage5_complete",
        "trajectory": _stitch_stage_entries + entries,
    }


def _save_storage_root_stitch(state: PipelineState, result: dict[str, Any]) -> None:
    """Persist stitch results to storage_root."""
    config = state.get("config", {}) or {}
    storage_root = str((config.get("paths", {}) or {}).get("storage_root", "") or "").strip()
    if not storage_root:
        return
    video_name = _get_video_name(state)
    if not video_name:
        return

    stitch_dir = Path(storage_root) / "stage5_stitch" / video_name
    stitch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = stitch_dir / "stitch_manifest.json"
    manifest_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Keep a same-name artifact as output dir for easier lookup.
    stitch_result_path = stitch_dir / "stitch_result.json"
    stitch_result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    final_video_path = str(result.get("final_video_path", "") or "").strip()
    if final_video_path and Path(final_video_path).exists():
        target_path = stitch_dir / Path(final_video_path).name
        if Path(final_video_path).resolve() != target_path.resolve():
            shutil.copy2(final_video_path, target_path)


# ---------------------------------------------------------------------------
# Helper: auto-select repair strategy from judge issues
# ---------------------------------------------------------------------------
def _auto_select_repair_strategy(judge_result: dict[str, Any]) -> dict[str, Any]:
    """Analyze judge issues and return independent per-dimension strategies.

    The legacy caller expects a single ``strategy`` field, so the first strategy
    remains the primary one while ``strategies`` carries the full non-mutually
    exclusive repair plan.
    """
    issues = judge_result.get("issues", [])
    dimension_observation = judge_result.get("dimension_observation")
    if not isinstance(dimension_observation, dict):
        dimension_observation = _stage4_dimension_observation(judge_result, pass_threshold=0.75)
    failed_dimensions = _stage4_failed_dimensions(dimension_observation)
    if not failed_dimensions:
        failed_dimensions = [
            _stage4_issue_dimension(issue)
            for issue in _dict_list(issues)
            if str(issue.get("severity", "")).lower() in {"critical", "warning"}
        ]
    failed_dimensions = [dim for dim in STAGE4_DIMENSIONS if dim in set(failed_dimensions)]
    strategies = _stage4_strategies_for_dimensions(failed_dimensions)
    return {
        "strategy": strategies[0],
        "strategies": strategies,
        "target_dimensions": failed_dimensions,
        "judge_issues": issues,
        "dimension_observation": dimension_observation,
    }


# ---------------------------------------------------------------------------
# Helper: persist decoration results to storage_root in reference format
# ---------------------------------------------------------------------------
def _save_storage_root_decoration(
    state: PipelineState,
    shot_id: str,
    dec_result: dict[str, Any],
) -> None:
    """Persist decoration results to storage_root in the reference format."""
    config = state.get("config", {}) or {}
    storage_root = str((config.get("paths", {}) or {}).get("storage_root", "") or "").strip()
    if not storage_root:
        return
    video_name = _get_video_name(state)
    if not video_name:
        return

    # Save in reference format: storage_root/stage3_trajectory/<video_name>/<shot_id>/
    stage3_dir = Path(storage_root) / "stage3_trajectory" / video_name / shot_id
    stage3_dir.mkdir(parents=True, exist_ok=True)
    traj_id = f"{shot_id}__traj00__best"
    candidate_dir = stage3_dir / "candidates" / traj_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = stage3_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    decoration = dec_result.get("decoration", {})
    verify_result = dec_result.get("verify_result", {})
    score = _cfg_float({"score": verify_result.get("score", 0.0)}, ("score",), 0.0) if isinstance(verify_result, dict) else 0.0
    issues = verify_result.get("issues", []) if isinstance(verify_result, dict) else []
    first_issue = issues[0] if isinstance(issues, list) and issues else None
    patch_priority = _decoration_patch_priority(issues, shot_id)
    observation = {
        "overall_score": score,
        "passed": bool(dec_result.get("passed", False)),
        "first_error": first_issue,
        "verifiers": [
            {
                "name": "Tool_DecorateShot_Check",
                "score": score,
                "passed": bool(dec_result.get("passed", False)),
                "issues": issues if isinstance(issues, list) else [],
            }
        ],
        "patch_priority": patch_priority,
    }
    reflection = {
        "error": "未发现需要修复的高优先级问题，当前装饰已达可接受质量。"
        if observation["passed"] else str((first_issue or {}).get("description") if isinstance(first_issue, dict) else first_issue or "装饰质量未通过"),
        "scope": "shot",
        "locus": f"shot:{shot_id}",
        "repair_strategy": "接受当前 decoration" if observation["passed"] else "按 patch_priority 执行 revise_slots/swap_structure/adjust_overlay 后重新检查",
        "chosen_op": "" if observation["passed"] or not patch_priority else patch_priority[0].get("op", ""),
    }
    plan_payload = {
        "structure_id": decoration.get("template_id", ""),
        "template_type": decoration.get("template_type", ""),
        "slots": decoration.get("slots", {}),
        "animation": decoration.get("animation", {}),
        "overlay_config": decoration.get("overlay_config", {}),
        "timing": decoration.get("timing", {}),
        "rationale": decoration.get("design_rationale_mapping", ""),
    }

    # template_plan.json
    plan_path = stage3_dir / "template_plan.json"
    plan_path.write_text(json.dumps(plan_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (candidate_dir / "template_plan.json").write_text(json.dumps(plan_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # verify_report.json
    verify_path = stage3_dir / "verify_report.json"
    verify_path.write_text(json.dumps({
        "passed": dec_result.get("passed", False),
        "verify_result": verify_result,
        "repair_history": dec_result.get("repair_history", []),
        "repair_rounds": dec_result.get("repair_rounds", 0),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (candidate_dir / "observation.json").write_text(json.dumps(observation, ensure_ascii=False, indent=2), encoding="utf-8")
    (candidate_dir / "slot_validation.json").write_text(json.dumps({
        "valid": observation["passed"],
        "issues": issues if isinstance(issues, list) else [],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (candidate_dir / "judge.json").write_text(json.dumps({
        "overall_score": score,
        "passed": observation["passed"],
        "issues": issues if isinstance(issues, list) else [],
        "backend": "Tool_DecorateShot_Check",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    round_log = {
        "schema_version": "aicut_decoration_round_log_v1",
        "round_index": int(dec_result.get("repair_rounds", 0) or 0),
        "shot_id": shot_id,
        "chain_id": traj_id,
        "policy_name": "best",
        "state_before": {},
        "observation": observation,
        "reflection": reflection,
        "patch": patch_priority[:1],
        "state_after": plan_payload,
        "decision": "accept" if observation["passed"] else "repair_exhausted",
        "reward": {"verifier_score": score, "passed": 1.0 if observation["passed"] else 0.0},
    }
    (candidate_dir / "round_logs.jsonl").write_text(json.dumps(round_log, ensure_ascii=False) + "\n", encoding="utf-8")
    repair_history = dec_result.get("repair_history", [])
    if isinstance(repair_history, list):
        (candidate_dir / "repair_history.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in repair_history if isinstance(item, dict)),
            encoding="utf-8",
        )
    trajectory = {
        "schema_version": "aicut_stage3_trajectory_v1",
        "trajectory_id": traj_id,
        "chain_id": traj_id,
        "route": "template",
        "shot_id": shot_id,
        "template_plan": plan_payload,
        "observation": observation,
        "reflection": reflection,
        "patch_priority": patch_priority,
        "round_log": round_log,
        "iterations": [round_log],
        "is_terminal": True,
        "is_accepted": bool(dec_result.get("passed", False)),
    }
    (candidate_dir / "trajectory.json").write_text(json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8")
    (trajectories_dir / f"{traj_id}.json").write_text(json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8")
    (stage3_dir / "route.json").write_text(json.dumps({
        "route": "template",
        "shot_id": shot_id,
        "is_spotlight": True,
        "candidates": [traj_id],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    label = "good" if observation["passed"] and score >= 0.75 else ("repairable" if score >= 0.45 else "bad")
    (stage3_dir / "classification.json").write_text(json.dumps({
        "selected_trajectory_id": traj_id,
        "best": {"trajectory_id": traj_id, "label": label, "overall_score": score},
        "candidates": [{"trajectory_id": traj_id, "label": label, "overall_score": score}],
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_storage_root_judge(
    state: PipelineState,
    shot_id: str,
    judge_round: int,
    judge_result: dict[str, Any],
) -> None:
    """Persist judge results to storage_root in the reference format."""
    config = state.get("config", {}) or {}
    storage_root = str((config.get("paths", {}) or {}).get("storage_root", "") or "").strip()
    if not storage_root:
        return
    video_name = _get_video_name(state)
    if not video_name:
        return

    stage4_dir = Path(storage_root) / "stage4_judge" / video_name / shot_id
    stage4_dir.mkdir(parents=True, exist_ok=True)

    judge_path = stage4_dir / f"judge_round_{judge_round:02d}.json"
    judge_path.write_text(
        json.dumps(judge_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _save_storage_root_render_check(
    state: PipelineState,
    shot_id: str,
    judge_round: int,
    cheap_check: dict[str, Any],
) -> None:
    """Persist cheap render-check results to storage_root next to VLM judge rounds."""
    config = state.get("config", {}) or {}
    storage_root = str((config.get("paths", {}) or {}).get("storage_root", "") or "").strip()
    if not storage_root:
        return
    video_name = _get_video_name(state)
    if not video_name:
        return

    stage4_dir = Path(storage_root) / "stage4_judge" / video_name / shot_id
    stage4_dir.mkdir(parents=True, exist_ok=True)
    check_path = stage4_dir / f"cheap_check_round_{judge_round:02d}.json"
    check_path.write_text(json.dumps(cheap_check, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_storage_root_judge_patch(
    state: PipelineState,
    shot_id: str,
    judge_round: int,
    patch_result: dict[str, Any],
) -> None:
    """Persist judge repair patch to storage_root next to judge rounds."""
    config = state.get("config", {}) or {}
    storage_root = str((config.get("paths", {}) or {}).get("storage_root", "") or "").strip()
    if not storage_root:
        return
    video_name = _get_video_name(state)
    if not video_name:
        return

    stage4_dir = Path(storage_root) / "stage4_judge" / video_name / shot_id
    stage4_dir.mkdir(parents=True, exist_ok=True)
    patch_path = stage4_dir / f"patch_round_{judge_round:02d}.json"
    patch_path.write_text(json.dumps(patch_result, ensure_ascii=False, indent=2), encoding="utf-8")


def _decoration_patch_priority(issues: Any, shot_id: str) -> list[dict[str, Any]]:
    if not isinstance(issues, list):
        return []
    out: list[dict[str, Any]] = []
    for issue in issues:
        if not isinstance(issue, dict):
            text = str(issue)
            op = "swap_structure" if "模板" in text else "revise_slots"
            slot_id = "subtitle" if "副标题" in text else ("title" if "标题" in text else "")
            out.append({
                "op": op,
                "target_shot_id": shot_id,
                "args": {"slot_ids": [slot_id] if slot_id else []},
                "scope": "shot",
                "rationale": text,
                "cost": 1.0,
            })
            continue
        desc = str(issue.get("description") or issue.get("detail") or "")
        action = str(issue.get("suggested_action") or "")
        if action not in {"revise_slots", "swap_structure", "adjust_overlay"}:
            action = "swap_structure" if "模板" in desc else ("adjust_overlay" if "透明度" in desc or "遮罩" in desc else "revise_slots")
        slot_id = str(issue.get("slot_id") or "")
        out.append({
            "op": action,
            "target_shot_id": shot_id,
            "args": {"slot_ids": [slot_id] if slot_id else [], "issue": desc},
            "scope": "shot",
            "source_verifier": "Tool_DecorateShot_Check",
            "rationale": str(issue.get("suggested_fix") or desc),
            "cost": 1.0,
        })
    return out


def _get_video_name(state: PipelineState) -> str:
    """Extract video name from state for storage path."""
    video_path = str(state.get("input_video_path", "") or "")
    if video_path:
        return Path(video_path).stem
    input_text = str(state.get("input_text", "") or "")
    if input_text:
        return hashlib.sha1(input_text.encode("utf-8")).hexdigest()[:12]
    return ""


# ---------------------------------------------------------------------------
# Node: save_trajectory
# ---------------------------------------------------------------------------
THOUGHT_POLISH_SYSTEM = """\
你是视频生成智能体的轨迹编辑器。你只润色已有 THINK 文本，让它更像清晰的导演决策记录。
硬性规则：
1. 只能输出 JSON object。
2. 不要添加未发生的工具调用、结果、分数或事实。
3. 不要输出 <think>、Markdown 或解释。
4. 每条 content 保持 20~90 字，保留原来的阶段意图和因果关系。"""

THOUGHT_POLISH_TEMPLATE = """\
请润色以下 pipeline THINK 文本。保持 index 不变，只改 content。

THINK 列表:
{thoughts_json}

输出 JSON:
{{
  "thoughts": [
    {{"index": 0, "content": "润色后的 THINK"}}
  ]
}}"""


def _polish_trajectory_thoughts(state: PipelineState, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    config = state.get("config", {})
    if not _cfg_bool(config, ("trajectory", "polish_thoughts"), True):
        return entries
    thought_rows: list[dict[str, Any]] = []
    entry_indices: list[int] = []
    for idx, entry in enumerate(entries):
        if entry.get("role") != "thought":
            continue
        content = str(entry.get("content", "") or "").strip()
        if not content:
            continue
        entry_indices.append(idx)
        thought_rows.append({
            "index": len(thought_rows),
            "stage": (entry.get("metadata") or {}).get("stage", ""),
            "tool": entry.get("tool") or (entry.get("metadata") or {}).get("tool", ""),
            "content": content[:260],
        })
    if not thought_rows:
        return entries
    max_rows = max(1, _cfg_int(config, ("trajectory", "polish_max_thoughts"), 120))
    clipped_rows = thought_rows[:max_rows]
    prompt = THOUGHT_POLISH_TEMPLATE.format(
        thoughts_json=json.dumps(clipped_rows, ensure_ascii=False, indent=2),
    )
    llm = _get_llm(state)
    try:
        raw = _as_mapping(llm.chat_json(THOUGHT_POLISH_SYSTEM, prompt, temperature=0.25))
    except Exception as exc:
        logger.warning("[save_trajectory] thought polish skipped: %s", exc)
        return entries
    finally:
        llm.close()
    polished_rows = raw.get("thoughts", [])
    if not isinstance(polished_rows, list):
        return entries
    replacements: dict[int, str] = {}
    for row in polished_rows:
        if not isinstance(row, dict):
            continue
        try:
            rel_idx = int(row.get("index"))
        except (TypeError, ValueError):
            continue
        content = str(row.get("content", "") or "").strip()
        if 0 <= rel_idx < len(clipped_rows) and content:
            replacements[entry_indices[rel_idx]] = content[:360]
    if not replacements:
        return entries
    out = [dict(entry) for entry in entries]
    for entry_idx, content in replacements.items():
        out[entry_idx] = {**out[entry_idx], "content": content}
        metadata = dict(out[entry_idx].get("metadata") or {})
        metadata["thought_polished"] = True
        out[entry_idx]["metadata"] = metadata
    return out


def node_save_trajectory(state: PipelineState) -> dict[str, Any]:
    """Save accumulated trajectory to SFT-ready format."""
    traj_dir = state.get("config", {}).get("trajectory", {}).get(
        "output_dir", "./outputs/trajectories"
    )
    logger = TrajectoryLogger(
        project_id=state.get("project_id", "unknown"),
        output_dir=traj_dir,
    )
    logger.entries = _polish_trajectory_thoughts(state, list(state.get("trajectory", [])))
    saved_path = logger.save(suffix=state.get("mode", "inference"))

    return {
        "trajectory": [{
            "role": "thought",
            "content": f"轨迹 THINK 已完成统一润色并保存至 {saved_path}，共 {len(logger.entries)} 条记录",
            "metadata": {"stage": "save_trajectory", "ts": datetime.now().isoformat()},
        }],
    }


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------
def route_after_modal_preflight(state: PipelineState) -> str:
    if state.get("mode") == "data_pipeline":
        return "data_stage1"
    return "segment_article"


def route_after_init(state: PipelineState) -> str:
    """Route after init. By default skip modal preflight to save time."""
    config = state.get("config", {})
    run_preflight = bool(_cfg_get(config, ("modal_probe", "run_before_pipeline"), False))
    if run_preflight:
        return "modal_tools_preflight"
    if state.get("mode") == "data_pipeline":
        return "data_stage1"
    return "segment_article"


def route_after_reflect(state: PipelineState) -> str:
    if state.get("plan_approved"):
        return "visual_audio_batch"
    config = state.get("config", {})
    max_rounds = _cfg_int(
        config,
        ("stage2_segment", "max_repair_rounds"),
        3,
    )
    if state.get("reflect_round", 0) >= max_rounds:
        if _cfg_bool(config, ("stage2_segment", "continue_on_reflect_failure"), False):
            logger.warning(
                "[route_after_reflect] self_reflect达到最大轮数(%d)但配置允许继续，进入后续阶段",
                max_rounds,
            )
            return "visual_audio_batch"
        return "reflect_failed"
    return "self_reflect"


# ---------------------------------------------------------------------------
# 标注末端真实物化：compose 全部完成后，一次性渲染全部镜头
# ---------------------------------------------------------------------------
def _final_hyperframes_render_pass(
    state: PipelineState,
    all_assets: dict[str, Any],
    all_entries: list[dict[str, Any]],
    output_dir: str,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """fake 模式末端：把全部 fake compose 结果物化成真实 MP4 后再拼接。

    spotlight 是否使用 HyperFrames 由 final_hyperframes_render 控制；
    普通镜头始终使用轻量 ffmpeg。这样 stitch_video 不会再收到不存在的 fake 路径。
    """
    shot_plan = [
        shot
        for shot in _dict_list(state.get("shot_plan", []))
        if str(shot.get("shot_id", "")) in all_assets
    ]
    if not shot_plan:
        logger.info("[final_render] 无可渲染镜头，跳过末端物化")
        return all_entries, all_assets

    use_hyperframes = _cfg_bool(
        config,
        ("execution", "fake_mode", "final_hyperframes_render"),
        False,
    )
    render_workers = max(
        1,
        _cfg_int(config, ("execution", "fake_mode", "final_render_workers"), 2),
    )
    logger.info(
        "[final_render] 开始: 物化 %d 个镜头 (spotlight_hyperframes=%s, workers=%d)",
        len(shot_plan),
        use_hyperframes,
        render_workers,
    )
    _add_stage_marker(
        all_entries,
        "stage5_final_render",
        "Stage 5a: 最终镜头物化",
        "start",
        summary=(
            f"{len(shot_plan)} 个镜头；spotlight "
            f"{'使用 HyperFrames' if use_hyperframes else '使用 ffmpeg'}"
        ),
    )

    render_results: dict[str, dict] = {}
    success_count = 0
    hyperframes_count = 0
    ffmpeg_count = 0

    def _render_one(shot: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any], str]:
        sid = str(shot.get("shot_id", ""))
        assets = all_assets.get(sid, {})
        if not isinstance(assets, dict):
            assets = {"shot_id": sid}
        backend = (
            "hyperframes"
            if use_hyperframes and shot.get("is_spotlight") and assets.get("decoration")
            else "ffmpeg"
        )
        logger.info("[final_render]   渲染镜头 %s backend=%s", sid, backend)
        try:
            compose_result = timeline_compose.run(
                shot,
                assets,
                output_dir,
                config=config,
                render_backend=backend,
            )
        except Exception as exc:
            logger.exception("[final_render] 渲染失败 shot=%s backend=%s: %s", sid, backend, exc)
            compose_result = {
                "shot_id": sid,
                "render_success": False,
                "render_status": f"exception: {exc}",
                "rendered_path": "",
                "timeline": assets.get("timeline", {}),
                "compose_status": "error",
                "render_backend": backend,
            }
        return sid, shot, compose_result, backend

    with ThreadPoolExecutor(max_workers=min(render_workers, len(shot_plan))) as executor:
        futures = {executor.submit(_render_one, shot): shot for shot in shot_plan}
        completed = 0
        for future in as_completed(futures):
            sid, shot, compose_result, backend = future.result()
            completed += 1
            _emit_progress(
                "Tool_FinalRender",
                f"{backend} 渲染",
                completed,
                len(shot_plan),
                f"镜头 {sid} {'成功' if compose_result.get('render_success') else '失败'}",
                "stage5_final_render",
            )
            if backend == "hyperframes":
                hyperframes_count += 1
            else:
                ffmpeg_count += 1
            if compose_result.get("render_success"):
                success_count += 1
                render_results[sid] = {
                    "rendered_path": compose_result.get("rendered_path", ""),
                    "render_status": compose_result.get("render_status", ""),
                    "render_success": True,
                    "render_backend": backend,
                }
            else:
                render_results[sid] = {
                    "rendered_path": "",
                    "render_status": compose_result.get("render_status", "render_failed"),
                    "render_success": False,
                    "render_backend": backend,
                }
            assets = all_assets.get(sid, {})
            if not isinstance(assets, dict):
                assets = {"shot_id": sid}
            all_assets[sid] = {
                **assets,
                "shot_id": sid,
                "timeline": compose_result.get("timeline", assets.get("timeline", {})),
                "rendered_path": compose_result.get("rendered_path", ""),
                "compose_result": compose_result,
            }

    # ── 轨迹：只记录一条批量渲染汇总 entry ──
    render_entry = _log_tool(
        "Tool_FinalRender",
        {
            "shot_ids": [s.get("shot_id", "") for s in shot_plan],
            "shot_count": len(shot_plan),
            "spotlight_hyperframes": use_hyperframes,
            "workers": render_workers,
            "mode": "fake_mode_final_render",
        },
        {
            "shots": render_results,
            "render_success_count": success_count,
            "render_failed_count": len(shot_plan) - success_count,
            "hyperframes_count": hyperframes_count,
            "ffmpeg_count": ffmpeg_count,
            "status": "success" if success_count == len(shot_plan) else "partial_failure",
        },
        "stage5_final_render",
        thought=(
            f"标注收尾——{len(shot_plan)} 个镜头跑完最终物化: "
            f"成功了 {success_count} 个，其中 HyperFrames 装饰了 {hyperframes_count} 个 spotlight，"
            f"剩下的 {ffmpeg_count} 个常规镜头用 ffmpeg 快速出了片，可以进拼接了"
        ),
    )
    all_entries.extend(render_entry)
    _add_stage_marker(
        all_entries,
        "stage5_final_render",
        "Stage 5a: 最终镜头物化",
        "end",
        summary=f"真实渲染成功 {success_count}/{len(shot_plan)}",
    )

    logger.info(
        "[final_render] 完成: %d/%d success",
        success_count,
        len(shot_plan),
    )
    return all_entries, all_assets


def node_final_render(state: PipelineState) -> dict[str, Any]:
    """Materialize all repaired fake-mode shots before final stitching."""
    entries: list[dict[str, Any]] = []
    entries, all_assets = _final_hyperframes_render_pass(
        state,
        dict(state.get("shot_assets", {})),
        entries,
        state.get("output_dir", ""),
        state.get("config", {}),
    )
    return {
        "shot_assets": all_assets,
        "current_stage": "stage5_final_render_done",
        "trajectory": entries,
    }


def route_after_compose(state: PipelineState) -> str:
    idx = state.get("compose_shot_idx", 0)
    total = len(state.get("shot_plan", []))
    if idx >= total:
        config = state.get("config", {})
        if (
            _is_fake_mode(config)
            and not _cfg_bool(config, ("execution", "fake_mode", "skip_final_stitch"), False)
        ):
            return "final_render"
        return "stitch_video"
    return "compose_and_judge"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_graph() -> StateGraph:
    """Construct the full LangGraph state machine.

    Style check is removed (redundant with per-shot VLM judge).
    stitch_video connects directly to save_trajectory.
    severity_rank inserted before compose_and_judge for dual-track routing.
    """
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("init", node_init)
    graph.add_node("modal_tools_preflight", node_modal_tools_preflight)
    graph.add_node("data_stage1", node_data_stage1)
    graph.add_node("segment_article", node_segment_article)
    graph.add_node("plan_shots", node_plan_shots)
    graph.add_node("self_reflect", node_self_reflect)
    graph.add_node("reflect_failed", node_reflect_failed)
    graph.add_node("visual_audio_batch", node_visual_audio_batch)
    graph.add_node("decorate_spotlights", node_decorate_spotlights)
    graph.add_node("music_search", node_music_search)
    graph.add_node("severity_rank", node_severity_rank)
    graph.add_node("compose_and_judge", node_compose_and_judge)
    graph.add_node("final_render", node_final_render)
    graph.add_node("stitch_video", node_stitch_video)
    graph.add_node("save_trajectory", node_save_trajectory)

    # Edges
    graph.add_edge(START, "init")
    graph.add_conditional_edges("init", route_after_init, {
        "modal_tools_preflight": "modal_tools_preflight",
        "data_stage1": "data_stage1",
        "segment_article": "segment_article",
    })
    graph.add_conditional_edges("modal_tools_preflight", route_after_modal_preflight, {
        "data_stage1": "data_stage1",
        "segment_article": "segment_article",
    })
    graph.add_edge("data_stage1", "segment_article")
    graph.add_edge("segment_article", "plan_shots")
    graph.add_edge("plan_shots", "self_reflect")
    graph.add_conditional_edges("self_reflect", route_after_reflect, {
        "visual_audio_batch": "visual_audio_batch",
        "self_reflect": "self_reflect",
        "reflect_failed": "reflect_failed",
    })
    graph.add_edge("reflect_failed", "save_trajectory")
    graph.add_edge("visual_audio_batch", "decorate_spotlights")
    graph.add_edge("decorate_spotlights", "music_search")
    graph.add_edge("music_search", "severity_rank")
    graph.add_edge("severity_rank", "compose_and_judge")
    graph.add_conditional_edges("compose_and_judge", route_after_compose, {
        "final_render": "final_render",
        "stitch_video": "stitch_video",
        "compose_and_judge": "compose_and_judge",
    })
    graph.add_edge("final_render", "stitch_video")
    graph.add_edge("stitch_video", "save_trajectory")
    graph.add_edge("save_trajectory", END)

    return graph


def compile_graph(checkpointer=None):
    """Compile the graph with optional checkpointer for HITL persistence."""
    graph = build_graph()
    return graph.compile(checkpointer=checkpointer)
