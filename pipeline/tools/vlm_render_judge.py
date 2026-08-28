"""Tool_VLMRenderJudge: Quality check rendered shot video with VLM scoring."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import subprocess
from pathlib import Path
from typing import Any

from ..llm import LLMClient

SYSTEM_PROMPT = """\
你是视频渲染质量审查专家。当前模式是“渲染 MP4 多模态审查”。
你必须主要依据随 prompt 提供的 MP4 抽帧画面判断可见结果，同时参考镜头规划和时间线元数据。

硬性输出规则:
1. 只能输出一个合法 JSON object，不能输出 JSON array、Markdown、解释文字、代码块或 <think>。
2. issues、options 等集合字段必须是 JSON array，array 中每一项必须是 JSON object。
3. 一旦存在 critical 或 warning 级别问题，passed 必须为 false。
4. 不要输出推理过程，只输出最终 JSON。"""

JUDGE_TEMPLATE = """\
请对以下渲染后的镜头进行质量检查:

镜头规划:
{shot_json}

时间线合成结果:
{timeline_json}

素材选择:
{clip_meta_json}

检查维度(每项0-1分):
1. 时序对齐: 音频与视频时长是否匹配
2. 字幕正确: 字幕分段是否与旁白对应
3. 素材匹配: 视觉素材是否匹配 visual_query 描述
4. 动效合理: 装饰动效是否适合镜头节奏
5. 整体质量: 综合观感评分

输出(严格JSON):
{{
  "passed": true/false,
  "overall_score": 0.0-1.0,
  "dimensions": {{
    "timing_alignment": 0.0-1.0,
    "subtitle_accuracy": 0.0-1.0,
    "visual_match": 0.0-1.0,
    "effect_quality": 0.0-1.0,
    "overall_impression": 0.0-1.0
  }},
  "issues": [
    {{
      "dimension": "检查维度",
      "severity": "critical/warning/info",
      "description": "问题描述",
      "suggested_fix": "修复建议"
    }}
  ],
  "verdict": "pass/rerender/adjust",
  "summary": "综合评价"
}}"""

BATCH_JUDGE_TEMPLATE = """\
请对以下一组高风险渲染镜头进行批量质量检查。每个镜头已经经过本地快检，
你需要重点根据 MP4 抽帧判断语义匹配、素材匹配、字幕/装饰是否符合观众可见结果。

批量镜头:
{items_json}

多模态 observation:
{observation_manifest_json}

检查维度(每项0-1分):
1. timing_alignment: 音频、字幕、视频时长是否匹配
2. subtitle_accuracy: 字幕/装饰文案是否与旁白对应，不能暴露内部占位符
3. visual_match: 视觉素材是否匹配 visual_query 和 vocal_script
4. effect_quality: 装饰模板、动效、遮罩是否适合镜头节奏
5. overall_impression: 综合观感评分

输出(严格JSON object):
{{
  "shots": [
    {{
      "shot_id": "镜头ID",
      "passed": true/false,
      "overall_score": 0.0-1.0,
      "dimensions": {{
        "timing_alignment": 0.0-1.0,
        "subtitle_accuracy": 0.0-1.0,
        "visual_match": 0.0-1.0,
        "effect_quality": 0.0-1.0,
        "overall_impression": 0.0-1.0
      }},
      "issues": [
        {{
          "dimension": "timing_alignment/subtitle_accuracy/visual_match/effect_quality/overall_impression",
          "severity": "critical/warning/info",
          "description": "问题描述",
          "suggested_fix": "修复建议"
        }}
      ],
      "verdict": "pass/rerender/adjust",
      "summary": "综合评价"
    }}
  ]
}}"""

OPTIONS_TEMPLATE = """\
对以下镜头的质量问题，请提供 {num_options} 个不同的修复方案:

质检结果:
{judge_json}

镜头信息:
{shot_json}

时间线:
{timeline_json}

输出要求(严格JSON object格式):
{{
  "options": [
    {{
      "option_id": 0,
      "label": "方案简称",
      "description": "详细修复说明",
      "changes": {{"修改内容": "..."}}
    }}
  ]
}}"""

PATCH_TEMPLATE = """\
请根据镜头质检问题和人工选择的修复方案，输出可执行的修复补丁。

镜头信息:
{shot_json}

时间线信息:
{timeline_json}

质检结果:
{judge_json}

人工修复选择:
{fix_changes_json}

输出(严格JSON):
{{
  "shot_patch": {{
    "vocal_script": "可选，修正后的旁白文本",
    "visual_query": "可选，修正后的检索词",
    "duration_sec": 8.0
  }},
  "asset_patch": {{
    "clip_path": "可选",
    "audio_path": "可选",
    "decoration": {{}}
  }},
  "timeline_patch": {{
    "transition": {{"type": "crossfade", "duration_ms": 400}}
  }},
  "repair_summary": "本轮修复说明"
}}"""


def run(
    llm: LLMClient,
    shot: dict[str, Any],
    timeline: dict[str, Any],
    clip_meta: dict[str, Any],
    pass_threshold: float = 0.75,
) -> dict[str, Any]:
    """Judge rendered shot quality."""
    prompt = JUDGE_TEMPLATE.format(
        shot_json=json.dumps(shot, ensure_ascii=False, indent=2),
        timeline_json=json.dumps(timeline, ensure_ascii=False, indent=2),
        clip_meta_json=json.dumps(clip_meta, ensure_ascii=False, indent=2),
    )
    llm_error = ""
    try:
        result = _as_dict(llm.chat_json(SYSTEM_PROMPT, prompt, temperature=0.2))
    except Exception as exc:
        result = {}
        llm_error = str(exc)
    if _is_empty_judge_result(result):
        result = _fallback_judge(shot, timeline, clip_meta, llm_error=llm_error)
        result["fallback_used"] = True
    else:
        result["fallback_used"] = False
    return _finalize_judge_result(
        result,
        shot,
        timeline,
        clip_meta,
        pass_threshold=pass_threshold,
        llm_error=llm_error,
        batch_judge=False,
    )


def batch_run(
    llm: LLMClient,
    items: list[dict[str, Any]],
    pass_threshold: float = 0.75,
) -> dict[str, dict[str, Any]]:
    """Judge multiple high-risk rendered shots in one LLM/VLM call."""
    if not items:
        return {}
    if any(_uses_render_observation(item) for item in items) and len(items) > 1:
        results: dict[str, dict[str, Any]] = {}
        for item in items:
            results.update(batch_run(llm, [item], pass_threshold=pass_threshold))
        return results

    compact_items: list[dict[str, Any]] = []
    shot_lookup: dict[str, dict[str, Any]] = {}
    observation_manifest: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        shot = item.get("shot") if isinstance(item.get("shot"), dict) else {}
        sid = str(shot.get("shot_id") or item.get("shot_id") or f"shot_{idx + 1}")
        timeline = item.get("timeline") if isinstance(item.get("timeline"), dict) else {}
        clip_meta = item.get("clip_meta") if isinstance(item.get("clip_meta"), dict) else {}
        cheap_check = item.get("cheap_check") if isinstance(item.get("cheap_check"), dict) else {}
        render_observation = item.get("render_observation") if isinstance(item.get("render_observation"), dict) else {}
        shot_lookup[sid] = {
            "shot": shot,
            "timeline": timeline,
            "clip_meta": clip_meta,
            "render_observation": render_observation,
            "require_render_observation": bool(item.get("require_render_observation", False)),
        }
        frames = _observation_frames(render_observation)
        observation_manifest.append({
            "shot_id": sid,
            "source": "mp4_frames" if frames else "missing",
            "rendered_path": render_observation.get("rendered_path", item.get("rendered_path", "")),
            "frame_count": len(frames),
            "timestamps_sec": [frame.get("timestamp_sec") for frame in frames],
            "status": render_observation.get("status", ""),
        })
        compact_items.append({
            "shot_id": sid,
            "shot": {
                "shot_id": sid,
                "chapter_title": shot.get("chapter_title", ""),
                "vocal_script": shot.get("vocal_script", ""),
                "visual_query": shot.get("visual_query", ""),
                "duration_sec": shot.get("duration_sec", 0),
                "is_spotlight": bool(shot.get("is_spotlight", False)),
                "design_rationale": shot.get("design_rationale", ""),
            },
            "timeline": _compact_timeline(timeline),
            "clip_meta": {
                "clip_id": clip_meta.get("clip_id", ""),
                "caption": clip_meta.get("caption", clip_meta.get("caption_zh", "")),
                "score": clip_meta.get("score", clip_meta.get("similarity", "")),
                "status": clip_meta.get("status", ""),
            },
            "cheap_check": {
                "passed": cheap_check.get("passed", False),
                "score": cheap_check.get("score", 0.0),
                "issues": cheap_check.get("issues", []),
                "high_risk_reasons": cheap_check.get("high_risk_reasons", []),
            },
            "render_observation": observation_manifest[-1],
        })

    missing_required = [
        sid for sid, source in shot_lookup.items()
        if source.get("require_render_observation") and not _observation_frames(source.get("render_observation", {}))
    ]
    if missing_required:
        return {
            sid: _finalize_judge_result(
                _observation_failure_judge(
                    source["shot"],
                    source["timeline"],
                    source["clip_meta"],
                    source.get("render_observation", {}),
                ),
                source["shot"],
                source["timeline"],
                source["clip_meta"],
                pass_threshold=pass_threshold,
                llm_error="render_observation_missing",
                batch_judge=True,
                render_observation=source.get("render_observation", {}),
            )
            for sid, source in shot_lookup.items()
        }

    llm_error = ""
    try:
        prompt = BATCH_JUDGE_TEMPLATE.format(
            items_json=json.dumps(compact_items, ensure_ascii=False, indent=2),
            observation_manifest_json=json.dumps(observation_manifest, ensure_ascii=False, indent=2),
        )
        if any(_observation_frames(source.get("render_observation", {})) for source in shot_lookup.values()):
            raw = _chat_json_with_render_frames(llm, prompt, shot_lookup, temperature=0.1)
        else:
            raw = llm.chat_json(SYSTEM_PROMPT, prompt, temperature=0.2)
    except Exception as exc:
        raw = {}
        llm_error = str(exc)

    rows = _extract_batch_rows(raw)
    by_sid = {
        str(row.get("shot_id") or "").strip(): row
        for row in rows
        if str(row.get("shot_id") or "").strip()
    }
    results: dict[str, dict[str, Any]] = {}
    for sid, source in shot_lookup.items():
        row = by_sid.get(sid)
        row_error = llm_error
        if row is None or _is_empty_judge_result(row):
            row_error = llm_error or "batch_result_missing"
            row = _fallback_judge(
                source["shot"],
                source["timeline"],
                source["clip_meta"],
                llm_error=row_error,
            )
            row["fallback_used"] = True
        else:
            row["fallback_used"] = False
        results[sid] = _finalize_judge_result(
            row,
            source["shot"],
            source["timeline"],
            source["clip_meta"],
            pass_threshold=pass_threshold,
            llm_error=row_error,
            batch_judge=True,
            render_observation=source.get("render_observation", {}),
        )
    return results


def _finalize_judge_result(
    result: dict[str, Any],
    shot: dict[str, Any],
    timeline: dict[str, Any],
    clip_meta: dict[str, Any],
    pass_threshold: float,
    llm_error: str,
    batch_judge: bool,
    render_observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    issues = _normalize_issues(result.get("issues", []))
    blocking_issues = [
        issue for issue in issues
        if str(issue.get("severity", "")).lower() in {"critical", "warning"}
    ]
    passed = bool(result.get("passed", False)) and not blocking_issues
    score = _as_float(result.get("overall_score", 0.0), 0.0)
    if bool(result.get("fallback_used", False)):
        passed = False
        score = min(score, max(0.0, pass_threshold - 0.05))
    elif score >= pass_threshold and not blocking_issues:
        passed = True
    frames = _observation_frames(render_observation or {})
    return {
        "shot_id": shot.get("shot_id", ""),
        "passed": passed,
        "overall_score": score,
        "dimensions": _normalize_dimensions(result.get("dimensions", {})),
        "issues": issues,
        "verdict": result.get("verdict", "pass" if passed else "rerender"),
        "summary": str(result.get("summary", "")).strip() or "VLM质检完成",
        "status": "approved" if passed else ("needs_review" if result.get("fallback_used") else "needs_fix"),
        "fallback_used": bool(result.get("fallback_used", False)),
        "llm_error": llm_error,
        "batch_judge": batch_judge,
        "judge_type": "render_vlm" if render_observation else "render_metadata",
        "observation_source": "mp4_frames" if frames else ("missing_mp4_frames" if render_observation else "metadata_only"),
        "observation_frame_count": len(frames),
        "rendered_path": (render_observation or {}).get("rendered_path", ""),
    }


def extract_render_observation(
    rendered_path: str,
    output_dir: str,
    *,
    num_frames: int = 6,
    timeout_sec: int = 90,
) -> dict[str, Any]:
    """Extract evenly spaced JPEG frames from a rendered MP4 for VLM observation."""
    src = Path(rendered_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not src.exists() or src.stat().st_size <= 0:
        return {
            "status": "missing_rendered_mp4",
            "rendered_path": rendered_path,
            "frames": [],
            "frame_count": 0,
        }

    duration = _probe_duration(src, timeout_sec=max(10, min(timeout_sec, 30)))
    if duration <= 0:
        duration = 1.0
    count = max(1, int(num_frames))
    if count == 1:
        timestamps = [duration * 0.5]
    else:
        start = max(0.0, duration * 0.10)
        end = max(start, duration * 0.90)
        timestamps = [start + (end - start) * i / (count - 1) for i in range(count)]

    frames: list[dict[str, Any]] = []
    errors: list[str] = []
    for idx, ts in enumerate(timestamps):
        frame_path = out_dir / f"frame_{idx:02d}_{int(ts * 1000):06d}ms.jpg"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{ts:.3f}",
            "-i",
            str(src),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(frame_path),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        if proc.returncode == 0 and frame_path.exists() and frame_path.stat().st_size > 0:
            frames.append({
                "index": idx,
                "timestamp_sec": round(ts, 3),
                "path": str(frame_path),
            })
        else:
            errors.append(proc.stderr[-300:])

    return {
        "status": "success" if frames else "frame_extract_failed",
        "rendered_path": str(src),
        "duration_sec": round(duration, 3),
        "frames": frames,
        "frame_count": len(frames),
        "errors": errors[:3],
    }


def _probe_duration(path: Path, timeout_sec: int) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        if proc.returncode == 0:
            return _as_float(proc.stdout.strip(), 0.0)
    except Exception:
        return 0.0
    return 0.0


def _uses_render_observation(item: dict[str, Any]) -> bool:
    if item.get("require_render_observation"):
        return True
    obs = item.get("render_observation")
    return isinstance(obs, dict) and bool(obs)


def _observation_frames(render_observation: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(render_observation, dict):
        return []
    frames = render_observation.get("frames", [])
    if not isinstance(frames, list):
        return []
    return [
        frame for frame in frames
        if isinstance(frame, dict) and frame.get("path") and Path(str(frame.get("path"))).exists()
    ]


def _chat_json_with_render_frames(
    llm: LLMClient,
    prompt: str,
    shot_lookup: dict[str, dict[str, Any]],
    temperature: float,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    frame_budget = max(1, int(getattr(llm, "max_frames_per_call", 8) or 8))
    used = 0
    for sid, source in shot_lookup.items():
        frames = _observation_frames(source.get("render_observation", {}))
        for frame in frames:
            if used >= frame_budget:
                break
            ts = frame.get("timestamp_sec", "")
            parts.append({"type": "text", "text": f"shot_id={sid} frame_index={frame.get('index')} timestamp_sec={ts}"})
            parts.append(_image_url_part(str(frame.get("path"))))
            used += 1
    if hasattr(llm, "chat_json_multimodal"):
        return llm.chat_json_multimodal(SYSTEM_PROMPT, parts, temperature=temperature)
    return llm.chat_json(SYSTEM_PROMPT, prompt, temperature=temperature)


def _image_url_part(path: str) -> dict[str, Any]:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{data}"},
    }


def _observation_failure_judge(
    shot: dict[str, Any],
    timeline: dict[str, Any],
    clip_meta: dict[str, Any],
    render_observation: dict[str, Any],
) -> dict[str, Any]:
    rendered_path = str(render_observation.get("rendered_path", "") or "")
    status = str(render_observation.get("status", "") or "missing")
    return {
        "passed": False,
        "overall_score": 0.0,
        "dimensions": {
            "timing_alignment": 0.0,
            "subtitle_accuracy": 0.0,
            "visual_match": 0.0,
            "effect_quality": 0.0,
            "overall_impression": 0.0,
        },
        "issues": [{
            "dimension": "render_observation",
            "severity": "critical",
            "description": f"VLM-Judge 要求使用渲染 MP4 抽帧，但未获得有效帧: status={status}, rendered_path={rendered_path}",
            "suggested_fix": "先修复 HyperFrames render/lint 或 MP4 抽帧失败，再重新执行 VLMRenderJudge",
        }],
        "verdict": "rerender",
        "summary": "缺少真实 MP4 抽帧，禁止退化为纯文本 VLM 判断。",
        "fallback_used": False,
        "shot_id": shot.get("shot_id", ""),
    }


def generate_fix_options(
    llm: LLMClient,
    judge_result: dict[str, Any],
    shot: dict[str, Any],
    timeline: dict[str, Any],
    num_options: int = 4,
) -> list[dict[str, Any]]:
    """Generate fix options for HITL decision when judge fails."""
    prompt = OPTIONS_TEMPLATE.format(
        num_options=num_options,
        judge_json=json.dumps(judge_result, ensure_ascii=False, indent=2),
        shot_json=json.dumps(shot, ensure_ascii=False, indent=2),
        timeline_json=json.dumps(timeline, ensure_ascii=False, indent=2),
    )
    try:
        result = llm.chat_json(SYSTEM_PROMPT, prompt, temperature=0.7)
    except Exception:
        return _fallback_options(judge_result, num_options)
    if isinstance(result, list):
        return [opt for opt in result if isinstance(opt, dict)][:num_options]
    result = _as_dict(result)
    options = result.get("options", result.get("fixes", []))
    usable = [opt for opt in options if isinstance(opt, dict)] if isinstance(options, list) else []
    if len(usable) < num_options:
        usable = (usable + _fallback_options(judge_result, num_options))[:num_options]
    return usable[:num_options]


def patch(
    llm: LLMClient,
    shot: dict[str, Any],
    timeline: dict[str, Any],
    judge_result: dict[str, Any],
    fix_changes: dict[str, Any],
) -> dict[str, Any]:
    """Generate structured patch instructions for judge-repair loop."""
    prompt = PATCH_TEMPLATE.format(
        shot_json=json.dumps(shot, ensure_ascii=False, indent=2),
        timeline_json=json.dumps(timeline, ensure_ascii=False, indent=2),
        judge_json=json.dumps(judge_result, ensure_ascii=False, indent=2),
        fix_changes_json=json.dumps(fix_changes, ensure_ascii=False, indent=2),
    )
    try:
        result = _as_dict(llm.chat_json(SYSTEM_PROMPT, prompt, temperature=0.2))
    except Exception:
        result = {}
    if not result:
        return _deterministic_patch(shot, timeline, judge_result, fix_changes)

    shot_patch = result.get("shot_patch", {})
    asset_patch = result.get("asset_patch", {})
    timeline_patch = result.get("timeline_patch", {})
    if not isinstance(shot_patch, dict):
        shot_patch = {}
    if not isinstance(asset_patch, dict):
        asset_patch = {}
    if not isinstance(timeline_patch, dict):
        timeline_patch = {}
    patch_payload = {
        "shot_patch": shot_patch,
        "asset_patch": asset_patch,
        "timeline_patch": timeline_patch,
        "repair_summary": result.get("repair_summary", ""),
    }
    if not any(patch_payload[k] for k in ("shot_patch", "asset_patch", "timeline_patch")):
        patch_payload = _deterministic_patch(shot, timeline, judge_result, fix_changes)
    return patch_payload


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, tuple):
        for item in value:
            if isinstance(item, dict):
                return item
            if isinstance(item, list):
                return {"options": item}
        return {}
    if isinstance(value, list):
        return {"options": value}
    return {}


def _extract_batch_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    mapped = _as_dict(value)
    for key in ("shots", "results", "items", "judgements", "judgments"):
        rows = mapped.get(key)
        if isinstance(rows, list):
            return [item for item in rows if isinstance(item, dict)]
    if mapped.get("shot_id"):
        return [mapped]
    return []


def _compact_timeline(timeline: dict[str, Any]) -> dict[str, Any]:
    tracks = timeline.get("tracks", {}) if isinstance(timeline.get("tracks"), dict) else {}
    visual = tracks.get("visual", {}) if isinstance(tracks.get("visual"), dict) else {}
    audio = tracks.get("audio", {}) if isinstance(tracks.get("audio"), dict) else {}
    subtitles = tracks.get("subtitles", [])
    decoration = tracks.get("decoration", {}) if isinstance(tracks.get("decoration"), dict) else {}
    subtitle_preview = subtitles[:4] if isinstance(subtitles, list) else []
    return {
        "duration_sec": timeline.get("duration_sec", 0),
        "visual": {
            "clip_path": visual.get("clip_path", ""),
            "start": visual.get("start", 0),
            "end": visual.get("end", 0),
        },
        "audio": {
            "audio_path": audio.get("audio_path", ""),
            "start": audio.get("start", 0),
            "end": audio.get("end", 0),
        },
        "subtitle_count": len(subtitles) if isinstance(subtitles, list) else 0,
        "subtitle_preview": subtitle_preview,
        "decoration": {
            "template_id": decoration.get("template_id", ""),
            "template_type": decoration.get("template_type", ""),
            "slots": decoration.get("slots", {}),
            "overlay_config": decoration.get("overlay_config", {}),
        },
    }


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_empty_judge_result(result: dict[str, Any]) -> bool:
    if not isinstance(result, dict) or not result:
        return True
    has_issues = isinstance(result.get("issues"), list) and len(result.get("issues", [])) > 0
    has_summary = bool(str(result.get("summary", "")).strip())
    has_score = isinstance(result.get("overall_score"), (int, float))
    has_dims = isinstance(result.get("dimensions"), dict) and bool(result.get("dimensions"))
    return not (has_issues or has_summary or has_score or has_dims)


def _normalize_dimensions(value: Any) -> dict[str, float]:
    dims = value if isinstance(value, dict) else {}
    return {
        "timing_alignment": _as_float(dims.get("timing_alignment", 0.0), 0.0),
        "subtitle_accuracy": _as_float(dims.get("subtitle_accuracy", 0.0), 0.0),
        "visual_match": _as_float(dims.get("visual_match", 0.0), 0.0),
        "effect_quality": _as_float(dims.get("effect_quality", 0.0), 0.0),
        "overall_impression": _as_float(dims.get("overall_impression", 0.0), 0.0),
    }


def _normalize_issues(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        value = [value] if value else []
    out: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            issue = dict(item)
            issue.setdefault("dimension", _infer_dimension(str(issue.get("description", ""))))
            issue.setdefault("severity", "warning")
            issue.setdefault("description", str(issue.get("detail") or issue.get("message") or ""))
            issue.setdefault("suggested_fix", "")
            out.append(issue)
        elif item:
            text = str(item).strip()
            out.append({
                "dimension": _infer_dimension(text),
                "severity": _infer_severity(text),
                "description": text,
                "suggested_fix": "",
            })
    return out


def _infer_dimension(text: str) -> str:
    if any(k in text for k in ("素材", "画面", "视觉", "clip", "visual")):
        return "visual_match"
    if any(k in text for k in ("字幕", "旁白", "文案", "标题", "槽位", "subtitle", "slot", "title")):
        return "subtitle_accuracy"
    if any(k in text for k in ("时序", "时间", "同步", "duration", "timing")):
        return "timing_alignment"
    if any(k in text for k in ("动效", "装饰", "模板", "effect")):
        return "effect_quality"
    return "overall_impression"


def _infer_severity(text: str) -> str:
    if any(k in text for k in ("严重", "无法", "为空", "缺少", "不匹配", "错误")):
        return "critical"
    return "warning"


def _deterministic_patch(
    shot: dict[str, Any],
    timeline: dict[str, Any],
    judge_result: dict[str, Any],
    fix_changes: dict[str, Any],
) -> dict[str, Any]:
    strategy = str(fix_changes.get("strategy") or "").strip()
    issues = _normalize_issues(fix_changes.get("judge_issues") or judge_result.get("issues") or [])
    shot_patch: dict[str, Any] = {}
    asset_patch: dict[str, Any] = {}
    timeline_patch: dict[str, Any] = {}
    summaries: list[str] = []

    if strategy in {"reretrieve_visual", "regenerate_shot_execution"} or any(
        issue.get("dimension") == "visual_match" for issue in issues
    ):
        shot_patch["visual_query"] = _refine_visual_query(shot, issues)
        asset_patch["clip_path"] = ""
        asset_patch["clip_meta"] = {
            "repair_request": "reretrieve_visual",
            "previous_issue_count": len(issues),
        }
        summaries.append("更新 visual_query 并请求重新检索当前镜头素材")

    if strategy in {"repair_subtitle_and_decoration", "regenerate_shot_execution"} or any(
        issue.get("dimension") in {"subtitle_accuracy", "effect_quality"} for issue in issues
    ):
        decoration = dict(((timeline.get("tracks") or {}).get("decoration") or {}))
        slots = dict(decoration.get("slots") or {})
        slots["title"] = _core_text(shot)
        slots["subtitle"] = _subtitle_text(shot)
        if not slots.get("eyebrow"):
            slots["eyebrow"] = _short_label(shot)
        decoration["slots"] = slots
        overlay = dict(decoration.get("overlay_config") or {})
        overlay["opacity"] = min(float(overlay.get("opacity", 0.7) or 0.7), 0.65)
        decoration["overlay_config"] = overlay
        asset_patch["decoration"] = decoration
        summaries.append("重填当前镜头装饰槽位并降低遮罩透明度")

    if strategy in {"retime_timeline", "regenerate_shot_execution"} or any(
        issue.get("dimension") == "timing_alignment" for issue in issues
    ):
        duration = max(
            _as_float(timeline.get("duration_sec"), 0.0),
            _as_float(shot.get("duration_sec"), 0.0),
            max(1.0, len(str(shot.get("vocal_script", ""))) / 4.5),
        )
        shot_patch["duration_sec"] = round(duration, 2)
        timeline_patch["duration_sec"] = round(duration, 2)
        summaries.append("调整当前镜头 duration_sec 以便重新合成时对齐")

    if not summaries:
        summaries.append("执行保守补丁：重填字幕/装饰槽位并重新合成")
        decoration = dict(((timeline.get("tracks") or {}).get("decoration") or {}))
        decoration["slots"] = {
            **dict(decoration.get("slots") or {}),
            "title": _core_text(shot),
            "subtitle": _subtitle_text(shot),
        }
        asset_patch["decoration"] = decoration

    return {
        "shot_patch": shot_patch,
        "asset_patch": asset_patch,
        "timeline_patch": timeline_patch,
        "repair_summary": "；".join(summaries),
    }


def _refine_visual_query(shot: dict[str, Any], issues: list[dict[str, Any]]) -> str:
    base = str(shot.get("visual_query") or "").strip()
    script = _core_text(shot)
    issue_hint = " ".join(str(i.get("description", "")) for i in issues[:2])
    query = "，".join(x for x in [base, script, issue_hint[:40]] if x)
    return query[:120]


def _core_text(shot: dict[str, Any]) -> str:
    text = str(shot.get("vocal_script") or shot.get("chapter_title") or "").strip()
    for sep in ("，", "。", "！", "？", "；"):
        idx = text.find(sep)
        if 4 <= idx <= 24:
            return text[:idx]
    return text[:24]


def _subtitle_text(shot: dict[str, Any]) -> str:
    text = str(shot.get("vocal_script") or "").strip()
    parts = [p.strip() for p in re.split(r"[，。！？；]", text) if len(p.strip()) >= 3]
    if len(parts) >= 2:
        return parts[1][:32]
    if parts:
        return parts[0][:32]
    return str(shot.get("chapter_title") or "")[:32]


def _short_label(shot: dict[str, Any]) -> str:
    title = str(shot.get("chapter_title") or "").strip()
    return title[:12] if title else str(shot.get("structure_label") or "重点镜头")[:12]


def _fallback_judge(
    shot: dict[str, Any],
    timeline: dict[str, Any],
    clip_meta: dict[str, Any],
    llm_error: str = "",
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    issues.append({
        "dimension": "vlm_availability",
        "severity": "warning",
        "description": "VLM质检输出不可用，当前结果只能作为结构化本地检查，不能自动标为高置信通过",
        "suggested_fix": "保留为 needs_review，或重试 VLMRenderJudge",
    })
    if not timeline:
        issues.append({
            "dimension": "时序对齐",
            "severity": "critical",
            "description": "时间线为空，无法确认合成结果",
            "suggested_fix": "重新生成 timeline",
        })
    if not clip_meta.get("clip_id"):
        issues.append({
            "dimension": "素材匹配",
            "severity": "warning",
            "description": "缺少可确认的视觉素材 clip_id",
            "suggested_fix": "检查检索结果或使用本地素材兜底",
        })
    if not shot.get("vocal_script"):
        issues.append({
            "dimension": "字幕正确",
            "severity": "critical",
            "description": "镜头旁白为空",
            "suggested_fix": "补充 vocal_script",
        })

    critical = any(issue.get("severity") == "critical" for issue in issues)
    score = 0.44 if critical else 0.65
    return {
        "passed": False,
        "overall_score": score,
        "dimensions": {
            "timing_alignment": 0.7 if timeline else 0.0,
            "subtitle_accuracy": 0.8 if shot.get("vocal_script") else 0.0,
            "visual_match": 0.6 if clip_meta.get("clip_id") else 0.4,
            "effect_quality": 0.7,
            "overall_impression": score,
        },
        "issues": issues,
        "verdict": "pass" if not critical else "adjust",
        "summary": "VLM 质检输出不可用，已使用本地结构化检查结果。",
        "llm_error": llm_error,
    }


def _fallback_options(judge_result: dict[str, Any], num_options: int) -> list[dict[str, Any]]:
    issues = judge_result.get("issues", [])
    options = [
        {
            "option_id": 0,
            "label": "重新检索视觉素材",
            "description": "针对素材匹配度不足，重新生成visual_query并检索更贴合旁白和镜头意图的素材。",
            "changes": {"strategy": "reretrieve_visual", "judge_issues": issues},
        },
        {
            "option_id": 1,
            "label": "调整时间线对齐",
            "description": "修复音频、字幕、视频时长不一致的问题，重新生成timeline并对齐关键帧。",
            "changes": {"strategy": "retime_timeline", "judge_issues": issues},
        },
        {
            "option_id": 2,
            "label": "重做字幕和装饰",
            "description": "保持素材不变，重新拆分字幕、调整模板装饰和动效节奏。",
            "changes": {"strategy": "repair_subtitle_and_decoration", "judge_issues": issues},
        },
        {
            "option_id": 3,
            "label": "重新生成本镜头",
            "description": "对该镜头从检索、TTS、装饰到时间线合成进行完整重做。",
            "changes": {"strategy": "regenerate_shot_execution", "judge_issues": issues},
        },
    ]
    return options[:num_options]
