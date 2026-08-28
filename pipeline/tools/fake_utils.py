"""Fake utilities for annotation (fake) mode.

These functions produce structurally-identical JSON outputs to the real
HyperFrames render / VLM judge / BGM search, but skip the expensive
pixel-level computation.  They are only used when ``execution.mode == fake``.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any


def fake_render_success(
    shot: dict[str, Any],
    assets: dict[str, Any],
    output_dir: str,
) -> dict[str, Any]:
    """Return a fake-but-structurally-valid timeline compose result.

    Mirrors the schema returned by ``timeline_compose.run()`` so that
    downstream trajectory entries are consistent.
    """
    sid = str(shot.get("shot_id", "") or "unknown")
    audio_meta = assets.get("audio_meta", {}) if isinstance(assets.get("audio_meta"), dict) else {}
    clip_meta = assets.get("clip_meta", {}) if isinstance(assets.get("clip_meta"), dict) else {}
    decoration = assets.get("decoration", {}) if isinstance(assets.get("decoration"), dict) else {}

    audio_dur = float(audio_meta.get("duration_sec", 0) or 0)
    clip_dur = float(clip_meta.get("duration_sec", 0) or 0)
    shot_dur = float(shot.get("duration_sec", 0) or 0)
    # 与真实 timeline_compose 保持一致：TTS 是时长真值，长素材会被裁剪，
    # 不能把完整素材时长直接当成成片镜头时长。
    duration = (audio_dur + 0.15) if audio_dur > 0 else max(shot_dur, min(clip_dur, 8.0), 1.0)

    fake_rendered = str(Path(output_dir) / "shots" / sid / f"fake_render_{uuid.uuid4().hex[:8]}.mp4")

    timeline = {
        "shot_id": sid,
        "duration_sec": round(duration, 2),
        "tracks": {
            "visual": {
                "clip_path": assets.get("clip_path", ""),
                "clip_id": clip_meta.get("clip_id", ""),
                "start": 0.0,
                "end": round(duration, 2),
            },
            "audio": {
                "audio_path": assets.get("audio_path", ""),
                "start": 0.0,
                "end": round(audio_dur or duration, 2),
            },
            "subtitles": [],
            "decoration": decoration,
        },
    }

    return {
        "shot_id": sid,
        "timeline": timeline,
        "rendered_path": fake_rendered,
        "render_success": True,
        "render_status": "fake_annotation",
        "render_backend": "fake",
        "compose_status": "success",
        "duration_sec": round(duration, 2),
        "subtitle_count": 0,
        "fake_mode": True,
    }


def fake_music_search(
    music_style: str,
    total_duration: float,
) -> dict[str, Any]:
    """Return a fake-but-structurally-valid BGM search result."""
    return {
        "music_style": music_style,
        "candidates_count": 1,
        "candidates": [],
        "selected_tracks": [{
            "track_id": "fake_bgm",
            "music_path": "",
            "score": 1.0,
            "duration_sec": total_duration + 10,
            "name": "fake_annotation_bgm",
            "caption": f"标注模式 BGM: {music_style[:60]}",
            "cdn_path": "",
            "rank": 1,
            "offset_sec": 0.0,
            "end_sec": round(total_duration, 2),
        }],
        "selected_count": 1,
        "total_bgm_duration_sec": round(total_duration + 10, 2),
        "target_duration_sec": round(total_duration, 2),
        "duration_covered": True,
        "minimum_track_count": 1,
        "playlist_policy_version": 2,
        "backend": "fake_annotation",
        "retrieval_diagnostics": [{"backend": "fake", "status": "success"}],
        "status": "success",
    }


def fake_vlm_analysis(
    shot: dict[str, Any],
    assets: dict[str, Any],
    compose_result: dict[str, Any],
) -> dict[str, Any]:
    """Generate a fake VLM judge result based on real metadata.

    The analysis text *simulates* the "from rendered screenshot" tone used by
    the real VLM judge, but the scoring is derived from verifiable metadata
    (clip duration, audio duration, decoration slots, etc.) so the scores are
    meaningful even though no frames were extracted.
    """
    sid = str(shot.get("shot_id", "") or "unknown")
    clip_meta = assets.get("clip_meta", {}) if isinstance(assets.get("clip_meta"), dict) else {}
    audio_meta = assets.get("audio_meta", {}) if isinstance(assets.get("audio_meta"), dict) else {}
    decoration = assets.get("decoration", {}) if isinstance(assets.get("decoration"), dict) else {}

    clip_dur = float(clip_meta.get("duration_sec", 0) or 0)
    audio_dur = float(audio_meta.get("duration_sec", 0) or 0)
    try:
        match_score = float(clip_meta.get("match_score", 0.0))
    except (TypeError, ValueError):
        match_score = 0.0

    issues: list[dict[str, Any]] = []

    # --- timing_alignment ---
    if clip_dur > 0 and audio_dur > 0:
        shortfall = max(0.0, audio_dur - clip_dur)
        if shortfall > 3.0:
            timing_score = max(0.3, 1.0 - shortfall * 0.12)
            issues.append({
                "dimension": "timing_alignment",
                "severity": "warning",
                "description": f"从渲染截图观察，素材时长 {clip_dur:.1f}s 短于旁白 {audio_dur:.1f}s，缺口 {shortfall:.1f}s",
                "suggested_fix": "重新检索能够完整覆盖旁白时长的素材",
            })
        elif shortfall > 1.5:
            timing_score = max(0.6, 1.0 - shortfall * 0.1)
        else:
            timing_score = 1.0
    else:
        timing_score = 0.8

    # --- subtitle_accuracy / slot_quality ---
    slots = decoration.get("slots", {}) if isinstance(decoration.get("slots"), dict) else {}
    slot_score = 1.0
    _JARGON = ("开篇定调", "视觉冲击", "叙事功能", "需强视觉", "段落 ", "design_rationale", "emphasis_target")
    for slot_name, val in slots.items():
        text = str(val or "")
        if not text.strip():
            slot_score = min(slot_score, 0.5)
            issues.append({
                "dimension": "subtitle_accuracy",
                "severity": "warning",
                "description": f"从渲染画面发现，装饰槽位 {slot_name} 为空",
                "suggested_fix": f"从 vocal_script 提炼内容填充槽位 {slot_name}",
            })
        elif sum(1 for j in _JARGON if j in text) >= 2:
            slot_score = min(slot_score, 0.4)
            issues.append({
                "dimension": "subtitle_accuracy",
                "severity": "critical",
                "description": f"观察渲染画面发现，字幕区域存在管线术语 '{text[:30]}'，建议替换为观众友好文案",
                "suggested_fix": "用 vocal_script/chapter_title 提炼观众可见文案",
            })

    # --- visual_match ---
    if match_score > 0.7:
        visual_score = min(1.0, match_score)
    elif match_score > 0.5:
        visual_score = match_score
        issues.append({
            "dimension": "visual_match",
            "severity": "info",
            "description": f"从渲染截图观察，素材匹配度 {match_score:.2f} 处于中等水平",
            "suggested_fix": "可考虑更换检索关键词获取更精准素材",
        })
    else:
        visual_score = max(0.3, match_score)
        issues.append({
            "dimension": "visual_match",
            "severity": "warning",
            "description": f"从渲染截图观察，素材与镜头需求匹配度偏低 ({match_score:.2f})",
            "suggested_fix": "重新检索视觉素材，调整 visual_query 关键词",
        })

    # --- effect_quality ---
    template_id = str(decoration.get("template_id", "") or "")
    effect_score = 0.7
    if template_id and "fallback" not in template_id.lower():
        effect_score += 0.2
    if slots:
        effect_score += 0.1
    effect_score = min(1.0, effect_score)

    # --- overall ---
    overall = round((timing_score + slot_score + visual_score + effect_score) / 4, 3)
    passed = overall >= 0.75 and not any(
        i.get("severity") == "critical" for i in issues
    )

    # Build analysis text in the "VLM screenshot observation" style
    analysis_parts = [
        f"从渲染截图观察，镜头 {sid} 的字幕与旁白对齐{'良好' if timing_score > 0.7 else '存在偏差'}",
        f"装饰模板 {template_id or '未设置'} 的{'标题动效自然' if effect_score > 0.7 else '视觉效果一般'}",
    ]
    if not issues:
        analysis_parts.append("整体渲染画面质量达标")

    return {
        "shot_id": sid,
        "passed": passed,
        "overall_score": overall,
        "dimensions": {
            "timing_alignment": round(timing_score, 3),
            "subtitle_accuracy": round(slot_score, 3),
            "visual_match": round(visual_score, 3),
            "effect_quality": round(effect_score, 3),
            "overall_impression": overall,
        },
        "issues": issues,
        "verdict": "pass" if passed else "adjust",
        "summary": "；".join(analysis_parts),
        "status": "approved" if passed else "needs_fix",
        "fallback_used": False,
        "batch_judge": False,
        "fake_mode": True,
        "source": "fake_vlm_metadata_analysis",
    }


def fake_stitch_result(
    shot_plan: list[dict[str, Any]],
    shot_assets: dict[str, Any],
    output_dir: str,
) -> dict[str, Any]:
    """Return a fake stitch result (no actual MP4 concatenation)."""
    total_duration = 0.0
    shot_list = []
    for shot in shot_plan:
        sid = str(shot.get("shot_id", ""))
        assets = shot_assets.get(sid, {})
        audio_meta = assets.get("audio_meta", {}) if isinstance(assets.get("audio_meta"), dict) else {}
        dur = float(audio_meta.get("duration_sec", 0) or 0) or 8.0
        total_duration += dur
        shot_list.append({
            "shot_id": sid,
            "duration_sec": round(dur, 2),
            "rendered_path": assets.get("rendered_path", ""),
            "approved": assets.get("approved", False),
        })

    return {
        "final_video_path": "",
        "total_duration_sec": round(total_duration, 2),
        "shot_count": len(shot_plan),
        "approved_count": sum(1 for s in shot_list if s["approved"]),
        "concat_status": "skipped_fake_mode",
        "concat_error": "",
        "shots": shot_list,
        "fake_mode": True,
    }
