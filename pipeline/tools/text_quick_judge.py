"""Tool_TextQuickJudge: Text-only quality judge for non-spotlight shots.

This tool has an independent text-only input contract. It never receives
rendered MP4 frames, screenshot observations, or decoration-render fields.
"""

from __future__ import annotations

import json
from typing import Any

from ..llm import LLMClient

SYSTEM_PROMPT = """\
你是视频渲染质量审查专家。你需要基于纯文本信息对单镜头的合理性进行快速判断。

当前模式: 独立文本快速审查（无渲染画面、无MP4抽帧、无装饰渲染结果可参考）。
你的判断完全基于以下文本信息:
- shot_caption: VLM 为该镜头已生成的画面内容描述
- visual_query: 用于检索素材视频的搜索词
- vocal_script: 旁白文案
- audio_duration_sec: TTS 音频时长
- video_duration_sec: 检索到的素材视频时长

硬性输出规则:
1. 只能输出一个合法 JSON object，不能输出 JSON array、Markdown、解释文字、代码块或 <think>。
2. issues、options 等集合字段必须是 JSON array，array 中每一项必须是 JSON object。
3. 一旦存在 critical 或 warning 级别问题，passed 必须为 false。
4. 不要输出推理过程，只输出最终 JSON。"""

TEXT_JUDGE_TEMPLATE = """\
请对以下镜头进行文本层面的合理性快速判断（无渲染画面可参考，纯文本分析）:

镜头ID: {shot_id}
章节标题: {chapter_title}

旁白文案 (vocal_script):
{vocal_script}

视觉检索词 (visual_query):
{visual_query}

画面内容描述 (shot_caption):
{shot_caption}

TTS 音频时长: {audio_duration_sec:.1f} 秒
素材视频时长: {video_duration_sec:.1f} 秒

检查维度(每项0-1分):
1. visual_match: 画面描述(shot_caption)是否匹配旁白文案(vocal_script)的语义，素材是否贴合内容
2. timing_alignment: 素材视频是否足够长以覆盖音频；长素材可以裁剪，不应因“过长”扣分
3. query_accuracy: visual_query 是否准确描述了画面需求，能否有效检索到合适素材
4. content_coherence: 旁白文案是否流畅自然，与章节标题的逻辑关系是否通顺

输出(严格JSON):
{{
  "passed": true/false,
  "overall_score": 0.0-1.0,
  "dimensions": {{
    "visual_match": 0.0-1.0,
    "timing_alignment": 0.0-1.0,
    "query_accuracy": 0.0-1.0,
    "content_coherence": 0.0-1.0
  }},
  "issues": [
    {{
      "dimension": "visual_match/timing_alignment/query_accuracy/content_coherence",
      "severity": "critical/warning/info",
      "description": "问题描述",
      "suggested_fix": "修复建议"
    }}
  ],
  "verdict": "pass/adjust",
  "summary": "综合评价"
}}"""

BATCH_TEXT_JUDGE_TEMPLATE = """\
请对以下一组镜头进行批量文本层面的合理性快速判断（无渲染画面可参考，纯文本分析）。

批量镜头:
{items_json}

检查维度(每项0-1分):
1. visual_match: 画面描述(shot_caption)是否匹配旁白文案(vocal_script)的语义
2. timing_alignment: 素材视频是否足够长以覆盖音频；素材过长可以裁剪，不应扣分
3. query_accuracy: visual_query 是否准确描述了画面需求
4. content_coherence: 旁白文案是否流畅自然，与章节标题的逻辑关系是否通顺

输出(严格JSON object):
{{
  "shots": [
    {{
      "shot_id": "镜头ID",
      "passed": true/false,
      "overall_score": 0.0-1.0,
      "dimensions": {{
        "visual_match": 0.0-1.0,
        "timing_alignment": 0.0-1.0,
        "query_accuracy": 0.0-1.0,
        "content_coherence": 0.0-1.0
      }},
      "issues": [
        {{
          "dimension": "visual_match/timing_alignment/query_accuracy/content_coherence",
          "severity": "critical/warning/info",
          "description": "问题描述",
          "suggested_fix": "修复建议"
        }}
      ],
      "verdict": "pass/adjust",
      "summary": "综合评价"
    }}
  ]
}}"""


DECORATION_SLOT_BATCH_TEXT_JUDGE_TEMPLATE = """\
请对以下一组镜头进行批量文本层面的装饰槽位质量审查（无渲染画面可参考，纯文本分析）。

当前目标不是评估视频素材是否真实匹配，而是重点检查 Stage3b 模板填槽是否准确、稳定、可渲染。

批量镜头:
{items_json}

重点检查维度(每项0-1分):
1. decoration_quality: 模板槽位是否符合槽位功能；文字/emoji 是否简短、完整、有效；是否存在 placeholder、内部变量、过长文本、语义错填、空槽位
2. subtitle_accuracy: 字幕/标题类槽位是否忠实表达 vocal_script，不编造事实，不截断关键信息
3. content_coherence: 槽位内容、章节标题、旁白是否语义一致，情绪和视频风格是否匹配
4. timing_alignment: 仅检查显著的音频时长/视频时长冲突；素材过长可以裁剪，不应扣分

忽略项:
- 不要因为 visual_match、query_accuracy、clip_path 为空或素材真实性扣分
- 如果没有 decoration 或 slots，请将 decoration_quality 判为 warning/critical

输出(严格JSON object):
{{
  "shots": [
    {{
      "shot_id": "镜头ID",
      "passed": true/false,
      "overall_score": 0.0-1.0,
      "dimensions": {{
        "decoration_quality": 0.0-1.0,
        "subtitle_accuracy": 0.0-1.0,
        "content_coherence": 0.0-1.0,
        "timing_alignment": 0.0-1.0,
        "visual_match": 1.0
      }},
      "issues": [
        {{
          "dimension": "decoration_quality/subtitle_accuracy/content_coherence/timing_alignment",
          "severity": "critical/warning/info",
          "description": "问题描述",
          "suggested_fix": "修复建议"
        }}
      ],
      "verdict": "pass/adjust",
      "summary": "综合评价"
    }}
  ]
}}"""


def run(
    llm: LLMClient,
    shot: dict[str, Any],
    assets: dict[str, Any],
    pass_threshold: float = 0.70,
) -> dict[str, Any]:
    """Text-only judge for a single non-spotlight shot."""
    sid = str(shot.get("shot_id", ""))
    clip_meta = assets.get("clip_meta", {}) if isinstance(assets.get("clip_meta"), dict) else {}
    audio_meta = assets.get("audio_meta", {}) if isinstance(assets.get("audio_meta"), dict) else {}

    prompt = TEXT_JUDGE_TEMPLATE.format(
        shot_id=sid,
        chapter_title=shot.get("chapter_title", ""),
        vocal_script=shot.get("vocal_script", ""),
        visual_query=shot.get("visual_query", ""),
        shot_caption=clip_meta.get("caption", clip_meta.get("caption_zh", "")),
        audio_duration_sec=_as_float(audio_meta.get("duration_sec"), 0.0),
        video_duration_sec=_as_float(clip_meta.get("duration", clip_meta.get("duration_sec")), 0.0),
    )

    llm_error = ""
    try:
        result = _as_dict(llm.chat_json(SYSTEM_PROMPT, prompt, temperature=0.2))
    except Exception as exc:
        result = {}
        llm_error = str(exc)

    if _is_empty_result(result):
        result = _fallback_judge(shot, assets, llm_error=llm_error)
        result["fallback_used"] = True
    else:
        result["fallback_used"] = False

    return _finalize_result(result, shot, assets, pass_threshold, llm_error)


def batch_run(
    llm: LLMClient,
    items: list[dict[str, Any]],
    pass_threshold: float = 0.70,
    focus_decoration_slots: bool = False,
) -> dict[str, dict[str, Any]]:
    """Batch text-only judge for multiple non-spotlight shots."""
    if not items:
        return {}

    compact_items: list[dict[str, Any]] = []
    shot_lookup: dict[str, dict[str, Any]] = {}

    for idx, item in enumerate(items):
        shot = item.get("shot") if isinstance(item.get("shot"), dict) else {}
        sid = str(shot.get("shot_id") or item.get("sid") or f"shot_{idx + 1}")
        assets = item.get("assets") if isinstance(item.get("assets"), dict) else {}
        clip_meta = assets.get("clip_meta", {}) if isinstance(assets.get("clip_meta"), dict) else {}
        audio_meta = assets.get("audio_meta", {}) if isinstance(assets.get("audio_meta"), dict) else {}

        shot_lookup[sid] = {"shot": shot, "assets": assets}
        compact_items.append({
            "shot_id": sid,
            "chapter_title": shot.get("chapter_title", ""),
            "vocal_script": shot.get("vocal_script", ""),
            "visual_query": shot.get("visual_query", ""),
            "shot_caption": clip_meta.get("caption", clip_meta.get("caption_zh", "")),
            "decoration": _compact_decoration(assets.get("decoration")),
            "decoration_verify": _compact_verify(assets.get("decoration_verify")),
            "audio_duration_sec": _as_float(audio_meta.get("duration_sec"), 0.0),
            "video_duration_sec": _as_float(clip_meta.get("duration", clip_meta.get("duration_sec")), 0.0),
        })

    llm_error = ""
    template = DECORATION_SLOT_BATCH_TEXT_JUDGE_TEMPLATE if focus_decoration_slots else BATCH_TEXT_JUDGE_TEMPLATE
    try:
        raw = llm.chat_json(
            SYSTEM_PROMPT,
            template.format(
                items_json=json.dumps(compact_items, ensure_ascii=False, indent=2)
            ),
            temperature=0.2,
        )
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
        if row is None or _is_empty_result(row):
            row_error = llm_error or "batch_result_missing"
            row = _fallback_judge(source["shot"], source["assets"], llm_error=row_error)
            row["fallback_used"] = True
        else:
            row["fallback_used"] = False
        results[sid] = _finalize_result(row, source["shot"], source["assets"], pass_threshold, row_error)

    return results


def _finalize_result(
    result: dict[str, Any],
    shot: dict[str, Any],
    assets: dict[str, Any],
    pass_threshold: float,
    llm_error: str,
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

    return {
        "shot_id": shot.get("shot_id", ""),
        "passed": passed,
        "overall_score": score,
        "dimensions": _normalize_dimensions(result.get("dimensions", {})),
        "issues": issues,
        "verdict": result.get("verdict", "pass" if passed else "adjust"),
        "summary": str(result.get("summary", "")).strip() or "文本快速质检完成",
        "status": "approved" if passed else "needs_fix",
        "fallback_used": bool(result.get("fallback_used", False)),
        "llm_error": llm_error,
        "judge_type": "text_quick",
        "observation_source": "text_metadata_only",
    }


def _fallback_judge(
    shot: dict[str, Any],
    assets: dict[str, Any],
    llm_error: str = "",
) -> dict[str, Any]:
    """Deterministic fallback when LLM is unavailable."""
    clip_meta = assets.get("clip_meta", {}) if isinstance(assets.get("clip_meta"), dict) else {}
    audio_meta = assets.get("audio_meta", {}) if isinstance(assets.get("audio_meta"), dict) else {}
    issues: list[dict[str, Any]] = []

    audio_dur = _as_float(audio_meta.get("duration_sec"), 0.0)
    video_dur = _as_float(clip_meta.get("duration", clip_meta.get("duration_sec")), 0.0)

    shortfall = max(0.0, audio_dur - video_dur)
    if audio_dur > 0 and video_dur > 0 and shortfall > max(1.0, audio_dur * 0.15):
        issues.append({
            "dimension": "timing_alignment",
            "severity": "warning",
            "description": f"素材视频({video_dur:.1f}s)短于音频({audio_dur:.1f}s)，缺口 {shortfall:.1f}s",
            "suggested_fix": "重新检索能够完整覆盖旁白时长的素材",
        })

    if not shot.get("vocal_script"):
        issues.append({
            "dimension": "content_coherence",
            "severity": "critical",
            "description": "镜头缺少旁白文案",
            "suggested_fix": "补充 vocal_script",
        })

    if not shot.get("visual_query"):
        issues.append({
            "dimension": "query_accuracy",
            "severity": "warning",
            "description": "镜头缺少 visual_query",
            "suggested_fix": "补充可检索的视觉关键词",
        })

    decoration = assets.get("decoration", {}) if isinstance(assets.get("decoration"), dict) else {}
    slots = decoration.get("slots", {}) if isinstance(decoration.get("slots"), dict) else {}
    if shot.get("is_spotlight") and not slots:
        issues.append({
            "dimension": "decoration_quality",
            "severity": "warning",
            "description": "spotlight 镜头缺少模板槽位填充结果",
            "suggested_fix": "重新生成模板槽位，确保每个必要槽位都有简短有效内容",
        })
    for slot_id, slot_value in slots.items():
        text = str(slot_value or "").strip()
        if not text:
            issues.append({
                "dimension": "decoration_quality",
                "severity": "warning",
                "description": f"槽位 {slot_id} 为空",
                "suggested_fix": "填入与旁白和槽位功能匹配的短文本或emoji",
            })
        elif len(text) > 48:
            issues.append({
                "dimension": "decoration_quality",
                "severity": "warning",
                "description": f"槽位 {slot_id} 文本过长，可能导致排版混乱",
                "suggested_fix": "压缩为短句，保留核心信息",
            })
        elif any(token in text.lower() for token in ("placeholder", "todo", "slot_", "待填写", "段落")):
            issues.append({
                "dimension": "decoration_quality",
                "severity": "critical",
                "description": f"槽位 {slot_id} 含内部占位文本",
                "suggested_fix": "替换为面向观众可见的最终文案",
            })

    caption = clip_meta.get("caption", clip_meta.get("caption_zh", ""))
    if not caption:
        issues.append({
            "dimension": "visual_match",
            "severity": "warning",
            "description": "缺少 shot_caption，无法评估素材画面与文案匹配度",
            "suggested_fix": "确保前置 VLM caption 步骤为该镜头生成了描述",
        })

    critical = any(issue.get("severity") == "critical" for issue in issues)
    score = 0.40 if critical else 0.65
    return {
        "passed": False,
        "overall_score": score,
        "dimensions": {
            "visual_match": 0.6 if caption else 0.3,
            "timing_alignment": 0.7 if (audio_dur > 0 and video_dur > 0) else 0.4,
            "query_accuracy": 0.7 if shot.get("visual_query") else 0.3,
            "content_coherence": 0.7 if shot.get("vocal_script") else 0.0,
            "decoration_quality": 0.85 if slots else (0.45 if shot.get("is_spotlight") else 0.7),
            "subtitle_accuracy": 0.7 if shot.get("vocal_script") else 0.0,
        },
        "issues": issues,
        "verdict": "adjust",
        "summary": "LLM 不可用，已使用本地文本规则检查。",
        "llm_error": llm_error,
    }


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, tuple):
        for item in value:
            if isinstance(item, dict):
                return item
        return {}
    if isinstance(value, list):
        return {"shots": value}
    return {}


def _extract_batch_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    mapped = _as_dict(value)
    for key in ("shots", "results", "items"):
        rows = mapped.get(key)
        if isinstance(rows, list):
            return [item for item in rows if isinstance(item, dict)]
    if mapped.get("shot_id"):
        return [mapped]
    return []


def _compact_decoration(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    slots = value.get("slots", {}) if isinstance(value.get("slots"), dict) else {}
    return {
        "template_id": value.get("template_id", ""),
        "template_type": value.get("template_type", ""),
        "slots": {
            str(key): str(slot_value)[:120]
            for key, slot_value in slots.items()
        },
        "overlay_config": value.get("overlay_config", {}),
        "timing": value.get("timing", {}),
    }


def _compact_verify(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    issues = value.get("issues", [])
    if not isinstance(issues, list):
        issues = []
    return {
        "passed": value.get("passed", None),
        "score": value.get("score", None),
        "issues": issues[:5],
    }


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_empty_result(result: dict[str, Any]) -> bool:
    if not isinstance(result, dict) or not result:
        return True
    has_issues = isinstance(result.get("issues"), list) and len(result.get("issues", [])) > 0
    has_summary = bool(str(result.get("summary", "")).strip())
    has_score = isinstance(result.get("overall_score"), (int, float))
    return not (has_issues or has_summary or has_score)


def _normalize_dimensions(value: Any) -> dict[str, float]:
    dims = value if isinstance(value, dict) else {}
    return {
        "visual_match": _as_float(dims.get("visual_match", 0.0), 0.0),
        "timing_alignment": _as_float(dims.get("timing_alignment", 0.0), 0.0),
        "query_accuracy": _as_float(dims.get("query_accuracy", 0.0), 0.0),
        "content_coherence": _as_float(dims.get("content_coherence", 0.0), 0.0),
        "decoration_quality": _as_float(dims.get("decoration_quality", 0.0), 0.0),
        "subtitle_accuracy": _as_float(dims.get("subtitle_accuracy", 0.0), 0.0),
        "effect_quality": _as_float(dims.get("effect_quality", 0.0), 0.0),
        "slot_fill_accuracy": _as_float(dims.get("slot_fill_accuracy", dims.get("slot_accuracy", 0.0)), 0.0),
        "layout_quality": _as_float(dims.get("layout_quality", dims.get("text_layout", 0.0)), 0.0),
    }


def _normalize_issues(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        value = [value] if value else []
    out: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            issue = dict(item)
            issue.setdefault("dimension", "visual_match")
            issue.setdefault("severity", "warning")
            issue.setdefault("description", "")
            issue.setdefault("suggested_fix", "")
            out.append(issue)
        elif item:
            out.append({
                "dimension": "visual_match",
                "severity": "warning",
                "description": str(item).strip(),
                "suggested_fix": "",
            })
    return out
