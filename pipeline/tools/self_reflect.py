"""Tool_SelfReflect: Verify → Reflect → Patch → Re-plan loop for shot plan quality.

Supports two modes:
  - Full mode (real): LLM verify + HITL interrupt + patch
  - Agentic lightweight mode (fake): vocal_script 质检（相似度+压缩率）+ Qwen 诊断 + 自动修复循环
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any, Callable

from ..llm import LLMClient, batch_chat_json

SYSTEM_PROMPT = """\
你是视频导演质检专家。你需要检查镜头规划的逻辑完整性、一致性和质量，找出问题并给出修复方案。

硬性输出规则:
1. 只能输出一个合法 JSON object，不能输出 JSON array、tuple、Markdown、解释文字或代码块。
2. 不要输出 <think>、分析过程、注释、前后缀文本。
3. 所有字段必须按用户要求的 schema 返回；无法判断时使用空数组、空字符串或 false。
4. issues、fix_options、shots 等集合字段必须是 JSON array，array 里的每一项必须是 JSON object。"""

VERIFY_TEMPLATE = """\
请验证以下镜头规划的质量，检查以下方面:
1. 逻辑连贯性: 镜头之间是否自然过渡
2. 内容完整性: 是否覆盖了所有段落的核心内容
3. 旁白质量: vocal_script 是否流畅自然
4. 视觉描述: visual_query 是否具体可检索
5. spotlight 合理性: 重点镜头选择是否恰当
6. 时长合理性: 各镜头时长是否符合旁白文本长度
7. 检索可用性: visual_query 是否只含实拍语义，未混入分屏/卡片/漫画/插画/字幕条等后期包装词

原始文章分段:
{segments_json}

当前镜头规划:
{shot_plan_json}

输出要求(严格JSON格式):
{{
  "passed": true/false,
  "overall_score": 0.0-1.0,
  "issues": [
    {{
      "issue_id": "issue_001",
      "severity": "critical/warning/info",
      "shot_id": "shot_001 或 global",
      "category": "逻辑/内容/旁白/视觉/spotlight/时长",
      "description": "问题描述",
      "suggested_fix": "修复建议"
    }}
  ],
  "summary": "整体评价"
}}"""

PATCH_TEMPLATE = """\
请根据以下验证问题修复镜头规划。

验证结果:
{verify_result_json}

当前镜头规划:
{shot_plan_json}

输出修复后的完整 shot_plan(保持相同JSON格式):
{{
  "shots": [...]
}}"""

OPTIONS_TEMPLATE = """\
对以下镜头规划的问题，请提供 {num_options} 个不同的修复方案:

验证问题:
{issues_json}

当前规划:
{shot_plan_json}

输出要求(严格JSON object格式):
{{
  "options": [
    {{
      "option_id": 0,
      "label": "方案简称",
      "description": "详细修复说明",
      "changes": {{}}
    }}
  ]
}}"""


def verify(
    llm: LLMClient,
    segments: list[dict[str, Any]],
    shot_plan: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run verification on the current shot plan."""
    segments = _dict_list(segments)
    shot_plan = _dict_list(shot_plan)
    if not shot_plan:
        return {
            "passed": False,
            "overall_score": 0.0,
            "issues": [{
                "issue_id": "issue_001",
                "severity": "critical",
                "shot_id": "global",
                "category": "内容",
                "description": "镜头规划为空，无法进入后续执行阶段",
                "suggested_fix": "重新生成文章分段和镜头规划",
            }],
            "summary": "shot_plan is empty",
        }

    prompt = VERIFY_TEMPLATE.format(
        segments_json=json.dumps(segments, ensure_ascii=False, indent=2),
        shot_plan_json=json.dumps(shot_plan, ensure_ascii=False, indent=2),
    )
    try:
        result = _as_dict(llm.chat_json(SYSTEM_PROMPT, prompt, temperature=0.2))
    except Exception:
        result = {}
    if not result or not isinstance(result, dict):
        return _local_verify(segments, shot_plan)
    return result


def generate_fix_options(
    llm: LLMClient,
    issues: list[dict[str, Any]],
    shot_plan: list[dict[str, Any]],
    num_options: int = 4,
) -> list[dict[str, Any]]:
    """Generate multiple fix options for HITL decision."""
    issues = _dict_list(issues)
    shot_plan = _dict_list(shot_plan)
    prompt = OPTIONS_TEMPLATE.format(
        num_options=num_options,
        issues_json=json.dumps(issues, ensure_ascii=False, indent=2),
        shot_plan_json=json.dumps(shot_plan, ensure_ascii=False, indent=2),
    )
    try:
        result = llm.chat_json(SYSTEM_PROMPT, prompt, temperature=0.7)
    except Exception:
        return []
    result = _as_dict(result)
    items = result.get("items") or result.get("options") or result.get("fixes") or []
    if not isinstance(items, list):
        items = []
    return [opt for opt in items if isinstance(opt, dict)][:num_options]


def patch(
    llm: LLMClient,
    verify_result: dict[str, Any],
    shot_plan: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply LLM-driven patch to fix issues."""
    verify_result = _as_dict(verify_result)
    shot_plan = _dict_list(shot_plan)
    prompt = PATCH_TEMPLATE.format(
        verify_result_json=json.dumps(verify_result, ensure_ascii=False, indent=2),
        shot_plan_json=json.dumps(shot_plan, ensure_ascii=False, indent=2),
    )
    try:
        result = _as_dict(llm.chat_json(SYSTEM_PROMPT, prompt, temperature=0.2))
    except Exception:
        result = {"shots": shot_plan}
    if not isinstance(result, dict):
        result = {"shots": shot_plan}
    if not isinstance(result.get("shots"), list):
        result["shots"] = shot_plan
    return result


def run(
    llm: LLMClient,
    segments: list[dict[str, Any]],
    shot_plan: list[dict[str, Any]],
    num_options: int = 4,
) -> dict[str, Any]:
    """Full verify → reflect cycle. Returns verification result with fix options."""
    segments = _dict_list(segments)
    shot_plan = _dict_list(shot_plan)

    try:
        vresult = verify(llm, segments, shot_plan)
    except Exception:
        vresult = _local_verify(segments, shot_plan)
    vresult = _as_dict(vresult)
    if _is_empty_verify_result(vresult):
        vresult = _local_verify(segments, shot_plan)

    issues = vresult.get("issues", [])
    if not isinstance(issues, list):
        issues = _dict_list(issues)
    else:
        issues = [i for i in issues if isinstance(i, dict)]
    if not issues and not bool(vresult.get("passed", False)):
        # Guardrail: do not surface empty HITL questions.
        issues = [{
            "issue_id": "issue_reflect_empty",
            "severity": "critical",
            "shot_id": "global",
            "category": "系统",
            "description": "反思结果为空或结构不完整，已自动回退本地校验",
            "suggested_fix": "重新执行本地校验并检查上游镜头规划字段完整性",
        }]
    if not str(vresult.get("summary", "")).strip():
        vresult["summary"] = "反思结果为空，已启用兜底校验"

    # Any reported issue means the plan is not ready. Warnings still require
    # HITL review because downstream generation is expensive and irreversible.
    passed = bool(vresult.get("passed", False)) and not issues
    if not passed and issues:
        try:
            options = generate_fix_options(llm, issues, shot_plan, num_options=num_options)
        except Exception:
            options = []
        if len(options) < num_options:
            options = (options + _fallback_fix_options(issues, shot_plan, num_options))[:num_options]
    else:
        options = []
    overall_score = _as_float(vresult.get("overall_score", 0.0), 0.0)
    if not passed and not issues:
        overall_score = min(overall_score, 0.2)
    return {
        "passed": passed,
        "overall_score": overall_score,
        "issues": issues,
        "summary": str(vresult.get("summary", "")),
        "fix_options": options,
        "status": "approved" if passed else "needs_fix",
    }


# ---------------------------------------------------------------------------
# Agentic lightweight self-reflect (fake mode)
# ---------------------------------------------------------------------------

SELF_REFLECT_OBSERVATION_TEMPLATE = """\
你是镜头规划质检专家。请审查每个镜头的 vocal_script 质量。

## 审查维度
1. 语义完整性：vocal_script 是否覆盖对应段落的核心论点？
   - 已预计算相似度（similarity_to_original），低于 0.75 为异常
2. 内容压缩度：vocal_script 是否过度压缩？
   - 已预计算压缩率（compression_ratio），高于 10% 为异常
3. Spotlight 合理性：spotlight 分布是否在关键位置？
4. 信息遗漏：段落核心信息是否在任一镜头中完全缺失？

## 镜头质检数据（含程序化指标）
{shots_with_metrics_json}

## 原始文章段落
{segments_summary}

## 输出 JSON
{{
  "has_issues": true,
  "failed_shots": [
    {{
      "shot_id": "...",
      "failed_checks": ["similarity_low", "over_compressed", "coverage_missing", "spotlight_misplaced"],
      "severity": "critical",
      "root_cause_analysis": "根因：为什么这个检查项失败？",
      "suggested_repair_direction": "修复方向"
    }}
  ],
  "global_issues": [],
  "summary": "诊断总结"
}}

## 规则
- similarity_to_original < 0.75 且你确认语义确实偏离 → failed_checks 加 similarity_low
- compression_ratio > 0.10 且你确认关键信息缺失 → failed_checks 加 over_compressed
- 两者都 OK 但有其他语义问题 → 按实际情况报 failed_checks
- 全部通过 → has_issues=false, failed_shots=[], global_issues=[]"""

SELF_REFLECT_REPAIR_TEMPLATE = """\
你是镜头规划修复专家。对诊断出的问题镜头做最小化修复。

## 诊断数据
{failed_shots_json}

## 原始文章段落（修复素材源）
{segments_full_text_json}

## 修复原则
1. vocal_script 相似度低 → 从原文 raw_span 提取核心论据，重写口语化
2. vocal_script 压缩过度 → 补充缺失的关键信息点，保证语义完整
3. 信息遗漏 → 从遗漏段落 key_points 生成新镜头
4. spotlight 位置错误 → 调整 is_spotlight 标记

## 输出 JSON
{{
  "reflection": "我的修复思路...",
  "repaired_shots": [
    {{
      "shot_id": "...",
      "changed_fields": {{
        "vocal_script": "新的口播文本",
        "is_spotlight": true
      }},
      "change_reason": "修复原因",
      "expected_similarity_after": 0.85,
      "expected_compression_after": 0.05
    }}
	  ]
	}}"""

SELF_REFLECT_SEGMENT_OBSERVATION_TEMPLATE = """\
你是文章分段质检专家。请先审查 Stage 2a 的文章分段本身，再决定是否需要修复。

## 审查目标
1. 分段边界是否符合原文的时间/论述顺序，是否存在该合并或拆开的段落
2. 每段是否有独立信息量，title/summary/key_points 是否互相支撑
3. raw_span 是否能作为原文锚点，是否过短、重复或与相邻段语义重叠
4. 分段是否能支撑当前 shot_plan 的章节归属；不要为了迎合某个坏镜头强行改分段

## 当前文章分段
{segments_json}

## 当前镜头规划摘要
{shot_plan_summary_json}

## 输出 JSON
{{
  "has_issues": true,
  "issues": [
    {{
      "segment_id": "seg_01 或 global",
      "severity": "critical/warning/info",
      "category": "边界/重复/信息量/锚点/章节映射",
      "description": "问题描述",
      "repair_direction": "修复方向"
    }}
  ],
  "repaired_segments": [
    {{
      "segment_id": "seg_01",
      "structure_label": "引子/概述/核心章节/总结升华",
      "title": "内容型章节标题",
      "summary": "本章叙事重点",
      "key_points": ["要点1", "要点2"],
      "raw_span": "原文锚点",
      "planner_strategy": "视觉表达策略"
    }}
  ],
  "summary": "分段反思总结"
}}

## 规则
- 如果分段没有结构性问题，输出 has_issues=false, issues=[], repaired_segments=[]。
- 如果需要修复，repaired_segments 必须是完整分段列表，按原文顺序排列，不能只输出局部 diff。
- 修复分段时保持 2~8 段，segment_id 使用 seg_01, seg_02 ... 连续编号。"""


def _compute_text_similarity(
    llm: LLMClient,
    text_a: str,
    text_b: str,
) -> float:
    """用 LLM paraphrase 判断两段文本是否表达相同意思，返回 0.0-1.0 相似度分数。"""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    if text_a.strip() == text_b.strip():
        return 1.0

    prompt = (
        "请判断以下两段文本是否表达了相同的核心意思，给出 0.0 到 1.0 的语义相似度分数。\n"
        "1.0 = 完全相同的意思，0.0 = 完全无关。\n"
        "只输出一个 JSON: {\"similarity\": 0.0}\n\n"
        f"文本A: {text_a[:500]}\n\n"
        f"文本B: {text_b[:500]}"
    )
    try:
        result = _as_dict(llm.chat_json(
            "你是文本语义分析专家。只输出 JSON。",
            prompt,
            temperature=0.1,
        ))
        score = float(result.get("similarity", 0.5))
        return max(0.0, min(1.0, score))
    except Exception:
        return _fallback_text_similarity(text_a, text_b)


def _fallback_text_similarity(text_a: str, text_b: str) -> float:
    """LLM 不可用时的轻量兜底；只用于避免把失败误当成固定 0.5。"""
    a = "".join(str(text_a or "").split())
    b = "".join(str(text_b or "").split())
    if not a or not b:
        return 0.0
    return round(SequenceMatcher(None, a, b).ratio(), 3)


def _compute_text_similarities(
    llm: LLMClient,
    pairs: list[dict[str, Any]],
    *,
    batch_size: int = 30,
    concurrency: int = 2,
) -> dict[str, float]:
    """批量执行 LLM paraphrase 判断，避免每个镜头串行调用一次 Qwen。"""
    if not pairs:
        return {}

    def _format_prompt(batch: list[dict[str, Any]]) -> str:
        rows = [
            {
                "pair_id": str(item.get("pair_id", "")),
                "text_a": str(item.get("text_a", ""))[:1200],
                "text_b": str(item.get("text_b", ""))[:1200],
            }
            for item in batch
        ]
        return (
            "请批量判断每组文本是否表达相同核心意思，给出 0.0-1.0 语义相似度。"
            "必须覆盖全部 pair_id，只输出 JSON："
            '{"scores":[{"pair_id":"...","similarity":0.0}]}。\n'
            f"文本对：{json.dumps(rows, ensure_ascii=False)}"
        )

    try:
        rows = batch_chat_json(
            llm,
            items=pairs,
            format_prompt=_format_prompt,
            extract_results=lambda raw: raw.get("scores", []),
            system_prompt="你是文本语义分析专家。只输出合法 JSON，不遗漏任何 pair_id。",
            temperature=0.1,
            max_batch_size=max(1, batch_size),
            max_batch_output_chars=12000,
            concurrency=max(1, concurrency),
            item_label="text_similarity_pairs",
            output_key="scores",
        )
    except Exception:
        rows = []

    scores: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id:
            continue
        try:
            score = float(row.get("similarity", 0.0))
        except (TypeError, ValueError):
            continue
        scores[pair_id] = max(0.0, min(1.0, score))

    for item in pairs:
        pair_id = str(item.get("pair_id", "")).strip()
        if pair_id and pair_id not in scores:
            scores[pair_id] = _fallback_text_similarity(
                str(item.get("text_a", "")),
                str(item.get("text_b", "")),
            )
    return scores


def _check_vocal_script_quality(
    llm: LLMClient,
    vocal_script: str,
    original_segment_text: str,
    *,
    min_similarity: float = 0.75,
    max_compression_ratio: float = 0.10,
    similarity_override: float | None = None,
) -> dict[str, Any]:
    """量化检查 vocal_script 质量（相似度 + 压缩率）。"""
    # 1. 语义相似度（LLM paraphrase）
    similarity = (
        _compute_text_similarity(llm, vocal_script, original_segment_text)
        if similarity_override is None
        else max(0.0, min(1.0, float(similarity_override)))
    )

    # 2. 长度压缩率
    orig_len = max(len(original_segment_text), 1)
    vocal_len = len(vocal_script)
    compression_ratio = max(0.0, (orig_len - vocal_len) / orig_len)

    issues: list[str] = []
    if similarity < min_similarity:
        issues.append(f"相似度过低: {similarity:.2f} < {min_similarity}")
    if compression_ratio > max_compression_ratio:
        issues.append(f"口播压缩过度: {compression_ratio:.1%} > {max_compression_ratio:.0%}")

    return {
        "passed": len(issues) == 0,
        "similarity": round(similarity, 3),
        "compression_ratio": round(compression_ratio, 3),
        "compression_ok": compression_ratio <= max_compression_ratio,
        "issues": issues,
    }


def _segment_text(segment: dict[str, Any]) -> str:
    parts = [
        str(segment.get("title", "") or ""),
        str(segment.get("summary", "") or ""),
        str(segment.get("raw_span", "") or segment.get("narration_script", "") or ""),
        " ".join(str(item) for item in (segment.get("key_points", []) or [])),
    ]
    return " ".join(part.strip() for part in parts if part and part.strip())


def _normalize_reflected_segments(value: Any, fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw = _dict_list(value)
    if not raw:
        return fallback
    normalized: list[dict[str, Any]] = []
    for item in raw[:8]:
        text = _segment_text(item)
        if not text:
            continue
        idx = len(normalized) + 1
        title = str(item.get("title") or item.get("chapter_title") or item.get("summary") or f"章节{idx}").strip()
        key_points = item.get("key_points", [])
        if not isinstance(key_points, list):
            key_points = [str(key_points)]
        normalized.append({
            **item,
            "segment_id": f"seg_{idx:02d}",
            "structure_label": str(item.get("structure_label") or _default_structure_label(idx, len(raw))).strip(),
            "title": title[:24],
            "summary": str(item.get("summary") or text).strip()[:300],
            "raw_span": str(item.get("raw_span") or item.get("narration_script") or text).strip()[:240],
            "narration_script": str(item.get("narration_script") or item.get("raw_span") or text).strip()[:400],
            "key_points": [str(point).strip() for point in key_points if str(point).strip()][:5],
            "planner_strategy": str(item.get("planner_strategy") or "").strip()[:240],
        })
    return normalized or fallback


def _default_structure_label(index_one_based: int, total: int) -> str:
    if index_one_based <= 1:
        return "引子"
    if index_one_based >= max(1, total):
        return "总结升华"
    if index_one_based == 2 and total > 3:
        return "概述"
    return "核心章节"


def _segments_equivalent(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    fields = ("title", "summary", "raw_span", "key_points", "structure_label")
    compact_left = [{field: item.get(field) for field in fields} for item in left]
    compact_right = [{field: item.get(field) for field in fields} for item in right]
    return compact_left == compact_right


def _local_segment_reflection(segments: list[dict[str, Any]], shot_plan: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if not segments:
        issues.append({
            "segment_id": "global",
            "severity": "critical",
            "category": "信息量",
            "description": "文章分段为空，无法支撑镜头规划",
            "repair_direction": "重新生成完整分段",
        })
    seen_texts: list[str] = []
    for idx, segment in enumerate(segments):
        sid = str(segment.get("segment_id") or f"seg_{idx + 1:02d}")
        text = _segment_text(segment)
        if len(text) < 12:
            issues.append({
                "segment_id": sid,
                "severity": "warning",
                "category": "信息量",
                "description": "分段文本过短，独立信息量不足",
                "repair_direction": "与相邻段合并或补充 summary/key_points",
            })
        for prev in seen_texts[-1:]:
            if prev and text and SequenceMatcher(None, prev, text).ratio() > 0.72:
                issues.append({
                    "segment_id": sid,
                    "severity": "warning",
                    "category": "重复",
                    "description": "相邻分段语义高度重叠，可能需要合并或重划边界",
                    "repair_direction": "合并重复段或重新划分段落边界",
                })
        seen_texts.append(text)

    segment_ids = {str(seg.get("segment_id") or "") for seg in segments if seg.get("segment_id")}
    used_ids = {str(shot.get("segment_id") or "") for shot in shot_plan if shot.get("segment_id")}
    missing = sorted(segment_ids - used_ids)
    if segment_ids and missing:
        issues.append({
            "segment_id": "global",
            "severity": "warning",
            "category": "章节映射",
            "description": f"有 {len(missing)} 个分段未被任何镜头覆盖: {', '.join(missing[:4])}",
            "repair_direction": "保持分段顺序，级联修正 shot_plan 的 segment_id",
        })
    return {
        "has_issues": bool(issues),
        "issues": issues,
        "repaired_segments": [],
        "summary": "本地分段反思完成",
        "source": "local_rules",
    }


def _reflect_and_repair_segments(
    llm: LLMClient,
    segments: list[dict[str, Any]],
    shot_plan: list[dict[str, Any]],
) -> dict[str, Any]:
    shot_summary = [
        {
            "shot_id": shot.get("shot_id", ""),
            "segment_id": shot.get("segment_id", ""),
            "chapter_title": shot.get("chapter_title", ""),
            "vocal_script": str(shot.get("vocal_script", ""))[:160],
            "is_spotlight": shot.get("is_spotlight", False),
        }
        for shot in shot_plan
    ]
    try:
        prompt = SELF_REFLECT_SEGMENT_OBSERVATION_TEMPLATE.format(
            segments_json=json.dumps(segments, ensure_ascii=False, indent=2),
            shot_plan_summary_json=json.dumps(shot_summary, ensure_ascii=False, indent=2),
        )
        result = _as_dict(llm.chat_json(SYSTEM_PROMPT, prompt, temperature=0.2))
    except Exception:
        result = {}
    if not result:
        result = _local_segment_reflection(segments, shot_plan)
    if not isinstance(result.get("issues"), list):
        result["issues"] = []
    repaired = _normalize_reflected_segments(result.get("repaired_segments", []), segments)
    if bool(result.get("has_issues")) and not _segments_equivalent(segments, repaired):
        result["repaired_segments"] = repaired
    else:
        result["repaired_segments"] = []
    result["has_issues"] = bool(result.get("has_issues")) or bool(result.get("issues"))
    return result


def _cascade_segments_to_shots(
    shot_plan: list[dict[str, Any]],
    old_segments: list[dict[str, Any]],
    new_segments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    if not shot_plan or not new_segments:
        return shot_plan, []
    old_index = {
        str(segment.get("segment_id") or ""): idx
        for idx, segment in enumerate(old_segments)
        if segment.get("segment_id")
    }
    new_texts = [_segment_text(segment) for segment in new_segments]
    affected: list[str] = []
    seen_segment_ids: set[str] = set()
    cascaded: list[dict[str, Any]] = []
    for idx, shot in enumerate(shot_plan):
        updated = dict(shot)
        shot_text = " ".join([
            str(shot.get("chapter_title", "") or ""),
            str(shot.get("vocal_script", "") or ""),
            str(shot.get("intent", "") or ""),
        ]).strip()
        best_idx = -1
        best_score = 0.0
        for seg_idx, seg_text in enumerate(new_texts):
            if not shot_text or not seg_text:
                continue
            score = SequenceMatcher(None, shot_text, seg_text).ratio()
            if score > best_score:
                best_idx = seg_idx
                best_score = score
        if best_idx < 0 or best_score < 0.10:
            old_seg_id = str(shot.get("segment_id") or "")
            best_idx = min(len(new_segments) - 1, max(0, old_index.get(old_seg_id, idx * len(new_segments) // max(1, len(shot_plan)))))
        segment = new_segments[best_idx]
        new_seg_id = str(segment.get("segment_id") or f"seg_{best_idx + 1:02d}")
        old_values = (
            updated.get("segment_id"),
            updated.get("chapter_title"),
            updated.get("structure_label"),
            updated.get("is_segment_first"),
        )
        is_first = new_seg_id not in seen_segment_ids
        seen_segment_ids.add(new_seg_id)
        updated["segment_id"] = new_seg_id
        updated["chapter_title"] = str(segment.get("title") or updated.get("chapter_title") or "")
        updated["structure_label"] = str(segment.get("structure_label") or updated.get("structure_label") or "")
        updated["planner_strategy"] = str(segment.get("planner_strategy") or updated.get("planner_strategy") or "")
        updated["key_points"] = segment.get("key_points", updated.get("key_points", []))
        updated["is_segment_first"] = is_first
        if is_first:
            updated["is_spotlight"] = True
            if not str(updated.get("spotlight_reason", "")).strip():
                updated["spotlight_reason"] = "分段级联后成为章节首镜头，需要强化结构锚点。"
        new_values = (
            updated.get("segment_id"),
            updated.get("chapter_title"),
            updated.get("structure_label"),
            updated.get("is_segment_first"),
        )
        if new_values != old_values:
            affected.append(str(updated.get("shot_id") or idx + 1))
        cascaded.append(updated)
    return cascaded, affected


def agentic_run(
    llm: LLMClient,
    segments: list[dict[str, Any]],
    shot_plan: list[dict[str, Any]],
    *,
    max_rounds: int = 2,
    min_similarity: float = 0.75,
    max_compression_ratio: float = 0.10,
    diagnose_batch_size: int = 6,
    diagnose_concurrency: int = 3,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Agentic 轻量自反思：分批诊断 → 批量修复 → 复检循环。

    优化策略（单镜头最大速度）:
      Phase 1 - 分批 Qwen 诊断: 所有镜头按 diagnose_batch_size 拆批，
                diagnose_concurrency 并发调 Qwen，每批独立诊断。
      Phase 2 - 批量修复: 收集 Phase 1 全部失败镜头，一次 Qwen 调用修复。
      Phase 3 - 复检: 只对修复过的镜头重算程序化指标。
      Phase 4 - 循环: 仍有问题 → Phase 1（仅诊断失败镜头）→ Phase 2 → ...
    """
    segments = _dict_list(segments)
    shot_plan = _dict_list(shot_plan)

    def _refresh_segment_map(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {
            str(s.get("segment_id", "")): s
            for s in items
            if isinstance(s, dict) and str(s.get("segment_id", ""))
        }

    segment_map = _refresh_segment_map(segments)
    trajectory: list[dict[str, Any]] = []

    def _progress(sub_tool: str, current: int, label: str) -> None:
        if progress_callback is not None:
            progress_callback(sub_tool, current, 100, label)

    # ── 纯数学程序化指标：压缩率（不调 LLM）──
    def _compute_compression(shots: list[dict]) -> dict[str, dict]:
        """计算每镜头的 vocal_script 压缩率（纯数学，无 LLM）。"""
        metrics: dict[str, dict] = {}
        # 按 segment 分组，计算聚合压缩率
        groups: dict[str, list[dict]] = {}
        for shot in shots:
            seg_id = str(shot.get("segment_id", "") or shot.get("shot_id", ""))
            groups.setdefault(seg_id, []).append(shot)

        for seg_id, g_shots in groups.items():
            all_script = " ".join(str(s.get("vocal_script", "")) for s in g_shots)
            seg = segment_map.get(seg_id, {})
            original = (
                str(seg.get("raw_span", "") or "").strip()
                or " ".join(str(x) for x in (seg.get("key_points", []) or []))
                or all_script
            )
            orig_len = max(len(original), 1)
            vocal_len = len(all_script)
            comp_ratio = max(0.0, (orig_len - vocal_len) / orig_len)
            comp_ok = comp_ratio <= max_compression_ratio
            for s in g_shots:
                sid = s.get("shot_id", "")
                metrics[sid] = {
                    "compression_ratio": round(comp_ratio, 3),
                    "compression_ok": comp_ok,
                    "vocal_length": vocal_len,
                    "original_length": orig_len,
                }
        return metrics

    # ── 合并诊断：一次 Qwen 调用同时判断相似度 + coverage + spotlight ──
    def _batch_diagnose(shots_to_check: list[dict], metrics: dict, round_label: str) -> tuple[set, set]:
        """分批 Qwen 诊断，同时做相似度判断 + 语义检查。返回 (failed_sids, all_failed)。

        每批 prompt 包含: shot 信息 + 原文 + 压缩率（参考值）
        Qwen 一次给出: 相似度判断 + coverage + spotlight 问题
        等价于合并原来的"相似度计算"和"语义诊断"两步。
        """
        if not shots_to_check:
            return set(), set()
        chunks = [shots_to_check[i:i + diagnose_batch_size] for i in range(0, len(shots_to_check), diagnose_batch_size)]
        _progress("Qwen诊断", 0, f"{round_label}：{len(chunks)} 批 × ~{diagnose_batch_size} 镜头，并发 {diagnose_concurrency}")

        all_failed_shots: list[dict] = []
        import concurrent.futures

        def _diag_one(chunk: list[dict]) -> list[dict]:
            try:
                # 构建紧凑 prompt：shot 信息 + 压缩率 + 原文
                compact = []
                for shot in chunk:
                    sid = shot.get("shot_id", "")
                    seg_id = str(shot.get("segment_id", "") or "")
                    seg = segment_map.get(seg_id, {})
                    compact.append({
                        "shot_id": sid,
                        "chapter_title": shot.get("chapter_title", ""),
                        "vocal_script": str(shot.get("vocal_script", ""))[:200],
                        "is_spotlight": shot.get("is_spotlight"),
                        "segment_id": seg_id,
                        "compression_ratio": metrics.get(sid, {}).get("compression_ratio", 0),
                        "compression_ok": metrics.get(sid, {}).get("compression_ok", True),
                        "original_text": (
                            str(seg.get("raw_span", "") or "").strip()[:300]
                            or " ".join(str(x) for x in (seg.get("key_points", []) or []))[:300]
                        ),
                    })
                prompt = SELF_REFLECT_OBSERVATION_TEMPLATE.format(
                    shots_with_metrics_json=json.dumps(compact, ensure_ascii=False),
                    segments_summary=json.dumps([
                        {"id": s.get("segment_id"), "key_points": s.get("key_points", [])}
                        for s in segments
                    ], ensure_ascii=False),
                )
                result = _as_dict(llm.chat_json(SYSTEM_PROMPT, prompt, temperature=0.2))
                return result.get("failed_shots", [])
            except Exception:
                return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=diagnose_concurrency) as ex:
            futures = [ex.submit(_diag_one, c) for c in chunks]
            done = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    all_failed_shots.extend(future.result())
                except Exception:
                    pass
                done += 1
                _progress("Qwen诊断", int(done / len(chunks) * 50),
                          f"{round_label}：诊断 {done}/{len(chunks)} 批完成")

        llm_failed = {f["shot_id"] for f in all_failed_shots if isinstance(f, dict) and f.get("shot_id")}
        # 压缩率超标也加入失败集合
        comp_failed = {sid for sid, m in metrics.items() if not m.get("compression_ok", True)}
        all_failed = comp_failed | llm_failed
        return all_failed, all_failed

    # ── Phase 2: 批量修复 ──
    def _batch_repair(failed_sids: set[str], round_label: str) -> dict:
        _progress("批量修复", 55,
                  f"{round_label}：收集到 {len(failed_sids)} 个问题镜头，一次 Qwen 调用修复")
        failed_shots_data = []
        for shot in shot_plan:
            sid = shot.get("shot_id", "")
            if sid not in failed_sids:
                continue
            seg = segment_map.get(str(shot.get("segment_id", "") or ""), {})
            failed_shots_data.append({
                "shot_id": sid,
                "current_vocal_script": shot.get("vocal_script", ""),
                "original_segment_full": (
                    str(seg.get("raw_span", "") or "").strip()
                    or " ".join(str(x) for x in (seg.get("key_points", []) or []))
                ),
            })
        try:
            repair_prompt = SELF_REFLECT_REPAIR_TEMPLATE.format(
                failed_shots_json=json.dumps(failed_shots_data, ensure_ascii=False),
                segments_full_text_json=json.dumps([
                    {"segment_id": s.get("segment_id", ""), "raw_span": s.get("raw_span", ""),
                     "key_points": s.get("key_points", [])}
                    for s in segments
                ], ensure_ascii=False),
            )
            return _as_dict(llm.chat_json(SYSTEM_PROMPT, repair_prompt, temperature=0.2))
        except Exception:
            return {"repaired_shots": [], "reflection": "LLM修复失败"}

    # ── Round 0: 先反思文章分段，再级联修正 shot → segment 映射 ──
    _progress("分段反思", 0, "先检查 Stage 2a 文章分段边界与信息独立性")
    segment_reflection = _reflect_and_repair_segments(llm, segments, shot_plan)
    segment_issues = _dict_list(segment_reflection.get("issues", []))
    repaired_segments = _dict_list(segment_reflection.get("repaired_segments", []))
    segment_changed = bool(repaired_segments) and not _segments_equivalent(segments, repaired_segments)
    affected_shots: list[str] = []
    if segment_changed:
        old_segments = segments
        segments = repaired_segments
        segment_map = _refresh_segment_map(segments)
        shot_plan, affected_shots = _cascade_segments_to_shots(shot_plan, old_segments, segments)
        _progress("分段级联", 20, f"分段已修复，级联更新 {len(affected_shots)} 个镜头章节归属")
    else:
        _progress("分段反思", 20, "文章分段结构可用，无需级联重映射")
    unresolved_segment_issues = segment_issues if segment_issues and not segment_changed else []
    trajectory.append({
        "role": "thought",
        "content": (
            f"[OBSERVATION] Stage 2a 分段反思完成：发现 {len(segment_issues)} 个分段问题；"
            + (
                f"已修复分段并级联更新 {len(affected_shots)} 个镜头的 segment_id。"
                if segment_changed
                else "分段边界保持不变，继续检查 shot_plan。"
            )
        ),
        "metadata": {
            "stage": "stage2_segmentation_reflect",
            "phase": "observation_repair_cascade",
            "segment_changed": segment_changed,
            "affected_shot_ids": affected_shots[:20],
        },
    })

    # ════════════ 主循环 ════════════
    for round_num in range(max_rounds + 1):
        round_label = f"第 {round_num + 1} 轮"

        # Phase 1: 纯数学压缩率 + Qwen 合并诊断（相似度+coverage+spotlight 一次搞定）
        _progress("压缩率计算", 5, f"{round_label}：纯数学计算压缩率")
        metrics = _compute_compression(shot_plan)
        comp_failed = {sid for sid, m in metrics.items() if not m.get("compression_ok")}

        if round_num == 0:
            all_failed, _ = _batch_diagnose(shot_plan, metrics, round_label)
        else:
            # 第 2+ 轮：只诊断上一轮修复后仍失败的
            still_failed = [s for s in shot_plan if s.get("shot_id") in comp_failed]
            all_failed, _ = _batch_diagnose(still_failed, metrics, round_label)
        _progress("Qwen诊断", 50, f"{round_label}：诊断完成——{len(all_failed)} 个问题镜头")

        # Trajectory: OBSERVATION
        trajectory.append({
            "role": "thought",
            "content": (
                f"第 {round_num+1} 轮诊断——压缩率超标 {len(comp_failed)} 个，"
                f"Qwen 诊断共发现 {len(all_failed)} 个有问题（含相似度、覆盖、spotlight）"
            ),
            "metadata": {"stage": "stage2_self_reflect", "round": round_num},
        })

        if not all_failed:
            if unresolved_segment_issues:
                issues = [
                    {
                        "shot_id": "global",
                        "severity": str(issue.get("severity", "warning")),
                        "category": f"分段/{issue.get('category', '结构')}",
                        "description": str(issue.get("description", "分段反思发现未修复问题")),
                        "suggested_fix": str(issue.get("repair_direction", "重新检查文章分段边界")),
                    }
                    for issue in unresolved_segment_issues
                ]
                return {
                    "passed": False, "overall_score": 0.55, "issues": issues,
                    "summary": f"分段反思发现 {len(issues)} 个未修复问题",
                    "fix_options": [], "status": "needs_fix", "segments": segments,
                    "shot_plan": shot_plan, "trajectory": trajectory,
                }
            _progress("完成", 100, "全部旁白通过自检")
            trajectory.append({
                "role": "thought", "content": "全部镜头通过自反思检查，plan_approved=true",
                "metadata": {"stage": "stage2_self_reflect", "round": round_num},
            })
            return {
                "passed": True, "overall_score": 1.0, "issues": [], "summary": "质检通过",
                "fix_options": [], "status": "approved", "segments": segments,
                "shot_plan": shot_plan, "trajectory": trajectory,
            }

        if round_num >= max_rounds:
            _progress("完成", 100, f"{max_rounds} 轮修复预算耗尽")
            issues = []
            for sid in sorted(all_failed):
                for t in metrics.get(sid, {}).get("issues", []):
                    issues.append({"shot_id": sid, "severity": "warning", "category": "旁白",
                                   "description": t, "suggested_fix": "从原文重写 vocal_script"})
            return {
                "passed": False, "overall_score": 0.4, "issues": issues,
                "summary": f"自反思 {max_rounds} 轮后仍有 {len(all_failed)} 个未通过",
                "fix_options": [], "status": "needs_fix", "segments": segments,
                "shot_plan": shot_plan, "trajectory": trajectory,
            }

        # Phase 2: 批量修复（所有失败镜头一次 Qwen 调用）
        repair_result = _batch_repair(all_failed, round_label)
        _progress("批量修复", 80, f"{round_label}：修复 {len(repair_result.get('repaired_shots', []))} 个镜头")

        trajectory.extend([
            {"role": "call", "tool": "Tool_SelfReflect_RepairShot",
             "tool_input": json.dumps({"failed_shot_ids": sorted(all_failed), "round": round_num}, ensure_ascii=False),
             "metadata": {"stage": "stage2_self_reflect_repair", "round": round_num}},
            {"role": "respond", "tool": "Tool_SelfReflect_RepairShot",
             "tool_output": json.dumps({
                 "repaired_count": len(repair_result.get("repaired_shots", [])),
                 "reflection": str(repair_result.get("reflection", ""))[:200],
             }, ensure_ascii=False),
             "metadata": {"stage": "stage2_self_reflect_repair", "round": round_num}},
        ])

        # 应用修复
        repair_map = {}
        for r in (repair_result.get("repaired_shots") or []):
            if isinstance(r, dict) and r.get("shot_id"):
                repair_map[r["shot_id"]] = r.get("changed_fields", {}) if isinstance(r.get("changed_fields"), dict) else {}
        for shot in shot_plan:
            if shot.get("shot_id") in repair_map:
                shot.update(repair_map[shot["shot_id"]])
        _progress("批量修复", 90, f"{round_label}：修复已应用，准备复检")

        # ──── 循环回到 Phase 1 ────

    return {
        "passed": False, "overall_score": 0.4, "issues": [],
        "summary": "自反思循环结束", "fix_options": [], "status": "needs_fix",
        "segments": segments, "shot_plan": shot_plan, "trajectory": trajectory,
    }


def _as_dict(value: Any) -> dict[str, Any]:
    """Normalize LLM outputs so callers never crash on tuple/list responses."""
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


def _dict_list(value: Any) -> list[dict[str, Any]]:
    """Keep only mapping items from common tuple/list wrapper shapes."""
    if isinstance(value, dict):
        return [value]
    if isinstance(value, tuple):
        items: list[dict[str, Any]] = []
        for part in value:
            items.extend(_dict_list(part))
        return items
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_empty_verify_result(vresult: dict[str, Any]) -> bool:
    """Detect unusable LLM verify output that would cause empty HITL prompts."""
    if not isinstance(vresult, dict) or not vresult:
        return True
    has_issues = isinstance(vresult.get("issues"), list) and len(vresult.get("issues", [])) > 0
    has_summary = bool(str(vresult.get("summary", "")).strip())
    has_score = isinstance(vresult.get("overall_score"), (int, float))
    # if all key signals are missing, treat as empty
    return not (has_issues or has_summary or has_score)


def _fallback_fix_options(
    issues: list[dict[str, Any]],
    shot_plan: list[dict[str, Any]],
    num_options: int = 4,
) -> list[dict[str, Any]]:
    """Deterministic HITL options when the LLM does not return usable options."""
    missing_desc = "；".join(
        str(issue.get("description", ""))
        for issue in issues
        if isinstance(issue, dict) and issue.get("category") == "内容"
    )
    shot_count = len(_dict_list(shot_plan))
    options = [
        {
            "option_id": 0,
            "label": "按ASR重分配段落",
            "description": "保持TransNetV2镜头数量不变，重新根据每个镜头ASR文本与文章分段的语义匹配分配segment_id。",
            "changes": {
                "strategy": "reassign_segments_by_asr_semantics",
                "preserve_shot_count": True,
                "target_shot_count": shot_count,
                "issues": missing_desc,
            },
        },
        {
            "option_id": 1,
            "label": "按时间顺序均衡覆盖",
            "description": "在语义匹配不稳定时，按镜头时间顺序将所有文章分段均衡覆盖，保证每个segment至少有一个镜头。",
            "changes": {
                "strategy": "rebalance_segments_by_time_order",
                "preserve_shot_count": True,
                "ensure_all_segments_covered": True,
            },
        },
        {
            "option_id": 2,
            "label": "保守补齐缺失段落",
            "description": "只调整当前未覆盖段落附近的镜头segment_id，尽量保持已有镜头规划内容不变。",
            "changes": {
                "strategy": "minimal_segment_coverage_patch",
                "preserve_existing_fields": True,
                "ensure_all_segments_covered": True,
            },
        },
        {
            "option_id": 3,
            "label": "重新生成完整镜头标注",
            "description": "基于文章分段和TransNetV2镜头边界重新生成完整shot_plan，适合当前规划覆盖关系整体错误的情况。",
            "changes": {
                "strategy": "regenerate_full_shot_plan",
                "preserve_shot_count": True,
                "target_shot_count": shot_count,
            },
        },
    ]
    return options[:num_options]


def _local_verify(
    segments: list[dict[str, Any]],
    shot_plan: list[dict[str, Any]],
) -> dict[str, Any]:
    segments = _dict_list(segments)
    shot_plan = _dict_list(shot_plan)
    issues: list[dict[str, Any]] = []
    segment_ids = {seg.get("segment_id") for seg in segments if seg.get("segment_id")}
    covered_ids = {shot.get("segment_id") for shot in shot_plan if shot.get("segment_id")}

    missing = sorted(segment_ids - covered_ids)
    if missing:
        coverage_ratio = len(covered_ids & segment_ids) / max(len(segment_ids), 1)
        severity = "critical" if coverage_ratio < 0.5 else "warning"
        issues.append({
            "issue_id": "issue_001",
            "severity": severity,
            "shot_id": "global",
            "category": "内容",
            "description": (
                f"以下段落未被镜头覆盖: {', '.join(missing)}"
                f" (覆盖率 {coverage_ratio:.0%})"
            ),
            "suggested_fix": "为缺失段落补充镜头",
        })

    # Detect degenerate plan: all shots identical vocal_script
    scripts = [shot.get("vocal_script", "") for shot in shot_plan]
    unique_scripts = set(s for s in scripts if s.strip())
    if len(unique_scripts) <= 1 and len(shot_plan) > 1:
        issues.append({
            "issue_id": "issue_degenerate",
            "severity": "critical",
            "shot_id": "global",
            "category": "旁白",
            "description": f"所有 {len(shot_plan)} 个镜头具有相同的 vocal_script，标注数据退化",
            "suggested_fix": "重新标注，确保每个镜头有独立的ASR文本",
        })

    non_retrieval_tokens = (
        "分屏", "转场", "字幕条", "数据卡片", "信息图", "图文排版",
        "漫画", "动漫", "插画", "贴纸", "速度线", "弹窗", "UI",
    )

    for i, shot in enumerate(shot_plan, start=1):
        sid = shot.get("shot_id", f"shot_{i:03d}")
        if not shot.get("vocal_script"):
            issues.append({
                "issue_id": f"issue_script_{i:03d}",
                "severity": "critical",
                "shot_id": sid,
                "category": "旁白",
                "description": "镜头缺少 vocal_script",
                "suggested_fix": "补充可朗读的旁白文本",
            })
        if not shot.get("visual_query"):
            issues.append({
                "issue_id": f"issue_visual_{i:03d}",
                "severity": "warning",
                "shot_id": sid,
                "category": "视觉",
                "description": "镜头缺少 visual_query",
                "suggested_fix": "补充可检索的视觉关键词",
            })
        else:
            visual_query = str(shot.get("visual_query", ""))
            bad_tokens = [token for token in non_retrieval_tokens if token in visual_query]
            if bad_tokens:
                issues.append({
                    "issue_id": f"issue_visual_style_{i:03d}",
                    "severity": "warning",
                    "shot_id": sid,
                    "category": "视觉",
                    "description": f"visual_query 含后期包装词，可能导致检索失败: {', '.join(bad_tokens[:4])}",
                    "suggested_fix": "visual_query 仅保留实拍主体/动作/场景，将包装描述放入 visual_direction",
                })

    critical_count = sum(1 for issue in issues if issue.get("severity") == "critical")
    warning_count = sum(1 for issue in issues if issue.get("severity") == "warning")
    passed = not issues
    score = 1.0 if not issues else max(0.1, 1.0 - 0.3 * critical_count - 0.1 * warning_count)
    return {
        "passed": passed,
        "overall_score": round(score, 2),
        "issues": issues,
        "summary": "本地自检通过" if passed else "本地自检发现镜头规划问题，需要人工审核或修复",
    }
