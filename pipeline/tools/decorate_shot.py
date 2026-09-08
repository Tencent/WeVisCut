"""Tool_DecorateShot: Template selection, slot filling, check & repair for spotlight shots.

Four cooperating stages:
  - template_planner: structure selection with design_rationale matching + diversity
  - slot_filler: slot filling from vocal_script/chapter_title with validation
  - verifiers: multi-dimensional check (render validity, timing, theme, continuity)
  - repair_planner: targeted repair (revise_slots / swap_structure / swap_motion)

Key improvements over the old implementation:
  1. Template selection uses gold_template_policies + instructions + design_rationale matching
  2. Slot filling extracts content from vocal_script/chapter_title (not placeholders)
  3. Check returns status="needs_repair" (not "success") when passed=false
  4. Self-repair loop: up to max_check_repair_iterations to auto-fix issues before HITL
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Optional

from ..llm import LLMClient

logger = logging.getLogger("aicut_pipeline")

SYSTEM_PROMPT = """\
你是视频模板装饰专家(WevisionCut Agent)。你为重点镜头(spotlight)选择合适的
Gold Template 结构，填充文本槽位，并设计叠加动效。

核心原则：
1. 选择 structure 时根据 design_rationale 的语义匹配（因果→cause_effect, 对比→comparison_split等）
2. 只能从候选 Gold Template structure_id 中选择，禁止自造 template_id
3. 同一视频中不同 shot 尽量选择不同的模板，充分利用模板多样性
4. 槽位填充必须来自 vocal_script / chapter_title，面向观众的事实传达
4. 禁止输出 design_rationale 里的管线内部描述性文字到展示槽位
5. 数字/专有名词必须在上下文中有字面依据
6. 只输出最终 JSON，不输出解释、Markdown、代码块或 <think>。"""

DECORATE_TEMPLATE = """\
为以下 spotlight 镜头设计模板装饰:

镜头信息:
- shot_id: {shot_id}
- chapter_title: {chapter_title}
- vocal_script: {vocal_script}
- visual_direction: {visual_direction}
- design_rationale: {design_rationale}
- emphasis_target: {emphasis_target}
- structure_label: {structure_label}

{template_policy_hint}
{instructions_hint}
可用 Gold Template 候选(只能选择其中一个 template_id):
{registry_hint}
已使用模板(禁止重复): {already_used}

输出(严格JSON):
{{
  "template_id": "结构标识(如 cause_effect_chain / comparison_split / stat_highlight 等)",
  "template_type": "lower_third|overlay|motion|caption_emphasis|title_card|split_screen",
  "slots": {{
    "title": "从vocal_script提炼的核心短句（面向观众）",
    "subtitle": "补充解释文案",
    "eyebrow": "从chapter_title提炼的简短标签",
    "accent_color": "#hex颜色",
    "font_style": "字体风格"
  }},
  "animation": {{
    "enter": "入场动画类型",
    "duration_ms": 500,
    "exit": "退场动画类型"
  }},
  "overlay_config": {{
    "position": "bottom-left|bottom-center|top-right|center",
    "opacity": 0.7,
    "z_index": 10
  }},
  "timing": {{
    "decor_start_sec": 0.3,
    "decor_duration_sec": 3.8
  }},
  "design_rationale_mapping": "design_rationale如何映射到所选structure"
}}

只返回上述 JSON，文案尽量短。"""

VERIFY_TEMPLATE = """\
请验证以下 spotlight 装饰方案的质量:

镜头信息:
{shot_json}

装饰方案:
{decoration_json}

检查维度:
1. 模板族是否匹配 design_rationale 的语义（因果/对比/流程/数据等）
2. 文本槽位是否从 vocal_script 提炼了面向观众的有效内容（非占位符/管线描述）
3. title 是否简短有力、直击要点（非截断半句话）
4. subtitle 是否提供了有意义的补充说明（非空）
5. eyebrow/label 是否为简短标签（来自 chapter_title）
6. 模板与同一视频其他镜头是否有差异（禁止重复 template_id）
7. overlay opacity 是否合理（信息密集画面建议 0.6-0.7）
8. 动画时长是否合理

输出(JSON):
{{
  "passed": true/false,
  "issues": [
    {{
      "dimension": "检查维度名",
      "severity": "critical/warning/info",
      "slot_id": "问题槽位(如有)",
      "description": "问题描述",
      "suggested_fix": "修复建议",
      "suggested_action": "revise_slots|swap_structure|adjust_overlay"
    }}
  ],
  "score": 0.0-1.0
}}"""

REPAIR_TEMPLATE = """\
请根据检查问题修复装饰方案。只修改有问题的部分，保留正确的部分。

镜头信息:
{shot_json}

当前装饰方案:
{decoration_json}

发现的问题:
{issues_json}

修复原则:
1. 优先做最小化修复：只修改导致问题的 slot，不改无关 slot
2. 槽位文本必须来自 vocal_script / chapter_title（面向观众）
3. 如果 title 是占位符/管线描述，从 vocal_script 提炼核心短句替换
4. 如果模板族不匹配，换一个更贴合 design_rationale 的 structure
5. 保留核心信息点，缩短文字时保留最重要的关键词

输出(严格JSON):
{{
  "repair_action": "revise_slots|swap_structure|adjust_overlay",
  "decoration": {{完整的修复后装饰方案}},
  "diff_summary": "修复了什么、为什么"
}}"""


# design_rationale keyword → structure family mapping
_RATIONALE_STRUCTURE_MAP: list[tuple[list[str], list[str]]] = [
    (["因果", "导致", "连锁", "传导"], ["cause_effect_chain", "vertical_flow"]),
    (["对比", "VS", "vs", "分屏", "利弊"], ["comparison_split", "pros_cons", "method_comparison"]),
    (["流程", "步骤", "先后", "顺序"], ["flow_steps", "vertical_flow", "timeline_step"]),
    (["层级", "分层", "三层"], ["layer_stack"]),
    (["循环", "反馈", "闭环"], ["cycle_loop"]),
    (["数字", "数据", "价格", "比例", "百分比"], ["stat_highlight", "metric_cards", "bar_chart"]),
    (["排名", "Top", "排序"], ["ranking_list"]),
    (["定义", "术语", "是什么"], ["definition_card"]),
    (["类比", "相当于", "像是"], ["analogy_card"]),
    (["三点", "三个", "几个要点"], ["three_points"]),
    (["清单", "回顾", "要点"], ["recap_checklist"]),
    (["引用", "金句", "名言"], ["quote_card"]),
    (["时间线", "演进", "历史"], ["timeline_step", "historical_moment"]),
    (["问题", "悬念", "为什么"], ["question_hook", "problem_hook"]),
    (["总结", "关键", "结论"], ["key_takeaway", "thesis_statement"]),
]
_REGISTRY_CACHE: dict[str, dict[str, Any]] = {}


def run(
    llm: LLMClient,
    shot: dict[str, Any],
    *,
    already_used_templates: list[str] | None = None,
    config: dict[str, Any] | None = None,
    max_repair_rounds: int = 2,
) -> dict[str, Any]:
    """Generate template decoration for a spotlight shot with check & self-repair loop."""
    if not shot.get("is_spotlight"):
        return {
            "shot_id": shot.get("shot_id", ""),
            "skipped": True,
            "reason": "非 spotlight 镜头，跳过装饰",
            "status": "skipped",
        }

    already_used = already_used_templates or []
    config = config or {}

    # Load policy/instructions hints from config paths
    template_policy_hint = _load_policy_hint(config)
    instructions_hint = _load_instructions_hint(config)
    registry = _load_structure_registry(config)
    registry_hint = _build_registry_hint(registry, shot, already_used)

    prompt = DECORATE_TEMPLATE.format(
        shot_id=shot.get("shot_id", ""),
        chapter_title=shot.get("chapter_title", ""),
        vocal_script=shot.get("vocal_script", ""),
        visual_direction=shot.get("visual_direction", ""),
        design_rationale=shot.get("design_rationale", ""),
        emphasis_target=shot.get("emphasis_target", ""),
        structure_label=shot.get("structure_label", ""),
        template_policy_hint=template_policy_hint,
        instructions_hint=instructions_hint,
        registry_hint=registry_hint,
        already_used=", ".join(already_used) if already_used else "无",
    )
    try:
        decoration = _as_dict(llm.chat_json(SYSTEM_PROMPT, prompt, temperature=0.3))
    except Exception as exc:
        logger.warning(
            "[decorate_shot] 镜头 %s LLM 装饰失败，将使用本地兜底: %s",
            shot.get("shot_id", ""),
            exc,
        )
        decoration = {}
    fallback_used = False
    if not decoration or not decoration.get("template_id"):
        decoration = _smart_fallback_decoration(shot, already_used, registry)
        fallback_used = True
    else:
        before_template_id = str(decoration.get("template_id", ""))
        decoration = _normalize_decoration(decoration, shot, already_used, registry)
        if before_template_id != str(decoration.get("template_id", "")):
            fallback_used = True

    # Ensure slots are properly filled from vocal_script and actual slot_schema.
    decoration = _ensure_slot_content(decoration, shot, registry)

    # Check & repair loop
    repair_history: list[dict[str, Any]] = []
    for repair_round in range(max_repair_rounds + 1):
        verify_result = _verify_decoration(llm, shot, decoration, registry)

        if verify_result.get("passed", False):
            return {
                "shot_id": shot.get("shot_id", ""),
                "decoration": decoration,
                "verify_result": verify_result,
                "passed": True,
                "fallback_used": fallback_used,
                "repair_history": repair_history,
                "repair_rounds": repair_round,
                "status": "success",
            }

        if repair_round >= max_repair_rounds:
            break

        # Auto-repair based on issues
        repair_result = _repair_decoration(llm, shot, decoration, verify_result, registry, already_used)
        repair_history.append({
            "round": repair_round,
            "issues": verify_result.get("issues", []),
            "repair_action": repair_result.get("repair_action", ""),
            "diff_summary": repair_result.get("diff_summary", ""),
        })

        new_decoration = repair_result.get("decoration", {})
        if new_decoration and isinstance(new_decoration, dict) and new_decoration.get("template_id"):
            decoration = _normalize_decoration(new_decoration, shot, already_used, registry)
            decoration = _ensure_slot_content(decoration, shot, registry)
        else:
            # Deterministic repair fallback
            decoration = _deterministic_repair(decoration, verify_result, shot, already_used, registry)

    return {
        "shot_id": shot.get("shot_id", ""),
        "decoration": decoration,
        "verify_result": verify_result,
        "passed": False,
        "fallback_used": fallback_used,
        "repair_history": repair_history,
        "repair_rounds": max_repair_rounds,
        "status": "needs_repair",
    }


# ---------------------------------------------------------------------------
# Batch decoration: 1 global LLM call for all spotlight shots (fake mode)
# ---------------------------------------------------------------------------

BATCH_DECORATE_TEMPLATE = """\
为以下全部 spotlight 镜头批量设计模板装饰方案。

## Gold Template 候选结构
{registry_hint}

## 模板选择策略
{template_policy_hint}

## 装饰指引
{instructions_hint}

## 待装饰镜头列表
{shots_json}

## 输出要求
对每个镜头输出一个装饰方案。严格按以下 JSON 格式:
{{
  "decorations": {{
    "<shot_id>": {{
      "template_id": "从候选结构中选一个",
      "template_type": "lower_third|overlay|motion|caption_emphasis|title_card|split_screen",
      "slots": {{
        "<slot_name>": "内容文本（严格不超过 slot_schema 中 max_chars 限制）",
        ...
      }},
      "animation": {{"enter": "...", "duration_ms": 500, "exit": "..."}},
      "overlay_config": {{"position": "...", "opacity": 0.7}},
      "timing": {{"decor_start_sec": 0.3, "decor_duration_sec": ...}},
      "design_rationale_mapping": "简短说明为什么选这个模板"
    }},
    ...
  }},
  "template_diversity_check": {{
    "duplicates": ["如果有重复的 template_id 列在这里"],
    "all_unique": true
  }},
  "global_summary": "一句话总结模板分配策略"
}}

## 硬性规则
1. 只能从候选 structure_id 中选择，禁止自造 template_id
2. 不同镜头尽量选不同模板（检查 template_diversity_check）
3. 每个 slot 的文本长度严格不超过对应的 max_chars
4. 不要使用管线内部术语（如 "开篇定调"、"设计原理" 等）
5. 文案面向观众，从 vocal_script / chapter_title 提炼
6. 只输出上述 JSON，不输出解释或 Markdown"""

TEXT_ONLY_REPAIR_TEMPLATE = """\
请根据检查问题修复装饰方案。只修改有问题的部分，保留正确的部分。
注意：当前为纯文本修复模式，不需要重新渲染验证。

镜头信息:
{shot_json}

当前装饰方案:
{decoration_json}

发现的问题:
{issues_json}

槽位 schema:
{slot_schema_json}

修复原则:
1. 优先做最小化修复：只修改导致问题的 slot，不改无关 slot
2. 槽位文本必须来自 vocal_script / chapter_title（面向观众）
3. 如果 title 是占位符/管线描述，从 vocal_script 提炼核心短句替换
4. 如果模板族不匹配，换一个更贴合 design_rationale 的 structure
5. 保留核心信息点，缩短文字时保留最重要的关键词
6. 所有 slot 文本长度不超过 max_chars

输出(严格JSON):
{{
  "repair_action": "revise_slots|swap_structure|adjust_overlay",
  "decoration": {{完整的修复后装饰方案}},
  "diff_summary": "修复了什么、为什么"
}}"""


def batch_run(
    llm: LLMClient,
    shots: list[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
    max_repair_rounds: int = 2,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    """一次性为全部 spotlight shots 生成装饰方案（批量模式）。

    1 次全局 LLM 调用选模板+填槽位 → 本地 rule-based 验证 → 纯文本 LLM repair（仅对失败的 shot）

    Returns:
        {
            "results": {"shot_id_1": {...}, "shot_id_2": {...}, ...},
            "summary": "批量装饰 N 个 spotlight 镜头",
            "status": "success",
        }
    """
    config = config or {}
    template_policy_hint = _load_policy_hint(config)
    instructions_hint = _load_instructions_hint(config)
    registry = _load_structure_registry(config)

    if not shots:
        return {"results": {}, "summary": "无 spotlight 镜头", "status": "success"}

    progress_total = len(shots) + 2

    def _progress(sub_tool: str, current: int, label: str) -> None:
        if progress_callback is not None:
            progress_callback(sub_tool, current, progress_total, label)

    # Step 1: 全局 decoration plan（1 次 LLM，一次性选模板 + 填槽位）
    _progress("全局模板规划", 0, f"一次请求规划 {len(shots)} 个 spotlight 镜头")
    global_result = _global_decoration_plan(
        llm, shots, template_policy_hint, instructions_hint, registry, config
    )
    _progress("全局模板规划", 1, "模板选择与槽位初稿已返回")

    # Step 2: 快速本地验证 + 定点修复
    results: dict[str, dict[str, Any]] = {}
    already_used: list[str] = []

    for shot_index, shot in enumerate(shots):
        sid = shot.get("shot_id", "")
        dec = global_result.get("decorations", {}).get(sid, {})
        if not dec or not isinstance(dec, dict) or not dec.get("template_id"):
            dec = _smart_fallback_decoration(shot, already_used, registry)

        # Normalize + ensure slot content with max_chars limits
        dec = _normalize_decoration(dec, shot, already_used, registry)
        dec = _ensure_slot_content_with_limits(dec, shot, registry)

        template_id = str(dec.get("template_id", ""))
        if template_id and template_id not in already_used:
            already_used.append(template_id)

        # Local verify
        verify_result = _local_verify(shot, dec, registry)
        passed = bool(verify_result.get("passed", False))
        repair_rounds = 0

        # Text-only repair loop (only for failed shots)
        if not passed and max_repair_rounds > 0:
            for repair_round in range(max_repair_rounds):
                issues = verify_result.get("issues", [])
                if not issues:
                    break
                _progress(
                    "定点文本修复",
                    shot_index + 1,
                    f"镜头 {sid} 第 {repair_round + 1}/{max_repair_rounds} 轮纯文本修复",
                )
                repaired_dec = _text_only_repair(llm, shot, dec, issues, registry)
                if repaired_dec and repaired_dec.get("template_id"):
                    dec = _normalize_decoration(repaired_dec, shot, already_used, registry)
                    dec = _ensure_slot_content_with_limits(dec, shot, registry)
                verify_result = _local_verify(shot, dec, registry)
                repair_rounds = repair_round + 1
                if verify_result.get("passed", False):
                    passed = True
                    break

        results[sid] = {
            "shot_id": sid,
            "decoration": dec,
            "verify_result": verify_result,
            "passed": passed,
            "repair_rounds": repair_rounds,
            "status": "success" if passed else "needs_repair",
        }
        _progress(
            "本地校验与修复",
            shot_index + 2,
            f"镜头 {sid} {'通过' if passed else '仍需复核'}",
        )

    passed_count = sum(1 for r in results.values() if r.get("passed"))
    _progress("批量装饰完成", progress_total, f"{passed_count}/{len(shots)} 个镜头通过")
    return {
        "results": results,
        "summary": f"批量装饰 {len(shots)} 个 spotlight 镜头，通过 {passed_count} 个",
        "status": "success",
    }


def _global_decoration_plan(
    llm: LLMClient,
    shots: list[dict[str, Any]],
    template_policy_hint: str,
    instructions_hint: str,
    registry: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """1 次 LLM 调用为全部 shots 生成 decoration plan。"""
    # Build a compact shot list for the prompt
    shots_compact = []
    for shot in shots:
        shots_compact.append({
            "shot_id": shot.get("shot_id", ""),
            "chapter_title": shot.get("chapter_title", ""),
            "vocal_script": str(shot.get("vocal_script", ""))[:200],
            "design_rationale": shot.get("design_rationale", ""),
            "structure_label": shot.get("structure_label", ""),
            "emphasis_target": shot.get("emphasis_target", ""),
        })

    # Build registry hint with top candidates per shot
    registry_rows = []
    for sid_key, item in list(registry.items())[:60]:
        schema = item.get("slot_schema", {}) if isinstance(item.get("slot_schema"), dict) else {}
        slots_list = list((schema.get("slots") or {}).keys())[:8]
        registry_rows.append({
            "id": sid_key,
            "family": item.get("family", ""),
            "use_for": item.get("use_for", []),
            "slots": slots_list,
        })
    registry_hint = json.dumps(registry_rows, ensure_ascii=False, indent=2)

    prompt = BATCH_DECORATE_TEMPLATE.format(
        registry_hint=registry_hint,
        template_policy_hint=template_policy_hint or "（无）",
        instructions_hint=instructions_hint or "（无）",
        shots_json=json.dumps(shots_compact, ensure_ascii=False, indent=2),
    )

    try:
        result = _as_dict(llm.chat_json(SYSTEM_PROMPT, prompt, temperature=0.3))
    except Exception:
        result = {}

    if not result or not isinstance(result.get("decorations"), dict):
        return {"decorations": {}, "global_summary": "全局 plan 失败，使用 fallback", "template_diversity_check": {}}
    return result


def _ensure_slot_content_with_limits(
    decoration: dict[str, Any],
    shot: dict[str, Any],
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """确保所有槽位内容有效且在字符限制内。

    在 _ensure_slot_content 基础上增加：读取 template slot_schema.json 的 max_chars，
    对 LLM 输出的每个 slot 文本做截断 + 校验。
    """
    decoration = _ensure_slot_content(decoration, shot, registry)

    template_id = str(decoration.get("template_id", ""))
    slots = decoration.get("slots", {})
    if not isinstance(slots, dict):
        slots = {}
    schema = _slot_schema_for(decoration, registry)
    if not schema or not isinstance(schema.get("slots"), dict):
        return decoration

    changed = False
    for slot_name, slot_spec in schema["slots"].items():
        if not isinstance(slot_spec, dict):
            continue
        slot_type = str(slot_spec.get("type", "")).lower()
        if slot_type == "json":
            continue

        max_chars = _as_int(slot_spec.get("max_chars"), 0)
        current_val = str(slots.get(slot_name, ""))

        if max_chars > 0 and len(current_val) > max_chars:
            slots[slot_name] = _trim_text(current_val, max_chars)
            changed = True

    if changed:
        decoration["slots"] = slots
    return decoration


def _text_only_repair(
    llm: LLMClient,
    shot: dict[str, Any],
    decoration: dict[str, Any],
    issues: list[dict],
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """纯文本修复：不渲染，只让 LLM 根据 check issues 修改 decoration JSON。"""
    repair_prompt = TEXT_ONLY_REPAIR_TEMPLATE.format(
        shot_json=json.dumps(shot, ensure_ascii=False, indent=2),
        decoration_json=json.dumps(decoration, ensure_ascii=False, indent=2),
        issues_json=json.dumps(issues, ensure_ascii=False, indent=2),
        slot_schema_json=json.dumps(
            _slot_schema_for(decoration, registry) or {}, ensure_ascii=False, indent=2
        ),
    )
    try:
        result = _as_dict(llm.chat_json(SYSTEM_PROMPT, repair_prompt, temperature=0.2))
    except Exception:
        result = {}
    if result and isinstance(result.get("decoration"), dict):
        return result["decoration"]
    return decoration


def _verify_decoration(
    llm: LLMClient,
    shot: dict[str, Any],
    decoration: dict[str, Any],
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify decoration quality. Returns passed=True/False with issues."""
    verify_prompt = VERIFY_TEMPLATE.format(
        shot_json=json.dumps(shot, ensure_ascii=False, indent=2),
        decoration_json=json.dumps(decoration, ensure_ascii=False, indent=2),
    )
    try:
        verify_result = _as_dict(llm.chat_json(SYSTEM_PROMPT, verify_prompt, temperature=0.2))
    except Exception:
        verify_result = {}
    if not verify_result:
        # Local deterministic check
        verify_result = _local_verify(shot, decoration, registry)
    else:
        llm_issues = _normalize_issues(verify_result.get("issues", []))
        local_result = _local_verify(shot, decoration, registry)
        merged_issues = _merge_issues(llm_issues, local_result.get("issues", []))
        llm_score = _as_float(verify_result.get("score"), None)
        local_score = _as_float(local_result.get("score"), 0.0)
        verify_result["issues"] = merged_issues
        verify_result["score"] = round(min(llm_score if llm_score is not None else 1.0, local_score), 2)

    # Enforce: if there are critical/warning issues, passed must be False
    blocking = [
        i for i in verify_result.get("issues", [])
        if isinstance(i, dict) and str(i.get("severity", "")).lower() in {"critical", "warning"}
    ]
    if blocking:
        verify_result["passed"] = False
    if _as_float(verify_result.get("score"), 1.0) < 0.75:
        verify_result["passed"] = False
    verify_result["status"] = "passed" if verify_result.get("passed", False) else "needs_repair"

    return verify_result


def _repair_decoration(
    llm: LLMClient,
    shot: dict[str, Any],
    decoration: dict[str, Any],
    verify_result: dict[str, Any],
    registry: dict[str, Any] | None = None,
    already_used: list[str] | None = None,
) -> dict[str, Any]:
    """Repair decoration based on verification issues."""
    repair_prompt = REPAIR_TEMPLATE.format(
        shot_json=json.dumps(shot, ensure_ascii=False, indent=2),
        decoration_json=json.dumps(decoration, ensure_ascii=False, indent=2),
        issues_json=json.dumps(verify_result.get("issues", []), ensure_ascii=False, indent=2),
    )
    try:
        result = _as_dict(llm.chat_json(SYSTEM_PROMPT, repair_prompt, temperature=0.2))
    except Exception:
        result = {}
    if not result or not isinstance(result.get("decoration"), dict):
        return {
            "repair_action": "none",
            "decoration": {},
            "diff_summary": "LLM修复输出不可用",
        }
    result["decoration"] = _normalize_decoration(result["decoration"], shot, already_used or [], registry or {})
    return result


def _deterministic_repair(
    decoration: dict[str, Any],
    verify_result: dict[str, Any],
    shot: dict[str, Any],
    already_used: list[str],
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic repair fallback when LLM repair fails."""
    issues = _normalize_issues(verify_result.get("issues", []))
    new_dec = dict(decoration)
    slots = dict(new_dec.get("slots", {}))

    for issue in issues:
        if not isinstance(issue, dict):
            continue
        action = issue.get("suggested_action", "")
        slot_id = issue.get("slot_id", "")
        severity = str(issue.get("severity", "")).lower()

        if severity not in ("critical", "warning"):
            continue

        if action == "swap_structure" or "模板" in issue.get("description", ""):
            new_dec = _smart_fallback_decoration(shot, already_used, registry)
            slots = dict(new_dec.get("slots", {}))
            break

        if slot_id and slot_id in slots:
            slots[slot_id] = _extract_slot_content(slot_id, shot)

    # Fix empty/placeholder slots
    for sid in ("title", "subtitle", "eyebrow"):
        val = str(slots.get(sid, "")).strip()
        if not val or _is_placeholder(val) or _is_pipeline_jargon(val):
            slots[sid] = _extract_slot_content(sid, shot)

    new_dec["slots"] = slots

    # Fix overlay opacity
    overlay = new_dec.get("overlay_config", {})
    if isinstance(overlay, dict) and float(overlay.get("opacity", 0.7)) > 0.8:
        overlay["opacity"] = 0.65
        new_dec["overlay_config"] = overlay

    return _ensure_slot_content(new_dec, shot, registry)


def _ensure_slot_content(
    decoration: dict[str, Any],
    shot: dict[str, Any],
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ensure all slots have meaningful content from vocal_script/chapter_title."""
    slots = decoration.get("slots", {})
    if not isinstance(slots, dict):
        slots = {}

    schema = _slot_schema_for(decoration, registry)
    if schema:
        schema_slots = schema.get("slots", {}) if isinstance(schema.get("slots"), dict) else {}
        data_slots = set(schema.get("data_slots") or [])
        filled: dict[str, Any] = {}
        for index, (sid, spec) in enumerate(schema_slots.items()):
            if not isinstance(spec, dict):
                spec = {}
            is_data = sid in data_slots or str(spec.get("type", "")).lower() == "json"
            val = slots.get(sid)
            if is_data:
                if not _valid_data_slot_value(val):
                    val = _extract_data_slot_content(sid, spec, shot, index)
                filled[sid] = val
                continue

            text = str(val or "").strip()
            if not text or _is_placeholder(text) or _is_pipeline_jargon(text) or _is_garbled(text):
                text = _extract_schema_slot_content(sid, spec, shot, index)
            max_chars = _as_int(spec.get("max_chars"), 0)
            if max_chars > 0:
                text = _trim_text(text, max_chars)
            filled[sid] = text
        slots = filled
    else:
        for sid in ("title", "subtitle", "eyebrow"):
            val = str(slots.get(sid, "")).strip()
            if not val or _is_placeholder(val) or _is_pipeline_jargon(val):
                slots[sid] = _extract_slot_content(sid, shot)

    decoration["slots"] = slots
    overlay = decoration.get("overlay_config", {})
    if not isinstance(overlay, dict):
        overlay = {}
    overlay.setdefault("position", "bottom-left")
    try:
        opacity = float(overlay.get("opacity", 0.7))
    except (TypeError, ValueError):
        opacity = 0.7
    overlay["opacity"] = min(max(opacity, 0.45), 0.75)
    overlay.setdefault("z_index", 10)
    decoration["overlay_config"] = overlay
    return decoration


def _extract_slot_content(slot_id: str, shot: dict[str, Any]) -> str:
    """Extract meaningful content for a slot from shot context."""
    vocal_script = shot.get("vocal_script", "")
    chapter_title = shot.get("chapter_title", "")
    visual_direction = shot.get("visual_direction", "")

    if slot_id in ("title", "headline", "thesis"):
        return _extract_core_statement(vocal_script, chapter_title)
    elif slot_id in ("subtitle", "caption", "explain"):
        return _extract_explanation(vocal_script, chapter_title)
    elif slot_id in ("eyebrow", "label", "badge", "tag"):
        return _extract_short_label(chapter_title, vocal_script)
    return _extract_core_statement(vocal_script, chapter_title)


def _extract_schema_slot_content(
    slot_id: str,
    spec: dict[str, Any],
    shot: dict[str, Any],
    index: int = 0,
) -> str:
    """Fill a real Gold Template text slot from shot context."""
    label = str(spec.get("label") or "")
    low = f"{slot_id} {label}".lower()
    clauses = _content_units(shot)
    item_index = max(0, _slot_number(slot_id) - 1)

    if any(k in low for k in ("eyebrow", "label", "badge", "tag", "顶部标签")):
        return _extract_short_label(str(shot.get("chapter_title", "")), str(shot.get("vocal_script", "")))
    if any(k in low for k in ("headline", "title", "thesis", "question", "主标题", "标题", "问题")):
        if any(k in low for k in ("point", "step", "takeaway", "要点", "阶段")) and clauses:
            return _make_short_phrase(clauses[item_index % len(clauses)], max_chars=10)
        return _extract_core_statement(str(shot.get("vocal_script", "")), str(shot.get("chapter_title", "")))
    if any(k in low for k in ("desc", "subtitle", "caption", "explain", "hint", "text", "说明", "描述", "补充", "答案", "正文")):
        if clauses:
            return clauses[item_index % len(clauses)]
        return _extract_explanation(str(shot.get("vocal_script", "")), str(shot.get("chapter_title", "")))
    if any(k in low for k in ("takeaway", "point", "step", "要点", "阶段")) and clauses:
        return clauses[item_index % len(clauses)]
    if clauses:
        return clauses[index % len(clauses)]
    return _extract_slot_content(slot_id, shot)


def _extract_data_slot_content(
    slot_id: str,
    spec: dict[str, Any],
    shot: dict[str, Any],
    index: int = 0,
) -> Any:
    """Fill JSON/data slots while preserving the slot's expected shape."""
    default = spec.get("default")
    clauses = _content_units(shot)
    if not clauses:
        clauses = [_extract_core_statement(str(shot.get("vocal_script", "")), str(shot.get("chapter_title", "")))]
    clauses = [c for c in clauses if c]

    if isinstance(default, list):
        max_items = _as_int(spec.get("max_items"), len(default) or 3) or (len(default) or 3)
        min_items = _as_int(spec.get("min_items"), min(2, max_items)) or min(2, max_items)
        count = max(1, min(max_items, max(min_items, min(len(clauses), len(default) or max_items))))
        if not default or isinstance(default[0], str):
            return [_make_short_phrase(clauses[i % len(clauses)], max_chars=12) for i in range(count)]
        if isinstance(default[0], dict):
            rows = []
            for i in range(count):
                sample = default[min(i, len(default) - 1)]
                rows.append(_fill_data_dict(sample, clauses, i))
            return rows
        return clauses[:count]
    if isinstance(default, dict):
        return _fill_data_dict(default, clauses, index)
    return [_make_short_phrase(c, max_chars=12) for c in clauses[:3]]


def _fill_data_dict(sample: dict[str, Any], clauses: list[str], index: int) -> dict[str, Any]:
    row: dict[str, Any] = {}
    numbers = _extract_numbers(" ".join(clauses))
    for key, value in sample.items():
        low = str(key).lower()
        clause = clauses[index % len(clauses)] if clauses else ""
        if isinstance(value, list):
            row[key] = [_make_short_phrase(c, max_chars=10) for c in clauses[: max(2, min(3, len(clauses)))]]
        elif isinstance(value, (int, float)):
            row[key] = _number_to_float(numbers[0], default=float(value)) if numbers else value
        elif any(k in low for k in ("dim", "label", "name", "title", "阶段", "维度")):
            row[key] = _make_short_phrase(clause, max_chars=8)
        elif any(k in low for k in ("value", "metric", "number", "score", "数值")):
            row[key] = numbers[index % len(numbers)] if numbers else _make_short_phrase(clause, max_chars=8)
        else:
            row[key] = _trim_text(clause, 20)
    return row


def _extract_core_statement(speech: str, chapter_title: str) -> str:
    """Extract the core statement from vocal_script for title-type slots."""
    if speech and 4 <= len(speech) <= 26:
        return speech
    if speech and len(speech) > 26:
        clause = _extract_first_clause(speech)
        if clause and len(clause) >= 4:
            return clause
    if chapter_title:
        return chapter_title
    return speech[:26] if speech else ""


def _extract_explanation(speech: str, chapter_title: str) -> str:
    """Extract explanation text for subtitle-type slots."""
    if speech:
        parts = re.split(r"[，。！？；]", speech)
        parts = [p.strip() for p in parts if p.strip() and len(p.strip()) >= 3]
        if len(parts) >= 2:
            return parts[1][:30] if len(parts[1]) > 30 else parts[1]
        if parts:
            return parts[0]
    return chapter_title or ""


def _extract_short_label(chapter_title: str, speech: str) -> str:
    """Extract short label for eyebrow/tag-type slots."""
    if chapter_title and 2 <= len(chapter_title) <= 14:
        return chapter_title
    if chapter_title and len(chapter_title) > 14:
        for sep in ("，", "、", "：", "——"):
            idx = chapter_title.find(sep)
            if 2 <= idx <= 14:
                return chapter_title[:idx]
        return chapter_title[:14]
    if speech:
        clause = _extract_first_clause(speech)
        if clause and len(clause) <= 14:
            return clause
    return ""


def _content_units(shot: dict[str, Any]) -> list[str]:
    """Return audience-facing content chunks, never internal planning labels."""
    raw_items: list[str] = []
    speech = str(shot.get("vocal_script", "") or "")
    raw_items.extend(re.split(r"[，。！？；\n]", speech))
    for key in ("key_points",):
        val = shot.get(key)
        if isinstance(val, list):
            raw_items.extend(str(x) for x in val)
    for key in ("chapter_title", "intent"):
        val = str(shot.get(key, "") or "")
        if val:
            raw_items.append(val)
    out: list[str] = []
    for item in raw_items:
        text = _trim_text(str(item).strip(), 30)
        if not text or _is_pipeline_jargon(text) or _is_placeholder(text):
            continue
        if text not in out:
            out.append(text)
    return out


def _make_short_phrase(text: str, max_chars: int = 12) -> str:
    text = str(text or "").strip()
    for sep in ("，", "。", "；", "：", "、"):
        idx = text.find(sep)
        if 2 <= idx <= max_chars:
            text = text[:idx]
            break
    return _trim_text(text, max_chars)


def _trim_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip("，,、；;：: ")


def _slot_number(slot_id: str) -> int:
    match = re.search(r"(\d+)", str(slot_id))
    if not match:
        return 1
    try:
        return int(match.group(1))
    except ValueError:
        return 1


def _extract_numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)?%?", str(text or ""))


def _number_to_float(value: Any, default: float = 0.0) -> float:
    text = str(value or "").strip()
    if not text:
        return default
    text = text.rstrip("%％").replace(",", "")
    try:
        return float(text)
    except ValueError:
        return default


def _extract_first_clause(text: str) -> str:
    """Extract first meaningful clause from text."""
    if not text:
        return ""
    for sep in ("，", "。", "！", "？", "；", "、", "："):
        idx = text.find(sep)
        if 4 <= idx <= 30:
            return text[:idx]
    return text[:30] if len(text) > 30 else text


def _local_verify(
    shot: dict[str, Any],
    decoration: dict[str, Any],
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Local deterministic verification of decoration quality."""
    issues: list[dict[str, Any]] = []
    slots = decoration.get("slots", {})
    schema = _slot_schema_for(decoration, registry)
    template_id = str(decoration.get("template_id", "")).strip()

    if not template_id:
        issues.append({
            "dimension": "模板选择",
            "severity": "critical",
            "slot_id": "",
            "description": "template_id 为空，无法映射到 Gold Template",
            "suggested_fix": "从 registry 中选择匹配 design_rationale 的 structure",
            "suggested_action": "swap_structure",
        })
    elif "fallback" in template_id.lower():
        issues.append({
            "dimension": "模板选择",
            "severity": "warning",
            "slot_id": "",
            "description": f"使用 fallback 模板 '{template_id}'，视觉冲击力不足",
            "suggested_fix": "根据 design_rationale 选择更匹配的模板",
            "suggested_action": "swap_structure",
        })
    elif registry and template_id not in registry:
        issues.append({
            "dimension": "模板选择",
            "severity": "critical",
            "slot_id": "",
            "description": f"template_id '{template_id}' 不在 Gold Template registry 中",
            "suggested_fix": "切换到合法 Gold Template structure_id",
            "suggested_action": "swap_structure",
        })

    if schema:
        schema_slots = schema.get("slots", {}) if isinstance(schema.get("slots"), dict) else {}
        data_slots = set(schema.get("data_slots") or [])
        for sid, spec in schema_slots.items():
            if not isinstance(spec, dict):
                spec = {}
            val = slots.get(sid) if isinstance(slots, dict) else None
            is_data = sid in data_slots or str(spec.get("type", "")).lower() == "json"
            if is_data:
                if not _valid_data_slot_value(val):
                    issues.append({
                        "dimension": "槽位内容",
                        "severity": "critical",
                        "slot_id": sid,
                        "description": f"数据槽位 {sid} 为空或结构无效",
                        "suggested_fix": "根据 vocal_script/key_points 重填数据槽位",
                        "suggested_action": "revise_slots",
                    })
                continue
            text = str(val or "").strip()
            if not text or _is_placeholder(text):
                issues.append({
                    "dimension": "槽位内容",
                    "severity": "critical",
                    "slot_id": sid,
                    "description": f"槽位 {sid} 为空或占位符",
                    "suggested_fix": "从 vocal_script 提炼面向观众的短文案",
                    "suggested_action": "revise_slots",
                })
            elif _is_pipeline_jargon(text):
                issues.append({
                    "dimension": "槽位内容",
                    "severity": "critical",
                    "slot_id": sid,
                    "description": f"槽位 {sid} 暴露内部逻辑: '{text}'",
                    "suggested_fix": "替换为观众可理解的内容摘要",
                    "suggested_action": "revise_slots",
                })
            elif _is_garbled(text):
                issues.append({
                    "dimension": "槽位内容",
                    "severity": "warning",
                    "slot_id": sid,
                    "description": f"槽位 {sid} 文本疑似异常: '{text}'",
                    "suggested_fix": "用 vocal_script 中的真实短句替换",
                    "suggested_action": "revise_slots",
                })
            max_chars = _as_int(spec.get("max_chars"), 0)
            if max_chars and len(text) > max_chars:
                issues.append({
                    "dimension": "槽位长度",
                    "severity": "warning",
                    "slot_id": sid,
                    "description": f"槽位 {sid} 超过 max_chars={max_chars}",
                    "suggested_fix": "压缩槽位文本但保留核心信息",
                    "suggested_action": "revise_slots",
                })
    else:
        if template_id:
            issues.append({
                "dimension": "模板选择",
                "severity": "warning",
                "slot_id": "",
                "description": f"模板 '{template_id}' 缺少 slot_schema，无法确认真实槽位覆盖",
                "suggested_fix": "切换到带 slot_schema 的 Gold Template structure",
                "suggested_action": "swap_structure",
            })

    # Check title slot
    title = str(slots.get("title", "")).strip()
    if not schema and (not title or _is_placeholder(title)):
        issues.append({
            "dimension": "槽位内容",
            "severity": "critical",
            "slot_id": "title",
            "description": f"标题槽位为空或占位符: '{title}'",
            "suggested_fix": "从 vocal_script 提炼核心短句",
            "suggested_action": "revise_slots",
        })
    elif not schema and _is_pipeline_jargon(title):
        issues.append({
            "dimension": "槽位内容",
            "severity": "critical",
            "slot_id": "title",
            "description": f"标题暴露内部逻辑: '{title}'，应基于 vocal_script 提炼语义",
            "suggested_fix": "从 vocal_script 提取面向观众的内容",
            "suggested_action": "revise_slots",
        })

    # Check subtitle
    subtitle = str(slots.get("subtitle", "")).strip()
    if not schema and not subtitle:
        issues.append({
            "dimension": "槽位内容",
            "severity": "warning",
            "slot_id": "subtitle",
            "description": "副标题为空，应利用副标题补充关键解说摘要",
            "suggested_fix": "从 vocal_script 提取补充说明",
            "suggested_action": "revise_slots",
        })

    # Check if fallback template is used for spotlight
    # Check overlay opacity
    overlay = decoration.get("overlay_config", {})
    opacity = float(overlay.get("opacity", 0.7)) if isinstance(overlay, dict) else 0.7
    if opacity > 0.8:
        issues.append({
            "dimension": "叠加层配置",
            "severity": "warning",
            "slot_id": "",
            "description": f"遮罩透明度 {opacity} 偏高，可能遮挡画面细节",
            "suggested_fix": "建议降至 0.6-0.7",
            "suggested_action": "adjust_overlay",
        })

    critical_count = sum(1 for i in issues if i["severity"] == "critical")
    warning_count = sum(1 for i in issues if i["severity"] == "warning")
    blocking = [i for i in issues if i["severity"] in ("critical", "warning")]
    score = max(0.0, 1.0 - critical_count * 0.25 - warning_count * 0.12)
    return {
        "passed": len(blocking) == 0,
        "issues": issues,
        "score": round(score, 2),
    }


def _smart_fallback_decoration(
    shot: dict[str, Any],
    already_used: list[str] | None = None,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Smart fallback: pick structure based on design_rationale keywords."""
    already_used = already_used or []
    template_id = _choose_structure_id(shot, already_used, registry or {})

    decoration = {
        "template_id": template_id,
        "template_type": _structure_to_type(template_id, registry),
        "slots": {},
        "animation": {
            "enter": "slide_in",
            "duration_ms": 500,
            "exit": "fade_out",
        },
        "overlay_config": {
            "position": "bottom-left",
            "opacity": 0.65,
            "z_index": 10,
        },
        "timing": {
            "decor_start_sec": 0.3,
            "decor_duration_sec": 3.8,
        },
        "design_rationale_mapping": f"基于 design_rationale 语义匹配选择 {template_id}",
    }
    return _ensure_slot_content(decoration, shot, registry)


def _structure_to_type(structure_id: str, registry: dict[str, Any] | None = None) -> str:
    """Map structure_id to template_type."""
    item = (registry or {}).get(structure_id, {})
    family = str(item.get("family", "")).lower()
    if family == "dataviz":
        return "overlay"
    if family == "narrative":
        return "motion"
    type_map = {
        "lower_third": ["definition_card", "key_takeaway", "thesis_statement", "lower_third"],
        "overlay": ["stat_highlight", "metric_cards", "quote_card"],
        "motion": ["cause_effect_chain", "flow_steps", "cycle_loop", "timeline_step", "three_points_stack"],
        "caption_emphasis": ["focus_zoom", "analogy_card"],
        "title_card": ["hook_title_intro", "chapter_transition", "question_hook_split"],
        "split_screen": ["comparison_split", "pros_cons", "method_comparison"],
    }
    for ttype, structures in type_map.items():
        if structure_id in structures:
            return ttype
    return "overlay"


def _normalize_decoration(
    decoration: dict[str, Any],
    shot: dict[str, Any],
    already_used: list[str],
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Coerce LLM output to a valid Gold Template decoration contract."""
    registry = registry or {}
    out = dict(decoration or {})
    tid = str(out.get("template_id", "")).strip()
    if not tid or "fallback" in tid.lower() or (registry and tid not in registry):
        tid = _choose_structure_id(shot, already_used, registry)
        out["template_id"] = tid
    out["template_type"] = str(out.get("template_type") or _structure_to_type(tid, registry))
    if not isinstance(out.get("animation"), dict):
        out["animation"] = {"enter": "slide_in", "duration_ms": 500, "exit": "fade_out"}
    if not isinstance(out.get("timing"), dict):
        out["timing"] = {"decor_start_sec": 0.3, "decor_duration_sec": 3.8}
    out.setdefault("design_rationale_mapping", f"基于 registry 选择 {tid}")
    return _ensure_slot_content(out, shot, registry)


def _choose_structure_id(
    shot: dict[str, Any],
    already_used: list[str],
    registry: dict[str, Any],
) -> str:
    ranked = _rank_structure_candidates(registry, shot)
    for sid, _score in ranked:
        if sid not in already_used:
            return sid
    if ranked:
        return ranked[0][0]
    for fallback in ("key_takeaway", "definition_card", "question_hook_split", "timeline_step"):
        if fallback not in already_used:
            return fallback
    return "key_takeaway"


def _rank_structure_candidates(registry: dict[str, Any], shot: dict[str, Any]) -> list[tuple[str, float]]:
    if not registry:
        return []
    haystack = " ".join(
        str(shot.get(k, "") or "")
        for k in ("design_rationale", "emphasis_target", "intent", "structure_label", "vocal_script", "chapter_title")
    )
    rows: list[tuple[str, float]] = []
    for sid, item in registry.items():
        score = 0.0
        item_text = " ".join([
            sid,
            str(item.get("family", "")),
            " ".join(str(x) for x in (item.get("use_for") or [])),
            str(item.get("display_name", "")),
        ])
        for keywords, structures in _RATIONALE_STRUCTURE_MAP:
            if any(kw in haystack for kw in keywords):
                if sid in structures:
                    score += 4.0
                if any(kw in item_text for kw in keywords):
                    score += 2.0
        for token in re.split(r"[\s,，。！？；、/|]+", haystack):
            token = token.strip()
            if len(token) >= 2 and token in item_text:
                score += 0.25
        if str(item.get("spotlight_level", "")).lower() == "high":
            score += 0.35
        if item.get("slot_schema"):
            score += 0.5
        rows.append((sid, score))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def _build_registry_hint(
    registry: dict[str, Any],
    shot: dict[str, Any],
    already_used: list[str],
    limit: int = 18,
) -> str:
    if not registry:
        return "registry 未加载成功；如无可用候选，将使用确定性 fallback。"
    rows = []
    for sid, _score in _rank_structure_candidates(registry, shot)[:limit]:
        item = registry.get(sid, {})
        slots = list(((item.get("slot_schema") or {}).get("slots") or {}).keys())
        rows.append({
            "id": sid,
            "family": item.get("family", ""),
            "use_for": item.get("use_for", []),
            "slots": slots[:8],
            "already_used": sid in already_used,
        })
    return json.dumps(rows, ensure_ascii=False, indent=2)


def _load_structure_registry(config: dict[str, Any]) -> dict[str, Any]:
    root = str((config.get("paths", {}) or {}).get("gold_template_root", "") or "").strip()
    if not root:
        return {}
    root_path = Path(root).expanduser().resolve()
    cache_key = str(root_path)
    if cache_key in _REGISTRY_CACHE:
        return _REGISTRY_CACHE[cache_key]

    gen_dir = root_path / "generated" / "structures"
    registry_json = _read_json(root_path / "registry" / "structures.json")
    registry_index = {
        str(item.get("id")): item
        for item in (registry_json.get("structures") or [])
        if isinstance(item, dict) and item.get("id")
    }
    out: dict[str, Any] = {}
    if gen_dir.exists():
        for struct_dir in sorted(p for p in gen_dir.iterdir() if p.is_dir()):
            sid = struct_dir.name
            card = _read_json(struct_dir / "structure_card.json")
            schema = _read_json(struct_dir / "slot_schema.json")
            merged = {**registry_index.get(sid, {}), **card}
            merged["id"] = sid
            merged["slot_schema"] = schema if isinstance(schema, dict) else {}
            if "data_slots" not in merged and isinstance(schema, dict):
                merged["data_slots"] = schema.get("data_slots", [])
            out[sid] = merged
    _REGISTRY_CACHE[cache_key] = out
    return out


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
 

def _slot_schema_for(
    decoration: dict[str, Any],
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tid = str(decoration.get("template_id", "") or "")
    item = (registry or {}).get(tid, {})
    schema = item.get("slot_schema")
    return schema if isinstance(schema, dict) else {}


def _valid_data_slot_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and not _is_placeholder(value)
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def _normalize_issues(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        value = [value] if value else []
    out: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            issue = dict(item)
            if "description" not in issue:
                issue["description"] = str(issue.get("detail") or issue.get("message") or "")
            issue.setdefault("dimension", "质量检查")
            issue.setdefault("severity", "warning")
            issue.setdefault("slot_id", "")
            issue.setdefault("suggested_fix", "")
            issue.setdefault("suggested_action", _infer_action_from_text(str(issue.get("description", ""))))
            out.append(issue)
        elif item:
            text = str(item).strip()
            out.append({
                "dimension": _infer_dimension_from_text(text),
                "severity": _infer_severity_from_text(text),
                "slot_id": _infer_slot_from_text(text),
                "description": text,
                "suggested_fix": "",
                "suggested_action": _infer_action_from_text(text),
            })
    return out


def _merge_issues(*groups: Any) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    merged: list[dict[str, Any]] = []
    for group in groups:
        for issue in _normalize_issues(group):
            key = (str(issue.get("slot_id", "")), str(issue.get("description", ""))[:80])
            if key in seen:
                continue
            seen.add(key)
            merged.append(issue)
    return merged


def _infer_dimension_from_text(text: str) -> str:
    if any(k in text for k in ("模板", "structure", "fallback")):
        return "模板选择"
    if any(k in text for k in ("标题", "副标题", "slot", "槽位", "文案", "文本")):
        return "槽位内容"
    if any(k in text for k in ("透明度", "遮罩", "opacity", "overlay")):
        return "叠加层配置"
    return "质量检查"


def _infer_severity_from_text(text: str) -> str:
    if any(k in text for k in ("暴露内部", "占位符", "无效", "严重", "缺失", "为空", "不匹配")):
        return "critical"
    return "warning"


def _infer_slot_from_text(text: str) -> str:
    if any(k in text for k in ("副标题", "subtitle")):
        return "subtitle"
    if any(k in text for k in ("标题", "title", "headline")):
        return "title"
    match = re.search(r"slot[:：]\s*([A-Za-z0-9_-]+)", text)
    return match.group(1) if match else ""


def _infer_action_from_text(text: str) -> str:
    if any(k in text for k in ("模板", "fallback", "structure", "换")):
        return "swap_structure"
    if any(k in text for k in ("透明度", "遮罩", "opacity", "overlay")):
        return "adjust_overlay"
    return "revise_slots"


def _as_float(value: Any, default: Optional[float]) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_garbled(text: str) -> bool:
    if not text:
        return False
    if len(text) >= 3 and len(set(text)) <= 2:
        return True
    return bool(re.search(r"[{}<>]{2,}|undefined|null|None", text, flags=re.IGNORECASE))


def _load_policy_hint(config: dict[str, Any]) -> str:
    """Load gold_template_policies if available."""
    try:
        path = config.get("paths", {}).get("gold_template_policies_file", "")
        if path and Path(path).exists():
            text = Path(path).read_text(encoding="utf-8")
            if len(text) > 2000:
                text = text[:2000] + "\n...(截断)"
            return f"模板策略参考:\n{text}"
    except Exception:
        pass
    return ""


def _load_instructions_hint(config: dict[str, Any]) -> str:
    """Load instructions if available."""
    try:
        path = config.get("paths", {}).get("instructions_file", "")
        if path and Path(path).exists():
            text = Path(path).read_text(encoding="utf-8")
            if len(text) > 1500:
                text = text[:1500] + "\n...(截断)"
            return f"装饰指令参考:\n{text}"
    except Exception:
        pass
    return ""


_PIPELINE_JARGON = (
    "开篇定调", "视觉冲击", "叙事功能", "承担", "铺垫", "呼应",
    "章节结构", "信息层级", "段落导入", "层级锚点",
    "structure_label", "design_rationale", "emphasis_target",
    "需强视觉", "建立危机感", "段落 ",
)


def _is_placeholder(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return True
    placeholders = {"text", "string", "...", "…", "<text>", "placeholder", "默认", "标题"}
    return cleaned.lower() in placeholders


def _is_pipeline_jargon(text: str) -> bool:
    if not text:
        return False
    # "段落 7" pattern (internal segment numbering)
    if re.match(r"^段落\s*\d+$", text.strip()):
        return True
    hit_count = sum(1 for m in _PIPELINE_JARGON if m in text)
    return hit_count >= 2


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, tuple):
        for item in value:
            if isinstance(item, dict):
                return item
        return {}
    return {}
