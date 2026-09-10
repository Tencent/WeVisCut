"""Tool_PlanShots: Generate shot plan from article/ASR sentence segments.

When video_shots are available in data_pipeline mode, the current Stage1
provides ASR sentence windows as the shot basis. Each sentence window may carry
covered TransNetV2 physical shot metadata for traceability.
Otherwise (inference mode), asks LLM to generate shots from segments.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ..llm import LLMClient

DEFAULT_MAX_SPOTLIGHT_RATIO = 0.30

# -- Prompt for inference mode: generate shots from segments --
SYSTEM_PROMPT = """\
你是视频导演，需要将文章的结构化段落转化为详细的镜头规划。
每个镜头需要包含旁白脚本、视觉画面描述、是否为重点镜头等信息。"""

USER_TEMPLATE = """\
基于以下文章分段结果，生成完整的镜头规划（shot plan）。

文章分段:
{segments_json}

输出要求(严格JSON格式):
{{
  "shots": [
    {{
      "shot_id": "shot_001",
      "segment_id": "seg_001",
      "chapter_title": "章节标题",
      "vocal_script": "该镜头的旁白文本，适合TTS朗读",
      "visual_query": "一段可用于素材库检索的真实画面描述",
      "visual_direction": "画面节奏、情绪、构图方向",
      "duration_sec": 8,
      "is_spotlight": false,
      "design_rationale": "为什么这样设计这个镜头"
    }}
  ]
}}

约束:
- 每个 segment 可拆分为 1-3 个 shots
- vocal_script 要与段落内容一致，不引入原文外事实
- visual_query 应是一句自然的真实画面描述，写清主体、动作和场景，不要只堆关键词
- visual_query 禁止写后期包装词（分屏/卡片/字幕条/信息图/漫画/动漫/插画/动效/UI等）
- visual_direction 才描述画面风格与包装（构图、节奏、情绪、转场建议等）
- is_spotlight=true 的镜头需要模板装饰，仅用于关键内容(总数不超过{max_spotlight_percent})
- duration_sec 根据旁白文本长度合理估算(约4.5字/秒)"""

# -- Prompt for data_pipeline mode: label ASR sentence shots --
LABEL_SYSTEM_PROMPT = """\
你是 shot-level 视觉规划员，需要为按 ASR 句子切分的分镜补充视觉与叙事标注。
每个分镜的 vocal_script 已由上游 ASR 校对得到，必须原样保留，不得改写。
你不看图片、不调用 VLM；visual_query 必须根据当前这句话想象一段可检索的真实画面描述。"""

LABEL_USER_TEMPLATE = """\
以下是按“纠错后的 ASR 一句话 = 一个分镜”切出的 {num_shots} 个分镜窗口，
以及文章的 {num_segments} 个结构化段落。每个分镜窗口可能覆盖多个 TransNetV2 物理镜头，
covered_source_shot_ids 仅用于理解时间范围，不代表要逐个物理镜头输出。

请为每个分镜补充视觉检索和导演标注。

ASR句子分镜:
{shots_json}

文章分段:
{segments_json}

输出要求(严格JSON格式):
{{
  "shots": [
    {{
      "shot_id": "1",
      "visual_query": "一句真实画面描述，而不是关键词列表",
      "visual_direction": "镜头语言、节奏、构图或包装方向",
      "emphasis_target": "主体/对比/数字/概念/流程/因果/结论/时间线",
      "intent": "该镜头的叙事目的",
      "design_rationale": "画面设计思路",
      "is_segment_first": true/false,
      "is_spotlight": true/false,
      "spotlight_reason": "spotlight理由或常规描述",
      "planner_strategy": "该段落的整体包装策略",
      "key_points": ["该段落的核心要点"]
    }}
  ]
}}

约束:
- 输出的 shot 数量必须与输入分镜数量完全一致（{num_shots}个）
- shot_id 必须与输入编号一致
- 章节归属和 vocal_script 由上游确定；不要输出 vocal_script，也不要改写 ASR 句子
- is_spotlight=true 仅用于视觉冲击力强或信息密度高的关键镜头(不超过{max_spotlight_percent})
- visual_query 必须是一句自然的真实画面描述，根据当前 ASR 句子想象画面，禁止直接复用整句 vocal_script
- visual_query 要写清真实可拍摄画面的主体、动作、环境和氛围，不要输出空格分隔的关键词列表
- visual_query 长度建议 24-80 个中文字符；可以像“镜头中，一名老门将在球场上飞身扑救，身后的看台人群紧张注视。”这样描述
- visual_query 禁止出现以下后期装饰词：分屏、转场、字幕条、数据卡片、信息图、图文排版、漫画、动漫、插画、贴纸、速度线、UI、弹窗
- visual_direction 才用于描述镜头语言与包装方向（节奏、情绪、构图、动效建议）；不要把这些包装词放进 visual_query
- intent / design_rationale / spotlight_reason 必须具体对应当前 shot，禁止空泛套话"""

def run(
    llm: LLMClient,
    segments: list[dict[str, Any]],
    video_shots: list[dict[str, Any]] | None = None,
    asr_segments: list[dict[str, Any]] | None = None,
    shot_captions: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate shot plan.

    If video_shots are provided (data_pipeline mode), uses them as the shot
    basis and assigns segment labels. Otherwise generates shots from segments.
    """
    if not segments:
        raise RuntimeError("plan_shots strict validation failed: no segments available")

    if video_shots:
        result = _run_with_video_shots(
            llm,
            segments,
            video_shots,
            asr_segments or [],
            shot_captions or [],
            config=config,
        )
    else:
        result = _run_from_segments(llm, segments, config=config)

    result["shots"] = _dedupe_adjacent_vocal_scripts(
        _post_normalize_shots(result.get("shots", []))
    )
    _validate_final_vocal_scripts(result.get("shots", []))
    result["script_polish_applied"] = False
    return result


def _is_0907(config: dict[str, Any] | None) -> bool:
    config = config or {}
    if str(config.get("policy_version", "") or "").strip() == "0907":
        return True
    newvis = config.get("newvis0907")
    return isinstance(newvis, dict) and bool(newvis.get("enabled"))


def postprocess_shot_plan_0907(
    result: dict[str, Any],
    config: dict[str, Any] | None = None,
    total_duration: float | None = None,
) -> dict[str, Any]:
    """0907-only post-process on ``run``'s output — never touches legacy.

    The training/inference schema is left unchanged; this runs *after* the model
    output to (a) split any shot planned longer than the 15s model-shot band on
    semantic/punctuation boundaries (character-conserving) and (b) attach the
    expected minimum model-shot count + a pacing analysis so downstream stages
    can enforce the 20+ rule. When 0907 is not opted in the result is returned
    verbatim.
    """
    if not _is_0907(config):
        return result
    if not isinstance(result, dict):
        return result
    from studio.newvis0907 import pacing as _pacing

    shots = result.get("shots", [])
    if not isinstance(shots, list):
        return result
    new_shots = _pacing.split_long_shot_plan({"shot_plan": shots})
    duration = float(total_duration or 0.0)
    analysis = _pacing.analyze_shot_pacing({
        "shot_plan": new_shots,
        "total_duration_sec": duration,
    })
    return {
        **result,
        "shots": new_shots,
        "num_shots": len(new_shots),
        "minimum_model_shots": _pacing.minimum_model_shots(duration),
        "pacing_analysis": analysis,
        "pacing_postprocessed": True,
    }


def _run_with_video_shots(
    llm: LLMClient,
    segments: list[dict[str, Any]],
    video_shots: list[dict[str, Any]],
    asr_segments: list[dict[str, Any]],
    shot_captions: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Data pipeline mode: use TransNetV2 shot boundaries as the basis.

    先生成完整的确定性标注，再并发分批调用 LLM 补充语义字段；
    批次缺项只重试缺失部分，最终保留确定性结果而不是整批失败。
    """
    annotated_shots = _assign_asr_to_shots(video_shots, asr_segments)
    caption_map = _build_caption_map(shot_captions)
    _attach_captions_to_annotated_shots(annotated_shots, caption_map)
    segment_binding = _assign_segments_to_annotated_shots(annotated_shots, segments)

    shots_for_prompt = []
    for i, shot in enumerate(annotated_shots):
        sid = str(i + 1)
        bound_seg = segment_binding.get(sid) or _segment_for_shot_index(
            segments, i, total_shots=len(annotated_shots),
        )
        shots_for_prompt.append({
            "shot_id": sid,
            "start_sec": round(shot.get("start_sec", 0), 2),
            "end_sec": round(shot.get("end_sec", 0), 2),
            "duration_sec": round(shot.get("duration_sec", 0), 2),
            "asr_text": shot.get("asr_text", ""),
            "caption_zh": shot.get("caption_zh", ""),
            "source_type": shot.get("source_type", ""),
            "covered_source_shot_ids": list(shot.get("source_shot_ids", []) or []),
            "physical_shot_count": int(shot.get("physical_shot_count", 0) or 0),
            "bound_segment_id": str(bound_seg.get("segment_id", "")) if bound_seg else "",
            "bound_structure_label": str(bound_seg.get("structure_label", "")) if bound_seg else "",
            "bound_chapter_title": str(bound_seg.get("title", "")) if bound_seg else "",
        })

    planner_cfg = ((config or {}).get("stage2_segment", {}) or {}).get("shot_planner", {}) or {}
    max_spotlight_percent = _format_max_spotlight_percent(config)
    batch_size = max(1, _as_int(planner_cfg.get("batch_size"), 20))
    request_concurrency = max(1, _as_int(planner_cfg.get("request_concurrency"), 4))
    field_retry_max = max(0, _as_int(planner_cfg.get("field_retry_max"), 1))
    field_retry_batch_size = max(1, _as_int(planner_cfg.get("field_retry_batch_size"), 10))

    expected_ids = [str(i + 1) for i in range(len(video_shots))]
    deterministic_shots = _deterministic_label_video_shots(shots_for_prompt, segments)
    if bool(planner_cfg.get("use_llm_labeling", True)):
        results_raw = _label_video_shots_resilient(
            llm=llm,
            shots_for_prompt=shots_for_prompt,
            segments=segments,
            batch_size=batch_size,
            request_concurrency=request_concurrency,
            expected_ids=expected_ids,
            field_retry_max=field_retry_max,
            field_retry_batch_size=field_retry_batch_size,
            max_spotlight_percent=max_spotlight_percent,
        )
        shots, fallback_used = _merge_llm_and_fallback_shots(
            results_raw,
            deterministic_shots,
            expected_count=len(expected_ids),
        )
        shots = _restore_deterministic_fields(shots, deterministic_shots)
    else:
        shots = deterministic_shots
        fallback_used = False

    try:
        _validate_complete_shot_labels(shots, expected_ids=expected_ids, context="video_shots")
    except RuntimeError:
        shots = deterministic_shots
        fallback_used = True

    for i, shot in enumerate(shots):
        if not shot.get("shot_id"):
            shot["shot_id"] = str(i + 1)
        if "is_spotlight" not in shot:
            shot["is_spotlight"] = False
        if not shot.get("vocal_script"):
            shot["vocal_script"] = annotated_shots[i].get("asr_text", "") if i < len(annotated_shots) else ""
        caption_zh = annotated_shots[i].get("caption_zh", "") if i < len(annotated_shots) else ""
        fallback_query = _visual_query_from_script(
            str(shot.get("vocal_script", "")),
            _segment_for_shot_index(segments, i, total_shots=len(shots)),
        )
        if not shot.get("visual_query"):
            shot["visual_query"] = caption_zh or fallback_query or shot.get("vocal_script", "")[:60]
        elif caption_zh and str(shot.get("visual_query", "")).strip() == str(shot.get("vocal_script", "")).strip():
            shot["visual_query"] = caption_zh
        elif str(shot.get("visual_query", "")).strip() == str(shot.get("vocal_script", "")).strip() and fallback_query:
            shot["visual_query"] = fallback_query
        if not str(shot.get("visual_direction", "")).strip():
            shot["visual_direction"] = _build_local_visual_direction(
                _segment_for_shot_index(segments, i, total_shots=len(shots)),
                str(shot.get("visual_query", "")),
                str(shot.get("emphasis_target", "主体") or "主体"),
                bool(shot.get("is_segment_first", False)),
            )
        shot["visual_query"] = _ensure_visual_query_description(
            shot.get("visual_query", ""),
            str(shot.get("vocal_script", "")),
            _segment_for_shot_index(segments, i, total_shots=len(shots)),
        )
        if not shot.get("duration_sec"):
            if i < len(annotated_shots):
                shot["duration_sec"] = max(1, round(annotated_shots[i].get("duration_sec", 3)))
            else:
                script_len = len(shot.get("vocal_script", ""))
                shot["duration_sec"] = max(3, round(script_len / 4.5))
        else:
            shot["duration_sec"] = max(1, round(_as_float(shot.get("duration_sec"), 3.0)))
        if i < len(annotated_shots):
            for key in ("start_sec", "end_sec", "source_type", "source_shot_ids", "covered_shots", "physical_shot_count"):
                if key not in shot or shot.get(key) in ("", None, []):
                    shot[key] = annotated_shots[i].get(key, [] if key.endswith("ids") or key == "covered_shots" else "")

    return {
        "shots": shots,
        "num_shots": len(shots),
        "num_spotlight": sum(1 for s in shots if s.get("is_spotlight")),
        "status": "success" if shots else "failed",
        "fallback_used": fallback_used,
        "mode": "video_shots",
    }


def _build_caption_map(shot_captions: list[dict[str, Any]]) -> dict[str, str]:
    """Build shot_id -> caption_zh map."""
    out: dict[str, str] = {}
    for item in shot_captions:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("shot_id", "")).strip()
        cap = str(item.get("caption_zh", "")).strip()
        if sid and cap:
            out[sid] = cap
    return out


def _segment_for_shot_index(
    segments: list[dict[str, Any]],
    shot_index_zero_based: int,
    *,
    total_shots: int | None = None,
) -> dict[str, Any]:
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        rng = seg.get("shot_index_range", [])
        if isinstance(rng, list) and len(rng) >= 2:
            try:
                start, end = int(rng[0]), int(rng[1])
            except (TypeError, ValueError):
                continue
            if start <= shot_index_zero_based <= end:
                return seg
    if not segments:
        return {}
    total = max(1, int(total_shots or (shot_index_zero_based + 1)))
    idx = min(len(segments) - 1, max(0, shot_index_zero_based * len(segments) // total))
    return segments[idx] if isinstance(segments[idx], dict) else {}


def _assign_segments_to_annotated_shots(
    annotated_shots: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Port of the old pipeline's ASR-anchor binding.

    Segment ownership is deterministic. LLM output is never allowed to change
    segment_id / structure_label / chapter_title.
    """
    if not segments:
        return {}

    transcript = "".join(_strip_ws(shot.get("asr_text", "")) for shot in annotated_shots)
    ranges = _compute_segment_ranges(transcript, segments)
    have_ranges = any(start >= 0 and end > start for start, end in ranges)
    if not transcript or not have_ranges:
        return {
            str(idx + 1): _segment_for_shot_index(
                segments, idx, total_shots=len(annotated_shots),
            )
            for idx in range(len(annotated_shots))
        }

    offsets: list[int] = []
    cursor = 0
    for shot in annotated_shots:
        text = _strip_ws(shot.get("asr_text", ""))
        if not text:
            offsets.append(-1)
            continue
        idx = transcript.find(text[:18], cursor)
        if idx < 0:
            idx = transcript.find(text[:18])
        offsets.append(idx)
        if idx >= 0:
            cursor = idx + len(text)

    binding: dict[str, dict[str, Any]] = {}
    last_segment: dict[str, Any] | None = None
    for idx, (shot, offset) in enumerate(zip(annotated_shots, offsets)):
        chosen: dict[str, Any] | None = None
        if offset >= 0:
            for seg, (start, end) in zip(segments, ranges):
                if start >= 0 and start <= offset < max(end, start + 1):
                    chosen = seg
                    break
            if chosen is None:
                candidates = [
                    (abs(offset - start), seg)
                    for seg, (start, _end) in zip(segments, ranges)
                    if start >= 0
                ]
                if candidates:
                    chosen = min(candidates, key=lambda item: item[0])[1]
        if chosen is None and not _strip_ws(shot.get("asr_text", "")) and last_segment is not None:
            chosen = last_segment
        if chosen is None:
            chosen = _segment_for_shot_index(
                segments, idx, total_shots=len(annotated_shots),
            )
        binding[str(idx + 1)] = chosen
        if _strip_ws(shot.get("asr_text", "")):
            last_segment = chosen
    return binding


def _compute_segment_ranges(
    transcript: str,
    segments: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    starts: list[int] = []
    cursor = 0
    for seg in segments:
        anchor = _strip_ws(
            seg.get("raw_span")
            or seg.get("narration_script")
            or seg.get("summary")
            or ""
        )[:24]
        if not anchor:
            starts.append(-1)
            continue
        idx = transcript.find(anchor, cursor)
        if idx < 0:
            idx = transcript.find(anchor)
        starts.append(idx)
        if idx >= 0:
            cursor = idx + len(anchor)

    ranges: list[tuple[int, int]] = []
    for idx, start in enumerate(starts):
        if start < 0:
            ranges.append((-1, -1))
            continue
        next_start = next((value for value in starts[idx + 1:] if value > start), len(transcript))
        ranges.append((start, next_start))
    return ranges


def _strip_ws(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _text_len(value: Any) -> int:
    return len(_strip_ws(value))


def _label_video_shots_resilient(
    *,
    llm: LLMClient,
    shots_for_prompt: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    batch_size: int,
    request_concurrency: int,
    expected_ids: list[str],
    field_retry_max: int = 1,
    field_retry_batch_size: int = 10,
    max_spotlight_percent: str = "30%",
) -> list[dict[str, Any]]:
    """Batch enrichment with partial retries; unresolved rows use local defaults."""

    def _format_prompt(batch: list[dict[str, Any]]) -> str:
        return LABEL_USER_TEMPLATE.format(
            num_shots=len(batch),
            num_segments=len(segments),
            shots_json=json.dumps(batch, ensure_ascii=False, indent=2),
            segments_json=json.dumps(_trim_segments_for_prompt(segments), ensure_ascii=False, indent=2),
            max_spotlight_percent=max_spotlight_percent,
        )

    def _run_batches(items: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
        chunks = [items[i:i + size] for i in range(0, len(items), size)]
        if not chunks:
            return []

        def _call(chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
            try:
                raw = llm.chat_json(LABEL_SYSTEM_PROMPT, _format_prompt(chunk), temperature=0.2)
            except Exception:
                return []
            allowed = {str(item.get("shot_id", "")).strip() for item in chunk}
            rows: list[dict[str, Any]] = []
            for row in _extract_shot_label_items(raw):
                sid = _normalize_returned_shot_id(
                    row.get("shot_id") or row.get("id") or row.get("index")
                )
                if sid and sid in allowed:
                    rows.append({**row, "shot_id": sid})
            return rows

        if len(chunks) == 1:
            return _call(chunks[0])
        by_index: dict[int, list[dict[str, Any]]] = {}
        workers = min(max(1, request_concurrency), len(chunks))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_call, chunk): idx for idx, chunk in enumerate(chunks)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    by_index[idx] = future.result()
                except Exception:
                    by_index[idx] = []
        return [
            row
            for idx in range(len(chunks))
            for row in by_index.get(idx, [])
        ]

    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in _run_batches(shots_for_prompt, max(1, batch_size)):
        rows_by_id[str(row.get("shot_id", "")).strip()] = row

    prompt_by_id = {
        str(item.get("shot_id", "")).strip(): item
        for item in shots_for_prompt
    }
    for _attempt in range(field_retry_max):
        missing_ids = [sid for sid in expected_ids if sid not in rows_by_id]
        if not missing_ids:
            break
        retry_items = [prompt_by_id[sid] for sid in missing_ids if sid in prompt_by_id]
        for row in _run_batches(retry_items, max(1, field_retry_batch_size)):
            rows_by_id[str(row.get("shot_id", "")).strip()] = row

    return [rows_by_id[sid] for sid in expected_ids if sid in rows_by_id]


def _trim_segments_for_prompt(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "segment_id": seg.get("segment_id", ""),
            "structure_label": seg.get("structure_label", ""),
            "chapter_title": seg.get("title", ""),
            "summary": str(seg.get("summary", "") or "")[:200],
            "key_points": list(seg.get("key_points", []) or [])[:5],
            "planner_strategy": str(seg.get("planner_strategy", "") or "")[:200],
        }
        for seg in segments
        if isinstance(seg, dict)
    ]


def _normalize_returned_shot_id(value: Any) -> str:
    sid = str(value or "").strip()
    if not sid:
        return ""
    match = re.search(r"(\d+)$", sid)
    return match.group(1) if match else sid


def _restore_deterministic_fields(
    shots: list[dict[str, Any]],
    deterministic_shots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge semantic LLM fields while preserving source alignment and chapter binding."""
    base_by_id = {
        str(shot.get("shot_id", "")).strip(): shot
        for shot in deterministic_shots
        if isinstance(shot, dict)
    }
    semantic_fields = {
        "visual_query",
        "visual_direction",
        "emphasis_target",
        "intent",
        "design_rationale",
        "is_spotlight",
        "spotlight_reason",
    }
    out: list[dict[str, Any]] = []
    for shot in shots:
        sid = str(shot.get("shot_id", "")).strip()
        base = dict(base_by_id.get(sid, {"shot_id": sid}))
        for field in semantic_fields:
            value = shot.get(field)
            if field == "is_spotlight":
                if isinstance(value, bool) and not base.get("is_segment_first"):
                    base[field] = value
            elif str(value or "").strip():
                base[field] = value
        if base.get("is_segment_first"):
            base["is_spotlight"] = True
            if not str(base.get("spotlight_reason", "")).strip():
                base["spotlight_reason"] = "章节开篇层级锚点，需要用 spotlight 强化结构分层。"
        out.append(base)
    return out


def _deterministic_label_video_shots(
    shots_for_prompt: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Plan every ASR sentence shot locally using script text and segment arcs."""
    seen_segment_ids: set[str] = set()
    segment_by_id = {
        str(seg.get("segment_id", "") or ""): seg
        for seg in segments
        if isinstance(seg, dict)
    }
    out: list[dict[str, Any]] = []

    for idx, shot in enumerate(shots_for_prompt):
        bound_id = str(shot.get("bound_segment_id", "") or "")
        seg = segment_by_id.get(bound_id) or _segment_for_shot_index(
            segments, idx, total_shots=len(shots_for_prompt),
        )
        seg_id = str(seg.get("segment_id", "") or "")
        is_first = bool(seg_id and seg_id not in seen_segment_ids)
        if seg_id:
            seen_segment_ids.add(seg_id)

        asr_text = _normalize_vocal_script_text(shot.get("asr_text", ""))
        caption_zh = _normalize_visual_query(shot.get("caption_zh", ""))
        summary = str(seg.get("summary", "") or "").strip()
        key_points = list(seg.get("key_points", []) or [])
        visual_query = (
            caption_zh
            or _visual_query_from_script(asr_text, seg)
            or _normalize_visual_query(summary)
            or str(seg.get("title", "") or "真实人物场景")
        )
        vocal_script = asr_text or _build_silent_shot_vocal_script(shot, seg, visual_query)
        emphasis_target = _infer_emphasis_target(vocal_script, visual_query, is_first)

        is_spotlight = is_first

        out.append({
            "shot_id": str(shot.get("shot_id", idx + 1)),
            "segment_id": seg_id,
            "structure_label": str(seg.get("structure_label", "") or ""),
            "chapter_title": str(seg.get("title", "") or ""),
            "vocal_script": vocal_script,
            "visual_query": visual_query,
            "visual_direction": _build_local_visual_direction(seg, visual_query, emphasis_target, is_first),
            "emphasis_target": emphasis_target,
            "intent": _build_local_intent(seg, emphasis_target, is_first),
            "design_rationale": _build_local_design_rationale(seg, visual_query, emphasis_target, is_first),
            "is_segment_first": is_first,
            "is_spotlight": is_spotlight,
            "spotlight_reason": _build_local_spotlight_reason(seg, is_spotlight, is_first),
            "planner_strategy": str(seg.get("planner_strategy", "") or "按原视频叙事节奏呈现关键画面"),
            "key_points": key_points,
            "duration_sec": max(1, round(_as_float(shot.get("duration_sec"), 3.0))),
            "start_sec": round(_as_float(shot.get("start_sec"), 0.0), 3),
            "end_sec": round(_as_float(shot.get("end_sec"), 0.0), 3),
            "source_type": str(shot.get("source_type", "") or ""),
            "source_shot_ids": list(shot.get("source_shot_ids", []) or []),
            "physical_shot_count": int(shot.get("physical_shot_count", 0) or 0),
        })

    return out


def _build_silent_shot_vocal_script(
    shot: dict[str, Any],
    seg: dict[str, Any],
    visual_query: str,
) -> str:
    """Create a short non-duplicated narration fallback for shots without ASR."""
    title = _normalize_vocal_script_text(seg.get("title", "")) or "当前章节"
    visual = _normalize_vocal_script_text(visual_query)[:48]
    summary = _normalize_vocal_script_text(seg.get("summary", ""))[:36]
    if visual:
        return _normalize_vocal_script_text(f"画面展示{visual}，承接章节「{title}」。")
    if summary:
        return _normalize_vocal_script_text(f"围绕「{title}」，继续铺陈{summary}。")
    start = _as_float(shot.get("start_sec"), 0.0)
    return _normalize_vocal_script_text(
        f"画面继续推进「{title}」的叙事节奏。{start:.1f}秒处进入下一层信息。"
    )


def _normalize_visual_query(value: Any) -> str:
    text = _normalize_vocal_script_text(value)
    if not text:
        return ""
    text = re.sub(r"\b(UI|ui)\b", "", text)
    banned = "分屏|转场|字幕条|数据卡片|信息图|图文排版|漫画|动漫|插画|贴纸|速度线|弹窗"
    text = re.sub(banned, "", text)
    text = re.sub(r"\s+", " ", text).strip(" ，；：、")
    return text[:160]


def _ensure_visual_query_description(query: Any, vocal_script: str, seg: dict[str, Any]) -> str:
    text = _normalize_visual_query(query)
    fallback = _visual_query_from_script(vocal_script, seg)
    if not text:
        return fallback
    if _strip_ws(text) == _strip_ws(vocal_script):
        return fallback or text[:80]
    keyword_like = (
        " " in text
        and not re.search(r"[，。！？；]", text)
        and len([part for part in text.split(" ") if part.strip()]) >= 2
    )
    too_short_phrase = _text_len(text) < 12 and not re.search(r"[，。！？；]", text)
    has_sentence_shape = bool(re.search(r"[，。！？；]", text)) or any(
        marker in text for marker in ("画面", "镜头", "一名", "一位", "一座", "一群", "工作人员", "运动员")
    )
    if has_sentence_shape and not keyword_like and not too_short_phrase:
        return text if text[-1] in "。！？；" else f"{text}。"

    phrase = re.sub(r"[\s，、；]+", "、", text).strip("、。")
    if phrase:
        return _normalize_visual_query(f"镜头中呈现{phrase}的真实场景，主体动作和环境关系清晰可见。")
    return fallback


def _visual_query_from_script(vocal_script: str, seg: dict[str, Any]) -> str:
    """Build a retrievable live-action scene description from the ASR sentence fallback."""
    text = _normalize_vocal_script_text(vocal_script)
    context = f"{text} {seg.get('title', '')} {seg.get('summary', '')}"
    if not text and not context.strip():
        return ""
    domain_queries = (
        (("足球", "世界杯", "门将", "进球", "球队", "球员", "比赛"), "镜头中，足球比赛现场的球员在草坪上激烈攻防，看台观众紧张注视。"),
        (("篮球", "投篮", "篮板", "NBA"), "镜头中，篮球比赛现场的球员在球场上快速攻防，观众席充满紧张气氛。"),
        (("实验", "科学", "研究", "技术", "芯片", "AI", "模型"), "镜头中，科研人员在实验室操作仪器设备，屏幕和台面呈现技术研究场景。"),
        (("城市", "街道", "交通", "商业", "公司"), "镜头中，城市街道上人群与车辆流动，建筑和商业环境共同构成现实生活场景。"),
        (("学校", "考试", "学习", "学生", "课堂"), "镜头中，学生在教室里低头学习，书本和课桌构成安静的课堂环境。"),
        (("医院", "医生", "患者", "健康", "医疗"), "镜头中，医生在医院诊室与患者交流，医疗设备和走廊营造真实诊疗环境。"),
        (("家庭", "亲人", "孩子", "父母", "人生"), "镜头中，家人在室内生活空间里交流陪伴，人物关系和情绪自然可见。"),
        (("海", "港口", "岛", "国家", "国旗"), "镜头中，海港城市的街头和岸线同时出现，国旗与人群呈现地域氛围。"),
    )
    for keywords, query in domain_queries:
        if any(keyword in context for keyword in keywords):
            return query

    cleaned = re.sub(
        r"(?:但是|然后|所以|因为|如果|其实|这个|那个|就是|我们|他们|你会发现|可以看到|"
        r"换句话说|也就是说|接下来|最终|同时)",
        "",
        text,
    )
    cleaned = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9，,、 ]+", "", cleaned)
    clauses = [c.strip() for c in re.split(r"[，,、。！？!?；;]", cleaned) if c.strip()]
    core = max(clauses, key=len) if clauses else cleaned
    core = core[:36].strip()
    if not core:
        title = str(seg.get("title", "") or "").strip()
        return _normalize_visual_query(f"镜头中呈现与「{title}」相关的真实人物和环境场景。") if title else "镜头中呈现真实人物与环境的纪实场景。"
    return _normalize_visual_query(f"镜头中呈现{core}相关的真实场景，主体和环境细节清晰可见。")


def _infer_emphasis_target(vocal_script: str, visual_query: str, is_first: bool) -> str:
    text = f"{vocal_script} {visual_query}"
    if re.search(r"\d", text):
        return "数字"
    if any(word in text for word in ("因为", "导致", "所以", "帮助", "证明", "实现")):
        return "因果"
    if any(word in text for word in ("对比", "比", "但", "然而", "却")):
        return "对比"
    if any(word in text for word in ("最后", "最终", "证明", "梦想成真", "不是限制")):
        return "结论"
    if is_first:
        return "主体"
    return "概念"


def _build_local_intent(seg: dict[str, Any], emphasis_target: str, is_first: bool) -> str:
    label = str(seg.get("structure_label", "") or "段落")
    summary = str(seg.get("summary", "") or "").strip()
    if is_first:
        return f"作为{label}的开篇层级锚点，先建立本章主题并突出{emphasis_target}信息。"
    if summary:
        return f"承接本章前后镜头，围绕叙事重点继续强化{emphasis_target}信息。"
    return f"延续原视频内容，突出{emphasis_target}信息。"


def _build_local_design_rationale(
    seg: dict[str, Any],
    visual_query: str,
    emphasis_target: str,
    is_first: bool,
) -> str:
    title = str(seg.get("title", "") or "当前章节")
    label = str(seg.get("structure_label", "") or "概述")
    summary = str(seg.get("summary", "") or "").strip()
    if label == "引子":
        narrative_role = "快速建立事件悬念和观看动机"
    elif label == "总结升华":
        narrative_role = "把前文信息收束为清晰结论和情绪落点"
    elif label == "概述":
        narrative_role = "交代背景、主体关系和后续观看顺序"
    else:
        narrative_role = "展开关键事实、证据或事件转折"
    prefix = f"作为章节「{title}」的开篇镜头，" if is_first else f"位于章节「{title}」的推进段，"
    return (
        f"{prefix}本章需要{narrative_role}。画面优先沿用原视频可见的「{visual_query}」，"
        f"把{emphasis_target}设为第一信息层级；口播对应的章节重点是「{summary[:72]}」。"
        "节奏上保持与相邻实拍镜头连续，避免为了包装打断原视频叙事。"
    )


def _build_local_visual_direction(
    seg: dict[str, Any],
    visual_query: str,
    emphasis_target: str,
    is_first: bool,
) -> str:
    title = str(seg.get("title", "") or "当前章节")
    opening = "章节开篇用更明确的建立镜头" if is_first else "章节推进段保持连续叙事"
    return (
        f"{opening}，围绕「{visual_query}」组织主体与环境关系；"
        f"构图上突出{emphasis_target}信息，并与「{title}」的口播节奏对齐。"
    )


def _build_local_spotlight_reason(seg: dict[str, Any], is_spotlight: bool, is_first: bool) -> str:
    title = str(seg.get("title", "") or "当前章节")
    if is_spotlight:
        return f"章节「{title}」的开篇镜头，用于标记段落层级并完成导入，需用 spotlight 强化结构感。"
    if is_first:
        return f"章节「{title}」开篇但 spotlight 配额已受控，保持常规镜头避免视觉疲劳。"
    return "普通叙事镜头，服务于原视频内容推进，无需 spotlight 以保持节奏。"


def _extract_shot_label_items(raw: Any) -> list[dict[str, Any]]:
    """Accept common model JSON shapes, including nested and id-keyed objects."""
    if isinstance(raw, list):
        direct = [item for item in raw if isinstance(item, dict) and _looks_like_shot(item)]
        if direct:
            return direct
        nested: list[dict[str, Any]] = []
        for item in raw:
            nested.extend(_extract_shot_label_items(item))
        return nested
    if not isinstance(raw, dict):
        return []
    if _looks_like_shot(raw):
        return [raw]

    for key in ("shots", "__array__", "items", "results", "data", "shot_plan", "shot_plans", "plans"):
        value = raw.get(key)
        if value is None:
            continue
        rows = _extract_shot_label_items(value)
        if rows:
            return rows

    keyed: list[dict[str, Any]] = []
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        item = dict(value)
        item.setdefault("shot_id", _normalize_returned_shot_id(key))
        if _looks_like_shot(item):
            keyed.append(item)
    return keyed


def _looks_like_shot(item: dict[str, Any]) -> bool:
    return bool(
        item.get("shot_id")
        or item.get("id")
        or item.get("index")
        or any(
            key in item
            for key in (
                "vocal_script",
                "visual_query",
                "emphasis_target",
                "design_rationale",
                "is_spotlight",
            )
        )
    )


def _attach_captions_to_annotated_shots(
    annotated_shots: list[dict[str, Any]],
    caption_map: dict[str, str],
) -> None:
    """Attach caption_zh to each annotated shot by shot id."""
    for idx, shot in enumerate(annotated_shots):
        sid = str(idx + 1)
        shot["caption_zh"] = caption_map.get(sid, "")


def _run_from_segments(
    llm: LLMClient,
    segments: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Inference mode: generate shots from article segments."""
    prompt = USER_TEMPLATE.format(
        segments_json=json.dumps(segments, ensure_ascii=False, indent=2),
        max_spotlight_percent=_format_max_spotlight_percent(config),
    )
    result = _as_dict(llm.chat_json(SYSTEM_PROMPT, prompt, temperature=0.2))
    shots = _extract_shots(result)
    if not shots:
        shots = _fallback_shots_from_segments(segments)
        if not shots:
            raise RuntimeError("plan_shots strict validation failed: LLM returned no shots")
    for i, shot in enumerate(shots):
        seg = _segment_for_shot_index(segments, i, total_shots=len(shots))
        if not shot.get("shot_id"):
            shot["shot_id"] = f"shot_{i+1:03d}"
        if not str(shot.get("chapter_title", "")).strip():
            shot["chapter_title"] = str(seg.get("title", "") or "")
        if not str(shot.get("segment_id", "")).strip():
            shot["segment_id"] = str(seg.get("segment_id", "") or "")
        if not str(shot.get("vocal_script", "")).strip():
            shot["vocal_script"] = _normalize_vocal_script_text(
                seg.get("narration_script")
                or seg.get("raw_span")
                or seg.get("summary")
                or shot.get("chapter_title", "")
            )
        shot["visual_query"] = _ensure_visual_query_description(
            shot.get("visual_query", ""),
            str(shot.get("vocal_script", "")),
            seg,
        )
        if "is_spotlight" not in shot:
            shot["is_spotlight"] = False
        if not shot.get("duration_sec"):
            script_len = len(shot.get("vocal_script", ""))
            shot["duration_sec"] = max(3, round(script_len / 4.5))
        else:
            shot["duration_sec"] = max(3, round(_as_float(shot.get("duration_sec"), 8.0)))
    return {
        "shots": shots,
        "num_shots": len(shots),
        "num_spotlight": sum(1 for s in shots if s.get("is_spotlight")),
        "status": "success" if shots else "failed",
        "fallback_used": False,
        "mode": "segments",
    }


def _fallback_shots_from_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic fallback for inference mode when LLM returns empty shots."""
    out: list[dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue
        sid = f"shot_{idx + 1:03d}"
        seg_id = str(seg.get("segment_id", "") or f"seg_{idx + 1:02d}")
        title = str(seg.get("title", "") or f"章节{idx + 1}")
        script = _normalize_vocal_script_text(
            seg.get("narration_script")
            or seg.get("raw_span")
            or seg.get("summary")
            or title
        )
        if not script:
            script = _normalize_vocal_script_text(f"围绕「{title}」继续推进当前叙事。")
        visual_query = _visual_query_from_script(script, seg)
        is_first = idx == 0
        emphasis_target = _infer_emphasis_target(script, visual_query, is_first)
        out.append({
            "shot_id": sid,
            "segment_id": seg_id,
            "chapter_title": title,
            "vocal_script": script,
            "visual_query": visual_query,
            "visual_direction": _build_local_visual_direction(seg, visual_query, emphasis_target, is_first),
            "duration_sec": max(3, round(len(script) / 4.5)),
            "is_spotlight": is_first,
            "design_rationale": _build_local_design_rationale(seg, visual_query, emphasis_target, is_first),
            "emphasis_target": emphasis_target,
        })
    return out


def _assign_asr_to_shots(
    video_shots: list[dict[str, Any]],
    asr_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign ASR text to each video shot without duplicating transcript segments.

    Each ASR segment is assigned to exactly one shot: the shot with the largest
    temporal overlap. This keeps downstream TTS continuous and non-overlapping
    across shot boundaries.
    """
    annotated = []
    for shot in video_shots:
        s_start = float(
            shot.get("start_time") or shot.get("start_sec") or shot.get("start", 0)
        )
        s_end = float(
            shot.get("end_time") or shot.get("end_sec") or shot.get("end", 0)
        )
        s_dur = float(
            shot.get("duration") or shot.get("duration_sec", 0)
        )
        if s_end <= s_start:
            s_end = s_start + max(s_dur, 1.0)

        annotated.append({
            "shot_id": str(shot.get("shot_id", len(annotated) + 1)),
            "start_sec": s_start,
            "end_sec": s_end,
            "duration_sec": s_end - s_start,
            "asr_text": _normalize_vocal_script_text(
                shot.get("asr_text")
                or shot.get("text")
                or shot.get("vocal_script")
                or ""
            ),
            "source_type": str(shot.get("source_type", "") or ""),
            "source_shot_ids": list(shot.get("source_shot_ids", []) or []),
            "covered_shots": list(shot.get("covered_shots", []) or []),
            "physical_shot_count": int(shot.get("physical_shot_count", 0) or 0),
        })

    assigned_texts: list[list[str]] = [[] for _ in annotated]
    for seg in asr_segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        try:
            a_start = float(seg.get("start", 0))
            a_end = float(seg.get("end", 0))
        except (TypeError, ValueError):
            continue
        if a_end <= a_start:
            continue

        best_idx = -1
        best_overlap = 0.0
        midpoint = (a_start + a_end) / 2.0
        midpoint_idx = -1
        for idx, shot in enumerate(annotated):
            s_start = float(shot.get("start_sec", 0.0) or 0.0)
            s_end = float(shot.get("end_sec", 0.0) or 0.0)
            overlap = max(0.0, min(s_end, a_end) - max(s_start, a_start))
            if s_start <= midpoint < s_end:
                midpoint_idx = idx
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx

        if best_idx < 0 and midpoint_idx >= 0:
            best_idx = midpoint_idx
        if best_idx >= 0:
            assigned_texts[best_idx].append(text)

    for shot, texts in zip(annotated, assigned_texts):
        assigned = _normalize_vocal_script_text(" ".join(texts))
        if assigned:
            shot["asr_text"] = assigned
    return annotated


def _merge_llm_and_fallback_shots(
    llm_shots: list[dict[str, Any]],
    fallback_shots: list[dict[str, Any]],
    *,
    expected_count: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Keep valid LLM labels and fill only missing shot ids with fallback rows."""
    fallback_by_id = {
        str(shot.get("shot_id", "")).strip(): shot
        for shot in fallback_shots
        if isinstance(shot, dict) and str(shot.get("shot_id", "")).strip()
    }
    llm_by_id: dict[str, dict[str, Any]] = {}
    for shot in llm_shots:
        if not isinstance(shot, dict):
            continue
        sid = str(shot.get("shot_id", "")).strip()
        if not sid:
            continue
        llm_by_id[sid] = shot

    merged: list[dict[str, Any]] = []
    fallback_used = len(llm_by_id) != expected_count
    for idx in range(expected_count):
        sid = str(idx + 1)
        llm_row = llm_by_id.get(sid)
        fallback_row = fallback_by_id.get(sid, {"shot_id": sid})
        if llm_row is None:
            merged.append(dict(fallback_row))
            fallback_used = True
            continue
        # Preserve deterministic timing/script fields from source alignment when
        # the LLM omits them, but keep real labels when it provides them.
        merged.append({**fallback_row, **llm_row})
    return merged, fallback_used


def _validate_complete_shot_labels(
    shots: list[dict[str, Any]],
    *,
    expected_ids: list[str],
    context: str,
) -> None:
    """Fail closed when LLM shot labeling is incomplete or low quality."""
    errors: list[str] = []
    if len(shots) != len(expected_ids):
        errors.append(f"expected {len(expected_ids)} shots, got {len(shots)}")

    seen: set[str] = set()
    expected = set(expected_ids)
    banned_markers = ("LLM标注失败", "兜底", "fallback", "按时间顺序分配段落")
    required_fields = (
        "shot_id",
        "segment_id",
        "chapter_title",
        "vocal_script",
        "visual_query",
        "emphasis_target",
        "intent",
        "design_rationale",
        "spotlight_reason",
        "planner_strategy",
    )
    for idx, shot in enumerate(shots, start=1):
        if not isinstance(shot, dict):
            errors.append(f"row {idx} is not object")
            continue
        sid = str(shot.get("shot_id", "")).strip()
        if not sid:
            errors.append(f"row {idx} missing shot_id")
            continue
        if sid in seen:
            errors.append(f"duplicate shot_id={sid}")
        seen.add(sid)
        if sid not in expected:
            errors.append(f"unexpected shot_id={sid}")
        for field in required_fields:
            value = shot.get(field)
            if isinstance(value, str):
                text = value.strip()
            elif value is None:
                text = ""
            else:
                text = str(value).strip()
            if not text:
                errors.append(f"shot {sid} missing {field}")
            if any(marker in text for marker in banned_markers):
                errors.append(f"shot {sid} contains fallback marker in {field}")
        if not isinstance(shot.get("is_spotlight"), bool):
            errors.append(f"shot {sid} is_spotlight must be boolean")
        if not isinstance(shot.get("is_segment_first"), bool):
            errors.append(f"shot {sid} is_segment_first must be boolean")

    missing_ids = sorted(expected - seen, key=lambda x: int(x) if x.isdigit() else x)
    if missing_ids:
        errors.append(f"missing shot_ids={missing_ids[:10]}")

    if errors:
        detail = "; ".join(errors[:20])
        raise RuntimeError(f"plan_shots strict validation failed ({context}): {detail}")


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, tuple):
        for item in value:
            if isinstance(item, dict):
                return item
            if isinstance(item, list):
                return {"shots": item}
        return {}
    if isinstance(value, list):
        return {"shots": value}
    return {}


def _extract_shots(result: dict[str, Any]) -> list[dict[str, Any]]:
    raw_shots = (
        result.get("shots")
        or result.get("shot_plan")
        or result.get("plans")
        or result.get("items")
        or result.get("results")
        or result.get("data")
        or result.get("__array__")
        or []
    )
    if not isinstance(raw_shots, list):
        raw_shots = _extract_shot_label_items(result)
        if not isinstance(raw_shots, list):
            return []

    shots: list[dict[str, Any]] = []
    for item in raw_shots:
        if isinstance(item, dict):
            script = (
                item.get("vocal_script")
                or item.get("narration_script")
                or item.get("script")
                or item.get("asr_text")
                or item.get("text")
                or ""
            )
            shots.append({**item, "vocal_script": _normalize_vocal_script_text(script)})
    return [shot for shot in shots if isinstance(shot, dict)]


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _format_max_spotlight_percent(config: dict[str, Any] | None) -> str:
    planner_cfg = ((config or {}).get("stage2_segment", {}) or {}).get("shot_planner", {}) or {}
    ratio = _as_float(planner_cfg.get("max_spotlight_ratio"), DEFAULT_MAX_SPOTLIGHT_RATIO)
    if ratio > 1.0:
        ratio = ratio / 100.0
    ratio = max(0.0, min(1.0, ratio))
    percent = ratio * 100.0
    if percent.is_integer():
        return f"{int(percent)}%"
    return f"{percent:.1f}".rstrip("0").rstrip(".") + "%"


def _normalize_vocal_script_text(value: Any) -> str:
    """Normalize TTS script spacing/punctuation for Chinese narration."""
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    # Remove spaces between Chinese characters (common ASR artifact).
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    # Remove spaces around Chinese punctuation.
    text = re.sub(r"\s*([，。！？；：、“”‘’（）《》【】])\s*", r"\1", text)
    # Normalize commas/periods repeated by noisy ASR formatting.
    text = re.sub(r"[，,]{2,}", "，", text)
    text = re.sub(r"[。\.]{2,}", "。", text)
    return text.strip()


def _post_normalize_shots(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for shot in shots:
        if not isinstance(shot, dict):
            continue
        out.append({
            **shot,
            "vocal_script": _normalize_vocal_script_text(shot.get("vocal_script", "")),
        })
    return out


def _dedupe_adjacent_vocal_scripts(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove repeated phrase overlaps across adjacent TTS scripts."""
    deduped: list[dict[str, Any]] = []
    previous = ""
    for shot in shots:
        if not isinstance(shot, dict):
            continue
        current = _normalize_vocal_script_text(shot.get("vocal_script", ""))
        original = current
        overlap = _longest_suffix_prefix_overlap(previous, current)
        if overlap > 0:
            current = _normalize_vocal_script_text(current[overlap:])
        if not current:
            current = _build_vocal_script_fallback(shot, original)
        updated = {**shot, "vocal_script": current}
        deduped.append(updated)
        if current:
            previous = current
    return deduped


def _build_vocal_script_fallback(shot: dict[str, Any], original_script: str) -> str:
    """Build a short non-empty fallback when dedupe trims a script to empty."""
    chapter = _normalize_vocal_script_text(shot.get("chapter_title", ""))[:24]
    visual = _normalize_vocal_script_text(shot.get("visual_query", ""))[:36]
    sid = str(shot.get("shot_id", "")).strip()
    if chapter and visual:
        text = f"接着进入「{chapter}」，画面转向{visual}。"
    elif chapter:
        text = f"接着进入「{chapter}」的下一层信息。"
    elif visual:
        text = f"画面继续推进，聚焦{visual}。"
    elif original_script:
        text = f"镜头继续推进，补充上一段的关键信息。"
    else:
        text = f"{sid or '当前镜头'}继续承接主线叙事。"
    return _normalize_vocal_script_text(text)


def _validate_final_vocal_scripts(shots: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    previous = ""
    previous_sid = ""
    for shot in shots:
        if not isinstance(shot, dict):
            continue
        sid = str(shot.get("shot_id", "")).strip()
        current = _normalize_vocal_script_text(shot.get("vocal_script", ""))
        if not current:
            errors.append(f"shot {sid or '?'} empty vocal_script after polish")
        overlap = _longest_suffix_prefix_overlap(previous, current)
        if overlap > 0:
            errors.append(
                f"adjacent vocal_script overlap: {previous_sid or '?'}->{sid or '?'} length={overlap}"
            )
        if current:
            previous = current
            previous_sid = sid
    if errors:
        raise RuntimeError("plan_shots strict validation failed (vocal_script): " + "; ".join(errors[:10]))


def _longest_suffix_prefix_overlap(previous: str, current: str) -> int:
    """Return exact char overlap length between previous suffix and current prefix."""
    previous = _normalize_vocal_script_text(previous)
    current = _normalize_vocal_script_text(current)
    max_len = min(len(previous), len(current), 80)
    # Avoid trimming common two-character discourse markers.
    min_len = 6
    for size in range(max_len, min_len - 1, -1):
        if previous[-size:] == current[:size]:
            return size
    return 0
