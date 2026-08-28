"""Split ASR/article text into compact, semantic video chapters."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from ..llm import LLMClient

SYSTEM_PROMPT = """\
你是资深视频导演编辑。请按原文时间顺序，把 ASR/文章切成少量语义章节。
章节标题必须概括内容，不能写“段落1”，也不能与 structure_label 相同。"""

USER_TEMPLATE = """\
请把下面的 ASR/文章拆成 4~{max_segments} 个章节。

【全文】
{article_text}

【Stage1 整片概要，仅作辅助】
{video_summary_json}

只输出严格 JSON：
{{
  "intent": "全片核心主旨",
  "music_style": "纯音乐风格：曲风+速度+主要乐器+情绪，只写音乐本身",
  "segments": [
    {{
      "segment_id": "seg_01",
      "structure_label": "引子|概述|核心章节|总结升华",
      "title": "6~18字的内容型章节标题",
      "summary": "本章叙事重点",
      "key_points": ["要点1", "要点2"],
      "raw_span": "从本章开头原样复制40~80字，作为定位锚点",
      "planner_strategy": "本章的视觉表达策略"
    }}
  ]
}}

约束：
1. 最多 {max_segments} 段，按原文顺序；核心章节通常 2~3 段。
2. title 必须是 4~12 字的凝练栏目标题，使用名词短语或有记忆点的短句；
   禁止摘要句，禁止“通过/展示/讲述/结合/制造悬念/铺垫/收尾”等导演过程描述，
   禁止“段落1/章节1/核心章节/总结升华”等结构占位词。
3. raw_span 必须直接复制原文，只保留开头锚点，不要复制整章。
4. structure_label 只允许：引子、概述、核心章节、总结升华。
5. music_style 只描述音乐：例如“中快速体育纪录片纯音乐，鼓点、贝斯与铜管，紧张振奋”。
   禁止出现“适合视频/贴合画面/服务叙事/用于背景/观众/主题内容”等非音乐描述。
6. 输出 JSON，禁止 Markdown 和解释。"""

_GENERIC_TITLE_RE = re.compile(r"^(?:第?[一二三四五六七八九十\d]+[章节段]|段落|章节)\s*[一二三四五六七八九十\d]*$")


def run(
    llm: LLMClient,
    article_text: str,
    max_segments: int = 6,
    video_summary: dict[str, Any] | None = None,
    video_shots: list[dict[str, Any]] | None = None,
    asr_segments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Use one compact LLM call, then fall back to Stage1 arcs or heuristics."""
    max_segments = max(2, min(int(max_segments or 6), 8))
    summary = video_summary if isinstance(video_summary, dict) else {}
    prompt = USER_TEMPLATE.format(
        article_text=str(article_text or "")[:50000],
        video_summary_json=json.dumps(_compact_video_summary(summary), ensure_ascii=False),
        max_segments=max_segments,
    )

    try:
        result = _as_dict(llm.chat_json(SYSTEM_PROMPT, prompt, temperature=0.3))
    except Exception:
        result = {}

    segments = _extract_segments(result)
    source = "llm"
    fallback_used = False
    if not segments:
        segments = _segments_from_video_summary(
            summary,
            video_shots or [],
            asr_segments or [],
            max_segments=max_segments,
        )
        source = "stage1_concat_summary.narrative_arc"
        fallback_used = True
    if not segments:
        segments = _fallback_segments(article_text, max_segments=max_segments)
        source = "heuristic"
        fallback_used = True

    segments = _normalize_segments(segments[:max_segments])
    music_style = _normalize_music_style(
        result.get("music_style"),
        video_summary=summary,
    )
    intent = str(result.get("intent") or "").strip()
    if not intent:
        intent = str(summary.get("video_intent") or summary.get("intent") or "").strip()
    if not intent and article_text:
        intent = _derive_title(article_text, "全片主题", max_chars=40)

    return {
        "intent": intent,
        "segments": segments,
        "music_style": music_style,
        "num_segments": len(segments),
        "status": "success" if segments else "failed",
        "fallback_used": fallback_used,
        "source": source,
    }


def _compact_video_summary(video_summary: dict[str, Any]) -> dict[str, Any]:
    style = video_summary.get("global_style")
    arcs = video_summary.get("narrative_arc")
    return {
        "video_intent": video_summary.get("video_intent") or video_summary.get("intent") or "",
        "global_style": style if isinstance(style, dict) else {},
        "narrative_arc": arcs[:8] if isinstance(arcs, list) else [],
        "topic": video_summary.get("topic") or "",
    }


def _normalize_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, raw in enumerate(segments):
        if not isinstance(raw, dict):
            continue
        summary = _normalize_text(
            raw.get("summary")
            or raw.get("narration_script")
            or raw.get("raw_span")
            or ""
        )
        raw_span = _normalize_text(raw.get("raw_span") or raw.get("narration_script") or summary)
        label = _normalize_structure_label(raw.get("structure_label"))
        title = _condense_chapter_title(
            str(raw.get("title") or ""),
            summary or raw_span,
            label,
            idx,
        )
        key_points = raw.get("key_points")
        if not isinstance(key_points, list):
            key_points = []
        key_points = [str(item).strip() for item in key_points if str(item).strip()][:5]
        if not key_points and summary:
            key_points = [summary[:80]]
        strategy = str(raw.get("planner_strategy") or "").strip()
        if not strategy:
            strategy = _default_planner_strategy(label, title)

        item = {
            **raw,
            "segment_id": f"seg_{len(normalized) + 1:02d}",
            "structure_label": label,
            "title": title[:24],
            "summary": summary[:300],
            "raw_span": raw_span[:120],
            "narration_script": _normalize_text(raw.get("narration_script") or raw_span or summary),
            "key_points": key_points,
            "planner_strategy": strategy[:240],
        }
        normalized.append(item)
    return normalized


def _segments_from_video_summary(
    video_summary: dict[str, Any],
    video_shots: list[dict[str, Any]],
    asr_segments: list[dict[str, Any]],
    *,
    max_segments: int,
) -> list[dict[str, Any]]:
    """Reliable fallback: Stage1 ranges provide boundaries, summaries provide titles."""
    arcs = video_summary.get("narrative_arc") if isinstance(video_summary, dict) else None
    if not isinstance(arcs, list) or not arcs:
        return []
    segments: list[dict[str, Any]] = []
    for arc in arcs[:max_segments]:
        if not isinstance(arc, dict):
            continue
        label = _normalize_structure_label(arc.get("phase"))
        start_idx, end_idx = _arc_range(arc.get("shot_index_range"))
        raw_span = _raw_span_for_shot_range(video_shots, asr_segments, start_idx, end_idx)
        summary = str(arc.get("phase_summary") or "").strip()
        title = _condense_chapter_title("", summary or raw_span, label, len(segments))
        segments.append({
            "structure_label": label,
            "title": title,
            "summary": summary or raw_span[:160],
            "raw_span": raw_span or summary,
            "narration_script": raw_span or summary,
            "key_points": [summary] if summary else [],
            "planner_strategy": _planner_strategy_for_arc(video_summary, arc),
            "shot_index_range": [start_idx, end_idx] if start_idx >= 0 and end_idx >= start_idx else [],
        })
    return segments


def _arc_range(value: Any) -> tuple[int, int]:
    if isinstance(value, list) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            pass
    return -1, -1


def _raw_span_for_shot_range(
    video_shots: list[dict[str, Any]],
    asr_segments: list[dict[str, Any]],
    start_idx: int,
    end_idx: int,
) -> str:
    if start_idx < 0 or end_idx < start_idx or not video_shots or not asr_segments:
        return ""
    if start_idx >= len(video_shots):
        return ""
    end_idx = min(end_idx, len(video_shots) - 1)
    start_shot = video_shots[start_idx]
    end_shot = video_shots[end_idx]
    start_sec = _safe_float(start_shot.get("start_sec", start_shot.get("start_time", 0)), 0.0)
    end_sec = _safe_float(end_shot.get("end_sec", end_shot.get("end_time", 0)), start_sec)
    texts = [
        str(seg.get("text") or "").strip()
        for seg in asr_segments
        if _safe_float(seg.get("end"), 0.0) > start_sec
        and _safe_float(seg.get("start"), 0.0) < end_sec
        and str(seg.get("text") or "").strip()
    ]
    return _normalize_text(" ".join(texts))


def _planner_strategy_for_arc(video_summary: dict[str, Any], arc: dict[str, Any]) -> str:
    style = video_summary.get("global_style") if isinstance(video_summary.get("global_style"), dict) else {}
    parts = [
        str(style.get("visual_style") or "").strip(),
        str(style.get("color_tone") or "").strip(),
        str(arc.get("phase_summary") or "").strip(),
    ]
    return "；".join(part for part in parts if part)[:240]


def _music_style_from_summary(video_summary: dict[str, Any]) -> str:
    context = " ".join([
        str(video_summary.get("video_intent") or video_summary.get("intent") or ""),
        str(video_summary.get("topic") or ""),
        str(video_summary.get("title") or ""),
    ])
    style = video_summary.get("global_style") if isinstance(video_summary.get("global_style"), dict) else {}
    audio_pace = str(style.get("audio_pace") or "").strip()
    pace = audio_pace if audio_pace in {"慢速", "中慢速", "中速", "中快速", "快速"} else "中速"
    if any(word in context for word in ("足球", "篮球", "比赛", "世界杯", "体育", "运动员")):
        return f"{pace}体育纪录片纯音乐，鼓点、贝斯与铜管，紧张振奋"
    if any(word in context for word in ("宇宙", "科学", "物理", "技术", "科普", "实验")):
        return f"{pace}科普纪录片纯音乐，合成器、弦乐与电子脉冲，理性神秘"
    if any(word in context for word in ("商业", "经济", "数据", "增长", "考试", "学习")):
        return f"{pace}现代电子纯音乐，轻打击乐、合成器与钢琴，清晰有推动感"
    if any(word in context for word in ("人物", "人生", "梦想", "亲人", "励志", "成长")):
        return f"{pace}温暖纪录片纯音乐，钢琴、弦乐与轻打击乐，克制感动"
    return f"{pace}现代纪录片纯音乐，钢琴、弦乐与轻电子节拍，克制清晰"


def _normalize_music_style(value: Any, *, video_summary: dict[str, Any]) -> str:
    text = _normalize_text(value)
    if not text:
        return _music_style_from_summary(video_summary)
    text = text.replace("背景纯音乐", "纯音乐").replace("背景音乐", "纯音乐").replace("配乐", "纯音乐")
    clauses = [
        clause.strip(" ，,。；;")
        for clause in re.split(r"[，,。；;]", text)
        if clause.strip(" ，,。；;")
    ]
    banned = ("适合", "贴合", "配合", "用于", "服务", "视频", "画面", "叙事", "观众", "主题", "内容", "背景")
    kept = [clause for clause in clauses if not any(word in clause for word in banned)]
    cleaned = "，".join(kept[:4]).strip("，")
    cleaned = re.sub(r"(?:纯音乐){2,}", "纯音乐", cleaned)
    if not cleaned:
        return _music_style_from_summary(video_summary)
    if "纯音乐" not in cleaned:
        cleaned = f"{cleaned}，纯音乐"
    return cleaned[:72]


def _normalize_structure_label(value: Any) -> str:
    raw = str(value or "").strip()
    if raw in {"引子", "概述", "核心章节", "总结升华"}:
        return raw
    if any(word in raw for word in ("总结", "升华", "结尾", "收束", "行动", "价值")):
        return "总结升华"
    if any(word in raw for word in ("引子", "引入", "开篇", "开场", "钩子", "悬念")):
        return "引子"
    if any(word in raw for word in ("概述", "背景", "铺垫", "过渡")):
        return "概述"
    return "核心章节"


def _is_generic_title(title: str, structure_label: str) -> bool:
    compact = re.sub(r"\s+", "", str(title or ""))
    return (
        not compact
        or compact == structure_label
        or compact in {"引子", "概述", "核心章节", "总结升华"}
        or bool(_GENERIC_TITLE_RE.match(compact))
    )


def _condense_chapter_title(
    proposed_title: str,
    summary: str,
    structure_label: str,
    index: int,
) -> str:
    """Turn summary-like phrases into a compact 4~12 character chapter title."""
    sources = [str(proposed_title or ""), *re.split(r"[，,。；;、]", str(summary or ""))]
    candidates: list[str] = []
    for source in sources:
        candidate = _clean_title_phrase(source)
        if candidate and not _is_generic_title(candidate, structure_label):
            candidates.append(candidate)

    if not candidates:
        return f"章节{index + 1}"

    def _score(text: str) -> tuple[int, int]:
        length = len(text)
        ideal = 1 if 4 <= length <= 10 else 0
        meta_penalty = sum(
            word in text
            for word in (
                "画面", "镜头", "数据", "元素", "背景", "信息", "叙事", "价值观",
                "结果", "全过程", "伏笔", "表现", "战术动画", "实拍",
            )
        )
        return ideal * 20 - meta_penalty * 8 - abs(length - 7), -length

    best = max(candidates, key=_score)
    if len(best) > 12:
        best = best[:12].rstrip("与和及到")
    return best or f"章节{index + 1}"


def _clean_title_phrase(value: Any) -> str:
    text = re.sub(r"\s+", "", _normalize_text(value)).strip("：:，,。；;“”\" ")
    text = re.sub(
        r"^(?:本段|本章|通过|展示|讲述|介绍|描述|分析|提出|强调|告知|结合|交代|深入挖掘|"
        r"以|将|复盘|呈现|定格|致敬|暗示|为)",
        "",
        text,
    )
    replacements = (
        ("社交媒体粉丝增长数据", "粉丝暴涨"),
        ("职业生涯起点", "职业起点"),
        ("退役危机到入围世界杯的转折", "退役到世界杯"),
        ("梦想与年龄的普世价值观", "梦想不问年龄"),
        ("梦想与年龄的价值", "梦想不问年龄"),
        ("个人故事上升至", ""),
        ("球员大器晚成的背景", "大器晚成"),
        ("身价对比与神扑画面", "身价悬殊与神扑"),
        ("庆祝画面与国旗元素", "国家荣耀"),
        ("世界杯西班牙平局结果", "西班牙意外战平"),
        ("佛德角港口与球队合影", "佛德角亮相"),
        ("这支小国球队并非易与之辈", "小国劲旅"),
        ("爆冷埋下伏笔", "爆冷伏笔"),
        ("西班牙进攻受阻", "西班牙久攻不下"),
        ("0比0比分", "0比0大冷门"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    text = re.sub(r"(?:迅速)?建立.*$", "", text)
    text = re.sub(r"(?:制造)?悬念.*$", "", text)
    text = re.sub(r"为后续.*$", "", text)
    text = re.sub(r"(?:铺垫|收尾|升华|吸引观众停留|情感浓度最高).*$", "", text)
    text = re.sub(r"的(?:背景|含义|转折|基础信息)$", "", text)
    text = re.sub(r"^(?:球员|人物|本章|本段)", "", text)
    text = re.sub(r"(?:画面|镜头|元素|数据)$", "", text)
    text = text.strip("：:，,。；;、与和 ")
    return text


def _derive_title(text: str, default: str, max_chars: int = 18) -> str:
    candidate = _normalize_text(text)
    candidate = re.sub(
        r"^(?:本段|本章)?(?:讲述|介绍|描述|分析|提出|强调|告知|展示|复盘|通过|围绕)",
        "",
        candidate,
    ).strip("：:，,。 ")
    candidate = re.split(r"[。！？；\n]|，(?=.{6,})", candidate, maxsplit=1)[0].strip()
    if not candidate:
        return default
    return candidate[:max_chars]


def _default_planner_strategy(label: str, title: str) -> str:
    if label == "引子":
        return f"用高信息密度画面快速建立「{title}」的悬念与观看动机"
    if label == "总结升华":
        return f"用克制节奏和结论性画面收束「{title}」"
    return f"围绕「{title}」按原视频顺序呈现主体、证据与关键动作"


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, tuple):
        for item in value:
            if isinstance(item, dict):
                return item
            if isinstance(item, list):
                return {"segments": item}
    if isinstance(value, list):
        return {"segments": value}
    return {}


def _extract_segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    raw_segments = (
        result.get("segments")
        or result.get("article_segments")
        or result.get("paragraphs")
        or result.get("sections")
        or []
    )
    if isinstance(raw_segments, dict):
        raw_segments = list(raw_segments.values())
    if not isinstance(raw_segments, list):
        return []
    segments: list[dict[str, Any]] = []
    for item in raw_segments:
        if isinstance(item, dict):
            raw_span = item.get("raw_span")
            if isinstance(raw_span, dict):
                raw_span = ""
            segments.append({
                **item,
                "raw_span": raw_span or item.get("text") or item.get("content") or "",
                "narration_script": item.get("narration_script") or raw_span or item.get("text") or "",
            })
        elif isinstance(item, str) and item.strip():
            segments.append({"raw_span": item.strip(), "narration_script": item.strip()})
    return [seg for seg in segments if seg.get("summary") or seg.get("raw_span") or seg.get("narration_script")]


def _fallback_segments(article_text: str, max_segments: int = 6) -> list[dict[str, Any]]:
    text = re.sub(r"\s+", " ", str(article_text or "")).strip()
    if not text:
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[。！？!?])\s*", text) if s.strip()]
    target = max_segments if len(sentences) <= 1 else min(max_segments, max(2, min(6, len(sentences))))
    if len(sentences) > 1:
        chunk_size = max(1, math.ceil(len(sentences) / target))
        chunks = ["".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    else:
        chunk_size = max(1, math.ceil(len(text) / target))
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()][:max_segments]
    segments: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        structure = "引子" if idx == 0 else ("总结升华" if idx == len(chunks) - 1 else "核心章节")
        segments.append({
            "structure_label": structure,
            "title": _condense_chapter_title("", chunk, structure, idx),
            "summary": chunk[:160],
            "raw_span": chunk[:120],
            "narration_script": chunk,
            "key_points": [chunk[:80]],
            "planner_strategy": "",
        })
    return segments
