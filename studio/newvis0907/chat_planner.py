"""Rule-based planning and model-plan validation for newvis0907."""

from __future__ import annotations

import json
import re
from typing import Any

from . import conversation
from .hierarchy import build_hierarchy
from .locator import resolve_targets

LEGACY_OPS = [
    "delete_shot",
    "reorder_shots",
    "replace_footage",
    "set_bgm",
    "set_narration",
    "set_spotlight_slots",
    "set_subtitle",
    "swap_template",
]
NEWVIS0907_OPS = [
    "remove_animation",
    "set_emphasis",
    "replace_subclip",
    "set_subclip_timing",
    "set_shot_duration",
    "optimize_cut",
]
SUPPORTED_OPS = sorted(LEGACY_OPS + NEWVIS0907_OPS)

_OP_FIELDS: dict[str, dict[str, set[str]]] = {
    "set_narration": {"required": {"op", "shot_id", "text"}, "optional": {"visual_query"}},
    "set_subtitle": {"required": {"op", "shot_id", "text"}, "optional": set()},
    "set_spotlight_slots": {"required": {"op", "shot_id", "slots"}, "optional": set()},
    "swap_template": {"required": {"op", "shot_id"}, "optional": {"structure_id", "motion_id", "visual_id", "slots"}},
    "replace_footage": {"required": {"op", "shot_id"}, "optional": {"query", "clip_path"}},
    "reorder_shots": {"required": {"op", "shot_ids"}, "optional": set()},
    "delete_shot": {"required": {"op", "shot_id"}, "optional": set()},
    "set_bgm": {"required": {"op"}, "optional": {"query", "music_style", "volume"}},
    "remove_animation": {"required": {"op", "shot_id"}, "optional": {"phrase", "anchor_id"}},
    "set_emphasis": {"required": {"op", "shot_id", "phrase", "emphasis_type"}, "optional": set()},
    "replace_subclip": {"required": {"op", "subclip_id"}, "optional": {"query", "clip_path"}},
    "set_subclip_timing": {"required": {"op", "subclip_id"}, "optional": {"duration_sec", "mode"}},
    "set_shot_duration": {"required": {"op", "shot_id", "duration_sec"}, "optional": set()},
    "optimize_cut": {"required": {"op"}, "optional": {"scope", "policy"}},
}

_QUOTE_PATTERNS = [
    re.compile(r"“([^”]+)”"),
    re.compile(r'"([^"]+)"'),
    re.compile(r"「([^」]+)」"),
]
_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*秒")
# Ordered longest-first so "替换成" wins over "换成".
_REPLACE_MARKERS = (
    "替换成", "替换为", "更换成", "更换为", "修改成", "修改为",
    "换成", "改成", "换为", "改为", "变成", "弄成", "换一个", "换个", "换",
)
# "把第 3 个素材为海边落日" / "第 3 个画面改成海边落日": the unit noun followed by
# a change marker is the most reliable place to cut the search query.
_UNIT_QUERY_RE = re.compile(
    r"(?:镜头|分镜|素材|画面|视频|片段)\s*(?:的)?\s*"
    r"(?:改|换|替换|更换|设置|设|调整)?\s*(?:成|为|到)\s*(.+)$"
)
_FOOTAGE_WORDS = ("镜头", "分镜", "画面", "素材", "视频", "片段")
# Trailing nouns the user adds for readability but that are not part of the
# search query itself: "大海的视频" is a request for 大海.
_QUERY_TAILS = (
    "的视频素材", "的画面素材", "的视频", "的画面", "的素材", "的片段", "的镜头",
    "视频素材", "画面素材", "视频", "画面", "素材", "片段", "镜头",
)
_QUERY_LEADS = ("一个", "一段", "一条", "个", "段")

# A model reply is a *plan*, not a report: it must not claim the edit already
# happened. These completion-tense phrases are scrubbed to a neutral planning
# statement so the UI never shows an optimistic "done" before render+verify.
_COMPLETION_PHRASES = (
    "已成功",
    "修改成功",
    "修改已生效",
    "已生效",
    "已经生效",
    "已修改",
    "已经修改",
    "修改完成",
    "已完成",
    "已经完成",
    "改好了",
    "已经改好",
)
_NEUTRAL_PLAN_REPLY = "已生成修改计划，将在执行并校验后生效。"


def _sanitize_reply(reply: str) -> str:
    """Neutralise completion-tense wording in a model reply."""
    text = str(reply or "").strip()
    if not text:
        return text
    if any(phrase in text for phrase in _COMPLETION_PHRASES):
        return _NEUTRAL_PLAN_REPLY
    return text


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_text(value: Any) -> str:
    return str(value or "")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _quoted_texts(message: str) -> list[str]:
    values: list[str] = []
    for pattern in _QUOTE_PATTERNS:
        values.extend(match.strip() for match in pattern.findall(message or "") if match.strip())
    return values


_GREETINGS = ("你好", "您好", "hi", "hello", "在吗", "在么", "嗨")
_CAPABILITY_WORDS = (
    "可以改什么", "能改什么", "改什么", "能做什么", "可以做什么", "怎么用",
    "如何使用", "有什么功能", "支持什么", "帮助",
    # The same question asked casually: 干啥 / 干嘛 / 会啥 / 帮我做什么.
    "干啥", "干嘛", "干什么", "会啥", "会什么", "做啥",
    "怎么改", "怎么用你", "能帮我做什么", "可以帮我做什么", "有啥用",
)
# A bare "?" is a request for help, not a target reference.
_HELP_ONLY = ("?", "？", "??", "？？", "???", "？？？")
_CAPABILITY_REPLY = (
    "我可以按你的话改这支片子，常用的有：\n"
    "· 换素材：「第 3 个镜头换成海边落日」\n"
    "· 调时长：「第 2 个镜头调到 5 秒」\n"
    "· 改文案字幕：「把第 1 个镜头的口播改成…」\n"
    "· 动画强调：「强调“真正的效率”这句」\n"
    "· 整体优化：「整体优化一下节奏」\n"
    "定位镜头可以用序号、章节名、台词原文或时间点。"
)
_GREETING_REPLY = (
    "你好，我是这支片子的剪辑助手。\n"
    "告诉我要改哪个镜头、改成什么就行，比如「第 3 个镜头换成海边落日」。\n"
    "想知道能改哪些内容，问我「可以改什么」。"
)
_THANKS_WORDS = ("谢谢", "感谢", "辛苦了", "可以了", "好的")
_CANCEL_WORDS = ("取消", "算了", "不用改了", "先别改", "撤销这次")
_CONFUSION_WORDS = ("什么意思", "没懂", "不明白", "解释一下")
_THANKS_REPLY = "不客气。你继续告诉我要改哪个镜头、改成什么，我会先生成修改计划再执行校验。"
_CANCEL_REPLY = "好的，这次不继续改。要撤销已经生效的修改，可以点对应事务的「回滚」。"
_CONFUSION_REPLY = (
    "我的意思是：需要先定位到唯一镜头，才能安全修改。\n"
    "你可以说「第 3 个镜头换成海边落日」、"
    "「《章节名》第一个镜头改成大海」或直接引用画面对应的台词。"
)


def _small_talk_reply(message: str) -> str:
    """A direct answer for greetings and capability questions."""
    text = _as_text(message).strip().lower()
    if not text:
        return ""
    if text in _HELP_ONLY:
        return _CAPABILITY_REPLY
    if any(word in text for word in _CAPABILITY_WORDS):
        return _CAPABILITY_REPLY
    if len(text) <= 12 and any(word in text for word in _CONFUSION_WORDS):
        return _CONFUSION_REPLY
    if len(text) <= 12 and any(word in text for word in _CANCEL_WORDS):
        return _CANCEL_REPLY
    if len(text) <= 8 and any(word in text for word in _THANKS_WORDS):
        return _THANKS_REPLY
    # Greetings only count when they are the whole message, so "你好，把第三个
    # 镜头换掉" still routes to the editor.
    if len(text) <= 8 and any(text.startswith(word) for word in _GREETINGS):
        return _GREETING_REPLY
    return ""


def _clean_query(raw: str) -> str:
    """Reduce "换成…" trailing text to the searchable subject."""
    text = _as_text(raw).strip().strip("。！!，,、 ")
    for lead in _QUERY_LEADS:
        if text.startswith(lead):
            text = text[len(lead):].strip()
            break
    for tail in _QUERY_TAILS:
        if text.endswith(tail) and len(text) > len(tail):
            return text[: -len(tail)].strip()
    return text


def _footage_replacement_query(message: str) -> str:
    """The subject the user wants a shot's footage replaced with, if any."""
    text = _as_text(message)
    if not any(word in text for word in _FOOTAGE_WORDS):
        return ""
    unit = _UNIT_QUERY_RE.search(text)
    if unit:
        cleaned = _clean_query(unit.group(1))
        if cleaned:
            return cleaned
    best_index, best_marker = len(text), ""
    for marker in _REPLACE_MARKERS:
        found = text.find(marker)
        if found != -1 and (found < best_index or (found == best_index and len(marker) > len(best_marker))):
            best_index, best_marker = found, marker
    if not best_marker:
        return ""
    return _clean_query(text[best_index + len(best_marker):])


def _after_change_marker(message: str) -> str:
    text = _as_text(message)
    best_index, best_marker = len(text), ""
    for marker in _REPLACE_MARKERS + ("调成", "设为", "设置成", "设置为"):
        found = text.find(marker)
        if found != -1 and (found < best_index or (found == best_index and len(marker) > len(best_marker))):
            best_index, best_marker = found, marker
    if not best_marker:
        return ""
    return text[best_index + len(best_marker):].strip(" 。！!，,、")


#: Clauses that qualify *how* to apply an edit rather than forming its content.
#: 「改成今晚早点睡，不重新配音」— the second clause is an instruction to us, not
#: words to put on screen.
_TRAILING_QUALIFIERS = (
    "不重新配音", "不用重新配音", "不改配音", "不动配音", "配音不变",
    "只改字幕", "只改显示", "只改文字", "仅字幕", "其他不动", "别的不用改",
)


def _strip_trailing_qualifier(text: str) -> str:
    """Drop trailing how-to-apply clauses from user-authored content."""
    value = _as_text(text).strip()
    changed = True
    while changed:
        changed = False
        for qualifier in _TRAILING_QUALIFIERS:
            stripped = value.rstrip(" 。.!！")
            if stripped.endswith(qualifier):
                value = stripped[: -len(qualifier)].rstrip(" ，,、；;。.")
                changed = True
    return value.strip()


def _extract_json_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = _as_text(raw).strip()
    if not text:
        return {}
    candidates = [text]
    fence = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidates.extend(chunk.strip() for chunk in fence if chunk.strip())
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _normalize_ops(raw_ops: Any) -> list[dict[str, Any]]:
    return [item for item in _as_list(raw_ops) if isinstance(item, dict)]


def _validate_target_membership(op: dict[str, Any], allowed_target_ids: set[str], errors: list[str]) -> None:
    if "shot_id" in op and _as_text(op.get("shot_id")) not in allowed_target_ids:
        errors.append(f'未知目标 shot_id={op.get("shot_id")!r}')
    if "subclip_id" in op and _as_text(op.get("subclip_id")) not in allowed_target_ids:
        errors.append(f'未知目标 subclip_id={op.get("subclip_id")!r}')
    if "shot_ids" in op:
        values = op.get("shot_ids")
        if not isinstance(values, list) or not values:
            errors.append("shot_ids 必须是非空数组")
        else:
            for shot_id in values:
                if _as_text(shot_id) not in allowed_target_ids:
                    errors.append(f"未知目标 shot_id={shot_id!r}")


def _normalize_intent(raw: Any, *, has_ops: bool) -> str:
    text = _as_text(raw).strip().lower()
    if text in {"edit", "question", "clarify"}:
        return text
    return "edit" if has_ops else "question"


def _shot_alias_map(state: dict[str, Any]) -> dict[str, str]:
    """Map display numbers / loose labels the model may emit onto real shot ids."""
    aliases: dict[str, str] = {}
    for idx, shot in enumerate(_as_list(_as_dict(state).get("shot_plan")), start=1):
        if not isinstance(shot, dict):
            continue
        sid = _as_text(shot.get("shot_id"))
        if not sid:
            continue
        aliases[sid] = sid
        aliases[str(idx)] = sid
        aliases[f"{idx:02d}"] = sid
        aliases[f"镜头{idx}"] = sid
        aliases[f"镜头 {idx}"] = sid
        aliases[f"镜头{idx:02d}"] = sid
        aliases[f"S{idx}"] = sid
        aliases[f"s{idx}"] = sid
        aliases[f"S{idx:02d}"] = sid
    return aliases


def _rewrite_model_targets(parsed: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Rewrite display-number shot ids the LoRA often emits (03 → S03)."""
    aliases = _shot_alias_map(state)
    if not aliases:
        return parsed
    rewritten = dict(parsed)
    ops: list[dict[str, Any]] = []
    for op in _normalize_ops(parsed.get("ops")):
        item = dict(op)
        sid = _as_text(item.get("shot_id"))
        if sid and sid not in aliases.values():
            mapped = aliases.get(sid) or aliases.get(sid.lstrip("0") or sid)
            if mapped:
                item["shot_id"] = mapped
        if "shot_ids" in item and isinstance(item.get("shot_ids"), list):
            mapped_ids = []
            for raw_id in item["shot_ids"]:
                token = _as_text(raw_id)
                mapped_ids.append(aliases.get(token) or aliases.get(token.lstrip("0") or token) or token)
            item["shot_ids"] = mapped_ids
        ops.append(item)
    rewritten["ops"] = ops
    return rewritten


def validate_model_plan(raw: Any, allowed_target_ids: set[str] | list[str] | tuple[str, ...], max_ops: int = 8) -> dict:
    """Validate an assistant-produced plan against the supported op schema."""
    allowed = {str(item) for item in allowed_target_ids}
    parsed = _extract_json_dict(raw)
    ops = _normalize_ops(parsed.get("ops"))
    reply = _sanitize_reply(_as_text(parsed.get("reply")))
    intent = _normalize_intent(parsed.get("intent"), has_ops=bool(ops))
    errors: list[str] = []

    if not parsed:
        errors.append("模型计划不是合法 JSON 对象")
    if not ops and not reply:
        errors.append("模型计划既没有操作也没有回复")
    if len(ops) > max_ops:
        errors.append(f"模型计划操作数超过上限 {max_ops}")

    clean_ops: list[dict[str, Any]] = []
    for index, op in enumerate(ops):
        name = _as_text(op.get("op")).strip()
        spec = _OP_FIELDS.get(name)
        if spec is None:
            errors.append(f"第 {index + 1} 条操作不支持: {name or '<empty>'}")
            continue

        keys = set(op.keys())
        missing = sorted(spec["required"] - keys)
        extra = sorted(keys - spec["required"] - spec["optional"])
        if missing:
            errors.append(f"第 {index + 1} 条操作缺少字段: {', '.join(missing)}")
        if extra:
            errors.append(f"第 {index + 1} 条操作包含非法字段: {', '.join(extra)}")
        _validate_target_membership(op, allowed, errors)

        if name == "set_subclip_timing":
            mode = _as_text(op.get("mode")).strip().lower()
            if mode == "auto":
                op = dict(op)
                op["mode"] = "auto"
            else:
                duration = _as_float(op.get("duration_sec"), -1.0)
                if duration <= 0:
                    errors.append(f"第 {index + 1} 条操作 duration_sec 必须大于 0，或使用 mode=auto")
                else:
                    op = dict(op)
                    op["duration_sec"] = duration
        elif name in {"set_shot_duration"}:
            duration = _as_float(op.get("duration_sec"), -1.0)
            if duration <= 0:
                errors.append(f"第 {index + 1} 条操作 duration_sec 必须大于 0")
            else:
                op = dict(op)
                op["duration_sec"] = duration
        clean_ops.append(dict(op))

    if errors:
        intent = "clarify"
    elif ops:
        intent = "edit"
    return {
        "ok": not errors,
        "reply": reply,
        "ops": clean_ops if not errors else [],
        "errors": errors,
        "intent": intent,
    }


# ---------------------------------------------------------------------------
# intent consistency
# ---------------------------------------------------------------------------
# The model reads intent; this layer only checks that the ops it produced are
# about the same *kind of thing* the user asked about. It is not a second
# intent parser — it never decides what the edit is, it only refuses an edit
# that answers a different question ("换个动画" answered with a footage search
# for「更加醒目的动画装饰」).
_OP_CATEGORY = {
    "set_emphasis": "animation",
    "remove_animation": "animation",
    "swap_template": "animation",
    "set_spotlight_slots": "animation",
    "replace_footage": "footage",
    "replace_subclip": "footage",
    "set_narration": "script",
    "set_subtitle": "script",
    "set_bgm": "audio",
    "set_shot_duration": "timing",
    "set_subclip_timing": "timing",
    "optimize_cut": "timing",
    "delete_shot": "structure",
    "reorder_shots": "structure",
}
# Words that name a *kind* of change. Target words ("镜头"/"分镜") are absent on
# purpose: they say where to edit, not what to edit.
_CATEGORY_WORDS = {
    "animation": ("动画", "动效", "特效", "花字", "hyperframes", "hyperframe",
                  "装饰", "强调", "重点突出", "模板", "包装"),
    "footage": ("画面", "素材", "视频", "片段", "空镜", "b-roll", "broll", "配图"),
    "script": ("字幕", "文案", "口播", "旁白", "解说", "台词", "配音", "念"),
    "audio": ("bgm", "配乐", "背景音乐", "音乐"),
    "timing": ("时长", "节奏", "快一点", "慢一点", "拖沓", "秒"),
    "structure": ("删掉", "删除", "顺序", "重排", "调换", "挪到"),
}
#: A footage search query is a description of a *picture*. These words mean the
#: model turned an animation or caption request into a stock-footage search.
_NON_FOOTAGE_QUERY_WORDS = (
    "动画", "动效", "特效", "花字", "装饰", "hyperframes", "模板", "字幕", "配乐", "音乐",
)
#: The user asking for captions only, with the voice-over left alone.
_DISPLAY_ONLY_WORDS = (
    "只改字幕", "只改显示", "不重新配音", "不用重新配音", "不改配音", "不动配音",
    "配音不变", "只改文字", "仅字幕",
)


def _message_categories(message: str) -> set[str]:
    """The kinds of change this message unambiguously mentions."""
    text = _as_text(message).lower()
    return {
        name for name, words in _CATEGORY_WORDS.items()
        if any(word in text for word in words)
    }


def wants_display_only_subtitle(message: str) -> bool:
    """True when the user explicitly wants captions to diverge from the audio."""
    text = _as_text(message)
    return any(word in text for word in _DISPLAY_ONLY_WORDS)


def _op_conflicts(op: dict[str, Any], wanted: set[str]) -> bool:
    name = _as_text(op.get("op"))
    category = _OP_CATEGORY.get(name)
    if category is None:
        return False
    if category == "footage":
        query = _as_text(op.get("query")).lower()
        if any(word in query for word in _NON_FOOTAGE_QUERY_WORDS):
            return True
    if not wanted:
        return False
    return category not in wanted


def reconcile_ops(message: str, ops: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Drop ops that answer a different question than the user asked.

    Returns the surviving ops and a human-readable note per dropped op. A
    caption edit is also upgraded to a narration edit unless the user asked for
    captions only, because「把字幕改成…」normally means the line as spoken, and
    silently leaving the voice-over saying the old words is the worst outcome.
    """
    wanted = _message_categories(message)
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    for op in ops:
        if _op_conflicts(op, wanted):
            dropped.append(_as_text(op.get("op")))
            continue
        if _as_text(op.get("op")) == "set_subtitle" and not wants_display_only_subtitle(message):
            op = {"op": "set_narration", "shot_id": op.get("shot_id"), "text": op.get("text")}
        kept.append(op)
    return kept, dropped


def _extract_allowed_ids(locator_result: dict[str, Any]) -> set[str]:
    allowed: set[str] = set()
    for bucket in ("targets", "candidates"):
        for item in _as_list(locator_result.get(bucket)):
            if not isinstance(item, dict):
                continue
            shot_id = _as_text(item.get("shot_id"))
            subclip_id = _as_text(item.get("subclip_id"))
            if shot_id:
                allowed.add(shot_id)
            if subclip_id:
                allowed.add(subclip_id)
    return allowed


def _all_target_ids(state: dict[str, Any]) -> set[str]:
    """Every real target id the model may name in a validated plan."""
    allowed: set[str] = set()
    for shot in _as_list(_as_dict(state).get("shot_plan")):
        if isinstance(shot, dict) and _as_text(shot.get("shot_id")):
            allowed.add(_as_text(shot.get("shot_id")))
    hierarchy = build_hierarchy(state or {})
    for shot in _as_list(_as_dict(hierarchy).get("model_shots")):
        for subclip in _as_list(_as_dict(shot).get("subclips")):
            if isinstance(subclip, dict) and _as_text(subclip.get("subclip_id")):
                allowed.add(_as_text(subclip.get("subclip_id")))
    for assets in _as_dict(_as_dict(state).get("shot_assets")).values():
        if not isinstance(assets, dict):
            continue
        for carrier in (assets, _as_dict(assets.get("clip_meta"))):
            playlist = _as_dict(carrier.get("visual_playlist"))
            for clip in _as_list(playlist.get("clips")):
                if isinstance(clip, dict) and _as_text(clip.get("subclip_id")):
                    allowed.add(_as_text(clip.get("subclip_id")))
    return allowed


SYSTEM_PROMPT = """你是 WevisionCut 的剪辑助手。用户已经有一支成片，正在用中文口语跟你说想怎么改。
你的工作是理解他这句话真正想做什么，然后要么直接给出编辑操作，要么像同事一样回答他。

只输出一个 JSON 对象，不要 Markdown、不要代码块、不要任何解释文字：
{"intent":"edit|question|clarify","reply":"给用户看的中文回复","ops":[...]}

【先判断意图】
· edit —— 他要改片子，而且你能确定改哪里、改成什么。
· question —— 打招呼、道谢、取消、问你能做什么、问片子里某段是什么。直接用中文回答，ops 留空。
· clarify —— 他确实想改，但目标不唯一或缺关键信息。给 2-3 个具体候选让他挑，ops 留空。
判不准时优先 clarify，不要瞎改；但只要能合理确定目标，就直接 edit，别为了保险反复追问。
输入里的「系统定位参考」只是一个粗略提示，可能是错的，最终以用户原话和片子结构为准。

【绝对不要重复提问】这是最重要的一条。
输入里的「对话记忆」告诉你：上一轮你问了什么、用户原本想做什么、已经问过几次。
· 目标已经确定（「已锁定镜头」非空，或用户用 @ 指定过）时，永远不要再问「你指的是哪个镜头」。
  这时候缺的是改动内容，就问「这个镜头你想怎么改」，并给出可以照着说的例子。
· 用户回「是的」「对」「就是这个」「第2个」「镜头03」时，那是在回答你上一个问题，
  不是新的请求。把它和「用户原本想做的事」拼起来，直接给出 ops。
· 「已经问过几次」≥2 时禁止再 clarify：挑最像的那个镜头直接 edit，
  并在 reply 里说明「我先按镜头 XX 改，不对的话告诉我序号」。
· 用户只说了目标没说改法（例如「@镜头03 这个镜头改一下动画」里的「改一下动画」
  已经足够），能自己决定的就自己决定，不要为了保险再问一轮。

【先分清用户要改的是哪一类东西，选错类别比选错镜头更严重】
· 动画/动效/特效/花字/装饰/hyperframes/模板 → set_emphasis 或 remove_animation。
· 画面/素材/视频/片段/空镜 → replace_footage 或 replace_subclip。
· 字幕/文案/口播/旁白/台词 → set_narration（或 set_subtitle）。
· 配乐/BGM/背景音乐 → set_bgm。
· 时长/节奏 → set_shot_duration / set_subclip_timing / optimize_cut。
绝对不要跨类别：用户说「换一个更醒目的动画装饰」，那是要换动画模板，
不是去搜一段叫「更醒目的动画装饰」的视频素材。replace_footage 的 query 里
永远不该出现「动画」「装饰」「特效」「字幕」这类词——出现了就说明你选错了操作。

【动画类请求】
· 「加动画/加个动画/加 hyperframes 动画/改一下动画/换个动画/换个更醒目的动画」
  → set_emphasis，phrase 从该镜头的口播原文里挑一句最值得强调的
  （必须是原文里真实出现的字），emphasis_type 用 key_point。系统会自动换一套
  与当前不同的模板，所以「换一个」也用这个操作。
· 「去掉动画/删掉动画」→ remove_animation，不带 phrase 表示清掉该镜头全部动画。

【字幕和配音是一件事】
用户说「把字幕改成…」「改一下文案」，默认用 set_narration —— 它会同时更新
口播文案、重新配音、并让字幕跟着新文案走。只有用户明确说「只改字幕」
「不重新配音」「配音不变」时，才用 set_subtitle。

【怎么定位目标】
用户可能用这些方式指镜头，都要认得：
· 序号：第3个镜头、第三个、镜头03
· 章节：《章节名》第一个镜头、跑步机这一章节第一个镜头、第2章开头
· 台词：直接引用口播或字幕里的原句
· 时间：0:12 那里、第 12 秒左右
· 子镜头：子镜头2.1、素材子镜头7.2
· 承接上文：那个、刚才那个、还是那里、再换一个
· @提及：@镜头03、@子镜头2.1、@《章节名》
注意：
· 用户说「素材/画面/视频/片段」时，通常指的就是一个镜头，例如「第3个素材」= 第 3 个镜头。
· 出现 @提及 时以它为准，忽略同一句里其他的序号说法。
· 承接上文时看 previous_targets 和 history 里最近改过的目标。
· shot_id 只能照抄 shots 里的 shot_id（如 S03），绝不能写成 03、第三个、镜头3。
· subclip_id 只能照抄 subclips 里的 subclip_id。

【可用操作和字段】只能用下面列出的字段，多写字段会被拒绝：
· replace_footage {shot_id, query} 换整个镜头的画面素材。query 是检索词，只留画面主体，如「海边落日」「摇滚乐队」，不要带「的视频」「素材」这类词。
· replace_subclip {subclip_id, query} 只换镜头里的某一小段素材。
· set_shot_duration {shot_id, duration_sec} 改镜头时长（秒）。
· set_subclip_timing {subclip_id, duration_sec} 或 {subclip_id, mode:"auto"} 改子镜头时长。
· set_narration {shot_id, text} 改口播/旁白/解说/文案。
· set_subtitle {shot_id, text} 只改屏幕字幕。
· set_bgm {query} 换配乐/背景音乐/BGM，全片级，不带 shot_id。
· set_emphasis {shot_id, phrase, emphasis_type:"key_point"} 把某句做成重点强调。
· remove_animation {shot_id, phrase} 去掉某句上的动画。
· delete_shot {shot_id} 删镜头，系统会让用户二次确认。
· reorder_shots {shot_ids} 重排镜头，系统会让用户二次确认。
· optimize_cut {scope:"global"} 整体优化节奏。
一句话里有多件事就给多条 ops，最多 8 条。

【回复怎么写】
· edit：先「好的，我来处理：」，每条操作单独一行，例如「· 镜头 03：换成「海边落日」的画面素材」，最后补一句改动会先渲染再校验、完成后可以预览。
· clarify：说人话，直接列出最可能的两三个镜头（带序号和章节名）让他挑。禁止输出「没找到唯一目标」这类机器话。
· question：正面回答他的问题，可以顺带给一个可以照着说的例子。
· 任何时候都不许说「已修改成功」「已生效」「改好了」，只能说会处理、将执行、完成后可预览。

【常见对话，照这个思路处理】
· 「你好」→ question：打个招呼，一句话说明怎么用。
· 「你可以干啥」「能改什么」「？」→ question：说明能换素材、调时长、改口播字幕、换配乐、做强调动画、整体优化。
· 「第3个镜头换成海边落日」→ edit：replace_footage S03。
· 「你帮我修改第3个素材为海边落日」→ 同上，素材就是镜头。
· 「跑步机这一章节第一个镜头换成大海」→ 先定位到那一章，再取该章第一个镜头。
· 「@镜头03 换成摇滚乐队」→ edit：目标就是 S03。
· 「那个也换成夜市灯火」→ 用 previous_targets 里的目标继续改。
· 「第2个镜头调到5秒」→ set_shot_duration。
· 「把第1个镜头口播改成…」→ set_narration；只说字幕就用 set_subtitle。
· 「配乐换成轻快电子乐」→ set_bgm。
· 「强调“真正的效率”这句」→ set_emphasis；「把动画“真正的效率”删掉」→ remove_animation。
· 「删掉第4个镜头」→ delete_shot。
· 「整体优化一下节奏」→ optimize_cut。
· 「后面那个画面不好看」→ clarify：列出最可能的两三个镜头。
· 「算了」「取消」→ question：确认这次不改，并说明可以回滚已生效的修改。
· 「谢谢」→ question：简短回应。
· 「什么意思」「没懂」→ question：把上一条解释清楚，并给一句可以直接照说的话。
· 他想要的事情你做不到（比如换配音音色、加片头包装）→ question：说明目前能改的范围，并给个替代做法。"""


def _shot_context(state: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The film's structure, shaped for the model: chapters and numbered shots."""
    hierarchy = build_hierarchy(state or {})
    raw_shots = [row for row in _as_list(_as_dict(state).get("shot_plan")) if isinstance(row, dict)]
    titles = {
        _as_text(seg.get("segment_id")): _as_text(seg.get("title") or seg.get("structure_label"))
        for seg in _as_list(_as_dict(state).get("segments"))
        if isinstance(seg, dict)
    }

    chapters: list[dict[str, Any]] = []
    for chapter in _as_list(_as_dict(hierarchy).get("chapters")):
        if not isinstance(chapter, dict):
            continue
        seg_id = _as_text(chapter.get("segment_id"))
        chapters.append({
            "display_index": int(chapter.get("display_index", 0) or 0),
            "segment_id": seg_id,
            "title": _as_text(chapter.get("title")) or titles.get(seg_id, seg_id),
        })

    shots: list[dict[str, Any]] = []
    for idx, shot_view in enumerate(_as_list(_as_dict(hierarchy).get("model_shots")), start=1):
        if not isinstance(shot_view, dict):
            continue
        sid = _as_text(shot_view.get("shot_id"))
        if not sid:
            continue
        raw = next((row for row in raw_shots if _as_text(row.get("shot_id")) == sid), {})
        seg_id = _as_text(shot_view.get("segment_id"))
        row: dict[str, Any] = {
            "display_number": _as_text(shot_view.get("display_number")) or f"{idx:02d}",
            "shot_id": sid,
            "chapter": titles.get(seg_id, seg_id),
            "start": round(_as_float(shot_view.get("start")), 2),
            "duration": round(_as_float(shot_view.get("duration"), _as_float(raw.get("duration_sec"))), 2),
            "narration": _as_text(shot_view.get("narration") or raw.get("vocal_script") or raw.get("subtitle_text"))[:160],
            "visual_query": _as_text(raw.get("visual_query")),
        }
        subclips = [
            {
                "subclip_id": _as_text(sub.get("subclip_id")),
                "display_number": _as_text(sub.get("display_number")),
                "query": _as_text(sub.get("query"))[:60],
            }
            for sub in _as_list(shot_view.get("subclips"))
            if isinstance(sub, dict) and _as_text(sub.get("subclip_id"))
        ]
        if subclips:
            row["subclips"] = subclips
        shots.append(row)

    if not chapters:
        for idx, seg in enumerate(_as_list(_as_dict(state).get("segments")), start=1):
            if isinstance(seg, dict) and _as_text(seg.get("segment_id")):
                chapters.append({
                    "display_index": idx,
                    "segment_id": _as_text(seg.get("segment_id")),
                    "title": _as_text(seg.get("title")) or _as_text(seg.get("segment_id")),
                })
    return chapters, shots


def build_model_plan_prompt(
    state: dict,
    message: str,
    previous_targets: Any = None,
    history: Any = None,
    memory: Any = None,
) -> tuple[str, str]:
    """Prompt the LoRA model to read the user's intent and answer or plan."""
    chapters, shots = _shot_context(state)
    locator = resolve_targets(state, message, previous_targets=previous_targets)
    turns = []
    for item in _as_list(history)[-8:]:
        if not isinstance(item, dict):
            continue
        role = _as_text(item.get("role"))
        content = _as_text(item.get("content")).strip()
        if role in {"user", "assistant"} and content:
            turns.append({"role": role, "content": content[:400]})

    memory_context = memory.prompt_context() if hasattr(memory, "prompt_context") else _as_dict(memory)

    user = json.dumps(
        {
            "用户这句话": message,
            "最近对话": turns,
            "对话记忆": memory_context,
            "上次改过的目标": previous_targets or [],
            "系统定位参考": {
                "status": locator.get("status"),
                "targets": locator.get("targets"),
                "candidates": locator.get("candidates"),
                "scope": locator.get("scope"),
            },
            "片子章节": chapters,
            "片子镜头": shots,
        },
        ensure_ascii=False,
    )
    return SYSTEM_PROMPT, user


def _find_shot_duration(state: dict[str, Any], shot_id: str) -> float:
    hierarchy = _as_list(_as_dict(state).get("shot_plan"))
    for shot in hierarchy:
        if isinstance(shot, dict) and _as_text(shot.get("shot_id")) == shot_id:
            return _as_float(shot.get("duration_sec"))
    return 0.0


def _needs_confirm(ops: list[dict[str, Any]], state: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    for op in ops:
        name = _as_text(op.get("op"))
        if name in {"delete_shot", "reorder_shots"} and name not in flags:
            flags.append(name)
        if name == "set_shot_duration":
            shot_id = _as_text(op.get("shot_id"))
            old_duration = _find_shot_duration(state, shot_id)
            new_duration = _as_float(op.get("duration_sec"))
            if old_duration > 0 and abs(new_duration - old_duration) / old_duration > 0.2 and name not in flags:
                flags.append(name)
    return flags


_candidate_label = conversation.candidate_label


def _clarify_from_locator(message: str, locator_result: dict[str, Any]) -> dict:
    """Ask *which shot* — only ever used when the target is genuinely unknown."""
    status = locator_result.get("status")
    candidates = _as_list(locator_result.get("candidates"))
    labels = [_candidate_label(item) for item in candidates[:3]]
    labels = [label for label in labels if label]
    if labels:
        head = "这几个镜头都对得上，你要改哪一个？" if status == "multiple" \
            else "我还不能确定你指的是哪个镜头，最接近的是这几个："
        reply = head + "\n" + "\n".join(f"· {label}" for label in labels) + "\n直接点一个，或者回我序号就行。"
    else:
        reply = (
            "我还没定位到你要改的地方。可以这样说：\n"
            "· 「第 3 个镜头」或「《章节名》第一个镜头」\n"
            "· 直接引用台词，例如「“真正的效率”这句」\n"
            "· 或者说时间点，例如「0:12 那里」"
        )
    return {
        "intent": "clarify",
        "reply": reply,
        "ops": [],
        "targets": [],
        "candidates": candidates,
        "located": None,
        "confidence": locator_result.get("confidence", 0.0),
        "needs_confirm": [],
        "scope": locator_result.get("scope", {}),
    }


def _clarify_missing_action(message: str, target: dict[str, Any], scope: Any = None) -> dict:
    """Ask *what to change* on a shot we have already pinned down.

    Re-asking "which shot?" once the shot is known is what turns a two-turn
    exchange into a loop: the user answers the question, the answer carries no
    new target reference, and the same question comes back.
    """
    label = _candidate_label(target)
    reply = (
        f"已经定位到{label}。这个镜头你想怎么改？\n"
        "· 换画面：「换成海边落日」\n"
        "· 调时长：「调到 5 秒」\n"
        "· 加动画：「加个强调动画」或指定要强调的那句话\n"
        "· 改文案：「口播改成…」"
    )
    return {
        "intent": "clarify",
        "reply": reply,
        "ops": [],
        "targets": [],
        "candidates": [target],
        "located": target,
        "confidence": 0.95,
        "needs_confirm": [],
        "scope": _as_dict(scope),
    }


class _Blanks(dict):
    """Format map that renders a missing field as empty rather than raising.

    ``remove_animation`` without a phrase means "clear every animation on this
    shot", so the sentence template must survive the absent field.
    """

    def __missing__(self, key: str) -> str:
        return ""


_OP_SENTENCE = {
    "replace_footage": "换成「{query}」的画面素材",
    "replace_subclip": "把子镜素材换成「{query}」",
    "remove_animation": "去掉动画「{phrase}」",
    "set_emphasis": "把「{phrase}」做成重点强调",
    "set_shot_duration": "时长调整为 {duration_sec} 秒",
    "set_subclip_timing": "子镜时长调整为 {duration_sec} 秒",
    "set_narration": "改写口播文案并重新配音",
    "set_subtitle": "只改屏幕字幕（配音不变）",
    "optimize_cut": "对整体节奏做一次优化",
}


def _describe_plan(ops: list[dict[str, Any]], locator_result: dict[str, Any]) -> str:
    """A plain-language summary of what will change, per target."""
    numbers: dict[str, str] = {}
    for bucket in ("targets", "candidates"):
        for item in _as_list(locator_result.get(bucket)):
            if isinstance(item, dict) and _as_text(item.get("shot_id")):
                numbers[_as_text(item["shot_id"])] = _as_text(item.get("display_number"))

    lines: list[str] = []
    for op in ops:
        name = _as_text(op.get("op"))
        template = _OP_SENTENCE.get(name)
        action = (
            template.format_map(_Blanks({k: _as_text(v) for k, v in op.items()})).replace("「」", "").strip()
            if template
            else name
        )
        shot_id = _as_text(op.get("shot_id"))
        label = numbers.get(shot_id) or _as_text(op.get("subclip_id")) or shot_id
        lines.append(f"· {('镜头 ' + label + '：') if label else ''}{action}")
    if not lines:
        return "已生成修改计划，将在执行并校验后生效。"
    return "好的，我来处理：\n" + "\n".join(lines) + "\n改动会先渲染再校验，完成后你可以直接预览。"


def _append_scope(message: str, scope: dict[str, Any]) -> str:
    title = _as_text(_as_dict(scope).get("title"))
    return f"《{title}》这章{message}" if title else message


def _resolve_for_phrase(state: dict[str, Any], message: str, scope: dict[str, Any], previous_targets: Any) -> dict:
    return resolve_targets(state, _append_scope(message, scope), previous_targets=previous_targets)


_ANIMATION_WORDS = ("动画", "动效", "特效", "花字", "hyperframes", "hyperframe")
_ANIMATION_REMOVE_WORDS = ("删", "删掉", "删除", "去掉", "取消", "不要", "拿掉", "关掉")
_ANIMATION_ADD_WORDS = (
    "加", "添加", "增加", "来个", "来一个", "做个", "做一个", "上个", "配个",
    "换", "改", "换个", "换一个", "重做", "调整", "优化", "弄个", "搞个",
)


def _wants_animation(message: str) -> bool:
    text = _as_text(message).lower()
    return any(word in text for word in _ANIMATION_WORDS)


def _shot_text(state: dict[str, Any], shot_id: str) -> str:
    for shot in _as_list(_as_dict(state).get("shot_plan")):
        if isinstance(shot, dict) and _as_text(shot.get("shot_id")) == shot_id:
            for key in ("vocal_script", "subtitle_text", "subtitle_display_text"):
                text = _as_text(shot.get(key)).strip()
                if text:
                    return text
    return ""


def _auto_emphasis_phrase(state: dict[str, Any], shot_id: str) -> tuple[str, str]:
    """Pick the phrase worth animating on a shot the user named without quoting.

    「@镜头03 加一个动画」names a shot but no phrase, and ``set_emphasis``
    refuses a phrase that is not really in the narration. Running the same
    classifier the generation pipeline uses keeps the choice consistent with
    how the film was decorated in the first place.
    """
    from . import animations as animations_mod

    text = _shot_text(state, shot_id)
    if not text:
        return "", ""
    for item in animations_mod.classify_semantics(text):
        phrase = _as_text(_as_dict(item).get("phrase")).strip()
        if phrase and phrase in text:
            return phrase, _as_text(item.get("type")) or "key_point"
    clauses = [part.strip() for part in re.split(r"[，,。！？!?；;：:\n、]", text) if part.strip()]
    return (max(clauses, key=len) if clauses else text.strip()), "key_point"


def _animation_ops(state: dict[str, Any], message: str, target: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn a bare animation request on a known shot into a real op."""
    shot_id = _as_text(_as_dict(target).get("shot_id"))
    if not shot_id:
        return []
    text = _as_text(message)
    if any(word in text for word in _ANIMATION_REMOVE_WORDS):
        return [{"op": "remove_animation", "shot_id": shot_id}]
    if not any(word in text for word in _ANIMATION_ADD_WORDS):
        return []
    phrase, kind = _auto_emphasis_phrase(state, shot_id)
    if not phrase:
        return []
    return [{
        "op": "set_emphasis",
        "shot_id": shot_id,
        "phrase": phrase,
        "emphasis_type": kind or "key_point",
    }]


def _rule_ops(state: dict[str, Any], message: str, previous_targets: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, dict[str, Any] | None]:
    ops: list[dict[str, Any]] = []
    locators: list[dict[str, Any]] = []
    root_scope = resolve_targets(state, message, previous_targets=previous_targets).get("scope", {})
    quoted = _quoted_texts(message)

    remove_phrase = quoted[0] if ("动画" in message and quoted and any(word in message for word in ("删", "删掉", "删除", "去掉"))) else ""
    emphasis_phrase = quoted[-1] if quoted and any(word in message for word in ("强调", "重点突出", "突出")) else ""

    if remove_phrase:
        locator = _resolve_for_phrase(state, f'动画“{remove_phrase}”', root_scope, previous_targets)
        locators.append(locator)
        if locator.get("status") != "unique":
            return [], locators, locator
        shot_id = _as_list(locator.get("targets"))[0]["shot_id"]
        ops.append({"op": "remove_animation", "shot_id": shot_id, "phrase": remove_phrase.rstrip("…。.")})

    if emphasis_phrase:
        locator = _resolve_for_phrase(state, f'强调“{emphasis_phrase}”', root_scope, previous_targets)
        locators.append(locator)
        if locator.get("status") not in {"unique", "multiple"}:
            return [], locators, locator
        if locator.get("status") == "multiple":
            return [], locators, locator
        shot_id = _as_list(locator.get("targets"))[0]["shot_id"]
        ops.append(
            {
                "op": "set_emphasis",
                "shot_id": shot_id,
                "phrase": emphasis_phrase.rstrip("…。."),
                "emphasis_type": "key_point",
            }
        )

    if ops:
        return ops, locators, None

    # An animation request that names a shot but quotes no phrase: pick the
    # phrase from the shot's own narration rather than asking again.
    if _wants_animation(message):
        locator = resolve_targets(state, message, previous_targets=previous_targets)
        locators.append(locator)
        if locator.get("status") != "unique":
            return [], locators, locator
        animation_ops = _animation_ops(state, message, _as_list(locator.get("targets"))[0])
        if animation_ops:
            return animation_ops, locators, None
        return [], locators, None

    if "整体优化" in message or "全片优化" in message or "优化一下" == message.strip():
        ops.append({"op": "optimize_cut", "scope": "global"})
        return ops, locators, None

    if any(word in message.lower() for word in ("bgm", "背景音乐", "配乐", "音乐")):
        query = _after_change_marker(message) or message
        if any(word in message for word in ("换", "改", "设置", "设为", "来一首", "加")):
            ops.append({"op": "set_bgm", "query": _clean_query(query)})
            return ops, locators, None

    if any(word in message for word in ("删掉", "删除", "去掉")) and "镜头" in message:
        locator = resolve_targets(state, message, previous_targets=previous_targets)
        locators.append(locator)
        if locator.get("status") != "unique":
            return [], locators, locator
        target = _as_list(locator.get("targets"))[0]
        ops.append({"op": "delete_shot", "shot_id": target["shot_id"]})
        return ops, locators, None

    if any(word in message for word in ("口播", "旁白", "解说", "文案")) and any(word in message for word in _REPLACE_MARKERS):
        locator = resolve_targets(state, message, previous_targets=previous_targets)
        locators.append(locator)
        if locator.get("status") != "unique":
            return [], locators, locator
        target = _as_list(locator.get("targets"))[0]
        text = _strip_trailing_qualifier(_after_change_marker(message))
        if text:
            ops.append({"op": "set_narration", "shot_id": target["shot_id"], "text": text})
            return ops, locators, None

    if "字幕" in message and any(word in message for word in _REPLACE_MARKERS):
        locator = resolve_targets(state, message, previous_targets=previous_targets)
        locators.append(locator)
        if locator.get("status") != "unique":
            return [], locators, locator
        target = _as_list(locator.get("targets"))[0]
        text = _strip_trailing_qualifier(_after_change_marker(message))
        if text:
            # Changing the caption normally means changing the line: leaving the
            # voice-over on the old wording ships a cut whose audio contradicts
            # its captions. Only an explicit "只改字幕" keeps them apart.
            name = "set_subtitle" if wants_display_only_subtitle(message) else "set_narration"
            ops.append({"op": name, "shot_id": target["shot_id"], "text": text})
            return ops, locators, None

    if "子镜头" in message and any(word in message for word in ("换成", "替换成", "换掉")):
        locator = resolve_targets(state, message, previous_targets=previous_targets)
        locators.append(locator)
        if locator.get("status") != "unique":
            return [], locators, locator
        target = _as_list(locator.get("targets"))[0]
        query = ""
        for marker in ("换成", "替换成", "换掉"):
            if marker in message:
                query = message.split(marker, 1)[1].strip(" 。")
                break
        ops.append({"op": "replace_subclip", "subclip_id": target["subclip_id"], "query": query})
        return ops, locators, None

    if "子镜头" not in message:
        footage_query = _footage_replacement_query(message)
        if footage_query and not _DURATION_RE.search(message):
            locator = resolve_targets(state, message, previous_targets=previous_targets)
            locators.append(locator)
            if locator.get("status") != "unique":
                return [], locators, locator
            target = _as_list(locator.get("targets"))[0]
            ops.append(
                {"op": "replace_footage", "shot_id": target["shot_id"], "query": footage_query}
            )
            return ops, locators, None

    if "子镜头" in message and ("调到" in message or "改到" in message or "时长" in message):
        locator = resolve_targets(state, message, previous_targets=previous_targets)
        locators.append(locator)
        if locator.get("status") != "unique":
            return [], locators, locator
        duration_match = _DURATION_RE.search(message)
        duration = _as_float(duration_match.group(1), 0.0) if duration_match else 0.0
        target = _as_list(locator.get("targets"))[0]
        ops.append({"op": "set_subclip_timing", "subclip_id": target["subclip_id"], "duration_sec": duration})
        return ops, locators, None

    if "镜头" in message and ("调到" in message or "改到" in message or "时长" in message):
        locator = resolve_targets(state, message, previous_targets=previous_targets)
        locators.append(locator)
        if locator.get("status") != "unique":
            return [], locators, locator
        duration_match = _DURATION_RE.search(message)
        duration = _as_float(duration_match.group(1), 0.0) if duration_match else 0.0
        target = _as_list(locator.get("targets"))[0]
        ops.append({"op": "set_shot_duration", "shot_id": target["shot_id"], "duration_sec": duration})
        return ops, locators, None

    locator = resolve_targets(state, message, previous_targets=previous_targets)
    locators.append(locator)
    if locator.get("status") != "unique":
        return [], locators, locator
    return [], locators, None


def _empty_turn(intent: str, reply: str, locator: dict[str, Any] | None = None) -> dict[str, Any]:
    loc = locator or {}
    return {
        "intent": intent,
        "reply": reply,
        "ops": [],
        "targets": [],
        "candidates": _as_list(loc.get("candidates")) if intent == "clarify" else [],
        "located": (_as_list(loc.get("targets")) or [None])[0],
        "confidence": loc.get("confidence", 1.0 if intent == "question" else 0.0),
        "needs_confirm": [],
        "scope": loc.get("scope", {}),
    }


def _edit_turn(
    ops: list[dict[str, Any]],
    reply: str,
    locator: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    targets: list[str] = []
    for op in ops:
        if "shot_id" in op:
            targets.append(_as_text(op.get("shot_id")))
        elif "subclip_id" in op:
            targets.append(_as_text(op.get("subclip_id")))
    return {
        "intent": "edit",
        "reply": reply or _describe_plan(ops, locator),
        "ops": ops,
        "targets": targets,
        "candidates": [],
        "located": (_as_list(locator.get("targets")) or _as_list(locator.get("candidates")) or [None])[0],
        "confidence": locator.get("confidence", 1.0),
        "needs_confirm": _needs_confirm(ops, state),
        "scope": locator.get("scope", {}),
    }


def _memory_anchor(memory: Any) -> dict[str, Any] | None:
    """The shot the conversation is already about, if any."""
    if memory is None:
        return None
    pending = getattr(memory, "pending", None)
    if pending is not None and getattr(pending, "target", None):
        return pending.target
    focus = getattr(memory, "focus", None)
    return focus if isinstance(focus, dict) and focus else None


def _autocommit(
    state: dict[str, Any],
    message: str,
    locator_result: dict[str, Any],
    memory: Any,
) -> dict[str, Any] | None:
    """Stop asking and act on the best candidate we have.

    Reached only after the same question has been asked ``MAX_CLARIFY_STREAK``
    times. Guessing one shot is recoverable — the user corrects it in a sentence
    — whereas another round of the same question is not.
    """
    pending = getattr(memory, "pending", None)
    # The shots we actually offered the user outrank a fresh fuzzy search: the
    # user has been choosing from that list, so the answer is in it.
    pool = (
        _as_list(locator_result.get("targets"))
        or (list(pending.candidates) if pending is not None else [])
        or _as_list(locator_result.get("candidates"))
    )
    target = next((item for item in pool if isinstance(item, dict) and item.get("shot_id")), None)
    target = target or _memory_anchor(memory)
    if not target:
        return None

    instruction = (pending.instruction if pending is not None else "") or message
    token = conversation.candidate_token(target)
    merged = f"{token} {instruction}".strip()
    previous = [{"shot_id": _as_text(target.get("shot_id")), "subclip_id": _as_text(target.get("subclip_id"))}]
    ops, locators, blocked = _rule_ops(state, merged, previous)
    if blocked is not None or not ops:
        return None
    locator = locators[-1] if locators else locator_result
    turn = _edit_turn(ops, "", locator, state)
    turn["located"] = target
    turn["reply"] = (
        f"我先按{_candidate_label(target)}来改，如果不是这个镜头，告诉我序号我马上换。\n"
        + turn["reply"]
    )
    return turn


def plan_edit(
    state: dict,
    message: str,
    previous_targets: Any = None,
    model_plan: Any = None,
    memory: Any = None,
) -> dict:
    """Prefer a validated model plan; fall back to deterministic rules.

    ``memory`` is an optional :class:`conversation.Memory`. The planner stays
    stateless with respect to it — memory only decides *which* clarification to
    ask (or whether to stop asking at all), never what the edit is.
    """
    raw_message = _as_text(message).strip()
    if not raw_message:
        return _empty_turn("clarify", "请说清楚要改哪一章、哪一句或哪个子镜头。")

    locator = resolve_targets(state, raw_message, previous_targets=previous_targets)
    rule_ops, locators, blocked = _rule_ops(state, raw_message, previous_targets)
    primary_locator = locators[-1] if locators else locator
    unique_rule_ops = rule_ops if blocked is None and rule_ops else []

    def _fallback(locator_result: dict[str, Any]) -> dict[str, Any]:
        """Every path that cannot produce ops funnels through here.

        Including the model's own ``clarify`` turns: a model that decides to ask
        a question has no memory of how many times it has already asked it, so
        letting its reply through unchecked reopens exactly the loop this
        function exists to close.

        Order matters: act if we have asked enough, otherwise ask the *narrow*
        question (what to change) whenever the shot is already known, and only
        ask "which shot" when the shot really is unknown.
        """
        if memory is not None and memory.should_autocommit(raw_message):
            committed = _autocommit(state, raw_message, locator_result, memory)
            if committed is not None:
                return committed
        located = (_as_list(locator_result.get("targets")) or [None])[0]
        if isinstance(located, dict) and located.get("shot_id"):
            return _clarify_missing_action(raw_message, located, locator_result.get("scope"))
        anchor = _memory_anchor(memory)
        # 「是的」「第2个」 answer a question; searching the narration for them
        # produces nonsense candidates and restarts the loop.
        if anchor and (conversation.is_pure_pick(raw_message)
                       or not _as_list(locator_result.get("candidates"))):
            return _clarify_missing_action(raw_message, anchor, locator_result.get("scope"))
        return _clarify_from_locator(raw_message, locator_result)

    if model_plan is not None:
        rewritten = _rewrite_model_targets(_extract_json_dict(model_plan), state)
        allowed_ids = _extract_allowed_ids(primary_locator) | _all_target_ids(state)
        validated = validate_model_plan(rewritten or model_plan, allowed_target_ids=allowed_ids)
        model_ops, dropped = reconcile_ops(raw_message, validated["ops"]) if validated["ok"] else ([], [])
        if validated["ok"] and model_ops:
            # A reply written for the ops we just discarded would describe an
            # edit that is not happening, so re-narrate from what survived.
            reply = validated["reply"] if not dropped else ""
            return _edit_turn(
                model_ops,
                reply or _describe_plan(model_ops, primary_locator),
                primary_locator,
                state,
            )
        if dropped and unique_rule_ops:
            # The model answered a different question; the rules answered this
            # one. Prefer being about the right thing over being model-authored.
            return _edit_turn(
                unique_rule_ops,
                _describe_plan(unique_rule_ops, primary_locator),
                primary_locator,
                state,
            )
        if unique_rule_ops:
            # Only narrate the edit with the model's words when the model also
            # decided this was an edit. Its clarify text asks a question, and
            # pairing "which shot did you mean?" with an edit we are about to
            # perform is worse than saying nothing.
            model_reply = (
                validated.get("reply")
                if validated.get("ok") and validated.get("intent") == "edit"
                else ""
            )
            return _edit_turn(
                unique_rule_ops,
                model_reply or _describe_plan(unique_rule_ops, primary_locator),
                primary_locator,
                state,
            )
        if validated["ok"] and validated["reply"] and validated.get("intent") == "question":
            return _empty_turn("question", validated["reply"], primary_locator)
        small_talk = _small_talk_reply(raw_message)
        if small_talk:
            return _empty_turn("question", small_talk)
        return _fallback(blocked if blocked is not None else primary_locator)

    small_talk = _small_talk_reply(raw_message)
    if small_talk:
        return _empty_turn("question", small_talk)
    if unique_rule_ops:
        return _edit_turn(unique_rule_ops, _describe_plan(unique_rule_ops, primary_locator), primary_locator, state)
    return _fallback(blocked if blocked is not None else primary_locator)
