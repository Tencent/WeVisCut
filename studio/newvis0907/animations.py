"""Semantic animation planning for the newvis0907 policy.

This module turns a shot's narration into *trustworthy* on-screen emphasis
plans. It is deliberately pure and side-effect free — it never mutates the
project, never touches disk except to read a real Gold Template registry, and
never fabricates a template id. The public surface is four functions:

``classify_semantics(text) -> list[dict]``
    Detect emphasis-worthy spans of seven semantic types (``quote``,
    ``key_point``, ``parallel_list``, ``proper_noun``, ``statistic``, ``chart``,
    ``timeline``). Each item carries the *exact* source phrase, its type, a
    human reason, and a priority used to pick a single primary anchor.

``locate_phrase_time(phrase, cues, duration) -> dict``
    Map a phrase to a ``{start, end, matched}`` interval using the subtitle
    cues. Timing comes from the cue a phrase lands in (never a uniform shot
    start of 0), refined *inside* a cue by the de-punctuated character ratio,
    and can span multiple cues. A phrase that is not present fails honestly
    (``matched=False`` with ``start``/``end`` ``None``) rather than defaulting
    to 0.

``select_template(anchor, registry, current=None) -> dict``
    Choose a real ``structure_id`` / ``motion_id`` / ``visual_id`` from the
    registry using per-type preferences. It can only ever return ids that exist
    in the registry; when nothing matches (or the registry is empty / missing a
    layer) it returns a structured failure instead of inventing an id.

``build_semantic_animation(shot, assets, registry, user_emphasis=None) -> dict``
    Compose a shot's ``{primary, anchors, quality}`` plan. There is at most one
    ``primary``; an explicit ``user_emphasis`` always wins (``source="user"``)
    over auto-detected anchors (``source="auto"``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .subtitles import build_subtitle_cues, normalize_lexical

# ---------------------------------------------------------------------------
# small coercion helpers
# ---------------------------------------------------------------------------
def _as_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _as_list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else []


def _as_text(v: Any) -> str:
    return str(v or "")


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# semantic types and priorities
# ---------------------------------------------------------------------------
SEMANTIC_TYPES: tuple[str, ...] = (
    "quote",
    "key_point",
    "parallel_list",
    "proper_noun",
    "statistic",
    "chart",
    "timeline",
)

# Higher = stronger claim to being the single primary anchor.
TYPE_PRIORITY: dict[str, int] = {
    "quote": 90,
    "statistic": 85,
    "chart": 80,
    "key_point": 75,
    "timeline": 70,
    "parallel_list": 60,
    "proper_noun": 50,
}

_KEY_POINT_WORDS = (
    "核心", "关键", "真正", "必须", "意味着", "本质", "重点",
    "归根结底", "最重要", "根本", "决定性", "务必", "一定要",
)
_QUOTE_SAID = ("说", "表示", "指出", "认为", "强调", "写道", "讲")
_SEQUENCE_WORDS = ("首先", "其次", "然后", "接着", "随后", "最后", "最终", "起初", "紧接着")
_ORDINAL_WORDS = ("第一", "第二", "第三", "第四", "一是", "二是", "三是", "其一", "其二", "其三")

# number + unit → a statistic (percent, multiplier, currency, counts, time spans…)
_UNIT = r"(?:%|％|‰|倍|万|亿|千|百分点|个|人|次|年|月|日|天|周|小时|分钟|秒|℃|度|元|美元|公里|千米|米|克|吨|GB|MB|KB|TB|K|W)"
_STAT_RE = re.compile(r"\d+(?:[.,]\d+)?\s*" + _UNIT)
_YEAR_RE = re.compile(r"\d{3,4}\s*年")
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[.\-+][A-Za-z0-9]+)*")
_TITLE_RE = re.compile(r"《([^》]+)》")
_QUOTE_RE_PATTERNS = (
    re.compile(r"“([^”]+)”"),
    re.compile(r"‘([^’]+)’"),
    re.compile(r"「([^」]+)」"),
    re.compile(r"『([^』]+)』"),
    re.compile(r"\"([^\"]+)\""),
)
_HARD_PUNCT = "。！？!?；;\n"


def _clauses(text: str) -> list[str]:
    """Split into clauses on hard/soft punctuation, keeping non-empty pieces."""
    parts = re.split(r"[，,。！？!?；;：:\n、]", str(text or ""))
    return [p.strip() for p in parts if p and p.strip()]


def _sentence_of(text: str, needle: str) -> str:
    """Return the smallest clause of ``text`` that contains ``needle``."""
    for clause in _clauses(text):
        if needle in clause:
            return clause
    return str(text or "").strip()


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


def classify_semantics(text: str) -> list[dict[str, Any]]:
    """Detect emphasis-worthy spans of the seven semantic types in ``text``.

    Returns a list of ``{"phrase", "type", "reason", "priority"}`` dicts sorted
    by descending priority then source order. Gates are strict: a lone number is
    a ``statistic`` not a ``chart``; a single sequence marker is not a
    ``timeline``; a single item is not a ``parallel_list``.
    """
    source = str(text or "")
    if not source.strip():
        return []
    results: list[dict[str, Any]] = []

    def add(phrase: str, kind: str, reason: str) -> None:
        phrase = str(phrase or "").strip().strip("、，,。 ")
        if not phrase:
            return
        results.append({
            "phrase": phrase,
            "type": kind,
            "reason": reason,
            "priority": TYPE_PRIORITY.get(kind, 0),
        })

    # --- quote: explicit quotation marks, or "X说……" -----------------------
    quoted: list[str] = []
    for pat in _QUOTE_RE_PATTERNS:
        quoted.extend(m.strip() for m in pat.findall(source) if m.strip())
    for q in _dedupe_keep_order(quoted):
        add(q, "quote", "被引号包裹的引述内容")
    if not quoted:
        for verb in _QUOTE_SAID:
            m = re.search(rf"{verb}[,，:：]?\s*([^。！？!?；;\n]{{2,40}})", source)
            if m:
                add(m.group(1), "quote", f"“{verb}”后接的引述内容")
                break

    # --- key_point: explicit emphasis keywords ------------------------------
    for word in _KEY_POINT_WORDS:
        if word in source:
            add(_sentence_of(source, word), "key_point", f"包含核心关键词“{word}”")
            break

    # --- parallel_list: >= 2 parallel items ---------------------------------
    dun_items = [p for p in re.split(r"[、]", source) if p.strip()]
    ordinal_hits = [w for w in _ORDINAL_WORDS if w in source]
    if source.count("、") >= 1 and len(dun_items) >= 2:
        add(source.strip(), "parallel_list", "顿号分隔的并列项 ≥ 2")
    elif len(ordinal_hits) >= 2:
        add(source.strip(), "parallel_list", "序数并列项 ≥ 2（第一/第二…）")

    # --- proper_noun: full ASCII tokens or 《...》 titles --------------------
    proper: list[str] = []
    for tok in _ASCII_TOKEN_RE.findall(source):
        if len(tok) >= 2:
            proper.append(tok)
    proper.extend(_TITLE_RE.findall(source))
    for p in _dedupe_keep_order(proper):
        add(p, "proper_noun", "完整的 ASCII/专有名词记号")

    # --- statistic: number + unit / percent ---------------------------------
    stats = _STAT_RE.findall(source)  # findall returns unit group; use finditer for text
    stat_spans = [m.group(0).strip() for m in _STAT_RE.finditer(source)]
    for s in _dedupe_keep_order(stat_spans):
        add(s, "statistic", "数字搭配单位/百分比")

    # --- chart: >= 2 comparable quantitative values -------------------------
    if len(stat_spans) >= 2:
        add(source.strip(), "chart", "至少两个可比较的定量值")

    # --- timeline: >= 2 ordered time nodes / sequence markers ---------------
    seq_hits = [w for w in _SEQUENCE_WORDS if w in source]
    year_hits = _YEAR_RE.findall(source)
    if len(seq_hits) >= 2 or len(year_hits) >= 2:
        reason = "顺序标记 ≥ 2（首先/其次/最后…）" if len(seq_hits) >= 2 else "有序时间节点 ≥ 2"
        add(source.strip(), "timeline", reason)

    results.sort(key=lambda item: (-item["priority"], source.find(item["phrase"])))
    return results


# ---------------------------------------------------------------------------
# phrase → time, from subtitle cues
# ---------------------------------------------------------------------------
def _norm_chars(text: str) -> list[str]:
    """De-punctuated, whitespace-free, lower-cased character list of ``text``."""
    return [ch.lower() for ch in str(text or "") if ch.isalnum()]


def _cue_bounds(cue: dict[str, Any]) -> tuple[float, float]:
    start = _as_float(cue.get("start"))
    end = _as_float(cue.get("end"), start)
    if end < start:
        end = start
    return start, end


def locate_phrase_time(phrase: str, cues: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    """Locate ``phrase`` in ``cues`` and return ``{start, end, matched, ...}``.

    * Timing is taken from the cue(s) the phrase lands in — never a uniform 0.
    * Inside a cue the position is refined by the de-punctuated character ratio.
    * A phrase can span multiple cues (start from the first, end in the last).
    * A phrase that is not present returns ``matched=False`` with ``start`` and
      ``end`` ``None`` (it never silently defaults to 0).
    * Both bounds are clamped into ``[0, duration]``.
    """
    dur = max(0.0, _as_float(duration))
    norm_phrase = "".join(_norm_chars(phrase))
    fail = {"start": None, "end": None, "matched": False}
    if not norm_phrase:
        return fail

    cue_list = [c for c in _as_list(cues) if isinstance(c, dict)]
    if not cue_list:
        return fail

    # Build a global normalized string with a per-character map back to its cue.
    global_chars: list[str] = []
    index_map: list[tuple[int, int, int]] = []  # (cue_idx, pos_in_cue, cue_norm_len)
    cue_norm_len: list[int] = []
    for ci, cue in enumerate(cue_list):
        chars = _norm_chars(cue.get("text"))
        cue_norm_len.append(len(chars))
        for pos, ch in enumerate(chars):
            global_chars.append(ch)
            index_map.append((ci, pos, len(chars)))
    global_str = "".join(global_chars)

    idx = global_str.find(norm_phrase)
    if idx < 0:
        return fail
    g0 = idx
    g1 = idx + len(norm_phrase) - 1

    ci0, pos0, len0 = index_map[g0]
    ci1, pos1, len1 = index_map[g1]
    s0, e0 = _cue_bounds(cue_list[ci0])
    s1, e1 = _cue_bounds(cue_list[ci1])

    start = s0 + (pos0 / len0) * (e0 - s0) if len0 else s0
    end = s1 + ((pos1 + 1) / len1) * (e1 - s1) if len1 else e1

    if dur > 0:
        start = min(max(0.0, start), dur)
        end = min(max(0.0, end), dur)
    else:
        start = max(0.0, start)
        end = max(0.0, end)
    if end <= start:
        end = min(dur, start + 0.5) if dur > 0 else start + 0.5

    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "matched": True,
        "cue_start_index": ci0,
        "cue_end_index": ci1,
    }


# ---------------------------------------------------------------------------
# template selection — real ids only
# ---------------------------------------------------------------------------
# Preferred *real* structure ids per semantic type (checked in order), plus a
# substring fallback used to find any equivalent real id in the registry.
_STRUCTURE_PREF: dict[str, tuple[list[str], list[str]]] = {
    "quote": (["quote_card"], ["quote"]),
    "key_point": (["key_takeaway", "key_takeaway_center", "thesis_statement"],
                  ["key_takeaway", "thesis", "takeaway"]),
    "parallel_list": (["three_points", "three_points_stack", "recap_checklist", "agenda_roadmap"],
                      ["three_points", "recap", "agenda", "icon_overview", "module_breakdown"]),
    "proper_noun": (["definition_card", "definition_card_split", "analogy_card"],
                    ["definition", "analogy"]),
    "statistic": (["stat_highlight", "metric_cards"], ["stat_highlight", "metric"]),
    "chart": (["bar_chart", "line_chart", "chart_callout", "donut_pie", "metric_cards"],
              ["chart", "donut", "pie", "ranking", "comparison"]),
    "timeline": (["timeline_step", "historical_moment", "flow_steps", "vertical_flow"],
                 ["timeline", "historical", "flow", "vertical_flow", "cause_effect", "cycle"]),
}
_MOTION_PREF: dict[str, list[str]] = {
    "quote": ["mask_reveal", "scale_pop_hold"],
    "key_point": ["scale_pop_hold", "slide_in_hold"],
    "parallel_list": ["card_stack_sequence", "slide_in_hold"],
    "proper_noun": ["slide_in_hold", "scale_pop_hold"],
    "statistic": ["scale_pop_hold", "slide_in_hold"],
    "chart": ["slide_in_hold", "parallax_pan"],
    "timeline": ["slide_in_hold", "parallax_pan"],
}
_VISUAL_PREF: list[str] = ["dark_tech", "clean_saas", "warm_editorial", "minimal_academic",
                           "documentary_caption", "neon_cyber"]
_FAMILY_TEMPLATE_TYPE = {
    "dataviz": "overlay",
    "narrative": "lower_third",
    "diagram": "motion",
}


def _normalize_registry(registry: Any) -> dict[str, list[str]]:
    """Coerce many registry shapes into ``{structures, motions, visuals}`` id lists.

    Accepts: ``None``; a filesystem path to a Gold Template root; the editor
    ``{"structure_id": [...], "motion_id": [...], "visual_id": [...]}`` shape; a
    ``{"structures": ..., "motions": ..., "visuals": ...}`` shape (lists or
    id-keyed dicts); or a flat ``id -> item`` dict (treated as structures only).
    """
    if registry is None:
        return {"structures": [], "motions": [], "visuals": []}
    if isinstance(registry, (str, Path)):
        return registry_from_gold_root(str(registry))

    def _ids(value: Any) -> list[str]:
        if isinstance(value, dict):
            return [str(k) for k in value.keys()]
        if isinstance(value, list):
            out: list[str] = []
            for item in value:
                if isinstance(item, dict) and item.get("id"):
                    out.append(str(item["id"]))
                elif isinstance(item, str):
                    out.append(item)
            return out
        return []

    if isinstance(registry, dict):
        if any(k in registry for k in ("structure_id", "motion_id", "visual_id")):
            return {
                "structures": _ids(registry.get("structure_id")),
                "motions": _ids(registry.get("motion_id")),
                "visuals": _ids(registry.get("visual_id")),
            }
        if any(k in registry for k in ("structures", "motions", "visuals")):
            return {
                "structures": _ids(registry.get("structures")),
                "motions": _ids(registry.get("motions")),
                "visuals": _ids(registry.get("visuals")),
            }
        # Flat id -> item dict: treat as a structures-only registry.
        return {"structures": [str(k) for k in registry.keys()], "motions": [], "visuals": []}
    if isinstance(registry, list):
        return {"structures": _ids(registry), "motions": [], "visuals": []}
    return {"structures": [], "motions": [], "visuals": []}


def registry_from_gold_root(root: str | Path) -> dict[str, Any]:
    """Read the available real ids per layer from a Gold Template library.

    Mirrors what the renderer (``timeline_compose._resolve_*_id``) probes: an id
    only counts when its marker file exists on disk, so this can never surface
    an id the renderer would silently fall back away from. It also loads each
    structure's ``slot_schema.json`` (when present) into ``schemas`` so slots can
    be filled to the *real* schema shape (F3).
    """
    empty: dict[str, Any] = {"structures": [], "motions": [], "visuals": [], "schemas": {}}
    base = Path(str(root or "")).expanduser()
    if not str(root) or not base.exists():
        return empty
    layers = {
        "structures": ("structures", "structure.html"),
        "motions": ("motions", "motion.js"),
        "visuals": ("visuals", "tokens.css"),
    }
    out: dict[str, Any] = {}
    for key, (subdir, marker) in layers.items():
        gen = base / "generated" / subdir
        try:
            out[key] = sorted(d.name for d in gen.iterdir() if (d / marker).is_file())
        except OSError:
            out[key] = []
    schemas: dict[str, dict[str, Any]] = {}
    for sid in out.get("structures", []):
        schema_path = base / "generated" / "structures" / sid / "slot_schema.json"
        if schema_path.is_file():
            try:
                data = json.loads(schema_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            if isinstance(data, dict):
                schemas[sid] = data
    out["schemas"] = schemas
    return out


def _registry_schemas(registry: Any) -> dict[str, dict[str, Any]]:
    """Return the ``{structure_id: slot_schema}`` map for a registry, if any."""
    if isinstance(registry, (str, Path)):
        return _as_dict(registry_from_gold_root(str(registry)).get("schemas"))
    if isinstance(registry, dict):
        return _as_dict(registry.get("schemas"))
    return {}


def _pick_structure(kind: str, available: list[str], current: str) -> str:
    prefs, substrs = _STRUCTURE_PREF.get(kind, ([], []))
    present = [sid for sid in prefs if sid in available]
    # Prefer a preferred id that is not the current one (encourage diversity).
    for sid in present:
        if sid != current:
            return sid
    if present:
        return present[0]
    for needle in substrs:
        for sid in available:
            if needle in sid and sid != current:
                return sid
        for sid in available:
            if needle in sid:
                return sid
    return ""


def _pick_motion(kind: str, available: list[str]) -> str:
    for mid in _MOTION_PREF.get(kind, []):
        if mid in available:
            return mid
    return available[0] if available else ""


def _pick_visual(available: list[str], current: str) -> str:
    if current and current in available:
        return current
    for vid in _VISUAL_PREF:
        if vid in available:
            return vid
    return available[0] if available else ""


def select_template(anchor: dict[str, Any], registry: Any, current: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pick a real ``structure_id`` / ``motion_id`` / ``visual_id`` for an anchor.

    Returns ``{"ok": True, "structure_id", "motion_id", "visual_id",
    "template_type", "reason"}`` on success, and a structured
    ``{"ok": False, "reason": ...}`` when the registry is empty, is missing a
    layer, or has no id matching the anchor's type. It never invents an id.
    """
    kind = _as_text(_as_dict(anchor).get("type")) or "key_point"
    reg = _normalize_registry(registry)
    structures = reg.get("structures") or []
    motions = reg.get("motions") or []
    visuals = reg.get("visuals") or []

    if not structures or not motions or not visuals:
        return {
            "ok": False,
            "reason": "registry 缺少可用的 structure/motion/visual 层，拒绝造 ID",
            "type": kind,
        }

    cur = _as_dict(current)
    cur_structure = _as_text(cur.get("structure_id") or cur.get("template_id"))
    cur_visual = _as_text(cur.get("visual_id"))

    structure_id = _pick_structure(kind, structures, cur_structure)
    if not structure_id:
        return {
            "ok": False,
            "reason": f"registry 中没有匹配语义类型“{kind}”的真实 structure_id",
            "type": kind,
        }
    motion_id = _pick_motion(kind, motions)
    visual_id = _pick_visual(visuals, cur_visual)
    if not motion_id or not visual_id:
        return {"ok": False, "reason": "registry 缺少可用的 motion/visual", "type": kind}

    return {
        "ok": True,
        "structure_id": structure_id,
        "motion_id": motion_id,
        "visual_id": visual_id,
        "template_type": _template_type_for(structure_id),
        "reason": f"按语义类型“{kind}”从真实模板库匹配",
    }


def _template_type_for(structure_id: str) -> str:
    # Without registry metadata we key off well-known dataviz/diagram ids.
    dataviz = ("bar_chart", "line_chart", "donut_pie", "chart_callout", "metric_cards",
               "ranking_list", "stat_highlight", "case_gallery", "method_comparison")
    diagram = ("flow_steps", "vertical_flow", "cause_effect", "cycle_loop", "branching_flow",
               "layer_stack", "module_breakdown", "focus_zoom", "venn_compare")
    if any(structure_id.startswith(p) for p in dataviz):
        return "overlay"
    if any(structure_id.startswith(p) for p in diagram):
        return "motion"
    return "lower_third"


# ---------------------------------------------------------------------------
# whole-shot semantic animation plan
# ---------------------------------------------------------------------------
def _shot_narration(shot: dict[str, Any]) -> str:
    return (
        _as_text(shot.get("vocal_script"))
        or _as_text(shot.get("narration"))
        or _as_text(shot.get("subtitle_text"))
        or _as_text(shot.get("subtitle_display_text"))
    )


def _shot_duration(shot: dict[str, Any], assets: dict[str, Any]) -> float:
    audio = _as_dict(assets.get("audio_meta"))
    return (
        _as_float(audio.get("duration_sec"))
        or _as_float(shot.get("duration_sec"))
        or 0.0
    )


def resolve_cues(shot: dict[str, Any], assets: dict[str, Any], duration: float) -> list[dict[str, Any]]:
    """Return subtitle cues for a shot, generating them from narration if absent.

    Prefers real cues written by the renderer (``assets``/``shot`` ->
    ``subtitle_cues``); otherwise derives deterministic cues from the narration
    so phrase timing is always cue-based rather than a uniform shot start.
    """
    for source in (assets, shot):
        cues = [c for c in _as_list(_as_dict(source).get("subtitle_cues")) if isinstance(c, dict)]
        if cues:
            return cues
    text = _shot_narration(shot)
    if not text or duration <= 0:
        return []
    try:
        return build_subtitle_cues(text, duration)["cues"]
    except Exception:
        return []


_TITLE_SLOT_HINTS = ("headline", "title", "thesis", "quote", "term", "stat", "question",
                     "主标题", "标题", "问题", "金句", "术语")


def _clause_units(text: str) -> list[str]:
    units = [c for c in _clauses(text) if c]
    return units


def _fill_text_slot(name: str, spec: dict[str, Any], phrase: str, index: int,
                    units: list[str]) -> str:
    """Pick audience-facing text for one schema *text* slot.

    The primary (title-like) slots receive the *exact* phrase; secondary text
    slots fall back to a narration clause so mandatory slots always get sensible
    content. ``max_chars`` from the schema is honoured.
    """
    label = f"{name} {_as_text(spec.get('label'))}".lower()
    is_titleish = any(h in label for h in _TITLE_SLOT_HINTS)
    if is_titleish or (index == 0 and not units):
        text = phrase
    elif units:
        text = units[index % len(units)]
    else:
        text = phrase
    max_chars = spec.get("max_chars")
    try:
        limit = int(max_chars)
    except (TypeError, ValueError):
        limit = 0
    if limit > 0 and len(text) > limit:
        text = text[:limit]
    return text


def _fill_data_slot(spec: dict[str, Any], phrase: str, units: list[str]) -> Any:
    """Fill a JSON/list schema slot, preserving the declared shape."""
    default = spec.get("default")
    items = units or [phrase]
    if isinstance(default, list):
        if default and isinstance(default[0], dict):
            template = default[0]
            rows: list[dict[str, Any]] = []
            for i, unit in enumerate(items[: max(len(default), 1)]):
                row = dict(template)
                for k, v in row.items():
                    if isinstance(v, str):
                        row[k] = unit
                rows.append(row)
            return rows or [dict(template)]
        return items[: max(len(default), len(items))] or [phrase]
    if isinstance(default, dict):
        return dict(default)
    # No usable default: a list of phrase-derived items is the safe shape.
    return items[:3] or [phrase]


def _slots_from_schema(phrase: str, schema: dict[str, Any], narration: str) -> dict[str, Any]:
    """Map the exact phrase onto the real ``slot_schema`` (F3)."""
    schema = _as_dict(schema)
    schema_slots = _as_dict(schema.get("slots"))
    if not schema_slots:
        return {}
    data_slots = {str(x) for x in _as_list(schema.get("data_slots"))}
    units = _clause_units(narration) or ([phrase] if phrase else [])
    slots: dict[str, Any] = {}
    text_index = 0
    for name, spec in schema_slots.items():
        spec = _as_dict(spec)
        is_data = name in data_slots or _as_text(spec.get("type")).lower() == "json"
        if is_data:
            slots[name] = _fill_data_slot(spec, phrase, units)
        else:
            slots[name] = _fill_text_slot(name, spec, phrase, text_index, units)
            text_index += 1
    # Guarantee the exact phrase is present somewhere in the display slots.
    if phrase and phrase not in json.dumps(slots, ensure_ascii=False):
        for name, spec in schema_slots.items():
            if name not in data_slots and _as_text(_as_dict(spec).get("type")).lower() != "json":
                slots[name] = phrase
                break
    return slots


def _slots_for(phrase: str, kind: str, schema: dict[str, Any] | None = None,
               narration: str = "") -> dict[str, Any]:
    """Fill display slots with the *exact* phrase (audience-facing text).

    When a real structure ``schema`` is available the slots follow its keys and
    types (F3); otherwise a generic set of phrase-filled slots is used.
    """
    phrase = _as_text(phrase)
    if schema:
        filled = _slots_from_schema(phrase, schema, narration or phrase)
        if filled:
            return filled
    slots: dict[str, Any] = {"title": phrase, "headline": phrase}
    if kind == "quote":
        slots["quote_text"] = phrase
    elif kind == "statistic":
        slots["stat"] = phrase
    elif kind == "proper_noun":
        slots["term"] = phrase
    return slots


def _make_anchor(
    phrase: str,
    kind: str,
    loc: dict[str, Any],
    template: dict[str, Any],
    *,
    source: str,
    priority: int,
    reason: str,
    schema: dict[str, Any] | None = None,
    narration: str = "",
) -> dict[str, Any]:
    return {
        "phrase": phrase,
        "type": kind,
        "start_sec": loc.get("start"),
        "end_sec": loc.get("end"),
        "structure_id": _as_text(template.get("structure_id")),
        "motion_id": _as_text(template.get("motion_id")),
        "visual_id": _as_text(template.get("visual_id")),
        "template_type": _as_text(template.get("template_type")),
        "slots": _slots_for(phrase, kind, schema=schema, narration=narration),
        "source": source,
        "priority": priority,
        "reason": reason,
    }


def anchor_to_decoration(anchor: dict[str, Any], duration: float) -> dict[str, Any]:
    """Build a renderer decoration from an anchor.

    Writes ``timing.decor_start_sec`` / ``timing.decor_duration_sec`` from the
    anchor's phrase interval so both the ffmpeg ``drawtext`` overlay and the
    HyperFrames assemble step play the decoration during the spoken phrase, and
    carries the real structure/motion/visual ids plus phrase-filled slots.
    """
    anchor = _as_dict(anchor)
    dur = max(0.0, _as_float(duration))
    start = _as_float(anchor.get("start_sec"))
    end = _as_float(anchor.get("end_sec"), start)
    if end <= start:
        end = min(dur, start + min(4.0, dur or 4.0)) if dur > 0 else start + 4.0
    if dur > 0:
        start = min(max(0.0, start), dur)
        end = min(max(start, end), dur)
    decor_duration = max(0.5, end - start)
    return {
        "template_id": _as_text(anchor.get("structure_id")),
        "structure_id": _as_text(anchor.get("structure_id")),
        "motion_id": _as_text(anchor.get("motion_id")),
        "visual_id": _as_text(anchor.get("visual_id")),
        "template_type": _as_text(anchor.get("template_type")) or "lower_third",
        "slots": dict(_as_dict(anchor.get("slots"))),
        "timing": {
            "decor_start_sec": round(start, 3),
            "decor_duration_sec": round(decor_duration, 3),
        },
        "animation": {"enter": "slide_in", "duration_ms": 500, "exit": "fade_out"},
        "source": _as_text(anchor.get("source")) or "auto",
        "phrase": _as_text(anchor.get("phrase")),
        "semantic_type": _as_text(anchor.get("type")),
    }


def build_semantic_animation(
    shot: dict[str, Any],
    assets: dict[str, Any],
    registry: Any,
    user_emphasis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose ``{primary, anchors, quality}`` for a shot.

    * Auto anchors come from :func:`classify_semantics` + :func:`locate_phrase_time`
      + :func:`select_template`; only phrases that both locate in the cues and
      match a real template become anchors.
    * There is at most one ``primary`` — the highest-priority auto anchor, unless
      an explicit ``user_emphasis`` is given, which always wins with
      ``source="user"``.
    """
    shot = _as_dict(shot)
    assets = _as_dict(assets)
    duration = _shot_duration(shot, assets)
    cues = resolve_cues(shot, assets, duration)
    narration = _shot_narration(shot)
    current = _as_dict(assets.get("decoration"))
    schemas = _registry_schemas(registry)

    anchors: list[dict[str, Any]] = []
    issues: list[str] = []

    for cand in classify_semantics(narration):
        loc = locate_phrase_time(cand["phrase"], cues, duration)
        if not loc.get("matched"):
            continue
        template = select_template(cand, registry, current=current)
        if not template.get("ok"):
            issues.append(f"{cand['type']}:{template.get('reason', 'no template')}")
            continue
        anchors.append(_make_anchor(
            cand["phrase"], cand["type"], loc, template,
            source="auto", priority=int(cand.get("priority", 0)),
            reason=_as_text(cand.get("reason")),
            schema=schemas.get(_as_text(template.get("structure_id"))),
            narration=narration,
        ))

    primary: dict[str, Any] = {}
    user = _as_dict(user_emphasis)
    user_phrase = _as_text(user.get("phrase")).strip()
    if user_phrase:
        user_type = _as_text(user.get("type") or user.get("emphasis_type")) or "key_point"
        loc = locate_phrase_time(user_phrase, cues, duration)
        template = select_template({"type": user_type}, registry, current=current)
        # User override still records the anchor even if the registry is absent;
        # the caller decides whether a missing real template is fatal.
        user_template = template if template.get("ok") else {}
        user_anchor = _make_anchor(
            user_phrase, user_type, loc,
            user_template,
            source="user",
            priority=TYPE_PRIORITY.get(user_type, 100) + 1000,  # always outranks auto
            reason="用户显式指定的强调",
            schema=schemas.get(_as_text(user_template.get("structure_id"))),
            narration=narration,
        )
        user_anchor["template_ok"] = bool(template.get("ok"))
        if not template.get("ok"):
            user_anchor["template_reason"] = _as_text(template.get("reason"))
        # Replace any auto anchor with the same phrase.
        anchors = [a for a in anchors if a["phrase"] != user_phrase]
        anchors.insert(0, user_anchor)
        primary = user_anchor
    elif anchors:
        primary = max(anchors, key=lambda a: (a.get("priority", 0), -_as_float(a.get("start_sec"))))

    quality = {
        "status": "ok" if primary else "empty",
        "anchor_count": len(anchors),
        "has_primary": bool(primary),
        "issues": issues,
    }
    return {"primary": primary, "anchors": anchors, "quality": quality}


__all__ = [
    "SEMANTIC_TYPES",
    "TYPE_PRIORITY",
    "anchor_to_decoration",
    "build_semantic_animation",
    "classify_semantics",
    "locate_phrase_time",
    "registry_from_gold_root",
    "resolve_cues",
    "select_template",
]
