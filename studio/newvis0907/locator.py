"""Deterministic target resolution for newvis0907 editing."""

from __future__ import annotations

import re
from typing import Any

from .hierarchy import build_hierarchy

_QUOTE_PATTERNS = [
    re.compile(r"“([^”]+)”"),
    re.compile(r'"([^"]+)"'),
    re.compile(r"「([^」]+)」"),
]
_TIME_MMSS_RE = re.compile(r"(\d+):(\d{1,2})")
_TIME_SEC_RE = re.compile(r"(?<![\d:])(\d+(?:\.\d+)?)\s*秒(?:左右)?")
_CN_DIGITS = {"零": 0, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}
_NUM_TOKEN = r"\d+|[零一两二三四五六七八九十百]+"
# Users say "第3个镜头" but just as often "第3个素材/画面/视频/片段" — all of
# them name the same model shot.
_SHOT_UNITS = r"镜头|分镜|镜|素材|画面|视频|片段"
_SHOT_ORDINAL_RE = re.compile(rf"第?\s*({_NUM_TOKEN})\s*(?:个|条|段)?(?:{_SHOT_UNITS})")
_SHOT_SIMPLE_RE = re.compile(rf"(?:镜头|分镜)\s*({_NUM_TOKEN})")
# Explicit targets the UI inserts when the user picks a shot with "@" or drags a
# timeline clip into the composer. These outrank every other reference in the
# same sentence.
_MENTION_SUBCLIP_RE = re.compile(r"@\s*(?:素材)?子镜头?\s*(\d+)\s*[.．]\s*(\d+)")
_MENTION_SHOT_RE = re.compile(rf"@\s*(?:镜头|分镜)\s*({_NUM_TOKEN})")
_MENTION_SHOT_ID_RE = re.compile(r"@\s*([A-Za-z]{1,3}\d{1,3})(?![\w.])")
_MENTION_CHAPTER_RE = re.compile(rf"@\s*(?:章节|段落|章)\s*({_NUM_TOKEN})")
_MENTION_CHAPTER_TITLE_RE = re.compile(r"@\s*《([^》]+)》")
_CHAPTER_ORDINAL_RE = re.compile(rf"第\s*({_NUM_TOKEN})\s*(?:章|章节|段|部分)")
# Wordings that mark the preceding text as a chapter reference, e.g.
# "跑步机上的原地踏步这一章节…".
_CHAPTER_SUFFIXES = (
    "这一章节", "这个章节", "这一章", "这一段", "这个部分", "这部分",
    "这章", "这段", "章节", "部分",
)
_SUBCLIP_RE = re.compile(r"(?:素材)?子镜头\s*(\d+)\.(\d+)")
_ALL_WORDS = ("全部", "所有", "都")
_PREVIOUS_WORDS = ("刚才那个", "还是那里", "还是那个", "前面那个")


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


def parse_number(token: str) -> int:
    """Read an Arabic or Chinese numeral such as ``3``/``三``/``十二``."""
    raw = str(token or "").strip()
    if raw.isdigit():
        return int(raw)
    if not raw or any(ch not in _CN_DIGITS and ch not in "十百" for ch in raw):
        return 0
    total, section, seen = 0, 0, False
    for ch in raw:
        if ch in _CN_DIGITS:
            section = _CN_DIGITS[ch]
            seen = True
        elif ch == "十":
            section = (section or 1) * 10
            seen = True
        elif ch == "百":
            total += (section or 1) * 100
            section = 0
            seen = True
    return total + section if seen else 0


def _quoted_texts(message: str) -> list[str]:
    values: list[str] = []
    for pattern in _QUOTE_PATTERNS:
        values.extend(match.strip() for match in pattern.findall(message or "") if match.strip())
    return values


def _trim_phrase(phrase: str) -> str:
    out = str(phrase or "").strip()
    out = out.rstrip("…。.")
    return out.strip()


def _contains_any(message: str, words: tuple[str, ...] | list[str]) -> bool:
    return any(word in message for word in words)


def _char_overlap_score(needle: str, haystack: str) -> float:
    left = {ch for ch in needle if not ch.isspace()}
    right = {ch for ch in haystack if not ch.isspace()}
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left), 1)


def _build_scope(chapter: dict[str, Any] | None) -> dict[str, Any]:
    if not chapter:
        return {}
    return {
        "segment_id": chapter["segment_id"],
        "title": chapter["title"],
        "display_index": chapter["display_index"],
        "start": chapter["start"],
        "end": chapter["end"],
    }


def _candidate_payload(
    shot: dict[str, Any],
    *,
    matched_text: str,
    reason: str,
    subclip: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "shot_id": shot["shot_id"],
        "display_number": shot["display_number"],
        "segment_id": shot["segment_id"],
        "chapter_title": shot["chapter_title"],
        "start": shot["start"],
        "end": shot["end"],
        "matched_text": matched_text,
        "reason": reason,
    }
    if subclip is not None:
        payload["subclip_id"] = subclip["subclip_id"]
        payload["subclip_display_number"] = subclip["display_number"]
    return payload


def strip_mentions(message: str) -> str:
    """Drop ``@…`` target tokens so only the user's instruction text remains."""
    text = _as_text(message)
    for pattern in (
        _MENTION_SUBCLIP_RE,
        _MENTION_SHOT_RE,
        _MENTION_CHAPTER_TITLE_RE,
        _MENTION_CHAPTER_RE,
        _MENTION_SHOT_ID_RE,
    ):
        text = pattern.sub(" ", text)
    return " ".join(text.split())


def _mention_chapter(message: str, chapters: list[dict[str, Any]]) -> dict[str, Any] | None:
    title = _MENTION_CHAPTER_TITLE_RE.search(message or "")
    if title:
        for chapter in chapters:
            if chapter["title"] == title.group(1):
                return chapter
    ordinal = _MENTION_CHAPTER_RE.search(message or "")
    if ordinal:
        wanted = parse_number(ordinal.group(1))
        for chapter in chapters:
            if chapter["display_index"] == wanted:
                return chapter
    return None


def _mention_ranked(
    message: str,
    shots: list[dict[str, Any]],
    scoped_shots: list[dict[str, Any]],
) -> list[tuple[float, dict[str, Any]]]:
    """Resolve an explicit ``@`` target to exactly one shot or subclip."""
    subclip_match = _MENTION_SUBCLIP_RE.search(message or "")
    if subclip_match:
        wanted = f"{int(subclip_match.group(1))}.{int(subclip_match.group(2))}"
        for shot in shots:
            for subclip in shot["subclips"]:
                if subclip["display_number"] == wanted:
                    return [(1.0, _candidate_payload(
                        shot, matched_text=f"@子镜头{wanted}",
                        reason="mention_subclip", subclip=subclip))]
        return []

    shot_id_match = _MENTION_SHOT_ID_RE.search(message or "")
    if shot_id_match:
        wanted_id = shot_id_match.group(1)
        for shot in shots:
            if shot["shot_id"].casefold() == wanted_id.casefold():
                return [(1.0, _candidate_payload(
                    shot, matched_text=f"@{shot['shot_id']}", reason="mention_shot_id"))]

    ordinal_match = _MENTION_SHOT_RE.search(message or "")
    if ordinal_match:
        wanted = parse_number(ordinal_match.group(1))
        ordered = sorted(scoped_shots or shots, key=lambda shot: shot["start"])
        for shot in ordered:
            if shot["display_number"].lstrip("0") == str(wanted):
                return [(1.0, _candidate_payload(
                    shot, matched_text=f"@镜头{wanted}", reason="mention_shot"))]
        if 1 <= wanted <= len(ordered):
            return [(1.0, _candidate_payload(
                ordered[wanted - 1], matched_text=f"@镜头{wanted}", reason="mention_shot"))]
    return []


def _extract_scope(message: str, chapters: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not message:
        return None

    quoted = re.findall(r"《([^》]+)》", message)
    for title in quoted:
        for chapter in chapters:
            if chapter["title"] == title:
                return chapter

    ordinal = _CHAPTER_ORDINAL_RE.search(message)
    if ordinal:
        wanted = parse_number(ordinal.group(1))
        for chapter in chapters:
            if chapter["display_index"] == wanted:
                return chapter

    for chapter in sorted(chapters, key=lambda item: len(item["title"]), reverse=True):
        title = chapter["title"]
        if not title:
            continue
        if any(f"{title}{suffix}" in message for suffix in _CHAPTER_SUFFIXES):
            return chapter
    return None


def _index_state(state: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = build_hierarchy(state or {})
    chapters_by_segment = {
        _as_text(ch["segment_id"]): {
            "segment_id": _as_text(ch["segment_id"]),
            "title": _as_text(ch["title"]),
            "display_index": int(ch.get("display_index", 0) or 0),
            "start": _as_float(ch.get("start")),
            "end": _as_float(ch.get("end")),
        }
        for ch in _as_list(payload.get("chapters"))
        if isinstance(ch, dict)
    }
    shot_plan = [row for row in _as_list(_as_dict(state).get("shot_plan")) if isinstance(row, dict)]
    shot_assets = _as_dict(_as_dict(state).get("shot_assets"))

    shots: list[dict[str, Any]] = []
    for idx, shot_view in enumerate(_as_list(payload.get("model_shots"))):
        if not isinstance(shot_view, dict):
            continue
        raw_shot = shot_plan[idx] if idx < len(shot_plan) else {}
        shot_id = _as_text(shot_view.get("shot_id"))
        raw_shot_id = _as_text(_as_dict(raw_shot).get("shot_id"))
        assets = _as_dict(shot_assets.get(raw_shot_id) or shot_assets.get(shot_id))
        chapter = chapters_by_segment.get(
            _as_text(shot_view.get("segment_id")),
            {
                "segment_id": _as_text(shot_view.get("segment_id")),
                "title": _as_text(shot_view.get("segment_id")),
                "display_index": 0,
                "start": _as_float(shot_view.get("start")),
                "end": _as_float(shot_view.get("end")),
            },
        )

        semantic = _as_dict(_as_dict(raw_shot).get("semantic_animations"))
        primary = _as_dict(semantic.get("primary"))
        decoration = _as_dict(assets.get("decoration"))
        slots = _as_dict(decoration.get("slots"))
        slot_values = [_as_text(value) for value in slots.values() if _as_text(value)]

        subclips = []
        for sub in _as_list(shot_view.get("subclips")):
            if isinstance(sub, dict):
                subclips.append(
                    {
                        "subclip_id": _as_text(sub.get("subclip_id")),
                        "display_number": _as_text(sub.get("display_number")),
                        "start": _as_float(sub.get("start")),
                        "end": _as_float(sub.get("end")),
                        "query": _as_text(sub.get("query")),
                    }
                )

        shots.append(
            {
                "shot_id": shot_id,
                "display_number": _as_text(shot_view.get("display_number")),
                "segment_id": _as_text(shot_view.get("segment_id")),
                "chapter_title": chapter["title"],
                "chapter_display_index": chapter["display_index"],
                "start": _as_float(shot_view.get("start")),
                "end": _as_float(shot_view.get("end")),
                "duration": _as_float(shot_view.get("duration"), _as_float(shot_view.get("end")) - _as_float(shot_view.get("start"))),
                "narration": _as_text(shot_view.get("narration")) or _as_text(_as_dict(raw_shot).get("vocal_script")),
                "display_subtitle": _as_text(shot_view.get("display_subtitle")),
                "animation_type": _as_text(shot_view.get("animation_type")),
                "animation_phrase": _as_text(primary.get("phrase")),
                "slot_values": slot_values,
                "subclips": subclips,
            }
        )
    chapters = sorted(chapters_by_segment.values(), key=lambda item: item["display_index"] or 9999)
    return shots, chapters


def _previous_target(previous_targets: Any, shots: list[dict[str, Any]]) -> dict[str, Any] | None:
    entries = [row for row in _as_list(previous_targets) if isinstance(row, dict)]
    if not entries:
        return None
    last = entries[-1]
    shot_id = _as_text(last.get("shot_id"))
    subclip_id = _as_text(last.get("subclip_id"))
    shot = next((row for row in shots if row["shot_id"] == shot_id), None)
    if not shot:
        return None
    subclip = None
    if subclip_id:
        subclip = next((row for row in shot["subclips"] if row["subclip_id"] == subclip_id), None)
    return _candidate_payload(
        shot,
        matched_text=subclip_id or shot["display_subtitle"] or shot["narration"],
        reason="previous_target",
        subclip=subclip,
    )


def _within_scope(shots: list[dict[str, Any]], scope: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not scope:
        return list(shots)
    return [shot for shot in shots if shot["segment_id"] == scope["segment_id"]]


def _time_candidates(shots: list[dict[str, Any]], second: float) -> list[tuple[float, dict[str, Any]]]:
    out: list[tuple[float, dict[str, Any]]] = []
    for shot in shots:
        if shot["start"] <= second <= shot["end"]:
            out.append((0.99, _candidate_payload(shot, matched_text=f"{second:g}s", reason="time_range")))
        else:
            distance = min(abs(shot["start"] - second), abs(shot["end"] - second))
            score = max(0.0, 0.7 - distance / 60.0)
            out.append((score, _candidate_payload(shot, matched_text=f"{second:g}s", reason="time_nearby")))
    return out


def _phrase_candidates(
    shots: list[dict[str, Any]],
    phrase: str,
    *,
    animation_only: bool,
) -> list[tuple[float, dict[str, Any]]]:
    exact: list[tuple[float, dict[str, Any]]] = []
    fallback: list[tuple[float, dict[str, Any]]] = []
    for shot in shots:
        animation_phrase = shot["animation_phrase"]
        if phrase and animation_phrase and phrase in animation_phrase:
            exact.append((1.0, _candidate_payload(shot, matched_text=animation_phrase, reason="animation_phrase")))
            continue

        slot_hit = next((value for value in shot["slot_values"] if phrase and phrase in value), "")
        if phrase and slot_hit:
            exact.append((0.99, _candidate_payload(shot, matched_text=slot_hit, reason="animation_slot")))
            continue

        if not animation_only:
            for text in (shot["display_subtitle"], shot["narration"]):
                if phrase and text and phrase in text:
                    exact.append((0.97, _candidate_payload(shot, matched_text=phrase, reason="exact_text")))
                    break
            else:
                corpus = " ".join(
                    part for part in [shot["display_subtitle"], shot["narration"], animation_phrase, " ".join(shot["slot_values"])] if part
                )
                score = _char_overlap_score(phrase, corpus)
                if score > 0:
                    fallback.append((0.5 + score * 0.3, _candidate_payload(shot, matched_text=corpus[:80], reason="fuzzy_text")))
        elif phrase:
            corpus = " ".join(part for part in [animation_phrase, " ".join(shot["slot_values"])] if part)
            score = _char_overlap_score(phrase, corpus)
            if score > 0:
                fallback.append((0.45 + score * 0.2, _candidate_payload(shot, matched_text=corpus[:80], reason="fuzzy_animation")))

    ranked = exact if exact else fallback
    ranked.sort(key=lambda item: (-item[0], item[1]["start"], item[1]["display_number"]))
    return ranked


def _finalize(
    *,
    status: str,
    ranked: list[tuple[float, dict[str, Any]]],
    scope: dict[str, Any] | None,
    explicit_all: bool,
) -> dict[str, Any]:
    candidates = [item[1] for item in ranked[:5]]
    targets: list[dict[str, Any]] = []
    confidence = ranked[0][0] if ranked else 0.0

    if status == "unique" and ranked:
        targets = [ranked[0][1]]
    elif status == "multiple" and explicit_all:
        if ranked:
            top_score = ranked[0][0]
            targets = [item[1] for item in ranked if abs(item[0] - top_score) < 1e-9]
            candidates = targets
    return {
        "status": status,
        "targets": targets,
        "candidates": candidates,
        "confidence": round(confidence, 4),
        "scope": _build_scope(scope),
    }


def resolve_targets(state: dict, message: str, previous_targets: Any = None) -> dict:
    """Resolve the user's free-form reference to real hierarchy ids only."""
    shots, chapters = _index_state(state or {})
    raw_message = _as_text(message).strip()
    scoped_chapter = _mention_chapter(raw_message, chapters) or _extract_scope(raw_message, chapters)
    scoped_shots = _within_scope(shots, scoped_chapter)
    explicit_all = _contains_any(raw_message, _ALL_WORDS)

    if not raw_message:
        return {"status": "none", "targets": [], "candidates": [], "confidence": 0.0, "scope": _build_scope(scoped_chapter)}

    # An explicit "@target" is the user pointing at the shot, so it settles the
    # question before any ordinal or phrase elsewhere in the sentence.
    mention_ranked = _mention_ranked(raw_message, shots, scoped_shots)
    if mention_ranked:
        return _finalize(status="unique", ranked=mention_ranked, scope=scoped_chapter, explicit_all=explicit_all)

    if "@" in raw_message:
        # The mention referenced something that no longer exists; fall through
        # on the remaining instruction text instead of matching "@" as prose.
        raw_message = strip_mentions(raw_message) or raw_message

    if _contains_any(raw_message, _PREVIOUS_WORDS):
        previous = _previous_target(previous_targets, shots)
        if previous is not None:
            return {
                "status": "unique",
                "targets": [previous],
                "candidates": [previous],
                "confidence": 0.98,
                "scope": _build_scope(scoped_chapter),
            }

    subclip_match = _SUBCLIP_RE.search(raw_message)
    if subclip_match:
        wanted = f"{int(subclip_match.group(1))}.{int(subclip_match.group(2))}"
        for shot in shots:
            for subclip in shot["subclips"]:
                if subclip["display_number"] == wanted:
                    ranked = [(1.0, _candidate_payload(shot, matched_text=wanted, reason="subclip_display_number", subclip=subclip))]
                    return _finalize(status="unique", ranked=ranked, scope=scoped_chapter, explicit_all=explicit_all)

    # A quoted phrase is a more specific target than "the start of this chapter",
    # so when the user quotes something, defer to phrase resolution below instead
    # of collapsing to the earliest shot in scope.
    has_quoted_phrase = bool(_quoted_texts(raw_message))
    if scoped_shots and not has_quoted_phrase and ("开头" in raw_message or "开始" in raw_message) and ("章" in raw_message or "部分" in raw_message or "《" in raw_message):
        earliest = min(scoped_shots, key=lambda shot: (shot["start"], shot["display_number"]))
        ranked = [(0.99, _candidate_payload(earliest, matched_text=earliest["chapter_title"], reason="chapter_start"))]
        return _finalize(status="unique", ranked=ranked, scope=scoped_chapter, explicit_all=explicit_all)

    ordinal_match = _SHOT_ORDINAL_RE.search(raw_message) or _SHOT_SIMPLE_RE.search(raw_message)
    if ordinal_match:
        wanted = parse_number(ordinal_match.group(1))
        ordered = sorted(scoped_shots or shots, key=lambda shot: shot["start"])
        if 1 <= wanted <= len(ordered):
            ranked = [(0.99, _candidate_payload(ordered[wanted - 1], matched_text=str(wanted), reason="shot_ordinal"))]
            return _finalize(status="unique", ranked=ranked, scope=scoped_chapter, explicit_all=explicit_all)

    mmss = _TIME_MMSS_RE.search(raw_message)
    sec = _TIME_SEC_RE.search(raw_message)
    if mmss or sec:
        second = (
            int(mmss.group(1)) * 60 + int(mmss.group(2))
            if mmss
            else _as_float(sec.group(1))
        )
        ranked = _time_candidates(scoped_shots or shots, second)
        ranked.sort(key=lambda item: (-item[0], item[1]["start"]))
        if ranked and ranked[0][0] >= 0.99:
            return _finalize(status="unique", ranked=ranked, scope=scoped_chapter, explicit_all=explicit_all)
        return _finalize(status="none", ranked=ranked, scope=scoped_chapter, explicit_all=explicit_all)

    phrases = [_trim_phrase(value) for value in _quoted_texts(raw_message) if _trim_phrase(value)]
    phrase = phrases[-1] if phrases else ""
    if phrase:
        ranked = _phrase_candidates(scoped_shots or shots, phrase, animation_only=("动画" in raw_message))
        if not ranked:
            return _finalize(status="none", ranked=[], scope=scoped_chapter, explicit_all=explicit_all)
        top_score = ranked[0][0]
        if top_score < 0.9:
            return _finalize(status="none", ranked=ranked, scope=scoped_chapter, explicit_all=explicit_all)
        top = [item for item in ranked if abs(item[0] - top_score) < 1e-9]
        if len(top) == 1:
            return _finalize(status="unique", ranked=ranked, scope=scoped_chapter, explicit_all=explicit_all)
        return _finalize(status="multiple", ranked=ranked, scope=scoped_chapter, explicit_all=explicit_all)

    corpus_ranked: list[tuple[float, dict[str, Any]]] = []
    for shot in scoped_shots or shots:
        corpus = " ".join(part for part in [shot["chapter_title"], shot["display_subtitle"], shot["narration"]] if part)
        score = _char_overlap_score(raw_message, corpus)
        if score > 0:
            corpus_ranked.append((0.35 + score * 0.3, _candidate_payload(shot, matched_text=corpus[:80], reason="fuzzy_text")))
    corpus_ranked.sort(key=lambda item: (-item[0], item[1]["start"]))
    return _finalize(status="none", ranked=corpus_ranked, scope=scoped_chapter, explicit_all=explicit_all)
