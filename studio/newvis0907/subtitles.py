"""Deterministic subtitle formatting for the 0907 policy."""

from __future__ import annotations

from dataclasses import dataclass

QUESTION_HINTS = (
    "如何",
    "怎样",
    "为什么",
    "吗",
    "呢",
    "是否",
    "能否",
    "可否",
    "何时",
    "哪里",
    "哪儿",
    "谁",
    "什么",
    "多少",
    "几时",
    "难道",
)
SEMANTIC_CONNECTORS = (
    "但是",
    "但",
    "不过",
    "而是",
    "然而",
    "然后",
    "再看",
    "再",
    "并且",
    "而且",
    "同时",
    "所以",
    "因此",
    "最后",
)
HARD_PUNCT = set("。！？!?…")
SOFT_PUNCT = set("，；：,;:")
ASCII_PUNCT = set(".,!?;:\"'()[]{}")
OPENERS = {"（": "）", "(": ")", "【": "】", "[": "]", "《": "》", "“": "”", "‘": "’", "「": "」", "『": "』"}
CLOSERS = {v: k for k, v in OPENERS.items()}
ASCII_TOKEN_CHARS = set("._+-/%")
NUMBER_UNIT_SUFFIXES = set("年月日天周次倍级版集章%％℃")
TRAILING_DASHES = set("—-－")


@dataclass(frozen=True)
class _Token:
    text: str
    kind: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return len(self.text)


def _compact_text(text: str) -> str:
    return "".join(str(text or "").split())


def _tokenize(text: str) -> list[_Token]:
    source = str(text or "")
    tokens: list[_Token] = []
    idx = 0
    while idx < len(source):
        ch = source[idx]
        if ch.isspace():
            start = idx
            while idx < len(source) and source[idx].isspace():
                idx += 1
            tokens.append(_Token(source[start:idx], "space", start, idx))
            continue
        if ch in OPENERS or ch in CLOSERS or ch in HARD_PUNCT or ch in SOFT_PUNCT or ch in ASCII_PUNCT:
            tokens.append(_Token(ch, "punct", idx, idx + 1))
            idx += 1
            continue
        if ord(ch) < 128 and (ch.isalnum() or ch == "%"):
            start = idx
            idx += 1
            while idx < len(source):
                cur = source[idx]
                if ord(cur) < 128 and (cur.isalnum() or cur in ASCII_TOKEN_CHARS):
                    idx += 1
                    continue
                break
            while idx < len(source) and source[idx] in NUMBER_UNIT_SUFFIXES:
                idx += 1
            tokens.append(_Token(source[start:idx], "text", start, idx))
            continue
        tokens.append(_Token(ch, "text", idx, idx + 1))
        idx += 1
    return tokens


def normalize_lexical(text: str) -> str:
    """Drop punctuation and whitespace without changing lexical character order."""
    return "".join(tok.text for tok in _tokenize(_compact_text(text)) if tok.kind == "text")


def equivalent_to_narration(display: str, narration: str) -> bool:
    return normalize_lexical(display) == normalize_lexical(narration)


def _looks_like_question(text: str) -> bool:
    compact = _compact_text(text)
    if not compact:
        return False
    if any(mark in compact for mark in ("？", "?")):
        return True
    return any(hint in compact for hint in QUESTION_HINTS)


def _deterministic_punctuate(text: str) -> str:
    compact = _compact_text(text)
    if not compact:
        return ""
    if compact.endswith("......") or compact.endswith("……"):
        return compact
    tail = compact[-1]
    if tail in HARD_PUNCT:
        return compact
    base = compact.rstrip("".join(SOFT_PUNCT | TRAILING_DASHES))
    if not base:
        base = compact
    if base.endswith("......") or base.endswith("……"):
        return base
    if base[-1] in HARD_PUNCT:
        return base
    return base + ("？" if _looks_like_question(base) else "。")


def _extract_punctuator_text(result: object) -> str:
    if isinstance(result, dict):
        return _compact_text(str(result.get("text", "") or ""))
    return _compact_text(str(result or ""))


def punctuate_chinese(text: str, punctuator=None) -> dict:
    """Apply an optional punctuator, but fall back if lexical characters drift."""
    source = _compact_text(text)
    fallback_used = False
    display = ""
    if punctuator is not None:
        try:
            display = _extract_punctuator_text(punctuator(source))
        except Exception:
            display = ""
        if display and equivalent_to_narration(display, source):
            return {"text": display, "valid": True, "fallback_used": False}
        fallback_used = True
    display = _deterministic_punctuate(source)
    return {
        "text": display,
        "valid": equivalent_to_narration(display, source),
        "fallback_used": fallback_used,
    }


def _candidate_score(prev_text: str, next_text: str, used_len: int, limit: int) -> tuple[int, int]:
    if prev_text and prev_text[-1] in HARD_PUNCT:
        return (400, used_len)
    if prev_text and prev_text[-1] in SOFT_PUNCT:
        return (300, used_len)
    if any(next_text.startswith(word) for word in SEMANTIC_CONNECTORS):
        return (220, used_len)
    if any(prev_text.endswith(word) for word in SEMANTIC_CONNECTORS):
        return (200, used_len)
    return (100 - abs(limit - used_len), used_len)


def _best_split_index(text: str, limit: int, prefer_balanced: bool) -> int:
    compact = _compact_text(text)
    tokens = [tok for tok in _tokenize(compact) if tok.kind != "space"]
    if not tokens or len(compact) <= limit:
        return len(compact)

    boundary_levels: dict[int, int] = {0: 0}
    stack: list[str] = []
    for pos, ch in enumerate(compact, start=1):
        if ch in OPENERS:
            stack.append(ch)
        elif ch in CLOSERS and stack and stack[-1] == CLOSERS[ch]:
            stack.pop()
        boundary_levels[pos] = len(stack)

    used = 0
    best_any: tuple[tuple[int, int], int] | None = None
    best_balanced: tuple[tuple[int, int], int] | None = None
    for idx, token in enumerate(tokens[:-1], start=1):
        used += token.length
        if used > limit:
            break
        split_at = token.end
        prev_text = compact[:split_at]
        next_text = compact[split_at:]
        if not next_text:
            continue
        score = _candidate_score(prev_text, next_text, used, limit)
        current = (score, split_at)
        if best_any is None or current[0] > best_any[0]:
            best_any = current
        if boundary_levels.get(split_at, 0) == 0:
            if best_balanced is None or current[0] > best_balanced[0]:
                best_balanced = current

    picked = best_balanced if prefer_balanced and best_balanced is not None else best_any
    if picked is not None:
        return picked[1]

    used = 0
    last_safe = 0
    for token in tokens:
        if used + token.length > limit:
            break
        used += token.length
        last_safe = token.end
    if last_safe:
        return last_safe
    first = tokens[0]
    return first.end or min(len(compact), limit)


def _split_greedy(text: str, limit: int) -> list[str]:
    compact = _compact_text(text)
    if not compact:
        return []
    parts: list[str] = []
    rest = compact
    while len(rest) > limit:
        split_at = _best_split_index(rest, limit, prefer_balanced=True)
        if split_at <= 0 or split_at >= len(rest):
            split_at = _best_split_index(rest, limit, prefer_balanced=False)
        head = rest[:split_at].strip()
        if not head:
            head = rest[:split_at or limit]
            split_at = len(head)
        parts.append(head)
        rest = rest[split_at:].strip()
    if rest:
        parts.append(rest)
    return parts


def _split_sentences(text: str) -> list[str]:
    compact = _compact_text(text)
    if not compact:
        return []
    parts: list[str] = []
    stack: list[str] = []
    start = 0
    for idx, ch in enumerate(compact):
        if ch in OPENERS:
            stack.append(ch)
        elif ch in CLOSERS and stack and stack[-1] == CLOSERS[ch]:
            stack.pop()
        if not stack and ch in HARD_PUNCT:
            part = compact[start:idx + 1].strip()
            if part:
                parts.append(part)
            start = idx + 1
    tail = compact[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _split_into_lines(text: str, max_chars_per_line: int, max_lines: int) -> list[str]:
    compact = _compact_text(text)
    if not compact:
        return []
    if _is_single_unsplittable_token(compact):
        return [compact]
    if len(compact) <= max_chars_per_line:
        return [compact]
    if max_lines <= 1:
        return [compact]

    boundary = _best_split_index(compact, max_chars_per_line, prefer_balanced=False)
    candidates = []
    if 0 < boundary < len(compact):
        candidates.append(boundary)
    fallback = min(len(compact), max_chars_per_line)
    if fallback not in candidates:
        candidates.append(fallback)

    for split_at in candidates:
        head = compact[:split_at].strip()
        tail = compact[split_at:].strip()
        if not head or not tail:
            continue
        tail_lines = _split_into_lines(tail, max_chars_per_line, max_lines - 1)
        if tail_lines:
            merged = [head, *tail_lines]
            if len(merged) <= max_lines:
                return merged
    return [compact]


def _is_single_unsplittable_token(text: str) -> bool:
    tokens = [tok for tok in _tokenize(text) if tok.kind != "space"]
    return len(tokens) == 1 and tokens[0].kind == "text"


def _lines_fit(lines: list[str], limit: int) -> bool:
    for line in lines:
        if len(line) <= limit:
            continue
        if not _is_single_unsplittable_token(line):
            return False
    return True


def _normalize_cue_capacity(text: str, max_chars_per_line: int, max_lines: int) -> list[str]:
    compact = _compact_text(text)
    if not compact:
        return []
    lines = _split_into_lines(compact, max_chars_per_line, max_lines)
    if "".join(lines) == compact and len(lines) <= max_lines and _lines_fit(lines, max_chars_per_line):
        return [compact]
    if _is_single_unsplittable_token(compact):
        return [compact]
    split_limit = max_chars_per_line
    split_at = _best_split_index(compact, split_limit, prefer_balanced=True)
    if split_at <= 0 or split_at >= len(compact):
        split_at = _best_split_index(compact, split_limit, prefer_balanced=False)
    if split_at <= 0 or split_at >= len(compact):
        split_at = min(len(compact), max_chars_per_line * max_lines)
    head = compact[:split_at].strip()
    tail = compact[split_at:].strip()
    if not head or not tail:
        return [compact]
    return [
        *_normalize_cue_capacity(head, max_chars_per_line, max_lines),
        *_normalize_cue_capacity(tail, max_chars_per_line, max_lines),
    ]


def _allocate_cue_times(cues: list[str], duration: float) -> list[tuple[float, float]]:
    total_duration = max(0.0, float(duration or 0.0))
    if not cues:
        return []
    if total_duration <= 0:
        return [(0.0, 0.0) for _ in cues]

    weights = [max(1, len(_compact_text(cue))) for cue in cues]
    total_weight = sum(weights) or 1
    times: list[tuple[float, float]] = []
    cursor = 0.0
    for idx, weight in enumerate(weights):
        if idx == len(weights) - 1:
            end = total_duration
        else:
            end = total_duration * sum(weights[:idx + 1]) / total_weight
        end = min(total_duration, max(cursor, end))
        times.append((round(cursor, 3), round(end, 3)))
        cursor = end
    if times:
        start, _ = times[-1]
        times[-1] = (start, round(total_duration, 3))
    return times


def build_subtitle_cues(
    text: str,
    duration: float,
    max_chars_per_line: int = 14,
    max_lines: int = 2,
    punctuator=None,
) -> dict:
    """Build 0907 subtitle cues with deterministic punctuation and cue timing."""
    punctuated = punctuate_chinese(text, punctuator=punctuator)
    display_text = str(punctuated.get("text", "") or "")
    max_chars_per_line = max(1, int(max_chars_per_line or 14))
    max_lines = max(1, int(max_lines or 2))
    max_chars_per_cue = max_chars_per_line * max_lines

    sentence_units = _split_sentences(display_text) or ([display_text] if display_text else [])
    cue_texts: list[str] = []
    for unit in sentence_units:
        for part in _split_greedy(unit, max_chars_per_cue):
            cue_texts.extend(_normalize_cue_capacity(part, max_chars_per_line, max_lines))

    times = _allocate_cue_times(cue_texts, duration)
    cues = []
    for (start, end), cue_text in zip(times, cue_texts):
        lines = _split_into_lines(cue_text, max_chars_per_line, max_lines)
        cues.append({
            "start": start,
            "end": end,
            "text": cue_text,
            "lines": [line for line in lines if line],
        })

    normalized_source = normalize_lexical(text)
    normalized_display = normalize_lexical(display_text)
    equivalent = normalized_source == normalized_display
    return {
        "display_text": display_text,
        "cues": cues,
        "validation": {
            "valid": bool(punctuated.get("valid", False)) and equivalent,
            "equivalent": equivalent,
            "fallback_used": bool(punctuated.get("fallback_used", False)),
            "normalized_source": normalized_source,
            "normalized_display": normalized_display,
        },
    }


__all__ = [
    "build_subtitle_cues",
    "equivalent_to_narration",
    "normalize_lexical",
    "punctuate_chinese",
]
