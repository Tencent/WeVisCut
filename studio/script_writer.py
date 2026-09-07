"""Draft a narration script from a creative brief.

The studio accepts two kinds of input. Mode 1 is a finished 口播稿, which goes
straight into the pipeline. Mode 2 is a brief — "剪一支 60 秒、300 字的科普短片，
讲城市早高峰" — and this module turns it into a script the pipeline can consume,
using the same policy that drives the agent loop.

Everything here is a pure function of the brief text except ``draft``, so the
length arithmetic and the output cleaning can be tested without a GPU.
"""

from __future__ import annotations

import re
from typing import Any

# Measured against the studio's default TTS voice and speaking rate. Chinese
# narration lands near 4.6 characters per second including pauses; the planner
# later re-times shots against the real audio, so this only needs to be close.
CHARS_PER_SECOND = 4.6

DEFAULT_DURATION_SEC = 60
MIN_DURATION_SEC = 8
MAX_DURATION_SEC = 1800
MIN_WORDS = 30
MAX_WORDS = 8000

_CN_DIGITS = {
    "零": 0, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}


def _cn_number(token: str) -> int | None:
    """Parse the small Chinese numerals that show up in durations (一~九十九)."""
    token = token.strip()
    if not token:
        return None
    if token.isdigit():
        return int(token)
    if "十" in token:
        head, _, tail = token.partition("十")
        tens = _CN_DIGITS.get(head, 1) if head else 1
        ones = _CN_DIGITS.get(tail, 0) if tail else 0
        if head and head not in _CN_DIGITS:
            return None
        if tail and tail not in _CN_DIGITS:
            return None
        return tens * 10 + ones
    return _CN_DIGITS.get(token)


_NUM = r"(\d+(?:\.\d+)?|[零一两二三四五六七八九十]{1,3})"

_MIN_SEC_RE = re.compile(_NUM + r"\s*分[钟]?\s*" + _NUM + r"\s*秒")
_MIN_HALF_RE = re.compile(_NUM + r"\s*分半")
_MIN_RE = re.compile(_NUM + r"\s*(?:分钟|分鐘|minutes?|mins?|min\b)")
_SEC_RE = re.compile(_NUM + r"\s*(?:秒钟|秒|seconds?|secs?|s\b)")
_WORD_RE = re.compile(_NUM + r"\s*(?:个字|字)")


def _value(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        n = _cn_number(token)
        return None if n is None else float(n)


def parse_brief(text: str) -> dict[str, Any]:
    """Pull an explicit duration and/or word count out of the brief.

    Whichever the user did not state is derived from the other, so the prompt
    always carries a concrete target — an open-ended "写一段口播稿" reliably
    produces something far too long for a short film.
    """
    brief = str(text or "")
    duration: float | None = None
    words: int | None = None

    if "半分钟" in brief:
        duration = 30.0
    m = _MIN_HALF_RE.search(brief)
    if m and (v := _value(m.group(1))) is not None:
        duration = v * 60 + 30
    if duration is None:
        m = _MIN_SEC_RE.search(brief)
        if m:
            mins, secs = _value(m.group(1)), _value(m.group(2))
            if mins is not None and secs is not None:
                duration = mins * 60 + secs
    if duration is None:
        m = _MIN_RE.search(brief)
        if m and (v := _value(m.group(1))) is not None:
            duration = v * 60
    if duration is None:
        m = _SEC_RE.search(brief)
        if m and (v := _value(m.group(1))) is not None:
            duration = v

    m = _WORD_RE.search(brief)
    if m and (v := _value(m.group(1))) is not None:
        words = int(round(v))

    explicit_duration = duration is not None
    explicit_words = words is not None

    if duration is None and words is not None:
        duration = words / CHARS_PER_SECOND
    if duration is None:
        duration = float(DEFAULT_DURATION_SEC)
    duration = max(MIN_DURATION_SEC, min(MAX_DURATION_SEC, duration))

    if words is None:
        # Round to a tidy number: the model follows "约 280 字" more closely
        # than "约 276 字", which reads like a hard constraint it cannot hit.
        words = int(round(duration * CHARS_PER_SECOND / 10.0)) * 10
    words = max(MIN_WORDS, min(MAX_WORDS, words))

    return {
        "duration_sec": int(round(duration)),
        "word_count": words,
        "explicit_duration": explicit_duration,
        "explicit_word_count": explicit_words,
    }


SYSTEM_PROMPT = """你是一名短视频口播稿撰稿人，为 AI 视频生成流水线供稿。
你的稿子会被直接送去配音，因此必须是可以逐字朗读的正文。

写作要求：
1. 总分总结构：开头一句话立住钩子或抛出问题，中间分 2-4 层递进展开，结尾收束回开头的主题。
2. 口语化：短句为主，避免书面长句、避免括号、避免分号。
3. 内容具体：给细节、给画面感、给可被检索成镜头的名词，不要空泛的形容词堆砌。
4. 全文一气呵成，前后不重复、不绕回已经说过的话。

输出格式（最重要，必须严格遵守）：
- 直接输出口播稿正文，第一个字就是稿子的第一个字。
- 全文只有一整段，中间不要换行、不要空行、不要分段。
- 不要输出任何思考过程、构思、提纲、草稿、字数统计、自我点评或修改说明。
- 绝对不要在句子后面标注字数，例如「。(29)」「（18字）」这类计数标记一律不许出现。
- 不要标题、不要小标题、不要分镜号、不要「镜头1」「画面：」「旁白：」这类标注。
- 不要 Markdown、不要列表符号、不要代码块、不要用引号包裹全文。
- 不要写「大家好」「今天我们来聊聊」这类套话开场，第一句就进入内容。
- 写完最后一句就停下，不要再补充任何说明。

篇幅是硬指标：字数不够就是不合格。宁可多写细节，也不要写不到目标字数。"""

# Used on the retry after a first attempt came back unusable.
STRICT_SUFFIX = """

特别注意：上一次的输出不合格，混进了思考过程或重复段落。
这一次请直接、一次性地写完整篇稿子，只输出一整段中文正文，不要有任何其他内容。"""


# Narration sentences run around this length, so a character target converts
# into a sentence count. A model tracks "write about 20 sentences" far more
# reliably than "write about 550 characters", which is the difference between
# landing on the requested duration and stopping half way.
CHARS_PER_SENTENCE = 28

# A draft lands near 80% of whatever single number the prompt names, and asking
# again does not fix it: the retry re-states the same script at the same length.
# So the requirement is stated as a range with the real target as its floor.
# Overshoot is cheap — ``trim_to_length`` cuts it back on a sentence boundary —
# while falling short is what the user sees in the finished video.
ASK_RATIO = 1.25


def sentence_target(word_count: int) -> int:
    return max(3, int(round(word_count / CHARS_PER_SENTENCE)))


def upper_bound(word_count: int) -> int:
    """Top of the stated range: a tidy number above the real target."""
    return int(round(word_count * ASK_RATIO / 10.0)) * 10


def build_user_prompt(brief: str, target: dict[str, Any]) -> str:
    lines = [
        "请根据下面的创作需求写一篇口播稿。",
        "",
        "【创作需求】",
        str(brief).strip(),
        "",
        "【篇幅要求】",
        f"- 全篇 {target['word_count']}-{upper_bound(target['word_count'])} 字，"
        f"少于 {target['word_count']} 字算不合格，宁可写到上限",
        f"- 大约 {sentence_target(target['word_count'])}-"
        f"{sentence_target(upper_bound(target['word_count']))} 句话，"
        "每句 25-35 字，写够句数再收尾",
        f"- 朗读时长约 {target['duration_sec']} 秒",
    ]
    if not target["explicit_duration"] and not target["explicit_word_count"]:
        lines.append("- 需求中没有指定时长，按上面的默认篇幅写")
    lines += [
        "",
        "【输出】",
        "只输出口播稿正文，一整段，不分段不换行，不要任何思考过程或说明文字。",
        f"务必写够 {target['word_count']} 字以上，不要提前收尾。",
        "现在开始：",
    ]
    return "\n".join(lines)


EXPAND_SUFFIX = """

这一轮的任务是扩写，不是重写。保留原稿的第一句和最后一句不动，
只在中间补充具体的年份、人名、数字、动作和场景细节，把篇幅撑到目标字数。
不要复述已经说过的内容，不要加入与主题无关的段落。"""


def build_expand_prompt(
    brief: str, target: dict[str, Any], current: str, current_len: int
) -> str:
    """Ask for a longer version of an existing draft.

    Expanding beats re-drafting: a fresh attempt at the same brief tends to
    come up short the same way, while a concrete "you wrote N, need M" gap
    gives the model something to close.
    """
    # Aim at the top of the range, not the target: an expansion that merely
    # reaches the target still undershoots it once the model comes up short.
    goal = upper_bound(target["word_count"])
    need = max(0, goal - current_len)
    return "\n".join([
        "下面这篇口播稿篇幅不够，需要扩写。",
        "",
        "【原始创作需求】",
        str(brief).strip(),
        "",
        "【当前稿件】",
        current,
        "",
        "【篇幅差距】",
        f"- 当前 {current_len} 字，要写到 {goal} 字，还差约 {need} 字",
        f"- 也就是在原文基础上再补大约 {sentence_target(need)} 句话，一句都不能少",
        f"- 朗读时长要凑到约 {target['duration_sec']} 秒",
        "",
        "【输出】",
        "输出扩写后的完整口播稿全文，一整段，不要只输出新增部分，",
        "不要任何说明文字，不要标注字数。",
        "现在开始：",
    ])


_FENCE_RE = re.compile(r"```[a-zA-Z]*")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<think>", re.IGNORECASE)
_ANY_TAG_RE = re.compile(r"</?(?:think|answer|output|response)>", re.IGNORECASE)
_LABEL_RE = re.compile(
    r"^\s*(?:【[^】]{0,16}】|\[[^\]]{0,16}\]|\*{1,2}[^*\n]{0,24}\*{1,2}\s*[:：]?|"
    r"(?:镜头|画面|旁白|配音|字幕|口播稿?|文案|正文|标题|开头|结尾|草稿|终稿|初稿)\s*\d*\s*[:：])\s*"
)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.、)]|\(\d+\))\s+")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s+\S")
# Per-sentence character tallies the model writes to itself mid-script, as in
# "…意外发现密室。(29)墙缝里漏出的泥丸…". Capped at three digits so a year in
# parentheses — "（1900）" — is left alone.
_INLINE_COUNT_RE = re.compile(r"[(（]\s*(?:共|约|计)?\s*\d{1,3}\s*(?:字|chars?)?\s*[)）]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
# Chinese asides the model writes to itself; a fully parenthesised line is
# never narration, and these prefixes only introduce commentary.
_CN_META_RE = re.compile(
    r"^\s*(?:[（(].*[)）]\s*$|"
    r"(?:注|注意|说明|备注|字数|统计|以上|下面|接下来我|我需要|我将|让我)\s*[:：])"
)
# 。！？…  plus the ASCII equivalents, keeping any trailing closing quote.
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?…；;])(?=[^”』」)）])|(?<=[。！？!?…])$")

# A line has to be mostly Chinese to be narration rather than the model
# talking to itself; reasoning traces are overwhelmingly English or symbols.
_MIN_NARRATION_CJK = 4
_MIN_NARRATION_RATIO = 0.55
# Short sentences may legitimately repeat for rhythm; long ones repeating means
# the model looped and pasted the same paragraph again.
_MIN_DEDUPE_LEN = 8


def _strip_markers(line: str) -> str:
    line = _ANY_TAG_RE.sub("", line)
    line = _INLINE_COUNT_RE.sub("", line)
    line = _FENCE_RE.sub("", line)
    line = re.sub(r"^\s*#{1,6}\s*", "", line)
    prev = None
    while prev != line:
        prev = line
        line = _LABEL_RE.sub("", line)
        line = _BULLET_RE.sub("", line)
    return line.strip()


def classify_line(line: str, *, strict: bool = True) -> str:
    """``blank`` | ``meta`` | ``text``.

    ``meta`` is the model talking to itself — English analysis, word-count
    tallies, parenthesised asides, titles. It both gets dropped and acts as a
    boundary between candidate drafts. Blank lines are only a soft separator so
    that a normally paragraphed script stays one block.

    ``strict=False`` keeps anything with Chinese in it, and is used as a second
    pass when the strict one rejected everything.
    """
    s = line.strip()
    if not s:
        return "blank"
    cjk = len(_CJK_RE.findall(s))
    if cjk == 0:
        return "meta"
    if not strict:
        return "text"
    if _CN_META_RE.match(s):
        return "meta"
    # A markdown heading is a title, not narration: keeping it means the voice
    # reads the title out loud before the script starts.
    if _HEADING_RE.match(s) and len(s) <= 28:
        return "meta"
    letters = len(_LATIN_RE.findall(s))
    if letters >= 3 and cjk / len(s) < _MIN_NARRATION_RATIO:
        return "meta"
    if cjk < _MIN_NARRATION_CJK:
        return "meta"
    return "text"


def _collect_blocks(text: str, *, strict: bool) -> list[list[str]]:
    """Split into candidate drafts at every meta line.

    A reasoning trace holds many partial drafts; concatenating them interleaves
    throwaway fragments with the real script, so the caller keeps only the
    longest run of narration — the best reconstruction available.
    """
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        kind = classify_line(line, strict=strict)
        if kind == "meta":
            if current:
                blocks.append(current)
                current = []
            continue
        if kind == "blank":
            continue
        stripped = _strip_markers(line)
        if stripped:
            current.append(stripped)
    if current:
        blocks.append(current)
    return blocks


def _dedupe_sentences(lines: list[str]) -> str:
    """Flatten lines into one paragraph, dropping sentences already said.

    A model that loops pastes the same paragraph several times; de-duplicating
    by sentence collapses that without needing to detect the loop itself.
    """
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        for sent in split_sentences(line):
            key = re.sub(r"[\s\W_]+", "", sent, flags=re.UNICODE)
            if len(key) >= _MIN_DEDUPE_LEN:
                if key in seen:
                    continue
                seen.add(key)
            out.append(sent)
    return "".join(out).strip()


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(str(text or "")) if p and p.strip()]
    return parts


def clean_draft(raw: str) -> str:
    """Reduce whatever the model returned to one paragraph of pure narration.

    This does more than tidy formatting. When a long draft runs out of tokens
    while still inside ``<think>`` there is no closing tag, so nothing upstream
    can strip the reasoning and the caller would otherwise receive the entire
    English trace — repeated draft attempts, word-count tallies and all — as
    the "script". Keeping only Chinese narration lines and then de-duplicating
    at the sentence level recovers the real script out of that trace, and is a
    no-op on a well-formed draft.
    """
    text = _THINK_RE.sub("", str(raw or ""))

    blocks = _collect_blocks(text, strict=True) or _collect_blocks(text, strict=False)
    if not blocks:
        return ""

    # One paragraph, per the output contract: the pipeline segments the script
    # semantically, so hard breaks only survive as odd pauses in the voiceover.
    out = max((_dedupe_sentences(b) for b in blocks), key=visible_length, default="")

    # A whole-script quote wrap is common and would be read aloud.
    for open_q, close_q in (('"', '"'), ("“", "”"), ("「", "」"), ("『", "』")):
        if out.startswith(open_q) and out.endswith(close_q) and len(out) > 2:
            out = out[1:-1].strip()
    return out


def ends_mid_sentence(text: str) -> bool:
    return bool(text) and text[-1] not in "。！？!?…”』」)）"


def trim_to_sentence(text: str) -> str:
    """Drop a trailing fragment so the voiceover never stops mid-thought."""
    if not ends_mid_sentence(text):
        return text
    cut = max(text.rfind(ch) for ch in "。！？!?…")
    return text[: cut + 1].strip() if cut > 0 else text


def trim_to_length(text: str, max_chars: int) -> str:
    """Shorten to ``max_chars`` by dropping sentences from the body.

    The closing sentence is kept: it carries the 总分总 payoff, and cutting the
    tail would leave the voiceover ending on a mid-argument beat.
    """
    if visible_length(text) <= max_chars:
        return text
    sentences = split_sentences(text)
    if len(sentences) < 3:
        return text

    head, tail = sentences[:-1], sentences[-1]
    budget = max_chars - visible_length(tail)
    if budget <= 0:
        return text

    kept = ""
    for sent in head:
        if kept and visible_length(kept + sent) > budget:
            break
        kept += sent
    return ((kept or head[0]) + tail).strip()


def visible_length(text: str) -> int:
    """Character count as a reader would judge it: no whitespace, no punctuation."""
    return len(re.sub(r"[\s\W_]+", "", str(text or ""), flags=re.UNICODE))


# One Chinese character is roughly one token for this tokenizer; punctuation and
# the odd multi-token word push it a little above. 2.6 leaves the model room to
# overshoot and be trimmed back, which is the direction we want to fail in.
TOKENS_PER_CHAR = 2.6
# ``compose`` runs with thinking off, so this is slack for a long opening or a
# run-on final sentence rather than a reasoning trace. It stays non-zero because
# a truncated draft is the one failure the user actually sees.
THINKING_HEADROOM = 1024
# Unused budget costs nothing — generation stops at EOS — so the ceiling only
# needs to be above the longest script anyone asks for.
MAX_OUTPUT_TOKENS = 16384
# Below this fraction of the target the draft is a fragment, not a short script.
MIN_USABLE_RATIO = 0.45
# Word count is a hard requirement: short means the finished video runs short of
# the requested duration, which is what the user feels. Accept only near-target
# drafts and expand otherwise; with thinking off an extra round is cheap.
MIN_ACCEPT_RATIO = 0.92
# The prompt asks for up to 1.25x, so trimming is the normal case rather than a
# rare correction. Keep the band tight: the point of asking high is to land on
# the requested duration, not to run over it.
MAX_KEEP_RATIO = 1.08
MAX_EXPAND_ROUNDS = 2


def token_budget(word_count: int) -> int:
    want = int(word_count * TOKENS_PER_CHAR) + THINKING_HEADROOM
    return max(2048, min(MAX_OUTPUT_TOKENS, want))


def _compose(model: Any, system: str, user: str, *, max_tokens: int, temperature: float):
    """Call ``compose`` when the backend has it, else fall back to ``generate``."""
    fn = getattr(model, "compose", None)
    if callable(fn):
        return fn(system, user, max_tokens=max_tokens, temperature=temperature)
    text = model.generate(system, user, max_tokens=max_tokens, temperature=temperature)
    return {"text": text, "raw": text, "truncated": False, "completion_tokens": 0}


async def draft(model: Any, brief: str, *, temperature: float = 0.7) -> dict[str, Any]:
    """Ask the policy for a script in 1-2 calls (speed-first)."""
    import asyncio

    text = str(brief or "").strip()
    if not text:
        raise ValueError("创作需求不能为空")

    target = parse_brief(text)
    # Budget for the top of the range we ask for, not the target we settle for.
    max_tokens = token_budget(upper_bound(target["word_count"]))
    floor = int(target["word_count"] * MIN_USABLE_RATIO)

    best = ""
    attempts: list[dict[str, Any]] = []
    # Call #1: full draft.
    result = await asyncio.to_thread(
        _compose, model, SYSTEM_PROMPT, build_user_prompt(text, target),
        max_tokens=max_tokens, temperature=temperature,
    )
    script = clean_draft(result.get("text") or "")
    if visible_length(script) < floor:
        salvaged = clean_draft(result.get("raw") or "")
        if visible_length(salvaged) > visible_length(script):
            script = salvaged
    script = trim_to_sentence(script)
    attempts.append({
        "chars": visible_length(script),
        "truncated": bool(result.get("truncated")),
    })
    best = script

    if not best:
        raise RuntimeError("模型没有产出可用的口播稿，请补充需求后重试")

    # Close the gap by expanding rather than re-drafting: a fresh attempt at the
    # same brief comes up short the same way, while "you wrote N, need M" gives
    # the model something concrete to close. Each round gets the full budget —
    # halving it was self-defeating, since the whole problem is length.
    want = int(target["word_count"] * MIN_ACCEPT_RATIO)
    for _ in range(MAX_EXPAND_ROUNDS):
        current = visible_length(best)
        if current >= want:
            break
        result = await asyncio.to_thread(
            _compose, model,
            SYSTEM_PROMPT + EXPAND_SUFFIX,
            build_expand_prompt(text, target, best, current),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        grown = trim_to_sentence(clean_draft(result.get("text") or ""))
        if visible_length(grown) < visible_length(clean_draft(result.get("raw") or "")):
            grown = trim_to_sentence(clean_draft(result.get("raw") or ""))
        attempts.append({
            "chars": visible_length(grown),
            "expand": True,
            "truncated": bool(result.get("truncated")),
        })
        # No growth means expansion is not working on this brief; another
        # identical round would only cost the user more waiting.
        if visible_length(grown) <= current:
            break
        best = grown

    best = trim_to_sentence(trim_to_length(best, int(target["word_count"] * MAX_KEEP_RATIO)))
    length = visible_length(best)
    return {
        "script": best,
        "brief": text,
        "target": target,
        "char_count": length,
        "estimated_sec": int(round(length / CHARS_PER_SECOND)),
        "attempts": attempts,
        "short": length < want,
    }
