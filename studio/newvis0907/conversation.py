"""Layered conversation memory for the newvis0907 editing chat.

The editing chat is multi-turn: a user says「@镜头03 改一下动画」, the assistant
may need one detail, the user answers「是的」, and the edit must then *happen*.
Without memory each turn is planned from scratch, so an answer to a question the
assistant just asked reads as a brand-new (and unresolvable) request and the
conversation loops.

Four layers, each with a different lifetime:

``working``
    The question currently on the table — which shot, or which change — plus the
    candidates that were offered. Lives until it is answered, superseded, or
    ``WORKING_TTL`` turns pass.
``focus``
    The shot the conversation is currently about, so「再短一点」「换一个」
    continue on the same target without repeating it.
``episodes``
    A bounded, compact log of recent turns (user text, intent, targets, ops).
    Older episodes are folded into ``summary`` rather than dropped, which keeps
    the model prompt small without losing the thread.
``preferences``
    Durable, distilled facts about how this user likes their film edited.

Everything is persisted next to the project as ``chat_memory0907.json`` so a page
reload — or a worker restart — does not amnesia the conversation.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .locator import parse_number, strip_mentions

MEMORY_FILENAME = "chat_memory0907.json"
MEMORY_VERSION = 1

#: A pending question older than this many turns is stale — the user moved on.
WORKING_TTL = 3
#: Episodes kept verbatim; anything older is folded into the rolling summary.
EPISODE_WINDOW = 12
#: How many times we may ask about the same instruction before committing to the
#: best candidate. Asking twice is helpful; asking three times is a loop.
MAX_CLARIFY_STREAK = 2

_AFFIRM = (
    "是的", "是", "对", "对的", "对对", "嗯", "恩", "没错", "就是它", "就是这个",
    "就它", "确认", "确定", "可以", "行", "好", "好的", "ok", "okay", "yes", "y",
    "改这个", "改它", "就改这个", "改那个",
)
_NEGATE = ("不是", "不对", "都不是", "不", "换一个", "不是这个", "no")
#: 「第2个」「第二个」「2」「选 3」 — a bare pick with no unit noun, which only
#: makes sense as an answer to a list we just printed.
_PICK_RE = re.compile(r"^(?:选|挑|要|用|就)?\s*第?\s*([0-9零一两二三四五六七八九十]+)\s*(?:个|条|项|号)?$")
#: The bullet prefix used when printing candidates, so pasting a line back works.
_BULLET_RE = re.compile(r"^[·\-•*\s]+")
_DISPLAY_RE = re.compile(r"(?:镜头|分镜)\s*0*(\d+)")
_SUBCLIP_DISPLAY_RE = re.compile(r"子镜头\s*(\d+)\s*[.．]\s*(\d+)")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_text(value: Any) -> str:
    return str(value or "")


def _normalize(message: str) -> str:
    """Strip list decoration and trailing punctuation from a short reply."""
    text = _BULLET_RE.sub("", _as_text(message).strip())
    return text.strip().strip("。.！!，,、 ").strip()


def is_affirmation(message: str) -> bool:
    text = _normalize(message).lower()
    if not text or len(text) > 6:
        return False
    if any(text.startswith(word) for word in _NEGATE):
        return False
    return text in _AFFIRM or any(text == word for word in _AFFIRM)


def is_negation(message: str) -> bool:
    text = _normalize(message).lower()
    return bool(text) and len(text) <= 8 and any(text.startswith(word) for word in _NEGATE)


def candidate_token(candidate: dict[str, Any]) -> str:
    """The ``@`` token that names this candidate unambiguously to the locator."""
    subclip = _as_text(_as_dict(candidate).get("subclip_display_number"))
    if subclip:
        return f"@子镜头{subclip}"
    number = _as_text(_as_dict(candidate).get("display_number"))
    return f"@镜头{number}" if number else ""


def candidate_label(candidate: dict[str, Any]) -> str:
    """A one-line, human description of a candidate shot."""
    item = _as_dict(candidate)
    number = _as_text(item.get("display_number"))
    chapter = _as_text(item.get("chapter_title"))
    line = _as_text(item.get("matched_text"))
    # The matched text is often the "@" token the user just typed, which says
    # nothing new; prefer real narration when it is available.
    if line.startswith("@"):
        line = _as_text(item.get("narration"))
    # Fuzzy matches quote a corpus that already begins with the chapter title,
    # which would print it twice in the same line.
    if chapter and line.startswith(chapter):
        line = line[len(chapter):].strip()
    if len(line) > 24:
        line = line[:24] + "…"
    parts = [part for part in (f"镜头 {number}" if number else "", chapter, line) if part]
    return " · ".join(parts)


def pick_candidate(message: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Bind a short follow-up reply to one of the candidates we just offered.

    Handles every way a user answers a printed list: the shot number, an ordinal
    into the list, a pasted copy of the option line, or — when there is only one
    option — a plain「是的」.
    """
    rows = [_as_dict(item) for item in _as_list(candidates) if isinstance(item, dict)]
    if not rows:
        return None
    text = _normalize(message)
    if not text:
        return None

    subclip = _SUBCLIP_DISPLAY_RE.search(text)
    if subclip:
        wanted = f"{int(subclip.group(1))}.{int(subclip.group(2))}"
        for row in rows:
            if _as_text(row.get("subclip_display_number")) == wanted:
                return row

    display = _DISPLAY_RE.search(text)
    if display:
        wanted = str(int(display.group(1)))
        for row in rows:
            if _as_text(row.get("display_number")).lstrip("0") == wanted:
                return row

    for row in rows:
        shot_id = _as_text(row.get("shot_id"))
        if shot_id and shot_id.casefold() in text.casefold():
            return row

    # A pasted option line: match on the label we printed.
    for row in rows:
        label = candidate_label(row)
        if label and (label in text or text in label):
            return row

    if is_affirmation(text):
        # "是的" only settles the question when there was nothing to choose
        # between; with several options it is not an answer.
        return rows[0] if len(rows) == 1 else None

    pick = _PICK_RE.match(text)
    if pick:
        index = parse_number(pick.group(1))
        if 1 <= index <= len(rows):
            return rows[index - 1]
    return None


@dataclass
class Pending:
    """The question currently awaiting an answer (the working-memory layer).

    ``kind`` is ``"target"`` when we know what to do but not where, and
    ``"action"`` when we know the shot but not the change.
    """

    kind: str = "target"
    instruction: str = ""
    candidates: list[dict[str, Any]] = field(default_factory=list)
    target: dict[str, Any] | None = None
    asked: int = 0
    turn: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Any) -> "Pending | None":
        data = _as_dict(raw)
        if not data:
            return None
        kind = _as_text(data.get("kind")) or "target"
        if kind not in {"target", "action"}:
            return None
        return cls(
            kind=kind,
            instruction=_as_text(data.get("instruction")),
            candidates=[c for c in _as_list(data.get("candidates")) if isinstance(c, dict)],
            target=_as_dict(data.get("target")) or None,
            asked=int(data.get("asked") or 0),
            turn=int(data.get("turn") or 0),
        )


@dataclass
class Turn:
    """What one user message means once conversation memory is applied."""

    message: str
    original: str
    previous_targets: list[dict[str, Any]] = field(default_factory=list)
    bound_target: dict[str, Any] | None = None
    #: A short line explaining the binding, prefixed to the assistant reply so
    #: the user can see (and correct) the assumption we made.
    note: str = ""
    resolved_pending: bool = False


class Memory:
    """The four memory layers for one project's editing conversation."""

    def __init__(self, path: Path | None = None, data: dict[str, Any] | None = None) -> None:
        raw = _as_dict(data)
        self.path = path
        self.turn = int(raw.get("turn") or 0)
        self.pending = Pending.from_dict(raw.get("pending"))
        self.focus = _as_dict(raw.get("focus")) or None
        self.episodes = [e for e in _as_list(raw.get("episodes")) if isinstance(e, dict)]
        self.summary = _as_text(raw.get("summary"))
        self.preferences = _as_dict(raw.get("preferences"))
        self.clarify_streak = int(raw.get("clarify_streak") or 0)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, project_dir: Any) -> "Memory":
        path = Path(str(project_dir)) / MEMORY_FILENAME
        if not path.is_file():
            return cls(path=path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls(path=path)
        if not isinstance(raw, dict) or int(raw.get("version") or 0) != MEMORY_VERSION:
            return cls(path=path)
        return cls(path=path, data=raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": MEMORY_VERSION,
            "turn": self.turn,
            "pending": self.pending.to_dict() if self.pending else None,
            "focus": self.focus,
            "episodes": self.episodes,
            "summary": self.summary,
            "preferences": self.preferences,
            "clarify_streak": self.clarify_streak,
        }

    def save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.write_text(
                json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError:  # pragma: no cover - persistence is best effort
            pass

    def reset(self) -> None:
        """Forget everything except distilled preferences (「新建创作」)."""
        self.turn = 0
        self.pending = None
        self.focus = None
        self.episodes = []
        self.summary = ""
        self.clarify_streak = 0

    # ------------------------------------------------------------------
    # reading a turn
    # ------------------------------------------------------------------
    def _expire_pending(self) -> None:
        if self.pending and self.turn - self.pending.turn > WORKING_TTL:
            self.pending = None

    def interpret(self, message: str) -> Turn:
        """Rewrite ``message`` into something the planner can resolve alone.

        The planner stays stateless: everything this conversation knows is folded
        into the returned message (as an explicit ``@`` target) and
        ``previous_targets``, so a reply of「是的」arrives at the planner as
       「@镜头03 改一下动画」.
        """
        self.turn += 1
        self._expire_pending()
        raw = _as_text(message).strip()
        turn = Turn(message=raw, original=raw, previous_targets=self._focus_rows())

        if not raw:
            return turn

        pending = self.pending
        if pending is None:
            return turn

        if is_negation(raw) and pending.kind == "target":
            # "不是这个" clears the offer but keeps the instruction alive.
            self.pending = Pending(
                kind="target", instruction=pending.instruction,
                candidates=[], target=None, asked=pending.asked, turn=self.turn,
            )
            return turn

        if pending.kind == "action" and pending.target:
            # We already know the shot, so this message is normally the missing
            # instruction.
            if not _names_a_target(raw):
                token = candidate_token(pending.target)
                turn.message = f"{token} {raw}".strip() if token else raw
                turn.bound_target = pending.target
                turn.previous_targets = _target_rows(pending.target)
                turn.resolved_pending = True
            elif not _new_instruction(raw, pending.target):
                # The user re-pointed at a target (often by pasting the line we
                # printed) without saying anything new. The new target wins, but
                # the instruction they gave before we interrupted them stands.
                turn.message = f"{raw} {pending.instruction}".strip()
                turn.resolved_pending = True
            return turn

        picked = pick_candidate(raw, pending.candidates)
        if picked is None:
            if is_affirmation(raw) and self.focus:
                picked = self.focus
        if picked is None:
            return turn

        token = candidate_token(picked)
        # Either the user restated what they want, or they only pointed at a
        # shot — in which case the instruction we interrupted still stands.
        # Merging both would produce a contradictory request.
        instruction = _new_instruction(raw, picked) or pending.instruction or raw
        turn.message = f"{token} {instruction}".strip()
        turn.bound_target = picked
        turn.previous_targets = _target_rows(picked)
        turn.resolved_pending = True
        turn.note = f"按{candidate_label(picked)}来改。"
        return turn

    def _focus_rows(self) -> list[dict[str, Any]]:
        return _target_rows(self.focus) if self.focus else []

    def should_autocommit(self, instruction: str) -> bool:
        """True when we have already asked enough and must now just act.

        Repeating a question the user has answered twice is worse than picking
        the best candidate and saying so — the user can always correct one shot,
        but they cannot escape a loop.
        """
        pending = self.pending
        if pending is None:
            return False
        if pending.asked < MAX_CLARIFY_STREAK:
            return False
        return bool(pending.instruction) and bool(_as_text(instruction))

    # ------------------------------------------------------------------
    # writing a turn
    # ------------------------------------------------------------------
    def record(self, turn: Turn, payload: dict[str, Any]) -> None:
        """Fold the planner's decision back into every memory layer."""
        result = _as_dict(payload)
        intent = _as_text(result.get("intent")) or "clarify"
        ops = [op for op in _as_list(result.get("ops")) if isinstance(op, dict)]
        located = _as_dict(result.get("located")) or turn.bound_target
        candidates = [c for c in _as_list(result.get("candidates")) if isinstance(c, dict)]

        if intent == "edit" and ops:
            self.pending = None
            self.clarify_streak = 0
            if located:
                self.focus = located
            self._learn_preferences(turn.original, ops)
        elif intent == "clarify":
            self.clarify_streak += 1
            asked = (self.pending.asked + 1) if self.pending else 1
            instruction = (
                (self.pending.instruction if self.pending else "")
                or strip_mentions(turn.original)
                or turn.original
            )
            if located:
                self.focus = located
                self.pending = Pending(
                    kind="action", instruction=instruction, candidates=candidates,
                    target=located, asked=asked, turn=self.turn,
                )
            else:
                self.pending = Pending(
                    kind="target", instruction=instruction, candidates=candidates,
                    target=None, asked=asked, turn=self.turn,
                )
        else:  # question / small talk leaves the open question untouched
            self.clarify_streak = 0

        self.episodes.append({
            "turn": self.turn,
            "user": turn.original[:200],
            "effective": turn.message[:200] if turn.message != turn.original else "",
            "intent": intent,
            "targets": [_as_text(t) for t in _as_list(result.get("targets"))],
            "ops": [_as_text(op.get("op")) for op in ops],
        })
        self._compact()

    def _learn_preferences(self, message: str, ops: list[dict[str, Any]]) -> None:
        """Distil the durable layer: what this user keeps asking for."""
        text = _as_text(message).lower()
        names = {_as_text(op.get("op")) for op in ops}
        if "hyperframes" in text:
            self.preferences["animation_engine"] = "hyperframes"
        if "set_bgm" in names:
            query = next((_as_text(op.get("query")) for op in ops if _as_text(op.get("op")) == "set_bgm"), "")
            if query:
                self.preferences["bgm"] = query[:40]
        if "set_emphasis" in names:
            self.preferences["likes_emphasis"] = True

    def _compact(self) -> None:
        """Trim the episodic layer, folding what falls out into ``summary``."""
        if len(self.episodes) <= EPISODE_WINDOW:
            return
        dropped = self.episodes[:-EPISODE_WINDOW]
        self.episodes = self.episodes[-EPISODE_WINDOW:]
        edits = [e for e in dropped if _as_text(e.get("intent")) == "edit"]
        lines = [_as_text(self.summary)] if self.summary else []
        for episode in edits:
            targets = "、".join(_as_list(episode.get("targets"))) or "全片"
            ops = "、".join(_as_list(episode.get("ops"))) or "调整"
            lines.append(f"第{episode.get('turn')}轮：{targets} {ops}")
        # Keep the summary itself bounded, or compaction just moves the problem.
        self.summary = "\n".join(lines[-20:])

    # ------------------------------------------------------------------
    # reading memory back out, for the model prompt
    # ------------------------------------------------------------------
    def prompt_context(self) -> dict[str, Any]:
        """The memory layers, shaped for the planner prompt."""
        pending = self.pending
        return {
            "当前待回答的问题": (
                {
                    "类型": "缺少目标镜头" if pending.kind == "target" else "缺少具体改动",
                    "用户原本想做的事": pending.instruction,
                    "已经问过几次": pending.asked,
                    "候选镜头": [candidate_label(c) for c in pending.candidates[:3]],
                    "已锁定镜头": candidate_label(pending.target) if pending.target else "",
                }
                if pending
                else None
            ),
            "当前聚焦镜头": candidate_label(self.focus) if self.focus else "",
            "最近几轮要点": self.episodes[-6:],
            "更早的对话摘要": self.summary,
            "用户偏好": self.preferences,
        }


def _target_rows(candidate: dict[str, Any] | None) -> list[dict[str, Any]]:
    item = _as_dict(candidate)
    shot_id = _as_text(item.get("shot_id"))
    if not shot_id:
        return []
    return [{"shot_id": shot_id, "subclip_id": _as_text(item.get("subclip_id"))}]


def is_pure_pick(message: str) -> bool:
    """True when the message only identifies a shot and carries no instruction.

    「是的」「第2个」「镜头03」 all answer a question without stating a new one,
    so they must be read against the conversation rather than searched for in
    the film's narration.
    """
    text = _normalize(message)
    if not text:
        return False
    if is_affirmation(text) or _PICK_RE.match(text):
        return True
    remainder = strip_mentions(text)
    remainder = _DISPLAY_RE.sub("", remainder)
    remainder = _SUBCLIP_DISPLAY_RE.sub("", remainder)
    return len(_normalize(remainder)) <= 2


def _new_instruction(message: str, picked: dict[str, Any]) -> str:
    """What the user asked for in *this* message, beyond naming a shot.

    Empty when the message only identifies a shot — a bare「是的」, an ordinal,
    or the candidate line pasted straight back — since none of those say
    anything about what should change.
    """
    if is_pure_pick(message):
        return ""
    # We print candidates as "· 镜头 03 · …", so a message starting with that
    # bullet is the user pasting one back — a selection, never an instruction.
    if _BULLET_RE.match(_as_text(message)) and _as_text(message).strip()[:1] in "·•*":
        return ""
    text = _normalize(strip_mentions(message))
    text = _DISPLAY_RE.sub(" ", text)
    text = _SUBCLIP_DISPLAY_RE.sub(" ", text)
    shot_id = _as_text(_as_dict(picked).get("shot_id"))
    if shot_id:
        text = re.sub(re.escape(shot_id), " ", text, flags=re.IGNORECASE)
    # Strip the pieces of the label we printed, so echoing it back reads as a
    # selection rather than as an instruction.
    for chunk in re.split(r"[·|、,，\s]+", candidate_label(picked)):
        if len(chunk) >= 2:
            text = text.replace(chunk, " ")
    remainder = _normalize(text)
    return remainder if len(remainder) >= 3 else ""


def _names_a_target(message: str) -> bool:
    text = _as_text(message)
    return bool(
        "@" in text
        or _DISPLAY_RE.search(text)
        or _SUBCLIP_DISPLAY_RE.search(text)
    )
