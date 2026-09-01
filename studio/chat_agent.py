"""Multi-turn conversational edit agent.

Editing by conversation only works if the agent remembers what was already
agreed. But a long session plus a full shot index will blow past the context
window, and the 9B policy degrades badly when its prompt is truncated
arbitrarily. So context here is *budgeted*, not merely capped:

**History** — recent turns stay verbatim; older turns are compressed by the
model itself into a running digest that keeps user preferences, what was already
changed, and what was rejected. Compression is triggered by an estimated token
count, and the digest is folded forward so nothing is summarised twice.

**Project** — a short cut injects its whole shot index. A long one injects only
the shots plausibly related to the current message, selected by
``Project.find_shots``, with a note telling the model the list is partial so it
asks instead of guessing.

The model's reply is a JSON envelope (see ``chat_prompts``). Parsing is
deliberately forgiving of code fences and stray prose, and a malformed envelope
is re-asked with the schema error attached rather than surfaced to the user.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from studio import chat_prompts, editor
from studio.project import Project

logger = logging.getLogger("aicut_studio.chat")

VALID_INTENTS = {"edit", "clarify", "question", "reject"}


# ---------------------------------------------------------------------------
# token estimation
# ---------------------------------------------------------------------------
def estimate_tokens(text: str, chars_per_token: float = 1.7) -> int:
    """Cheap token estimate.

    Avoids loading a tokenizer just to decide when to compress. CJK runs about
    1.0-1.5 chars/token and ASCII about 4, so a single divisor is approximate by
    design — it only needs to be monotonic and roughly calibrated, since it
    gates compression rather than sizing a hard limit.
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return int(cjk / max(0.1, chars_per_token) + other / 4) + 1


# ---------------------------------------------------------------------------
# conversation memory
# ---------------------------------------------------------------------------
@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str
    at: str = field(default_factory=lambda: datetime.now().isoformat())
    meta: dict[str, Any] = field(default_factory=dict)

    def as_message(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    def as_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content, "at": self.at, "meta": self.meta}


@dataclass
class Memory:
    """Rolling conversation memory with model-driven compression."""

    max_history_tokens: int = 12000
    keep_recent_turns: int = 6
    summary_target_tokens: int = 600
    chars_per_token: float = 1.7

    turns: list[Turn] = field(default_factory=list)
    summary: str = ""
    compressions: int = 0

    def add(self, role: str, content: str, **meta: Any) -> None:
        self.turns.append(Turn(role=role, content=content, meta=meta or {}))

    def recent(self) -> list[dict[str, str]]:
        return [t.as_message() for t in self.turns[-self.keep_recent_turns:]]

    def token_estimate(self) -> int:
        body = self.summary + "".join(t.content for t in self.turns)
        return estimate_tokens(body, self.chars_per_token)

    def needs_compression(self) -> bool:
        return (
            self.token_estimate() > self.max_history_tokens
            and len(self.turns) > self.keep_recent_turns
        )

    def turns_to_compress(self) -> list[dict[str, str]]:
        return [t.as_message() for t in self.turns[: -self.keep_recent_turns]]

    def apply_compression(self, new_summary: str) -> None:
        """Replace compressed turns with their summary, keeping recent ones."""
        summary = str(new_summary or "").strip()
        if summary:
            self.summary = summary
        # Drop the turns now represented by the summary. Without this the
        # estimate never falls and compression would fire every turn.
        self.turns = self.turns[-self.keep_recent_turns:]
        self.compressions += 1

    def drop_oldest(self) -> None:
        """Fallback when compression is unavailable: forget the oldest turns."""
        if len(self.turns) > self.keep_recent_turns:
            self.turns = self.turns[-self.keep_recent_turns:]

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "turns": [t.as_dict() for t in self.turns],
            "compressions": self.compressions,
            "token_estimate": self.token_estimate(),
            "budget": self.max_history_tokens,
        }

    # ---- disk persistence ----
    def save(self, path: str | Path) -> None:
        """Persist memory to a JSON file (atomic write)."""
        import json, tempfile
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.summary,
            "compressions": self.compressions,
            "turns": [t.as_dict() for t in self.turns],
        }
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=str(p.parent))
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            os.replace(tmp_path, str(p))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: str | Path, ctx: dict[str, Any] | None = None) -> Memory:
        """Restore memory from disk. Returns a fresh Memory if file is absent."""
        import json
        p = Path(path)
        mem = cls.from_config(ctx or {})
        if not p.exists():
            return mem
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            mem.summary = str(data.get("summary", "") or "")
            mem.compressions = int(data.get("compressions", 0) or 0)
            for t in data.get("turns", []):
                mem.turns.append(Turn(
                    role=str(t.get("role", "user")),
                    content=str(t.get("content", "")),
                    at=str(t.get("at", "")),
                    meta=t.get("meta", {}) or {},
                ))
        except Exception as exc:
            logger.warning("Failed to load chat memory from %s: %s", path, exc)
        return mem

    @classmethod
    def from_config(cls, ctx: dict[str, Any]) -> Memory:
        return cls(
            max_history_tokens=int(ctx.get("max_history_tokens", 12000) or 12000),
            keep_recent_turns=int(ctx.get("keep_recent_turns", 6) or 6),
            summary_target_tokens=int(ctx.get("summary_target_tokens", 600) or 600),
            chars_per_token=float(ctx.get("chars_per_token", 1.7) or 1.7),
        )


# ---------------------------------------------------------------------------
# response parsing
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull one JSON object out of a model response.

    Tries, in order: the whole string, any fenced block, then a brace-matched
    span. Brace matching is used rather than a greedy regex because replies
    legitimately contain nested objects inside ``ops``.
    """
    raw = (text or "").strip()
    if not raw:
        return None

    for candidate in _json_candidates(raw):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _json_candidates(raw: str) -> list[str]:
    out = [raw]
    out.extend(m.strip() for m in _FENCE_RE.findall(raw))

    start = raw.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(raw)):
            ch = raw[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out.append(raw[start:i + 1])
                    break
        break
    return out


@dataclass
class ChatReply:
    intent: str
    reply: str
    ops: list[dict[str, Any]] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    confidence: float = 0.0
    parse_error: str = ""
    raw: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent, "reply": self.reply, "ops": self.ops,
            "targets": self.targets, "confidence": self.confidence,
            "parse_error": self.parse_error,
        }


def parse_reply(text: str) -> ChatReply:
    """Parse the model's envelope, or return a reply carrying a schema error."""
    obj = _extract_json_object(text)
    if obj is None:
        return ChatReply(
            intent="", reply="", raw=text,
            parse_error="输出不是合法 JSON 对象，必须只输出一个 JSON",
        )

    intent = str(obj.get("intent", "") or "").strip()
    if intent not in VALID_INTENTS:
        return ChatReply(
            intent="", reply="", raw=text,
            parse_error=f"intent 必须是 {sorted(VALID_INTENTS)} 之一，收到 {intent!r}",
        )

    reply = str(obj.get("reply", "") or "").strip()
    if not reply:
        return ChatReply(
            intent="", reply="", raw=text,
            parse_error="reply 不能为空，需要给用户一句中文说明",
        )

    ops_raw = obj.get("ops")
    ops = [o for o in ops_raw if isinstance(o, dict)] if isinstance(ops_raw, list) else []
    if intent != "edit" and ops:
        # A non-edit intent carrying operations is contradictory; trust the
        # intent and drop the ops rather than silently editing the video.
        ops = []

    targets_raw = obj.get("targets")
    targets = [str(t) for t in targets_raw] if isinstance(targets_raw, list) else []

    try:
        confidence = float(obj.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    if intent == "edit" and not ops:
        return ChatReply(
            intent="", reply="", raw=text,
            parse_error="intent=edit 时 ops 不能为空；若无法定位请改用 clarify",
        )

    return ChatReply(
        intent=intent, reply=reply, ops=ops, targets=targets,
        confidence=max(0.0, min(1.0, confidence)), raw=text,
    )


# ---------------------------------------------------------------------------
# the agent
# ---------------------------------------------------------------------------
class ChatEditAgent:
    """Turns a user message into validated edit operations.

    ``generate`` is injected as ``(system_prompt, user_prompt, **kw) -> str`` so
    the agent can run against the in-process model, a remote endpoint, or a stub
    in tests without changing this class.
    """

    def __init__(
        self,
        generate: Callable[..., str],
        chat_config: dict[str, Any],
        editor_flags: dict[str, Any] | None = None,
    ) -> None:
        self._generate = generate
        self.config = chat_config or {}
        self.editor_flags = editor_flags or {}

        ctx = self.config.get("context")
        self.memory = Memory.from_config(ctx if isinstance(ctx, dict) else {})

        safety = self.config.get("safety")
        self.safety = safety if isinstance(safety, dict) else {}
        self.max_ops_per_turn = int(self.safety.get("max_ops_per_turn", 12) or 12)
        self.max_parse_retries = int(self.safety.get("max_parse_retries", 2) or 2)
        self.require_confirm_for = list(self.safety.get("require_confirm_for") or [])
        self.allow_rerender = bool(self.safety.get("allow_rerender", True))

        self.temperature = float(self.config.get("temperature", 0.2) or 0.2)
        self.max_tokens = int(self.config.get("max_tokens", 4096) or 4096)

        self._digest_max_shots = int(
            (ctx or {}).get("project_digest_max_shots", 60) or 60
        ) if isinstance(ctx, dict) else 60
        self._relevant_topk = int(
            (ctx or {}).get("relevant_shot_topk", 20) or 20
        ) if isinstance(ctx, dict) else 20

    # ------------------------------------------------------------------
    def available_ops(self) -> list[str]:
        """Operations the config leaves enabled, so the prompt can't offer more."""
        return [
            name for name in editor.ALL_OPS
            if bool(self.editor_flags.get(editor.OP_TO_FLAG.get(name, ""), True))
        ]

    def _build_context(self, project: Project, user_message: str) -> tuple[dict, list]:
        """Choose between a full and a focused project view."""
        digest = project.digest(max_shots=self._digest_max_shots)
        focused: list[dict[str, Any]] = []
        if digest.get("shots_truncated"):
            focused = project.find_shots(user_message, limit=self._relevant_topk)
        return digest, focused

    async def _compress_if_needed(self) -> dict[str, Any] | None:
        """Fold older turns into the running summary when over budget."""
        if not self.memory.needs_compression():
            return None

        before = self.memory.token_estimate()
        older = self.memory.turns_to_compress()
        system, user = chat_prompts.build_summary_request(
            older_turns=older,
            previous_summary=self.memory.summary,
            target_chars=int(self.memory.summary_target_tokens * self.memory.chars_per_token),
        )
        try:
            summary = await self._call_model(system, user, max_tokens=1024, temperature=0.1)
            self.memory.apply_compression(summary)
        except Exception as exc:
            # Never fail a user's edit because summarisation failed; forgetting
            # the oldest turns degrades quality but keeps the session usable.
            logger.warning("history compression failed, dropping oldest turns: %s", exc)
            self.memory.drop_oldest()

        return {
            "compressed": True,
            "turns_compressed": len(older),
            "tokens_before": before,
            "tokens_after": self.memory.token_estimate(),
            "summary_chars": len(self.memory.summary),
        }

    async def _call_model(
        self, system: str, user: str, *, max_tokens: int, temperature: float
    ) -> str:
        import asyncio
        import inspect

        if inspect.iscoroutinefunction(self._generate):
            return await self._generate(
                system, user, max_tokens=max_tokens, temperature=temperature
            )
        return await asyncio.to_thread(
            self._generate, system, user, max_tokens=max_tokens, temperature=temperature
        )

    # ------------------------------------------------------------------
    async def handle(self, project: Project, user_message: str) -> dict[str, Any]:
        """Run one conversational turn.

        Returns a payload for the frontend. Nothing is applied to the project
        here — operations are validated and returned so the caller decides
        whether to apply, and whether confirmation is required first.
        """
        message = str(user_message or "").strip()
        if not message:
            return {
                "intent": "clarify",
                "reply": "想改哪里？可以说「开头那句改成…」或「0:15 的画面换掉」。",
                "ops": [], "targets": [], "confidence": 0.0,
                "applied": False, "needs_confirm": [], "memory": self.memory.as_dict(),
            }

        compression = await self._compress_if_needed()
        digest, focused = self._build_context(project, message)

        base_payload = dict(
            project_digest=digest,
            focused_shots=focused,
            history_summary=self.memory.summary,
            recent_turns=self.memory.recent(),
            user_message=message,
            available_ops=self.available_ops(),
            max_ops_per_turn=self.max_ops_per_turn,
            template_registry=editor.template_registry(project),
            subtitle_drift=editor.subtitle_drift(project),
        )

        reply: ChatReply | None = None
        attempts: list[str] = []
        system = self._system_prompt()

        for attempt in range(self.max_parse_retries + 1):
            user_prompt = chat_prompts.build_user_payload(**base_payload)
            if attempts:
                # Re-ask with the schema error attached so the model repairs its
                # own output instead of the user seeing a failure.
                user_prompt += (
                    "\n\n【上一次输出不合规，请修正后重新输出】\n"
                    + "\n".join(f"- {a}" for a in attempts)
                )
            raw = await self._call_model(
                system, user_prompt,
                max_tokens=self.max_tokens, temperature=self.temperature,
            )
            candidate = parse_reply(raw)
            if not candidate.parse_error:
                reply = candidate
                break
            attempts.append(candidate.parse_error)
            logger.info("chat parse retry %d: %s", attempt + 1, candidate.parse_error)

        if reply is None:
            self.memory.add("user", message)
            fallback = "我没能理解这次的修改要求，能换个说法再讲一遍吗？比如指明大概时间点或原话片段。"
            self.memory.add("assistant", fallback, parse_failed=True)
            return {
                "intent": "clarify", "reply": fallback, "ops": [], "targets": [],
                "confidence": 0.0, "applied": False, "needs_confirm": [],
                "parse_errors": attempts, "memory": self.memory.as_dict(),
                "compression": compression,
            }

        # --- enforce the per-turn op ceiling ---
        trimmed = 0
        ops = reply.ops
        if len(ops) > self.max_ops_per_turn:
            trimmed = len(ops) - self.max_ops_per_turn
            ops = ops[: self.max_ops_per_turn]

        # --- validate against the real project without mutating it ---
        validation: list[dict[str, Any]] = []
        rejected: list[str] = []
        if ops:
            try:
                results = editor.precheck_ops(project, ops, self.editor_flags)
                validation = [r.as_dict() for r in results]
                rejected = [f"{r.op}: {r.message}" for r in results if not r.ok]
            except editor.EditError as exc:
                rejected = [str(exc)]
                validation = []

        if rejected:
            # The model located something that does not hold up. Tell the user
            # plainly rather than applying a partially-valid batch.
            note = "；".join(rejected[:3])
            reply_text = f"{reply.reply}\n\n不过这次改动没能通过校验：{note}。要不要换个说法或指明具体镜头？"
            self.memory.add("user", message)
            self.memory.add("assistant", reply_text, rejected=rejected)
            return {
                "intent": "clarify", "reply": reply_text, "ops": [],
                "targets": reply.targets, "confidence": reply.confidence,
                "applied": False, "needs_confirm": [], "validation": validation,
                "rejected": rejected, "memory": self.memory.as_dict(),
                "compression": compression,
            }

        needs_confirm = editor.high_risk_ops(ops, self.require_confirm_for)

        reply_text = reply.reply
        if trimmed:
            reply_text += f"\n\n（本轮先处理 {len(ops)} 处，还有 {trimmed} 处没做，可以继续告诉我。）"

        self.memory.add("user", message)
        self.memory.add(
            "assistant", reply_text,
            intent=reply.intent, targets=reply.targets,
            op_names=[str(o.get("op", "")) for o in ops],
        )

        return {
            "intent": reply.intent,
            "reply": reply_text,
            "ops": ops,
            "targets": reply.targets,
            "confidence": reply.confidence,
            "applied": False,
            "needs_confirm": needs_confirm,
            "validation": validation,
            "trimmed": trimmed,
            "allow_rerender": self.allow_rerender,
            "memory": self.memory.as_dict(),
            "compression": compression,
        }

    # ------------------------------------------------------------------
    def _system_prompt(self) -> str:
        """System prompt plus few-shots, rendered as a single instruction block.

        The few-shots are embedded here rather than sent as prior turns so they
        cannot be evicted by history compression.
        """
        examples = "\n\n".join(
            f"用户: {ex['user']}\n输出: {ex['assistant']}"
            for ex in chat_prompts.FEW_SHOT_EXAMPLES
        )
        return (
            chat_prompts.SYSTEM_PROMPT
            + "\n\n════════════════════════════════════════════════════════════\n"
            + "七、示例\n"
            + "════════════════════════════════════════════════════════════\n"
            + examples
        )

    # ------------------------------------------------------------------
    def reset(self) -> None:
        ctx = self.config.get("context")
        self.memory = Memory.from_config(ctx if isinstance(ctx, dict) else {})
