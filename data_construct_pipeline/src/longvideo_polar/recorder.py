"""Three-part trajectory recorder for long-horizon 口播稿 → 成片 rollouts.

The recorder is the single write path for everything a rollout produces. It is
fed from two seams that together cover 100% of the run:

1. :mod:`longvideo_polar.model_client` calls :meth:`TrajectoryRecorder.record_llm`
   for **every** model completion, tagged with the role that made it. Nothing a
   model says can bypass this, because the pipeline reaches its models only
   through that client.
2. :mod:`longvideo_polar.runner` calls :meth:`TrajectoryRecorder.record_tool_call`
   for every orchestrator step, from the pipeline's own broadcast events.

Streams written to ``<output>/trajectories``:

``orchestrator.jsonl``
    One multi-turn conversation: the 主流程 deciding tool after tool. Directly
    trainable as a long-horizon tool-calling episode.
``generator.jsonl`` / ``critic.jsonl``
    Lists of single-turn samples, one per skill invocation, bucketed by
    ``stage_task``. Directly trainable as content / reflection skills.
``interactions.jsonl``
    The edges between the streams: which orchestrator step handed off to which
    role, and what came back. This is what makes the three dumps a single causal
    trajectory rather than three unrelated logs.
``llm_calls.jsonl``
    Raw, untruncated-by-default model I/O for every role. The audit trail.
``episode.json``
    Run manifest: config snapshot, per-role stats, tool-call tally, final video,
    errors, timings.

Entry schema inside each role stream matches AICutPipeline's
``pipeline/trajectory.py`` (``user`` / ``thought`` / ``call`` / ``respond``) so the
existing SFT and RL builders can consume these files unchanged.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from longvideo_polar.roles import ALL_ROLES, ROLE_LABELS_ZH, Role, normalize_role

logger = logging.getLogger(__name__)

_MAX_RAW_OUTPUT_CHARS_DEFAULT = 200_000


def _ts() -> str:
    return datetime.now().isoformat()


# ---------------------------------------------------------------------------
# Payload compaction and reward extraction
#
# AICutPipeline owns the canonical implementations. Prefer them so dumps stay
# byte-compatible with the existing training builders; fall back to local
# equivalents when the pipeline tree is not on the path.
# ---------------------------------------------------------------------------


def _load_pipeline_helpers() -> tuple[Any, Any]:
    try:
        from pipeline.trajectory import (  # type: ignore[import-not-found]
            compact_payload,
            extract_reward_scores,
        )
    except Exception:  # noqa: BLE001 - pipeline tree simply not reachable
        return _local_compact_payload, _local_extract_reward_scores
    return compact_payload, extract_reward_scores


def _local_compact_payload(
    value: Any,
    *,
    max_depth: int = 5,
    max_items: int = 6,
    max_string: int = 240,
    _depth: int = 0,
) -> Any:
    """Bound payload size while preserving status, ids, paths and errors."""
    if isinstance(value, str):
        if len(value) <= max_string:
            return value
        return value[:max_string] + f"…<省略 {len(value) - max_string} 字符>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if _depth >= max_depth:
        if isinstance(value, dict):
            return {"_summary": f"dict({len(value)})"}
        if isinstance(value, list):
            return {"_summary": f"list({len(value)})"}
        return str(value)[:max_string]
    kwargs = {"max_depth": max_depth, "max_items": max_items, "max_string": max_string}
    if isinstance(value, list):
        items = [
            _local_compact_payload(item, _depth=_depth + 1, **kwargs)
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            items.append({"_truncated_count": len(value) - max_items})
        return items
    if isinstance(value, dict):
        priority = (
            "status", "shot_id", "segment_id", "render_success", "rendered_path",
            "output_path", "error", "message", "reasons", "issues",
            "failed_count", "success_count",
        )
        keys = list(value)
        ordered = [key for key in priority if key in value]
        ordered.extend(key for key in keys if key not in ordered)
        selected = ordered[: max_items * 2]
        result: dict[str, Any] = {
            str(key): _local_compact_payload(value[key], _depth=_depth + 1, **kwargs)
            for key in selected
        }
        if len(ordered) > len(selected):
            result["_truncated_keys"] = len(ordered) - len(selected)
        return result
    return _local_compact_payload(str(value), _depth=_depth + 1, **kwargs)


_REWARD_KEY_SUFFIX = "_score"
_REWARD_KEYS = {
    "overall_score", "quality_score", "match_score", "severity_score",
    "similarity_score", "semantic_similarity_score", "score", "similarity",
}


def _as_reward_score(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return max(0.0, min(1.0, score))


def _local_extract_reward_scores(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Scan ``respond`` entries for bounded numeric quality scores."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def walk(value: Any, path: list[str], ctx: dict[str, str]) -> None:
        if isinstance(value, dict):
            shot_id = str(value.get("shot_id") or ctx.get("shot_id") or "")
            for key, child in value.items():
                key_s = str(key)
                child_path = [*path, key_s]
                lowered = key_s.lower()
                if lowered in _REWARD_KEYS or lowered.endswith(_REWARD_KEY_SUFFIX):
                    score = _as_reward_score(child)
                    if score is not None:
                        source = ".".join(child_path)
                        dedupe = (ctx["entry_index"], ctx["tool"], shot_id, source)
                        if dedupe not in seen:
                            seen.add(dedupe)
                            rows.append({
                                "entry_index": ctx["entry_index"],
                                "stage": ctx["stage"],
                                "tool": ctx["tool"],
                                "shot_id": shot_id,
                                "score_name": source,
                                "reward_score": round(score, 4),
                                "source_path": source,
                            })
                if isinstance(child, (dict, list)):
                    walk(child, child_path, {**ctx, "shot_id": shot_id})
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, (dict, list)):
                    walk(item, [*path, str(idx)], ctx)

    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("role") != "respond":
            continue
        output = entry.get("tool_output")
        if not isinstance(output, dict):
            continue
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        walk(output, [], {
            "entry_index": idx,
            "stage": str(metadata.get("stage") or ""),
            "tool": str(entry.get("tool") or ""),
            "shot_id": "",
        })
    return rows


def summarize_reward_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate flat reward rows into a compact summary."""
    scores = [
        score for row in rows
        if (score := _as_reward_score(row.get("reward_score"))) is not None
    ]
    if not scores:
        return {"count": 0, "mean_reward": None, "min_reward": None, "max_reward": None}
    by_stage: dict[str, list[float]] = {}
    for row in rows:
        score = _as_reward_score(row.get("reward_score"))
        if score is None:
            continue
        by_stage.setdefault(str(row.get("stage") or "unknown"), []).append(score)
    return {
        "count": len(scores),
        "mean_reward": round(sum(scores) / len(scores), 4),
        "min_reward": round(min(scores), 4),
        "max_reward": round(max(scores), 4),
        "by_stage": {
            stage: {
                "count": len(values),
                "mean_reward": round(sum(values) / len(values), 4),
            }
            for stage, values in sorted(by_stage.items())
        },
    }


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class LLMCall:
    """One model completion, attributed to a role."""

    index: int
    role: Role
    stage: str
    stage_task: str
    tool: str
    step: int
    system_prompt: str
    user_content: str
    raw_output: str
    cleaned_output: str
    thinking: str
    tool_name: str
    arguments: dict[str, Any]
    parse_ok: bool
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    model: str
    provider: str = ""
    # ``length`` marks a completion the server cut off at max_tokens. A sample
    # that failed to parse because it was truncated is a budget problem, not a
    # model-capability one, and QC has to be able to tell them apart.
    finish_reason: str = ""
    error: str = ""
    timestamp: str = field(default_factory=_ts)

    def to_json(self, *, max_raw_chars: int) -> dict[str, Any]:
        def clip(text: str) -> str:
            if max_raw_chars <= 0 or len(text) <= max_raw_chars:
                return text
            return text[:max_raw_chars] + f"…<truncated {len(text) - max_raw_chars} chars>"

        return {
            "index": self.index,
            "role": self.role.value,
            "role_label": ROLE_LABELS_ZH[self.role],
            "stage": self.stage,
            "stage_task": self.stage_task,
            "tool": self.tool,
            "step": self.step,
            "model": self.model,
            "provider": self.provider,
            "system_prompt": clip(self.system_prompt),
            "user_content": clip(self.user_content),
            "raw_output": clip(self.raw_output),
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "parse_ok": self.parse_ok,
            "latency_ms": round(self.latency_ms, 1),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "truncated": self.finish_reason == "length",
            "error": self.error,
            "timestamp": self.timestamp,
        }


@dataclass
class ToolCallRecord:
    """One orchestrator step: decision, dispatch, observation."""

    step: int
    tool: str
    stage: str
    executed_by: Role
    arguments: dict[str, Any]
    output: dict[str, Any]
    thought: str
    status: str
    precondition_missing: list[str]
    external: bool
    timestamp: str = field(default_factory=_ts)


@dataclass
class Interaction:
    """A handoff edge from the orchestrator to a generator/critic invocation."""

    step: int
    tool: str
    from_role: Role
    to_role: Role
    stage_task: str
    llm_call_index: int
    request_chars: int
    response_chars: int
    parse_ok: bool
    latency_ms: float
    status: str
    timestamp: str = field(default_factory=_ts)

    def to_json(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "tool": self.tool,
            "from_role": self.from_role.value,
            "to_role": self.to_role.value,
            "from_role_label": ROLE_LABELS_ZH[self.from_role],
            "to_role_label": ROLE_LABELS_ZH[self.to_role],
            "stage_task": self.stage_task,
            "llm_call_index": self.llm_call_index,
            "request_chars": self.request_chars,
            "response_chars": self.response_chars,
            "parse_ok": self.parse_ok,
            "latency_ms": round(self.latency_ms, 1),
            "status": self.status,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class TrajectoryRecorder:
    """Thread-safe collector for a single rollout's three-part trajectory.

    The pipeline runs model calls via ``asyncio.to_thread``, so every mutating
    method takes an internal lock.
    """

    def __init__(
        self,
        *,
        episode_id: str,
        output_dir: str | Path,
        task_description: str = "",
        input_script: str = "",
        max_raw_output_chars: int = _MAX_RAW_OUTPUT_CHARS_DEFAULT,
        store_full_prompts: bool = True,
        metadata: dict[str, Any] | None = None,
        require_final_video: bool = True,
        benign_error: Callable[[str], bool] | None = None,
    ) -> None:
        self.episode_id = episode_id
        self.output_dir = Path(output_dir)
        self.traj_dir = self.output_dir / "trajectories"
        self.task_description = task_description or "将口播稿转化为完整成片"
        self.input_script = input_script
        self.max_raw_output_chars = int(max_raw_output_chars)
        self.store_full_prompts = bool(store_full_prompts)
        self.metadata: dict[str, Any] = dict(metadata or {})
        # A text-only run is asked for the trajectory, not for a file. Judging it
        # by ``final_video_exists`` marks every episode failed and makes the
        # exported corpus look like a total loss.
        self.require_final_video = bool(require_final_video)
        self._benign_error = benign_error or (lambda _message: False)

        self._lock = threading.RLock()
        self._start_time = time.time()
        self._llm_calls: list[LLMCall] = []
        self._tool_calls: list[ToolCallRecord] = []
        self._interactions: list[Interaction] = []
        self._errors: list[str] = []
        # Set by the runner before each dispatch so model calls made inside the
        # dispatch can be attributed to the owning orchestrator step.
        self._current_step: int = 0
        self._current_tool: str = ""
        self._compact_payload, self._extract_reward_scores = _load_pipeline_helpers()

    # -- context ----------------------------------------------------------

    def begin_step(self, step: int, tool: str) -> None:
        """Mark the orchestrator step that subsequent model calls belong to."""
        with self._lock:
            self._current_step = int(step)
            self._current_tool = str(tool or "")

    def add_error(self, message: str) -> None:
        with self._lock:
            text = str(message or "").strip()
            if text:
                self._errors.append(text)

    def update_metadata(self, **values: Any) -> None:
        with self._lock:
            self.metadata.update(values)

    # -- recording --------------------------------------------------------

    def record_llm(
        self,
        *,
        role: Role | str,
        system_prompt: str,
        user_content: str,
        raw_output: str,
        cleaned_output: str = "",
        thinking: str = "",
        tool_name: str = "",
        arguments: dict[str, Any] | None = None,
        parse_ok: bool = False,
        latency_ms: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model: str = "",
        provider: str = "",
        stage: str = "",
        stage_task: str = "",
        finish_reason: str = "",
        error: str = "",
    ) -> LLMCall:
        """Record one model completion and, for skill roles, its handoff edge.

        An orchestrator completion tagged ``stage="agent_loop"`` *is* the start of a
        new long-horizon step, so it advances the step counter. The runner later
        confirms the authoritative step index via :meth:`begin_step` once the
        pipeline announces its dispatch. Skill calls inherit whichever step is
        current, which is what links them back to the decision that caused them.
        """
        resolved = normalize_role(role)
        with self._lock:
            resolved_stage = stage or ("agent_loop" if resolved is Role.ORCHESTRATOR else "")
            if resolved is Role.ORCHESTRATOR and resolved_stage == "agent_loop":
                self._current_step += 1
                self._current_tool = tool_name or ""

            call = LLMCall(
                index=len(self._llm_calls),
                role=resolved,
                stage=resolved_stage,
                stage_task=stage_task or _infer_stage_task(user_content),
                tool=self._current_tool,
                step=self._current_step,
                system_prompt=system_prompt if self.store_full_prompts else "",
                user_content=user_content,
                raw_output=raw_output,
                cleaned_output=cleaned_output or raw_output,
                thinking=thinking,
                tool_name=tool_name,
                arguments=dict(arguments or {}),
                parse_ok=bool(parse_ok),
                latency_ms=float(latency_ms),
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                model=model,
                provider=provider,
                finish_reason=finish_reason,
                error=error,
            )
            self._llm_calls.append(call)

            if resolved is not Role.ORCHESTRATOR:
                self._interactions.append(Interaction(
                    step=call.step,
                    tool=call.tool,
                    from_role=Role.ORCHESTRATOR,
                    to_role=resolved,
                    stage_task=call.stage_task,
                    llm_call_index=call.index,
                    request_chars=len(user_content or ""),
                    response_chars=len(raw_output or ""),
                    parse_ok=call.parse_ok,
                    latency_ms=call.latency_ms,
                    status="error" if error else "ok",
                ))
            return call

    def record_tool_call(
        self,
        *,
        step: int,
        tool: str,
        stage: str,
        arguments: dict[str, Any] | None,
        output: dict[str, Any] | None,
        thought: str = "",
        precondition_missing: list[str] | None = None,
        external: bool = False,
        executed_by: Role | str = Role.ORCHESTRATOR,
    ) -> ToolCallRecord:
        """Record one completed orchestrator step."""
        payload = output if isinstance(output, dict) else {}
        with self._lock:
            record = ToolCallRecord(
                step=int(step),
                tool=str(tool or ""),
                stage=str(stage or ""),
                executed_by=normalize_role(executed_by),
                arguments=dict(arguments or {}),
                output=payload,
                thought=str(thought or ""),
                status=str(payload.get("status") or "unknown"),
                precondition_missing=list(precondition_missing or []),
                external=bool(external),
            )
            self._tool_calls.append(record)
            return record

    def merge_agent_trace(self, rows: list[dict[str, Any]]) -> int:
        """Upgrade recorded steps with full payloads from ``agent_trace.jsonl``.

        Broadcast events only carry the compacted observation the orchestrator will
        see next turn. AICutPipeline separately writes each step's untruncated
        arguments and output to ``agent_trace.jsonl``, so merging that file after
        the run gives the dump full fidelity without changing pipeline code.

        Args:
            rows: Parsed ``agent_trace.jsonl`` entries.

        Returns:
            How many step records were upgraded or added.
        """
        with self._lock:
            by_key = {(record.step, record.tool): record for record in self._tool_calls}
            changed = 0
            for row in rows:
                if not isinstance(row, dict) or row.get("event") is not None:
                    continue  # skill_raw_output rows carry no step decision
                tool = str(row.get("tool_name") or "")
                if not tool:
                    continue
                try:
                    step = int(row.get("step") or 0)
                except (TypeError, ValueError):
                    continue
                output = row.get("output") if isinstance(row.get("output"), dict) else {}
                arguments = row.get("arguments") if isinstance(row.get("arguments"), dict) else {}
                record = by_key.get((step, tool))
                if record is None:
                    record = ToolCallRecord(
                        step=step,
                        tool=tool,
                        stage="",
                        executed_by=Role.ORCHESTRATOR,
                        arguments=arguments,
                        output=output,
                        thought="",
                        status=str(output.get("status") or "unknown"),
                        precondition_missing=[],
                        external=False,
                        timestamp=str(row.get("timestamp") or _ts()),
                    )
                    self._tool_calls.append(record)
                    by_key[(step, tool)] = record
                    changed += 1
                    continue
                if output:
                    record.output = output
                    record.status = str(output.get("status") or record.status)
                if arguments:
                    record.arguments = arguments
                changed += 1
            self._tool_calls.sort(key=lambda item: (item.step, item.timestamp))
            return changed

    # -- views ------------------------------------------------------------

    @property
    def llm_calls(self) -> list[LLMCall]:
        with self._lock:
            return list(self._llm_calls)

    @property
    def tool_calls(self) -> list[ToolCallRecord]:
        with self._lock:
            return list(self._tool_calls)

    @property
    def interactions(self) -> list[Interaction]:
        with self._lock:
            return list(self._interactions)

    def role_call_counts(self) -> dict[str, int]:
        with self._lock:
            counts = {role.value: 0 for role in ALL_ROLES}
            for call in self._llm_calls:
                counts[call.role.value] += 1
            return counts

    def role_providers(self) -> dict[str, list[str]]:
        """Providers that actually answered for each role, from the recorded calls."""
        with self._lock:
            providers: dict[str, set[str]] = {role.value: set() for role in ALL_ROLES}
            for call in self._llm_calls:
                if call.provider:
                    providers[call.role.value].add(call.provider)
            return {role: sorted(names) for role, names in providers.items()}

    # -- stream construction ---------------------------------------------

    def build_orchestrator_entries(self) -> list[dict[str, Any]]:
        """Build the 主流程 stream as one multi-turn tool-calling episode."""
        with self._lock:
            entries: list[dict[str, Any]] = [{
                "role": "user",
                "content": self.task_description,
                "metadata": {
                    "stage": "init",
                    "episode_id": self.episode_id,
                    "input_script_chars": len(self.input_script),
                    "ts": _ts(),
                },
            }]
            thoughts = {
                call.step: call.thinking
                for call in self._llm_calls
                if call.role is Role.ORCHESTRATOR and call.thinking
            }
            for record in self._tool_calls:
                thought = record.thought or thoughts.get(record.step, "")
                if thought:
                    entries.append({
                        "role": "thought",
                        "content": thought,
                        "tool": record.tool,
                        "metadata": {
                            "tool": record.tool,
                            "stage": record.stage,
                            "step": record.step,
                            "ts": record.timestamp,
                        },
                    })
                entries.append({
                    "role": "call",
                    "tool": record.tool,
                    "tool_input": self._compact_payload(record.arguments),
                    "metadata": {
                        "stage": record.stage,
                        "step": record.step,
                        "executed_by": record.executed_by.value,
                        "external_tool": record.external,
                        "ts": record.timestamp,
                    },
                })
                entries.append({
                    "role": "respond",
                    "tool": record.tool,
                    "content": _summarize_output(record.tool, record.output),
                    "tool_output": self._compact_payload(record.output),
                    "metadata": {
                        "stage": record.stage,
                        "step": record.step,
                        "status": record.status,
                        "precondition_missing": record.precondition_missing,
                        "ts": record.timestamp,
                    },
                })
            return entries

    def build_skill_samples(self, role: Role) -> list[dict[str, Any]]:
        """Build single-turn samples for the generator or critic stream."""
        if role is Role.ORCHESTRATOR:
            raise ValueError("orchestrator uses build_orchestrator_entries()")
        with self._lock:
            samples: list[dict[str, Any]] = []
            for call in self._llm_calls:
                if call.role is not role:
                    continue
                messages = []
                if call.system_prompt:
                    messages.append({"role": "system", "content": call.system_prompt})
                messages.append({"role": "user", "content": call.user_content})
                messages.append({"role": "assistant", "content": call.cleaned_output})
                samples.append({
                    "messages": messages,
                    "metadata": {
                        "episode_id": self.episode_id,
                        "role": role.value,
                        "role_label": ROLE_LABELS_ZH[role],
                        "stage_task": call.stage_task,
                        "tool": call.tool,
                        "step": call.step,
                        "llm_call_index": call.index,
                        "parse_ok": call.parse_ok,
                        "latency_ms": round(call.latency_ms, 1),
                        "completion_tokens": call.completion_tokens,
                        # Provenance travels with the sample, not just the manifest:
                        # a distilled Opus critic sample and a self-generated Qwen one
                        # must never be indistinguishable once shards are merged.
                        "model": call.model,
                        "provider": call.provider,
                        "error": call.error,
                        "ts": call.timestamp,
                    },
                })
            return samples

    def build_orchestrator_sft_messages(self) -> list[dict[str, Any]]:
        """Convert the 主流程 stream into OpenAI tool-calling messages."""
        entries = self.build_orchestrator_entries()
        messages: list[dict[str, Any]] = []
        system_prompt = next(
            (call.system_prompt for call in self.llm_calls
             if call.role is Role.ORCHESTRATOR and call.system_prompt),
            "",
        )
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for entry in entries:
            kind = entry.get("role")
            if kind == "user":
                messages.append({"role": "user", "content": entry.get("content", "")})
            elif kind == "thought":
                messages.append({
                    "role": "assistant",
                    "content": f"<think>{entry.get('content', '')}</think>",
                })
            elif kind == "call":
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": entry.get("tool", ""),
                            "arguments": json.dumps(
                                entry.get("tool_input", {}), ensure_ascii=False
                            ),
                        },
                    }],
                })
            elif kind == "respond":
                messages.append({
                    "role": "tool",
                    "name": entry.get("tool", ""),
                    "content": json.dumps(entry.get("tool_output", {}), ensure_ascii=False),
                })
        return messages

    # -- persistence ------------------------------------------------------

    def save(self, *, result: dict[str, Any] | None = None) -> dict[str, str]:
        """Write every stream to disk and return a name → path map."""
        with self._lock:
            self.traj_dir.mkdir(parents=True, exist_ok=True)
            (self.traj_dir / "sft").mkdir(exist_ok=True)
            (self.traj_dir / "rl").mkdir(exist_ok=True)
            paths: dict[str, str] = {}

            orch_entries = self.build_orchestrator_entries()
            reward_scores = self._extract_reward_scores(orch_entries)
            reward_summary = summarize_reward_scores(reward_scores)

            paths["orchestrator"] = self._write_json(
                self.traj_dir / "orchestrator.jsonl",
                {
                    "episode_id": self.episode_id,
                    "role": Role.ORCHESTRATOR.value,
                    "role_label": ROLE_LABELS_ZH[Role.ORCHESTRATOR],
                    "timestamp": _ts(),
                    "elapsed_seconds": round(time.time() - self._start_time, 1),
                    "num_entries": len(orch_entries),
                    "num_tool_calls": len(self._tool_calls),
                    "raw_trajectory": orch_entries,
                    "sft_messages": self.build_orchestrator_sft_messages(),
                    "reward_scores": reward_scores,
                    "reward_summary": reward_summary,
                    "metadata": dict(self.metadata),
                },
            )

            for role in (Role.GENERATOR, Role.CRITIC):
                samples = self.build_skill_samples(role)
                by_task: dict[str, int] = {}
                for sample in samples:
                    task = str(sample["metadata"].get("stage_task") or "unknown")
                    by_task[task] = by_task.get(task, 0) + 1
                paths[role.value] = self._write_json(
                    self.traj_dir / f"{role.value}.jsonl",
                    {
                        "episode_id": self.episode_id,
                        "role": role.value,
                        "role_label": ROLE_LABELS_ZH[role],
                        "timestamp": _ts(),
                        "num_samples": len(samples),
                        "samples_by_stage_task": by_task,
                        "samples": samples,
                        "metadata": dict(self.metadata),
                    },
                )

            paths["interactions"] = self._write_lines(
                self.traj_dir / "interactions.jsonl",
                [item.to_json() for item in self._interactions],
            )
            paths["llm_calls"] = self._write_lines(
                self.traj_dir / "llm_calls.jsonl",
                [
                    call.to_json(max_raw_chars=self.max_raw_output_chars)
                    for call in self._llm_calls
                ],
            )

            # Role-scoped SFT shards, ready for the training builders.
            paths["sft_orchestrator"] = self._write_lines(
                self.traj_dir / "sft" / "orchestrator_messages.jsonl",
                [{
                    "messages": self.build_orchestrator_sft_messages(),
                    "metadata": {
                        "episode_id": self.episode_id,
                        "role": Role.ORCHESTRATOR.value,
                        "num_tool_calls": len(self._tool_calls),
                        "reward_summary": reward_summary,
                        # A 主流程 trajectory is shaped by which model played critic
                        # inside the loop, so the shard has to say which mix produced
                        # it. Skill shards carry this per row; here it is per episode.
                        "annotation_profile": self.metadata.get("annotation_profile", ""),
                        "providers": self.role_providers(),
                    },
                }],
            )
            for role in (Role.GENERATOR, Role.CRITIC):
                paths[f"sft_{role.value}"] = self._write_lines(
                    self.traj_dir / "sft" / f"{role.value}_messages.jsonl",
                    self.build_skill_samples(role),
                )

            paths["rl_rewards"] = self._write_lines(
                self.traj_dir / "rl" / "reward_scores.jsonl", reward_scores
            )

            paths["episode"] = self._write_json(
                self.output_dir / "episode.json",
                self.build_episode_manifest(result=result, reward_summary=reward_summary, paths=paths),
            )
            logger.info(
                "Recorded episode %s: %d orchestrator steps, %s LLM calls, %d interactions",
                self.episode_id, len(self._tool_calls), self.role_call_counts(),
                len(self._interactions),
            )
            return paths

    def build_episode_manifest(
        self,
        *,
        result: dict[str, Any] | None = None,
        reward_summary: dict[str, Any] | None = None,
        paths: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Assemble the top-level run manifest."""
        with self._lock:
            run_result = dict(result or {})
            reported = _dedupe_errors([
                *self._errors,
                *(str(item) for item in run_result.get("errors", [])),
            ])
            # A switched-off stage announcing itself describes the configuration,
            # not a defect. Keep it in the record as a notice so the run stays
            # auditable, but do not let it decide the episode's status.
            notices = [item for item in reported if self._benign_error(item)]
            errors = [item for item in reported if not self._benign_error(item)]
            tool_tally: dict[str, int] = {}
            for record in self._tool_calls:
                tool_tally[record.tool] = tool_tally.get(record.tool, 0) + 1
            final_video = str(run_result.get("final_video_path") or "")
            video_exists = bool(final_video) and Path(final_video).is_file()
            return {
                "episode_id": self.episode_id,
                "timestamp": _ts(),
                "elapsed_seconds": round(time.time() - self._start_time, 1),
                "task_description": self.task_description,
                "input_script_chars": len(self.input_script),
                "status": self._status(video_exists, errors),
                "final_video_path": final_video,
                "final_video_exists": video_exists,
                "project_id": run_result.get("project_id", ""),
                "output_dir": str(self.output_dir),
                "roles": {
                    role.value: {
                        "label": ROLE_LABELS_ZH[role],
                        "llm_calls": self.role_call_counts()[role.value],
                        "completion_tokens": sum(
                            call.completion_tokens for call in self._llm_calls
                            if call.role is role
                        ),
                        "parse_failures": sum(
                            1 for call in self._llm_calls
                            if call.role is role and not call.parse_ok
                        ),
                        "errors": sum(
                            1 for call in self._llm_calls if call.role is role and call.error
                        ),
                        # Observed, not configured: this is what actually answered.
                        "models_used": sorted({
                            call.model for call in self._llm_calls
                            if call.role is role and call.model
                        }),
                        "providers_used": self.role_providers().get(role.value, []),
                    }
                    for role in ALL_ROLES
                },
                "orchestrator_steps": len(self._tool_calls),
                "tool_call_counts": tool_tally,
                "stages_completed": list(run_result.get("stages_completed", [])),
                "interactions": {
                    "total": len(self._interactions),
                    "to_generator": sum(
                        1 for item in self._interactions if item.to_role is Role.GENERATOR
                    ),
                    "to_critic": sum(
                        1 for item in self._interactions if item.to_role is Role.CRITIC
                    ),
                },
                "reward_summary": reward_summary or {},
                "errors": errors,
                "notices": notices,
                "truncated_calls": sum(
                    1 for call in self._llm_calls if call.finish_reason == "length"
                ),
                "artifacts": paths or {},
                "metadata": dict(self.metadata),
            }

    def _status(self, video_exists: bool, errors: list[str]) -> str:
        """Grade the run against what this configuration was asked to produce."""
        if not self.require_final_video:
            return "completed" if not errors else "completed_with_errors"
        if video_exists:
            return "completed" if not errors else "completed_with_errors"
        return "failed"

    @staticmethod
    def _write_json(path: Path, payload: Any) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return str(path)

    @staticmethod
    def _write_lines(path: Path, rows: list[Any]) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return str(path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dedupe_errors(messages: list[str]) -> list[str]:
    """Collapse the same failure reported twice, keeping the more specific text.

    The pipeline both broadcasts an ``error`` event and appends the same reason to
    its result dict, and the broadcast form carries a ``[stage]`` prefix. Matching
    on containment keeps the prefixed message and drops the bare duplicate.
    """
    kept: list[str] = []
    for message in messages:
        text = str(message or "").strip()
        if not text:
            continue
        replaced = False
        for index, existing in enumerate(kept):
            if text == existing or text in existing:
                replaced = True
                break
            if existing in text:
                kept[index] = text  # the longer form names the stage
                replaced = True
                break
        if not replaced:
            kept.append(text)
    return kept


def _infer_stage_task(user_content: str) -> str:
    """Recover the ``stage_task`` label from a skill payload, if present."""
    text = str(user_content or "").lstrip()
    if not text.startswith("{"):
        return ""
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    for key in ("stage_task", "task", "skill", "stage"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _summarize_output(tool: str, output: dict[str, Any]) -> str:
    """Build the short ``respond`` summary line for a tool result."""
    parts = [tool]
    status = str(output.get("status") or "").strip()
    if status:
        parts.append(status)
    for label, keys in (
        ("总数", ("shot_count", "num_shots", "total", "total_count")),
        ("成功", ("render_success_count", "success_count", "passed_count")),
        ("失败", ("render_failed_count", "failed_count", "error_count")),
    ):
        for key in keys:
            if output.get(key) is not None:
                parts.append(f"{label} {output[key]}")
                break
    if len(parts) == 1:
        parts.append("已返回结果")
    return "；".join(str(part) for part in parts)
