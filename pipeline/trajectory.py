"""SFT trajectory logging in standard Tool-use format.

Each pipeline run produces a trajectory file that can be directly used for
SFT training. Format:
    [
      {"role": "user",           "content": "<task description>"},
      {"role": "thought",        "content": "<reasoning>"},
      {"role": "call",           "tool": "Tool_X", "tool_input": {...}},
      {"role": "respond",        "content": "<tool output summary>", "tool_output": {...}},
      {"role": "human_feedback", "content": "<HITL correction>", "metadata": {...}},
      ...
    ]

Human feedback entries are formatted as exemplary self-reflection demonstrations
or negative-feedback correction trajectories for training.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def compact_payload(
    value: Any,
    *,
    max_depth: int = 5,
    max_items: int = 6,
    max_string: int = 240,
    _depth: int = 0,
) -> Any:
    """Bound trajectory payload size while preserving status, IDs, paths and errors."""
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
    if isinstance(value, list):
        items = [
            compact_payload(
                item,
                max_depth=max_depth,
                max_items=max_items,
                max_string=max_string,
                _depth=_depth + 1,
            )
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            items.append({"_truncated_count": len(value) - max_items})
        return items
    if isinstance(value, dict):
        priority = (
            "status", "shot_id", "segment_id", "clip_id", "render_success",
            "render_status", "render_backend", "rendered_path", "output_path",
            "error", "message", "reasons", "issues", "failed_count",
            "success_count", "render_failed_count", "render_success_count",
        )
        keys = list(value)
        id_keyed_map = (
            len(keys) > max_items
            and all(isinstance(item, dict) for item in value.values())
            and all(str(key).replace("_", "").replace("-", "").isalnum() for key in keys)
        )
        if id_keyed_map:
            selected_keys = keys[:max_items]
            result = {
                str(key): compact_payload(
                    value[key],
                    max_depth=max_depth,
                    max_items=max_items,
                    max_string=max_string,
                    _depth=_depth + 1,
                )
                for key in selected_keys
            }
            result["_truncated_items"] = len(keys) - len(selected_keys)
            return result
        ordered = [key for key in priority if key in value]
        ordered.extend(key for key in keys if key not in ordered)
        selected = ordered[: max_items * 2]
        result = {
            str(key): compact_payload(
                value[key],
                max_depth=max_depth,
                max_items=max_items,
                max_string=max_string,
                _depth=_depth + 1,
            )
            for key in selected
        }
        if len(ordered) > len(selected):
            result["_truncated_keys"] = len(ordered) - len(selected)
        return result
    return compact_payload(
        str(value),
        max_depth=max_depth,
        max_items=max_items,
        max_string=max_string,
        _depth=_depth + 1,
    )


_REWARD_SCORE_KEYS = {
    "overall_score",
    "quality_score",
    "match_score",
    "severity_score",
    "similarity_score",
    "semantic_similarity_score",
    "score",
    "similarity",
}


def extract_reward_scores_from_output(
    tool_output: dict[str, Any],
    *,
    entry_index: int = -1,
    stage: str = "",
    tool: str = "",
) -> list[dict[str, Any]]:
    """Extract model/quality scores from one tool output for RL reward export."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def emit(
        *,
        score_name: str,
        value: Any,
        source_path: str,
        shot_id: str = "",
        dimension: str = "",
    ) -> None:
        score = _as_reward_score(value)
        if score is None:
            return
        key = (entry_index, stage, tool, shot_id, score_name, source_path)
        if key in seen:
            return
        seen.add(key)
        rows.append({
            "entry_index": entry_index,
            "stage": stage,
            "tool": tool,
            "shot_id": shot_id,
            "dimension": dimension,
            "score_name": score_name,
            "reward_score": round(score, 4),
            "source_path": source_path,
        })

    def walk(value: Any, path: list[str], shot_id: str = "") -> None:
        if isinstance(value, dict):
            current_shot = str(value.get("shot_id") or shot_id or "")

            dim_scores = value.get("dimension_scores")
            if isinstance(dim_scores, dict):
                for dim, score in dim_scores.items():
                    emit(
                        score_name=f"dimension_scores.{dim}",
                        value=score,
                        source_path=".".join([*path, "dimension_scores", str(dim)]),
                        shot_id=current_shot,
                        dimension=str(dim),
                    )

            dim_obs = value.get("dimension_observation")
            if isinstance(dim_obs, dict):
                for dim, obs in dim_obs.items():
                    if isinstance(obs, dict) and "score" in obs:
                        emit(
                            score_name=f"dimension_observation.{dim}",
                            value=obs.get("score"),
                            source_path=".".join([*path, "dimension_observation", str(dim), "score"]),
                            shot_id=current_shot,
                            dimension=str(dim),
                        )

            for key, child in value.items():
                key_s = str(key)
                child_path = [*path, key_s]
                child_shot = current_shot
                if isinstance(child, dict) and _path_names_id_map(path, key_s):
                    child_shot = key_s
                if _is_reward_score_key(key_s, path):
                    emit(
                        score_name=".".join(child_path),
                        value=child,
                        source_path=".".join(child_path),
                        shot_id=current_shot,
                    )
                if key_s in {"dimension_scores", "dimension_observation"}:
                    continue
                if isinstance(child, (dict, list)):
                    walk(child, child_path, child_shot)
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                item_path = [*path, str(idx)]
                if isinstance(item, (dict, list)):
                    walk(item, item_path, shot_id)

    if isinstance(tool_output, dict):
        walk(tool_output, [], "")
    return rows


def extract_reward_scores(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract all reward-capable model scores from a saved trajectory."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def add_row(row: dict[str, Any]) -> None:
        score = _as_reward_score(row.get("reward_score"))
        if score is None:
            score = _as_reward_score(row.get("score"))
        if score is None:
            return
        normalized = dict(row)
        normalized["reward_score"] = round(score, 4)
        normalized.setdefault("entry_index", -1)
        normalized.setdefault("stage", "")
        normalized.setdefault("tool", "")
        normalized.setdefault("shot_id", "")
        normalized.setdefault("dimension", "")
        normalized.setdefault("score_name", "score")
        normalized.setdefault("source_path", "")
        key = (
            normalized.get("entry_index"),
            normalized.get("stage"),
            normalized.get("tool"),
            normalized.get("shot_id"),
            normalized.get("score_name"),
            normalized.get("source_path"),
        )
        if key in seen:
            return
        seen.add(key)
        rows.append(normalized)

    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        stage = str(metadata.get("stage", "") or "")
        tool = str(entry.get("tool") or metadata.get("tool") or "")

        for row in metadata.get("reward_scores", []) if isinstance(metadata.get("reward_scores"), list) else []:
            if not isinstance(row, dict):
                continue
            patched = dict(row)
            patched["entry_index"] = idx
            patched.setdefault("stage", stage)
            patched.setdefault("tool", tool)
            add_row(patched)

        if entry.get("role") != "respond":
            continue
        output = entry.get("tool_output")
        if isinstance(output, dict):
            for row in extract_reward_scores_from_output(
                output,
                entry_index=idx,
                stage=stage,
                tool=tool,
            ):
                add_row(row)
    return rows


def summarize_reward_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [
        score
        for row in rows
        if (score := _as_reward_score(row.get("reward_score"))) is not None
    ]
    if not scores:
        return {"count": 0, "mean_reward": None, "min_reward": None, "max_reward": None}
    by_stage: dict[str, list[float]] = {}
    for row in rows:
        score = _as_reward_score(row.get("reward_score"))
        if score is None:
            continue
        stage = str(row.get("stage") or "unknown")
        by_stage.setdefault(stage, []).append(score)
    return {
        "count": len(scores),
        "mean_reward": round(sum(scores) / len(scores), 4),
        "min_reward": round(min(scores), 4),
        "max_reward": round(max(scores), 4),
        "by_stage": {
            stage: {
                "count": len(stage_scores),
                "mean_reward": round(sum(stage_scores) / len(stage_scores), 4),
            }
            for stage, stage_scores in sorted(by_stage.items())
            if stage_scores
        },
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


def _is_reward_score_key(key: str, path: list[str]) -> bool:
    key_l = key.lower()
    if key_l in _REWARD_SCORE_KEYS:
        return True
    if key_l.endswith("_score"):
        return True
    return key_l == "score" and any(
        part in {
            "scores",
            "shots",
            "visual",
            "severity_scores",
            "dimension_observation",
            "tracks",
            "candidates",
        }
        for part in path
    )


def _path_names_id_map(path: list[str], key: str) -> bool:
    if not path:
        return False
    return path[-1] in {
        "shots",
        "visual",
        "audio",
        "severity_scores",
        "shot_scores",
        "dimension_scores_by_shot",
    } and bool(key)


class TrajectoryLogger:
    """Accumulates trajectory entries and serializes to SFT-ready JSONL."""

    def __init__(self, project_id: str, output_dir: str) -> None:
        self.project_id = project_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict[str, Any]] = []
        self._start_time = time.time()

    def log_user(self, content: str, stage: str = "") -> dict[str, Any]:
        entry = {
            "role": "user",
            "content": content,
            "metadata": {"stage": stage, "ts": _ts()},
        }
        self.entries.append(entry)
        return entry

    def log_thought(self, content: str, tool: str = "", stage: str = "") -> dict[str, Any]:
        entry = {
            "role": "thought",
            "content": content,
            "metadata": {"tool": tool, "stage": stage, "ts": _ts()},
        }
        self.entries.append(entry)
        return entry

    def log_call(
        self,
        tool: str,
        tool_input: dict[str, Any],
        stage: str = "",
    ) -> dict[str, Any]:
        entry = {
            "role": "call",
            "tool": tool,
            "tool_input": tool_input,
            "metadata": {"stage": stage, "ts": _ts()},
        }
        self.entries.append(entry)
        return entry

    def log_respond(
        self,
        tool: str,
        tool_output: dict[str, Any],
        content: str = "",
        stage: str = "",
    ) -> dict[str, Any]:
        entry = {
            "role": "respond",
            "content": content or f"{tool} 执行完成",
            "tool": tool,
            "tool_output": tool_output,
            "metadata": {"stage": stage, "ts": _ts()},
        }
        self.entries.append(entry)
        return entry

    def log_human_feedback(
        self,
        node_name: str,
        question: str,
        options: list[dict[str, Any]],
        selected_option: int,
        custom_input: str = "",
        feedback_type: str = "self_reflect_demo",
    ) -> dict[str, Any]:
        """Log HITL intervention as a training-ready trajectory entry.

        feedback_type:
          - "self_reflect_demo": packaged as exemplary self-reflection
          - "negative_correction": packaged as negative feedback → correction
        """
        chosen = options[selected_option] if 0 <= selected_option < len(options) else {}
        if feedback_type == "self_reflect_demo":
            content = (
                f"[自反思] 在 {node_name} 节点，系统提出了 {len(options)} 个方案。\n"
                f"问题: {question}\n"
                f"经过审查，选择方案 {selected_option}: {chosen.get('label', '')}\n"
                f"理由: {chosen.get('description', custom_input)}"
            )
        else:
            content = (
                f"[修正反馈] 在 {node_name} 节点发现问题。\n"
                f"问题: {question}\n"
                f"原方案不佳，人工修正为: {custom_input or chosen.get('description', '')}"
            )

        entry = {
            "role": "human_feedback",
            "content": content,
            "metadata": {
                "node_name": node_name,
                "question": question,
                "options": options,
                "selected_option": selected_option,
                "custom_input": custom_input,
                "feedback_type": feedback_type,
                "ts": _ts(),
            },
        }
        self.entries.append(entry)
        return entry

    def to_sft_messages(self, compact: bool = False) -> list[dict[str, Any]]:
        """Convert trajectory into standard SFT message format."""
        system_msg = {
            "role": "system",
            "content": (
                "你是 AICut 视频生成导演智能体。你的任务是将长文章转化为完整的长视频，"
                "优先通过批量工具（Tool_BatchVisualAudio、Tool_BatchDecorateSpotlights、"
                "Tool_BatchTimelineCompose、Tool_BatchVLMJudge）一次处理镜头列表，"
                "工具内部并发执行并返回按 shot_id 汇总的 JSON；只有出现问题时才定点读取和修复对应镜头。"
                "最终使用 Tool_FinalRender 物化修复后的镜头，再由 Tool_StitchVideo 合成预览。"
                "每一步都要先思考（thought）再调用工具（call），并基于工具返回结果（respond）决定下一步。"
                "遇到质量问题时进行自反思修复循环。"
            ),
        }
        messages = [system_msg]
        for entry in self.entries:
            role = entry["role"]
            if role == "user":
                messages.append({"role": "user", "content": entry["content"]})
            elif role == "thought":
                messages.append({
                    "role": "assistant",
                    "content": f"<think>{entry['content']}</think>",
                })
            elif role == "call":
                tool_input = compact_payload(entry["tool_input"]) if compact else entry["tool_input"]
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": entry["tool"],
                            "arguments": json.dumps(tool_input, ensure_ascii=False),
                        },
                    }],
                })
            elif role == "respond":
                tool_output = compact_payload(entry["tool_output"]) if compact else entry["tool_output"]
                messages.append({
                    "role": "tool",
                    "name": entry.get("tool", ""),
                    "content": json.dumps(tool_output, ensure_ascii=False),
                })
            elif role == "human_feedback":
                # Rewrite HITL as assistant self-reflection for SFT
                messages.append({
                    "role": "assistant",
                    "content": entry["content"],
                })
        return messages

    def save(self, suffix: str = "") -> Path:
        """Save trajectory as JSONL for SFT training."""
        name = f"{self.project_id}{f'_{suffix}' if suffix else ''}.jsonl"
        path = self.output_dir / name
        elapsed = time.time() - self._start_time
        reward_scores = extract_reward_scores(self.entries)
        reward_summary = summarize_reward_scores(reward_scores)
        record = {
            "project_id": self.project_id,
            "timestamp": _ts(),
            "elapsed_seconds": round(elapsed, 1),
            "num_entries": len(self.entries),
            "raw_trajectory": self.entries,
            "sft_messages": self.to_sft_messages(),
            "reward_scores": reward_scores,
            "reward_summary": reward_summary,
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        # Also save compact version
        compact_path = self.output_dir / f"{self.project_id}{f'_{suffix}' if suffix else ''}_compact.jsonl"
        with compact_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({
                "messages": self.to_sft_messages(compact=True),
                "metadata": {
                    "project_id": self.project_id,
                    "num_tool_calls": sum(1 for e in self.entries if e["role"] == "call"),
                    "num_hitl": sum(1 for e in self.entries if e["role"] == "human_feedback"),
                    "elapsed_seconds": round(elapsed, 1),
                    "reward_summary": reward_summary,
                },
            }, ensure_ascii=False) + "\n")
        rewards_path = self.output_dir / f"{self.project_id}{f'_{suffix}' if suffix else ''}_rewards.jsonl"
        with rewards_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({
                "project_id": self.project_id,
                "timestamp": _ts(),
                "reward_summary": reward_summary,
                "reward_scores": reward_scores,
            }, ensure_ascii=False) + "\n")
        return path

    def get_entries(self) -> list[dict[str, Any]]:
        """Return entries as dicts for state serialization."""
        return list(self.entries)


def _ts() -> str:
    return datetime.now().isoformat()
