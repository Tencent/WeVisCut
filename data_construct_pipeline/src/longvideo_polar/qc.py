"""Measure the corpus, then publish it.

Two claims about this dataset need evidence rather than assertion: that spotlight
templates are used across the whole registry instead of a favoured handful, and
that slot text is written from the shot's visual query and its segment summary
instead of copied out of the narration. Both are measured here from the recorded
trajectories — not from the policy's intentions — so the numbers describe what the
annotator actually produced.

:func:`export_dataset` then assembles the publication layout: per-role SFT shards,
the reward rows, the QC report and a dataset card carrying the selection filter,
the model and machine provenance, and the known limitations.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from longvideo_polar.decoration_policy import VERBATIM_SPAN_LIMIT, longest_common_span

logger = logging.getLogger(__name__)

# Slot values at or below this length are exempt from the copy check: a 4-character
# label that happens to appear in the narration is the correct answer, not a lift.
MIN_SLOT_LEN_FOR_COPY_CHECK = 6

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


# ---------------------------------------------------------------------------
# Reading recorded samples
# ---------------------------------------------------------------------------


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield SFT samples from a role shard.

    Two shapes reach this function. Aggregated shards are true JSON Lines, one
    sample per line. Per-episode role files written by
    :class:`~longvideo_polar.recorder.TrajectoryRecorder` keep the ``.jsonl``
    suffix but are a single pretty-printed object whose ``samples`` array holds
    the rows — reading those line by line silently yields nothing, which is
    indistinguishable from an episode that never chose a template.
    """
    file_path = Path(str(path))
    if not file_path.is_file():
        return

    text = file_path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{") and "\n " in text[:400]:
        try:
            document = json.loads(text)
        except json.JSONDecodeError:
            document = None
        if isinstance(document, dict):
            for sample in document.get("samples") or []:
                if isinstance(sample, dict):
                    yield sample
            return

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            yield row


def _messages_of(row: dict[str, Any]) -> tuple[str, str]:
    """Return ``(user_content, assistant_content)`` from an SFT row."""
    messages = row.get("messages")
    if not isinstance(messages, list):
        return "", ""
    user = next(
        (str(m.get("content") or "") for m in reversed(messages)
         if isinstance(m, dict) and m.get("role") == "user"),
        "",
    )
    assistant = next(
        (str(m.get("content") or "") for m in reversed(messages)
         if isinstance(m, dict) and m.get("role") == "assistant"),
        "",
    )
    return user, assistant


def _parse_json_blob(text: str) -> dict[str, Any]:
    """Best-effort parse of a model turn that should contain one JSON object."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", str(text or "")).strip()
    # Reasoning models are usually primed with `<think>` by the chat template, so
    # the recorded completion starts mid-thought and only the closing tag is
    # present. Without this the scan below locks onto the first brace inside the
    # reasoning — often a literal `{}` the model is describing — and every sample
    # reads as a parse failure.
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[1].strip()
    cleaned = re.sub(r"```(?:json)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start < 0:
        return {}
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(cleaned[start:index + 1])
                except json.JSONDecodeError:
                    return {}
                return payload if isinstance(payload, dict) else {}
    return {}


# ---------------------------------------------------------------------------
# Template coverage
# ---------------------------------------------------------------------------


def analyze_templates(
    generator_shard: str | Path, *, registry_size: int = 90
) -> dict[str, Any]:
    """Distribution of realized template choices across the registry.

    Args:
        generator_shard: ``generator_sft.jsonl`` from the aggregate directory.
        registry_size: Total templates available, used to normalize entropy and to
            count the ones never chosen.

    Returns:
        Coverage metrics plus the full histogram.
    """
    chosen: Counter[str] = Counter()
    per_episode: dict[str, list[str]] = {}

    for row in iter_jsonl(generator_shard):
        metadata = row.get("metadata") or {}
        if metadata.get("stage_task") != "template_retrieve":
            continue
        _, assistant = _messages_of(row)
        template_id = str(_parse_json_blob(assistant).get("template_id") or "")
        if not template_id:
            continue
        chosen[template_id] += 1
        episode = str(row.get("episode_id") or metadata.get("episode_id") or "")
        per_episode.setdefault(episode, []).append(template_id)

    total = sum(chosen.values())
    if not total:
        return {"total_choices": 0, "distinct_templates": 0, "note": "no template_retrieve samples"}

    probabilities = [count / total for count in chosen.values()]
    entropy = -sum(p * math.log(p, 2) for p in probabilities)
    max_entropy = math.log(max(2, registry_size), 2)

    # Gini is computed over the whole registry, with unused templates counted as
    # zeros: concentration among the templates that were used says nothing about
    # the ones that never were.
    padded = sorted(list(chosen.values()) + [0] * max(0, registry_size - len(chosen)))
    n = len(padded)
    gini = (
        sum((2 * i - n + 1) * v for i, v in enumerate(padded)) / (n * sum(padded))
        if sum(padded) else 0.0
    )

    distinctness = [
        len(set(picks)) / len(picks) for picks in per_episode.values() if picks
    ]

    return {
        "total_choices": total,
        "registry_size": registry_size,
        "distinct_templates": len(chosen),
        "coverage_ratio": round(len(chosen) / registry_size, 4),
        "never_used": max(0, registry_size - len(chosen)),
        "normalized_entropy": round(entropy / max_entropy, 4),
        "gini": round(gini, 4),
        "top5_share": round(sum(c for _, c in chosen.most_common(5)) / total, 4),
        "mean_intra_episode_distinctness": (
            round(sum(distinctness) / len(distinctness), 4) if distinctness else 0.0
        ),
        "episodes_measured": len(per_episode),
        "histogram": dict(chosen.most_common()),
    }


# ---------------------------------------------------------------------------
# Slot quality
# ---------------------------------------------------------------------------


def analyze_slots(generator_shard: str | Path) -> dict[str, Any]:
    """How much filled slot text was written versus lifted.

    The headline number is ``verbatim_rate``: the share of slots sharing a run of
    at least :data:`~longvideo_polar.decoration_policy.VERBATIM_SPAN_LIMIT`
    characters with the shot's narration. Also reports duplicate slot values and
    ``max_chars`` overruns, both of which the fill contract forbids.
    """
    total_slots = 0
    verbatim_slots = 0
    over_limit = 0
    empty_slots = 0
    duplicate_samples = 0
    samples = 0
    parse_failures = 0
    span_histogram: Counter[int] = Counter()
    worst: list[dict[str, Any]] = []

    for row in iter_jsonl(generator_shard):
        metadata = row.get("metadata") or {}
        if metadata.get("stage_task") != "template_fill":
            continue
        samples += 1
        user, assistant = _messages_of(row)
        payload = _parse_json_blob(assistant)
        slots = payload.get("slots")
        if not isinstance(slots, dict) or not slots:
            parse_failures += 1
            continue

        request = _parse_json_blob(user)
        narration = _narration_of(request)
        limits = _slot_limits(request)

        values: list[str] = []
        for name, value in slots.items():
            if not isinstance(value, str):
                continue
            text = value.strip()
            total_slots += 1
            if not text:
                empty_slots += 1
                continue
            values.append(text)

            limit = limits.get(str(name), 0)
            if limit and len(text) > limit:
                over_limit += 1

            if narration and len(text) >= MIN_SLOT_LEN_FOR_COPY_CHECK:
                span = longest_common_span(text, narration)
                span_histogram[min(span, 20)] += 1
                if span >= VERBATIM_SPAN_LIMIT:
                    verbatim_slots += 1
                    if len(worst) < 25:
                        worst.append({
                            "episode_id": row.get("episode_id", ""),
                            "slot": str(name),
                            "value": text[:60],
                            "shared_span": span,
                        })

        if len(values) != len(set(values)) and values:
            duplicate_samples += 1

    return {
        "fill_samples": samples,
        "parse_failures": parse_failures,
        "total_slots": total_slots,
        "empty_slots": empty_slots,
        "verbatim_slots": verbatim_slots,
        "verbatim_rate": round(verbatim_slots / total_slots, 4) if total_slots else 0.0,
        "verbatim_threshold_chars": VERBATIM_SPAN_LIMIT,
        "max_chars_violations": over_limit,
        "max_chars_violation_rate": round(over_limit / total_slots, 4) if total_slots else 0.0,
        "samples_with_duplicate_slots": duplicate_samples,
        "duplicate_rate": round(duplicate_samples / samples, 4) if samples else 0.0,
        "shared_span_histogram": {str(k): v for k, v in sorted(span_histogram.items())},
        "worst_examples": worst,
    }


def _narration_of(request: dict[str, Any]) -> str:
    memory = request.get("memory") if isinstance(request.get("memory"), dict) else {}
    shot = memory.get("shot") if isinstance(memory.get("shot"), dict) else {}
    segment = memory.get("segment") if isinstance(memory.get("segment"), dict) else {}
    return " ".join(
        str(part) for part in (
            shot.get("vocal_script", ""), segment.get("narration_excerpt", "")
        ) if part
    )


def _slot_limits(request: dict[str, Any]) -> dict[str, int]:
    template = request.get("template") if isinstance(request.get("template"), dict) else {}
    slots = template.get("slots") if isinstance(template.get("slots"), dict) else {}
    limits: dict[str, int] = {}
    for name, spec in slots.items():
        if isinstance(spec, dict):
            try:
                limit = int(spec.get("max_chars") or 0)
            except (TypeError, ValueError):
                limit = 0
            if limit > 0:
                limits[str(name)] = limit
    return limits


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def aggregate_shards(episodes_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Concatenate per-episode trajectories into per-role shards.

    Every row is tagged with its ``episode_id`` so a training run can be traced
    back to the source video, and so shards from separate batches stay separable
    after concatenation.
    """
    episodes_path = Path(str(episodes_dir))
    target = Path(str(out_dir))
    target.mkdir(parents=True, exist_ok=True)

    manifests = sorted(episodes_path.glob("*/episode.json"))
    index: list[dict[str, Any]] = []
    counts = {"orchestrator": 0, "generator": 0, "critic": 0, "interactions": 0, "rewards": 0}

    handles = {
        "orchestrator": (target / "orchestrator_sft.jsonl").open("w", encoding="utf-8"),
        "generator": (target / "generator_sft.jsonl").open("w", encoding="utf-8"),
        "critic": (target / "critic_sft.jsonl").open("w", encoding="utf-8"),
        "interactions": (target / "interactions.jsonl").open("w", encoding="utf-8"),
        "rewards": (target / "rewards.jsonl").open("w", encoding="utf-8"),
    }
    try:
        for manifest_path in manifests:
            manifest = _read_json(manifest_path)
            if not manifest:
                continue
            episode_id = str(manifest.get("episode_id") or manifest_path.parent.name)
            traj = manifest_path.parent / "trajectories"

            orchestrator = _read_json(traj / "orchestrator.jsonl")
            if orchestrator:
                roles_info = manifest.get("roles") or {}
                handles["orchestrator"].write(json.dumps({
                    "episode_id": episode_id,
                    "messages": orchestrator.get("sft_messages", []),
                    "metadata": {
                        "episode_id": episode_id,
                        "num_tool_calls": orchestrator.get("num_tool_calls", 0),
                        "status": manifest.get("status"),
                        "annotation_profile": (manifest.get("metadata") or {}).get(
                            "annotation_profile", ""
                        ),
                        "providers": {
                            role: (info or {}).get("providers_used") or []
                            for role, info in roles_info.items()
                        },
                    },
                }, ensure_ascii=False) + "\n")
                counts["orchestrator"] += 1
                for reward_row in orchestrator.get("reward_scores") or []:
                    if isinstance(reward_row, dict):
                        handles["rewards"].write(json.dumps(
                            {"episode_id": episode_id, **reward_row}, ensure_ascii=False
                        ) + "\n")
                        counts["rewards"] += 1

            for role in ("generator", "critic"):
                payload = _read_json(traj / f"{role}.jsonl")
                for sample in (payload.get("samples") or []) if payload else []:
                    if not isinstance(sample, dict):
                        continue
                    metadata = sample.get("metadata") or {}
                    handles[role].write(json.dumps({
                        "episode_id": episode_id,
                        "messages": sample.get("messages", []),
                        "metadata": {**metadata, "episode_id": episode_id},
                    }, ensure_ascii=False) + "\n")
                    counts[role] += 1

            for row in iter_jsonl(traj / "interactions.jsonl"):
                handles["interactions"].write(json.dumps(
                    {"episode_id": episode_id, **row}, ensure_ascii=False
                ) + "\n")
                counts["interactions"] += 1

            index.append({
                "episode_id": episode_id,
                "status": manifest.get("status"),
                "orchestrator_steps": manifest.get("orchestrator_steps"),
                "roles": manifest.get("roles"),
                "reward_summary": manifest.get("reward_summary"),
                "final_video_exists": manifest.get("final_video_exists"),
                "elapsed_seconds": manifest.get("elapsed_seconds"),
                "errors": manifest.get("errors"),
            })
    finally:
        for handle in handles.values():
            handle.close()

    (target / "index.json").write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "episode_count": len(index),
                "samples": counts,
                "episodes": index,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    logger.info("aggregated %d episodes → %s", len(index), target)
    return {"episode_count": len(index), "samples": counts, "aggregate_dir": str(target)}


def export_dataset(
    output_root: str | Path,
    *,
    registry_size: int = 90,
    extra_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the publication layout under ``output_root``.

    Writes ``aggregate/`` (per-role shards), ``qc/`` (coverage and slot metrics)
    and the dataset card. Safe to re-run: everything is regenerated from the
    per-episode trajectories, which are the source of truth.
    """
    root = Path(str(output_root)).expanduser()
    episodes_dir = root / "episodes"
    aggregate_dir = root / "aggregate"
    qc_dir = root / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    aggregate = aggregate_shards(episodes_dir, aggregate_dir)
    generator_shard = aggregate_dir / "generator_sft.jsonl"

    templates = analyze_templates(generator_shard, registry_size=registry_size)
    slots = analyze_slots(generator_shard)

    (qc_dir / "template_coverage.json").write_text(
        json.dumps(templates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (qc_dir / "slot_quality.json").write_text(
        json.dumps(slots, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    worklist_header = _worklist_header(root / "worklist.jsonl")
    status = _read_json(root / "_state" / "campaign_status.json")

    card = {
        "name": root.name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "selection": worklist_header.get("selection", {}),
        "naming_scheme": worklist_header.get("naming_scheme", ""),
        "rejected_at_selection": worklist_header.get("rejected", {}),
        "run_settings": status.get("settings", {}),
        "run_summary": status.get("summary", {}),
        "machines": [
            {
                "host": endpoint.get("host"),
                "model": endpoint.get("model"),
                "completed": endpoint.get("completed"),
                "failed": endpoint.get("failed"),
            }
            for endpoint in (status.get("pool") or {}).get("endpoints", [])
        ],
        "aggregate": aggregate,
        "quality": {"template_coverage": templates, "slot_quality": slots},
        "provenance": dict(extra_provenance or {}),
    }
    (root / "dataset_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (qc_dir / "qc_report.md").write_text(_render_report(card), encoding="utf-8")
    (root / "README.md").write_text(_render_readme(card), encoding="utf-8")
    logger.info("exported dataset card + QC → %s", root)
    return card


def _worklist_header(path: Path) -> dict[str, Any]:
    for row in iter_jsonl(path):
        if row.get("_type") == "worklist_header":
            return row
        break
    return {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _render_report(card: dict[str, Any]) -> str:
    templates = card["quality"]["template_coverage"]
    slots = card["quality"]["slot_quality"]
    samples = card["aggregate"]["samples"]
    histogram = templates.get("histogram") or {}
    rows = "\n".join(
        f"| {tid} | {count} | {count / max(1, templates.get('total_choices', 1)) * 100:.2f}% |"
        for tid, count in list(histogram.items())[:20]
    )
    return f"""# QC 报告 — {card['name']}

生成时间：{card['generated_at']}

## 1. 规模

| 项 | 值 |
| --- | --- |
| episode 数 | {card['aggregate']['episode_count']} |
| 主流程样本（多轮） | {samples['orchestrator']} |
| 内容生成器样本（单轮） | {samples['generator']} |
| 反思修复样本（单轮） | {samples['critic']} |
| 跨角色交接边 | {samples['interactions']} |
| reward 行 | {samples['rewards']} |

## 2. 模版泛化性

衡量的是**实际落盘轨迹里的模版选择分布**，不是策略的意图。

| 指标 | 值 | 说明 |
| --- | --- | --- |
| 覆盖模版数 | {templates.get('distinct_templates', 0)} / {templates.get('registry_size', 0)} | 越接近满值越好 |
| 覆盖率 | {templates.get('coverage_ratio', 0) * 100:.1f}% | |
| 从未被选中 | {templates.get('never_used', 0)} | 目标 0 |
| 归一化熵 | {templates.get('normalized_entropy', 0):.3f} | 1.0 = 完全均匀 |
| Gini | {templates.get('gini', 0):.3f} | 0 = 均匀，1 = 极度集中 |
| Top-5 占比 | {templates.get('top5_share', 0) * 100:.1f}% | 越低越分散 |
| 单视频内模版相异率 | {templates.get('mean_intra_episode_distinctness', 0) * 100:.1f}% | 同一视频不同镜头用不同模版的比例 |
| 统计样本 | {templates.get('total_choices', 0)} 次检索 / {templates.get('episodes_measured', 0)} 条视频 | |

### 使用频次（Top 20）

| template_id | 次数 | 占比 |
| --- | --- | --- |
{rows}

## 3. 槽位文案质量

`verbatim_rate` 是核心指标：槽位文案与该镜头口播稿共享连续
{slots.get('verbatim_threshold_chars', 8)} 字及以上的比例，衡量「直接复制」而非「联想改写」。

| 指标 | 值 |
| --- | --- |
| template_fill 样本 | {slots.get('fill_samples', 0)} |
| 槽位总数 | {slots.get('total_slots', 0)} |
| 照抄槽位数 | {slots.get('verbatim_slots', 0)} |
| **照抄率** | **{slots.get('verbatim_rate', 0) * 100:.2f}%** |
| 超 max_chars 槽位 | {slots.get('max_chars_violations', 0)}（{slots.get('max_chars_violation_rate', 0) * 100:.2f}%） |
| 含重复槽位值的样本 | {slots.get('samples_with_duplicate_slots', 0)}（{slots.get('duplicate_rate', 0) * 100:.2f}%） |
| 空槽位 | {slots.get('empty_slots', 0)} |
| 解析失败 | {slots.get('parse_failures', 0)} |

## 4. 机器

| 机器 | 模型 | 完成 | 失败 |
| --- | --- | --- | --- |
""" + "\n".join(
        f"| {m.get('host')} | {m.get('model')} | {m.get('completed')} | {m.get('failed')} |"
        for m in card.get("machines", [])
    ) + "\n"


def _render_readme(card: dict[str, Any]) -> str:
    selection = card.get("selection") or {}
    summary = card.get("run_summary") or {}
    settings = card.get("run_settings") or {}
    templates = card["quality"]["template_coverage"]
    slots = card["quality"]["slot_quality"]
    samples = card["aggregate"]["samples"]
    rejected = "\n".join(
        f"| {reason} | {count} |" for reason, count in (card.get("rejected_at_selection") or {}).items()
    )
    return f"""# {card['name']}

bilibili 长视频口播稿的长程 Agent 轨迹标注集，按主流程 / 内容生成器 / 反思修复
三个角色切分，可直接用于 SFT 与 RL。

生成时间：{card['generated_at']}

## 目录结构

```
{card['name']}/
├── README.md                 本文件
├── dataset_card.json         机器可读的数据集卡片
├── worklist.jsonl            入选清单 + 筛选条件（可复现）
├── episodes/                 每条视频一个目录
│   └── <序号>__<BV号>__<asset_id前8位>/
│       ├── episode.json          运行清单：状态、分角色统计、reward、错误、耗时
│       ├── worker.log            该条 rollout 的完整日志
│       └── trajectories/
│           ├── orchestrator.jsonl    主流程：一条多轮 tool-calling episode
│           ├── generator.jsonl       内容生成器：单轮样本，按 stage_task 分桶
│           ├── critic.jsonl          反思修复：单轮样本，按 stage_task 分桶
│           ├── interactions.jsonl    跨角色交接边（把三份流串成一条因果轨迹）
│           ├── llm_calls.jsonl       全量原始模型 I/O，审计用
│           ├── sft/                  三个角色的 messages 格式
│           └── rl/reward_scores.jsonl
├── aggregate/                合并分片，每行带 episode_id
│   ├── orchestrator_sft.jsonl   {samples['orchestrator']} 行
│   ├── generator_sft.jsonl      {samples['generator']} 行
│   ├── critic_sft.jsonl         {samples['critic']} 行
│   ├── interactions.jsonl       {samples['interactions']} 行
│   ├── rewards.jsonl            {samples['rewards']} 行
│   └── index.json
├── qc/                       质量报告
│   ├── template_coverage.json
│   ├── slot_quality.json
│   └── qc_report.md
└── _state/                   运行状态、机器池、模版使用账本、会话配置
```

## 命名

`<序号>__<BV号>__<asset_id 前8位>`，例如 `000123__BV1Qx411V7yh__0001c616`。
序号用于排序与分页，BV 号是可公开引用的来源视频句柄，asset_id 前缀保证与原数据集
`videos/` `scripts/` 的连接唯一。完整 asset_id 见 `worklist.jsonl` 与
`episode.json`。

## 筛选口径

| 条件 | 值 |
| --- | --- |
| 语种 | {', '.join(selection.get('languages') or []) or '不限'} |
| 口播稿字数 | {selection.get('min_chars', 0)} – {selection.get('max_chars', 0)} |
| 条数上限 | {selection.get('limit') or '不限'} |
| 分层抽样 | {'是' if selection.get('stratify') else '否'} |
| 随机种子 | {selection.get('seed')} |

被排除的条目及原因：

| 原因 | 条数 |
| --- | --- |
{rejected}

## 运行设置

| 项 | 值 |
| --- | --- |
| 模式 | {settings.get('mode')} |
| 标注 profile | {settings.get('profile')} |
| 单条最大步数 | {settings.get('max_steps')} |
| 完成 / 失败 / 跳过 | {summary.get('done', 0)} / {summary.get('failed', 0)} / {summary.get('skipped', 0)} |

## 质量摘要

- 模版覆盖：**{templates.get('distinct_templates', 0)}/{templates.get('registry_size', 0)}**
  （归一化熵 {templates.get('normalized_entropy', 0):.3f}，Top-5 占比
  {templates.get('top5_share', 0) * 100:.1f}%，单视频内相异率
  {templates.get('mean_intra_episode_distinctness', 0) * 100:.1f}%）
- 槽位照抄率：**{slots.get('verbatim_rate', 0) * 100:.2f}%**
  （与口播稿共享连续 {slots.get('verbatim_threshold_chars', 8)} 字及以上）
- 超长槽位：{slots.get('max_chars_violation_rate', 0) * 100:.2f}%

完整指标见 [`qc/qc_report.md`](qc/qc_report.md)。

## 已知限制

- 模式为 `text_only` 时媒体工具关闭，不产出成片，`final_video_exists` 恒为 false；
  依赖成片硬门槛的 RL reward 在这批数据上不可用，需要另跑 `full_media` 子集。
- 失败轨迹一并保留（`recording.save_on_failure`）。反思修复训练需要它们，
  但按 `status` 过滤前不要直接当正样本用。
- 口播稿来自字幕，含 ASR / 字幕组噪声；超过 {selection.get('max_chars', 0)} 字与
  少于 {selection.get('min_chars', 0)} 字的条目已在选取阶段排除。
"""
