"""Enrich Gold_Template's structures.json so the pipeline sees semantically rich candidates.

背景
----
AICutPipeline 的 ``_load_template_registry``
(``Infer_Vis_new/server/infer_graph.py``) 读取
``<gold_template_root>/registry/structures.json``，并期望每个条目含有:

  - ``template_id`` 或 ``id``      → candidate.template_id
  - ``keywords`` 或 ``tags``       → candidate.keywords
  - ``category``                   → candidate.category
  - ``name``                       → candidate.name
  - ``slots`` 为 **字典**          → slot_schemas[tid]

但 Gold_Template 的 ``structures.json`` 与之偏离:

  - 用 ``display_name`` 而非 ``name``
  - 用 ``use_for`` (短语列表) 而非 ``keywords``/``tags``
  - ``slots`` 是 **字符串列表** (仅有槽位名，无 schema)

净效果: loader 返回 90 个 candidate，但其 ``keywords``/``name`` 全空;
90 个 ``slot_schemas[tid]`` 也是空 dict。下游 ``template_retrieve`` 因此
拿到 90 个光秃秃的 template_id，毫无语义抓手 → 返回空 template_id →
每个 shot 进入 3 次重试死亡循环。

修复
----
在 ``InferenceOrchestrator._load_template_registry`` 外面包一层 monkey-patch，
对其输出做事后富化:

  1. 直接重读 ``structures.json`` (沿用 original loader 解析出的
     ``gold_template_root`` 路径)。
  2. 对每个 template item，富化对应 candidate:
       - ``keywords`` ← ``use_for`` ∪ ``keywords`` ∪ ``tags``
       - ``name`` ← ``display_name`` ∪ ``name``
       - 附带 ``family``/``duration_range``/``spotlight_level``/``anim_roles``
         等额外语义上下文，提升 LLM 检索多样性
  3. 对每个 template item，从 ``slots`` 字符串列表合成
     ``slot_schemas[tid]``: 每个槽位变成
       ``{slot_name: {"name": slot_name, "type": "text",
                      "max_chars": 32, "description": ""}}``
     若 ``slots`` 已是 dict，则保留原状。

由此保证每条视频都能做出多样、有语义依据的模版选择。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_DEFAULT_SLOT_MAX_CHARS = 32


def _resolve_gold_root(config: dict[str, Any]) -> Path | None:
    """Resolve ``paths.gold_template_root`` from the runtime pipeline config."""
    if not isinstance(config, dict):
        return None
    paths = config.get("paths") or {}
    raw = paths.get("gold_template_root") if isinstance(paths, dict) else None
    if not raw:
        return None
    return Path(str(raw)).expanduser().resolve()


def _build_items_by_id(gold_root: Path) -> dict[str, dict]:
    """Re-read structures.json and index items by template id."""
    structures_path = gold_root / "registry" / "structures.json"
    if not structures_path.is_file():
        return {}
    try:
        data = json.loads(structures_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("template_enricher: failed to read structures.json: %s", exc)
        return {}
    items = (
        data
        if isinstance(data, list)
        else data.get("structures", data.get("templates", []))
    )
    indexed: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        tid = str(item.get("template_id", item.get("id", "")))
        if tid:
            indexed[tid] = item
    return indexed


def _enrich_candidates(
    candidates: list[dict[str, Any]],
    items_by_id: dict[str, dict],
) -> list[dict[str, Any]]:
    """Fold use_for/display_name/family/etc. back into each candidate."""
    for cand in candidates:
        tid = str(cand.get("template_id") or cand.get("id") or "")
        item = items_by_id.get(tid, {})

        # keywords ← use_for ∪ keywords ∪ tags (deduped, order-preserving)
        kw: list[str] = []
        seen: set[str] = set()
        for k in ("use_for", "keywords", "tags"):
            v = item.get(k)
            if isinstance(v, list):
                for x in v:
                    sx = str(x)
                    if sx and sx not in seen:
                        seen.add(sx)
                        kw.append(sx)
            elif isinstance(v, str) and v and v not in seen:
                seen.add(v)
                kw.append(v)
        cand["keywords"] = kw

        # name ← display_name ∪ name
        nm = item.get("display_name") or item.get("name") or ""
        cand["name"] = str(nm)

        # Carry extra semantic context for richer LLM retrieval.
        for fld in (
            "family",
            "category",
            "duration_range",
            "spotlight_level",
            "anim_roles",
            "use_for",
            "template_code",
            "owner_prompt",
        ):
            if fld in item:
                cand[fld] = item[fld]
    return candidates


def _synthesize_slot_schemas(
    slot_schemas: dict[str, dict[str, Any]],
    items_by_id: dict[str, dict],
) -> dict[str, dict[str, Any]]:
    """Convert string-list slots into proper slot_schema dicts."""
    for tid, existing in list(slot_schemas.items()):
        item = items_by_id.get(tid, {})
        slots_raw = item.get("slots", item.get("slot_schema", {}))

        if isinstance(slots_raw, list):
            synth: dict[str, dict[str, Any]] = {}
            for s in slots_raw:
                if isinstance(s, str) and s:
                    synth[s] = {
                        "name": s,
                        "type": "text",
                        "max_chars": _DEFAULT_SLOT_MAX_CHARS,
                        "description": "",
                    }
                elif isinstance(s, dict):
                    sn = str(s.get("name", s.get("id", "")))
                    if sn:
                        synth[sn] = s
            if synth:
                slot_schemas[tid] = synth
            # else: leave the existing (possibly empty) entry untouched
        elif isinstance(slots_raw, dict) and slots_raw:
            slot_schemas[tid] = slots_raw
    return slot_schemas


def enrich_registry(
    registry: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Post-process ``_load_template_registry`` output to compensate for format drift.

    Args:
        registry: Baseline output from the original loader
            (``{"candidates": [...], "slot_schemas": {...}}``).
        config: Runtime pipeline config (must contain
            ``paths.gold_template_root``).

    Returns:
        Enriched registry with non-empty ``keywords``, ``name``, and
        ``slot_schemas`` per candidate.
    """
    gold_root = _resolve_gold_root(config)
    if not gold_root:
        logger.debug(
            "template_enricher: no gold_template_root in config; skipping enrichment"
        )
        return registry

    items_by_id = _build_items_by_id(gold_root)
    if not items_by_id:
        logger.debug(
            "template_enricher: structures.json yielded no items; skipping enrichment"
        )
        return registry

    candidates = registry.setdefault("candidates", [])
    registry["candidates"] = _enrich_candidates(candidates, items_by_id)

    slot_schemas = registry.setdefault("slot_schemas", {})
    registry["slot_schemas"] = _synthesize_slot_schemas(slot_schemas, items_by_id)

    n_cand = len(registry["candidates"])
    n_slots = sum(1 for v in registry["slot_schemas"].values() if v)
    logger.info(
        "template_enricher: enriched %d candidates (%d with non-empty slot_schema) from %s",
        n_cand,
        n_slots,
        gold_root,
    )
    return registry


def install_template_enricher(orchestrator_cls: type) -> None:
    """Monkey-patch ``InferenceOrchestrator._load_template_registry`` to enrich its output.

    Idempotent: safe to call multiple times — subsequent calls are no-ops.

    Args:
        orchestrator_cls: The ``InferenceOrchestrator`` class (imported from
            ``server.infer_graph``).
    """
    if getattr(orchestrator_cls, "_template_enricher_installed", False):
        return

    # Grab the original staticmethod (unwrap descriptor for Py<3.10 compat).
    original = orchestrator_cls._load_template_registry
    if hasattr(original, "__func__"):
        original = original.__func__

    def patched(config: dict[str, Any]) -> dict[str, Any]:
        registry = original(config) if callable(original) else {}
        try:
            return enrich_registry(registry, config or {})
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "template_enricher: enrichment failed (using baseline): %s", exc
            )
            return registry

    # Reassign as staticmethod so ``self._load_template_registry(...)`` still works.
    orchestrator_cls._load_template_registry = staticmethod(patched)
    orchestrator_cls._template_enricher_installed = True
    logger.info(
        "template_enricher: installed on %s", orchestrator_cls.__name__
    )
