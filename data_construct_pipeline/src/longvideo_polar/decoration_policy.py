"""Coverage-driven template selection and association-first slot filling.

Why this module exists
----------------------
``Infer_Vis_new`` decorates spotlight shots in two LLM steps — ``template_retrieve``
picks one of the Gold_Template structures, then ``template_fill`` writes its slots.
Both steps are built by static methods on ``InferenceOrchestrator``, and both have
defects that show up immediately at annotation scale:

1. **Candidates arrive unlabeled.** ``_load_template_registry`` reads ``keywords``
   and ``name``, but ``structures.json`` stores ``use_for`` and ``display_name``.
   All 90 candidates therefore reach the model as bare ids with empty semantics.

2. **Nothing spreads the choice.** All 90 candidates are offered to every shot with
   no ranking, no rotation and no memory of what the episode already used. Measured
   on a real rollout: 32 retrieve calls collapsed onto **7 distinct templates**, with
   ``problem_hook`` alone taking 11 of 32. A corpus annotated that way teaches the
   model that six templates exist.

3. **Slot schemas are degraded.** The list-of-names form gets expanded by name
   heuristics even though ``generated/structures/<id>/slot_schema.json`` holds the
   real labels, defaults and per-slot ``max_chars``.

4. **Fill invites copying.** The instruction only forbids verbatim copy for
   ``headline``/``eyebrow``, so the remaining slots tend to become sentence
   fragments lifted out of ``vocal_script``.

What it does
------------
Three static methods are wrapped, so AICutPipeline itself stays untouched:

``_load_template_registry``
    Rebuilt from the registry *and* the per-structure slot schemas, emitting the
    exact ``candidate_card`` / ``normalize_slot_schema`` shapes that
    ``preprocess_ori_data/src/gold_template_registry.py`` uses at SFT build time.
    Inference payloads then match training payloads field for field.

``_build_skill_payload_template_retrieve``
    Replaces the flat 90-candidate list with a per-shot pool (default 14) mixing
    semantic relevance, a global under-use bonus and family/category stratification,
    rotated by a stable hash of the shot so no template owns the first slot. Adds
    ``already_used_templates`` and an explicit diversity policy.

``_build_skill_payload_template_fill``
    Records the realized choice into the shared ledger, surfaces the association
    sources (``visual_query`` + segment summary/key points) as their own payload
    block, and swaps in an instruction that treats slot filling as re-expression
    rather than extraction.

Coverage is tracked in SQLite because every rollout is a separate subprocess; the
ledger is what lets template 87 of 90 get its turn on episode 6,000.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

POLICY_VERSION = "coverage_v1"

# Pool size per shot. Training ranked to top-12; 14 leaves room for the coverage and
# stratification slices without drifting far from the distribution the model saw.
DEFAULT_POOL_SIZE = 14

# How the pool is split. Semantic picks keep choices defensible, coverage picks pull
# the long tail in, stratification guarantees every family stays representable.
DEFAULT_SEMANTIC_SLICE = 6
DEFAULT_COVERAGE_SLICE = 5

# Longest run of characters a slot may share with the narration before it counts as
# copied rather than written. Also stated to the model in the fill instruction.
VERBATIM_SPAN_LIMIT = 8

# Category hints, mirroring gold_template_registry._KEYWORD_CATEGORY so retrieval
# ranks the same way it did when the training rows were built.
_KEYWORD_CATEGORY: tuple[tuple[tuple[str, ...], str], ...] = (
    (("因果", "导致", "所以", "因为", "结果"), "C3"),
    (("对比", "相比", "差异", "区别", "vs"), "C4"),
    (("流程", "步骤", "阶段", "第一步", "第二步", "顺序"), "C3"),
    (("数据", "数字", "百分", "%", "增长", "占比", "排行"), "C4"),
    (("定义", "概念", "是指", "术语", "机制", "原理"), "C2"),
    (("总结", "回顾", "综上", "小结", "启示"), "C6"),
    (("开场", "钩子", "问题", "为什么", "你知道", "想象"), "C1"),
    (("引述", "转场", "接下来", "章节", "历史"), "C5"),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class GoldTemplateRegistry:
    """Read Gold_Template into the shapes the unified v5 payloads expect.

    Per-structure slot schemas are 90 separate files on a shared filesystem where a
    single ``stat`` costs ~2s, so reading them all up front adds minutes to *every*
    rollout subprocess. They are instead served from a one-file consolidated cache
    when present, and otherwise read lazily per template id.
    """

    #: Consolidated ``{template_id: slot_schema.json}`` written by :meth:`build_cache`.
    CACHE_FILENAME = "slot_schema_cache.json"

    def __init__(self, gold_root: str | Path, *, cache_path: str | Path | None = None) -> None:
        self.root = Path(str(gold_root)).expanduser().resolve()
        self.cache_path = Path(str(cache_path)).expanduser() if cache_path else None
        self.structures: dict[str, dict[str, Any]] = {}
        self.category_names: dict[str, str] = {}
        self._slot_cache: dict[str, dict[str, Any]] = {}
        self._cache_loaded = False
        self._load()
        self._load_cache()

    def _load(self) -> None:
        index_path = self.root / "registry" / "structures.json"
        if not index_path.is_file():
            logger.warning("decoration_policy: no structures.json at %s", index_path)
            return
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - a broken registry must not kill the run
            logger.warning("decoration_policy: unreadable structures.json: %s", exc)
            return

        items = data.get("structures") if isinstance(data, dict) else data
        for item in items or []:
            if not isinstance(item, dict):
                continue
            # structures.json keys on `id`; `template_id` is accepted for forward
            # compatibility with a renamed registry.
            sid = str(item.get("id") or item.get("template_id") or "").strip()
            if sid:
                self.structures[sid] = dict(item)

        for category in (data.get("categories") if isinstance(data, dict) else None) or []:
            if isinstance(category, dict) and category.get("id"):
                self.category_names[str(category["id"])] = str(category.get("name") or "")

    def __len__(self) -> int:
        return len(self.structures)

    @property
    def template_ids(self) -> list[str]:
        return sorted(self.structures)

    def _load_cache(self) -> None:
        """Warm the slot-schema cache from the consolidated file, if there is one."""
        if not self.cache_path or not self.cache_path.is_file():
            return
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - a stale cache must not be fatal
            logger.warning("decoration_policy: slot cache unreadable: %s", exc)
            return
        schemas = payload.get("schemas") if isinstance(payload, dict) else None
        if isinstance(schemas, dict):
            self._slot_cache.update(
                {str(sid): spec for sid, spec in schemas.items() if isinstance(spec, dict)}
            )
            self._cache_loaded = True
            logger.debug("decoration_policy: %d slot schemas from cache", len(self._slot_cache))

    def build_cache(self, cache_path: str | Path | None = None) -> Path:
        """Read every slot schema once and write the consolidated cache.

        Run this at batch preparation time so the thousands of rollout subprocesses
        that follow each pay one file read instead of ninety.
        """
        target = Path(str(cache_path or self.cache_path or (self.root / self.CACHE_FILENAME)))
        target.parent.mkdir(parents=True, exist_ok=True)
        schemas = {sid: self.slot_schema_file(sid) for sid in self.template_ids}
        target.write_text(
            json.dumps(
                {
                    "gold_template_root": str(self.root),
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "count": len(schemas),
                    "schemas": schemas,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        self.cache_path = target
        self._cache_loaded = True
        logger.info("decoration_policy: wrote slot schema cache (%d) → %s", len(schemas), target)
        return target

    def slot_schema_file(self, sid: str) -> dict[str, Any]:
        """Read ``generated/structures/<sid>/slot_schema.json``, cached per id."""
        if sid in self._slot_cache:
            return self._slot_cache[sid]
        path = self.root / "generated" / "structures" / sid / "slot_schema.json"
        payload: dict[str, Any] = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload = loaded
            except Exception as exc:  # noqa: BLE001
                logger.debug("decoration_policy: bad slot_schema for %s: %s", sid, exc)
        self._slot_cache[sid] = payload
        return payload

    def candidate_card(self, sid: str) -> dict[str, Any]:
        """The candidate shape used by ``build_unified_sft_v5``.

        Field-for-field identical to ``gold_template_registry.candidate_card`` in
        ``preprocess_ori_data``, so a recorded ``template_retrieve`` turn is drawn
        from the same input distribution the SFT corpus was built on.

        The pipeline's own loader instead reads ``item["name"]`` and
        ``item["keywords"]`` — names ``structures.json`` does not use, since it
        calls them ``display_name`` and ``use_for`` — so every candidate reached
        the model as a bare id under an empty label. Installing this registry
        replaces that loader outright; nothing downstream reads the old spellings.
        """
        entry = self.structures.get(sid) or {}
        return {
            "template_id": sid,
            "display_name": entry.get("display_name", ""),
            "family": entry.get("family", ""),
            "category": entry.get("category", ""),
            "category_name": self.category_names.get(str(entry.get("category") or ""), ""),
            "use_for": list(entry.get("use_for") or [])[:6],
            "spotlight_level": entry.get("spotlight_level", ""),
            "slot_names": list(entry.get("slots") or [])[:12],
            "duration_range": entry.get("duration_range", []),
        }

    def normalized_slot_schema(self, sid: str) -> dict[str, Any]:
        """The ``template`` payload shape used by ``build_unified_sft_v5``.

        Prefers the authored per-structure schema (real labels and ``max_chars``)
        and falls back to bare slot names from the index.
        """
        detailed = self.slot_schema_file(sid)
        entry = self.structures.get(sid) or {}
        slots_out: dict[str, Any] = {}
        detailed_slots = detailed.get("slots")
        if isinstance(detailed_slots, dict):
            for name, spec in detailed_slots.items():
                if isinstance(spec, dict):
                    slots_out[str(name)] = {
                        "type": spec.get("type", "string"),
                        "label": spec.get("label", ""),
                        "max_chars": spec.get("max_chars", 0),
                    }
                else:
                    slots_out[str(name)] = {"type": "string"}
        else:
            for name in entry.get("slots") or []:
                slots_out[str(name)] = {"type": "string"}
        return {
            "template_id": sid,
            "display_name": entry.get("display_name", ""),
            "family": entry.get("family", detailed.get("family", "")),
            "duration_default": detailed.get("duration_default"),
            "slots": slots_out,
            "data_slots": list(detailed.get("data_slots") or []),
        }


# ---------------------------------------------------------------------------
# Usage ledger
# ---------------------------------------------------------------------------


class TemplateUsageLedger:
    """Cross-process counts of how often each template was offered and chosen.

    Batch annotation runs one subprocess per episode, so in-memory counters cannot
    see each other. SQLite in WAL mode gives concurrent readers, a serialized
    writer, and durability across the days a 9k-episode batch takes.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(str(db_path)).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            # A long timeout is cheaper than losing a count: writes are tiny and
            # rare (one per decorated shot) but many processes contend.
            conn = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS template_usage (
                template_id TEXT PRIMARY KEY,
                offered     INTEGER NOT NULL DEFAULT 0,
                chosen      INTEGER NOT NULL DEFAULT 0,
                updated_at  REAL    NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS template_choice_log (
                episode_id  TEXT NOT NULL,
                shot_id     TEXT NOT NULL,
                template_id TEXT NOT NULL,
                created_at  REAL NOT NULL,
                PRIMARY KEY (episode_id, shot_id)
            )
            """
        )

    def _write(self, statement: str, rows: list[tuple[Any, ...]]) -> None:
        if not rows:
            return
        for attempt in range(5):
            try:
                conn = self._connect()
                conn.executemany(statement, rows)
                return
            except sqlite3.OperationalError as exc:  # locked / busy
                if attempt == 4:
                    logger.warning("decoration_policy: ledger write dropped: %s", exc)
                    return
                time.sleep(0.2 * (attempt + 1))

    def record_offered(self, template_ids: list[str]) -> None:
        now = time.time()
        self._write(
            """
            INSERT INTO template_usage (template_id, offered, chosen, updated_at)
            VALUES (?, 1, 0, ?)
            ON CONFLICT(template_id) DO UPDATE SET
                offered = offered + 1, updated_at = excluded.updated_at
            """,
            [(tid, now) for tid in template_ids if tid],
        )

    def record_choice(self, template_id: str, *, episode_id: str = "", shot_id: str = "") -> None:
        if not template_id:
            return
        now = time.time()
        self._write(
            """
            INSERT INTO template_usage (template_id, offered, chosen, updated_at)
            VALUES (?, 0, 1, ?)
            ON CONFLICT(template_id) DO UPDATE SET
                chosen = chosen + 1, updated_at = excluded.updated_at
            """,
            [(template_id, now)],
        )
        if episode_id and shot_id:
            self._write(
                """
                INSERT OR REPLACE INTO template_choice_log
                    (episode_id, shot_id, template_id, created_at)
                VALUES (?, ?, ?, ?)
                """,
                [(episode_id, shot_id, template_id, now)],
            )

    def chosen_counts(self) -> dict[str, int]:
        try:
            conn = self._connect()
            rows = conn.execute("SELECT template_id, chosen FROM template_usage").fetchall()
        except sqlite3.Error as exc:
            logger.debug("decoration_policy: ledger read failed: %s", exc)
            return {}
        return {str(tid): int(count) for tid, count in rows}

    def snapshot(self) -> dict[str, dict[str, int]]:
        try:
            conn = self._connect()
            rows = conn.execute(
                "SELECT template_id, offered, chosen FROM template_usage"
            ).fetchall()
        except sqlite3.Error:
            return {}
        return {
            str(tid): {"offered": int(offered), "chosen": int(chosen)}
            for tid, offered, chosen in rows
        }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _haystack(shot: dict[str, Any], segment: dict[str, Any]) -> str:
    """Concatenate the shot/segment text a template could plausibly match.

    ``visual_query`` is included here but not in the training-time equivalent: it
    is the most concrete description of what the shot actually shows, which is
    exactly the signal a diagram-vs-dataviz choice turns on.
    """
    parts = [
        str(shot.get(key, "") or "")
        for key in (
            "vocal_script", "chapter_title", "design_rationale", "emphasis_target",
            "intent", "structure_label", "planner_strategy", "visual_query",
            "spotlight_reason",
        )
    ]
    seg = segment or {}
    parts.append(str(seg.get("summary", "") or ""))
    parts.append(str(seg.get("title", "") or ""))
    parts.append(str(seg.get("structure_label", "") or ""))
    key_points = seg.get("key_points")
    if isinstance(key_points, list):
        parts.extend(str(point) for point in key_points)
    return " ".join(part for part in parts if part)


def _semantic_score(entry: dict[str, Any], haystack: str, hinted: set[str]) -> float:
    """Rank one template against the shot text.

    Weights follow ``gold_template_registry._score_structure`` so ranking stays
    comparable with the training rows; the bigram term is added because Chinese
    ``use_for`` phrases rarely appear verbatim in narration.
    """
    score = 0.0
    for token in entry.get("use_for") or []:
        token = str(token).strip()
        if not token:
            continue
        if token in haystack:
            score += 3.0
        elif len(token) >= 2 and _bigram_overlap(token, haystack) >= 0.5:
            score += 1.0
    name = str(entry.get("display_name") or "")
    if name and name in haystack:
        score += 1.5
    if str(entry.get("category") or "") in hinted:
        score += 2.0
    if str(entry.get("spotlight_level") or "") == "high":
        score += 0.3
    return score


def _bigram_overlap(needle: str, haystack: str) -> float:
    """Fraction of ``needle``'s character bigrams present in ``haystack``."""
    if len(needle) < 2:
        return 0.0
    bigrams = {needle[i:i + 2] for i in range(len(needle) - 1)}
    if not bigrams:
        return 0.0
    hit = sum(1 for bigram in bigrams if bigram in haystack)
    return hit / len(bigrams)


def _hinted_categories(haystack: str) -> set[str]:
    return {
        category
        for keywords, category in _KEYWORD_CATEGORY
        if any(keyword in haystack for keyword in keywords)
    }


def _stable_offset(*parts: str) -> int:
    """Deterministic rotation seed.

    Deliberately not ``hash()``: PYTHONHASHSEED randomization would make the same
    episode produce different pools on re-run, and this corpus has to be
    reproducible.
    """
    digest = hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def longest_common_span(text: str, other: str) -> int:
    """Longest contiguous run of characters shared by both strings.

    Used to measure how much of a filled slot was lifted from the narration.
    """
    if not text or not other:
        return 0
    previous = [0] * (len(other) + 1)
    best = 0
    for i in range(1, len(text) + 1):
        current = [0] * (len(other) + 1)
        char = text[i - 1]
        for j in range(1, len(other) + 1):
            if char == other[j - 1]:
                current[j] = previous[j - 1] + 1
                if current[j] > best:
                    best = current[j]
        previous = current
    return best


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class DecorationPolicy:
    """Builds per-shot candidate pools and the association-first fill contract."""

    def __init__(
        self,
        registry: GoldTemplateRegistry,
        ledger: TemplateUsageLedger | None = None,
        *,
        pool_size: int = DEFAULT_POOL_SIZE,
        semantic_slice: int = DEFAULT_SEMANTIC_SLICE,
        coverage_slice: int = DEFAULT_COVERAGE_SLICE,
        episode_id: str = "",
    ) -> None:
        self.registry = registry
        self.ledger = ledger
        self.pool_size = max(4, int(pool_size))
        self.semantic_slice = max(1, int(semantic_slice))
        self.coverage_slice = max(0, int(coverage_slice))
        self.episode_id = episode_id
        self._episode_used: list[str] = []
        self._lock = threading.Lock()

    # -- selection ------------------------------------------------------

    def candidate_pool(self, shot: dict[str, Any], segment: dict[str, Any]) -> list[dict[str, Any]]:
        """Choose which templates this shot gets to pick from."""
        structures = self.registry.structures
        if not structures:
            return []

        haystack = _haystack(shot, segment)
        hinted = _hinted_categories(haystack)
        scores = {
            sid: _semantic_score(entry, haystack, hinted)
            for sid, entry in structures.items()
        }
        global_counts = self.ledger.chosen_counts() if self.ledger else {}
        with self._lock:
            episode_used = list(self._episode_used)

        # Templates this episode already used go last. Dropping them outright would
        # break short videos where the pool is larger than what is left.
        used = set(episode_used)
        available = [sid for sid in structures if sid not in used]
        if len(available) < self.pool_size:
            available = list(structures)

        picked: list[str] = []

        def take(sid: str) -> None:
            if sid and sid not in picked and len(picked) < self.pool_size:
                picked.append(sid)

        # 1. Semantic relevance — keeps every pool defensible on its merits.
        by_score = sorted(available, key=lambda sid: (-scores.get(sid, 0.0), sid))
        for sid in by_score[: self.semantic_slice]:
            take(sid)

        # 2. Global under-use — this is what walks the long tail of the 90 over the
        #    course of the corpus. Ties break toward the more relevant template.
        by_usage = sorted(
            (sid for sid in available if sid not in picked),
            key=lambda sid: (global_counts.get(sid, 0), -scores.get(sid, 0.0), sid),
        )
        for sid in by_usage[: self.coverage_slice]:
            take(sid)

        # 3. Stratify — a pool missing a whole family cannot choose one.
        for family in ("narrative", "diagram", "dataviz"):
            if len(picked) >= self.pool_size:
                break
            if any(str(structures[sid].get("family")) == family for sid in picked):
                continue
            for sid in by_usage:
                if str(structures[sid].get("family")) == family:
                    take(sid)
                    break

        # 4. Top up from the remaining least-used.
        for sid in by_usage:
            if len(picked) >= self.pool_size:
                break
            take(sid)

        # 5. Rotate. Without this the highest-scoring template sits at position 0 for
        #    every shot, and position bias alone reproduces the collapse.
        if picked:
            offset = _stable_offset(self.episode_id, str(shot.get("shot_id", ""))) % len(picked)
            picked = picked[offset:] + picked[:offset]

        if self.ledger:
            self.ledger.record_offered(picked)
        return [self.registry.candidate_card(sid) for sid in picked]

    def record_choice(self, template_id: str, shot_id: str = "") -> None:
        tid = str(template_id or "").strip()
        if not tid:
            return
        with self._lock:
            if tid not in self._episode_used:
                self._episode_used.append(tid)
        if self.ledger:
            self.ledger.record_choice(tid, episode_id=self.episode_id, shot_id=str(shot_id or ""))

    @property
    def episode_used(self) -> list[str]:
        with self._lock:
            return list(self._episode_used)

    # -- payload fragments ----------------------------------------------

    def diversity_block(self) -> dict[str, Any]:
        """Tell the model, in-band, that spreading choices is part of the task."""
        return {
            "already_used_in_this_video": self.episode_used,
            "prefer_unused_template": True,
            "candidates_are_prefiltered": True,
            "note": (
                "candidates 已按语义相关度与全局覆盖度预筛，池内每个模版都可选；"
                "同一条视频内不同镜头应选不同模版，除非语义上确实只有一个合适。"
            ),
        }

    @staticmethod
    def association_block(shot: dict[str, Any], segment: dict[str, Any]) -> dict[str, Any]:
        """The material slot text must be derived from, hoisted for salience.

        These fields already exist elsewhere in the payload, but buried inside
        ``memory.shot`` / ``memory.segment`` the model treats them as background and
        falls back on copying ``vocal_script``.
        """
        seg = segment or {}
        key_points = seg.get("key_points")
        return {
            "visual_query": str(shot.get("visual_query", "") or ""),
            "segment_summary": str(seg.get("summary", "") or ""),
            "key_points": [str(point) for point in key_points][:6]
            if isinstance(key_points, list)
            else [],
            "chapter_title": str(shot.get("chapter_title", "") or ""),
            "design_rationale": str(shot.get("design_rationale", "") or ""),
        }

    @staticmethod
    def fill_instruction(shot_duration: float) -> str:
        """The association-first contract for ``template_fill``.

        Stated as a rewriting task with a worked example, because "不要照抄" alone
        measurably does not stop extraction — the model needs to see what the
        transformed output looks like.
        """
        max_duration = min(4.0, max(2.0, shot_duration * 0.6))
        return (
            "根据 template.slots 把全部槽位填满。这是【联想改写】任务，不是【摘抄】任务。\n"
            "\n"
            "一、联想，不复述（最重要）\n"
            "  每个槽位都要由 association.visual_query（这一镜头观众看得见的画面）\n"
            "  与 association.segment_summary / key_points（这一段要立住的观点）\n"
            "  重新组织成一句观众一眼能懂的新表达。\n"
            f"  硬性禁止：从 memory.shot.vocal_script 连续照抄 {VERBATIM_SPAN_LIMIT} 个及以上汉字。\n"
            "  硬性禁止：把口播原句整句搬进任何槽位。\n"
            "\n"
            "二、槽位各司其职，彼此不重复\n"
            "  eyebrow / *_label：类目或视角标签，短词组，不成句。\n"
            "  headline / *_title：一句话结论，是观点不是标题党。\n"
            "  *_text / *_desc：支撑说明，补充 headline 没说完的那一层。\n"
            "  *_value / 数字槽：具体数值与单位。\n"
            "  emoji 槽：恰好一个 emoji，不要文字。\n"
            "  严禁多个槽位填相同或近义的文字。\n"
            "\n"
            "三、可核查\n"
            "  数字、单位、专有名词必须能在 vocal_script 或 key_points 里找到依据，不许臆造。\n"
            "  口播稿不足 30 字时，从 segment.narration_excerpt 与 key_points 提炼。\n"
            "\n"
            "四、长度\n"
            "  严格遵守 template.slots 里每个槽位的 max_chars；宁可更短，不许超。\n"
            "\n"
            "五、示例\n"
            "  visual_query=「城市夜景航拍，写字楼灯光」\n"
            "  segment_summary=「加班文化推高一线城市写字楼空置率」\n"
            "  ✗ 照抄： headline=「加班文化推高一线城市写字楼空置率」\n"
            "  ✓ 联想： eyebrow=「职场观察」 headline=「灯还亮着，楼却空了」\n"
            "           explain_text=「空置率与加班时长同步走高」\n"
            "\n"
            f"六、时序：timing.decor_start_sec ∈ [0.2, 0.8]，"
            f"timing.decor_duration_sec ≤ {max_duration:.1f}s（本镜头时长 {shot_duration:.1f}s）。"
        )

    @staticmethod
    def fill_constraints() -> dict[str, Any]:
        return {
            "fill_all_slots": True,
            "respect_max_chars": True,
            "derive_from_visual_query_and_segment_summary": True,
            "no_verbatim_span_from_vocal_script": VERBATIM_SPAN_LIMIT,
            "no_duplicate_slot_values": True,
            "each_slot_plays_a_distinct_role": True,
            "numbers_must_be_grounded_in_source": True,
            "avoid_placeholder_or_pipeline_jargon": True,
        }


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


class _LazySlotSchemas(dict):
    """``{template_id: normalized_slot_schema}`` that materializes on lookup.

    The pipeline only ever asks for the one template a shot chose
    (``slot_schemas.get(chosen_id, {})``), so building all 90 up front would read 90
    files to use one. Both ``get`` and ``__getitem__`` are overridden because
    ``dict.get`` does not route through ``__missing__``.
    """

    def __init__(self, registry: "GoldTemplateRegistry") -> None:
        super().__init__()
        self._registry = registry

    def _materialize(self, key: Any) -> dict[str, Any] | None:
        sid = str(key)
        if sid not in self._registry.structures:
            return None
        schema = self._registry.normalized_slot_schema(sid)
        super().__setitem__(sid, schema)
        return schema

    def get(self, key: Any, default: Any = None) -> Any:  # type: ignore[override]
        if super().__contains__(key):
            return super().__getitem__(key)
        return self._materialize(key) or default

    def __getitem__(self, key: Any) -> Any:
        if super().__contains__(key):
            return super().__getitem__(key)
        schema = self._materialize(key)
        if schema is None:
            raise KeyError(key)
        return schema

    def __contains__(self, key: Any) -> bool:
        return super().__contains__(key) or str(key) in self._registry.structures

    def __len__(self) -> int:
        return len(self._registry.structures)

    def keys(self):  # type: ignore[override]
        return self._registry.structures.keys()


_ACTIVE: dict[str, Any] = {"policy": None}


def active_policy() -> DecorationPolicy | None:
    return _ACTIVE.get("policy")  # type: ignore[return-value]


def install_decoration_policy(
    orchestrator_cls: type,
    *,
    ledger_path: str | Path | None = None,
    slot_cache_path: str | Path | None = None,
    episode_id: str = "",
    pool_size: int = DEFAULT_POOL_SIZE,
    semantic_slice: int = DEFAULT_SEMANTIC_SLICE,
    coverage_slice: int = DEFAULT_COVERAGE_SLICE,
    enabled: bool = True,
) -> DecorationPolicy | None:
    """Wrap the three decoration static methods on ``InferenceOrchestrator``.

    Idempotent, and safe to call when the registry is missing: every wrapper falls
    back to the original implementation on any failure, so a bad registry degrades
    decoration quality instead of ending the rollout.

    Args:
        orchestrator_cls: ``server.infer_graph.InferenceOrchestrator``.
        ledger_path: SQLite file for cross-episode coverage counts. Coverage
            balancing is skipped when omitted.
        slot_cache_path: Consolidated slot-schema cache; avoids 90 shared-filesystem
            reads per subprocess.
        episode_id: Recorded alongside each realized choice.
        pool_size: Candidates offered per shot.
        semantic_slice: Of those, how many come from relevance ranking.
        coverage_slice: How many come from the global under-use ranking.
        enabled: When false, install nothing and return ``None``.

    Returns:
        The installed policy, or ``None`` when disabled or already installed.
    """
    if not enabled:
        return None
    if getattr(orchestrator_cls, "_decoration_policy_installed", False):
        return active_policy()

    original_registry = _unwrap(orchestrator_cls._load_template_registry)
    original_retrieve = _unwrap(orchestrator_cls._build_skill_payload_template_retrieve)
    original_fill = _unwrap(orchestrator_cls._build_skill_payload_template_fill)

    state: dict[str, Any] = {"registry": None, "policy": None}

    def ensure_policy(config: dict[str, Any]) -> DecorationPolicy | None:
        if state["policy"] is not None:
            return state["policy"]
        gold_root = _resolve_gold_root(config)
        if not gold_root:
            return None
        registry = GoldTemplateRegistry(gold_root, cache_path=slot_cache_path)
        if not len(registry):
            return None
        ledger = None
        if ledger_path:
            try:
                ledger = TemplateUsageLedger(ledger_path)
            except Exception as exc:  # noqa: BLE001 - coverage is best-effort
                logger.warning("decoration_policy: ledger unavailable (%s)", exc)
        policy = DecorationPolicy(
            registry,
            ledger,
            pool_size=pool_size,
            semantic_slice=semantic_slice,
            coverage_slice=coverage_slice,
            episode_id=episode_id,
        )
        state["registry"] = registry
        state["policy"] = policy
        _ACTIVE["policy"] = policy
        logger.info(
            "decoration_policy(%s): %d templates, pool=%d (semantic=%d, coverage=%d), ledger=%s",
            POLICY_VERSION, len(registry), pool_size, semantic_slice, coverage_slice,
            ledger_path or "off",
        )
        return policy

    def patched_registry(config: dict[str, Any]) -> dict[str, Any]:
        """Rebuild candidates and slot schemas in the training-time shapes."""
        baseline = original_registry(config) if callable(original_registry) else {}
        try:
            policy = ensure_policy(config or {})
            if policy is None:
                return baseline
            registry = policy.registry
            return {
                "candidates": [registry.candidate_card(sid) for sid in registry.template_ids],
                "slot_schemas": _LazySlotSchemas(registry),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("decoration_policy: registry rebuild failed (%s)", exc)
            return baseline

    def patched_retrieve(
        shot: dict[str, Any],
        segment: dict[str, Any],
        tool_args: dict[str, Any],
        *,
        candidate_templates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload = original_retrieve(
            shot, segment, tool_args, candidate_templates=candidate_templates
        )
        try:
            policy = state["policy"]
            if policy is None:
                return payload
            pool = policy.candidate_pool(shot or {}, segment or {})
            if pool:
                payload["candidates"] = pool
            payload["diversity_policy"] = policy.diversity_block()
            constraints = payload.get("constraints")
            if isinstance(constraints, dict):
                constraints["select_from_candidates_only"] = True
                constraints["prefer_template_not_used_in_this_video"] = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("decoration_policy: retrieve payload untouched (%s)", exc)
        return payload

    def patched_fill(
        shot: dict[str, Any],
        segment: dict[str, Any],
        tool_args: dict[str, Any],
        *,
        template_id: str,
        slot_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = original_fill(
            shot, segment, tool_args, template_id=template_id, slot_schema=slot_schema
        )
        try:
            policy = state["policy"]
            if policy is None:
                return payload
            # This is where the retrieve decision becomes observable, so it is also
            # where the ledger learns what was actually chosen.
            policy.record_choice(template_id, str(shot.get("shot_id", "")))
            payload["template_id"] = str(template_id or "")
            payload["association"] = policy.association_block(shot or {}, segment or {})
            payload["instruction"] = policy.fill_instruction(
                float(shot.get("duration_sec", 0) or 0)
            )
            payload["constraints"] = policy.fill_constraints()
        except Exception as exc:  # noqa: BLE001
            logger.warning("decoration_policy: fill payload untouched (%s)", exc)
        return payload

    orchestrator_cls._load_template_registry = staticmethod(patched_registry)
    orchestrator_cls._build_skill_payload_template_retrieve = staticmethod(patched_retrieve)
    orchestrator_cls._build_skill_payload_template_fill = staticmethod(patched_fill)
    orchestrator_cls._decoration_policy_installed = True
    logger.info("decoration_policy: installed on %s", orchestrator_cls.__name__)
    return state["policy"]


def _unwrap(method: Any) -> Any:
    """Get the plain function behind a ``staticmethod`` descriptor."""
    return method.__func__ if hasattr(method, "__func__") else method


def _resolve_gold_root(config: dict[str, Any]) -> str:
    if isinstance(config, dict):
        paths = config.get("paths")
        if isinstance(paths, dict) and paths.get("gold_template_root"):
            return str(paths["gold_template_root"])
    return os.environ.get("GOLD_TEMPLATE_ROOT", "")
