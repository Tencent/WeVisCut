"""Offline checks for the batch-annotation control plane.

Exercises the pieces that must be right before a large batch starts, none of
which need a model endpoint:

1. curation  — the manifest filters, stratified sampling and folder naming
2. decoration — whether the candidate pool actually spreads across all 90
   templates instead of collapsing onto the handful the retriever likes
3. qc        — template/slot analysis over an already-recorded trajectory

Run::

    python3 tests/smoke_wevisioncut.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.append(str(PKG_SRC))

DATASET_ROOT = Path(os.environ.get("LVP_SMOKE_DATASET", ""))
GOLD_ROOT = Path(os.environ.get("WEVISIONCUT_GOLD_TEMPLATE", ""))
SAMPLE_EPISODE = Path(os.environ.get("LVP_SMOKE_EPISODE", ""))

_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        _failures.append(label)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def test_curation() -> None:
    section("curation")
    from longvideo_polar.curation import (
        CurationFilter,
        build_worklist,
        episode_dir_name,
        read_worklist,
        write_worklist,
    )

    check(
        "folder naming is seq__BV__asset8",
        episode_dir_name(123, "0a1b2c3d4e5f67890123456789abcdef", "example_src_001")
        == "000123__example_src_001__0a1b2c3d",
        episode_dir_name(123, "0a1b2c3d4e5f67890123456789abcdef", "example_src_001"),
    )
    check(
        "naming degrades gracefully without a source id",
        episode_dir_name(7, "0a1b2c3d4e5f67890123456789abcdef") == "000007__0a1b2c3d",
        episode_dir_name(7, "0a1b2c3d4e5f67890123456789abcdef"),
    )

    manifest = DATASET_ROOT / "manifest.jsonl"
    if not manifest.is_file():
        check("manifest reachable", False, f"missing {manifest}")
        return

    selection = CurationFilter(min_chars=800, max_chars=8000, languages=["zh"], limit=200)
    items, rejected = build_worklist(manifest, DATASET_ROOT, selection)
    check("worklist honours the limit", len(items) == 200, f"{len(items)} items")
    check("rejection reasons are tallied", bool(rejected), json.dumps(rejected, ensure_ascii=False))
    check(
        "every item respects the char window",
        all(800 <= it.script_chars <= 8000 for it in items),
        f"chars {min(i.script_chars for i in items)}..{max(i.script_chars for i in items)}",
    )
    check(
        "folder names are unique",
        len({it.episode_id for it in items}) == len(items),
    )
    check(
        "script files exist on disk",
        all(Path(it.script_path).is_file() for it in items[:20]),
    )

    repeat, _ = build_worklist(manifest, DATASET_ROOT, selection)
    check(
        "selection is deterministic across runs",
        [i.asset_id for i in repeat] == [i.asset_id for i in items],
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = write_worklist(tmp, items, selection, manifest_path=str(manifest), rejected=rejected)
        loaded, meta = read_worklist(path)
        check(
            "worklist round-trips through disk",
            [i.asset_id for i in loaded] == [i.asset_id for i in items] and bool(meta),
        )

    print(f"  sample: {items[0].episode_id}  ({items[0].script_chars} chars)")


def test_decoration() -> None:
    section("decoration policy")
    from longvideo_polar.decoration_policy import DecorationPolicy, GoldTemplateRegistry

    if not GOLD_ROOT.is_dir():
        check("gold template root reachable", False, f"missing {GOLD_ROOT}")
        return

    from longvideo_polar.decoration_policy import TemplateUsageLedger

    with tempfile.TemporaryDirectory() as tmp:
        registry = GoldTemplateRegistry(GOLD_ROOT, cache_path=Path(tmp) / "cache.json")
        size = len(registry.template_ids)
        check("registry loads the full template set", size >= 80, f"{size} templates")

        # The recorded turn has to be drawn from the same input distribution the
        # SFT corpus was built on, so the card must match the offline builder
        # field for field — no extras, none missing.
        canonical = {
            "template_id", "display_name", "family", "category", "category_name",
            "use_for", "spotlight_level", "slot_names", "duration_range",
        }
        card = registry.candidate_card(registry.template_ids[0])
        check(
            "candidate card matches build_unified_sft_v5 exactly",
            set(card) == canonical,
            f"extra={sorted(set(card) - canonical)} missing={sorted(canonical - set(card))}",
        )
        check(
            "candidate cards carry a real label and use_for",
            bool(card.get("display_name")) and bool(card.get("use_for")),
            json.dumps(card, ensure_ascii=False)[:170],
        )
        check(
            "no candidate card is left unlabeled",
            all(registry.candidate_card(sid).get("display_name") for sid in registry.template_ids),
        )
        schema_keys = {
            "template_id", "display_name", "family", "duration_default",
            "slots", "data_slots",
        }
        check(
            "slot schema matches build_unified_sft_v5 exactly",
            set(registry.normalized_slot_schema(registry.template_ids[0])) == schema_keys,
            str(sorted(set(registry.normalized_slot_schema(registry.template_ids[0])))),
        )

        ledger = TemplateUsageLedger(Path(tmp) / "ledger.db")
        policy = DecorationPolicy(registry, ledger, episode_id="smoke")

        # Every shot here carries the *same* topic. That is the pathological input:
        # a purely semantic retriever answers it with the same narrow head every
        # time, which is what produced 7-of-90 coverage on the real rollout.
        segment = {
            "summary": "讲产品发布会上的三个亮点和随后的数据增长",
            "key_points": ["新芯片性能翻倍", "售价下调", "首周销量破百万"],
        }
        chosen: Counter[str] = Counter()
        pools: Counter[str] = Counter()
        per_episode_distinct: list[float] = []

        for episode in range(30):
            policy = DecorationPolicy(registry, ledger, episode_id=f"ep{episode:03d}")
            picks: list[str] = []
            for shot in range(12):
                shot_payload = {
                    "shot_id": f"{episode}-{shot}",
                    "visual_query": "产品发布会 舞台 大屏幕 数据图表",
                    "chapter_title": "发布会亮点",
                    "duration": 5.0,
                }
                pool = policy.candidate_pool(shot_payload, segment)
                ids = [c["template_id"] for c in pool]
                pools.update(ids)
                # Stand in for the model: always take the first candidate offered,
                # so the measurement reflects the pool alone, not model taste.
                policy.record_choice(ids[0], shot_id=shot_payload["shot_id"])
                picks.append(ids[0])
                chosen[ids[0]] += 1
            per_episode_distinct.append(len(set(picks)) / len(picks))

        check(
            "candidate pools span most of the registry",
            len(pools) >= size * 0.8,
            f"{len(pools)}/{size} templates offered",
        )
        check(
            "choices spread across the registry (was 7/90)",
            len(chosen) >= size * 0.5,
            f"{len(chosen)}/{size} templates chosen",
        )
        top_share = chosen.most_common(1)[0][1] / sum(chosen.values())
        check(
            "no single template dominates (was 34%)",
            top_share < 0.10,
            f"top template = {top_share:.1%} of choices",
        )
        within = sum(per_episode_distinct) / len(per_episode_distinct)
        check(
            "shots within one video differ from each other",
            within > 0.8,
            f"{within:.0%} of shots in a video use a distinct template",
        )

        instruction = policy.fill_instruction(5.0)
        check(
            "fill instruction frames the task as rewriting, not extraction",
            "联想改写" in instruction and "摘抄" in instruction,
        )
        association = policy.association_block(
            {"visual_query": "城市夜景 车流 延时", "chapter_title": "通勤"}, segment
        )
        check(
            "association block hoists visual query + summary",
            association["visual_query"] == "城市夜景 车流 延时"
            and association["segment_summary"] == segment["summary"]
            and association["key_points"] == segment["key_points"],
        )
        diversity = policy.diversity_block()
        check(
            "diversity block reports what the video already used",
            diversity["prefer_unused_template"] is True
            and len(diversity["already_used_in_this_video"]) > 1,
        )

        schema = registry.normalized_slot_schema(chosen.most_common(1)[0][0])
        check(
            "slot schema is retrievable and shaped for SFT",
            isinstance(schema, dict) and bool(schema.get("slots")),
            f"slots={len(schema.get('slots') or [])}",
        )

        cache = registry.build_cache()
        check("consolidated slot cache is written", Path(cache).is_file())


def test_qc() -> None:
    section("qc")
    from longvideo_polar.qc import analyze_slots, analyze_templates

    shard = SAMPLE_EPISODE / "trajectories" / "generator.jsonl"
    if not shard.is_file():
        check("sample trajectory reachable", False, f"missing {shard}")
        return

    templates = analyze_templates(shard, registry_size=90)
    check(
        "template choices are read out of the recorded episode",
        templates.get("total_choices", 0) > 0,
        json.dumps(
            {k: v for k, v in templates.items() if not isinstance(v, (list, dict))},
            ensure_ascii=False,
        ),
    )
    slots = analyze_slots(shard)
    check(
        "template_fill samples parse into slots",
        slots.get("total_slots", 0) > 0
        and slots.get("parse_failures", 1) < slots.get("fill_samples", 0),
        json.dumps(
            {k: v for k, v in slots.items() if not isinstance(v, (list, dict))},
            ensure_ascii=False,
        ),
    )

    # This episode predates the decoration policy, so it is the "before" column:
    # the numbers here are the collapse the policy exists to remove.
    print(
        f"  baseline (pre-policy) rollout: {templates.get('distinct_templates')}"
        f"/{templates.get('registry_size')} templates, "
        f"gini={templates.get('gini')}, top5={templates.get('top5_share')}, "
        f"intra-episode distinctness={templates.get('mean_intra_episode_distinctness')}"
    )
    print(f"  baseline slot verbatim rate: {slots.get('verbatim_rate')}")


def test_endpoints() -> None:
    section("endpoint pool")
    from longvideo_polar.endpoints import EndpointPool

    with tempfile.TemporaryDirectory() as tmp:
        pool = EndpointPool(state_path=Path(tmp) / "pool.json")
        ep = pool.add("10.0.0.1:8080", probe=False)
        check("bare ip:port normalizes to a base url", ep.base_url.startswith("http://"), ep.base_url)
        pool.add("http://10.0.0.2:8080/v1", probe=False)
        check("pool holds both machines", len(pool.all()) == 2)
        check("duplicate add is idempotent", len(pool.all()) == 2 and (
            pool.add("10.0.0.1:8080", probe=False) is not None and len(pool.all()) == 2
        ))
        pool.save()
        reloaded = EndpointPool(state_path=Path(tmp) / "pool.json")
        reloaded.load()
        check("pool survives a restart", len(reloaded.all()) == 2)

        for endpoint in pool.all():
            endpoint.healthy = True
        first = pool.acquire()
        check("acquire hands out a healthy machine", first is not None)
        check("in-flight is tracked", first is not None and first.in_flight == 1)
        pool.release(first.base_url, success=True)
        check("release returns capacity", first.in_flight == 0 and first.completed == 1)
        pool.mark_unhealthy(first.base_url, "probe failed")
        check("unhealthy machines are skipped", not first.healthy)


def main() -> int:
    for test in (test_curation, test_decoration, test_qc, test_endpoints):
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - smoke test reports, never raises
            import traceback

            print(f"  [FAIL] {test.__name__} raised {type(exc).__name__}: {exc}")
            traceback.print_exc()
            _failures.append(test.__name__)

    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed:")
        for name in _failures:
            print(f"  - {name}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
