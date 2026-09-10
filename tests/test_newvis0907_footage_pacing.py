"""Tests for the newvis0907 footage identity + pacing planner (Task 7).

These lock the trustworthy-cut contract for the 0907 visual playlist:

* footage: canonical identity + global occupancy dedup by clip_id / real path /
  file fingerprint / high-similarity visual fingerprint, with safe degradation
  when a file is missing.
* pacing: ``plan_visual_playlist`` never freezes / loops / holds a last frame,
  keeps every subclip final duration >= 2s inside a 2-6s target band, uses each
  source once, prefers natural speed then slow motion down to a 1/3 floor, and
  reports ``coverage_status=short`` (with an explicit deficit + needs_retrieval /
  needs_split) instead of faking coverage.
* pacing model-shot maths: ``minimum_model_shots`` 20+ formula, long-shot
  splitting on semantic/punctuation boundaries with character conservation, and
  ``analyze_shot_pacing`` quality issues.

Run directly: ``python3 tests/test_newvis0907_footage_pacing.py``.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studio.newvis0907 import footage, pacing  # noqa: E402

PASS: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASS.append(name)
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}{': ' + detail if detail else ''}")
        raise AssertionError(name)


# ---------------------------------------------------------------------------
# footage identity + occupancy
# ---------------------------------------------------------------------------
def test_source_identity_and_content_signal(tmp: Path) -> None:
    print("\n[footage] source_identity / 文件内容信号 / canonical path")
    real = tmp / "clip_a.mp4"
    real.write_bytes(b"abc" * 1000)
    link = tmp / "alias.mp4"
    try:
        link.symlink_to(real)
        have_symlink = True
    except OSError:
        have_symlink = False

    ident_real = footage.source_identity({"clip_id": "c1", "clip_path": str(real)})
    check("clip_id extracted", ident_real.clip_id == "c1")
    check("content signal present for real file", bool(ident_real.fingerprint))

    if have_symlink:
        ident_link = footage.source_identity({"clip_id": "c1", "clip_path": str(link)})
        check("canonical path resolves symlink to same real path",
              ident_link.canonical_path == ident_real.canonical_path,
              f"{ident_link.canonical_path} vs {ident_real.canonical_path}")
        check("same real file yields same fingerprint",
              ident_link.fingerprint == ident_real.fingerprint)

    # Missing file degrades safely (no crash, no fingerprint), identity still usable.
    ident_missing = footage.source_identity({"clip_id": "c2", "clip_path": str(tmp / "gone.mp4")})
    check("missing file degrades to no fingerprint", ident_missing.fingerprint is None)
    check("missing file still has a clip_id", ident_missing.clip_id == "c2")


def test_visual_probe_is_injectable_and_optional() -> None:
    print("\n[footage] 视觉指纹可注入/可选")
    calls: list[str] = []

    def probe(item):
        calls.append(str(item.get("clip_path")))
        return [1.0, 0.0, 0.0]

    ident = footage.source_identity({"clip_id": "c1", "clip_path": "/x/a.mp4"}, visual_probe=probe)
    check("probe invoked", calls == ["/x/a.mp4"], str(calls))
    check("visual fingerprint stored", ident.visual == [1.0, 0.0, 0.0])

    # No probe => no visual fingerprint, and an inline fingerprint is honoured.
    ident_none = footage.source_identity({"clip_id": "c1", "clip_path": "/x/a.mp4"})
    check("visual fingerprint optional", ident_none.visual is None)
    ident_inline = footage.source_identity(
        {"clip_id": "c1", "clip_path": "/x/a.mp4", "visual_fingerprint": [0.0, 1.0]}
    )
    check("inline visual fingerprint honoured", ident_inline.visual == [0.0, 1.0])


def test_occupancy_dedup_by_every_signal(tmp: Path) -> None:
    print("\n[footage] 同 clip_id / 真实路径 / 文件指纹 / 高相似视觉指纹都视为重复")
    real = tmp / "shared.mp4"
    real.write_bytes(b"z" * 4096)
    link = tmp / "shared_link.mp4"
    symlinked = False
    try:
        link.symlink_to(real)
        symlinked = True
    except OSError:
        pass

    state = {
        "shot_plan": [{"shot_id": "S1"}, {"shot_id": "S2"}],
        "shot_assets": {
            "S1": {"clip_meta": {"visual_playlist": {"clips": [
                {"subclip_id": "S1::1", "clip_id": "used_a", "path": str(real),
                 "visual_fingerprint": [1.0, 0.0]},
            ]}}},
            "S2": {"clip_meta": {"visual_playlist": {"clips": [
                {"subclip_id": "S2::1", "clip_id": "used_b", "path": "/other/b.mp4"},
            ]}}},
        },
    }
    occ = footage.build_occupancy(state)

    candidates = [
        {"clip_id": "used_a", "clip_path": "/totally/different.mp4"},          # same clip_id
        {"clip_id": "new1", "clip_path": str(link) if symlinked else str(real)},  # same real path
        {"clip_id": "new2", "clip_path": "/other/b.mp4"},                      # same path as S2
        {"clip_id": "sim", "clip_path": "/n/sim.mp4", "visual_fingerprint": [0.99, 0.01]},  # near S1 visual
        {"clip_id": "fresh", "clip_path": str(tmp / "fresh.mp4"),
         "visual_fingerprint": [0.0, 1.0]},                                     # genuinely new
    ]
    kept = footage.filter_unused_candidates(candidates, occ, visual_threshold=0.9)
    kept_ids = [c.get("clip_id") for c in kept]
    check("same clip_id filtered", "used_a" not in kept_ids, str(kept_ids))
    check("same real path filtered", "new1" not in kept_ids, str(kept_ids))
    check("same literal path filtered", "new2" not in kept_ids, str(kept_ids))
    check("high visual similarity filtered", "sim" not in kept_ids, str(kept_ids))
    check("genuinely new candidate kept", kept_ids == ["fresh"], str(kept_ids))


def test_occupancy_can_exclude_a_shot(tmp: Path) -> None:
    print("\n[footage] build_occupancy 可排除当前重规划的镜头")
    state = {
        "shot_plan": [{"shot_id": "S1"}, {"shot_id": "S2"}],
        "shot_assets": {
            "S1": {"clip_meta": {"visual_playlist": {"clips": [
                {"clip_id": "own", "path": "/a.mp4"}]}}},
            "S2": {"clip_meta": {"visual_playlist": {"clips": [
                {"clip_id": "other", "path": "/b.mp4"}]}}},
        },
    }
    occ_all = footage.build_occupancy(state)
    occ_excl = footage.build_occupancy(state, exclude_shot_ids=["S1"])
    own = {"clip_id": "own", "clip_path": "/a.mp4"}
    check("own clip occupied globally", footage.filter_unused_candidates([own], occ_all, 0.9) == [])
    check("own clip free when its shot excluded",
          [c["clip_id"] for c in footage.filter_unused_candidates([own], occ_excl, 0.9)] == ["own"])


# ---------------------------------------------------------------------------
# pacing: plan_visual_playlist
# ---------------------------------------------------------------------------
def _persistent_fields_ok(clip: dict) -> bool:
    required = {
        "subclip_id", "clip_id", "path", "source_duration_sec", "source_in_sec",
        "source_out_sec", "source_used_duration_sec", "final_duration_sec",
        "playback_rate", "offset_sec", "start_sec", "end_sec", "score", "query",
        "fingerprint", "user_override", "replaced",
    }
    return required.issubset(clip.keys())


def test_natural_speed_fill_covers_and_is_unique() -> None:
    print("\n[pacing] 自然速度优先，每 source 只出现一次，字段齐全")
    primary = {"clip_id": "p", "clip_path": "/p.mp4", "source_duration_sec": 6.0, "score": 0.9}
    candidates = [
        {"clip_id": "c1", "clip_path": "/c1.mp4", "source_duration_sec": 6.0, "score": 0.8},
        {"clip_id": "c2", "clip_path": "/c2.mp4", "source_duration_sec": 6.0, "score": 0.7},
    ]
    plan = pacing.plan_visual_playlist(primary, candidates, target_duration=12.0)
    check("coverage covered", plan["coverage_status"] == "covered", str(plan))
    clips = plan["clips"]
    check("no <2s subclip", all(c["final_duration_sec"] >= 2.0 - 1e-6 for c in clips), str(clips))
    check("each subclip within 2-6s band",
          all(2.0 - 1e-6 <= c["final_duration_sec"] <= 6.0 + 1e-6 for c in clips), str(clips))
    ids = [c["clip_id"] for c in clips]
    check("each source used at most once", len(ids) == len(set(ids)), str(ids))
    check("natural speed => playback_rate 1.0",
          all(abs(c["playback_rate"] - 1.0) < 1e-6 for c in clips), str(clips))
    check("covered duration ~= target", abs(sum(c["final_duration_sec"] for c in clips) - 12.0) < 1e-6)
    check("all persistent fields present", all(_persistent_fields_ok(c) for c in clips), str(clips[0].keys()))
    # offsets are contiguous and non-overlapping.
    cursor = 0.0
    contiguous = True
    for c in clips:
        contiguous = contiguous and abs(c["offset_sec"] - cursor) < 1e-6
        cursor = c["end_sec"]
    check("offsets contiguous / no loop", contiguous, str(clips))


def test_slow_motion_floor_is_one_third() -> None:
    print("\n[pacing] 覆盖不足时慢放，最低 playback_rate=1/3（最多拉长3倍）")
    # One 4s source must cover 12s => would need 3x stretch (rate 1/3), no loop.
    primary = {"clip_id": "solo", "clip_path": "/solo.mp4", "source_duration_sec": 4.0, "score": 0.9}
    plan = pacing.plan_visual_playlist(primary, [], target_duration=12.0)
    rates = [c["playback_rate"] for c in plan["clips"]]
    check("never below the 1/3 floor", all(r >= 1.0 / 3.0 - 1e-6 for r in rates), str(rates))
    check("no freeze/loop: final <= 3x source_used",
          all(c["final_duration_sec"] <= 3.0 * c["source_used_duration_sec"] + 1e-6 for c in plan["clips"]),
          str(plan["clips"]))
    # 4s * 3 = 12s exactly coverable via slow motion.
    check("slow-motion covers exactly", plan["coverage_status"] == "covered", str(plan))
    check("slow-motion flags a retrieval hint", plan.get("needs_retrieval") is True, str(plan))


def test_coverage_short_is_honest_not_faked() -> None:
    print("\n[pacing] 覆盖仍不足 => coverage_status=short + deficit + needs_retrieval，不伪 covered")
    # 3s source, max stretch 9s, target 20s => impossible without loop/freeze.
    primary = {"clip_id": "tiny", "clip_path": "/tiny.mp4", "source_duration_sec": 3.0}
    plan = pacing.plan_visual_playlist(primary, [], target_duration=20.0)
    check("status is short", plan["coverage_status"] == "short", str(plan))
    check("explicit positive deficit", plan.get("deficit_sec", 0.0) > 0.0, str(plan))
    check("asks for retrieval or split",
          bool(plan.get("needs_retrieval")) or bool(plan.get("needs_split")), str(plan))
    check("does not fake covered", plan["coverage_status"] != "covered")
    check("no subclip loops beyond 3x",
          all(c["final_duration_sec"] <= 3.0 * c["source_used_duration_sec"] + 1e-6 for c in plan["clips"]),
          str(plan["clips"]))


def test_remainder_below_two_seconds_is_rebalanced() -> None:
    print("\n[pacing] 剩余<2s 通过重平衡吸收，绝不产生 <2s 子镜")
    # target 7s with 6s-capable sources: naive 6+1 would make a 1s subclip.
    primary = {"clip_id": "a", "clip_path": "/a.mp4", "source_duration_sec": 6.0}
    candidates = [{"clip_id": "b", "clip_path": "/b.mp4", "source_duration_sec": 6.0}]
    plan = pacing.plan_visual_playlist(primary, candidates, target_duration=7.0)
    clips = plan["clips"]
    check("no <2s subclip after rebalance",
          all(c["final_duration_sec"] >= 2.0 - 1e-6 for c in clips), str(clips))
    check("still covers target", abs(sum(c["final_duration_sec"] for c in clips) - 7.0) < 1e-6, str(clips))
    check("covered", plan["coverage_status"] == "covered", str(plan))


def test_target_below_two_allows_one_flagged_short_model_shot() -> None:
    print("\n[pacing] target<2 时允许一个带 quality warning 的短镜，或返回 short；不破坏正常 2s 规则")
    primary = {"clip_id": "s", "clip_path": "/s.mp4", "source_duration_sec": 6.0}
    plan = pacing.plan_visual_playlist(primary, [], target_duration=1.4)
    if plan["coverage_status"] == "covered":
        check("exactly one short special subclip", len(plan["clips"]) == 1, str(plan))
        check("short subclip covers the target",
              abs(plan["clips"][0]["final_duration_sec"] - 1.4) < 1e-6, str(plan))
        check("short subclip carries a quality warning",
              bool(plan.get("quality_warning")) or any("2" in str(w) for w in plan.get("warnings", [])),
              str(plan))
    else:
        check("otherwise returns short honestly", plan["coverage_status"] == "short", str(plan))


def test_occupied_sources_are_not_reused_across_playlists(tmp: Path) -> None:
    print("\n[pacing] 全局占用：已用 source 不会再次进入新 playlist（相邻/全片不重复）")
    state = {
        "shot_plan": [{"shot_id": "S1"}],
        "shot_assets": {"S1": {"clip_meta": {"visual_playlist": {"clips": [
            {"clip_id": "used", "path": "/used.mp4"}]}}}},
    }
    occ = footage.build_occupancy(state)
    primary = {"clip_id": "used", "clip_path": "/used.mp4", "source_duration_sec": 6.0}
    candidates = [{"clip_id": "fresh", "clip_path": "/fresh.mp4", "source_duration_sec": 6.0}]
    plan = pacing.plan_visual_playlist(primary, candidates, target_duration=6.0, occupied=occ)
    used_ids = [c["clip_id"] for c in plan["clips"]]
    check("occupied primary excluded", "used" not in used_ids, str(used_ids))
    check("fresh source used instead", "fresh" in used_ids, str(used_ids))


def test_prefer_retrieval_order_consumes_ranked_videos_as_units() -> None:
    print("\n[pacing] 检索序号优先：按 retrieval_rank 消费，每个视频作为一个可编辑单元")
    sources = [
        {"clip_id": "c3", "clip_path": "/c3.mp4", "source_duration_sec": 4.0, "retrieval_rank": 3, "score": 0.99, "query": "大海"},
        {"clip_id": "c1", "clip_path": "/c1.mp4", "source_duration_sec": 4.0, "retrieval_rank": 1, "score": 0.2, "query": "大海"},
        {"clip_id": "c2", "clip_path": "/c2.mp4", "source_duration_sec": 4.0, "retrieval_rank": 2, "score": 0.3, "query": "大海"},
    ]
    plan = pacing.plan_visual_playlist(
        None,
        sources,
        target_duration=10.0,
        prefer_retrieval_order=True,
        shot_id="S88",
    )
    clips = plan["clips"]
    check("covered", plan["coverage_status"] == "covered", str(plan))
    check("consumes by retrieval_rank", [c["clip_id"] for c in clips][:2] == ["c1", "c2"], str(clips))
    check("every emitted unit carries retrieval_rank", all(int(c.get("retrieval_rank", 0)) > 0 for c in clips), str(clips))
    check("every emitted unit keeps query", all(c.get("query") == "大海" for c in clips), str(clips))


# ---------------------------------------------------------------------------
# pacing: model-shot maths / analysis / splitting
# ---------------------------------------------------------------------------
def test_minimum_model_shots_formula() -> None:
    print("\n[pacing] minimum_model_shots：>=240s 时 max(20, ceil(dur/15))")
    check("240s => 20", pacing.minimum_model_shots(240) == 20, str(pacing.minimum_model_shots(240)))
    check("300s => 20", pacing.minimum_model_shots(300) == 20, str(pacing.minimum_model_shots(300)))
    check("301s => 21", pacing.minimum_model_shots(301) == 21, str(pacing.minimum_model_shots(301)))
    check("600s => 40", pacing.minimum_model_shots(600) == 40, str(pacing.minimum_model_shots(600)))
    check("short video under 240s not forced to 20",
          pacing.minimum_model_shots(120) < 20, str(pacing.minimum_model_shots(120)))
    check("matches ceil(dur/15) for large",
          pacing.minimum_model_shots(1000) == max(20, math.ceil(1000 / 15)))


def test_split_long_shot_plan_conserves_characters() -> None:
    print("\n[pacing] split_long_shot_plan：在标点/语义边界拆 >15s 镜头，合并旁白零丢字")
    narration = (
        "人工智能在最近几年里发展得非常快，"
        "很多公司都开始使用大模型来提升效率。"
        "但是这也带来了新的挑战，"
        "比如数据安全和算力成本的问题。"
        "所以我们需要更谨慎地部署这些系统。"
    )
    state = {
        "shot_plan": [
            {"shot_id": "L1", "segment_id": "seg1", "order": 0,
             "vocal_script": narration, "visual_query": "科技公司办公室",
             "duration_sec": 24.0, "is_spotlight": False},
            {"shot_id": "L2", "segment_id": "seg1", "order": 1,
             "vocal_script": "这是一个短镜头。", "visual_query": "城市夜景",
             "duration_sec": 6.0},
        ],
    }
    new_plan = pacing.split_long_shot_plan(state)
    split_children = [s for s in new_plan if str(s.get("shot_id")).startswith("L1")]
    check("long shot was split into >=2 pieces", len(split_children) >= 2, str([s["shot_id"] for s in new_plan]))
    merged = "".join(s["vocal_script"] for s in split_children)
    check("merged narration is character-identical (no loss, no dup)",
          merged == narration, f"{merged!r} != {narration!r}")
    check("children keep the segment", all(s["segment_id"] == "seg1" for s in split_children))
    check("children keep the visual query",
          all(s.get("visual_query") == "科技公司办公室" for s in split_children))
    check("short shot L2 is left untouched",
          any(s["shot_id"] == "L2" and s["vocal_script"] == "这是一个短镜头。" for s in new_plan))
    # each split child should be within the model-shot band (<=15s planned)
    check("no split child still exceeds 15s",
          all(s.get("duration_sec", 0) <= 15.0 + 1e-6 for s in split_children), str(split_children))
    # order preserved: split children come before L2.
    ids = [s["shot_id"] for s in new_plan]
    check("split children precede the next shot", ids.index(split_children[-1]["shot_id"]) < ids.index("L2"))


def test_analyze_shot_pacing_reports_issues() -> None:
    print("\n[pacing] analyze_shot_pacing：报告过长镜头/过短子镜/镜头数不足")
    state = {
        "input_text": "x",
        "total_duration_sec": 300.0,
        "shot_plan": [
            {"shot_id": "A", "duration_sec": 22.0, "vocal_script": "很长的镜头。" * 5},
        ],
        "shot_assets": {
            "A": {"clip_meta": {"visual_playlist": {"clips": [
                {"subclip_id": "A::1", "final_duration_sec": 1.0},
            ]}}},
        },
    }
    report = pacing.analyze_shot_pacing(state)
    codes = {i.get("code") for i in report.get("issues", [])}
    check("flags an over-long shot", "shot_too_long" in codes, str(codes))
    check("flags a sub-2s subclip", "subclip_too_short" in codes, str(codes))
    check("flags insufficient model shots", "too_few_model_shots" in codes, str(codes))
    check("status is not ok", report.get("status") != "ok", str(report))


def test_content_signal_ignores_mtime_and_path(tmp: Path) -> None:
    print("\n[footage][M2] file_content_signal 去 mtime：同字节副本(异路径/异mtime)去重")
    data = b"identical-bytes-payload" * 4096
    a = tmp / "a.mp4"
    sub = tmp / "nested"
    sub.mkdir(parents=True, exist_ok=True)
    b = sub / "b.mp4"
    a.write_bytes(data)
    b.write_bytes(data)
    # Force a clearly different mtime on the copy.
    old = time.time() - 100000
    os.utime(b, (old, old))

    sig_a = footage.file_content_signal(str(a))
    sig_b = footage.file_content_signal(str(b))
    check("content signal exists", bool(sig_a) and bool(sig_b))
    check("same bytes across path + mtime => identical content signal",
          sig_a == sig_b, f"{sig_a} vs {sig_b}")
    # The signal must be size:hash only (no mtime component) => exactly one ':'.
    check("content signal carries no mtime component (size:hash)",
          sig_a.count(":") == 1, sig_a)

    # And the identity-level dedup must treat the copy as already-occupied even
    # though clip_id and path differ.
    occ = footage.build_occupancy({
        "shot_plan": [{"shot_id": "S1"}],
        "shot_assets": {"S1": {"clip_meta": {"visual_playlist": {"clips": [
            {"clip_id": "orig", "path": str(a)}]}}}},
    })
    kept = footage.filter_unused_candidates(
        [{"clip_id": "different_id", "clip_path": str(b)}], occ, 0.9)
    check("content-identical copy at a new path is deduped by fingerprint",
          kept == [], str(kept))


def test_split_long_shot_plan_cleans_physical_range_fields() -> None:
    print("\n[pacing][L6] 拆分清理/重算 start_sec/end_sec/source_shot_ids，保业务元数据")
    narration = (
        "第一句话讲清楚背景信息。第二句话补充关键的细节内容。"
        "第三句话给出更完整的推理过程。第四句话继续展开说明。"
        "第五句话负责收束并总结全部要点。第六句话再给一个例子。"
    )
    state = {
        "shot_plan": [
            {
                "shot_id": "L1", "segment_id": "seg1", "order": 3,
                "vocal_script": narration, "visual_query": "科技公司办公室",
                "chapter_title": "开场", "is_spotlight": True,
                "duration_sec": 30.0,
                # physical provenance fields that become inconsistent if blindly copied
                "start_sec": 100.0, "end_sec": 130.0,
                "source_shot_ids": ["phys_7", "phys_8"],
            },
        ],
    }
    out = pacing.split_long_shot_plan(state)
    children = [s for s in out if str(s.get("shot_id")).startswith("L1")]
    check("long shot split into >=2 children", len(children) >= 2, str([s["shot_id"] for s in out]))

    # narration still conserved (guard against regressions from the cleanup)
    check("narration conserved after split",
          "".join(c["vocal_script"] for c in children) == narration)

    # business metadata preserved on every child
    check("segment preserved", all(c.get("segment_id") == "seg1" for c in children))
    check("visual_query preserved", all(c.get("visual_query") == "科技公司办公室" for c in children))
    check("chapter preserved", all(c.get("chapter_title") == "开场" for c in children))

    # physical ranges recomputed: contiguous, non-overlapping, spanning the parent range
    starts_ends = [(c.get("start_sec"), c.get("end_sec")) for c in children]
    check("first child keeps parent range start",
          abs(_f(children[0].get("start_sec")) - 100.0) < 1e-3, str(starts_ends))
    check("last child keeps parent range end",
          abs(_f(children[-1].get("end_sec")) - 130.0) < 1e-3, str(starts_ends))
    contiguous = all(
        abs(_f(a[1]) - _f(b[0])) < 1e-3 and _f(b[1]) > _f(b[0])
        for a, b in zip(starts_ends, starts_ends[1:])
    ) and _f(starts_ends[0][1]) > _f(starts_ends[0][0])
    check("child physical ranges are contiguous & non-overlapping", contiguous, str(starts_ends))
    check("parent's identical range is not copied verbatim onto every child",
          not all(se == (100.0, 130.0) for se in starts_ends), str(starts_ends))

    # stale physical source-shot provenance must be cleaned, never duplicated wholesale
    check("stale source_shot_ids cleaned on children",
          all(not c.get("source_shot_ids") for c in children),
          str([c.get("source_shot_ids") for c in children]))

    # fully JSON serializable
    json.dumps(out, allow_nan=False)


def test_plan_visual_playlist_rejects_non_finite_target() -> None:
    print("\n[pacing][L7] target_duration NaN/inf/负数 => 结构化 invalid/short 且 JSON 可序列化")
    primary = {"clip_id": "c", "clip_path": "", "source_duration_sec": 6.0}
    for bad in (float("nan"), float("inf"), float("-inf"), -5.0):
        pl = pacing.plan_visual_playlist(primary, [], bad)
        check(f"non-finite/neg target => structured invalid/short ({bad})",
              pl.get("coverage_status") in ("invalid", "short"), str(pl.get("coverage_status")))
        check(f"no clips emitted for a bogus target ({bad})", pl.get("clips") == [], str(pl.get("clips")))
        # strict JSON: must not leak NaN / Infinity tokens
        try:
            json.dumps(pl, allow_nan=False)
            serializable = True
        except ValueError:
            serializable = False
        check(f"result is strict-JSON serializable ({bad})", serializable, str(pl))


def _f(v: object) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def main() -> int:
    import shutil

    tmp = Path(tempfile.mkdtemp(prefix="aicut_footage_pacing_"))

    def sub(name: str) -> Path:
        d = tmp / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    try:
        test_source_identity_and_content_signal(sub("a"))
        test_visual_probe_is_injectable_and_optional()
        test_occupancy_dedup_by_every_signal(sub("b"))
        test_occupancy_can_exclude_a_shot(sub("c"))
        test_natural_speed_fill_covers_and_is_unique()
        test_slow_motion_floor_is_one_third()
        test_coverage_short_is_honest_not_faked()
        test_remainder_below_two_seconds_is_rebalanced()
        test_target_below_two_allows_one_flagged_short_model_shot()
        test_occupied_sources_are_not_reused_across_playlists(sub("d"))
        test_prefer_retrieval_order_consumes_ranked_videos_as_units()
        test_minimum_model_shots_formula()
        test_split_long_shot_plan_conserves_characters()
        test_split_long_shot_plan_cleans_physical_range_fields()
        test_analyze_shot_pacing_reports_issues()
        test_content_signal_ignores_mtime_and_path(sub("m2"))
        test_plan_visual_playlist_rejects_non_finite_target()
    except AssertionError as exc:
        print(f"\n失败: {exc}")
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n全部通过: {len(PASS)} 项断言")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
