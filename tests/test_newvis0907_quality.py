"""Unit + fixture tests for the newvis0907 unified quality strategy (Task 10).

Run: python3 tests/test_newvis0907_quality.py

These lock the contract of ``studio.newvis0907.quality``:

* ``score_from_issues`` uses explicit, stable severity weights;
* ``evaluate_project`` checks every trustworthy dimension (subtitles, footage
  identity/coverage/rate/freeze, 20+ model-shot rule + 8-15s band, narration +
  final loudness, semantic animation phrase/time/type/template/primary/gate);
* a clean, internally-consistent global fixture (4-5 min, 20+ shots, one 3-subclip
  shot) evaluates to ``pass``, and each injected defect surfaces its own issue;
* ``propose_fixes`` only proposes safe actions and never rewrites narration.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studio.newvis0907 import quality  # noqa: E402
from studio.newvis0907.subtitles import build_subtitle_cues  # noqa: E402

PASS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASS.append(name)
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}{': ' + detail if detail else ''}")
        raise AssertionError(name)


def _codes(report: dict) -> set:
    return {i["code"] for i in report["issues"]}


def _within_target_loudness() -> dict:
    return {
        "status": "normalized",
        "normalized": True,
        "target": {"integrated_lufs": -16.0, "true_peak_dbtp": -1.5,
                   "tolerance_lu": 1.0, "lra_lu": 11.0},
        "output": {"output_i": -16.1, "output_tp": -1.6},
    }


def _make_shot(sid, segment, narration, duration, clips, *, loudness=None,
               semantic=None):
    """Build a fully-consistent shot + assets using the real helpers."""
    built = build_subtitle_cues(narration, duration)
    shot = {
        "shot_id": sid,
        "segment_id": segment,
        "vocal_script": narration,
        "subtitle_display_text": built["display_text"],
        "duration_sec": duration,
    }
    assets = {
        "subtitle_cues": built["cues"],
        "audio_meta": {"duration_sec": duration,
                       "loudness": loudness if loudness is not None else _within_target_loudness()},
        "clip_meta": {"visual_playlist": {"clips": clips, "coverage_status": "covered"}},
    }
    if semantic is not None:
        shot["semantic_animations"] = semantic
        assets["semantic_animations"] = semantic
    return shot, assets


def _clip(cid, path, source_dur, final, start, *, rate=None, **extra):
    used = source_dur
    r = rate if rate is not None else (used / final if final else 1.0)
    d = {
        "subclip_id": f"{cid}",
        "clip_id": cid,
        "path": path,
        "source_duration_sec": source_dur,
        "source_used_duration_sec": used,
        "final_duration_sec": final,
        "playback_rate": r,
        "start_sec": start,
        "offset_sec": start,
        "end_sec": start + final,
    }
    d.update(extra)
    return d


def _two_clips(prefix, seed, base=100.0):
    """Two distinct 3s subclips (natural speed, >=2s)."""
    return [
        _clip(f"{prefix}_a{seed}", f"/clips/{prefix}_a{seed}.mp4", 3.0, 3.0, 0.0),
        _clip(f"{prefix}_b{seed}", f"/clips/{prefix}_b{seed}.mp4", 3.0, 3.0, 3.0),
    ]


def _global_pass_state():
    """4-5 min, 22 model shots (each 12s -> 264s), one shot has 3 subclips."""
    shots, assets_map, segments = [], {}, []
    narrations = [
        "我们先来看第一个关键的问题背景。",
        "接着深入分析它背后的真正原因。",
        "这里有一个非常重要的核心结论。",
        "再看看另一个容易被忽略的细节。",
    ]
    n = 22
    for i in range(1, n + 1):
        sid = f"S{i:02d}"
        seg = f"seg{(i - 1) // 6 + 1}"
        narration = narrations[(i - 1) % len(narrations)]
        if i == 5:
            clips = [
                _clip(f"c{i}_a", f"/clips/c{i}_a.mp4", 4.0, 4.0, 0.0),
                _clip(f"c{i}_b", f"/clips/c{i}_b.mp4", 4.0, 4.0, 4.0),
                _clip(f"c{i}_c", f"/clips/c{i}_c.mp4", 4.0, 4.0, 8.0),
            ]
        else:
            clips = [
                _clip(f"c{i}_a", f"/clips/c{i}_a.mp4", 6.0, 6.0, 0.0),
                _clip(f"c{i}_b", f"/clips/c{i}_b.mp4", 6.0, 6.0, 6.0),
            ]
        shot, assets = _make_shot(sid, seg, narration, 12.0, clips)
        shots.append(shot)
        assets_map[sid] = assets
    seen = set()
    for s in shots:
        if s["segment_id"] not in seen:
            seen.add(s["segment_id"])
            segments.append({"segment_id": s["segment_id"], "title": s["segment_id"]})
    return {
        "segments": segments,
        "shot_plan": shots,
        "shot_assets": assets_map,
        "total_duration_sec": 264.0,
        "final_video_path": "/tmp/final.mp4",
        "final_measurement": _within_target_loudness(),
    }


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def test_score_weights_stable_and_ordered():
    print("\n[quality] score 用稳定 severity 权重")
    check("no issues => 100", quality.score_from_issues([]) == 100)
    one_warn = quality.score_from_issues([{"severity": "warning"}])
    one_fail = quality.score_from_issues([{"severity": "fail"}])
    check("one warning => 95", one_warn == 95, str(one_warn))
    check("one fail => 80", one_fail == 80, str(one_fail))
    check("fail costs more than warning", one_fail < one_warn)
    two_fail = quality.score_from_issues([{"severity": "fail"}, {"severity": "fail"}])
    check("two fails lower than one", two_fail < one_fail, str(two_fail))
    check("score never below 0",
          quality.score_from_issues([{"severity": "fail"}] * 20) == 0)
    check("deterministic",
          quality.score_from_issues([{"severity": "fail"}, {"severity": "warning"}])
          == quality.score_from_issues([{"severity": "warning"}, {"severity": "fail"}]))
    # weights exposed for auditability
    check("weights exposed", quality.SEVERITY_WEIGHTS["fail"] > quality.SEVERITY_WEIGHTS["warning"] > 0)


def test_status_mapping():
    print("\n[quality] status pass/warning/fail 映射")
    check("empty => pass", quality.status_from_issues([]) == "pass")
    check("warning => warning", quality.status_from_issues([{"severity": "warning"}]) == "warning")
    check("fail dominates", quality.status_from_issues(
        [{"severity": "warning"}, {"severity": "fail"}]) == "fail")
    check("info only => pass", quality.status_from_issues([{"severity": "info"}]) == "pass")


# ---------------------------------------------------------------------------
# global pass fixture
# ---------------------------------------------------------------------------
def test_global_fixture_passes():
    print("\n[quality] 干净的全局 fixture => pass")
    report = quality.evaluate_project(_global_pass_state())
    check("status pass", report["status"] == "pass", str(_codes(report)))
    check("score 100", report["score"] == 100, str(report["score"]))
    check("no issues", report["issues"] == [], str(_codes(report)))
    check("summary counts model shots", report["summary"]["model_shot_count"] == 22)
    check("summary total duration", report["summary"]["total_duration_sec"] >= 240)
    check("report is policy 0907", report["policy_version"] == "0907")
    # required stable issue schema keys exist even if empty
    check("report has status/score/summary/issues",
          all(k in report for k in ("status", "score", "summary", "issues")))


# ---------------------------------------------------------------------------
# per-dimension defect injection
# ---------------------------------------------------------------------------
def test_subtitle_defects():
    print("\n[quality] 字幕：字符守恒 / 标点 / 行容量")
    state = _global_pass_state()
    # tamper display text to drop a character (char conservation break)
    state["shot_plan"][0]["subtitle_display_text"] = "我们先来看第一个关键。"
    report = quality.evaluate_project(state)
    check("char mismatch detected", "subtitle_char_mismatch" in _codes(report), str(_codes(report)))

    state2 = _global_pass_state()
    narr = state2["shot_plan"][1]["vocal_script"]
    # a display without any terminal punctuation
    from studio.newvis0907.subtitles import _compact_text, HARD_PUNCT
    state2["shot_plan"][1]["subtitle_display_text"] = _compact_text(narr).rstrip("".join(HARD_PUNCT))
    r2 = quality.evaluate_project(state2)
    check("missing punctuation detected", "subtitle_punctuation" in _codes(r2), str(_codes(r2)))

    state3 = _global_pass_state()
    long_line = "这是一句非常非常长的字幕需要超过十四个字符的限制来触发问题"
    state3["shot_plan"][2]["vocal_script"] = long_line
    state3["shot_plan"][2]["subtitle_display_text"] = long_line
    state3["shot_assets"]["S03"]["subtitle_cues"] = [
        {"start": 0, "end": 12, "text": long_line, "lines": [long_line]}
    ]
    r3 = quality.evaluate_project(state3)
    check("over-long line detected", "subtitle_line_too_long" in _codes(r3), str(_codes(r3)))


def test_footage_duplicate_and_rate_and_freeze():
    print("\n[quality] 素材：重复/覆盖/子镜数/时长/慢放/冻结")
    # same-playlist duplicate: both clips share a clip_id
    state = _global_pass_state()
    dup = state["shot_assets"]["S01"]["clip_meta"]["visual_playlist"]["clips"]
    dup[1]["clip_id"] = dup[0]["clip_id"]
    dup[1]["path"] = dup[0]["path"]
    r = quality.evaluate_project(state)
    check("playlist duplicate", "footage_duplicate_in_playlist" in _codes(r), str(_codes(r)))

    # adjacent duplicate
    s2 = _global_pass_state()
    a = s2["shot_assets"]["S02"]["clip_meta"]["visual_playlist"]["clips"][0]
    b = s2["shot_assets"]["S03"]["clip_meta"]["visual_playlist"]["clips"][0]
    b["clip_id"] = a["clip_id"]
    b["path"] = a["path"]
    r2 = quality.evaluate_project(s2)
    check("adjacent duplicate", "footage_duplicate_adjacent" in _codes(r2), str(_codes(r2)))

    # whole-film duplicate (non-adjacent)
    s3 = _global_pass_state()
    a = s3["shot_assets"]["S02"]["clip_meta"]["visual_playlist"]["clips"][0]
    b = s3["shot_assets"]["S10"]["clip_meta"]["visual_playlist"]["clips"][0]
    b["clip_id"] = a["clip_id"]
    b["path"] = a["path"]
    r3 = quality.evaluate_project(s3)
    check("whole-film duplicate", "footage_duplicate_global" in _codes(r3), str(_codes(r3)))

    # sub-2s subclip
    s4 = _global_pass_state()
    s4["shot_assets"]["S04"]["clip_meta"]["visual_playlist"]["clips"][0]["final_duration_sec"] = 1.2
    r4 = quality.evaluate_project(s4)
    check("sub-2s subclip", "subclip_too_short" in _codes(r4), str(_codes(r4)))

    # playback rate below 1/3 (final way beyond 3x source_used)
    s5 = _global_pass_state()
    c = s5["shot_assets"]["S06"]["clip_meta"]["visual_playlist"]["clips"][0]
    c["source_used_duration_sec"] = 1.0
    c["final_duration_sec"] = 6.0
    c["playback_rate"] = 1.0 / 6.0
    r5 = quality.evaluate_project(s5)
    check("rate below 1/3", "subclip_rate_too_slow" in _codes(r5), str(_codes(r5)))

    # freeze marker
    s6 = _global_pass_state()
    s6["shot_assets"]["S07"]["clip_meta"]["visual_playlist"]["clips"][0]["freeze"] = True
    r6 = quality.evaluate_project(s6)
    check("freeze marker", "footage_freeze_or_loop" in _codes(r6), str(_codes(r6)))

    # coverage short / invalid
    s7 = _global_pass_state()
    s7["shot_assets"]["S08"]["clip_meta"]["visual_playlist"]["coverage_status"] = "short"
    s7["shot_assets"]["S08"]["clip_meta"]["visual_playlist"]["deficit_sec"] = 2.5
    r7 = quality.evaluate_project(s7)
    check("coverage short", "footage_coverage_short" in _codes(r7), str(_codes(r7)))
    s8 = _global_pass_state()
    s8["shot_assets"]["S09"]["clip_meta"]["visual_playlist"]["coverage_status"] = "invalid"
    r8 = quality.evaluate_project(s8)
    check("coverage invalid", "footage_coverage_invalid" in _codes(r8), str(_codes(r8)))

    # single subclip shot
    s9 = _global_pass_state()
    s9["shot_assets"]["S11"]["clip_meta"]["visual_playlist"]["clips"] = [
        _clip("solo", "/clips/solo.mp4", 12.0, 12.0, 0.0)
    ]
    r9 = quality.evaluate_project(s9)
    check("too few subclips", "shot_too_few_subclips" in _codes(r9), str(_codes(r9)))


def test_pacing_defects():
    print("\n[quality] 分镜：20+ 规则 / 过长 fail")
    # too few model shots for a >=240s video
    state = _global_pass_state()
    state["shot_plan"] = state["shot_plan"][:10]
    keep = {s["shot_id"] for s in state["shot_plan"]}
    state["shot_assets"] = {k: v for k, v in state["shot_assets"].items() if k in keep}
    report = quality.evaluate_project(state)
    check("too few model shots (fail for long video)",
          "too_few_model_shots" in _codes(report), str(_codes(report)))
    check("too few is a fail",
          any(i["code"] == "too_few_model_shots" and i["severity"] == "fail" for i in report["issues"]))

    # over-long single shot => fail
    s2 = _global_pass_state()
    s2["shot_plan"][0]["duration_sec"] = 22.0
    s2["shot_assets"]["S01"]["audio_meta"]["duration_sec"] = 22.0
    r2 = quality.evaluate_project(s2)
    check("over-long shot fails", any(
        i["code"] == "shot_too_long" and i["severity"] == "fail" for i in r2["issues"]), str(_codes(r2)))


def test_loudness_defects():
    print("\n[quality] loudness：未归一化 / 超标 / 成品实测")
    state = _global_pass_state()
    state["shot_assets"]["S01"]["audio_meta"]["loudness"] = {
        "status": "normalized", "normalized": True,
        "target": {"integrated_lufs": -16.0, "true_peak_dbtp": -1.5,
                   "tolerance_lu": 1.0, "lra_lu": 11.0},
        "output": {"output_i": -12.0, "output_tp": -0.5},  # too loud, TP over
    }
    r = quality.evaluate_project(state)
    check("loudness out of target", "loudness_out_of_target" in _codes(r), str(_codes(r)))

    s2 = _global_pass_state()
    s2["shot_assets"]["S02"]["audio_meta"]["loudness"] = {
        "status": "failed", "target": {"integrated_lufs": -16.0, "true_peak_dbtp": -1.5,
                                        "tolerance_lu": 1.0, "lra_lu": 11.0},
        "error": {"code": "ffmpeg_failed", "message": "boom"},
    }
    r2 = quality.evaluate_project(s2)
    check("loudness not normalized", "loudness_not_normalized" in _codes(r2), str(_codes(r2)))

    s3 = _global_pass_state()
    s3.pop("final_measurement", None)
    r3 = quality.evaluate_project(s3)
    check("final measurement missing warning",
          "final_measurement_missing" in _codes(r3), str(_codes(r3)))


def test_animation_defects():
    print("\n[quality] 语义动画：短语/时间/类型/门槛/primary")
    narr = "这里有一个真正重要的核心结论。"
    dur = 12.0
    built = build_subtitle_cues(narr, dur)
    # phrase not in narration
    sem_bad_phrase = {
        "primary": {"phrase": "不存在的短语", "type": "key_point",
                    "start_sec": 1.0, "end_sec": 3.0,
                    "structure_id": "s", "motion_id": "m", "visual_id": "v"},
        "anchors": [],
    }
    shot, assets = _make_shot("S01", "seg1", narr, dur, _two_clips("x", 1), semantic=sem_bad_phrase)
    state = {"shot_plan": [shot], "shot_assets": {"S01": assets}, "total_duration_sec": 12.0}
    r = quality.evaluate_project(state)
    check("phrase missing", "animation_phrase_missing" in _codes(r), str(_codes(r)))

    # invalid type
    sem_bad_type = {
        "primary": {"phrase": "核心结论", "type": "made_up_type",
                    "start_sec": 1.0, "end_sec": 3.0,
                    "structure_id": "s", "motion_id": "m", "visual_id": "v"},
        "anchors": [],
    }
    shot2, assets2 = _make_shot("S01", "seg1", narr, dur, _two_clips("y", 2), semantic=sem_bad_type)
    r2 = quality.evaluate_project({"shot_plan": [shot2], "shot_assets": {"S01": assets2}})
    check("invalid type", "animation_type_invalid" in _codes(r2), str(_codes(r2)))

    # invalid time (end<=start)
    sem_bad_time = {
        "primary": {"phrase": "核心结论", "type": "key_point",
                    "start_sec": 0.0, "end_sec": 0.0,
                    "structure_id": "s", "motion_id": "m", "visual_id": "v"},
        "anchors": [],
    }
    shot3, assets3 = _make_shot("S01", "seg1", narr, dur, _two_clips("z", 3), semantic=sem_bad_time)
    r3 = quality.evaluate_project({"shot_plan": [shot3], "shot_assets": {"S01": assets3}})
    check("invalid time", "animation_time_invalid" in _codes(r3), str(_codes(r3)))

    # chart gate failure (single number, not >=2)
    narr_gate = "这个数字大约是百分之五十。"
    built_g = build_subtitle_cues(narr_gate, dur)
    sem_gate = {
        "primary": {"phrase": "百分之五十", "type": "chart",
                    "start_sec": 1.0, "end_sec": 3.0,
                    "structure_id": "s", "motion_id": "m", "visual_id": "v"},
        "anchors": [],
    }
    shot4, assets4 = _make_shot("S01", "seg1", narr_gate, dur, _two_clips("g", 4), semantic=sem_gate)
    r4 = quality.evaluate_project({"shot_plan": [shot4], "shot_assets": {"S01": assets4}})
    check("chart gate failed", "animation_gate_failed" in _codes(r4), str(_codes(r4)))

    # incomplete template
    sem_tpl = {
        "primary": {"phrase": "核心结论", "type": "key_point",
                    "start_sec": 1.0, "end_sec": 3.0, "structure_id": "s"},
        "anchors": [],
    }
    shot5, assets5 = _make_shot("S01", "seg1", narr, dur, _two_clips("t", 5), semantic=sem_tpl)
    r5 = quality.evaluate_project({"shot_plan": [shot5], "shot_assets": {"S01": assets5}})
    check("incomplete template", "animation_template_incomplete" in _codes(r5), str(_codes(r5)))


def test_user_override_respected_but_reported():
    print("\n[quality] 尊重 user_override，但硬阈值仍报告")
    state = _global_pass_state()
    clip = state["shot_assets"]["S04"]["clip_meta"]["visual_playlist"]["clips"][0]
    clip["final_duration_sec"] = 1.2  # sub-2s
    clip["user_override"] = True
    report = quality.evaluate_project(state)
    issue = next((i for i in report["issues"] if i["code"] == "subclip_too_short"), None)
    check("override still reported", issue is not None, str(_codes(report)))
    check("override marked", issue and issue["user_override"] is True)
    check("override not auto-fixable", issue and issue["auto_fixable"] is False)
    check("hard breach stays at least warning", issue and issue["severity"] in ("warning", "fail"))


# ---------------------------------------------------------------------------
# propose_fixes
# ---------------------------------------------------------------------------
def test_propose_fixes_safe_only():
    print("\n[quality] propose_fixes：只提安全修复，不动旁白")
    state = _global_pass_state()
    # inject a subtitle punctuation issue + a loudness issue
    from studio.newvis0907.subtitles import _compact_text, HARD_PUNCT
    state["shot_plan"][1]["subtitle_display_text"] = _compact_text(
        state["shot_plan"][1]["vocal_script"]).rstrip("".join(HARD_PUNCT))
    state["shot_assets"]["S03"]["audio_meta"]["loudness"] = {
        "status": "failed", "target": {"integrated_lufs": -16.0, "true_peak_dbtp": -1.5,
                                        "tolerance_lu": 1.0, "lra_lu": 11.0},
        "error": {"code": "ffmpeg_failed", "message": "boom"},
    }
    report = quality.evaluate_project(state)
    fixes = quality.propose_fixes(state, report)
    kinds = [f.get("action") or f.get("op") for f in fixes]
    check("proposes subtitle rebuild", "rebuild_subtitles" in kinds, str(kinds))
    check("proposes audio normalize", "normalize_audio" in kinds, str(kinds))
    # never a narration rewrite
    check("never rewrites narration",
          all(f.get("op") not in ("set_narration",) and f.get("action") != "rewrite_narration"
              for f in fixes), str(kinds))


def test_propose_fixes_skips_user_override():
    print("\n[quality] propose_fixes：不自动回退 user_override")
    state = _global_pass_state()
    clip = state["shot_assets"]["S04"]["clip_meta"]["visual_playlist"]["clips"][0]
    clip["final_duration_sec"] = 1.2
    clip["user_override"] = True
    report = quality.evaluate_project(state)
    fixes = quality.propose_fixes(state, report)
    check("no auto fix for overridden subclip",
          all(f.get("subclip_id") != clip["subclip_id"] for f in fixes), str(fixes))


# ---------------------------------------------------------------------------
# persistent user overrides (I1): stable issue_id + annotate + exclude/restore
# ---------------------------------------------------------------------------
def test_issue_id_is_stable():
    print("\n[quality] issue_id 稳定")
    state = _global_pass_state()
    r1 = quality.evaluate_project(state)
    r2 = quality.evaluate_project(_global_pass_state())
    # every issue carries a stable id derived from (code, shot, subclip)
    for i in r1["issues"]:
        check("issue_id present", bool(i.get("issue_id")), str(i))
        check("issue_id matches formula",
              i["issue_id"] == quality.issue_id_for(i["code"], i["shot_id"], i["subclip_id"]))
    check("ids stable across runs",
          {i["issue_id"] for i in r1["issues"]} == {i["issue_id"] for i in r2["issues"]})


def test_normalize_overrides_forms():
    print("\n[quality] normalize_overrides：list/dict/auto")
    key = quality.issue_id_for("x", "S1", "")
    m1 = quality.normalize_overrides([{"code": "x", "shot_id": "S1", "action": "ignore"}])
    check("list ignore kept", key in m1, str(m1))
    m2 = quality.normalize_overrides([{"issue_id": key, "action": "auto"}])
    check("auto restore dropped", key not in m2, str(m2))
    m3 = quality.normalize_overrides({key: {"action": "ignore", "code": "x", "shot_id": "S1"}})
    check("dict form kept", key in m3, str(m3))


def test_soft_override_annotates_softens_and_excludes():
    print("\n[quality] 软项 ignore：标记/降级/退出 auto-fix，restore 回归")
    state = _global_pass_state()
    state["shot_assets"]["S01"]["audio_meta"].pop("loudness", None)  # soft warning
    base = quality.evaluate_project(state)
    lm = next(i for i in base["issues"] if i["code"] == "loudness_missing" and i["shot_id"] == "S01")
    check("soft issue starts as warning", lm["severity"] == "warning")
    check("soft issue auto-fixable", lm["auto_fixable"] is True)

    ov = [{"code": "loudness_missing", "shot_id": "S01", "action": "ignore"}]
    rep = quality.evaluate_project(state, overrides=ov)
    lm2 = next(i for i in rep["issues"] if i["code"] == "loudness_missing" and i["shot_id"] == "S01")
    check("override annotated", lm2["user_override"] is True)
    check("soft override downgraded to info", lm2["severity"] == "info", str(lm2))
    check("override not auto-fixable", lm2["auto_fixable"] is False)
    fixes = quality.propose_fixes(state, rep)
    check("override excluded from proposals",
          not any(f.get("action") == "normalize_audio" and f.get("shot_id") == "S01" for f in fixes),
          str(fixes))

    rep3 = quality.evaluate_project(
        state, overrides=[{"code": "loudness_missing", "shot_id": "S01", "action": "auto"}])
    lm3 = next(i for i in rep3["issues"] if i["code"] == "loudness_missing" and i["shot_id"] == "S01")
    check("restore removes override", lm3["user_override"] is False)
    fixes3 = quality.propose_fixes(state, rep3)
    check("restore re-adds to proposals",
          any(f.get("action") == "normalize_audio" and f.get("shot_id") == "S01" for f in fixes3),
          str(fixes3))


def test_hard_override_keeps_severity_but_marks():
    print("\n[quality] 硬阈值 ignore：保留原 severity + 标记，仍进入 fail")
    state = _global_pass_state()
    clip = state["shot_assets"]["S04"]["clip_meta"]["visual_playlist"]["clips"][0]
    clip["final_duration_sec"] = 1.2  # subclip_too_short => hard FAIL
    ov = [{"code": "subclip_too_short", "shot_id": "S04",
           "subclip_id": clip["subclip_id"], "action": "ignore"}]
    rep = quality.evaluate_project(state, overrides=ov)
    issue = next(i for i in rep["issues"] if i["code"] == "subclip_too_short")
    check("hard override keeps fail severity", issue["severity"] == "fail", str(issue))
    check("hard override marked", issue["user_override"] is True)
    check("hard override not auto-fixable", issue["auto_fixable"] is False)
    check("hard override still fails project", rep["status"] == "fail", rep["status"])


def test_animation_uniform_zero_dedup():
    print("\n[quality] M4：同一 anchor 不同时报 uniform_zero 与 time_mismatch")
    narr = "先做一个铺垫说明，怎样让AI成为自己的工具。"
    dur = 12.0
    built = build_subtitle_cues(narr, dur)
    sem = {
        "primary": {"phrase": "怎样让AI成为自己的工具", "type": "key_point",
                    "start_sec": 0.0, "end_sec": 3.0,
                    "structure_id": "s", "motion_id": "m", "visual_id": "v"},
        "anchors": [],
    }
    shot, assets = _make_shot("S01", "seg1", narr, dur, _two_clips("uz", 1), semantic=sem)
    assets["subtitle_cues"] = built["cues"]
    rep = quality.evaluate_project({"shot_plan": [shot], "shot_assets": {"S01": assets}})
    codes = [i["code"] for i in rep["issues"]]
    check("uniform_zero reported", "animation_time_uniform_zero" in codes, str(codes))
    check("time_mismatch not double-counted",
          "animation_time_mismatch" not in codes, str(codes))


def main() -> int:
    tests = [
        test_score_weights_stable_and_ordered,
        test_status_mapping,
        test_global_fixture_passes,
        test_subtitle_defects,
        test_footage_duplicate_and_rate_and_freeze,
        test_pacing_defects,
        test_loudness_defects,
        test_animation_defects,
        test_user_override_respected_but_reported,
        test_propose_fixes_safe_only,
        test_propose_fixes_skips_user_override,
        test_issue_id_is_stable,
        test_normalize_overrides_forms,
        test_soft_override_annotates_softens_and_excludes,
        test_hard_override_keeps_severity_but_marks,
        test_animation_uniform_zero_dedup,
    ]
    try:
        for t in tests:
            t()
    except AssertionError as exc:
        print(f"\n失败: {exc}")
        return 1
    print(f"\n全部通过: {len(PASS)} 项断言")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
