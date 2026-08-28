"""Offline checks for AICut Studio — no GPU, no model weights, no ffmpeg.

Covers the parts that are easy to get wrong and expensive to discover at
runtime: config projection, the track projection the UI renders, edit-operation
validation and atomicity, chat-reply parsing plus memory compression, and the
HTTP surface the frontend actually calls.

Run:  python tests/test_studio_offline.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

STUDIO_DIR = Path(__file__).resolve().parents[1]
if str(STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIO_DIR))

os.environ.setdefault("AICUT_STUDIO_SKIP_MODEL", "1")

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(f"{name}{(' — ' + detail) if detail and not cond else ''}")
    print(("  ok   " if cond else "  FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# a synthetic finished project: 4 shots, 2 spotlight, one shot missing audio
# ---------------------------------------------------------------------------
def make_state() -> dict:
    shots = []
    for i in range(1, 5):
        shots.append({
            "shot_id": f"S{i:02d}",
            "segment_id": f"SEG{(i + 1) // 2:02d}",
            "vocal_script": f"第{i}句口播文本，用于测试定位与改写。",
            "visual_query": f"画面{i}",
            "duration_sec": 4.0,
            "is_spotlight": i in (2, 4),
        })

    assets = {}
    for i, shot in enumerate(shots, start=1):
        sid = shot["shot_id"]
        entry = {
            "audio_path": f"/tmp/aicut/{sid}.wav",
            "audio_meta": {"duration_sec": 4.0 + i * 0.5},
            "clip_path": f"/tmp/aicut/{sid}.mp4",
            "clip_source": "retrieval",
            "rendered_path": f"/tmp/aicut/{sid}_rendered.mp4",
        }
        if shot["is_spotlight"]:
            entry["decoration"] = {
                "structure_id": "S_center_stack",
                "motion_id": "M_slide_in",
                "visual_id": "V_dark_grid",
                "slots": {"title": f"标题{i}", "subtitle": f"副标题{i}"},
            }
        if sid == "S03":
            # a half-produced shot, so pending lanes are exercised
            entry.pop("rendered_path")
            entry["audio_path"] = ""
        assets[sid] = entry

    return {
        "project_id": "aicut_test_0001",
        "input_text": "这是一段用于离线测试的口播稿。" * 3,
        "segments": [
            {"segment_id": "SEG01", "title": "开场", "summary": "引入话题"},
            {"segment_id": "SEG02", "title": "展开", "summary": "给出论据"},
        ],
        "shot_plan": shots,
        "shot_assets": assets,
        "bgm_tracks": [{"track_id": "BGM01", "name": "calm_tech", "path": "/tmp/aicut/bgm.mp3"}],
        "bgm_volume": 0.45,
        "music_style": "calm tech",
        "final_video_path": "/tmp/aicut/final.mp4",
        "stitch_result": {
            "status": "success", "concat_status": "ok",
            "ready_shot_count": 3, "shot_count": 4,
        },
        "errors": [],
    }


# ---------------------------------------------------------------------------
def test_config() -> None:
    section("config")
    from studio import config as studio_config

    cfg = studio_config.load()
    check("studio_config.yaml 可加载", cfg is not None)
    check("无致命配置错误", not cfg.errors(),
          "; ".join(f"{i.where}: {i.message}" for i in cfg.errors()))

    for key in ("agent_mode", "model", "agent", "tool_switches"):
        check(f"infer_config 含 {key}", key in cfg.infer_config)
    for key in ("paths", "services", "llm"):
        check(f"pipeline_config 含 {key}", key in cfg.pipeline_config)

    # node_severity_rank reads these; when the block is missing it silently
    # falls back to allow_heuristic_fallback=False and every shot scores 0.0.
    sev = cfg.pipeline_config.get("stage3_trajectory", {}).get("severity_rank", {})
    check("pipeline_config 含 severity_rank", bool(sev))
    check("severity_rank 允许启发式兜底",
          sev.get("allow_heuristic_fallback") is True,
          f"allow_heuristic_fallback={sev.get('allow_heuristic_fallback')!r}")
    check("severity_rank top_ratio 合法",
          0.0 < float(sev.get("top_ratio", 0)) <= 1.0, str(sev.get("top_ratio")))
    check("severity_rank min_shots_to_process >= 1",
          int(sev.get("min_shots_to_process", 0)) >= 1, str(sev.get("min_shots_to_process")))

    check("active_profile 非空", bool(cfg.active_profile))
    check("system_prompt 已载入", len(cfg.system_prompt) > 200,
          f"{len(cfg.system_prompt)} chars")
    check("media_roots 非空", len(cfg.media_roots) > 0)

    autonomy = cfg.autonomy
    check("默认完全自主", bool(autonomy.get("fully_autonomous")))

    if cfg.warnings():
        print("  (warnings)")
        for issue in cfg.warnings():
            print(f"    - {issue.where}: {issue.message}")


def test_concat_codec_guard() -> None:
    """Stream-copy concat must never ship a codec browsers cannot decode.

    Regression: with fast_concat_copy on, mpeg4 (mp4v) shots at mixed frame
    rates were copied straight through; ffmpeg reported success but the
    browser played audio over a black frame.
    """
    section("concat codec guard")
    from studio import config as studio_config

    studio_config.install_import_paths()
    from pipeline.tools import stitch_video as sv

    check("mpeg4 不在浏览器可解编码白名单",
          "mpeg4" not in sv._BROWSER_SAFE_VIDEO_CODECS)
    check("h264 在白名单", "h264" in sv._BROWSER_SAFE_VIDEO_CODECS)

    uniform_h264 = {"codec": "h264", "width": 1920, "height": 1080, "fps": 24.0}
    mixed_fps = {"codec": "h264", "width": 1920, "height": 1080, "fps": 14.0}
    mpeg4 = {"codec": "mpeg4", "width": 1920, "height": 1080, "fps": 24.0}

    probes: dict[str, dict] = {}
    original = sv._probe_video_params
    sv._probe_video_params = lambda path: probes.get(path, {})
    try:
        probes = {"a.mp4": uniform_h264, "b.mp4": dict(uniform_h264)}
        ok, reason = sv._concat_copy_is_safe(["a.mp4", "b.mp4"])
        check("同参数 h264 允许 stream copy", ok, reason)

        probes = {"a.mp4": uniform_h264, "b.mp4": mpeg4}
        ok, reason = sv._concat_copy_is_safe(["a.mp4", "b.mp4"])
        check("含 mpeg4 时拒绝 stream copy", not ok, reason)
        check("拒绝原因指向编码", "codec_not_browser_safe" in reason, reason)

        probes = {"a.mp4": uniform_h264, "b.mp4": mixed_fps}
        ok, reason = sv._concat_copy_is_safe(["a.mp4", "b.mp4"])
        check("帧率不一致时拒绝 stream copy", not ok, reason)

        probes = {"a.mp4": uniform_h264, "b.mp4": {}}
        ok, reason = sv._concat_copy_is_safe(["a.mp4", "b.mp4"])
        check("探测失败时拒绝 stream copy", not ok, reason)

        ok, reason = sv._concat_copy_is_safe([])
        check("空输入拒绝 stream copy", not ok, reason)
    finally:
        sv._probe_video_params = original


def test_tracks() -> None:
    section("tracks projection")
    from studio import tracks

    state = make_state()
    snap = tracks.build_snapshot(state, ["segments", "shots", "narration", "subtitle",
                                         "footage", "spotlight", "bgm", "render"])
    clips = snap["clips"]

    check("segments 轨 2 条", len(clips["segments"]) == 2, str(len(clips["segments"])))
    check("shots 轨 4 条", len(clips["shots"]) == 4, str(len(clips["shots"])))
    check("spotlight 轨 2 条", len(clips["spotlight"]) == 2, str(len(clips["spotlight"])))
    check("bgm 轨 1 条", len(clips["bgm"]) == 1)

    render_ready = [c for c in clips["render"] if c.get("ready")]
    check("render 轨 3/4 就绪", len(render_ready) == 3, f"{len(render_ready)}/4")

    narration_ready = [c for c in clips["narration"] if c.get("ready")]
    check("narration 轨 3/4 就绪（S03 无音频）", len(narration_ready) == 3,
          f"{len(narration_ready)}/4")

    starts = [c["start"] for c in clips["shots"]]
    check("镜头时间单调递增", starts == sorted(starts))
    check("总时长 = 各镜头时长之和",
          abs(snap["total_duration"] - clips["shots"][-1]["end"]) < 0.05,
          f'{snap["total_duration"]} vs {clips["shots"][-1]["end"]}')

    check("成片存在时总进度 100", snap["overall_pct"] == 100, str(snap["overall_pct"]))

    # BGM should be clamped to video total_duration
    state_bgm = dict(state, bgm_tracks=[
        {"track_id": "BGM01", "name": "long_song", "duration_sec": 300, "path": "/bgm.mp3"}
    ])
    snap_bgm = tracks.build_snapshot(state_bgm)
    bgm_clips = snap_bgm["clips"]["bgm"]
    check("配乐轨截断到视频总时长",
          bgm_clips[0]["end"] <= snap_bgm["total_duration"],
          f'bgm_end={bgm_clips[0]["end"]}, total={snap_bgm["total_duration"]}')
    check("配乐轨 end == total_duration",
          abs(bgm_clips[0]["end"] - snap_bgm["total_duration"]) < 0.01,
          f'{bgm_clips[0]["end"]} != {snap_bgm["total_duration"]}')
    check("配乐轨 duration 相应调整",
          abs(bgm_clips[0]["duration"] - snap_bgm["total_duration"]) < 0.01)

    no_final = dict(state, final_video_path="")
    mid = tracks.build_snapshot(no_final)
    check("未收尾时总进度 < 100", mid["overall_pct"] < 100, str(mid["overall_pct"]))

    empty = tracks.build_snapshot({})
    check("空状态不报错", empty["overall_pct"] == 0 and empty["total_duration"] == 0)

    for name, items in clips.items():
        if name == "total_duration":
            continue
        bad = [c for c in items if not isinstance(c, dict) or "start" not in c]
        check(f"{name} 轨字段完整", not bad, str(bad[:1]))

    check("stats 覆盖所有轨", set(snap["stats"]) == set(snap["track_order"]))
    check("snapshot 可 JSON 序列化", bool(json.dumps(snap)))


def test_project_and_editor(tmp: Path) -> tuple:
    section("project + editor")
    from studio import editor, project as project_mod

    pdir = tmp / "projects" / "aicut_test_0001"
    pdir.mkdir(parents=True)
    (pdir / "final_state.json").write_text(
        json.dumps(make_state(), ensure_ascii=False), encoding="utf-8")

    proj = project_mod.load(pdir, {})
    check("工程可载入", proj.project_id == "aicut_test_0001")
    check("镜头数 4", len(proj.shot_plan) == 4)

    digest = proj.digest()
    check("digest 含 shots", len(digest["shots"]) == 4)
    check("digest 标出重点镜头",
          sum(1 for s in digest["shots"] if s.get("is_spotlight")) == 2)
    check("digest 不泄露文件路径", "/tmp/aicut" not in json.dumps(digest, ensure_ascii=False))

    found = proj.find_shots("第3句", limit=3)
    check("find_shots 有返回", len(found) > 0)

    flags: dict = {}

    # -- narration edit invalidates downstream lanes --
    results = editor.apply_ops(proj, [
        {"op": "set_narration", "shot_id": "S01", "text": "改写后的第一句，冲击力更强。"},
    ], flags)
    check("改旁白成功", all(r.ok for r in results), str([r.message for r in results]))
    check("改旁白标脏 S01", "S01" in proj.dirty)
    dirty_stages = proj.dirty.get("S01", set())
    check("改旁白连带重配音", "tts" in dirty_stages, str(dirty_stages))
    check("改旁白连带重渲染", "render" in dirty_stages, str(dirty_stages))

    # -- subtitle-only edit must not re-run TTS --
    editor.apply_ops(proj, [{"op": "set_subtitle", "shot_id": "S02", "text": "只改字幕"}], flags)
    check("改字幕不重配音", "tts" not in proj.dirty.get("S02", set()),
          str(proj.dirty.get("S02")))
    check("改字幕要重渲染", "render" in proj.dirty.get("S02", set()))

    # -- regeneration plan ordering --
    steps = editor.plan_regeneration(proj)
    tools = [s["tool"] for s in steps]
    check("重生成计划非空", bool(tools), str(tools))
    if "Tool_GenerateNarration" in tools and "Tool_RenderShots" in tools:
        check("配音先于渲染",
              tools.index("Tool_GenerateNarration") < tools.index("Tool_RenderShots"),
              str(tools))
    check("重生成计划不含拼接", "Tool_StitchVideo" not in tools, str(tools))
    check("stitch_step 单独提供", editor.stitch_step(proj)["tool"] == "Tool_StitchVideo")

    # -- validation --
    for bad, why in [
        ({"op": "set_narration", "shot_id": "S99", "text": "x"}, "未知 shot_id"),
        ({"op": "set_narration", "shot_id": "S01"}, "缺 text"),
        ({"op": "no_such_op", "shot_id": "S01"}, "未知 op"),
        ({"op": "replace_footage", "shot_id": "S01"}, "缺 query/clip_path"),
        ({"op": "swap_template", "shot_id": "S02"}, "缺模板 id"),
        ({"op": "set_narration", "shot_id": "S01", "text": "x", "bogus": 1}, "未知参数"),
        ({"op": "set_spotlight_slots", "shot_id": "S01", "slots": {"t": "x"}}, "非重点镜头"),
    ]:
        try:
            editor.apply_ops(proj, [bad], flags)
            ok = False
        except editor.EditError:
            ok = True
        check(f"拒绝非法操作: {why}", ok)

    # -- atomicity: a bad op in a batch must not leave the first one applied --
    before = proj.shot("S04")["vocal_script"]
    try:
        editor.apply_ops(proj, [
            {"op": "set_narration", "shot_id": "S04", "text": "这条不该生效"},
            {"op": "set_narration", "shot_id": "S99", "text": "非法"},
        ], flags)
    except editor.EditError:
        pass
    check("批次失败时不留半改状态",
          proj.shot("S04")["vocal_script"] == before,
          proj.shot("S04")["vocal_script"])

    # -- config flags gate operations --
    try:
        editor.apply_ops(proj, [{"op": "delete_shot", "shot_id": "S04"}],
                         {"delete_shot": False})
        gated = False
    except editor.EditError:
        gated = True
    check("配置关闭的操作被拒绝", gated)

    # -- dry run leaves state untouched --
    snapshot = json.dumps(proj.state, ensure_ascii=False, sort_keys=True)
    editor.precheck_ops(proj, [{"op": "delete_shot", "shot_id": "S04"}], flags)
    check("precheck 不改状态",
          json.dumps(proj.state, ensure_ascii=False, sort_keys=True) == snapshot)

    # -- high-risk gating --
    risky = editor.high_risk_ops(
        [{"op": "delete_shot"}, {"op": "set_subtitle"}],
        ["delete_shot", "reorder_shots"])
    check("高风险识别 delete_shot", risky == ["delete_shot"], str(risky))

    # -- reorder + delete + bgm --
    editor.apply_ops(proj, [{"op": "set_bgm", "query": "更安静的配乐", "volume": 0.3}], flags)
    check("配乐改动标脏", proj.bgm_dirty)

    order = [s["shot_id"] for s in proj.shot_plan]
    editor.apply_ops(proj, [{"op": "reorder_shots", "shot_ids": list(reversed(order))}], flags)
    check("镜头顺序已反转",
          [s["shot_id"] for s in proj.shot_plan] == list(reversed(order)))

    editor.apply_ops(proj, [{"op": "delete_shot", "shot_id": "S04"}], flags)
    check("删除镜头生效", "S04" not in proj.shot_ids())
    check("删除后资产同步清理", "S04" not in proj.shot_assets)

    # -- persistence + versioning --
    proj.save("manual_edit", max_versions=3)
    versions = proj.list_versions()
    check("保存后产生版本", len(versions) >= 1, str(len(versions)))

    reloaded = project_mod.load(pdir, {})
    check("保存内容可回读", "S04" not in reloaded.shot_ids())

    for i in range(5):
        proj.save(f"manual_edit_{i}", max_versions=3)
    check("版本数量受上限约束", len(proj.list_versions()) <= 3,
          str(len(proj.list_versions())))

    latest = proj.list_versions()[0]["name"]
    proj.restore_version(latest)
    check("版本可回滚", True)

    discovered = project_mod.discover(tmp / "projects")
    check("discover 能发现工程", any(d["project_id"] == "aicut_test_0001" for d in discovered),
          str([d["project_id"] for d in discovered]))

    test_project_delete(tmp)
    return proj, pdir


def test_project_delete(tmp: Path) -> None:
    """Deleting a project removes rendered video, so the guards matter more
    than the happy path."""
    section("project delete")
    from studio import project as project_mod

    root = tmp / "projects"
    roots = [root, tmp]

    victim = root / "aicut_delete_me"
    victim.mkdir(parents=True)
    (victim / "final_state.json").write_text(
        json.dumps(make_state(), ensure_ascii=False), encoding="utf-8")
    (victim / "final.mp4").write_bytes(b"x" * 2048)
    (victim / "versions").mkdir()
    (victim / "versions" / "v1.json").write_text("{}", encoding="utf-8")

    result = project_mod.delete(victim, roots)
    check("删除返回释放空间", result["freed_bytes"] > 2000, str(result["freed_bytes"]))
    check("目录已移除", not victim.exists())
    check("删除返回 project_id", result["project_id"] == "aicut_test_0001",
          result["project_id"])

    # -- refuses anything that is not a project --
    plain = root / "not_a_project"
    plain.mkdir(parents=True)
    (plain / "notes.txt").write_text("hi", encoding="utf-8")
    for target, why in [
        (plain, "缺少 final_state.json"),
        (root, "工作区根目录"),
        (tmp.parent, "工作区之外"),
        (root / "does_not_exist", "不存在"),
    ]:
        try:
            project_mod.delete(target, roots)
            ok = False
        except project_mod.ProjectError:
            ok = True
        check(f"拒绝删除: {why}", ok)
    check("被拒绝的目录仍在", plain.is_dir())

    # -- a symlink inside the workspace must not escape it --
    outside = tmp / "outside_target"
    outside.mkdir()
    (outside / "final_state.json").write_text("{}", encoding="utf-8")
    link = root / "sneaky"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        link = None
    if link is not None:
        # resolves to <tmp>/outside_target, which is directly under an allowed
        # root, so this one is legitimately allowed — the point is that the
        # check runs on the resolved path, not the link path
        resolved_ok = True
        try:
            project_mod.delete(link, [root])   # only <root> allowed → must refuse
            resolved_ok = False
        except project_mod.ProjectError:
            pass
        check("软链接按解析后的真实路径判定", resolved_ok)
        check("软链接指向的目录未被删除", outside.is_dir())
    shutil.rmtree(outside, ignore_errors=True)


def test_chat_agent(proj) -> None:
    section("chat agent")
    import asyncio

    from studio import chat_agent, chat_prompts

    # -- prompt sanity: the contract must actually name the ops --
    from studio import editor
    missing = [op for op in editor.ALL_OPS if op not in chat_prompts.SYSTEM_PROMPT]
    check("系统提示覆盖全部操作", not missing, str(missing))
    check("提示要求只输出 JSON", "JSON" in chat_prompts.SYSTEM_PROMPT)
    check("few-shot 示例存在", len(chat_prompts.FEW_SHOT_EXAMPLES) >= 4,
          str(len(chat_prompts.FEW_SHOT_EXAMPLES)))
    for i, ex in enumerate(chat_prompts.FEW_SHOT_EXAMPLES):
        try:
            obj = json.loads(ex["assistant"])
            ok = isinstance(obj, dict) and obj.get("intent") in chat_agent.VALID_INTENTS
        except json.JSONDecodeError:
            ok = False
        check(f"示例 {i} 输出合法", ok, ex["assistant"][:60])

    # -- reply parsing --
    cases = [
        ('{"intent":"edit","reply":"好","ops":[{"op":"set_subtitle","shot_id":"S01","text":"x"}]}',
         "edit", 1),
        ('```json\n{"intent":"clarify","reply":"哪一句？","ops":[]}\n```', "clarify", 0),
        ('思考中...\n{"intent":"question","reply":"共 4 个镜头","ops":[]}\n以上。', "question", 0),
        ('{"intent":"edit","reply":"好","ops":[]}', "", 0),          # edit without ops
        ('{"intent":"nonsense","reply":"x"}', "", 0),                # bad intent
        ('not json at all', "", 0),
        ('{"intent":"question","reply":"x","ops":[{"op":"delete_shot","shot_id":"S01"}]}',
         "question", 0),                                             # ops dropped
    ]
    for raw, want_intent, want_ops in cases:
        r = chat_agent.parse_reply(raw)
        ok = r.intent == want_intent and len(r.ops) == want_ops
        check(f"解析: {raw[:38]}…", ok, f"got intent={r.intent!r} ops={len(r.ops)}")

    # -- memory compression --
    mem = chat_agent.Memory(max_history_tokens=200, keep_recent_turns=2)
    for i in range(12):
        mem.add("user", f"第{i}轮的请求内容，需要占用一些上下文空间。" * 3)
        mem.add("assistant", f"第{i}轮的回复内容。" * 3)
    check("超预算触发压缩", mem.needs_compression(), str(mem.token_estimate()))
    before = mem.token_estimate()
    mem.apply_compression("摘要：用户改了开头与字幕。")
    check("压缩后 token 下降", mem.token_estimate() < before,
          f"{before} -> {mem.token_estimate()}")
    check("压缩后保留最近轮次", len(mem.turns) == 2, str(len(mem.turns)))
    check("压缩后不再需要压缩", not mem.needs_compression())
    check("摘要已保留", "摘要" in mem.summary)

    # -- full turn against a stubbed model --
    calls: list[tuple[str, str]] = []

    def fake_generate(system, user, **kw):
        calls.append((system, user))
        return json.dumps({
            "intent": "edit",
            "reply": "已把第一句改得更有冲击力。",
            "targets": ["S01"],
            "confidence": 0.9,
            "ops": [{"op": "set_subtitle", "shot_id": proj.shot_ids()[0], "text": "新的字幕"}],
        }, ensure_ascii=False)

    agent = chat_agent.ChatEditAgent(
        generate=fake_generate,
        chat_config={"context": {"max_history_tokens": 4000, "keep_recent_turns": 4},
                     "safety": {"require_confirm_for": ["delete_shot"], "max_ops_per_turn": 5}},
        editor_flags={},
    )
    payload = asyncio.run(agent.handle(proj, "开头那句太平淡了"))
    check("对话轮返回 edit", payload["intent"] == "edit", str(payload["intent"]))
    check("对话轮产出 ops", len(payload["ops"]) == 1)
    check("低风险不需确认", not payload["needs_confirm"])
    check("对话轮未自行落盘", payload["applied"] is False)
    check("记忆记录了本轮", len(payload["memory"]["turns"]) == 2)
    check("提示注入了工程摘要", "shot_id" in calls[0][1])

    # -- schema retry --
    attempts = {"n": 0}

    def flaky(system, user, **kw):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return "抱歉，我先说点别的。"
        return json.dumps({"intent": "clarify", "reply": "想改哪一句？", "ops": []},
                          ensure_ascii=False)

    agent2 = chat_agent.ChatEditAgent(generate=flaky, chat_config={}, editor_flags={})
    payload2 = asyncio.run(agent2.handle(proj, "改一下"))
    check("非法输出会重问", attempts["n"] == 2, str(attempts["n"]))
    check("重问后恢复正常", payload2["intent"] == "clarify")

    # -- persistent garbage degrades gracefully --
    agent3 = chat_agent.ChatEditAgent(
        generate=lambda s, u, **kw: "完全不是 JSON",
        chat_config={"safety": {"max_parse_retries": 1}}, editor_flags={})
    payload3 = asyncio.run(agent3.handle(proj, "改一下"))
    check("持续非法输出转为澄清", payload3["intent"] == "clarify")
    check("持续非法输出不产生 ops", not payload3["ops"])

    # -- ops rejected by the real project surface as clarify, not 500 --
    agent4 = chat_agent.ChatEditAgent(
        generate=lambda s, u, **kw: json.dumps({
            "intent": "edit", "reply": "改这里",
            "ops": [{"op": "set_narration", "shot_id": "S_NOT_EXIST", "text": "x"}],
        }, ensure_ascii=False),
        chat_config={}, editor_flags={})
    payload4 = asyncio.run(agent4.handle(proj, "改那个不存在的镜头"))
    check("校验不过转为澄清", payload4["intent"] == "clarify", str(payload4["intent"]))
    check("校验不过不返回 ops", not payload4["ops"])

    # -- disabled ops are hidden from the prompt --
    agent5 = chat_agent.ChatEditAgent(
        generate=lambda s, u, **kw: "{}", chat_config={},
        editor_flags={"delete_shot": False})
    check("被禁操作不进可用列表", "delete_shot" not in agent5.available_ops(),
          str(agent5.available_ops()))

    # -- per-turn op ceiling --
    many = [{"op": "set_subtitle", "shot_id": sid, "text": "t"} for sid in proj.shot_ids()]
    agent6 = chat_agent.ChatEditAgent(
        generate=lambda s, u, **kw: json.dumps(
            {"intent": "edit", "reply": "批量改", "ops": many}, ensure_ascii=False),
        chat_config={"safety": {"max_ops_per_turn": 2}}, editor_flags={})
    payload6 = asyncio.run(agent6.handle(proj, "全部字幕都改"))
    check("单轮操作数受限", len(payload6["ops"]) == 2, str(len(payload6["ops"])))
    check("超出部分有提示", "还有" in payload6["reply"], payload6["reply"])


def test_orchestrator_overrides() -> None:
    section("autonomous orchestrator")
    try:
        from studio.orchestrator import AutonomousOrchestrator
    except ImportError as exc:
        check("AutonomousOrchestrator 可导入", False, str(exc))
        return

    check("AutonomousOrchestrator 可导入", True)
    check("提供 run_autonomous", hasattr(AutonomousOrchestrator, "run_autonomous"))

    # The point of the subclass: no forced tool rewriting.
    for name in ("_next_missing_agent_tool", "_force_tool_for_stage"):
        check(f"已覆盖扶正逻辑 {name}",
              name in AutonomousOrchestrator.__dict__, "未在子类中覆盖")

    import inspect
    src = inspect.getsource(AutonomousOrchestrator.run_autonomous)
    check("自主循环不调用 _force_tool_for_stage", "_force_tool_for_stage(" not in src)
    check("自主循环把错误回灌为 observation", "_set_feedback" in src)

    test_error_observation()
    test_failure_flags()
    test_env_failure_stops_fast()
    test_stage4_repair_budget()
    test_total_tool_call_cap()


def test_error_observation() -> None:
    """A failing tool must reach the model with its reason attached.

    The parent builds ``summary`` from a key whitelist that omits ``error``,
    which is what let a dead TTS service look like an unexplained failure and
    turned it into a retry loop.
    """
    section("failed tool observation")
    from studio.orchestrator import AutonomousOrchestrator

    orch = object.__new__(AutonomousOrchestrator)
    reason = ("RuntimeError: Qwen3-TTS 服务不可达 (http://127.0.0.1:7865/health "
              "探测失败: URLError: Connection refused)")

    obs = orch._make_agent_observation(
        "Tool_BatchVisualAudio", {"status": "error", "error": reason, "_elapsed_sec": 0.0})
    check("失败观测 status=error", obs.get("status") == "error", str(obs.get("status")))
    blob = json.dumps(obs, ensure_ascii=False)
    check("失败原因传达给模型", "Qwen3-TTS" in blob, blob[:300])
    check("原因落在 summary.message（训练时的同分布字段）",
          "Qwen3-TTS" in str(obs.get("summary", {}).get("message", "")))
    check("附带不要重试的提示", "重复同样的调用" in str(obs.get("hint", "")))

    # A parent-built observation for the same failure must NOT contain it —
    # this is the bug being fixed, and pins it so it cannot silently return.
    from server.infer_graph import InferenceOrchestrator
    bare = InferenceOrchestrator._make_agent_observation(
        orch, "Tool_BatchVisualAudio", {"status": "error", "error": reason})
    check("（回归）父类实现确实会丢掉原因",
          "Qwen3-TTS" not in json.dumps(bare, ensure_ascii=False),
          "父类已修好，子类覆盖可以去掉了")

    ok = orch._make_agent_observation(
        "Tool_BatchVisualAudio",
        {"status": "success", "tts_success_count": 13, "visual_success_count": 13})
    check("成功观测不加 hint", "hint" not in ok, str(ok.get("hint")))
    check("成功观测保留计数", ok["summary"].get("tts_success_count") == 13, str(ok["summary"]))


def test_failure_flags() -> None:
    """A crashed tool must not leave its stage marked complete."""
    section("completion flags on failure")
    from server.infer_graph import InferenceOrchestrator
    from studio.orchestrator import AutonomousOrchestrator

    orch = object.__new__(AutonomousOrchestrator)
    orch.visual_audio_done = False
    orch.decorated_done = False
    orch.stitched_done = False

    before = orch._completion_flags()
    check("能快照完成标记", before.get("visual_audio_done") is False, str(before))

    # The parent flips the flag purely on tool name, ignoring status.
    InferenceOrchestrator.update_state_from_respond(
        orch, "Tool_BatchVisualAudio", {"status": "error", "error": "boom"})
    check("（回归）父类在失败时也置位完成标记", orch.visual_audio_done is True,
          "父类已修好，子类的撤销逻辑可以去掉了")

    reverted = orch._restore_completion_flags(before)
    check("失败后撤销误置的标记", orch.visual_audio_done is False)
    check("报告被撤销的标记名", reverted == ["visual_audio_done"], str(reverted))

    # An already-done stage must not be reverted by a later unrelated failure.
    orch.decorated_done = True
    snapshot = orch._completion_flags()
    orch.visual_audio_done = True
    check("只撤销本次翻转的标记",
          orch._restore_completion_flags(snapshot) == ["visual_audio_done"])
    check("先前已完成的阶段不受影响", orch.decorated_done is True)


def _bare_orchestrator(model, dispatch, events: list[dict]):
    """An `AutonomousOrchestrator` with only the agent loop wired up.

    Built with `object.__new__` so the loop can be driven without a model, a
    pipeline config or a GPU. Every cost bound is read from the module
    constants rather than hard-coded here, so adding a new ceiling cannot
    silently leave these tests running against a stale default.
    """
    import types

    from studio import orchestrator as orch_mod
    from server.infer_graph import InferenceOrchestrator

    orch = object.__new__(orch_mod.AutonomousOrchestrator)
    orch.model = model
    orch.system_prompt = ""
    orch.pipeline_config = {}
    orch.agent_mode = "long_horizon"
    orch.fully_autonomous = True
    orch._is_multi_model = False
    orch._progress_fn = None
    orch.last_observation = {}
    orch.agent_history = []
    orch.max_agent_steps = 30
    orch.max_same_tool = 10
    orch.max_consecutive_parse_failures = orch_mod.DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES
    orch.max_consecutive_blocked = orch_mod.DEFAULT_MAX_CONSECUTIVE_BLOCKED
    orch.max_identical_failures = orch_mod.DEFAULT_MAX_IDENTICAL_FAILURES
    orch.max_identical_calls = orch_mod.DEFAULT_MAX_IDENTICAL_CALLS
    orch.max_stage4_repairs = orch_mod.DEFAULT_MAX_STAGE4_REPAIRS
    orch.max_stage4_attempts = orch_mod.DEFAULT_MAX_STAGE4_ATTEMPTS
    orch.max_tool_calls = orch_mod.DEFAULT_MAX_TOOL_CALLS
    orch.visual_audio_done = False
    orch.decorated_done = False
    orch.stitched_done = False
    orch.quality_checked = False
    orch.repaired_done = False

    async def _broadcast(run_id, ev):
        events.append(ev)

    async def _wait(run_id):
        return None

    orch.broadcast = _broadcast
    orch._wait_if_paused = _wait
    orch._dispatch_tool = dispatch
    orch.build_agentic_state_json = lambda script, state: "{}"
    orch._phase_for_tool = lambda tool: "asset_and_compose"
    orch._thinking_event = lambda **kw: {"type": "thinking"}
    orch._agent_precondition_missing = lambda tool, state: []
    orch._apply_tool_output = lambda tool, out, state: None
    orch._append_trace = lambda path, rec: None
    orch._build_preview = lambda state: {}
    orch.update_state_from_respond = types.MethodType(
        InferenceOrchestrator.update_state_from_respond, orch)
    orch._record_agent_history = types.MethodType(
        InferenceOrchestrator._record_agent_history, orch)
    return orch


def test_env_failure_stops_fast() -> None:
    """End-to-end: a tool that always fails the same way ends the run early,
    with the real cause reported instead of a generic loop message."""
    section("environment failure short-circuit")
    import asyncio

    reason = "RuntimeError: Qwen3-TTS 服务不可达 (Connection refused)"

    class _Pred:
        parse_ok = True
        agent_trace: dict = {}
        raw_output = ""
        completion_tokens = 0

        def __init__(self, tool: str, args: dict) -> None:
            self.tool_name = tool
            self.arguments = args

    class _Model:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, *a, **kw) -> _Pred:
            self.calls += 1
            # Mimic the observed trajectory: same tool, varying arguments.
            return _Pred("Tool_BatchVisualAudio",
                         {"shot_ids": json.dumps([str(i) for i in range(1, self.calls + 2)])})

    events: list[dict] = []

    async def _dispatch(tool_name, args, state, run_id):
        return {"status": "error", "error": reason, "_elapsed_sec": 0.0}

    orch = _bare_orchestrator(_Model(), _dispatch, events)

    with tempfile.TemporaryDirectory() as td:
        results = asyncio.run(orch.run_autonomous("测试稿件", "run_env_fail", td))

    steps = results["steps_used"]
    check("环境故障时快速停止（≤4 步）", steps <= 4, f"steps_used={steps}")
    check("没有耗尽重复预算", steps < orch.max_same_tool, f"steps_used={steps}")

    errs = " | ".join(results.get("errors") or [])
    check("终止原因带上真实报错", "Qwen3-TTS" in errs, errs[:300])
    check("终止原因指明是环境故障", "环境故障" in errs, errs[:300])

    check("失败后未标记素材阶段完成", orch.visual_audio_done is False)

    obs_blob = json.dumps(orch.last_observation, ensure_ascii=False)
    check("模型最后看到的观测含失败原因", "Qwen3-TTS" in obs_blob, obs_blob[:300])

    logs = [e for e in events if e.get("type") == "console_log"]
    check("控制台逐次输出失败原因",
          any("Qwen3-TTS" in str(e.get("text", "")) for e in logs),
          str([e.get("text") for e in logs])[:300])
    check("控制台提示撤销了完成标记",
          any("撤销" in str(e.get("text", "")) for e in logs))

    errors_ev = [e for e in events if e.get("type") == "error"]
    check("向前端广播了错误事件", len(errors_ev) >= 1)
    check("完成事件标记为 completed_with_errors",
          any(e.get("type") == "pipeline_complete"
              and e.get("status") == "completed_with_errors" for e in events))


def test_stage4_repair_budget() -> None:
    """The observed Stage4 loop: repair re-sent 11 times, always accepted.

    `Tool_Stage4DimensionalRepair` answers `completed_with_warnings` rather than
    `error`, so nothing in the failure accounting ever fired and the run only
    stopped once it burned the whole repeat budget. Two things end it now: the
    repair budget refuses to execute a third call, and identical calls are
    counted regardless of the status they came back with.
    """
    section("Stage4 修复预算")
    import asyncio

    class _Pred:
        parse_ok = True
        agent_trace: dict = {}
        raw_output = ""
        completion_tokens = 0

        def __init__(self, tool: str, args: dict) -> None:
            self.tool_name = tool
            self.arguments = args

    # Verbatim from the failing run: the same repair call, over and over.
    repair_args = {
        "repair_round": "0",
        "failed_shot_ids": '["6", "11", "17", "18"]',
        "target_dimensions": '["decoration_quality", "visual_match"]',
        "minimal_repair": "True",
        "rejudge_after_repair": "True",
    }

    class _Model:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, *a, **kw) -> _Pred:
            self.calls += 1
            if self.calls == 1:
                return _Pred("Tool_Stage4DimensionJudge", {"threshold": "0.75"})
            return _Pred("Tool_Stage4DimensionalRepair", dict(repair_args))

    events: list[dict] = []
    executed: list[str] = []

    async def _dispatch(tool_name, args, state, run_id):
        executed.append(tool_name)
        return {"status": "completed_with_warnings", "_elapsed_sec": 0.0}

    orch = _bare_orchestrator(_Model(), _dispatch, events)
    with tempfile.TemporaryDirectory() as td:
        results = asyncio.run(orch.run_autonomous("测试稿件", "run_stage4", td))

    repairs_run = executed.count("Tool_Stage4DimensionalRepair")
    check("修复实际执行次数不超过预算",
          repairs_run <= orch.max_stage4_repairs,
          f"executed={repairs_run}, budget={orch.max_stage4_repairs}")
    check("远早于 max_same_tool 就停下",
          results["steps_used"] < orch.max_same_tool,
          f"steps_used={results['steps_used']}")
    check("拒绝次数也有上限，不会一直空转",
          executed.count("Tool_Stage4DimensionalRepair") + 3 >= results["steps_used"],
          f"steps={results['steps_used']}, executed={len(executed)}")

    logs = [str(e.get("text", "")) for e in events if e.get("type") == "console_log"]
    check("超预算的修复被拒绝执行",
          any("拒绝执行" in t for t in logs), str(logs)[:400])
    check("提示模型去收尾",
          "Tool_StitchVideo" in json.dumps(orch.last_observation, ensure_ascii=False),
          json.dumps(orch.last_observation, ensure_ascii=False)[:300])
    check("放开收尾前置条件", orch.quality_checked and orch.repaired_done)

    errs = " | ".join(results.get("errors") or [])
    check("终止原因点名修复预算，而非泛泛的死循环",
          "修复预算" in errs and "死循环" not in errs, errs[:300])
    check("终止原因告诉用户成片还能救",
          "编辑器" in errs, errs[:300])
    check("记录了工具调用总数", results.get("tool_calls_used", 0) > 0,
          str(results.get("tool_calls_used")))


def test_total_tool_call_cap() -> None:
    """A model that keeps making progress-looking calls still hits a hard cap."""
    section("工具调用总上限")
    import asyncio

    class _Pred:
        parse_ok = True
        agent_trace: dict = {}
        raw_output = ""
        completion_tokens = 0

        def __init__(self, tool: str, args: dict) -> None:
            self.tool_name = tool
            self.arguments = args

    class _Model:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, *a, **kw) -> _Pred:
            # Alternate tools with fresh arguments every time, so neither the
            # repeat ceiling nor the identical-call ceiling can catch it.
            self.calls += 1
            tool = "Tool_MusicSearch" if self.calls % 2 else "Tool_SeverityRank"
            return _Pred(tool, {"nonce": str(self.calls)})

    events: list[dict] = []
    executed: list[str] = []

    async def _dispatch(tool_name, args, state, run_id):
        executed.append(tool_name)
        return {"status": "success", "_elapsed_sec": 0.0}

    orch = _bare_orchestrator(_Model(), _dispatch, events)
    orch.max_tool_calls = 6
    orch.max_agent_steps = 50
    with tempfile.TemporaryDirectory() as td:
        results = asyncio.run(orch.run_autonomous("测试稿件", "run_cap", td))

    check("执行的工具调用不超过总上限",
          len(executed) <= orch.max_tool_calls,
          f"executed={len(executed)}, cap={orch.max_tool_calls}")
    check("未走到步数上限才停",
          results["steps_used"] < orch.max_agent_steps,
          f"steps_used={results['steps_used']}")
    errs = " | ".join(results.get("errors") or [])
    check("终止原因点名工具调用总上限", "max_tool_calls" in errs, errs[:300])


def test_scoped_dispatch() -> None:
    """Selective regeneration only processes dirty shots, leaving others intact."""
    section("scoped dispatch (selective regen)")
    import asyncio
    from studio.app import _dispatch_scoped

    full_plan = [
        {"shot_id": "1", "vocal_script": "镜头1"},
        {"shot_id": "2", "vocal_script": "镜头2"},
        {"shot_id": "3", "vocal_script": "镜头3"},
        {"shot_id": "4", "vocal_script": "镜头4"},
    ]
    full_assets = {
        "1": {"audio_path": "/a/1.wav", "clip_path": "/v/1.mp4", "rendered_path": "/r/1.mp4"},
        "2": {"audio_path": "/a/2.wav", "clip_path": "/v/2.mp4", "rendered_path": "/r/2.mp4"},
        "3": {"audio_path": "/a/3.wav", "clip_path": "/v/3.mp4", "rendered_path": "/r/3.mp4"},
        "4": {"audio_path": "/a/4.wav", "clip_path": "/v/4.mp4", "rendered_path": "/r/4.mp4"},
    }
    state = {"shot_plan": list(full_plan), "shot_assets": dict(full_assets)}

    # Track what the tool actually saw
    seen_plan: list[str] = []
    seen_assets: list[str] = []

    class _Orch:
        async def _dispatch_tool(self, tool, args, st, run_id):
            seen_plan.extend(str(s.get("shot_id", "")) for s in st.get("shot_plan", []))
            seen_assets.extend(st.get("shot_assets", {}).keys())
            # Simulate writing new results for the processed shots
            for sid in seen_assets:
                st["shot_assets"][sid] = {**st["shot_assets"].get(sid, {}),
                                          "rendered_path": f"/r/{sid}_new.mp4"}
            return {"status": "success"}

    logs: list[str] = []

    async def _broadcast(rid, ev):
        if ev.get("type") == "console_log":
            logs.append(ev.get("text", ""))

    result = asyncio.run(_dispatch_scoped(
        _Orch(), "Tool_BatchTimelineCompose", {"shot_ids": ["2", "3"]}, state, "test_run",
        shot_ids_filter=["2", "3"], broadcast_fn=_broadcast,
    ))

    check("工具只看到脏镜头", sorted(seen_plan) == ["2", "3"], str(seen_plan))
    check("工具只看到脏资产", sorted(seen_assets) == ["2", "3"], str(seen_assets))
    check("执行后返回 success", result.get("status") == "success")

    # After: state must have all 4 shots back with mixed assets
    check("state 恢复为完整 shot_plan",
          [str(s.get("shot_id", "")) for s in state["shot_plan"]] == ["1", "2", "3", "4"],
          str([s.get("shot_id") for s in state["shot_plan"]]))
    check("未改动镜头保留原始 rendered_path",
          state["shot_assets"]["1"]["rendered_path"] == "/r/1.mp4",
          str(state["shot_assets"]["1"]))
    check("改动镜头写入新 rendered_path",
          state["shot_assets"]["2"]["rendered_path"] == "/r/2_new.mp4",
          str(state["shot_assets"]["2"]))
    check("未改动镜头4保留原始",
          state["shot_assets"]["4"]["rendered_path"] == "/r/4.mp4")
    check("广播了跳过信息", any("跳过" in s and "2" in s for s in logs), str(logs))

    # Stitch should NOT be scoped
    seen_plan.clear()
    seen_assets.clear()
    state2 = {"shot_plan": list(full_plan), "shot_assets": dict(full_assets)}
    asyncio.run(_dispatch_scoped(
        _Orch(), "Tool_StitchVideo", {"shot_ids": ["2"]}, state2, "test_run2",
        shot_ids_filter=["2"], broadcast_fn=_broadcast,
    ))
    check("StitchVideo 不做 scope（看到全部镜头）",
          sorted(seen_plan) == ["1", "2", "3", "4"], str(seen_plan))

    # Regression: replacing footage stripped the HyperFrames animation.
    # pipeline/graph.py rebuilds each asset dict with a literal
    # `"decoration": {}` placeholder, so the key exists but is empty. The
    # restore used setdefault, which is a no-op on an existing key — the empty
    # placeholder survived, timeline_compose saw no decoration and rendered the
    # spotlight with plain ffmpeg instead of HyperFrames.
    state3 = {
        "shot_plan": [{"shot_id": "1", "is_spotlight": True}],
        "shot_assets": {"1": {"decoration": {"template_id": "hook_title_intro"},
                              "clip_path": "/old.mp4"}},
    }

    class _WipeOrch:
        async def _dispatch_tool(self, tool, args, st, run_id):
            # Mimic node_visual_audio_batch rebuilding the asset dict.
            st["shot_assets"]["1"] = {"clip_path": "/new.mp4", "decoration": {}}
            return {"status": "success"}

    asyncio.run(_dispatch_scoped(
        _WipeOrch(), "Tool_BatchVisualAudio", {}, state3, "test_run3",
        shot_ids_filter=["1"], broadcast_fn=None,
    ))
    deco = state3["shot_assets"]["1"].get("decoration") or {}
    check("换素材后 decoration 未被空占位覆盖",
          deco.get("template_id") == "hook_title_intro", str(deco))
    check("换素材后新 clip_path 生效",
          state3["shot_assets"]["1"]["clip_path"] == "/new.mp4")

    # The inverse must still hold: a node that produces real new decoration
    # (swap_template / set_spotlight_slots) wins over the stale snapshot.
    state4 = {
        "shot_plan": [{"shot_id": "1", "is_spotlight": True}],
        "shot_assets": {"1": {"decoration": {"template_id": "OLD"}}},
    }

    class _FreshOrch:
        async def _dispatch_tool(self, tool, args, st, run_id):
            st["shot_assets"]["1"] = {"decoration": {"template_id": "NEW"}}
            return {"status": "success"}

    asyncio.run(_dispatch_scoped(
        _FreshOrch(), "Tool_BatchDecorateSpotlights", {}, state4, "test_run4",
        shot_ids_filter=["1"], broadcast_fn=None,
    ))
    check("新生成的 decoration 优先于旧快照",
          state4["shot_assets"]["1"]["decoration"].get("template_id") == "NEW",
          str(state4["shot_assets"]["1"]))


def test_http(tmp: Path) -> None:
    section("http surface")
    from fastapi.testclient import TestClient

    from studio.app import app

    with TestClient(app) as client:
        r = client.get("/api/status")
        check("GET /api/status", r.status_code == 200, r.text[:200])
        body = r.json()
        check("status 含 model", "model" in body)
        check("跳过模型加载时未就绪", body["model"]["loaded"] is False)

        r = client.get("/api/config")
        check("GET /api/config", r.status_code == 200, r.text[:200])
        cfg = r.json()
        for key in ("tracks", "features", "editor", "agent"):
            check(f"config 含 {key}", key in cfg)
        check("轨道定义带 label",
              all("key" in t and "label" in t for t in cfg["tracks"]))
        check("editor.ops 与后端一致", isinstance(cfg["editor"]["ops"], list)
              and len(cfg["editor"]["ops"]) >= 8)

        r = client.get("/")
        check("GET / 返回页面", r.status_code == 200 and "AICut" in r.text, r.text[:120])

        for asset in ("/static/css/studio.css", "/static/js/studio.js",
                      "/static/js/timeline.js", "/static/js/editor.js",
                      "/static/js/chat.js"):
            r = client.get(asset)
            check(f"静态资源 {asset}", r.status_code == 200, str(r.status_code))

        r = client.get("/api/runs")
        check("GET /api/runs", r.status_code == 200 and "runs" in r.json())

        r = client.get("/api/runs/nope")
        check("未知 run 返回 404", r.status_code == 404, str(r.status_code))

        r = client.post("/api/runs", json={"input_text": ""})
        check("空稿件被拒", r.status_code == 400, str(r.status_code))

        r = client.post("/api/runs", json={"input_text": "测试稿件"})
        check("模型未就绪时启动返回 503", r.status_code == 503, str(r.status_code))

        r = client.get("/api/projects")
        check("GET /api/projects", r.status_code == 200 and "projects" in r.json())

        r = client.get("/api/project", params={"dir": str(tmp / "does_not_exist")})
        check("未知工程返回 404", r.status_code == 404, str(r.status_code))

        # a real project, placed under the configured workspace
        from studio import config as studio_config
        cfg_obj = studio_config.load()
        pdir = cfg_obj.workspace_root / "projects" / "aicut_httptest_0001"
        pdir.mkdir(parents=True, exist_ok=True)
        state = make_state()
        state["project_id"] = "aicut_httptest_0001"
        (pdir / "final_state.json").write_text(
            json.dumps(state, ensure_ascii=False), encoding="utf-8")

        try:
            r = client.get("/api/project", params={"dir": str(pdir)})
            check("GET /api/project", r.status_code == 200, r.text[:200])
            payload = r.json()
            for key in ("progress", "digest", "dirty", "pending_steps", "versions"):
                check(f"project payload 含 {key}", key in payload)
            check("progress 可直接喂给时间轴",
                  "clips" in payload["progress"] and "track_order" in payload["progress"])

            r = client.get("/api/projects")
            check("新工程出现在列表",
                  any(p["project_id"] == "aicut_httptest_0001" for p in r.json()["projects"]))

            r = client.post("/api/project/ops", json={
                "project_dir": str(pdir), "dry_run": True,
                "ops": [{"op": "set_subtitle", "shot_id": "S01", "text": "试改"}],
            })
            check("dry_run 校验通过", r.status_code == 200 and r.json()["ok"], r.text[:200])

            r = client.post("/api/project/ops", json={
                "project_dir": str(pdir),
                "ops": [{"op": "set_narration", "shot_id": "S01", "text": "正式改写"}],
            })
            check("POST ops 应用成功", r.status_code == 200, r.text[:200])
            if r.status_code == 200:
                body = r.json()
                check("应用后返回脏镜头", "S01" in body["dirty"], str(body["dirty"]))
                check("应用后给出重生成计划", len(body["pending_steps"]) > 0)

            r = client.post("/api/project/ops", json={
                "project_dir": str(pdir),
                "ops": [{"op": "set_narration", "shot_id": "S404", "text": "x"}],
            })
            check("非法 op 返回 400", r.status_code == 400, str(r.status_code))

            r = client.post("/api/project/regenerate", json={"project_dir": str(pdir)})
            check("模型未就绪时重渲染返回 503", r.status_code == 503, str(r.status_code))

            r = client.get("/api/project/versions", params={"dir": str(pdir)})
            check("GET versions", r.status_code == 200, r.text[:150])

            r = client.get("/api/project/chat/memory", params={"dir": str(pdir)})
            check("GET chat memory", r.status_code == 200, r.text[:150])

            r = client.post("/api/project/chat", json={
                "project_dir": str(pdir), "message": "把开头改一下"})
            check("chat 在模型未就绪时返回 503（或已关闭 403）",
                  r.status_code in (503, 403), str(r.status_code) + r.text[:150])

            _http_delete_project_checks(client, cfg_obj, pdir)

            r = client.get("/api/media", params={"path": "/etc/passwd"})
            check("媒体接口拒绝越权路径", r.status_code in (403, 404, 415),
                  str(r.status_code))

            r = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "hi"}]})
            check("模型未就绪时 /v1 返回 503", r.status_code == 503, str(r.status_code))
        finally:
            shutil.rmtree(pdir, ignore_errors=True)


def test_event_stream() -> None:
    """The live-progress path: fan-out ordering, throttling, and late-join replay.

    Exercised through ``RunStore`` with a stub socket rather than a real
    WebSocket, so the assertions cover the logic that can actually be wrong
    without depending on the test client's event loop.
    """
    section("event stream")
    import asyncio

    from studio import tracks
    from studio.runs import RunStore

    class FakeSocket:
        def __init__(self, fail_after: int = -1) -> None:
            self.sent: list[dict] = []
            self.fail_after = fail_after

        async def send_json(self, payload: dict) -> None:
            if 0 <= self.fail_after <= len(self.sent):
                raise ConnectionResetError("client gone")
            self.sent.append(payload)

    snapshot = tracks.build_snapshot(make_state())

    async def scenario() -> None:
        store = RunStore(max_events=100, throttle_ms=0, raw_buffer=5)
        run = store.create("事件流测试")
        live = FakeSocket()
        store.subscribe(run.id, live)

        await store.emit(run.id, {"type": "run_started", "run_id": run.id})
        await store.emit(run.id, {
            "type": "thinking", "step": 1, "tool_name": "Tool_PlanShots",
            "thinking": "先规划镜头", "parse_ok": True,
        })
        await store.emit(run.id, {
            "type": "tool_call", "step": 1, "tool_name": "Tool_PlanShots",
            "arguments": {"max_shots": 4}, "precondition_missing": [],
        })
        await store.emit(run.id, {"type": "progress", "progress": snapshot})
        await store.emit(run.id, {"type": "console_log", "level": "system", "text": "一条日志"})
        await store.emit(run.id, {"type": "error", "stage": "render", "message": "渲染失败"})

        types = [m["type"] for m in live.sent]
        check("事件按序送达",
              types == ["run_started", "thinking", "tool_call", "progress",
                        "console_log", "error"], str(types))
        check("console_log 自动补时间戳",
              bool(next(m for m in live.sent if m["type"] == "console_log").get("timestamp")))
        check("error 记入 run.errors", run.errors == ["渲染失败"], str(run.errors))
        check("progress 存为最新快照", run.progress.get("overall_pct") == 100)
        check("progress 不进事件日志",
              all(e["type"] != "progress" for e in run.events))

        # -- late join replays history and lands on current state --
        late = FakeSocket()
        await store.replay(run.id, late)
        rtypes = [m["type"] for m in late.sent]
        check("回放以 replay_start 开始", rtypes[0] == "replay_start", str(rtypes[:2]))
        check("回放以 replay_end 结束", rtypes[-1] == "replay_end", str(rtypes[-2:]))
        check("回放包含历史事件", "tool_call" in rtypes and "console_log" in rtypes, str(rtypes))
        check("回放补发最新进度", "progress" in rtypes, str(rtypes))

        # -- throttling drops intermediate snapshots but not status events --
        thr = RunStore(max_events=100, throttle_ms=10_000, raw_buffer=5)
        trun = thr.create("限流测试")
        tsock = FakeSocket()
        thr.subscribe(trun.id, tsock)
        for _ in range(5):
            await thr.emit(trun.id, {"type": "progress", "progress": snapshot})
        await thr.emit(trun.id, {"type": "stage_update", "stage": "render", "status": "started"})
        sent_types = [m["type"] for m in tsock.sent]
        check("进度事件被限流", sent_types.count("progress") == 1, str(sent_types))
        check("状态事件不受限流", "stage_update" in sent_types, str(sent_types))
        check("限流仍保留最新快照", bool(trun.progress))

        await thr.flush_progress(trun.id)
        check("收尾强制补发最新进度",
              [m["type"] for m in tsock.sent].count("progress") == 2,
              str([m["type"] for m in tsock.sent]))

        # -- a dead socket is dropped, not retried forever --
        dead = FakeSocket(fail_after=0)
        thr.subscribe(trun.id, dead)
        await thr.emit(trun.id, {"type": "console_log", "text": "x"})
        check("断开的连接被移除", dead not in thr.sockets.get(trun.id, []))

        # -- raw output buffer is bounded --
        for i in range(12):
            await thr.emit(trun.id, {"type": "raw_output", "text": f"out {i}", "step": i})
        check("raw_output 缓冲有上限", len(trun.raw_outputs) == 5, str(len(trun.raw_outputs)))

        # -- running runs cannot be deleted --
        check("运行中不可删除", thr.delete(trun.id) is False)
        trun.status = "completed"
        check("结束后可删除", thr.delete(trun.id) is True)

    asyncio.run(scenario())


def _http_delete_project_checks(client, cfg_obj, loaded_dir: Path) -> None:
    """DELETE /api/project: guards, cache invalidation, happy path."""
    from studio.app import studio

    r = client.delete("/api/project", params={"dir": "/etc"})
    check("拒绝删除工作区外目录", r.status_code == 400, str(r.status_code))

    r = client.delete("/api/project", params={"dir": str(cfg_obj.workspace_root)})
    check("拒绝删除工作区根目录", r.status_code == 400, str(r.status_code))

    # a running job on that project blocks deletion
    blocked = studio.store.create("占用中的任务")
    blocked.project_dir = str(loaded_dir)
    blocked.status = "running"
    r = client.delete("/api/project", params={"dir": str(loaded_dir)})
    check("有任务运行时拒绝删除", r.status_code == 409, str(r.status_code))
    blocked.status = "completed"

    # the loaded project is cached; deleting must drop that cache
    check("删除前工程已被缓存",
          str(loaded_dir.resolve()) in studio.projects, str(list(studio.projects)))

    victim = cfg_obj.workspace_root / "projects" / "aicut_http_delete"
    victim.mkdir(parents=True, exist_ok=True)
    (victim / "final_state.json").write_text(
        json.dumps(make_state(), ensure_ascii=False), encoding="utf-8")
    (victim / "final.mp4").write_bytes(b"y" * 4096)

    r = client.get("/api/projects")
    check("待删工程出现在列表",
          any(p["project_dir"] == str(victim) for p in r.json()["projects"]))

    r = client.delete("/api/project", params={"dir": str(victim)})
    check("DELETE /api/project 成功", r.status_code == 200, r.text[:200])
    if r.status_code == 200:
        check("返回释放的空间", r.json().get("freed_bytes", 0) > 4000,
              str(r.json().get("freed_bytes")))
    check("目录已从磁盘移除", not victim.exists())

    r = client.get("/api/projects")
    check("删除后不再出现在列表",
          all(p["project_dir"] != str(victim) for p in r.json()["projects"]))

    r = client.delete("/api/project", params={"dir": str(victim)})
    check("重复删除返回 400", r.status_code == 400, str(r.status_code))

    # deleting the cached project must clear both caches
    r = client.delete("/api/project", params={"dir": str(loaded_dir)})
    check("删除已加载的工程成功", r.status_code == 200, r.text[:200])
    check("工程缓存已清理",
          str(loaded_dir.resolve()) not in studio.projects, str(list(studio.projects)))
    check("对话缓存已清理",
          str(loaded_dir.resolve()) not in studio.chats, str(list(studio.chats)))

    r = client.get("/api/project", params={"dir": str(loaded_dir)})
    check("删除后再打开返回 404", r.status_code == 404, str(r.status_code))

    studio.store.runs.pop(blocked.id, None)


def test_script_cutting() -> None:
    """The script -> coverage unit -> shot contract.

    `_fill_shots_with_full_script` drops any shot beyond `len(units)`, so the
    unit count is a hard ceiling on the shot count. A script written without
    sentence-final punctuation used to collapse to one unit and cap a whole
    narration at a one-shot video.
    """
    section("口播稿切分与槽位契约")
    try:
        from server.infer_graph import InferenceOrchestrator as IO
    except ImportError as exc:
        check("InferenceOrchestrator 可导入", False, str(exc))
        return
    check("InferenceOrchestrator 可导入", True)

    cut = IO._script_coverage_units

    def lossless(text: str) -> bool:
        """Slicing the text at unit starts must reproduce it verbatim."""
        starts = [u["char_start"] for u in cut(text)]
        if not starts:
            return not text.strip()
        bounds = starts + [len(text)]
        joined = "".join(text[a:b] for a, b in zip(bounds, bounds[1:]))
        return "".join(joined.split()) == "".join(text[starts[0]:].split())

    run_on = "森林看起来安静" + "，但地下的真菌网络一直在传递信号" * 12
    units = cut(run_on)
    check("无句末标点的长稿不再塌成 1 个句元", len(units) > 5, f"units={len(units)}")
    check("长稿切分无损", lossless(run_on))

    lines = "\n".join(f"第{i}段口播内容" for i in range(1, 21))
    check("换行分隔的稿件按行切分", len(cut(lines)) == 20, f"units={len(cut(lines))}")

    # A well-punctuated script must cut exactly as before, or the trained
    # model starts seeing an input distribution it was never trained on.
    normal = "森林很安静。地下藏着通信系统。科学家叫它木维网！你信吗？"
    old_style = [s.strip() for s in re.split(r"(?<=[。！？!?])\s*", normal) if s.strip()]
    check("正常标点稿件切分与旧行为一致",
          [u["text"] for u in cut(normal)] == old_style,
          f"{[u['text'] for u in cut(normal)]} != {old_style}")

    check("空稿返回空", cut("   ") == [])
    check("无任何标点的稿件仍可切分", len(cut("啊" * 300)) > 1)
    check("单句稿件切成 1 个句元", len(cut("就这一句话。")) == 1)

    # Unit count really is the shot ceiling.
    shots = [{"shot_id": str(i), "covered_unit_ids": []} for i in range(1, 9)]
    info = IO._fill_shots_with_full_script(shots, cut(run_on), run_on)
    check("镜头数不超过句元数", len(shots) <= len(cut(run_on)))
    check("铺满后口播稿完整", bool(info.get("complete")), str(info))

    spec = IO._slot_spec_from_name
    expected = {
        "year": ("short_text", 12),
        "emoji_icon": ("emoji", 2),
        "person_name": ("short_text", 12),
        "headline": ("headline", 16),
        "explain_text": ("text", 40),
        "concept_desc": ("text", 40),   # must not match the `con` label bucket
        "left_label": ("label", 12),
        "tag1": ("label", 12),
    }
    for name, (want_type, want_max) in expected.items():
        got = spec(name)
        check(f"槽位 {name} -> {want_type}",
              got.get("type") == want_type and got.get("max_chars") == want_max,
              str(got))
    check("列表型槽位不当成文本", spec("nodes").get("type") == "list", str(spec("nodes")))
    check("槽位带填写提示", bool(spec("year").get("hint")))


def test_plan_shots_alignment() -> None:
    """The plan_shots generator payload must match the v6 training rows.

    The model was trained to plan one segment per call. Handing it a
    whole-article payload puts it off-distribution: it runs until it exhausts
    its token budget and never closes the JSON, and the runtime then falls back
    to a couple of stub shots. The fixture is a real training row, trimmed.
    """
    section("plan_shots 与训练对齐")
    fixture = Path(__file__).parent / "fixtures" / "v6_plan_shots_training_row.json"
    if not fixture.exists():
        check("训练样本 fixture 存在", False, str(fixture))
        return
    try:
        from server.infer_graph import InferenceOrchestrator as IO
    except ImportError as exc:
        check("InferenceOrchestrator 可导入", False, str(exc))
        return

    gold = json.loads(fixture.read_text(encoding="utf-8"))["user_payload"]
    mem, cons = gold["memory"], gold["constraints"]
    target = mem["target_segment"]
    segments = [dict(s) for s in mem["segments"]]
    for i, seg in enumerate(segments):
        if seg.get("segment_id") == target.get("segment_id"):
            segments[i] = target
        elif seg.get("segment_id") == (mem.get("next_segment") or {}).get("segment_id"):
            segments[i] = mem["next_segment"]

    mine = IO._build_skill_payload_plan_shots(
        {
            "input_text": gold["input_script"],
            "intent": mem["intent"],
            "music_style": mem["music_style"],
            "segments": segments,
        },
        target,
        segments=segments,
        planned_shots=[],
        chunk_index=cons["chunk_index"],
        num_chunks=cons["num_chunks_for_segment"],
        expected_shots=cons["expected_num_shots"],
        total_segment_shots=cons["total_segment_shots"],
        total_target_shots=cons["num_video_shots"],
    )

    def paths(obj, prefix: str = "") -> set:
        out = set()
        if isinstance(obj, dict):
            for k, v in obj.items():
                out.add(prefix + k)
                out |= paths(v, f"{prefix}{k}.")
        elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
            out |= paths(obj[0], f"{prefix}[].")
        return out

    check("顶层字段与顺序一致", list(gold.keys()) == list(mine.keys()),
          f"{list(mine.keys())}")
    check("role 标为 generator", mine.get("role") == "generator", str(mine.get("role")))
    missing = sorted(paths(gold) - paths(mine))
    extra = sorted(paths(mine) - paths(gold))
    check("没有训练里有而推理缺失的字段", not missing, str(missing))
    check("没有训练里没有的多余字段", not extra, str(extra))
    check("instruction 与训练逐字一致", gold["instruction"] == mine["instruction"],
          mine["instruction"][:80])
    check("output_schema 与训练一致", gold["output_schema"] == mine["output_schema"])
    for key, want in cons.items():
        # num_segments is derived from the segment list, which the fixture
        # truncates to keep it small.
        if key == "num_segments" or not isinstance(want, (str, int, float, bool)):
            continue
        check(f"约束 {key} 一致", mine["constraints"].get(key) == want,
              f"{mine['constraints'].get(key)!r} != {want!r}")
    check("约束 num_segments 取自段落数",
          mine["constraints"]["num_segments"] == len(segments),
          str(mine["constraints"]["num_segments"]))

    # Shots come back without unit ids; anchoring by position is what lets the
    # fill step order them.
    script = "第一句话。第二句话。第三句话。第四句话。"
    units = IO._script_coverage_units(script)
    shots = [
        {"shot_id": "1", "vocal_script": "第一句话。第二句话。"},
        {"shot_id": "2", "vocal_script": "第三句话。第四句话。"},
    ]
    anchored = IO._anchor_shots_to_units(shots, units, script)
    check("全部镜头都能在原文中定位", anchored == 2, str(anchored))
    check("锚点按顺序切分句元",
          shots[0]["covered_unit_ids"] == ["u01", "u02"]
          and shots[1]["covered_unit_ids"] == ["u03", "u04"],
          f"{shots[0]['covered_unit_ids']} / {shots[1]['covered_unit_ids']}")
    stray = [{"shot_id": "1", "vocal_script": "这句话原文里没有"}]
    check("定位不到的镜头不报错也不乱锚",
          IO._anchor_shots_to_units(stray, units, script) == 0
          and "covered_unit_ids" not in stray[0])


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="aicut_studio_test_"))
    try:
        test_config()
        test_concat_codec_guard()
        test_tracks()
        proj, _ = test_project_and_editor(tmp)
        test_chat_agent(proj)
        test_orchestrator_overrides()
        test_script_cutting()
        test_plan_shots_alignment()
        test_scoped_dispatch()
        test_event_stream()
        test_http(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'=' * 60}")
    print(f"通过 {len(PASS)} / {len(PASS) + len(FAIL)}")
    if FAIL:
        print("\n失败项:")
        for f in FAIL:
            print("  - " + f)
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
