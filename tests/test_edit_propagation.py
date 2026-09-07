"""Regression tests for edits that reached state but never reached the video.

Every case here corresponds to an edit the UI reported as applied while the
rendered output was unchanged, so each test asserts on the value the *renderer*
or the *tool call* ends up seeing — not merely that the project dict was
mutated. Run directly: ``python3 tests/test_edit_propagation.py``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studio import editor  # noqa: E402
from studio import project as project_mod  # noqa: E402

def _find_gold_template_root() -> Path:
    """Locate the template library without hardcoding a checkout layout.

    ``external_tool/gold_template`` sits beside the app in the published repo
    and one level up in the development tree, so probe for the marker directory
    the registry actually reads instead of assuming either.
    """
    env = os.environ.get("AICUT_GOLD_TEMPLATE_ROOT")
    if env:
        return Path(env)
    candidates = [
        ROOT / "external_tool" / "gold_template",
        ROOT.parent / "external_tool" / "gold_template",
    ]
    for candidate in candidates:
        if (candidate / "generated" / "structures").is_dir():
            return candidate
    return candidates[0]


GOLD_ROOT = _find_gold_template_root()

PASS: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASS.append(name)
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}{': ' + detail if detail else ''}")
        raise AssertionError(name)


def make_project(tmp: Path) -> project_mod.Project:
    state = {
        "output_dir": str(tmp),
        "config": {"paths": {"gold_template_root": str(GOLD_ROOT)}},
        "input_text": "demo",
        "segments": [],
        "shot_plan": [
            {"shot_id": "1", "vocal_script": "第一句旁白", "is_spotlight": True,
             "duration_sec": 4.0},
            {"shot_id": "2", "vocal_script": "第二句旁白", "is_spotlight": False,
             "duration_sec": 4.0},
        ],
        "shot_assets": {
            "1": {
                "clip_path": str(tmp / "a.mp4"),
                "rendered_path": str(tmp / "shots/1/shot.mp4"),
                "decoration": {
                    "structure_id": "key_takeaway",
                    "motion_id": "scale_pop_hold",
                    "visual_id": "dark_tech",
                    "slots": {"headline": "旧标题"},
                },
            },
            "2": {"clip_path": str(tmp / "b.mp4")},
        },
        "bgm_tracks": [{"name": "old_track"}],
        "music_style": "舒缓钢琴",
    }
    return project_mod.Project(project_id="p1", project_dir=tmp, state=state)


# ---------------------------------------------------------------------------
def test_template_registry_rejects_invented_ids(tmp: Path) -> None:
    print("\n[swap_template] 只接受模板库里真实存在的 id")
    proj = make_project(tmp)
    registry = editor.template_registry(proj)
    check("registry lists structures", len(registry["structure_id"]) > 10,
          str(len(registry["structure_id"])))
    check("registry lists motions", len(registry["motion_id"]) >= 5)
    check("registry lists visuals", len(registry["visual_id"]) >= 5)

    # These are the exact ids the chat agent emitted in production.
    for bad in ({"motion_id": "slide_in"},
                {"visual_id": "motion_zoom"},
                {"structure_id": "split_screen"}):
        rejected = False
        try:
            editor.apply_ops(proj, [{"op": "swap_template", "shot_id": "1", **bad}])
        except editor.EditError as exc:
            rejected = "不存在于模板库" in str(exc)
        check(f"rejects {bad}", rejected)

    # A real id still applies, and the suggestion points at it.
    try:
        editor.apply_ops(proj, [{"op": "swap_template", "shot_id": "1",
                                 "motion_id": "slide_in"}])
    except editor.EditError as exc:
        check("suggests the real id", "slide_in_hold" in str(exc), str(exc))

    editor.apply_ops(proj, [{"op": "swap_template", "shot_id": "1",
                             "motion_id": "slide_in_hold",
                             "visual_id": "warm_editorial"}])
    deco = proj.assets_for("1")["decoration"]
    check("valid ids applied",
          deco["motion_id"] == "slide_in_hold" and deco["visual_id"] == "warm_editorial")
    check("template edit needs a render", "render" in proj.dirty.get("1", set()))


def test_subtitle_reaches_the_renderer(tmp: Path) -> None:
    print("\n[set_subtitle] 新字幕进入渲染时间轴")
    from pipeline.tools import timeline_compose

    proj = make_project(tmp)
    editor.apply_ops(proj, [{"op": "set_subtitle", "shot_id": "1",
                             "text": "修正后的字幕"}])
    shot = proj.shot("1")
    check("subtitle_text stored", shot["subtitle_text"] == "修正后的字幕")
    check("narration untouched", shot["vocal_script"] == "第一句旁白")

    # timeline_compose.run() shells out to ffmpeg; assert on the same expression
    # it now uses to pick the subtitle source.
    source = str(shot.get("subtitle_text") or "").strip() or shot["vocal_script"]
    track = timeline_compose.build_subtitle_track(source, 4.0, config={})
    rendered = "".join(str(seg.get("text", "")) for seg in track)
    check("renderer would burn the new text", "修正后的字幕" in rendered, rendered)
    check("renderer drops the old text", "第一句旁白" not in rendered, rendered)

    plain = proj.shot("2")
    fallback = str(plain.get("subtitle_text") or "").strip() or plain["vocal_script"]
    check("unedited shot still uses narration", fallback == "第二句旁白")

    digest_row = next(r for r in proj.digest()["shots"] if r["shot_id"] == "1")
    check("digest exposes the diverged subtitle",
          digest_row.get("subtitle") == "修正后的字幕")


def test_subtitles_stay_in_sync_with_the_voice_over(tmp: Path) -> None:
    print("\n[字幕/配音一致性] 改口播后字幕跟随，模板编辑不动字幕")
    from pipeline.tools import timeline_compose

    def burned(shot: dict) -> str:
        source = str(shot.get("subtitle_text") or "").strip() or shot["vocal_script"]
        track = timeline_compose.build_subtitle_track(source, 4.0, config={})
        return "".join(str(seg.get("text", "")) for seg in track)

    # A clean cut has nothing to warn about.
    proj = make_project(tmp)
    check("clean project has no drift", editor.subtitle_drift(proj) == [])

    # A deliberate subtitle-only fix is allowed, but must be reported as a
    # divergence rather than a quiet success.
    res = editor.apply_ops(proj, [{"op": "set_subtitle", "shot_id": "1",
                                   "text": "第一句"}])[0]
    check("divergence flagged on the result",
          res.details.get("diverges_from_narration") is True)
    check("message says the audio is unchanged", "配音仍是原文" in res.message, res.message)
    drift = editor.subtitle_drift(proj)
    check("drift reports the shot",
          len(drift) == 1 and drift[0]["shot_id"] == "1", str(drift))
    check("drift carries both texts",
          drift[0]["subtitle"] == "第一句" and drift[0]["narration"] == "第一句旁白")

    # Changing the narration must not leave the old override captioning the new
    # audio: that is the state that shipped captions the voice never said.
    res = editor.apply_ops(proj, [{"op": "set_narration", "shot_id": "1",
                                   "text": "全新的一句旁白内容"}])[0]
    shot = proj.shot("1")
    check("stale override dropped", "subtitle_text" not in shot, str(shot.keys()))
    check("drift resolved", editor.subtitle_drift(proj) == [])
    check("renderer burns the new narration", "全新的一句旁白内容" in burned(shot))
    check("renderer drops the stale subtitle", "第一句" not in burned(shot).replace(
        "全新的一句旁白内容", ""))
    check("user is told the subtitle was resynced",
          "字幕已同步" in res.message, res.message)
    check("TTS will be regenerated", "tts" in proj.dirty.get("1", set()))
    check("old audio invalidated", "audio_path" not in proj.assets_for("1"))

    # Setting the subtitle back to the narration is a resync, not a divergence.
    proj = make_project(tmp)
    editor.apply_ops(proj, [{"op": "set_subtitle", "shot_id": "1", "text": "截断版"}])
    res = editor.apply_ops(proj, [{"op": "set_subtitle", "shot_id": "1",
                                   "text": "第一句旁白"}])[0]
    check("resync clears the override", res.details.get("synced") is True)
    check("resync leaves no drift", editor.subtitle_drift(proj) == [])
    check("resync stores no duplicate", "subtitle_text" not in proj.shot("1"))

    # Whitespace-only differences are not a divergence.
    proj = make_project(tmp)
    editor.apply_ops(proj, [{"op": "set_subtitle", "shot_id": "1",
                             "text": " 第一句旁白 "}])
    check("whitespace is not drift", editor.subtitle_drift(proj) == [])


def test_narration_edit_regenerates_voice_visual_query_and_footage(tmp_path: Path) -> None:
    print("\n[set_narration] 改镜头文案会级联重配音、重检索、重渲染")
    proj = make_project(tmp_path)
    assets = proj.assets_for("1")
    assets["audio_path"] = "/old/audio.wav"
    assets["audio_meta"] = {"duration_sec": 4.0}
    assets["clip_path"] = "/old/video.mp4"
    assets["clip_source"] = "retrieval"
    old_query = str(proj.shot("1").get("visual_query", "") or "")

    result = editor.apply_ops(proj, [{
        "op": "set_narration",
        "shot_id": "1",
        "text": "量子芯片在极低温环境里维持脆弱量子态",
        "visual_query": "极低温量子芯片实验室特写",
    }])[0]

    shot = proj.shot("1")
    dirty = proj.dirty.get("1", set())
    check("narration updated", shot["vocal_script"] == "量子芯片在极低温环境里维持脆弱量子态")
    check("visual query updated", shot["visual_query"] == "极低温量子芯片实验室特写")
    check("visual query changed", shot["visual_query"] != old_query)
    check("tts marked dirty", "tts" in dirty, str(dirty))
    check("footage marked dirty", "footage" in dirty, str(dirty))
    check("render marked dirty", "render" in dirty, str(dirty))
    check("old audio removed", "audio_path" not in assets and "audio_meta" not in assets)
    check("old footage removed", "clip_path" not in assets)
    check("clip source reset to retrieval", assets.get("clip_source") == "retrieval")
    check("message mentions retrieval", "重新检索" in result.message, result.message)

    tools = [step["tool"] for step in editor.plan_regeneration(proj)]
    check("visual/audio regenerates before render",
          tools[:2] == ["Tool_BatchVisualAudio", "Tool_BatchTimelineCompose"],
          str(tools))


def test_production_regression_template_edit_keeps_audio_in_sync(tmp: Path) -> None:
    print("\n[线上回归] 换模板那一轮不应该让字幕和配音脱节")
    proj = make_project(tmp)
    long_script = proj.shot("1")["vocal_script"]

    # The exact op batch the chat agent emitted for "镜头1的模版换一个趋势图":
    # a template swap plus its slots, in one operation now that `slots` is
    # accepted — the split form was rejected as an unknown parameter.
    editor.apply_ops(proj, [{
        "op": "swap_template", "shot_id": "1",
        "structure_id": "bar_chart", "visual_id": "clean_saas",
        "slots": {"headline": "人工智能的十二年巨变"},
    }])
    deco = proj.assets_for("1")["decoration"]
    check("structure swapped", deco["structure_id"] == "bar_chart")
    check("slots filled in the same op",
          deco["slots"] == {"headline": "人工智能的十二年巨变"}, str(deco.get("slots")))
    check("template edit left the narration alone",
          proj.shot("1")["vocal_script"] == long_script)
    check("template edit created no caption drift", editor.subtitle_drift(proj) == [])
    check("template edit only needs a re-render",
          [s["tool"] for s in editor.plan_regeneration(proj)]
          == ["Tool_BatchTimelineCompose"])

    # Swapping structure without providing slots still clears the stale ones.
    proj2 = make_project(tmp)
    editor.apply_ops(proj2, [{"op": "swap_template", "shot_id": "1",
                              "structure_id": "bar_chart"}])
    check("stale slots cleared on structure change",
          "slots" not in proj2.assets_for("1")["decoration"])

    # Slots without a structure change merge into the existing ones.
    proj3 = make_project(tmp)
    editor.apply_ops(proj3, [{"op": "swap_template", "shot_id": "1",
                              "motion_id": "mask_reveal",
                              "slots": {"headline": "新标题"}}])
    check("slots merge when structure is unchanged",
          proj3.assets_for("1")["decoration"]["slots"]["headline"] == "新标题")

    bad = False
    try:
        editor.apply_ops(make_project(tmp), [{"op": "swap_template", "shot_id": "1",
                                              "slots": "not a dict"}])
    except editor.EditError:
        bad = True
    check("non-dict slots rejected", bad)


def test_restitch_only_edits_are_scheduled(tmp: Path) -> None:
    print("\n[reorder / delete / 音量] 无镜头重渲染时仍然重新拼接")
    for ops, label in (
        ([{"op": "reorder_shots", "shot_ids": ["2", "1"]}], "reorder_shots"),
        ([{"op": "delete_shot", "shot_id": "2"}], "delete_shot"),
        ([{"op": "set_bgm", "volume": 0.2}], "set_bgm(volume)"),
    ):
        proj = make_project(tmp)
        editor.apply_ops(proj, ops)
        steps = editor.plan_regeneration(proj)
        check(f"{label}: no per-shot steps", steps == [], str(steps))
        check(f"{label}: still needs a restitch", editor.needs_restitch(proj, steps))
        stitch = editor.stitch_step(proj, bgm_changed=False)
        check(f"{label}: rebuilds the mix instead of remuxing",
              stitch["args"]["skip_bgm_remix"] is False)

    # A pure render edit keeps the fast remux, which is the whole point of it.
    proj = make_project(tmp)
    editor.apply_ops(proj, [{"op": "set_subtitle", "shot_id": "1", "text": "新字幕"}])
    steps = editor.plan_regeneration(proj)
    check("subtitle edit renders the shot",
          [s["tool"] for s in steps] == ["Tool_BatchTimelineCompose"], str(steps))
    check("subtitle edit keeps the BGM fast path",
          editor.stitch_step(proj, bgm_changed=False)["args"]["skip_bgm_remix"] is True)

    proj = make_project(tmp)
    check("clean project needs nothing",
          not editor.needs_restitch(proj, editor.plan_regeneration(proj)))


def test_bgm_volume_reaches_the_stitcher(tmp: Path) -> None:
    print("\n[set_bgm] 音量与风格进入拼接/检索参数")
    proj = make_project(tmp)
    editor.apply_ops(proj, [{"op": "set_bgm", "volume": 0.15,
                             "query": "摇滚风格纯音乐"}])
    check("volume in state", proj.state["bgm_volume"] == 0.15)
    check("query in state", proj.state["bgm_query"] == "摇滚风格纯音乐")
    check("music_style follows the query",
          proj.state["music_style"] == "摇滚风格纯音乐")
    check("old tracks cleared", proj.state["bgm_tracks"] == [])
    check("bgm marked dirty", proj.bgm_dirty)

    steps = editor.plan_regeneration(proj)
    check("music search scheduled",
          [s["tool"] for s in steps] == ["Tool_MusicSearch"], str(steps))
    args = steps[0]["args"]
    check("search gets the new query", args["query"] == "摇滚风格纯音乐", str(args))

    stitch = editor.stitch_step(proj, bgm_changed=True)
    check("stitch gets the new volume", stitch["args"]["bgm_volume"] == 0.15)
    check("stitch rebuilds the mix", stitch["args"]["skip_bgm_remix"] is False)


def test_stitch_dispatch_forwards_mix_args() -> None:
    print("\n[Tool_StitchVideo] dispatch 把 bgm_volume 传给 stitch_video.run")
    src = (ROOT / "server" / "infer_graph.py").read_text(encoding="utf-8")
    call = src.index("sv_tool.run,")
    block = src[max(0, call - 2000):call + 800]
    check("bgm_volume forwarded to stitch_video.run", "bgm_volume=bgm_volume" in block)
    check("duck_under_voice forwarded", "duck_under_voice=duck_under_voice" in block)
    check("volume read from the edit's tool args", 'tool_args.get("bgm_volume"' in block)
    check("falls back to project state", 'state.get("bgm_volume"' in block)
    check("volume clamped to a valid gain",
          "max(0.0, min(1.0, bgm_volume))" in block)


def test_music_search_cache_respects_the_request(tmp: Path) -> None:
    print("\n[node_music_search] 换风格不会命中旧缓存")
    import pipeline.graph as graph_mod
    from pipeline.tools import music_search

    calls: list[str] = []

    def fake_run(*, music_style, total_video_duration, config):
        calls.append(music_style)
        return {
            "selected_tracks": [{"name": f"track_for_{music_style}"}],
            "selected_count": 1,
            "playlist_policy_version": 2,
            "music_style": music_style,
            "status": "success",
        }

    artifacts: dict[str, dict] = {}

    def fake_save(state, stage_dir, filename, data):
        artifacts[f"{stage_dir}/{filename}"] = json.loads(json.dumps(data))
        return f"{stage_dir}/{filename}"

    def fake_load(state, stage_dir, filename):
        got = artifacts.get(f"{stage_dir}/{filename}")
        return dict(got) if got else None

    orig = (music_search.run, graph_mod._save_json_artifact,
            graph_mod._load_cached_json)
    music_search.run = fake_run
    graph_mod._save_json_artifact = fake_save
    graph_mod._load_cached_json = fake_load
    try:
        state = {
            "output_dir": str(tmp), "config": {}, "input_text": "demo",
            "shot_plan": [{"shot_id": "1"}], "shot_assets": {},
            "music_style": "舒缓钢琴",
        }
        first = graph_mod.node_music_search(dict(state))
        check("first search runs", calls == ["舒缓钢琴"], str(calls))
        check("first search returns its track",
              first["bgm_tracks"][0]["name"] == "track_for_舒缓钢琴")

        same = graph_mod.node_music_search(dict(state))
        check("same style hits the cache", calls == ["舒缓钢琴"], str(calls))
        check("cache hit still returns tracks", bool(same["bgm_tracks"]))

        changed = dict(state)
        changed["bgm_query"] = "摇滚风格纯音乐"
        result = graph_mod.node_music_search(changed)
        check("new query re-runs the search",
              calls == ["舒缓钢琴", "摇滚风格纯音乐"], str(calls))
        check("new query returns new tracks",
              result["bgm_tracks"][0]["name"] == "track_for_摇滚风格纯音乐",
              str(result["bgm_tracks"]))
    finally:
        (music_search.run, graph_mod._save_json_artifact,
         graph_mod._load_cached_json) = orig


def test_visual_audio_cache_checks_visual_query() -> None:
    print("\n[node_visual_audio_batch] 素材缓存必须校验 visual_query")
    src = (ROOT / "pipeline" / "graph.py").read_text(encoding="utf-8")
    start = src.index("def _script_matches")
    block = src[start:start + 1700]
    check("cache reads cached visual query",
          "cached_query" in block and "visual_query" in block, block)
    check("cache reads planned visual query",
          "planned_query" in block and "visual_query" in block, block)
    check("cache rejects visual query mismatch",
          "cached_query != planned_query" in block, block)


def test_bgm_settings_survive_reload(tmp: Path) -> None:
    print("\n[final_state.json] bgm_query / bgm_volume 持久化")
    proj = make_project(tmp)
    editor.apply_ops(proj, [{"op": "set_bgm", "volume": 0.33, "query": "电子节拍"}])
    proj.save()

    payload = json.loads((tmp / "final_state.json").read_text(encoding="utf-8"))
    check("query persisted", payload["bgm_query"] == "电子节拍")
    check("volume persisted", payload["bgm_volume"] == 0.33)

    reloaded = project_mod.load(tmp, {})
    check("query restored", reloaded.state["bgm_query"] == "电子节拍")
    check("volume restored", reloaded.state["bgm_volume"] == 0.33)
    check("volume reaches a fresh stitch",
          editor.stitch_step(reloaded)["args"]["bgm_volume"] == 0.33)


def test_edit_versions_keep_playable_media_snapshots(tmp_path: Path) -> None:
    print("\n[versions] 每次编辑前的成片效果可独立回看")
    proj = make_project(tmp_path)
    final = tmp_path / "final.mp4"
    clip = tmp_path / "shots" / "1" / "shot.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    final.write_bytes(b"old final")
    clip.write_bytes(b"old clip")
    proj.state["final_video_path"] = str(final)
    proj.assets_for("1")["rendered_path"] = str(clip)
    proj.save()

    editor.apply_ops(proj, [{"op": "set_narration", "shot_id": "1",
                             "text": "全新的第一句旁白"}])
    proj.save("manual_edit")

    versions = proj.list_versions()
    check("version listed", bool(versions), str(versions))
    first = versions[0]
    check("version is a mini project", first.get("kind") == "project", str(first))
    check("version exposes preview video", bool(first.get("final_video_path")), str(first))

    snapshot = Path(first["project_dir"])
    payload = json.loads((snapshot / "final_state.json").read_text(encoding="utf-8"))
    snap_final = Path(payload["final_video_path"])
    snap_clip = Path(payload["shot_assets"]["1"]["rendered_path"])
    check("final video copied into snapshot", snap_final.is_file() and snap_final.read_bytes() == b"old final")
    check("shot clip copied into snapshot", snap_clip.is_file() and snap_clip.read_bytes() == b"old clip")
    check("snapshot paths stay inside version dir",
          snapshot in snap_final.parents and snapshot in snap_clip.parents,
          f"{snap_final} / {snap_clip}")


def main() -> int:
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="aicut_edit_test_"))
    try:
        test_template_registry_rejects_invented_ids(tmp / "t1")
        test_subtitle_reaches_the_renderer(tmp / "t2")
        test_subtitles_stay_in_sync_with_the_voice_over(tmp / "t2b")
        test_narration_edit_regenerates_voice_visual_query_and_footage(tmp / "t2ba")
        test_production_regression_template_edit_keeps_audio_in_sync(tmp / "t2c")
        test_restitch_only_edits_are_scheduled(tmp / "t3")
        test_bgm_volume_reaches_the_stitcher(tmp / "t4")
        test_stitch_dispatch_forwards_mix_args()
        test_music_search_cache_respects_the_request(tmp / "t6")
        test_visual_audio_cache_checks_visual_query()
        test_bgm_settings_survive_reload(tmp / "t7")
        test_edit_versions_keep_playable_media_snapshots(tmp / "t8")
    except AssertionError as exc:
        print(f"\n失败: {exc}")
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n全部通过: {len(PASS)} 项断言")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
