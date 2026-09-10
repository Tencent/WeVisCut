"""Project agent state onto the studio's multi-track progress model.

The agent's ``state`` dict is the authority on what exists. The frontend needs
two views of it:

  * **clips** — per-track time-positioned blocks for the timeline
  * **stats** — per-track "how far along is this lane" for the progress rail

Both are derived here as pure functions of ``state`` so they can be unit-tested
without a GPU, a model, or a running pipeline.

Track semantics
---------------
=============  ==============================================================
segments       chapter division produced by Tool_SegmentArticle
shots          shot plan produced by Tool_PlanShots
narration      TTS audio per shot (Tool_BatchVisualAudio)
subtitle       spoken text per shot (exists as soon as the shot is planned)
footage        retrieved video material per shot (Tool_BatchVisualAudio)
spotlight      template decoration for spotlight shots
               (Tool_BatchDecorateSpotlights)
bgm            background music (Tool_MusicSearch)
render         composed per-shot video (Tool_BatchTimelineCompose)
=============  ==============================================================
"""

from __future__ import annotations

from typing import Any

ALL_TRACKS: tuple[str, ...] = (
    "segments",
    "shots",
    "narration",
    "subtitle",
    "footage",
    "spotlight",
    "bgm",
    "render",
)

TRACK_LABELS: dict[str, str] = {
    "segments": "分段",
    "shots": "镜头",
    "narration": "旁白",
    "subtitle": "字幕",
    "footage": "素材",
    "spotlight": "装饰",
    "bgm": "配乐",
    "render": "渲染",
}

# Which tool most recently advances each track — used to highlight the active lane.
TRACK_OWNER_TOOL: dict[str, str] = {
    "segments": "Tool_SegmentArticle",
    "shots": "Tool_PlanShots",
    "narration": "Tool_BatchVisualAudio",
    "subtitle": "Tool_PlanShots",
    "footage": "Tool_BatchVisualAudio",
    "spotlight": "Tool_BatchDecorateSpotlights",
    "bgm": "Tool_MusicSearch",
    "render": "Tool_BatchTimelineCompose",
}


def _as_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _as_list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else []


def _shot_rows(state: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any], float, float]]:
    """Lay shots out on a timeline. Returns (shot, assets, start, end) per shot.

    Duration prefers the real TTS audio length once it exists, because that is
    what the final cut will actually be; before TTS it falls back to the
    planner's estimate so the timeline is still meaningful mid-run.
    """
    shot_plan = [s for s in _as_list(state.get("shot_plan")) if isinstance(s, dict)]
    shot_assets = _as_dict(state.get("shot_assets"))

    rows: list[tuple[dict[str, Any], dict[str, Any], float, float]] = []
    offset = 0.0
    for shot in shot_plan:
        sid = str(shot.get("shot_id", "") or "")
        assets = _as_dict(shot_assets.get(sid))
        audio_dur = float(_as_dict(assets.get("audio_meta")).get("duration_sec", 0) or 0)
        planned = float(shot.get("duration_sec", 0) or 0) or 8.0
        duration = max(audio_dur, planned)
        rows.append((shot, assets, offset, offset + duration))
        offset += duration
    return rows


def build_clips(state: dict[str, Any]) -> dict[str, Any]:
    """Build per-track clip lists for the timeline."""
    rows = _shot_rows(state)
    total_duration = round(rows[-1][3], 2) if rows else 0.0

    shots: list[dict[str, Any]] = []
    narration: list[dict[str, Any]] = []
    subtitle: list[dict[str, Any]] = []
    footage: list[dict[str, Any]] = []
    spotlight: list[dict[str, Any]] = []
    render: list[dict[str, Any]] = []

    for shot, assets, start, end in rows:
        sid = str(shot.get("shot_id", "") or "")
        seg_id = str(shot.get("segment_id", "") or "")
        text = str(shot.get("vocal_script", "") or "")
        is_spot = bool(shot.get("is_spotlight"))
        base = {"shot_id": sid, "segment_id": seg_id,
                "start": round(start, 2), "end": round(end, 2),
                "duration": round(end - start, 2)}

        shots.append({
            **base,
            "title": str(shot.get("chapter_title", "") or sid),
            "is_spotlight": is_spot,
            "visual_query": str(shot.get("visual_query", "") or ""),
        })

        audio_path = str(assets.get("audio_path", "") or "")
        audio_dur = float(_as_dict(assets.get("audio_meta")).get("duration_sec", 0) or 0)
        narration.append({
            **base,
            "end": round(start + audio_dur, 2) if audio_dur else base["end"],
            "text": text,
            "audio_path": audio_path,
            "ready": bool(audio_path),
        })

        subtitle.append({**base, "text": text, "ready": bool(text)})

        clip_path = str(assets.get("clip_path", "") or "")
        footage.append({
            **base,
            "clip_path": clip_path,
            "ready": bool(clip_path),
            "source": str(assets.get("clip_source", "") or ""),
        })

        if is_spot:
            decoration = _as_dict(assets.get("decoration"))
            spotlight.append({
                **base,
                "ready": bool(decoration),
                "structure_id": str(decoration.get("structure_id", "") or ""),
                "motion_id": str(decoration.get("motion_id", "") or ""),
                "visual_id": str(decoration.get("visual_id", "") or ""),
                "template_id": str(decoration.get("template_id", "") or ""),
                "slots": _as_dict(decoration.get("slots")),
            })

        rendered = str(assets.get("rendered_path", "") or "")
        render.append({**base, "rendered_path": rendered, "ready": bool(rendered)})

    # ---- segments span their member shots ----
    by_segment: dict[str, list[tuple[float, float, str]]] = {}
    for shot, _assets, start, end in rows:
        seg_id = str(shot.get("segment_id", "") or "")
        if seg_id:
            by_segment.setdefault(seg_id, []).append(
                (start, end, str(shot.get("shot_id", "") or ""))
            )

    raw_segments = [s for s in _as_list(state.get("segments")) if isinstance(s, dict)]
    segment_meta: dict[str, dict[str, Any]] = {}
    for idx, seg in enumerate(raw_segments):
        seg_id = str(seg.get("segment_id", "") or f"seg_{idx + 1:02d}")
        if seg_id in segment_meta:
            continue
        segment_meta[seg_id] = {
            "title": str(seg.get("title") or seg.get("structure_label") or seg_id),
            "summary": str(seg.get("summary", "") or ""),
        }

    segments: list[dict[str, Any]] = []
    for seg_id, members in by_segment.items():
        meta = segment_meta.get(seg_id, {"title": seg_id, "summary": ""})
        start = min((m[0] for m in members), default=0.0)
        end = max((m[1] for m in members), default=0.0)
        segments.append({
            "segment_id": seg_id,
            "title": str(meta.get("title", seg_id) or seg_id),
            "summary": str(meta.get("summary", "") or ""),
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(end - start, 2),
            "shot_count": len(members),
            "shot_ids": [m[2] for m in members],
            "ready": bool(members),
        })
    segments.sort(key=lambda row: (row["start"], row["end"], row["segment_id"]))

    # ---- bgm ----
    bgm: list[dict[str, Any]] = []
    for t in _as_list(state.get("bgm_tracks")):
        if not isinstance(t, dict):
            continue
        start = float(t.get("offset_sec", 0) or 0)
        duration = float(t.get("duration_sec", 0) or 0)
        end = float(t.get("end_sec", 0) or 0) or (start + duration)
        # Clamp to video duration: the rendered video fades/cuts the BGM at
        # the last frame, so the timeline should reflect that.
        if total_duration > 0 and end > total_duration:
            end = total_duration
            duration = max(0.0, end - start)
        bgm.append({
            "track_id": str(t.get("track_id", "") or ""),
            "name": str(t.get("name", "") or ""),
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(duration, 2),
            "path": str(t.get("path", "") or t.get("audio_path", "") or ""),
            "ready": True,
        })

    return {
        "total_duration": total_duration,
        "segments": segments,
        "shots": shots,
        "narration": narration,
        "subtitle": subtitle,
        "footage": footage,
        "spotlight": spotlight,
        "bgm": bgm,
        "render": render,
    }


def build_stats(state: dict[str, Any], clips: dict[str, Any] | None = None) -> dict[str, Any]:
    """Per-track completion counters driving the progress rail."""
    c = clips if clips is not None else build_clips(state)

    def ratio(done: int, total: int) -> int:
        return int(round(done * 100 / total)) if total else 0

    stats: dict[str, Any] = {}
    for name in ("segments", "shots", "narration", "subtitle", "footage", "spotlight", "render"):
        items = c.get(name, [])
        total = len(items)
        done = sum(1 for i in items if i.get("ready"))
        stats[name] = {"total": total, "done": done, "pct": ratio(done, total)}

    # shots/segments are "done" as soon as they exist
    for name in ("segments", "shots"):
        total = len(c.get(name, []))
        stats[name] = {"total": total, "done": total, "pct": 100 if total else 0}

    n_bgm = len(c.get("bgm", []))
    stats["bgm"] = {"total": max(1, n_bgm), "done": n_bgm, "pct": 100 if n_bgm else 0}

    return stats


def build_overall_progress(state: dict[str, Any], stats: dict[str, Any] | None = None) -> int:
    """A single 0-100 number for the headline progress bar.

    Weighted by roughly how much wall-clock each phase costs, so the bar does
    not sit at 90% through the entire (slow) render stage.
    """
    s = stats if stats is not None else build_stats(state)
    weights = {
        "segments": 5,
        "shots": 10,
        "narration": 15,
        "footage": 15,
        "spotlight": 10,
        "render": 30,
        "bgm": 5,
    }
    total_w = sum(weights.values())
    acc = sum(weights[k] * s.get(k, {}).get("pct", 0) for k in weights)
    pct = int(round(acc / total_w))
    stitch = _as_dict(state.get("stitch_result"))
    if (
        state.get("final_video_path")
        or state.get("stitch_pending")
        or stitch.get("deferred")
    ):
        return 100
    # Reserve the last 2% for a verified final stitch.
    return min(98, pct)


def build_snapshot(state: dict[str, Any], selected: list[str] | None = None) -> dict[str, Any]:
    """Full progress payload pushed to the frontend."""
    clips = build_clips(state)
    stats = build_stats(state, clips)
    order = [t for t in (selected or list(ALL_TRACKS)) if t in ALL_TRACKS]
    stitch = _as_dict(state.get("stitch_result"))
    return {
        "track_order": order,
        "labels": {t: TRACK_LABELS[t] for t in order},
        "clips": {t: clips.get(t, []) for t in order},
        "stats": {t: stats.get(t, {}) for t in order},
        "total_duration": clips["total_duration"],
        "overall_pct": build_overall_progress(state, stats),
        "music_style": str(state.get("music_style", "") or ""),
        "final_video_path": str(state.get("final_video_path", "") or ""),
        "stitch": {
            "status": str(stitch.get("status", "") or ""),
            "concat_status": str(stitch.get("concat_status", "") or ""),
            "ready_shot_count": stitch.get("ready_shot_count", 0),
            "shot_count": stitch.get("shot_count", 0),
            "validation_errors": _as_list(stitch.get("validation_errors"))[:10],
        },
    }
