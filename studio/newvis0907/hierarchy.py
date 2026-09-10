"""Pure hierarchy projection for newvis0907."""

from __future__ import annotations

from typing import Any

from .schemas import ChapterView, FootageSubclipView, HierarchyPayload, ModelShotView

UNASSIGNED_SEGMENT_ID = "__unassigned__"
UNASSIGNED_SEGMENT_TITLE = "未分章"
DEFAULT_SHOT_DURATION_SEC = 8.0


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_text(value: Any) -> str:
    return str(value or "")


def _as_bool(value: Any) -> bool:
    return bool(value)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _round_sec(value: float) -> float:
    return round(float(value), 4)


def _dump_model(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model.dict()


def _safe_duration(audio_duration: float, planned_duration: float) -> float:
    if audio_duration > 0:
        return audio_duration
    if planned_duration > 0:
        return planned_duration
    return DEFAULT_SHOT_DURATION_SEC


def _display_subtitle(shot: dict[str, Any]) -> str:
    return (
        _as_text(shot.get("subtitle_display_text"))
        or _as_text(shot.get("subtitle_text"))
        or _as_text(shot.get("vocal_script"))
    )


def _animation_type(shot: dict[str, Any], assets: dict[str, Any]) -> str:
    semantic = _as_dict(shot.get("semantic_animations"))
    primary = _as_dict(semantic.get("primary"))
    kind = _as_text(primary.get("type"))
    if kind:
        return kind
    decoration = _as_dict(assets.get("decoration"))
    return _as_text(decoration.get("structure_id"))


def _quality_status(shot: dict[str, Any], assets: dict[str, Any]) -> str:
    shot_status = _as_text(shot.get("quality_status"))
    if shot_status:
        return shot_status
    quality_summary = _as_dict(assets.get("quality_summary"))
    asset_status = _as_text(quality_summary.get("status"))
    return asset_status or "unknown"


def _pick_playlist_clips(assets: dict[str, Any]) -> list[dict[str, Any]]:
    clip_meta = _as_dict(assets.get("clip_meta"))
    playlist = _as_dict(clip_meta.get("visual_playlist"))
    clips = [c for c in _as_list(playlist.get("clips")) if isinstance(c, dict)]
    if clips:
        return clips

    direct = assets.get("visual_playlist")
    if isinstance(direct, list):
        direct_clips = [c for c in direct if isinstance(c, dict)]
        if direct_clips:
            return direct_clips
    elif isinstance(direct, dict):
        direct_clips = [c for c in _as_list(direct.get("clips")) if isinstance(c, dict)]
        if direct_clips:
            return direct_clips
    return []


def _fallback_subclip(shot_id: str, assets: dict[str, Any]) -> dict[str, Any]:
    clip_meta = _as_dict(assets.get("clip_meta"))
    source_duration = (
        _as_float(clip_meta.get("source_duration_sec"))
        or _as_float(clip_meta.get("duration_sec"))
        or _as_float(assets.get("duration_sec"))
    )
    fallback_id = _as_text(clip_meta.get("subclip_id")) or f"{shot_id}::subclip::1"
    return {
        "subclip_id": fallback_id,
        "clip_id": _as_text(clip_meta.get("clip_id")) or fallback_id,
        "path": _as_text(assets.get("clip_path")) or _as_text(clip_meta.get("path")),
        "source_duration_sec": source_duration,
        "source_in_sec": _as_float(clip_meta.get("source_in_sec")),
        "source_out_sec": _as_float(clip_meta.get("source_out_sec")),
        "used_duration_sec": _as_float(clip_meta.get("used_duration_sec")),
        "query": _as_text(clip_meta.get("query")),
        "score": clip_meta.get("score"),
        "fingerprint": _as_text(clip_meta.get("fingerprint")),
    }


def _infer_playback_rate(explicit: Any, source_used: float, final_duration: float) -> float:
    explicit_rate = _as_float(explicit, 0.0)
    if explicit_rate > 0:
        return explicit_rate
    if source_used > 0 and final_duration > 0:
        return source_used / final_duration
    return 1.0


def _resolve_subclip_start(clip: dict[str, Any], running_cursor: float, shot_duration: float) -> float:
    for key in ("offset_sec", "start_sec", "start"):
        if clip.get(key) is not None:
            start = _as_float(clip.get(key), running_cursor)
            if start < 0:
                return 0.0
            if start > shot_duration:
                return shot_duration
            return start
    return max(0.0, min(running_cursor, shot_duration))


def _resolve_unique_shot_ids(shot_plan: list[dict[str, Any]]) -> list[str]:
    used: set[str] = set()
    resolved: list[str] = []
    for index, shot in enumerate(shot_plan, start=1):
        raw = _as_text(shot.get("shot_id"))
        base = raw if raw else f"shot_{index:02d}"
        candidate = base
        suffix = 2
        while not candidate or candidate in used:
            candidate = f"{base}__{suffix}"
            suffix += 1
        used.add(candidate)
        resolved.append(candidate)
    return resolved


def _build_subclips(
    shot_id: str,
    shot_display_index: int,
    assets: dict[str, Any],
    shot_duration: float,
) -> list[FootageSubclipView]:
    clips = _pick_playlist_clips(assets)
    if not clips:
        fallback = _fallback_subclip(shot_id, assets)
        # If there is no path and no metadata either, treat as no subclips.
        if not fallback["path"] and not fallback["source_duration_sec"] and not fallback["used_duration_sec"]:
            return []
        clips = [fallback]

    subclips: list[FootageSubclipView] = []
    running_cursor = 0.0
    for idx, raw in enumerate(clips, start=1):
        clip = _as_dict(raw)
        path = _as_text(clip.get("path")) or _as_text(clip.get("clip_path")) or _as_text(assets.get("clip_path"))

        source_duration = (
            _as_float(clip.get("source_duration_sec"))
            or _as_float(clip.get("source_duration"))
            or _as_float(clip.get("duration_sec"))
            or _as_float(clip.get("clip_duration_sec"))
        )
        source_in = max(0.0, _as_float(clip.get("source_in_sec"), _as_float(clip.get("source_in"), 0.0)))
        source_out = _as_float(clip.get("source_out_sec"), _as_float(clip.get("source_out"), 0.0))
        if source_out <= 0 and source_duration > 0:
            source_out = source_duration
        if source_out < source_in:
            source_out = source_in
        source_used = max(0.0, source_out - source_in)

        wanted_duration = (
            _as_float(clip.get("used_duration_sec"))
            or _as_float(clip.get("final_duration_sec"))
            or _as_float(clip.get("final_duration"))
            or _as_float(clip.get("duration_sec"))
            or source_used
        )
        if wanted_duration <= 0:
            wanted_duration = max(0.0, shot_duration - running_cursor)

        start = _resolve_subclip_start(clip, running_cursor, shot_duration)
        end = start + max(0.0, wanted_duration)
        if start > shot_duration:
            start = shot_duration
            end = shot_duration
        if end > shot_duration:
            end = shot_duration
        final_duration = max(0.0, end - start)
        running_cursor = max(running_cursor, end)

        subclip_id = _as_text(clip.get("subclip_id")) or f"{shot_id}::subclip::{idx}"
        clip_id = _as_text(clip.get("clip_id")) or subclip_id
        playback_rate = _infer_playback_rate(clip.get("playback_rate"), source_used, final_duration)

        score_val = clip.get("score")
        score = _as_float(score_val) if score_val is not None else None

        subclips.append(
            FootageSubclipView(
                subclip_id=subclip_id,
                display_number=f"{shot_display_index}.{idx}",
                clip_id=clip_id,
                path=path,
                source_duration=_round_sec(source_duration),
                source_in=_round_sec(source_in),
                source_out=_round_sec(source_out),
                final_duration=_round_sec(final_duration),
                playback_rate=_round_sec(playback_rate),
                score=score,
                query=_as_text(clip.get("query")),
                start=_round_sec(start),
                end=_round_sec(end),
                replaced=_as_bool(clip.get("replaced")),
                user_override=_as_bool(clip.get("user_override")),
                fingerprint=_as_text(clip.get("fingerprint")),
            )
        )
    return subclips


def build_hierarchy(state: dict) -> dict:
    """Project ``final_state`` into chapter -> shot -> subclip hierarchy."""
    payload = _as_dict(state)
    raw_segments = [s for s in _as_list(payload.get("segments")) if isinstance(s, dict)]
    shot_plan = [s for s in _as_list(payload.get("shot_plan")) if isinstance(s, dict)]
    shot_assets = _as_dict(payload.get("shot_assets"))
    resolved_ids = _resolve_unique_shot_ids(shot_plan)

    rows: list[dict[str, Any]] = []
    offset = 0.0
    for index, shot in enumerate(shot_plan, start=1):
        sid_raw = _as_text(shot.get("shot_id"))
        sid = resolved_ids[index - 1]
        assets = _as_dict(shot_assets.get(sid_raw))
        if not assets and sid_raw != sid:
            assets = _as_dict(shot_assets.get(sid))
        audio_duration = _as_float(_as_dict(assets.get("audio_meta")).get("duration_sec"))
        planned_duration = _as_float(shot.get("duration_sec"))
        duration = _safe_duration(audio_duration, planned_duration)
        start = offset
        end = start + duration
        rows.append(
            {
                "index": index,
                "shot": shot,
                "assets": assets,
                "shot_id": sid,
                "shot_id_raw": sid_raw,
                "segment_id": _as_text(shot.get("segment_id")),
                "start": start,
                "end": end,
                "duration": duration,
            }
        )
        offset = end

    model_shots: list[ModelShotView] = []
    for row in rows:
        shot = _as_dict(row["shot"])
        assets = _as_dict(row["assets"])
        shot_id = _as_text(row["shot_id"])
        shot_duration = _as_float(row["duration"])
        subclips = _build_subclips(shot_id, int(row["index"]), assets, shot_duration)

        model_shots.append(
            ModelShotView(
                shot_id=shot_id,
                segment_id=_as_text(row["segment_id"]),
                narration=_as_text(shot.get("vocal_script")),
                display_number=f"{int(row['index']):02d}",
                display_subtitle=_display_subtitle(shot),
                start=_round_sec(_as_float(row["start"])),
                end=_round_sec(_as_float(row["end"])),
                duration=_round_sec(shot_duration),
                is_spotlight=_as_bool(shot.get("is_spotlight")),
                animation_type=_animation_type(shot, assets),
                quality_status=_quality_status(shot, assets),
                subclips=subclips,
            )
        )

    by_segment: dict[str, list[ModelShotView]] = {}
    for shot_view in model_shots:
        by_segment.setdefault(shot_view.segment_id, []).append(shot_view)

    known_segment_ids: set[str] = set()
    chapters: list[ChapterView] = []
    display_index = 1
    for seg in raw_segments:
        seg_id = _as_text(seg.get("segment_id")) or f"segment_{display_index:02d}"
        known_segment_ids.add(seg_id)
        shot_views = by_segment.get(seg_id, [])
        if shot_views:
            start = min(s.start for s in shot_views)
            end = max(s.end for s in shot_views)
        else:
            start = 0.0
            end = 0.0

        chapters.append(
            ChapterView(
                segment_id=seg_id,
                title=_as_text(seg.get("title")) or _as_text(seg.get("structure_label")) or seg_id,
                summary=_as_text(seg.get("summary")),
                display_index=display_index,
                model_shot_ids=[s.shot_id for s in shot_views],
                start=_round_sec(start),
                end=_round_sec(end),
            )
        )
        display_index += 1

    unknown_shots = [
        s for s in model_shots
        if not s.segment_id or s.segment_id not in known_segment_ids
    ]
    if unknown_shots:
        chapters.append(
            ChapterView(
                segment_id=UNASSIGNED_SEGMENT_ID,
                title=UNASSIGNED_SEGMENT_TITLE,
                summary="",
                display_index=display_index,
                model_shot_ids=[s.shot_id for s in unknown_shots],
                start=_round_sec(min(s.start for s in unknown_shots)),
                end=_round_sec(max(s.end for s in unknown_shots)),
            )
        )

    total_duration = _round_sec(rows[-1]["end"]) if rows else 0.0
    final_payload = HierarchyPayload(
        chapters=chapters,
        model_shots=model_shots,
        total_duration=total_duration,
    )
    return _dump_model(final_payload)
