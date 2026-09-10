"""Visual pacing planner for the newvis0907 trustworthy cut.

Two responsibilities, both pure:

1. :func:`plan_visual_playlist` — turn a primary clip + retrieval candidates
   into an ordered, non-repeating visual playlist that covers a shot's TTS
   duration. It obeys the trustworthy-cut rules:

   * every subclip's final duration is ``>= 2s`` inside a ``2-6s`` band;
   * each source is used at most once (and never a globally-occupied one);
   * natural speed is preferred, then slow motion down to a ``1/3`` floor
     (a source is stretched at most 3x — never a freeze, last-frame hold or
     modulo loop);
   * a leftover ``< 2s`` is rebalanced into siblings rather than emitted;
   * when coverage is still impossible it returns ``coverage_status="short"``
     with an explicit ``deficit_sec`` and ``needs_retrieval`` / ``needs_split``
     instead of faking ``covered``.

2. Model-shot maths: :func:`minimum_model_shots`, :func:`analyze_shot_pacing`
   and :func:`split_long_shot_plan` (character-conserving split of over-long
   shots on semantic / punctuation boundaries).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from . import footage as footage_mod
from .subtitles import _compact_text, _split_sentences

DEFAULT_MIN_SUBCLIP = 2.0
DEFAULT_MAX_SUBCLIP = 6.0
DEFAULT_MIN_PLAYBACK_RATE = 1.0 / 3.0  # source stretched at most 3x
MODEL_SHOT_MAX_SEC = 15.0
MODEL_SHOT_MIN_SEC = 8.0
LONG_VIDEO_THRESHOLD_SEC = 240.0
_TOL = 1e-6


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_text(value: Any) -> str:
    return str(value or "")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _source_duration(item: dict[str, Any]) -> float:
    for key in ("source_duration_sec", "source_duration", "duration_sec", "duration", "clip_duration_sec"):
        val = _as_float(item.get(key), 0.0)
        if val > 0:
            return val
    return 0.0


def _score_of(item: dict[str, Any]) -> float:
    for key in ("score", "match_score"):
        if item.get(key) is not None:
            return _as_float(item.get(key), 0.0)
    return 0.0


# ---------------------------------------------------------------------------
# playlist planning
# ---------------------------------------------------------------------------
def _emit_clip(
    *,
    source: dict[str, Any],
    idx: int,
    shot_id: str,
    source_used: float,
    final: float,
    cursor: float,
) -> dict[str, Any]:
    source_in = max(0.0, _as_float(source.get("source_in_sec"), _as_float(source.get("source_in"), 0.0)))
    source_out = source_in + max(0.0, source_used)
    rate = (source_used / final) if final > 0 else 1.0
    path = _as_text(source.get("clip_path") or source.get("path"))
    sid_prefix = f"{shot_id}::subclip::" if shot_id else "subclip::"
    return {
        "subclip_id": _as_text(source.get("subclip_id")) or f"{sid_prefix}{idx}",
        "clip_id": footage_mod.clip_id(source) or path or f"{sid_prefix}{idx}",
        "path": path,
        "clip_path": path,
        "source_duration_sec": round(_source_duration(source), 4),
        "source_in_sec": round(source_in, 4),
        "source_out_sec": round(source_out, 4),
        "source_used_duration_sec": round(source_used, 4),
        "final_duration_sec": round(final, 4),
        "playback_rate": round(rate, 6),
        "offset_sec": round(cursor, 4),
        "start_sec": round(cursor, 4),
        "end_sec": round(cursor + final, 4),
        "score": source.get("score", source.get("match_score")),
        "query": _as_text(source.get("query") or source.get("visual_query")),
        "retrieval_rank": int(_as_float(source.get("retrieval_rank", source.get("rank")), 0.0)) or idx,
        "fingerprint": _as_text(source.get("fingerprint")),
        "user_override": bool(source.get("user_override", False)),
        "replaced": bool(source.get("replaced", False)),
    }


def _order_sources(
    sources: list[dict[str, Any]],
    min_subclip: float,
    *,
    prefer_retrieval_order: bool = False,
) -> list[dict[str, Any]]:
    """Primary stays first; the rest prefer natural-speed capable, higher score."""
    if not sources:
        return []
    if prefer_retrieval_order:
        # "combined videos as shot units": consume retrieval list in stable order.
        return sorted(
            sources,
            key=lambda s: (
                int(_as_float(s.get("retrieval_rank", s.get("rank")), 10_000)),
                -_score_of(s),
                _as_text(s.get("clip_id") or s.get("clip_path") or s.get("path")),
            ),
        )
    head, tail = sources[0], sources[1:]
    tail_sorted = sorted(
        tail,
        key=lambda s: (0 if _source_duration(s) >= min_subclip - _TOL else 1, -_score_of(s)),
    )
    return [head, *tail_sorted]


def _chunk_for(remaining: float, has_more: bool, min_subclip: float, max_subclip: float) -> float:
    """Pick a target chunk that never strands a ``< min_subclip`` remainder."""
    if remaining <= max_subclip + _TOL:
        return remaining
    leftover = remaining - max_subclip
    if leftover < min_subclip - _TOL and has_more:
        # Leave exactly ``min_subclip`` for the next source instead of a tiny tail.
        return remaining - min_subclip
    return max_subclip


def _apply_slowdown(clips: list[dict[str, Any]], deficit: float, min_rate: float) -> float:
    """Stretch existing subclips (rate down to ``min_rate``) to absorb a deficit."""
    remaining = deficit
    for clip in clips:
        if remaining <= _TOL:
            break
        source_used = _as_float(clip["source_used_duration_sec"])
        if source_used <= 0:
            continue
        max_final = source_used / min_rate  # 3x when min_rate == 1/3
        headroom = max_final - _as_float(clip["final_duration_sec"])
        if headroom <= _TOL:
            continue
        add = min(headroom, remaining)
        new_final = _as_float(clip["final_duration_sec"]) + add
        clip["final_duration_sec"] = round(new_final, 4)
        clip["playback_rate"] = round(source_used / new_final, 6)
        remaining -= add
    return max(0.0, remaining)


def _reflow_offsets(clips: list[dict[str, Any]]) -> float:
    cursor = 0.0
    for clip in clips:
        final = _as_float(clip["final_duration_sec"])
        clip["offset_sec"] = round(cursor, 4)
        clip["start_sec"] = round(cursor, 4)
        clip["end_sec"] = round(cursor + final, 4)
        cursor += final
    return cursor


def plan_visual_playlist(
    primary: dict[str, Any] | None,
    candidates: list[dict[str, Any]] | None,
    target_duration: float,
    occupied: "footage_mod.Occupancy | None" = None,
    *,
    shot_id: str = "",
    visual_threshold: float = 0.9,
    min_subclip: float = DEFAULT_MIN_SUBCLIP,
    max_subclip: float = DEFAULT_MAX_SUBCLIP,
    min_playback_rate: float = DEFAULT_MIN_PLAYBACK_RATE,
    prefer_retrieval_order: bool = False,
    visual_probe: Any = None,
) -> dict[str, Any]:
    """Plan the visual playlist for one shot. See module docstring for rules."""
    # Finiteness guard: a NaN / inf / negative target is never a real duration.
    # Return a structured, strict-JSON-serializable ``invalid`` result (no
    # NaN/Inf leaks) instead of silently coercing to a bogus ``covered``.
    raw_target = _as_float(target_duration, float("nan"))
    if not math.isfinite(raw_target) or raw_target < 0:
        return {
            "clips": [],
            "target_duration_sec": 0.0,
            "covered_duration_sec": 0.0,
            "coverage_status": "invalid",
            "deficit_sec": 0.0,
            "needs_retrieval": True,
            "needs_split": False,
            "uses_multiple_clips": False,
            "quality_warning": True,
            "warnings": ["target_duration 非有限（NaN/Inf）或为负，判为 invalid"],
        }
    target = max(0.0, raw_target)

    raw_sources: list[dict[str, Any]] = []
    if isinstance(primary, dict) and (primary.get("clip_path") or primary.get("path") or primary.get("clip_id")):
        raw_sources.append(primary)
    raw_sources.extend([c for c in (candidates or []) if isinstance(c, dict)])

    # Drop globally-occupied and duplicate sources (never twice in the film).
    sources = footage_mod.filter_unused_candidates(
        raw_sources, occupied, visual_threshold=visual_threshold, visual_probe=visual_probe
    )
    sources = _order_sources(
        sources,
        min_subclip,
        prefer_retrieval_order=prefer_retrieval_order,
    )

    warnings: list[str] = []

    def _empty(status: str, deficit: float, needs_retrieval: bool, needs_split: bool) -> dict[str, Any]:
        return {
            "clips": [],
            "target_duration_sec": round(target, 4),
            "covered_duration_sec": 0.0,
            "coverage_status": status,
            "deficit_sec": round(max(0.0, deficit), 4),
            "needs_retrieval": needs_retrieval,
            "needs_split": needs_split,
            "uses_multiple_clips": False,
            "quality_warning": False,
            "warnings": warnings,
        }

    if target <= _TOL:
        return _empty("covered", 0.0, False, False)
    if not sources:
        warnings.append("没有可用素材，需检索或拆分")
        return _empty("short", target, True, target > MODEL_SHOT_MAX_SEC)

    # ── target below the 2s floor: one flagged short model shot ─────────────
    if target < min_subclip - _TOL:
        src = sources[0]
        src_dur = _source_duration(src)
        used = min(src_dur, target) if src_dur > 0 else target
        clip = _emit_clip(
            source=src, idx=1, shot_id=shot_id,
            source_used=used if used > 0 else target, final=target, cursor=0.0,
        )
        warnings.append(
            f"目标时长 {target:.2f}s < 2s 正常子镜规则，使用一个覆盖目标的短模型镜（quality warning）"
        )
        return {
            "clips": [clip],
            "target_duration_sec": round(target, 4),
            "covered_duration_sec": round(target, 4),
            "coverage_status": "covered",
            "deficit_sec": 0.0,
            "needs_retrieval": False,
            "needs_split": False,
            "uses_multiple_clips": False,
            "quality_warning": True,
            "warnings": warnings,
        }

    # ── natural-speed greedy fill ───────────────────────────────────────────
    clips: list[dict[str, Any]] = []
    cursor = 0.0
    idx = 0
    for pos, src in enumerate(sources):
        if cursor >= target - _TOL:
            break
        src_dur = _source_duration(src)
        if src_dur <= 0:
            continue
        remaining = target - cursor
        has_more = pos < len(sources) - 1
        desired = _chunk_for(remaining, has_more, min_subclip, max_subclip)

        if src_dur >= desired - _TOL:
            source_used, final = desired, desired            # natural, exact chunk
        elif src_dur >= min_subclip - _TOL:
            source_used, final = src_dur, src_dur            # natural, full (shorter) clip
        elif src_dur * (1.0 / min_playback_rate) >= min_subclip - _TOL:
            # Too short for a 2s natural cut: stretch just enough (>=2s, within 3x).
            source_used = src_dur
            final = min(max(desired, min_subclip), src_dur / min_playback_rate)
        else:
            continue  # unusably short even at max stretch — skip, never loop
        idx += 1
        clips.append(_emit_clip(
            source=src, idx=idx, shot_id=shot_id,
            source_used=source_used, final=final, cursor=cursor,
        ))
        cursor += final

    if not clips:
        warnings.append("素材过短，无法组成合法子镜，需检索或拆分")
        return _empty("short", target, True, target > MODEL_SHOT_MAX_SEC)

    covered = sum(_as_float(c["final_duration_sec"]) for c in clips)
    slowed_in_fill = any(_as_float(c["playback_rate"]) < 1.0 - _TOL for c in clips)

    # ── slow-motion fixup to absorb any shortfall (never a loop/freeze) ─────
    needs_retrieval = slowed_in_fill
    if covered < target - _TOL:
        leftover = _apply_slowdown(clips, target - covered, min_playback_rate)
        covered = sum(_as_float(c["final_duration_sec"]) for c in clips)
        needs_retrieval = True
        if leftover > _TOL:
            warnings.append(
                f"素材总量不足，慢放至 {1.0/min_playback_rate:.0f}x 上限后仍缺 {leftover:.2f}s"
            )
            _reflow_offsets(clips)
            return {
                "clips": clips,
                "target_duration_sec": round(target, 4),
                "covered_duration_sec": round(covered, 4),
                "coverage_status": "short",
                "deficit_sec": round(leftover, 4),
                "needs_retrieval": True,
                "needs_split": target > MODEL_SHOT_MAX_SEC,
                "uses_multiple_clips": len({c["clip_id"] for c in clips}) > 1,
                "quality_warning": bool(warnings),
                "warnings": warnings,
            }

    _reflow_offsets(clips)
    return {
        "clips": clips,
        "target_duration_sec": round(target, 4),
        "covered_duration_sec": round(covered, 4),
        "coverage_status": "covered",
        "deficit_sec": 0.0,
        "needs_retrieval": needs_retrieval,
        "needs_split": False,
        "uses_multiple_clips": len({c["clip_id"] for c in clips}) > 1,
        "quality_warning": bool(warnings),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# model-shot maths
# ---------------------------------------------------------------------------
def minimum_model_shots(total_duration: float) -> int:
    """Minimum model-shot count for a video of ``total_duration`` seconds.

    For >= 240s the floor is ``max(20, ceil(duration / 15))`` so long videos are
    never covered by a handful of over-long shots; shorter videos just use
    ``ceil(duration / 15)`` (target model shots 8-15s).
    """
    duration = max(0.0, _as_float(total_duration, 0.0))
    base = max(1, math.ceil(duration / MODEL_SHOT_MAX_SEC)) if duration > 0 else 0
    if duration >= LONG_VIDEO_THRESHOLD_SEC:
        return max(20, base)
    return base


def _total_duration_of(state: dict[str, Any]) -> float:
    for key in ("total_duration_sec", "total_duration", "final_duration_sec"):
        val = _as_float(state.get(key), 0.0)
        if val > 0:
            return val
    total = 0.0
    shot_assets = _as_dict(state.get("shot_assets"))
    for shot in _as_list(state.get("shot_plan")):
        if not isinstance(shot, dict):
            continue
        assets = _as_dict(shot_assets.get(_as_text(shot.get("shot_id"))))
        audio = _as_float(_as_dict(assets.get("audio_meta")).get("duration_sec"))
        total += audio if audio > 0 else _as_float(shot.get("duration_sec"))
    return total


def analyze_shot_pacing(state: dict[str, Any]) -> dict[str, Any]:
    """Report pacing quality issues (pure): over-long shots, sub-2s subclips,
    and too-few model shots for the video length."""
    state = _as_dict(state)
    shot_plan = [s for s in _as_list(state.get("shot_plan")) if isinstance(s, dict)]
    shot_assets = _as_dict(state.get("shot_assets"))
    issues: list[dict[str, Any]] = []

    for shot in shot_plan:
        sid = _as_text(shot.get("shot_id"))
        duration = _as_float(shot.get("duration_sec"))
        if duration > MODEL_SHOT_MAX_SEC + _TOL:
            issues.append({
                "code": "shot_too_long",
                "severity": "warn",
                "field": sid,
                "message": f"镜头 {sid} 时长 {duration:.1f}s 超过 {MODEL_SHOT_MAX_SEC:.0f}s，建议拆分",
            })
        for clip in footage_mod.iter_playlist_clips(shot_assets.get(sid)):
            final = _as_float(clip.get("final_duration_sec"), _as_float(clip.get("used_duration_sec")))
            if 0 < final < DEFAULT_MIN_SUBCLIP - _TOL:
                issues.append({
                    "code": "subclip_too_short",
                    "severity": "warn",
                    "field": _as_text(clip.get("subclip_id")) or sid,
                    "message": f"子镜 {clip.get('subclip_id') or sid} 时长 {final:.2f}s < 2s",
                })

    total_duration = _total_duration_of(state)
    required = minimum_model_shots(total_duration)
    if required and len(shot_plan) < required:
        issues.append({
            "code": "too_few_model_shots",
            "severity": "warn",
            "field": "",
            "message": (
                f"模型镜数量 {len(shot_plan)} 少于 {total_duration:.0f}s 视频所需的最少 {required} 个"
            ),
        })

    return {
        "status": "ok" if not issues else "needs_attention",
        "issues": issues,
        "total_duration_sec": round(total_duration, 3),
        "model_shot_count": len(shot_plan),
        "minimum_model_shots": required,
    }


# ---------------------------------------------------------------------------
# long-shot splitting (character-conserving)
# ---------------------------------------------------------------------------
def _unique_child_id(base: str, index: int, used: set[str]) -> str:
    candidate = base if index == 0 else f"{base}__{index + 1}"
    suffix = index + 1
    while candidate in used or not candidate:
        suffix += 1
        candidate = f"{base}__{suffix}"
    used.add(candidate)
    return candidate


def _group_sentences_by_budget(sentences: list[str], budget: int) -> list[str]:
    """Greedily merge sentences into chunks of at most ``budget`` characters."""
    if not sentences:
        return []
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + len(sentence) <= budget:
            current += sentence
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def split_long_shot_plan(
    state: dict[str, Any],
    max_shot_sec: float = MODEL_SHOT_MAX_SEC,
) -> list[dict[str, Any]]:
    """Split shots longer than ``max_shot_sec`` on semantic/punctuation
    boundaries, keeping the merged narration character-identical and preserving
    segment / order / visual query. Returns a new shot plan list."""
    state = _as_dict(state)
    shot_plan = [s for s in _as_list(state.get("shot_plan")) if isinstance(s, dict)]
    used_ids: set[str] = {_as_text(s.get("shot_id")) for s in shot_plan if _as_text(s.get("shot_id"))}

    out: list[dict[str, Any]] = []
    for shot in shot_plan:
        duration = _as_float(shot.get("duration_sec"))
        narration = _as_text(shot.get("vocal_script"))
        compact = _compact_text(narration)
        if duration <= max_shot_sec + _TOL or len(compact) <= 1:
            out.append(shot)
            continue

        sentences = _split_sentences(compact) or [compact]
        total_chars = sum(len(s) for s in sentences) or 1
        # chars that fit inside one <=max_shot_sec chunk
        budget = max(1, int(math.floor(total_chars * max_shot_sec / duration)))
        chunks = _group_sentences_by_budget(sentences, budget)
        if len(chunks) <= 1:
            out.append(shot)
            continue

        base_id = _as_text(shot.get("shot_id")) or f"shot_{len(out) + 1}"
        used_ids.discard(base_id)
        chunk_chars = [len(c) for c in chunks]
        assigned = 0.0

        # Physical range provenance recompute: a shot may carry a timeline
        # ``[start_sec, end_sec]`` and physical ``source_shot_ids``. Copied
        # verbatim onto every child these become inconsistent (overlapping /
        # duplicated). Recompute start/end proportionally within the parent range
        # and clear source-shot provenance that can't be attributed per chunk.
        p_start = _as_float(shot.get("start_sec"))
        p_end = _as_float(shot.get("end_sec"))
        has_range = (
            "start_sec" in shot and "end_sec" in shot
            and math.isfinite(p_start) and math.isfinite(p_end) and p_end > p_start
        )
        span = (p_end - p_start) if has_range else 0.0
        dur_cursor = 0.0
        for i, chunk in enumerate(chunks):
            child = dict(shot)
            child["shot_id"] = _unique_child_id(base_id, i, used_ids)
            child["vocal_script"] = chunk
            if i == len(chunks) - 1:
                child_duration = round(max(0.0, duration - assigned), 4)
            else:
                child_duration = round(duration * chunk_chars[i] / total_chars, 4)
                assigned += child_duration
            child["duration_sec"] = child_duration

            if has_range and duration > 0:
                cs = p_start + span * (dur_cursor / duration)
                ce = p_end if i == len(chunks) - 1 else p_start + span * ((dur_cursor + child_duration) / duration)
                child["start_sec"] = round(cs, 4)
                child["end_sec"] = round(ce, 4)
            else:
                child.pop("start_sec", None)
                child.pop("end_sec", None)
            dur_cursor += child_duration

            for phys in ("source_shot_ids", "physical_shot_ids", "physical_shots",
                         "shot_windows", "source_shot_id", "covered_shot_ids"):
                if phys in child:
                    child[phys] = [] if isinstance(child.get(phys), list) else None

            child["split_from"] = base_id
            child["split_index"] = i
            out.append(child)
    return out


__all__ = [
    "analyze_shot_pacing",
    "minimum_model_shots",
    "plan_visual_playlist",
    "split_long_shot_plan",
]
