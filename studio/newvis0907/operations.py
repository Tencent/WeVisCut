"""State mutations for newvis0907 edits.

Two responsibilities:

1. **apply_operations** — mutate project state for a batch of ops, atomically.
   New 0907 ops (``remove_animation``, ``set_emphasis``, ``replace_subclip``,
   ``set_subclip_timing``, ``set_shot_duration``, ``optimize_cut``) are handled
   here; legacy ops are delegated to :mod:`studio.editor` so the manual UI and
   the chat surface keep sharing one vocabulary. A batch is all-or-nothing: if
   any op fails, the project's state / dirty / history are restored exactly.

2. **compute_digest / verify_postconditions / target_shot_ids** — pure helpers
   the verification layer reads to decide whether an edit actually took effect
   and left everything else untouched.

Rendering / quality strategy is *not* implemented here; these ops only mark the
affected shots dirty so a later regeneration pass knows what to rebuild.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat as stat_mod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from studio import editor as editor_mod
from studio.newvis0907 import animations as animations_mod
from studio.project import Project


def normalize_lexical(text: Any) -> str:
    """Punctuation/whitespace-insensitive key (alphanumeric + CJK only).

    Unlike ``subtitles.normalize_lexical`` this also drops CJK punctuation such
    as ``、`` so that phrase matching is truly punctuation-equivalent.
    """
    return "".join(ch.lower() for ch in str(text or "") if ch.isalnum())

# A cheap-but-real file signal: size + mtime_ns + a hash of the first/last block.
# This distinguishes a freshly regenerated file from a stale one without hashing
# an entire (potentially huge) video.
_FINGERPRINT_BLOCK = 65536


def file_fingerprint(path: str | None) -> str | None:
    """``size:mtime_ns:headtailhash`` for a regular non-empty file, else ``None``.

    ``None`` for a missing / empty / non-regular file, so "no output" and "an
    empty output" both read as *no signal* — a verifier can then require the
    signal to appear (or change) before trusting a render.
    """
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    if not stat_mod.S_ISREG(st.st_mode) or st.st_size == 0:
        return None
    try:
        with open(path, "rb") as fh:
            head = fh.read(_FINGERPRINT_BLOCK)
            if st.st_size > _FINGERPRINT_BLOCK:
                fh.seek(max(0, st.st_size - _FINGERPRINT_BLOCK))
                tail = fh.read(_FINGERPRINT_BLOCK)
            else:
                tail = b""
    except OSError:
        return None
    digest = hashlib.sha256()
    digest.update(head)
    digest.update(tail)
    return f"{st.st_size}:{st.st_mtime_ns}:{digest.hexdigest()[:16]}"


def capture_file_signals(project: Project, shot_ids: list[str] | None = None) -> dict[str, Any]:
    """Real on-disk signals for the final video and each shot's rendered file."""
    ids = shot_ids if shot_ids is not None else project.shot_ids()
    rendered: dict[str, str | None] = {}
    for sid in ids:
        assets = _as_dict(project.shot_assets.get(sid))
        rendered[sid] = file_fingerprint(_as_text(assets.get("rendered_path")) or None)
    return {
        "final": file_fingerprint(project.final_video_path or None),
        "rendered": rendered,
    }

# Ops that mutate footage/render/animation state — everything supported here
# invalidates at least the affected shot's render.
NEWVIS0907_OPS: tuple[str, ...] = (
    "replace_footage",
    "remove_animation",
    "set_emphasis",
    "replace_subclip",
    "set_subclip_timing",
    "set_shot_duration",
    "optimize_cut",
)
LEGACY_OPS: tuple[str, ...] = tuple(sorted(editor_mod.OP_SPEC.keys()))
SUPPORTED_OPS: tuple[str, ...] = tuple(sorted(set(NEWVIS0907_OPS) | set(LEGACY_OPS)))

# Legacy ops that re-render at least one shot clip (per-shot render invalidated).
LEGACY_RENDER_OPS: tuple[str, ...] = (
    "set_narration",
    "set_subtitle",
    "set_spotlight_slots",
    "swap_template",
    "replace_footage",
)
# Ops that only rebuild the concat/mix (restitch) — no shot clip is re-rendered.
# ``reorder_shots`` / ``delete_shot`` are *structural* (they change the shot set
# or order); ``set_bgm`` is restitch-only with no shot target.
STRUCTURAL_OPS: tuple[str, ...] = ("reorder_shots", "delete_shot")
RESTITCH_ONLY_OPS: tuple[str, ...] = ("set_bgm",)
# The union of ops for which verification must require a *fresh per-shot render*.
RENDER_INVALIDATING_OPS: tuple[str, ...] = tuple(
    sorted(set(NEWVIS0907_OPS) | set(LEGACY_RENDER_OPS))
)


class OperationError(ValueError):
    """A batch could not be applied; the project was left unchanged."""


# Project.state keys written by 0907 ops that ``Project.restore_version`` does
# not know about, so a rollback must restore them explicitly.
EXTRA_STATE_KEYS: tuple[str, ...] = ("quality_requests0907",)


def snapshot_extra_state(project: Project) -> dict[str, Any]:
    """Capture the pre-edit value of 0907-only state keys (for rollback)."""
    return {k: copy.deepcopy(project.state[k]) for k in EXTRA_STATE_KEYS if k in project.state}


def restore_extra_state(project: Project, snapshot: dict[str, Any]) -> None:
    """Restore (or clear) the 0907-only state keys from a pre-edit snapshot."""
    for key in EXTRA_STATE_KEYS:
        if key in snapshot:
            project.state[key] = copy.deepcopy(snapshot[key])
        else:
            project.state.pop(key, None)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _as_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _as_list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else []


def _as_text(v: Any) -> str:
    return str(v or "")


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _now_iso() -> str:
    return datetime.now().isoformat()


def _require_shot(project: Project, shot_id: str) -> dict[str, Any]:
    shot = project.shot(shot_id)
    if shot is None:
        raise OperationError(f"镜头不存在: {shot_id}")
    return shot


def _playlist_clips(assets: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Return the *mutable* visual_playlist clip list for a shot's assets."""
    clip_meta = _as_dict(assets.get("clip_meta"))
    playlist = _as_dict(clip_meta.get("visual_playlist"))
    if isinstance(playlist.get("clips"), list):
        return playlist["clips"]
    direct = assets.get("visual_playlist")
    if isinstance(direct, list):
        return direct
    if isinstance(direct, dict) and isinstance(direct.get("clips"), list):
        return direct["clips"]
    return None


def _subclip_id_for(shot_id: str, clip: dict[str, Any], idx: int) -> str:
    return _as_text(clip.get("subclip_id")) or f"{shot_id}::subclip::{idx}"


def _find_subclip(project: Project, subclip_id: str) -> tuple[str, dict[str, Any]] | None:
    """Locate ``(shot_id, clip_dict)`` for a subclip id across all shots."""
    for shot in project.shot_plan:
        sid = _as_text(shot.get("shot_id"))
        assets = _as_dict(project.shot_assets.get(sid))
        clips = _playlist_clips(assets)
        if not clips:
            continue
        for idx, raw in enumerate(clips, start=1):
            if not isinstance(raw, dict):
                continue
            if _subclip_id_for(sid, raw, idx) == subclip_id:
                return sid, raw
    return None


def _subclip_signature(assets: dict[str, Any], shot_id: str) -> list[dict[str, Any]]:
    clips = _playlist_clips(assets) or []
    out: list[dict[str, Any]] = []
    for idx, raw in enumerate(clips, start=1):
        clip = _as_dict(raw)
        out.append({
            "subclip_id": _subclip_id_for(shot_id, clip, idx),
            "path": _as_text(clip.get("path")) or _as_text(clip.get("clip_path")),
            "query": _as_text(clip.get("query")),
            "used_duration_sec": _as_float(clip.get("used_duration_sec")),
            "final_duration_sec": _as_float(clip.get("final_duration_sec")),
            "replaced": bool(clip.get("replaced")),
        })
    return out


# ---------------------------------------------------------------------------
# semantic-animation helpers (shared by set_emphasis / remove_animation)
# ---------------------------------------------------------------------------
def _load_semantic_registry(project: Project) -> dict[str, list[str]]:
    """Read the real Gold Template id lists from the project config (may be empty)."""
    cfg = _as_dict(project.state.get("config"))
    paths = _as_dict(cfg.get("paths"))
    root = _as_text(paths.get("gold_template_root")).strip()
    if not root:
        return {"structures": [], "motions": [], "visuals": []}
    return animations_mod.registry_from_gold_root(root)


def _registry_available(reg: dict[str, list[str]]) -> bool:
    reg = _as_dict(reg)
    return bool(reg.get("structures") and reg.get("motions") and reg.get("visuals"))


def _verify_template_ids(project: Project, primary: dict[str, Any]) -> tuple[bool, str]:
    """Validate a primary anchor's three template ids against the gold registry.

    * No ids claimed → nothing to verify (pure phrase emphasis) → ``(True, "")``.
    * Ids claimed but registry unreachable → fail closed with a reason.
    * Ids claimed and registry reachable → every id must be a real registry id.
    """
    primary = _as_dict(primary)
    structure_id = _as_text(primary.get("structure_id"))
    motion_id = _as_text(primary.get("motion_id"))
    visual_id = _as_text(primary.get("visual_id"))
    if not any((structure_id, motion_id, visual_id)):
        return True, ""
    reg = _load_semantic_registry(project)
    if not _registry_available(reg):
        return False, "registry unreachable, cannot verify template ids"
    ok = (
        structure_id in _as_list(reg.get("structures"))
        and motion_id in _as_list(reg.get("motions"))
        and visual_id in _as_list(reg.get("visuals"))
    )
    return (ok, "" if ok else "template id not found in gold registry")


def _shot_semantic_duration(shot: dict[str, Any], assets: dict[str, Any]) -> float:
    audio = _as_dict(assets.get("audio_meta"))
    return _as_float(audio.get("duration_sec")) or _as_float(shot.get("duration_sec")) or 0.0


def _phrase_present(shot: dict[str, Any], assets: dict[str, Any], phrase: str) -> bool:
    """True when ``phrase`` occurs (punctuation-insensitively) in vocal / display / cues."""
    norm = normalize_lexical(phrase)
    if not norm:
        return False
    haystacks: list[str] = [
        _as_text(shot.get("vocal_script")),
        _as_text(shot.get("subtitle_text")),
        _as_text(shot.get("subtitle_display_text")),
    ]
    for source in (assets, shot):
        for cue in _as_list(_as_dict(source).get("subtitle_cues")):
            if isinstance(cue, dict):
                haystacks.append(_as_text(cue.get("text")))
    return any(norm in normalize_lexical(text) for text in haystacks if text)


def _semantic_anchor_matches(anchor: dict[str, Any], norm_phrase: str, anchor_id: str) -> bool:
    if not isinstance(anchor, dict):
        return False
    if anchor_id:
        if anchor_id in (_as_text(anchor.get("anchor_id")), _as_text(anchor.get("id"))):
            return True
    if norm_phrase:
        anchor_norm = normalize_lexical(_as_text(anchor.get("phrase")))
        legacy_norm = normalize_lexical(_as_text(anchor.get("anchor")))
        if anchor_norm and (norm_phrase in anchor_norm or anchor_norm == norm_phrase):
            return True
        if legacy_norm and norm_phrase == legacy_norm:
            return True
    return False


def _pick_primary(anchors: list[dict[str, Any]]) -> dict[str, Any]:
    real = [a for a in anchors if isinstance(a, dict) and _as_text(a.get("phrase"))]
    if not real:
        return {}
    return max(real, key=lambda a: (_as_float(a.get("priority")), -_as_float(a.get("start_sec"))))


# ---------------------------------------------------------------------------
# new-op appliers
# ---------------------------------------------------------------------------
def _op_remove_animation(project: Project, op: dict[str, Any]) -> dict[str, Any]:
    sid = _as_text(op.get("shot_id"))
    shot = _require_shot(project, sid)
    phrase = _as_text(op.get("phrase")).strip()
    anchor_id = _as_text(op.get("anchor_id")).strip()
    norm_phrase = normalize_lexical(phrase) if phrase else ""
    removed: list[str] = []

    assets = project.assets_for(sid)

    def _match(anchor: dict[str, Any]) -> bool:
        if not phrase and not anchor_id:
            return True  # no selector => clear every animation on the shot
        return _semantic_anchor_matches(anchor, norm_phrase, anchor_id)

    # Filter both the shot-level and asset-level semantic stores.
    for container in (shot, assets):
        sem = container.get("semantic_animations")
        if not isinstance(sem, dict):
            continue
        anchors = [a for a in _as_list(sem.get("anchors")) if isinstance(a, dict)]
        kept = [a for a in anchors if not _match(a)]
        if len(kept) != len(anchors):
            removed.append("anchors")
        primary = _as_dict(sem.get("primary"))
        if primary and _match(primary):
            removed.append("primary")
            primary = {}
        if not primary.get("phrase") and kept:
            primary = _pick_primary(kept)
        sem["anchors"] = kept
        sem["primary"] = primary

    # Legacy decoration lists / slots that embed the matched phrase.
    deco = assets.get("decoration")
    if isinstance(deco, dict) and (phrase or anchor_id):
        for key in ("animations", "semantic_animations", "decorations"):
            val = deco.get(key)
            if isinstance(val, list):
                kept_list = [
                    x for x in val
                    if not (isinstance(x, dict) and norm_phrase
                            and norm_phrase in normalize_lexical(json.dumps(x, ensure_ascii=False)))
                ]
                if len(kept_list) != len(val):
                    removed.append(f"decoration.{key}")
                deco[key] = kept_list
        slots = deco.get("slots")
        if isinstance(slots, dict) and norm_phrase:
            kept_slots = {k: v for k, v in slots.items()
                          if norm_phrase not in normalize_lexical(_as_text(v))}
            if len(kept_slots) != len(slots):
                removed.append("decoration.slots")
            deco["slots"] = kept_slots

    # Re-derive the shot's decoration / spotlight from what remains.
    shot_sem = _as_dict(shot.get("semantic_animations"))
    remaining = _as_list(shot_sem.get("anchors"))
    new_primary = _as_dict(shot_sem.get("primary"))
    duration = _shot_semantic_duration(shot, assets)
    if not remaining and not new_primary.get("phrase"):
        if "decoration" in assets:
            assets.pop("decoration", None)
            removed.append("decoration")
        shot["is_spotlight"] = False
    elif new_primary.get("structure_id"):
        assets["decoration"] = animations_mod.anchor_to_decoration(new_primary, duration)

    project.mark_dirty(sid, "render")
    return {"op": "remove_animation", "ok": True, "shot_id": sid,
            "message": f"{sid} 已移除匹配动画",
            "details": {"phrase": phrase, "anchor_id": anchor_id, "removed": sorted(set(removed))}}


def _op_set_emphasis(project: Project, op: dict[str, Any]) -> dict[str, Any]:
    sid = _as_text(op.get("shot_id"))
    shot = _require_shot(project, sid)
    phrase = _as_text(op.get("phrase")).strip()
    if not phrase:
        raise OperationError("set_emphasis 需要 phrase")
    etype = _as_text(op.get("emphasis_type") or op.get("type")).strip() or "key_point"

    assets = project.assets_for(sid)
    if not _phrase_present(shot, assets, phrase):
        raise OperationError(f"未在旁白/字幕/字幕分句中找到短语「{phrase}」，拒绝标记强调")

    reg = _load_semantic_registry(project)
    plan = animations_mod.build_semantic_animation(
        shot, assets, reg, user_emphasis={"phrase": phrase, "type": etype}
    )
    primary = _as_dict(plan.get("primary"))
    if _registry_available(reg) and not _as_text(primary.get("structure_id")):
        raise OperationError(
            f"未能在真实模板库中为「{phrase}」匹配 structure/motion/visual，拒绝造 ID"
        )

    semantic = {"primary": primary, "anchors": _as_list(plan.get("anchors")), "quality": plan.get("quality")}
    shot["semantic_animations"] = semantic
    assets["semantic_animations"] = semantic

    duration = _shot_semantic_duration(shot, assets)
    if _as_text(primary.get("structure_id")):
        assets["decoration"] = animations_mod.anchor_to_decoration(primary, duration)
    shot["is_spotlight"] = True

    project.mark_dirty(sid, "render")
    return {"op": "set_emphasis", "ok": True, "shot_id": sid,
            "message": f"{sid} 已标记强调「{phrase}」",
            "details": {
                "phrase": phrase,
                "type": etype,
                "structure_id": _as_text(primary.get("structure_id")),
                "decor_start_sec": _as_float(_as_dict(_as_dict(assets.get("decoration")).get("timing")).get("decor_start_sec")),
            }}


def _op_replace_subclip(project: Project, op: dict[str, Any]) -> dict[str, Any]:
    subclip_id = _as_text(op.get("subclip_id"))
    located = _find_subclip(project, subclip_id)
    if located is None:
        raise OperationError(f"子镜头不存在: {subclip_id}")
    sid, clip = located
    query = _as_text(op.get("query")).strip()
    clip_path = _as_text(op.get("clip_path")).strip()
    if not query and not clip_path:
        raise OperationError("replace_subclip 需要 query 或 clip_path")
    if clip_path:
        candidate = Path(clip_path).expanduser().resolve()
        if not candidate.is_file() or candidate.suffix.lower() not in {
            ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v",
        }:
            raise OperationError("replace_subclip 的 clip_path 必须是存在的视频文件")
        clip["clip_path"] = clip_path
        clip["path"] = clip_path
        clip["needs_retrieval"] = False
    elif query:
        # Query-only replacement is a pinned placeholder. The visual retrieval
        # step fills this exact slot; compose must not silently reuse the old
        # path or rebuild the user's sibling subclips.
        clip.pop("clip_path", None)
        clip.pop("path", None)
        clip.pop("clip_id", None)
        clip["needs_retrieval"] = True
    if query:
        clip["query"] = query
        _require_shot(project, sid)["visual_query"] = query
    clip["replaced"] = True
    clip["user_override"] = True
    project.mark_dirty(sid, "footage")
    return {"op": "replace_subclip", "ok": True, "shot_id": sid,
            "message": f"{subclip_id} 素材将被替换",
            "details": {"subclip_id": subclip_id, "query": query, "clip_path": clip_path}}


def _op_replace_footage(project: Project, op: dict[str, Any]) -> dict[str, Any]:
    """0907 replace_footage override.

    We treat retrieved videos as the editable shot units. So a shot-level
    replacement must drop stale playlist/candidates and force a fresh retrieval
    pass under the new query; otherwise old pinned subclips can leak through and
    make "replace shot 2" look like the same video repeating.
    """
    sid = _as_text(op.get("shot_id"))
    shot = _require_shot(project, sid)
    assets = project.assets_for(sid)
    clip_path = _as_text(op.get("clip_path")).strip()
    query = _as_text(op.get("query")).strip()
    if not clip_path and not query:
        raise OperationError("replace_footage 需要 query 或 clip_path")

    clip_meta = _as_dict(assets.get("clip_meta"))
    if clip_path:
        candidate = Path(clip_path).expanduser().resolve()
        if not candidate.is_file() or candidate.suffix.lower() not in {
            ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v",
        }:
            raise OperationError("replace_footage 的 clip_path 必须是存在的视频文件")
        assets["clip_path"] = str(candidate)
        assets["clip_source"] = "manual"
        # Manual path replacement invalidates any generated playlist.
        clip_meta.pop("visual_playlist", None)
        assets.pop("visual_playlist", None)
        assets["retrieval_candidates"] = []
        if query:
            shot["visual_query"] = query
        clip_meta["query"] = shot.get("visual_query", "")
        assets["clip_meta"] = clip_meta
        project.mark_dirty(sid, "render")
        return {
            "op": "replace_footage",
            "ok": True,
            "shot_id": sid,
            "message": f"{sid} 素材已替换为指定文件",
            "details": {"clip_path": str(candidate), "query": _as_text(shot.get("visual_query"))},
        }

    # Query-driven replacement: reset all old visual units and re-retrieve.
    shot["visual_query"] = query
    assets.pop("clip_path", None)
    assets["clip_source"] = "retrieval"
    assets["retrieval_candidates"] = []
    assets.pop("visual_playlist", None)
    clip_meta.pop("visual_playlist", None)
    clip_meta["query"] = query
    assets["clip_meta"] = clip_meta
    project.mark_dirty(sid, "footage")
    return {
        "op": "replace_footage",
        "ok": True,
        "shot_id": sid,
        "message": f"{sid} 将按新描述重新检索素材",
        "details": {"query": query},
    }


# Subclip timing rules shared with the pacing planner.
MIN_SUBCLIP_SEC = 2.0
MIN_PLAYBACK_RATE = 1.0 / 3.0  # source stretched at most 3x
# TTS speed band in which a shot duration change is applied by re-timing the
# narration automatically; outside it the caller must change the narration.
TTS_SPEED_MIN = 0.90
TTS_SPEED_MAX = 1.10


def _clip_final(clip: dict[str, Any]) -> float:
    return (
        _as_float(clip.get("final_duration_sec"))
        or _as_float(clip.get("used_duration_sec"))
        or _as_float(clip.get("duration_sec"))
    )


def _clip_source_used(clip: dict[str, Any]) -> float:
    return (
        _as_float(clip.get("source_used_duration_sec"))
        or _as_float(clip.get("source_duration_sec"))
        or _as_float(clip.get("duration_sec"))
    )


def _set_clip_final(clip: dict[str, Any], final: float) -> None:
    clip["final_duration_sec"] = round(final, 4)
    source_used = _clip_source_used(clip)
    if source_used > 0:
        clip["playback_rate"] = round(source_used / final, 6) if final > 0 else 1.0


def _rebalance_subclips(clips: list[dict[str, Any]], edited_idx: int, new_final: float) -> None:
    """Keep the shot total constant by re-timing sibling subclips.

    No sibling may fall below 2s or below the 1/3 playback-rate floor
    (``final <= 3 * source_used``). Raises :class:`OperationError` if the manual
    duration cannot be absorbed without breaking those constraints.
    """
    finals_before = [_clip_final(c) for c in clips]
    total = sum(finals_before)
    others = [i for i in range(len(clips)) if i != edited_idx]
    if not others:
        _set_clip_final(clips[edited_idx], new_final)
        return

    target = total - new_final
    mins = [MIN_SUBCLIP_SEC for _ in others]
    maxs: list[float] = []
    for i in others:
        su = _clip_source_used(clips[i])
        maxs.append(su / MIN_PLAYBACK_RATE if su > 0 else float("inf"))
    if target < sum(mins) - 1e-6 or target > sum(maxs) + 1e-6:
        raise OperationError(
            "无法在不破坏兄弟子镜约束（每个 >=2s、playback_rate >=1/3）的前提下重平衡该时长"
        )

    cur = [finals_before[i] for i in others]
    cur_sum = sum(cur) or 1.0
    alloc = [max(mins[k], min(maxs[k], target * cur[k] / cur_sum)) for k in range(len(others))]
    diff = target - sum(alloc)
    for _ in range(200):
        if abs(diff) < 1e-6:
            break
        progressed = False
        for k in range(len(others)):
            if abs(diff) < 1e-6:
                break
            if diff > 0:
                room = maxs[k] - alloc[k]
                if room > 1e-9:
                    add = min(room, diff)
                    alloc[k] += add
                    diff -= add
                    progressed = True
            else:
                room = alloc[k] - mins[k]
                if room > 1e-9:
                    sub = min(room, -diff)
                    alloc[k] -= sub
                    diff += sub
                    progressed = True
        if not progressed:
            break
    if abs(diff) > 1e-6:
        raise OperationError(
            "无法在不破坏兄弟子镜约束（每个 >=2s、playback_rate >=1/3）的前提下重平衡该时长"
        )

    _set_clip_final(clips[edited_idx], new_final)
    for k, i in enumerate(others):
        _set_clip_final(clips[i], alloc[k])


def _op_set_subclip_timing(project: Project, op: dict[str, Any]) -> dict[str, Any]:
    subclip_id = _as_text(op.get("subclip_id"))
    located = _find_subclip(project, subclip_id)
    if located is None:
        raise OperationError(f"子镜头不存在: {subclip_id}")
    sid, clip = located
    mode = _as_text(op.get("mode")).strip().lower()
    if mode == "auto":
        clip.pop("used_duration_sec", None)
        clip.pop("final_duration_sec", None)
        clip.pop("playback_rate", None)
        clip["user_override"] = False
        project.mark_dirty(sid, "footage")
        return {"op": "set_subclip_timing", "ok": True, "shot_id": sid,
                "message": f"{subclip_id} 已恢复自动时长",
                "details": {"subclip_id": subclip_id, "mode": "auto"}}

    duration = _as_float(op.get("duration_sec"), -1.0)
    if duration < MIN_SUBCLIP_SEC:
        raise OperationError(f"手工子镜时长必须 >= {MIN_SUBCLIP_SEC:g} 秒，或使用 mode=auto")
    source_used = _clip_source_used(clip)
    if source_used > 0 and duration > source_used / MIN_PLAYBACK_RATE + 1e-6:
        raise OperationError(
            f"{subclip_id} 时长 {duration:g}s 超过素材 3 倍慢放上限（playback_rate 会低于 1/3）"
        )

    assets = _as_dict(project.shot_assets.get(sid))
    clips = _playlist_clips(assets) or []
    try:
        edited_idx = clips.index(clip)
    except ValueError:
        edited_idx = -1
    if edited_idx >= 0 and len(clips) > 1:
        _rebalance_subclips(clips, edited_idx, duration)
    else:
        _set_clip_final(clip, duration)
    # ``used_duration_sec`` records the manual target (what verification checks).
    clip["used_duration_sec"] = round(duration, 4)
    clip["user_override"] = True
    project.mark_dirty(sid, "footage")
    return {"op": "set_subclip_timing", "ok": True, "shot_id": sid,
            "message": f"{subclip_id} 时长调整为 {duration}s，已自动重平衡同镜子镜",
            "details": {"subclip_id": subclip_id, "duration_sec": duration}}


def _op_set_shot_duration(project: Project, op: dict[str, Any]) -> dict[str, Any]:
    sid = _as_text(op.get("shot_id"))
    shot = _require_shot(project, sid)
    duration = _as_float(op.get("duration_sec"), -1.0)
    if duration <= 0:
        raise OperationError("set_shot_duration 的 duration_sec 必须大于 0")

    new_narration = _as_text(op.get("vocal_script") or op.get("narration")).strip()
    if new_narration:
        # An explicit narration change makes the new duration legitimate: the
        # voice is re-recorded rather than time-stretched.
        shot["vocal_script"] = new_narration
        shot["duration_sec"] = duration
        shot["tts_speed"] = 1.0
        project.mark_dirty(sid, "tts")
        project.mark_dirty(sid, "render")
        return {"op": "set_shot_duration", "ok": True, "shot_id": sid,
                "message": f"{sid} 时长调整为 {duration}s（配合新旁白重新配音）",
                "details": {"duration_sec": duration, "narration_changed": True}}

    assets = _as_dict(project.shot_assets.get(sid))
    base = _as_float(_as_dict(assets.get("audio_meta")).get("duration_sec"))
    if base <= 0:
        # No narration audio exists yet (pre-TTS planning): there is nothing to
        # "silently stretch", so the target duration is accepted and TTS will be
        # generated fresh to fit it at natural speed. The band guard only bites
        # once real audio exists and a change would require stretching it.
        shot["tts_speed"] = 1.0
        shot["duration_sec"] = duration
        project.mark_dirty(sid, "tts")
        project.mark_dirty(sid, "render")
        return {"op": "set_shot_duration", "ok": True, "shot_id": sid,
                "message": f"{sid} 时长调整为 {duration}s（尚无配音，按自然语速重配）",
                "details": {"duration_sec": duration, "tts_speed": 1.0}}
    speed = base / duration if duration > 0 else 1.0
    if not (TTS_SPEED_MIN - 1e-6 <= speed <= TTS_SPEED_MAX + 1e-6):
        raise OperationError(
            f"目标时长需要 TTS 语速 {speed:.2f}x，超出自动范围 "
            f"{TTS_SPEED_MIN:.2f}-{TTS_SPEED_MAX:.2f}；请改写旁白或重新分镜，"
            "不能静默拉伸音频"
        )
    shot["tts_speed"] = round(speed, 4)
    shot["duration_sec"] = duration
    project.mark_dirty(sid, "tts")
    project.mark_dirty(sid, "render")
    return {"op": "set_shot_duration", "ok": True, "shot_id": sid,
            "message": f"{sid} 时长调整为 {duration}s（TTS 语速 {speed:.2f}x）",
            "details": {"duration_sec": duration, "tts_speed": round(speed, 4)}}


def _op_optimize_cut(project: Project, op: dict[str, Any]) -> dict[str, Any]:
    scope = _as_text(op.get("scope")).strip() or "global"
    policy = op.get("policy")

    # Resolve the target set *before* recording the request, and reject a scope
    # that names neither "global"/"all" nor a real segment / shot — silently
    # falling back to the whole film would let a typo re-render everything.
    if scope in ("global", "all"):
        target_ids = project.shot_ids()
    else:
        seg_ids = {sid for sid, shot in ((_as_text(s.get("shot_id")), s) for s in project.shot_plan)
                   if _as_text(shot.get("segment_id")) == scope}
        if seg_ids:
            target_ids = sorted(seg_ids)
        elif project.shot(scope) is not None:
            target_ids = [scope]
        else:
            raise OperationError(f"未知优化范围: {scope}（既不是段落也不是镜头）")

    requests = project.state.setdefault("quality_requests0907", [])
    if not isinstance(requests, list):
        requests = []
        project.state["quality_requests0907"] = requests
    requests.append({"scope": scope, "policy": policy, "at": _now_iso()})

    for sid in target_ids:
        project.mark_dirty(sid, "render")
    return {"op": "optimize_cut", "ok": True, "shot_id": "",
            "message": f"已记录质量优化请求（{scope}）",
            "details": {"scope": scope, "quality_requested": True, "shots": target_ids}}


_NEW_APPLIERS: dict[str, Callable[[Project, dict[str, Any]], dict[str, Any]]] = {
    "replace_footage": _op_replace_footage,
    "remove_animation": _op_remove_animation,
    "set_emphasis": _op_set_emphasis,
    "replace_subclip": _op_replace_subclip,
    "set_subclip_timing": _op_set_subclip_timing,
    "set_shot_duration": _op_set_shot_duration,
    "optimize_cut": _op_optimize_cut,
}


# ---------------------------------------------------------------------------
# batch apply (atomic)
# ---------------------------------------------------------------------------
def _snapshot(project: Project) -> dict[str, Any]:
    return {
        "state": copy.deepcopy(project.state),
        "dirty": {k: set(v) for k, v in project.dirty.items()},
        "bgm_dirty": project.bgm_dirty,
        "restitch_dirty": project.restitch_dirty,
        "history": copy.deepcopy(project.history),
    }


def _restore(project: Project, snap: dict[str, Any]) -> None:
    project.state.clear()
    project.state.update(snap["state"])
    project.dirty.clear()
    project.dirty.update({k: set(v) for k, v in snap["dirty"].items()})
    project.bgm_dirty = snap["bgm_dirty"]
    project.restitch_dirty = snap["restitch_dirty"]
    project.history[:] = snap["history"]


def _apply_one(project: Project, op: dict[str, Any], flags: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(op, dict):
        raise OperationError("操作必须是对象")
    name = _as_text(op.get("op"))
    if name in _NEW_APPLIERS:
        return _NEW_APPLIERS[name](project, op)
    if name in editor_mod.OP_SPEC:
        try:
            editor_mod.validate_op(op, flags)
            result = editor_mod._APPLY[name](project, op)
        except editor_mod.EditError as exc:
            raise OperationError(str(exc)) from exc
        return result.as_dict()
    raise OperationError(f"未知操作 {name!r}")


def apply_operations(
    project: Project,
    ops: list[dict[str, Any]],
    enabled_flags: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Apply a batch of ops atomically; return per-op result dicts.

    On any failure the project's ``state``/``dirty``/``history`` are rolled back
    to exactly their pre-batch values and :class:`OperationError` is raised.
    """
    if not ops:
        return []

    snap = _snapshot(project)
    results: list[dict[str, Any]] = []
    try:
        for op in ops:
            results.append(_apply_one(project, op, enabled_flags))
    except OperationError:
        _restore(project, snap)
        raise
    except Exception as exc:  # defensive: never leave a half-applied batch
        _restore(project, snap)
        raise OperationError(f"{type(exc).__name__}: {exc}") from exc

    project.history.append({"at": _now_iso(), "ops": results})
    return results


# ---------------------------------------------------------------------------
# digests & verification helpers (pure)
# ---------------------------------------------------------------------------
def _shot_signature(shot: dict[str, Any], assets: dict[str, Any]) -> str:
    sid = _as_text(shot.get("shot_id"))
    payload = {
        "vocal_script": _as_text(shot.get("vocal_script")),
        "subtitle_text": _as_text(shot.get("subtitle_text")),
        "duration_sec": _as_float(shot.get("duration_sec")),
        "is_spotlight": bool(shot.get("is_spotlight")),
        "segment_id": _as_text(shot.get("segment_id")),
        "semantic_animations": shot.get("semantic_animations"),
        "decoration": _as_dict(assets.get("decoration")),
        "subclips": _subclip_signature(assets, sid),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def compute_digest(project: Project) -> dict[str, Any]:
    """A comparable fingerprint of the edit-relevant project state."""
    shots: dict[str, str] = {}
    for shot in project.shot_plan:
        sid = _as_text(shot.get("shot_id"))
        assets = _as_dict(project.shot_assets.get(sid))
        shots[sid] = _shot_signature(shot, assets)
    return {
        "shots": shots,
        "shot_order": project.shot_ids(),
        "final_video_path": project.final_video_path,
        "final_fingerprint": _as_text(project.state.get("final_fingerprint")),
        "quality_requests": len(_as_list(project.state.get("quality_requests0907"))),
        # Real on-disk signals, so verification can require freshly written
        # outputs rather than trusting a non-empty file or a bumped version.
        "file_signals": capture_file_signals(project),
    }


def target_shot_ids(project: Project, ops: list[dict[str, Any]]) -> set[str]:
    """Shot ids an op batch is expected to touch (owning shots for subclips)."""
    ids: set[str] = set()
    for op in ops:
        name = _as_text(op.get("op"))
        if name in ("optimize_cut",):
            scope = _as_text(op.get("scope")).strip()
            if scope in ("", "global", "all"):
                ids |= set(project.shot_ids())
            else:
                matched = {_as_text(s.get("shot_id")) for s in project.shot_plan
                           if _as_text(s.get("segment_id")) == scope}
                if matched:
                    ids |= matched
                elif project.shot(scope) is not None:
                    ids.add(scope)
                else:
                    ids |= set(project.shot_ids())
        elif name == "reorder_shots":
            ids |= set(project.shot_ids())
        elif "shot_id" in op:
            ids.add(_as_text(op.get("shot_id")))
        elif "shot_ids" in op:
            ids |= {_as_text(x) for x in _as_list(op.get("shot_ids"))}
        elif "subclip_id" in op:
            located = _find_subclip(project, _as_text(op.get("subclip_id")))
            if located is not None:
                ids.add(located[0])
    return {i for i in ids if i}


def ops_need_render(ops: list[dict[str, Any]]) -> bool:
    """Whether any op invalidates rendered output (needs a regeneration pass)."""
    return any(_as_text(o.get("op")) in SUPPORTED_OPS for o in ops)


def _check(name: str, ok: bool, detail: str = "") -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def verify_postconditions(project: Project, ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Confirm, from current state, that each op's intended change took effect."""
    checks: list[dict[str, Any]] = []
    for i, op in enumerate(ops):
        name = _as_text(op.get("op"))
        tag = f"op[{i}]:{name}"
        if name == "set_shot_duration":
            shot = project.shot(_as_text(op.get("shot_id")))
            want = _as_float(op.get("duration_sec"))
            ok = shot is not None and abs(_as_float(shot.get("duration_sec")) - want) < 1e-6
            checks.append(_check(tag, ok, "duration matches" if ok else "duration mismatch"))
        elif name == "set_emphasis":
            shot = project.shot(_as_text(op.get("shot_id"))) or {}
            primary = _as_dict(_as_dict(shot.get("semantic_animations")).get("primary"))
            want = normalize_lexical(_as_text(op.get("phrase")))
            have = normalize_lexical(_as_text(primary.get("phrase")))
            phrase_ok = bool(want) and want == have
            has_type = bool(_as_text(primary.get("type")))
            # time must come from the phrase's real cue interval, never a bare 0.
            time_ok = _as_float(primary.get("end_sec")) > _as_float(primary.get("start_sec"))
            # Template ids, when claimed, must resolve against the real gold
            # registry — never trusted blindly. If the registry is unreachable
            # we fail closed rather than pass implicitly.
            template_ok, template_reason = _verify_template_ids(project, primary)
            ok = phrase_ok and has_type and time_ok and template_ok
            detail = "emphasis set" if ok else (
                "phrase mismatch" if not phrase_ok else
                "type missing" if not has_type else
                "time not from cue" if not time_ok else
                (template_reason or "template invalid")
            )
            checks.append(_check(tag, ok, detail))
        elif name == "remove_animation":
            shot = project.shot(_as_text(op.get("shot_id"))) or {}
            sem = _as_dict(shot.get("semantic_animations"))
            phrase = _as_text(op.get("phrase")).strip()
            norm_phrase = normalize_lexical(phrase)
            primary_norm = normalize_lexical(_as_text(_as_dict(sem.get("primary")).get("phrase")))
            anchors = _as_list(sem.get("anchors"))
            still = bool(norm_phrase) and (
                (primary_norm and norm_phrase in primary_norm)
                or any(isinstance(a, dict)
                       and norm_phrase in normalize_lexical(_as_text(a.get("phrase")))
                       for a in anchors)
            )
            # the decoration must not still carry the removed phrase.
            assets = _as_dict(project.shot_assets.get(_as_text(op.get("shot_id"))))
            deco_norm = normalize_lexical(json.dumps(_as_dict(assets.get("decoration")), ensure_ascii=False))
            deco_still = bool(norm_phrase) and norm_phrase in deco_norm
            ok = (not still) and (not deco_still)
            checks.append(_check(tag, ok,
                                 "animation removed" if ok else
                                 ("anchor still present" if still else "decoration still present")))
        elif name in ("replace_subclip", "set_subclip_timing"):
            located = _find_subclip(project, _as_text(op.get("subclip_id")))
            if located is None:
                checks.append(_check(tag, False, "subclip missing"))
            elif name == "replace_subclip":
                clip = located[1]
                ok = bool(clip.get("replaced")) or _as_text(clip.get("query")) == _as_text(op.get("query"))
                checks.append(_check(tag, ok, "subclip replaced" if ok else "not replaced"))
            else:
                clip = located[1]
                if _as_text(op.get("mode")).strip().lower() == "auto":
                    auto_ok = (
                        "used_duration_sec" not in clip
                        and "final_duration_sec" not in clip
                        and not bool(clip.get("user_override"))
                    )
                    checks.append(_check(tag, auto_ok,
                                         "timing restored to auto" if auto_ok else "auto timing not restored"))
                else:
                    want = _as_float(op.get("duration_sec"))
                    ok = abs(_as_float(clip.get("used_duration_sec")) - want) < 1e-6
                    checks.append(_check(tag, ok, "timing set" if ok else "timing mismatch"))
        elif name == "optimize_cut":
            requests = _as_list(project.state.get("quality_requests0907"))
            ok = bool(requests)
            checks.append(_check(tag, ok, "quality request recorded" if ok else "no request"))
        elif name == "delete_shot":
            sid = _as_text(op.get("shot_id"))
            ok = bool(sid) and project.shot(sid) is None
            checks.append(_check(tag, ok, "shot deleted" if ok else "still present"))
        elif name == "reorder_shots":
            want = [_as_text(x) for x in _as_list(op.get("shot_ids"))]
            ok = project.shot_ids() == want
            checks.append(_check(tag, ok, "order applied" if ok else "order mismatch"))
        else:
            # legacy / generic: target shot must still exist
            sid = _as_text(op.get("shot_id"))
            ok = (not sid) or project.shot(sid) is not None
            checks.append(_check(tag, ok, "target present" if ok else "target missing"))
    return checks
