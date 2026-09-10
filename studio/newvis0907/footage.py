"""Footage identity + global occupancy for the newvis0907 trustworthy cut.

This module is deliberately dependency-free (no ``studio.project`` /
``studio.editor`` imports) so both the offline planner and the render pipeline
can reason about *which real source a clip is* without pulling in the heavy
project machinery.

A single retrieved source is identified by four independent signals, any one of
which makes two clips "the same footage":

1. ``clip_id`` — the retrieval id, when both sides carry one.
2. canonical path — the real on-disk path (symlinks + ``..`` resolved).
3. file content fingerprint — ``size:headtailhash`` of the bytes (no path / no
   mtime, so byte-identical copies dedup).
4. visual fingerprint — an optional embedding/hash; two sources whose
   similarity is ``>= visual_threshold`` are treated as duplicates.

Everything degrades safely: a missing / empty file yields *no* fingerprint (so
"no file" never accidentally equals another "no file"), and a source with only
a clip_id is still a usable identity.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat as stat_mod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

_FINGERPRINT_BLOCK = 65536


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_text(value: Any) -> str:
    return str(value or "")


def _clip_path_of(item: dict[str, Any]) -> str:
    return _as_text(item.get("clip_path") or item.get("path"))


def clip_id(item: dict[str, Any]) -> str:
    """Best-effort retrieval id for a clip/candidate/playlist entry."""
    item = _as_dict(item)
    return _as_text(item.get("clip_id") or item.get("source_id") or item.get("id"))


def canonical_path(path: str | None) -> str:
    """Resolve a path to its real, absolute location without requiring it exist.

    Symlinks and ``..`` are resolved so an alias and its target compare equal;
    a missing file still normalises to a stable absolute string (safe degrade).
    """
    raw = _as_text(path).strip()
    if not raw:
        return ""
    expanded = os.path.expanduser(raw)
    try:
        return os.path.realpath(expanded)
    except OSError:
        return os.path.abspath(expanded)


def file_content_signal(path: str | None) -> str | None:
    """``size:headtailhash`` for a regular non-empty file, else ``None``.

    Deliberately *content-only* (no path, no mtime): two byte-identical copies at
    different paths / with different mtimes must dedup as the same footage. This
    is distinct from ``operations.file_fingerprint`` (which keeps ``mtime_ns`` on
    purpose so a re-encode at the same size is still seen as a fresh render).

    ``None`` for missing / empty / non-regular files, so absence of output is
    never mistaken for a matching signal.
    """
    raw = _as_text(path).strip()
    if not raw:
        return None
    target = os.path.expanduser(raw)
    try:
        st = os.stat(target)
    except OSError:
        return None
    if not stat_mod.S_ISREG(st.st_mode) or st.st_size == 0:
        return None
    try:
        with open(target, "rb") as fh:
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
    return f"{st.st_size}:{digest.hexdigest()[:16]}"


def _coerce_vector(value: Any) -> list[float] | None:
    if isinstance(value, (list, tuple)) and value:
        try:
            return [float(x) for x in value]
        except (TypeError, ValueError):
            return None
    return None


def visual_similarity(a: Any, b: Any) -> float:
    """Similarity in ``[0, 1]`` between two visual fingerprints.

    * numeric vectors -> cosine similarity (clamped to ``[0, 1]``);
    * strings -> 1.0 when equal else 0.0 (perceptual-hash style exact match);
    * anything missing / mismatched -> 0.0.
    """
    if a is None or b is None:
        return 0.0
    va, vb = _coerce_vector(a), _coerce_vector(b)
    if va is not None and vb is not None and len(va) == len(vb):
        dot = sum(x * y for x, y in zip(va, vb))
        na = math.sqrt(sum(x * x for x in va))
        nb = math.sqrt(sum(y * y for y in vb))
        if na <= 0 or nb <= 0:
            return 0.0
        return max(0.0, min(1.0, dot / (na * nb)))
    if isinstance(a, str) and isinstance(b, str):
        return 1.0 if a and a == b else 0.0
    return 0.0


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceIdentity:
    """The dedup-relevant identity of one footage source."""

    clip_id: str = ""
    canonical_path: str = ""
    fingerprint: str | None = None
    visual: Any = None

    def matches(self, other: "SourceIdentity", visual_threshold: float = 0.9) -> bool:
        if self.clip_id and other.clip_id and self.clip_id == other.clip_id:
            return True
        if self.canonical_path and other.canonical_path and self.canonical_path == other.canonical_path:
            return True
        if self.fingerprint and other.fingerprint and self.fingerprint == other.fingerprint:
            return True
        if self.visual is not None and other.visual is not None:
            if visual_similarity(self.visual, other.visual) >= visual_threshold:
                return True
        return False


def source_identity(
    item: dict[str, Any],
    visual_probe: Callable[[dict[str, Any]], Any] | None = None,
) -> SourceIdentity:
    """Build a :class:`SourceIdentity` from a clip / candidate / playlist entry.

    ``visual_probe`` is optional and injectable: when provided it is called with
    the item to obtain a visual fingerprint (embedding/hash). An inline
    ``visual_fingerprint`` on the item is honoured when no probe is given.
    """
    item = _as_dict(item)
    path = _clip_path_of(item)
    visual: Any = item.get("visual_fingerprint")
    if visual_probe is not None:
        try:
            visual = visual_probe(item)
        except Exception:
            visual = item.get("visual_fingerprint")
    fingerprint = _as_text(item.get("fingerprint")) or None
    if fingerprint is None:
        fingerprint = file_content_signal(path)
    return SourceIdentity(
        clip_id=clip_id(item),
        canonical_path=canonical_path(path),
        fingerprint=fingerprint,
        visual=visual,
    )


# ---------------------------------------------------------------------------
# occupancy
# ---------------------------------------------------------------------------
@dataclass
class Occupancy:
    """The set of footage sources already used somewhere in the film."""

    identities: list[SourceIdentity] = field(default_factory=list)

    def add(self, identity: SourceIdentity) -> None:
        self.identities.append(identity)

    def is_occupied(self, identity: SourceIdentity, visual_threshold: float = 0.9) -> bool:
        return any(identity.matches(other, visual_threshold) for other in self.identities)

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self.identities)


def iter_playlist_clips(assets: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield the visual-playlist clip dicts recorded on a shot's assets."""
    assets = _as_dict(assets)
    clip_meta = _as_dict(assets.get("clip_meta"))
    playlist = _as_dict(clip_meta.get("visual_playlist"))
    clips = [c for c in _as_list(playlist.get("clips")) if isinstance(c, dict)]
    if clips:
        yield from clips
        return
    direct = assets.get("visual_playlist")
    if isinstance(direct, list):
        yield from (c for c in direct if isinstance(c, dict))
    elif isinstance(direct, dict):
        yield from (c for c in _as_list(direct.get("clips")) if isinstance(c, dict))


def build_occupancy(
    state: dict[str, Any],
    exclude_shot_ids: Sequence[str] | None = None,
    visual_probe: Callable[[dict[str, Any]], Any] | None = None,
) -> Occupancy:
    """Scan every shot's visual playlist to record globally-used sources.

    ``exclude_shot_ids`` drops a shot's own clips from the occupancy so that
    re-planning that shot does not treat its previous choices as conflicts.
    Because occupancy is global, an unused candidate is by construction absent
    from every other playlist — which is what guarantees "never twice in the
    whole film", "never in the same playlist", and "never on adjacent shots".
    """
    state = _as_dict(state)
    excluded = {str(x) for x in (exclude_shot_ids or [])}
    shot_plan = [s for s in _as_list(state.get("shot_plan")) if isinstance(s, dict)]
    shot_assets = _as_dict(state.get("shot_assets"))

    occ = Occupancy()
    # Prefer iterating the shot plan (so exclusion by shot_id is precise), but
    # also cover any assets whose shot is not in the plan.
    seen_keys: set[str] = set()
    for shot in shot_plan:
        sid = _as_text(shot.get("shot_id"))
        seen_keys.add(sid)
        if sid in excluded:
            continue
        for clip in iter_playlist_clips(shot_assets.get(sid)):
            occ.add(source_identity(clip, visual_probe=visual_probe))
    for sid, assets in shot_assets.items():
        if sid in seen_keys or sid in excluded:
            continue
        for clip in iter_playlist_clips(assets):
            occ.add(source_identity(clip, visual_probe=visual_probe))
    return occ


def filter_unused_candidates(
    candidates: list[dict[str, Any]],
    occupancy: Occupancy | None,
    visual_threshold: float = 0.9,
    visual_probe: Callable[[dict[str, Any]], Any] | None = None,
) -> list[dict[str, Any]]:
    """Return candidates whose source is not already occupied.

    Dedup also happens *within* the returned list, so the same source never
    survives twice even if the caller passed duplicates.
    """
    kept: list[dict[str, Any]] = []
    running = Occupancy(list(occupancy.identities) if occupancy else [])
    for raw in candidates or []:
        if not isinstance(raw, dict):
            continue
        ident = source_identity(raw, visual_probe=visual_probe)
        if running.is_occupied(ident, visual_threshold):
            continue
        kept.append(raw)
        running.add(ident)
    return kept


__all__ = [
    "Occupancy",
    "SourceIdentity",
    "build_occupancy",
    "canonical_path",
    "clip_id",
    "file_content_signal",
    "filter_unused_candidates",
    "iter_playlist_clips",
    "source_identity",
    "visual_similarity",
]
