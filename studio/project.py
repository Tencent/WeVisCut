"""Editable project: the durable artifact a finished run leaves behind.

A run writes ``final_state.json`` into its project directory. That file is the
handoff point between "the agent generated a video" and "the user edits it":
it carries the shot plan, per-shot assets, BGM and the final video path.

This module owns reading it back, turning it into a state dict the agent's own
tools can operate on, and versioning it so every edit is reversible.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("aicut_studio.project")

STATE_FILENAME = "final_state.json"
VERSIONS_DIRNAME = "versions"


class ProjectError(RuntimeError):
    """Project could not be loaded or saved."""


def _as_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _as_list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else []


@dataclass
class Project:
    """A loaded, editable project."""

    project_id: str
    project_dir: Path
    state: dict[str, Any]
    # shot_id -> set of reasons needing regeneration ("tts"|"footage"|"decoration"|"render")
    dirty: dict[str, set[str]] = field(default_factory=dict)
    bgm_dirty: bool = False
    # Set by edits that change the final cut without invalidating any single
    # shot's render: reordering, deleting, and BGM/volume changes. Without this
    # such an edit produces no regeneration steps at all and the concatenated
    # video is never rebuilt.
    restitch_dirty: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)

    # ---------------- accessors ----------------
    @property
    def shot_plan(self) -> list[dict[str, Any]]:
        return [s for s in _as_list(self.state.get("shot_plan")) if isinstance(s, dict)]

    @property
    def shot_assets(self) -> dict[str, Any]:
        assets = self.state.setdefault("shot_assets", {})
        return assets if isinstance(assets, dict) else {}

    @property
    def final_video_path(self) -> str:
        return str(self.state.get("final_video_path", "") or "")

    def shot(self, shot_id: str) -> dict[str, Any] | None:
        for s in self.shot_plan:
            if str(s.get("shot_id", "")) == str(shot_id):
                return s
        return None

    def shot_ids(self) -> list[str]:
        return [str(s.get("shot_id", "")) for s in self.shot_plan]

    def assets_for(self, shot_id: str) -> dict[str, Any]:
        a = self.shot_assets.setdefault(str(shot_id), {})
        return a if isinstance(a, dict) else {}

    # ---------------- dirty tracking ----------------
    def mark_dirty(self, shot_id: str, *reasons: str) -> None:
        """Record that a shot needs specific regeneration steps.

        ``render`` is implied by every other reason, since changing audio,
        footage or decoration all invalidate the composed clip.
        """
        bucket = self.dirty.setdefault(str(shot_id), set())
        for r in reasons:
            bucket.add(r)
        if bucket - {"render"}:
            bucket.add("render")
        if "render" in bucket:
            assets = self.assets_for(str(shot_id))
            assets.pop("rendered_path", None)
            assets.pop("approved", None)

    def dirty_for(self, reason: str) -> list[str]:
        return sorted(sid for sid, reasons in self.dirty.items() if reason in reasons)

    def clear_dirty(self) -> None:
        self.dirty.clear()
        self.bgm_dirty = False
        self.restitch_dirty = False

    def is_dirty(self) -> bool:
        return bool(self.dirty) or self.bgm_dirty or self.restitch_dirty

    # ---------------- summaries ----------------
    def digest(self, max_shots: int = 60) -> dict[str, Any]:
        """Compact, LLM-friendly index of the cut.

        Deliberately excludes file paths and raw template HTML: the chat agent
        needs to *locate* content, and long paths would burn context budget.
        """
        shots = self.shot_plan
        rows: list[dict[str, Any]] = []
        offset = 0.0
        for s in shots:
            sid = str(s.get("shot_id", ""))
            assets = _as_dict(self.shot_assets.get(sid))
            audio_dur = float(_as_dict(assets.get("audio_meta")).get("duration_sec", 0) or 0)
            dur = max(audio_dur, float(s.get("duration_sec", 0) or 0) or 8.0)
            row: dict[str, Any] = {
                "shot_id": sid,
                "segment_id": str(s.get("segment_id", "") or ""),
                "start": round(offset, 2),
                "end": round(offset + dur, 2),
                "text": str(s.get("vocal_script", "") or ""),
                "visual_query": str(s.get("visual_query", "") or "")[:60],
                "is_spotlight": bool(s.get("is_spotlight")),
            }
            # Only surfaced once `set_subtitle` made the on-screen text diverge
            # from the narration, so the agent does not re-propose a fix that
            # has already been applied.
            subtitle = str(s.get("subtitle_text", "") or "")
            if subtitle and subtitle != row["text"]:
                row["subtitle"] = subtitle
            if s.get("is_spotlight"):
                deco = _as_dict(assets.get("decoration"))
                row["template"] = {
                    "structure_id": str(deco.get("structure_id", "") or ""),
                    "motion_id": str(deco.get("motion_id", "") or ""),
                    "visual_id": str(deco.get("visual_id", "") or ""),
                }
                slots = _as_dict(deco.get("slots"))
                if slots:
                    row["slots"] = {k: str(v)[:80] for k, v in list(slots.items())[:8]}
            rows.append(row)
            offset += dur

        truncated = len(rows) > max_shots
        return {
            "project_id": self.project_id,
            "total_shots": len(rows),
            "total_duration_sec": round(offset, 2),
            "segments": [
                {
                    "segment_id": str(sg.get("segment_id", "") or ""),
                    "title": str(sg.get("title", "") or ""),
                    "summary": str(sg.get("summary", "") or "")[:120],
                }
                for sg in _as_list(self.state.get("segments")) if isinstance(sg, dict)
            ],
            "music_style": str(self.state.get("music_style", "") or ""),
            "bgm_tracks": [
                str(t.get("name", "") or "") for t in _as_list(self.state.get("bgm_tracks"))
                if isinstance(t, dict)
            ],
            "shots": rows[:max_shots],
            "shots_truncated": truncated,
        }

    def find_shots(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Rank shots by overlap with a query.

        Used to keep the chat agent's context small on long cuts: instead of
        injecting all shots, inject the ones plausibly related to what the user
        just asked about. Character-level overlap works well enough for Chinese
        without pulling in a tokenizer or embedding model.
        """
        q = str(query or "").strip()
        if not q:
            return self.digest()["shots"][:limit]

        q_chars = {ch for ch in q if not ch.isspace()}
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in self.digest(max_shots=10_000)["shots"]:
            haystack = " ".join([
                row.get("text", ""),
                row.get("shot_id", ""),
                row.get("segment_id", ""),
                " ".join(str(v) for v in _as_dict(row.get("slots")).values()),
            ])
            if not haystack.strip():
                continue
            h_chars = {ch for ch in haystack if not ch.isspace()}
            if not h_chars:
                continue
            overlap = len(q_chars & h_chars) / max(1, len(q_chars))
            # exact substring is a much stronger signal than char overlap
            if q in haystack:
                overlap += 1.0
            if overlap > 0:
                scored.append((overlap, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [row for _score, row in scored[:limit]]

    # ---------------- persistence ----------------
    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "project_id": self.project_id,
            "input_text": self.state.get("input_text", ""),
            "segments": self.state.get("segments", []),
            "shot_plan": self.state.get("shot_plan", []),
            "shot_assets": self.state.get("shot_assets", {}),
            "bgm_tracks": self.state.get("bgm_tracks", []),
            "stitch_result": self.state.get("stitch_result", {}),
            "music_style": self.state.get("music_style", ""),
            # Both are set by `set_bgm`. Leaving them out meant a reload
            # resurrected the original style and volume.
            "bgm_query": self.state.get("bgm_query", ""),
            "bgm_volume": self.state.get("bgm_volume"),
            "final_video_path": self.state.get("final_video_path", ""),
            "edit_history": self.history[-200:],
            "saved_at": datetime.now().isoformat(),
        }

    def save(self, snapshot_label: str = "", max_versions: int = 10) -> Path:
        """Write state back, optionally snapshotting the previous version first."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        target = self.project_dir / STATE_FILENAME

        if snapshot_label and target.exists():
            versions = self.project_dir / VERSIONS_DIRNAME
            versions.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in snapshot_label)[:40]
            try:
                self._write_version_project(target, versions / f"{stamp}_{safe}")
            except OSError as exc:
                logger.warning("版本快照失败（继续保存主状态）: %s", exc)
            self._prune_versions(versions, max_versions)

        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self.to_payload(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        tmp.replace(target)  # atomic: never leave a half-written project
        return target

    def _write_version_project(self, state_path: Path, snapshot_dir: Path) -> None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            shutil.copy2(state_path, snapshot_dir / STATE_FILENAME)
            return
        _copy_media_snapshot(payload, snapshot_dir)
        (snapshot_dir / STATE_FILENAME).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _prune_versions(versions_dir: Path, keep: int) -> None:
        try:
            files = sorted(
                [p for p in versions_dir.iterdir() if p.is_dir() or p.suffix == ".json"],
                key=lambda p: p.name,
            )
        except OSError:
            return
        for stale in files[: max(0, len(files) - max(1, keep))]:
            try:
                if stale.is_dir():
                    shutil.rmtree(stale)
                else:
                    stale.unlink()
            except OSError:
                pass

    def list_versions(self) -> list[dict[str, Any]]:
        versions = self.project_dir / VERSIONS_DIRNAME
        if not versions.is_dir():
            return []
        out: list[dict[str, Any]] = []
        candidates = [p for p in versions.iterdir() if p.is_dir() or p.suffix == ".json"]
        for p in sorted(candidates, key=lambda x: x.name, reverse=True):
            try:
                stat = p.stat()
            except OSError:
                continue
            item = {
                "name": p.name,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "kind": "project" if p.is_dir() else "state",
            }
            if p.is_dir():
                item["project_dir"] = str(p)
                try:
                    payload = json.loads((p / STATE_FILENAME).read_text(encoding="utf-8"))
                    item["final_video_path"] = str(payload.get("final_video_path", "") or "")
                    item["has_video"] = bool(item["final_video_path"])
                except (OSError, json.JSONDecodeError):
                    item["final_video_path"] = ""
                    item["has_video"] = False
            out.append(item)
        return out

    def restore_version(self, name: str) -> None:
        """Roll the project back to a snapshot."""
        root = (self.project_dir / VERSIONS_DIRNAME).resolve()
        candidate = (root / name).resolve()
        if root not in candidate.parents:
            raise ProjectError(f"unknown version: {name}")
        state_file = candidate / STATE_FILENAME if candidate.is_dir() else candidate
        if not state_file.is_file():
            raise ProjectError(f"unknown version: {name}")
        payload = json.loads(state_file.read_text(encoding="utf-8"))
        self.state.update({
            "segments": payload.get("segments", []),
            "shot_plan": payload.get("shot_plan", []),
            "shot_assets": payload.get("shot_assets", {}),
            "bgm_tracks": payload.get("bgm_tracks", []),
            "stitch_result": payload.get("stitch_result", {}),
            "music_style": payload.get("music_style", ""),
            "bgm_query": payload.get("bgm_query", ""),
            "final_video_path": payload.get("final_video_path", ""),
        })
        if payload.get("bgm_volume") is not None:
            try:
                self.state["bgm_volume"] = float(payload["bgm_volume"])
            except (TypeError, ValueError):
                pass
        self.clear_dirty()


def _copy_media_snapshot(payload: dict[str, Any], snapshot_dir: Path) -> None:
    media_dir = snapshot_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    def copy_one(path_value: Any, label: str) -> str:
        src = Path(str(path_value or ""))
        if not src.is_file():
            return str(path_value or "")
        suffix = src.suffix or ".bin"
        dst = media_dir / f"{label}{suffix}"
        try:
            shutil.copy2(src, dst)
            return str(dst)
        except OSError as exc:
            logger.warning("复制版本媒体失败 %s -> %s: %s", src, dst, exc)
            return str(path_value or "")

    if payload.get("final_video_path"):
        payload["final_video_path"] = copy_one(payload.get("final_video_path"), "final_video")

    assets = _as_dict(payload.get("shot_assets"))
    for sid, raw_asset in list(assets.items()):
        asset = raw_asset if isinstance(raw_asset, dict) else {}
        safe_sid = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(sid))[:60] or "shot"
        for key in ("rendered_path", "clip_path", "audio_path"):
            if asset.get(key):
                asset[key] = copy_one(asset.get(key), f"{safe_sid}_{key}")
    payload["shot_assets"] = assets


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load(project_dir: str | Path, pipeline_config: dict[str, Any] | None = None) -> Project:
    """Load a project from its directory."""
    pdir = Path(project_dir).resolve()
    state_path = pdir / STATE_FILENAME
    if not state_path.is_file():
        raise ProjectError(f"{STATE_FILENAME} not found in {pdir}")

    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectError(f"cannot read {state_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProjectError(f"{state_path} must contain a JSON object")

    project_id = str(payload.get("project_id") or pdir.name)

    # Rebuild the shape the agent's tools expect.
    state: dict[str, Any] = {
        "mode": "agent_long_horizon",
        "project_id": project_id,
        "output_dir": str(pdir),
        "config": pipeline_config or {},
        "input_text": payload.get("input_text", ""),
        "segments": _as_list(payload.get("segments")),
        "shot_plan": _as_list(payload.get("shot_plan")),
        "shot_assets": _as_dict(payload.get("shot_assets")),
        "bgm_tracks": _as_list(payload.get("bgm_tracks")),
        "stitch_result": _as_dict(payload.get("stitch_result")),
        "music_style": payload.get("music_style", ""),
        "bgm_query": payload.get("bgm_query", ""),
        "final_video_path": payload.get("final_video_path", ""),
        "severity_scores": {},
        "severity_ranked_shots": [],
        "current_stage": "editing",
        "errors": [],
        "trajectory": [],
    }
    if payload.get("bgm_volume") is not None:
        try:
            state["bgm_volume"] = float(payload["bgm_volume"])
        except (TypeError, ValueError):
            pass

    return Project(
        project_id=project_id,
        project_dir=pdir,
        state=state,
        history=_as_list(payload.get("edit_history")),
    )


def delete(project_dir: str | Path, allowed_roots: list[Path]) -> dict[str, Any]:
    """Permanently remove a project directory.

    This deletes rendered video, so it is deliberately hard to point at the
    wrong thing: the target must resolve inside a configured workspace root and
    must look like a project (contain ``final_state.json``). Symlinks are
    resolved before the containment check, so a link planted inside the
    workspace cannot be used to delete something outside it.
    """
    try:
        target = Path(project_dir).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProjectError(f"工程不存在: {project_dir}") from exc

    if not target.is_dir():
        raise ProjectError(f"不是目录: {target}")

    roots = [Path(r).resolve() for r in allowed_roots]
    if not any(root in target.parents for root in roots):
        raise ProjectError(f"拒绝删除工作区之外的路径: {target}")
    if target in roots:
        raise ProjectError("拒绝删除工作区根目录")
    if not (target / STATE_FILENAME).is_file():
        raise ProjectError(f"目录不像一个成片工程（缺少 {STATE_FILENAME}）: {target}")

    freed = _dir_size(target)
    payload = {}
    try:
        payload = json.loads((target / STATE_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass

    shutil.rmtree(target)
    logger.info("deleted project %s (%d bytes)", target, freed)
    return {
        "project_dir": str(target),
        "project_id": str(_as_dict(payload).get("project_id") or target.name),
        "freed_bytes": freed,
    }


def _dir_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def discover(workspace_root: str | Path, limit: int = 100) -> list[dict[str, Any]]:
    """List editable projects under a workspace, newest first."""
    root = Path(workspace_root)
    if not root.is_dir():
        return []

    found: list[dict[str, Any]] = []
    # Projects live at <workspace>/**/<project_id>/final_state.json; bound the
    # walk depth so a large RESULT tree cannot stall the request.
    for state_path in root.glob(f"*/*/{STATE_FILENAME}"):
        found.append(state_path)
    for state_path in root.glob(f"*/{STATE_FILENAME}"):
        found.append(state_path)

    out: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for state_path in found:
        pdir = state_path.parent
        if pdir in seen:
            continue
        seen.add(pdir)
        try:
            stat = state_path.stat()
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        shots = _as_list(payload.get("shot_plan"))
        out.append({
            "project_id": str(payload.get("project_id") or pdir.name),
            "project_dir": str(pdir),
            "title": str(payload.get("input_text", ""))[:60].replace("\n", " ") or "(无标题)",
            "shot_count": len(shots),
            "final_video_path": str(payload.get("final_video_path", "") or ""),
            "has_video": bool(payload.get("final_video_path")),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "modified_ts": stat.st_mtime,
        })

    out.sort(key=lambda x: x["modified_ts"], reverse=True)
    return out[:limit]
