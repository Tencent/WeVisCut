"""Post-production edit operations.

One operation vocabulary serves both the manual UI and the chat agent, so a
change requested in conversation and the same change made by dragging a clip go
through identical validation and regeneration. That is what keeps the two paths
from drifting.

Two phases, deliberately separate:

1. **apply** — mutate project state and mark affected shots dirty. Pure data,
   no subprocesses, instant. This is what the UI previews.
2. **regenerate** — re-run the *agent's own tools* over the dirty shots, then
   re-stitch. Slow, streams progress.

Splitting them means a user can queue several edits and pay the render cost
once, and it means a bad edit can be rejected before anything expensive runs.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from studio.project import Project

logger = logging.getLogger("aicut_studio.editor")

# ── operation vocabulary ───────────────────────────────────────────────────
# name -> (required args, optional args, human label)
OP_SPEC: dict[str, dict[str, Any]] = {
    "set_narration": {
        "required": ["shot_id", "text"],
        "optional": ["visual_query"],
        "label": "改旁白（重新配音并重检索素材）",
        "risk": "low",
    },
    "set_subtitle": {
        "required": ["shot_id", "text"],
        "optional": [],
        "label": "改字幕（不重新配音）",
        "risk": "low",
    },
    "set_spotlight_slots": {
        "required": ["shot_id", "slots"],
        "optional": [],
        "label": "改模板文字",
        "risk": "low",
    },
    "swap_template": {
        "required": ["shot_id"],
        # ``slots`` is accepted here because switching structure_id clears the
        # old slots: refilling them is part of the same edit, and rejecting the
        # combined form threw away the whole batch.
        "optional": ["structure_id", "motion_id", "visual_id", "slots"],
        "label": "换模板",
        "risk": "high",
    },
    "replace_footage": {
        "required": ["shot_id"],
        "optional": ["query", "clip_path"],
        "label": "换素材",
        "risk": "low",
    },
    "reorder_shots": {
        "required": ["shot_ids"],
        "optional": [],
        "label": "调整镜头顺序",
        "risk": "high",
    },
    "delete_shot": {
        "required": ["shot_id"],
        "optional": [],
        "label": "删除镜头",
        "risk": "high",
    },
    "set_bgm": {
        "required": [],
        "optional": ["query", "music_style", "volume"],
        "label": "换配乐 / 调音量",
        "risk": "low",
    },
}

ALL_OPS = sorted(OP_SPEC.keys())

# Which manual_ops config flag gates each operation.
OP_TO_FLAG: dict[str, str] = {
    "set_narration": "edit_narration",
    "set_subtitle": "edit_subtitle",
    "set_spotlight_slots": "edit_spotlight_slots",
    "swap_template": "swap_template",
    "replace_footage": "replace_footage",
    "reorder_shots": "reorder_shots",
    "delete_shot": "delete_shot",
    "set_bgm": "adjust_bgm",
}


class EditError(ValueError):
    """An operation was rejected before any state was changed."""


@dataclass
class OpResult:
    op: str
    ok: bool
    message: str
    shot_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "op": self.op, "ok": self.ok, "message": self.message,
            "shot_id": self.shot_id, "details": self.details,
        }


def _as_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


# ---------------------------------------------------------------------------
# Gold Template registry
# ---------------------------------------------------------------------------
# Which file proves a template layer actually exists on disk. The renderer
# (`timeline_compose._resolve_*_id`) probes exactly these paths and silently
# falls back to a default when one is missing, so an id that is not here is not
# a template — it is a no-op that looks like a successful edit.
_TEMPLATE_LAYERS: dict[str, tuple[str, str]] = {
    "structure_id": ("structures", "structure.html"),
    "motion_id": ("motions", "motion.js"),
    "visual_id": ("visuals", "tokens.css"),
}

_REGISTRY_CACHE: dict[str, dict[str, list[str]]] = {}


def _gold_template_root(project: Project) -> str:
    paths = _as_dict(_as_dict(project.state.get("config")).get("paths"))
    return str(paths.get("gold_template_root", "") or "").strip()


def template_registry(project: Project) -> dict[str, list[str]]:
    """Available ids per template layer, read from the Gold Template library.

    Returns empty lists when the library is not configured or reachable; callers
    treat that as "cannot verify" and skip validation rather than blocking edits
    on a deployment that renders through some other backend.
    """
    root = _gold_template_root(project)
    if not root:
        return {key: [] for key in _TEMPLATE_LAYERS}

    cached = _REGISTRY_CACHE.get(root)
    if cached is not None:
        return cached

    registry: dict[str, list[str]] = {}
    for key, (subdir, marker) in _TEMPLATE_LAYERS.items():
        base = Path(root) / "generated" / subdir
        try:
            registry[key] = sorted(
                d.name for d in base.iterdir() if (d / marker).is_file()
            )
        except OSError:
            registry[key] = []
    _REGISTRY_CACHE[root] = registry
    return registry


def _suggest(value: str, options: list[str], limit: int = 6) -> list[str]:
    """Ids a caller plausibly meant, so a rejection is actionable."""
    import difflib

    close = difflib.get_close_matches(value, options, n=limit, cutoff=0.4)
    if close:
        return close
    prefix = value.split("_")[0]
    return [o for o in options if o.startswith(prefix)][:limit]


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def validate_op(op: dict[str, Any], enabled_flags: dict[str, Any] | None = None) -> None:
    """Raise `EditError` if the operation is malformed or disabled."""
    if not isinstance(op, dict):
        raise EditError("操作必须是对象")

    name = str(op.get("op", "") or "")
    if name not in OP_SPEC:
        raise EditError(f"未知操作 {name!r}，可用: {', '.join(ALL_OPS)}")

    spec = OP_SPEC[name]
    if enabled_flags is not None:
        flag = OP_TO_FLAG.get(name, "")
        if flag and not bool(enabled_flags.get(flag, True)):
            raise EditError(f"操作 {name} 已在配置中禁用 (editor.manual_ops.{flag})")

    for key in spec["required"]:
        if key not in op or op[key] in (None, ""):
            raise EditError(f"{name} 缺少必填参数 {key}")

    allowed = set(spec["required"]) | set(spec["optional"]) | {"op", "reason"}
    unknown = [k for k in op if k not in allowed]
    if unknown:
        raise EditError(f"{name} 含未知参数: {', '.join(sorted(unknown))}")

    if name == "set_spotlight_slots" and not isinstance(op.get("slots"), dict):
        raise EditError("set_spotlight_slots 的 slots 必须是对象")
    if name == "reorder_shots" and not isinstance(op.get("shot_ids"), list):
        raise EditError("reorder_shots 的 shot_ids 必须是数组")
    if name == "swap_template" and not any(
        op.get(k) for k in ("structure_id", "motion_id", "visual_id")
    ):
        raise EditError("swap_template 至少要指定 structure_id / motion_id / visual_id 之一")
    if name == "replace_footage" and not (op.get("query") or op.get("clip_path")):
        raise EditError("replace_footage 需要 query 或 clip_path")


def high_risk_ops(ops: list[dict[str, Any]], require_confirm_for: list[str]) -> list[str]:
    """Names of submitted operations that the config says need confirmation."""
    gate = set(require_confirm_for or [])
    return sorted({str(o.get("op", "")) for o in ops if str(o.get("op", "")) in gate})


# ---------------------------------------------------------------------------
# phase 1: apply to state
# ---------------------------------------------------------------------------
def _run_sequence(project: Project, ops: list[dict[str, Any]]) -> list[OpResult]:
    """Apply ops in order, converting per-op failures into results."""
    results: list[OpResult] = []
    for op in ops:
        name = str(op["op"])
        try:
            results.append(_APPLY[name](project, op))
        except EditError as exc:
            results.append(OpResult(op=name, ok=False, message=str(exc),
                                    shot_id=str(op.get("shot_id", "") or "")))
        except Exception as exc:  # defensive: never 500 on a bad edit
            logger.exception("edit op %s failed", name)
            results.append(OpResult(op=name, ok=False,
                                    message=f"{type(exc).__name__}: {exc}",
                                    shot_id=str(op.get("shot_id", "") or "")))
    return results


def _trial_copy(project: Project) -> Project:
    """A throwaway clone for dry-running a batch.

    Only project *metadata* is copied — shot plan, asset records, BGM. Media
    files are referenced by path and never duplicated, so this stays cheap even
    for a long cut.
    """
    return Project(
        project_id=project.project_id,
        project_dir=project.project_dir,
        state=copy.deepcopy(project.state),
        dirty={k: set(v) for k, v in project.dirty.items()},
        bgm_dirty=project.bgm_dirty,
        restitch_dirty=project.restitch_dirty,
        history=[],
    )


def precheck_ops(
    project: Project,
    ops: list[dict[str, Any]],
    enabled_flags: dict[str, Any] | None = None,
) -> list[OpResult]:
    """Validate a batch against real project state without mutating anything.

    Runs the batch on a clone so later ops are checked against the state earlier
    ops would produce — otherwise a legitimate pair like
    ``[delete_shot s2, reorder_shots [s1,s3]]`` would be wrongly rejected.
    """
    for op in ops:
        validate_op(op, enabled_flags)
    return _run_sequence(_trial_copy(project), ops)


def apply_ops(
    project: Project,
    ops: list[dict[str, Any]],
    enabled_flags: dict[str, Any] | None = None,
) -> list[OpResult]:
    """Validate and apply operations, marking shots dirty. No rendering.

    A batch is all-or-nothing: it is first dry-run on a clone, and only
    committed if every operation succeeds. A single bad shot id therefore
    cannot leave the project half-edited.
    """
    if not ops:
        return []

    trial = precheck_ops(project, ops, enabled_flags)
    failures = [r for r in trial if not r.ok]
    if failures:
        detail = "; ".join(f"{r.op}: {r.message}" for r in failures[:5])
        raise EditError(f"操作被整批拒绝（{len(failures)}/{len(ops)} 个失败）: {detail}")

    results = _run_sequence(project, ops)
    project.history.append({
        "at": datetime.now().isoformat(),
        "ops": [r.as_dict() for r in results],
    })
    return results


def _require_shot(project: Project, op: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    sid = str(op.get("shot_id", "") or "")
    shot = project.shot(sid)
    if shot is None:
        raise EditError(f"镜头不存在: {sid}（现有 {len(project.shot_ids())} 个镜头）")
    return sid, shot


def _op_set_narration(project: Project, op: dict[str, Any]) -> OpResult:
    sid, shot = _require_shot(project, op)
    text = str(op["text"]).strip()
    if not text:
        raise EditError("旁白文字不能为空")
    old = str(shot.get("vocal_script", "") or "")
    old_query = str(shot.get("visual_query", "") or "")
    shot["vocal_script"] = text
    new_query = str(op.get("visual_query", "") or "").strip() or _visual_query_from_narration(text)
    if new_query:
        shot["visual_query"] = new_query
    # A subtitle override was written against the *old* narration, so keeping it
    # would caption the new audio with the previous wording. Narration is the
    # source of truth: drop the override and let subtitles follow the new script.
    stale_subtitle = str(shot.pop("subtitle_text", "") or "")
    # Invalidate recorded audio and retrieved footage so regeneration cannot
    # silently reuse assets matched to the previous narration.
    assets = project.assets_for(sid)
    assets.pop("audio_path", None)
    assets.pop("audio_meta", None)
    assets.pop("clip_path", None)
    assets.pop("clip_meta", None)
    assets.pop("visual_candidates", None)
    assets["clip_source"] = "retrieval"
    project.mark_dirty(sid, "tts", "footage")
    details = {
        "old": old[:120],
        "new": text[:120],
        "old_visual_query": old_query[:120],
        "visual_query": str(shot.get("visual_query", "") or "")[:120],
    }
    if stale_subtitle and stale_subtitle != text:
        return OpResult("set_narration", True,
                        f"{sid} 旁白已更新，将重新配音并按新画面描述重新检索素材（原先单独设置的字幕已同步为新旁白）",
                        sid, {**details, "dropped_subtitle": stale_subtitle[:120]})
    return OpResult("set_narration", True, f"{sid} 旁白已更新，将重新配音并重新检索素材", sid,
                    details)


def _visual_query_from_narration(text: str, limit: int = 30) -> str:
    """Deterministic fallback for manual edits that only provide narration."""
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit]


def _op_set_subtitle(project: Project, op: dict[str, Any]) -> OpResult:
    sid, shot = _require_shot(project, op)
    text = str(op["text"]).strip()
    if not text:
        raise EditError("字幕文字不能为空")
    old = str(shot.get("subtitle_text") or shot.get("vocal_script", "") or "")
    script = str(shot.get("vocal_script", "") or "")
    # Kept separate from vocal_script so display text can diverge from spoken
    # audio without triggering a TTS regeneration. An override equal to the
    # narration is not a divergence, so store nothing rather than a duplicate
    # that later narration edits would have to remember to clean up.
    if _same_text(text, script):
        shot.pop("subtitle_text", None)
        project.mark_dirty(sid, "render")
        return OpResult("set_subtitle", True, f"{sid} 字幕已与配音保持一致", sid,
                        {"old": old[:120], "new": text[:120], "synced": True})

    shot["subtitle_text"] = text
    project.mark_dirty(sid, "render")
    # This op deliberately skips TTS, so the audience now reads one thing and
    # hears another. That is legitimate for a typo or a length trim, but it is
    # the single easiest way to ship a cut whose captions do not match the
    # voice-over, so say so instead of reporting a quiet success.
    return OpResult("set_subtitle", True,
                    f"{sid} 字幕已更新（不重新配音，配音仍是原文）", sid,
                    {"old": old[:120], "new": text[:120],
                     "diverges_from_narration": True,
                     "narration": script[:120]})


def _same_text(a: str, b: str) -> bool:
    """Compare display/spoken text ignoring whitespace-only differences."""
    return " ".join(str(a or "").split()) == " ".join(str(b or "").split())


def _op_set_spotlight_slots(project: Project, op: dict[str, Any]) -> OpResult:
    sid, shot = _require_shot(project, op)
    if not shot.get("is_spotlight"):
        raise EditError(f"{sid} 不是 spotlight 镜头，没有模板槽位")
    assets = project.assets_for(sid)
    deco = _as_dict(assets.get("decoration"))
    if not deco:
        raise EditError(f"{sid} 尚无装饰数据，请先生成")
    slots = _as_dict(deco.get("slots"))
    incoming = {str(k): ("" if v is None else str(v)) for k, v in op["slots"].items()}
    unknown = [k for k in incoming if slots and k not in slots]
    if unknown and slots:
        raise EditError(
            f"{sid} 模板无这些槽位: {', '.join(unknown)}；可用: {', '.join(sorted(slots))}"
        )
    slots.update(incoming)
    deco["slots"] = slots
    assets["decoration"] = deco
    project.mark_dirty(sid, "decoration")
    return OpResult("set_spotlight_slots", True,
                    f"{sid} 模板文字已更新 ({len(incoming)} 个槽位)", sid,
                    {"slots": incoming})


def _op_swap_template(project: Project, op: dict[str, Any]) -> OpResult:
    sid, shot = _require_shot(project, op)
    if not shot.get("is_spotlight"):
        raise EditError(f"{sid} 不是 spotlight 镜头，无法换模板")
    assets = project.assets_for(sid)
    deco = _as_dict(assets.get("decoration"))
    registry = template_registry(project)
    changed: dict[str, Any] = {}
    for key in ("structure_id", "motion_id", "visual_id"):
        val = op.get(key)
        if not val:
            continue
        val = str(val)
        # The renderer resolves an unknown id to a built-in default without
        # complaining, so an unvalidated swap reports success while the cut keeps
        # the old template. Reject it here instead, and say what is available.
        known = registry.get(key) or []
        if known and val not in known:
            hint = _suggest(val, known)
            raise EditError(
                f"{key}={val!r} 不存在于模板库。"
                + (f"是否想用: {', '.join(hint)}？" if hint else "")
                + f"（共 {len(known)} 个可选，例如 {', '.join(known[:8])}）"
            )
        changed[key] = val
        deco[key] = val
    if not changed:
        raise EditError("没有提供任何模板字段")
    # A different structure exposes different slots; stale values would render
    # into the wrong places, so require them to be refilled.
    if "structure_id" in changed:
        deco.pop("slots", None)
        deco.pop("template_id", None)

    incoming_slots = op.get("slots")
    if incoming_slots is not None:
        if not isinstance(incoming_slots, dict):
            raise EditError("slots 必须是对象，形如 {\"headline\": \"新标题\"}")
        slots = _as_dict(deco.get("slots"))
        # After a structure swap the old slot names are gone, so there is
        # nothing to validate against and any name is provisional; the renderer
        # maps them onto the new structure's schema.
        slots.update({str(k): ("" if v is None else str(v)) for k, v in incoming_slots.items()})
        deco["slots"] = slots
        changed["slots"] = sorted(slots)

    assets["decoration"] = deco
    project.mark_dirty(sid, "decoration")
    return OpResult("swap_template", True, f"{sid} 模板已切换: {changed}", sid, changed)


def _op_replace_footage(project: Project, op: dict[str, Any]) -> OpResult:
    sid, shot = _require_shot(project, op)
    assets = project.assets_for(sid)
    clip_path = str(op.get("clip_path", "") or "")
    if clip_path:
        p = Path(clip_path)
        if not p.is_file():
            raise EditError(f"素材文件不存在: {clip_path}")
        assets["clip_path"] = str(p)
        assets["clip_source"] = "manual"
        project.mark_dirty(sid, "render")
        return OpResult("replace_footage", True, f"{sid} 素材已替换为指定文件", sid,
                        {"clip_path": str(p)})

    query = str(op.get("query", "") or "").strip()
    shot["visual_query"] = query
    assets.pop("clip_path", None)
    assets["clip_source"] = "retrieval"
    project.mark_dirty(sid, "footage")
    return OpResult("replace_footage", True, f"{sid} 将按新描述重新检索素材", sid,
                    {"query": query})


def _op_reorder_shots(project: Project, op: dict[str, Any]) -> OpResult:
    new_order = [str(s) for s in op["shot_ids"]]
    current = project.shot_ids()
    if sorted(new_order) != sorted(current):
        missing = sorted(set(current) - set(new_order))
        extra = sorted(set(new_order) - set(current))
        raise EditError(
            "reorder_shots 必须是现有镜头的一个完整排列"
            + (f"；缺少 {missing}" if missing else "")
            + (f"；多出 {extra}" if extra else "")
        )
    by_id = {str(s.get("shot_id", "")): s for s in project.shot_plan}
    project.state["shot_plan"] = [by_id[sid] for sid in new_order]
    # Clip content is unchanged; only concat order and subtitle timing shift,
    # so a re-stitch is enough and no shot needs re-rendering.
    project.restitch_dirty = True
    return OpResult("reorder_shots", True, f"镜头顺序已调整（{len(new_order)} 个）", "",
                    {"order": new_order})


def _op_delete_shot(project: Project, op: dict[str, Any]) -> OpResult:
    sid, _shot = _require_shot(project, op)
    if len(project.shot_plan) <= 1:
        raise EditError("不能删除最后一个镜头")
    project.state["shot_plan"] = [
        s for s in project.shot_plan if str(s.get("shot_id", "")) != sid
    ]
    project.shot_assets.pop(sid, None)
    project.dirty.pop(sid, None)
    project.restitch_dirty = True
    return OpResult("delete_shot", True, f"{sid} 已删除", sid)


def _op_set_bgm(project: Project, op: dict[str, Any]) -> OpResult:
    details: dict[str, Any] = {}
    if op.get("volume") is not None:
        try:
            vol = float(op["volume"])
        except (TypeError, ValueError):
            raise EditError("volume 必须是数字")
        if not 0.0 <= vol <= 1.0:
            raise EditError("volume 必须在 0.0 ~ 1.0 之间")
        project.state["bgm_volume"] = vol
        details["volume"] = vol

    query = str(op.get("query", "") or "").strip()
    style = str(op.get("music_style", "") or "").strip()
    if query or style:
        if style:
            project.state["music_style"] = style
            details["music_style"] = style
        if query:
            project.state["bgm_query"] = query
            details["query"] = query
            if not style:
                # Keep the field the digest and the search node read in sync with
                # the request, otherwise the UI keeps showing the style chosen at
                # generation time.
                project.state["music_style"] = query
        project.state["bgm_tracks"] = []
        project.bgm_dirty = True

    if not details:
        raise EditError("set_bgm 需要 query / music_style / volume 至少一项")
    # Volume is applied by the stitch step, so either path needs a re-stitch —
    # and the mix has to be rebuilt rather than remuxed from the previous one.
    project.restitch_dirty = True
    return OpResult("set_bgm", True, f"配乐设置已更新: {details}", "", details)


_APPLY: dict[str, Callable[[Project, dict[str, Any]], OpResult]] = {
    "set_narration": _op_set_narration,
    "set_subtitle": _op_set_subtitle,
    "set_spotlight_slots": _op_set_spotlight_slots,
    "swap_template": _op_swap_template,
    "replace_footage": _op_replace_footage,
    "reorder_shots": _op_reorder_shots,
    "delete_shot": _op_delete_shot,
    "set_bgm": _op_set_bgm,
}


# ---------------------------------------------------------------------------
# phase 2: regenerate via the agent's own tools
# ---------------------------------------------------------------------------
def plan_regeneration(project: Project, scope: str = "only_dirty") -> list[dict[str, Any]]:
    """Decide which tools to re-run, in dependency order.

    Reuses the same tools the agent used to build the video, so edits and
    generation share one execution path instead of a parallel re-implementation.

    Decoration edits (``swap_template`` / ``set_spotlight_slots``) mutate the
    shot's ``decoration`` metadata in place. They only need a re-render, *not* a
    fresh ``Tool_BatchDecorateSpotlights`` pass: that tool regenerates the
    decoration from scratch through the LLM and would silently discard the
    user's template/slot changes, which is why edited templates never showed up
    in the final cut.
    """
    all_ids = project.shot_ids()
    if scope == "all":
        av_ids = render_ids = all_ids
        bgm = True
    else:
        av_ids = sorted(set(project.dirty_for("tts")) | set(project.dirty_for("footage")))
        # ``mark_dirty`` already implies ``render`` for every other reason, so
        # decoration-dirty shots are included here without re-decorating them.
        render_ids = project.dirty_for("render")
        bgm = project.bgm_dirty

    # Re-running assets invalidates those shots' composed clips as well.
    render_ids = sorted(set(render_ids) | set(av_ids))

    steps: list[dict[str, Any]] = []
    if av_ids:
        steps.append({
            "tool": "Tool_BatchVisualAudio",
            "args": {"shot_ids": av_ids, "mode": "repair"},
            "label": f"重新生成配音/素材（{len(av_ids)} 镜）",
        })
    if render_ids:
        steps.append({
            "tool": "Tool_BatchTimelineCompose",
            # No reuse flag: the scoped dispatch already narrows the plan to the
            # dirty shots, so everything reaching the renderer must be rebuilt.
            "args": {
                "shot_ids": render_ids,
                "require_tts_visual_audio": True,
                "require_spotlight_decoration": True,
            },
            "label": f"重新渲染镜头（{len(render_ids)} 镜）",
        })
    if bgm:
        steps.append({
            "tool": "Tool_MusicSearch",
            "args": {
                "query": str(project.state.get("bgm_query", "") or ""),
                "music_style": str(project.state.get("music_style", "") or ""),
                "must_finish_before_stitch": True,
            },
            "label": "重新检索配乐",
        })
    return steps


def subtitle_drift(project: Project) -> list[dict[str, str]]:
    """Shots whose on-screen text no longer matches what the voice-over says.

    Only ``set_subtitle`` can create this state, and it is invisible in the
    player until someone reads along with the audio — which is how a cut went
    out with captions showing a shortened version of the narration while the
    TTS still read the full script. Surfaced so the UI and the chat agent can
    warn instead of the mismatch being discovered in the finished video.
    """
    drift: list[dict[str, str]] = []
    for shot in project.shot_plan:
        override = str(shot.get("subtitle_text") or "").strip()
        if not override:
            continue
        script = str(shot.get("vocal_script", "") or "")
        if not _same_text(override, script):
            drift.append({
                "shot_id": str(shot.get("shot_id", "")),
                "subtitle": override[:160],
                "narration": script[:160],
            })
    return drift


def needs_restitch(project: Project, steps: list[dict[str, Any]]) -> bool:
    """Whether the final cut has to be re-concatenated.

    True for any regeneration step, and also for edits that produce no steps at
    all — reordering, deleting and BGM changes leave every shot's render valid
    but change the cut that is assembled from them.
    """
    return bool(steps) or project.restitch_dirty


def stitch_step(project: Project, *, bgm_changed: bool = False) -> dict[str, Any]:
    """Build the stitch command. Always uses the full shot list because stitch
    concatenates clips from all shots — but clips for unchanged shots already
    exist from the initial generation, so this is effectively a fast re-concat.

    The BGM fast path remuxes the previous mix onto the new video by stream copy
    instead of re-reading BGM files from disk, which matters on slow NFS mounts.
    It is only sound when the audio the old mix baked in is still correct, so it
    is disabled whenever the track list, the volume, or the shot order changed
    (*bgm_changed*, or anything that set ``restitch_dirty``). Reordering keeps
    the total duration identical, so the remux's duration guard would not catch
    it and the narration would play in the old order under the new picture.
    """
    shot_ids = project.shot_ids()
    reuse_old_mix = not (bgm_changed or project.restitch_dirty)
    args: dict[str, Any] = {
        "shot_count": len(shot_ids),
        "approved_count": len(shot_ids),
        "final_gate_required": False,
        "skip_rendered_check": True,
        "skip_bgm_remix": reuse_old_mix,
    }
    vol = project.state.get("bgm_volume")
    if vol is not None:
        args["bgm_volume"] = float(vol)
    return {"tool": "Tool_StitchVideo", "args": args, "label": "快速拼接成片（仅改动镜头已重渲染）"}
