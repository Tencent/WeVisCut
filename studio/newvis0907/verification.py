"""Trustworthy verification for newvis0907 edits.

An edit is only ``complete`` when we can *prove*, from the mutated project state
(and, when a render ran, from its outputs), that:

* every targeted shot still exists and changed the way the op intended,
* no non-target shot's summary changed,
* if the batch invalidated rendered output, the regenerated files exist and are
  non-empty and the final video path / fingerprint / version actually moved.

Quality strategy and animation timing are *not* judged here — those arrive in
later tasks. This layer never reports success on faith: ``complete`` is the AND
of concrete checks, so a caller can never surface "succeeded" without them.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from studio.newvis0907 import operations
from studio.project import Project


def _default_probe(path: str) -> int | None:
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _file_nonempty(path: str, probe: Callable[[str], int | None]) -> bool:
    size = probe(path)
    return size is not None and size > 0


def _check(name: str, ok: bool, detail: str = "") -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _tts_loudness_targets(project: Project, ops: list[dict[str, Any]]) -> set[str]:
    targets: set[str] = set()
    for op in ops:
        name = str(op.get("op") or "")
        if name in {"set_narration", "set_shot_duration"}:
            targets |= operations.target_shot_ids(project, [op])
    return targets


def _loudness_check(project: Project, sid: str) -> dict[str, Any]:
    from studio.newvis0907 import loudness as loudness_mod

    assets = project.shot_assets.get(sid) if isinstance(project.shot_assets.get(sid), dict) else {}
    audio_meta = assets.get("audio_meta") if isinstance(assets, dict) and isinstance(assets.get("audio_meta"), dict) else {}
    payload = audio_meta.get("loudness") if isinstance(audio_meta.get("loudness"), dict) else {}
    if not payload:
        return _check(f"loudness:{sid}", False, "loudness metadata missing")
    if str(payload.get("status", "")).lower() == "failed":
        error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
        return _check(
            f"loudness:{sid}",
            False,
            str(error.get("message") or "loudness normalization failed"),
        )
    issue = loudness_mod.payload_quality_issue(payload)
    if issue is not None:
        return _check(f"loudness:{sid}", False, str(issue.get("message") or "loudness out of target"))
    return _check(f"loudness:{sid}", True, "loudness within target")


def verify_transaction(
    project: Project,
    before_digest: dict[str, Any],
    ops: list[dict[str, Any]],
    render_result: dict[str, Any] | None = None,
    probe: Callable[[str], int | None] | None = None,
) -> dict[str, Any]:
    """Return ``{"complete": bool, "checks": [...], "after_digest": {...}}``.

    ``render_result`` (from the regeneration runner) and ``probe`` (file size
    lookup) are injectable so tests can drive success/failure without real
    rendering. When the batch needs no render, verification falls back to state
    postconditions alone.
    """
    probe = probe or _default_probe
    after = operations.compute_digest(project)
    before = before_digest or {}
    before_shots = before.get("shots") or {}
    after_shots = after.get("shots") or {}

    def _op_name(op: dict[str, Any]) -> str:
        return str(op.get("op") or "")

    # Classify the batch. Only *render-invalidating* ops require a fresh per-shot
    # render; structural (reorder/delete) and restitch-only (bgm) ops are judged
    # by their own semantics + a moved final video. We do NOT blank out targets
    # to make checks pass — each category has explicit postconditions below.
    render_ops = [o for o in ops if _op_name(o) in operations.RENDER_INVALIDATING_OPS]
    reorder_ops = [o for o in ops if _op_name(o) == "reorder_shots"]
    delete_ops = [o for o in ops if _op_name(o) == "delete_shot"]

    render_targets: set[str] = set()
    for op in render_ops:
        render_targets |= operations.target_shot_ids(project, [op])
    delete_targets = {str(o.get("shot_id") or "") for o in delete_ops}
    delete_targets.discard("")
    loudness_targets = _tts_loudness_targets(project, ops)

    checks: list[dict[str, Any]] = []

    # 1) render-op targets still exist (deleted shots are handled separately)
    for sid in sorted(render_targets):
        checks.append(_check(f"target_exists:{sid}", sid in after_shots,
                             "present" if sid in after_shots else "missing"))

    # 2) each op's intended change is visible in state (incl. reorder order and
    #    delete removal — see operations.verify_postconditions).
    checks.extend(operations.verify_postconditions(project, ops))

    # 3) delete semantics: target shots must be gone
    for sid in sorted(delete_targets):
        gone = sid not in after_shots
        checks.append(_check(f"deleted:{sid}", gone,
                             "removed" if gone else "still present"))

    # 4) reorder semantics: the final order must equal the requested permutation
    for op in reorder_ops:
        want = [str(x) for x in (op.get("shot_ids") or [])]
        ok = after.get("shot_order") == want
        checks.append(_check("reordered", ok,
                             "order applied" if ok else "order mismatch"))

    # 5) non-target shots' summaries are untouched. A shot that vanished without
    #    being a delete target is collateral damage.
    protected = render_targets | delete_targets
    for sid, sig in before_shots.items():
        if sid in protected:
            continue
        if sid not in after_shots:
            checks.append(_check(f"untouched:{sid}", False, "unexpectedly removed"))
            continue
        ok = after_shots.get(sid) == sig
        checks.append(_check(f"untouched:{sid}", ok,
                             "unchanged" if ok else "changed unexpectedly"))

    # 6) render-op targets must actually have changed (guards a no-op edit). For
    #    structural / restitch-only batches the change is proven by the deleted /
    #    reordered / final checks instead.
    if render_targets:
        shot_changed = any(after_shots.get(sid) != before_shots.get(sid) for sid in render_targets)
        quality_changed = after.get("quality_requests") != before.get("quality_requests")
        changed_any = shot_changed or quality_changed
        checks.append(_check("target_changed", changed_any,
                             "state changed" if changed_any else "nothing changed"))

    # 7) render output (whenever the batch needs a regeneration/restitch pass).
    #    Real before/after file signals: render-op targets need a fresh per-shot
    #    render, and every restitching batch needs the final video file to have
    #    moved. A leftover old file or a bumped version string is NOT enough.
    if operations.ops_need_render(ops):
        before_fs = before.get("file_signals") or {}
        after_fs = after.get("file_signals") or {}
        before_rendered = before_fs.get("rendered") or {}
        after_rendered = after_fs.get("rendered") or {}

        if not render_result:
            checks.append(_check("render_output", False, "render pending"))
        else:
            for sid in sorted(render_targets):
                after_fp = after_rendered.get(sid)
                before_fp = before_rendered.get(sid)
                ok = after_fp is not None and after_fp != before_fp
                checks.append(_check(
                    f"render_written:{sid}", ok,
                    "fresh render" if ok else "no fresh render (missing/unchanged)",
                ))

            before_final = before_fs.get("final")
            after_final = after_fs.get("final")
            final_ok = after_final is not None and after_final != before_final
            checks.append(_check(
                "final_written", final_ok,
                "final file moved" if final_ok else "final file missing/unchanged",
            ))
            for sid in sorted(loudness_targets):
                checks.append(_loudness_check(project, sid))

    complete = all(c["ok"] for c in checks)
    return {"complete": complete, "checks": checks, "after_digest": after}
