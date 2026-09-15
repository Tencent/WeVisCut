"""Dedicated FastAPI router for the newvis0907 editing surface.

This module is intentionally free of any import from ``studio.app`` — it must not
create a circular import. Instead, :func:`configure_router` is handed a
:class:`RouterDeps` bundle of callbacks by ``app.py`` (after every legacy helper
is defined), and the router reaches back into the existing project/regeneration
machinery only through those callbacks.

Endpoints all live under ``/api/0907`` and never report an optimistic success:
a chat edit that has only been persisted returns ``persisted`` (or ``rendering``)
and a ``transaction`` view — the ``succeeded`` state appears only once the
background runner has finished render *and* verification.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from studio.newvis0907 import conversation, operations, schemas
from studio.newvis0907.chat_planner import plan_edit as _plan_edit
from studio.newvis0907.hierarchy import build_hierarchy as _build_hierarchy
from studio.newvis0907.transactions import TransactionError, TransactionManager

POLICY_VERSION = "0907"
CHAT_HISTORY_FILENAME = "chat_history0907.json"

# States in which the background runner may be (re)started.
_SCHEDULABLE = {
    schemas.TransactionState.PERSISTED.value,
    schemas.TransactionState.RENDER_FAILED.value,
    schemas.TransactionState.VERIFICATION_FAILED.value,
}
# States that mean the ops were committed to project state.
_APPLIED = {
    schemas.TransactionState.PERSISTED.value,
    schemas.TransactionState.RENDERING.value,
    schemas.TransactionState.VERIFYING.value,
    schemas.TransactionState.SUCCEEDED.value,
    schemas.TransactionState.RENDER_FAILED.value,
    schemas.TransactionState.VERIFICATION_FAILED.value,
}


def _default_runner(manager: TransactionManager, transaction: Any) -> None:
    """No-op default so a misconfigured server never fakes progress."""
    return None


@dataclass
class RouterDeps:
    """Everything the 0907 router needs from the host application.

    ``load_project`` / ``project_payload`` / ``discover_projects`` /
    ``delete_project`` / ``export_project`` are supplied by ``app.py`` and reuse
    the existing project machinery. ``schedule_runner`` starts the dedicated
    background regeneration+verification runner (async in production, a
    synchronous fake in tests).
    """

    load_project: Callable[[str], Any]
    project_payload: Callable[[Any], dict[str, Any]]
    discover_projects: Callable[[], list[dict[str, Any]]] = field(default=lambda: [])
    delete_project: Callable[[str], dict[str, Any]] = field(
        default=lambda target: {"deleted": True, "project_dir": target}
    )
    export_project: Callable[[Any], dict[str, Any]] = field(
        default=lambda project: {"status": "not_implemented"}
    )
    editor_flags: Callable[[], dict[str, Any]] = field(default=lambda: {})
    plan_edit: Callable[..., dict[str, Any]] = _plan_edit
    model_plan: Callable[..., Any] | None = None
    build_hierarchy: Callable[[dict[str, Any]], dict[str, Any]] = _build_hierarchy
    schedule_runner: Callable[[TransactionManager, Any], None] = _default_runner
    manager_factory: Callable[[Any], TransactionManager] | None = None


router = APIRouter()

_DEPS: RouterDeps | None = None
_MANAGERS: dict[str, TransactionManager] = {}


def configure_router(deps: RouterDeps) -> APIRouter:
    """Install dependencies. Called by ``app.py`` before ``include_router``."""
    global _DEPS
    _DEPS = deps
    _MANAGERS.clear()
    return router


def _require_deps() -> RouterDeps:
    if _DEPS is None:
        raise HTTPException(500, "0907 router 未配置")
    return _DEPS


def _manager_for(project: Any) -> TransactionManager:
    deps = _require_deps()
    key = str(Path(project.project_dir).resolve())
    mgr = _MANAGERS.get(key)
    if mgr is None:
        if deps.manager_factory is not None:
            mgr = deps.manager_factory(project)
        else:
            mgr = TransactionManager(project, editor_flags=deps.editor_flags())
        _MANAGERS[key] = mgr
    return mgr


def _load(project_dir: str) -> Any:
    if not project_dir:
        raise HTTPException(400, "必须提供 project_dir")
    return _require_deps().load_project(project_dir)


def _state_digest(project: Any) -> dict[str, Any]:
    """A light, state-derived digest for the idempotency fallback (I5)."""
    try:
        digest = operations.compute_digest(project)
        return {"shots": digest.get("shots"), "order": digest.get("shot_order")}
    except Exception:  # pragma: no cover - digest is best-effort
        return {}


def _idempotency_key(seed: str, ops: list[dict[str, Any]], project: Any = None) -> str:
    """Server fallback idempotency key.

    When the client does not supply an explicit key, we still want a *genuine*
    double-submit (same phrasing, same project state) to dedupe — but the same
    phrasing *after the state moved on* (or after a terminal edit) must mint a
    new transaction rather than replay an old success. Mixing the current
    project digest into the key gives exactly that (I5).
    """
    parts: list[Any] = [seed, ops]
    if project is not None:
        parts.append(_state_digest(project))
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return "chat_" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _recent_chat_turns(project: Any, limit: int = 8) -> list[dict[str, str]]:
    """Last user/assistant turns so the LoRA can keep multi-turn context."""
    turns: list[dict[str, str]] = []
    for msg in _as_message_list(_load_chat(project).get("messages"))[-limit:]:
        role = str(msg.get("role") or "")
        content = str(msg.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            turns.append({"role": role, "content": content[:400]})
    return turns


def _as_message_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _invoke_model_plan(
    fn: Callable[..., Any],
    state: dict[str, Any],
    message: str,
    *,
    previous_targets: Any,
    history: list[dict[str, str]],
    memory: Any = None,
) -> tuple[Any, str]:
    """Call the model-plan hook, tolerating older/simpler signatures.

    Returns ``(plan, thinking)``: a hook may hand back the plan alone, or a
    ``(plan, reasoning_text)`` pair when the backend exposed the model's
    reasoning stream.
    """
    attempts = (
        lambda: fn(state, message, previous_targets=previous_targets, history=history, memory=memory),
        lambda: fn(state, message, previous_targets=previous_targets, history=history),
        lambda: fn(state, message, previous_targets=previous_targets),
    )
    result: Any = None
    for attempt in attempts:
        try:
            result = attempt()
            break
        except TypeError:
            continue
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], str(result[1] or "")
    return result, ""


def _invoke_plan_edit(
    fn: Callable[..., Any],
    state: dict[str, Any],
    message: str,
    *,
    previous_targets: Any,
    model_plan: Any,
    memory: Any,
) -> dict[str, Any]:
    """Call the planner, tolerating a hook that predates conversation memory."""
    try:
        return fn(state, message, previous_targets=previous_targets,
                  model_plan=model_plan, memory=memory)
    except TypeError:
        return fn(state, message, previous_targets=previous_targets, model_plan=model_plan)


def _recent_targets(project: Any, manager: TransactionManager) -> list[dict[str, Any]] | None:
    """Previous edit targets so "刚才那个" resolves (I10).

    Prefer the most recently updated transaction's op targets; fall back to the
    last assistant chat turn's recorded targets. Returns rows shaped
    ``{"shot_id", "subclip_id"}`` (what the locator's previous-target path reads).
    """
    txns = [t for t in manager.all() if t.ops]
    if txns:
        latest = max(txns, key=lambda t: (t.updated_at or "", t.created_at or ""))
        rows: list[dict[str, Any]] = []
        for op in latest.ops:
            sid = str(op.get("shot_id") or "")
            scid = str(op.get("subclip_id") or "")
            if scid and not sid:
                located = operations._find_subclip(project, scid)
                if located is not None:
                    sid = located[0]
            if sid:
                rows.append({"shot_id": sid, "subclip_id": scid})
        if rows:
            return rows
    for msg in reversed(_load_chat(project).get("messages", [])):
        meta = msg.get("meta") if isinstance(msg, dict) else None
        targets = (meta or {}).get("targets") if isinstance(meta, dict) else None
        if targets:
            return [{"shot_id": str(t), "subclip_id": ""} for t in targets if isinstance(t, str)]
    return None


def _merged_payload(project: Any, manager: TransactionManager) -> dict[str, Any]:
    deps = _require_deps()
    payload = dict(deps.project_payload(project))
    hierarchy = deps.build_hierarchy(project.state)
    payload["hierarchy"] = hierarchy
    payload["quality"] = _quality_payload(project.state, hierarchy)
    payload["active_transactions"] = [tx.to_view() for tx in manager.active()]
    payload["policy_version"] = POLICY_VERSION
    return payload


def _maybe_schedule(manager: TransactionManager, transaction: Any) -> str | None:
    """Start the background runner and surface the job id it created.

    The scheduler creates a *unique* run/journal id synchronously and returns it
    so the response can hand the frontend a ``regen_job_id`` to attach to (the
    old code created the id deep inside the async task, so the UI never saw the
    running job). A scheduler that returns ``None`` (legacy/no-op) is tolerated.
    """
    if transaction is not None and transaction.state in _SCHEDULABLE:
        job_id = _require_deps().schedule_runner(manager, transaction)
        return str(job_id) if job_id else None
    return None


def _editor_enabled() -> bool:
    flags = _require_deps().editor_flags() or {}
    if not isinstance(flags, dict):
        return True
    # Default to enabled only when the flag is entirely absent; an explicit
    # false must block writes.
    return bool(flags.get("enabled", True))


def _require_editor_enabled() -> None:
    if not _editor_enabled():
        raise HTTPException(403, "编辑器已禁用（editor.enabled=false），只读模式不可写入")


def _quality_payload(state: dict[str, Any], hierarchy: dict[str, Any] | None = None) -> dict[str, Any]:
    """The live, unified quality report for a project (Task 10).

    Delegates entirely to :func:`studio.newvis0907.quality.evaluate_project`, so
    the router never invents an optimistic "ok" for a dimension it did not check.
    """
    from studio.newvis0907 import quality as quality_mod

    return quality_mod.evaluate_project(state if isinstance(state, dict) else {})


# --------------------------- chat history log ---------------------------
def _chat_path(project: Any) -> Path:
    return Path(project.project_dir) / CHAT_HISTORY_FILENAME


def _load_chat(project: Any) -> dict[str, Any]:
    path = _chat_path(project)
    if not path.is_file():
        return {"messages": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"messages": []}
    return data if isinstance(data, dict) else {"messages": []}


def _append_chat(project: Any, role: str, content: str, meta: dict[str, Any]) -> None:
    from datetime import datetime

    data = _load_chat(project)
    messages = data.setdefault("messages", [])
    messages.append({"role": role, "content": content, "at": datetime.now().isoformat(), "meta": meta})
    try:
        _chat_path(project).write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    except OSError:
        pass


# --------------------------- request bodies ---------------------------
class ChatBody(BaseModel):
    project_dir: str
    message: str = ""
    apply: bool = False
    ops: list[dict[str, Any]] | None = None
    idempotency_key: str | None = None


class OpsBody(BaseModel):
    project_dir: str
    ops: list[dict[str, Any]] = []
    idempotency_key: str | None = None


class TxBody(BaseModel):
    project_dir: str


class RegenBody(BaseModel):
    project_dir: str
    transaction_id: str = ""


class ExportBody(BaseModel):
    project_dir: str


class QualityBody(BaseModel):
    project_dir: str
    scope: str = ""
    shot_ids: list[str] | None = None
    apply: bool = False
    idempotency_key: str | None = None
    # Persistent user overrides: each record is {issue_id|code, shot_id,
    # subclip_id, action: "ignore"|"auto"}. "ignore" hides the issue from
    # auto-fix (and softens soft breaches); "auto" restores it.
    overrides: list[dict[str, Any]] | None = None


# --------------------------- endpoints ---------------------------
@router.get("/api/0907/project")
async def get_project_0907(dir: str) -> dict[str, Any]:
    project = _load(dir)
    manager = _manager_for(project)
    return _merged_payload(project, manager)


@router.get("/api/0907/projects")
async def list_projects_0907() -> dict[str, Any]:
    import asyncio

    # Discovery walks the (possibly networked) filesystem; keep it off the loop.
    projects = await asyncio.to_thread(_require_deps().discover_projects)
    return {"projects": projects, "policy_version": POLICY_VERSION}


@router.post("/api/0907/project/chat")
async def project_chat_0907(body: ChatBody) -> dict[str, Any]:
    _require_editor_enabled()
    project = _load(body.project_dir)
    manager = _manager_for(project)

    # Confirmation-free apply path (frontend "应用" button posts ops directly).
    if body.apply:
        ops = body.ops or []
        if not ops:
            raise HTTPException(400, "apply=true 时必须带上 ops")
        key = body.idempotency_key or _idempotency_key("apply:" + body.message, ops, project)
        try:
            tx = manager.open_planned(key, ops)
        except operations.OperationError as exc:
            raise HTTPException(400, str(exc)) from exc
        regen_job_id = _maybe_schedule(manager, tx)
        return {
            "intent": "edit",
            "reply": "",
            "ops": ops,
            "needs_confirm": [],
            "targets": sorted(operations.target_shot_ids(project, ops)),
            "applied": tx.state in _APPLIED,
            "results": tx.applied_results,
            "transaction": tx.to_view(),
            "regen_job_id": regen_job_id,
            "project": _merged_payload(project, manager),
            "policy_version": POLICY_VERSION,
        }

    deps = _require_deps()
    memory = conversation.Memory.load(project.project_dir)
    # Fold the conversation into this one message: a reply of「是的」arrives at
    # the planner as「@镜头03 改一下动画」, so the planner never has to guess
    # what an answer to its own question meant.
    turn = memory.interpret(body.message)
    previous_targets = turn.previous_targets or _recent_targets(project, manager)
    history = _recent_chat_turns(project)
    model_plan = None
    thinking = ""
    if deps.model_plan is not None:
        try:
            model_plan, thinking = _invoke_model_plan(
                deps.model_plan,
                project.state,
                turn.message,
                previous_targets=previous_targets,
                history=history,
                memory=memory,
            )
        except Exception:
            model_plan = None
    payload = _invoke_plan_edit(
        deps.plan_edit,
        project.state,
        turn.message,
        previous_targets=previous_targets,
        model_plan=model_plan,
        memory=memory,
    )
    _append_chat(project, "user", body.message, {})

    intent = str(payload.get("intent") or "clarify")
    ops = list(payload.get("ops") or [])
    needs_confirm = list(payload.get("needs_confirm") or [])
    reply = str(payload.get("reply") or "")
    plan_targets = list(payload.get("targets") or [])
    # When memory (not the user's words) chose the target, say so up front so a
    # wrong guess is visible and correctable in one sentence.
    if turn.note and intent == "edit":
        reply = f"{turn.note}\n{reply}".strip()

    transaction = None
    applied = False
    results: list[dict[str, Any]] = []
    regen_job_id: str | None = None
    if intent == "edit" and ops:
        key = body.idempotency_key or _idempotency_key(body.message, ops, project)
        try:
            transaction = manager.open_planned(key, ops, needs_confirm=needs_confirm, reply=reply)
        except operations.OperationError as exc:
            intent = "clarify"
            reply = f"{reply}（应用时被拒绝：{exc}）".strip()
        else:
            if transaction.state in _APPLIED:
                applied = True
                results = transaction.applied_results
                regen_job_id = _maybe_schedule(manager, transaction)

    payload["intent"] = intent
    memory.record(turn, payload)
    memory.save()

    candidates = list(payload.get("candidates") or [])
    _append_chat(project, "assistant", reply,
                 {"intent": intent, "targets": plan_targets,
                  "candidates": candidates,
                  "thinking": thinking,
                  "scope": payload.get("scope", {})})

    return {
        "intent": intent,
        "reply": reply,
        "ops": ops,
        "needs_confirm": needs_confirm,
        "candidates": candidates,
        "choices": [
            {
                "token": conversation.candidate_token(item),
                "label": conversation.candidate_label(item),
                "shot_id": str(item.get("shot_id") or ""),
            }
            for item in candidates[:3]
            if conversation.candidate_token(item)
        ],
        "thinking": thinking,
        "understood_as": turn.message if turn.message != turn.original else "",
        "scope": payload.get("scope", {}),
        "targets": plan_targets,
        "applied": applied,
        "results": results,
        "transaction": transaction.to_view() if transaction is not None else None,
        "regen_job_id": regen_job_id,
        "project": _merged_payload(project, manager),
        "policy_version": POLICY_VERSION,
    }


@router.post("/api/0907/project/ops")
async def apply_ops_0907(body: OpsBody) -> dict[str, Any]:
    _require_editor_enabled()
    project = _load(body.project_dir)
    manager = _manager_for(project)
    if not body.ops:
        raise HTTPException(400, "必须提供 ops")
    key = body.idempotency_key or _idempotency_key("ops", body.ops, project)
    try:
        tx = manager.open_planned(key, body.ops)
    except operations.OperationError as exc:
        raise HTTPException(400, str(exc)) from exc
    regen_job_id = _maybe_schedule(manager, tx)
    return {
        "applied": tx.state in _APPLIED,
        "results": tx.applied_results,
        "targets": sorted(operations.target_shot_ids(project, body.ops)),
        "transaction": tx.to_view(),
        "regen_job_id": regen_job_id,
        "project": _merged_payload(project, manager),
        "policy_version": POLICY_VERSION,
    }


@router.post("/api/0907/project/transactions/{txid}/confirm")
async def confirm_transaction_0907(txid: str, body: TxBody) -> dict[str, Any]:
    _require_editor_enabled()
    project = _load(body.project_dir)
    manager = _manager_for(project)
    try:
        tx = manager.confirm(txid)
    except TransactionError as exc:
        raise HTTPException(409, str(exc)) from exc
    except operations.OperationError as exc:
        raise HTTPException(400, str(exc)) from exc
    regen_job_id = _maybe_schedule(manager, tx)
    return {"transaction": tx.to_view(), "regen_job_id": regen_job_id,
            "targets": sorted(operations.target_shot_ids(project, tx.ops)),
            "project": _merged_payload(project, manager),
            "policy_version": POLICY_VERSION}


@router.post("/api/0907/project/transactions/{txid}/retry")
async def retry_transaction_0907(txid: str, body: TxBody) -> dict[str, Any]:
    _require_editor_enabled()
    project = _load(body.project_dir)
    manager = _manager_for(project)
    try:
        tx = manager.retry(txid)
    except TransactionError as exc:
        raise HTTPException(409, str(exc)) from exc
    regen_job_id = _maybe_schedule(manager, tx)
    return {"transaction": tx.to_view(), "regen_job_id": regen_job_id,
            "targets": sorted(operations.target_shot_ids(project, tx.ops)),
            "project": _merged_payload(project, manager),
            "policy_version": POLICY_VERSION}


@router.post("/api/0907/project/transactions/{txid}/rollback")
async def rollback_transaction_0907(txid: str, body: TxBody) -> dict[str, Any]:
    from studio.newvis0907 import quality as quality_mod

    project = _load(body.project_dir)
    manager = _manager_for(project)
    try:
        tx = manager.rollback(txid)
    except TransactionError as exc:
        raise HTTPException(409, str(exc)) from exc
    # After a rollback the persisted report could be stale; recompute it from the
    # restored state and store it so state and report never disagree.
    try:
        store_quality_report(project, quality_mod.evaluate_project(project.state))
        project.save()
    except Exception:  # pragma: no cover - best effort persistence
        pass
    return {"transaction": tx.to_view(), "project": _merged_payload(project, manager),
            "policy_version": POLICY_VERSION}


@router.post("/api/0907/project/regenerate")
async def regenerate_0907(body: RegenBody) -> dict[str, Any]:
    _require_editor_enabled()
    project = _load(body.project_dir)
    manager = _manager_for(project)
    if not body.transaction_id:
        raise HTTPException(400, "必须提供 transaction_id")
    tx = manager.get(body.transaction_id)
    if tx is None:
        raise HTTPException(404, "事务不存在")
    regen_job_id = _maybe_schedule(manager, tx)
    return {"transaction": tx.to_view(), "regen_job_id": regen_job_id,
            "targets": sorted(operations.target_shot_ids(project, tx.ops)),
            "project": _merged_payload(project, manager),
            "policy_version": POLICY_VERSION}


@router.get("/api/0907/project/chat/history")
async def chat_history_0907(dir: str) -> dict[str, Any]:
    project = _load(dir)
    return {"messages": _load_chat(project).get("messages", []), "policy_version": POLICY_VERSION}


@router.post("/api/0907/project/chat/reset")
async def chat_reset_0907(body: TxBody) -> dict[str, Any]:
    """Drop the conversation context without touching the film or its history.

    Useful when a thread has gone sideways: the next message is planned from a
    clean slate rather than against a stale pending question. Distilled user
    preferences survive, since they are about taste, not about this thread.
    """
    project = _load(body.project_dir)
    memory = conversation.Memory.load(project.project_dir)
    memory.reset()
    memory.save()
    return {"reset": True, "policy_version": POLICY_VERSION}


def _filter_report(report: dict[str, Any], scope: str, shot_ids: list[str] | None) -> dict[str, Any]:
    """Narrow a whole-project report to a scope / shot id set (schema-preserving)."""
    wanted: set[str] = set(shot_ids or [])
    if not wanted and not scope:
        return report
    issues = report.get("issues") or []
    if wanted:
        issues = [i for i in issues if not str(i.get("shot_id") or "") or str(i.get("shot_id")) in wanted]
    from studio.newvis0907 import quality as quality_mod

    filtered = dict(report)
    filtered["issues"] = issues
    filtered["status"] = quality_mod.status_from_issues(issues)
    filtered["score"] = quality_mod.score_from_issues(issues)
    filtered["scope"] = scope or None
    filtered["shot_ids"] = sorted(wanted) if wanted else None
    return filtered


def store_quality_report(project: Any, report: dict[str, Any]) -> None:
    """Persist the latest quality report into the project's 0907 namespace.

    ``Project.to_payload`` whitelists the ``quality_report0907`` state key, so
    this survives a reload without leaking into any legacy reader (which only
    reads fixed keys).
    """
    try:
        project.state["quality_report0907"] = report
    except Exception:  # pragma: no cover - defensive
        pass


def _merge_overrides(project: Any, incoming: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold posted override records into the persisted ``quality_overrides0907``.

    ``action == "auto"`` (restore) removes the mapping; anything else is stored
    as an ``ignore``. The merged map is written back onto the project state
    (a whitelisted 0907 key) so it survives reload and drives every later
    ``evaluate_project`` for this project.
    """
    from studio.newvis0907 import quality as quality_mod

    merged = quality_mod.normalize_overrides(project.state.get("quality_overrides0907"))
    for rec in incoming or []:
        rec = dict(rec or {})
        action = str(rec.get("action") or "ignore").strip().lower()
        key = str(rec.get("issue_id") or quality_mod.issue_id_for(
            str(rec.get("code") or ""), str(rec.get("shot_id") or ""),
            str(rec.get("subclip_id") or ""),
        ))
        if action == "auto":
            merged.pop(key, None)
        else:
            merged.update(quality_mod.normalize_overrides([rec]))
    project.state["quality_overrides0907"] = merged
    return merged


@router.post("/api/0907/project/quality-check")
async def quality_check_0907(body: QualityBody) -> dict[str, Any]:
    """Run the unified quality strategy and (optionally) apply safe auto-fixes.

    ``apply=false`` returns the report plus the safe fix proposals. ``apply=true``
    opens a real transaction (``optimize_cut`` scoped to the affected shots) that
    marks the cut dirty and re-runs the trustworthy pipeline — an issue that
    needs external footage/ffmpeg stays visible until it is really resolved.
    """
    from studio.newvis0907 import quality as quality_mod

    project = _load(body.project_dir)
    manager = _manager_for(project)
    # Fold in any user override changes *before* evaluating, so the report and
    # its proposals immediately reflect ignore / restore-auto choices.
    if body.overrides:
        _merge_overrides(project, body.overrides)
    report = quality_mod.evaluate_project(project.state)
    store_quality_report(project, report)
    try:
        project.save()
    except Exception:  # pragma: no cover - persistence is best-effort here
        pass

    scoped = _filter_report(report, body.scope, body.shot_ids)
    proposals = quality_mod.propose_fixes(project.state, report)

    result: dict[str, Any] = {
        **scoped,
        "complete": scoped.get("status") == "pass",
        "proposals": proposals,
        "policy_version": POLICY_VERSION,
    }

    if body.apply and proposals:
        # A read-only quality check is allowed in read-only mode, but applying
        # fixes is a write and must honour the editor flag.
        _require_editor_enabled()
        ops = _fixes_to_ops(proposals, body.scope, body.shot_ids)
        if ops:
            key = body.idempotency_key or _idempotency_key(
                "quality-fix:" + (body.scope or "global"), ops, project)
            try:
                tx = manager.open_planned(key, ops)
            except operations.OperationError as exc:
                raise HTTPException(400, str(exc)) from exc
            result["regen_job_id"] = _maybe_schedule(manager, tx)
            result["applied"] = tx.state in _APPLIED
            result["transaction"] = tx.to_view()
            result["ops"] = ops
    result["project"] = _merged_payload(project, manager)
    return result


def _fixes_to_ops(proposals: list[dict[str, Any]], scope: str, shot_ids: list[str] | None) -> list[dict[str, Any]]:
    """Translate safe fix proposals into an applicable op batch.

    Every proposal (direct ``optimize_cut`` op or an internal re-plan / normalize
    action) is turned into a scoped ``optimize_cut`` that marks the affected
    shot(s) dirty; the background runner then rebuilds them through the real
    trustworthy pipeline. Narration is never rewritten here.
    """
    scopes: list[str] = []
    seen: set[str] = set()

    def _add(sc: str) -> None:
        if sc not in seen:
            seen.add(sc)
            scopes.append(sc)

    for proposal in proposals:
        if proposal.get("op") == "optimize_cut":
            _add(str(proposal.get("scope") or "global"))
            continue
        sid = str(proposal.get("shot_id") or "")
        if sid:
            _add(sid)
        else:
            _add("global")
    if scope and not shot_ids:
        scopes = [scope]
    return [{"op": "optimize_cut", "scope": sc, "policy": "quality_autofix"} for sc in scopes]


@router.post("/api/0907/project/export")
async def export_project_0907(body: ExportBody) -> dict[str, Any]:
    project = _load(body.project_dir)
    return _require_deps().export_project(project)


@router.delete("/api/0907/project")
@router.post("/api/0907/project/delete")
async def delete_project_0907(dir: str = "", project_dir: str = "") -> dict[str, Any]:
    target = dir or project_dir
    if not target:
        raise HTTPException(400, "必须提供 dir 或 project_dir 参数")
    key = str(Path(target).resolve())
    _MANAGERS.pop(key, None)
    return _require_deps().delete_project(target)
