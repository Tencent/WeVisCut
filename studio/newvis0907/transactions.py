"""Per-project transaction manager for trustworthy 0907 edits.

One :class:`TransactionManager` is bound to a single ``project_dir`` and owns the
lifecycle of every edit against that project:

* transactions are persisted to ``project_dir/edit_transactions0907.json`` with
  an atomic write (the temp file name carries a uuid, so concurrent writers on
  the same directory never fight over a fixed ``.tmp`` path),
* an *idempotency key* makes a repeated request return the *same* transaction
  without mutating the project twice,
* state transitions are validated against
  :data:`studio.newvis0907.schemas.ALLOWED_TRANSITIONS`,
* the pre-edit project state is snapshotted through :meth:`Project.save`'s
  version mechanism before any mutation, so a failed edit can be rolled back to
  exactly the version it started from,
* a render failure leaves the project *dirty* (nothing is cleared); a retry
  resumes from the failed state and never re-applies the ops.
"""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from studio import project as project_mod
from studio.newvis0907 import operations, verification
from studio.newvis0907 import schemas

STORE_FILENAME = "edit_transactions0907.json"
LOCK_SUFFIX = ".lock"


class TransactionError(RuntimeError):
    """An illegal transition or operation on a transaction."""


@dataclass
class Transaction:
    id: str
    idempotency_key: str
    state: str
    ops: list[dict[str, Any]] = field(default_factory=list)
    needs_confirm: list[str] = field(default_factory=list)
    reply: str = ""
    before_digest: dict[str, Any] = field(default_factory=dict)
    pre_edit_extra: dict[str, Any] = field(default_factory=dict)
    applied_results: list[dict[str, Any]] = field(default_factory=list)
    render_result: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    pre_edit_version: str = ""
    error: str = ""
    created_at: str = ""
    updated_at: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "idempotency_key": self.idempotency_key,
            "state": self.state,
            "ops": self.ops,
            "needs_confirm": self.needs_confirm,
            "reply": self.reply,
            "before_digest": self.before_digest,
            "pre_edit_extra": self.pre_edit_extra,
            "applied_results": self.applied_results,
            "render_result": self.render_result,
            "verification": self.verification,
            "pre_edit_version": self.pre_edit_version,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "history": self.history,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transaction":
        return cls(
            id=str(data.get("id", "")),
            idempotency_key=str(data.get("idempotency_key", "")),
            state=str(data.get("state", "")),
            ops=list(data.get("ops") or []),
            needs_confirm=list(data.get("needs_confirm") or []),
            reply=str(data.get("reply", "")),
            before_digest=dict(data.get("before_digest") or {}),
            pre_edit_extra=dict(data.get("pre_edit_extra") or {}),
            applied_results=list(data.get("applied_results") or []),
            render_result=data.get("render_result"),
            verification=data.get("verification"),
            pre_edit_version=str(data.get("pre_edit_version", "")),
            error=str(data.get("error", "")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            history=list(data.get("history") or []),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_view(self) -> dict[str, Any]:
        return {
            "transaction_id": self.id,
            "state": self.state,
            "ops": self.ops,
            "needs_confirm": self.needs_confirm,
            "reply": self.reply,
            "applied_results": self.applied_results,
            "verification": self.verification,
            "render_result": self.render_result,
            "error": self.error,
            "pre_edit_version": self.pre_edit_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "history": self.history,
        }


class TransactionManager:
    """Owns edit transactions for one project directory."""

    def __init__(
        self,
        project: project_mod.Project,
        *,
        apply_ops: Callable[..., list[dict[str, Any]]] | None = None,
        verify: Callable[..., dict[str, Any]] | None = None,
        editor_flags: dict[str, Any] | None = None,
        clock: Callable[[], str] | None = None,
        max_versions: int = 50,
    ) -> None:
        self.project = project
        self.project_dir = Path(project.project_dir)
        self._apply_ops = apply_ops or operations.apply_operations
        self._verify = verify or verification.verify_transaction
        self._editor_flags = editor_flags or {}
        self._clock = clock or (lambda: datetime.now().isoformat())
        self._max_versions = max_versions
        self._txns: dict[str, Transaction] = {}
        self._by_key: dict[str, str] = {}
        self._load()

    # ---------------- persistence ----------------
    @property
    def _path(self) -> Path:
        return self.project_dir / STORE_FILENAME

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        """Exclusive ``flock`` around a read-modify-write of the store.

        ``flock`` is keyed by the open file description, so two managers in the
        same process (each opening the lock file separately) exclude each other,
        and so do two processes. The single-process ``open_planned`` /
        ``_persist`` interface is unchanged; the lock is an internal guarantee.
        """
        self.project_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.project_dir / (STORE_FILENAME + LOCK_SUFFIX)
        fh = open(lock_path, "a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()

    def _read_disk(self) -> list[dict[str, Any]]:
        path = self._path
        if not path.is_file():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return [raw for raw in (data.get("transactions") or []) if isinstance(raw, dict)]

    def _merge_locked(self) -> None:
        """Fold on-disk transactions into memory (call inside ``_file_lock``).

        A transaction we have never seen is adopted; for one we already hold,
        the record with the later ``updated_at`` wins so another instance's
        progress is not silently reverted.
        """
        for raw in self._read_disk():
            tx = Transaction.from_dict(raw)
            if not tx.id:
                continue
            current = self._txns.get(tx.id)
            if current is None or (tx.updated_at or "") > (current.updated_at or ""):
                self._txns[tx.id] = tx
        self._by_key = {
            tx.idempotency_key: tx.id
            for tx in self._txns.values()
            if tx.idempotency_key
        }

    def _write_locked(self) -> None:
        """Write the merged union to disk atomically (call inside ``_file_lock``)."""
        payload = {
            "transactions": [tx.to_dict() for tx in self._txns.values()],
            "by_key": self._by_key,
            "saved_at": self._clock(),
        }
        # The uuid in the temp name is what lets two concurrent writers on the
        # same project_dir avoid clobbering a shared fixed ``.tmp`` path.
        tmp = self.project_dir / f".{STORE_FILENAME}.{uuid.uuid4().hex}.tmp"
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                           encoding="utf-8")
            os.replace(tmp, self._path)
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise

    def _load(self) -> None:
        with self._file_lock():
            self._merge_locked()

    def _persist(self) -> None:
        with self._file_lock():
            self._merge_locked()
            self._write_locked()

    # ---------------- accessors ----------------
    def get(self, txid: str) -> Transaction | None:
        return self._txns.get(txid)

    def all(self) -> list[Transaction]:
        return list(self._txns.values())

    def active(self) -> list[Transaction]:
        return [tx for tx in self._txns.values() if tx.state in schemas.ACTIVE_STATES]

    def _require(self, txid: str) -> Transaction:
        tx = self._txns.get(txid)
        if tx is None:
            raise TransactionError(f"事务不存在: {txid}")
        return tx

    def _resolve(self, tx_or_id: "Transaction | str") -> Transaction:
        if isinstance(tx_or_id, Transaction):
            return tx_or_id
        return self._require(str(tx_or_id))

    # ---------------- state machine ----------------
    def transition(self, tx: Transaction, target: str) -> Transaction:
        allowed = schemas.ALLOWED_TRANSITIONS.get(tx.state, frozenset())
        if target not in allowed:
            raise TransactionError(f"非法状态转换: {tx.state} -> {target}")
        tx.history.append({"from": tx.state, "to": target, "at": self._clock()})
        tx.state = target
        tx.updated_at = self._clock()
        return tx

    # ---------------- persistence of project state ----------------
    def _snapshot_pre_edit(self, tx: Transaction) -> None:
        if tx.pre_edit_version:
            return
        try:
            before = {v.get("name") for v in self.project.list_versions()}
            # save() with a label snapshots the current on-disk (pre-edit) state
            # into versions/ before rewriting final_state.json.
            self.project.save(f"preedit_{tx.id}", max_versions=self._max_versions)
            after = self.project.list_versions()
            new = [v.get("name") for v in after if v.get("name") not in before]
            tx.pre_edit_version = str(new[0]) if new else (str(after[0].get("name")) if after else "")
        except (OSError, project_mod.ProjectError):
            tx.pre_edit_version = ""

    def _persist_state(self) -> None:
        try:
            self.project.save()  # no label => rewrite only, no new snapshot
        except (OSError, project_mod.ProjectError):
            pass

    # ---------------- high-level lifecycle ----------------
    def open_planned(
        self,
        idempotency_key: str,
        ops: list[dict[str, Any]],
        needs_confirm: list[str] | None = None,
        reply: str = "",
    ) -> Transaction:
        """Create (or return, if the key was seen) a planned transaction.

        Low-risk edits are applied and moved to ``persisted``; edits needing
        confirmation stop at ``confirmation_required`` without mutating anything.
        """
        key = str(idempotency_key or uuid.uuid4().hex)
        # Reserve the idempotency key under the lock so a second manager instance
        # racing on the same project_dir cannot mint a duplicate transaction.
        with self._file_lock():
            self._merge_locked()
            existing_id = self._by_key.get(key)
            if existing_id is not None:
                return self._txns[existing_id]

            txid = f"tx_{uuid.uuid4().hex[:12]}"
            now = self._clock()
            tx = Transaction(
                id=txid,
                idempotency_key=key,
                state=schemas.TransactionState.LOCATED.value,
                ops=list(ops or []),
                needs_confirm=list(needs_confirm or []),
                reply=reply,
                before_digest=operations.compute_digest(self.project),
                pre_edit_extra=operations.snapshot_extra_state(self.project),
                created_at=now,
                updated_at=now,
            )
            self._txns[txid] = tx
            self._by_key[key] = txid
            self._write_locked()

        self.transition(tx, schemas.TransactionState.PLANNED.value)
        if tx.needs_confirm:
            self.transition(tx, schemas.TransactionState.CONFIRMATION_REQUIRED.value)
            self._persist()
            return tx

        self._apply_and_persist(tx)
        return tx

    def confirm(self, txid: str) -> Transaction:
        tx = self._require(txid)
        if tx.state != schemas.TransactionState.CONFIRMATION_REQUIRED.value:
            raise TransactionError(f"事务无需确认或已处理: {tx.state}")
        self._apply_and_persist(tx)
        return tx

    def _apply_and_persist(self, tx: Transaction) -> None:
        self._snapshot_pre_edit(tx)
        try:
            results = self._apply_ops(self.project, tx.ops)
        except operations.OperationError as exc:
            tx.error = str(exc)
            self.transition(tx, schemas.TransactionState.REJECTED.value)
            self._persist()
            raise
        tx.applied_results = list(results)
        self._persist_state()
        self.transition(tx, schemas.TransactionState.PERSISTED.value)
        self._persist()

    def enter_rendering(self, tx: Transaction) -> Transaction:
        if tx.state not in (
            schemas.TransactionState.PERSISTED.value,
            schemas.TransactionState.RENDER_FAILED.value,
            schemas.TransactionState.VERIFICATION_FAILED.value,
        ):
            raise TransactionError(f"无法从 {tx.state} 进入 rendering")
        self.transition(tx, schemas.TransactionState.RENDERING.value)
        return tx

    # -- lifecycle primitives shared by execute() and the async prod runner --
    def begin_render(self, tx: Transaction) -> Transaction:
        """Move into ``rendering`` (from persisted or a failed state) and persist."""
        self.enter_rendering(tx)
        self._persist()
        return tx

    def record_render_failure(self, tx: Transaction, error: str) -> Transaction:
        """Mark a render step failed. Dirty flags are deliberately left intact."""
        tx.error = str(error)
        self.transition(tx, schemas.TransactionState.RENDER_FAILED.value)
        self._persist()
        return tx

    def record_render_success(
        self,
        tx: Transaction,
        render_result: dict[str, Any],
        probe: Callable[[str], int | None] | None = None,
    ) -> Transaction:
        """Store render output, verify, and settle to succeeded / verification_failed."""
        tx.render_result = render_result if isinstance(render_result, dict) else {}
        self.transition(tx, schemas.TransactionState.VERIFYING.value)
        self._persist()
        outcome = self._verify(
            self.project, tx.before_digest, tx.ops,
            render_result=tx.render_result, probe=probe,
        )
        tx.verification = outcome
        if outcome.get("complete"):
            self.transition(tx, schemas.TransactionState.SUCCEEDED.value)
        else:
            self.transition(tx, schemas.TransactionState.VERIFICATION_FAILED.value)
        self._persist()
        return tx

    def execute(
        self,
        tx_or_id: "Transaction | str",
        executor: Callable[[project_mod.Project, Transaction], dict[str, Any]],
        probe: Callable[[str], int | None] | None = None,
    ) -> Transaction:
        """Run render + verify for a transaction using ``executor``.

        ``executor(project, tx)`` performs the regeneration and returns a render
        result dict (``{"outputs": [...], "final_video_path": ..., ...}``). Any
        exception, or a returned ``{"error": ...}``, marks the transaction
        ``render_failed`` and leaves the project dirty. Re-apply never happens
        here, so a retry simply calls ``execute`` again.
        """
        tx = self._resolve(tx_or_id)
        self.begin_render(tx)

        try:
            result = executor(self.project, tx)
        except Exception as exc:  # noqa: BLE001 - any failure => render_failed
            return self.record_render_failure(tx, f"{type(exc).__name__}: {exc}")

        if isinstance(result, dict) and result.get("error"):
            return self.record_render_failure(tx, str(result.get("error")))

        return self.record_render_success(tx, result if isinstance(result, dict) else {}, probe=probe)

    def retry(self, txid: str) -> Transaction:
        """Validate that a transaction can be resumed from a failed state."""
        tx = self._require(txid)
        if tx.state not in (
            schemas.TransactionState.RENDER_FAILED.value,
            schemas.TransactionState.VERIFICATION_FAILED.value,
        ):
            raise TransactionError(f"事务不可重试（当前 {tx.state}）")
        return tx

    def rollback(self, txid: str) -> Transaction:
        tx = self._require(txid)
        if schemas.TransactionState.ROLLED_BACK.value not in schemas.ALLOWED_TRANSITIONS.get(
            tx.state, frozenset()
        ):
            raise TransactionError(f"事务不可回滚（当前 {tx.state}）")
        if tx.pre_edit_version:
            try:
                self.project.restore_version(tx.pre_edit_version)
                # restore_version does not know about 0907-only state keys
                # (e.g. quality_requests0907), so undo those explicitly.
                operations.restore_extra_state(self.project, tx.pre_edit_extra)
                self.project.save(f"rollback_{tx.id}", max_versions=self._max_versions)
            except (OSError, project_mod.ProjectError) as exc:
                tx.error = f"rollback restore failed: {exc}"
        else:
            operations.restore_extra_state(self.project, tx.pre_edit_extra)
        self.transition(tx, schemas.TransactionState.ROLLED_BACK.value)
        self._persist()
        return tx
