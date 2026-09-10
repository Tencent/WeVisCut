"""Public helpers for newvis0907 offline planning and trustworthy editing."""

from . import animations, footage, loudness, pacing
from .chat_planner import SUPPORTED_OPS, plan_edit, validate_model_plan
from .hierarchy import build_hierarchy
from .locator import resolve_targets
from .operations import OperationError, apply_operations, compute_digest
from .transactions import Transaction, TransactionError, TransactionManager
from .verification import verify_transaction

__all__ = [
    "OperationError",
    "SUPPORTED_OPS",
    "Transaction",
    "TransactionError",
    "TransactionManager",
    "animations",
    "apply_operations",
    "build_hierarchy",
    "compute_digest",
    "footage",
    "loudness",
    "pacing",
    "plan_edit",
    "resolve_targets",
    "validate_model_plan",
    "verify_transaction",
]
