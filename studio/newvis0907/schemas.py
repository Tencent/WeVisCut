"""Pydantic schemas for newvis0907 payloads."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict
except Exception:  # pragma: no cover - pydantic v1 compatibility
    ConfigDict = None  # type: ignore[assignment]


def _is_pydantic_v2() -> bool:
    return hasattr(BaseModel, "model_dump")


if _is_pydantic_v2():
    class _CompatModel(BaseModel):
        model_config = ConfigDict(extra="ignore") if ConfigDict is not None else {}
else:
    class _CompatModel(BaseModel):
        class Config:
            extra = "ignore"


class TransactionState(str, Enum):
    """Lifecycle of a trustworthy 0907 edit transaction."""

    LOCATED = "located"
    PLANNED = "planned"
    CONFIRMATION_REQUIRED = "confirmation_required"
    PERSISTED = "persisted"
    RENDERING = "rendering"
    VERIFYING = "verifying"
    SUCCEEDED = "succeeded"
    CLARIFICATION_REQUIRED = "clarification_required"
    REJECTED = "rejected"
    RENDER_FAILED = "render_failed"
    VERIFICATION_FAILED = "verification_failed"
    ROLLED_BACK = "rolled_back"


# Terminal states never transition further.
TERMINAL_STATES: frozenset[str] = frozenset(
    {
        TransactionState.SUCCEEDED.value,
        TransactionState.REJECTED.value,
        TransactionState.ROLLED_BACK.value,
        TransactionState.CLARIFICATION_REQUIRED.value,
    }
)

# States a caller may treat as "still in flight" for the active list.
ACTIVE_STATES: frozenset[str] = frozenset(
    {
        TransactionState.LOCATED.value,
        TransactionState.PLANNED.value,
        TransactionState.CONFIRMATION_REQUIRED.value,
        TransactionState.PERSISTED.value,
        TransactionState.RENDERING.value,
        TransactionState.VERIFYING.value,
        TransactionState.RENDER_FAILED.value,
        TransactionState.VERIFICATION_FAILED.value,
    }
)

# Allowed state transitions. A move not listed here is rejected.
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    TransactionState.LOCATED.value: frozenset(
        {TransactionState.PLANNED.value, TransactionState.CLARIFICATION_REQUIRED.value,
         TransactionState.REJECTED.value}
    ),
    TransactionState.PLANNED.value: frozenset(
        {TransactionState.CONFIRMATION_REQUIRED.value, TransactionState.PERSISTED.value,
         TransactionState.REJECTED.value}
    ),
    TransactionState.CONFIRMATION_REQUIRED.value: frozenset(
        {TransactionState.PERSISTED.value, TransactionState.REJECTED.value}
    ),
    TransactionState.PERSISTED.value: frozenset(
        {TransactionState.RENDERING.value, TransactionState.ROLLED_BACK.value}
    ),
    TransactionState.RENDERING.value: frozenset(
        {TransactionState.VERIFYING.value, TransactionState.RENDER_FAILED.value}
    ),
    TransactionState.VERIFYING.value: frozenset(
        {TransactionState.SUCCEEDED.value, TransactionState.VERIFICATION_FAILED.value}
    ),
    TransactionState.RENDER_FAILED.value: frozenset(
        {TransactionState.RENDERING.value, TransactionState.ROLLED_BACK.value}
    ),
    TransactionState.VERIFICATION_FAILED.value: frozenset(
        {TransactionState.RENDERING.value, TransactionState.ROLLED_BACK.value}
    ),
    TransactionState.SUCCEEDED.value: frozenset(),
    TransactionState.CLARIFICATION_REQUIRED.value: frozenset(),
    TransactionState.REJECTED.value: frozenset(),
    TransactionState.ROLLED_BACK.value: frozenset(),
}


class TransactionView(_CompatModel):
    transaction_id: str = ""
    state: TransactionState = TransactionState.LOCATED
    created_at: str = ""
    updated_at: str = ""
    message: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class QualityIssue(_CompatModel):
    code: str = ""
    message: str = ""
    severity: str = ""
    field: str = ""


class QualitySummary(_CompatModel):
    status: str = "unknown"
    score: float | None = None
    issues: list[QualityIssue] = Field(default_factory=list)


class FootageSubclipView(_CompatModel):
    subclip_id: str
    display_number: str
    clip_id: str
    path: str
    source_duration: float
    source_in: float
    source_out: float
    final_duration: float
    playback_rate: float
    score: float | None = None
    query: str = ""
    start: float
    end: float
    replaced: bool = False
    user_override: bool = False
    fingerprint: str = ""


class ModelShotView(_CompatModel):
    shot_id: str
    segment_id: str
    narration: str
    display_number: str
    display_subtitle: str
    start: float
    end: float
    duration: float
    is_spotlight: bool = False
    animation_type: str = ""
    quality_status: str = "unknown"
    subclips: list[FootageSubclipView] = Field(default_factory=list)


class ChapterView(_CompatModel):
    segment_id: str
    title: str
    summary: str = ""
    display_index: int
    model_shot_ids: list[str] = Field(default_factory=list)
    start: float
    end: float


class HierarchyPayload(_CompatModel):
    chapters: list[ChapterView] = Field(default_factory=list)
    model_shots: list[ModelShotView] = Field(default_factory=list)
    total_duration: float = 0.0
