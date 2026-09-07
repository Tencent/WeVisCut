"""Long-horizon 口播稿 → 成片 trajectory capture for AICutPipeline.

Records one rollout of AICutPipeline's real long-horizon agent loop as three
causally linked trajectory streams:

- ``orchestrator`` (主流程) — drives progress, one tool call per turn
- ``generator`` (内容生成器) — segmentation, shot planning, decoration, repairs
- ``critic`` (反思修复) — self-reflection, quality judging, severity ranking

Core usage (no Polar required)::

    from longvideo_polar import LongVideoConfig, LongVideoRunner

    config = LongVideoConfig.load("config/longvideo_polar.yaml")
    result = await LongVideoRunner(config).run(script=script_text)

``LongVideoPolarHarness`` and ``LongVideoRewardEvaluator`` are resolved lazily so
this package stays importable where ``polar`` is not installed (the runtime side
only needs the core).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from longvideo_polar.batch import BatchRunner, aggregate_episodes
from longvideo_polar.config import (
    ROLE_NAMES,
    LongVideoConfig,
    ModelEndpoint,
    resolve_profile,
)
from longvideo_polar.model_client import PredictionResult, RoleRoutedModelClient
from longvideo_polar.recorder import TrajectoryRecorder
from longvideo_polar.roles import ALL_ROLES, ROLE_LABELS_ZH, Role, load_tool_schema
from longvideo_polar.runner import LongVideoRunner, RolloutResult

if TYPE_CHECKING:  # pragma: no cover
    from longvideo_polar.harness import LongVideoPolarHarness
    from longvideo_polar.reward import LongVideoRewardEvaluator

__all__ = [
    "ALL_ROLES",
    "ROLE_LABELS_ZH",
    "BatchRunner",
    "LongVideoConfig",
    "LongVideoPolarHarness",
    "LongVideoRewardEvaluator",
    "LongVideoRunner",
    "ModelEndpoint",
    "PredictionResult",
    "ROLE_NAMES",
    "Role",
    "RoleRoutedModelClient",
    "RolloutResult",
    "TrajectoryRecorder",
    "aggregate_episodes",
    "load_tool_schema",
    "resolve_profile",
]

_LAZY: dict[str, tuple[str, str]] = {
    "LongVideoPolarHarness": ("longvideo_polar.harness", "LongVideoPolarHarness"),
    "LongVideoRewardEvaluator": ("longvideo_polar.reward", "LongVideoRewardEvaluator"),
}


def __getattr__(name: str) -> Any:
    """Import Polar-dependent symbols only when they are actually requested."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    try:
        module = __import__(module_name, fromlist=[attribute])
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            f"{name} requires the 'polar' package; install this integration with "
            "the [polar] extra on the Polar control plane"
        ) from exc
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
