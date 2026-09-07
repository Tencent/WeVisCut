"""Polar evaluator scoring a long-horizon 口播稿 → 成片 rollout.

The reward reads the manifest the in-runtime recorder already wrote
(``episode.json``) plus one ``ffprobe`` measurement of the final MP4, so scoring
costs a single shell command and needs no host round-trip
(``refresh_runtime: false``).

Shape of the signal:

- **Hard gate on delivery.** No playable final video means reward ``0``. Producing
  a real 成片 is the behaviour that has to be learned first; quality terms only
  matter once the run clears that gate.
- **Dense terms afterwards.** Stage coverage, per-shot judge pass rate, repair
  efficiency, duration fit and step economy all vary within a GRPO group, so
  leave-one-out advantages stay non-degenerate.
- **Per-role visibility.** The breakdown reports orchestrator / generator / critic
  parse-failure and error counts, which is what tells you *which* of the three
  parts regressed when the reward drops.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from polar.runtime.base import BaseRuntime
from polar.trajectory.evaluator.base import BaseTrajectoryEvaluator
from polar.trajectory.models import EvalResult, Trajectory

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"__LVP_REWARD_JSON__(?P<body>\{.*?\})__END__", re.DOTALL)

# Measurement script. Placeholders are substituted in Python so the embedded
# awk/printf blocks do not need brace escaping.
_MEASURE_SCRIPT = r"""
EPISODE_GLOB='@EPISODE_GLOB@'
manifest=""
for candidate in $EPISODE_GLOB; do
  if [ -f "$candidate" ]; then manifest="$candidate"; break; fi
done

video=""
if [ -n "$manifest" ]; then
  video="$(sed -n 's/.*"final_video_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$manifest" | head -1)"
fi
[ -n "$video" ] || video='@FALLBACK_VIDEO@'

render=0; dur=0; width=0; height=0; bytes=0
if [ -f "$video" ]; then
  bytes="$(wc -c < "$video" 2>/dev/null | tr -d ' ')"; bytes="${bytes:-0}"
  dur="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$video" 2>/dev/null | head -1)"
  dur="${dur:-0}"; case "$dur" in ''|*[!0-9.]*) dur=0;; esac
  wh="$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=,:p=0 "$video" 2>/dev/null | head -1)"
  width="${wh%%,*}"; height="${wh##*,}"
  case "$width" in ''|*[!0-9]*) width=0;; esac
  case "$height" in ''|*[!0-9]*) height=0;; esac
  if awk -v d="$dur" -v w="$width" -v b="$bytes" 'BEGIN{exit !(d>0 && w>0 && b>1024)}'; then
    render=1
  fi
fi

printf '__LVP_REWARD_JSON__{"render":%s,"duration":%s,"width":%s,"height":%s,"bytes":%s,"video":"%s","manifest":"%s"}__END__\n' \
  "$render" "$dur" "$width" "$height" "$bytes" "$video" "$manifest"

if [ -n "$manifest" ]; then
  echo "__LVP_MANIFEST_BEGIN__"
  cat "$manifest"
  echo "__LVP_MANIFEST_END__"
fi
""".strip()

_MANIFEST_RE = re.compile(
    r"__LVP_MANIFEST_BEGIN__\s*(?P<body>.*?)\s*__LVP_MANIFEST_END__", re.DOTALL
)

# Stages a complete 口播稿 → 成片 run must pass through. Coverage against this
# list is the main dense progress signal before the video exists.
_REQUIRED_TOOLS: tuple[str, ...] = (
    "Tool_SegmentArticle",
    "Tool_PlanShots",
    "Tool_SelfReflect",
    "Tool_BatchVisualAudio",
    "Tool_BatchTimelineCompose",
    "Tool_MusicSearch",
    "Tool_Stage4DimensionJudge",
    "Tool_StitchVideo",
)


class LongVideoRewardEvaluator(BaseTrajectoryEvaluator):
    """Score a long-horizon rollout in ``[0, 1]`` from its manifest and final MP4."""

    def __init__(
        self,
        *,
        output_root: str = "/polar/outputs/longvideo",
        fallback_video: str = "",
        min_seconds: float = 30.0,
        max_seconds: float = 600.0,
        target_steps: int = 14,
        measure_timeout_seconds: float = 300.0,
        require_completed: bool = False,
        weight_render: float = 0.35,
        weight_stage_coverage: float = 0.20,
        weight_judge: float = 0.15,
        weight_duration: float = 0.10,
        weight_role_health: float = 0.10,
        weight_step_economy: float = 0.10,
    ) -> None:
        self.output_root = output_root
        self.fallback_video = fallback_video
        self.min_seconds = float(min_seconds)
        self.max_seconds = float(max_seconds)
        self.target_steps = max(1, int(target_steps))
        self.measure_timeout_seconds = float(measure_timeout_seconds)
        # The pipeline can deliver a good video while the *agent transcript* ends
        # in a timeout, so this defaults off: the artifact is the ground truth.
        self.require_completed = bool(require_completed)
        self.weights = {
            "render": float(weight_render),
            "stage_coverage": float(weight_stage_coverage),
            "judge": float(weight_judge),
            "duration": float(weight_duration),
            "role_health": float(weight_role_health),
            "step_economy": float(weight_step_economy),
        }

    async def evaluate(self, trajectory: Trajectory, **runtime: Any) -> EvalResult:
        source_runtime = runtime.get("runtime")
        if not isinstance(source_runtime, BaseRuntime):
            return EvalResult(
                outcome_reward=0.0,
                metadata={"reward_error": "runtime unavailable", "output_root": self.output_root},
            )

        if self.require_completed and trajectory.status != "COMPLETED":
            return EvalResult(
                outcome_reward=0.0,
                metadata={"reward_error": f"trajectory status={trajectory.status}"},
            )

        timeout = self.measure_timeout_seconds
        cap = runtime.get("timeout_seconds")
        if isinstance(cap, (int, float)) and cap > 0:
            timeout = min(timeout, float(cap))

        try:
            result = await source_runtime.exec(
                self._build_measure_command(),
                env=dict(runtime.get("env") or {}) or None,
                timeout_sec=timeout,
            )
        except Exception as exc:  # noqa: BLE001 - reward must never kill a session
            logger.exception("long-video reward measurement failed")
            return EvalResult(
                outcome_reward=0.0, metadata={"reward_error": f"measure exec failed: {exc}"}
            )

        measurements = _extract_json(_JSON_RE, result.stdout)
        if measurements is None:
            return EvalResult(
                outcome_reward=0.0,
                metadata={
                    "reward_error": "no measurement JSON",
                    "measure_return_code": result.return_code,
                    "measure_stderr": (result.stderr or "")[-500:],
                },
            )
        manifest = _extract_json(_MANIFEST_RE, result.stdout) or {}

        score, breakdown = self._score(measurements, manifest)
        breakdown["measure_return_code"] = result.return_code
        return EvalResult(outcome_reward=score, metadata=breakdown)

    # -- scoring ---------------------------------------------------------

    def _build_measure_command(self) -> str:
        script = _MEASURE_SCRIPT
        root = self.output_root.rstrip("/")
        for token, value in (
            # Ordered most-specific first; the script takes the first match.
            ("@EPISODE_GLOB@", f"{root}/episode/episode.json {root}/*/episode.json {root}/episode.json"),
            ("@FALLBACK_VIDEO@", self.fallback_video),
        ):
            script = script.replace(token, value)
        return script

    def _score(
        self, measurements: dict[str, Any], manifest: dict[str, Any]
    ) -> tuple[float, dict[str, Any]]:
        rendered = _as_float(measurements.get("render")) >= 1.0
        if not rendered:
            # Still surface progress so failed rollouts are diagnosable.
            return 0.0, {
                "score": 0.0,
                "gate": "render_failed",
                "measurements": measurements,
                "stage_coverage": self._stage_coverage(manifest),
                "roles": manifest.get("roles", {}),
                "errors": manifest.get("errors", [])[:10],
            }

        components = {
            "render": 1.0,
            "stage_coverage": self._stage_coverage(manifest),
            "judge": self._judge_score(manifest),
            "duration": self._duration_score(_as_float(measurements.get("duration"))),
            "role_health": self._role_health(manifest),
            "step_economy": self._step_economy(manifest),
        }
        weight_sum = sum(self.weights.values()) or 1.0
        score = sum(self.weights[key] * components[key] for key in components) / weight_sum
        score = max(0.0, min(1.0, score))
        return score, {
            "score": score,
            "gate": "passed",
            "components": components,
            "weights": self.weights,
            "measurements": measurements,
            "roles": manifest.get("roles", {}),
            "interactions": manifest.get("interactions", {}),
            "orchestrator_steps": manifest.get("orchestrator_steps", 0),
            "errors": manifest.get("errors", [])[:10],
        }

    @staticmethod
    def _stage_coverage(manifest: dict[str, Any]) -> float:
        """Fraction of the required pipeline stages the orchestrator actually ran."""
        tally = manifest.get("tool_call_counts")
        if not isinstance(tally, dict) or not tally:
            return 0.0
        hit = sum(1 for tool in _REQUIRED_TOOLS if int(_as_float(tally.get(tool))) > 0)
        return hit / len(_REQUIRED_TOOLS)

    @staticmethod
    def _judge_score(manifest: dict[str, Any]) -> float:
        """Mean quality-judge reward, defaulting to neutral when nothing judged."""
        summary = manifest.get("reward_summary")
        if not isinstance(summary, dict):
            return 0.5
        mean = summary.get("mean_reward")
        value = _as_float(mean) if mean is not None else None
        if value is None or not summary.get("count"):
            return 0.5
        return max(0.0, min(1.0, value))

    def _duration_score(self, duration: float) -> float:
        if duration <= 0:
            return 0.0
        if self.min_seconds <= duration <= self.max_seconds:
            return 1.0
        if duration < self.min_seconds:
            return max(0.0, duration / self.min_seconds)
        overshoot = (duration - self.max_seconds) / max(self.max_seconds, 1e-6)
        return max(0.0, 1.0 - overshoot)

    @staticmethod
    def _role_health(manifest: dict[str, Any]) -> float:
        """Mean per-role success rate across the three parts.

        A role with no calls scores neutral rather than zero: skipping the critic
        entirely is a coverage problem, already penalized by stage coverage, not a
        reliability problem.
        """
        roles = manifest.get("roles")
        if not isinstance(roles, dict) or not roles:
            return 0.5
        scores: list[float] = []
        for info in roles.values():
            if not isinstance(info, dict):
                continue
            calls = int(_as_float(info.get("llm_calls")))
            if calls <= 0:
                scores.append(0.5)
                continue
            bad = int(_as_float(info.get("parse_failures"))) + int(_as_float(info.get("errors")))
            scores.append(max(0.0, 1.0 - bad / calls))
        return sum(scores) / len(scores) if scores else 0.5

    def _step_economy(self, manifest: dict[str, Any]) -> float:
        """Reward finishing near the target step count, penalizing thrash."""
        steps = int(_as_float(manifest.get("orchestrator_steps")))
        if steps <= 0:
            return 0.0
        if steps <= self.target_steps:
            return 1.0
        overshoot = (steps - self.target_steps) / self.target_steps
        return max(0.0, 1.0 - overshoot)


def _extract_json(pattern: re.Pattern[str], stdout: str | None) -> dict[str, Any] | None:
    if not stdout:
        return None
    match = pattern.search(stdout)
    if not match:
        return None
    try:
        payload = json.loads(match.group("body"))
    except (ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
