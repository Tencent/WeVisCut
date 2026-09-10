"""Unified project quality evaluation for the newvis0907 trustworthy cut.

``evaluate_project(state) -> dict`` is the single source of truth for whether a
finished (or in-progress) 0907 cut is trustworthy. It never reports ``pass`` for
a dimension it did not actually check, and it never *fakes* a fix: an issue that
depends on external footage / ffmpeg stays visible until it is really resolved.

The report shape is stable::

    {
      "status": "pass" | "warning" | "fail",
      "score": 0-100,                       # deterministic severity-weighted
      "summary": {"pass": bool, "counts": {severity: n}, "total_issues": n,
                  "model_shot_count": n, "total_duration_sec": float, ...},
      "issues": [ {code, severity, message, shot_id, subclip_id, field,
                   actual, expected, auto_fixable, user_override}, ... ],
    }

Every dimension named in the task is checked here: subtitle punctuation /
character conservation / cue capacity; footage identity duplicates (whole-film /
adjacent / same-playlist), coverage short/invalid, per-shot subclip count,
sub-2s subclips, sub-1/3 playback rate, freeze/loop markers; the 20+ model-shot
rule and the 8-15s per-shot band; narration + final loudness; and the semantic
animation contract (phrase present, phrase-derived timing, valid type, at most
one primary, real template ids, chart/timeline gates). A user override is
respected (severity is softened and auto-fix is disabled) but the issue is still
reported when it breaches a hard trustworthy threshold.
"""

from __future__ import annotations

import math
from typing import Any

from . import animations as animations_mod
from . import footage as footage_mod
from . import loudness as loudness_mod
from . import pacing as pacing_mod
from . import subtitles as subtitles_mod

# --------------------------------------------------------------------------
# severity model + stable scoring
# --------------------------------------------------------------------------
FAIL = "fail"
WARNING = "warning"
INFO = "info"

_SEVERITY_ORDER = {INFO: 0, WARNING: 1, FAIL: 2}

# Deterministic, stable weights: the whole point of a numeric score is that two
# reports with the same issues always produce the same number, and that a more
# severe mix always scores lower. A single ``fail`` costs far more than a
# ``warning`` so a hard trustworthy breach can never hide behind a high score.
SEVERITY_WEIGHTS: dict[str, float] = {FAIL: 20.0, WARNING: 5.0, INFO: 1.0}

# Auto-fix convergence: an issue that is still present after this many applied
# fix cycles is declared manual-only — it is no longer proposed and no longer
# opens a transaction, so the auto-fix loop can never spin forever (I2).
FIX_ATTEMPT_LIMIT = 2

MIN_SUBCLIP_SEC = pacing_mod.DEFAULT_MIN_SUBCLIP
MAX_SUBCLIP_SEC = pacing_mod.DEFAULT_MAX_SUBCLIP
MIN_PLAYBACK_RATE = pacing_mod.DEFAULT_MIN_PLAYBACK_RATE
MODEL_SHOT_MIN_SEC = pacing_mod.MODEL_SHOT_MIN_SEC
MODEL_SHOT_MAX_SEC = pacing_mod.MODEL_SHOT_MAX_SEC
LONG_VIDEO_THRESHOLD_SEC = pacing_mod.LONG_VIDEO_THRESHOLD_SEC

# Target integrated-loudness window: -16 ± 1 == [-17, -15] LUFS, TP <= -1.5.
LOUDNESS_TARGET = loudness_mod.target_profile()

_TOL = 1e-6


def score_from_issues(issues: list[dict[str, Any]]) -> int:
    """Deterministic 0-100 score from a list of issues.

    ``score = round(clamp(100 - sum(weight[severity])))``. Exposed (and tested)
    directly so the weighting can never silently drift.
    """
    penalty = 0.0
    for issue in issues or []:
        sev = str((issue or {}).get("severity") or WARNING)
        penalty += SEVERITY_WEIGHTS.get(sev, SEVERITY_WEIGHTS[WARNING])
    return int(round(max(0.0, min(100.0, 100.0 - penalty))))


def status_from_issues(issues: list[dict[str, Any]]) -> str:
    worst = INFO
    has_any = False
    for issue in issues or []:
        sev = str((issue or {}).get("severity") or WARNING)
        has_any = True
        if _SEVERITY_ORDER.get(sev, 1) > _SEVERITY_ORDER.get(worst, 0):
            worst = sev
    if not has_any:
        return "pass"
    if worst == FAIL:
        return "fail"
    if worst == WARNING:
        return "warning"
    return "pass"  # only info-level notes


# --------------------------------------------------------------------------
# coercion helpers
# --------------------------------------------------------------------------
def _as_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _as_list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else []


def _as_text(v: Any) -> str:
    return str(v or "")


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return f


def issue_id_for(code: str, shot_id: str = "", subclip_id: str = "") -> str:
    """Stable, human-legible id for an issue's ``(code, shot, subclip)`` target.

    The same defect on the same target always yields the same id across
    re-evaluations, so a persisted user override keeps matching it.
    """
    return "|".join([_as_text(code), _as_text(shot_id), _as_text(subclip_id)])


def normalize_overrides(raw: Any) -> dict[str, dict[str, Any]]:
    """Coerce a stored/posted overrides collection into ``{issue_id: record}``.

    Accepts either the persisted mapping (``{issue_id: {...}}``) or a list of
    records (``[{"issue_id"|"code","shot_id","subclip_id","action"}, ...]``).
    Only ``action == "ignore"`` records are kept; ``"auto"`` (restore) records
    are dropped so a restore simply removes the mapping.
    """
    out: dict[str, dict[str, Any]] = {}

    def _consume(rec: Any) -> None:
        rec = _as_dict(rec)
        action = _as_text(rec.get("action")).strip().lower() or "ignore"
        key = _as_text(rec.get("issue_id")) or issue_id_for(
            _as_text(rec.get("code")), _as_text(rec.get("shot_id")), _as_text(rec.get("subclip_id"))
        )
        if not _as_text(rec.get("code")) and key.count("|") != 2:
            return
        if action == "auto":
            out.pop(key, None)
            return
        out[key] = {
            "action": "ignore",
            "issue_id": key,
            "code": _as_text(rec.get("code")) or key.split("|", 1)[0],
            "shot_id": _as_text(rec.get("shot_id")),
            "subclip_id": _as_text(rec.get("subclip_id")),
        }

    if isinstance(raw, dict):
        for key, rec in raw.items():
            rec = _as_dict(rec)
            rec.setdefault("issue_id", key)
            _consume(rec)
    elif isinstance(raw, list):
        for rec in raw:
            _consume(rec)
    return out


def normalize_fix_attempts(raw: Any) -> dict[str, int]:
    """Coerce the persisted ``quality_fix_attempts0907`` map to ``{issue_id: n}``."""
    out: dict[str, int] = {}
    if isinstance(raw, dict):
        for key, val in raw.items():
            try:
                n = int(val)
            except (TypeError, ValueError):
                continue
            if n > 0 and isinstance(key, str):
                out[key] = n
    return out


class _Issues:
    """Small accumulator that applies the user-override softening uniformly."""

    def __init__(
        self,
        overrides: dict[str, dict[str, Any]] | None = None,
        fix_attempts: dict[str, int] | None = None,
    ) -> None:
        self.items: list[dict[str, Any]] = []
        self.overrides = overrides or {}
        self.fix_attempts = fix_attempts or {}

    def add(
        self,
        code: str,
        severity: str,
        message: str,
        *,
        shot_id: str = "",
        subclip_id: str = "",
        field: str = "",
        actual: Any = None,
        expected: Any = None,
        auto_fixable: bool = False,
        user_override: bool = False,
        hard: bool = False,
    ) -> dict[str, Any]:
        issue_id = issue_id_for(code, shot_id, subclip_id)
        # A persistent user override (``quality_overrides0907``) counts the same
        # as an inline clip-level override flag.
        persisted = issue_id in self.overrides
        eff_override = bool(user_override) or persisted

        eff_severity = severity
        eff_auto = auto_fixable
        # An explicit user override is respected: a *soft* breach is downgraded
        # one severity level and never auto-fixed. A *hard* trustworthy breach
        # keeps its original severity (the risk stays fully visible) but is also
        # removed from the auto-fix set and marked overridden.
        if eff_override:
            eff_auto = False
            if not hard:
                if severity == FAIL:
                    eff_severity = WARNING
                elif severity == WARNING:
                    eff_severity = INFO
        # Auto-fix convergence: once an issue has survived FIX_ATTEMPT_LIMIT
        # applied fixes it is manual-only — never auto-fixed / proposed again.
        needs_manual = self.fix_attempts.get(issue_id, 0) >= FIX_ATTEMPT_LIMIT
        if needs_manual:
            eff_auto = False

        issue = {
            "issue_id": issue_id,
            "code": code,
            "severity": eff_severity,
            "message": message,
            "shot_id": shot_id,
            "subclip_id": subclip_id,
            "field": field or shot_id or subclip_id,
            "actual": actual,
            "expected": expected,
            "auto_fixable": bool(eff_auto),
            "user_override": bool(eff_override),
            "needs_manual": bool(needs_manual),
        }
        self.items.append(issue)
        return issue


# --------------------------------------------------------------------------
# per-shot data extraction
# --------------------------------------------------------------------------
def _shot_narration(shot: dict[str, Any]) -> str:
    return (
        _as_text(shot.get("vocal_script"))
        or _as_text(shot.get("narration"))
        or _as_text(shot.get("subtitle_text"))
    )


def _shot_duration(shot: dict[str, Any], assets: dict[str, Any]) -> float:
    audio = _as_dict(assets.get("audio_meta"))
    return _as_float(audio.get("duration_sec")) or _as_float(shot.get("duration_sec")) or 0.0


def _display_text(shot: dict[str, Any], assets: dict[str, Any]) -> str:
    for source in (shot, assets):
        val = _as_text(_as_dict(source).get("subtitle_display_text"))
        if val:
            return val
    return _as_text(shot.get("subtitle_text"))


def _shot_cues(shot: dict[str, Any], assets: dict[str, Any]) -> list[dict[str, Any]]:
    for source in (assets, shot):
        cues = [c for c in _as_list(_as_dict(source).get("subtitle_cues")) if isinstance(c, dict)]
        if cues:
            return cues
    return []


def _playlist_clips(assets: dict[str, Any]) -> list[dict[str, Any]]:
    return list(footage_mod.iter_playlist_clips(assets))


def _clip_final(clip: dict[str, Any]) -> float:
    return (
        _as_float(clip.get("final_duration_sec"))
        or _as_float(clip.get("used_duration_sec"))
        or _as_float(clip.get("final_duration"))
        or _as_float(clip.get("duration_sec"))
    )


def _clip_source_used(clip: dict[str, Any]) -> float:
    return (
        _as_float(clip.get("source_used_duration_sec"))
        or _as_float(clip.get("source_duration_sec"))
        or _as_float(clip.get("source_duration"))
        or _as_float(clip.get("duration_sec"))
    )


def _clip_rate(clip: dict[str, Any]) -> float:
    explicit = _as_float(clip.get("playback_rate"))
    if explicit > 0:
        return explicit
    source_used = _clip_source_used(clip)
    final = _clip_final(clip)
    if source_used > 0 and final > 0:
        return source_used / final
    return 1.0


def _clip_subclip_id(shot_id: str, clip: dict[str, Any], idx: int) -> str:
    return _as_text(clip.get("subclip_id")) or f"{shot_id}::subclip::{idx}"


# --------------------------------------------------------------------------
# subtitle checks
# --------------------------------------------------------------------------
def _check_subtitles(shots: list[dict[str, Any]], issues: _Issues) -> None:
    for shot in shots:
        sid = _as_text(shot["shot_id"])
        assets = shot["assets"]
        raw = shot["raw"]
        narration = _shot_narration(raw)
        if not narration.strip():
            continue
        duration = shot["duration"]

        display = _display_text(raw, assets)
        cues = _shot_cues(raw, assets)
        if not display and (cues or narration):
            built = subtitles_mod.build_subtitle_cues(narration, duration)
            display = built.get("display_text", "")
            if not cues:
                cues = built.get("cues", [])

        # character conservation: on-screen text must be lexically identical to
        # the spoken narration (only punctuation / whitespace may differ).
        norm_src = subtitles_mod.normalize_lexical(narration)
        norm_disp = subtitles_mod.normalize_lexical(display)
        if display and norm_src != norm_disp:
            issues.add(
                "subtitle_char_mismatch", FAIL,
                f"{sid} 字幕文字与旁白不守恒（增删改字）",
                shot_id=sid, field="subtitle", actual=norm_disp, expected=norm_src,
                auto_fixable=True, hard=True,
            )

        # punctuation: a natural cut punctuates the on-screen text; the display
        # should end on a hard sentence terminator.
        if display:
            tail = subtitles_mod._compact_text(display)[-1:]
            if tail and tail not in subtitles_mod.HARD_PUNCT:
                issues.add(
                    "subtitle_punctuation", WARNING,
                    f"{sid} 字幕缺少自然标点（结尾无句末标点）",
                    shot_id=sid, field="subtitle", actual=tail, expected="。/！/？/…",
                    auto_fixable=True,
                )

        # cue capacity: <= 2 lines, each line <= 14 chars (an unsplittable token
        # such as a URL / long ASCII id is the sole exception).
        for cue in cues:
            cue_text = _as_text(cue.get("text"))
            lines = [ln for ln in _as_list(cue.get("lines")) if _as_text(ln)]
            if not lines:
                lines = subtitles_mod._split_into_lines(cue_text, 14, 2)
            if len(lines) > 2:
                issues.add(
                    "subtitle_too_many_lines", FAIL,
                    f"{sid} 单条字幕超过 2 行",
                    shot_id=sid, field="subtitle", actual=len(lines), expected="<=2",
                    auto_fixable=True, hard=True,
                )
            for line in lines:
                if len(line) > 14 and not subtitles_mod._is_single_unsplittable_token(line):
                    issues.add(
                        "subtitle_line_too_long", FAIL,
                        f"{sid} 字幕单行超过 14 字：{line}",
                        shot_id=sid, field="subtitle", actual=len(line), expected="<=14",
                        auto_fixable=True, hard=True,
                    )


# --------------------------------------------------------------------------
# footage / source checks
# --------------------------------------------------------------------------
def _check_footage(shots: list[dict[str, Any]], issues: _Issues, visual_threshold: float = 0.9) -> None:
    # gather every subclip with a stable identity + position
    catalogue: list[dict[str, Any]] = []
    for shot in shots:
        sid = _as_text(shot["shot_id"])
        assets = shot["assets"]
        raw = shot["raw"]
        clips = _playlist_clips(assets)
        shot_duration = shot["duration"]

        # coverage status recorded on the playlist (short / invalid) — honest,
        # never faked.
        clip_meta = _as_dict(assets.get("clip_meta"))
        playlist = _as_dict(clip_meta.get("visual_playlist"))
        status = _as_text(playlist.get("coverage_status")).strip().lower()
        if status == "invalid":
            issues.add(
                "footage_coverage_invalid", FAIL,
                f"{sid} 素材覆盖无效（invalid）",
                shot_id=sid, field="footage.coverage",
                actual="invalid", expected="covered",
            )
        elif status == "short":
            issues.add(
                "footage_coverage_short", WARNING,
                f"{sid} 素材覆盖不足，缺口 {_as_float(playlist.get('deficit_sec')):.2f}s",
                shot_id=sid, field="footage.coverage",
                actual=_as_float(playlist.get("deficit_sec")), expected=0.0,
                auto_fixable=True,
            )

        # per-shot subclip count: a model shot should be a multi-source cut
        # (>=2 subclips) unless it is legitimately below the 2s subclip floor.
        if clips and len(clips) < 2 and shot_duration >= MIN_SUBCLIP_SEC:
            issues.add(
                "shot_too_few_subclips", WARNING,
                f"{sid} 只有 {len(clips)} 个子镜，建议 >=2 个不同素材",
                shot_id=sid, field="footage", actual=len(clips), expected=">=2",
                auto_fixable=True,
            )

        for idx, clip in enumerate(clips, start=1):
            subclip_id = _clip_subclip_id(sid, clip, idx)
            override = bool(clip.get("user_override"))
            final = _clip_final(clip)
            source_used = _clip_source_used(clip)
            rate = _clip_rate(clip)

            # freeze / loop / last-frame hold markers are never allowed.
            if any(bool(clip.get(k)) for k in ("freeze", "loop", "hold_last_frame", "frozen", "looped")):
                issues.add(
                    "footage_freeze_or_loop", FAIL,
                    f"{subclip_id} 使用了冻结/循环/定格帧",
                    shot_id=sid, subclip_id=subclip_id, field="footage",
                    actual="freeze/loop", expected="none",
                    user_override=override, hard=True,
                )

            if 0 < final < MIN_SUBCLIP_SEC - _TOL:
                issues.add(
                    "subclip_too_short", FAIL,
                    f"{subclip_id} 子镜时长 {final:.2f}s < 2s",
                    shot_id=sid, subclip_id=subclip_id, field="footage",
                    actual=round(final, 3), expected=">=2.0",
                    auto_fixable=True, user_override=override, hard=True,
                )

            # playback rate floor: final may be at most 3x source_used (rate 1/3).
            if source_used > 0 and final > 0 and rate < MIN_PLAYBACK_RATE - _TOL:
                issues.add(
                    "subclip_rate_too_slow", FAIL,
                    f"{subclip_id} playback_rate {rate:.3f} < 1/3（超过 3 倍慢放=近似定格）",
                    shot_id=sid, subclip_id=subclip_id, field="footage",
                    actual=round(rate, 4), expected=">=0.333",
                    user_override=override, hard=True,
                )

            ident = footage_mod.source_identity(clip)
            catalogue.append({
                "shot_index": shot["index"], "shot_id": sid,
                "subclip_id": subclip_id, "identity": ident, "override": override,
            })

    # duplicate detection across the catalogue.
    reported_global: set[tuple[int, int]] = set()
    n = len(catalogue)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = catalogue[i], catalogue[j]
            if not a["identity"].matches(b["identity"], visual_threshold):
                continue
            same_shot = a["shot_id"] == b["shot_id"]
            adjacent = abs(a["shot_index"] - b["shot_index"]) == 1
            if same_shot:
                issues.add(
                    "footage_duplicate_in_playlist", FAIL,
                    f"{a['subclip_id']} 与 {b['subclip_id']} 在同一镜头重复使用同一素材",
                    shot_id=a["shot_id"], subclip_id=b["subclip_id"], field="footage",
                    actual="duplicate", expected="unique",
                    auto_fixable=True, hard=True,
                )
            elif adjacent:
                issues.add(
                    "footage_duplicate_adjacent", FAIL,
                    f"{a['subclip_id']} 与相邻镜头 {b['subclip_id']} 重复使用同一素材",
                    shot_id=b["shot_id"], subclip_id=b["subclip_id"], field="footage",
                    actual="duplicate", expected="unique",
                    auto_fixable=True, hard=True,
                )
            else:
                key = (i, j)
                if key not in reported_global:
                    reported_global.add(key)
                    issues.add(
                        "footage_duplicate_global", WARNING,
                        f"{a['subclip_id']} 与 {b['subclip_id']} 全片重复使用同一素材",
                        shot_id=b["shot_id"], subclip_id=b["subclip_id"], field="footage",
                        actual="duplicate", expected="unique",
                        auto_fixable=True,
                    )


# --------------------------------------------------------------------------
# pacing / model-shot checks
# --------------------------------------------------------------------------
def _check_pacing(shots: list[dict[str, Any]], total_duration: float, issues: _Issues) -> None:
    required = pacing_mod.minimum_model_shots(total_duration)
    count = len(shots)
    if required and count < required:
        severity = FAIL if total_duration >= LONG_VIDEO_THRESHOLD_SEC else WARNING
        issues.add(
            "too_few_model_shots", severity,
            f"模型镜数量 {count} 少于 {total_duration:.0f}s 视频所需的最少 {required} 个",
            field="pacing", actual=count, expected=f">={required}",
            auto_fixable=True,
        )
    for shot in shots:
        sid = _as_text(shot["shot_id"])
        duration = shot["duration"]
        if duration <= 0:
            continue
        if duration > MODEL_SHOT_MAX_SEC + _TOL:
            issues.add(
                "shot_too_long", FAIL,
                f"{sid} 时长 {duration:.1f}s 超过 15s 模型镜上限，必须拆分",
                shot_id=sid, field="pacing", actual=round(duration, 2), expected="<=15",
                auto_fixable=True,
            )
        elif duration < MODEL_SHOT_MIN_SEC - _TOL:
            issues.add(
                "shot_short", INFO,
                f"{sid} 时长 {duration:.1f}s 短于建议的 8s（可接受）",
                shot_id=sid, field="pacing", actual=round(duration, 2), expected=">=8",
            )


# --------------------------------------------------------------------------
# loudness checks
# --------------------------------------------------------------------------
def _check_loudness(shots: list[dict[str, Any]], state: dict[str, Any], issues: _Issues) -> None:
    for shot in shots:
        sid = _as_text(shot["shot_id"])
        assets = shot["assets"]
        raw = shot["raw"]
        if not _shot_narration(raw).strip():
            continue
        audio_meta = _as_dict(assets.get("audio_meta"))
        if not audio_meta:
            # no audio yet (pre-TTS) — nothing rendered to measure.
            issues.add(
                "loudness_missing", WARNING,
                f"{sid} 口播尚未生成/缺少 loudness 测量",
                shot_id=sid, field="audio_meta.loudness", actual=None, expected="normalized/skipped",
                auto_fixable=True,
            )
            continue
        payload = _as_dict(audio_meta.get("loudness"))
        if not payload:
            issues.add(
                "loudness_missing", WARNING,
                f"{sid} 缺少 loudness 测量",
                shot_id=sid, field="audio_meta.loudness", actual=None, expected="normalized/skipped",
                auto_fixable=True,
            )
            continue
        status = _as_text(payload.get("status")).strip().lower()
        if status not in {"normalized", "skipped"}:
            issue = loudness_mod.payload_quality_issue(payload) or {}
            issues.add(
                "loudness_not_normalized", FAIL,
                _as_text(issue.get("message")) or f"{sid} loudness 未归一化（status={status or 'unknown'}）",
                shot_id=sid, field="audio_meta.loudness", actual=status or None,
                expected="normalized/skipped", auto_fixable=True,
            )
            continue
        if not loudness_mod.payload_is_within_target(payload):
            issue = loudness_mod.payload_quality_issue(payload) or {}
            issues.add(
                "loudness_out_of_target", FAIL,
                _as_text(issue.get("message")) or f"{sid} 口播响度超出目标 -16±1 LUFS / TP<=-1.5",
                shot_id=sid, field="audio_meta.loudness",
                actual=issue.get("integrated_lufs", issue.get("true_peak_dbtp")),
                expected="[-17,-15] LUFS, TP<=-1.5", auto_fixable=True,
            )

    # final (成品) measurement. The production stitch graph records the real
    # measured loudness of the finished film under ``stitch_result.loudness``
    # (integrated/true-peak + narration_target + ducking), so that is a valid
    # source of the final measurement — the report must never be "missing"
    # forever once the film has actually been stitched.
    final_path = _as_text(state.get("final_video_path"))
    stitch = _as_dict(state.get("stitch_result"))
    final_meas = _as_dict(
        state.get("final_measurement")
        or state.get("final_loudness")
        or stitch.get("loudness")
    )
    if final_path:
        if not final_meas:
            issues.add(
                "final_measurement_missing", WARNING,
                "成品视频缺少最终 loudness 实测",
                field="final_measurement", actual=None, expected="present",
                auto_fixable=True,
            )
        elif not loudness_mod.payload_is_within_target(final_meas) and not loudness_mod.is_within_target(
            final_meas.get("measurement", final_meas), final_meas.get("target")
        ):
            issues.add(
                "final_loudness_out_of_target", FAIL,
                "成品 loudness 实测超出目标 -16±1 LUFS / TP<=-1.5",
                field="final_measurement", actual=None,
                expected="[-17,-15] LUFS, TP<=-1.5", auto_fixable=True,
            )


# --------------------------------------------------------------------------
# semantic animation checks
# --------------------------------------------------------------------------
def _iter_semantic(shot_raw: dict[str, Any], assets: dict[str, Any]) -> dict[str, Any]:
    for source in (shot_raw, assets):
        sem = _as_dict(_as_dict(source).get("semantic_animations"))
        if sem:
            return sem
    return {}


def _check_animations(shots: list[dict[str, Any]], issues: _Issues) -> None:
    for shot in shots:
        sid = _as_text(shot["shot_id"])
        assets = shot["assets"]
        raw = shot["raw"]
        sem = _iter_semantic(raw, assets)
        if not sem:
            continue
        narration = _shot_narration(raw)
        norm_narr = subtitles_mod.normalize_lexical(narration)
        duration = shot["duration"]
        cues = _shot_cues(raw, assets) or animations_mod.resolve_cues(raw, assets, duration)

        primary = _as_dict(sem.get("primary"))
        anchors = [a for a in _as_list(sem.get("anchors")) if isinstance(a, dict)]

        # at most one primary anchor.
        primary_like = [a for a in anchors if bool(a.get("is_primary"))]
        if isinstance(sem.get("primary"), list) and len(sem.get("primary")) > 1:
            issues.add(
                "animation_multiple_primary", FAIL,
                f"{sid} 声明了多个 primary 动画（最多 1 个）",
                shot_id=sid, field="semantic_animations", actual=len(sem.get("primary")), expected="<=1",
            )
        elif len(primary_like) > 1:
            issues.add(
                "animation_multiple_primary", FAIL,
                f"{sid} 有 {len(primary_like)} 个 primary 动画（最多 1 个）",
                shot_id=sid, field="semantic_animations", actual=len(primary_like), expected="<=1",
            )

        check_anchors = list(anchors)
        if primary and primary not in check_anchors:
            check_anchors.append(primary)

        for anchor in check_anchors:
            phrase = _as_text(anchor.get("phrase")).strip()
            atype = _as_text(anchor.get("type")).strip()
            is_semantic = bool(phrase)

            if is_semantic and phrase:
                if subtitles_mod.normalize_lexical(phrase) not in norm_narr:
                    issues.add(
                        "animation_phrase_missing", FAIL,
                        f"{sid} 动画短语「{phrase}」不在旁白中",
                        shot_id=sid, field="semantic_animations", actual=phrase, expected="in narration",
                        auto_fixable=True,
                    )
                    continue

            if atype and atype not in animations_mod.SEMANTIC_TYPES:
                issues.add(
                    "animation_type_invalid", FAIL,
                    f"{sid} 动画语义类型「{atype}」非法",
                    shot_id=sid, field="semantic_animations", actual=atype,
                    expected=list(animations_mod.SEMANTIC_TYPES), auto_fixable=True,
                )

            start = _as_float(anchor.get("start_sec"))
            end = _as_float(anchor.get("end_sec"))
            if is_semantic:
                # semantic timing must come from the phrase's cue interval — a
                # bare 0/0 or end<=start is not phrase-derived.
                if end <= start + _TOL:
                    issues.add(
                        "animation_time_invalid", FAIL,
                        f"{sid} 动画时间无效（end<=start）",
                        shot_id=sid, field="semantic_animations",
                        actual=[start, end], expected="end>start", auto_fixable=True,
                    )
                else:
                    loc = animations_mod.locate_phrase_time(phrase, cues, duration)
                    if loc.get("matched") and loc.get("start") is not None:
                        want = _as_float(loc.get("start"))
                        # An anchor whose start was uniformly zeroed (start≈0 but
                        # the phrase actually lands later) is the more specific,
                        # more severe defect — report ONLY that, never also the
                        # generic mismatch, so the same anchor is not penalised
                        # twice.
                        if start <= _TOL and want > 0.5:
                            issues.add(
                                "animation_time_uniform_zero", FAIL,
                                f"{sid} 语义动画起点被统一置 0，未按短语定位",
                                shot_id=sid, field="semantic_animations",
                                actual=0.0, expected=round(want, 2), auto_fixable=True,
                            )
                        elif abs(want - start) > 0.75:
                            issues.add(
                                "animation_time_mismatch", WARNING,
                                f"{sid} 动画起点 {start:.2f}s 与短语所在字幕 {want:.2f}s 不符",
                                shot_id=sid, field="semantic_animations",
                                actual=round(start, 2), expected=round(want, 2), auto_fixable=True,
                            )
            else:
                # non-semantic decoration must start at 0.
                if start > _TOL:
                    issues.add(
                        "animation_nonsemantic_start", WARNING,
                        f"{sid} 非语义动画起点应统一为 0，实际 {start:.2f}s",
                        shot_id=sid, field="semantic_animations",
                        actual=round(start, 2), expected=0.0, auto_fixable=True,
                    )

            # chart / timeline gate: these types demand >=2 comparable data
            # points; a phrase that does not satisfy the gate must not use them.
            if atype in {"chart", "timeline"} and phrase:
                detected = {c["type"] for c in animations_mod.classify_semantics(narration)}
                if atype not in detected:
                    issues.add(
                        "animation_gate_failed", FAIL,
                        f"{sid} 使用 {atype} 但旁白不满足其门槛（<2 个可比较节点）",
                        shot_id=sid, field="semantic_animations",
                        actual=atype, expected=">=2 data points", auto_fixable=True,
                    )

            # a claimed template must carry all three real ids.
            claimed = any(_as_text(anchor.get(k)) for k in ("structure_id", "motion_id", "visual_id"))
            if claimed:
                missing = [k for k in ("structure_id", "motion_id", "visual_id") if not _as_text(anchor.get(k))]
                if missing:
                    issues.add(
                        "animation_template_incomplete", FAIL,
                        f"{sid} 动画模板缺少 {', '.join(missing)}",
                        shot_id=sid, field="semantic_animations",
                        actual=missing, expected="structure/motion/visual", auto_fixable=True,
                    )


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def _prepare_shots(state: dict[str, Any]) -> list[dict[str, Any]]:
    shot_plan = [s for s in _as_list(state.get("shot_plan")) if isinstance(s, dict)]
    shot_assets = _as_dict(state.get("shot_assets"))
    rows: list[dict[str, Any]] = []
    for index, shot in enumerate(shot_plan, start=1):
        sid = _as_text(shot.get("shot_id")) or f"shot_{index:02d}"
        assets = _as_dict(shot_assets.get(sid)) or _as_dict(shot_assets.get(_as_text(shot.get("shot_id"))))
        rows.append({
            "index": index,
            "shot_id": sid,
            "raw": shot,
            "assets": assets,
            "duration": _shot_duration(shot, assets),
        })
    return rows


def evaluate_project(
    state: dict[str, Any], overrides: Any = None
) -> dict[str, Any]:
    """Evaluate the whole 0907 cut and return a stable quality report.

    ``overrides`` may be passed explicitly (tests / callers); otherwise the
    persisted ``quality_overrides0907`` map on ``state`` is used. Every reported
    issue carries a stable ``issue_id`` and an ``user_override`` annotation.
    """
    state = _as_dict(state)
    if overrides is None:
        overrides = state.get("quality_overrides0907")
    override_map = normalize_overrides(overrides)
    fix_attempts = normalize_fix_attempts(state.get("quality_fix_attempts0907"))

    shots = _prepare_shots(state)
    total_duration = pacing_mod._total_duration_of(state)

    issues = _Issues(override_map, fix_attempts)
    _check_subtitles(shots, issues)
    _check_footage(shots, issues)
    _check_pacing(shots, total_duration, issues)
    _check_loudness(shots, state, issues)
    _check_animations(shots, issues)

    items = issues.items
    counts = {FAIL: 0, WARNING: 0, INFO: 0}
    for issue in items:
        counts[issue["severity"]] = counts.get(issue["severity"], 0) + 1

    status = status_from_issues(items)
    score = score_from_issues(items)
    return {
        "status": status,
        "score": score,
        "summary": {
            "pass": status == "pass",
            "counts": counts,
            "total_issues": len(items),
            "auto_fixable": sum(1 for i in items if i["auto_fixable"]),
            "model_shot_count": len(shots),
            "minimum_model_shots": pacing_mod.minimum_model_shots(total_duration),
            "total_duration_sec": round(total_duration, 3),
        },
        "issues": items,
        "policy_version": "0907",
    }


# --------------------------------------------------------------------------
# auto-fix proposal — safe, deterministic, never silently touches narration
# --------------------------------------------------------------------------
# Which issue codes map to which safe internal recovery action. Anything that
# would rewrite the spoken narration is deliberately absent.
_SUBTITLE_CODES = {"subtitle_char_mismatch", "subtitle_punctuation",
                   "subtitle_too_many_lines", "subtitle_line_too_long"}
_FOOTAGE_CODES = {"footage_coverage_short", "footage_coverage_invalid",
                  "footage_duplicate_in_playlist", "footage_duplicate_adjacent",
                  "footage_duplicate_global", "footage_freeze_or_loop",
                  "subclip_too_short", "subclip_rate_too_slow", "shot_too_few_subclips"}
_LOUDNESS_CODES = {"loudness_missing", "loudness_not_normalized",
                   "loudness_out_of_target", "final_measurement_missing",
                   "final_loudness_out_of_target"}
_ANIMATION_CODES = {"animation_phrase_missing", "animation_type_invalid",
                    "animation_time_invalid", "animation_time_mismatch",
                    "animation_time_uniform_zero", "animation_nonsemantic_start",
                    "animation_gate_failed", "animation_template_incomplete",
                    "animation_multiple_primary"}
_PACING_CODES = {"too_few_model_shots", "shot_too_long"}


def propose_fixes(state: dict[str, Any], report: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Propose only *provably safe* automatic recovery actions.

    Returns a list of items shaped either as an applicable op
    (``{"op": ..., ...}``, ready for the transaction manager) or an internal
    action (``{"action": ..., ...}``) the runner turns into a scoped
    regeneration. Narration is never silently rewritten; a duplicate / coverage
    fix is a *re-plan / re-retrieve* request, and loudness is a *normalize*
    marker — if the external footage or ffmpeg turns out to be unavailable the
    underlying issue simply stays visible (nothing here fakes a fix).
    """
    state = _as_dict(state)
    if report is None:
        report = evaluate_project(state)
    issues = [i for i in _as_list(report.get("issues")) if isinstance(i, dict)]

    proposals: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _emit(item: dict[str, Any], key: str) -> None:
        if key in seen:
            return
        seen.add(key)
        proposals.append(item)

    subtitle_shots: set[str] = set()
    animation_shots: set[str] = set()
    footage_shots: set[str] = set()
    loudness_shots: set[str] = set()

    for issue in issues:
        if not issue.get("auto_fixable"):
            continue
        if issue.get("user_override"):
            # respect the user's explicit choice — never auto-revert it.
            continue
        code = _as_text(issue.get("code"))
        sid = _as_text(issue.get("shot_id"))
        subclip_id = _as_text(issue.get("subclip_id"))
        if code in _SUBTITLE_CODES:
            subtitle_shots.add(sid)
        elif code in _ANIMATION_CODES:
            animation_shots.add(sid)
        elif code in _FOOTAGE_CODES:
            if subclip_id and code in {"subclip_too_short", "footage_duplicate_in_playlist",
                                       "footage_duplicate_adjacent", "footage_duplicate_global"}:
                _emit(
                    {"action": "replan_subclip", "subclip_id": subclip_id, "shot_id": sid,
                     "reason": code},
                    f"replan_subclip:{subclip_id}",
                )
            else:
                footage_shots.add(sid or "__global__")
        elif code in _LOUDNESS_CODES:
            if sid:
                loudness_shots.add(sid)
            else:
                _emit({"action": "normalize_final_audio", "reason": code}, "normalize_final")
        elif code in _PACING_CODES:
            _emit({"op": "optimize_cut", "scope": "global", "policy": "pacing"},
                  "optimize_cut:pacing")

    for sid in sorted(subtitle_shots):
        _emit({"action": "rebuild_subtitles", "shot_id": sid}, f"rebuild_subtitles:{sid}")
    for sid in sorted(animation_shots):
        _emit({"action": "rebuild_semantic_animation", "shot_id": sid},
              f"rebuild_semantic_animation:{sid}")
    for sid in sorted(footage_shots):
        if sid == "__global__":
            _emit({"op": "optimize_cut", "scope": "global", "policy": "footage"},
                  "optimize_cut:footage")
        else:
            _emit({"action": "replan_playlist", "shot_id": sid}, f"replan_playlist:{sid}")
    for sid in sorted(loudness_shots):
        _emit({"action": "normalize_audio", "shot_id": sid}, f"normalize_audio:{sid}")

    return proposals


def advance_fix_attempts(
    state: dict[str, Any],
    prev_report: dict[str, Any] | None,
    new_report: dict[str, Any] | None,
) -> dict[str, int]:
    """Update ``quality_fix_attempts0907`` after a regeneration cycle (I2).

    An auto-fixable issue that was present *before* a render and is still present
    *after* it counts as one failed fix attempt; an issue that disappeared has
    its counter cleared (a later recurrence starts fresh). Persisted on ``state``
    (whitelisted key) so it survives reload / version restore.
    """
    attempts = dict(normalize_fix_attempts(state.get("quality_fix_attempts0907")))
    prev_fixable = {
        _as_text(i.get("issue_id"))
        for i in _as_list(_as_dict(prev_report).get("issues"))
        if isinstance(i, dict) and i.get("auto_fixable")
    }
    new_ids = {
        _as_text(i.get("issue_id"))
        for i in _as_list(_as_dict(new_report).get("issues"))
        if isinstance(i, dict)
    }
    for iid in prev_fixable:
        if iid and iid in new_ids:
            attempts[iid] = attempts.get(iid, 0) + 1
    for iid in list(attempts.keys()):
        if iid not in new_ids:
            attempts.pop(iid, None)
    state["quality_fix_attempts0907"] = attempts
    return attempts


__all__ = [
    "SEVERITY_WEIGHTS",
    "FIX_ATTEMPT_LIMIT",
    "evaluate_project",
    "propose_fixes",
    "score_from_issues",
    "status_from_issues",
    "issue_id_for",
    "normalize_overrides",
    "normalize_fix_attempts",
    "advance_fix_attempts",
]
