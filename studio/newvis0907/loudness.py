from __future__ import annotations

import json
import math
import os
import stat
import subprocess
import uuid
from pathlib import Path
from typing import Any

TARGET_INTEGRATED_LUFS = -16.0
TARGET_TRUE_PEAK_DBTP = -1.5
TARGET_TOLERANCE_LU = 1.0
TARGET_LRA_LU = 11.0

_NUMERIC_FIELDS = (
    "input_i",
    "input_tp",
    "input_lra",
    "input_thresh",
    "output_i",
    "output_tp",
    "output_lra",
    "output_thresh",
    "target_offset",
)


class LoudnessError(RuntimeError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.details = dict(details)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            **self.details,
        }


def target_profile() -> dict[str, float]:
    return {
        "integrated_lufs": TARGET_INTEGRATED_LUFS,
        "true_peak_dbtp": TARGET_TRUE_PEAK_DBTP,
        "tolerance_lu": TARGET_TOLERANCE_LU,
        "lra_lu": TARGET_LRA_LU,
    }


def target_matches(target: Any) -> bool:
    if not isinstance(target, dict):
        return False
    expected = target_profile()
    for key, value in expected.items():
        try:
            got = float(target.get(key))
        except (TypeError, ValueError):
            return False
        if abs(got - value) > 1e-6:
            return False
    return True


def _structured_failure(
    code: str,
    message: str,
    *,
    status: str = "failed",
    input_measurement: dict[str, Any] | None = None,
    output_measurement: dict[str, Any] | None = None,
    normalized: bool = False,
    quality: dict[str, Any] | None = None,
    **details: Any,
) -> dict[str, Any]:
    payload = {
        "input": input_measurement,
        "output": output_measurement,
        "status": status,
        "normalized": normalized,
        "target": target_profile(),
        "error": {"code": code, "message": message, **details},
    }
    if quality:
        payload["quality_issue"] = quality
    return payload


def _extract_json_candidates(text: str) -> list[str]:
    out: list[str] = []
    depth = 0
    start: int | None = None
    for idx, ch in enumerate(text or ""):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth <= 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                out.append(text[start:idx + 1])
                start = None
    return out


def _finite_number(data: dict[str, Any], field: str) -> float:
    if field not in data:
        raise LoudnessError(
            "missing_measurement_field",
            f"ffmpeg loudnorm JSON missing field: {field}",
            field=field,
        )
    try:
        value = float(data[field])
    except (TypeError, ValueError):
        raise LoudnessError(
            "invalid_measurement_field",
            f"ffmpeg loudnorm JSON field is not numeric: {field}",
            field=field,
            value=data.get(field),
        ) from None
    if not math.isfinite(value):
        raise LoudnessError(
            "nonfinite_measurement_field",
            f"ffmpeg loudnorm JSON field is non-finite: {field}",
            field=field,
            value=data.get(field),
        )
    return value


def parse_loudnorm_json(stderr: str) -> dict[str, Any]:
    candidates = _extract_json_candidates(stderr or "")
    if not candidates:
        raise LoudnessError("invalid_loudnorm_json", "ffmpeg stderr contains no loudnorm JSON")

    best_error: LoudnessError | None = None

    def _priority(err: LoudnessError) -> int:
        if err.code in {"missing_measurement_field", "invalid_measurement_field", "nonfinite_measurement_field"}:
            return 3
        if err.code == "invalid_loudnorm_json":
            return 1
        return 2

    for raw in reversed(candidates):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            current = LoudnessError(
                "invalid_loudnorm_json",
                "ffmpeg stderr contains a non-JSON loudnorm candidate",
                detail=str(exc),
            )
            if best_error is None or _priority(current) > _priority(best_error):
                best_error = current
            continue
        if not isinstance(data, dict):
            current = LoudnessError(
                "invalid_loudnorm_json",
                "ffmpeg stderr loudnorm candidate is not a JSON object",
                candidate_type=type(data).__name__,
            )
            if best_error is None or _priority(current) > _priority(best_error):
                best_error = current
            continue
        try:
            measurement: dict[str, Any] = {}
            for field in _NUMERIC_FIELDS:
                measurement[field] = _finite_number(data, field)
            measurement["normalization_type"] = str(data.get("normalization_type", "") or "")
            return measurement
        except LoudnessError as exc:
            if best_error is None or _priority(exc) > _priority(best_error):
                best_error = exc
            continue

    if best_error is not None:
        raise best_error
    raise LoudnessError("invalid_loudnorm_json", "ffmpeg stderr contains no valid loudnorm JSON object")


def _run_ffmpeg(
    cmd: list[str],
    *,
    runner: Any,
    timeout_sec: int,
) -> subprocess.CompletedProcess[str] | Any:
    return runner(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )


def measure_loudness(
    path: str,
    ffmpeg_binary: str = "ffmpeg",
    runner: Any = subprocess.run,
    timeout_sec: int = 30,
) -> dict[str, Any]:
    src = Path(path).expanduser()
    if not src.is_file():
        return {
            "ok": False,
            "status": "failed",
            "target": target_profile(),
            "error": {
                "code": "audio_file_missing",
                "message": f"audio file does not exist: {src}",
                "path": str(src),
            },
        }

    cmd = [
        ffmpeg_binary,
        "-hide_banner",
        "-i",
        str(src),
        "-af",
        (
            f"loudnorm=I={TARGET_INTEGRATED_LUFS:g}:"
            f"TP={TARGET_TRUE_PEAK_DBTP:g}:"
            f"LRA={TARGET_LRA_LU:g}:print_format=json"
        ),
        "-f",
        "null",
        "-",
    ]
    try:
        proc = _run_ffmpeg(cmd, runner=runner, timeout_sec=timeout_sec)
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "status": "failed",
            "target": target_profile(),
            "error": {
                "code": "ffmpeg_timeout",
                "message": "ffmpeg loudness measurement timed out",
                "path": str(src),
            },
        }
    except FileNotFoundError:
        return {
            "ok": False,
            "status": "failed",
            "target": target_profile(),
            "error": {
                "code": "ffmpeg_not_found",
                "message": f"ffmpeg binary not found: {ffmpeg_binary}",
                "path": str(src),
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "status": "failed",
            "target": target_profile(),
            "error": {
                "code": "ffmpeg_runner_error",
                "message": str(exc),
                "path": str(src),
            },
        }

    if int(getattr(proc, "returncode", 1) or 0) != 0:
        return {
            "ok": False,
            "status": "failed",
            "target": target_profile(),
            "error": {
                "code": "ffmpeg_failed",
                "message": "ffmpeg loudness measurement failed",
                "path": str(src),
                "returncode": getattr(proc, "returncode", None),
                "stderr": str(getattr(proc, "stderr", "") or "")[-1200:],
            },
        }
    try:
        measurement = parse_loudnorm_json(str(getattr(proc, "stderr", "") or ""))
    except LoudnessError as exc:
        return {
            "ok": False,
            "status": "failed",
            "target": target_profile(),
            "error": {**exc.to_dict(), "path": str(src)},
        }
    return {
        "ok": True,
        "status": "measured",
        "target": target_profile(),
        "measurement": measurement,
    }


def _measurement_for_target_check(measurement: Any) -> tuple[float, float] | None:
    if not isinstance(measurement, dict):
        return None
    if "input_i" in measurement and "input_tp" in measurement:
        try:
            return float(measurement["input_i"]), float(measurement["input_tp"])
        except (TypeError, ValueError):
            return None
    if "output_i" in measurement and "output_tp" in measurement:
        try:
            return float(measurement["output_i"]), float(measurement["output_tp"])
        except (TypeError, ValueError):
            return None
    return None


def is_within_target(measurement: Any, target: dict[str, Any] | None = None) -> bool:
    target = target if target is not None else target_profile()
    pair = _measurement_for_target_check(measurement)
    if pair is None:
        return False
    integrated, true_peak = pair
    try:
        want_i = float(target["integrated_lufs"])
        tolerance = float(target["tolerance_lu"])
        ceiling = float(target["true_peak_dbtp"])
    except (KeyError, TypeError, ValueError):
        return False
    return abs(integrated - want_i) <= tolerance and true_peak <= ceiling + 1e-9


def quality_issue(measurement: Any, target: dict[str, Any] | None = None) -> dict[str, Any] | None:
    target = target if target is not None else target_profile()
    pair = _measurement_for_target_check(measurement)
    if pair is None:
        return {
            "code": "loudness_measurement_missing",
            "message": "缺少可用于 loudness 校验的 output/input measurement",
            "severity": "error",
            "field": "audio_meta.loudness",
        }
    integrated, true_peak = pair
    want_i = float(target["integrated_lufs"])
    tolerance = float(target["tolerance_lu"])
    ceiling = float(target["true_peak_dbtp"])
    if abs(integrated - want_i) > tolerance:
        return {
            "code": "loudness_out_of_target",
            "message": (
                f"口播 integrated loudness {integrated:.2f} LUFS 超出目标 "
                f"{want_i:.1f}±{tolerance:.1f}"
            ),
            "severity": "error",
            "field": "audio_meta.loudness",
            "integrated_lufs": round(integrated, 3),
            "target_integrated_lufs": want_i,
            "tolerance_lu": tolerance,
        }
    if true_peak > ceiling:
        return {
            "code": "true_peak_exceeded",
            "message": (
                f"口播 true peak {true_peak:.2f} dBTP 超过上限 {ceiling:.1f} dBTP"
            ),
            "severity": "error",
            "field": "audio_meta.loudness",
            "true_peak_dbtp": round(true_peak, 3),
            "target_true_peak_dbtp": ceiling,
        }
    return None


def _payload_measurement_for_target_check(payload: Any) -> tuple[Any, dict[str, Any] | None, str]:
    if not isinstance(payload, dict):
        return None, None, ""
    status = str(payload.get("status", "")).strip().lower()
    target = payload.get("target") if isinstance(payload.get("target"), dict) else None
    if status in {"normalized", "skipped"}:
        measurement = payload.get("output")
        if not isinstance(measurement, dict):
            measurement = payload.get("input")
        return measurement, target, status
    measurement = payload.get("input")
    if not isinstance(measurement, dict):
        measurement = payload.get("output")
    return measurement, target, status


def payload_is_within_target(payload: Any) -> bool:
    measurement, target, status = _payload_measurement_for_target_check(payload)
    if status not in {"normalized", "skipped"}:
        return False
    return is_within_target(measurement, target)


def payload_quality_issue(payload: Any) -> dict[str, Any] | None:
    measurement, target, status = _payload_measurement_for_target_check(payload)
    if status == "failed":
        error = payload.get("error") if isinstance(payload, dict) and isinstance(payload.get("error"), dict) else {}
        return {
            "code": str(error.get("code") or "loudness_normalization_failed"),
            "message": str(error.get("message") or "loudness normalization failed"),
            "severity": "error",
            "field": "audio_meta.loudness",
        }
    if status and status not in {"normalized", "skipped"}:
        return {
            "code": "loudness_status_invalid",
            "message": f"loudness status not verifiable: {status}",
            "severity": "error",
            "field": "audio_meta.loudness",
        }
    return quality_issue(measurement, target)


def _cleanup(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def normalize_narration(
    path: str,
    ffmpeg_binary: str = "ffmpeg",
    runner: Any = subprocess.run,
    timeout_sec: int = 30,
) -> dict[str, Any]:
    src = Path(path).expanduser()
    if not src.is_file():
        return _structured_failure(
            "audio_file_missing",
            f"audio file does not exist: {src}",
            path=str(src),
        )

    first = measure_loudness(
        str(src),
        ffmpeg_binary=ffmpeg_binary,
        runner=runner,
        timeout_sec=timeout_sec,
    )
    if not first.get("ok"):
        err = first.get("error", {}) if isinstance(first.get("error"), dict) else {}
        return _structured_failure(
            str(err.get("code", "measure_failed")),
            str(err.get("message", "initial loudness measurement failed")),
            input_measurement=None,
            output_measurement=None,
            **{k: v for k, v in err.items() if k not in {"code", "message"}},
        )

    input_measurement = first.get("measurement", {})
    if is_within_target(input_measurement):
        return {
            "input": input_measurement,
            "output": input_measurement,
            "status": "skipped",
            "normalized": False,
            "target": target_profile(),
        }

    temp_path = src.with_name(f"{src.stem}.loudnorm.{uuid.uuid4().hex}{src.suffix}")
    cmd = [
        ffmpeg_binary,
        "-y",
        "-hide_banner",
        "-i",
        str(src),
        "-af",
        (
            f"loudnorm=I={TARGET_INTEGRATED_LUFS:g}:"
            f"TP={TARGET_TRUE_PEAK_DBTP:g}:"
            f"LRA={TARGET_LRA_LU:g}:"
            f"measured_I={float(input_measurement['input_i']):g}:"
            f"measured_TP={float(input_measurement['input_tp']):g}:"
            f"measured_LRA={float(input_measurement['input_lra']):g}:"
            f"measured_thresh={float(input_measurement['input_thresh']):g}:"
            f"offset={float(input_measurement['target_offset']):g}:"
            "linear=true:print_format=json"
        ),
        str(temp_path),
    ]

    try:
        proc = _run_ffmpeg(cmd, runner=runner, timeout_sec=timeout_sec)
    except subprocess.TimeoutExpired:
        _cleanup(temp_path)
        return _structured_failure(
            "ffmpeg_timeout",
            "ffmpeg loudness normalization timed out",
            input_measurement=input_measurement,
            path=str(src),
        )
    except FileNotFoundError:
        _cleanup(temp_path)
        return _structured_failure(
            "ffmpeg_not_found",
            f"ffmpeg binary not found: {ffmpeg_binary}",
            input_measurement=input_measurement,
            path=str(src),
        )
    except Exception as exc:  # noqa: BLE001
        _cleanup(temp_path)
        return _structured_failure(
            "ffmpeg_runner_error",
            str(exc),
            input_measurement=input_measurement,
            path=str(src),
        )

    if int(getattr(proc, "returncode", 1) or 0) != 0:
        _cleanup(temp_path)
        return _structured_failure(
            "ffmpeg_failed",
            "ffmpeg loudness normalization failed",
            input_measurement=input_measurement,
            path=str(src),
            returncode=getattr(proc, "returncode", None),
            stderr=str(getattr(proc, "stderr", "") or "")[-1200:],
        )
    if not temp_path.is_file() or temp_path.stat().st_size <= 0:
        _cleanup(temp_path)
        return _structured_failure(
            "normalized_file_missing",
            "ffmpeg normalization produced no output file",
            input_measurement=input_measurement,
            path=str(src),
        )

    second = measure_loudness(
        str(temp_path),
        ffmpeg_binary=ffmpeg_binary,
        runner=runner,
        timeout_sec=timeout_sec,
    )
    if not second.get("ok"):
        err = second.get("error", {}) if isinstance(second.get("error"), dict) else {}
        _cleanup(temp_path)
        return _structured_failure(
            str(err.get("code", "measure_failed")),
            str(err.get("message", "post-normalization measurement failed")),
            input_measurement=input_measurement,
            output_measurement=None,
            path=str(src),
            **{k: v for k, v in err.items() if k not in {"code", "message", "path"}},
        )

    output_measurement = second.get("measurement", {})
    issue = quality_issue(output_measurement)
    if issue is not None:
        _cleanup(temp_path)
        return _structured_failure(
            "loudness_verification_failed",
            issue["message"],
            input_measurement=input_measurement,
            output_measurement=output_measurement,
            quality=issue,
            path=str(src),
        )

    try:
        try:
            os.chmod(temp_path, stat.S_IMODE(src.stat().st_mode))
        except OSError:
            pass
        os.replace(temp_path, src)
    except OSError as exc:
        _cleanup(temp_path)
        return _structured_failure(
            "atomic_replace_failed",
            f"failed to atomically replace normalized file: {exc}",
            input_measurement=input_measurement,
            output_measurement=output_measurement,
            path=str(src),
        )

    return {
        "input": input_measurement,
        "output": output_measurement,
        "status": "normalized",
        "normalized": True,
        "target": target_profile(),
    }
