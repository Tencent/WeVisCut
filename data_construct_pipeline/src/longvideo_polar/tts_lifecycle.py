"""TTS service lifecycle: probe health, auto-start if down, wait for readiness.

The Qwen3-TTS service (``AICutPipeline/server/qwen3_tts_service.py``) is a
FastAPI + uvicorn app that exposes:

  - ``GET /health``  → ``{"status": "ok", "loaded": bool, ...}``
  - ``POST /tts``    → single-item synthesis
  - ``POST /tts_batch`` → batch synthesis

Without TTS running, ``Tool_BatchVisualAudio`` cannot produce audio tracks,
and ``Tool_StitchVideo`` will never succeed — so the pipeline cannot produce
a final video. This module ensures TTS is alive before a rollout begins.

Usage::

    from longvideo_polar.tts_lifecycle import ensure_tts_running
    status = ensure_tts_running(
        service_url="http://127.0.0.1:7865",
        launch_script="/path/to/run_qwen3_tts_service.sh",
        auto_start=True,
        startup_timeout_sec=300,
    )
    if not status.healthy:
        logger.warning("TTS not ready; rollout will proceed but may fail at StitchVideo")
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class TTSServiceStatus:
    """Outcome of a TTS lifecycle check."""

    healthy: bool = False
    service_url: str = ""
    loaded: bool = False
    free_model_replicas: int = 0
    auto_started: bool = False
    process_pid: int | None = None
    error: str = ""

    def __repr__(self) -> str:
        parts = [
            f"healthy={self.healthy}",
            f"url={self.service_url}",
            f"loaded={self.loaded}",
            f"free_replicas={self.free_model_replicas}",
        ]
        if self.auto_started:
            parts.append(f"auto_started(pid={self.process_pid})")
        if self.error:
            parts.append(f"error={self.error}")
        return f"TTSServiceStatus({', '.join(parts)})"


def probe_tts_health(
    service_url: str,
    *,
    timeout: float = 5.0,
) -> dict[str, Any] | None:
    """GET ``/health`` on the TTS service.

    Returns:
        Parsed JSON dict if the service responded with 200, else ``None``.
    """
    url = f"{service_url.rstrip('/')}/health"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict):
                return data
        logger.debug("TTS health probe: %s → %s", url, resp.status_code)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("TTS health probe failed: %s", exc)
        return None


def is_tts_healthy(service_url: str, *, timeout: float = 5.0) -> bool:
    """Check if TTS service is alive AND models are loaded."""
    health = probe_tts_health(service_url, timeout=timeout)
    if not health:
        return False
    return health.get("status") == "ok" and health.get("loaded", False)


def launch_tts_service(
    launch_script: str | Path,
    *,
    env_overrides: dict[str, str] | None = None,
    log_path: str | Path | None = None,
    python_bin: str | None = None,
) -> subprocess.Popen:
    """Launch the TTS service in the background.

    Args:
        launch_script: Path to ``run_qwen3_tts_service.sh``.
        env_overrides: Additional env vars to set (e.g., GPU visibility,
            batch size, IO workers).
        log_path: Where to redirect stdout/stderr. Defaults to
            ``/tmp/qwen3_tts_service.log``.
        python_bin: Override the Python interpreter used by the script.

    Returns:
        The ``subprocess.Popen`` handle for the launched service.
    """
    script_path = Path(launch_script).expanduser().resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"TTS launch script not found: {script_path}")

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    if python_bin:
        env["AICUT_QWEN3_TTS_PYTHON"] = python_bin

    log_file = Path(log_path) if log_path else Path("/tmp/qwen3_tts_service.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Launching TTS service: %s (log: %s)", script_path, log_file)
    fh = log_file.open("a", buffering=1)
    fh.write(f"\n{'='*60}\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] TTS service launch\n{'='*60}\n")

    proc = subprocess.Popen(
        ["bash", str(script_path)],
        env=env,
        stdout=fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    logger.info("TTS service launched: pid=%d", proc.pid)
    return proc


def ensure_tts_running(
    *,
    service_url: str,
    launch_script: str | Path | None = None,
    auto_start: bool = True,
    startup_timeout_sec: int = 300,
    env_overrides: dict[str, str] | None = None,
    python_bin: str | None = None,
    log_path: str | Path | None = None,
) -> TTSServiceStatus:
    """Ensure the TTS service is running and healthy before a rollout.

    Flow:
      1. Probe ``GET /health`` on ``service_url``.
      2. If healthy → return immediately.
      3. If not healthy AND ``auto_start`` → launch the TTS service script,
         then poll ``/health`` until healthy or ``startup_timeout_sec`` elapses.
      4. If still not healthy → return status with ``healthy=False`` and an
         error message. Caller decides whether to proceed anyway.

    Args:
        service_url: TTS service base URL (e.g., ``http://127.0.0.1:7865``).
        launch_script: Path to ``run_qwen3_tts_service.sh``. Required if
            ``auto_start=True`` and TTS is not already running.
        auto_start: Whether to launch the TTS service if it's not running.
        startup_timeout_sec: Maximum seconds to wait for the service to become
            healthy after launch.
        env_overrides: Additional env vars for the TTS service (e.g.,
            ``CUDA_VISIBLE_DEVICES``, ``AICUT_QWEN3_TTS_MAX_BATCH_SIZE``).
        python_bin: Override Python interpreter for the TTS script.
        log_path: Log file path for the TTS service.

    Returns:
        :class:`TTSServiceStatus` describing the outcome.
    """
    status = TTSServiceStatus(service_url=service_url)

    # Step 1: Quick probe — is TTS already running?
    health = probe_tts_health(service_url, timeout=3.0)
    if health and health.get("status") == "ok":
        status.healthy = True
        status.loaded = health.get("loaded", False)
        status.free_model_replicas = health.get("free_model_replicas", 0)
        logger.info(
            "TTS service already running: %s (loaded=%s, free_replicas=%d)",
            service_url,
            status.loaded,
            status.free_model_replicas,
        )
        return status

    # Step 2: If not running, decide whether to auto-start
    if not auto_start:
        status.error = "TTS not running and auto_start=False"
        logger.warning(status.error)
        return status

    if not launch_script:
        status.error = "TTS not running and no launch_script provided"
        logger.warning(status.error)
        return status

    # Step 3: Launch TTS service
    try:
        proc = launch_tts_service(
            launch_script,
            env_overrides=env_overrides,
            log_path=log_path,
            python_bin=python_bin,
        )
        status.auto_started = True
        status.process_pid = proc.pid
    except Exception as exc:  # noqa: BLE001
        status.error = f"Failed to launch TTS service: {exc}"
        logger.error(status.error)
        return status

    # Step 4: Poll /health until healthy or timeout
    deadline = time.time() + startup_timeout_sec
    poll_interval = 5.0
    last_health: dict[str, Any] | None = None

    while time.time() < deadline:
        # Check if process died
        if proc.poll() is not None:
            status.error = f"TTS service process exited prematurely (code={proc.returncode})"
            logger.error(status.error)
            return status

        # Probe health
        last_health = probe_tts_health(service_url, timeout=5.0)
        if last_health and last_health.get("status") == "ok":
            loaded = last_health.get("loaded", False)
            if loaded:
                status.healthy = True
                status.loaded = True
                status.free_model_replicas = last_health.get("free_model_replicas", 0)
                logger.info(
                    "TTS service became healthy after %.1fs: %s (free_replicas=%d)",
                    startup_timeout_sec - (deadline - time.time()),
                    service_url,
                    status.free_model_replicas,
                )
                return status
            else:
                logger.debug("TTS service responding but models not yet loaded; waiting...")
        else:
            logger.debug("TTS service not yet responding; waiting...")

        time.sleep(poll_interval)

    # Timeout reached
    status.error = (
        f"TTS service did not become healthy within {startup_timeout_sec}s "
        f"(last health: {last_health})"
    )
    logger.error(status.error)
    return status


# ─── High-concurrency presets for full-batch annotation ──────────────────

_CONCURRENCY_PRESETS: dict[str, dict[str, str]] = {
    "conservative": {
        "AICUT_QWEN3_TTS_MAX_BATCH_SIZE": "32",
        "AICUT_QWEN3_TTS_IO_WORKERS": "4",
        "AICUT_QWEN3_TTS_MODEL_REPLICAS": "1",
    },
    "balanced": {
        "AICUT_QWEN3_TTS_MAX_BATCH_SIZE": "64",
        "AICUT_QWEN3_TTS_IO_WORKERS": "8",
        "AICUT_QWEN3_TTS_MODEL_REPLICAS": "1",
    },
    "max_gpu": {
        "AICUT_QWEN3_TTS_MAX_BATCH_SIZE": "128",
        "AICUT_QWEN3_TTS_IO_WORKERS": "16",
        "AICUT_QWEN3_TTS_MODEL_REPLICAS": "2",
    },
}


def get_concurrency_preset(preset: str) -> dict[str, str]:
    """Get env-var overrides for a named concurrency preset.

    Presets:
      - ``conservative``: Small batches, few workers. Safe for shared GPUs.
      - ``balanced``: Default. Matches the shipped script's defaults.
      - ``max_gpu``: Large batches, many workers, 2 model replicas.
        Saturates a dedicated GPU. Use for full-batch annotation.
    """
    if preset not in _CONCURRENCY_PRESETS:
        raise ValueError(
            f"Unknown concurrency preset: {preset!r}. "
            f"Available: {list(_CONCURRENCY_PRESETS.keys())}"
        )
    return dict(_CONCURRENCY_PRESETS[preset])


# ─── Preflight check with batch-mode concurrency safety ──────────────────

import threading

_tts_lock = threading.Lock()
_tts_checked = False


def preflight_tts(
    *,
    service_url: str | None = None,
    launch_script: str | Path | None = None,
    auto_start: bool | None = None,
    startup_timeout_sec: int | None = None,
    concurrency_preset: str | None = None,
    python_bin: str | None = None,
    log_path: str | Path | None = None,
) -> TTSServiceStatus:
    """Pre-rollout TTS liveness check with batch-mode concurrency safety.

    In batch mode with ``workers=N``, N rollouts may start simultaneously.
    The first one acquires the lock, probes TTS health, and launches the TTS
    service if needed. Subquent rollouts see ``_tts_checked=True`` and skip
    the check entirely, avoiding port conflicts and redundant launches.

    Configuration is driven by environment variables (with sensible defaults):

      - ``LVP_TTS_AUTO_START`` (default: ``"1"``) — whether to auto-start TTS
      - ``LVP_TTS_LAUNCH_SCRIPT`` (default: local copy under
        ``longvideo_polar/services/tts/run_qwen3_tts_service.sh``)
      - ``LVP_TTS_STARTUP_TIMEOUT`` (default: ``"300"``) — max seconds to wait
      - ``LVP_TTS_CONCURRENCY_PRESET`` (default: ``"balanced"``) — one of
        ``conservative``, ``balanced``, ``max_gpu``
      - ``LVP_TTS_PYTHON`` (default: ``""`` = inherit) — Python bin for TTS
      - ``LVP_TTS_LOG`` (default: ``/tmp/qwen3_tts_service.log``) — log path

    Args:
        service_url: Override TTS service URL.
        launch_script: Override TTS launch script path.
        auto_start: Override auto-start behavior.
        startup_timeout_sec: Override startup timeout.
        concurrency_preset: Override concurrency preset.
        python_bin: Override Python interpreter for TTS script.
        log_path: Override log file path.

    Returns:
        :class:`TTSServiceStatus` describing the TTS service state.
    """
    global _tts_checked

    svc_url = service_url or os.environ.get(
        "AICUT_TTS_URL", "http://127.0.0.1:7865"
    )

    # Resolve auto_start
    if auto_start is None:
        auto_start = os.environ.get("LVP_TTS_AUTO_START", "1").lower() in (
            "1", "true", "yes", "on",
        )

    # Resolve launch_script — default to the local copy bundled with longvideo_polar
    if launch_script is None:
        env_script = os.environ.get("LVP_TTS_LAUNCH_SCRIPT", "")
        if env_script:
            launch_script = env_script
        else:
            # Compute path relative to this module's location
            _mod_dir = Path(__file__).resolve().parent
            _local_script = _mod_dir.parent.parent / "services" / "tts" / "run_qwen3_tts_service.sh"
            launch_script = str(_local_script)

    # Resolve startup_timeout
    if startup_timeout_sec is None:
        startup_timeout_sec = int(
            os.environ.get("LVP_TTS_STARTUP_TIMEOUT", "300")
        )

    # Resolve concurrency preset
    if concurrency_preset is None:
        concurrency_preset = os.environ.get(
            "LVP_TTS_CONCURRENCY_PRESET", "balanced"
        )

    # Resolve python_bin
    if python_bin is None:
        python_bin = os.environ.get("LVP_TTS_PYTHON", "") or None

    # Resolve log_path
    if log_path is None:
        log_path = os.environ.get(
            "LVP_TTS_LOG", "/tmp/qwen3_tts_service.log"
        )

    # Build env_overrides from concurrency preset
    env_overrides = get_concurrency_preset(concurrency_preset)

    # Concurrency-safe check: skip if already checked by another worker
    with _tts_lock:
        if _tts_checked:
            # Another worker already did the preflight; just do a quick probe
            health = probe_tts_health(svc_url, timeout=3.0)
            if health and health.get("status") == "ok":
                return TTSServiceStatus(
                    healthy=True,
                    service_url=svc_url,
                    loaded=health.get("loaded", False),
                    free_model_replicas=health.get("free_model_replicas", 0),
                )
            # TTS not responding — fall through to try launching
            logger.warning(
                "TTS not responding after another worker's preflight; "
                "attempting launch"
            )

        status = ensure_tts_running(
            service_url=svc_url,
            launch_script=launch_script if auto_start else None,
            auto_start=auto_start,
            startup_timeout_sec=startup_timeout_sec,
            env_overrides=env_overrides,
            python_bin=python_bin,
            log_path=log_path,
        )
        _tts_checked = True

    if not status.healthy:
        logger.warning(
            "TTS preflight: service not healthy at %s — rollout will proceed "
            "but may fail at StitchVideo (error: %s)",
            svc_url,
            status.error,
        )
    else:
        logger.info(
            "TTS preflight: service healthy at %s (loaded=%s, free_replicas=%d%s)",
            svc_url,
            status.loaded,
            status.free_model_replicas,
            ", auto_started" if status.auto_started else "",
        )

    return status
