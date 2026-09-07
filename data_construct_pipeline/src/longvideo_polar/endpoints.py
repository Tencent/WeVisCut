"""A pool of served-model machines that rollouts are scheduled across.

The batch runner spawns one subprocess per episode. Single-endpoint config points
every one of them at the same vLLM, which caps a 9k-episode corpus at whatever one
machine can do. This module turns that single ``base_url`` into a pool:

- machines are added by ``ip:port`` alone — the scheme, the ``/v1`` suffix and the
  served model name are filled in by probing ``/v1/models``
- each machine carries a rollout capacity, and episodes are handed to whichever
  machine has the most free capacity, so a slow or newly added machine drains at
  its own rate instead of by fixed round-robin
- unreachable machines are quarantined rather than removed, and probed again on
  the next cycle, so a vLLM restart heals itself without touching the batch
- the pool is a JSON file, which is what lets the web UI edit it while a batch is
  running and lets the next process see the change

Assignment is per *episode*, not per request: a rollout is one subprocess with one
``--base-url``, and keeping an episode on one machine also keeps its prefix cache
warm across the ~140 calls it makes.

Proxy handling is load-bearing. Bare-IP intranet endpoints are often not on
a corporate proxy allow-list, so a request that honors ``HTTP_PROXY`` comes
back as a 403 — indistinguishable from a dead server. Bare-IP and loopback
hosts therefore bypass the environment proxy entirely.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8080
DEFAULT_PROBE_TIMEOUT = 8.0

# Rollouts per machine. Each rollout issues bursts of concurrent skill calls, so
# this multiplies against the server's own --max-num-seqs; 4 is a safe start for a
# 397B served with --max-num-seqs 16.
DEFAULT_ROLLOUT_CAPACITY = 4

_HOST_PORT_RE = re.compile(r"^(?P<host>[^/:\s]+)(?::(?P<port>\d+))?$")


def normalize_base_url(raw: str, *, default_port: int = DEFAULT_PORT) -> str:
    """Turn user input into a full OpenAI-compatible base URL.

    Accepts ``10.0.0.1``, ``10.0.0.1:8080``, ``http://host:8080`` and
    ``http://host:8080/v1``; always returns a ``.../v1`` form.

    Raises:
        ValueError: If nothing resembling a host can be parsed.
    """
    text = str(raw or "").strip().rstrip("/")
    if not text:
        raise ValueError("endpoint is empty")

    if "://" not in text:
        match = _HOST_PORT_RE.match(text.split("/")[0])
        if match:
            port = match.group("port") or str(default_port)
            text = f"http://{match.group('host')}:{port}"
        else:
            text = f"http://{text}"

    parsed = urlparse(text)
    if not parsed.hostname:
        raise ValueError(f"cannot parse host from {raw!r}")
    port = parsed.port or default_port
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return f"{parsed.scheme or 'http'}://{parsed.hostname}:{port}{path}"


def bypasses_proxy(base_url: str) -> bool:
    """Whether this host must be reached directly, ignoring ``HTTP_PROXY``."""
    host = urlparse(base_url).hostname or ""
    if host in {"localhost", "0.0.0.0"}:
        return True
    try:
        ipaddress.ip_address(host)
    except ValueError:
        # A real DNS name is presumably a gateway that the proxy is meant to serve.
        return False
    return True


@dataclass
class Endpoint:
    """One served-model machine."""

    base_url: str
    model: str = ""
    api_key: str = "EMPTY"
    label: str = ""
    capacity: int = DEFAULT_ROLLOUT_CAPACITY
    enabled: bool = True

    # Probe state, refreshed by the pool rather than configured.
    healthy: bool = False
    last_error: str = ""
    last_probe_at: float = 0.0
    latency_ms: float = 0.0
    served_models: list[str] = field(default_factory=list)

    # Live counters, reset per process.
    in_flight: int = 0
    completed: int = 0
    failed: int = 0

    @property
    def host(self) -> str:
        parsed = urlparse(self.base_url)
        return f"{parsed.hostname}:{parsed.port}" if parsed.hostname else self.base_url

    @property
    def free_slots(self) -> int:
        if not (self.enabled and self.healthy):
            return 0
        return max(0, int(self.capacity) - int(self.in_flight))

    def display_name(self) -> str:
        return self.label or self.host

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["host"] = self.host
        payload["free_slots"] = self.free_slots
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Endpoint":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in (payload or {}).items() if key in fields})


class EndpointPool:
    """Health-checked machines plus per-episode capacity accounting."""

    def __init__(self, state_path: str | Path | None = None) -> None:
        self.state_path = Path(str(state_path)).expanduser() if state_path else None
        self._endpoints: dict[str, Endpoint] = {}
        self._lock = threading.RLock()
        self._round_robin = 0
        if self.state_path and self.state_path.is_file():
            self.load()

    # -- membership ------------------------------------------------------

    def add(
        self,
        raw: str,
        *,
        model: str = "",
        api_key: str = "EMPTY",
        label: str = "",
        capacity: int = DEFAULT_ROLLOUT_CAPACITY,
        probe: bool = True,
    ) -> Endpoint:
        """Register a machine from ``ip:port`` (or a full URL).

        Re-adding an existing machine updates it in place rather than duplicating,
        which is what makes the UI's add button idempotent.
        """
        base_url = normalize_base_url(raw)
        with self._lock:
            endpoint = self._endpoints.get(base_url)
            if endpoint is None:
                endpoint = Endpoint(base_url=base_url, api_key=api_key, label=label)
                self._endpoints[base_url] = endpoint
            if model:
                endpoint.model = model
            if api_key:
                endpoint.api_key = api_key
            if label:
                endpoint.label = label
            endpoint.capacity = max(1, int(capacity))
            endpoint.enabled = True
        if probe:
            self.probe(base_url)
        self.save()
        return self._endpoints[base_url]

    def remove(self, base_url: str) -> bool:
        with self._lock:
            removed = self._endpoints.pop(normalize_base_url(base_url), None) is not None
        if removed:
            self.save()
        return removed

    def set_enabled(self, base_url: str, enabled: bool) -> bool:
        with self._lock:
            endpoint = self._endpoints.get(normalize_base_url(base_url))
            if endpoint is None:
                return False
            endpoint.enabled = bool(enabled)
        self.save()
        return True

    def set_capacity(self, base_url: str, capacity: int) -> bool:
        with self._lock:
            endpoint = self._endpoints.get(normalize_base_url(base_url))
            if endpoint is None:
                return False
            endpoint.capacity = max(1, int(capacity))
        self.save()
        return True

    def get(self, base_url: str) -> Endpoint | None:
        with self._lock:
            return self._endpoints.get(normalize_base_url(base_url))

    def all(self) -> list[Endpoint]:
        with self._lock:
            return list(self._endpoints.values())

    def healthy(self) -> list[Endpoint]:
        return [ep for ep in self.all() if ep.enabled and ep.healthy]

    @property
    def total_capacity(self) -> int:
        return sum(ep.capacity for ep in self.healthy())

    # -- health ----------------------------------------------------------

    def probe(self, base_url: str, *, timeout: float = DEFAULT_PROBE_TIMEOUT) -> Endpoint | None:
        """Ask one machine for ``/models`` and record what came back."""
        endpoint = self.get(base_url)
        if endpoint is None:
            return None
        started = time.monotonic()
        try:
            with httpx.Client(
                base_url=endpoint.base_url,
                timeout=timeout,
                trust_env=not bypasses_proxy(endpoint.base_url),
                headers=_auth_headers(endpoint.api_key),
            ) as client:
                response = client.get("/models")
                response.raise_for_status()
                payload = response.json()
            models = [
                str(item.get("id"))
                for item in (payload.get("data") or [])
                if isinstance(item, dict) and item.get("id")
            ]
            endpoint.served_models = models
            endpoint.healthy = True
            endpoint.last_error = ""
            # Adopting the served name avoids the "model not found" class of failure
            # when the pool config and the deployment drift apart.
            if not endpoint.model and models:
                endpoint.model = models[0]
        except Exception as exc:  # noqa: BLE001 - a probe must never raise
            endpoint.healthy = False
            endpoint.served_models = []
            endpoint.last_error = _short_error(exc)
        endpoint.latency_ms = round((time.monotonic() - started) * 1000.0, 1)
        endpoint.last_probe_at = time.time()
        return endpoint

    def probe_all(self, *, timeout: float = DEFAULT_PROBE_TIMEOUT) -> list[Endpoint]:
        """Probe every machine concurrently; slow machines must not serialize."""
        endpoints = self.all()
        if not endpoints:
            return []
        threads = [
            threading.Thread(target=self.probe, args=(ep.base_url,), kwargs={"timeout": timeout})
            for ep in endpoints
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=timeout + 5.0)
        self.save()
        return self.all()

    # -- scheduling ------------------------------------------------------

    def acquire(self) -> Endpoint | None:
        """Reserve a rollout slot on the least-loaded healthy machine.

        Returns ``None`` when every machine is full or unhealthy; callers wait and
        retry rather than queueing, so a machine coming back online is picked up
        without restarting the batch.
        """
        with self._lock:
            available = [ep for ep in self._endpoints.values() if ep.free_slots > 0]
            if not available:
                return None
            # Most free slots first; round-robin only breaks ties, so two idle
            # machines alternate but a busy one is skipped.
            self._round_robin += 1
            available.sort(key=lambda ep: (-ep.free_slots, ep.base_url))
            top = [ep for ep in available if ep.free_slots == available[0].free_slots]
            endpoint = top[self._round_robin % len(top)]
            endpoint.in_flight += 1
            return endpoint

    def release(self, base_url: str, *, success: bool = True) -> None:
        with self._lock:
            endpoint = self._endpoints.get(normalize_base_url(base_url))
            if endpoint is None:
                return
            endpoint.in_flight = max(0, endpoint.in_flight - 1)
            if success:
                endpoint.completed += 1
            else:
                endpoint.failed += 1

    def mark_unhealthy(self, base_url: str, reason: str) -> None:
        """Quarantine a machine after a rollout failed against it."""
        with self._lock:
            endpoint = self._endpoints.get(normalize_base_url(base_url))
            if endpoint is None:
                return
            endpoint.healthy = False
            endpoint.last_error = str(reason)[:300]
        logger.warning("endpoint quarantined: %s (%s)", base_url, str(reason)[:160])

    # -- persistence -----------------------------------------------------

    def load(self) -> None:
        if not self.state_path or not self.state_path.is_file():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("endpoint pool unreadable (%s): %s", self.state_path, exc)
            return
        with self._lock:
            self._endpoints.clear()
            for item in payload.get("endpoints", []):
                if not isinstance(item, dict) or not item.get("base_url"):
                    continue
                endpoint = Endpoint.from_dict(item)
                # In-flight counts belong to a process, not to the file.
                endpoint.in_flight = 0
                self._endpoints[endpoint.base_url] = endpoint

    def save(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "endpoints": [ep.to_dict() for ep in self.all()],
        }
        # Atomic replace: the UI may read this file while a batch worker writes it.
        temp_path = self.state_path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temp_path.replace(self.state_path)

    def describe(self) -> dict[str, Any]:
        endpoints = self.all()
        return {
            "total": len(endpoints),
            "healthy": sum(1 for ep in endpoints if ep.enabled and ep.healthy),
            "total_capacity": self.total_capacity,
            "in_flight": sum(ep.in_flight for ep in endpoints),
            "endpoints": [ep.to_dict() for ep in endpoints],
        }


def _auth_headers(api_key: str) -> dict[str, str]:
    key = str(api_key or "").strip()
    return {"Authorization": f"Bearer {key}"} if key and key != "EMPTY" else {}


def _short_error(exc: Exception) -> str:
    """Readable one-liner, including the body that explains a 403 or 404."""
    response = getattr(exc, "response", None)
    detail = ""
    if response is not None:
        try:
            detail = (response.text or "").strip().replace("\n", " ")[:200]
        except Exception:  # noqa: BLE001
            detail = ""
    message = f"{type(exc).__name__}: {exc}"[:200]
    return f"{message} | {detail}" if detail else message
