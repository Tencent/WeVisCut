"""Least-in-flight router across several single-GPU vLLM servers.

One vLLM process per GPU gives N independent engines, but the studio expects a
single OpenAI-compatible endpoint. This is that endpoint: it forwards
``/v1/*`` to whichever backend currently has the fewest requests in flight,
which is the right policy here because AICut requests differ wildly in cost (a
30-token tool call next to a 9k-token script) and round-robin would happily
queue a cheap call behind an expensive one.

Deliberately small: no retries across backends for streaming responses, no
sticky sessions, no health-based ejection beyond a periodic probe. Run it with

    python3 -m studio.gpu_router --backends http://127.0.0.1:8100,... --port 8099
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("gpu_router")

# A backend that fails is parked for this long before it is tried again, so one
# dead GPU does not soak up a share of every request.
COOLDOWN_SECONDS = 20.0
PROBE_INTERVAL_SECONDS = 15.0


@dataclass
class Backend:
    url: str
    inflight: int = 0
    total: int = 0
    failures: int = 0
    healthy: bool = True
    blocked_until: float = 0.0
    last_latency_ms: float = 0.0

    @property
    def available(self) -> bool:
        return self.healthy and time.monotonic() >= self.blocked_until

    def stats(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "inflight": self.inflight,
            "total": self.total,
            "failures": self.failures,
            "healthy": self.healthy,
            "available": self.available,
            "last_latency_ms": round(self.last_latency_ms, 1),
        }


@dataclass
class Router:
    backends: list[Backend]
    timeout: float = 1800.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def pick(self) -> Backend | None:
        async with self._lock:
            usable = [b for b in self.backends if b.available]
            if not usable:
                # Everything is cooling down: fall back to the whole set rather
                # than refusing, since a cooldown is a guess, not a fact.
                usable = list(self.backends)
            if not usable:
                return None
            chosen = min(usable, key=lambda b: (b.inflight, b.total))
            chosen.inflight += 1
            chosen.total += 1
            return chosen

    async def release(self, backend: Backend, *, ok: bool, latency_ms: float) -> None:
        async with self._lock:
            backend.inflight = max(0, backend.inflight - 1)
            backend.last_latency_ms = latency_ms
            if ok:
                backend.failures = 0
                backend.healthy = True
                # Also lift the cooldown: a request that just succeeded is
                # better evidence than the failure that parked it.
                backend.blocked_until = 0.0
            else:
                backend.failures += 1
                backend.blocked_until = time.monotonic() + COOLDOWN_SECONDS

    async def probe(self, client: httpx.AsyncClient) -> None:
        for b in self.backends:
            try:
                resp = await client.get(f"{b.url}/v1/models", timeout=5.0)
                alive = resp.status_code < 500
            except Exception:
                alive = False
            async with self._lock:
                if alive and not b.healthy:
                    logger.info("backend back online: %s", b.url)
                    b.blocked_until = 0.0
                    b.failures = 0
                elif not alive and b.healthy:
                    logger.warning("backend unreachable: %s", b.url)
                b.healthy = alive


def build_app(backend_urls: list[str], timeout: float = 1800.0) -> FastAPI:
    router = Router([Backend(url=u.rstrip("/")) for u in backend_urls], timeout=timeout)
    app = FastAPI(title="WeVisCut GPU router", version="1.0.0")
    state: dict[str, Any] = {}

    def _client() -> httpx.AsyncClient:
        """The shared client, created on first use.

        Lazy rather than startup-only so the router still works when it is
        mounted inside another ASGI app, which does not propagate lifespan
        events to its sub-applications.
        """
        client = state.get("client")
        if client is None:
            # trust_env=False on purpose: the backends are always loopback, and
            # an inherited http_proxy would send every inference request to a
            # corporate proxy that answers 403.
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout), trust_env=False)
            state["client"] = client
        return client

    @app.on_event("startup")
    async def _startup() -> None:
        _client()
        state["probe"] = asyncio.create_task(_probe_loop())
        logger.info("router ready over %d backend(s)", len(router.backends))

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        task = state.get("probe")
        if task is not None:
            task.cancel()
        client = state.get("client")
        if client is not None:
            await client.aclose()

    async def _probe_loop() -> None:
        client = _client()
        while True:
            try:
                await router.probe(client)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.debug("probe failed", exc_info=True)
            await asyncio.sleep(PROBE_INTERVAL_SECONDS)

    @app.get("/router/stats")
    async def stats() -> dict[str, Any]:
        return {
            "backends": [b.stats() for b in router.backends],
            "inflight_total": sum(b.inflight for b in router.backends),
            "available": sum(1 for b in router.backends if b.available),
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        ready = sum(1 for b in router.backends if b.available)
        return {"ok": ready > 0, "available": ready, "total": len(router.backends)}

    @app.api_route("/{full_path:path}",
                   methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy(full_path: str, request: Request) -> Any:
        backend = await router.pick()
        if backend is None:
            return JSONResponse({"error": {"message": "no backend configured"}},
                                status_code=503)

        client = _client()
        body = await request.body()
        # Hop-by-hop and length headers belong to this connection, not the next.
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "connection",
                                 "keep-alive", "transfer-encoding")
        }
        url = f"{backend.url}/{full_path}"
        started = time.monotonic()
        # Forwarded chunk by chunk rather than buffered: an SSE response
        # (``stream: true``) is one long-lived body, and reading it to
        # completion first delayed every token until the answer was already
        # finished — which is what stopped the live thinking panel from
        # updating while the model was still writing.
        ctx = client.stream(
            request.method, url,
            content=body or None,
            headers=headers,
            params=dict(request.query_params),
        )
        try:
            resp = await ctx.__aenter__()
        except Exception as exc:
            await router.release(backend, ok=False,
                                 latency_ms=(time.monotonic() - started) * 1000)
            logger.warning("backend %s failed: %s", backend.url, exc)
            return JSONResponse(
                {"error": {"message": f"backend {backend.url} failed: {exc}"}},
                status_code=502)

        released = False

        async def body() -> Any:
            nonlocal released
            ok = resp.status_code < 500
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            except Exception:
                ok = False
                raise
            finally:
                if not released:
                    released = True
                    await router.release(
                        backend, ok=ok,
                        latency_ms=(time.monotonic() - started) * 1000)
                await ctx.__aexit__(None, None, None)

        passthrough = {
            k: v for k, v in resp.headers.items()
            if k.lower() not in ("content-length", "transfer-encoding", "connection")
        }
        passthrough["x-wed-backend"] = backend.url
        return StreamingResponse(
            body(),
            status_code=resp.status_code,
            headers=passthrough,
            media_type=resp.headers.get("content-type"),
        )

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="WeVisCut GPU router")
    ap.add_argument("--backends", required=True,
                    help="逗号分隔的后端地址，例如 http://127.0.0.1:8100,http://127.0.0.1:8101")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [gpu_router] %(levelname)s: %(message)s")
    urls = [u.strip() for u in args.backends.split(",") if u.strip()]
    if not urls:
        raise SystemExit("--backends 不能为空")

    import uvicorn

    uvicorn.run(build_app(urls, timeout=args.timeout),
                host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
