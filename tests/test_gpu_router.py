"""Offline tests for the multi-GPU router (no GPU, no vLLM).

The router is what turns N single-GPU vLLM instances into the one
OpenAI-compatible endpoint the studio expects, so its two jobs are worth
pinning down: send each request to the least busy backend, and stop sending
work to a backend that is failing.

Fake backends are real FastAPI apps mounted behind the router's HTTP client, so
the proxying path (headers, body, status codes) is exercised for real.

Run:  python3 tests/test_gpu_router.py
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Module scope on purpose: `from __future__ import annotations` makes the route
# annotations strings, and FastAPI resolves them against module globals.
from fastapi import Request  # noqa: E402

from studio.gpu_router import Backend, Router, build_app  # noqa: E402


class PickTest(unittest.TestCase):
    """Least-in-flight, because AICut requests differ wildly in cost."""

    def setUp(self):
        self.router = Router([Backend(url=f"http://b{i}") for i in range(3)])

    def test_spreads_across_idle_backends(self):
        picked = [asyncio.run(self.router.pick()).url for _ in range(3)]
        self.assertEqual(sorted(picked), ["http://b0", "http://b1", "http://b2"])

    def test_prefers_the_least_busy_not_the_next_in_line(self):
        self.router.backends[0].inflight = 5
        self.router.backends[1].inflight = 0
        self.router.backends[2].inflight = 3
        self.assertEqual(asyncio.run(self.router.pick()).url, "http://b1")

    def test_inflight_is_released(self):
        b = asyncio.run(self.router.pick())
        self.assertEqual(b.inflight, 1)
        asyncio.run(self.router.release(b, ok=True, latency_ms=12.0))
        self.assertEqual(b.inflight, 0)
        self.assertEqual(b.last_latency_ms, 12.0)

    def test_a_failure_parks_the_backend(self):
        b = asyncio.run(self.router.pick())
        asyncio.run(self.router.release(b, ok=False, latency_ms=1.0))
        self.assertFalse(b.available)
        self.assertEqual(b.failures, 1)

        # The next 4 picks must all avoid the parked one.
        for _ in range(4):
            self.assertNotEqual(asyncio.run(self.router.pick()).url, b.url)

    def test_success_clears_a_previous_failure(self):
        b = self.router.backends[0]
        asyncio.run(self.router.release(b, ok=False, latency_ms=1.0))
        asyncio.run(self.router.release(b, ok=True, latency_ms=1.0))
        self.assertEqual(b.failures, 0)
        self.assertTrue(b.available)

    def test_all_parked_still_returns_a_backend(self):
        # A cooldown is a guess. Refusing every request would be worse than
        # retrying a backend that may well have recovered.
        for b in self.router.backends:
            asyncio.run(self.router.release(b, ok=False, latency_ms=1.0))
        self.assertIsNotNone(asyncio.run(self.router.pick()))

    def test_no_backends_returns_none(self):
        self.assertIsNone(asyncio.run(Router([]).pick()))


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _FakeBackend:
    """A real uvicorn server on loopback, so the proxy hop is genuine."""

    def __init__(self, name: str, hits: dict) -> None:
        import threading

        import uvicorn
        from fastapi import FastAPI

        self.name = name
        self.port = _free_port()
        app = FastAPI()

        @app.get("/v1/models")
        async def models():
            return {"data": [{"id": "weviscut-9b"}]}

        @app.post("/v1/chat/completions")
        async def chat(payload: dict):
            hits[name] = hits.get(name, 0) + 1
            return {"backend": name, "echo": payload.get("messages")}

        # vLLM rejects unauthenticated calls when VLLM_API_KEY is set, so the
        # router has to hand the header on or the whole cluster 401s.
        @app.post("/v1/echo-auth")
        async def echo_auth(request: Request):
            return {"auth": request.headers.get("authorization", "")}

        self._server = uvicorn.Server(uvicorn.Config(
            app, host="127.0.0.1", port=self.port, log_level="error"))
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        import time as _t

        self._thread.start()
        for _ in range(100):
            if getattr(self._server, "started", False):
                return
            _t.sleep(0.05)
        raise RuntimeError(f"fake backend {self.name} did not start")

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)


class ProxyTest(unittest.TestCase):
    """End-to-end over real HTTP, including the proxy hop."""

    @classmethod
    def setUpClass(cls):
        cls.hits: dict = {}
        cls.backends = [_FakeBackend("a", cls.hits), _FakeBackend("b", cls.hits)]
        for b in cls.backends:
            b.start()

    @classmethod
    def tearDownClass(cls):
        for b in cls.backends:
            b.stop()

    def setUp(self):
        from fastapi.testclient import TestClient

        self.hits.clear()
        self.client = TestClient(build_app([b.url for b in self.backends]))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_body_and_response_pass_through(self):
        res = self.client.post("/v1/chat/completions",
                               json={"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn(body["backend"], ("a", "b"))
        self.assertEqual(body["echo"], [{"role": "user", "content": "hi"}])
        self.assertIn("x-wed-backend", res.headers)

    def test_load_is_shared_between_backends(self):
        for _ in range(8):
            self.client.post("/v1/chat/completions", json={"messages": []})
        self.assertEqual(sum(self.hits.values()), 8)
        # Sequential requests each release before the next picks, so the tie is
        # broken on total count and the split must be even.
        self.assertEqual(self.hits.get("a"), self.hits.get("b"))

    def test_stats_and_health(self):
        self.client.post("/v1/chat/completions", json={"messages": []})
        stats = self.client.get("/router/stats").json()
        self.assertEqual(len(stats["backends"]), 2)
        self.assertEqual(stats["inflight_total"], 0)
        self.assertEqual(sum(b["total"] for b in stats["backends"]), 1)

        health = self.client.get("/health").json()
        self.assertTrue(health["ok"])
        self.assertEqual(health["total"], 2)

    def test_authorization_reaches_the_backend(self):
        res = self.client.post("/v1/echo-auth", json={},
                               headers={"Authorization": "Bearer sk-secret"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["auth"], "Bearer sk-secret")

    def test_unreachable_backend_returns_502_not_a_hang(self):
        app = build_app(["http://127.0.0.1:9"])     # discard port
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            res = client.post("/v1/chat/completions", json={"messages": []})
            self.assertEqual(res.status_code, 502)
            self.assertIn("failed", res.json()["error"]["message"])


class StudioClusterEndpointTest(unittest.TestCase):
    """`/api/cluster` is the only way to see per-GPU load from outside.

    The router binds loopback, and the box exposes one port, so if this
    endpoint breaks the cluster becomes unobservable off-box.
    """

    @classmethod
    def setUpClass(cls):
        import os

        os.environ.setdefault("AICUT_STUDIO_SKIP_MODEL", "1")

        from fastapi.testclient import TestClient

        from studio import app as app_mod

        cls.app_mod = app_mod
        cls.client = TestClient(app_mod.app)
        cls.client.__enter__()

        # Cluster stats sit behind auth, so the test needs a session. Minting
        # one beats shipping a fixture password.
        if app_mod.studio.config.auth_enabled:
            from studio import auth as auth_mod

            app_mod.studio.users.users.setdefault(
                "cluster_test", auth_mod.User(username="cluster_test", role="admin"))
            # In-memory only: pin the store so a users.yaml reload cannot drop
            # the fixture account mid-run and turn every call into 401.
            app_mod.studio.users.maybe_reload = lambda: None
            token = app_mod.studio.sessions.issue("cluster_test")
            cls.client.cookies.set(app_mod.SESSION_COOKIE, token)

        cls.model = app_mod.studio.config.infer_config.setdefault("model", {})
        cls.saved = dict(cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def setUp(self):
        self.model.clear()
        self.model.update(self.saved)
        self.app_mod._CLUSTER_CACHE.update(at=0.0, value=None)

    def _point_at(self, base: str) -> None:
        self.model["infer_backend"] = "openai"
        self.model["openai_base_url"] = base
        self.model["openai_model"] = "weviscut-9b"

    def test_single_gpu_reports_disabled(self):
        self.model["infer_backend"] = "vllm"
        self.model["openai_base_url"] = ""
        body = self.client.get("/api/cluster").json()
        self.assertFalse(body["enabled"])
        self.assertEqual(body["backends"], [])

    def test_reports_per_backend_load(self):
        backends = [_FakeBackend("x", {}), _FakeBackend("y", {})]
        for b in backends:
            b.start()
        router = _FakeRouterServer([b.url for b in backends])
        router.start()
        try:
            self._point_at(f"{router.url}/v1")
            body = self.client.get("/api/cluster").json()
            self.assertTrue(body["enabled"])
            self.assertTrue(body["reachable"], body.get("error"))
            self.assertEqual(body["total"], 2)
            self.assertEqual(body["healthy"], 2)
            self.assertEqual(body["inflight"], 0)
            self.assertEqual(body["model"], "weviscut-9b")
            self.assertEqual(len(body["backends"]), 2)
        finally:
            router.stop()
            for b in backends:
                b.stop()

    def test_dead_router_is_a_status_not_a_500(self):
        self._point_at("http://127.0.0.1:9/v1")     # discard port
        res = self.client.get("/api/cluster")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertTrue(body["enabled"])
        self.assertFalse(body["reachable"])
        self.assertTrue(body["error"])

    def test_result_is_cached_between_polls(self):
        # Every open tab polls on a 4s timer; the router must not see them all.
        backends = [_FakeBackend("z", {})]
        backends[0].start()
        router = _FakeRouterServer([backends[0].url])
        router.start()
        try:
            self._point_at(f"{router.url}/v1")
            self.client.get("/api/cluster")
            before = router.stats_hits
            self.client.get("/api/cluster")
            self.assertEqual(router.stats_hits, before)
        finally:
            router.stop()
            backends[0].stop()


class _FakeRouterServer:
    """The real router app on loopback, with a counter on /router/stats."""

    def __init__(self, backend_urls: list[str]) -> None:
        import threading

        import uvicorn

        self.stats_hits = 0
        self.port = _free_port()
        app = build_app(backend_urls)
        outer = self

        @app.middleware("http")
        async def count(request, call_next):
            if request.url.path == "/router/stats":
                outer.stats_hits += 1
            return await call_next(request)

        self._server = uvicorn.Server(uvicorn.Config(
            app, host="127.0.0.1", port=self.port, log_level="error"))
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        import time as _t

        self._thread.start()
        for _ in range(100):
            if getattr(self._server, "started", False):
                return
            _t.sleep(0.05)
        raise RuntimeError("fake router did not start")

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
