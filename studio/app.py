"""AICut Studio HTTP/WebSocket server.

Route groups:

* ``/api/status``, ``/api/config`` — what the frontend needs to render itself
* ``/api/runs`` + ``/ws/{run_id}`` — start a generation and watch it live
* ``/api/projects/...`` — load a finished cut, edit it, regenerate, re-stitch
* ``/api/projects/{id}/chat`` — conversational editing
* ``/v1/chat/completions`` — OpenAI-compatible endpoint the *pipeline tools*
  call, so the same 9B weights serve the agent and the pipeline's internal LLM
  calls without a second deployment

Model loading is a background task rather than a blocking startup step: weights
take 30-60s, and the UI should be reachable (showing "loading") the whole time.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import mimetypes
import os
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from contextvars import ContextVar

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from studio import auth as auth_mod
from studio import config as studio_config
from studio import chat_agent, editor, project as project_mod, script_writer, tracks
from studio.model_client import ModelNotReady, ModelService, build_chat_backend
from studio.runs import RunStore
from studio.newvis0907 import operations as newvis0907_ops
from studio.newvis0907 import router as newvis0907_router
from studio.newvis0907 import schemas as newvis0907_schemas
from studio.newvis0907.chat_planner import build_model_plan_prompt as _build_model_plan_prompt_0907
from studio.newvis0907.transactions import TransactionError as Newvis0907TxError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("aicut_studio")

STUDIO_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = STUDIO_DIR / "static"


# ---------------------------------------------------------------------------
# application state
# ---------------------------------------------------------------------------
class Studio:
    """Process-wide state, assembled at startup."""

    def __init__(self) -> None:
        self.cfg: studio_config.StudioConfig | None = None
        self.model: ModelService | None = None
        self.runs: RunStore | None = None
        # project_dir -> loaded project (kept so edits accumulate across requests)
        self.projects: dict[str, project_mod.Project] = {}
        # project_dir -> chat agent (one conversation per project)
        self.chats: dict[str, chat_agent.ChatEditAgent] = {}
        self.load_task: asyncio.Task | None = None
        self.regen_locks: dict[str, asyncio.Lock] = {}
        self.users: auth_mod.UserStore | None = None
        self.sessions = auth_mod.SessionStore()
        self.owners: auth_mod.OwnerIndex | None = None

    # -- config-derived helpers --
    @property
    def config(self) -> studio_config.StudioConfig:
        if self.cfg is None:
            raise RuntimeError("studio not initialised")
        return self.cfg

    @property
    def store(self) -> RunStore:
        if self.runs is None:
            raise RuntimeError("studio not initialised")
        return self.runs

    def editor_flags(self) -> dict[str, Any]:
        manual = self.config.editor.get("manual_ops")
        return manual if isinstance(manual, dict) else {}

    def track_order(self) -> list[str]:
        requested = self.config.progress.get("tracks") or list(tracks.ALL_TRACKS)
        return [str(t) for t in requested if str(t) in tracks.ALL_TRACKS]

    def regen_lock(self, key: str) -> asyncio.Lock:
        return self.regen_locks.setdefault(key, asyncio.Lock())

    @property
    def owner_index(self) -> auth_mod.OwnerIndex:
        if self.owners is None:
            raise RuntimeError("studio not initialised")
        return self.owners


studio = Studio()
app = FastAPI(title="WevisionCut Studio", version="1.1.0")

# The signed-in user for the request being handled. A context var rather than a
# FastAPI dependency so that ~30 existing handler signatures stay untouched.
CURRENT_USER: ContextVar[auth_mod.User | None] = ContextVar("current_user", default=None)

SESSION_COOKIE = "wevisioncut_session"
# Paths reachable without a session: the login screen itself, its API, and the
# static assets that screen needs to render.
PUBLIC_PREFIXES = ("/login", "/api/auth/", "/static/")
# The pipeline's own tools call this over loopback with no cookie to offer.
LOOPBACK_ONLY_PATHS = ("/v1/chat/completions", "/chat/completions")
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def current_user() -> auth_mod.User | None:
    return CURRENT_USER.get()


def require_user() -> auth_mod.User:
    """The signed-in user, or a synthetic admin when auth is switched off."""
    user = CURRENT_USER.get()
    if user is not None:
        return user
    if not studio.config.auth_enabled:
        return auth_mod.User(username="local", role=auth_mod.ADMIN_ROLE,
                             display_name="本地用户")
    raise HTTPException(401, "请先登录")


def _resolve_user(token: str) -> auth_mod.User | None:
    username = studio.sessions.resolve(token)
    if not username or studio.users is None:
        return None
    return studio.users.get(username)


@app.middleware("http")
async def auth_middleware(request: Request, call_next: Any) -> Any:
    """Gate every route behind a session when ``auth.enabled``."""
    if not studio.cfg:
        CURRENT_USER.set(None)
        return await call_next(request)

    path = request.url.path
    if path in LOOPBACK_ONLY_PATHS:
        host = (request.client.host if request.client else "") or ""
        if host not in LOOPBACK_HOSTS:
            return JSONResponse({"detail": "仅允许本机调用"}, status_code=403)
        CURRENT_USER.set(None)
        return await call_next(request)

    if not studio.config.auth_enabled:
        CURRENT_USER.set(None)
        return await call_next(request)

    if path.startswith(PUBLIC_PREFIXES):
        CURRENT_USER.set(None)
        return await call_next(request)

    user = _resolve_user(request.cookies.get(SESSION_COOKIE, ""))
    if user is None:
        # HTML navigations get a redirect they can follow; XHR gets a 401 the
        # frontend turns into a redirect itself.
        if request.method == "GET" and "text/html" in request.headers.get("accept", ""):
            return RedirectResponse("/login", status_code=302)
        return JSONResponse({"detail": "请先登录"}, status_code=401)

    token = CURRENT_USER.set(user)
    try:
        return await call_next(request)
    finally:
        CURRENT_USER.reset(token)


# ---------------------------------------------------------------------------
# startup / shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup() -> None:
    cfg = studio_config.load(os.environ.get("AICUT_STUDIO_CONFIG"))
    studio.cfg = cfg

    for issue in cfg.issues:
        (logger.error if issue.level == "error" else logger.warning)("config %s", issue)
    if cfg.errors():
        raise RuntimeError(
            "studio_config.yaml 有致命错误: "
            + "; ".join(f"{i.where}: {i.message}" for i in cfg.errors())
        )

    progress_cfg = cfg.progress
    studio.runs = RunStore(
        max_events=int(progress_cfg.get("max_events_per_run", 5000) or 5000),
        throttle_ms=int(progress_cfg.get("push_throttle_ms", 120) or 120),
        raw_buffer=int(progress_cfg.get("raw_output_buffer", 400) or 400),
        journal_root=cfg.workspace_root / "runs",
    )

    cfg.workspace_root.mkdir(parents=True, exist_ok=True)
    # Run history used to live only in this process, so a restart wiped every
    # conversation and trajectory. Reload what previous processes journalled.
    studio.store.rehydrate()

    studio.owners = auth_mod.OwnerIndex(cfg.workspace_root / "owners.json")
    if cfg.auth_enabled:
        studio.users = auth_mod.UserStore(
            cfg.users_file, passwordless=cfg.auth_passwordless)
        studio.sessions = auth_mod.SessionStore(
            ttl_seconds=float(cfg.auth.get("session_hours", 72) or 72) * 3600)
        if studio.users.load_error:
            logger.error(
                "鉴权已开启但账号不可用: %s。请运行 "
                "python3 scripts/manage_users.py init", studio.users.load_error)
        else:
            admins = [u for u in studio.users.users.values() if u.is_admin]
            logger.info("鉴权已开启：%d 个账号，%d 个超级管理员",
                        len(studio.users.users), len(admins))
    else:
        logger.warning("鉴权未开启 (auth.enabled=false)：所有人都能看到所有作品")
    _wire_pipeline_runtime(cfg)

    studio.model = ModelService(cfg.infer_config, cfg.system_prompt)
    if os.environ.get("AICUT_STUDIO_SKIP_MODEL") == "1":
        studio.model.skipped = True
        logger.warning("AICUT_STUDIO_SKIP_MODEL=1 — 跳过模型加载（仅供接口自检）")
    else:
        studio.load_task = asyncio.create_task(_load_model_background())

    logger.info("AICut Studio ready on port %s", _effective_port(cfg))


async def _load_model_background() -> None:
    assert studio.model is not None
    try:
        await studio.model.load_async()
    except Exception:
        logger.exception("模型加载失败；界面可用但无法启动推理")


def _effective_port(cfg: studio_config.StudioConfig) -> int:
    """The port actually bound, which the launcher may override via --port."""
    return int(os.environ.get("AICUT_STUDIO_PORT") or cfg.server.get("port", 8095))


def _wire_pipeline_runtime(cfg: studio_config.StudioConfig) -> None:
    """Point the pipeline's internal LLM calls at this process, and install
    the transparent-overlay render accelerator if enabled."""
    local_llm = (os.environ.get("AICUT_LOCAL_LLM_BASE_URL")
                 or f"http://127.0.0.1:{_effective_port(cfg)}/v1")

    llm_block = {
        "base_url": local_llm,
        "endpoint": "/chat/completions",
        "api_key": "local",
        "model": "local-aicut-9b",
        "temperature": 0.2,
        "max_tokens": 2048,
        "timeout": 600,
    }
    cfg.pipeline_config["llm"] = llm_block
    # Pipeline tools resolve their LLM through services.qwen; keep it in sync so
    # no tool silently reaches out to a remote 397B endpoint.
    services = cfg.pipeline_config.setdefault("services", {})
    services["qwen"] = {**llm_block, "offline": False, "max_prompt_chars": 50000}

    render_split = cfg.pipeline_config.get("render_split") or {}
    if render_split.get("enabled"):
        try:
            studio_config.install_import_paths()
            from server.render_split import install as install_render_split

            if install_render_split(cfg.pipeline_config):
                logger.info("render-split 透明层加速已启用")
        except Exception as exc:
            logger.warning("render-split 未启用: %s", exc)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    if studio.load_task and not studio.load_task.done():
        studio.load_task.cancel()
    for run in list(studio.store.runs.values()) if studio.runs else []:
        if run.task and not run.task.done():
            run.task.cancel()
    if studio.model is not None:
        studio.model.unload()


# ---------------------------------------------------------------------------
# static
# ---------------------------------------------------------------------------
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/0901_newvis")
async def index_newvis() -> FileResponse:
    """Conversation-first shell. Served alongside — not instead of — ``/``."""
    return FileResponse(str(STATIC_DIR / "newvis" / "index.html"))


@app.get("/0907_newvis")
async def index_newvis0907() -> FileResponse:
    """Dedicated 0907 shell, isolated from the 0901 surface."""
    return FileResponse(str(STATIC_DIR / "newvis0907" / "index.html"))


# ---------------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------------
@app.get("/login")
async def login_page() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "newvis" / "login.html"))


class LoginRequest(BaseModel):
    username: str = ""
    password: str = ""


@app.post("/api/auth/login")
async def auth_login(req: LoginRequest) -> JSONResponse:
    if not studio.config.auth_enabled:
        return JSONResponse({"detail": "服务未开启鉴权"}, status_code=400)
    if studio.users is None or studio.users.load_error:
        detail = studio.users.load_error if studio.users else "账号系统未初始化"
        return JSONResponse({"detail": detail}, status_code=503)

    if studio.config.auth_passwordless:
        try:
            name = auth_mod.normalize_username(req.username)
            role = (auth_mod.ADMIN_ROLE if name in studio.config.auth_admins
                    else "user")
            user = studio.users.ensure(name, role=role)
        except auth_mod.InvalidUsername as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
    else:
        user = studio.users.authenticate(req.username, req.password)
        if user is None:
            # One message for both cases so it cannot be used to enumerate
            # accounts.
            return JSONResponse({"detail": "账号或密码不对"}, status_code=401)

    token = studio.sessions.issue(user.username)
    studio.sessions.purge()
    resp = JSONResponse({"user": user.public()})
    resp.set_cookie(
        SESSION_COOKIE, token,
        max_age=int(studio.sessions.ttl_seconds),
        httponly=True, samesite="lax",
        # No `secure`: this is served over plain HTTP on an internal address,
        # and a secure cookie would simply never be sent back.
    )
    logger.info("login: %s (%s)", user.username, user.role)
    return resp


@app.post("/api/auth/logout")
async def auth_logout(request: Request) -> JSONResponse:
    studio.sessions.revoke(request.cookies.get(SESSION_COOKIE, ""))
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE)
    return resp


@app.get("/api/auth/me")
async def auth_me(request: Request) -> dict[str, Any]:
    if not studio.config.auth_enabled:
        return {"auth_enabled": False, "passwordless": False,
                "user": require_user().public()}
    user = _resolve_user(request.cookies.get(SESSION_COOKIE, ""))
    return {
        "auth_enabled": True,
        # The login page needs this before anyone is logged in, to know whether
        # to ask for a password at all.
        "passwordless": studio.config.auth_passwordless,
        "user": user.public() if user else None,
    }


# ---------------------------------------------------------------------------
# status / config
# ---------------------------------------------------------------------------
@app.get("/api/status")
async def get_status() -> dict[str, Any]:
    model = studio.model.describe() if studio.model else {"loaded": False}
    return {
        "model": model,
        "active_runs": studio.store.active_count(),
        "agent_mode": studio.config.infer_config.get("agent_mode", ""),
        "autonomy": studio.config.autonomy,
    }


_CLUSTER_CACHE: dict[str, Any] = {"at": 0.0, "value": None}
_CLUSTER_TTL = 2.0
_cluster_client: Any = None


def _router_root() -> str:
    """Base URL of the GPU router, or '' on a single-GPU deployment."""
    model = studio.config.infer_config.get("model") or {}
    base = str(model.get("openai_base_url") or "")
    if str(model.get("infer_backend") or "") != "openai" or not base:
        return ""
    return base.rsplit("/v1", 1)[0].rstrip("/")


@app.get("/api/cluster")
async def get_cluster() -> dict[str, Any]:
    """Per-GPU load, proxied from the router.

    The router and the vLLM instances bind loopback only, so 8080 is the single
    way in from outside. Without this the cluster is unobservable off-box.
    """
    global _cluster_client

    root = _router_root()
    if not root:
        return {"enabled": False, "backends": []}

    # Every open tab polls this on the status timer; one fetch serves them all.
    now = time.time()
    if _CLUSTER_CACHE["value"] is not None and now - _CLUSTER_CACHE["at"] < _CLUSTER_TTL:
        return _CLUSTER_CACHE["value"]

    try:
        import httpx

        if _cluster_client is None:
            # trust_env would route a loopback call through a corporate proxy.
            _cluster_client = httpx.AsyncClient(trust_env=False, timeout=3.0)
        resp = await _cluster_client.get(f"{root}/router/stats")
        resp.raise_for_status()
        stats = resp.json()
    except Exception as exc:  # router down is a status, not a server error
        out = {"enabled": True, "reachable": False, "error": str(exc), "backends": []}
    else:
        backends = stats.get("backends") or []
        model = studio.config.infer_config.get("model") or {}
        out = {
            "enabled": True,
            "reachable": True,
            "model": str(model.get("openai_model") or ""),
            "healthy": sum(1 for b in backends if b.get("healthy")),
            "total": len(backends),
            "inflight": sum(int(b.get("inflight") or 0) for b in backends),
            "backends": backends,
        }

    _CLUSTER_CACHE.update(at=now, value=out)
    return out


@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    """Only what the UI needs to render — never secrets or full paths."""
    cfg = studio.config
    services = cfg.raw.get("services", {})
    services = services if isinstance(services, dict) else {}

    def enabled(name: str) -> bool:
        block = services.get(name)
        return bool(block.get("enabled", True)) if isinstance(block, dict) else False

    order = studio.track_order()
    return {
        "name": cfg.studio.get("name", "AICut Studio"),
        "locale": cfg.studio.get("locale", "zh"),
        "tracks": [
            {"key": t, "label": tracks.TRACK_LABELS.get(t, t)} for t in order
        ],
        "user": require_user().public(),
        "features": {
            "editor": cfg.editor_enabled,
            "chat_edit": cfg.chat_enabled,
            "script_draft": True,
            "auth": cfg.auth_enabled,
            "manual_ops": studio.editor_flags(),
            "services": {
                "tts": enabled("tts"),
                "visual_retrieval": enabled("visual_retrieval"),
                "bgm": enabled("bgm"),
                "hyperframes": enabled("hyperframes"),
            },
        },
        "editor": {
            "ops": editor.ALL_OPS,
            "op_labels": {k: v["label"] for k, v in editor.OP_SPEC.items()},
            "op_risk": {k: v["risk"] for k, v in editor.OP_SPEC.items()},
            "require_confirm_for": (cfg.chat_edit.get("safety") or {}).get(
                "require_confirm_for", []
            ),
            "auto_restitch": bool((cfg.editor.get("rerender") or {}).get("auto_restitch", True)),
        },
        "agent": {
            "max_steps": studio.config.infer_config.get("agent", {}).get("max_steps", 0),
            "fully_autonomous": bool(cfg.autonomy.get("fully_autonomous", True)),
        },
        "warnings": [f"{i.where}: {i.message}" for i in cfg.warnings()],
    }


# ---------------------------------------------------------------------------
# script drafting (brief -> 口播稿)
# ---------------------------------------------------------------------------
class ScriptDraftRequest(BaseModel):
    brief: str = ""
    temperature: float = 0.7


@app.post("/api/script/draft")
async def draft_script(req: ScriptDraftRequest) -> dict[str, Any]:
    """Turn a creative brief into a narration script the pipeline can run.

    Deliberately separate from ``/api/runs``: the draft is shown to the user
    (and kept in the run's input) before the expensive pipeline starts, so a
    wrong-length or off-topic script is visible immediately rather than after
    a full render.
    """
    brief = (req.brief or "").strip()
    if not brief:
        raise HTTPException(400, "创作需求不能为空")
    if studio.model is None or not studio.model.is_loaded:
        raise HTTPException(
            503,
            studio.model.load_error if studio.model and studio.model.load_error
            else "模型尚未加载完成，请稍候",
        )

    backend = build_chat_backend(studio.config.chat_edit, studio.model)
    try:
        return await script_writer.draft(backend, brief, temperature=req.temperature)
    except ModelNotReady as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("script draft failed")
        raise HTTPException(500, f"生成口播稿失败: {type(exc).__name__}: {exc}") from exc


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------
class StartRunRequest(BaseModel):
    input_text: str = ""


@app.post("/api/runs")
async def start_run(req: StartRunRequest) -> dict[str, Any]:
    text = (req.input_text or "").strip()
    if not text:
        raise HTTPException(400, "input_text 不能为空")
    if studio.model is None or not studio.model.is_loaded:
        raise HTTPException(
            503,
            studio.model.load_error if studio.model and studio.model.load_error
            else "模型尚未加载完成，请稍候",
        )

    run = studio.store.create(text)
    run.owner = require_user().username
    studio.store.save_meta(run)
    run.task = asyncio.create_task(_execute_run(run.id, text))
    return {"run_id": run.id, "status": run.status}


@app.get("/api/runs")
async def list_runs() -> dict[str, Any]:
    return {"runs": _visible_runs(require_user())}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    return _require_run_access(run_id).detail()


@app.get("/api/runs/{run_id}/events")
async def get_run_events(run_id: str) -> dict[str, Any]:
    """The recorded trajectory, from memory or from the on-disk journal.

    The WebSocket replays this too, but an HTTP read lets the frontend rebuild
    a finished run's narrative without opening a socket to something that will
    never emit again.
    """
    run = _require_run_access(run_id)
    events = list(run.events) if run.events else studio.store.read_journal(run_id)
    return {
        "run_id": run_id,
        "events": events,
        "event_count": len(events),
        # The input chapter is drawn by the frontend, not emitted by the run, so
        # a rebuild needs the original text back.
        "input_text": run.input_text if run is not None else "",
        "status": run.status if run is not None else "",
    }


@app.delete("/api/runs/{run_id}")
@app.post("/api/runs/{run_id}/delete")
async def delete_run(run_id: str) -> dict[str, Any]:
    _require_run_access(run_id)
    if not studio.store.delete(run_id):
        raise HTTPException(409, "运行中或不存在，无法删除")
    return {"deleted": run_id}


def _require_orchestrator(run_id: str) -> Any:
    run = _require_run_access(run_id)
    if run.orchestrator is None:
        raise HTTPException(404, "run not found or already finished")
    return run


@app.post("/api/runs/{run_id}/pause")
async def pause_run(run_id: str) -> dict[str, Any]:
    run = _require_orchestrator(run_id)
    run.orchestrator.pause()
    run.status = "paused"
    await studio.store.emit(run_id, {"type": "run_paused"})
    return {"status": "paused"}


@app.post("/api/runs/{run_id}/resume")
async def resume_run(run_id: str) -> dict[str, Any]:
    run = _require_orchestrator(run_id)
    run.orchestrator.resume()
    run.status = "running"
    await studio.store.emit(run_id, {"type": "run_resumed"})
    return {"status": "running"}


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str, force: bool = False) -> dict[str, Any]:
    """Stop a run.

    The cooperative path only takes effect between steps, so a click during a
    long render would appear to do nothing for minutes. ``force`` additionally
    cancels the driving task, which stops the loop at its next await instead of
    at its next step boundary. Already-spawned ffmpeg/Chrome children are left
    to exit on their own — they write into the project dir and killing them
    mid-write would corrupt a shot that is otherwise reusable.
    """
    run = _require_orchestrator(run_id)
    run.orchestrator.cancel()
    run.status = "cancelling"
    await studio.store.emit(run_id, {
        "type": "console_log", "level": "warn",
        "text": "已强制中断，正在收尾" if force else "已发送取消请求，当前步骤结束后停止",
    })
    if force and run.task is not None and not run.task.done():
        run.task.cancel()
    return {"status": "cancelling", "forced": bool(force)}


# ---------------------------------------------------------------------------
# run execution
# ---------------------------------------------------------------------------
def _pipeline_config_for_policy(base_config: dict[str, Any] | None, policy_version: str | None) -> dict[str, Any]:
    """Return a per-run config copy, applying 0907-only runtime policy.

    ``render.defer_stitch`` is deliberately left at its configured value. Forcing
    a real concat during generation makes ``Tool_StitchVideo`` fail validation
    whenever any shot has no rendered file yet, and the model then retries the
    identical call until the loop guard aborts the run. Preview audio is handled
    by the player instead, which cannot break generation.
    """
    if policy_version is None:
        return base_config or {}
    pipeline_config = copy.deepcopy(base_config or {})
    pipeline_config["policy_version"] = str(policy_version)
    return pipeline_config


async def _execute_run(run_id: str, input_text: str, policy_version: str | None = None) -> None:
    """Drive a generation run on the deepagents harness.

    The control loop is a ``create_deep_agent`` ReAct graph; the AICut engine
    (state, dispatch, observation, preview) is reused via ``AICutRuntime``.

    ``policy_version`` is opt-in (the 0907 surface passes ``"0907"``). When set,
    the runtime is handed a *deep copy* of the pipeline config carrying the
    marker so ``plan_shots`` / ``generate_audio`` / ``decorate_shot`` /
    ``timeline_compose`` / ``stitch_video`` take their trustworthy-cut branches —
    without ever mutating the shared global ``studio.config.pipeline_config``.
    """
    from aicut_agent.runner import create_runtime, run_aicut_generation

    run = studio.store.get(run_id)
    if run is None:
        return
    cfg = studio.config
    store = studio.store

    pipeline_config = _pipeline_config_for_policy(cfg.pipeline_config, policy_version)

    async def broadcast(rid: str, event: dict[str, Any]) -> None:
        await store.emit(rid, event)

    async def progress(rid: str, snapshot: dict[str, Any]) -> None:
        await store.emit(rid, {"type": "progress", "progress": snapshot})

    output_dir = str(cfg.workspace_root / "projects")

    await broadcast(run_id, {"type": "run_started", "run_id": run_id})
    await broadcast(run_id, {
        "type": "console_log", "level": "system",
        "text": f"模型档位: {cfg.active_profile} · 步数上限 "
                f"{cfg.infer_config.get('agent', {}).get('max_steps')} · deepagents 驱动",
    })

    runtime = create_runtime(
        studio.model.loader,
        infer_config=cfg.infer_config,
        pipeline_config=pipeline_config,
        stage_config=cfg.stage_config,
        system_prompt=cfg.system_prompt,
        input_text=input_text,
        run_id=run_id,
        output_dir=output_dir,
        track_order=studio.track_order(),
        broadcast_fn=broadcast,
        progress_fn=progress,
    )
    orchestrator = runtime.orchestrator
    run.orchestrator = orchestrator  # pause/resume/cancel target

    try:
        result = await run_aicut_generation(runtime, studio.model.loader)
        run.result = result
        run.project_dir = str(result.get("output_dir", "") or "")
        run.status = "completed" if not result.get("errors") else "completed_with_errors"
    except asyncio.CancelledError:
        run.status = "cancelled"
        # A forced stop cancels this very task. Clearing the pending-cancel
        # state lets the terminal notification below still go out, which is
        # what releases the composer in the UI; without it the page would sit
        # on "正在生成中" until the socket dropped.
        task = asyncio.current_task()
        if task is not None and hasattr(task, "uncancel"):
            task.uncancel()
        await broadcast(run_id, {
            "type": "console_log", "level": "warn", "text": "管线已被用户中断",
        })
    except Exception as exc:
        run.status = "error"
        detail = f"{type(exc).__name__}: {exc}"
        run.errors.append(detail)
        logger.exception("run %s failed", run_id)
        await broadcast(run_id, {"type": "error", "stage": "agent_loop", "message": detail})
        await broadcast(run_id, {
            "type": "console_log", "level": "stderr",
            "text": traceback.format_exc()[-2000:],
        })
    finally:
        run.orchestrator = None
        run.finished_at = _now_iso()
        if orchestrator.project_dir and not run.project_dir:
            run.project_dir = str(orchestrator.project_dir)
        # The project directory only exists once the pipeline has created it,
        # so ownership can only be recorded here, not when the run started.
        if run.project_dir and run.owner and studio.owners is not None:
            studio.owner_index.claim(run.project_dir, run.owner)
        if policy_version == "0907" and run.project_dir:
            try:
                _refresh_quality_report_0907(_load_project(run.project_dir))
            except Exception:
                logger.exception("0907 post-generation quality refresh failed")

    await store.flush_progress(run_id)
    await broadcast(run_id, {
        "type": "run_complete",
        "status": run.status,
        "elapsed_seconds": run.elapsed,
        "result": run.result,
        "project_dir": run.project_dir,
    })


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now().isoformat()


# ---------------------------------------------------------------------------
# websocket
# ---------------------------------------------------------------------------
@app.websocket("/ws/{run_id}")
async def run_socket(websocket: WebSocket, run_id: str) -> None:
    # HTTP middleware does not run for WebSockets, so the session cookie has to
    # be checked here or the event stream would be readable by anyone.
    if studio.cfg is not None and studio.config.auth_enabled:
        user = _resolve_user(websocket.cookies.get(SESSION_COOKIE, ""))
        if user is None:
            await websocket.close(code=4401)
            return
        run = studio.store.get(run_id)
        if run is not None and not user.is_admin:
            owner = getattr(run, "owner", "")
            allowed = owner == user.username or (
                run.project_dir
                and studio.owner_index.can_access(run.project_dir, user))
            if not allowed:
                await websocket.close(code=4403)
                return

    await websocket.accept()
    studio.store.subscribe(run_id, websocket)
    try:
        await studio.store.replay(run_id, websocket)
        while True:
            raw = await websocket.receive_text()
            if raw == "ping" or '"ping"' in raw:
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.debug("websocket closed for %s", run_id, exc_info=True)
    finally:
        studio.store.unsubscribe(run_id, websocket)


# ---------------------------------------------------------------------------
# projects
# ---------------------------------------------------------------------------
def _load_project(project_dir: str) -> project_mod.Project:
    """Load (and cache) a project so edits accumulate across requests.

    The ownership check lives here rather than in each handler: every project
    route reaches a project through this function, so a new route cannot
    accidentally ship without isolation.
    """
    key = str(Path(project_dir).resolve())
    _require_access(key)
    cached = studio.projects.get(key)
    if cached is not None:
        return cached
    try:
        proj = project_mod.load(key, studio.config.pipeline_config)
    except project_mod.ProjectError as exc:
        raise HTTPException(404, str(exc)) from exc
    studio.projects[key] = proj
    return proj


def _shot_playlist(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-shot clips in cut order, for playback without a stitched file.

    Each clip already carries its narration audio and burned-in subtitles, so
    playing them back to back previews the cut; only cross-shot transitions are
    missing, and those arrive with the real export. BGM is not in these clips
    either — see :func:`_preview_bgm`, which hands the viewer the raw tracks to
    play alongside.
    """
    assets = state.get("shot_assets", {}) or {}
    out: list[dict[str, Any]] = []
    for shot in state.get("shot_plan", []) or []:
        sid = str(shot.get("shot_id", "") or "")
        item = assets.get(sid) if isinstance(assets.get(sid), dict) else {}
        path = str((item or {}).get("rendered_path", "") or "")
        if not path:
            continue
        timeline = (item or {}).get("timeline") or {}
        out.append({
            "shot_id": sid,
            "path": path,
            "duration_sec": timeline.get("duration_sec") or shot.get("duration_sec") or 0,
        })
    return out


def _preview_bgm(
    state: dict[str, Any],
    roots: list[Path],
    default_volume: float = 0.5,
) -> dict[str, Any]:
    """The BGM tracks the viewer can play under a playlist preview.

    The mix into the video only happens at export, so a deferred cut previews
    as pure narration. Handing the raw tracks to the player and letting it run
    them in parallel is not the export mix — no ducking, no trim to length —
    but it is the difference between hearing the music and not.

    Tracks that fall outside ``media_roots`` are dropped here rather than
    handed over and 403'd one request later.
    """
    out: list[dict[str, Any]] = []
    for track in state.get("bgm_tracks") or []:
        if not isinstance(track, dict):
            continue
        raw = str(
            track.get("music_path")
            or track.get("path")
            or track.get("audio_path")
            or track.get("file_path")
            or ""
        )
        if not raw:
            continue
        try:
            target = Path(raw).resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if not target.is_file() or target.suffix.lower() not in ALLOWED_MEDIA_EXT:
            continue
        if not any(target == root or root in target.parents for root in roots):
            continue
        out.append({
            "path": str(target),
            "title": str(track.get("title") or track.get("name") or ""),
            "duration_sec": float(track.get("duration_sec") or 0) or 0.0,
        })

    try:
        volume = float(state.get("bgm_volume") or default_volume)
    except (TypeError, ValueError):
        volume = default_volume
    return {"tracks": out, "volume": max(0.0, min(1.0, volume))}


def _run_id_for_project(project_dir: str) -> str:
    """The generation run that produced a project, if its journal is still around.

    Lets the UI put the original trajectory back on screen when a project is
    reopened, instead of showing only the finished artefacts.
    """
    target = str(project_dir)
    best: tuple[str, str] = ("", "")
    for run in studio.store.runs.values():
        if run.project_dir != target or run.id.startswith(("chat_", "edit_", "export_")):
            continue
        if run.started_at > best[1]:
            best = (run.id, run.started_at)
    return best[0]


def _require_access(project_dir: str | Path) -> auth_mod.User:
    """Refuse a project the signed-in user does not own.

    404 rather than 403: a user who cannot see a project should not be able to
    confirm it exists by probing directory names.
    """
    user = require_user()
    if not studio.config.auth_enabled:
        return user
    if not studio.owner_index.can_access(project_dir, user):
        target = Path(project_dir).resolve()
        inherited = any(
            (parent / project_mod.STATE_FILENAME).is_file()
            and studio.owner_index.can_access(parent, user)
            for parent in target.parents
        )
        if not inherited:
            raise HTTPException(404, "工程不存在或无权访问")
    return user


def _require_run_access(run_id: str) -> Any:
    run = studio.store.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    user = require_user()
    if not studio.config.auth_enabled or user.is_admin:
        return run
    owner = getattr(run, "owner", "")
    if owner and owner == user.username:
        return run
    if run.project_dir and studio.owner_index.can_access(run.project_dir, user):
        return run
    raise HTTPException(404, "run not found")


def _visible_runs(user: auth_mod.User) -> list[dict[str, Any]]:
    if not studio.config.auth_enabled or user.is_admin:
        return studio.store.list()
    out = []
    for run in sorted(studio.store.runs.values(),
                      key=lambda x: x.started_at, reverse=True):
        owner = getattr(run, "owner", "")
        if owner == user.username or (
                run.project_dir and studio.owner_index.can_access(run.project_dir, user)):
            out.append(run.summary())
    return out


def _project_payload(proj: project_mod.Project) -> dict[str, Any]:
    snapshot = tracks.build_snapshot(proj.state, studio.track_order())
    playlist = _shot_playlist(proj.state)
    return {
        "project_id": proj.project_id,
        "project_dir": str(proj.project_dir),
        "run_id": _run_id_for_project(str(proj.project_dir)),
        "progress": snapshot,
        "digest": proj.digest(),
        "final_video_path": proj.final_video_path,
        "playlist": playlist,
        "preview_bgm": _preview_bgm(
            proj.state,
            studio.config.media_roots,
            float((studio.config.pipeline_config.get("music") or {}).get("bgm_volume", 0.5)),
        ),
        # True when the cut is validated but never concatenated, i.e. the
        # viewer should play the playlist and offer an export.
        "stitch_pending": bool(
            not proj.final_video_path
            and playlist
            and (proj.state.get("stitch_pending")
                 or (proj.state.get("stitch_result") or {}).get("deferred"))),
        "dirty": {sid: sorted(r) for sid, r in proj.dirty.items()},
        "bgm_dirty": proj.bgm_dirty,
        "restitch_dirty": proj.restitch_dirty,
        "subtitle_drift": editor.subtitle_drift(proj),
        "pending_steps": [
            {"tool": s["tool"], "label": s["label"]}
            for s in editor.plan_regeneration(proj)
        ],
        "versions": proj.list_versions(),
        "edit_history": proj.history[-20:],
    }


def _discover_projects() -> list[dict[str, Any]]:
    roots = [studio.config.workspace_root / "projects", studio.config.workspace_root]
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for root in roots:
        for item in project_mod.discover(root):
            if item["project_dir"] in seen:
                continue
            seen.add(item["project_dir"])
            out.append(item)
    out.sort(key=lambda x: x["modified_ts"], reverse=True)
    return out


@app.get("/api/projects")
async def list_projects() -> dict[str, Any]:
    user = require_user()
    # The scan stats every project directory and takes over a minute on the
    # network filesystem. Run in a thread: on the event loop it froze the whole
    # server — including other users' runs — for the duration.
    found = await asyncio.to_thread(_discover_projects)
    out: list[dict[str, Any]] = []
    for item in found:
        if studio.config.auth_enabled \
                and not studio.owner_index.can_access(item["project_dir"], user):
            continue
        # Admins see everything, so they need to know whose it is.
        item["owner"] = studio.owner_index.owner_of(item["project_dir"])
        out.append(item)
    return {"projects": out}


@app.get("/api/project")
async def get_project(dir: str) -> dict[str, Any]:
    if not studio.config.editor_enabled:
        raise HTTPException(403, "编辑功能已在配置中关闭 (editor.enabled=false)")
    return _project_payload(_load_project(dir))


@app.delete("/api/project")
@app.post("/api/project/delete")
async def delete_project(dir: str = "", project_dir: str = "") -> dict[str, Any]:
    """Delete a finished project, including its rendered video.

    Refused while anything is still writing to it, since removing the directory
    under a running job would fail halfway and leave a corrupt project behind.

    Accepts both DELETE /api/project?dir=... and POST /api/project/delete?dir=...
    (the POST fallback exists because some corporate proxies block DELETE).
    """
    target = dir or project_dir
    if not target:
        raise HTTPException(400, "必须提供 dir 或 project_dir 参数")
    key = str(Path(target).resolve())
    _require_access(key)

    lock = studio.regen_locks.get(key)
    if lock is not None and lock.locked():
        raise HTTPException(409, "该工程正在重新渲染，请等待结束后再删除")
    for run in studio.store.runs.values():
        if run.is_active and run.project_dir and str(Path(run.project_dir).resolve()) == key:
            raise HTTPException(409, "该工程有任务正在运行，无法删除")

    roots = [studio.config.workspace_root / "projects", studio.config.workspace_root]
    try:
        result = project_mod.delete(target, roots)
    except project_mod.ProjectError as exc:
        raise HTTPException(400, str(exc)) from exc

    # Drop cached state so a later project at the same path starts clean.
    studio.projects.pop(key, None)
    studio.chats.pop(key, None)
    studio.regen_locks.pop(key, None)
    if studio.owners is not None:
        studio.owner_index.release(key)
    for run in studio.store.runs.values():
        if run.project_dir and str(Path(run.project_dir).resolve()) == key:
            run.project_dir = ""
    return {"deleted": True, **result}


class OpsRequest(BaseModel):
    project_dir: str
    ops: list[dict[str, Any]] = []
    dry_run: bool = False


@app.post("/api/project/ops")
async def apply_ops(req: OpsRequest) -> dict[str, Any]:
    if not studio.config.editor_enabled:
        raise HTTPException(403, "编辑功能已关闭")
    proj = _load_project(req.project_dir)
    flags = studio.editor_flags()

    try:
        if req.dry_run:
            results = editor.precheck_ops(proj, req.ops, flags)
            return {
                "dry_run": True,
                "results": [r.as_dict() for r in results],
                "ok": all(r.ok for r in results),
            }
        results = editor.apply_ops(proj, req.ops, flags)
    except editor.EditError as exc:
        raise HTTPException(400, str(exc)) from exc

    rerender = studio.config.editor.get("rerender") or {}
    proj.save("manual_edit", max_versions=int(rerender.get("max_versions", 10) or 10))
    return {
        "dry_run": False,
        "results": [r.as_dict() for r in results],
        "ok": True,
        **_project_payload(proj),
    }


class RegenRequest(BaseModel):
    project_dir: str
    scope: str = ""
    restitch: bool | None = None


def _tool_result_event(
    orchestrator: Any,
    tool: str,
    step: int,
    output: Any,
) -> dict[str, Any]:
    status = (output or {}).get("status", "done") if isinstance(output, dict) else "done"
    observation: dict[str, Any] = {}
    if isinstance(output, dict):
        try:
            observation = orchestrator._make_agent_observation(tool, output)
        except Exception:
            observation = {"status": status, "summary": dict(output)}
    return {
        "type": "tool_result",
        "stage": tool,
        "step": step,
        "tool_name": tool,
        "status": status,
        "observation": observation,
    }


@app.post("/api/project/regenerate")
async def regenerate(req: RegenRequest) -> dict[str, Any]:
    """Re-run the agent's tools over edited shots, then re-stitch.

    Progress streams over ``/ws/{job_id}`` using the same event vocabulary as a
    generation run, so the frontend reuses one renderer for both.
    """
    if not studio.config.editor_enabled:
        raise HTTPException(403, "编辑功能已关闭")
    # Authorise before reporting service state, so the answer for someone
    # else's project does not depend on whether the model happens to be up.
    _require_access(req.project_dir)
    if studio.model is None or not studio.model.is_loaded:
        raise HTTPException(503, "模型尚未加载完成，无法重渲染")

    proj = _load_project(req.project_dir)
    rerender_cfg = studio.config.editor.get("rerender") or {}
    scope = req.scope or str(rerender_cfg.get("scope", "only_dirty"))
    restitch = (
        bool(rerender_cfg.get("auto_restitch", True)) if req.restitch is None else req.restitch
    )

    steps = editor.plan_regeneration(proj, scope)
    if not steps and not restitch:
        return {"status": "noop", "message": "没有需要重新生成的内容"}

    job_id = f"edit_{uuid.uuid4().hex[:8]}"
    job = studio.store.create(f"[重渲染] {proj.project_id}", run_id=job_id)
    job.project_dir = str(proj.project_dir)

    job.task = asyncio.create_task(_execute_regeneration(job_id, proj, steps, restitch))
    return {
        "status": "started",
        "job_id": job_id,
        "steps": [{"tool": s["tool"], "label": s["label"]} for s in steps],
        "restitch": restitch,
    }


class ExportRequest(BaseModel):
    project_dir: str


@app.post("/api/project/export")
async def export_project(req: ExportRequest) -> dict[str, Any]:
    """Run the real stitch for a project whose generation deferred it.

    Generation stops at a validated manifest (``render.defer_stitch``), so this
    is where the per-shot transcode, ffmpeg concat and BGM mix actually happen.
    Progress streams over ``/ws/{job_id}`` like any other job, which is what
    puts the export in the conversation and in the run history.
    """
    proj = _load_project(req.project_dir)

    job_id = f"export_{uuid.uuid4().hex[:8]}"
    job = studio.store.create(f"[导出成片] {proj.project_id}", run_id=job_id)
    job.project_dir = str(proj.project_dir)
    job.task = asyncio.create_task(_execute_export(job_id, proj))
    return {"status": "started", "job_id": job_id}


async def _execute_export(job_id: str, proj: project_mod.Project) -> None:
    from pipeline.tools import stitch_video as sv_tool

    store = studio.store
    job = store.get(job_id)
    cfg = studio.config
    loop = asyncio.get_running_loop()

    async def broadcast(event: dict[str, Any]) -> None:
        await store.emit(job_id, event)

    def on_progress(event: dict[str, Any]) -> None:
        text = (f"  {event.get('stage', 'stitch')} "
                f"[{event.get('current', 0)}/{event.get('total', 0)}] "
                f"{event.get('label', '')}")
        loop.call_soon_threadsafe(
            lambda: asyncio.create_task(
                broadcast({"type": "console_log", "level": "stdout", "text": text})))

    await broadcast({"type": "run_started", "run_id": job_id})
    await broadcast({
        "type": "tool_call", "stage": "Tool_StitchVideo", "step": 1,
        "tool_name": "Tool_StitchVideo",
    })

    lock = studio.regen_lock(str(proj.project_dir))
    async with lock:
        try:
            music_cfg = cfg.pipeline_config.get("music") or {}
            state = proj.state
            result = await asyncio.to_thread(
                sv_tool.run,
                state.get("shot_plan", []),
                state.get("shot_assets", {}),
                str(proj.project_dir),
                bgm_tracks=state.get("bgm_tracks", []),
                bgm_volume=float(state.get("bgm_volume") or music_cfg.get("bgm_volume", 0.5)),
                duck_under_voice=bool(music_cfg.get("duck_under_voice", True)),
                config=cfg.pipeline_config,
                progress_callback=on_progress,
                dry_run=False,
            )
            final_path = str(
                result.get("final_video_path") or result.get("output_path") or "")
            state["stitch_result"] = result
            state["final_video_path"] = final_path
            state["stitch_pending"] = False
            proj.save("export")

            if job is not None:
                job.status = "completed" if final_path else "error"
                job.result = {
                    "final_video_path": final_path,
                    "project_dir": str(proj.project_dir),
                }
            await broadcast({
                "type": "tool_result", "stage": "Tool_StitchVideo", "step": 1,
                "tool_name": "Tool_StitchVideo",
                "status": "success" if final_path else "error",
                "observation": {"summary": {
                    "final_video_path": final_path,
                    "shot_count": result.get("ready_shot_count", 0),
                    "total_duration_sec": result.get("total_duration_sec", 0),
                }},
            })
            if not final_path:
                await broadcast({
                    "type": "error", "stage": "export",
                    "message": f"合成未产出成片 (concat_status="
                               f"{result.get('concat_status', 'unknown')})",
                })
        except asyncio.CancelledError:
            if job is not None:
                job.status = "cancelled"
            await broadcast({"type": "console_log", "level": "warn", "text": "导出已取消"})
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            logger.exception("export %s failed", job_id)
            if job is not None:
                job.status = "error"
                job.errors.append(detail)
            await broadcast({"type": "error", "stage": "export", "message": detail})
        finally:
            if job is not None:
                job.finished_at = _now_iso()

    await broadcast({
        "type": "run_complete",
        "status": job.status if job else "completed",
        "elapsed_seconds": job.elapsed if job else 0,
        "result": job.result if job else {},
        "project_dir": str(proj.project_dir),
    })


async def _execute_regeneration(
    job_id: str,
    proj: project_mod.Project,
    steps: list[dict[str, Any]],
    restitch: bool,
    strict: bool = False,
    policy_version: str | None = None,
) -> None:
    """Regenerate ``steps`` for ``proj``.

    ``strict`` is opt-in and used only by the 0907 runner.
    """
    from studio.orchestrator import AutonomousOrchestrator

    store = studio.store
    job = store.get(job_id)
    cfg = studio.config
    pipeline_config = _pipeline_config_for_policy(cfg.pipeline_config, policy_version)

    async def broadcast(rid: str, event: dict[str, Any]) -> None:
        await store.emit(rid, event)

    async def progress(rid: str, snapshot: dict[str, Any]) -> None:
        await store.emit(rid, {"type": "progress", "progress": snapshot})

    lock = studio.regen_lock(str(proj.project_dir))
    if lock.locked():
        await broadcast(job_id, {
            "type": "console_log", "level": "warn",
            "text": "该工程已有重渲染任务在执行，排队等待...",
        })

    async with lock:
        orchestrator = AutonomousOrchestrator(
            studio.model.loader,
            pipeline_config,
            cfg.stage_config,
            cfg.system_prompt,
            broadcast,
            cfg.infer_config,
            progress_fn=progress,
        )
        if job is not None:
            job.orchestrator = orchestrator

        all_steps = list(steps)
        if restitch:
            bgm_changed = any(s["tool"] == "Tool_MusicSearch" for s in steps)
            all_steps.append(editor.stitch_step(proj, bgm_changed=bgm_changed))

        regen_timeout = int(cfg.editor.get("rerender", {}).get("timeout_sec", 900) or 900)
        await broadcast(job_id, {"type": "run_started", "run_id": job_id})
        try:
            for idx, step in enumerate(all_steps, start=1):
                tool = step["tool"]
                args = step["args"]
                shot_ids_filter = args.get("shot_ids")

                await broadcast(job_id, {
                    "type": "stage_update",
                    "stage": tool,
                    "status": "started",
                    "message": f"[{idx}/{len(all_steps)}] {step['label']}",
                    "progress_pct": int(round((idx - 1) * 100 / len(all_steps))),
                })

                try:
                    output = await asyncio.wait_for(
                        _dispatch_scoped(
                            orchestrator, tool, args, proj.state, job_id,
                            shot_ids_filter=shot_ids_filter,
                            broadcast_fn=broadcast,
                        ),
                        timeout=regen_timeout,
                    )
                except asyncio.TimeoutError:
                    await broadcast(job_id, {
                        "type": "console_log", "level": "error",
                        "text": f"  ✗ {tool} 超时 ({regen_timeout}s)，跳过",
                    })
                    output = {"status": "timeout", "error": f"{tool} timed out after {regen_timeout}s"}

                if strict and isinstance(output, dict) \
                        and str(output.get("status", "")).lower() in {"timeout", "error", "failed"}:
                    raise RuntimeError(
                        f"strict regeneration aborted: {tool} status="
                        f"{output.get('status')} ({output.get('error', '')})"
                    )

                if isinstance(output, dict):
                    orchestrator.update_state_from_respond(tool, output)
                    orchestrator._apply_tool_output(tool, output, proj.state)
                await broadcast(job_id, _tool_result_event(orchestrator, tool, idx, output))
                await progress(job_id, tracks.build_snapshot(proj.state, studio.track_order()))

            proj.clear_dirty()
            rerender_cfg = cfg.editor.get("rerender") or {}
            if bool(rerender_cfg.get("keep_version_history", True)):
                proj.save("rerender", max_versions=int(rerender_cfg.get("max_versions", 10) or 10))
            else:
                proj.save()

            if job is not None:
                job.status = "completed"
                job.result = {
                    "final_video_path": proj.final_video_path,
                    "project_dir": str(proj.project_dir),
                }
        except asyncio.CancelledError:
            if job is not None:
                job.status = "cancelled"
            await broadcast(job_id, {
                "type": "console_log", "level": "warn", "text": "重渲染已取消",
            })
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            logger.exception("regeneration %s failed", job_id)
            if job is not None:
                job.status = "error"
                job.errors.append(detail)
            await broadcast(job_id, {"type": "error", "stage": "rerender", "message": detail})
        finally:
            if job is not None:
                job.orchestrator = None
                job.finished_at = _now_iso()

    await store.flush_progress(job_id)
    await broadcast(job_id, {
        "type": "run_complete",
        "status": job.status if job else "completed",
        "elapsed_seconds": job.elapsed if job else 0,
        "result": job.result if job else {},
        "project_dir": str(proj.project_dir),
    })


def _invalidate_visual_audio_cache(state: dict[str, Any]) -> None:
    """Remove pipeline-level disk cache for shot_assets so retrieval reruns."""
    config = state.get("config", {}) or {}
    storage_root = str((config.get("paths", {}) or {}).get("storage_root", "") or "").strip()
    if not storage_root:
        return
    output_dir = state.get("output_dir", "")
    if not output_dir:
        return
    cache_file = Path(output_dir) / "stage3_visual_audio" / "shot_assets.json"
    if cache_file.exists():
        try:
            cache_file.unlink()
            logger.info("已删除视频素材缓存: %s", cache_file)
        except OSError:
            pass
    # Also check pipeline's own cache root (keyed by project)
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from pipeline.graph import _cache_root_dir  # type: ignore
        cache_root = _cache_root_dir(state)
        if cache_root:
            global_cache = cache_root / "stage3_visual_audio" / "shot_assets.json"
            if global_cache.exists():
                global_cache.unlink()
                logger.info("已删除全局素材缓存: %s", global_cache)
    except Exception:
        pass


_PRESERVED_ASSET_KEYS = ("decoration", "decoration_verify", "template_retrieve", "semantic_animations")


async def _dispatch_scoped(
    orchestrator: Any,
    tool: str,
    args: dict[str, Any],
    state: dict[str, Any],
    run_id: str,
    *,
    shot_ids_filter: list[str] | None = None,
    broadcast_fn: Any = None,
) -> dict[str, Any] | None:
    """Run a tool against only the specified shots, preserving untouched ones.

    The upstream dispatch (`_dispatch_tool`) always processes every shot in
    ``state["shot_plan"]`` regardless of arguments.  To get selective behaviour
    without rewriting the pipeline nodes, we temporarily narrow the plan and
    assets visible in the state, dispatch, then merge the results back.

    **Decoration preservation**: pipeline nodes like ``node_visual_audio_batch``
    overwrite ``shot_assets`` wholesale via ``_merge_state_patch``, which would
    discard the decoration metadata already generated for a spotlight shot.  We
    snapshot those keys before dispatch and restore them afterwards so that the
    subsequent ``Tool_BatchTimelineCompose`` can still render HyperFrames.

    The restore must treat an *empty* value as missing, not just an absent key:
    ``pipeline/graph.py`` rebuilds each asset dict with a literal
    ``"decoration": {}`` placeholder, so the key is always present after a
    footage/TTS re-run.  Restoring with ``setdefault`` silently kept that empty
    placeholder, the renderer saw ``0 个 Spotlight 有装饰`` and fell back to
    plain ffmpeg — the shot came back with its HyperFrames animation stripped.

    For ``Tool_StitchVideo`` this scoping is skipped — stitching always needs
    the full shot sequence to produce the final video.
    """
    if not shot_ids_filter or tool == "Tool_StitchVideo":
        return await orchestrator._dispatch_tool(tool, args, state, run_id)

    # When re-running Tool_BatchVisualAudio for specific shots (e.g. after
    # replace_footage), the pipeline graph node has its own disk cache that is
    # keyed by project, NOT by visual_query. If the old cache file still exists
    # it would happily return stale assets. Delete it so the retrieval actually
    # runs with the new query.
    if tool == "Tool_BatchVisualAudio":
        _invalidate_visual_audio_cache(state)

    filter_set = set(shot_ids_filter)
    full_plan = state.get("shot_plan", [])
    full_assets = state.get("shot_assets", {})

    scoped_plan = [s for s in full_plan if str(s.get("shot_id", "")) in filter_set]
    scoped_assets = {k: v for k, v in full_assets.items() if k in filter_set}

    # Save decoration metadata that pipeline nodes would overwrite.
    saved_deco: dict[str, dict[str, Any]] = {}
    for sid in filter_set:
        sa = full_assets.get(sid)
        if isinstance(sa, dict):
            kept = {k: sa[k] for k in _PRESERVED_ASSET_KEYS if k in sa}
            clip_meta = sa.get("clip_meta") if isinstance(sa.get("clip_meta"), dict) else {}
            playlist = (
                clip_meta.get("visual_playlist")
                if isinstance(clip_meta.get("visual_playlist"), dict)
                else None
            )
            clips = playlist.get("clips", []) if playlist else []
            if any(
                isinstance(clip, dict) and (clip.get("user_override") or clip.get("replaced"))
                for clip in clips
            ):
                kept["_visual_playlist0907"] = copy.deepcopy(playlist)
            if kept:
                saved_deco[sid] = kept

    skipped_count = len(full_plan) - len(scoped_plan)
    if broadcast_fn and skipped_count > 0:
        await broadcast_fn(run_id, {
            "type": "console_log",
            "level": "stdout",
            "text": (
                f"  跳过 {skipped_count} 个未改动镜头，"
                f"只处理 {len(scoped_plan)} 个: "
                f"{', '.join(str(s.get('shot_id', '')) for s in scoped_plan[:8])}"
                + ("..." if len(scoped_plan) > 8 else "")
            ),
        })

    state["shot_plan"] = scoped_plan
    state["shot_assets"] = scoped_assets
    old_pipeline_config = getattr(orchestrator, "pipeline_config", None)
    if (
        isinstance(old_pipeline_config, dict)
        and str(old_pipeline_config.get("policy_version") or "") == "0907"
    ):
        try:
            from studio.newvis0907 import footage as footage_0907
            full_occupancy = footage_0907.build_occupancy(
                {"shot_plan": full_plan, "shot_assets": full_assets},
                exclude_shot_ids=list(filter_set),
            )
            runtime = dict(old_pipeline_config.get("newvis0907_runtime") or {})
            runtime["base_occupancy"] = full_occupancy
            orchestrator.pipeline_config = {
                **old_pipeline_config,
                "newvis0907_runtime": runtime,
            }
        except Exception:
            pass

    try:
        output = await orchestrator._dispatch_tool(tool, args, state, run_id)
    finally:
        if isinstance(old_pipeline_config, dict):
            orchestrator.pipeline_config = old_pipeline_config
        state["shot_plan"] = full_plan
        merged_assets = dict(full_assets)
        merged_assets.update(state.get("shot_assets", {}))
        for sid, deco in saved_deco.items():
            sa = merged_assets.get(sid)
            if isinstance(sa, dict):
                for k, v in deco.items():
                    if k == "_visual_playlist0907":
                        clip_meta = sa.get("clip_meta") if isinstance(sa.get("clip_meta"), dict) else {}
                        sa["clip_meta"] = {**clip_meta, "visual_playlist": v}
                        sa["visual_playlist"] = v
                        continue
                    if not sa.get(k):
                        sa[k] = v
        state["shot_assets"] = merged_assets

    return output


class VersionRequest(BaseModel):
    project_dir: str
    name: str


@app.get("/api/project/versions")
async def list_versions(dir: str) -> dict[str, Any]:
    return {"versions": _load_project(dir).list_versions()}


@app.post("/api/project/restore")
async def restore_version(req: VersionRequest) -> dict[str, Any]:
    proj = _load_project(req.project_dir)
    try:
        proj.restore_version(req.name)
    except project_mod.ProjectError as exc:
        raise HTTPException(404, str(exc)) from exc
    proj.save("pre_restore")
    return {"restored": req.name, **_project_payload(proj)}


# ---------------------------------------------------------------------------
# conversational editing
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    project_dir: str
    message: str = ""
    apply: bool = False
    ops: list[dict[str, Any]] | None = None


def _chat_memory_path(proj: project_mod.Project) -> Path:
    return proj.project_dir / "chat_memory.json"


def _chat_for(proj: project_mod.Project) -> chat_agent.ChatEditAgent:
    key = str(proj.project_dir)
    existing = studio.chats.get(key)
    if existing is not None:
        return existing

    cfg = studio.config
    backend = build_chat_backend(cfg.chat_edit, studio.model)
    # Restore conversation memory from disk so it survives server restarts.
    mem_path = _chat_memory_path(proj)
    memory = chat_agent.Memory.load(mem_path, cfg.chat_edit)
    agent = chat_agent.ChatEditAgent(
        generate=backend.generate,
        chat_config=cfg.chat_edit,
        editor_flags=studio.editor_flags(),
    )
    agent.memory = memory
    studio.chats[key] = agent
    return agent


def _save_chat_memory(proj: project_mod.Project, agent: chat_agent.ChatEditAgent) -> None:
    try:
        agent.memory.save(_chat_memory_path(proj))
    except Exception as exc:
        logger.warning("Failed to save chat memory: %s", exc)


@app.post("/api/project/chat")
async def project_chat(req: ChatRequest) -> dict[str, Any]:
    cfg = studio.config
    if not cfg.chat_enabled:
        raise HTTPException(403, "对话编辑已在配置中关闭 (chat_edit.enabled=false)")

    proj = _load_project(req.project_dir)
    agent = _chat_for(proj)

    # Confirmation path: the user approved ops the agent proposed earlier.
    if req.apply:
        ops = req.ops or []
        if not ops:
            raise HTTPException(400, "apply=true 时必须带上 ops")
        try:
            results = editor.apply_ops(proj, ops, studio.editor_flags())
        except editor.EditError as exc:
            raise HTTPException(400, str(exc)) from exc
        rerender = cfg.editor.get("rerender") or {}
        proj.save("chat_edit", max_versions=int(rerender.get("max_versions", 10) or 10))
        resp: dict[str, Any] = {
            "applied": True,
            "results": [r.as_dict() for r in results],
            **_project_payload(proj),
        }

        chat_cfg = cfg.chat_edit or {}
        safety = chat_cfg.get("safety") if isinstance(chat_cfg.get("safety"), dict) else {}
        if bool(safety.get("allow_rerender", True)) and studio.model and studio.model.is_loaded:
            scope = str(rerender.get("scope", "only_dirty"))
            steps = editor.plan_regeneration(proj, scope)
            # `reorder_shots`, `delete_shot` and BGM/volume changes need no
            # per-shot regeneration, only a re-stitch. Gating on `steps` alone
            # left them applied in state but never rendered into the cut.
            if editor.needs_restitch(proj, steps):
                restitch = bool(rerender.get("auto_restitch", True))
                job_id = f"chat_{uuid.uuid4().hex[:8]}"
                job = studio.store.create(
                    f"[对话确认修改] {', '.join(o.get('op','') for o in ops[:3])}", run_id=job_id)
                job.project_dir = str(proj.project_dir)
                job.task = asyncio.create_task(
                    _execute_regeneration(job_id, proj, steps, restitch))
                resp["regen_job_id"] = job_id
                resp["regen_steps"] = [
                    {"tool": s["tool"], "label": s["label"]} for s in steps
                ]
        return resp

    try:
        payload = await agent.handle(proj, req.message)
    except ModelNotReady as exc:
        raise HTTPException(503, str(exc)) from exc

    # Low-risk edits apply immediately; risky ones wait for confirmation.
    if payload.get("intent") == "edit" and payload.get("ops") and not payload.get("needs_confirm"):
        try:
            results = editor.apply_ops(proj, payload["ops"], studio.editor_flags())
            rerender = cfg.editor.get("rerender") or {}
            proj.save("chat_edit", max_versions=int(rerender.get("max_versions", 10) or 10))
            payload["applied"] = True
            payload["results"] = [r.as_dict() for r in results]

            # Auto-trigger regeneration so the user sees the change in the video
            # without needing to manually click "重新渲染".
            chat_cfg = cfg.chat_edit or {}
            safety = chat_cfg.get("safety") if isinstance(chat_cfg.get("safety"), dict) else {}
            if bool(safety.get("allow_rerender", True)) and studio.model and studio.model.is_loaded:
                scope = str(rerender.get("scope", "only_dirty"))
                steps = editor.plan_regeneration(proj, scope)
                if editor.needs_restitch(proj, steps):
                    restitch = bool(rerender.get("auto_restitch", True))
                    job_id = f"chat_{uuid.uuid4().hex[:8]}"
                    job = studio.store.create(
                        f"[对话修改] {payload.get('reply', '')[:30]}", run_id=job_id)
                    job.project_dir = str(proj.project_dir)
                    job.task = asyncio.create_task(
                        _execute_regeneration(job_id, proj, steps, restitch))
                    payload["regen_job_id"] = job_id
                    payload["regen_steps"] = [
                        {"tool": s["tool"], "label": s["label"]} for s in steps
                    ]
                    dirty_ids = sorted(proj.dirty.keys())
                    payload["reply"] += (
                        f"\n\n正在自动重渲染改动的 {len(dirty_ids)} 个镜头，"
                        "完成后视频会自动更新。"
                        if dirty_ids else
                        "\n\n正在重新拼接成片，完成后视频会自动更新。"
                    )
        except editor.EditError as exc:
            payload["applied"] = False
            payload["intent"] = "clarify"
            payload["reply"] += f"\n\n（应用时被拒绝：{exc}）"

    payload["project"] = _project_payload(proj)
    _save_chat_memory(proj, agent)
    return payload


@app.post("/api/project/chat/reset")
async def reset_chat(dir: str) -> dict[str, Any]:
    proj = _load_project(dir)
    agent = studio.chats.get(str(proj.project_dir))
    if agent is not None:
        agent.reset()
        _save_chat_memory(proj, agent)
    else:
        mem_path = _chat_memory_path(proj)
        if mem_path.exists():
            mem_path.unlink(missing_ok=True)
    return {"reset": True}


def _memory_for(proj: project_mod.Project) -> chat_agent.Memory:
    """Conversation memory for a project, from cache or from disk.

    The read-only endpoints must not go through ``_chat_for``: that builds a
    chat agent (and needs a model backend) just to answer "what was said". They
    used to read the in-process cache alone, so reopening a project after a
    restart showed an empty conversation even though ``chat_memory.json`` was
    sitting right there.
    """
    agent = studio.chats.get(str(proj.project_dir))
    if agent is not None:
        return agent.memory
    return chat_agent.Memory.load(_chat_memory_path(proj), studio.config.chat_edit)


@app.get("/api/project/chat/memory")
async def chat_memory(dir: str) -> dict[str, Any]:
    return _memory_for(_load_project(dir)).as_dict()


@app.get("/api/project/chat/history")
async def chat_history(dir: str) -> dict[str, Any]:
    """Return conversation turns for the frontend to re-render on page load."""
    memory = _memory_for(_load_project(dir))
    return {
        "messages": [
            {"role": t.role, "content": t.content, "at": t.at, "meta": t.meta}
            for t in memory.turns
        ],
        "summary": memory.summary,
        "compressions": memory.compressions,
    }


# ---------------------------------------------------------------------------
# media
# ---------------------------------------------------------------------------
ALLOWED_MEDIA_EXT = {
    ".mp4", ".webm", ".mov", ".mkv",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg",
    ".jpg", ".jpeg", ".png", ".webp",
}


def _media_is_accessible(target: Path, user: auth_mod.User) -> bool:
    """True when ``target`` is outside the project tree, or inside an owned one.

    Shared assets — the footage library, BGM, cached render output — are not
    per-user and stay readable; anything under ``<workspace>/projects`` must
    belong to the caller.
    """
    projects_root = (studio.config.workspace_root / "projects").resolve()
    try:
        rel = target.resolve().relative_to(projects_root)
    except ValueError:
        return True
    if not rel.parts:
        return False
    return studio.owner_index.can_access(projects_root / rel.parts[0], user)


@app.get("/api/media")
async def serve_media(path: str) -> FileResponse:
    """Serve a media file, restricted to configured roots.

    Resolves symlinks before the containment check so a link inside a media root
    cannot be used to read arbitrary files.
    """
    try:
        target = Path(path).resolve(strict=True)
    except (OSError, RuntimeError):
        raise HTTPException(404, "file not found")

    if not target.is_file():
        raise HTTPException(404, "not a file")
    if target.suffix.lower() not in ALLOWED_MEDIA_EXT:
        raise HTTPException(415, f"unsupported media type: {target.suffix}")

    roots = studio.config.media_roots
    if not any(target == root or root in target.parents for root in roots):
        raise HTTPException(403, "path outside configured media_roots")

    # Containment in a media root is not authorisation: every user's projects
    # live under the same workspace, so without this a guessed project id would
    # stream somebody else's cut.
    if studio.config.auth_enabled:
        user = require_user()
        if not user.is_admin and not _media_is_accessible(target, user):
            raise HTTPException(404, "file not found")

    mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    # `no-cache` means "revalidate before reuse", not "never cache": the final cut
    # is rewritten in place at the same path on every edit, so a cached body would
    # keep showing the pre-edit video. Range requests still work for seeking.
    return FileResponse(
        str(target),
        media_type=mime,
        headers={"Accept-Ranges": "bytes", "Cache-Control": "no-cache, must-revalidate"},
    )


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoint used by pipeline tools
# ---------------------------------------------------------------------------
class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[dict[str, Any]] = []
    temperature: float = 0.2
    max_tokens: int = 2048
    top_p: float = 1.0
    stream: bool = False


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif item.get("type"):
                    parts.append(f"[{item['type']}]")
            elif item is not None:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> Any:
    if studio.model is None or not studio.model.is_loaded:
        return JSONResponse({"error": {"message": "model not loaded"}}, status_code=503)

    system = ""
    user = ""
    for msg in req.messages:
        role = msg.get("role")
        if role == "system":
            system = _content_to_text(msg.get("content"))
        elif role == "user":
            user = _content_to_text(msg.get("content"))

    prediction = await asyncio.to_thread(
        studio.model.predict_text, system, user,
        max_tokens=req.max_tokens, temperature=req.temperature, top_p=req.top_p,
    )
    text = str(getattr(prediction, "cleaned_output", "") or "")
    return {
        "id": f"chatcmpl-studio-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or "local-aicut-9b",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": getattr(prediction, "prompt_tokens", 0),
            "completion_tokens": getattr(prediction, "completion_tokens", 0),
            "total_tokens": (
                getattr(prediction, "prompt_tokens", 0)
                + getattr(prediction, "completion_tokens", 0)
            ),
        },
    }


# newvis0907 dedicated surface (isolated router, additively wired here)
# ---------------------------------------------------------------------------
# The 0907 router is configured *after* every legacy helper above is defined, so
# it can reuse them through injected callbacks without importing this module (no
# circular import). Nothing above is modified: this is pure addition.
POLICY_VERSION_0907 = "0907"


@app.post("/api/0907/runs")
async def start_run_0907(req: StartRunRequest) -> dict[str, Any]:
    """Dedicated 0907 generation endpoint.

    Accepts the same body as the legacy ``POST /api/runs`` and shares the same
    run store + ``/ws/{run_id}`` fan-out, but binds the job to
    ``policy_version="0907"`` so the runner drives the trustworthy-cut pipeline
    from a *config copy*. The global studio config is never mutated, and the
    legacy ``/api/runs`` path is completely untouched.
    """
    text = (req.input_text or "").strip()
    if not text:
        raise HTTPException(400, "input_text 不能为空")
    if studio.model is None or not studio.model.is_loaded:
        raise HTTPException(
            503,
            studio.model.load_error if studio.model and studio.model.load_error
            else "模型尚未加载完成，请稍候",
        )

    run = studio.store.create(text)
    run.owner = require_user().username
    run.policy_version = POLICY_VERSION_0907
    studio.store.save_meta(run)
    run.task = asyncio.create_task(_execute_run(run.id, text, policy_version=POLICY_VERSION_0907))
    return {"run_id": run.id, "status": run.status, "policy_version": POLICY_VERSION_0907}


def _discover_projects_0907() -> list[dict[str, Any]]:
    """The 0907 library, filtered to the signed-in user like /api/projects."""
    user = require_user()
    out: list[dict[str, Any]] = []
    for item in _discover_projects():
        if studio.config.auth_enabled \
                and not studio.owner_index.can_access(item["project_dir"], user):
            continue
        item["owner"] = studio.owner_index.owner_of(item["project_dir"])
        out.append(item)
    return out


def _delete_project_0907(target: str) -> dict[str, Any]:
    """Delete a project for the 0907 surface, reusing the legacy safeguards."""
    if not target:
        raise HTTPException(400, "必须提供 dir 或 project_dir 参数")
    key = str(Path(target).resolve())
    _require_access(key)

    lock = studio.regen_locks.get(key)
    if lock is not None and lock.locked():
        raise HTTPException(409, "该工程正在重新渲染，请等待结束后再删除")
    for run in studio.store.runs.values():
        if run.is_active and run.project_dir and str(Path(run.project_dir).resolve()) == key:
            raise HTTPException(409, "该工程有任务正在运行，无法删除")

    roots = [studio.config.workspace_root / "projects", studio.config.workspace_root]
    try:
        result = project_mod.delete(target, roots)
    except project_mod.ProjectError as exc:
        raise HTTPException(400, str(exc)) from exc

    studio.projects.pop(key, None)
    studio.chats.pop(key, None)
    studio.regen_locks.pop(key, None)
    if studio.owners is not None:
        studio.owner_index.release(key)
    for run in studio.store.runs.values():
        if run.project_dir and str(Path(run.project_dir).resolve()) == key:
            run.project_dir = ""
    return {"deleted": True, **result}


def _export_project_0907(proj: project_mod.Project) -> dict[str, Any]:
    """Start the deferred stitch for a project (same job machinery as export)."""
    job_id = f"export_{uuid.uuid4().hex[:8]}"
    job = studio.store.create(f"[导出成片] {proj.project_id}", run_id=job_id)
    job.project_dir = str(proj.project_dir)
    job.task = asyncio.create_task(_execute_export(job_id, proj))
    return {"status": "started", "job_id": job_id}


def _refresh_quality_report_0907(proj: project_mod.Project) -> dict[str, Any] | None:
    """Recompute the unified quality report and persist it into project state.

    Stored under the ``quality_report0907`` namespace (kept by
    ``Project.to_payload``) so the trustworthy-cut UI can show a failed hard
    quality item without ever letting an unrelated legacy edit be blocked.
    """
    try:
        from studio.newvis0907 import quality as quality_mod

        # Convergence bookkeeping (I2): an auto-fixable issue that survives this
        # regeneration cycle is one failed fix attempt; resolved ones are cleared.
        prev_report = proj.state.get("quality_report0907")
        interim = quality_mod.evaluate_project(proj.state)
        quality_mod.advance_fix_attempts(proj.state, prev_report, interim)
        # Re-evaluate so needs_manual reflects the updated attempt counts.
        report = quality_mod.evaluate_project(proj.state)
        proj.state["quality_report0907"] = report
        try:
            proj.save()
        except (OSError, project_mod.ProjectError):
            pass
        return report
    except Exception:  # noqa: BLE001 - quality refresh must never crash a runner
        logger.exception("0907 quality report refresh failed")
        return None


def _capture_dirty_snapshot(proj: project_mod.Project) -> dict[str, Any]:
    """A restorable copy of every dirty reason (per shot + bgm + restitch)."""
    return {
        "shots": {sid: set(reasons) for sid, reasons in proj.dirty.items()},
        "bgm": bool(proj.bgm_dirty),
        "restitch": bool(proj.restitch_dirty),
    }


def _restore_dirty_snapshot(proj: project_mod.Project, snapshot: dict[str, Any]) -> None:
    """Put the dirty flags back exactly, so a retry regenerates the same steps."""
    proj.dirty.clear()
    for sid, reasons in (snapshot.get("shots") or {}).items():
        proj.dirty[str(sid)] = set(reasons)
    proj.bgm_dirty = bool(snapshot.get("bgm"))
    proj.restitch_dirty = bool(snapshot.get("restitch"))


async def _run_0907_regeneration(
    manager: Any,
    transaction: Any,
    job_id: str,
) -> None:
    """Dedicated 0907 background runner: render (strict) then verify.

    A step timeout/error becomes ``render_failed`` and can never slide into
    ``succeeded`` (``_execute_regeneration`` is called with ``strict=True``).
    Both a render failure and a verification failure restore the pre-render dirty
    snapshot so a retry re-generates the same steps. Verification compares real
    before/after file signals (see ``newvis0907.verification``): the targeted
    shots' rendered files and the final video must be freshly written.
    """
    try:
        manager.begin_render(transaction)
    except Newvis0907TxError:
        return

    proj = manager.project
    dirty_snapshot = _capture_dirty_snapshot(proj)
    try:
        if studio.model is None or not studio.model.is_loaded:
            _restore_dirty_snapshot(proj, dirty_snapshot)
            manager.record_render_failure(transaction, "模型尚未加载完成，无法重渲染")
            return

        scope = str((studio.config.editor.get("rerender") or {}).get("scope", "only_dirty"))
        steps = editor.plan_regeneration(proj, scope)

        # The run/journal job was created synchronously by ``_schedule_0907_runner``
        # (so the API response could hand the UI a ``regen_job_id`` to attach to);
        # here we only stream progress into it.
        try:
            await _execute_regeneration(
                job_id,
                proj,
                steps,
                restitch=True,
                strict=True,
                policy_version="0907",
            )
        except Exception as exc:  # noqa: BLE001 - any failure => render_failed
            _restore_dirty_snapshot(proj, dirty_snapshot)
            manager.record_render_failure(transaction, f"{type(exc).__name__}: {exc}")
            return

        finished = studio.store.get(job_id)
        status = getattr(finished, "status", "") if finished is not None else ""
        if status != "completed":
            _restore_dirty_snapshot(proj, dirty_snapshot)
            manager.record_render_failure(transaction, f"重渲染未成功（status={status}）")
            return

        # Audit record only — verification itself re-reads the files from disk.
        outputs: list[str] = []
        for sid in sorted(newvis0907_ops.target_shot_ids(proj, transaction.ops)):
            assets = proj.shot_assets.get(sid) if isinstance(proj.shot_assets.get(sid), dict) else {}
            rendered = str((assets or {}).get("rendered_path", "") or "")
            if rendered:
                outputs.append(rendered)
        render_result = {"outputs": outputs, "final_video_path": proj.final_video_path}

        manager.record_render_success(transaction, render_result)
        if transaction.state == newvis0907_schemas.TransactionState.VERIFICATION_FAILED.value:
            _restore_dirty_snapshot(proj, dirty_snapshot)
        elif transaction.state == newvis0907_schemas.TransactionState.SUCCEEDED.value:
            # Refresh the persisted project quality report after a verified edit.
            _refresh_quality_report_0907(proj)
    except Exception as exc:  # noqa: BLE001 - never leak into an optimistic state
        logger.exception("0907 runner failed for %s", getattr(transaction, "id", "?"))
        _restore_dirty_snapshot(proj, dirty_snapshot)
        try:
            manager.record_render_failure(transaction, f"{type(exc).__name__}: {exc}")
        except Newvis0907TxError:
            pass


# Fire-and-forget runner tasks are held here so the event loop cannot garbage
# collect a still-running task; each removes itself on completion.
_BACKGROUND_0907_TASKS: set[asyncio.Task] = set()


def _schedule_0907_runner(manager: Any, transaction: Any) -> str:
    """Create the regeneration job *synchronously* and return its id.

    A unique id is minted every call (a retry never reuses/overwrites the prior
    job) and it is never ``tx_tx...`` — the transaction id already starts with
    ``tx_``, so the job is ``regen_<tx-id>_<rand>``. The id is returned so the
    HTTP response can hand the frontend a ``regen_job_id`` to ``attachRun`` to,
    instead of the id being buried inside the async task where the UI never saw
    it.
    """
    proj = manager.project
    job_id = f"regen_{transaction.id}_{uuid.uuid4().hex[:6]}"
    job = studio.store.create(f"[0907修改] {proj.project_id}", run_id=job_id)
    job.project_dir = str(proj.project_dir)
    job.policy_version = POLICY_VERSION_0907
    task = asyncio.create_task(_run_0907_regeneration(manager, transaction, job_id))
    _BACKGROUND_0907_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_0907_TASKS.discard)
    return job_id


def _model_plan_0907(
    state: dict[str, Any],
    message: str,
    previous_targets: Any = None,
    history: Any = None,
) -> Any:
    """Ask the configured LoRA chat backend for a structured 0907 edit plan."""
    if studio.model is None or not studio.model.is_loaded:
        raise ModelNotReady(
            studio.model.load_error if studio.model and studio.model.load_error
            else "模型尚未加载完成，请稍候"
        )
    backend = build_chat_backend(studio.config.chat_edit, studio.model)
    system, user = _build_model_plan_prompt_0907(
        state if isinstance(state, dict) else {},
        message,
        previous_targets=previous_targets,
        history=history,
    )
    raw = backend.generate(system, user, max_tokens=1400, temperature=0.2)
    try:
        return json.loads(str(raw or ""))
    except json.JSONDecodeError:
        return raw


newvis0907_router.configure_router(
    newvis0907_router.RouterDeps(
        load_project=_load_project,
        project_payload=_project_payload,
        discover_projects=_discover_projects_0907,
        delete_project=_delete_project_0907,
        export_project=_export_project_0907,
        editor_flags=studio.editor_flags,
        model_plan=_model_plan_0907,
        schedule_runner=_schedule_0907_runner,
    )
)
app.include_router(newvis0907_router.router)
