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
import logging
import mimetypes
import os
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from studio import config as studio_config
from studio import chat_agent, editor, project as project_mod, tracks
from studio.model_client import ModelNotReady, ModelService, build_chat_backend
from studio.runs import RunStore

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


studio = Studio()
app = FastAPI(title="AICut Studio", version="1.0.0")


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
    )

    cfg.workspace_root.mkdir(parents=True, exist_ok=True)
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
        "features": {
            "editor": cfg.editor_enabled,
            "chat_edit": cfg.chat_enabled,
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
    run.task = asyncio.create_task(_execute_run(run.id, text))
    return {"run_id": run.id, "status": run.status}


@app.get("/api/runs")
async def list_runs() -> dict[str, Any]:
    return {"runs": studio.store.list()}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    run = studio.store.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return run.detail()


@app.delete("/api/runs/{run_id}")
@app.post("/api/runs/{run_id}/delete")
async def delete_run(run_id: str) -> dict[str, Any]:
    if not studio.store.delete(run_id):
        raise HTTPException(409, "运行中或不存在，无法删除")
    return {"deleted": run_id}


def _require_orchestrator(run_id: str) -> Any:
    run = studio.store.get(run_id)
    if run is None or run.orchestrator is None:
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
async def cancel_run(run_id: str) -> dict[str, Any]:
    run = _require_orchestrator(run_id)
    run.orchestrator.cancel()
    run.status = "cancelling"
    await studio.store.emit(run_id, {
        "type": "console_log", "level": "warn",
        "text": "已发送取消请求，当前步骤结束后停止",
    })
    return {"status": "cancelling"}


# ---------------------------------------------------------------------------
# run execution
# ---------------------------------------------------------------------------
async def _execute_run(run_id: str, input_text: str) -> None:
    """Drive a generation run on the deepagents harness.

    The control loop is a ``create_deep_agent`` ReAct graph; the AICut engine
    (state, dispatch, observation, preview) is reused via ``AICutRuntime``.
    """
    from aicut_agent.runner import create_runtime, run_aicut_generation

    run = studio.store.get(run_id)
    if run is None:
        return
    cfg = studio.config
    store = studio.store

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
        pipeline_config=cfg.pipeline_config,
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
        await broadcast(run_id, {
            "type": "console_log", "level": "warn", "text": "管线已被用户取消",
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
    """Load (and cache) a project so edits accumulate across requests."""
    key = str(Path(project_dir).resolve())
    cached = studio.projects.get(key)
    if cached is not None:
        return cached
    try:
        proj = project_mod.load(key, studio.config.pipeline_config)
    except project_mod.ProjectError as exc:
        raise HTTPException(404, str(exc)) from exc
    studio.projects[key] = proj
    return proj


def _project_payload(proj: project_mod.Project) -> dict[str, Any]:
    snapshot = tracks.build_snapshot(proj.state, studio.track_order())
    return {
        "project_id": proj.project_id,
        "project_dir": str(proj.project_dir),
        "progress": snapshot,
        "digest": proj.digest(),
        "final_video_path": proj.final_video_path,
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


@app.get("/api/projects")
async def list_projects() -> dict[str, Any]:
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


@app.post("/api/project/regenerate")
async def regenerate(req: RegenRequest) -> dict[str, Any]:
    """Re-run the agent's tools over edited shots, then re-stitch.

    Progress streams over ``/ws/{job_id}`` using the same event vocabulary as a
    generation run, so the frontend reuses one renderer for both.
    """
    if not studio.config.editor_enabled:
        raise HTTPException(403, "编辑功能已关闭")
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


async def _execute_regeneration(
    job_id: str,
    proj: project_mod.Project,
    steps: list[dict[str, Any]],
    restitch: bool,
) -> None:
    from studio.orchestrator import AutonomousOrchestrator

    store = studio.store
    job = store.get(job_id)
    cfg = studio.config

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
            cfg.pipeline_config,
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

                if isinstance(output, dict):
                    orchestrator.update_state_from_respond(tool, output)
                    orchestrator._apply_tool_output(tool, output, proj.state)
                await broadcast(job_id, {
                    "type": "tool_result",
                    "stage": tool,
                    "step": idx,
                    "tool_name": tool,
                    "status": (output or {}).get("status", "done") if isinstance(output, dict) else "done",
                })
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


_PRESERVED_ASSET_KEYS = ("decoration", "decoration_verify", "template_retrieve")


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

    try:
        output = await orchestrator._dispatch_tool(tool, args, state, run_id)
    finally:
        state["shot_plan"] = full_plan
        merged_assets = dict(full_assets)
        merged_assets.update(state.get("shot_assets", {}))
        # Restore decoration metadata that was wiped by the pipeline node.
        # Only a non-empty fresh value wins: the node either produced real new
        # decoration (keep it) or left the `{}` placeholder (restore the old).
        for sid, deco in saved_deco.items():
            sa = merged_assets.get(sid)
            if isinstance(sa, dict):
                for k, v in deco.items():
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


@app.get("/api/project/chat/memory")
async def chat_memory(dir: str) -> dict[str, Any]:
    proj = _load_project(dir)
    agent = studio.chats.get(str(proj.project_dir))
    if agent is None:
        return {"summary": "", "turns": [], "compressions": 0, "token_estimate": 0}
    return agent.memory.as_dict()


@app.get("/api/project/chat/history")
async def chat_history(dir: str) -> dict[str, Any]:
    """Return conversation turns for the frontend to re-render on page load."""
    proj = _load_project(dir)
    agent = studio.chats.get(str(proj.project_dir))
    if agent is None:
        return {"messages": []}
    msgs: list[dict[str, Any]] = []
    for turn in agent.memory.turns:
        msgs.append({
            "role": turn.role,
            "content": turn.content,
            "at": turn.at,
            "meta": turn.meta,
        })
    return {
        "messages": msgs,
        "summary": agent.memory.summary,
        "compressions": agent.memory.compressions,
    }


# ---------------------------------------------------------------------------
# media
# ---------------------------------------------------------------------------
ALLOWED_MEDIA_EXT = {
    ".mp4", ".webm", ".mov", ".mkv",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg",
    ".jpg", ".jpeg", ".png", ".webp",
}


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
