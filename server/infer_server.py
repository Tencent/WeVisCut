"""FastAPI server for AICut Inference Visualization.

Loads the fine-tuned LoRA model once at startup, then serves:
  - WebSocket real-time streaming of pipeline execution
  - REST endpoints for inference control and preview data
  - Static frontend files
  - Media file serving (video/audio)
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("aicut_infer")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
INFER_VIS_DIR = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = INFER_VIS_DIR.parent
STATIC_DIR = INFER_VIS_DIR / "static"

# Add pipeline root to sys.path so we can import pipeline modules
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
_config_path = Path(
    os.environ.get("AICUT_INFER_CONFIG", str(INFER_VIS_DIR / "infer_config.yaml"))
).resolve()


def _load_infer_config(path: Path | None = None) -> dict[str, Any]:
    p = path or _config_path
    if p.exists():
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {}


def _apply_model_profile(infer_cfg: dict[str, Any]) -> dict[str, Any]:
    """Overlay model.profiles[active_profile] onto model config."""
    if not isinstance(infer_cfg, dict):
        return infer_cfg
    model_cfg = infer_cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        return infer_cfg

    profiles = model_cfg.get("profiles", {})
    if not isinstance(profiles, dict) or not profiles:
        return infer_cfg

    env_profile = os.environ.get("AICUT_MODEL_PROFILE", "").strip()
    active = env_profile or str(model_cfg.get("active_profile", "") or "").strip()
    if not active:
        return infer_cfg

    profile_cfg = profiles.get(active)
    if not isinstance(profile_cfg, dict):
        logger.warning(
            "Model profile %r not found. Available profiles: %s",
            active,
            ", ".join(profiles.keys()),
        )
        return infer_cfg

    merged_model_cfg = dict(model_cfg)
    merged_model_cfg.update(profile_cfg)
    merged_model_cfg["profiles"] = profiles
    merged_model_cfg["active_profile"] = active

    resolved = dict(infer_cfg)
    resolved["model"] = merged_model_cfg
    logger.info(
        "Using model profile %s: backend=%s base=%s adapter=%s",
        active,
        merged_model_cfg.get("infer_backend", "transformers"),
        merged_model_cfg.get("base_model", ""),
        merged_model_cfg.get("adapter", "") or "-",
    )
    return resolved


def _load_pipeline_config(infer_cfg: dict[str, Any]) -> dict[str, Any]:
    rel = infer_cfg.get("pipeline_config", "../config.yaml")
    cfg_path = (INFER_VIS_DIR / rel).resolve()
    if cfg_path.exists():
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return {}


def _load_system_prompt(infer_cfg: dict[str, Any]) -> str:
    rel = infer_cfg.get("system_prompt_file", "../preprocess_ori_data/configs/sft_system_prompt.txt")
    path = (INFER_VIS_DIR / rel).resolve()
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def _load_stage_config(infer_cfg: dict[str, Any]) -> dict[str, Any]:
    rel = infer_cfg.get("stage_config_file", "../preprocess_ori_data/configs/stage_allowed_tools.yaml")
    path = (INFER_VIS_DIR / rel).resolve()
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
infer_config: dict[str, Any] = {}
pipeline_config: dict[str, Any] = {}
system_prompt: str = ""
stage_config: dict[str, Any] = {}
model_loader = None

active_runs: dict[str, dict[str, Any]] = {}
active_orchestrators: dict[str, Any] = {}  # run_id -> InferenceOrchestrator
ws_connections: dict[str, list[WebSocket]] = {}
model_reload_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="AICut Inference Visualization", version="1.0.0")


@app.on_event("startup")
async def startup_event():
    """Load model and configs at server startup."""
    global infer_config, pipeline_config, system_prompt, stage_config, model_loader

    logger.info("Loading configurations...")
    infer_config = _load_infer_config()
    infer_config = _apply_model_profile(infer_config)
    pipeline_config = _load_pipeline_config(infer_config)
    system_prompt = _load_system_prompt(infer_config)
    stage_config = _load_stage_config(infer_config)

    pipeline_config.setdefault("execution", {})["mode"] = "real"

    logger.info("System prompt loaded: %d chars", len(system_prompt))
    logger.info("Stage config: %d stages", len(stage_config.get("allowed_tools", {})))

    from server.model_loader import UnifiedModelLoader

    model_loader = UnifiedModelLoader(infer_config)
    logger.info("Loading unified single model (30-60s)...")
    model_loader.load(orchestrator_system_prompt=system_prompt)
    logger.info("Unified model ready!")

    # Override pipeline config LLM to point to our local /v1 endpoint.
    # Prefer the actual uvicorn port exported by run_infer_vis.sh over YAML,
    # otherwise a CLI --port override can accidentally point tools at another
    # process on the YAML port.
    server_cfg = infer_config.setdefault("server", {})
    local_port = int(os.environ.get("AICUT_INFER_PORT") or server_cfg.get("port", 8090))
    server_cfg["port"] = local_port
    local_llm_url = os.environ.get("AICUT_LOCAL_LLM_BASE_URL") or f"http://127.0.0.1:{local_port}/v1"
    pipeline_config["llm"] = {
        "base_url": local_llm_url,
        "endpoint": "/chat/completions",
        "api_key": "local",
        "model": "local-qwen3.5-9b-unified-v5",
        "temperature": 0.2,
        "max_tokens": 2048,
        "timeout": 300,
    }
    for svc, svc_cfg in list((pipeline_config.get("services", {}) or {}).items()):
        if not isinstance(svc_cfg, dict):
            continue
        is_llm_service = svc in {"qwen", "gemini"} or "model" in svc_cfg or "api_type" in svc_cfg
        if not is_llm_service:
            continue
        svc_cfg["base_url"] = local_llm_url
        svc_cfg["endpoint"] = "/chat/completions"
        svc_cfg["api_key"] = "local"
        svc_cfg["model"] = "local-qwen3.5-9b-unified-v5"
        svc_cfg["offline"] = False

    switches = infer_config.get("tool_switches", {}) or {}
    pipeline_config.setdefault("tts", {})["service_url"] = infer_config.get(
        "tts_service_url",
        pipeline_config.get("tts", {}).get("service_url", "http://127.0.0.1:7865"),
    )
    pipeline_config.setdefault("music", {})["enabled"] = bool(switches.get("bgm_search", True))
    hf_accel = (
        pipeline_config.setdefault("services", {})
        .setdefault("hyperframes", {})
        .setdefault("accelerated_render", {})
    )
    hf_accel["enabled"] = bool(switches.get("hyperframes_render", True))
    fake_mode = pipeline_config.setdefault("execution", {}).setdefault("fake_mode", {})
    fake_mode["skip_final_stitch"] = not bool(switches.get("final_stitch", True))
    fake_mode["final_hyperframes_render"] = bool(switches.get("hyperframes_render", True))
    fake_mode["skip_hyperframes_render"] = not bool(switches.get("hyperframes_render", True))

    # render-split transparent overlay acceleration
    render_split_cfg = infer_config.get("render_split", {}) or {}
    if render_split_cfg.get("enabled"):
        pipeline_config.setdefault("render_split", {}).update(render_split_cfg)
        try:
            from server.render_split import install as install_render_split
            if install_render_split(pipeline_config):
                logger.info("render-split overlay pipeline active")
        except Exception as exc:
            logger.warning("render-split install skipped: %s", exc)

    logger.info("Pipeline LLM overridden to local endpoint: %s", local_llm_url)


@app.on_event("shutdown")
async def shutdown_event():
    """Release model resources and terminate multi-model child processes."""
    global model_loader
    try:
        if model_loader is not None:
            if hasattr(model_loader, "unload"):
                model_loader.unload()
            elif hasattr(model_loader, "shutdown"):
                model_loader.shutdown()
            logger.info("Model loader released on shutdown.")
    except Exception:
        logger.exception("Failed to release model loader on shutdown.")


# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class InferStartRequest(BaseModel):
    input_text: str = ""
    config_overrides: dict[str, Any] | None = None


class ModelProfileRequest(BaseModel):
    profile: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/agent")
@app.get("/agent/")
async def agent_index():
    return FileResponse(str(STATIC_DIR / "agent.html"))


@app.get("/api/status")
async def get_status():
    """Server and model status."""
    model_cfg = infer_config.get("model", {})
    model_cfg = model_cfg if isinstance(model_cfg, dict) else {}
    profiles = model_cfg.get("profiles", {})
    profiles = profiles if isinstance(profiles, dict) else {}
    deployment_mode = str(infer_config.get("deployment_mode", "single_model"))

    status = {
        "model_loaded": model_loader is not None and model_loader.is_loaded,
        "deployment_mode": deployment_mode,
        "model_config": model_cfg,
        "active_model_profile": model_cfg.get("active_profile", ""),
        "available_model_profiles": list(profiles.keys()),
        "model_backend": model_cfg.get("infer_backend", "transformers"),
        "agent_mode": infer_config.get("agent_mode", "tool_call"),
        "system_prompt_file": infer_config.get("system_prompt_file", ""),
        "active_runs": len(active_runs),
        "pipeline_llm": pipeline_config.get("llm", {}).get("model", "N/A"),
    }
    return status


def _busy_run_count() -> int:
    done = {"completed", "completed_with_errors", "cancelled", "error"}
    return sum(1 for run in active_runs.values() if run.get("status") not in done)


@app.post("/api/model/profile")
async def switch_model_profile(req: ModelProfileRequest):
    """Reload the local model with another configured profile."""
    global infer_config, model_loader

    profile = str(req.profile or "").strip()
    raw_config = _load_infer_config()
    raw_model_cfg = raw_config.get("model", {})
    profiles = (
        raw_model_cfg.get("profiles", {})
        if isinstance(raw_model_cfg, dict)
        else {}
    )
    if not profile or not isinstance(profiles, dict) or profile not in profiles:
        raise HTTPException(
            status_code=400,
            detail={"error": "unknown_profile", "available_profiles": list(profiles.keys()) if isinstance(profiles, dict) else []},
        )

    busy = _busy_run_count()
    if busy > 0:
        raise HTTPException(
            status_code=409,
            detail={"error": "active_runs_in_progress", "active_runs": busy},
        )

    async with model_reload_lock:
        current = str((infer_config.get("model", {}) or {}).get("active_profile", "") or "")
        if current == profile and model_loader is not None and model_loader.is_loaded:
            return await get_status()

        raw_model_cfg["active_profile"] = profile
        new_config = _apply_model_profile(raw_config)

        old_loader = model_loader
        if old_loader is not None:
            logger.info("Unloading current model profile before switching to %s", profile)
            await asyncio.to_thread(old_loader.unload)

        from server.model_loader import UnifiedModelLoader

        new_loader = UnifiedModelLoader(new_config)
        infer_config = new_config
        model_loader = new_loader
        try:
            logger.info("Reloading model profile %s...", profile)
            await asyncio.to_thread(lambda: new_loader.load(orchestrator_system_prompt=system_prompt))
        except Exception:
            logger.exception("Failed to load model profile %s", profile)
            model_loader = None
            raise

        return await get_status()


@app.post("/api/infer/start")
async def start_inference(req: InferStartRequest):
    """Start a new inference run with the given input script."""
    if not req.input_text.strip():
        return {"error": "input_text is required"}
    if model_loader is None or not model_loader.is_loaded:
        return {"error": "Model not loaded yet"}

    run_id = f"infer_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"
    active_runs[run_id] = {
        "id": run_id,
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "started_monotonic": time.monotonic(),
        "input_text": req.input_text,
        "trajectory": [],
        "console_logs": [],
        "stages": [],
        "agent_events": [],
        "preview": {},
        "errors": [],
        "result": None,
    }

    asyncio.create_task(_run_inference(run_id, req.input_text))
    return {"run_id": run_id, "status": "started"}


@app.get("/api/infer/{run_id}")
async def get_run(run_id: str):
    """Get current run state."""
    run = active_runs.get(run_id)
    if not run:
        return {"error": "Run not found"}
    return {
        "id": run["id"],
        "status": run["status"],
        "started_at": run["started_at"],
        "elapsed_seconds": round(time.monotonic() - run["started_monotonic"], 2),
        "stages": run["stages"],
        "errors": run["errors"],
    }


@app.get("/api/infer/{run_id}/result")
async def get_run_result(run_id: str):
    """
    Return the full pipeline result dict for a finished run.

    Added for batch evaluation: `/api/infer/{run_id}` intentionally returns only
    a light status payload for the UI's polling loop, so the actual result
    (project_id, output_dir, final_video_path, stages_completed, errors) had no
    way out of the process. Batch drivers need those to locate artifacts on disk.

    Read-only: it just exposes what `_run_inference` already stored.
    """
    run = active_runs.get(run_id)
    if not run:
        return {"error": "Run not found"}
    return {
        "id": run["id"],
        "status": run["status"],
        "elapsed_seconds": round(time.monotonic() - run["started_monotonic"], 2),
        "result": run.get("result"),
        "errors": run.get("errors", []),
        "stages": run.get("stages", []),
    }


@app.get("/api/infer/{run_id}/preview")
async def get_preview(run_id: str):
    """Get 5-track timeline preview data."""
    run = active_runs.get(run_id)
    if not run:
        return {"error": "Run not found"}
    return run.get("preview", {})


@app.post("/api/infer/{run_id}/pause")
async def pause_run(run_id: str):
    """Pause a running inference pipeline."""
    orch = active_orchestrators.get(run_id)
    if not orch:
        return {"error": "Run not found or already finished"}
    orch.pause()
    run = active_runs.get(run_id, {})
    run["status"] = "paused"
    await _broadcast(run_id, {"type": "run_paused"})
    return {"status": "paused"}


@app.post("/api/infer/{run_id}/resume")
async def resume_run(run_id: str):
    """Resume a paused inference pipeline."""
    orch = active_orchestrators.get(run_id)
    if not orch:
        return {"error": "Run not found or already finished"}
    orch.resume()
    run = active_runs.get(run_id, {})
    run["status"] = "running"
    await _broadcast(run_id, {"type": "run_resumed"})
    return {"status": "running"}


@app.post("/api/infer/{run_id}/cancel")
async def cancel_run(run_id: str):
    """Cancel a running (or paused) inference pipeline."""
    orch = active_orchestrators.get(run_id)
    if not orch:
        return {"error": "Run not found or already finished"}
    orch.cancel()
    run = active_runs.get(run_id, {})
    run["status"] = "cancelling"
    await _broadcast(run_id, {
        "type": "console_log",
        "text": "取消请求已发送，当前步骤结束后会停止",
        "level": "warn",
    })
    return {"status": "cancelling"}


@app.get("/api/media")
async def serve_media(path: str):
    """Serve video/audio files for in-browser playback."""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return {"error": "File not found"}

    allowed_ext = {".mp4", ".webm", ".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac", ".jpg", ".png"}
    if file_path.suffix.lower() not in allowed_ext:
        return {"error": "Unsupported media type"}

    mime, _ = mimetypes.guess_type(str(file_path))
    if not mime:
        mime = "application/octet-stream"

    return FileResponse(str(file_path), media_type=mime, headers={"Accept-Ranges": "bytes"})


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[dict[str, Any]] = []
    temperature: float = 0.2
    max_tokens: int = 2048
    top_p: float = 1.0
    stream: bool = False


def _message_content_to_text(content: Any) -> str:
    """Convert OpenAI text/multimodal content into a text-only prompt."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
                elif item.get("type"):
                    parts.append(f"[{item.get('type')}]")
            elif item is not None:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def local_chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions using the loaded local model.

    This endpoint is called internally by pipeline tools (LLMClient)
    so they can use the local LoRA model instead of a remote API.
    """
    if model_loader is None or not model_loader.is_loaded:
        return {"error": "Model not loaded"}

    system_msg = ""
    user_msg = ""
    for m in req.messages:
        if m.get("role") == "system":
            system_msg = _message_content_to_text(m.get("content", ""))
        elif m.get("role") == "user":
            user_msg = _message_content_to_text(m.get("content", ""))

    is_multi = hasattr(model_loader, "predict_orchestrator")
    if is_multi:
        prediction = await asyncio.to_thread(
            model_loader.predict_orchestrator,
            system_msg,
            user_msg,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
    else:
        prediction = await asyncio.to_thread(
            model_loader.predict,
            system_msg,
            user_msg,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )

    return {
        "id": f"chatcmpl-local-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "model": req.model or "local-qwen3.5-9b-lora",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": prediction.cleaned_output,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prediction.prompt_tokens,
            "completion_tokens": prediction.completion_tokens,
            "total_tokens": prediction.prompt_tokens + prediction.completion_tokens,
        },
    }


@app.get("/api/runs")
async def list_runs():
    """List all inference runs."""
    return {
        "runs": [
            {
                "id": r["id"],
                "status": r["status"],
                "started_at": r["started_at"],
                "elapsed_seconds": round(time.monotonic() - r["started_monotonic"], 2),
                "title": (r.get("input_text") or "")[:60].split("\n")[0] or "(无标题)",
                "input_text_preview": (r.get("input_text") or "")[:120],
            }
            for r in sorted(
                active_runs.values(),
                key=lambda x: x.get("started_at", ""),
                reverse=True,
            )
        ]
    }


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str):
    """Delete a completed run from history."""
    run = active_runs.get(run_id)
    if not run:
        return {"error": "Run not found"}
    if run["status"] in ("running", "paused"):
        return {"error": "Cannot delete a running or paused run. Cancel it first."}
    del active_runs[run_id]
    if run_id in active_orchestrators:
        del active_orchestrators[run_id]
    if run_id in ws_connections:
        del ws_connections[run_id]
    return {"status": "deleted", "id": run_id}


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------
@app.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    """Real-time streaming for inference progress."""
    await websocket.accept()
    ws_connections.setdefault(run_id, []).append(websocket)

    try:
        # Send existing data on reconnect
        run = active_runs.get(run_id)
        if run:
            for log in run.get("console_logs", []):
                await websocket.send_json({"type": "console_log", **log})
            for stage in run.get("stages", []):
                await websocket.send_json({"type": "stage_update", **stage})
            for event in run.get("agent_events", []):
                await websocket.send_json(event)
            if run.get("preview"):
                await websocket.send_json({"type": "preview_update", "preview": run["preview"]})
            if run["status"] in ("completed", "completed_with_errors", "cancelled", "error"):
                await websocket.send_json({
                    "type": "run_complete",
                    "status": run["status"],
                    "elapsed_seconds": round(time.monotonic() - run["started_monotonic"], 2),
                    "preview": run.get("preview", {}),
                })

        while True:
            data = await websocket.receive_text()
            # Client can send pings or commands
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        pass
    finally:
        if run_id in ws_connections:
            ws_connections[run_id] = [
                ws for ws in ws_connections[run_id] if ws != websocket
            ]


# ---------------------------------------------------------------------------
# Broadcast helpers
# ---------------------------------------------------------------------------
async def _broadcast(run_id: str, message: dict[str, Any]):
    """Broadcast message to all WebSocket clients for a run."""
    # Store in run state
    run = active_runs.get(run_id)
    if run:
        msg_type = message.get("type", "")

        if msg_type == "console_log":
            run.setdefault("console_logs", []).append({
                "text": message.get("text", ""),
                "level": message.get("level", "stdout"),
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })

        if msg_type == "stage_update":
            run.setdefault("stages", []).append(message)

        if msg_type in {"thinking", "tool_call", "tool_result", "error", "pipeline_complete"}:
            run.setdefault("agent_events", []).append(message)

        if msg_type == "preview_update":
            run["preview"] = message.get("preview", {})

        if msg_type == "error":
            run.setdefault("errors", []).append(message.get("message", ""))

    conns = ws_connections.get(run_id, [])
    for ws in list(conns):
        try:
            await ws.send_json(message)
        except Exception:
            conns.remove(ws)


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
async def _run_inference(run_id: str, input_text: str):
    """Execute the inference pipeline."""
    from server.infer_graph import InferenceOrchestrator

    run = active_runs[run_id]
    output_dir = str(
        Path(pipeline_config.get("paths", {}).get("output_root", "/tmp/aicut_infer"))
        / "infer_vis"
    )

    await _broadcast(run_id, {
        "type": "console_log",
        "text": "初始化推理管线...",
        "level": "system",
    })
    await _broadcast(run_id, {
        "type": "console_log",
        "text": f"部署模式: 统一单模型 (Unified v5)",
        "level": "system",
    })
    await _broadcast(run_id, {
        "type": "console_log",
        "text": f"模型: {infer_config.get('model', {}).get('base_model', 'N/A')}",
        "level": "system",
    })
    await _broadcast(run_id, {
        "type": "console_log",
        "text": f"适配器: {infer_config.get('model', {}).get('adapter', 'N/A')}",
        "level": "system",
    })

    switches = infer_config.get("tool_switches", {})
    sw_labels = {
        "tts": "TTS语音合成",
        "visual_retrieval": "视频素材检索",
        "bgm_search": "配乐检索",
        "timeline_compose": "时间轴渲染",
        "hyperframes_render": "HyperFrames",
        "final_stitch": "最终拼接",
    }
    sw_parts = [
        f"{label}: {'✓' if switches.get(key, False) else '✗'}"
        for key, label in sw_labels.items()
    ]
    await _broadcast(run_id, {
        "type": "console_log",
        "text": f"工具开关 → {' | '.join(sw_parts)}",
        "level": "system",
    })

    orchestrator = InferenceOrchestrator(
        model_loader=model_loader,
        pipeline_config=pipeline_config,
        stage_config=stage_config,
        system_prompt=system_prompt,
        broadcast_fn=_broadcast,
        infer_config=infer_config,
    )
    active_orchestrators[run_id] = orchestrator

    try:
        result = await orchestrator.run_pipeline(input_text, run_id, output_dir)
        if not result.get("final_video_path"):
            result.setdefault("errors", []).append("最终视频未生成，请查看拼接 manifest 或 validation_errors")
        run["result"] = result
        run["status"] = "completed" if not result.get("errors") else "completed_with_errors"
    except asyncio.CancelledError:
        run["status"] = "cancelled"
        await _broadcast(run_id, {
            "type": "console_log",
            "text": "管线已被用户取消",
            "level": "warn",
        })
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {str(exc)[:500]}"
        logger.exception("Inference pipeline failed")
        run["status"] = "error"
        run["errors"].append(err_msg)
        await _broadcast(run_id, {
            "type": "console_log",
            "text": f"管线异常: {err_msg}",
            "level": "stderr",
        })
        await _broadcast(run_id, {
            "type": "console_log",
            "text": traceback.format_exc()[-2000:],
            "level": "stderr",
        })
    finally:
        active_orchestrators.pop(run_id, None)

    elapsed = round(time.monotonic() - run["started_monotonic"], 2)
    await _broadcast(run_id, {
        "type": "console_log",
        "text": f"推理完成 — 状态: {run['status']}, 耗时: {elapsed}s",
        "level": "success" if run["status"] == "completed" else "stderr",
    })
    await _broadcast(run_id, {
        "type": "run_complete",
        "status": run["status"],
        "elapsed_seconds": elapsed,
        "preview": run.get("preview", {}),
    })


# ---------------------------------------------------------------------------
# Config override endpoint for setting config path at startup via env
# ---------------------------------------------------------------------------
def set_config_path(path: str):
    global _config_path
    _config_path = Path(path).resolve()
