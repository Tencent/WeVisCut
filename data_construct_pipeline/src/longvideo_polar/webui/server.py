"""Control plane for the annotation campaign.

A 9k-episode batch runs for days across machines that come and go, so it needs an
operator surface rather than a terminal command: preview what a filter selects
before committing to it, add a machine by typing ``ip:port``, watch which stage
each live episode is on, and pause without losing the work already done.

The server owns exactly one campaign at a time. State lives in files under
``<output_root>/_state`` — the endpoint pool, the frozen session config, the status
document — so restarting this process does not lose the batch, and the batch does
not depend on a browser being open.

Run it with ``longvideo-polar ui``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from longvideo_polar.campaign import (
    MODE_LABELS_ZH,
    MODE_TOOL_SWITCHES,
    CampaignRunner,
    CampaignSettings,
)
from longvideo_polar.curation import (
    CurationFilter,
    build_worklist,
    read_worklist,
    summarize,
    write_worklist,
)
from longvideo_polar.endpoints import EndpointPool, normalize_base_url
from longvideo_polar.qc import export_dataset

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class EndpointBody(BaseModel):
    address: str = Field(..., description="ip:port, or a full base URL")
    model: str = ""
    api_key: str = "EMPTY"
    label: str = ""
    capacity: int = 4


class CapacityBody(BaseModel):
    address: str
    capacity: int


class ToggleBody(BaseModel):
    address: str
    enabled: bool


class SelectionBody(BaseModel):
    min_chars: int = 200
    max_chars: int = 20000
    languages: list[str] = Field(default_factory=lambda: ["zh", "zh-CN", "zh-Hans"])
    min_duration_sec: float = 0.0
    max_duration_sec: float = 0.0
    limit: int = 0
    stratify: bool = False
    seed: int = 42

    def to_filter(self) -> CurationFilter:
        return CurationFilter(
            min_chars=self.min_chars,
            max_chars=self.max_chars,
            languages=list(self.languages),
            min_duration_sec=self.min_duration_sec,
            max_duration_sec=self.max_duration_sec,
            limit=self.limit,
            stratify=self.stratify,
            seed=self.seed,
        )


class StartBody(BaseModel):
    mode: str = "text_only"
    profile: str = "all_qwen"
    max_steps: int = 48
    resume: bool = True
    retry_failed: bool = False
    limit: int = 0


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


class CampaignService:
    """Owns the pool, the worklist and at most one running campaign."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        output_root: Path,
        config_path: Path,
        gold_template_root: str = "",
    ) -> None:
        self.dataset_root = dataset_root
        self.output_root = output_root
        self.config_path = config_path
        self.gold_template_root = gold_template_root

        self.state_dir = output_root / "_state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.worklist_path = output_root / "worklist.jsonl"
        self.manifest_path = dataset_root / "manifest.jsonl"

        self.pool = EndpointPool(self.state_dir / "endpoints.json")
        self.runner: CampaignRunner | None = None
        self._task: asyncio.Task | None = None
        # Selection preview walks a 14k-row manifest; cache it so the UI's live
        # preview does not re-read the file on every keystroke.
        self._preview_cache: dict[str, Any] = {}
        self._lock = threading.Lock()

    # -- selection -------------------------------------------------------

    def preview(self, selection: CurationFilter) -> dict[str, Any]:
        key = json.dumps(selection.as_dict(), sort_keys=True)
        cached = self._preview_cache.get(key)
        if cached is not None:
            return cached
        items, rejected = build_worklist(self.manifest_path, self.dataset_root, selection)
        payload = {
            "selection": selection.as_dict(),
            "rejected": rejected,
            "stats": summarize(items),
            "preview": [item.as_dict() for item in items[:8]],
        }
        # One entry is enough: the UI only ever asks about the filter on screen.
        self._preview_cache = {key: payload}
        return payload

    def commit(self, selection: CurationFilter) -> dict[str, Any]:
        items, rejected = build_worklist(self.manifest_path, self.dataset_root, selection)
        if not items:
            raise ValueError("selection matched 0 episodes")
        write_worklist(
            self.output_root, items, selection,
            manifest_path=str(self.manifest_path), rejected=rejected,
        )
        return {"selected": len(items), "rejected": rejected, "stats": summarize(items)}

    def worklist_summary(self) -> dict[str, Any]:
        if not self.worklist_path.is_file():
            return {"exists": False}
        items, header = read_worklist(self.worklist_path)
        return {
            "exists": True,
            "count": len(items),
            "selection": header.get("selection", {}),
            "generated_at": header.get("generated_at", ""),
            "stats": summarize(items),
        }

    # -- campaign --------------------------------------------------------

    def build_runner(self, settings: CampaignSettings) -> CampaignRunner:
        runner = CampaignRunner(
            worklist_path=self.worklist_path,
            output_root=self.output_root,
            config_path=self.config_path,
            pool=self.pool,
            settings=settings,
        )
        runner.load()
        return runner

    async def start(self, settings: CampaignSettings) -> dict[str, Any]:
        with self._lock:
            if self._task is not None and not self._task.done():
                raise RuntimeError("a campaign is already running")
        if not self.worklist_path.is_file():
            raise RuntimeError("no worklist yet — apply a selection first")
        if not self.pool.healthy():
            await asyncio.to_thread(self.pool.probe_all)
        if not self.pool.healthy():
            raise RuntimeError("no healthy machine in the pool")

        runner = self.build_runner(settings)
        self.runner = runner
        self._task = asyncio.create_task(self._supervise(runner))
        return runner.status_document()

    async def _supervise(self, runner: CampaignRunner) -> None:
        try:
            await runner.run()
        except Exception:  # noqa: BLE001 - surfaced through status, never crashes the server
            logger.exception("campaign aborted")
            return
        try:
            await asyncio.to_thread(
                export_dataset,
                self.output_root,
                extra_provenance={
                    "dataset_root": str(self.dataset_root),
                    "gold_template_root": self.gold_template_root,
                    "config_path": str(self.config_path),
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("export failed; per-episode trajectories are unaffected")

    def status(self) -> dict[str, Any]:
        if self.runner is not None:
            return self.runner.status_document()
        # No campaign in this process: serve the last document written to disk so a
        # restarted server still shows where the batch got to.
        status_path = self.state_dir / "campaign_status.json"
        if status_path.is_file():
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
                payload["stale"] = True
                payload["pool"] = self.pool.describe()
                return payload
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "summary": {"phase": "idle", "total": 0, "settled": 0, "progress_pct": 0.0},
            "pool": self.pool.describe(),
            "episodes": [],
        }


def create_app(
    *,
    dataset_root: str | Path,
    output_root: str | Path,
    config_path: str | Path,
    gold_template_root: str = "",
    bootstrap_endpoints: list[str] | None = None,
) -> FastAPI:
    """Build the control-plane app.

    Args:
        dataset_root: Directory holding ``manifest.jsonl`` and ``scripts/``.
        output_root: Where the corpus is written.
        config_path: Base longvideo_polar YAML config.
        gold_template_root: Recorded in the dataset card for provenance.
        bootstrap_endpoints: ``ip:port`` machines registered on first start.
    """
    service = CampaignService(
        dataset_root=Path(str(dataset_root)).expanduser(),
        output_root=Path(str(output_root)).expanduser(),
        config_path=Path(str(config_path)).expanduser(),
        gold_template_root=gold_template_root,
    )
    for address in bootstrap_endpoints or []:
        try:
            service.pool.add(address, probe=False)
        except ValueError as exc:
            logger.warning("skipping bootstrap endpoint %r: %s", address, exc)

    app = FastAPI(title="WeDirector 批量轨迹标注", version="1.0")
    app.state.service = service

    # -- machines --------------------------------------------------------

    @app.get("/api/endpoints")
    async def list_endpoints() -> dict[str, Any]:
        return service.pool.describe()

    @app.post("/api/endpoints")
    async def add_endpoint(body: EndpointBody) -> dict[str, Any]:
        try:
            endpoint = await asyncio.to_thread(
                service.pool.add,
                body.address,
                model=body.model,
                api_key=body.api_key,
                label=body.label,
                capacity=body.capacity,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return endpoint.to_dict()

    @app.delete("/api/endpoints")
    async def remove_endpoint(address: str) -> dict[str, Any]:
        return {"removed": service.pool.remove(address)}

    @app.post("/api/endpoints/probe")
    async def probe_endpoints() -> dict[str, Any]:
        await asyncio.to_thread(service.pool.probe_all)
        return service.pool.describe()

    @app.post("/api/endpoints/toggle")
    async def toggle_endpoint(body: ToggleBody) -> dict[str, Any]:
        if not service.pool.set_enabled(body.address, body.enabled):
            raise HTTPException(status_code=404, detail="unknown endpoint")
        return service.pool.describe()

    @app.post("/api/endpoints/capacity")
    async def set_capacity(body: CapacityBody) -> dict[str, Any]:
        if not service.pool.set_capacity(body.address, body.capacity):
            raise HTTPException(status_code=404, detail="unknown endpoint")
        return service.pool.describe()

    # -- selection -------------------------------------------------------

    @app.get("/api/modes")
    async def list_modes() -> dict[str, Any]:
        return {
            "modes": [
                {
                    "id": mode,
                    "label": MODE_LABELS_ZH.get(mode, mode),
                    "tool_switches": switches,
                }
                for mode, switches in MODE_TOOL_SWITCHES.items()
            ]
        }

    @app.post("/api/selection/preview")
    async def preview_selection(body: SelectionBody) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(service.preview, body.to_filter())
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/selection/commit")
    async def commit_selection(body: SelectionBody) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(service.commit, body.to_filter())
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/worklist")
    async def worklist() -> dict[str, Any]:
        return await asyncio.to_thread(service.worklist_summary)

    # -- campaign --------------------------------------------------------

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return service.status()

    @app.post("/api/campaign/start")
    async def start_campaign(body: StartBody) -> dict[str, Any]:
        settings = CampaignSettings(
            mode=body.mode,
            profile=body.profile,
            max_steps=body.max_steps,
            resume=body.resume,
            retry_failed=body.retry_failed,
            limit=body.limit,
        )
        try:
            return await service.start(settings)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/campaign/pause")
    async def pause_campaign() -> dict[str, Any]:
        if service.runner is None:
            raise HTTPException(status_code=400, detail="no campaign running")
        service.runner.pause()
        return {"phase": service.runner.phase}

    @app.post("/api/campaign/resume")
    async def resume_campaign() -> dict[str, Any]:
        if service.runner is None:
            raise HTTPException(status_code=400, detail="no campaign running")
        service.runner.resume()
        return {"phase": service.runner.phase}

    @app.post("/api/campaign/stop")
    async def stop_campaign() -> dict[str, Any]:
        if service.runner is None:
            raise HTTPException(status_code=400, detail="no campaign running")
        service.runner.stop()
        return {"phase": service.runner.phase}

    @app.post("/api/export")
    async def export() -> dict[str, Any]:
        card = await asyncio.to_thread(
            export_dataset,
            service.output_root,
            extra_provenance={
                "dataset_root": str(service.dataset_root),
                "gold_template_root": service.gold_template_root,
                "config_path": str(service.config_path),
            },
        )
        return {"quality": card["quality"], "aggregate": card["aggregate"]}

    @app.get("/api/episode/{episode_id}/log")
    async def episode_log(episode_id: str, tail: int = 200) -> JSONResponse:
        # Constrain to the episodes directory: episode_id reaches this handler
        # straight from the URL.
        log_path = (service.output_root / "episodes" / episode_id / "worker.log").resolve()
        episodes_root = (service.output_root / "episodes").resolve()
        if not str(log_path).startswith(str(episodes_root)) or not log_path.is_file():
            raise HTTPException(status_code=404, detail="log not found")
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return JSONResponse({"lines": lines[-tail:]})

    # -- static ----------------------------------------------------------

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(str(STATIC_DIR / "index.html"))

    return app
