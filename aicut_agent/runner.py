"""Drive one AICut generation run on the deepagents harness.

This replaces the hand-rolled ``AutonomousOrchestrator.run_autonomous`` loop with
a ``create_deep_agent`` ReAct graph, while reusing the *same* orchestrator engine
for state management, tool dispatch, observation recording and preview building.

The function owns the run's lifecycle: build the runtime + agent, stream the loop
to completion (or to a guardrail/stitch termination), emit the closing events the
frontend expects, and persist ``final_state.json``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from studio.config import install_import_paths

install_import_paths()

from aicut_agent.agent import build_aicut_agent
from aicut_agent.runtime import AICutRuntime

logger = logging.getLogger("aicut_online.runner")


def create_runtime(
    loader: Any,
    *,
    infer_config: dict[str, Any],
    pipeline_config: dict[str, Any],
    stage_config: dict[str, Any],
    system_prompt: str,
    input_text: str,
    run_id: str,
    output_dir: str,
    track_order: list[str],
    broadcast_fn: Callable[..., Any],
    progress_fn: Callable[..., Any] | None = None,
) -> AICutRuntime:
    """Create the orchestrator engine + per-run runtime bridge."""
    from studio.orchestrator import AutonomousOrchestrator

    orchestrator = AutonomousOrchestrator(
        loader,
        pipeline_config,
        stage_config,
        system_prompt,
        broadcast_fn,
        infer_config,
        progress_fn=progress_fn,
    )
    runtime = AICutRuntime(
        orchestrator,
        input_script=input_text,
        run_id=run_id,
        output_dir=output_dir,
        track_order=track_order,
        broadcast_fn=broadcast_fn,
        progress_fn=progress_fn,
    )
    return runtime


async def run_aicut_generation(runtime: AICutRuntime, loader: Any) -> dict[str, Any]:
    """Run the deepagents ReAct loop to completion and return the run results.

    Raises ``asyncio.CancelledError`` if the user cancelled; other exceptions
    propagate after being surfaced as an ``error`` event.
    """
    from langchain_core.messages import HumanMessage

    rt = runtime
    agent = build_aicut_agent(rt, loader)

    # Unexpected exceptions propagate to the caller (studio.app sets status=error),
    # matching the original run_autonomous. Guardrail terminations are not
    # exceptions — they end the loop normally with the reason in results["errors"].
    # CancelledError propagates so the caller can mark the run cancelled.
    await agent.ainvoke({"messages": [HumanMessage(content=rt.input_script)]})

    # Closing events (mirror run_autonomous's tail).
    if rt.terminated_reason:
        await rt.broadcast({"type": "error", "stage": "agent_loop", "message": rt.terminated_reason})
    results = rt.finalize()
    await rt.emit_progress("done")
    await rt.broadcast({
        "type": "pipeline_complete",
        "status": "completed" if not results["errors"] else "completed_with_errors",
        "results": results,
        "preview": rt.orchestrator._build_preview(rt.state),
    })
    return results
