"""Assemble the AICut deep agent on the deepagents harness.

``create_deep_agent`` normally ships a built-in filesystem/shell/subagent tool
suite plus a summarization middleware. None of that is in the AICut training
distribution, so we register a dedicated ``HarnessProfile`` (keyed to the
``aicut`` provider that :class:`~aicut_agent.chat_model.AICutChatModel` reports)
that:

* strips every built-in tool (``ls/read_file/write_file/edit_file/glob/grep/
  execute/task``) — the model only ever sees the 12 training tools;
* disables the auto-added general-purpose subagent (removes the ``task`` tool);
* drops ``SummarizationMiddleware`` (it would otherwise call the model to
  summarise history — meaningless and harmful for a state-authority policy).

The result is a LangGraph ``CompiledStateGraph`` driven per-run.
"""

from __future__ import annotations

import logging
from typing import Any

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)

from aicut_agent.chat_model import AICutChatModel
from aicut_agent.middleware import AICutGuardrailMiddleware
from aicut_agent.tools import build_aicut_tools

logger = logging.getLogger("aicut_online.agent")

AICUT_PROVIDER = "aicut"

# Built-in tools that must not leak into the policy's action space.
_BUILTIN_TOOLS = frozenset({
    "ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute", "task",
})

_PROFILE_REGISTERED = False


def ensure_aicut_profile() -> None:
    """Register the AICut harness profile once (idempotent)."""
    global _PROFILE_REGISTERED
    if _PROFILE_REGISTERED:
        return
    register_harness_profile(
        AICUT_PROVIDER,
        HarnessProfile(
            excluded_tools=_BUILTIN_TOOLS,
            excluded_middleware=frozenset({"SummarizationMiddleware"}),
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        ),
    )
    _PROFILE_REGISTERED = True
    logger.info("已注册 aicut harness profile（剔除内置工具/subagent/summarization）")


def build_aicut_agent(
    runtime: Any,
    loader: Any,
    *,
    checkpointer: Any = None,
    recursion_limit: int | None = None,
    debug: bool = False,
) -> Any:
    """Build a fresh deep agent graph bound to one run's runtime.

    A new graph is compiled per run; the expensive model loader is shared. Graph
    compilation cost is negligible next to model inference.
    """
    ensure_aicut_profile()

    model = AICutChatModel()
    model.attach_loader(loader)
    model.attach_runtime(runtime)

    tools = build_aicut_tools(runtime)
    guardrails = AICutGuardrailMiddleware(runtime)

    # Bound the ReAct loop generously; the real cost ceiling is enforced by the
    # guardrail middleware (tool-call / repeat / Stage4 budgets) and the model's
    # own step counter, not by LangGraph's recursion limit.
    limit = recursion_limit or max(100, int(runtime.max_agent_steps) * 4 + 20)

    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=runtime.system_prompt,
        middleware=[guardrails],
        checkpointer=checkpointer,
        debug=debug,
        name="aicut_deep_agent",
    )
    # Recursion limit is applied via config at call time; stash the default.
    agent = agent.with_config({"recursion_limit": limit})
    return agent
