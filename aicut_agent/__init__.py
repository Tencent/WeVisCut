"""deepagents integration layer for the AICut RL policy.

Public surface:

* :class:`AICutChatModel` — LangChain ``BaseChatModel`` over the in-process vLLM
  loader, doing the LangChain↔training-format conversion.
* :class:`AICutRuntime` — per-run bridge owning the authoritative pipeline state.
* :class:`AICutGuardrailMiddleware` — the cost guardrails (ported from
  ``run_autonomous``).
* :func:`build_aicut_tools` — the 12 training tools as LangChain tools.
* :func:`build_aicut_agent` / :func:`ensure_aicut_profile` — assemble the graph.
"""

from aicut_agent.agent import build_aicut_agent, ensure_aicut_profile
from aicut_agent.chat_model import AICutChatModel
from aicut_agent.middleware import AICutGuardrailMiddleware
from aicut_agent.runner import create_runtime, run_aicut_generation
from aicut_agent.runtime import AICutRuntime
from aicut_agent.tools import build_aicut_tools

__all__ = [
    "AICutChatModel",
    "AICutRuntime",
    "AICutGuardrailMiddleware",
    "build_aicut_tools",
    "build_aicut_agent",
    "ensure_aicut_profile",
    "create_runtime",
    "run_aicut_generation",
]
