"""Expose the 12 AICut long-horizon tools as LangChain ``StructuredTool``s.

The tool *schema* shown to the model is NOT derived from these objects — the
``AICutChatModel`` asks ``model_loader._make_agent_tools`` to build it from
``tool_schema.LIGHT_TOOL_SCHEMAS`` / ``REQUIRED_TOOL_ARGS`` so it is byte-identical
to training. These LangChain tools exist so the harness can *execute* a call:
each one simply forwards the (lightly-typed) arguments to
``AICutRuntime.execute_tool``, which runs the real pipeline implementation via
``InferenceOrchestrator._dispatch_tool``.

Argument handling is deliberately permissive (``Any``, extra keys ignored) so the
harness never coerces or rejects a value the policy produced — the pipeline layer
does its own parsing/coercion, exactly as it does today.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import ConfigDict, create_model

from studio.config import install_import_paths

install_import_paths()

from server.infer_graph import LONG_HORIZON_ACTION_SPACE  # noqa: E402
from server.tool_schema import (  # noqa: E402
    LIGHT_TOOL_SCHEMAS,
    REQUIRED_TOOL_ARGS,
    TOOL_CARDS,
    TOOL_PRECONDITIONS,
)

logger = logging.getLogger("aicut_online.tools")


def _tool_description(name: str) -> str:
    """Match the description the model sees in training (card + precondition)."""
    card = TOOL_CARDS.get(name, "")
    precondition = TOOL_PRECONDITIONS.get(name, "")
    if card and precondition:
        return f"{card} 前置条件：{precondition}"
    return card or f"AICut long-horizon agent action: {name}."


def _build_args_schema(name: str) -> type:
    """Build a permissive pydantic model for the tool's light control params.

    Every field is ``Any`` with a ``None`` default so the harness performs no
    coercion; ``extra='ignore'`` silently drops anything outside the whitelist
    (the pipeline reads only known keys anyway).
    """
    props = LIGHT_TOOL_SCHEMAS.get(name, {})
    fields: dict[str, Any] = {key: (Any, None) for key in props}
    model = create_model(f"{name}_Args", **fields)  # type: ignore[call-overload]
    model.model_config = ConfigDict(extra="ignore")
    return model


def build_aicut_tools(runtime: Any) -> list[StructuredTool]:
    """Create the 12 LangChain tools bound to a per-run runtime."""
    tools: list[StructuredTool] = []
    for name in LONG_HORIZON_ACTION_SPACE:
        args_schema = _build_args_schema(name)

        async def _run(_tool_name: str = name, **kwargs: Any) -> str:
            # Drop unset (None) fields; forward the rest verbatim.
            args = {k: v for k, v in kwargs.items() if v is not None}
            output = await runtime.execute_tool(_tool_name, args)
            return runtime.observation_text(_tool_name, output)

        tool = StructuredTool.from_function(
            func=None,
            coroutine=_run,
            name=name,
            description=_tool_description(name),
            args_schema=args_schema,
        )
        tools.append(tool)
    return tools
