"""``AICutChatModel`` — a LangChain ``BaseChatModel`` over the in-process vLLM policy.

This is the piece that lets the RL-trained 9B policy plug into the deepagents /
LangGraph harness *without* changing what the model sees. The harness drives a
multi-turn message list; the policy was trained on single-turn calls where the
whole world is serialised into one user JSON. The conversion therefore is:

* **LangChain → training (input):** ignore the accumulated chat messages and
  rebuild the user payload from the authoritative runtime state via
  ``build_agentic_state_json`` (the runtime *is* the state authority — see
  ``agent_contract.runtime_is_state_authority``). The system prompt is the exact
  training prompt held by the runtime, not whatever the harness assembled.
* **training → LangChain (output):** parse the model's ``<think> + tool_call``
  output into an ``AIMessage`` carrying a single structured ``tool_calls`` entry.

Parse failures and unknown-tool requests are handled *inside* ``_generate`` by
recording the observation and re-predicting (up to the configured ceiling), so
every ``AIMessage`` handed back to the ReAct loop is either a valid tool call or
a terminal message — exactly the self-correction loop the policy was trained
with.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Iterator, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import ConfigDict, Field, PrivateAttr

from studio.config import install_import_paths

install_import_paths()

from server.infer_graph import (  # noqa: E402
    LONG_HORIZON_ACTION_SPACE,
    LONG_HORIZON_TOOL_NAME_ALIASES,
)

logger = logging.getLogger("aicut_online.chat_model")


class AICutChatModel(BaseChatModel):
    """Wraps ``UnifiedModelLoader`` as a LangChain chat model.

    Two modes:

    * **agent mode** (a runtime is attached): rebuild the training user-JSON from
      runtime state and return a tool call. Used by the generation loop.
    * **plain mode** (no runtime): a plain ``(system, user) -> text`` completion.
      Used by the conversational editor and the ``/v1/chat/completions`` shim.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Serializable configuration.
    model_name: str = "aicut-9b"
    system_prompt_text: str = ""
    allowed_tool_names: list[str] = Field(
        default_factory=lambda: list(LONG_HORIZON_ACTION_SPACE)
    )
    max_consecutive_parse_failures: int = 4
    default_max_tokens: int = 24576
    default_temperature: float = 0.0

    # Heavy / non-serializable objects.
    _loader: Any = PrivateAttr(default=None)
    _runtime: Any = PrivateAttr(default=None)
    _bound_tool_names: list[str] = PrivateAttr(default_factory=list)

    # ------------------------------------------------------------------
    # wiring
    # ------------------------------------------------------------------
    def attach_loader(self, loader: Any) -> "AICutChatModel":
        self._loader = loader
        return self

    def attach_runtime(self, runtime: Any) -> "AICutChatModel":
        """Bind a per-run runtime (agent mode)."""
        self._runtime = runtime
        if runtime is not None:
            self.system_prompt_text = getattr(runtime.orchestrator, "system_prompt", "") or self.system_prompt_text
            self.max_consecutive_parse_failures = int(
                getattr(runtime, "max_consecutive_parse_failures", 4)
            )
        return self

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        """Record the bound tool names and return self.

        The schema actually sent to the model is built by
        ``model_loader._make_agent_tools`` from the canonical
        ``LONG_HORIZON_ACTION_SPACE`` — not from these LangChain tool objects —
        so the model sees byte-identical tool definitions to training.
        """
        names: list[str] = []
        for t in tools:
            name = getattr(t, "name", None)
            if name is None and isinstance(t, dict):
                name = t.get("name") or (t.get("function") or {}).get("name")
            if name:
                names.append(str(name))
        self._bound_tool_names = names
        return self

    @property
    def _llm_type(self) -> str:
        return "aicut-chat-model"

    def _get_ls_params(self, stop: Any = None, **kwargs: Any) -> dict[str, Any]:
        """Report a stable provider so the AICut harness profile resolves.

        deepagents selects its ``HarnessProfile`` (which strips the built-in
        filesystem/shell/task tools and the summarization middleware) by the
        model's provider. Reporting ``aicut`` lets ``agent.py`` register that
        profile under a deterministic key.
        """
        return {"ls_provider": "aicut", "ls_model_name": "aicut-9b", "ls_model_type": "chat"}

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------
    def _require_loader(self) -> Any:
        loader = self._loader
        if loader is None or not bool(getattr(loader, "is_loaded", False)):
            raise RuntimeError("AICut 模型尚未加载完成")
        return loader

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._runtime is None:
            return self._generate_plain(messages, stop=stop, **kwargs)
        return self._generate_agent(messages, stop=stop, run_manager=run_manager, **kwargs)

    # ------------------------------------------------------------------
    # agent mode (tool-calling, training-format)
    # ------------------------------------------------------------------
    def _generate_agent(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        loader = self._require_loader()
        runtime = self._runtime

        # Backstop terminator: if a guardrail or a successful stitch already ended
        # the run, return a terminal message (no tool call) without spending another
        # model call. The middleware normally ends the loop via Command(goto=END);
        # this covers the case where the loop still comes back to the model.
        if runtime.stitch_succeeded or runtime.terminated_reason:
            reason = runtime.terminated_reason or "模型已完成收尾"
            return self._terminal_chat_result(reason, None)

        runtime.step_idx += 1
        runtime.results["steps_used"] = runtime.step_idx

        # Step ceiling (mirrors run_autonomous's for-else branch).
        if runtime.step_idx > runtime.max_agent_steps:
            runtime.terminated_reason = (
                f"达到步数上限 max_steps={runtime.max_agent_steps}，模型未完成收尾"
            )
            return self._terminal_chat_result(runtime.terminated_reason, None)

        allowed = list(self.allowed_tool_names)
        last_prediction: Any = None
        for _attempt in range(1, self.max_consecutive_parse_failures + 1):
            user_payload = runtime.build_user_payload()
            prediction = loader.predict_orchestrator(
                self.system_prompt_text,
                user_payload,
                stage="agent_loop",
                allowed_tools=allowed,
            )
            last_prediction = prediction

            raw_tool = str(getattr(prediction, "tool_name", "") or "")
            tool_name = LONG_HORIZON_TOOL_NAME_ALIASES.get(raw_tool, raw_tool)
            parse_ok = bool(getattr(prediction, "parse_ok", False))
            unknown = parse_ok and tool_name not in allowed

            if parse_ok and not unknown:
                runtime.reset_parse_failures()
                # Stash the prediction so the async middleware can broadcast the
                # "thinking" event (this sync method runs in a worker thread).
                runtime.last_prediction = prediction
                runtime.last_tool_name = tool_name
                return self._agent_chat_result(prediction, tool_name)

            # Invalid / unknown: record the observation and re-predict.
            should_stop = runtime.record_parse_failure(raw_tool, prediction, unknown=unknown)
            logger.info(
                "agent step %s 输出无法解析/未知工具 (attempt %s): raw=%r",
                runtime.step_idx, runtime.consecutive_parse_failures, raw_tool,
            )
            if should_stop:
                runtime.terminated_reason = (
                    f"模型连续 {runtime.consecutive_parse_failures} 次输出无法解析或请求未知工具，停止执行"
                )
                return self._terminal_chat_result(runtime.terminated_reason, last_prediction)

        # Defensive: the loop above always returns, but never emit an invalid call.
        runtime.terminated_reason = "模型输出无法解析，停止执行"
        return self._terminal_chat_result(runtime.terminated_reason, last_prediction)

    def _agent_chat_result(self, prediction: Any, tool_name: str) -> ChatResult:
        args = getattr(prediction, "arguments", {})
        if not isinstance(args, dict):
            args = {}
        tool_call = {
            "name": tool_name,
            "args": args,
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "tool_call",
        }
        message = AIMessage(
            content=str(getattr(prediction, "thinking_summary", "") or ""),
            tool_calls=[tool_call],
            response_metadata={
                "raw_output": getattr(prediction, "raw_output", ""),
                "cleaned_output": getattr(prediction, "cleaned_output", ""),
                "thinking_summary": getattr(prediction, "thinking_summary", ""),
                "latency_ms": getattr(prediction, "latency_ms", 0.0),
                "prompt_tokens": getattr(prediction, "prompt_tokens", 0),
                "completion_tokens": getattr(prediction, "completion_tokens", 0),
                "tokens_per_sec": getattr(prediction, "tokens_per_sec", 0.0),
                "agent_trace": getattr(prediction, "agent_trace", {}),
                "tool_name": tool_name,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _terminal_chat_result(self, reason: str, prediction: Any) -> ChatResult:
        """A final assistant message with no tool calls — this ends the ReAct loop."""
        message = AIMessage(
            content=reason,
            tool_calls=[],
            response_metadata={
                "terminated": True,
                "reason": reason,
                "raw_output": getattr(prediction, "raw_output", "") if prediction else "",
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    # ------------------------------------------------------------------
    # plain mode (single-turn text completion)
    # ------------------------------------------------------------------
    def _generate_plain(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        loader = self._require_loader()
        system = ""
        user_parts: list[str] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system = self._text_of(msg)
            else:
                user_parts.append(self._text_of(msg))
        user = "\n".join(p for p in user_parts if p)

        max_tokens = int(kwargs.get("max_tokens") or self.default_max_tokens)
        temperature = float(kwargs.get("temperature") if kwargs.get("temperature") is not None else self.default_temperature)
        prediction = loader.predict(
            system, user, max_tokens=max_tokens, temperature=temperature,
        )
        text = str(getattr(prediction, "cleaned_output", "") or "")
        message = AIMessage(
            content=text,
            response_metadata={
                "raw_output": getattr(prediction, "raw_output", ""),
                "prompt_tokens": getattr(prediction, "prompt_tokens", 0),
                "completion_tokens": getattr(prediction, "completion_tokens", 0),
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    @staticmethod
    def _text_of(message: BaseMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content
        parts: list[str] = []
        for item in content if isinstance(content, list) else []:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)

    def generate_text(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: float = 1.0,
    ) -> str:
        """Direct plain-mode ``(system, user) -> text`` completion.

        Used by the conversational editor and the ``/v1/chat/completions`` shim —
        the same weights serve both the agent loop and these single-turn calls.
        """
        loader = self._require_loader()
        fn = getattr(loader, "predict_orchestrator", None) or loader.predict
        prediction = fn(system, user, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        return str(getattr(prediction, "cleaned_output", "") or "")

    # ------------------------------------------------------------------
    # streaming (optional): emit the whole message as one chunk
    # ------------------------------------------------------------------
    def _stream(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        from langchain_core.outputs import ChatGenerationChunk
        from langchain_core.messages import AIMessageChunk

        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        message = result.generations[0].message
        if not isinstance(message, AIMessage):
            raise TypeError("expected AIMessage")
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(
                content=message.content if isinstance(message.content, str) else "",
                tool_calls=getattr(message, "tool_calls", []),
                id=message.id,
                chunk_position="last",
            )
        )
        if run_manager:
            run_manager.on_llm_new_token("", chunk=chunk)
        yield chunk
