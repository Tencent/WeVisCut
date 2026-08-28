"""Unified Model Loader — single 9B model with role-based routing.

Loads ONE model at startup and routes orchestrator/generator/critic calls
by switching system prompts and inference parameters per role. Provides the
same interface as the old MultiModelLoader so infer_graph.py works unchanged.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("aicut_infer")

_THINK_RE = re.compile(r"<think>([\s\S]*?)</think>\s*", re.DOTALL)


@dataclass
class PredictionResult:
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    agent_trace: dict[str, Any] = field(default_factory=dict)
    raw_output: str = ""
    cleaned_output: str = ""
    thinking_summary: str = ""
    parse_ok: bool = False
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_sec: float = 0.0


class UnifiedModelLoader:
    """Single model, three roles. Drop-in replacement for MultiModelLoader."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.engine = None
        self._request_config_cls = None
        self._loaded = False
        self._infer_lock = threading.Lock()
        self._infer_backend = "transformers"
        self._default_max_tokens = 2048
        self._temperature = 0.0
        self._top_p = 1.0
        self._top_k: int | None = None
        self._repetition_penalty: float | None = None
        self._stop_words: list[str] = []
        self._max_tokens_by_role: dict[str, int] = {}
        self._max_tokens_by_stage: dict[str, int] = {}
        self._orchestrator_prompt: str = ""
        self._skill_prompt: str = ""
        self._enable_thinking: bool = True

    def load(self, orchestrator_system_prompt: str = "") -> None:
        if self._loaded:
            return

        model_cfg = self._resolve_model_config()
        base_model = str(model_cfg.get("base_model", ""))
        adapter = str(model_cfg.get("adapter", ""))
        model_type = str(model_cfg.get("model_type", "qwen3_5"))
        template_type = str(model_cfg.get("template_type", "qwen3_5"))
        attn_impl = str(model_cfg.get("attn_impl", "flash_attention_2"))
        device_map = str(model_cfg.get("device_map", "auto"))
        self._infer_backend = str(model_cfg.get("infer_backend", "transformers")).lower()
        self._default_max_tokens = int(model_cfg.get("max_new_tokens", 2048))
        self._temperature = float(model_cfg.get("temperature", 0.0))
        self._top_p = float(model_cfg.get("top_p", 1.0))
        self._top_k = _optional_int(model_cfg.get("top_k"))
        self._repetition_penalty = _optional_float(model_cfg.get("repetition_penalty"))
        stop_words = model_cfg.get("stop_words", model_cfg.get("stop", [])) or []
        self._stop_words = [str(s) for s in stop_words if str(s)]
        self._max_tokens_by_role = {
            str(k): int(v)
            for k, v in (model_cfg.get("max_new_tokens_by_role", {}) or {}).items()
            if v is not None
        }
        self._max_tokens_by_stage = {
            str(k): int(v)
            for k, v in (model_cfg.get("max_new_tokens_by_stage", {}) or {}).items()
            if v is not None
        }

        self._orchestrator_prompt = orchestrator_system_prompt or ""
        self._skill_prompt = orchestrator_system_prompt or ""
        # Default True: Qwen3.5's non_thinking_prefix is a closed empty <think>,
        # which makes thinking_summary always empty and mismatches SFT/DAPO.
        self._enable_thinking = _as_bool(model_cfg.get("enable_thinking"), True)

        # OpenAI-compatible endpoint: no local weights, no swift, no CUDA context.
        # Used both for remote APIs (302AI upper bounds) and for locally served
        # vLLM checkpoints, so every policy reaches predict() through one path.
        if self._infer_backend in {"openai", "api"}:
            self._request_config_cls = _PlainRequestConfig
            base_url = str(model_cfg.get("openai_base_url", "")).strip()
            served = str(model_cfg.get("openai_model", "")).strip() or base_model
            if not base_url:
                raise ValueError("infer_backend=openai requires model.openai_base_url")
            if not served:
                raise ValueError("infer_backend=openai requires model.openai_model")
            api_key, key_source = _resolve_openai_key(model_cfg)
            logger.info(
                "[UnifiedModelLoader] backend=openai endpoint=%s model=%s key=%s",
                base_url, served, key_source,
            )
            self.engine = _OpenAICompatEngine(
                base_url=base_url,
                api_key=api_key,
                model=served,
                timeout=float(model_cfg.get("openai_timeout", 600)),
                max_retries=int(model_cfg.get("openai_max_retries", 3)),
                extra_body=_with_thinking_extra_body(
                    dict(model_cfg.get("openai_extra_body") or {}),
                    self._enable_thinking,
                ),
                send_tool_choice=_as_bool(model_cfg.get("openai_send_tool_choice"), True),
            )
            self._loaded = True
            logger.info("[UnifiedModelLoader] enable_thinking=%s (openai extra_body)", self._enable_thinking)
            return

        # ms-swift 优先用 pip 安装;也允许通过环境变量 AICUT_MS_SWIFT_PATH 或
        # model.ms_swift_path 指向一份源码 checkout(完整拷贝后不再默认相对路径)。
        import os
        ms_swift_candidates = [
            os.environ.get("AICUT_MS_SWIFT_PATH", ""),
            str(self.config.get("model", {}).get("ms_swift_path", "") or ""),
        ]
        for cand in ms_swift_candidates:
            if cand and Path(cand).exists() and cand not in sys.path:
                sys.path.insert(0, cand)
                break

        import torch
        from swift.infer_engine import RequestConfig

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._request_config_cls = RequestConfig

        logger.info(
            "[UnifiedModelLoader] backend=%s base=%s adapter=%s dtype=%s",
            self._infer_backend, base_model, adapter or "(merged)", dtype,
        )
        t0 = time.monotonic()

        adapters = [adapter] if adapter else []
        if self._infer_backend == "vllm":
            from swift.infer_engine import VllmEngine
            self.engine = VllmEngine(
                base_model,
                adapters=adapters,
                torch_dtype=dtype,
                model_type=model_type,
                template_type=template_type,
                gpu_memory_utilization=float(model_cfg.get("vllm_gpu_memory_utilization", 0.9)),
                tensor_parallel_size=int(model_cfg.get("vllm_tensor_parallel_size", 1)),
                max_model_len=_optional_int(model_cfg.get("vllm_max_model_len")),
                max_num_seqs=int(model_cfg.get("vllm_max_num_seqs", 32)),
                enable_prefix_caching=_optional_bool(model_cfg.get("vllm_enable_prefix_caching")),
                max_lora_rank=int(model_cfg.get("vllm_max_lora_rank", 32)),
                enforce_eager=_as_bool(model_cfg.get("vllm_enforce_eager"), False),
            )
        else:
            from swift.infer_engine import TransformersEngine
            self.engine = TransformersEngine(
                base_model,
                adapters=adapters,
                torch_dtype=dtype,
                model_type=model_type,
                template_type=template_type,
                attn_impl=attn_impl,
                device_map=device_map,
                max_batch_size=1,
            )

        elapsed = time.monotonic() - t0
        self._apply_thinking_mode()
        self._loaded = True
        logger.info(
            "[UnifiedModelLoader] Model loaded in %.1fs enable_thinking=%s",
            elapsed, self._enable_thinking,
        )

    def _resolve_model_config(self) -> dict[str, Any]:
        model_cfg = dict(self.config.get("model", {}) or {})
        active_profile = str(model_cfg.get("active_profile", ""))
        profiles = model_cfg.get("profiles", {}) or {}
        if active_profile and active_profile in profiles:
            profile_cfg = profiles[active_profile]
            if isinstance(profile_cfg, dict):
                for k, v in profile_cfg.items():
                    if v not in (None, ""):
                        model_cfg[k] = v
        return model_cfg

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def orchestrator_prompt(self) -> str:
        return self._orchestrator_prompt

    @property
    def skill_prompt(self) -> str:
        return self._skill_prompt

    # ─── Public predict methods ───

    def predict(
        self,
        system_prompt: str,
        user_content: str,
        *,
        role: str = "orchestrator",
        stage: str = "",
        allowed_tools: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        timeout: float | None = None,
    ) -> PredictionResult:
        if not self._loaded or self.engine is None:
            raise RuntimeError("Model not loaded")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        tools = _make_agent_tools(allowed_tools) if role == "orchestrator" else None
        effective_max = self._effective_max_tokens(role, stage, max_tokens)
        effective_temp = temperature if temperature is not None else self._temperature
        effective_top_p = top_p if top_p is not None else self._top_p
        use_stop = list(self._stop_words) if role == "orchestrator" else []

        request_config = self._request_config_cls(
            max_tokens=effective_max,
            temperature=effective_temp,
            top_p=effective_top_p,
            top_k=self._top_k,
            repetition_penalty=self._repetition_penalty,
            stop=use_stop,
        )

        with self._infer_lock:
            t0 = time.monotonic()
            resp = self.engine.infer(
                [{
                    "messages": messages,
                    "tools": tools,
                    "chat_template_kwargs": {"enable_thinking": self._enable_thinking},
                }],
                request_config=request_config,
            )
            latency_ms = (time.monotonic() - t0) * 1000.0

        return _build_prediction_result(resp, latency_ms=latency_ms, stage=stage, allowed_tools=allowed_tools)

    def predict_orchestrator(
        self,
        system_prompt: str,
        user_content: str,
        **kwargs: Any,
    ) -> PredictionResult:
        return self.predict(system_prompt, user_content, role="orchestrator", **kwargs)

    def predict_generator(
        self,
        user_content: str,
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> PredictionResult:
        sp = system_prompt or self._skill_prompt
        content = _ensure_role_field(user_content, "generator")
        return self.predict(sp, content, role="generator", **kwargs)

    def predict_critic(
        self,
        user_content: str,
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> PredictionResult:
        sp = system_prompt or self._skill_prompt
        content = _ensure_role_field(user_content, "critic")
        return self.predict(sp, content, role="critic", **kwargs)

    def engine_for_tool(self, tool_name: str) -> str:
        from server.tool_schema import GENERATOR_TOOLS, CRITIC_TOOLS
        if tool_name in GENERATOR_TOOLS:
            return "generator"
        if tool_name in CRITIC_TOOLS:
            return "critic"
        return "orchestrator"

    def _apply_thinking_mode(self) -> None:
        """Pin the loaded swift template to the same thinking mode as SFT/DAPO."""
        template = getattr(self.engine, "template", None)
        if template is None:
            return
        template.enable_thinking = self._enable_thinking
        kwargs = getattr(template, "chat_template_kwargs", None)
        if isinstance(kwargs, dict):
            kwargs["enable_thinking"] = self._enable_thinking
        else:
            template.chat_template_kwargs = {"enable_thinking": self._enable_thinking}

    def _effective_max_tokens(self, role: str, stage: str, explicit: int | None) -> int:
        if explicit:
            return explicit
        role_max = self._max_tokens_by_role.get(role)
        if role_max:
            return role_max
        stage_max = self._max_tokens_by_stage.get(stage)
        if stage_max:
            return stage_max
        return self._default_max_tokens

    def unload(self) -> None:
        if self.engine is not None:
            if hasattr(self.engine, "shutdown"):
                self.engine.shutdown()
            self.engine = None
        self._loaded = False

    def shutdown(self) -> None:
        self.unload()


# ─── Helpers ───

def _ensure_role_field(content: str, role: str) -> str:
    """Inject role field into user content JSON if not present."""
    content = str(content or "").strip()
    if not content:
        return json.dumps({"role": role})
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and "role" not in obj:
            obj["role"] = role
            return json.dumps(obj, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        pass
    return content


def _build_prediction_result(
    resp: Any,
    *,
    latency_ms: float = 0.0,
    stage: str = "",
    allowed_tools: list[str] | None = None,
) -> PredictionResult:
    result = PredictionResult(latency_ms=latency_ms)

    if not resp:
        return result
    resp_item = resp[0] if isinstance(resp, (list, tuple)) and len(resp) > 0 else resp

    choices = getattr(resp_item, "choices", None)
    if choices and len(choices) > 0:
        msg = choices[0].message if hasattr(choices[0], "message") else choices[0]
        raw = getattr(msg, "content", "") or ""
    elif isinstance(resp_item, dict):
        raw = str(resp_item.get("response", resp_item.get("choices", [{}])[0].get("message", {}).get("content", "")))
        msg = None
    else:
        raw = str(getattr(resp_item, "response", ""))
        msg = None

    result.raw_output = raw

    usage = getattr(resp_item, "usage", None)
    if usage:
        result.prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        result.completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        if result.completion_tokens > 0 and latency_ms > 0:
            result.tokens_per_sec = result.completion_tokens / (latency_ms / 1000.0)

    reasoning = ""
    if msg is not None:
        reasoning = str(getattr(msg, "reasoning_content", "") or "")
    elif isinstance(resp_item, dict):
        msg_dict = ((resp_item.get("choices") or [{}])[0] or {}).get("message") or {}
        reasoning = str(msg_dict.get("reasoning_content") or "")
    result.thinking_summary = _extract_thinking(raw, reasoning)
    if result.thinking_summary and not _THINK_RE.search(raw) and reasoning:
        # vLLM reasoning parser moved think out of content; keep raw_output intact
        # for the terminal panel so the UI still shows a full <think> block.
        result.raw_output = f"<think>\n{result.thinking_summary}\n</think>\n{raw}".strip()
    cleaned = _THINK_RE.sub("", result.raw_output).strip()
    result.cleaned_output = cleaned

    tc = _extract_structured_tool_call(msg) if msg else None
    if tc:
        result.tool_name = tc["name"]
        result.arguments = tc["arguments"]
        result.parse_ok = True
    elif cleaned:
        parsed = _try_parse_tool_json(cleaned)
        if parsed:
            result.tool_name = parsed.get("name", parsed.get("tool_name", ""))
            result.arguments = parsed.get("arguments", parsed.get("args", {}))
            result.parse_ok = bool(result.tool_name)

    return result


def _extract_structured_tool_call(msg: Any) -> dict[str, Any] | None:
    if msg is None:
        return None
    tool_calls = getattr(msg, "tool_calls", None)
    if not tool_calls:
        return None
    if isinstance(tool_calls, list) and len(tool_calls) > 0:
        tc = tool_calls[0]
        fn = getattr(tc, "function", None) or (tc if isinstance(tc, dict) else None)
        if fn is None:
            return None
        if hasattr(fn, "name"):
            name = str(fn.name or "")
            args_raw = getattr(fn, "arguments", "{}")
        elif isinstance(fn, dict):
            name = str(fn.get("name") or "")
            args_raw = fn.get("arguments", "{}")
        else:
            return None
        if not name:
            return None
        if isinstance(args_raw, str):
            try:
                arguments = json.loads(args_raw)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
        elif isinstance(args_raw, dict):
            arguments = args_raw
        else:
            arguments = {}
        return {"name": name, "arguments": arguments if isinstance(arguments, dict) else {}}
    return None


def _try_parse_tool_json(text: str) -> dict[str, Any] | None:
    text = re.sub(r"```(?:json)?\s*\n?", "", text).strip()
    text = re.sub(r"```\s*$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, TypeError):
        pass
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    best_end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    best_end = i
                    break
    if best_end > start:
        try:
            obj = json.loads(text[start:best_end + 1])
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _make_agent_tools(allowed_tools: list[str] | None) -> list[dict[str, Any]] | None:
    """Build tool schemas matching the training data format exactly.

    Uses LIGHT_TOOL_SCHEMAS (input control params) + REQUIRED_TOOL_ARGS
    with additionalProperties=False, identical to build_orchestrator_sft_v4.py.
    """
    if not allowed_tools:
        return None
    from server.tool_schema import (
        TOOL_CARDS, TOOL_PRECONDITIONS,
        LIGHT_TOOL_SCHEMAS, REQUIRED_TOOL_ARGS,
    )
    def _sanitize(props):
        """Ensure array/object params carry items/properties (strict-proxy fix)."""
        out = {}
        for k, v in (props or {}).items():
            nv = dict(v) if isinstance(v, dict) else {"type": str(v)}
            if nv.get("type") == "array" and "items" not in nv:
                nv["items"] = {}
            if nv.get("type") == "object" and "properties" not in nv:
                nv["properties"] = {}
            out[k] = nv
        return out

    return [
        {
            "type": "function",
            "function": {
                "name": str(name),
                "description": _tool_description(str(name), TOOL_CARDS, TOOL_PRECONDITIONS),
                "parameters": {
                    "type": "object",
                    "properties": _sanitize(LIGHT_TOOL_SCHEMAS.get(str(name), {})),
                    "required": REQUIRED_TOOL_ARGS.get(str(name), []),
                    "additionalProperties": False,
                },
            },
        }
        for name in allowed_tools
    ]


def _tool_description(name: str, cards: dict[str, str], preconditions: dict[str, str]) -> str:
    card = cards.get(name)
    precondition = preconditions.get(name)
    if card and precondition:
        return f"{card} 前置条件：{precondition}"
    if card:
        return card
    return f"AICut long-horizon agent action: {name}."


_THINK_SUMMARY_LIMIT = 2000


def _extract_thinking(raw: str, reasoning_content: str = "") -> str:
    """Pull the think body from <think> tags or from a split reasoning field.

    An empty ``<think></think>`` (Qwen3.5 non-thinking prefix) is treated as
    missing, so the frontend does not render a blank think card.
    """
    match = _THINK_RE.search(str(raw or ""))
    if match:
        body = match.group(1).strip()
        if body:
            return body[:_THINK_SUMMARY_LIMIT]
    extra = str(reasoning_content or "").strip()
    if extra:
        return extra[:_THINK_SUMMARY_LIMIT]
    return ""


def _with_thinking_extra_body(extra_body: dict[str, Any], enable_thinking: bool) -> dict[str, Any]:
    merged = dict(extra_body or {})
    kwargs = dict(merged.get("chat_template_kwargs") or {})
    kwargs.setdefault("enable_thinking", enable_thinking)
    merged["chat_template_kwargs"] = kwargs
    return merged


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    return _as_bool(value)


# ─── OpenAI-compatible policy backend ───
#
# Why this exists: the Table2 evaluation has to drive the SAME orchestrator and
# the SAME tool set with 9 different policy LLMs — 6 local checkpoints and 3
# remote APIs. Loading weights in-process only covers the local ones, and it
# also pins a large GPU allocation inside the inference server. Routing every
# policy through an OpenAI-compatible endpoint instead means:
#   * remote models (302AI) become usable as policies at all;
#   * local checkpoints are served by a standalone `vllm serve` whose
#     gpu_memory_utilization is controlled outside this process (so a concurrent
#     training job is not squeezed);
#   * prompt construction, tool schemas and output parsing stay byte-identical
#     across all 9 rows, because predict() below is untouched.
#
# The shim deliberately returns the SDK's ChatCompletion object as-is: it
# already exposes .choices[0].message.{content,tool_calls} and .usage, which is
# precisely what _build_prediction_result() reads.


@dataclass
class _PlainRequestConfig:
    """Stand-in for swift's RequestConfig so the openai backend needs no swift."""
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    stop: list[str] = field(default_factory=list)


def _resolve_openai_key(model_cfg: dict[str, Any]) -> tuple[str, str]:
    """Resolve the endpoint credential, preferring the process environment.

    Returns (key, provenance) — provenance is logged so a run manifest can
    record which credential source was used without ever printing the key.
    A locally served vLLM needs no real auth, hence the "EMPTY" sentinel.
    """
    import os

    env_name = str(model_cfg.get("openai_api_key_env", "AICUT_POLICY_API_KEY")).strip()
    if env_name:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value, f"env:{env_name}"

    inline = str(model_cfg.get("openai_api_key", "")).strip()
    if inline:
        return inline, "config:model.openai_api_key"

    return "EMPTY", "sentinel:EMPTY (unauthenticated local endpoint)"


class _OpenAICompatEngine:
    """Minimal `.infer()` adapter over an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 600.0,
        max_retries: int = 3,
        extra_body: dict[str, Any] | None = None,
        send_tool_choice: bool = True,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._model = model
        self._extra_body = dict(extra_body or {})
        self._send_tool_choice = send_tool_choice
        self.model_name = model
        self.base_url = base_url

    def infer(self, infer_requests: list[dict[str, Any]], request_config: Any = None) -> list[Any]:
        if not infer_requests:
            return []
        request = infer_requests[0]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": request.get("messages", []),
        }

        tools = request.get("tools")
        if tools:
            kwargs["tools"] = tools
            if self._send_tool_choice:
                kwargs["tool_choice"] = "auto"

        max_tokens = getattr(request_config, "max_tokens", None)
        if max_tokens:
            kwargs["max_tokens"] = int(max_tokens)
        temperature = getattr(request_config, "temperature", None)
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        top_p = getattr(request_config, "top_p", None)
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
        stop = getattr(request_config, "stop", None) or []
        if stop:
            kwargs["stop"] = list(stop)

        # top_k / repetition_penalty are not OpenAI parameters. Only forward them
        # when explicitly configured, and only through extra_body — a strict
        # proxy rejects the whole request for an unknown top-level field.
        extra_body = dict(self._extra_body)
        top_k = getattr(request_config, "top_k", None)
        if top_k is not None:
            extra_body["top_k"] = int(top_k)
        repetition_penalty = getattr(request_config, "repetition_penalty", None)
        if repetition_penalty is not None:
            extra_body["repetition_penalty"] = float(repetition_penalty)
        if extra_body:
            kwargs["extra_body"] = extra_body

        completion = self._client.chat.completions.create(**kwargs)
        return [completion]

    def shutdown(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - shutdown must never raise
                pass
