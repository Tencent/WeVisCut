"""Role-routed, recording model client for the long-horizon pipeline.

:class:`RoleRoutedModelClient` is a drop-in replacement for AICutPipeline's
``Infer_Vis_new/server/model_loader.UnifiedModelLoader``. It exposes the same
surface (``predict``, ``predict_orchestrator``, ``predict_generator``,
``predict_critic``, ``engine_for_tool``, ``orchestrator_prompt``, ``skill_prompt``)
so ``InferenceOrchestrator`` drives it unchanged, but instead of loading local
weights it speaks the OpenAI chat-completions protocol.

Two things follow from that swap:

- **Polar sees every completion.** Point ``base_url`` at Polar's injected gateway
  and the wire-level traffic is captured for trajectory building and GRPO, with no
  changes inside AICutPipeline.
- **Nothing escapes recording.** Every call funnels through :meth:`predict`, which
  writes to the :class:`~longvideo_polar.recorder.TrajectoryRecorder` before
  returning. Since the pipeline can only reach a model through this client, the
  three role streams are complete by construction.

Each role may target a different endpoint (e.g. a large orchestrator and a small
generator), or all three may share one served model — that is a config choice.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from longvideo_polar.config import ModelEndpoint
from longvideo_polar.recorder import TrajectoryRecorder
from longvideo_polar.roles import Role, ToolSchema

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>([\s\S]*?)</think>\s*", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*\n?")
_THINK_CLOSE = "</think>"


def _split_reasoning(raw: str) -> tuple[str, str]:
    """Separate a completion's reasoning from its answer.

    Returns ``(reasoning, answer)``.

    Qwen3.5's chat template ends the prompt with a bare ``<think>``, so the model
    continues an already-open block and its completion carries only the closing
    tag. A paired ``<think>…</think>`` regex therefore matches nothing and the
    whole reasoning block survives into the answer — which is not a cosmetic
    problem, because every downstream JSON parser (ours and the pipeline's)
    falls back to scanning for the first ``{``. Any brace the model happened to
    write while reasoning gets extracted instead of the real answer, and the
    caller sees a well-formed dict with none of the keys it asked for. That is
    what produced ``template_fill returned empty decoration`` and
    ``Tool_PlanShots produced no shots`` on outputs that were, in fact, complete
    and valid.
    """
    text = str(raw or "")
    paired = _THINK_RE.findall(text)
    stripped = _THINK_RE.sub("", text)
    if _THINK_CLOSE not in stripped:
        return ("\n".join(paired).strip(), stripped.strip())
    # A closing tag with no opener: everything up to the last one is reasoning.
    head, _, tail = stripped.rpartition(_THINK_CLOSE)
    return ("\n".join([*paired, head]).strip(), tail.strip())


@dataclass
class PredictionResult:
    """Mirror of AICutPipeline's ``model_loader.PredictionResult``."""

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
    # ``length`` means the answer was cut off at max_tokens rather than finished.
    # Without it a truncated reasoning block is indistinguishable in the record
    # from a model that simply produced bad JSON.
    finish_reason: str = ""


class RoleRoutedModelClient:
    """Serve the orchestrator / generator / critic roles over HTTP, recording all I/O."""

    def __init__(
        self,
        *,
        endpoints: dict[str, ModelEndpoint],
        tool_schema: ToolSchema,
        recorder: TrajectoryRecorder | None = None,
        orchestrator_system_prompt: str = "",
        skill_system_prompt: str = "",
    ) -> None:
        self._endpoints = dict(endpoints)
        self._tool_schema = tool_schema
        self._recorder = recorder
        self._orchestrator_prompt = orchestrator_system_prompt
        self._skill_prompt = skill_system_prompt or orchestrator_system_prompt
        self._clients: dict[str, httpx.Client] = {}
        self._semaphores: dict[str, threading.Semaphore] = {}
        self._lock = threading.Lock()
        self._loaded = False

    # -- UnifiedModelLoader compatibility --------------------------------

    def load(self, orchestrator_system_prompt: str = "") -> None:
        """Open HTTP clients. Named ``load`` to match the local loader's contract."""
        if orchestrator_system_prompt:
            self._orchestrator_prompt = orchestrator_system_prompt
            if not self._skill_prompt:
                self._skill_prompt = orchestrator_system_prompt
        with self._lock:
            for role, endpoint in self._endpoints.items():
                if role in self._clients:
                    continue
                self._clients[role] = build_http_client(endpoint)
                self._semaphores[role] = threading.Semaphore(
                    max(1, int(endpoint.max_concurrency))
                )
            self._loaded = True
        logger.info(
            "Role endpoints ready: %s",
            {
                role: f"{ep.provider}:{ep.model}@{ep.base_url}"
                for role, ep in self._endpoints.items() if role != "default"
            },
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def orchestrator_prompt(self) -> str:
        return self._orchestrator_prompt

    @property
    def skill_prompt(self) -> str:
        return self._skill_prompt

    def engine_for_tool(self, tool_name: str) -> str:
        """Route a tool to its owning role (the pipeline calls this directly)."""
        return self._tool_schema.role_for_tool(tool_name).value

    def describe_endpoints(self) -> dict[str, dict[str, Any]]:
        """Secret-free per-role endpoint snapshot, for the episode manifest."""
        return {
            role: endpoint.describe()
            for role, endpoint in self._endpoints.items()
            if role != "default"
        }

    def unload(self) -> None:
        with self._lock:
            for client in self._clients.values():
                client.close()
            self._clients.clear()
            self._semaphores.clear()
            self._loaded = False

    def shutdown(self) -> None:
        self.unload()

    def attach_recorder(self, recorder: TrajectoryRecorder | None) -> None:
        """Point the client at a recorder (used when reusing a client across runs)."""
        self._recorder = recorder

    # -- prediction ------------------------------------------------------

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
        """Run one completion for ``role`` and record it.

        Args:
            system_prompt: Role system prompt.
            user_content: Serialized state payload (orchestrator) or skill payload.
            role: ``orchestrator`` | ``generator`` | ``critic``.
            stage: Pipeline stage label, recorded for bucketing.
            allowed_tools: Action space for the orchestrator; ignored otherwise.
            max_tokens: Override the endpoint default.
            temperature: Override the endpoint default.
            top_p: Override the endpoint default.
            timeout: Per-request timeout override.

        Returns:
            The parsed :class:`PredictionResult`. Transport failures return a
            result with ``parse_ok=False`` rather than raising, so the pipeline's
            own fallback logic can advance the run; the error is recorded.
        """
        if not self._loaded:
            self.load()

        role_key = role if role in self._endpoints else "default"
        endpoint = self._endpoints[role_key]
        client = self._clients[role_key]
        semaphore = self._semaphores[role_key]

        resolved_role = Role(role) if role in {r.value for r in Role} else Role.ORCHESTRATOR
        is_orchestrator = resolved_role is Role.ORCHESTRATOR

        # The unified v5 system prompt routes on `user.role`, so skill payloads
        # must carry it exactly as the local loader injects it.
        content = user_content if is_orchestrator else _ensure_role_field(user_content, role)

        payload: dict[str, Any] = {
            "model": endpoint.model,
            "messages": [
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": content},
            ],
            "temperature": endpoint.temperature if temperature is None else temperature,
            "top_p": endpoint.top_p if top_p is None else top_p,
            "max_tokens": int(max_tokens or endpoint.max_tokens),
            "stream": False,
        }
        if endpoint.extra_body:
            payload.update(endpoint.extra_body)

        tools = (
            _build_tool_specs(allowed_tools, self._tool_schema)
            if is_orchestrator and endpoint.native_tool_calls
            else None
        )
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "required"

        # Third-party gateways reject params vLLM accepts (and vice versa); dropping
        # them here beats a 400 that looks like a model failure.
        for key in endpoint.drop_params:
            payload.pop(key, None)

        started = time.monotonic()
        response_json: dict[str, Any] | None = None
        error = ""
        with semaphore:
            for attempt in range(1, max(1, endpoint.max_retries) + 1):
                try:
                    http_response = client.post(
                        endpoint.endpoint_path or "/chat/completions",
                        json=payload,
                        timeout=timeout or endpoint.timeout,
                    )
                    http_response.raise_for_status()
                    response_json = http_response.json()
                    error = ""
                    break
                except Exception as exc:  # noqa: BLE001 - transport must not kill the run
                    error = f"{type(exc).__name__}: {_error_detail(exc)}"
                    if attempt >= max(1, endpoint.max_retries):
                        logger.error(
                            "%s (%s:%s) completion failed after %d attempts: %s",
                            role, endpoint.provider, endpoint.model, attempt, error,
                        )
                        break
                    backoff = min(2.0 * attempt, 10.0)
                    logger.warning(
                        "%s (%s) completion attempt %d failed (%s); retrying in %.1fs",
                        role, endpoint.provider, attempt, error, backoff,
                    )
                    time.sleep(backoff)
        latency_ms = (time.monotonic() - started) * 1000.0

        result = (
            _parse_response(
                response_json, latency_ms=latency_ms, expect_tool_call=is_orchestrator,
            )
            if response_json is not None
            else PredictionResult(latency_ms=latency_ms)
        )
        if tools and result.tool_name:
            result.tool_name = _canonicalize_tool_name(result.tool_name, self._tool_schema)

        if result.finish_reason == "length" and not result.parse_ok:
            # The pipeline only sees an unparseable blob and retries with the same
            # budget, so the second attempt truncates in the same place. Say so.
            logger.warning(
                "%s/%s hit max_tokens (%d) before finishing its answer; raise "
                "models.%s.max_tokens or the retry will truncate identically",
                resolved_role, stage or "-", result.completion_tokens, resolved_role,
            )

        if self._recorder is not None:
            self._recorder.record_llm(
                role=resolved_role,
                system_prompt=system_prompt or "",
                user_content=content,
                raw_output=result.raw_output,
                cleaned_output=result.cleaned_output,
                thinking=result.thinking_summary,
                tool_name=result.tool_name,
                arguments=result.arguments,
                parse_ok=result.parse_ok,
                latency_ms=result.latency_ms,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                model=endpoint.model,
                provider=endpoint.provider,
                stage=stage,
                finish_reason=result.finish_reason,
                error=error,
            )
        return result

    def predict_orchestrator(
        self, system_prompt: str, user_content: str, **kwargs: Any
    ) -> PredictionResult:
        return self.predict(
            system_prompt or self._orchestrator_prompt,
            user_content,
            role=Role.ORCHESTRATOR.value,
            **kwargs,
        )

    def predict_generator(
        self, user_content: str, *, system_prompt: str | None = None, **kwargs: Any
    ) -> PredictionResult:
        return self.predict(
            system_prompt or self._skill_prompt,
            user_content,
            role=Role.GENERATOR.value,
            **kwargs,
        )

    def predict_critic(
        self, user_content: str, *, system_prompt: str | None = None, **kwargs: Any
    ) -> PredictionResult:
        return self.predict(
            system_prompt or self._skill_prompt,
            user_content,
            role=Role.CRITIC.value,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Payload / response helpers
# ---------------------------------------------------------------------------


def build_http_client(endpoint: ModelEndpoint) -> httpx.Client:
    """Open a transport for one endpoint, honoring gateway-specific quirks.

    Two details are load-bearing:

    - ``auth_header_prefix`` may be empty. Some gateways want the bare key and
      reject ``Authorization: Bearer <key>``.
    - Environment proxies are bypassed for loopback. A corporate ``HTTP_PROXY`` will
      otherwise intercept ``127.0.0.1`` as an SSRF target and fail every local vLLM
      call, which is indistinguishable from the server being down. AICutPipeline's
      own ``LLMClient`` does the same thing.
    """
    prefix = str(endpoint.auth_header_prefix or "").strip()
    key = str(endpoint.api_key or "")
    headers = {"Content-Type": "application/json", **dict(endpoint.extra_headers or {})}
    if key:
        headers.setdefault("Authorization", f"{prefix} {key}".strip() if prefix else key)

    base_url = endpoint.base_url.rstrip("/")
    is_loopback = any(host in base_url for host in ("127.0.0.1", "localhost", "0.0.0.0", "[::1]"))
    kwargs: dict[str, Any] = {
        "base_url": base_url,
        "headers": headers,
        "timeout": httpx.Timeout(endpoint.timeout, connect=endpoint.connect_timeout),
    }
    if is_loopback:
        kwargs["trust_env"] = False
    return httpx.Client(**kwargs)


def _error_detail(exc: Exception) -> str:
    """Include the response body for HTTP errors; a bare 400 says nothing useful."""
    response = getattr(exc, "response", None)
    body = ""
    if response is not None:
        try:
            body = (response.text or "").strip()
        except Exception:  # noqa: BLE001 - diagnostics must not raise
            body = ""
    message = str(exc)
    if body:
        return f"{message} | body={body[:600]}"
    return message


def _ensure_role_field(content: str, role: str) -> str:
    """Inject ``role`` into a JSON skill payload, matching UnifiedModelLoader."""
    text = str(content or "").strip()
    if not text:
        return json.dumps({"role": role})
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
    if isinstance(payload, dict) and "role" not in payload:
        payload["role"] = role
        return json.dumps(payload, ensure_ascii=False)
    return text


def _build_tool_specs(
    allowed_tools: list[str] | None, schema: ToolSchema
) -> list[dict[str, Any]] | None:
    """Build OpenAI function specs identical to the training-time schema.

    Uses AICutPipeline's ``LIGHT_TOOL_SCHEMAS`` / ``REQUIRED_TOOL_ARGS`` so the
    orchestrator sees the same tool surface it was trained on. Falls back to a
    permissive object schema when the pipeline module is unavailable.
    """
    names = [str(name) for name in (allowed_tools or schema.action_space) if str(name)]
    if not names:
        return None
    try:
        from server.tool_schema import (  # type: ignore[import-not-found]
            LIGHT_TOOL_SCHEMAS,
            REQUIRED_TOOL_ARGS,
            TOOL_CARDS,
            TOOL_PRECONDITIONS,
        )
    except Exception:  # noqa: BLE001
        LIGHT_TOOL_SCHEMAS, REQUIRED_TOOL_ARGS = {}, {}
        TOOL_CARDS, TOOL_PRECONDITIONS = {}, {}

    specs: list[dict[str, Any]] = []
    for name in names:
        properties = LIGHT_TOOL_SCHEMAS.get(name, {})
        spec_params: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "required": REQUIRED_TOOL_ARGS.get(name, []),
        }
        if properties:
            spec_params["additionalProperties"] = False
        specs.append({
            "type": "function",
            "function": {
                "name": name,
                "description": _tool_description(name, TOOL_CARDS, TOOL_PRECONDITIONS),
                "parameters": spec_params,
            },
        })
    return specs


def _tool_description(
    name: str, cards: dict[str, str], preconditions: dict[str, str]
) -> str:
    card = cards.get(name)
    precondition = preconditions.get(name)
    if card and precondition:
        return f"{card} 前置条件：{precondition}"
    if card:
        return card
    return f"AICut long-horizon agent action: {name}."


def _canonicalize_tool_name(name: str, schema: ToolSchema) -> str:
    """Map model-emitted aliases onto canonical action-space names."""
    if name in schema.action_space:
        return name
    try:
        from server.tool_schema import TOOL_ALIASES  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return name
    return TOOL_ALIASES.get(name, name)


def _parse_response(
    payload: dict[str, Any], *, latency_ms: float, expect_tool_call: bool = True
) -> PredictionResult:
    """Convert an OpenAI chat-completions response into a PredictionResult.

    Args:
        payload: Decoded chat-completions body.
        latency_ms: Wall time of the request.
        expect_tool_call: True for 主流程 turns, whose output must name a tool. False
            for skill turns, whose output is a bare JSON object — for those,
            ``parse_ok`` means "the pipeline's ``_parse_skill_json`` will succeed".
            Judging a skill turn by the tool-call rule would mark every generator and
            critic sample unparseable, which silently poisons any filtering on
            ``parse_ok`` downstream.
    """
    result = PredictionResult(latency_ms=latency_ms)

    usage = payload.get("usage") or {}
    if isinstance(usage, dict):
        result.prompt_tokens = int(usage.get("prompt_tokens") or 0)
        result.completion_tokens = int(usage.get("completion_tokens") or 0)
        if result.completion_tokens > 0 and latency_ms > 0:
            result.tokens_per_sec = result.completion_tokens / (latency_ms / 1000.0)

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return result
    first = choices[0] if isinstance(choices[0], dict) else {}
    result.finish_reason = str(first.get("finish_reason") or "")
    message = first.get("message") if isinstance(first.get("message"), dict) else {}

    raw = str(message.get("content") or "")
    result.raw_output = raw

    # Qwen-style servers may split reasoning into its own field.
    reasoning = str(message.get("reasoning_content") or "")
    inline_reasoning, answer = _split_reasoning(raw)
    result.thinking_summary = (reasoning or inline_reasoning).strip()[:500]
    result.cleaned_output = answer

    structured = _extract_tool_call(message.get("tool_calls"))
    if structured:
        result.tool_name = structured["name"]
        result.arguments = structured["arguments"]
        result.parse_ok = True
        return result

    if not result.cleaned_output:
        return result

    parsed = _parse_tool_json(result.cleaned_output)
    if not expect_tool_call:
        result.parse_ok = parsed is not None
        return result
    if parsed:
        result.tool_name = str(parsed.get("name") or parsed.get("tool_name") or "")
        arguments = parsed.get("arguments", parsed.get("args", {}))
        result.arguments = arguments if isinstance(arguments, dict) else {}
        result.parse_ok = bool(result.tool_name)
    return result


def _extract_tool_call(tool_calls: Any) -> dict[str, Any] | None:
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    first = tool_calls[0]
    function = first.get("function") if isinstance(first, dict) else None
    if not isinstance(function, dict):
        return None
    name = str(function.get("name") or "")
    if not name:
        return None
    raw_args = function.get("arguments", "{}")
    if isinstance(raw_args, str):
        try:
            arguments = json.loads(raw_args or "{}")
        except (json.JSONDecodeError, TypeError):
            arguments = {}
    elif isinstance(raw_args, dict):
        arguments = raw_args
    else:
        arguments = {}
    return {"name": name, "arguments": arguments if isinstance(arguments, dict) else {}}


def _parse_tool_json(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of the first complete JSON object in ``text``."""
    stripped = _FENCE_RE.sub("", text).strip()
    stripped = re.sub(r"```\s*$", "", stripped).strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except (json.JSONDecodeError, TypeError):
        pass

    start = stripped.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(stripped[start:index + 1])
                except (json.JSONDecodeError, TypeError):
                    return None
                return payload if isinstance(payload, dict) else None
    return None
