"""Unified LLM client for the AICut pipeline.

Wraps OpenAI-compatible API (Qwen by default) with retry, timeout,
and structured JSON output parsing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from typing import Any, Callable

import httpx

logger = logging.getLogger("aicut_pipeline")


# ─── 远端 Qwen stage 级别闸门（进程内、跨线程共享）──────────────────────
# 不在单个 HTTP 请求级别限流（会破坏 batch_chat_json 内部的并发和 vLLM 连续批处理），
# 而是提供一个 stage 级别的上下文管理器：同一时刻最多 N 个 slot 同时进行 Qwen-stage，
# 但每个 slot 内的 batch 并发（ThreadPoolExecutor/batch_chat_json）不受限。
# 由 batch_worker 在 _run_one_claim 入口处使用。
_QWEN_STAGE_GATE: "threading.Semaphore | None" = None
_QWEN_STAGE_GATE_LOCK = threading.Lock()


def get_qwen_stage_gate() -> "threading.Semaphore | None":
    """返回 Qwen stage 级别闸门（Semaphore），未配置则返回 None。"""
    global _QWEN_STAGE_GATE
    if _QWEN_STAGE_GATE is None:
        with _QWEN_STAGE_GATE_LOCK:
            if _QWEN_STAGE_GATE is None:
                try:
                    n = int(os.environ.get("AICUT_QWEN_MAX_CONCURRENCY", "0") or 0)
                except ValueError:
                    n = 0
                if n > 0:
                    _QWEN_STAGE_GATE = threading.Semaphore(n)
                    logger.info("[LLMClient] Qwen stage 闸门启用：进程内最多 %d 个 slot 同时做 Qwen stage", n)
    return _QWEN_STAGE_GATE


class LLMClient:
    def __init__(self, config: dict[str, Any], service: str | None = None) -> None:
        self.service_name = service or "default"
        if service:
            services = config.get("services", {})
            service_cfg = services.get(service, {}) if isinstance(services, dict) else {}
            if isinstance(service_cfg, dict) and service_cfg:
                llm_cfg = dict(service_cfg)
            else:
                logger.warning(
                    "[LLMClient] service=%s is not configured; falling back to config.llm",
                    service,
                )
                llm_cfg = dict(config.get("llm", {}))
        else:
            llm_cfg = dict(config.get("llm", {}))
        # At startup the studio rewrites `config.llm` to its own /v1 endpoint so
        # the loaded policy model also serves the pipeline's internal LLM calls.
        # These defaults only apply when the pipeline runs standalone.
        self.base_url = llm_cfg.get("base_url") or os.environ.get(
            "WEDIRECTOR_LLM_BASE_URL", "http://127.0.0.1:8081/v1"
        )
        self.api_key = llm_cfg.get("api_key") or os.environ.get(
            "WEDIRECTOR_LLM_API_KEY", "EMPTY"
        )
        self.model = llm_cfg.get("model") or os.environ.get(
            "WEDIRECTOR_LLM_MODEL", "wedirector-9b"
        )
        self.temperature = llm_cfg.get("temperature", 0.3)
        self.max_tokens = llm_cfg.get("max_tokens", 8192)
        self.timeout = llm_cfg.get("timeout", 600)
        self.endpoint = llm_cfg.get("endpoint", "/chat/completions")
        self.offline = bool(llm_cfg.get("offline", False))
        self.max_prompt_chars = int(llm_cfg.get("max_prompt_chars", 262144) or 262144)
        self.max_frames_per_call = int(
            llm_cfg.get("max_frames_per_call", llm_cfg.get("max_frames", 8)) or 8
        )
        auth_prefix = str(llm_cfg.get("auth_header_prefix", "Bearer")).strip()
        auth_value = f"{auth_prefix} {self.api_key}".strip() if auth_prefix else str(self.api_key)
        headers = {"Authorization": auth_value} if auth_value else {}
        client_kwargs = {
            "base_url": self.base_url,
            "headers": headers,
            "timeout": httpx.Timeout(self.timeout, connect=llm_cfg.get("connect_timeout", 20.0)),
        }
        proxy_url = str(llm_cfg.get("proxy_url") or "").strip()
        if proxy_url:
            client_kwargs["proxy"] = proxy_url
        # Bypass environment proxy for localhost/loopback to avoid corporate
        # proxies blocking 127.0.0.1 as "forbidden host" (SSRF protection).
        _is_local = any(
            h in self.base_url
            for h in ("127.0.0.1", "localhost", "0.0.0.0")
        )
        if _is_local and not proxy_url:
            client_kwargs["proxy"] = None
            client_kwargs["trust_env"] = False
        self._client = httpx.Client(**client_kwargs)

    def chat_messages(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if self.offline:
            return "{}"
        resp = self._client.post(self.endpoint, json=payload)
        if resp.is_error:
            body_preview = (resp.text or "").strip()
            if len(body_preview) > 1000:
                body_preview = body_preview[:1000] + "...(truncated)"
            logger.warning(
                "[LLMClient] %s %s failed: status=%s body=%s",
                self.service_name,
                self.model,
                resp.status_code,
                body_preview or "(empty)",
            )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def chat(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return self.chat_messages(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        raw = self.chat(system, user, temperature=temperature, max_tokens=max_tokens)
        parsed = _extract_json(raw)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"items": parsed}
        if isinstance(parsed, tuple):
            for item in parsed:
                if isinstance(item, dict):
                    return item
            return {}
        return {}

    def chat_json_multimodal(
        self,
        system: str,
        user_parts: list[dict[str, Any]],
        temperature: float | None = None,
    ) -> dict[str, Any]:
        raw = self.chat_messages(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_parts},
            ],
            temperature=temperature,
        )
        parsed = _extract_json(raw)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"items": parsed}
        return {}

    def generate_options(
        self,
        system: str,
        user: str,
        num_options: int = 4,
    ) -> list[dict[str, Any]]:
        """Generate N diverse solution options for HITL decision points."""
        prompt = (
            f"{user}\n\n"
            f"请生成 {num_options} 个不同的解决方案选项，每个方案用 JSON 对象表示，"
            f"包含 'option_id'(int), 'label'(str简短标题), 'description'(str详细说明), "
            f"'changes'(dict具体修改内容)。"
            f"返回一个 JSON 数组。"
        )
        raw = self.chat(system, prompt, temperature=0.7)
        parsed = _extract_json(raw)
        if isinstance(parsed, list):
            return parsed[:num_options]
        if isinstance(parsed, dict) and "options" in parsed:
            return parsed["options"][:num_options]
        return [parsed] if parsed else []

    def close(self) -> None:
        self._client.close()


# ─── 通用批量 LLM 调用 ────────────────────────────────────────────────

def _log_progress(current: int, total: int, label: str) -> None:
    """向 stderr 输出简易进度条。"""
    import sys
    if total <= 0:
        return
    pct = current / total * 100
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
    msg = f"\r  [{bar}] {pct:5.1f}%  {label}  ({current}/{total})"
    sys.stderr.write(msg)
    sys.stderr.flush()
    if current >= total:
        sys.stderr.write("\n")


def batch_chat_json(
    llm: "LLMClient",
    items: list[dict[str, Any]],
    *,
    format_prompt: Callable[[list[dict[str, Any]]], str],
    extract_results: Callable[[dict[str, Any]], list[dict[str, Any]]],
    system_prompt: str = "",
    temperature: float = 0.2,
    max_batch_size: int = 30,
    max_batch_output_chars: int = 40000,
    concurrency: int = 4,
    item_label: str = "items",
    output_key: str = "items",
) -> list[dict[str, Any]]:
    """通用批量 LLM 调用：自动分批 → 并发执行 → 合并 → 校验。

    工作流:
    1. 根据 max_batch_size 将 items 拆分为 chunks
    2. 以 concurrency 并发调 llm.chat_json() 处理每个 chunk
    3. 各 chunk 的输出按 extract_results() 提取 item list
    4. 按原始索引顺序合并
    5. 校验总数是否正确 → 不完整则降级到逐条顺序调用

    Returns:
        合并后的 item list（顺序与输入一致）

    Example:
        shots = batch_chat_json(
            llm, annotated_shots,
            format_prompt=lambda batch: LABEL_USER_TEMPLATE.format(
                num_shots=len(batch), shots_json=json.dumps(batch)
            ),
            extract_results=lambda raw: raw.get("shots", []),
            system_prompt=LABEL_SYSTEM_PROMPT,
            concurrency=4, max_batch_size=30,
            output_key="shots",
        )
    """
    if not items:
        return []

    n = len(items)
    # 估算每批大小：max_batch_size 和基于输出量估计的较小值
    est_per_item = max(150, min(400, max_batch_output_chars // max(1, max_batch_size)))
    batch_size = min(max_batch_size, max(1, max_batch_output_chars // est_per_item))
    if batch_size <= 0:
        batch_size = max_batch_size
    chunks = [items[i:i + batch_size] for i in range(0, n, batch_size)]

    if len(chunks) == 1:
        return _execute_single_batch(
            llm, items, format_prompt, extract_results,
            system_prompt, temperature, item_label, output_key,
        )

    # ── 并发执行多个 chunk ──
    results_by_chunk: dict[int, list[dict[str, Any]]] = {}
    errors: list[str] = []
    _log_progress(0, len(chunks), item_label)

    def _do_chunk(idx: int, chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            return _execute_single_batch(
                llm, chunk, format_prompt, extract_results,
                system_prompt, temperature, item_label, output_key,
            )
        except Exception as exc:
            errors.append(f"chunk_{idx}: {exc}")
            return []

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(_do_chunk, i, c): i for i, c in enumerate(chunks)}
        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results_by_chunk[idx] = future.result()
            except Exception as exc:
                errors.append(f"chunk_{idx}: {exc}")
                results_by_chunk[idx] = []
            completed += 1
            _log_progress(completed, len(chunks), item_label)

    # 按索引顺序合并
    merged: list[dict[str, Any]] = []
    for i in range(len(chunks)):
        merged.extend(results_by_chunk.get(i, []))

    # 校验完整性
    if len(merged) != n:
        logger.warning(
            "[batch_chat_json] 输出不完整: expected %d %s, got %d. 降级到逐条顺序调用.",
            n, item_label, len(merged),
        )
        if errors:
            logger.warning("[batch_chat_json] 分批错误: %s", "; ".join(errors[:3]))
        # 降级：逐条单次调用（最慢但最可靠）
        return _sequential_fallback(
            llm, items, format_prompt, extract_results,
            system_prompt, temperature, item_label, output_key,
        )
    return merged


def _execute_single_batch(
    llm: "LLMClient",
    items: list[dict[str, Any]],
    format_prompt: Callable[[list[dict[str, Any]]], str],
    extract_results: Callable[[dict[str, Any]], list[dict[str, Any]]],
    system_prompt: str,
    temperature: float,
    item_label: str,
    output_key: str,
) -> list[dict[str, Any]]:
    """执行单批 LLM 调用并提取结果。"""
    prompt = format_prompt(items)
    raw = llm.chat_json(system_prompt, prompt, temperature=temperature)
    results = extract_results(raw)
    if not isinstance(results, list):
        raise ValueError(f"{output_key} 不是 list: {type(results).__name__}")
    return results


def _sequential_fallback(
    llm: "LLMClient",
    items: list[dict[str, Any]],
    format_prompt: Callable[[list[dict[str, Any]]], str],
    extract_results: Callable[[dict[str, Any]], list[dict[str, Any]]],
    system_prompt: str,
    temperature: float,
    item_label: str,
    output_key: str,
) -> list[dict[str, Any]]:
    """降级方案：逐条顺序调用 LLM（最慢但最可靠）。"""
    results: list[dict[str, Any]] = []
    for item in items:
        try:
            batch = _execute_single_batch(
                llm, [item], format_prompt, extract_results,
                system_prompt, temperature, item_label, output_key,
            )
        except Exception:
            batch = []
        results.extend(batch)
    return results


def _extract_json(text: str) -> Any:
    """Extract JSON from LLM output that may contain markdown fences or extra text."""
    text = text.strip()
    # Strip Qwen3 <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if m := re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL):
        text = m.group(1).strip()
    # Try parsing the cleaned text directly
    for candidate in [text, text.split("\n\n")[0]]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    # Last resort: find first { or [ and parse from there
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break
    return {}
