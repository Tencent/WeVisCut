"""Model access for the studio.

Two consumers with different needs share one 9B policy:

* the **agent loop**, which needs the loader object itself (the orchestrator
  calls role-specific predict methods on it), and
* the **chat edit agent**, which just needs ``(system, user) -> text``.

``ModelService`` owns the loader and exposes both. The heavy imports (``swift``,
``torch``, vLLM) happen inside ``load()`` rather than at module import, so the
rest of the studio — config, editor, project, tracks, routes — can be imported
and tested on a machine with no GPU and no ms-swift installed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger("aicut_studio.model")


class ModelNotReady(RuntimeError):
    """Raised when generation is requested before the model finished loading."""


class ModelService:
    """Wraps the unified single-model loader."""

    def __init__(self, infer_config: dict[str, Any], system_prompt: str) -> None:
        self.infer_config = infer_config
        self.system_prompt = system_prompt
        self.loader: Any | None = None
        self._load_error: str = ""
        # Set when the server was started with --no-model, so the UI can say
        # "skipped" instead of showing a load spinner that will never finish.
        self.skipped: bool = False
        # A plain-mode AICutChatModel over the same weights (created on first use).
        self._chat_model: Any | None = None

    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self.loader is not None and bool(getattr(self.loader, "is_loaded", False))

    @property
    def load_error(self) -> str:
        return self._load_error

    def describe(self) -> dict[str, Any]:
        model_cfg = self.infer_config.get("model", {})
        model_cfg = model_cfg if isinstance(model_cfg, dict) else {}
        profiles = model_cfg.get("profiles", {})
        return {
            "loaded": self.is_loaded,
            "load_error": self._load_error,
            "skipped": self.skipped,
            "active_profile": model_cfg.get("active_profile", ""),
            "available_profiles": list(profiles) if isinstance(profiles, dict) else [],
            "base_model": model_cfg.get("base_model", ""),
            "adapter": model_cfg.get("adapter", "") or "",
            "backend": model_cfg.get("infer_backend", ""),
            "device": model_cfg.get("device_map", ""),
        }

    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load weights. Blocking and slow; call via a thread from async code."""
        from studio.config import install_import_paths

        install_import_paths()
        try:
            from server.model_loader import UnifiedModelLoader
        except ImportError as exc:
            self._load_error = (
                f"无法导入 UnifiedModelLoader ({exc})。"
                "请确认 ms-swift / torch 已安装在当前 Python 环境。"
            )
            raise

        try:
            loader = UnifiedModelLoader(self.infer_config)
            loader.load(orchestrator_system_prompt=self.system_prompt)
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            raise

        self.loader = loader
        self._load_error = ""
        logger.info("unified model ready (profile=%s)", self.describe()["active_profile"])

    async def load_async(self) -> None:
        await asyncio.to_thread(self.load)

    def unload(self) -> None:
        loader, self.loader = self.loader, None
        if loader is None:
            return
        for name in ("unload", "shutdown", "close"):
            fn = getattr(loader, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    logger.exception("model unload via %s() failed", name)
                return

    # ------------------------------------------------------------------
    def _require_loader(self) -> Any:
        if not self.is_loaded:
            raise ModelNotReady(self._load_error or "模型尚未加载完成")
        return self.loader

    def predict_text(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: float = 1.0,
    ) -> Any:
        """Single-turn completion. Returns the loader's prediction object."""
        loader = self._require_loader()
        # multi-model loaders route by role; the unified loader has plain predict
        fn = getattr(loader, "predict_orchestrator", None) or loader.predict
        return fn(system, user, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    def as_chat_model(self) -> Any:
        """A plain-mode ``AICutChatModel`` over the same loaded weights.

        Both the deepagents generation loop and the conversational editor go
        through ``AICutChatModel`` — one model abstraction over the in-process
        vLLM policy.
        """
        if self._chat_model is None:
            from aicut_agent.chat_model import AICutChatModel

            self._chat_model = AICutChatModel()
            self._chat_model.attach_loader(self._require_loader())
        return self._chat_model

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> str:
        """``(system, user) -> text``, the signature `ChatEditAgent` expects.

        Routed through ``AICutChatModel`` (plain mode) for a single model
        abstraction; identical output to calling the loader directly.
        """
        return self.as_chat_model().generate_text(
            system, user, max_tokens=max_tokens, temperature=temperature
        )


class RemoteModelService:
    """OpenAI-compatible client, for ``chat_edit.backend: openai``.

    Lets the conversational editor run on a separate endpoint so a long editing
    session does not contend with the agent loop for GPU memory.
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "EMPTY"
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> str:
        import requests

        url = f"{self.base_url}/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        choices = payload.get("choices") or []
        if not choices:
            return ""
        return str((choices[0].get("message") or {}).get("content", "") or "")


def build_chat_backend(
    chat_config: dict[str, Any], model_service: ModelService
) -> ModelService | RemoteModelService:
    """Pick the chat backend named in config."""
    backend = str((chat_config or {}).get("backend", "inproc") or "inproc")
    if backend == "openai":
        return RemoteModelService(
            base_url=str(chat_config.get("openai_base_url", "") or ""),
            api_key=str(chat_config.get("openai_api_key", "") or ""),
            model=str(chat_config.get("openai_model", "") or ""),
        )
    return model_service
