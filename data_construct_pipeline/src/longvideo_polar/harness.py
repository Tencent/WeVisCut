"""Polar harness that runs a long-horizon 口播稿 → 成片 rollout in a runtime.

The harness is intentionally thin. Polar prepares the container, injects
``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` for its gateway proxy, and then this
harness writes a session-local config and invokes ``longvideo-polar run``. Because
all three roles reach their model through that injected endpoint, Polar captures
wire-level completions for trajectory building and GRPO **and** the in-runtime
recorder writes the role-split dumps — the same rollout serves both purposes.

The task instruction is treated as the 口播稿 itself, unless
``settings.script_path`` points at a file already staged in the runtime (the usual
choice for long scripts, which do not belong in a prompt field).
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from typing import Any

from polar.agent.base import BaseHarness
from polar.agent.models import AgentSpec
from polar.runtime.base import RUNTIME_AGENT_LOG_DIR, RUNTIME_SESSION_DIR, BaseRuntime
from polar.runtime.models import ExecInput

from longvideo_polar.config import ROLE_NAMES

DEFAULT_OUTPUT_DIR = "/polar/outputs/longvideo"
"""Runtime path for episode artifacts. Bind-mount it to keep dumps after teardown."""


class LongVideoPolarHarness(BaseHarness):
    """Launch ``longvideo-polar run`` inside a Polar-prepared runtime."""

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)
        self._config_dir = f"{RUNTIME_SESSION_DIR}/.longvideo-polar"
        self._config_path = f"{self._config_dir}/config.yaml"
        self._script_path = f"{self._config_dir}/script.txt"

    async def setup(self, runtime: BaseRuntime) -> None:
        """Create the session config directory and the artifact output root."""
        await runtime.exec(
            f"mkdir -p {shlex.quote(self._config_dir)} "
            f"{shlex.quote(self._output_dir())} {shlex.quote(RUNTIME_AGENT_LOG_DIR)}"
        )

    def run_steps(self, instruction: str) -> list[ExecInput]:
        """Write the config (and script), then run the rollout.

        Args:
            instruction: The 口播稿 text, unless ``settings.script_path`` is set.

        Returns:
            Two ``ExecInput`` steps: config materialization, then the rollout.
        """
        env = {**self.env, "PYTHONUNBUFFERED": "1"}
        script_path = _setting_str(self.settings, "script_path")
        inline_script = "" if script_path else instruction

        return [
            ExecInput(
                command=self._write_config_command(inline_script=inline_script),
                env=env,
            ),
            ExecInput(
                command=self._run_command(script_path=script_path or self._script_path),
                env=env,
            ),
        ]

    def postrun_steps(self) -> list[ExecInput]:
        """Copy artifacts to a durable location when one is configured."""
        durable = _setting_str(self.settings, "persist_dir")
        if not durable:
            return []
        source = shlex.quote(self._output_dir())
        target = shlex.quote(durable)
        return [ExecInput(
            command=(
                f"mkdir -p {target} && cp -rf {source}/. {target}/ 2>/dev/null || true"
            ),
        )]

    # -- command construction --------------------------------------------

    def _output_dir(self) -> str:
        return _setting_str(self.settings, "output_dir") or DEFAULT_OUTPUT_DIR

    def _write_config_command(self, *, inline_script: str) -> str:
        """Emit a heredoc that materializes the rollout config in the runtime."""
        config = self._build_config()
        body = json.dumps(config, ensure_ascii=False, indent=2)
        command = (
            f"mkdir -p {shlex.quote(self._config_dir)} && "
            f"cat > {shlex.quote(self._config_path)} <<'LVPCONFIG'\n{body}\nLVPCONFIG"
        )
        if inline_script:
            command += (
                f"\ncat > {shlex.quote(self._script_path)} <<'LVPSCRIPT'\n"
                f"{inline_script}\nLVPSCRIPT"
            )
        return command

    def _build_config(self) -> dict[str, Any]:
        """Build the config document.

        YAML is a superset of JSON, so the harness writes JSON and the loader reads
        it as YAML. That avoids hand-rolling YAML quoting for arbitrary 口播稿 and
        path values.

        Polar's injected gateway variables are referenced as ``${VAR}`` and expanded
        by the config loader at run time, so the port Polar chose does not need to be
        known here.

        A per-role override in ``settings["models"]`` bypasses the gateway for that
        role. That is exactly what you want when 反思修复 runs on Gemini or Claude:
        those completions are a teacher signal, not policy actions, and routing them
        through the gateway would attribute them to the trained model. Give the
        critic override an explicit ``base_url``/``api_key`` and leave the roles you
        are training on the injected endpoint.
        """
        settings = self.settings
        endpoint_override: dict[str, Any] = {
            "provider": _setting_str(settings, "provider") or "qwen",
            "base_url": "${OPENAI_BASE_URL}/v1"
            if _setting_bool(settings, "append_v1_suffix", default=False)
            else "${OPENAI_BASE_URL}",
            "api_key": "${OPENAI_API_KEY:-EMPTY}",
            "model": self.model_name or _setting_str(settings, "model") or "",
        }
        for key in ("temperature", "top_p", "max_tokens", "timeout", "connect_timeout",
                    "max_retries", "max_concurrency", "native_tool_calls",
                    "endpoint_path", "auth_header_prefix", "extra_headers",
                    "extra_body", "drop_params"):
            if key in settings:
                endpoint_override[key] = settings[key]

        models: dict[str, Any] = {"default": endpoint_override}
        per_role = settings.get("models")
        if isinstance(per_role, Mapping):
            for role in ROLE_NAMES:
                override = per_role.get(role)
                if isinstance(override, Mapping):
                    models[role] = dict(override)

        config: dict[str, Any] = {
            "pipeline": {
                "root": _setting_str(settings, "pipeline_root"),
                "pipeline_config": _setting_str(settings, "pipeline_config") or "config.yaml",
                "tts_service_url": _setting_str(settings, "tts_service_url")
                or "http://127.0.0.1:7865",
            },
            "agent": {
                "agent_mode": "long_horizon",
                "max_steps": int(settings.get("max_steps", 48) or 48),
                "max_same_tool_repeats": int(settings.get("max_same_tool_repeats", 10) or 10),
            },
            "recording": {
                "output_root": self._output_dir(),
                "store_full_prompts": _setting_bool(settings, "store_full_prompts", default=True),
                "save_on_failure": True,
                "strict_tool_schema": _setting_bool(settings, "strict_tool_schema", default=True),
            },
            "models": models,
        }

        for key, section in (
            ("pipeline_config_overrides", ("pipeline", "config_overrides")),
            ("render_split", ("pipeline", "render_split")),
        ):
            value = settings.get(key)
            if isinstance(value, Mapping):
                config[section[0]][section[1]] = dict(value)

        switches = settings.get("tool_switches")
        if isinstance(switches, Mapping):
            config["tool_switches"] = {str(k): bool(v) for k, v in switches.items()}
        return config

    def _run_command(self, *, script_path: str) -> str:
        """Build the rollout command, teeing output into the agent log."""
        flags = [
            "--log-level", _setting_str(self.settings, "log_level") or "INFO",
            "run",
            "--config", shlex.quote(self._config_path),
            "--script-file", shlex.quote(script_path),
            "--output", shlex.quote(f"{self._output_dir()}/episode"),
        ]
        episode_id = _setting_str(self.settings, "episode_id")
        if episode_id:
            flags.extend(["--episode-id", shlex.quote(episode_id)])

        log_path = f"{shlex.quote(RUNTIME_AGENT_LOG_DIR)}/longvideo_polar.txt"
        # Prefer the console script; fall back to the module form when the package
        # was installed somewhere that does not put entry points on PATH.
        return (
            "LVP=longvideo-polar; "
            'command -v longvideo-polar >/dev/null 2>&1 || LVP="python3 -m longvideo_polar.cli"; '
            f"$LVP {' '.join(flags)} 2>&1 </dev/null | tee {log_path}"
        )


def _setting_str(settings: Mapping[str, Any], key: str) -> str:
    value = settings.get(key)
    return str(value).strip() if isinstance(value, str) and value.strip() else ""


def _setting_bool(settings: Mapping[str, Any], key: str, *, default: bool) -> bool:
    value = settings.get(key)
    return default if value is None else bool(value)
