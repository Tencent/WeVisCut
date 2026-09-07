"""Self-contained smoke test for the whole rollout path.

The point of this module is to answer "is the wiring actually intact?" without a
GPU, a served model, a TTS service, or a third-party API key. It stands up scripted
OpenAI-compatible servers on loopback, runs the **real** AICutPipeline long-horizon
agent loop against them, and then asserts on what landed on disk.

Two scenarios:

``all_qwen``
    One fake endpoint serves all three roles, mirroring a single served model with
    ``user.role`` routing.
``split_critic``
    A second endpoint serves 反思修复 only, with the awkward settings real Gemini /
    Claude gateways need (bare auth header, no ``top_p``, no native tool calls).
    This is what proves the Gemini/Opus path works — including that critic traffic
    never leaks to the Qwen endpoint and that samples are tagged per provider.

The strongest assertion here is call conservation: the fake servers count every
request they answer, and that count must equal the number of recorded LLM calls. If
the recorder ever silently dropped a completion, that check fails.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from longvideo_polar.config import LongVideoConfig
from longvideo_polar.roles import Role

logger = logging.getLogger(__name__)

SMOKE_SCRIPT = (
    "今天聊一个反常识的事实：决定一支短片好不好看的，往往不是画面，而是节奏。"
    "观众在前三秒就会决定要不要留下来，所以开头必须给出一个具体的钩子，而不是铺垫。"
    "接下来是三段论：先给冲突，再给证据，最后给一个可以立刻用的结论。"
    "如果你把这三段各控制在二十秒以内，整条片子就会一直往前推，不会塌下来。"
    "最后提醒一句：节奏不是快，而是让每一秒都有新信息。"
)

# The shortest path that exercises all three roles through real pipeline dispatch.
# The media stages need TTS/ffmpeg/retrieval and are switched off in smoke configs.
DEFAULT_TOOL_PLAN: tuple[str, ...] = (
    "Tool_SegmentArticle",
    "Tool_PlanShots",
    "Tool_SelfReflect",
)


# ---------------------------------------------------------------------------
# Scripted OpenAI-compatible server
# ---------------------------------------------------------------------------


@dataclass
class RecordedRequest:
    """One request as the fake server saw it, for wire-level assertions."""

    path: str
    model: str
    role: str
    stage_task: str
    has_tools: bool
    tool_choice: str
    params: list[str]
    auth_header: str
    extra_header_seen: str


class FakeModelServer:
    """A scripted chat-completions endpoint that plays all three roles.

    Role detection matches how the pipeline actually calls a model: an orchestrator
    request carries ``tools``, while a skill request carries a JSON payload whose
    ``role`` field the model client injected.
    """

    def __init__(
        self,
        *,
        model: str,
        tool_plan: tuple[str, ...] = DEFAULT_TOOL_PLAN,
        serve_roles: tuple[str, ...] = ("orchestrator", "generator", "critic"),
        extra_header_name: str = "",
    ) -> None:
        self.model = model
        self.tool_plan = tool_plan
        self.serve_roles = serve_roles
        self.extra_header_name = extra_header_name
        self.base_url = ""
        self.requests: list[RecordedRequest] = []
        self._lock = threading.Lock()
        self._step = 0
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle -------------------------------------------------------

    def start(self) -> str:
        """Start serving on an ephemeral loopback port and return the base URL."""
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: Any) -> None:  # noqa: N802 - stdlib hook
                return  # keep smoke output readable

            def do_POST(self) -> None:  # noqa: N802 - stdlib hook
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    payload = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    payload = {}
                status, body = outer._handle(self.path, payload, dict(self.headers))
                raw = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, port = self._server.server_address[:2]
        self.base_url = f"http://{host}:{port}/v1"
        return self.base_url

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "FakeModelServer":
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()

    # -- request handling ------------------------------------------------

    def count(self, role: str | None = None) -> int:
        with self._lock:
            if role is None:
                return len(self.requests)
            return sum(1 for item in self.requests if item.role == role)

    def snapshot(self) -> list[RecordedRequest]:
        with self._lock:
            return list(self.requests)

    def _handle(
        self, path: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> tuple[int, dict[str, Any]]:
        user_content = ""
        for message in payload.get("messages") or []:
            if isinstance(message, dict) and message.get("role") == "user":
                user_content = str(message.get("content") or "")
        skill = _safe_json(user_content)
        has_tools = bool(payload.get("tools"))
        role = "orchestrator" if has_tools else str(skill.get("role") or "orchestrator")
        stage_task = str(skill.get("stage_task") or ("agent_loop" if has_tools else ""))

        with self._lock:
            self.requests.append(RecordedRequest(
                path=path,
                model=str(payload.get("model") or ""),
                role=role,
                stage_task=stage_task,
                has_tools=has_tools,
                tool_choice=str(payload.get("tool_choice") or ""),
                params=sorted(payload),
                auth_header=str(headers.get("Authorization") or ""),
                extra_header_seen=(
                    str(headers.get(self.extra_header_name) or "")
                    if self.extra_header_name else ""
                ),
            ))

        if role not in self.serve_roles:
            return 403, {"error": {"message": f"role {role} not served by {self.model}"}}
        if role == "orchestrator":
            return 200, self._orchestrator_response()
        return 200, _completion(_skill_output(role, stage_task, skill), model=self.model)

    def _orchestrator_response(self) -> dict[str, Any]:
        with self._lock:
            index = min(self._step, len(self.tool_plan) - 1)
            self._step += 1
        tool = self.tool_plan[index]
        arguments = {"Tool_SegmentArticle": {"num_segments_target": 3}}.get(tool, {})
        return {
            "choices": [{
                "message": {
                    "content": f"<think>step {index + 1}: 选择 {tool}</think>",
                    "tool_calls": [{
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {
                            "name": tool,
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "model": self.model,
            "usage": {"prompt_tokens": 128, "completion_tokens": 24},
        }


def _completion(content: str, *, model: str) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "model": model,
        "usage": {"prompt_tokens": 256, "completion_tokens": max(1, len(content) // 4)},
    }


def _safe_json(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    if not stripped.startswith("{"):
        return {}
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


# ---------------------------------------------------------------------------
# Canned skill outputs, shaped like the pipeline's parsers expect
# ---------------------------------------------------------------------------


def _skill_output(role: str, stage_task: str, payload: dict[str, Any]) -> str:
    """Return canned JSON for one skill call.

    Content is derived from the incoming payload (coverage units, segment anchors)
    rather than hardcoded, so downstream preconditions such as "all segments
    planned" are satisfied for real instead of being faked around.
    """
    if stage_task == "segment_article":
        return json.dumps(_segments_output(payload), ensure_ascii=False)
    if stage_task == "plan_shots":
        return json.dumps(_shots_output(payload), ensure_ascii=False)
    if stage_task == "repair_plan":
        return json.dumps({"shots": _existing_shots(payload)}, ensure_ascii=False)
    if stage_task == "self_reflect_plan":
        return json.dumps({
            "summary": "整体节奏可用，两处镜头时长偏长。",
            "passed": False,
            "overall_score": 0.72,
            "issues": [{
                "shot_id": "shot_001",
                "dimension": "pacing",
                "severity": "medium",
                "detail": "首镜信息量不足，建议压缩到 4 秒内。",
            }],
            "fix_options": ["缩短首镜时长", "把结论句提前"],
        }, ensure_ascii=False)
    if stage_task == "quality_judge":
        return json.dumps({
            "passed": True, "overall_score": 0.86, "failed_dimensions": [],
        }, ensure_ascii=False)
    if role == Role.CRITIC.value:
        return json.dumps(
            {"passed": True, "overall_score": 0.8, "issues": []}, ensure_ascii=False
        )
    return json.dumps({"status": "ok", "shots": []}, ensure_ascii=False)


def _units(payload: dict[str, Any]) -> list[dict[str, Any]]:
    memory = payload.get("memory") if isinstance(payload.get("memory"), dict) else {}
    digest = memory.get("article_digest") if isinstance(memory.get("article_digest"), dict) else {}
    for source in (digest.get("source_units"), memory.get("coverage_units")):
        if isinstance(source, list) and source:
            return [item for item in source if isinstance(item, dict)]
    text = str(payload.get("input_script") or "")
    sentences = [part.strip() for part in re.split(r"(?<=[。！？!?])\s*", text) if part.strip()]
    return [
        {"unit_id": f"u{index + 1:02d}", "text": sentence}
        for index, sentence in enumerate(sentences)
    ]


def _segments_output(payload: dict[str, Any]) -> dict[str, Any]:
    units = _units(payload)
    constraints = payload.get("constraints") if isinstance(payload.get("constraints"), dict) else {}
    target = max(1, int(constraints.get("target_num_segments") or 3))
    target = min(target, max(1, len(units)))
    per_segment = max(1, -(-len(units) // target)) if units else 1

    segments: list[dict[str, Any]] = []
    for index in range(target):
        chunk = units[index * per_segment:(index + 1) * per_segment]
        if not chunk and index:
            break
        text = "".join(str(unit.get("text") or "") for unit in chunk)
        segments.append({
            "segment_id": f"seg_{index + 1:03d}",
            "structure_label": ["开场", "展开", "收束"][min(index, 2)],
            "title": f"第 {index + 1} 段",
            "summary": text[:60] or "内容摘要",
            "narration_script": text or "口播内容",
            "raw_span": text,
            "key_points": [text[:20] or "要点"],
            "coverage_unit_ids": [str(unit.get("unit_id") or "") for unit in chunk],
            "planner_strategy": "balanced",
        })
    return {
        "intent": "讲清短片节奏的三段论方法",
        "music_style": "upbeat_electronic",
        "segments": segments,
    }


def _shots_output(payload: dict[str, Any]) -> dict[str, Any]:
    memory = payload.get("memory") if isinstance(payload.get("memory"), dict) else {}
    anchors = memory.get("segment_anchors") or memory.get("segments") or []
    segment_ids = [
        str(item.get("segment_id"))
        for item in anchors if isinstance(item, dict) and item.get("segment_id")
    ] or ["seg_001"]
    units = _units(payload)

    shots: list[dict[str, Any]] = []
    for index, segment_id in enumerate(segment_ids):
        unit = units[index] if index < len(units) else {}
        text = str(unit.get("text") or "镜头口播内容")
        shots.append({
            "shot_id": f"shot_{index + 1:03d}",
            "segment_id": segment_id,
            "chapter_title": f"第 {index + 1} 段",
            "vocal_script": text,
            "visual_query": f"{text[:12]} 相关画面",
            "is_spotlight": index == 0,
            "duration_sec": 5.0,
            "coverage_unit_ids": [str(unit.get("unit_id") or f"u{index + 1:02d}")],
        })
    return {"shots": shots, "coverage_report": {"covered_ratio": 1.0}}


def _existing_shots(payload: dict[str, Any]) -> list[dict[str, Any]]:
    memory = payload.get("memory") if isinstance(payload.get("memory"), dict) else {}
    for key in ("shot_plan", "current_plan", "shots"):
        value = memory.get(key) or payload.get(key)
        if isinstance(value, list) and value:
            return [item for item in value if isinstance(item, dict)]
    return _shots_output(payload)["shots"]


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------


@dataclass
class SmokeReport:
    """Outcome of one smoke scenario."""

    scenario: str
    passed: bool = False
    checks: list[tuple[bool, str]] = field(default_factory=list)
    output_dir: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)

    def add(self, ok: bool, label: str) -> None:
        self.checks.append((bool(ok), label))

    @property
    def failures(self) -> list[str]:
        return [label for ok, label in self.checks if not ok]

    def render(self) -> str:
        lines = [f"[{self.scenario}]"]
        lines.extend(f"  {'PASS' if ok else 'FAIL'}  {label}" for ok, label in self.checks)
        verdict = "PASSED" if self.passed else f"FAILED ({len(self.failures)} check(s))"
        lines.append(f"  => {verdict}")
        return "\n".join(lines)


def build_smoke_config(
    *,
    pipeline_root: str | Path,
    output_root: str | Path,
    qwen_base_url: str,
    critic_base_url: str = "",
    qwen_model: str = "smoke-qwen",
    critic_model: str = "smoke-critic",
    critic_provider: str = "gemini",
    max_steps: int = len(DEFAULT_TOOL_PLAN),
) -> LongVideoConfig:
    """Build a config that runs the real pipeline with media tools switched off."""
    models: dict[str, Any] = {
        "default": {
            "provider": "qwen",
            "base_url": qwen_base_url,
            "api_key": "EMPTY",
            "model": qwen_model,
            "max_retries": 1,
            "timeout": 60.0,
            "max_concurrency": 2,
        }
    }
    if critic_base_url:
        # Deliberately awkward, the way real third-party gateways are.
        models["critic"] = {
            "provider": critic_provider,
            "base_url": critic_base_url,
            "api_key": "smoke-critic-key",
            "model": critic_model,
            "auth_header_prefix": "",
            "native_tool_calls": False,
            "drop_params": ["top_p"],
            "extra_headers": {"X-Smoke-Provider": critic_provider},
            "max_retries": 1,
            "timeout": 60.0,
            "max_concurrency": 2,
        }

    root = Path(pipeline_root)
    if (root / "config.yaml").is_file():
        pipeline_config = "config.yaml"
    elif (root / "data_construct_pipeline" / "config" / "pipeline_runtime.yaml").is_file():
        pipeline_config = "data_construct_pipeline/config/pipeline_runtime.yaml"
    else:
        pipeline_config = "config/pipeline_runtime.yaml"

    return LongVideoConfig.from_dict({
        "pipeline": {
            "root": str(pipeline_root),
            "pipeline_config": pipeline_config,
            "render_split": {"enabled": False},
        },
        "models": models,
        "agent": {"agent_mode": "long_horizon", "max_steps": max_steps},
        "tool_switches": {
            "tts": False,
            "visual_retrieval": False,
            "bgm_search": False,
            "timeline_compose": False,
            "hyperframes_render": False,
            "final_stitch": False,
        },
        "recording": {
            "output_root": str(output_root),
            "save_on_failure": True,
            "strict_tool_schema": True,
        },
    })


async def run_scenario(
    *,
    scenario: str,
    pipeline_root: str | Path,
    output_dir: str | Path,
    script: str = SMOKE_SCRIPT,
) -> SmokeReport:
    """Run one smoke scenario end to end and check the artifacts.

    Args:
        scenario: ``all_qwen`` or ``split_critic``.
        pipeline_root: AICutPipeline checkout.
        output_dir: Directory for this scenario's episode.
        script: 口播稿 text to annotate.

    Returns:
        A :class:`SmokeReport`; inspect ``passed`` and ``failures``.
    """
    from longvideo_polar.runner import LongVideoRunner

    report = SmokeReport(scenario=scenario, output_dir=str(output_dir))
    split = scenario == "split_critic"

    qwen = FakeModelServer(
        model="smoke-qwen",
        serve_roles=("orchestrator", "generator") if split
        else ("orchestrator", "generator", "critic"),
    )
    critic = FakeModelServer(
        model="smoke-critic",
        serve_roles=("critic",),
        extra_header_name="X-Smoke-Provider",
    ) if split else None

    qwen_url = qwen.start()
    critic_url = critic.start() if critic else ""
    try:
        config = build_smoke_config(
            pipeline_root=pipeline_root,
            output_root=Path(output_dir).parent,
            qwen_base_url=qwen_url,
            critic_base_url=critic_url,
        )
        result = await LongVideoRunner(config).run(
            script=script, episode_id=scenario, output_dir=output_dir, source="smoke",
        )
    finally:
        qwen.stop()
        if critic:
            critic.stop()

    report.manifest = result.manifest
    _check_artifacts(report, Path(output_dir), result)
    _check_call_conservation(report, Path(output_dir), qwen, critic)
    if split:
        _check_provider_split(report, Path(output_dir), qwen, critic)
    report.passed = not report.failures
    return report


def _check_artifacts(report: SmokeReport, output_dir: Path, result: Any) -> None:
    """Assert the three streams, their linkage, and the manifest."""
    traj = output_dir / "trajectories"
    manifest = result.manifest

    report.add((output_dir / "episode.json").is_file(), "episode.json written")
    report.add(bool(manifest), "manifest readable")

    orchestrator = _read_json(traj / "orchestrator.jsonl")
    steps = int(orchestrator.get("num_tool_calls") or 0)
    report.add(steps >= 2, f"主流程 executed >=2 steps (got {steps})")

    tool_messages = [
        message for message in orchestrator.get("sft_messages", [])
        if isinstance(message, dict) and message.get("tool_calls")
    ]
    report.add(
        len(tool_messages) == steps,
        f"主流程 SFT has one tool_call per step ({len(tool_messages)}/{steps})",
    )
    tools_run = set(manifest.get("tool_call_counts", {}))
    report.add(
        {"Tool_SegmentArticle", "Tool_PlanShots"} <= tools_run,
        f"分段+镜头规划 both dispatched (ran {sorted(tools_run)})",
    )

    generator = _read_json(traj / "generator.jsonl")
    gen_tasks = set(generator.get("samples_by_stage_task", {}))
    report.add(
        int(generator.get("num_samples") or 0) >= 2,
        f"内容生成器 produced >=2 samples (got {generator.get('num_samples')})",
    )
    report.add(
        {"segment_article", "plan_shots"} <= gen_tasks,
        f"内容生成器 covered segment_article+plan_shots (got {sorted(gen_tasks)})",
    )

    critic_stream = _read_json(traj / "critic.jsonl")
    critic_samples = int(critic_stream.get("num_samples") or 0)
    report.add(critic_samples >= 1, f"反思修复 produced >=1 sample (got {critic_samples})")
    report.add(
        "self_reflect_plan" in set(critic_stream.get("samples_by_stage_task", {})),
        "反思修复 covered self_reflect_plan",
    )

    for role, stream in (("generator", generator), ("critic", critic_stream)):
        samples = stream.get("samples") or []
        well_formed = all(
            isinstance(sample, dict)
            and [message.get("role") for message in sample.get("messages", [])][-2:]
            == ["user", "assistant"]
            and str(sample["messages"][-1].get("content") or "").strip()
            for sample in samples
        )
        report.add(bool(samples) and well_formed, f"{role} samples are usable single-turn SFT rows")
        # Skill outputs here are valid JSON by construction, so anything marked
        # unparseable means the parse verdict itself is wrong — and a wrong verdict
        # would quietly delete these rows from any parse_ok-filtered training set.
        unparsed = [
            sample.get("metadata", {}).get("stage_task", "?")
            for sample in samples if not sample.get("metadata", {}).get("parse_ok")
        ]
        report.add(not unparsed, f"{role} samples marked parse_ok (bad: {unparsed})")

    interactions = _read_lines(traj / "interactions.jsonl")
    report.add(bool(interactions), "interactions recorded")
    linked = {row.get("to_role") for row in interactions}
    report.add(
        {"generator", "critic"} <= linked,
        f"主流程 handed off to both skills (saw {sorted(linked)})",
    )
    steps_seen = {row.get("step") for row in interactions}
    report.add(
        bool(steps_seen) and all(isinstance(step, int) and step >= 1 for step in steps_seen),
        f"every handoff is attributed to a real step (steps={sorted(steps_seen)})",
    )
    report.add(
        bool(interactions) and all(row.get("tool") for row in interactions),
        "every handoff names its triggering tool",
    )

    report.add(
        bool(orchestrator.get("reward_scores")),
        f"reward scores extracted ({len(orchestrator.get('reward_scores') or [])} rows)",
    )
    report.add(
        bool(manifest.get("metadata", {}).get("models", {}).get("roles")),
        "per-role model provenance in manifest",
    )

    # The run is expected to stop at the step cap (media tools are off, so it never
    # reaches StitchVideo). Any *other* error means something genuinely broke.
    unexpected = [
        message for message in manifest.get("errors", [])
        if "max_agent_steps" not in str(message)
    ]
    report.add(not unexpected, f"no unexpected errors (got {unexpected})")


def _check_call_conservation(
    report: SmokeReport,
    output_dir: Path,
    qwen: FakeModelServer,
    critic: FakeModelServer | None,
) -> None:
    """Every completion the servers answered must appear in the recorder's log.

    This is the no-silent-loss guarantee: a mismatch means some code path reached a
    model without going through the recording client.
    """
    served = qwen.count() + (critic.count() if critic else 0)
    recorded = _read_lines(output_dir / "trajectories" / "llm_calls.jsonl")
    report.add(
        served > 0 and served == len(recorded),
        f"all model calls recorded ({len(recorded)} recorded / {served} served)",
    )
    report.add(
        bool(recorded) and all(row.get("provider") and row.get("model") for row in recorded),
        "every recorded call carries provider + model",
    )
    report.add(
        bool(recorded) and all(int(row.get("step") or 0) >= 1 for row in recorded),
        "every recorded call is attributed to a step",
    )


def _check_provider_split(
    report: SmokeReport,
    output_dir: Path,
    qwen: FakeModelServer,
    critic: FakeModelServer | None,
) -> None:
    """Assert 反思修复 ran on the second provider and nothing leaked across."""
    if critic is None:
        report.add(False, "critic endpoint missing")
        return

    critic_requests = critic.snapshot()
    report.add(bool(critic_requests), f"critic endpoint received traffic ({len(critic_requests)})")
    report.add(
        all(item.role == Role.CRITIC.value for item in critic_requests),
        "critic endpoint saw only critic-role calls",
    )
    report.add(
        qwen.count(Role.CRITIC.value) == 0,
        f"no critic call leaked to the Qwen endpoint ({qwen.count(Role.CRITIC.value)} leaked)",
    )
    report.add(
        all(item.model == "smoke-critic" for item in critic_requests),
        "critic requests used the critic model name",
    )
    # Gateway compatibility knobs must actually take effect on the wire.
    report.add(
        all("top_p" not in item.params for item in critic_requests),
        "drop_params removed top_p from critic requests",
    )
    report.add(
        all(item.auth_header == "smoke-critic-key" for item in critic_requests),
        "auth_header_prefix='' sent the bare key (no 'Bearer ')",
    )
    report.add(
        all(item.extra_header_seen == "gemini" for item in critic_requests),
        "extra_headers reached the critic endpoint",
    )
    report.add(
        all(not item.has_tools for item in critic_requests),
        "critic requests carry no tool schema",
    )

    samples = _read_json(output_dir / "trajectories" / "critic.jsonl").get("samples") or []
    providers = {
        str((sample.get("metadata") or {}).get("provider") or "") for sample in samples
    }
    report.add(
        providers == {"gemini"},
        f"critic samples tagged provider=gemini (got {sorted(providers)})",
    )
    gen_providers = {
        str((sample.get("metadata") or {}).get("provider") or "")
        for sample in (_read_json(output_dir / "trajectories" / "generator.jsonl").get("samples") or [])
    }
    report.add(
        gen_providers == {"qwen"},
        f"generator samples stayed provider=qwen (got {sorted(gen_providers)})",
    )
    roles = report.manifest.get("roles", {})
    report.add(
        roles.get("critic", {}).get("providers_used") == ["gemini"]
        and roles.get("generator", {}).get("providers_used") == ["qwen"],
        "manifest records the observed per-role providers",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows
