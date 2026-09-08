"""Command-line entry point for longvideo_polar.

Subcommands:

``run``
    One 口播稿 → one 成片 + three trajectory streams. This is also what the Polar
    harness executes inside the runtime.
``batch``
    Many scripts, process-isolated, resumable, aggregated.
``aggregate``
    Re-aggregate an existing rollout directory without re-running anything.
``check``
    Validate the config, show the model mix, and probe each role endpoint. Run this
    first — it catches a bad AICutPipeline path, a missing third-party key or an
    unreachable model before a long rollout does.
``smoke``
    Offline end-to-end verification against scripted local models. No GPU, no TTS
    service, no API key; proves the plumbing including per-provider routing.
``prepare``
    One-time setup for a corpus run: build the consolidated slot-schema cache and
    register the initial machines, so the thousands of rollout subprocesses that
    follow do not each pay for 90 shared-filesystem reads.
``ui``
    Serve the campaign control plane — machine pool, selection, live progress.
``export``
    Aggregate an annotated corpus and write its QC report and dataset card.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from longvideo_polar.batch import BatchRunner, aggregate_episodes
from longvideo_polar.config import DEFAULT_CONFIG_PATH, ROLE_NAMES, LongVideoConfig
from longvideo_polar.roles import ALL_ROLES, ROLE_LABELS_ZH, load_tool_schema
from longvideo_polar.runner import LongVideoRunner

logger = logging.getLogger("longvideo_polar")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="longvideo-polar",
        description=(
            "Capture long-horizon 口播稿→成片 trajectories from AICutPipeline, "
            "split into 主流程 / 内容生成器 / 反思修复 streams."
        ),
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config_arg(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "--config", default=None,
            help=f"Config YAML path (default: $LONGVIDEO_POLAR_CONFIG or {DEFAULT_CONFIG_PATH}).",
        )
        target.add_argument(
            "--profile", default=None,
            help=(
                "Model-mix profile from the config's `profiles` block, e.g. all_qwen "
                "(全 Qwen) or critic_gemini / critic_opus (反思修复走三方模型). "
                "Defaults to $LVP_PROFILE, then annotation_profile."
            ),
        )

    run = sub.add_parser("run", help="Run one rollout.")
    add_config_arg(run)
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--script-file", help="Path to a 口播稿 text file.")
    source.add_argument("--script", help="Inline 口播稿 text.")
    run.add_argument("--output", default=None, help="Output directory for this episode.")
    run.add_argument("--episode-id", default=None, help="Stable episode id.")
    run.add_argument(
        "--pipeline-root", default=None,
        help="Override pipeline.root (the WevisionCut / environment checkout).",
    )
    run.add_argument("--max-steps", type=int, default=None, help="Override agent.max_steps.")
    run.add_argument(
        "--model", default=None,
        help="Override the model name for all three roles.",
    )
    run.add_argument(
        "--base-url", default=None,
        help="Override the OpenAI-compatible base URL for all three roles.",
    )

    batch = sub.add_parser("batch", help="Run a batch of rollouts.")
    add_config_arg(batch)
    batch.add_argument("--input-dir", default=None, help="Override batch.input_dir.")
    batch.add_argument("--glob", default=None, help="Override batch.input_glob.")
    batch.add_argument("--workers", type=int, default=None, help="Override batch.workers.")
    batch.add_argument("--limit", type=int, default=None, help="Override batch.limit.")
    batch.add_argument("--output-root", default=None, help="Override recording.output_root.")
    batch.add_argument(
        "--no-resume", action="store_true",
        help="Re-run episodes that already have an episode.json.",
    )

    agg = sub.add_parser("aggregate", help="Aggregate an existing rollout directory.")
    agg.add_argument("root", help="Directory containing <episode>/episode.json entries.")

    check = sub.add_parser("check", help="Validate config and probe role endpoints.")
    add_config_arg(check)
    check.add_argument(
        "--skip-probe", action="store_true",
        help="Validate config and paths only; do not contact the model endpoints.",
    )

    smoke = sub.add_parser(
        "smoke",
        help="Run the real pipeline against scripted local models and verify the dumps.",
        description=(
            "Stands up fake OpenAI-compatible endpoints on loopback, drives the real "
            "AICutPipeline agent loop, and asserts the three streams, their linkage, "
            "per-provider routing and that no model call went unrecorded. Needs no GPU, "
            "no TTS service and no API key."
        ),
    )
    smoke.add_argument(
        "--pipeline-root", default=None,
        help="WevisionCut / environment checkout (default: pipeline.root from the config).",
    )
    smoke.add_argument("--config", default=None, help="Config used only to locate pipeline.root.")
    smoke.add_argument(
        "--output", default=None,
        help="Where to write smoke artifacts (default: a temp dir, removed afterwards).",
    )
    smoke.add_argument(
        "--scenario", action="append", choices=["all_qwen", "split_critic"], default=None,
        help="Scenario to run; repeatable. Default: both.",
    )
    smoke.add_argument(
        "--keep", action="store_true",
        help="Keep the smoke output directory for inspection.",
    )

    prepare = sub.add_parser(
        "prepare", help="One-time corpus setup: slot-schema cache + machine pool."
    )
    add_config_arg(prepare)
    prepare.add_argument("--output-root", required=True, help="Corpus output directory.")
    prepare.add_argument(
        "--gold-template-root", default=None,
        help="Gold_Template checkout (default: paths.gold_template_root from the pipeline config).",
    )
    prepare.add_argument(
        "--endpoint", action="append", default=None, metavar="IP:PORT",
        help="Machine to register; repeatable.",
    )
    prepare.add_argument(
        "--capacity", type=int, default=4,
        help="Concurrent rollouts per machine (default: 4).",
    )

    ui = sub.add_parser("ui", help="Serve the campaign control plane.")
    add_config_arg(ui)
    ui.add_argument("--dataset-root", required=True, help="Directory holding manifest.jsonl.")
    ui.add_argument("--output-root", required=True, help="Corpus output directory.")
    ui.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0).")
    ui.add_argument("--port", type=int, default=8099, help="Port (default: 8099).")
    ui.add_argument(
        "--endpoint", action="append", default=None, metavar="IP:PORT",
        help="Machine registered on first start; repeatable.",
    )

    export = sub.add_parser("export", help="Aggregate a corpus and write its QC report.")
    export.add_argument("output_root", help="Corpus output directory.")
    export.add_argument(
        "--registry-size", type=int, default=90,
        help="Templates in the Gold_Template registry, for coverage metrics (default: 90).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "batch":
            return _cmd_batch(args)
        if args.command == "aggregate":
            return _cmd_aggregate(args)
        if args.command == "check":
            return _cmd_check(args)
        if args.command == "smoke":
            return _cmd_smoke(args)
        if args.command == "prepare":
            return _cmd_prepare(args)
        if args.command == "ui":
            return _cmd_ui(args)
        if args.command == "export":
            return _cmd_export(args)
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        logger.error("%s: %s", type(exc).__name__, exc)
        if args.log_level == "DEBUG":
            raise
        return 1
    parser.error(f"unknown command {args.command!r}")
    return 2


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    config = LongVideoConfig.load(
        args.config, overrides=_run_overrides(args), profile=args.profile
    )
    script = (
        Path(args.script_file).expanduser().read_text(encoding="utf-8")
        if args.script_file
        else str(args.script or "")
    )
    runner = LongVideoRunner(config)
    result = asyncio.run(runner.run(
        script=script,
        episode_id=args.episode_id,
        output_dir=args.output,
        source=args.script_file or "inline",
    ))
    print(result.summary())
    print(f"artifacts: {result.output_dir}")
    for name in ("orchestrator", "generator", "critic", "interactions", "episode"):
        path = result.artifacts.get(name)
        if path:
            print(f"  {name:<14} {path}")
    for error in result.errors[:10]:
        print(f"  error: {error}", file=sys.stderr)
    return 0 if result.success else 1


def _cmd_batch(args: argparse.Namespace) -> int:
    overrides: dict[str, Any] = {"batch": {}, "recording": {}}
    if args.input_dir is not None:
        overrides["batch"]["input_dir"] = args.input_dir
    if args.glob is not None:
        overrides["batch"]["input_glob"] = args.glob
    if args.workers is not None:
        overrides["batch"]["workers"] = args.workers
    if args.limit is not None:
        overrides["batch"]["limit"] = args.limit
    if args.no_resume:
        overrides["batch"]["resume"] = False
    if args.output_root is not None:
        overrides["recording"]["output_root"] = args.output_root

    config = LongVideoConfig.load(
        args.config, overrides=_prune(overrides), profile=args.profile
    )
    config_path = config.raw.get("_config_path") or args.config or DEFAULT_CONFIG_PATH
    runner = BatchRunner(config, config_path, profile=config.profile)
    report = asyncio.run(runner.run(output_root=args.output_root))
    print(report.summary())
    if report.aggregate_dir:
        print(f"aggregate: {report.aggregate_dir}")
    for item in report.items:
        if item.status == "failed":
            print(f"  failed: {item.script_path} — {item.error[:200]}", file=sys.stderr)
    return 0 if report.failed == 0 else 1


def _cmd_aggregate(args: argparse.Namespace) -> int:
    out_dir = aggregate_episodes(args.root)
    if not out_dir:
        print("nothing to aggregate", file=sys.stderr)
        return 1
    print(out_dir)
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    config = LongVideoConfig.load(args.config, profile=args.profile)
    print(f"config: {config.raw.get('_config_path', '?')}")
    print(f"profile: {config.profile or '(none)'}")

    failures: list[str] = []
    try:
        config.validate()
        print(f"pipeline.root: {config.pipeline.resolved_root()}  OK")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"config: {exc}")
        print(f"config: FAILED — {exc}", file=sys.stderr)

    print("model mix:")
    for role in ALL_ROLES:
        endpoint = config.endpoint_for(role.value)
        print(
            f"  {role.value:<13} {ROLE_LABELS_ZH[role]:<7} "
            f"{endpoint.provider}:{endpoint.model} @ {endpoint.base_url}"
        )
    if config.describe_models()["mixed_providers"]:
        print("  (mixed providers — samples are tagged per provider for traceability)")

    # Resolving the schema strictly proves the role split matches the pipeline.
    try:
        from longvideo_polar.pipeline_bridge import bootstrap_pipeline_paths, load_pipeline_assets

        bootstrap_pipeline_paths(config.pipeline.root)
        schema = load_tool_schema(strict=True)
        print(f"tool schema: {schema.source}  ({len(schema.action_space)} actions)")
        for role in ALL_ROLES:
            tools = schema.tools_for_role(role)
            print(f"  {role.value:<13} {ROLE_LABELS_ZH[role]:<7} {len(tools):>2} tools: {', '.join(tools)}")
        _, prompt, stage_config = load_pipeline_assets(config)
        print(f"system prompt: {len(prompt)} chars")
        print(f"stage table: {len(stage_config.get('allowed_tools', {}))} stages")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"pipeline: {exc}")
        print(f"pipeline: FAILED — {exc}", file=sys.stderr)

    if not args.skip_probe:
        print("endpoint probe:")
        for role in ROLE_NAMES:
            try:
                print(f"  {role:<13} {_probe_endpoint(config.endpoint_for(role))}")
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{role} endpoint: {exc}")
                print(f"  {role:<13} FAILED — {exc}", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} check(s) failed", file=sys.stderr)
        return 1
    print("\nall checks passed")
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    """Pay the one-time costs before thousands of subprocesses each pay them."""
    from longvideo_polar.decoration_policy import GoldTemplateRegistry
    from longvideo_polar.endpoints import EndpointPool

    output_root = Path(args.output_root).expanduser()
    state_dir = output_root / "_state"
    state_dir.mkdir(parents=True, exist_ok=True)

    gold_root = args.gold_template_root
    if not gold_root:
        config = LongVideoConfig.load(args.config, profile=args.profile)
        pipeline_config_path = config.pipeline.resolve(config.pipeline.pipeline_config)
        import yaml

        loaded = yaml.safe_load(pipeline_config_path.read_text(encoding="utf-8")) or {}
        gold_root = str((loaded.get("paths") or {}).get("gold_template_root") or "")
    if not gold_root:
        logger.error("no gold_template_root; pass --gold-template-root")
        return 1

    registry = GoldTemplateRegistry(gold_root)
    if not len(registry):
        logger.error("no templates found under %s", gold_root)
        return 1
    cache_path = registry.build_cache(state_dir / "slot_schema_cache.json")
    print(f"slot schema cache: {len(registry)} templates -> {cache_path}")

    pool = EndpointPool(state_dir / "endpoints.json")
    for address in args.endpoint or []:
        try:
            endpoint = pool.add(address, capacity=args.capacity)
        except ValueError as exc:
            print(f"  skip {address}: {exc}", file=sys.stderr)
            continue
        status = "up" if endpoint.healthy else f"DOWN ({endpoint.last_error[:70]})"
        print(f"  {endpoint.host:<24} {endpoint.model or '(unknown)':<28} {status}")
    if pool.all():
        print(f"machine pool: {len(pool.healthy())}/{len(pool.all())} reachable -> {pool.state_path}")
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:
    """Serve the control plane."""
    try:
        import uvicorn
    except ImportError:
        logger.error(
            "the UI needs the web extra: pip install -e "
            "'data_construct_pipeline[web]'"
        )
        return 1

    from longvideo_polar.webui import create_app

    config = LongVideoConfig.load(args.config, profile=args.profile)
    config_path = config.raw.get("_config_path") or args.config or DEFAULT_CONFIG_PATH

    app = create_app(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        config_path=config_path,
        bootstrap_endpoints=args.endpoint or [],
    )
    print(f"control plane: http://{args.host}:{args.port}")
    print(f"  dataset : {args.dataset_root}")
    print(f"  output  : {args.output_root}")
    print(f"  config  : {config_path}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    """Aggregate a corpus and report what it actually contains."""
    from longvideo_polar.qc import export_dataset

    card = export_dataset(args.output_root, registry_size=args.registry_size)
    aggregate = card["aggregate"]
    coverage = card["quality"]["template_coverage"]
    slots = card["quality"]["slot_quality"]

    print(f"episodes: {aggregate['episode_count']}")
    for role, count in aggregate["samples"].items():
        print(f"  {role:<14} {count}")
    print(
        f"template coverage: {coverage.get('distinct_templates', 0)}"
        f"/{coverage.get('registry_size', 0)}"
        f"  entropy={coverage.get('normalized_entropy', 0):.3f}"
        f"  top5={coverage.get('top5_share', 0) * 100:.1f}%"
    )
    print(
        f"slot verbatim rate: {slots.get('verbatim_rate', 0) * 100:.2f}%"
        f"  (over {slots.get('total_slots', 0)} slots)"
    )
    print(f"report: {Path(args.output_root) / 'qc' / 'qc_report.md'}")
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {"pipeline": {}, "agent": {}, "models": {"default": {}}}
    if args.pipeline_root is not None:
        overrides["pipeline"]["root"] = args.pipeline_root
    if args.max_steps is not None:
        overrides["agent"]["max_steps"] = args.max_steps
    if args.model is not None:
        overrides["models"]["default"]["model"] = args.model
    if args.base_url is not None:
        overrides["models"]["default"]["base_url"] = args.base_url
    return _prune(overrides)


def _prune(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop empty nested sections so they do not shadow file values."""
    pruned: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            nested = _prune(value)
            if nested:
                pruned[key] = nested
        elif value is not None:
            pruned[key] = value
    return pruned


def _probe_endpoint(endpoint: Any) -> str:
    """Send a 1-token completion to confirm the endpoint answers.

    Reuses the client builder so the probe exercises the same auth-header shape and
    proxy handling as the real calls — a probe that passes with different headers
    than production would be worse than no probe at all.
    """
    from longvideo_polar.model_client import build_http_client

    payload: dict[str, Any] = {
        "model": endpoint.model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0.0,
    }
    for key in endpoint.drop_params:
        payload.pop(key, None)
    with build_http_client(endpoint) as client:
        response = client.post(
            endpoint.endpoint_path or "/chat/completions",
            json=payload,
            timeout=min(60.0, endpoint.timeout),
        )
        response.raise_for_status()
        body = response.json()
    served = str(body.get("model") or "") if isinstance(body, dict) else ""
    return f"OK{f' (served: {served})' if served else ''}"


def _cmd_smoke(args: argparse.Namespace) -> int:
    """Run the offline end-to-end verification scenarios."""
    import shutil
    import tempfile

    from longvideo_polar.smoke import run_scenario

    pipeline_root = args.pipeline_root
    if not pipeline_root:
        config = LongVideoConfig.load(args.config)
        pipeline_root = str(config.pipeline.resolved_root())

    scenarios = args.scenario or ["all_qwen", "split_critic"]
    output_root = Path(args.output).expanduser() if args.output else Path(
        tempfile.mkdtemp(prefix="lvp-smoke-")
    )
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"pipeline root: {pipeline_root}")
    print(f"smoke output:  {output_root}\n")

    reports = []
    try:
        for scenario in scenarios:
            print(f"running {scenario} ...", flush=True)
            report = asyncio.run(run_scenario(
                scenario=scenario,
                pipeline_root=pipeline_root,
                output_dir=output_root / scenario,
            ))
            reports.append(report)
            print(report.render() + "\n", flush=True)
    finally:
        if not args.keep and not args.output:
            shutil.rmtree(output_root, ignore_errors=True)

    failed = [report for report in reports if not report.passed]
    if failed:
        print(
            f"{len(failed)}/{len(reports)} scenario(s) failed: "
            + ", ".join(report.scenario for report in failed),
            file=sys.stderr,
        )
        return 1
    print(f"all {len(reports)} scenario(s) passed")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
