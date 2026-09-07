"""Batch discovery, resume behaviour and aggregation."""

from __future__ import annotations

import json

import pytest

from longvideo_polar.batch import BatchRunner, aggregate_episodes
from longvideo_polar.config import LongVideoConfig


def write_episode(root, episode_id: str, *, video: str = "") -> None:
    """Write a minimal but realistic per-episode dump."""
    out = root / episode_id
    traj = out / "trajectories"
    traj.mkdir(parents=True, exist_ok=True)
    (out / "episode.json").write_text(
        json.dumps({
            "episode_id": episode_id,
            "status": "completed" if video else "failed",
            "final_video_path": video,
            "final_video_exists": bool(video),
            "orchestrator_steps": 2,
            "roles": {"orchestrator": {"llm_calls": 2}, "generator": {"llm_calls": 1},
                      "critic": {"llm_calls": 1}},
            "interactions": {"total": 2, "to_generator": 1, "to_critic": 1},
            "reward_summary": {"count": 1, "mean_reward": 0.7},
            "errors": [],
            "elapsed_seconds": 12.0,
        }),
        encoding="utf-8",
    )
    (traj / "orchestrator.jsonl").write_text(
        json.dumps({
            "sft_messages": [{"role": "user", "content": "x"}],
            "num_tool_calls": 2,
            "reward_summary": {"count": 1, "mean_reward": 0.7},
            "reward_scores": [{"reward_score": 0.7, "stage": "quality_repair"}],
        }),
        encoding="utf-8",
    )
    for role, task in (("generator", "segment_article"), ("critic", "self_reflect_plan")):
        (traj / f"{role}.jsonl").write_text(
            json.dumps({
                "samples": [{
                    "messages": [{"role": "user", "content": "p"},
                                 {"role": "assistant", "content": "{}"}],
                    "metadata": {"stage_task": task, "role": role},
                }]
            }),
            encoding="utf-8",
        )
    (traj / "interactions.jsonl").write_text(
        json.dumps({"step": 1, "from_role": "orchestrator", "to_role": "generator"}) + "\n",
        encoding="utf-8",
    )


def test_aggregate_combines_all_three_streams(tmp_path):
    write_episode(tmp_path, "ep_a", video=str(tmp_path / "a.mp4"))
    write_episode(tmp_path, "ep_b")

    out = aggregate_episodes(tmp_path)
    assert out
    agg = tmp_path / "aggregate"

    index = json.loads((agg / "index.json").read_text(encoding="utf-8"))
    assert index["episode_count"] == 2
    assert index["succeeded"] == 1
    assert index["samples"] == {"orchestrator": 2, "generator": 2, "critic": 2}

    for name in ("orchestrator_sft.jsonl", "generator_sft.jsonl", "critic_sft.jsonl"):
        rows = [
            json.loads(line)
            for line in (agg / name).read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        assert len(rows) == 2
        # Every aggregated row is traceable back to its episode.
        assert {row["episode_id"] for row in rows} == {"ep_a", "ep_b"}

    rewards = [
        json.loads(line)
        for line in (agg / "rewards.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert rewards and rewards[0]["episode_id"] in {"ep_a", "ep_b"}


def test_aggregate_returns_empty_when_nothing_present(tmp_path):
    assert aggregate_episodes(tmp_path) == ""


def test_aggregate_skips_unreadable_manifest(tmp_path):
    write_episode(tmp_path, "ep_ok", video=str(tmp_path / "a.mp4"))
    broken = tmp_path / "ep_broken"
    broken.mkdir()
    (broken / "episode.json").write_text("{not json", encoding="utf-8")

    aggregate_episodes(tmp_path)
    index = json.loads((tmp_path / "aggregate" / "index.json").read_text(encoding="utf-8"))
    assert index["episode_count"] == 1


def make_batch_config(tmp_path, input_dir) -> LongVideoConfig:
    return LongVideoConfig.from_dict({
        "pipeline": {"root": str(tmp_path)},
        "models": {"default": {"base_url": "http://x/v1", "model": "m"}},
        "batch": {"input_dir": str(input_dir), "input_glob": "*.txt"},
        "recording": {"output_root": str(tmp_path / "out")},
    })


def test_discover_scripts_sorted_and_limited(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("c.txt", "a.txt", "b.txt"):
        (scripts / name).write_text("稿件", encoding="utf-8")
    (scripts / "ignore.md").write_text("no", encoding="utf-8")

    config = make_batch_config(tmp_path, scripts)
    found = BatchRunner(config, tmp_path / "cfg.yaml").discover_scripts()
    assert [path.name for path in found] == ["a.txt", "b.txt", "c.txt"]

    config.batch.limit = 2
    assert len(BatchRunner(config, tmp_path / "cfg.yaml").discover_scripts()) == 2


def test_discover_scripts_errors_are_actionable(tmp_path):
    config = make_batch_config(tmp_path, tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="input_dir does not exist"):
        BatchRunner(config, tmp_path / "cfg.yaml").discover_scripts()

    empty = tmp_path / "empty"
    empty.mkdir()
    config = make_batch_config(tmp_path, empty)
    with pytest.raises(FileNotFoundError, match="no inputs matched"):
        BatchRunner(config, tmp_path / "cfg.yaml").discover_scripts()

    config.batch.input_dir = ""
    with pytest.raises(ValueError, match="input_dir is not set"):
        BatchRunner(config, tmp_path / "cfg.yaml").discover_scripts()


async def test_resume_skips_completed_episodes_without_spawning(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "ep_done.txt").write_text("稿件", encoding="utf-8")
    (scripts / "ep_todo.txt").write_text("稿件", encoding="utf-8")

    out_root = tmp_path / "out"
    write_episode(out_root, "ep_done", video=str(tmp_path / "v.mp4"))

    config = make_batch_config(tmp_path, scripts)
    runner = BatchRunner(config, tmp_path / "cfg.yaml")

    spawned: list[str] = []

    async def fake_spawn(item):
        spawned.append(item.episode_id)
        write_episode(out_root, item.episode_id, video=str(tmp_path / "v.mp4"))
        return 0, ""

    runner._spawn = fake_spawn  # noqa: SLF001 - test seam
    report = await runner.run(output_root=out_root)

    assert spawned == ["ep_todo"]
    assert report.skipped == 1
    assert report.done == 2
    assert report.failed == 0
    status = json.loads((out_root / "batch_status.json").read_text(encoding="utf-8"))
    assert status["done"] == 2


async def test_resume_retries_an_episode_that_never_produced_a_video(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "ep_half.txt").write_text("稿件", encoding="utf-8")

    out_root = tmp_path / "out"
    write_episode(out_root, "ep_half")  # dumped a trajectory, no video

    config = make_batch_config(tmp_path, scripts)
    runner = BatchRunner(config, tmp_path / "cfg.yaml")

    spawned: list[str] = []

    async def fake_spawn(item):
        spawned.append(item.episode_id)
        write_episode(out_root, item.episode_id, video=str(tmp_path / "v.mp4"))
        return 0, ""

    runner._spawn = fake_spawn  # noqa: SLF001 - test seam
    report = await runner.run(output_root=out_root)

    assert spawned == ["ep_half"]
    assert report.skipped == 0
    assert report.done == 1


async def test_failure_reason_comes_from_the_manifest_not_the_log_tail(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "ep_x.txt").write_text("稿件", encoding="utf-8")

    out_root = tmp_path / "out"
    config = make_batch_config(tmp_path, scripts)
    runner = BatchRunner(config, tmp_path / "cfg.yaml")

    async def fake_spawn(item):
        write_episode(out_root, item.episode_id)
        manifest_path = out_root / item.episode_id / "episode.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["errors"] = ["拼接验证失败 (concat_status=partial)"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return 1, "INFO some noisy log line"

    runner._spawn = fake_spawn  # noqa: SLF001 - test seam
    report = await runner.run(output_root=out_root)

    assert report.failed == 1
    assert report.items[0].error == "拼接验证失败 (concat_status=partial)"


async def test_failed_rollout_is_reported_not_raised(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "ep_bad.txt").write_text("稿件", encoding="utf-8")

    config = make_batch_config(tmp_path, scripts)
    runner = BatchRunner(config, tmp_path / "cfg.yaml")

    async def fake_spawn(item):  # noqa: ARG001
        return 1, "boom"

    runner._spawn = fake_spawn  # noqa: SLF001 - test seam
    report = await runner.run(output_root=tmp_path / "out")

    assert report.failed == 1
    assert report.done == 0
    assert "boom" in report.items[0].error
