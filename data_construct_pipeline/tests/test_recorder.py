"""Trajectory recording, three-stream splitting and persistence."""

from __future__ import annotations

import json

from longvideo_polar.recorder import TrajectoryRecorder, summarize_reward_scores
from longvideo_polar.roles import Role


def make_recorder(tmp_path, **kwargs) -> TrajectoryRecorder:
    return TrajectoryRecorder(
        episode_id="ep_test",
        output_dir=tmp_path,
        task_description="将口播稿转化为完整成片",
        input_script="口播稿正文",
        **kwargs,
    )


def drive_two_steps(recorder: TrajectoryRecorder) -> None:
    """Simulate the loop: orchestrator decides, generator/critic execute."""
    recorder.record_llm(
        role=Role.ORCHESTRATOR,
        system_prompt="SYS",
        user_content='{"task": "script_to_long_video"}',
        raw_output="<think>先分段</think>",
        thinking="先分段",
        tool_name="Tool_SegmentArticle",
        parse_ok=True,
        stage="agent_loop",
        completion_tokens=20,
    )
    recorder.begin_step(1, "Tool_SegmentArticle")
    recorder.record_llm(
        role=Role.GENERATOR,
        system_prompt="SKILL",
        user_content='{"stage_task": "segment_article"}',
        raw_output='{"segments": [{"segment_id": "seg_001"}]}',
        cleaned_output='{"segments": [{"segment_id": "seg_001"}]}',
        parse_ok=True,
        completion_tokens=120,
    )
    recorder.record_tool_call(
        step=1, tool="Tool_SegmentArticle", stage="plan",
        arguments={"num_segments_target": 6},
        output={"status": "success", "num_segments": 1},
        executed_by=Role.GENERATOR,
    )

    recorder.record_llm(
        role=Role.ORCHESTRATOR,
        system_prompt="SYS",
        user_content='{"task": "script_to_long_video"}',
        raw_output="<think>该自检了</think>",
        thinking="该自检了",
        tool_name="Tool_SelfReflect",
        parse_ok=True,
        stage="agent_loop",
    )
    recorder.begin_step(2, "Tool_SelfReflect")
    recorder.record_llm(
        role=Role.CRITIC,
        system_prompt="SKILL",
        user_content='{"stage_task": "self_reflect_plan"}',
        raw_output='{"passed": true, "overall_score": 0.8}',
        cleaned_output='{"passed": true, "overall_score": 0.8}',
        parse_ok=True,
    )
    recorder.record_tool_call(
        step=2, tool="Tool_SelfReflect", stage="reflect_repair_plan",
        arguments={"scope": "full"},
        output={"status": "approved", "overall_score": 0.8},
        executed_by=Role.CRITIC,
    )


def test_orchestrator_call_advances_step_and_skills_inherit_it(tmp_path):
    recorder = make_recorder(tmp_path)
    drive_two_steps(recorder)

    calls = recorder.llm_calls
    assert [call.role for call in calls] == [
        Role.ORCHESTRATOR, Role.GENERATOR, Role.ORCHESTRATOR, Role.CRITIC
    ]
    assert [call.step for call in calls] == [1, 1, 2, 2]
    # The skill call is linked to the tool the orchestrator chose.
    assert calls[1].tool == "Tool_SegmentArticle"
    assert calls[3].tool == "Tool_SelfReflect"


def test_stage_task_is_recovered_from_skill_payload(tmp_path):
    recorder = make_recorder(tmp_path)
    drive_two_steps(recorder)
    tasks = {call.role: call.stage_task for call in recorder.llm_calls if call.role is not Role.ORCHESTRATOR}
    assert tasks[Role.GENERATOR] == "segment_article"
    assert tasks[Role.CRITIC] == "self_reflect_plan"


def test_interactions_capture_orchestrator_handoffs(tmp_path):
    recorder = make_recorder(tmp_path)
    drive_two_steps(recorder)

    interactions = recorder.interactions
    assert len(interactions) == 2
    assert all(item.from_role is Role.ORCHESTRATOR for item in interactions)
    assert [item.to_role for item in interactions] == [Role.GENERATOR, Role.CRITIC]
    assert [item.step for item in interactions] == [1, 2]
    # Orchestrator's own decisions are not handoffs.
    assert not any(item.to_role is Role.ORCHESTRATOR for item in interactions)


def test_orchestrator_stream_is_thought_call_respond(tmp_path):
    recorder = make_recorder(tmp_path)
    drive_two_steps(recorder)

    entries = recorder.build_orchestrator_entries()
    assert entries[0]["role"] == "user"
    assert [entry["role"] for entry in entries[1:]] == [
        "thought", "call", "respond", "thought", "call", "respond",
    ]
    assert entries[2]["tool"] == "Tool_SegmentArticle"
    assert entries[2]["metadata"]["executed_by"] == "generator"
    assert entries[3]["metadata"]["status"] == "success"


def test_orchestrator_sft_messages_use_tool_calls(tmp_path):
    recorder = make_recorder(tmp_path)
    drive_two_steps(recorder)

    messages = recorder.build_orchestrator_sft_messages()
    assert messages[0]["role"] == "system"
    tool_calls = [msg for msg in messages if msg.get("tool_calls")]
    assert len(tool_calls) == 2
    assert tool_calls[0]["tool_calls"][0]["function"]["name"] == "Tool_SegmentArticle"
    assert json.loads(tool_calls[0]["tool_calls"][0]["function"]["arguments"])
    assert [msg for msg in messages if msg["role"] == "tool"]


def test_skill_streams_are_single_turn_samples(tmp_path):
    recorder = make_recorder(tmp_path)
    drive_two_steps(recorder)

    for role, task in ((Role.GENERATOR, "segment_article"), (Role.CRITIC, "self_reflect_plan")):
        samples = recorder.build_skill_samples(role)
        assert len(samples) == 1
        assert [msg["role"] for msg in samples[0]["messages"]] == ["system", "user", "assistant"]
        assert samples[0]["metadata"]["stage_task"] == task
        assert samples[0]["metadata"]["role"] == role.value


def test_store_full_prompts_false_omits_system_message(tmp_path):
    recorder = make_recorder(tmp_path, store_full_prompts=False)
    drive_two_steps(recorder)
    samples = recorder.build_skill_samples(Role.GENERATOR)
    assert [msg["role"] for msg in samples[0]["messages"]] == ["user", "assistant"]


def test_save_writes_all_three_streams_plus_manifest(tmp_path):
    recorder = make_recorder(tmp_path)
    drive_two_steps(recorder)

    paths = recorder.save(result={
        "project_id": "aicut_x",
        "final_video_path": "",
        "stages_completed": ["01:Tool_SegmentArticle", "02:Tool_SelfReflect"],
        "errors": [],
    })
    for key in ("orchestrator", "generator", "critic", "interactions", "llm_calls", "episode"):
        assert key in paths

    traj = tmp_path / "trajectories"
    assert (traj / "orchestrator.jsonl").is_file()
    assert (traj / "generator.jsonl").is_file()
    assert (traj / "critic.jsonl").is_file()
    assert (traj / "sft" / "generator_messages.jsonl").is_file()
    assert (traj / "rl" / "reward_scores.jsonl").is_file()

    generator = json.loads((traj / "generator.jsonl").read_text(encoding="utf-8"))
    assert generator["role"] == "generator"
    assert generator["role_label"] == "内容生成器"
    assert generator["samples_by_stage_task"] == {"segment_article": 1}


def test_manifest_reports_per_role_stats_and_failure(tmp_path):
    recorder = make_recorder(tmp_path)
    drive_two_steps(recorder)
    recorder.record_llm(
        role=Role.GENERATOR, system_prompt="S", user_content="{}",
        raw_output="", parse_ok=False, error="Timeout: boom",
    )
    recorder.save(result={"final_video_path": "", "errors": []})

    manifest = json.loads((tmp_path / "episode.json").read_text(encoding="utf-8"))
    assert manifest["episode_id"] == "ep_test"
    # No final video means the episode failed, regardless of step count.
    assert manifest["status"] == "failed"
    assert manifest["final_video_exists"] is False
    assert manifest["orchestrator_steps"] == 2
    assert manifest["roles"]["generator"]["llm_calls"] == 2
    assert manifest["roles"]["generator"]["errors"] == 1
    assert manifest["roles"]["generator"]["parse_failures"] == 1
    assert manifest["interactions"] == {"total": 3, "to_generator": 2, "to_critic": 1}
    assert manifest["tool_call_counts"] == {"Tool_SegmentArticle": 1, "Tool_SelfReflect": 1}


def test_manifest_collapses_the_same_error_reported_twice(tmp_path):
    """The pipeline broadcasts an error *and* returns it; the prefixed form wins."""
    recorder = make_recorder(tmp_path)
    drive_two_steps(recorder)
    recorder.add_error("[agent_loop] 达到 max_agent_steps=2，Agent 未完成 StitchVideo")
    recorder.add_error("渲染失败")
    recorder.save(result={"errors": [
        "达到 max_agent_steps=2，Agent 未完成 StitchVideo",  # bare duplicate
        "渲染失败",
        "拼接验证失败 (concat_status=partial)",
    ]})

    manifest = json.loads((tmp_path / "episode.json").read_text(encoding="utf-8"))
    assert manifest["errors"] == [
        "[agent_loop] 达到 max_agent_steps=2，Agent 未完成 StitchVideo",
        "渲染失败",
        "拼接验证失败 (concat_status=partial)",
    ]


def test_manifest_marks_success_when_video_exists(tmp_path):
    video = tmp_path / "final.mp4"
    video.write_bytes(b"x" * 2048)
    recorder = make_recorder(tmp_path)
    drive_two_steps(recorder)
    recorder.save(result={"final_video_path": str(video), "errors": []})

    manifest = json.loads((tmp_path / "episode.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["final_video_exists"] is True


def test_merge_agent_trace_upgrades_compacted_output(tmp_path):
    recorder = make_recorder(tmp_path)
    recorder.record_tool_call(
        step=1, tool="Tool_SegmentArticle", stage="plan",
        arguments={}, output={"status": "success"},
    )
    changed = recorder.merge_agent_trace([
        {
            "step": 1,
            "tool_name": "Tool_SegmentArticle",
            "arguments": {"num_segments_target": 8},
            "output": {"status": "success", "segments": [{"segment_id": "seg_001"}]},
        },
        # skill_raw_output rows carry no step decision and must be ignored.
        {"event": "skill_raw_output", "tool": "Tool_SegmentArticle", "raw_output": "..."},
    ])
    assert changed == 1
    record = recorder.tool_calls[0]
    assert record.arguments == {"num_segments_target": 8}
    assert record.output["segments"] == [{"segment_id": "seg_001"}]


def test_merge_agent_trace_adds_steps_missing_from_broadcast(tmp_path):
    recorder = make_recorder(tmp_path)
    recorder.merge_agent_trace([
        {"step": 3, "tool_name": "Tool_StitchVideo", "arguments": {}, "output": {"status": "success"}},
    ])
    assert [record.tool for record in recorder.tool_calls] == ["Tool_StitchVideo"]
    assert recorder.tool_calls[0].step == 3


def test_reward_scores_extracted_from_tool_outputs(tmp_path):
    recorder = make_recorder(tmp_path)
    recorder.record_tool_call(
        step=1, tool="Tool_Stage4DimensionJudge", stage="quality_repair",
        arguments={},
        output={
            "status": "success",
            "shots": {"shot_001": {"shot_id": "shot_001", "overall_score": 0.75}},
        },
    )
    recorder.save(result={})
    rows = [
        json.loads(line)
        for line in (tmp_path / "trajectories" / "rl" / "reward_scores.jsonl")
        .read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert rows, "expected at least one reward row"
    assert any(abs(float(row["reward_score"]) - 0.75) < 1e-6 for row in rows)


def test_summarize_reward_scores_handles_empty_and_populated():
    assert summarize_reward_scores([])["count"] == 0
    summary = summarize_reward_scores([
        {"reward_score": 0.5, "stage": "a"},
        {"reward_score": 1.0, "stage": "a"},
        {"reward_score": 0.0, "stage": "b"},
    ])
    assert summary["count"] == 3
    assert summary["mean_reward"] == 0.5
    assert summary["by_stage"]["a"]["mean_reward"] == 0.75


def test_raw_output_truncation_respects_limit(tmp_path):
    recorder = make_recorder(tmp_path, max_raw_output_chars=50)
    recorder.record_llm(
        role=Role.GENERATOR, system_prompt="S",
        user_content="u", raw_output="x" * 500,
    )
    recorder.save(result={})
    line = (tmp_path / "trajectories" / "llm_calls.jsonl").read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert len(payload["raw_output"]) < 500
    assert "truncated" in payload["raw_output"]
