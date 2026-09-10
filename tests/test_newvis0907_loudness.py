from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _import_loudness():
    try:
        return import_module("studio.newvis0907.loudness")
    except ModuleNotFoundError as exc:
        raise AssertionError("studio.newvis0907.loudness is missing") from exc


def _target_dict() -> dict[str, float]:
    loudness = _import_loudness()
    return {
        "integrated_lufs": float(loudness.TARGET_INTEGRATED_LUFS),
        "true_peak_dbtp": float(loudness.TARGET_TRUE_PEAK_DBTP),
        "tolerance_lu": float(loudness.TARGET_TOLERANCE_LU),
        "lra_lu": float(loudness.TARGET_LRA_LU),
    }


def _meas(
    integrated: float = -16.0,
    true_peak: float = -1.5,
    lra: float = 6.0,
    thresh: float = -26.0,
    offset: float = 0.0,
) -> dict[str, float | str]:
    return {
        "input_i": float(integrated),
        "input_tp": float(true_peak),
        "input_lra": float(lra),
        "input_thresh": float(thresh),
        "target_offset": float(offset),
        "output_i": float(integrated),
        "output_tp": float(true_peak),
        "output_lra": float(lra),
        "output_thresh": float(thresh),
        "normalization_type": "dynamic",
    }


def _loudness_meta(
    output_i: float = -16.0,
    output_tp: float = -1.5,
    *,
    input_i: float | None = None,
    input_tp: float | None = None,
    status: str = "normalized",
    normalized: bool = True,
    with_output: bool = True,
    with_target: bool = True,
) -> dict:
    payload = {
        "input": _meas(
            integrated=output_i if input_i is None else input_i,
            true_peak=output_tp if input_tp is None else input_tp,
        ),
        "output": _meas(integrated=output_i, true_peak=output_tp) if with_output else None,
        "status": status,
        "normalized": normalized,
    }
    if with_target:
        payload["target"] = _target_dict()
    return payload


def _project_payload() -> dict:
    return {
        "project_id": "proj_loudness",
        "input_text": "demo",
        "final_video_path": "",
        "segments": [{"segment_id": "seg1", "title": "章节"}],
        "shot_plan": [
            {
                "shot_id": "S01",
                "segment_id": "seg1",
                "duration_sec": 4,
                "vocal_script": "第一句旁白",
                "subtitle_text": "第一句旁白",
            },
            {
                "shot_id": "S02",
                "segment_id": "seg1",
                "duration_sec": 4,
                "vocal_script": "第二句旁白",
                "subtitle_text": "第二句旁白",
                "semantic_animations": {"primary": {"type": "spotlight", "phrase": "第二句"}},
            },
        ],
        "shot_assets": {
            "S01": {
                "audio_meta": {"duration_sec": 4.0},
            },
            "S02": {
                "audio_meta": {"duration_sec": 4.0},
            },
        },
    }


def _write_project(tmp: Path):
    project_mod = import_module("studio.project")
    (tmp / "final_state.json").write_text(
        json.dumps(_project_payload(), ensure_ascii=False),
        encoding="utf-8",
    )
    return project_mod.load(tmp)


class ParseLoudnormJsonTest(unittest.TestCase):
    def test_parse_mixed_logs_extracts_last_valid_json(self) -> None:
        loudness = _import_loudness()
        stderr = """
ffmpeg banner
[Parsed_loudnorm_0 @ 0x1] noise before
{
  "input_i": "-21.10",
  "input_tp": "-4.00",
  "input_lra": "4.10",
  "input_thresh": "-31.00",
  "output_i": "-16.90",
  "output_tp": "-1.60",
  "output_lra": "4.00",
  "output_thresh": "-26.90",
  "normalization_type": "dynamic",
  "target_offset": "0.90"
}
more logs in the middle
{
  "input_i": "-16.20",
  "input_tp": "-1.70",
  "input_lra": "5.20",
  "input_thresh": "-26.20",
  "output_i": "-16.10",
  "output_tp": "-1.40",
  "output_lra": "5.10",
  "output_thresh": "-26.10",
  "normalization_type": "dynamic",
  "target_offset": "0.10"
}
"""
        parsed = loudness.parse_loudnorm_json(stderr)
        self.assertAlmostEqual(float(parsed["input_i"]), -16.2, places=3)
        self.assertAlmostEqual(float(parsed["output_tp"]), -1.4, places=3)

    def test_parse_prefers_last_valid_json_when_trailing_block_is_broken(self) -> None:
        loudness = _import_loudness()
        stderr = """
{
  "input_i": "-16.00",
  "input_tp": "-1.40",
  "input_lra": "6.00",
  "input_thresh": "-26.00",
  "output_i": "-16.00",
  "output_tp": "-1.40",
  "output_lra": "6.00",
  "output_thresh": "-26.00",
  "normalization_type": "dynamic",
  "target_offset": "0.00"
}
{
  "input_i": "-12.0",
"""
        parsed = loudness.parse_loudnorm_json(stderr)
        self.assertAlmostEqual(float(parsed["input_i"]), -16.0, places=3)

    def test_parse_falls_back_when_last_json_is_complete_but_missing_measurement_fields(self) -> None:
        loudness = _import_loudness()
        stderr = """
noise
{
  "input_i": "-16.40",
  "input_tp": "-1.80",
  "input_lra": "5.00",
  "input_thresh": "-26.40",
  "output_i": "-16.10",
  "output_tp": "-1.60",
  "output_lra": "4.90",
  "output_thresh": "-26.10",
  "normalization_type": "dynamic",
  "target_offset": "0.10"
}
tail
{
  "normalization_type": "dynamic",
  "target_offset": "0.00"
}
"""
        parsed = loudness.parse_loudnorm_json(stderr)
        self.assertAlmostEqual(float(parsed["input_i"]), -16.4, places=3)
        self.assertAlmostEqual(float(parsed["output_tp"]), -1.6, places=3)

    def test_parse_invalid_json_raises_structured_error(self) -> None:
        loudness = _import_loudness()
        with self.assertRaises(loudness.LoudnessError) as ctx:
            loudness.parse_loudnorm_json("ffmpeg stderr with no valid json")
        self.assertEqual(ctx.exception.code, "invalid_loudnorm_json")

    def test_parse_missing_or_nonfinite_field_raises_structured_error(self) -> None:
        loudness = _import_loudness()
        missing = """
{
  "input_i": "-16.00",
  "input_tp": "-1.20",
  "input_lra": "5.0",
  "output_i": "-16.00",
  "output_tp": "-1.20",
  "output_lra": "5.0",
  "output_thresh": "-26.0",
  "normalization_type": "dynamic",
  "target_offset": "0.0"
}
"""
        with self.assertRaises(loudness.LoudnessError) as ctx_missing:
            loudness.parse_loudnorm_json(missing)
        self.assertEqual(ctx_missing.exception.code, "missing_measurement_field")

        nonfinite = """
{
  "input_i": "-16.00",
  "input_tp": "inf",
  "input_lra": "5.0",
  "input_thresh": "-26.0",
  "output_i": "-16.00",
  "output_tp": "-1.20",
  "output_lra": "5.0",
  "output_thresh": "-26.0",
  "normalization_type": "dynamic",
  "target_offset": "0.0"
}
"""
        with self.assertRaises(loudness.LoudnessError) as ctx_nonfinite:
            loudness.parse_loudnorm_json(nonfinite)
        self.assertEqual(ctx_nonfinite.exception.code, "nonfinite_measurement_field")


class NormalizeNarrationTest(unittest.TestCase):
    def test_already_good_audio_is_skipped(self) -> None:
        loudness = _import_loudness()
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d) / "narration.wav"
            audio.write_bytes(b"original-audio")
            calls: list[list[str]] = []

            def fake_runner(cmd, **kwargs):
                calls.append(list(cmd))
                stderr = json.dumps(_meas(integrated=-16.3, true_peak=-1.7))
                return mock.Mock(returncode=0, stdout="", stderr=stderr)

            result = loudness.normalize_narration(str(audio), runner=fake_runner)
            self.assertEqual(result["status"], "skipped")
            self.assertFalse(result["normalized"])
            self.assertEqual(audio.read_bytes(), b"original-audio")
            self.assertEqual(len(calls), 1)

    def test_already_good_check_uses_input_measurement_not_output_prediction(self) -> None:
        loudness = _import_loudness()
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d) / "narration.wav"
            audio.write_bytes(b"before")
            calls: list[list[str]] = []

            def fake_runner(cmd, **kwargs):
                calls.append([str(x) for x in cmd])
                if cmd[-1] == "-":
                    stderr = json.dumps(
                        {
                            **_meas(integrated=-18.0, true_peak=-0.8),
                            "output_i": -16.0,
                            "output_tp": -1.6,
                        }
                        if len(calls) == 1 else
                        _meas(integrated=-16.0, true_peak=-1.6)
                    )
                    return mock.Mock(returncode=0, stdout="", stderr=stderr)
                Path(cmd[-1]).write_bytes(b"after")
                return mock.Mock(returncode=0, stdout="", stderr=json.dumps(_meas(integrated=-16.0, true_peak=-1.6)))

            result = loudness.normalize_narration(str(audio), runner=fake_runner)
            self.assertEqual(result["status"], "normalized")
            self.assertTrue(result["normalized"])
            self.assertEqual(len(calls), 3)
            self.assertEqual(audio.read_bytes(), b"after")

    def test_two_pass_uses_measured_fields_and_linear_true(self) -> None:
        loudness = _import_loudness()
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d) / "narration.wav"
            audio.write_bytes(b"input")
            calls: list[list[str]] = []

            def fake_runner(cmd, **kwargs):
                calls.append([str(x) for x in cmd])
                if cmd[-1] == "-":
                    stderr = json.dumps(
                        _meas(integrated=-21.2, true_peak=-3.4, lra=7.7, thresh=-31.5, offset=1.1)
                        if len(calls) == 1 else
                        _meas(integrated=-16.1, true_peak=-1.6, lra=7.2, thresh=-26.2, offset=0.0)
                    )
                    return mock.Mock(returncode=0, stdout="", stderr=stderr)
                Path(cmd[-1]).write_bytes(b"normalized")
                stderr = json.dumps(_meas(integrated=-16.1, true_peak=-1.6, lra=7.2, thresh=-26.2, offset=0.0))
                return mock.Mock(returncode=0, stdout="", stderr=stderr)

            result = loudness.normalize_narration(str(audio), runner=fake_runner)
            self.assertEqual(result["status"], "normalized")
            self.assertTrue(result["normalized"])
            second = " ".join(calls[1])
            self.assertIn("I=-16", second)
            self.assertIn("TP=-1.5", second)
            self.assertIn("LRA=11", second)
            self.assertIn("measured_I=-21.2", second)
            self.assertIn("measured_TP=-3.4", second)
            self.assertIn("measured_LRA=7.7", second)
            self.assertIn("measured_thresh=-31.5", second)
            self.assertIn("offset=1.1", second)
            self.assertIn("linear=true", second)

    def test_successful_normalize_atomically_replaces_original(self) -> None:
        loudness = _import_loudness()
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d) / "narration.wav"
            audio.write_bytes(b"before")
            audio.chmod(0o640)
            calls: list[list[str]] = []

            def fake_runner(cmd, **kwargs):
                calls.append([str(x) for x in cmd])
                if cmd[-1] == "-":
                    stderr = json.dumps(
                        _meas(integrated=-20.0, true_peak=-3.0)
                        if len(calls) == 1 else
                        _meas(integrated=-16.0, true_peak=-1.6)
                    )
                    return mock.Mock(returncode=0, stdout="", stderr=stderr)
                out = Path(cmd[-1])
                out.write_bytes(b"after")
                return mock.Mock(returncode=0, stdout="", stderr=json.dumps(_meas(integrated=-16.0, true_peak=-1.6)))

            result = loudness.normalize_narration(str(audio), runner=fake_runner)
            self.assertEqual(result["status"], "normalized")
            self.assertEqual(audio.read_bytes(), b"after")
            self.assertEqual(audio.stat().st_mode & 0o777, 0o640)
            temps = list(audio.parent.glob("narration.loudnorm*.wav"))
            self.assertEqual(temps, [])

    def test_second_pass_out_of_target_preserves_original_and_cleans_temp(self) -> None:
        loudness = _import_loudness()
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d) / "narration.wav"
            audio.write_bytes(b"before")

            def fake_runner(cmd, **kwargs):
                if cmd[-1] == "-":
                    return mock.Mock(returncode=0, stdout="", stderr=json.dumps(_meas(integrated=-20.0, true_peak=-3.0)))
                Path(cmd[-1]).write_bytes(b"after")
                return mock.Mock(returncode=0, stdout="", stderr=json.dumps(_meas(integrated=-18.5, true_peak=-1.2)))

            result = loudness.normalize_narration(str(audio), runner=fake_runner)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["normalized"])
            self.assertEqual(audio.read_bytes(), b"before")
            self.assertEqual(list(audio.parent.glob("narration.loudnorm*.wav")), [])

    def test_timeout_missing_binary_and_returncode_fail_closed(self) -> None:
        loudness = _import_loudness()
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d) / "narration.wav"
            audio.write_bytes(b"before")

            def timeout_runner(cmd, **kwargs):
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=5)

            timeout_result = loudness.normalize_narration(str(audio), runner=timeout_runner)
            self.assertEqual(timeout_result["status"], "failed")
            self.assertEqual(timeout_result["error"]["code"], "ffmpeg_timeout")
            self.assertEqual(audio.read_bytes(), b"before")

            def missing_runner(cmd, **kwargs):
                raise FileNotFoundError("ffmpeg")

            missing_result = loudness.normalize_narration(str(audio), runner=missing_runner)
            self.assertEqual(missing_result["status"], "failed")
            self.assertEqual(missing_result["error"]["code"], "ffmpeg_not_found")
            self.assertEqual(audio.read_bytes(), b"before")

            def bad_return_runner(cmd, **kwargs):
                return mock.Mock(returncode=2, stdout="", stderr="boom")

            bad_return = loudness.normalize_narration(str(audio), runner=bad_return_runner)
            self.assertEqual(bad_return["status"], "failed")
            self.assertEqual(bad_return["error"]["code"], "ffmpeg_failed")
            self.assertEqual(audio.read_bytes(), b"before")

    def test_missing_file_and_invalid_json_return_structured_failure(self) -> None:
        loudness = _import_loudness()
        with tempfile.TemporaryDirectory() as d:
            missing = Path(d) / "missing.wav"
            missing_result = loudness.normalize_narration(str(missing))
            self.assertEqual(missing_result["status"], "failed")
            self.assertEqual(missing_result["error"]["code"], "audio_file_missing")

            audio = Path(d) / "narration.wav"
            audio.write_bytes(b"before")

            def invalid_json_runner(cmd, **kwargs):
                return mock.Mock(returncode=0, stdout="", stderr="not json")

            invalid_result = loudness.normalize_narration(str(audio), runner=invalid_json_runner)
            self.assertEqual(invalid_result["status"], "failed")
            self.assertEqual(invalid_result["error"]["code"], "invalid_loudnorm_json")
            self.assertEqual(audio.read_bytes(), b"before")


class GenerateAudioIntegrationTest(unittest.TestCase):
    def test_run_only_0907_normalizes_and_records_metadata(self) -> None:
        generate_audio = import_module("pipeline.tools.generate_audio")
        _import_loudness()
        with tempfile.TemporaryDirectory() as d:
            shot = {"shot_id": "S01", "vocal_script": "一句旁白"}
            created = Path(d) / "shots" / "S01" / "narration.wav"
            created.parent.mkdir(parents=True, exist_ok=True)
            created.write_bytes(b"tts-audio")

            def fake_run_tts(**kwargs):
                return {
                    "audio_path": kwargs["output_path"],
                    "duration_sec": 3.0,
                    "voice": kwargs["voice"],
                    "backend": "fake",
                    "status": "success",
                    "attempts": [],
                    "speed": kwargs["speed"],
                }

            normalize_calls: list[str] = []

            def fake_normalize(path, **kwargs):
                normalize_calls.append(path)
                return _loudness_meta(output_i=-16.2, output_tp=-1.6, status="normalized")

            with mock.patch.object(generate_audio, "run_tts", fake_run_tts), \
                 mock.patch("studio.newvis0907.loudness.normalize_narration", fake_normalize):
                out_0907 = generate_audio.run(shot, d, {"policy_version": "0907", "tts": {}})
                out_legacy = generate_audio.run(shot, d, {"tts": {}})

            self.assertEqual(len(normalize_calls), 1)
            self.assertIn("loudness", out_0907["audio_meta"])
            self.assertEqual(out_0907["audio_meta"]["loudness"]["status"], "normalized")
            self.assertNotIn("loudness", out_legacy["audio_meta"])

    def test_batch_run_only_0907_normalizes_each_generated_file(self) -> None:
        generate_audio = import_module("pipeline.tools.generate_audio")
        _import_loudness()
        with tempfile.TemporaryDirectory() as d:
            shots = [
                {"shot_id": "S01", "vocal_script": "第一句"},
                {"shot_id": "S02", "vocal_script": "第二句"},
            ]

            def fake_batch(requests, service_url="", timeout_sec=1200):
                out: dict[str, dict] = {}
                for req in requests:
                    out_path = Path(str(req["output_path"]))
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(f"audio-{req['_shot_id']}".encode("utf-8"))
                    out[str(req["_shot_id"])] = {
                        "audio_path": str(out_path),
                        "duration_sec": 2.5,
                        "backend": "qwen3_service_batch",
                        "voice": "fake",
                        "status": "success",
                    }
                return out

            normalized: list[str] = []

            def fake_normalize(path, **kwargs):
                normalized.append(Path(path).name)
                return _loudness_meta(output_i=-16.0, output_tp=-1.6, status="normalized")

            with mock.patch.object(generate_audio, "_tts_via_qwen3_service_batch", fake_batch), \
                 mock.patch.object(generate_audio, "_qwen3_service_health", lambda **kwargs: {"true_batch_inference": True}), \
                 mock.patch("studio.newvis0907.loudness.normalize_narration", fake_normalize):
                out_0907 = generate_audio.batch_run(
                    shots,
                    d,
                    {"policy_version": "0907", "tts": {"backend": "qwen3_service", "dispatch_mode": "batch"}},
                )
                out_legacy = generate_audio.batch_run(
                    shots,
                    d,
                    {"tts": {"backend": "qwen3_service", "dispatch_mode": "batch"}},
                )

            self.assertEqual(len(normalized), 2)
            self.assertTrue(all("loudness" in out_0907[s]["audio_meta"] for s in ("S01", "S02")))
            self.assertTrue(all("loudness" not in out_legacy[s]["audio_meta"] for s in ("S01", "S02")))


class GraphCacheAndStitchIntegrationTest(unittest.TestCase):
    def test_0907_cache_rejects_missing_or_wrong_loudness_target(self) -> None:
        graph = import_module("pipeline.graph")

        shot = {"shot_id": "S01", "vocal_script": "旁白", "visual_query": "图"}
        missing = {"audio_meta": {"text": "旁白", "speed": 1.0}}
        wrong = {"audio_meta": {"text": "旁白", "speed": 1.0, "loudness": _loudness_meta(output_i=-18.0, output_tp=-1.4)}}
        good = {"audio_meta": {"text": "旁白", "speed": 1.0, "loudness": _loudness_meta(output_i=-16.1, output_tp=-1.6)}}
        predicted_only = {
            "audio_meta": {
                "text": "旁白",
                "speed": 1.0,
                "loudness": {
                    "status": "normalized",
                    "normalized": True,
                    "target": _target_dict(),
                    "input": _meas(integrated=-21.0, true_peak=-3.0),
                    "output": {
                        **_meas(integrated=-18.0, true_peak=-0.8),
                        "output_i": -16.0,
                        "output_tp": -1.6,
                    },
                },
            }
        }

        self.assertFalse(graph._audio_cache_loudness_matches(shot, missing, {"policy_version": "0907"}))
        self.assertFalse(graph._audio_cache_loudness_matches(shot, wrong, {"policy_version": "0907"}))
        self.assertFalse(graph._audio_cache_loudness_matches(shot, predicted_only, {"policy_version": "0907"}))
        self.assertTrue(graph._audio_cache_loudness_matches(shot, good, {"policy_version": "0907"}))
        self.assertTrue(graph._audio_cache_loudness_matches(shot, missing, {}))

    def test_stitch_only_measures_final_output_without_normalizing(self) -> None:
        stitch_video = import_module("pipeline.tools.stitch_video")
        _import_loudness()
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            shot = {"shot_id": "S01", "duration_sec": 4.0, "vocal_script": "旁白"}
            rendered = tmp / "shot.mp4"
            audio = tmp / "narration.wav"
            rendered.write_bytes(b"video")
            audio.write_bytes(b"audio")
            def fake_concat(paths, output_path, **kwargs):
                Path(output_path).write_bytes(b"final")
                return {"status": "success", "output_path": output_path, "duration_sec": 4.0}

            measures: list[str] = []

            def fake_measure(path, **kwargs):
                measures.append(path)
                return {"ok": True, "measurement": _meas(integrated=-16.3, true_peak=-1.7), "target": _target_dict()}

            with mock.patch.object(stitch_video, "_get_video_duration", lambda path: 4.0), \
                 mock.patch.object(stitch_video, "_ffmpeg_concat", fake_concat), \
                 mock.patch.object(stitch_video, "_has_audio_stream", lambda path: True), \
                 mock.patch("studio.newvis0907.loudness.measure_loudness", fake_measure), \
                 mock.patch("studio.newvis0907.loudness.normalize_narration", side_effect=AssertionError("must not normalize final output")):
                result = stitch_video.run(
                    [shot],
                    {"S01": {
                        "rendered_path": str(rendered),
                        "audio_path": str(audio),
                        "audio_meta": {"duration_sec": 4.0},
                        "compose_result": {"render_success": True, "render_backend": "fake"},
                    }},
                    str(tmp),
                    bgm_tracks=[],
                    bgm_volume=0.25,
                    duck_under_voice=False,
                    config={"policy_version": "0907", "stitch": {"sanitize_shots": False, "fast_concat_copy": False}},
                )

            self.assertEqual(result["concat_status"], "success")
            self.assertEqual(measures, [result["final_video_path"]])
            self.assertIn("loudness", result)
            self.assertEqual(result["narration_target"], _target_dict())
            self.assertEqual(result["duck_under_voice"], False)
            self.assertEqual(result["bgm_volume"], 0.25)


class VerificationAndQualityPayloadTest(unittest.TestCase):
    def test_tts_edit_fails_verification_when_loudness_out_of_target(self) -> None:
        operations = import_module("studio.newvis0907.operations")
        verification = import_module("studio.newvis0907.verification")
        with tempfile.TemporaryDirectory() as d:
            proj = _write_project(Path(d))
            before = operations.compute_digest(proj)
            ops = [{"op": "set_narration", "shot_id": "S01", "text": "改写后的新旁白"}]
            operations.apply_operations(proj, ops)

            rendered = Path(d) / "rendered_new.mp4"
            final = Path(d) / "final_new.mp4"
            rendered.write_bytes(b"x" * 32)
            final.write_bytes(b"y" * 64)
            proj.shot_assets.setdefault("S01", {})["rendered_path"] = str(rendered)
            proj.shot_assets["S01"]["audio_meta"] = {
                "duration_sec": 5.0,
                "loudness": _loudness_meta(output_i=-18.3, output_tp=-1.4),
            }
            proj.state["final_video_path"] = str(final)

            result = verification.verify_transaction(
                proj,
                before,
                ops,
                render_result={"outputs": [str(rendered)], "final_video_path": str(final)},
            )
            self.assertFalse(result["complete"])
            self.assertTrue(any(c["name"].startswith("loudness:") and not c["ok"] for c in result["checks"]))

    def test_quality_payload_uses_input_measurement_not_output_prediction(self) -> None:
        router_mod = import_module("studio.newvis0907.router")
        state = _project_payload()
        state["shot_assets"]["S01"]["audio_meta"]["loudness"] = {
            "status": "normalized",
            "normalized": True,
            "target": _target_dict(),
            "input": _meas(integrated=-21.0, true_peak=-3.0),
            "output": {
                **_meas(integrated=-18.0, true_peak=-0.8),
                "output_i": -16.0,
                "output_tp": -1.6,
            },
        }
        hierarchy = import_module("studio.newvis0907.hierarchy").build_hierarchy(state)
        payload = router_mod._quality_payload(state, hierarchy)
        self.assertTrue(any("loudness" in str(item.get("code", "")) for item in payload["issues"]), str(payload))

    def test_quality_payload_accepts_normalized_top_level_output_shape(self) -> None:
        router_mod = import_module("studio.newvis0907.router")
        state = _project_payload()
        state["shot_assets"]["S01"]["audio_meta"]["loudness"] = {
            "status": "normalized",
            "normalized": True,
            "target": _target_dict(),
            "input": _meas(integrated=-21.0, true_peak=-3.0),
            "output": {
                **_meas(integrated=-16.1, true_peak=-1.6),
                "output_i": -15.2,
                "output_tp": -1.0,
            },
        }
        hierarchy = import_module("studio.newvis0907.hierarchy").build_hierarchy(state)
        payload = router_mod._quality_payload(state, hierarchy)
        # The unified report now spans every dimension, so we assert specifically
        # that a within-target normalized loudness raises no out-of-target /
        # not-normalized loudness issue for that shot (unrelated dimensions of
        # this tiny fixture — subtitles / pacing / a second shot without a
        # measurement — are out of scope here).
        codes = {str(item.get("code")) for item in payload["issues"]}
        self.assertNotIn("loudness_out_of_target", codes, str(payload))
        self.assertNotIn("loudness_not_normalized", codes, str(payload))

    def test_failed_payload_never_passes_even_if_output_looks_good(self) -> None:
        router_mod = import_module("studio.newvis0907.router")
        state = _project_payload()
        state["shot_assets"]["S01"]["audio_meta"]["loudness"] = {
            "status": "failed",
            "normalized": False,
            "target": _target_dict(),
            "input": _meas(integrated=-21.0, true_peak=-3.0),
            "output": _meas(integrated=-16.0, true_peak=-1.6),
            "error": {"code": "ffmpeg_failed", "message": "mock fail"},
        }
        hierarchy = import_module("studio.newvis0907.hierarchy").build_hierarchy(state)
        payload = router_mod._quality_payload(state, hierarchy)
        self.assertTrue(payload["issues"], str(payload))

    def test_pure_visual_edit_does_not_require_audio_loudness_remeasure(self) -> None:
        operations = import_module("studio.newvis0907.operations")
        verification = import_module("studio.newvis0907.verification")
        with tempfile.TemporaryDirectory() as d:
            proj = _write_project(Path(d))
            before = operations.compute_digest(proj)
            ops = [{"op": "set_emphasis", "shot_id": "S02", "phrase": "第二句", "emphasis_type": "key_point"}]
            operations.apply_operations(proj, ops)

            rendered = Path(d) / "rendered_s02.mp4"
            final = Path(d) / "final_new.mp4"
            rendered.write_bytes(b"x" * 32)
            final.write_bytes(b"y" * 64)
            proj.shot_assets.setdefault("S02", {})["rendered_path"] = str(rendered)
            proj.state["final_video_path"] = str(final)

            result = verification.verify_transaction(
                proj,
                before,
                ops,
                render_result={"outputs": [str(rendered)], "final_video_path": str(final)},
            )
            self.assertTrue(result["complete"], str(result["checks"]))

    def test_quality_payload_reads_shot_loudness_issue(self) -> None:
        router_mod = import_module("studio.newvis0907.router")
        state = _project_payload()
        state["shot_assets"]["S01"]["audio_meta"]["loudness"] = _loudness_meta(output_i=-18.2, output_tp=-1.2)
        hierarchy = import_module("studio.newvis0907.hierarchy").build_hierarchy(state)
        payload = router_mod._quality_payload(state, hierarchy)
        self.assertTrue(any("loudness" in str(item.get("code", "")) for item in payload["issues"]), str(payload))


class OptionalSmokeTest(unittest.TestCase):
    def test_real_ffmpeg_measure_smoke(self) -> None:
        loudness = _import_loudness()
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("ffmpeg not available in PATH")
        with tempfile.TemporaryDirectory() as d:
            wav = Path(d) / "tone.wav"
            proc = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=880:duration=0.15",
                    str(wav),
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if proc.returncode != 0 or not wav.exists():
                self.skipTest("ffmpeg tone generation unavailable")
            result = loudness.measure_loudness(str(wav), ffmpeg_binary=ffmpeg)
            self.assertTrue(result["ok"], str(result))
            self.assertTrue(math.isfinite(float(result["measurement"]["input_i"])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
