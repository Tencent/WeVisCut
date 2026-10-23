"""Stage-1 tool entry points backed by the external DataPipeline runtime.

These five tools are only reachable in data-pipeline mode, which reproduces the
training-data preparation flow. They require a separate `data_pipeline` checkout
pointed at by `WEVISCUT_PIPELINE_ROOT`; that checkout is not part of this
repository and the studio's generation path never calls into it.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import traceback
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from tqdm import tqdm


AICUT_PIPELINE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEVISCUT_PIPELINE_ROOT = Path(os.environ.get(
    "WEVISCUT_PIPELINE_ROOT",
    str(AICUT_PIPELINE_ROOT / "external_tool" / "data_pipeline"),
))


def _prepare_external_data_pipeline() -> None:
    """Make the external `data_pipeline` package win over any local namesake."""
    local_root = str(AICUT_PIPELINE_ROOT)
    external_root = str(DEFAULT_WEVISCUT_PIPELINE_ROOT)

    sys.path[:] = [p for p in sys.path if p not in {"", local_root}]
    if external_root in sys.path:
        sys.path.remove(external_root)
    sys.path.insert(0, external_root)

    loaded = sys.modules.get("data_pipeline")
    loaded_file = str(getattr(loaded, "__file__", "")) if loaded else ""
    if loaded_file and not loaded_file.startswith(external_root):
        for name in list(sys.modules):
            if name == "data_pipeline" or name.startswith("data_pipeline."):
                sys.modules.pop(name, None)


def _load_cfg(config_path: str | None = None) -> Any:
    _prepare_external_data_pipeline()
    from data_pipeline.utils.config import PipelineConfig

    cfg = PipelineConfig.load(Path(config_path).expanduser().resolve())
    storage_root = os.getenv("WEVISCUT_STORAGE_ROOT")
    if storage_root:
        cfg.paths.storage_root = storage_root

    qwen_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("WEVISCUT_QWEN_BASE_URL")
    qwen_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("WEVISCUT_QWEN_API_KEY")
    qwen_model = os.getenv("WEVISCUT_QWEN_MODEL")
    if qwen_base_url:
        cfg.services.qwen.base_url = qwen_base_url
    if qwen_api_key:
        cfg.services.qwen.api_key = qwen_api_key
    if qwen_model:
        cfg.services.qwen.model = qwen_model
    return cfg


def _clone_cfg(config_path: str) -> Any:
    return copy.deepcopy(_load_cfg(config_path))


def _run_stage1(cfg: Any, video_path: str, force: bool) -> Any:
    _prepare_external_data_pipeline()
    from data_pipeline.stage1_parse.pipeline import Stage1Runner

    video = Path(video_path).expanduser().resolve()
    return Stage1Runner(cfg).run(video, video_id=video.stem, force=force)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _raw_vlm_caption_cfg(config_path: str) -> dict[str, Any]:
    try:
        import yaml

        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    cfg = raw.get("stage1_parse", {}).get("vlm_caption", {})
    return cfg if isinstance(cfg, dict) else {}


def _int_cfg(raw: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(raw.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _bool_cfg(raw: dict[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    size = max(1, size)
    return [items[i:i + size] for i in range(0, len(items), size)]


def _caption_text_for_cache(row: dict[str, Any]) -> str:
    payload = row.get("caption", row)
    if isinstance(payload, dict):
        text = (
            payload.get("caption_zh")
            or payload.get("visual_caption_zh")
            or payload.get("description_zh")
            or payload.get("description")
            or payload.get("caption")
            or row.get("caption_zh")
            or row.get("visual_caption_zh")
            or ""
        )
        return str(text).strip()
    return str(payload or "").strip()


def _captions_cover_shots(captions: list[dict[str, Any]], shots: list[Any]) -> tuple[bool, list[str]]:
    def _shot_id(shot: Any, idx: int) -> str:
        if isinstance(shot, dict):
            return str(shot.get("shot_id", idx + 1))
        return str(getattr(shot, "shot_id", idx + 1))

    def _shot_duration(shot: Any) -> float:
        value = shot.get("duration", shot.get("duration_sec", 0.0)) if isinstance(shot, dict) else getattr(shot, "duration", 0.0)
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _shot_is_skip(shot: Any) -> bool:
        return bool(shot.get("is_skip", False)) if isinstance(shot, dict) else bool(getattr(shot, "is_skip", False))

    required_ids = {
        _shot_id(shot, idx)
        for idx, shot in enumerate(shots)
        if _shot_duration(shot) > 0 and not _shot_is_skip(shot)
    }
    caption_ids = {
        str(row.get("shot_id", idx + 1))
        for idx, row in enumerate(captions)
        if isinstance(row, dict) and _caption_text_for_cache(row)
    }
    if required_ids:
        missing = sorted(required_ids - caption_ids, key=lambda x: int(x) if x.isdigit() else x)
        return not missing, missing
    return bool(captions) and all(_caption_text_for_cache(row) for row in captions), []


def _single_caption_job(
    captioner: Any,
    job: tuple[int, list[Path], str, dict[str, Any]],
    shot_records: list[Any],
) -> tuple[int, Any]:
    rec_idx, frame_paths_for_job, asr_text_for_job, meta_hint = job
    rec = shot_records[rec_idx]
    res = captioner.caption_shot(
        frame_paths=frame_paths_for_job,
        asr_text=rec.asr_text_for_vlm or asr_text_for_job,
        meta_hint=meta_hint,
    )
    return rec_idx, res


def _batch_caption_jobs(
    captioner: Any,
    jobs: list[tuple[int, list[Path], str, dict[str, Any]]],
    shot_records: list[Any],
    *,
    max_frames_per_shot: int,
    max_batch_images: int,
    fallback_to_single: bool,
    split_on_failure: bool,
    defer_failed: bool = False,
    depth: int = 0,
) -> list[tuple[int, Any]]:
    """Caption multiple shots in one Qwen-VL request, then split/fallback on misses."""
    _prepare_external_data_pipeline()
    from data_pipeline.stage1_parse.vlm_captioner import (
        VLMCaptionResult,
        _normalize_caption,
        validate_caption,
    )

    if not jobs:
        return []
    if len(jobs) == 1:
        return [_single_caption_job(captioner, jobs[0], shot_records)]

    max_frames_per_shot = max(1, int(max_frames_per_shot or 1))
    max_batch_images = max(max_frames_per_shot, int(max_batch_images or max_frames_per_shot))
    image_budget_jobs: list[tuple[int, list[Path], str, dict[str, Any]]] = []
    image_count = 0
    for job in jobs:
        frames = list(job[1])[:max_frames_per_shot]
        if image_budget_jobs and image_count + len(frames) > max_batch_images:
            break
        image_budget_jobs.append(job)
        image_count += len(frames)
    if len(image_budget_jobs) < len(jobs):
        first = _batch_caption_jobs(
            captioner,
            image_budget_jobs,
            shot_records,
            max_frames_per_shot=max_frames_per_shot,
            max_batch_images=max_batch_images,
            fallback_to_single=fallback_to_single,
            split_on_failure=split_on_failure,
            defer_failed=defer_failed,
            depth=depth,
        )
        rest = _batch_caption_jobs(
            captioner,
            jobs[len(image_budget_jobs):],
            shot_records,
            max_frames_per_shot=max_frames_per_shot,
            max_batch_images=max_batch_images,
            fallback_to_single=fallback_to_single,
            split_on_failure=split_on_failure,
            defer_failed=defer_failed,
            depth=depth,
        )
        return first + rest

    all_frames: list[Path] = []
    shot_inputs: list[dict[str, Any]] = []
    asr_window_chars = int(getattr(captioner._step_cfg, "asr_window_chars", 800) or 800)
    for job in jobs:
        rec_idx, frame_paths_for_job, asr_text_for_job, meta_hint = job
        rec = shot_records[rec_idx]
        frames = list(frame_paths_for_job)[:max_frames_per_shot]
        frame_labels = []
        for frame_idx, frame_path in enumerate(frames, start=1):
            all_frames.append(frame_path)
            frame_labels.append(f"image_{len(all_frames)}")
        shot_inputs.append({
            "shot_id": str(rec.shot_id),
            "shot_index": int(rec.shot_index),
            "duration_sec": round(float(getattr(rec, "duration", 0.0) or 0.0), 2),
            "asr_text": str(rec.asr_text_for_vlm or asr_text_for_job or "")[:asr_window_chars],
            "meta_hint": meta_hint,
            "frames": frame_labels,
        })

    if not all_frames:
        return [
            (job[0], VLMCaptionResult(parsed={}, error="no_frames"))
            for job in jobs
        ]

    original_max_frames = int(getattr(captioner._client.cfg, "max_frames", 0) or 0)
    captioner._client.cfg.max_frames = min(max(original_max_frames, len(all_frames)), max_batch_images)
    batch_system = (
        f"{captioner._sys}\n\n"
        "【批量模式补充规则】\n"
        "现在一次输入多个连续切镜。用户消息中的 shot_inputs 会声明每个 shot_id 对应哪些图片编号；"
        "图片会按 image_1、image_2... 的顺序紧跟在文本后。请分别观察每个 shot 的图片，"
        "不要把不同 shot 的画面/ASR 混在一起。\n"
        "严格只输出一个 JSON object，格式为："
        "{\"captions\":[{\"shot_id\":\"...\",\"caption\":{单镜头完整caption schema}}]}。"
        "captions 必须覆盖所有输入 shot_id，不能漏项、不能新增。"
    )
    batch_user = (
        "请为以下多个切镜批量生成结构化 caption。\n"
        "每个 caption 字段必须完全符合 system 中的单镜头 JSON schema；"
        "caption_zh、semantic、visual_tags 等中文字段必须使用中文；strict_tags 只能选中文枚举。\n\n"
        f"shot_inputs:\n{json.dumps(shot_inputs, ensure_ascii=False, indent=2)}"
    )

    try:
        res = captioner._client.chat(
            system=batch_system,
            user=batch_user,
            frame_paths=all_frames,
            temperature=getattr(captioner._step_cfg, "temperature", 0.2),
            json_only=True,
        )
    finally:
        captioner._client.cfg.max_frames = original_max_frames

    parsed = res.parsed or {}
    rows = parsed.get("captions") or parsed.get("items") or parsed.get("__array__") or []
    if not isinstance(rows, list):
        rows = []
    by_sid: dict[str, dict[str, Any]] = {}
    raw_errors: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("shot_id", "") or "")
        caption = row.get("caption") if isinstance(row.get("caption"), dict) else row
        if not sid or not isinstance(caption, dict):
            continue
        normalized = _normalize_caption(dict(caption))
        ok, issues = validate_caption(normalized)
        if ok:
            normalized["validation"] = {
                "ok": True,
                "schema": "strict_518_v1",
                "checked_by": "aicut_batch_caption",
                "batch_size": len(jobs),
            }
            by_sid[sid] = normalized
        else:
            raw_errors[sid] = "caption_validation_failed: " + "; ".join(issues[:12])

    results: list[tuple[int, Any]] = []
    failed_jobs: list[tuple[int, list[Path], str, dict[str, Any]]] = []
    for job in jobs:
        rec_idx = job[0]
        sid = str(shot_records[rec_idx].shot_id)
        if sid in by_sid:
            results.append((rec_idx, VLMCaptionResult(parsed=by_sid[sid], raw_text=res.text)))
        else:
            err = raw_errors.get(sid) or res.error or "batch_caption_missing_or_bad_json"
            failed_jobs.append(job)
            results.append((rec_idx, VLMCaptionResult(parsed={}, raw_text=res.text, error=err)))

    if not failed_jobs:
        return results
    if defer_failed:
        return results
    if split_on_failure and len(failed_jobs) > 1 and depth < 8:
        successful = [(idx, r) for idx, r in results if not getattr(r, "error", None)]
        mid = max(1, len(failed_jobs) // 2)
        repaired = _batch_caption_jobs(
            captioner,
            failed_jobs[:mid],
            shot_records,
            max_frames_per_shot=max_frames_per_shot,
            max_batch_images=max_batch_images,
            fallback_to_single=fallback_to_single,
            split_on_failure=split_on_failure,
            defer_failed=defer_failed,
            depth=depth + 1,
        ) + _batch_caption_jobs(
            captioner,
            failed_jobs[mid:],
            shot_records,
            max_frames_per_shot=max_frames_per_shot,
            max_batch_images=max_batch_images,
            fallback_to_single=fallback_to_single,
            split_on_failure=split_on_failure,
            defer_failed=defer_failed,
            depth=depth + 1,
        )
        return sorted(successful + repaired, key=lambda item: item[0])
    if fallback_to_single:
        successful = [(idx, r) for idx, r in results if not getattr(r, "error", None)]
        single = [_single_caption_job(captioner, job, shot_records) for job in failed_jobs]
        return sorted(successful + single, key=lambda item: item[0])
    return results


def _ok(tool_name: str, payload: dict[str, Any], artifacts: dict[str, str], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "success",
        "stage": "stage1",
        "tool_name": tool_name,
        "artifacts": artifacts,
        "metrics": metrics,
        **payload,
    }


def _err(tool_name: str, exc: Exception) -> dict[str, Any]:
    return {
        "status": "fatal_error",
        "stage": "stage1",
        "tool_name": tool_name,
        "error": str(exc),
        "exception_type": type(exc).__name__,
        "trace": {
            "exception_type": type(exc).__name__,
            "stacktrace": "".join(traceback.format_exception(exc)),
        },
        "artifacts": {},
        "metrics": {},
    }


def validate_environment(config_path: str) -> dict[str, Any]:
    try:
        _prepare_external_data_pipeline()
        required = ["faster_whisper", "scenedetect", "decord", "av", "cv2", "torch"]
        missing = []
        for module_name in required:
            try:
                __import__(module_name)
            except Exception:
                missing.append(module_name)
        if missing:
            raise RuntimeError(f"Missing Stage1 dependencies: {', '.join(missing)}")
        _load_cfg(config_path)
        from data_pipeline.stage1_parse.pipeline import Stage1Runner  # noqa: F401
        return {"status": "success", "missing": []}
    except Exception as exc:
        return _err("validate_environment", exc)


def transcribe_video(video_path: str, config_path: str, force: bool = False) -> dict[str, Any]:
    tool_name = "transcribe_video"
    try:
        cfg = _clone_cfg(config_path)
        cfg.stage1_parse.enabled_steps.run_asr = True
        cfg.stage1_parse.enabled_steps.run_shot_cutting = False
        cfg.stage1_parse.enabled_steps.run_frame_extract = False
        cfg.stage1_parse.enabled_steps.run_vlm_caption = False
        cfg.stage1_parse.enabled_steps.run_concat_summary = False
        res = _run_stage1(cfg, video_path, force)
        video_id = Path(video_path).stem
        transcript_path = cfg.storage_paths().transcript_path(video_id)
        duration = float(res.transcript.get("duration", 0.0))
        segments_count = len(res.transcript.get("segments", []))
        return _ok(
            tool_name,
            {
                "video_id": video_id,
                "transcript_path": str(transcript_path),
                "duration_sec": duration,
                "segments_count": segments_count,
            },
            {"transcript_path": str(transcript_path)},
            {"duration_sec": duration, "segments_count": segments_count},
        )
    except Exception as exc:
        return _err(tool_name, exc)


def cut_shots(video_path: str, config_path: str, force: bool = False) -> dict[str, Any]:
    tool_name = "cut_shots"
    try:
        cfg = _clone_cfg(config_path)
        cfg.stage1_parse.enabled_steps.run_asr = False
        cfg.stage1_parse.enabled_steps.run_shot_cutting = True
        cfg.stage1_parse.enabled_steps.run_frame_extract = False
        cfg.stage1_parse.enabled_steps.run_vlm_caption = False
        cfg.stage1_parse.enabled_steps.run_concat_summary = False
        _run_stage1(cfg, video_path, force)
        video_id = Path(video_path).stem
        shots_path = cfg.storage_paths().shots_index_path(video_id)
        shots = _read_jsonl(shots_path)
        return _ok(
            tool_name,
            {"video_id": video_id, "shots_path": str(shots_path), "shots_count": len(shots)},
            {"shots_path": str(shots_path)},
            {"shots_count": len(shots)},
        )
    except Exception as exc:
        return _err(tool_name, exc)


def extract_frames(video_path: str, config_path: str, force: bool = False) -> dict[str, Any]:
    tool_name = "extract_frames"
    try:
        cfg = _clone_cfg(config_path)
        cfg.stage1_parse.enabled_steps.run_asr = False
        cfg.stage1_parse.enabled_steps.run_shot_cutting = True
        cfg.stage1_parse.enabled_steps.run_frame_extract = True
        cfg.stage1_parse.enabled_steps.run_vlm_caption = False
        cfg.stage1_parse.enabled_steps.run_concat_summary = False
        _run_stage1(cfg, video_path, force)
        video_id = Path(video_path).stem
        frames_dir = cfg.storage_paths().stage1_root(video_id) / "frames"
        frame_count = len(list(frames_dir.glob("**/*.jpg"))) if frames_dir.exists() else 0
        return _ok(
            tool_name,
            {"video_id": video_id, "frames_dir": str(frames_dir), "frame_count": frame_count},
            {"frames_dir": str(frames_dir)},
            {"frame_count": frame_count},
        )
    except Exception as exc:
        return _err(tool_name, exc)


def caption_shots(video_path: str, config_path: str, force: bool = False) -> dict[str, Any]:
    tool_name = "caption_shots"
    try:
        cfg = _clone_cfg(config_path)
        video_id = Path(video_path).stem
        paths = cfg.storage_paths()
        shots_path = paths.shots_index_path(video_id)
        captions_path = paths.captions_index_path(video_id)
        if captions_path.exists() and force:
            captions_path.unlink()
        cached_captions = _read_jsonl(captions_path) if captions_path.exists() and not force else []
        raw_shots = _read_jsonl(shots_path) if shots_path.exists() else []
        cache_complete, missing_shot_ids = _captions_cover_shots(cached_captions, raw_shots)
        if cache_complete:
            return _ok(
                tool_name,
                {
                    "video_id": video_id,
                    "captions_path": str(captions_path),
                    "captioned_count": len(cached_captions),
                    "skipped": True,
                    "skip_reason": "existing_captions_index_complete",
                },
                {"captions_path": str(captions_path)},
                {"captioned_count": len(cached_captions), "pending": 0},
            )

        _prepare_external_data_pipeline()
        from data_pipeline.stage1_parse.asr import align_transcript_to_shots
        from data_pipeline.stage1_parse.frame_extractor import video_dimensions
        from data_pipeline.stage1_parse.pipeline import (
            Stage1ShotRecord,
            _attach_subtitle_to_caption,
            _caption_has_current_subtitle_alignment,
            _compact_subtitle_for_meta,
            _empty_subtitle,
            _read_cached_caption_record,
            _record_to_dict,
        )
        from data_pipeline.stage1_parse.shot_cutter import ShotCutterRunner
        from data_pipeline.stage1_parse.vlm_captioner import VLMCaptioner, caption_is_valid
        from data_pipeline.utils.io import read_json, write_json, write_jsonl

        transcript_path = paths.transcript_path(video_id)
        shots_path = paths.shots_index_path(video_id)
        if not transcript_path.exists():
            raise FileNotFoundError(f"transcript missing before caption: {transcript_path}")
        if not shots_path.exists():
            raise FileNotFoundError(f"shots missing before caption: {shots_path}")

        transcript = read_json(transcript_path)
        shots = ShotCutterRunner._parse_metadata(shots_path)
        kept_shots = [s for s in shots if s.duration > 0 and s.video_path.exists()]
        dims = video_dimensions(Path(video_path))
        width, height, fps = dims if dims else (1920, 1080, 30.0)
        subtitle_alignment = align_transcript_to_shots(
            transcript,
            [
                {
                    "shot_id": sh.shot_id,
                    "start_sec": float(sh.start_frame / max(sh.fps, 1e-3)),
                    "end_sec": float(sh.end_frame / max(sh.fps, 1e-3)),
                }
                for sh in kept_shots
            ],
            context_shots=1,
        ) if transcript else {}

        frame_cfg = cfg.stage1_parse.frame_extract
        shot_records = []
        pending_jobs = []

        for idx, sh in enumerate(kept_shots):
            shot_id = sh.shot_id
            frame_dir = paths.frames_dir(video_id, shot_id)
            frame_paths = sorted(frame_dir.glob(f"*.{frame_cfg.image_format}"))
            subtitle = subtitle_alignment.get(shot_id, _empty_subtitle())
            asr_text = str(subtitle.get("asr_text_exact") or "")
            asr_text_for_vlm = str(subtitle.get("asr_text_for_vlm") or asr_text)
            caption_path = paths.caption_path(video_id, shot_id)
            caption = {}
            cached_record = _read_cached_caption_record(caption_path) if not force else None

            if cached_record and cached_record.get("caption"):
                cached_caption = cached_record.get("caption") or {}
                if caption_is_valid(cached_caption) and _caption_has_current_subtitle_alignment(cached_caption):
                    caption = cached_caption
                else:
                    pending_jobs.append((
                        len(shot_records),
                        frame_paths,
                        asr_text,
                        {
                            "video_id": video_id,
                            "shot_index": idx,
                            "duration": round(sh.duration, 2),
                            "retry_reason": "cached_caption_failed_strict_518_validation",
                            "subtitle_alignment": _compact_subtitle_for_meta(subtitle),
                        },
                    ))
            elif frame_paths:
                pending_jobs.append((
                    len(shot_records),
                    frame_paths,
                    asr_text,
                    {
                        "video_id": video_id,
                        "shot_index": idx,
                        "duration": round(sh.duration, 2),
                        "subtitle_alignment": _compact_subtitle_for_meta(subtitle),
                    },
                ))

            record = Stage1ShotRecord(
                shot_id=shot_id,
                shot_index=idx,
                clip_path=str(sh.video_path),
                frames_dir=str(frame_dir),
                frame_paths=[str(p) for p in frame_paths],
                start_sec=float(sh.start_frame / max(sh.fps, 1e-3)),
                end_sec=float(sh.end_frame / max(sh.fps, 1e-3)),
                duration=float(sh.duration),
                fps=float(sh.fps if sh.fps > 0 else fps),
                width=width,
                height=height,
                asr_text=asr_text,
                asr_text_exact=str(subtitle.get("asr_text_exact") or ""),
                asr_text_overlap=str(subtitle.get("asr_text_overlap") or ""),
                asr_text_context_prev=str(subtitle.get("asr_text_context_prev") or ""),
                asr_text_context_next=str(subtitle.get("asr_text_context_next") or ""),
                asr_text_for_vlm=asr_text_for_vlm,
                asr_text_for_planner=str(subtitle.get("asr_text_for_planner") or ""),
                subtitle=subtitle,
                caption=caption,
            )
            shot_records.append(record)

        print(
            json.dumps(
                {
                    "event": "caption_jobs_prepared",
                    "video_id": video_id,
                    "shots": len(shot_records),
                    "pending": len(pending_jobs),
                    "missing_cached_shot_ids": missing_shot_ids[:20],
                    "request_concurrency": cfg.stage1_parse.vlm_caption.request_concurrency,
                    "max_frames": cfg.stage1_parse.vlm_caption.max_frames,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        if pending_jobs:
            raw_vlm_cfg = _raw_vlm_caption_cfg(config_path)
            batch_mode = _bool_cfg(raw_vlm_cfg, "batch_mode", True)
            requested_batch_size = max(1, _int_cfg(raw_vlm_cfg, "batch_size", 8))
            batch_concurrency = max(
                1,
                _int_cfg(raw_vlm_cfg, "batch_concurrency", int(cfg.stage1_parse.vlm_caption.request_concurrency)),
            )
            max_frames_per_shot = max(1, int(cfg.stage1_parse.vlm_caption.max_frames))
            service_image_limit = max(
                1,
                _int_cfg(
                    raw_vlm_cfg,
                    "server_image_limit",
                    int(getattr(cfg.services.qwen, "max_frames", 8) or 8),
                ),
            )
            max_frames_per_shot = min(max_frames_per_shot, service_image_limit)
            configured_max_batch_images = max(
                1,
                _int_cfg(raw_vlm_cfg, "max_batch_images", requested_batch_size * max_frames_per_shot),
            )
            max_batch_images = min(configured_max_batch_images, service_image_limit)
            effective_batch_size = max(1, min(requested_batch_size, max_batch_images // max_frames_per_shot))
            fallback_to_single = _bool_cfg(raw_vlm_cfg, "fallback_to_single", True)
            split_on_failure = _bool_cfg(raw_vlm_cfg, "split_on_failure", True)
            defer_failed_to_next_round = batch_mode and _bool_cfg(raw_vlm_cfg, "defer_failed_to_next_round", True)
            batch_retry_rounds = max(0, _int_cfg(raw_vlm_cfg, "batch_retry_rounds", 2))
            work_units = _chunks(pending_jobs, effective_batch_size) if batch_mode else [[job] for job in pending_jobs]
            max_workers = batch_concurrency if batch_mode else int(cfg.stage1_parse.vlm_caption.request_concurrency)
            max_workers = min(max(1, max_workers), len(work_units))
            progress = tqdm(total=len(pending_jobs), desc=f"Qwen-VL caption/{video_id}", unit="shot", dynamic_ncols=True)
            progress.set_postfix({"ok": 0, "fail": 0})
            ok_count = 0
            fail_count = 0

            print(
                json.dumps(
                    {
                        "event": "caption_batch_config",
                        "video_id": video_id,
                        "batch_mode": batch_mode,
                        "requested_batch_size": requested_batch_size,
                        "effective_batch_size": effective_batch_size if batch_mode else 1,
                        "batch_concurrency": max_workers,
                        "server_image_limit": service_image_limit,
                        "max_frames_per_shot": max_frames_per_shot,
                        "configured_max_batch_images": configured_max_batch_images,
                        "max_batch_images": max_batch_images,
                        "defer_failed_to_next_round": defer_failed_to_next_round,
                        "batch_retry_rounds": batch_retry_rounds,
                        "num_batches": len(work_units),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

            def _one_batch(
                batch_idx: int,
                jobs: list[tuple[int, list[Path], str, dict[str, Any]]],
                *,
                final_repair: bool = False,
            ) -> list[tuple[int, Any]]:
                captioner_local = VLMCaptioner(cfg)
                shot_ids = [str(shot_records[job[0]].shot_id) for job in jobs]
                print(
                    json.dumps(
                        {
                            "event": "caption_batch_start",
                            "batch_index": batch_idx,
                            "shot_ids": shot_ids,
                            "shots": len(jobs),
                            "frames": sum(len(job[1][:max_frames_per_shot]) for job in jobs),
                            "final_repair": final_repair,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if batch_mode:
                    return _batch_caption_jobs(
                        captioner_local,
                        jobs,
                        shot_records,
                        max_frames_per_shot=max_frames_per_shot,
                        max_batch_images=max_batch_images,
                        fallback_to_single=fallback_to_single,
                        split_on_failure=split_on_failure,
                        defer_failed=defer_failed_to_next_round and not final_repair,
                    )
                return [_single_caption_job(captioner_local, job, shot_records) for job in jobs]

            try:
                if defer_failed_to_next_round:
                    normal_queue = deque(pending_jobs)
                    final_repair_queue: deque[tuple[int, list[Path], str, dict[str, Any]]] = deque()
                    retry_counts: dict[int, int] = {}
                    futures: dict[Any, tuple[int, list[tuple[int, list[Path], str, dict[str, Any]]], bool]] = {}
                    next_batch_idx = 0
                    done_count = 0

                    def _submit_available(pool: ThreadPoolExecutor) -> None:
                        nonlocal next_batch_idx
                        while len(futures) < max_workers and (final_repair_queue or normal_queue):
                            final_repair = bool(final_repair_queue)
                            source = final_repair_queue if final_repair else normal_queue
                            jobs: list[tuple[int, list[Path], str, dict[str, Any]]] = []
                            while source and len(jobs) < effective_batch_size:
                                jobs.append(source.popleft())
                            if not jobs:
                                break
                            batch_idx = next_batch_idx
                            next_batch_idx += 1
                            futures[pool.submit(_one_batch, batch_idx, jobs, final_repair=final_repair)] = (
                                batch_idx,
                                jobs,
                                final_repair,
                            )

                    with ThreadPoolExecutor(max_workers=max_workers) as pool:
                        _submit_available(pool)
                        while futures:
                            done_futures, _ = wait(list(futures), return_when=FIRST_COMPLETED)
                            for future in done_futures:
                                batch_idx, jobs, final_repair = futures.pop(future)
                                jobs_by_idx = {job[0]: job for job in jobs}
                                requeued_jobs: list[tuple[int, list[Path], str, dict[str, Any]]] = []
                                final_jobs: list[tuple[int, list[Path], str, dict[str, Any]]] = []
                                finalized = 0
                                batch_results: list[tuple[int, Any]]
                                batch_exception = ""
                                try:
                                    batch_results = future.result()
                                except Exception as exc:
                                    batch_results = []
                                    batch_exception = str(exc) or "caption_batch_exception"

                                seen_result_indices: set[int] = set()
                                for rec_idx, res in batch_results:
                                    seen_result_indices.add(rec_idx)
                                    rec = shot_records[rec_idx]
                                    if res.error and not final_repair:
                                        job = jobs_by_idx.get(rec_idx)
                                        if job is not None and retry_counts.get(rec_idx, 0) < batch_retry_rounds:
                                            retry_counts[rec_idx] = retry_counts.get(rec_idx, 0) + 1
                                            requeued_jobs.append(job)
                                        elif job is not None:
                                            final_jobs.append(job)
                                        continue

                                    rec.caption = res.parsed
                                    _attach_subtitle_to_caption(rec.caption, rec.subtitle)
                                    if res.error:
                                        fail_count += 1
                                    else:
                                        ok_count += 1
                                    write_json(paths.caption_path(video_id, rec.shot_id), _record_to_dict(rec))
                                    finalized += 1

                                for job in jobs:
                                    rec_idx = job[0]
                                    if rec_idx in seen_result_indices:
                                        continue
                                    if final_repair:
                                        rec = shot_records[rec_idx]
                                        rec.caption = {}
                                        _attach_subtitle_to_caption(rec.caption, rec.subtitle)
                                        fail_count += 1
                                        write_json(paths.caption_path(video_id, rec.shot_id), _record_to_dict(rec))
                                        finalized += 1
                                        continue
                                    if retry_counts.get(rec_idx, 0) < batch_retry_rounds:
                                        retry_counts[rec_idx] = retry_counts.get(rec_idx, 0) + 1
                                        requeued_jobs.append(job)
                                    else:
                                        final_jobs.append(job)

                                for job in reversed(requeued_jobs):
                                    normal_queue.appendleft(job)
                                final_repair_queue.extend(final_jobs)

                                done_count += finalized
                                print(
                                    json.dumps(
                                        {
                                            "event": "caption_batch_done",
                                            "batch_index": batch_idx,
                                            "shots": len(jobs),
                                            "ok": sum(1 for _, res in batch_results if not bool(res.error)),
                                            "fail": sum(1 for _, res in batch_results if bool(res.error)),
                                            "requeued": len(requeued_jobs),
                                            "final_repair_queued": len(final_jobs),
                                            "final_repair": final_repair,
                                            "error": batch_exception,
                                            "done": done_count,
                                            "total": len(pending_jobs),
                                        },
                                        ensure_ascii=False,
                                    ),
                                    flush=True,
                                )
                                if finalized:
                                    progress.update(finalized)
                                progress.set_postfix({"ok": ok_count, "fail": fail_count})
                            _submit_available(pool)
                else:
                    with ThreadPoolExecutor(max_workers=max_workers) as pool:
                        futures = {
                            pool.submit(_one_batch, batch_idx, jobs): (batch_idx, jobs)
                            for batch_idx, jobs in enumerate(work_units)
                        }
                        done_count = 0
                        while futures:
                            done_futures, _ = wait(list(futures), return_when=FIRST_COMPLETED)
                            for future in done_futures:
                                batch_idx, jobs = futures.pop(future)
                                try:
                                    batch_results = future.result()
                                except Exception:
                                    fail_count += len(jobs)
                                    done_count += len(jobs)
                                    progress.update(len(jobs))
                                    progress.set_postfix({"ok": ok_count, "fail": fail_count})
                                    continue
                                for rec_idx, res in batch_results:
                                    rec = shot_records[rec_idx]
                                    rec.caption = res.parsed
                                    _attach_subtitle_to_caption(rec.caption, rec.subtitle)
                                    if res.error:
                                        fail_count += 1
                                    else:
                                        ok_count += 1
                                    write_json(paths.caption_path(video_id, rec.shot_id), _record_to_dict(rec))
                                done_count += len(batch_results)
                                print(
                                    json.dumps(
                                        {
                                            "event": "caption_batch_done",
                                            "batch_index": batch_idx,
                                            "shots": len(jobs),
                                            "ok": sum(1 for _, res in batch_results if not bool(res.error)),
                                            "fail": sum(1 for _, res in batch_results if bool(res.error)),
                                            "done": done_count,
                                            "total": len(pending_jobs),
                                        },
                                        ensure_ascii=False,
                                    ),
                                    flush=True,
                                )
                                progress.update(len(batch_results))
                                progress.set_postfix({"ok": ok_count, "fail": fail_count})
            finally:
                progress.close()

        for rec in shot_records:
            _attach_subtitle_to_caption(rec.caption, rec.subtitle)
            write_json(paths.caption_path(video_id, rec.shot_id), _record_to_dict(rec))

        write_jsonl(captions_path, [_record_to_dict(r) for r in shot_records])
        captions = _read_jsonl(captions_path)
        return _ok(
            tool_name,
            {"video_id": video_id, "captions_path": str(captions_path), "captioned_count": len(captions)},
            {"captions_path": str(captions_path)},
            {"captioned_count": len(captions)},
        )
    except Exception as exc:
        return _err(tool_name, exc)


def summarize_video(video_path: str, config_path: str, force: bool = False) -> dict[str, Any]:
    tool_name = "summarize_video"
    try:
        cfg = _clone_cfg(config_path)
        _prepare_external_data_pipeline()
        from data_pipeline.stage1_parse.asr import transcript_text
        from data_pipeline.stage1_parse.concat_summarizer import ConcatSummarizer
        from data_pipeline.stage1_parse.pipeline import Stage1ShotRecord, _to_summarizer_input
        from data_pipeline.utils.io import read_json, write_json

        video_id = Path(video_path).stem
        paths = cfg.storage_paths()
        transcript_path = paths.transcript_path(video_id)
        captions_path = paths.captions_index_path(video_id)
        summary_path = paths.concat_summary_path(video_id)
        if not transcript_path.exists():
            raise FileNotFoundError(f"transcript missing before summary: {transcript_path}")
        if not captions_path.exists():
            raise FileNotFoundError(f"captions missing before summary: {captions_path}")

        if summary_path.exists() and not force:
            summary = read_json(summary_path)
        else:
            transcript = read_json(transcript_path)
            rows = _read_jsonl(captions_path)
            shot_records = []
            for row in rows:
                shot_records.append(Stage1ShotRecord(
                    shot_id=str(row.get("shot_id", "")),
                    shot_index=int(row.get("shot_index", 0) or 0),
                    clip_path=str(row.get("clip_path", "")),
                    frames_dir=str(row.get("frames_dir", "")),
                    frame_paths=list(row.get("frame_paths") or []),
                    start_sec=float(row.get("start_sec", 0) or 0),
                    end_sec=float(row.get("end_sec", 0) or 0),
                    duration=float(row.get("duration", 0) or 0),
                    fps=float(row.get("fps", 0) or 0),
                    width=int(row.get("width", 0) or 0),
                    height=int(row.get("height", 0) or 0),
                    asr_text=str(row.get("asr_text", "")),
                    asr_text_exact=str(row.get("asr_text_exact", "")),
                    asr_text_overlap=str(row.get("asr_text_overlap", "")),
                    asr_text_context_prev=str(row.get("asr_text_context_prev", "")),
                    asr_text_context_next=str(row.get("asr_text_context_next", "")),
                    asr_text_for_vlm=str(row.get("asr_text_for_vlm", "")),
                    asr_text_for_planner=str(row.get("asr_text_for_planner", "")),
                    subtitle=dict(row.get("subtitle") or {}),
                    caption=dict(row.get("caption") or {}),
                ))
            print(
                json.dumps(
                    {"event": "summary_request_start", "video_id": video_id, "shots": len(shot_records)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            summarizer = ConcatSummarizer(cfg)
            shot_descs = [_to_summarizer_input(r) for r in shot_records]
            asr_full = transcript_text(transcript) if transcript else ""
            summary = {}
            max_summary_retries = 2
            for attempt in range(max_summary_retries + 1):
                res = summarizer.summarize(asr_full=asr_full, shot_descriptions=shot_descs)
                summary = res.parsed
                if summary:
                    break
                if attempt == max_summary_retries and res.error:
                    raise RuntimeError(f"summary failed: {res.error}")
            write_json(summary_path, summary)
            print(
                json.dumps(
                    {"event": "summary_request_done", "video_id": video_id, "ok": bool(summary)},
                    ensure_ascii=False,
                ),
                flush=True,
            )

        intent = summary.get("video_intent") or summary.get("intent") or summary.get("summary") or ""
        return _ok(
            tool_name,
            {"video_id": video_id, "summary_path": str(summary_path), "intent": intent},
            {"summary_path": str(summary_path)},
            {"intent_length": len(str(intent))},
        )
    except Exception as exc:
        return _err(tool_name, exc)
