"""Pure semantic matching utilities — no heavy deps, extracted from Crayotter."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


# ── text normalisation ────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[`*_>#\-]+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


_STOP = {
    "的", "了", "和", "是", "在", "与", "及", "并", "中", "上", "下", "对", "将", "为", "以", "或",
    "this", "that", "with", "from", "into", "for", "the", "and", "or", "to", "of", "in", "on",
}


def tokenize(text: str) -> set[str]:
    norm = normalize_text(text)
    if not norm:
        return set()
    tokens = re.findall(r"[a-z0-9]+|[一-鿿]", norm)
    return {tk for tk in tokens if tk and tk not in _STOP}


# ── duration bonus ────────────────────────────────────────────────────────────

def duration_bonus(duration: float) -> float:
    if 4 <= duration <= 12:
        return 0.04
    if 2 <= duration <= 20:
        return 0.015
    return 0.0


# ── similarity score ──────────────────────────────────────────────────────────

def similarity_score(query: str, semantic_text: str, duration: float) -> float:
    q_norm = normalize_text(query)
    t_norm = normalize_text(semantic_text)
    if not q_norm or not t_norm:
        return 0.0
    q_tok = tokenize(q_norm)
    t_tok = tokenize(t_norm)
    if not q_tok or not t_tok:
        return 0.0
    overlap = len(q_tok & t_tok)
    if overlap == 0 and q_norm not in t_norm:
        return 0.0
    recall = overlap / max(1, len(q_tok))
    precision = overlap / max(1, len(t_tok))
    contains_bonus = 0.35 if q_norm in t_norm else 0.0
    return round(0.7 * recall + 0.2 * precision + contains_bonus + duration_bonus(duration) * 2, 4)


# ── segment extraction from free-text analysis ───────────────────────────────

_TIME_RANGE_RE = re.compile(
    r"t\s*=\s*(\d+(?:\.\d+)?)s?\s*(?:-|~|—|–|至|到)\s*t?\s*=\s*(\d+(?:\.\d+)?)s?",
    re.IGNORECASE,
)


def extract_segments_from_text(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    matches = list(_TIME_RANGE_RE.finditer(text))
    results: list[dict[str, Any]] = []
    for idx, m in enumerate(matches):
        try:
            start, end = float(m.group(1)), float(m.group(2))
        except Exception:
            continue
        if end <= start:
            continue
        desc_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chunk = text[m.end():desc_end]
        lines = [re.sub(r"^\s*[-*•\d.\)\]]+\s*", "", l.strip()) for l in chunk.splitlines() if l.strip()]
        sem = normalize_text(" ".join(lines)) or normalize_text(text[m.start(): min(len(text), m.end() + 120)])
        results.append({"start": round(start, 2), "end": round(end, 2),
                        "duration": round(end - start, 2), "semantic_text": sem[:320]})
    seen: set[tuple[float, float]] = set()
    deduped = []
    for seg in results:
        key = (seg["start"], seg["end"])
        if key not in seen:
            seen.add(key)
            deduped.append(seg)
    return deduped


# ── analysis JSON helpers ─────────────────────────────────────────────────────

def ensure_semantic_index(analysis_path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return semantic_segments, writing back to JSON if they were missing."""
    segs = payload.get("semantic_segments", [])
    if not isinstance(segs, list) or not segs:
        segs = extract_segments_from_text(str(payload.get("analysis_text", "")))
    segs = [
        {**s, "semantic_text": normalize_text(str(s.get("semantic_text", "")))}
        for s in segs if isinstance(s, dict)
    ]
    if segs != payload.get("semantic_segments"):
        payload["semantic_segments"] = segs
        payload["semantic_index"] = {
            "index_version": 2,
            "segment_count": len(segs),
            "retrieval_mode": "text",
            "updated_at": datetime.now().isoformat(),
        }
        try:
            analysis_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    return segs


def iter_analysis_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*_analysis.json"):
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                files.append(p)
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
