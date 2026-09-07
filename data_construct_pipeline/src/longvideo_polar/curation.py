"""Turn the bilibili manifest into an explicit, reproducible annotation worklist.

Selecting inputs with a glob makes a corpus you cannot describe in a paper: the
scripts run range from 4 characters to 161,933, span five language tags, and the
episode folders come out named after 32-hex asset ids. This module replaces the
glob with a declared selection.

A worklist is written once and then obeyed. It records, per episode, the exact
input file, the output directory name and the manifest facts behind it; and it
records, once at the top, the filter that produced it. Re-running the same
:class:`CurationFilter` reproduces the same worklist, and a reader of the dataset
can see what was excluded and why.

Episode directories are named ``<seq>__<bv_id>__<asset8>``:

    000123__BV1Qx411V7yh__0001c616

The sequence number sorts and pages the 9k directories, the BV id is the public
handle that identifies the source video, and the asset-id prefix keeps the join
back to ``videos/`` and ``scripts/`` unambiguous.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

logger = logging.getLogger(__name__)

WORKLIST_FILENAME = "worklist.jsonl"
EXCLUDED_FILENAME = "excluded.jsonl"

# Scripts shorter than this cannot support segmentation into chapters; scripts
# longer than it overflow the context window and reliably time out. Both are
# defaults, overridable from the UI.
DEFAULT_MIN_CHARS = 200
DEFAULT_MAX_CHARS = 20_000

_BV_RE = re.compile(r"(BV[0-9A-Za-z]{10})")


def _extract_bv(row: dict[str, Any]) -> str:
    """Find the BV id in ``bv_url`` or the ``title`` field that mirrors it."""
    for key in ("bv_url", "title", "video_relpath"):
        match = _BV_RE.search(str(row.get(key) or ""))
        if match:
            return match.group(1)
    return ""


def episode_dir_name(index: int, asset_id: str, bv_id: str = "") -> str:
    """Build ``<seq>__<bv_id>__<asset8>``.

    ``bv_id`` is omitted when unknown rather than padded, so the name never claims
    a provenance it does not have.
    """
    parts = [f"{index:06d}"]
    if bv_id:
        parts.append(bv_id)
    parts.append(str(asset_id)[:8])
    return "__".join(parts)


@dataclass
class CurationFilter:
    """The declared inclusion criteria for one annotation batch."""

    min_chars: int = DEFAULT_MIN_CHARS
    max_chars: int = DEFAULT_MAX_CHARS
    languages: list[str] = field(default_factory=lambda: ["zh", "zh-CN", "zh-Hans"])
    min_duration_sec: float = 0.0
    max_duration_sec: float = 0.0  # 0 disables the bound
    limit: int = 0                 # 0 keeps everything that passed
    # Even sampling across the length range. Without it a truncating ``limit``
    # just takes the alphabetically-first asset ids, which correlates with nothing
    # but looks like a sample.
    stratify: bool = False
    strata: int = 8
    require_script_file: bool = True
    seed: int = 20260821

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def reject_reason(self, row: dict[str, Any]) -> str:
        """Why this row is out, or ``""`` if it is in."""
        if not row.get("script_relpath"):
            return "no_script"
        language = str(row.get("primary_script_lang") or "")
        if self.languages and language not in self.languages:
            return f"language:{language or 'unknown'}"
        chars = int(row.get("script_char_count") or 0)
        if self.min_chars and chars < self.min_chars:
            return f"too_short:{chars}"
        if self.max_chars and chars > self.max_chars:
            return f"too_long:{chars}"
        duration = float(row.get("duration_sec") or 0.0)
        if self.min_duration_sec and duration < self.min_duration_sec:
            return f"too_brief:{duration:.0f}s"
        if self.max_duration_sec and duration > self.max_duration_sec:
            return f"too_lengthy:{duration:.0f}s"
        return ""


@dataclass
class WorkItem:
    """One episode to annotate."""

    index: int
    episode_id: str
    asset_id: str
    bv_id: str
    script_path: str
    script_chars: int
    duration_sec: float
    language: str
    title: str
    category: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkItem":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in (payload or {}).items() if key in fields})


def read_manifest(manifest_path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream ``manifest.jsonl``, skipping unparseable lines."""
    path = Path(str(manifest_path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _stable_rank(asset_id: str, seed: int) -> int:
    """Deterministic shuffle key, stable across processes and Python runs."""
    digest = hashlib.blake2b(f"{seed}:{asset_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _stratified_sample(
    rows: list[dict[str, Any]], limit: int, strata: int, seed: int
) -> list[dict[str, Any]]:
    """Take ``limit`` rows spread evenly over script-length bands.

    Length is the variable that drives episode cost and shot count, so an even
    spread over it is what keeps a subset representative of the whole corpus.
    """
    if limit <= 0 or limit >= len(rows):
        return rows
    ordered = sorted(rows, key=lambda row: int(row.get("script_char_count") or 0))
    bands = max(1, min(int(strata), limit))
    band_size = len(ordered) / bands
    per_band = limit // bands
    remainder = limit - per_band * bands

    picked: list[dict[str, Any]] = []
    for band in range(bands):
        start = int(band * band_size)
        end = int((band + 1) * band_size) if band < bands - 1 else len(ordered)
        members = sorted(
            ordered[start:end],
            key=lambda row: _stable_rank(str(row.get("asset_id") or ""), seed),
        )
        take = per_band + (1 if band < remainder else 0)
        picked.extend(members[:take])
    return picked


def _existing_scripts(root: Path, relpaths: Iterable[str]) -> set[str]:
    """Resolve which ``script_relpath`` values exist, using one listing per dir.

    The corpus keeps all 9,876 scripts in a single flat directory. Stat'ing them
    one by one costs a round trip each on the shared mount — minutes of wall time
    for a selection preview the UI expects to answer instantly — whereas listing
    the directory once answers every membership question from memory.
    """
    by_dir: dict[str, set[str]] = {}
    for rel in relpaths:
        if not rel:
            continue
        parent, _, name = rel.rpartition("/")
        by_dir.setdefault(parent, set()).add(name)

    found: set[str] = set()
    for parent, names in by_dir.items():
        directory = root / parent if parent else root
        try:
            with os.scandir(directory) as entries:
                present = {entry.name for entry in entries if entry.is_file()}
        except OSError:
            continue
        prefix = f"{parent}/" if parent else ""
        found.update(f"{prefix}{name}" for name in names & present)
    return found


def build_worklist(
    manifest_path: str | Path,
    dataset_root: str | Path,
    selection: CurationFilter,
) -> tuple[list[WorkItem], dict[str, int]]:
    """Apply ``selection`` to the manifest.

    Args:
        manifest_path: The dataset's ``manifest.jsonl``.
        dataset_root: Directory that ``script_relpath`` is relative to.
        selection: Declared inclusion criteria.

    Returns:
        ``(items, rejection_counts)``. Rejection counts are keyed by reason family
        (``too_long``, ``language``, ...) so the dataset card can report them.
    """
    root = Path(str(dataset_root)).expanduser()
    rows = list(read_manifest(manifest_path))
    available: set[str] = set()
    if selection.require_script_file:
        available = _existing_scripts(root, (str(r.get("script_relpath") or "") for r in rows))

    kept: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}

    for row in rows:
        reason = selection.reject_reason(row)
        if not reason and selection.require_script_file:
            if str(row.get("script_relpath") or "") not in available:
                reason = "script_missing"
        if reason:
            family = reason.split(":", 1)[0]
            rejected[family] = rejected.get(family, 0) + 1
            continue
        kept.append(row)

    if selection.limit and selection.limit < len(kept):
        if selection.stratify:
            kept = _stratified_sample(kept, selection.limit, selection.strata, selection.seed)
        else:
            kept = sorted(
                kept, key=lambda row: _stable_rank(str(row.get("asset_id") or ""), selection.seed)
            )[: selection.limit]

    # Order by asset_id so the sequence numbers are stable for a given selection,
    # regardless of how the sampler happened to emit them.
    kept.sort(key=lambda row: str(row.get("asset_id") or ""))

    items: list[WorkItem] = []
    for index, row in enumerate(kept, start=1):
        asset_id = str(row.get("asset_id") or "")
        bv_id = _extract_bv(row)
        items.append(
            WorkItem(
                index=index,
                episode_id=episode_dir_name(index, asset_id, bv_id),
                asset_id=asset_id,
                bv_id=bv_id,
                script_path=str(root / str(row.get("script_relpath") or "")),
                script_chars=int(row.get("script_char_count") or 0),
                duration_sec=round(float(row.get("duration_sec") or 0.0), 2),
                language=str(row.get("primary_script_lang") or ""),
                title=str(row.get("title") or ""),
                category=str(row.get("category") or ""),
            )
        )
    return items, rejected


def write_worklist(
    output_root: str | Path,
    items: list[WorkItem],
    selection: CurationFilter,
    *,
    manifest_path: str = "",
    rejected: dict[str, int] | None = None,
) -> Path:
    """Persist the worklist and the filter that produced it."""
    root = Path(str(output_root)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    path = root / WORKLIST_FILENAME

    header = {
        "_type": "worklist_header",
        "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "manifest_path": str(manifest_path),
        "selection": selection.as_dict(),
        "selected": len(items),
        "rejected": dict(rejected or {}),
        "naming_scheme": "<seq>__<bv_id>__<asset_id[:8]>",
    }
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, ensure_ascii=False) + "\n")
        for item in items:
            handle.write(json.dumps(item.as_dict(), ensure_ascii=False) + "\n")
    logger.info("worklist: %d episodes → %s", len(items), path)
    return path


def read_worklist(path: str | Path) -> tuple[list[WorkItem], dict[str, Any]]:
    """Load a worklist written by :func:`write_worklist`.

    Returns:
        ``(items, header)``; ``header`` is empty for a headerless file.
    """
    file_path = Path(str(path)).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"worklist not found: {file_path}")
    items: list[WorkItem] = []
    header: dict[str, Any] = {}
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("_type") == "worklist_header":
            header = payload
            continue
        items.append(WorkItem.from_dict(payload))
    return items, header


def summarize(items: list[WorkItem]) -> dict[str, Any]:
    """Corpus statistics for the dataset card and the UI's pre-launch preview."""
    if not items:
        return {"count": 0}
    chars = sorted(item.script_chars for item in items)
    durations = sorted(item.duration_sec for item in items)
    languages: dict[str, int] = {}
    for item in items:
        languages[item.language] = languages.get(item.language, 0) + 1

    def percentile(values: list[Any], fraction: float) -> Any:
        return values[min(len(values) - 1, int(fraction * (len(values) - 1)))]

    return {
        "count": len(items),
        "languages": languages,
        "script_chars": {
            "min": chars[0],
            "p50": percentile(chars, 0.5),
            "p90": percentile(chars, 0.9),
            "max": chars[-1],
            "total": sum(chars),
        },
        "duration_sec": {
            "min": durations[0],
            "p50": percentile(durations, 0.5),
            "p90": percentile(durations, 0.9),
            "max": durations[-1],
            "total_hours": round(sum(durations) / 3600.0, 1),
        },
    }
