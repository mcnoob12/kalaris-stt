"""Clip manifests: JSONL with one clip per line.

Required fields: id, audio_filepath, text, duration, language, dataset, split, license.
speaker is required on training and dev sets, and strongly preferred elsewhere.
lattice is optional: a list of slots, each listing accepted spellings. When it's
present, scoring uses the official OIWER scorer instead of plain WER.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

REQUIRED_FIELDS = ("id", "audio_filepath", "text", "duration", "language", "dataset", "split", "license")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


REPO_ROOT = Path(__file__).resolve().parents[1]


def relative_to_repo(path: str | Path) -> str:
    """Path to store in a manifest, so it works on any machine with this repo."""
    return Path(path).resolve().relative_to(REPO_ROOT).as_posix()


def read_manifest(path: str | Path) -> list[dict]:
    """Read and validate a manifest. Relative audio paths resolve against the repo."""
    rows = read_jsonl(path)
    for row in rows:
        if "audio_filepath" in row and not Path(row["audio_filepath"]).is_absolute():
            row["audio_filepath"] = str(REPO_ROOT / row["audio_filepath"])
    seen: set[str] = set()
    for number, row in enumerate(rows, start=1):
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"{path}, line {number}: missing {missing}")
        if row["id"] in seen:
            raise ValueError(f"{path}, line {number}: duplicate id {row['id']}")
        seen.add(row["id"])
    return rows
