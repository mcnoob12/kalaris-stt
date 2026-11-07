"""Clip manifests: JSONL with one clip per line.

Required fields: id, audio_filepath, text, duration, language, dataset, split, license.
speaker is required on training and dev sets, and strongly preferred elsewhere.
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


def read_manifest(path: str | Path) -> list[dict]:
    rows = read_jsonl(path)
    seen: set[str] = set()
    for number, row in enumerate(rows, start=1):
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"{path}, line {number}: missing {missing}")
        if row["id"] in seen:
            raise ValueError(f"{path}, line {number}: duplicate id {row['id']}")
        seen.add(row["id"])
    return rows
