"""Train/test leakage checks, run before any score is reported.

A checksum finds byte-identical audio. A speaker check finds the same voice.
Re-encoded copies of one recording pass both, so also check transcripts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .normalize import normalize_hi


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class LeakageReport:
    duplicate_audio: list[tuple[str, str]] = field(default_factory=list)
    shared_speakers: list[str] = field(default_factory=list)
    shared_transcripts: list[str] = field(default_factory=list)
    clips_without_speaker: int = 0

    @property
    def is_clean(self) -> bool:
        return not (self.duplicate_audio or self.shared_speakers or self.shared_transcripts)

    def as_dict(self) -> dict:
        return {
            "is_clean": self.is_clean,
            "duplicate_audio": self.duplicate_audio[:50],
            "duplicate_audio_count": len(self.duplicate_audio),
            "shared_speakers": self.shared_speakers[:50],
            "shared_speaker_count": len(self.shared_speakers),
            "shared_transcript_count": len(self.shared_transcripts),
            "clips_without_speaker": self.clips_without_speaker,
        }


def check_leakage(
    train: Sequence[dict],
    held_out: Sequence[dict],
    *,
    check_transcripts: bool = True,
    min_transcript_words: int = 6,
) -> LeakageReport:
    """Compare a training manifest with a dev or test manifest.

    Speaker IDs are namespaced by dataset, because "spk1" in two corpora is not
    the same person. Short transcripts are ignored since common phrases repeat
    naturally.
    """
    report = LeakageReport()

    train_audio = {sha256_file(row["audio_filepath"]): row["id"] for row in train}
    for row in held_out:
        match = train_audio.get(sha256_file(row["audio_filepath"]))
        if match is not None:
            report.duplicate_audio.append((match, row["id"]))

    def speakers(rows: Sequence[dict]) -> set[str]:
        return {f"{r['dataset']}:{r['speaker']}" for r in rows if r.get("speaker")}

    report.shared_speakers = sorted(speakers(train) & speakers(held_out))
    report.clips_without_speaker = sum(1 for r in held_out if not r.get("speaker"))

    if check_transcripts:
        def texts(rows: Sequence[dict]) -> set[str]:
            normalized = (normalize_hi(r["text"]) for r in rows)
            return {t for t in normalized if len(t.split()) >= min_transcript_words}

        report.shared_transcripts = sorted(texts(train) & texts(held_out))

    return report
