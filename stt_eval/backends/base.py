from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Backend(ABC):
    name = "base"

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def load(self) -> None:
        """Load the model. Timed separately from transcription."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """Return the transcript of one 16 kHz mono audio file."""

    def accelerator_peak_bytes(self) -> int | None:
        """Peak accelerator memory, when the runtime exposes it."""
        return None

    def model_bytes(self) -> int | None:
        """On-disk model size, when known."""
        return None


def huggingface_cache_bytes(repo_id: str) -> int | None:
    """Size of a downloaded Hugging Face repo, without counting symlinks twice."""
    from huggingface_hub import scan_cache_dir

    for repo in scan_cache_dir().repos:
        if repo.repo_id == repo_id:
            return repo.size_on_disk
    return None


def local_path_bytes(path: str | Path) -> int:
    from ..bench import directory_size_bytes

    return directory_size_bytes(path)
