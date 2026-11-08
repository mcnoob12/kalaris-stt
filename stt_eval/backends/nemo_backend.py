from __future__ import annotations

from pathlib import Path

from .base import Backend, huggingface_cache_bytes, local_path_bytes


class NemoBackend(Backend):
    """Any NeMo ASR model: a Hugging Face ID or a local .nemo file.

    Run from the training environment (.venv-train), which has NeMo installed.
    """

    name = "nemo"

    def load(self) -> None:
        import torch
        from nemo.collections.asr.models import ASRModel

        if Path(self.model).exists():
            self._model = ASRModel.restore_from(self.model, map_location="cpu")
        else:
            self._model = ASRModel.from_pretrained(self.model, map_location="cpu")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(device).eval()

    def transcribe(self, audio_path: str) -> str:
        output = self._model.transcribe([audio_path], batch_size=1, verbose=False)
        # Depending on the model, NeMo returns a list or a (best, all) pair.
        first = output[0] if not isinstance(output, tuple) else output[0][0]
        if isinstance(first, list):
            first = first[0]
        return getattr(first, "text", first)

    def accelerator_peak_bytes(self) -> int | None:
        import torch

        return torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None

    def model_bytes(self) -> int | None:
        if Path(self.model).exists():
            return local_path_bytes(self.model)
        return huggingface_cache_bytes(self.model)
