from __future__ import annotations

from .base import Backend, huggingface_cache_bytes


class ParakeetMLX(Backend):
    name = "parakeet-mlx"

    def load(self) -> None:
        from parakeet_mlx import from_pretrained

        self._model = from_pretrained(self.model)

    def transcribe(self, audio_path: str) -> str:
        return self._model.transcribe(audio_path).text

    def accelerator_peak_bytes(self) -> int | None:
        import mlx.core as mx

        return mx.get_peak_memory()

    def model_bytes(self) -> int | None:
        return huggingface_cache_bytes(self.model)
