from __future__ import annotations

from .base import Backend, huggingface_cache_bytes


class WhisperMLX(Backend):
    name = "whisper-mlx"

    def __init__(self, model: str, language: str | None):
        super().__init__(model)
        # Set the language explicitly so results don't depend on auto-detection.
        self.language = language

    def load(self) -> None:
        import mlx_whisper

        self._mlx_whisper = mlx_whisper
        # mlx-whisper loads lazily on first use. Force it here so load time and
        # transcription time stay separate.
        from mlx_whisper.load_models import load_model

        load_model(self.model)

    def transcribe(self, audio_path: str) -> str:
        result = self._mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=self.model,
            language=self.language,
            task="transcribe",
            temperature=0.0,
            condition_on_previous_text=False,
            verbose=None,
        )
        return result["text"].strip()

    def accelerator_peak_bytes(self) -> int | None:
        import mlx.core as mx

        return mx.get_peak_memory()

    def model_bytes(self) -> int | None:
        return huggingface_cache_bytes(self.model)
