"""Inference adapters. Each imports its runtime only when it is selected."""

from __future__ import annotations

from .base import Backend


def load_backend(name: str, model: str, language: str | None = None) -> Backend:
    if name == "parakeet-mlx":
        from .parakeet_mlx_backend import ParakeetMLX

        return ParakeetMLX(model)
    if name == "whisper-mlx":
        from .whisper_mlx_backend import WhisperMLX

        return WhisperMLX(model, language)
    if name == "sherpa-onnx-transducer":
        from .sherpa_onnx_backend import SherpaOnnxTransducer

        return SherpaOnnxTransducer(model)
    if name == "indicconformer-onnx":
        from .indicconformer_onnx_backend import IndicConformerOnnx

        return IndicConformerOnnx(model, language)
    if name == "fluidaudio-coreml":
        from .fluidaudio_backend import FluidAudioCoreML

        return FluidAudioCoreML(model)
    raise ValueError(f"Unknown backend: {name}")
