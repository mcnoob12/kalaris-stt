from __future__ import annotations

from pathlib import Path

from .base import Backend, local_path_bytes


class SherpaOnnxTransducer(Backend):
    """Parakeet-style transducer on sherpa-onnx CPU, the Android-first path."""

    name = "sherpa-onnx-transducer"

    def __init__(self, model: str, num_threads: int = 4):
        super().__init__(model)
        self.num_threads = num_threads

    def load(self) -> None:
        import sherpa_onnx

        root = Path(self.model)

        def pick(stem: str) -> str:
            matches = sorted(root.glob(f"{stem}*.onnx"))
            if not matches:
                raise FileNotFoundError(f"No {stem}*.onnx in {root}")
            return str(matches[0])

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=pick("encoder"),
            decoder=pick("decoder"),
            joiner=pick("joiner"),
            tokens=str(root / "tokens.txt"),
            num_threads=self.num_threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            model_type="nemo_transducer",
            provider="cpu",
        )

    def transcribe(self, audio_path: str) -> str:
        import soundfile

        samples, rate = soundfile.read(audio_path, dtype="float32")
        stream = self._recognizer.create_stream()
        stream.accept_waveform(rate, samples)
        self._recognizer.decode_stream(stream)
        return stream.result.text.strip()

    def model_bytes(self) -> int | None:
        return local_path_bytes(self.model)
