from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import Backend, local_path_bytes

SPIECE_SPACE = "▁"


def ctc_greedy(token_ids: np.ndarray, blank: int) -> list[int]:
    """Collapse repeats, then remove blanks."""
    result: list[int] = []
    previous = -1
    for token in token_ids.tolist():
        if token != previous and token != blank:
            result.append(token)
        previous = token
    return result


class IndicConformerOnnx(Backend):
    """AI4Bharat IndicConformer 600M through the CTC head of an ONNX repack.

    The repack ships CTC only. The upstream model also has an RNNT head, so these
    scores may understate what the official model can reach.
    """

    name = "indicconformer-onnx"

    def __init__(self, model: str, language: str | None):
        super().__init__(model)
        if not language:
            raise ValueError("IndicConformer needs a language to mask its vocabulary")
        self.language = language

    def _root(self) -> Path:
        root = Path(self.model)
        if root.is_dir():
            return root
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(self.model))

    def load(self) -> None:
        import onnxruntime as ort

        root = self._root()
        self._root_path = root
        options = ort.SessionOptions()
        providers = ["CPUExecutionProvider"]
        self._frontend = ort.InferenceSession(str(root / "nemo128.onnx"), options, providers=providers)
        self._encoder = ort.InferenceSession(str(root / "encoder-model.onnx"), options, providers=providers)
        self._ctc = ort.InferenceSession(str(root / "ctc_decoder-model.onnx"), options, providers=providers)

        spans = json.loads((root / "language_spans.json").read_text(encoding="utf-8"))
        self._blank = spans["blank_token_id"]
        span = spans["languages"][self.language]
        self._start, self._length = span["start"], span["length"]
        self._vocab = (root / "vocab.txt").read_text(encoding="utf-8").split("\n")

    def transcribe(self, audio_path: str) -> str:
        import soundfile

        samples, rate = soundfile.read(audio_path, dtype="float32")
        if rate != 16000:
            raise ValueError(f"Expected 16 kHz audio, got {rate} Hz: {audio_path}")

        waveform = samples[np.newaxis, :]
        lengths = np.array([waveform.shape[1]], dtype=np.int64)
        names = [i.name for i in self._frontend.get_inputs()]
        features, feature_lengths = self._frontend.run(None, dict(zip(names, (waveform, lengths))))
        names = [i.name for i in self._encoder.get_inputs()]
        encoded, encoded_lengths = self._encoder.run(None, dict(zip(names, (features, feature_lengths))))
        logits = self._ctc.run(None, {self._ctc.get_inputs()[0].name: encoded})[0][0]

        # The vocabulary axis is identified by size, so export layout can't flip it.
        if logits.shape[0] == self._blank + 1:
            logits = logits.T
        logits = logits[: int(encoded_lengths[0])]

        # Mask to this language's token span plus the shared blank.
        allowed = np.concatenate([logits[:, self._start : self._start + self._length], logits[:, [self._blank]]], axis=1)
        picks = allowed.argmax(axis=1)
        local_blank = self._length
        tokens = ctc_greedy(picks, local_blank)
        text = "".join(self._vocab[self._start + token] for token in tokens)
        return text.replace(SPIECE_SPACE, " ").strip()

    def model_bytes(self) -> int | None:
        return local_path_bytes(self._root_path)
