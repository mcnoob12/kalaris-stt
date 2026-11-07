"""Evaluators turn a bit-width policy into metrics.

`StubEvaluator` is a placeholder with invented numbers. It exists to test the
environment's mechanics, and nothing it produces describes a real model. A real
evaluator must quantize the checkpoint, decode a fixed dev shard and measure
speed, and must refuse any manifest containing test clips.
"""

from dataclasses import dataclass
from typing import Protocol

# Share of parameters per layer group. Illustrative values, not measured from Parakeet.
STUB_GROUP_PARAMETERS_M = {
    "encoder_early": 150.0,
    "encoder_mid": 200.0,
    "encoder_late": 230.0,
    "decoder": 15.0,
    "joint": 5.0,
}
# Invented WER cost, in points, for each bit width in each group.
STUB_BIT_PENALTY = {16: 0.0, 8: 0.02, 4: 0.15, 2: 0.6}
STUB_GROUP_SENSITIVITY = {
    "encoder_early": 1.5,
    "encoder_mid": 0.8,
    "encoder_late": 1.0,
    "decoder": 2.0,
    "joint": 2.5,
}


@dataclass(frozen=True)
class Metrics:
    wer: float
    size_mb: float
    rtfx: float


class Evaluator(Protocol):
    name: str
    groups: list[str]
    baseline_wer: float

    def evaluate(self, policy: dict[str, int]) -> Metrics: ...


class StubEvaluator:
    """Deterministic invented metrics, for testing the environment only."""

    name = "stub"
    groups = list(STUB_GROUP_PARAMETERS_M)
    baseline_wer = 5.0

    def evaluate(self, policy: dict[str, int]) -> Metrics:
        size_mb = sum(STUB_GROUP_PARAMETERS_M[g] * policy[g] / 8 for g in self.groups)
        wer = self.baseline_wer + sum(STUB_BIT_PENALTY[policy[g]] * STUB_GROUP_SENSITIVITY[g] for g in self.groups)
        return Metrics(wer=round(wer, 3), size_mb=round(size_mb, 1), rtfx=50.0)
