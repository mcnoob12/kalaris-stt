"""WER, CER and paired significance testing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Errors:
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    reference_length: int = 0

    @property
    def edits(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def rate(self) -> float:
        """Error rate in percent. Undefined for an empty reference."""
        if self.reference_length == 0:
            return float("nan")
        return 100.0 * self.edits / self.reference_length

    def __add__(self, other: "Errors") -> "Errors":
        return Errors(
            self.substitutions + other.substitutions,
            self.insertions + other.insertions,
            self.deletions + other.deletions,
            self.reference_length + other.reference_length,
        )

    def as_dict(self) -> dict:
        return {**asdict(self), "rate": self.rate}


def align(reference: Sequence[str], hypothesis: Sequence[str]) -> Errors:
    """Count edits with Levenshtein alignment.

    Among minimum-edit alignments, this prefers substitutions over paired
    insertions and deletions, matching standard scoring tools.
    """
    # Each cell holds (edits, deletions, insertions) for a prefix pair.
    previous = [(j, 0, j) for j in range(len(hypothesis) + 1)]
    for i, ref_token in enumerate(reference, start=1):
        current = [(i, i, 0)]
        for j, hyp_token in enumerate(hypothesis, start=1):
            diagonal = previous[j - 1]
            cost = 0 if ref_token == hyp_token else 1
            options = (
                (diagonal[0] + cost, diagonal[1], diagonal[2]),
                (previous[j][0] + 1, previous[j][1] + 1, previous[j][2]),
                (current[j - 1][0] + 1, current[j - 1][1], current[j - 1][2] + 1),
            )
            current.append(min(options, key=lambda cell: cell[0]))
        previous = current
    edits, deletions, insertions = previous[-1]
    return Errors(edits - deletions - insertions, insertions, deletions, len(reference))


def word_errors(reference: str, hypothesis: str) -> Errors:
    return align(reference.split(), hypothesis.split())


def char_errors(reference: str, hypothesis: str) -> Errors:
    """CER over characters, with spaces removed so WER and CER stay independent."""
    return align(list(reference.replace(" ", "")), list(hypothesis.replace(" ", "")))


def total(errors: Sequence[Errors]) -> Errors:
    result = Errors()
    for item in errors:
        result = result + item
    return result


def corpus_wer(references: Sequence[str], hypotheses: Sequence[str]) -> Errors:
    """Corpus WER: total edits over total words, not a mean of per-clip rates."""
    if len(references) != len(hypotheses):
        raise ValueError(f"{len(references)} references but {len(hypotheses)} hypotheses")
    return total([word_errors(r, h) for r, h in zip(references, hypotheses)])


@dataclass(frozen=True)
class BootstrapResult:
    wer_a: float
    wer_b: float
    difference: float  # WER(B) - WER(A), in points. Negative means B is better.
    ci_low: float
    ci_high: float
    resamples: int
    unit: str

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0

    def as_dict(self) -> dict:
        return {**asdict(self), "excludes_zero": self.excludes_zero}


def paired_bootstrap(
    errors_a: Sequence[Errors],
    errors_b: Sequence[Errors],
    *,
    groups: Sequence[str] | None = None,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Confidence interval for WER(B) - WER(A) on the same clips.

    This is the repo's promotion gate. Pass speaker IDs as `groups` whenever
    they exist: clips from one speaker aren't independent, and resampling single
    clips then gives intervals that are too narrow.
    """
    if len(errors_a) != len(errors_b):
        raise ValueError("Paired bootstrap needs the same clips for both models")
    if not errors_a:
        raise ValueError("Paired bootstrap needs at least one clip")
    if any(a.reference_length != b.reference_length for a, b in zip(errors_a, errors_b)):
        raise ValueError("Clips are misaligned: reference lengths differ")

    edits_a = np.array([e.edits for e in errors_a], dtype=np.int64)
    edits_b = np.array([e.edits for e in errors_b], dtype=np.int64)
    lengths = np.array([e.reference_length for e in errors_a], dtype=np.int64)

    if groups is not None:
        if len(groups) != len(errors_a):
            raise ValueError("Need one group per clip")
        names, inverse = np.unique(np.array(groups), return_inverse=True)
        edits_a = np.bincount(inverse, weights=edits_a, minlength=len(names))
        edits_b = np.bincount(inverse, weights=edits_b, minlength=len(names))
        lengths = np.bincount(inverse, weights=lengths, minlength=len(names))

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(lengths), size=(resamples, len(lengths)))
    words = lengths[picks].sum(axis=1)
    usable = words > 0
    diffs = (
        100.0 * (edits_b[picks].sum(axis=1)[usable] - edits_a[picks].sum(axis=1)[usable])
        / words[usable]
    )
    tail = 100.0 * (1.0 - confidence) / 2.0
    low, high = np.percentile(diffs, [tail, 100.0 - tail])

    total_words = lengths.sum()
    wer_a = 100.0 * edits_a.sum() / total_words
    wer_b = 100.0 * edits_b.sum() / total_words
    return BootstrapResult(
        float(wer_a), float(wer_b), float(wer_b - wer_a), float(low), float(high),
        resamples, "group" if groups is not None else "clip",
    )
