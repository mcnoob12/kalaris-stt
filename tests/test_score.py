import pytest

from stt_eval.score import Errors, char_errors, corpus_wer, paired_bootstrap, word_errors


@pytest.mark.parametrize(
    "ref, hyp, subs, ins, dels",
    [
        ("a b c", "a b c", 0, 0, 0),
        ("a b c", "a x c", 1, 0, 0),
        ("a b c", "a c", 0, 0, 1),
        ("a b", "a b c", 0, 1, 0),
        ("a b c d", "x c d e", 1, 1, 1),
        ("", "a b", 0, 2, 0),
        ("a b", "", 0, 0, 2),
    ],
)
def test_word_error_counts(ref, hyp, subs, ins, dels):
    errors = word_errors(ref, hyp)
    assert (errors.substitutions, errors.insertions, errors.deletions) == (subs, ins, dels)


def test_known_answer_against_textbook_example():
    # "kitten" -> "sitting" has a Levenshtein distance of exactly 3.
    assert char_errors("kitten", "sitting").edits == 3


def test_wer_can_exceed_100_percent():
    assert word_errors("a", "x y z").rate == 300.0


def test_corpus_wer_weights_by_words_not_by_clip():
    # A mean of per-clip rates would give 50%. Corpus WER is 1 edit over 11 words.
    errors = corpus_wer(["a", "b b b b b b b b b b"], ["x", "b b b b b b b b b b"])
    assert errors.rate == pytest.approx(100 / 11)


def test_corpus_wer_rejects_misaligned_lists():
    with pytest.raises(ValueError):
        corpus_wer(["a", "b"], ["a"])


def test_cer_ignores_spaces():
    assert char_errors("a b", "ab").edits == 0


def test_empty_reference_rate_is_nan_not_zero():
    assert Errors().rate != Errors().rate


def clip(edits, length=10):
    return Errors(substitutions=edits, reference_length=length)


def test_bootstrap_detects_a_real_improvement():
    baseline = [clip(3)] * 200
    candidate = [clip(1)] * 200
    result = paired_bootstrap(baseline, candidate)
    assert result.difference == pytest.approx(-20.0)
    assert result.excludes_zero


def test_bootstrap_does_not_call_identical_models_different():
    errors = [clip(i % 4) for i in range(200)]
    result = paired_bootstrap(errors, errors)
    assert result.difference == 0
    assert not result.excludes_zero


def test_bootstrap_noise_does_not_pass_the_gate():
    # Wins and losses cancel, so the gate must stay shut.
    a = [clip(2), clip(4)] * 100
    b = [clip(4), clip(2)] * 100
    assert not paired_bootstrap(a, b).excludes_zero


def test_grouping_by_speaker_widens_the_interval():
    # One lucky speaker drives the whole gain. Per-clip resampling is overconfident.
    a = [clip(2)] * 200
    b = [clip(0)] * 20 + [clip(2)] * 180
    speakers = ["lucky"] * 20 + [f"s{i % 5}" for i in range(180)]
    per_clip = paired_bootstrap(a, b)
    per_speaker = paired_bootstrap(a, b, groups=speakers)
    assert (per_speaker.ci_high - per_speaker.ci_low) > (per_clip.ci_high - per_clip.ci_low)


def test_bootstrap_is_reproducible():
    a = [clip(i % 5) for i in range(50)]
    b = [clip((i + 1) % 5) for i in range(50)]
    assert paired_bootstrap(a, b, seed=7) == paired_bootstrap(a, b, seed=7)


def test_bootstrap_rejects_misaligned_clips():
    with pytest.raises(ValueError):
        paired_bootstrap([clip(1, length=10)], [clip(1, length=11)])


def test_lattice_comparison_allows_different_reference_lengths():
    # OIWER measures each model against the spellings it matched.
    a = [clip(3, length=10)] * 200
    b = [clip(1, length=11)] * 200
    result = paired_bootstrap(a, b, lattice=True)
    assert result.wer_a == pytest.approx(30.0)
    assert result.wer_b == pytest.approx(100 / 11)
    assert result.excludes_zero
