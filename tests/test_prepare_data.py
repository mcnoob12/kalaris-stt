import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_data", ROOT / "training" / "prepare_data.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

LIMITS = {"min_seconds": 0.5, "max_seconds": 20, "min_chars_per_second": 2, "max_chars_per_second": 40}
REGISTRY = {"datasets": [
    {"name": "Good", "status": "confirmed"},
    {"name": "Unreviewed", "status": "listed"},
    {"name": "Banned", "status": "excluded"},
]}
BENCHMARKS = {
    "primary": {"hi": {"repo": "VoiceArena/Monsoon"}},
    "english_average": [{"repo": "hf-audio/open-asr-leaderboard"}],
}


def source(**overrides):
    return {"name": "s", "repo": "org/data", "split": "train", "registry": "Good", **overrides}


def test_confirmed_source_is_allowed():
    prepare.check_source_allowed(source(), REGISTRY, BENCHMARKS, smoke=False)


def test_unreviewed_license_is_refused_unless_smoke():
    with pytest.raises(SystemExit, match="status: confirmed"):
        prepare.check_source_allowed(source(registry="Unreviewed"), REGISTRY, BENCHMARKS, smoke=False)
    prepare.check_source_allowed(source(registry="Unreviewed"), REGISTRY, BENCHMARKS, smoke=True)


def test_excluded_data_is_refused_even_for_smoke():
    with pytest.raises(SystemExit):
        prepare.check_source_allowed(source(registry="Banned"), REGISTRY, BENCHMARKS, smoke=True)


def test_unregistered_data_is_refused():
    with pytest.raises(SystemExit, match="isn't in"):
        prepare.check_source_allowed(source(registry="Mystery"), REGISTRY, BENCHMARKS, smoke=True)


@pytest.mark.parametrize("split", ["test", "test.clean"])
def test_test_splits_are_refused(split):
    with pytest.raises(SystemExit, match="test split"):
        prepare.check_source_allowed(source(split=split), REGISTRY, BENCHMARKS, smoke=True)


@pytest.mark.parametrize("repo", ["VoiceArena/Monsoon", "hf-audio/open-asr-leaderboard"])
def test_benchmarks_cant_be_used_for_training(repo):
    with pytest.raises(SystemExit, match="benchmark"):
        prepare.check_source_allowed(source(repo=repo), REGISTRY, BENCHMARKS, smoke=True)


@pytest.mark.parametrize(
    "text, duration, reason",
    [
        ("", 5, "empty_text"),
        ("abc", 0.1, "too_short"),
        ("abc " * 50, 30, "too_long"),
        ("a", 10, "bad_alignment"),
        ("a" * 500, 5, "bad_alignment"),
        ("यह एक सामान्य वाक्य है", 2.5, None),
    ],
)
def test_rejection_reasons(text, duration, reason):
    assert prepare.rejection_reason(text, duration, LIMITS) == reason


def test_clean_text_fixes_encoding_but_keeps_what_users_see():
    assert prepare.clean_text("  मेरा‍  Phone, ठीक है। ") == "मेरा Phone, ठीक है।"


def test_speaker_assignment_is_stable_and_never_splits_a_speaker():
    first = prepare.is_dev_speaker("d", "speaker-7", 0.5)
    assert all(prepare.is_dev_speaker("d", "speaker-7", 0.5) == first for _ in range(5))


def test_dev_fraction_is_about_right():
    picked = sum(prepare.is_dev_speaker("d", f"s{i}", 0.1) for i in range(5000))
    assert 400 < picked < 600


def test_clips_without_speakers_stay_in_training():
    assert prepare.is_dev_speaker("d", None, 0.5) is False
