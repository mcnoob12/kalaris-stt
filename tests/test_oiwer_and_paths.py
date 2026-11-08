import json

import pytest

from stt_eval.cli import main
from stt_eval.manifest import REPO_ROOT, read_manifest, relative_to_repo, write_jsonl
from stt_eval.score import oiwer_errors

LATTICE = [["इंप्रूव", "इम्प्रूव", "improve"], ["होता", "hota"], ["है", "hai"]]


@pytest.mark.parametrize("hypothesis", ["इंप्रूव होता है", "improve होता है", "improve hota hai"])
def test_any_accepted_spelling_is_correct(hypothesis):
    errors = oiwer_errors(hypothesis, LATTICE)
    assert errors.edits == 0
    assert errors.reference_length == 3


def test_wrong_word_still_counts():
    assert oiwer_errors("खराब होता है", LATTICE).substitutions == 1


def test_missing_word_counts_as_deletion():
    assert oiwer_errors("होता है", LATTICE).deletions == 1


def test_empty_output_counts_every_word():
    assert oiwer_errors("", LATTICE).deletions == 3


def clip(lattice=LATTICE):
    return {
        "id": "a", "audio_filepath": "/x.flac", "text": "इंप्रूव होता है", "duration": 1.0, "language": "hi",
        "speaker": "s", "dataset": "demo", "split": "test", "license": "CC-BY-4.0", "lattice": lattice,
    }


def test_score_uses_oiwer_when_a_lattice_is_present(tmp_path, capsys):
    manifest, hypotheses = tmp_path / "m.jsonl", tmp_path / "h.jsonl"
    write_jsonl(manifest, [clip()])
    write_jsonl(hypotheses, [{"id": "a", "text": "improve hota hai"}])
    assert main(["score", "--manifest", str(manifest), "--hypotheses", str(hypotheses), "--language", "hi"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["metric"] == "oiwer"
    assert report["wer"]["rate"] == 0.0


def test_relative_audio_paths_resolve_against_the_repo(tmp_path):
    row = clip()
    row["audio_filepath"] = "data/x.flac"
    manifest = tmp_path / "m.jsonl"
    write_jsonl(manifest, [row])
    assert read_manifest(manifest)[0]["audio_filepath"] == str(REPO_ROOT / "data" / "x.flac")


def test_relative_to_repo_round_trip():
    assert relative_to_repo(REPO_ROOT / "data" / "x.flac") == "data/x.flac"
