import json

import pytest

from stt_eval.cli import main
from stt_eval.leakage import check_leakage
from stt_eval.manifest import read_manifest, write_jsonl


def make_clip(tmp_path, clip_id, content, speaker="s1", text="short words", split="train"):
    path = tmp_path / f"{clip_id}.bin"
    path.write_bytes(content)
    return {
        "id": clip_id,
        "audio_filepath": str(path),
        "text": text,
        "duration": 1.0,
        "language": "hi",
        "speaker": speaker,
        "dataset": "demo",
        "split": split,
        "license": "CC-BY-4.0",
    }


def test_clean_split_passes(tmp_path):
    train = [make_clip(tmp_path, "a", b"one", speaker="s1")]
    test = [make_clip(tmp_path, "b", b"two", speaker="s2", split="test")]
    assert check_leakage(train, test).is_clean


def test_catches_planted_duplicate_audio(tmp_path):
    train = [make_clip(tmp_path, "a", b"same bytes", speaker="s1")]
    test = [make_clip(tmp_path, "b", b"same bytes", speaker="s2", split="test")]
    report = check_leakage(train, test)
    assert not report.is_clean
    assert report.duplicate_audio == [("a", "b")]


def test_catches_shared_speaker(tmp_path):
    train = [make_clip(tmp_path, "a", b"one", speaker="s1")]
    test = [make_clip(tmp_path, "b", b"two", speaker="s1", split="test")]
    report = check_leakage(train, test)
    assert not report.is_clean
    assert report.shared_speakers == ["demo:s1"]


def test_same_speaker_id_in_another_corpus_is_not_a_match(tmp_path):
    train = [make_clip(tmp_path, "a", b"one", speaker="s1")]
    other = make_clip(tmp_path, "b", b"two", speaker="s1", split="test")
    other["dataset"] = "other"
    assert check_leakage(train, [other]).is_clean


def test_catches_reencoded_copy_through_its_transcript(tmp_path):
    sentence = "यह एक लंबा वाक्य है जो दोहराया गया है"
    train = [make_clip(tmp_path, "a", b"flac", speaker="s1", text=sentence)]
    test = [make_clip(tmp_path, "b", b"opus", speaker="s2", text=sentence + "।", split="test")]
    assert check_leakage(train, test).shared_transcripts


def test_short_common_phrases_are_not_flagged(tmp_path):
    train = [make_clip(tmp_path, "a", b"one", speaker="s1", text="धन्यवाद")]
    test = [make_clip(tmp_path, "b", b"two", speaker="s2", text="धन्यवाद", split="test")]
    assert check_leakage(train, test).is_clean


def test_manifest_rejects_missing_fields_and_duplicates(tmp_path):
    path = tmp_path / "bad.jsonl"
    write_jsonl(path, [{"id": "a"}])
    with pytest.raises(ValueError, match="missing"):
        read_manifest(path)

    clip = make_clip(tmp_path, "a", b"one")
    write_jsonl(path, [clip, clip])
    with pytest.raises(ValueError, match="duplicate"):
        read_manifest(path)


def setup_run(tmp_path, split):
    manifest = tmp_path / "m.jsonl"
    write_jsonl(manifest, [make_clip(tmp_path, "a", b"one", text="नमस्ते दुनिया", split=split)])
    hypotheses = tmp_path / "h.jsonl"
    write_jsonl(hypotheses, [{"id": "a", "text": "नमस्ते"}])
    return str(manifest), str(hypotheses)


def test_score_reports_aggregates(tmp_path, capsys):
    manifest, hypotheses = setup_run(tmp_path, "test")
    assert main(["score", "--manifest", manifest, "--hypotheses", hypotheses, "--language", "hi"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["wer"]["rate"] == 50.0
    assert "ref" not in json.dumps(report).replace("reference_length", "")


def test_score_fails_when_a_clip_has_no_hypothesis(tmp_path):
    manifest, _ = setup_run(tmp_path, "test")
    empty = tmp_path / "empty.jsonl"
    write_jsonl(empty, [])
    with pytest.raises(SystemExit, match="missing 1 clips"):
        main(["score", "--manifest", manifest, "--hypotheses", str(empty), "--language", "hi"])


def test_worst_refuses_test_sets(tmp_path, capsys):
    manifest, hypotheses = setup_run(tmp_path, "test")
    code = main(["worst", "--manifest", manifest, "--hypotheses", hypotheses, "--language", "hi"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "Refused" in captured.err


def test_worst_works_on_dev_sets(tmp_path, capsys):
    manifest, hypotheses = setup_run(tmp_path, "dev")
    assert main(["worst", "--manifest", manifest, "--hypotheses", hypotheses, "--language", "hi"]) == 0
    assert json.loads(capsys.readouterr().out)["edits"] == 1


def test_leakage_command_exit_code_fails_on_a_leak(tmp_path):
    train, test = tmp_path / "train.jsonl", tmp_path / "test.jsonl"
    write_jsonl(train, [make_clip(tmp_path, "a", b"same")])
    write_jsonl(test, [make_clip(tmp_path, "b", b"same", speaker="s2", split="test")])
    assert main(["leakage", "--train", str(train), "--held-out", str(test)]) == 1
