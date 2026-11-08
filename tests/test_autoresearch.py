import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("trial", ROOT / "autoresearch" / "trial.py")
trial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trial)


def test_shipped_experiment_is_allowed():
    trial.check_allowed(yaml.safe_load((ROOT / "autoresearch" / "experiment.yaml").read_text()))


@pytest.mark.parametrize(
    "experiment",
    [
        {"dev_manifest": "manifests/hi_monsoon.jsonl"},
        {"train_manifest": "x"},
        {"tokenizer_dir": "x"},
        {"base_model": "x"},
        {"stages": {"full": {"max_steps": 1}}},
        {"stages": {"head": {"lr": 1}}},
        {"data": "x"},
        {"spec_augment": {"nested": True}},
    ],
)
def test_agent_cant_reach_outside_the_allowlist(experiment):
    with pytest.raises(SystemExit):
        trial.check_allowed(experiment)


def test_merge_overrides_only_what_it_names():
    base = {"stages": {"full": {"lr": 1, "max_steps": 5}}, "seed": 0}
    assert trial.merge(base, {"stages": {"full": {"lr": 2}}}) == {"stages": {"full": {"lr": 2, "max_steps": 5}}, "seed": 0}
    assert base["stages"]["full"]["lr"] == 1


def write_manifest(path, split):
    path.write_text(json.dumps({"id": "a", "split": split}) + "\n", encoding="utf-8")


def test_test_clips_cant_be_a_reward(tmp_path):
    manifest = tmp_path / "m.json"
    write_manifest(manifest, "test")
    with pytest.raises(SystemExit, match="test clips"):
        trial.check_not_test(manifest)


def test_dev_clips_are_fine(tmp_path):
    manifest = tmp_path / "m.json"
    write_manifest(manifest, "dev")
    trial.check_not_test(manifest)
