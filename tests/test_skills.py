"""Checks skills against the Agent Skills format, so agents can discover them."""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / ".agents" / "skills"
EXPECTED = {
    "asr-data-audit", "license-provenance", "indic-text-normalize", "speaker-disjoint-split",
    "nemo-transducer-finetune", "rl-sequence-finetune", "experiment-tracking",
    "asr-eval-wer", "asr-error-analysis",
    "quant-search", "edge-export", "device-bench", "autoresearch",
    "dataset-sourcing", "kaggle-gpu-training",
}


def frontmatter(path: Path) -> dict:
    match = re.match(r"---\n(.*?)\n---\n", path.read_text(encoding="utf-8"), re.DOTALL)
    assert match, f"{path} has no frontmatter"
    return yaml.safe_load(match.group(1))


def test_all_planned_skills_exist():
    assert {p.name for p in SKILLS.iterdir() if p.is_dir()} == EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_skill_format(name):
    meta = frontmatter(SKILLS / name / "SKILL.md")
    assert meta["name"] == name
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name) and len(name) <= 64
    assert 0 < len(meta["description"]) <= 1024


def test_skills_only_cite_stt_eval_commands_that_exist():
    from stt_eval.cli import build_parser

    commands = set(build_parser()._subparsers._group_actions[0].choices)
    for path in SKILLS.glob("*/SKILL.md"):
        for command in re.findall(r"stt-eval (\w+)", path.read_text(encoding="utf-8")):
            assert command in commands, f"{path} cites unknown command: {command}"
