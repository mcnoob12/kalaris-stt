"""Run one fixed-budget experiment and report one metric. Agents never edit this.

Adapted from https://github.com/karpathy/autoresearch (MIT): one editable file, a
fixed time budget, one metric that decides. Here the model is a NeMo speech model
and the metric is dev WER from the same scorer as every baseline.

Usage:
    .venv-train/bin/python autoresearch/trial.py --base training/configs/smoke.yaml \
        --head models/smoke/head.nemo --language hi --budget-seconds 300
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Everything the agent may change. Data, tokenizer, metric and budget aren't here.
ALLOWED = {
    "spec_augment": None,
    "stages": {"full": {"lr": None, "min_lr": None, "layer_decay": None, "weight_decay": None, "warmup_steps": None}},
    "data": {"batch_size": None, "accumulate": None, "max_duration": None},
}


def check_allowed(experiment: dict, allowed: dict = ALLOWED, path: str = "") -> None:
    for key, value in experiment.items():
        where = f"{path}{key}"
        if key not in allowed:
            raise SystemExit(f"experiment.yaml can't set {where}. Allowed keys are in trial.py.")
        if isinstance(value, dict):
            if not isinstance(allowed[key], dict):
                raise SystemExit(f"experiment.yaml: {where} must be a single value")
            check_allowed(value, allowed[key], f"{where}.")
        elif isinstance(allowed[key], dict):
            raise SystemExit(f"experiment.yaml: {where} must be a section")


def merge(base: dict, overrides: dict) -> dict:
    result = dict(base)
    for key, value in overrides.items():
        result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


def check_not_test(manifest: Path) -> None:
    """The reward must never come from a test set."""
    with open(manifest, encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and json.loads(line).get("split") == "test":
                raise SystemExit(f"{manifest} contains test clips. Trials may only use dev data.")


def run(command: list[str], log: Path, timeout: int) -> None:
    with open(log, "a", encoding="utf-8") as handle:
        subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=timeout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Training config the experiment applies to")
    parser.add_argument("--head", required=True, help="Fixed starting model, so trials compare fairly")
    parser.add_argument("--language", required=True, choices=["en", "hi", "hinglish"])
    parser.add_argument("--budget-seconds", type=int, default=300)
    parser.add_argument("--experiment", default="autoresearch/experiment.yaml")
    args = parser.parse_args()

    experiment = yaml.safe_load((ROOT / args.experiment).read_text(encoding="utf-8")) or {}
    check_allowed(experiment)
    base = yaml.safe_load((ROOT / args.base).read_text(encoding="utf-8"))
    check_not_test(ROOT / base["dev_manifest"])

    trial_dir = ROOT / "runs" / "autoresearch" / time.strftime("%Y%m%d-%H%M%S")
    trial_dir.mkdir(parents=True)
    config = merge(base, experiment)
    config["output_dir"] = str(trial_dir)
    config["init_from"] = str(ROOT / args.head)
    config["stages"]["full"]["max_steps"] = 10_000_000  # The budget ends training, not a step count.
    # Cosine decay needs a horizon, so a budgeted trial holds its rate constant unless told otherwise.
    config["stages"]["full"].setdefault("min_lr", config["stages"]["full"]["lr"])
    config["stages"]["full"]["checkpoint_every"] = 10_000_000
    config_path = trial_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    log = trial_dir / "run.log"
    python = sys.executable
    started = time.time()
    # A trial past twice its budget plus evaluation has hung, so it fails.
    timeout = args.budget_seconds * 2 + 600
    run([python, "training/finetune.py", str(config_path), "--stage", "full", "--max-seconds", str(args.budget_seconds)], log, timeout)
    train_seconds = time.time() - started

    hypotheses = trial_dir / "dev.jsonl"
    run([python, "-m", "stt_eval.cli", "transcribe", "--backend", "nemo", "--model", str(trial_dir / "full.nemo"),
         "--manifest", base["dev_manifest"], "--out", str(hypotheses), "--language", args.language], log, timeout)
    score = subprocess.run(
        [python, "-m", "stt_eval.cli", "score", "--manifest", base["dev_manifest"],
         "--hypotheses", str(hypotheses), "--language", args.language],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    wer = json.loads(score.stdout)["wer"]["rate"]

    print("---")
    print(f"dev_wer:          {wer:.4f}")
    print(f"train_seconds:    {train_seconds:.1f}")
    print(f"peak_child_rss_mb: {resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20:.0f}")
    print(f"hypotheses:       {hypotheses.relative_to(ROOT)}")
    print(f"log:              {log.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
