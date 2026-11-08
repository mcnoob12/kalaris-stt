"""Build NeMo training and dev manifests from Hugging Face datasets.

Usage:
    .venv/bin/python training/prepare_data.py training/configs/wave1.yaml

Safeguards:
- Refuses any source whose license isn't `confirmed` in configs/datasets.yaml.
  `--smoke` allows `listed` sources, for pipeline tests whose weights are thrown away.
- Refuses test splits and anything listed in configs/benchmarks.yaml.
- Keeps dev speakers out of training, by hashing each speaker to one side.
"""

from __future__ import annotations

import argparse
import hashlib
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import soundfile
import yaml

from stt_eval.hf_audio import SAMPLE_RATE, iter_records, to_16k_mono
from stt_eval.manifest import REPO_ROOT, write_jsonl

ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"))


def clean_text(text: str) -> str:
    """Training text keeps what users should see. It only fixes encoding noise."""
    return " ".join(unicodedata.normalize("NFC", text).translate(ZERO_WIDTH).split())


def rejection_reason(text: str, duration: float, limits: dict) -> str | None:
    if not text:
        return "empty_text"
    if duration < limits["min_seconds"]:
        return "too_short"
    if duration > limits["max_seconds"]:
        return "too_long"
    rate = len(text) / duration
    if rate < limits["min_chars_per_second"] or rate > limits["max_chars_per_second"]:
        return "bad_alignment"
    return None


def is_dev_speaker(dataset: str, speaker: str | None, dev_fraction: float) -> bool:
    """Assign a speaker to one side, stably. Clips with no speaker stay in training."""
    if not speaker or dev_fraction <= 0:
        return False
    digest = hashlib.sha256(f"{dataset}:{speaker}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < dev_fraction


def check_source_allowed(source: dict, registry: dict, benchmarks: dict, *, smoke: bool) -> None:
    name, repo, split = source["name"], source["repo"], source["split"]
    if split.startswith("test"):
        raise SystemExit(f"{name}: refusing to train on test split {split}")

    benchmark_repos = {b["repo"] for b in benchmarks["primary"].values()}
    benchmark_repos |= {b["repo"] for b in benchmarks["english_average"]}
    if repo in benchmark_repos:
        raise SystemExit(f"{name}: {repo} is a benchmark, so it can't be used for training")

    entry = next((d for d in registry["datasets"] if d["name"] == source["registry"]), None)
    if entry is None:
        raise SystemExit(f"{name}: {source['registry']} isn't in configs/datasets.yaml")
    allowed = {"confirmed", "listed"} if smoke else {"confirmed"}
    if entry["status"] not in allowed:
        raise SystemExit(
            f"{name}: {source['registry']} has status '{entry['status']}'. A person must review its "
            "license and set status: confirmed in configs/datasets.yaml."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--smoke", action="store_true", help="Allow unreviewed licenses for a throwaway pipeline test")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    registry = yaml.safe_load((REPO_ROOT / "configs" / "datasets.yaml").read_text(encoding="utf-8"))
    benchmarks = yaml.safe_load((REPO_ROOT / "configs" / "benchmarks.yaml").read_text(encoding="utf-8"))
    sources = [s for s in config["sources"] if s.get("enabled", True)]
    for source in sources:
        check_source_allowed(source, registry, benchmarks, smoke=args.smoke)

    output = REPO_ROOT / "data" / "training" / config["name"]
    limits = config["limits"]
    train: list[dict] = []
    dev: dict[str, list[dict]] = defaultdict(list)
    report = []

    for source in sources:
        name = source["name"]
        audio_dir = output / "audio" / name
        rejected: Counter = Counter()
        seconds = 0.0
        kept = 0
        max_seconds = source.get("max_hours", float("inf")) * 3600
        text_column, speaker_column = source["text_column"], source.get("speaker_column")

        for record in iter_records(source["repo"], source.get("config", "default"), source["split"], audio_dir):
            if seconds >= max_seconds:
                break
            samples = to_16k_mono(record["audio"]["bytes"])
            duration = len(samples) / SAMPLE_RATE
            text = clean_text(record[text_column] or "")
            reason = rejection_reason(text, duration, limits)
            if reason:
                rejected[reason] += 1
                continue

            path = audio_dir / f"{kept:07d}.flac"
            soundfile.write(path, samples, SAMPLE_RATE, format="FLAC")
            speaker = str(record[speaker_column]) if speaker_column and record.get(speaker_column) is not None else None
            row = {
                "id": f"{name}-{kept:07d}",
                "audio_filepath": str(path),
                "text": text,
                "duration": duration,
                "language": source["language"],
                "speaker": speaker,
                "dataset": name,
                "split": "train",
                "license": source["license"],
            }
            if is_dev_speaker(name, speaker, source.get("dev_fraction", 0.0)):
                dev[source["language"]].append({**row, "split": "dev"})
            else:
                # `repeat` oversamples scarce sources, such as real Hinglish.
                train.extend([row] * source.get("repeat", 1))
            seconds += duration
            kept += 1
        report.append(f"{name}: {kept} clips, {seconds / 3600:.2f} hours, rejected {dict(rejected)}")

    write_jsonl(output / "train.json", train)
    for language, rows in dev.items():
        write_jsonl(output / f"dev_{language}.json", rows)

    summary = "\n".join(report + [f"train rows: {len(train)}", f"dev rows: { {k: len(v) for k, v in dev.items()} }"])
    (output / "report.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
