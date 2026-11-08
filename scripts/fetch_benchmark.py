"""Fetch a benchmark from Hugging Face and write a manifest.

Reads the benchmark from configs/benchmarks.yaml. Only clip audio, reference text
and speaker ID are kept: Monsoon also carries personal details such as income and
birth date, which scoring doesn't need.

Usage: scripts/fetch_benchmark.py hi_monsoon
"""

from __future__ import annotations

import argparse
import json

import soundfile
import yaml

from stt_eval.hf_audio import SAMPLE_RATE, iter_records, to_16k_mono
from stt_eval.manifest import REPO_ROOT, relative_to_repo, write_jsonl


def reference_fields(record: dict, spec: dict) -> dict:
    if spec.get("lattice_column"):
        lattice = record[spec["lattice_column"]]
        if isinstance(lattice, str):
            lattice = json.loads(lattice)
        return {"lattice": lattice, "text": " ".join(slot[0] for slot in lattice)}
    for column in (spec.get("text_column"), "text", "raw_transcription", "transcription", "sentence"):
        if column and column in record:
            return {"text": record[column]}
    raise SystemExit(f"No text column found. Columns: {sorted(record)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark")
    parser.add_argument("--limit", type=int, help="Keep only the first N clips")
    args = parser.parse_args()

    config = yaml.safe_load((REPO_ROOT / "configs" / "benchmarks.yaml").read_text(encoding="utf-8"))
    specs = {**config["primary"], **config["secondary"]}
    if args.benchmark not in specs:
        raise SystemExit(f"Unknown benchmark. Choose from: {sorted(specs)}")
    spec = specs[args.benchmark]

    audio_dir = REPO_ROOT / "data" / "benchmarks" / args.benchmark
    rows: list[dict] = []
    for record in iter_records(spec["repo"], spec.get("config", "default"), spec["split"], audio_dir):
        if args.limit and len(rows) >= args.limit:
            break
        samples = to_16k_mono(record["audio"]["bytes"])
        number = len(rows)
        path = audio_dir / f"{number:06d}.flac"
        soundfile.write(path, samples, SAMPLE_RATE, format="FLAC")
        speaker_column = spec.get("speaker_column")
        rows.append({
            "id": f"{args.benchmark}-{number:06d}",
            "audio_filepath": relative_to_repo(path),
            **reference_fields(record, spec),
            "duration": len(samples) / SAMPLE_RATE,
            "language": spec["language"],
            "speaker": str(record[speaker_column]) if speaker_column else None,
            "dataset": args.benchmark,
            "split": "test",
            "license": spec["license"],
        })

    manifest = REPO_ROOT / "manifests" / f"{args.benchmark}.jsonl"
    write_jsonl(manifest, rows)
    hours = sum(r["duration"] for r in rows) / 3600
    print(f"{len(rows)} clips, {hours:.2f} hours -> {relative_to_repo(manifest)}")


if __name__ == "__main__":
    main()
