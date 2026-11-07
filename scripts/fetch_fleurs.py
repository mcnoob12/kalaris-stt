"""Download a FLEURS language split (Hindi test is about 300 MB) and write a manifest."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pyarrow.parquet as pq
import soundfile
from huggingface_hub import hf_hub_download

from stt_eval.manifest import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="hi_in")
    parser.add_argument("--language", default="hi")
    parser.add_argument("--split", default="test")
    parser.add_argument("--root", default="data/fleurs")
    parser.add_argument("--out", default="manifests/fleurs_hi_test.jsonl")
    args = parser.parse_args()

    parquet = hf_hub_download(
        "google/fleurs",
        f"parquet-data/{args.config}/{args.split}-00000-of-00001.parquet",
        repo_type="dataset",
    )
    audio_dir = (Path(args.root) / args.config / args.split).resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(parquet)
    print("columns:", table.column_names)

    rows = []
    for index, record in enumerate(table.to_pylist()):
        path = audio_dir / f"{index:05d}.flac"
        samples, rate = soundfile.read(io.BytesIO(record["audio"]["bytes"]), dtype="float32")
        if rate != 16000:
            raise ValueError(f"Clip {index} is {rate} Hz. This script expects 16 kHz.")
        soundfile.write(path, samples, rate, format="FLAC")
        rows.append({
            "id": f"fleurs-{args.config}-{args.split}-{index:05d}",
            "audio_filepath": str(path),
            "text": record["raw_transcription"],
            "duration": len(samples) / rate,
            "language": args.language,
            "speaker": str(record["speaker_id"]) if record.get("speaker_id") is not None else None,
            "dataset": f"fleurs-{args.config}",
            "split": args.split,
            "license": "CC-BY-4.0",
        })
    write_jsonl(args.out, rows)
    print(f"{len(rows)} clips, {sum(r['duration'] for r in rows) / 3600:.2f} hours -> {args.out}")


if __name__ == "__main__":
    main()
