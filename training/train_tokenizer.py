"""Train one SentencePiece BPE tokenizer shared across scripts.

Wave 2 scripts are included from day one by mixing in FLEURS text for those
languages. No audio is needed for this, and it means later languages won't force
a decoder reset.

Usage:
    .venv-train/bin/python training/train_tokenizer.py \
        --manifest data/training/wave1/train.json --output models/tokenizers/wave1 \
        --vocab-size 4096 --extra-fleurs bn_in,mr_in,te_in,ta_in,gu_in,kn_in,ml_in,pa_in,or_in
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

FLEURS_TSV = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/{config}/train.tsv"


def fleurs_text(config: str) -> list[str]:
    request = urllib.request.Request(FLEURS_TSV.format(config=config), headers={"User-Agent": "kalaris-stt"})
    with urllib.request.urlopen(request, timeout=60) as response:
        rows = csv.reader(response.read().decode("utf-8").splitlines(), delimiter="\t", quotechar='"')
        # Columns: id, file name, raw transcription, ...
        return list({row[2] for row in rows if len(row) > 2})


def script_counts(pieces: list[str]) -> Counter:
    counts: Counter = Counter()
    for piece in pieces:
        for character in piece:
            if character.isalpha():
                counts[unicodedata.name(character, "UNKNOWN").split(" ")[0]] += 1
                break
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--extra-fleurs", default="", help="Comma-separated FLEURS configs for later-wave scripts")
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as handle:
        lines = list({json.loads(line)["text"] for line in handle if line.strip()})
    for config in filter(None, args.extra_fleurs.split(",")):
        lines.extend(fleurs_text(config))

    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import create_spt_model

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as corpus:
        corpus.write("\n".join(lines))

    create_spt_model(
        data_file=corpus.name,
        vocab_size=args.vocab_size,
        sample_size=-1,
        do_lower_case=False,
        tokenizer_type="bpe",
        output_dir=str(output),
    )
    Path(corpus.name).unlink()

    pieces = (output / "tokenizer.vocab").read_text(encoding="utf-8").splitlines()
    print(f"{len(lines)} text lines, {len(pieces)} tokens -> {output}")
    print("tokens per script:", dict(script_counts([p.split("\t")[0] for p in pieces])))


if __name__ == "__main__":
    main()
