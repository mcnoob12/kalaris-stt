"""Command-line entry point: transcribe, score, compare, leakage, worst."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .bench import BenchResult, peak_rss_bytes
from .leakage import check_leakage
from .manifest import read_jsonl, read_manifest, write_jsonl
from .normalize import normalize, switching_level, to_script_insensitive
from .score import char_errors, oiwer_errors, paired_bootstrap, total, word_errors


def _options(args: argparse.Namespace) -> dict:
    options = {}
    if getattr(args, "fold_nukta", False):
        options["fold_nukta"] = True
    if getattr(args, "fold_nasal", False):
        options["fold_nasal"] = True
    return options


def _paired(manifest: list[dict], hypotheses_path: str) -> list[tuple[dict, str]]:
    """Join clips to hypotheses. A missing clip is an error, never a skip,
    because dropping hard clips would silently flatter a model."""
    hypotheses = {row["id"]: row["text"] for row in read_jsonl(hypotheses_path)}
    missing = [row["id"] for row in manifest if row["id"] not in hypotheses]
    if missing:
        raise SystemExit(f"{hypotheses_path} is missing {len(missing)} clips, e.g. {missing[:3]}")
    return [(row, hypotheses[row["id"]]) for row in manifest]


def command_transcribe(args: argparse.Namespace) -> int:
    from .backends import load_backend

    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]

    backend = load_backend(args.backend, args.model, args.language)
    started = time.perf_counter()
    backend.load()
    load_seconds = time.perf_counter() - started

    outputs, latencies = [], []
    for index, row in enumerate(rows, start=1):
        started = time.perf_counter()
        text = backend.transcribe(row["audio_filepath"])
        latency = time.perf_counter() - started
        latencies.append(latency)
        outputs.append({"id": row["id"], "text": text, "seconds": round(latency, 4)})
        if index % 200 == 0:
            print(f"{index}/{len(rows)}", file=sys.stderr, flush=True)

    write_jsonl(args.out, outputs)
    result = BenchResult(
        backend=args.backend,
        model=args.model,
        clips=len(rows),
        audio_seconds=sum(row["duration"] for row in rows),
        load_seconds=load_seconds,
        transcribe_seconds=sum(latencies),
        latency_p50_seconds=float(np.percentile(latencies, 50)),
        latency_p90_seconds=float(np.percentile(latencies, 90)),
        peak_rss_bytes=peak_rss_bytes(),
        accelerator_peak_bytes=backend.accelerator_peak_bytes(),
        model_bytes=backend.model_bytes(),
    )
    bench_path = Path(args.out).with_suffix(".bench.json")
    bench_path.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    print(json.dumps(result.as_dict(), indent=2))
    return 0


def command_score(args: argparse.Namespace) -> int:
    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    pairs = _paired(rows, args.hypotheses)
    options = _options(args)

    if any("lattice" in row for row, _ in pairs):
        if not all("lattice" in row for row, _ in pairs):
            raise SystemExit("Manifest mixes clips with and without lattices")
        result = total([oiwer_errors(hypothesis, row["lattice"]) for row, hypothesis in pairs])
        print(json.dumps({
            "manifest": args.manifest,
            "hypotheses": args.hypotheses,
            "language": args.language,
            "metric": "oiwer",
            "clips": len(pairs),
            "wer": result.as_dict(),
        }, indent=2, ensure_ascii=False))
        return 0

    word, char, loose = [], [], []
    by_level: dict[str, list] = defaultdict(list)
    for row, hypothesis in pairs:
        reference_text = normalize(row["text"], args.language, **options)
        hypothesis_text = normalize(hypothesis, args.language, **options)
        errors = word_errors(reference_text, hypothesis_text)
        word.append(errors)
        char.append(char_errors(reference_text, hypothesis_text))
        if args.language == "hinglish":
            loose.append(word_errors(to_script_insensitive(reference_text), to_script_insensitive(hypothesis_text)))
            by_level[switching_level(row["text"])].append(errors)

    report = {
        "manifest": args.manifest,
        "hypotheses": args.hypotheses,
        "language": args.language,
        "normalization": options or "strict",
        "clips": len(pairs),
        "wer": total(word).as_dict(),
        "cer": total(char).as_dict(),
    }
    if loose:
        report["script_insensitive_wer"] = total(loose).as_dict()
        report["wer_by_switching_level"] = {
            level: {"clips": len(items), **total(items).as_dict()} for level, items in sorted(by_level.items())
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def command_compare(args: argparse.Namespace) -> int:
    rows = read_manifest(args.manifest)
    options = _options(args)

    def errors_for(path: str) -> list:
        return [
            oiwer_errors(text, row["lattice"]) if "lattice" in row
            else word_errors(normalize(row["text"], args.language, **options), normalize(text, args.language, **options))
            for row, text in _paired(rows, path)
        ]

    has_speakers = all(row.get("speaker") for row in rows)
    groups = [f"{row['dataset']}:{row['speaker']}" for row in rows] if has_speakers else None
    lattice = any("lattice" in row for row in rows)
    result = paired_bootstrap(errors_for(args.a), errors_for(args.b), groups=groups, seed=args.seed, lattice=lattice)
    print(json.dumps({"a": args.a, "b": args.b, "language": args.language, **result.as_dict()}, indent=2))
    return 0


def command_leakage(args: argparse.Namespace) -> int:
    report = check_leakage(read_manifest(args.train), read_manifest(args.held_out))
    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    return 0 if report.is_clean else 1


def command_worst(args: argparse.Namespace) -> int:
    """List the worst clips for error analysis. Refuses test sets by design."""
    rows = read_manifest(args.manifest)
    test_clips = sum(1 for row in rows if row["split"] == "test")
    if test_clips:
        print(
            f"Refused: {args.manifest} has {test_clips} test clips. "
            "Per-clip test results would let tuning leak into the test set.",
            file=sys.stderr,
        )
        return 2

    scored = []
    for row, hypothesis in _paired(rows, args.hypotheses):
        reference = normalize(row["text"], args.language)
        output = normalize(hypothesis, args.language)
        errors = word_errors(reference, output)
        scored.append((errors.edits, row["id"], reference, output))
    scored.sort(key=lambda item: item[0], reverse=True)
    for edits, clip_id, reference, output in scored[: args.count]:
        print(json.dumps({"id": clip_id, "edits": edits, "ref": reference, "hyp": output}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stt-eval", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_normalization(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--language", required=True, choices=["en", "hi", "hinglish"])
        sub.add_argument("--fold-nukta", action="store_true", help="Treat ज़ and ज as equal")
        sub.add_argument("--fold-nasal", action="store_true", help="Treat ँ and ं as equal")

    transcribe = commands.add_parser("transcribe", help="Run a model and record speed and memory")
    transcribe.add_argument("--backend", required=True)
    transcribe.add_argument("--model", required=True)
    transcribe.add_argument("--manifest", required=True)
    transcribe.add_argument("--out", required=True)
    transcribe.add_argument("--language", help="Language code for backends that need one")
    transcribe.add_argument("--limit", type=int, help="Use only the first N clips")
    transcribe.set_defaults(handler=command_transcribe)

    score = commands.add_parser("score", help="Report aggregate WER and CER")
    score.add_argument("--manifest", required=True)
    score.add_argument("--hypotheses", required=True)
    score.add_argument("--limit", type=int, help="Score only the first N clips")
    add_normalization(score)
    score.set_defaults(handler=command_score)

    compare = commands.add_parser("compare", help="Paired bootstrap between two models")
    compare.add_argument("--manifest", required=True)
    compare.add_argument("--a", required=True, help="Baseline hypotheses")
    compare.add_argument("--b", required=True, help="Candidate hypotheses")
    compare.add_argument("--seed", type=int, default=0)
    add_normalization(compare)
    compare.set_defaults(handler=command_compare)

    leakage = commands.add_parser("leakage", help="Check a held-out set against training data")
    leakage.add_argument("--train", required=True)
    leakage.add_argument("--held-out", required=True)
    leakage.set_defaults(handler=command_leakage)

    worst = commands.add_parser("worst", help="Show worst clips (dev sets only)")
    worst.add_argument("--manifest", required=True)
    worst.add_argument("--hypotheses", required=True)
    worst.add_argument("--language", required=True, choices=["en", "hi", "hinglish"])
    worst.add_argument("--count", type=int, default=20)
    worst.set_defaults(handler=command_worst)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
