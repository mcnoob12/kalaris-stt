# Evaluation harness

`stt_eval` turns a clip list and a model into scores you can compare. A run has three steps:

```
manifest -> transcribe -> runs/<name>.jsonl and runs/<name>.bench.json
                               |
                               v
                       score or compare
```

## Manifests

A manifest is a JSONL file with one clip per line, checked by `stt_eval/manifest.py`.

| Field | Meaning |
|---|---|
| `id` | Unique clip ID |
| `audio_filepath` | Path to the audio, relative to the repo in committed manifests |
| `lattice` | Optional accepted spellings per word; switches scoring to OIWER |
| `text` | Reference transcript |
| `duration` | Seconds |
| `language` | `en`, `hi` and so on |
| `dataset`, `split`, `license` | Where the clip came from, and its terms |
| `speaker` | Optional in the file format, but needed for speaker-level checks |

## Commands

| Command | What it does |
|---|---|
| `stt-eval transcribe` | Runs a backend, then writes transcripts and speed and memory numbers |
| `stt-eval score` | Reports aggregate WER and CER |
| `stt-eval compare` | Paired bootstrap between two runs on the same clips |
| `stt-eval leakage` | Checks a held-out set against training data; exits 1 on a leak |
| `stt-eval worst` | Lists the worst clips for error analysis; refuses test sets |

Use the commands from `.venv/bin/`, or call them as `.venv/bin/python -m stt_eval.cli`.

## How scores are computed

- **WER** is corpus-level: total edits over total words, not a mean of per-clip rates.
- **CER** ignores spaces.
- **Missing clips fail the run,** because dropping hard clips would flatter a model.
- **English** uses the Whisper normalizer that leaderboards use.
- **Hindi benchmarks with lattices** use the official OIWER scorer, which normalizes text itself.
- **Hindi without lattices** uses NFC, removes zero-width characters and punctuation, and converts Devanagari digits. `--fold-nukta` and `--fold-nasal` are optional, and reports must state them.
- **Hinglish** adds WER by switching level (`mono`, `word`, `phrase`, `sentence`) and script-insensitive WER.

Script-insensitive WER compares consonant skeletons, so `computer` and `कंप्यूटर` match. Unrelated words can collide, which makes it a lower bound on real errors. Report it beside strict WER, never instead of it. Romanized Hinglish can't be detected by script and shows as `mono`.

## The promotion gate

`compare` reports `WER(B) - WER(A)` with a 95% interval, so a negative difference means B is better. B wins only if `excludes_zero` is true and the difference is negative.

It resamples by speaker when every clip has a speaker ID, and by clip otherwise. The output's `unit` field says which one ran. Clip-level intervals can be too narrow, because clips from one speaker aren't independent.

## Benchmarks

Each `.bench.json` records RTFx, p50 and p90 latency per clip, load time, peak RSS, peak accelerator memory, model size and hardware. RTFx excludes model load.

These are whole-clip results on the machine that ran them. They aren't live-microphone latency or phone numbers, and the harness doesn't run on phones yet.

## Leakage checks

`leakage` compares a training manifest with a held-out one in three ways:
- identical audio, by SHA-256
- shared speakers, namespaced by dataset
- shared transcripts of six words or more

Re-encoded audio with a different transcript and unknown speaker would still get through.

## Backends

| Backend | Runtime | Status |
|---|---|---|
| `parakeet-mlx` | MLX | Baselines measured |
| `sherpa-onnx-transducer` | ONNX on CPU | Baselines measured |
| `indicconformer-onnx` | ONNX Runtime on CPU, CTC only | Baselines measured |
| `whisper-mlx` | MLX | Baselines measured |
| `nemo` | NeMo, from `.venv-train` | Written for fine-tuned models, not yet run |

There's no Core ML backend yet. The entry was removed until an adapter exists.

To add a backend, subclass `Backend` in `stt_eval/backends/base.py` with `load()` and `transcribe()`, then register it in `stt_eval/backends/__init__.py`. Import its runtime inside the methods so other backends don't need it installed.

## Quantization environment

`envs/quant_env` is an OpenEnv environment. Each episode assigns a bit width (2, 4, 8 or 16) to each of five layer groups, then scores the policy:

```
reward = -(WER increase) - 5 x (fraction over size target) - 5 x (fraction under speed target)
```

The defaults are 200 MB and 10x realtime. Beating a target earns nothing extra.

**The evaluator is a stub with invented numbers.** Observations carry `evaluator: "stub"` as a reminder. See `envs/quant_env/README.md` to run it.

## Agent skills

`.agents/skills/` holds 13 skills in the Agent Skills format, and `AGENTS.md` holds the hard rules. Pi loads project files only after you trust them, either through its prompt or with `pi --approve`.

| Area | Skills |
|---|---|
| Data | `asr-data-audit`, `license-provenance`, `indic-text-normalize`, `speaker-disjoint-split` |
| Training | `nemo-transducer-finetune`, `rl-sequence-finetune`, `experiment-tracking` |
| Evaluation | `asr-eval-wer`, `asr-error-analysis` |
| Edge | `quant-search`, `edge-export`, `device-bench` |
| Tuning | `autoresearch` |

## Tests

```bash
.venv/bin/python -m pytest
```

Tests cover known-answer WER cases, Hindi normalization, a planted leak, the test-set refusal, the bootstrap gate, the environment's mechanics and the skill format. They don't run real models.
