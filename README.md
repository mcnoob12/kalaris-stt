# Kalaris STT

Evaluation harness, agent skills and search environment for the Kalaris Labs speech-to-text model. Wave 1 covers Hindi, Hinglish and English.

No production model has been trained here yet. Phase 0 measures the baselines future models must beat, and [`training/`](training/README.md) holds the fine-tuning stack.

| Read this | For |
|---|---|
| [Project status](docs/status.md) | What's done, known issues, and what needs a person |
| [Decisions](docs/decisions.md) | What we chose, why, and what would change it |
| [Evaluation sets](docs/evaluation.md) | Where evals come from and what they can't prove |
| [Evaluation harness](docs/harness.md) | How scoring, benchmarks, skills and the environment work |
| [Baseline results](reports/baseline.md) | Measured scores |
| [Leaderboard snapshot](reports/leaderboard_snapshot.md) | Current public standings |
| [Can we be SOTA?](docs/sota.md) | Where we can realistically win |
| [Training guide](training/README.md) | From Hugging Face data to a fine-tuned model |
| [Agent rules](AGENTS.md) | Hard rules for agents |
| [Dataset and model registry](configs/datasets.yaml) | Sources and license status |
| [Strategy document](https://claude.ai/artifact/6XuzBJBNuZJofz7BxF1aC9) | The full plan, with sources |

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install -e '.[dev,mlx,env,onnx,data]'
.venv/bin/python -m pytest
```

## Reproduce the baselines

This downloads about 9 GB, and stops if free disk space falls below 10 GB. Finished runs are skipped when it's re-run.

```bash
./scripts/run_baselines.sh
```

## Score a run

```bash
.venv/bin/stt-eval score --manifest manifests/fleurs_hi_test.jsonl --hypotheses runs/whisper_turbo_hi.jsonl --language hi
```

## Use the skills in Pi

Pi ignores project files until you trust them. Approve the prompt in an interactive session, or pass `--approve`:

```bash
pi --approve
```

## Free up disk space

Downloads live inside this folder and are gitignored. Deleting them is safe, but the next baseline run downloads them again.

```bash
rm -rf .cache models data
```
