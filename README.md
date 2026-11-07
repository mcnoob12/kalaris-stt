# Kalaris STT

Evaluation harness, agent skills and search environment for the Kalaris Labs speech-to-text model. Wave 1 covers Hindi, Hinglish and English.

No model has been trained here yet. This is Phase 0: it measures the baselines future models must beat.

- [Baseline results](reports/baseline.md)
- [Agent rules](AGENTS.md)
- [Dataset and model registry](configs/datasets.yaml)
- [Strategy document](https://claude.ai/artifact/6XuzBJBNuZJofz7BxF1aC9)

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install -e '.[dev,mlx,env,onnx,data]'
.venv/bin/python -m pytest
```

## Reproduce the baselines

This downloads about 8 GB, and stops if free disk space falls below 10 GB. Finished runs are skipped when it's re-run.

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
