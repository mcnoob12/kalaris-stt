# Kalaris STT

We are building an on-device speech-to-text model. Wave 1 covers Hindi, Hinglish and English: a flagship first, then a quantized mid-range model.

## Hard rules

1. **Never look at test-set audio or per-clip test results.** Use only aggregates from `stt-eval score` on test manifests. `stt-eval worst` refuses test sets, so don't work around it.
2. **Run a leakage check before reporting any score from a model we trained.** Use `stt-eval leakage --train ... --held-out ...` and report its result.
3. **Only a human can promote a model or start paid compute.** Propose these and wait for a yes. This includes GPU rentals, paid APIs, leaderboard submissions and publishing weights.
4. **Don't use a dataset or model until it has a confirmed license** in `configs/datasets.yaml`. Don't build on Bark 3.0 weights (CC BY-NC) or Krutrim IndicST (conflicting terms).
5. **Keep at least 10 GB of disk free.** This Mac's disk is nearly full. Check with `df -h .` before any download.

## Claims

- A model beats another only if `stt-eval compare` gives a 95% CI that excludes zero.
- Label every number as measured, vendor-reported or estimated.
- Report the machine, normalization options, clip count and commit with every score.
- Compare a new model only with baselines rescored by this repo, or with Open ASR Leaderboard numbers on the same set and metric.
- Never train or tune on anything in `configs/benchmarks.yaml`.

## Machines

| Machine | Use for | Don't use for |
|---|---|---|
| M1 Pro and M4, 16 GB | Evaluation, Core ML and MLX export, quantization search, RNNT smoke tests | Anything needing the TDT loss, which requires CUDA |
| i7 with RTX 3050 Ti 4 GB | TDT smoke tests on Parakeet 110M, ONNX CPU benchmarks | Training a 0.6B model |
| Kaggle free tier | Bake-offs that resume from checkpoints | Flagship runs |
| Rented RTX 4090 or A100 | Flagship training (human approval needed) | |

## Commands

```bash
uv pip install -e '.[dev,mlx,env,onnx,data]'
.venv/bin/python -m pytest
./scripts/run_baselines.sh
.venv/bin/stt-eval score --manifest manifests/fleurs_hi_test.jsonl --hypotheses runs/whisper_turbo_hi.jsonl --language hi
```

## Layout

- `stt_eval/`: scoring, normalization, leakage checks, benchmarks and backends
- `manifests/`: benchmark clip lists (JSONL); `data/`, `models/` and `runs/` are gitignored
- `configs/datasets.yaml`: dataset and model registry with license status
- `configs/benchmarks.yaml`: what we report against, following the Open ASR Leaderboard
- `training/`: data preparation, tokenizer and fine-tuning. Read `training/README.md` first.
- `autoresearch/`: agent loop for tuning experiments (`program.md`, `experiment.yaml`, `trial.py`)
- `envs/quant_env/`: OpenEnv quantization environment (its evaluator is a stub for now)
- `docs/`: project status, decisions, evaluation sets and harness guide
- `reports/`: reports, starting with `baseline.md`
- `.agents/skills/`: procedures for specific jobs. Read the matching skill first.
