# Training

This folder takes Hugging Face datasets to a fine-tuned `.nemo` model. See [`docs/status.md`](../docs/status.md) for what has been tested.

## Before spending money

| Step | Who | Done when |
|---|---|---|
| Run `hf auth login`, and accept gated terms for the datasets in `configs/wave1.yaml` | Person | `prepare_data.py` can read them |
| Review licenses and set `status: confirmed` in `configs/datasets.yaml` | Person | `prepare_data.py` stops refusing |
| Replace the `VERIFY` column names in `configs/wave1.yaml` | Agent | Possible once access is granted |
| Pass the TDT smoke test on a CUDA GPU | Agent or person | Near-zero WER on the overfit clips |
| Approve a GPU budget | Person | Required by `AGENTS.md` |

## The stages

```bash
# 1. Machine setup, on a rented GPU or Kaggle
./training/setup_gpu.sh

# 2. Build manifests. It refuses unconfirmed licenses, test splits and benchmarks.
.venv/bin/python training/prepare_data.py training/configs/wave1.yaml

# 3. Train the tokenizer, including later-wave scripts from FLEURS text
.venv-train/bin/python training/train_tokenizer.py --manifest data/training/wave1/train.json \
    --output models/tokenizers/wave1 --vocab-size 4096 \
    --extra-fleurs bn_in,mr_in,te_in,ta_in,gu_in,kn_in,ml_in,pa_in,or_in

# 4. Train the new decoder and joint with the encoder frozen, then everything
.venv-train/bin/python training/finetune.py training/configs/wave1.yaml --stage head
.venv-train/bin/python training/finetune.py training/configs/wave1.yaml --stage full

# 5. Score on dev with the same harness as every baseline
.venv-train/bin/python -m stt_eval.cli transcribe --backend nemo --model models/wave1/full.nemo \
    --manifest data/training/wave1/dev_hi.json --out runs/wave1_dev.jsonl
.venv/bin/stt-eval score --manifest data/training/wave1/dev_hi.json --hypotheses runs/wave1_dev.jsonl --language hi
```

Resume an interrupted stage with `--resume models/wave1/checkpoints_full/last.ckpt`.

## Train v0 on Kaggle

v0 is the first real model, trained on open data the owner approved: MUCS Hinglish, FLEURS Hindi and LibriSpeech. [`docs/data-sources.md`](../docs/data-sources.md) describes each one.

```bash
.venv/bin/python training/kaggle/build_notebook.py
```

Upload `training/kaggle/kalaris_v0.ipynb` to Kaggle, set a GPU and turn Internet on, then Save & Run All. The `kaggle-gpu-training` skill covers resuming and limits.

## Smoke tests

| Test | Where | Covers |
|---|---|---|
| `configs/smoke.yaml` | Mac or CPU | Data, tokenizer, stages, saving, overfitting on an RNNT model |
| `configs/smoke_tdt.yaml` | CUDA GPU with 4 GB or more | All of that on the real TDT loss |

```bash
.venv/bin/python training/prepare_data.py training/configs/smoke.yaml --smoke
.venv-train/bin/python training/train_tokenizer.py --manifest data/training/smoke/train.json \
    --output models/tokenizers/smoke --vocab-size 64
.venv-train/bin/python training/finetune.py training/configs/smoke_tdt.yaml --stage head
.venv-train/bin/python training/finetune.py training/configs/smoke_tdt.yaml --stage full
```

## Tuning with autoresearch

After the smoke tests pass and real dev data exists, [`autoresearch/program.md`](../autoresearch/program.md) runs an agent loop. It's adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch): the agent edits only `autoresearch/experiment.yaml`, each trial gets a fixed time budget, and a change stays only if it beats the best dev WER with a 95% CI that excludes zero. Upstream's code isn't installed, because it trains a text GPT and needs NVIDIA. Paid runs need agreed stop limits first.

## Things that will bite you

- **NeMo drops the TDT loss.** In NeMo 3.0, `change_vocabulary` rebuilds a plain RNNT loss with the wrong class count and raises no error. `finetune.py` rebuilds the right loss and checks it before training. Keep that if you change the trainer.
- **TDT training needs CUDA.** NeMo raises an error on CPU and Mac.
- **Tokenizer swaps can collapse.** A public Parakeet fine-tune on 50 hours fell into repeating one token, reaching 109.5% WER. `CollapseGuard` stops runs like that, and head training goes first for this reason. Don't unfreeze until the head stage is clearly learning.
- **Don't train on benchmarks.** Training on Monsoon or other leaderboard sets would invalidate every score we publish.

## Untested on purpose

`configs/wave1.yaml` has never been run. Its learning rates and step counts are starting points, and batch size must be measured on the first rental.
