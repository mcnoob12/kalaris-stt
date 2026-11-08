---
name: kaggle-gpu-training
description: Train Kalaris STT models on a free Kaggle GPU with the generated notebook. Use when a model needs an NVIDIA GPU (any TDT training) and no paid GPU is approved.
---

# Kaggle GPU training

TDT training needs CUDA, and Kaggle's free tier gives a T4 or P100 with sessions of up to 12 hours.

## Build the notebook

```bash
.venv/bin/python training/kaggle/build_notebook.py
```

It writes `training/kaggle/kalaris_v0.ipynb` with copies of the repo's code, so Kaggle never needs access to the private repo. Rebuild after changing any file it lists, and never edit the notebook directly.

## Run it (a person does this)

1. Upload the notebook at kaggle.com, under Code, then New Notebook, then Import.
2. In the right-hand panel, set Accelerator to a GPU and turn Internet on. Internet needs a phone-verified account.
3. Choose Save Version, then Save & Run All, so it keeps running after the tab closes.

Agents don't enter Kaggle credentials or push notebooks unless a person has set up the API token themselves.

## Limits to plan around

- **12 hours per session.** The notebook budgets 11 and stops cleanly. To continue, add the previous output as an input and set `RESUME` (and `HEAD_FROM`, `SKIP_HEAD`) in the settings cell.
- **No bf16 on T4 or P100,** so configs use `16-mixed`.
- **20 GB of saved output.** The notebook keeps only `last.ckpt`. A 0.6B checkpoint with optimizer state is roughly 7 GB.
- **Weekly GPU quota.** About 30 hours is commonly reported, but this is unconfirmed, so check your account settings.

## After a run

Download `/kaggle/working/v0`, then run `stt-eval leakage` against the training manifest before reporting any score. `dev_scores.json` holds dev results only. Benchmark runs need a person's approval.
