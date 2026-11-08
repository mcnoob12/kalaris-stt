---
name: autoresearch
description: Run an autonomous loop of fixed-budget speech-model training experiments, keeping only changes that beat the best dev WER. Use when tuning training settings after the pipeline passes its smoke tests.
---

# Autoresearch

Read `autoresearch/program.md` and follow it. It adapts karpathy/autoresearch to our trainer and scorer.

Check these first:
- `configs/smoke.yaml` and, for TDT, `configs/smoke_tdt.yaml` have passed their overfit gates. Tuning a broken pipeline wastes every trial.
- A dev manifest exists with real speaker-disjoint data. Overfit data proves mechanics only.
- The human has agreed the budget per trial and the stop limits, and approved any paid compute.
