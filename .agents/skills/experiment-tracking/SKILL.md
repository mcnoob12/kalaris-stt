---
name: experiment-tracking
description: Record ASR experiments so results can be reproduced and compared. Use when starting any training run, bake-off or quantization search.
---

# Experiment tracking

Every run writes one folder, `runs/<date>-<short-name>/`, and one line in `reports/experiments.md`.

**Before the run**, record the question it answers and the result that would change a decision. If no result would change a decision, don't spend the compute.

**Record in the folder:**
- commit hash, and whether the working tree was clean
- config, tokenizer and manifest hashes
- machine, GPU, runtime versions and seed
- cost: GPU-hours and money
- metrics per language, not just blended

**Log line:** date, question, compared-to, result with CI, decision.

Log failed runs too. They stop the next agent from repeating them.

Any tracker works (SwanLab, W&B, TensorBoard), but the folder and log line are the source of truth. Don't send audio or transcripts to a hosted tracker without human approval.
