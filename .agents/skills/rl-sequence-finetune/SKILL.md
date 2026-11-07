---
name: rl-sequence-finetune
description: Run reinforcement-learning sequence training (MWER, EDRL) after supervised ASR training. Use only for a final fine-tuning stage on a converged model, and never as the main training method.
---

# RL sequence fine-tuning

**Status:** Research risk. Published gains are on RNN-T and seq2seq, and no TDT result has been found. TDT's duration actions are the open problem.

## Why this comes last

RL needs a model that already decodes mostly correctly. On an untrained model, the reward carries almost no signal.

## Order of work

1. Start from the best supervised checkpoint, and record its dev WER as the baseline.
2. Implement **MWER** first. It works on decoded hypotheses, so it doesn't need per-action duration handling. It samples hypotheses, rewards those better than average, and penalizes those worse.
3. Try **EDRL** (per-action edit-distance reward) only if MWER shows a gain or you have a specific reason. Its RNN-T formulation must be adapted to TDT's durations, so write the adaptation down before coding.
4. Use a low learning rate, and blend with the transducer loss. Pure policy gradient tends to be unstable.

## Gate

```bash
.venv/bin/stt-eval compare --manifest <dev manifest> --a <supervised hyps> --b <rl hyps> --language hi
```

Keep RL only if `excludes_zero` is true and `difference` is negative for the target language, with no significant regression in the others. Otherwise, write up the result and ship the supervised model.

## Where OpenEnv fits

OpenEnv doesn't belong in this gradient loop. Run this stage as plain PyTorch inside NeMo. See `quant-search` for where OpenEnv does fit.
