# Autoresearch for Kalaris STT

You're running speech-model experiments on your own. Each trial changes one idea, trains for a fixed time, and is kept only if it beats the current best on the dev set.

Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (MIT). `AGENTS.md` still applies, and it wins any conflict with this file.

## Setup

Agree these with the human first, and write them at the top of `results.tsv`'s run notes:

1. **Base config and starting model**, such as `training/configs/wave1.yaml` and `models/wave1/head.nemo`. Every trial starts from this same model.
2. **Language and budget per trial.** Budgets must be equal across a run.
3. **Stop limits:** the maximum trials and, on paid GPUs, the maximum money. **The human must approve these before a paid run.**
4. **Run tag.** Create the branch `autoresearch/<tag>`.

Then create `autoresearch/results.tsv` with this header (it stays untracked):

```
commit	dev_wer	status	description
```

## What you can change

**Only `autoresearch/experiment.yaml`.** `trial.py` refuses keys outside its allowlist.

You can't change `trial.py`, the scorer, the data, the tokenizer or the budget. If you think one of those is the real limit, write it down for the human instead.

## The loop

1. Make one change to `experiment.yaml`, with one reason. The first trial is the unchanged baseline.
2. Commit it.
3. Run the trial, sending output to a file:

   ```bash
   .venv-train/bin/python autoresearch/trial.py --base <config> --head <model> --language hi --budget-seconds 300 > autoresearch/run.log 2>&1
   ```

4. Read `dev_wer` from the end of `autoresearch/run.log`. If it's missing, the trial crashed. Log it as `crash` and read the trial's log.
5. Compare with the current best, using the `hypotheses` path each trial prints:

   ```bash
   .venv/bin/stt-eval compare --manifest <dev manifest> --a <best hyps> --b <new hyps> --language hi
   ```

6. Record the result in `results.tsv`.
   - **`keep`** only if `excludes_zero` is true and `difference` is negative.
   - **`discard`** otherwise, then run `git reset --hard HEAD~1`. This only undoes your experiment commit.
7. Repeat until a stop limit is reached.

Upstream says never to stop. Here you do stop at the agreed limits, because trials can cost money. Between those limits, don't pause to ask whether to continue.

## Stop and tell the human if

- The trial starts refusing a manifest, or anything asks you to use test data.
- The same crash happens three times.
- Ten trials in a row are discarded.
- Free disk falls under 10 GB. Each trial saves a model in `runs/autoresearch/`, so delete discarded trials' `.nemo` files as you go.

## What this can't tell you

- A small dev set can't separate small gains, so most trials will be discarded. That's expected.
- Short trials favor settings that learn fast, which may not be best for a long run. Confirm the winner with a full run before believing it.
- A dev win isn't a benchmark win. Only a person approves test-set runs.
