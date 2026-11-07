---
name: asr-eval-wer
description: Score ASR models and decide whether one beats another. Use whenever reporting WER or CER, comparing models, or making a claim about accuracy.
---

# ASR evaluation

```bash
.venv/bin/stt-eval transcribe --backend parakeet-mlx --model <repo> --manifest <m> --out runs/<name>.jsonl --language hi
.venv/bin/stt-eval score --manifest <m> --hypotheses runs/<name>.jsonl --language hi
.venv/bin/stt-eval compare --manifest <m> --a runs/<base>.jsonl --b runs/<new>.jsonl --language hi
```

**Rules**
- Use corpus WER: total edits over total words. `stt-eval` does this.
- Missing clips fail the score. Never drop clips to make a run finish.
- Compare models only on identical clips with the identical normalizer.
- `compare` resamples by speaker when speakers exist, because clips from one speaker aren't independent. Use the CI for decisions, not the point estimate.
- Hinglish: report strict WER, script-insensitive WER and WER by switching level together.
- Use dev sets to choose. Run a test set only for a final, human-approved candidate, and report only aggregates.

**A complete result states:** model and commit, dataset and clip count, language, normalization options, WER, CER, CI against the baseline, and machine. Label each number as measured, vendor-reported or estimated.

**Don't compare across scorers.** Parakeet v3's 7-set average is 4.94 on the Bark card and 6.32 in the October 2025 leaderboard paper. Rescore baselines here instead.
