---
name: asr-error-analysis
description: Find out why an ASR model makes errors and what to fix next. Use after scoring a dev set, or when deciding what data or training change to try.
---

# ASR error analysis

Dev sets only. `stt-eval worst` refuses test manifests, and that refusal must stand.

```bash
.venv/bin/stt-eval worst --manifest <dev manifest> --hypotheses runs/<name>.jsonl --language hi --count 50
```

1. Split the error rate before reading examples: by language, switching level, duration, speaker and dataset. Look for a few speakers or one corpus dominating.
2. Read about 50 errors, and tag each one:
   - bad reference transcript
   - normalization only (nukta, numbers, spelling)
   - English in Devanagari, or Hindi in Latin
   - named entity or rare word
   - noise, overlap or far-field audio
   - clipped audio or hallucinated text on silence
3. Report counts per tag, with the share of total edits each one could remove.
4. Propose the single change with the largest expected gain, and name the dev metric that will prove it.

If many errors are bad references, fix the data before changing the model. A noisy dev set hides real progress.
