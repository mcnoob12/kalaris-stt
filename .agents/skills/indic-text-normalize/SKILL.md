---
name: indic-text-normalize
description: Normalize Hindi and Hinglish text for scoring and training. Use when preparing transcripts, comparing WER across models, or adding a new Indic language.
---

# Indic text normalization

Scoring text and training text have different jobs. Keep them separate.

**Scoring text** uses `stt_eval/normalize.py`:
- NFC is required. It makes `क़` (U+0958) and `क + ़` compare equal.
- Report strict WER first. `--fold-nukta` and `--fold-nasal` show how much of the error is spelling variation, so report them beside strict WER, never instead of it.
- Script-insensitive WER is a heuristic and a lower bound. Always report it with strict WER.

**Training text** keeps what a user should see: punctuation and casing where the corpus has them reliably. Hindi stays in Devanagari and English in Latin.

Watch for these:
- A `\w` regex can split Devanagari words at vowel signs. Test any new rule against `किताब`.
- Numbers: `२०२६`, `2026` and `दो हज़ार छब्बीस` are all legitimate. Digit forms are unified, but spelled-out numbers aren't, so count those errors separately in analysis.
- Romanized Hinglish (`mera phone kharab hai`) can't be detected by script. It shows as `mono`, so don't treat `mono` as "not Hinglish".
- Changing a normalizer changes every score. Rescore all baselines and note it in the report.

Adding a language: add its script's digits and punctuation, add tests with real words, and confirm `pytest` passes.
