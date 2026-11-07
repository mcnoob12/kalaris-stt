---
name: asr-data-audit
description: Audit a speech dataset before training or evaluation. Use when adding a corpus, or when a dataset's quality, duration, transcripts or language mix is unknown.
---

# ASR data audit

Produce `reports/data/<dataset>.md` before a dataset enters training.

1. Check `configs/datasets.yaml`. If the dataset isn't there, add it and run `license-provenance` first.
2. Build a manifest with the fields in `stt_eval/manifest.py`. Sample 200 clips with a fixed seed, not the first 200.
3. Measure these and record them:
   - total hours, clips, speakers, and hours per speaker
   - duration percentiles; flag clips under 0.5 s or over 30 s
   - sample rate and channels, and whether audio was upsampled from telephony
   - characters per second; values far from the median suggest bad alignment
   - script mix: share of Devanagari, Latin and digit tokens
   - exact duplicate audio by checksum, and duplicate transcripts
4. Listen to, or run a baseline model over, the sample. Report how often transcripts disagree with the audio. Use dev data only, never test data.
5. Conclude with one of: use, use after filtering (state the rule), or reject (state why).

Report hours per speaker and script mix even when the dataset looks fine. They predict most later surprises.
