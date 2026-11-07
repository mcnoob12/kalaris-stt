# Evaluation sets

Phase 0 uses two public test sets, chosen because they're open and small enough for this Mac's disk. They can show that the harness works, but they can't support a SOTA claim.

| Language | Set | Clips | Audio | Source | Status |
|---|---|---|---|---|---|
| English | LibriSpeech test-clean | 2,620 | 5.40 hours | [OpenSLR](https://www.openslr.org/12) | In use |
| Hindi | FLEURS `hi_in` test | 418 | 1.34 hours | [google/fleurs](https://huggingface.co/datasets/google/fleurs) | In use |
| Hinglish | None | | | | **Blocked** |

## Limits of these sets

- **LibriSpeech** is clean read speech, and only one of the seven English sets behind the Bark comparison.
- **FLEURS Hindi** is read speech, so it says little about conversational or noisy Hindi. It has no speaker IDs, so comparisons resample by clip and intervals may be too narrow. We also don't know whether any baseline model trained on it, which could flatter that model.

## What a SOTA claim needs

| Claim | Sets needed | What's blocking it |
|---|---|---|
| English against Bark's 4.29 and 4.94 | LibriSpeech test-clean and test-other, AMI, Earnings-22, GigaSpeech, SPGISpeech, VoxPopuli | Disk space and download work |
| Hindi | IndicVoices and Kathbath Hindi, plus Vistaar's Hindi sets | A person must accept the gated terms; Vistaar's terms are unconfirmed |
| Hinglish | HiACC, MUCS-2021 or our own collection | Original sources and licenses not found |

## Published numbers

Published numbers are context only, because each maker scores differently. Parakeet v3's seven-set average is 4.94 on the Bark card and 6.32 in the October 2025 leaderboard paper. Any comparison we publish must come from rescoring baselines with this repo.

## Rules for using test sets

- Agents see aggregate test scores only, never per-clip results. `stt-eval worst` refuses test manifests.
- Tuning decisions use dev sets, and **no dev sets exist yet**. They arrive with the Phase 1 data work.
- A test run for a candidate model needs human approval.
- A leakage check runs before any score from a model we trained is reported.

The registry in [`configs/datasets.yaml`](../configs/datasets.yaml) tracks every dataset and model with its license status. Only a person can mark a license `confirmed`, and none are confirmed yet.
