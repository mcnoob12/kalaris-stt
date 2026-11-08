# Evaluation sets

Our benchmarks now follow the Hugging Face Open ASR Leaderboard, so our numbers can sit beside its table. [`configs/benchmarks.yaml`](../configs/benchmarks.yaml) lists them, and [`reports/leaderboard_snapshot.md`](../reports/leaderboard_snapshot.md) holds the standings.

| Language | Benchmark | Metric | Size | Access | Status |
|---|---|---|---|---|---|
| Hindi | [Monsoon hi-IN](https://huggingface.co/datasets/VoiceArena/MonsoonASR-Open-ASR-leaderboard-hi-IN) | OIWER | 753 clips, 1.33 hours, conversational | Open | **Primary, in use** |
| Indian English | [Monsoon en-IN](https://huggingface.co/datasets/VoiceArena/MonsoonASR-Open-ASR-leaderboard-en-IN) | WER | 5.62 hours | Open | Configured, not fetched |
| English | [Leaderboard average](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard) | WER | Six more sets | Open | Configured, not fetched (disk) |
| English and Hindi | LibriSpeech test-clean and FLEURS Hindi | WER | 2,620 and 418 clips, read speech | Open | Secondary checks |
| Hindi | Lahaja | WER | 12.5 hours, 132 speakers | Gated | Needs terms accepted |
| Hinglish | HiACC (on Zenodo, not Hugging Face) | WER | 5.24 hours, 44 speakers | Open, non-commercial | Evaluation only, not fetched |

## Why Monsoon

- It's real phone audio of spontaneous speech from 468 speakers, rather than read speech.
- It's what the leaderboard's Hindi column ranks.
- Its answer key accepts spelling variants, including Latin spellings of Hindi and loanwords. A model writing `improve` or `इंप्रूव` is correct either way, which suits a code-switching model.

## OIWER

`stt-eval score` uses the official `voi-oiwer` scorer automatically when a manifest has lattices. The scorer normalizes text itself, and the corpus rate is total edits over total reference words.

## Limits

- **The public Hindi set is small.** At 1.33 hours, a small lead may not be significant, and the private set (4.47 hours) is only scored by the maintainers after submission.
- **The leaderboard's RTFx runs on an H200 GPU.** It doesn't measure on-device speed, so edge claims need our own phone benchmarks.
- **Hindi and code-switching overlap only partly.** Monsoon has some English loanwords, but it doesn't measure heavy Hinglish. HiACC is the only real test we found, and it's small.
- **Whether a training set overlaps Monsoon is unknown.** Check speakers and transcripts with `stt-eval leakage` before claiming a score.

## Rules

- Agents see aggregate test scores only, never per-clip results. `stt-eval worst` refuses test manifests.
- `training/prepare_data.py` refuses test splits and anything in `configs/benchmarks.yaml`.
- Tuning decisions use speaker-disjoint dev sets, which `prepare_data.py` builds.
- Published comparisons must use the same set and metric, because set mixes shift averages. Parakeet v3's English average is 4.94 on the Bark card, 6.32 in the October 2025 paper and 7.04 on today's leaderboard.

[`configs/datasets.yaml`](../configs/datasets.yaml) tracks every dataset and model with its license status. Only a person can mark a license `confirmed`, and none are confirmed yet.
