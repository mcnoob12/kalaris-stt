# Phase 0 baseline report

**Status: in progress. No scores have been recorded yet.** This page fills in from real runs only.

| | |
|---|---|
| Machine | Apple M1 Pro, 16 GB (not the base M4 named in the plan) |
| Command | `./scripts/run_baselines.sh` |
| Scoring | `stt_eval`, strict normalization |
| Started | 2026-11-07 |

## Runs

| Run | Model | Runtime | Eval set | Clips | WER | CER | RTFx | Peak RSS | Model size |
|---|---|---|---|---|---|---|---|---|---|
| `parakeet_mlx_en` | Parakeet TDT 0.6B v3 | MLX | LibriSpeech test-clean | 2,620 | Pending | Pending | Pending | Pending | Pending |
| `parakeet_onnx_en` | Parakeet TDT 0.6B v3, int8 | sherpa-onnx CPU | LibriSpeech test-clean | 2,620 | Pending | Pending | Pending | Pending | Pending |
| `whisper_turbo_en` | Whisper large-v3-turbo | MLX | LibriSpeech test-clean | 2,620 | Pending | Pending | Pending | Pending | Pending |
| `parakeet_mlx_hi` | Parakeet TDT 0.6B v3 | MLX | FLEURS Hindi test | 418 | Pending | Pending | Pending | Pending | Pending |
| `indicconformer_hi` | IndicConformer 600M, CTC repack | ONNX Runtime CPU | FLEURS Hindi test | 418 | Pending | Pending | Pending | Pending | Pending |
| `whisper_turbo_hi` | Whisper large-v3-turbo | MLX | FLEURS Hindi test | 418 | Pending | Pending | Pending | Pending | Pending |

Parakeet v3 has no Hindi vocabulary, so `parakeet_mlx_hi` isn't a real contender. It records the starting point for fine-tuning.

## Not run

- **Bark 3.0:** its non-commercial license isn't cleared for internal benchmarking.
- **Official IndicConformer:** gated, so someone must accept its terms.
- **Core ML:** the adapter isn't built.
- **Hinglish:** no eval set yet.

## Reproduction check

The plan requires Parakeet v3 on LibriSpeech test-clean to match a published number within a set tolerance, or an explanation of the gap. The published reference is 1.52% WER on the [Bark card](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0/blob/main/README.md), marked there as a leaderboard value.

Result: pending. The tolerance is 0.3 points, chosen before seeing the result. That reference used a different runtime, so a small gap is plausible.

## Conditions

Other light work ran on this Mac during the runs, including writing these docs, so speed numbers are approximate. Whether each number holds up will be noted with its result.
