# Phase 0 baseline report

Measured on 2026-11-07 on an Apple M1 Pro with 16 GB of memory (not the base M4 named in the plan). Reproduce with `./scripts/run_baselines.sh`.

## Hindi

| Run | Model | Set | Clips | Metric | Score | Leaderboard |
|---|---|---|---|---|---|---|
| `indicconformer_monsoon` | IndicConformer 600M, CTC-only repack | Monsoon hi-IN | 753 | OIWER | **11.06** | 8.78 (official model) |
| `whisper_turbo_monsoon` | Whisper large-v3-turbo | Monsoon hi-IN | 753 | OIWER | **41.69** | 41.55 |
| `indicconformer_hi` | IndicConformer 600M, CTC-only repack | FLEURS Hindi test | 418 | WER | 12.20 | |
| `whisper_turbo_hi` | Whisper large-v3-turbo | FLEURS Hindi test | 418 | WER | 32.22 | |
| `parakeet_mlx_hi` | Parakeet TDT 0.6B v3 | FLEURS Hindi test | 418 | WER | 99.94 | |

- IndicConformer beats Whisper turbo on Monsoon by 30.63 points (95% CI -34.38 to -27.49, resampled by speaker).
- Parakeet v3 has no Hindi vocabulary, so 99.94 is just its starting point before fine-tuning.
- FLEURS has no speaker IDs, and its numbers use strict normalization.

## English

| Run | Model | Set | Clips | WER | CER |
|---|---|---|---|---|---|
| `whisper_turbo_en` | Whisper large-v3-turbo, MLX | LibriSpeech test-clean | 2,620 | 1.91 | 0.65 |
| `parakeet_mlx_en` | Parakeet TDT 0.6B v3, MLX | LibriSpeech test-clean | 2,620 | 1.99 | 0.68 |
| `parakeet_onnx_en` | Parakeet TDT 0.6B v3, int8, sherpa-onnx CPU | LibriSpeech test-clean | 2,620 | 2.32 | 0.85 |

These use only one clean-speech set, so they say little about the leaderboard's English average.

## Speed and size

| Run | Runtime | RTFx | p50 per clip | Peak RSS | Peak GPU | Model on disk |
|---|---|---|---|---|---|---|
| `parakeet_onnx_en` | ONNX int8, 4 CPU threads | 31.2 | 0.19 s | 1,848 MB | | 640 MB |
| `parakeet_mlx_en` | MLX | 24.8 | 0.22 s | 2,105 MB | 2,684 MB | 2,392 MB |
| `indicconformer_hi` | ONNX FP32, CPU | 18.2 | 0.58 s | 2,620 MB | | 2,381 MB |
| `indicconformer_monsoon` | ONNX FP32, CPU | 14.0 | 0.38 s | 2,642 MB | | 2,381 MB |
| `whisper_turbo_hi` | MLX | 9.9 | 1.12 s | 1,771 MB | 2,370 MB | 1,539 MB |
| `whisper_turbo_en` | MLX | 8.3 | 0.80 s | 1,810 MB | 2,379 MB | 1,539 MB |
| `whisper_turbo_monsoon` | MLX | 3.1 | 1.96 s | 2,105 MB | 2,366 MB | 1,539 MB |

**Treat these as rough.** The Parakeet runs had the machine mostly to themselves, but the IndicConformer and Whisper runs shared it with NeMo installs and training tests. Whisper's 3.1 on Monsoon is the most affected. Rerun them alone before quoting any speed.

The ONNX int8 build is 640 MB and costs 0.33 points on this set, an early sign of what compression can save. It's far above the 200 MB target.

## Reproduction checks

| Check | Ours | Published | Result |
|---|---|---|---|
| Whisper turbo, Monsoon Hindi, OIWER | 41.69 | 41.55 | **Pass:** 0.14 apart, so our Hindi scoring matches the leaderboard |
| Parakeet v3, LibriSpeech test-clean, WER | 1.99 | 1.52 on the Bark card | **Fail:** 0.47 apart, over the 0.3 tolerance set beforehand |
| IndicConformer, Monsoon Hindi, OIWER | 11.06 | 8.78 | **Not comparable:** the repack is CTC-only and we don't know which head the leaderboard ran |

**The Parakeet gap is unexplained.** It isn't the normalizer: the leaderboard's own normalizer gives 1.97. The likeliest cause is the MLX runtime differing from NeMo, but that's untested. NeMo can test it on a GPU machine, since the 2.5 GB download doesn't fit this disk's safety margin. Until then, don't compare MLX English scores with published NeMo scores.

## Not run

- **Bark 3.0:** its non-commercial license isn't cleared for internal benchmarking. It also isn't on the leaderboard.
- **Official IndicConformer:** gated, so someone must accept its terms.
- **Core ML:** no adapter yet.
- **Hinglish:** HiACC hasn't been fetched yet.
- **Other English leaderboard sets:** there's no disk room for them on this Mac.
