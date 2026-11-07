---
name: quant-search
description: Search per-layer quantization for an ASR model using the OpenEnv environment. Use when shrinking a trained flagship toward the 200 MB mid-range target.
---

# Quantization search

**Status:** `envs/quant_env` is a working skeleton with a **stub evaluator**, so its numbers aren't real. Replace the stub before trusting any policy.

## Order of work

1. Start from a frozen flagship checkpoint.
2. Try uniform settings first (FP16, int8, 4-bit). They set the baseline a search must beat.
3. Run the environment: an agent assigns bits to each layer group, and reward combines dev WER, size and latency.
4. Run quantization-aware fine-tuning on the best two or three policies. Search results alone understate the final accuracy.
5. Confirm with `stt-eval compare` on the dev set, then `device-bench` on the reference phone.

## Guardrails

- Reward uses dev-set metrics only. The environment takes manifest paths, and test manifests must never be passed to it.
- Score policies on a fixed dev shard, and confirm winners on the full dev set, since small shards reward luck.
- Pass paths and metrics across the OpenEnv boundary, never audio.
- Keep the environment thin and its OpenEnv version pinned, because the API may change.

## What to measure

Download size, peak RAM and speed are separate goals. Bark's loader expands packed weights to FP32 in memory, so a small file can still use a lot of RAM. One independent test found 4-bit MLX lost 0.6 to 1.0 points on LibriSpeech with no speed gain on that Mac. Lower bits are only a win if the target device gains something.
