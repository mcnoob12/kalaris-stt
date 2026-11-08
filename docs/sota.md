# Can we be SOTA?

**Yes in some categories, but not overall.** The strongest openings are on-device Hindi and Hinglish, where no one is clearly competing. Beating everyone on Hindi outright, or on the English average, is unlikely at first. These odds are our judgment from the [leaderboard snapshot](../reports/leaderboard_snapshot.md) of 2026-11-07, since no model has been trained yet.

| Goal | Bar to beat | Odds | Why |
|---|---|---|---|
| Best Hindi at about 200 MB on device | No competitor at that size on the leaderboard | **Good** | Requires strong Hindi to survive compression. A quantized IndicConformer-class model would have a large lead over small models. |
| Best open real Hinglish result | No public benchmark found | **Good, but weak as a claim** | We'd largely define the test ourselves, and data is the blocker |
| Best open-weights Hindi | 8.78 (IndicConformer 600M, MIT) | **Plausible** | Needs more or better data than AI4Bharat used, which isn't guaranteed |
| Number one Hindi outright | 6.54 (fonada/kapture-v3) | **Long shot** | That's a 25% error cut on the open leader, and other players keep improving |
| English average, overall | 4.87 (zoom/scribe_v2.1_pro) | **Unlikely** | Parakeet v3 sits at 7.04, and the leaders are mostly closed |
| English parity with Parakeet | 7.04 | **Likely** | Should hold if English stays in the training mix |

## Where small models stand on Hindi

| Model | Parameters | Hindi OIWER |
|---|---|---|
| ai4bharat/indic-conformer-600m | 600M | 8.78 |
| nvidia/nemotron-3.5-asr-streaming-0.6b | 0.6B | 14.18 |
| facebook/omniASR-LLM-300M-v2 | 300M | 18.28 |
| Qwen/Qwen3-ASR-0.6B-hf | 0.6B | 18.57 |
| facebook/omniASR-CTC-300M-v2 | 300M | 37.58 |

For reference, Whisper large-v3 scores 29.86 and turbo scores 41.55. A 200 MB model under about 14 would beat every model of this size class here except IndicConformer. No published 200 MB Hindi figure exists on the board, and we haven't measured compression's cost on Hindi.

## What the claim should say

The honest first headline is: "the most accurate Hindi speech model that runs on a mid-range phone," with Monsoon scores and phone benchmarks published together.

The leaderboard measures speed on an H200 GPU and has no on-device track, so the edge half of that claim needs our own published benchmark.

## Goals these numbers changed

- **English targets:** our 4.29 and 4.94 came from the Bark card's set mix, which isn't today's average. Replace them with parity against Parakeet on the current sets.
- **Base model:** IndicConformer is now the Hindi favorite, and Parakeet still looks stronger on English and speed. The Phase 2 bake-off decides. IndicConformer's export used AI4Bharat's NeMo fork, so loading it in NeMo 3.0 needs testing.

## What decides the outcome

1. **Hindi data.** IndicVoices, Kathbath, Shrutilipi and Vaani need their terms accepted. These may be what IndicConformer trained on already, so extra data must come from somewhere else to gain an edge.
2. **Hinglish data we may train on commercially.** We haven't found any. HiACC and the Sonexis sample are non-commercial, and addyo07's source is unknown.
3. **Whether quantization holds up on Hindi.** Nobody has shown this at 200 MB, and it's the heart of the strongest claim.
4. **GPU budget.** Nothing about the flagship can happen without it.

## Limits of a win

- The public Hindi set is 1.33 hours, and the leaderboard also keeps a private 4.47-hour set. A public lead can shrink there.
- Models and scores change weekly. Rerun `scripts/snapshot_leaderboard.py` before any claim.
- Monsoon has real limits, as its authors note: speakers concentrate in the Hindi Belt, and it doesn't cover all Indian languages.
