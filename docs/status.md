# Project status

As of 2026-11-07, the baselines are done and the training stack is built. **Real training is blocked on data access, license review, a GPU test and a budget, all of which need a person.** See [Can we be SOTA?](sota.md) for where this can win.

## What's been tested

| Item | State | Evidence |
|---|---|---|
| Hindi scoring | **Works** | Whisper turbo scores 41.69, against the leaderboard's 41.55 |
| Baselines | **Done** | [`reports/baseline.md`](../reports/baseline.md) |
| Benchmarks from Hugging Face | **Works** | Monsoon hi-IN fetched and scored |
| Data prep, tokenizer, head and full stages | **Works on RNNT** | A small model overfits 7 clips to 0.00% WER on this Mac |
| TDT training | **Not tested** | NeMo requires CUDA, so it needs the 3050 Ti or Kaggle |
| TDT loss fix | **Partly tested** | Log shows the wrong loss before the fix and TDT after it, but no TDT training step has run |
| v0 model (`training/configs/v0.yaml`) | **Ready to run on Kaggle** | Open data approved by the owner. The notebook builds and its code imports; it hasn't run on Kaggle yet. |
| `training/configs/wave1.yaml` | **Never run** | Needs gated data and a GPU; column names are unverified |
| Autoresearch loop | **Written, not run** | Unit tests cover the allowlist and test-data refusal. No trial has run end to end yet. |
| OpenEnv quantization environment | Skeleton, stub evaluator | Unit tests, `openenv validate`, one live episode |
| Skills in Pi | Pi lists them all | Not run through Pi, because its Radius account has a $0.00 balance |
| Core ML | **Not built** | Backend entry removed until an adapter exists |

## Known issues

| Issue | Effect | Plan |
|---|---|---|
| Parakeet on LibriSpeech is 0.47 above published | MLX English scores can't be compared with published NeMo ones | Test NeMo on a GPU machine |
| The IndicConformer repack is CTC only | Its 11.06 may understate the model | Rerun the official model once its terms are accepted |
| Most baseline speeds shared the machine | Speeds are rough | Rerun alone, and on the base M4 |
| FLEURS has no speaker IDs | Its intervals may be too narrow | Treat it as a secondary check |
| Disk is about 97% full | Blocks more datasets and NeMo Parakeet here | Free space, or use the GPU machine |
| Hinglish data is MUCS only: technical tutorials under share-alike terms | v0 may be weak on everyday Hinglish | License or collect conversational data |

## What needs a person, in order

- [ ] **Run the v0 notebook on Kaggle** (`training/kaggle/kalaris_v0.ipynb`). It's also the first real TDT test.
- [ ] **Run `hf auth login`** and accept terms for IndicVoices, Kathbath, Shrutilipi, Vaani, Lahaja and official IndicConformer
- [ ] **Review the remaining licenses.** FLEURS, LibriSpeech and MUCS are confirmed; MUCS's share-alike terms still need a legal check before release.
- [ ] **Approve a GPU budget**, or apply for IndiaAI compute
- [ ] **Decide how to get Hinglish data**: license it or collect it
- [ ] Choose a reference mid-range Android phone
- [ ] Add Radius credits if you want agents running through Pi
- [ ] Decide whether Bark can be benchmarked internally

"pi harness" is confirmed to mean the Pi coding agent.

## Next steps once unblocked

1. Fill in the `VERIFY` columns in `training/configs/wave1.yaml`, and audit the data.
2. Run the Phase 2 bake-off: Parakeet v3 against IndicConformer on Hindi dev data.
3. Use autoresearch to tune settings on short trials, within agreed limits.
4. Train the flagship, then score it on Monsoon after a leakage check and human approval.
5. Run the quantization search with a real evaluator.
