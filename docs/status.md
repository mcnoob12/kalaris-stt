# Project status

Phase 0 is in progress as of 2026-11-07. No model has been trained, and baseline scores aren't in yet.

## Phase 0 progress

| Item | State | How it was checked |
|---|---|---|
| Strategy document | Done | Published as a [doc](https://claude.ai/artifact/6XuzBJBNuZJofz7BxF1aC9) |
| Scoring, normalization, leakage checks | Done | Unit tests |
| Benchmark code and four backends | Written, not yet proven | Their first real runs are the baselines |
| Core ML adapter | **Not built** | The FluidAudio CLI was looked at, but not built or run |
| 12 agent skills and `AGENTS.md` | Written | Format tests pass and Pi lists all 12 |
| Skills run through Pi | **Not tested** | Pi's Radius account has a $0.00 balance |
| OpenEnv environment | Skeleton done, stub evaluator | Unit tests, `openenv validate` on files and server, one live episode |
| Baseline runs | **Running** | Results land in `runs/` |
| Baseline report | **Pending results** | See [`reports/baseline.md`](../reports/baseline.md) |
| Reproducing a published baseline | **Pending results** | Parakeet v3 on LibriSpeech test-clean |

The Docker image for the environment hasn't been built, because Docker isn't installed on this Mac. The NeMo training recipe is a design and has never been run.

## Known issues

| Issue | Effect | Plan |
|---|---|---|
| `fluidaudio-coreml` has no adapter file | Selecting it crashes | Build it or remove the entry after the baselines |
| Manifests use absolute paths | They won't work on the M4 or a rented GPU without regenerating | Switch to repo-relative paths after the baselines |
| FLEURS has no speaker IDs | Hindi comparisons resample by clip, so intervals may be too narrow | Don't use FLEURS alone for a promotion decision |
| The IndicConformer repack is CTC only | Its score may understate the official model | Rerun with the official model once its terms are accepted |
| Baselines run on an M1 Pro | Speed numbers don't apply to the base M4 | Rerun on the M4 |
| Disk is 97% full, with 15 GB free mid-run | Limits which datasets fit, and the baseline run stops under 10 GB free | Free space or use external storage before adding sets |

## What needs a person

- [ ] Accept gated terms on Hugging Face for IndicVoices, Kathbath, Shrutilipi and official IndicConformer
- [ ] Review licenses and set `status: confirmed` in `configs/datasets.yaml`
- [ ] Find the original sources for HiACC and MUCS-2021, or budget for our own Hinglish data
- [ ] Decide on a GPU budget, or apply for IndiaAI compute, before Phase 3
- [ ] Choose a reference mid-range Android phone
- [ ] Add Radius credits if you want the skills tested through Pi
- [ ] Clear Bark's license for internal benchmarking, or keep using its published numbers
- [ ] Confirm that "pi harness" means the Pi coding agent, not a Raspberry Pi

## Next steps

1. Finish the baselines, write the report and check the Parakeet result against published numbers.
2. Fix the manifest paths and resolve the Core ML adapter.
3. Rerun the speed baselines on the base M4.
4. Start Phase 1: Hindi and Hinglish data audit, tokenizer and a smoke test.
