# Contributor handoff

Snapshot from 2026-11-08, on the M1 Pro. It goes stale quickly, so check it against `git log` and [`status.md`](status.md). Full scores are in [`reports/baseline.md`](../reports/baseline.md).

## Where the project stands

Phase 0 is finished. No production model has been trained. The training code is written and has passed its Mac smoke test, but it hasn't run on the real TDT loss.

| Item | State |
|---|---|
| Git | Branch `feat/training-stack`, pushed to GitHub. Not yet merged into `main`. |
| Baselines | All 8 finished |
| Mac smoke test (RNNT, FLEURS validation) | **Passed:** 7 clips overfit to 0.00% WER (measured) |
| TDT smoke test | Not run. It needs a CUDA GPU. |
| `training/configs/wave1.yaml` | Never run. Its `VERIFY` column names are unresolved, and the datasets are gated. |
| Tests | 118 passed |
| Licenses | **No entry in `configs/datasets.yaml` is `confirmed`.** Baselines used `listed` entries, and `--smoke` allows them for throwaway weights. Real training stays blocked until a person reviews them. |
| Disk on the M1 Pro | 11 GB free, against a 10 GB floor |

## Baseline scores

All measured with `stt-eval score` at the default strict normalization (no `--fold-nukta` or `--fold-nasal`), on an Apple M1 Pro with 16 GB. These are third-party models, so no leakage check applies. The only comparison run so far is on Monsoon: IndicConformer beats Whisper turbo by 30.63 points (95% CI -34.38 to -27.49, resampled by speaker).

| Run | Eval set | Clips | Error rate | CER | RTFx |
|---|---|---|---|---|---|
| `parakeet_mlx_en` | LibriSpeech test-clean | 2,620 | 1.99 WER | 0.68 | 24.8 |
| `parakeet_onnx_en` (int8) | LibriSpeech test-clean | 2,620 | 2.32 WER | 0.85 | 31.2 |
| `parakeet_mlx_hi` | FLEURS Hindi test | 418 | 99.94 WER | 102.01 | 24.9 |
| `indicconformer_hi` (CTC repack) | FLEURS Hindi test | 418 | 12.20 WER | 5.28 | 18.2 |
| `whisper_turbo_hi` | FLEURS Hindi test | 418 | 32.22 WER | 14.03 | 9.9 |
| `indicconformer_monsoon` (CTC repack) | Monsoon hi-IN | 753 | 11.06 OIWER | n/a | 14.0 |
| `whisper_turbo_monsoon` | Monsoon hi-IN | 753 | 41.69 OIWER | n/a | 3.1 |
| `whisper_turbo_en` | LibriSpeech test-clean | 2,620 | 1.91 WER | 0.65 | 8.3 |

Reading these:

- **The Parakeet reproduction check misses its tolerance.** We measured 1.99 against a published 1.52 (reported on the Bark card as a leaderboard value). The 0.47-point gap is over the 0.3 limit set in `reports/baseline.md`. The cause is unknown, though the runtime differs.
- **Whisper turbo on Monsoon agrees with the leaderboard:** 41.69 measured, 41.55 leaderboard. The harness looks sound for this set.
- **IndicConformer on Monsoon is worse than its leaderboard entry:** 11.06 measured, 8.78 leaderboard. We ran the CTC-only repack and the leaderboard lists the official model, so they aren't the same model.
- **Treat RTFx as approximate.** Other work ran during these runs. The smoke test overlaps `whisper_turbo_en`, so don't use that run's speed.
- **The leaderboard's "LS Clean" column isn't WER,** so it can't settle the Parakeet gap.

## Setting up on another machine

The repo is private and `mcnoob12` is the only collaborator. The owner needs to push the branch, then invite you:

```bash
gh api -X PUT repos/mcnoob12/kalaris-stt/collaborators/THEIR_GITHUB_LOGIN -f permission=push
```

Then:

```bash
git clone https://github.com/mcnoob12/kalaris-stt.git
```

```bash
cd kalaris-stt && git checkout feat/training-stack
```

```bash
uv venv && uv pip install -e '.[dev,mlx,env,onnx,data]'
```

```bash
.venv/bin/python -m pytest
```

`data/`, `models/` and `runs/` are gitignored, so a clone has no audio, weights or baseline outputs. Baseline runs from the M1 Pro exist only on that Mac.

`run_baselines.sh` skips fetching audio when a manifest already exists, and manifests are committed. On a clone, run `scripts/fetch_librispeech.py`, `scripts/fetch_fleurs.py` and `scripts/fetch_benchmark.py hi_monsoon` first. This comes from reading the script and hasn't been tested on a clean clone.

## Open work

| Task | Needs |
|---|---|
| Run the TDT smoke test (`training/configs/smoke_tdt.yaml`) | CUDA GPU with 4 GB or more, such as the i7 with RTX 3050 Ti |
| Explain the Parakeet gap using aggregates only | Any Mac with the data |
| Build a Core ML adapter (the broken entry was removed) | A Mac |
| Rerun speed baselines | The base M4 |
| Run one autoresearch trial end to end (`autoresearch/program.md`) | Any machine with `.venv-train` |
| Phase 1: Hindi and Hinglish data audit | Gated access and license review first |

Only a person can do these:

- Accept the gated terms on Hugging Face, and review licenses to set `status: confirmed`.
- Approve GPU budget or any other paid compute.
- Find Hinglish data we may train on commercially (none found yet; HiACC is evaluation-only), and pick a reference Android phone.

The full list is in [`status.md`](status.md).

## Caveman and Ponytail

**Not installed yet.** The automated install was blocked by a permission check, so the owner needs to run it once and commit the resulting `.claude/settings.json`. Everyone who opens the repo after that is offered the same plugins. Run these from the repo root:

```bash
claude plugin marketplace add 'JuliusBrussee/caveman#v4.0.1' --scope project
```

```bash
claude plugin marketplace add 'DietrichGebert/ponytail#v5.3.3' --scope project
```

```bash
claude plugin install caveman@caveman --scope project
```

```bash
claude plugin install ponytail@ponytail --scope project
```

The `#tag` pinning syntax hasn't been run here. These are the newest real tags. Caveman's `INSTALL.md` mentions `v4.1.0`, but that tag doesn't exist upstream.

| | Caveman | Ponytail |
|---|---|---|
| Source | [JuliusBrussee/caveman](https://github.com/JuliusBrussee/caveman), Apache-2.0 | [DietrichGebert/ponytail](https://github.com/DietrichGebert/ponytail), MIT |
| Effect | Terse agent replies, always on | Smallest-change coding style, always on |
| Hooks | Session start, subagent start, prompt submit, session end | Session start, subagent start, prompt submit |
| Local writes | Mode and usage stats | Mode, and a per-task cost ledger in `~/.config/ponytail/band/` |
| Network from hooks | None found by searching the hook code | None unless a webhook is configured. Leave it empty. |

Hooks need `node` on your PATH. That review was a pattern search plus reading the manifests and main skill text, not a line-by-line audit.

If they're on, `AGENTS.md` wins over both. Caveman shortens chat only: docs, reports and every score keep full labels (measured, vendor-reported or estimated) plus machine, normalization, clip count and commit. Ponytail's minimalism never removes a leakage check, license check or test.

## Prompt for the next contributor

Paste this into Claude Code from the repo root:

```text
You're joining Kalaris STT, an on-device speech-to-text project. Wave 1 covers Hindi, Hinglish and English. Repo: https://github.com/mcnoob12/kalaris-stt (private), branch feat/training-stack.

Before changing anything, read AGENTS.md, docs/handoff.md, docs/status.md, docs/decisions.md and training/README.md, plus the matching skill in .agents/skills/ for your task. Then tell me:
1. This machine's chip, memory, GPU and free disk (`df -h .`), and which AGENTS.md machine role it fits.
2. Whether the tests pass (`.venv/bin/python -m pytest`), setting up the environment first if needed.
3. Which open task from docs/handoff.md fits this machine, and your plan for it. Wait for my go-ahead before starting.

Rules that always apply:
- Never look at test-set audio or per-clip test results. Use only `stt-eval score` aggregates on test manifests, and never read per-clip hypothesis files for them. Never train or tune on anything in configs/benchmarks.yaml.
- Run `stt-eval leakage` before reporting a score from a model we trained, and report its result.
- Only a person can start paid compute, promote a model, submit to a leaderboard or publish weights. Propose and wait for a yes.
- Don't use a dataset or model for real training until it's `status: confirmed` in configs/datasets.yaml. None is yet. Don't build on Bark 3.0 weights or Krutrim IndicST.
- Keep at least 10 GB of disk free. Check before any download.
- A model beats another only if `stt-eval compare` gives a 95% CI excluding zero. Label every number as measured, vendor-reported or estimated. Report machine, normalization options, clip count and commit with every score.

Working together:
- Work on your own branch off feat/training-stack and open a PR. Don't push to main or feat/training-stack.
- Say which task you're taking before you start, so two people don't take the same one. Baseline outputs in runs/ live only on the owner's M1 Pro, so those report updates stay there.
- If caveman or ponytail are enabled, AGENTS.md overrides them. Docs, reports and scores stay in full prose with all labels.
```
