# Data, tools and services used

This lists everything the first model (v0) uses or that was searched for it, as of 2026-11-08. [`configs/datasets.yaml`](../configs/datasets.yaml) is the record of each license decision.

## Training data for v0

The owner approved all three for training on 2026-11-08.

| Dataset | What it is | Size used | License | Role |
|---|---|---|---|---|
| [MUCS 2021 Hinglish](https://www.openslr.org/104/) (OpenSLR SLR104), via its [Hugging Face mirror](https://huggingface.co/datasets/dianavdavidson/MUCS-Hinglish) | Spoken tutorials on technical topics, with Hindi in Devanagari and English terms in Latin in the same sentence | About 90 hours, 52,825 clips; 3% of tutorials held out for dev | **CC BY-SA 4.0** at the original. The mirror's CC BY 4.0 tag is wrong. | Real Hinglish, the core of v0 |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) Hindi (`hi_in`) | Read Wikipedia sentences, recorded by native speakers | About 7 hours of train, used twice; validation as Hindi dev | CC BY 4.0 | Pure Hindi |
| [LibriSpeech](https://huggingface.co/datasets/openslr/librispeech_asr) `train.100` | Read English audiobooks | Capped at 50 hours; 0.5 hours of validation as English dev | CC BY 4.0 | Keeps English from being forgotten |

Notes on this data:
- **v0 is light on pure Hindi.** About 7 hours of FLEURS against 90 of Hinglish, so it should be strongest on technical Hinglish. The gated corpora below fix this for v1.
- **MUCS's `speaker_id` is a tutorial ID.** Dev is split by tutorial, which is the closest speaker proxy available.
- **Share-alike:** CC BY-SA 4.0 may extend to models trained on MUCS. Legal opinion varies, so this is recorded for a legal check before release.
- **Text is made consistent:** LibriSpeech is lower-cased, and FLEURS uses its normalized column without punctuation, matching MUCS.

## Searched but not used

| Candidate | Where | Why not |
|---|---|---|
| IndicVoices, Kathbath, Shrutilipi, Vaani | Hugging Face, gated | Need the owner's login and accepted terms. Planned for v1. |
| [nvidia/Granary-v2](https://huggingface.co/datasets/nvidia/Granary-v2) | Hugging Face | Labels only. Its Hindi comes from Common Voice and FLEURS, so it adds cleaner labels rather than new audio. |
| [aman-hf/indic_asr](https://huggingface.co/datasets/aman-hf/indic_asr) | Hugging Face | Claims about 10,000 hours but names no sources, so licenses can't be checked |
| Shrutilipi re-uploads (`skesiraju`, `amithm3`) | Hugging Face | Copies of a gated AI4Bharat set. Use the original after accepting its terms. |
| HiACC, Sonexis, IndicSynth, Vartalaap, WorldSpeech | Hugging Face and Zenodo | Non-commercial licenses |
| YODAS-2 | Hugging Face | YouTube captions are often automatic, so the labels are noisy. Revisit with filtering. |
| Hindi speech datasets | Kaggle | Non-commercial, re-uploads without provenance, or text only |
| Speech environments | OpenEnv Hub | None exist. The two "audio" results are unrelated demos. |

Rerun the search with `scripts/source_datasets.py`. Its latest output is [`reports/data_sources_search.md`](../reports/data_sources_search.md).

## Model and software

| Item | Version | What it does here | License |
|---|---|---|---|
| [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | v3 | Base model: its encoder is kept, and a new tokenizer, decoder and joint are trained | CC BY 4.0 |
| [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) | 3.0.0 | Training toolkit | Apache 2.0 |
| SentencePiece, via NeMo | 0.2.x | Shared tokenizer (2,048 tokens) for Devanagari, Latin and later-wave scripts | Apache 2.0 |
| PyTorch Lightning | 2.4 | Training loop, checkpoints and time limits | Apache 2.0 |
| `voi-oiwer` | 0.1.4 | Official Hindi scorer for the leaderboard | MIT |
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | method only | Pattern for the tuning loop in `autoresearch/`; its code isn't used | MIT |
| [OpenEnv](https://github.com/meta-pytorch/OpenEnv) | 0.3.0 | Quantization search environment; not used for data | BSD-3-Clause |

## Services

| Service | Use | Account needed |
|---|---|---|
| Hugging Face Hub | Datasets, base model, benchmarks and search | Only for gated data |
| Kaggle | Free GPU to train v0, through `training/kaggle/kalaris_v0.ipynb` | Yes, phone-verified for internet |
| Kaggle API | Dataset search, which works without a login | No |
| OpenSLR | Original source and license for MUCS | No |

The Hugging Face, Firecrawl and Adaption connectors were attached to this Claude session but hadn't finished connecting, so none of this used them.
