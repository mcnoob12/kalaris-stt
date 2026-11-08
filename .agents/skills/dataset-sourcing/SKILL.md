---
name: dataset-sourcing
description: Find new speech training data and evaluation sets on Hugging Face, Kaggle and the OpenEnv Hub, and trace each one to its real source and license. Use when looking for more Hindi, Hinglish or English data, or before adding anything to configs/datasets.yaml.
---

# Dataset sourcing

```bash
.venv/bin/python scripts/source_datasets.py
```

This writes `reports/data_sources_search.md`, a list of leads. Then check each promising lead:

1. **Find the original source.** Many Hugging Face datasets are re-uploads. Read the card and follow its links to the original page.
2. **Read the license at the original source.** Re-uploads often mislabel it. MUCS is tagged CC BY 4.0 on Hugging Face but is CC BY-SA 4.0 on OpenSLR, and the original governs.
3. **Reject aggregators that don't name their sources.** Their license can't be checked. `aman-hf/indic_asr` was rejected for this.
4. **Check for benchmark overlap.** Training on anything in `configs/benchmarks.yaml` invalidates our scores, and aggregators can quietly include test sets.
5. **Check the data:** real or synthetic audio, hours, speaker IDs, script conventions (Devanagari, Latin or both), and which column holds clean text. `https://datasets-server.huggingface.co/first-rows?dataset=<id>&config=<c>&split=<s>` shows the schema.
6. **Record it** in `configs/datasets.yaml` with `status: listed`. Only a person sets `confirmed`.

## What each source is good for

| Source | Use it for | Found so far |
|---|---|---|
| Hugging Face | Most datasets and all our benchmarks | The useful sets, plus gated AI4Bharat and Vaani corpora |
| Kaggle | Free GPU time, and occasionally data | Searched 2026-11-08: Hindi speech results were non-commercial, re-uploads or text only |
| OpenEnv Hub | Environments, not data | No speech environments. We use OpenEnv for quantization search only. |

Gated datasets need a person to log in (`hf auth login`) and accept each dataset's terms. Never accept terms on someone's behalf.
