# Decisions

These are the project's decisions as of 2026-11-07, why we made them, and what would change them. The [strategy document](https://claude.ai/artifact/6XuzBJBNuZJofz7BxF1aC9) has the full reasoning and sources.

Evidence labels: **vendor** (a maker's own claim), **independent** (a third party's test), **estimate** (our judgment, not measured). No decision here rests on a score measured in this repo yet.

| Decision | Why | Evidence | What would change it |
|---|---|---|---|
| Wave 1 is Hindi, Hinglish and English | A narrow first wave makes a real benchmark win more likely | Team decision | Wave 1 targets met |
| Build a flagship first, then quantize to about 200 MB | Bark 3.0 Pro and Bark 3.0 follow this path | Team decision; vendor | Quantization can't reach the mid-range targets |
| Start from Parakeet TDT 0.6B v3 | Bark is a fine-tune of it, it's CC-BY-4.0, and it's fast | Vendor | IndicConformer wins the Phase 2 bake-off |
| Keep IndicConformer 600M as the challenger | It already covers Hindi and is listed as MIT | Listed license | Bake-off result |
| Replace the tokenizer, decoder and joint | Parakeet v3 has no Indic vocabulary | Vendor | Nothing expected |
| One tokenizer with Wave 2 scripts from day one | New languages then won't force another decoder reset | Estimate | Hurts Wave 1 accuracy |
| Whisper is a baseline only | It trails Parakeet on the Bark card and is much larger | Vendor | Rescored results disagree |
| Don't build on Bark's weights | They are CC BY-NC | Model card | A commercial license |
| Supervised training first, RL as a gated last stage | RL has published gains as fine-tuning, not as a replacement | Published papers; none on TDT | RL passes its gate by a wide margin |
| OpenEnv for outer loops only | It has no trainer and no audio environments, and is in early development | OpenEnv docs | OpenEnv adds audio training support |
| ONNX with sherpa-onnx first, then Core ML | No format is universal, and ONNX reaches Android | Estimate | Reference phone results |
| GGUF deferred | parakeet.cpp isn't in mainline whisper.cpp, and parity is unverified | Secondary sources | Independent parity results |
| Train on rented GPUs | 4 GB of VRAM can't train a 0.6B model, and no MPS or MLX training path was found | Estimate | A tested Mac training path, or owned GPUs |
| Kaggle for bake-offs, not as the data source | No strong Indic corpus was found there, and its sessions and quota are limited | Search results | A strong corpus on Kaggle |

## Choices made while building Phase 0

| Choice | Why |
|---|---|
| Baselines run on this M1 Pro, not the base M4 in the plan | This session runs on the M1 Pro. Results are labeled with the machine. |
| FLEURS is the first Hindi eval set | It's open and small, while IndicVoices and Kathbath are gated |
| IndicConformer runs through a third-party ONNX repack | The official repo is gated. The repack has the CTC head only, so its score may understate the model. |
| LibriSpeech downloads from an OpenSLR mirror | About 12x faster when tested, and verified against OpenSLR's checksum |
| Baselines run one model per process, one at a time | Peak memory and speed stay clean |
| Downloads stay inside the project and the run stops under 10 GB free | The disk is nearly full (97% during the first run) |
| Bark isn't run here | Its non-commercial license hasn't been cleared for internal benchmarking |
