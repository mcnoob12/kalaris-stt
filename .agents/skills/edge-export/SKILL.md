---
name: edge-export
description: Export a trained ASR model to ONNX, Core ML, MLX or GGUF. Use when preparing a model for a device or runtime, or when choosing a deployment format.
---

# Edge export

The source of truth is one NeMo checkpoint, safetensors weights and a quantization spec. Export every format from those. Never convert one exported format into another.

| Priority | Format and runtime | Target |
|---|---|---|
| 1 | ONNX with sherpa-onnx | Android, Windows, Linux, iOS fallback |
| 2 | Core ML with FluidAudio | iPhone, Mac (ANE) |
| 3 | MLX | Mac development and desktop |
| 4 | GGUF with parakeet.cpp | Deferred: not in mainline whisper.cpp, parity unverified |
| 5 | LiteRT or QNN | Android NPU, later |

A transducer exports as separate encoder, decoder and joint graphs, plus a decoding loop in the runtime. Check that the runtime supports TDT's duration handling, not only RNN-T.

**Parity check for every export:** Run the same 200 dev clips through the source model and the export, then:

```bash
.venv/bin/stt-eval compare --manifest <dev> --a runs/<source>.jsonl --b runs/<export>.jsonl --language hi
```

Differences in WER beyond the quantization you intended usually mean a mismatched frontend, such as the mel settings, windowing or normalization. Check that first.

Record for each export: format, runtime version, size, quantization, parity result, and compile or first-load time. Core ML can spend minutes compiling on first load, which is a user-facing cost.
