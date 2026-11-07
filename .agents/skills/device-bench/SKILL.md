---
name: device-bench
description: Benchmark ASR speed, memory and size on real devices. Use when reporting edge performance, comparing runtimes, or checking a mid-range target.
---

# Device benchmarks

```bash
.venv/bin/stt-eval transcribe --backend sherpa-onnx-transducer --model models/<dir> --manifest <m> --out runs/<name>.jsonl --language en
```

This writes `runs/<name>.bench.json` with RTFx, latency percentiles, peak RSS, accelerator peak memory and model size.

**Rules**
- One model per process. Peak RSS never falls, so a second model in one process inherits the first one's peak.
- On MLX, read `accelerator_peak_bytes` along with RSS. RSS can miss GPU allocations in unified memory.
- Stop other heavy work first, and plug in laptops. Note it if the run shared the machine.
- Report the exact device and threads. An M1 Pro number is not an M4 number, and neither is a phone number.
- Report model load and first-load compile time separately from RTFx.

**Phones:** Run at least 10 minutes and watch for thermal throttling, because short runs flatter phones. Record the speed in the first and last minute. On Android, test both the reference mid-range phone and CPU-only mode. The harness doesn't run on phones yet, so phone numbers need an on-device runner.

**Report:** device, OS, runtime version, threads, accelerator, clips, audio hours, RTFx, p50 and p90 latency, peak RAM, download size and WER on the same run.

Live-microphone latency is a different number from these whole-clip results. Don't present one as the other.
