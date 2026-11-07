# Quantization search environment

An OpenEnv environment where an agent assigns a bit width to each layer group of a speech model, then earns a reward based on dev WER, size and speed.

**The evaluator is a stub.** `server/evaluator.py` returns invented numbers so the mechanics can be tested. Observations carry `evaluator: "stub"` until a real evaluator replaces it. Don't draw conclusions from these results.

## Run it

```bash
cd envs/quant_env
../../.venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Each episode goes like this:
1. `reset()` returns the first layer group.
2. `step(QuantAction(bits=4))` assigns bits to that group and moves to the next.
3. The last step returns `done=True`, the metrics and the reward.

## To make it real

Add an evaluator with the same `evaluate(policy)` method that quantizes the checkpoint, decodes a fixed dev shard, and measures speed on the target device. It must refuse any manifest containing test clips.
