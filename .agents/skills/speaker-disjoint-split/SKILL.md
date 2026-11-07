---
name: speaker-disjoint-split
description: Create train, dev and test splits with no shared speakers or recordings. Use when building splits or when a dataset ships without speaker-safe ones.
---

# Speaker-disjoint splits

1. Keep any official test split unchanged, so scores stay comparable with published work.
2. Split by speaker, then assign all of that speaker's clips to one side. Hash the namespaced ID (`dataset:speaker`) for a stable result.
3. Balance dev by gender, region and duration where metadata allows, and record what metadata was missing.
4. If speaker IDs are missing, split by recording or session and record that the split is weaker.
5. Verify before use:

```bash
.venv/bin/stt-eval leakage --train manifests/train.jsonl --held-out manifests/dev.jsonl
.venv/bin/stt-eval leakage --train manifests/train.jsonl --held-out manifests/test.jsonl
```

Exit code 1 means a leak, so fix the split rather than the check. Shared transcripts can be legitimate in read-speech corpora where many speakers read one script. Report them, and let a human accept them.

A checksum misses re-encoded copies, and speaker IDs miss the same person under two IDs. The transcript check catches some of these, but not all.
