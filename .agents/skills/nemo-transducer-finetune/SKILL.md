---
name: nemo-transducer-finetune
description: Fine-tune a Parakeet or IndicConformer transducer in NVIDIA NeMo for Hindi, Hinglish and English. Use when planning or running supervised training, or diagnosing a run.
---

# NeMo transducer fine-tuning

**Status:** This recipe is a design and hasn't been run in this repo. Confirm NeMo config keys against the installed version. Rented GPUs need human approval.

## Order of work

1. **Smoke test on the 3050 Ti** with Parakeet 110M and short clips. Gate: it overfits about 20 clips to near-zero WER. If it can't, stop and debug before spending money.
2. **Tokenizer:** Train one SentencePiece BPE model on Devanagari, Latin and the Wave 2 scripts, so later languages don't force a decoder reset.
3. **Head training:** Restore the encoder, build a new decoder and joint for the new vocabulary, and freeze the encoder.
4. **Full fine-tune:** Unfreeze with layer-wise learning-rate decay, lowest near the input. Use mixed precision.
5. **Mixing:** Keep English in every batch to prevent forgetting. Oversample real Hinglish, and use synthetic splices only as a supplement.
6. **Augmentation:** SpecAugment, noise, room impulses, codecs and cheap-mic filters, applied to training data only.

## Rules

- Checkpoint often and resume cleanly, because Kaggle sessions and rentals end without warning.
- Track WER on separate Hindi, Hinglish and English dev sets. A blended number can hide an English regression.
- Start each rental with a short run to measure cost per epoch, and report the projected total before continuing.
- Save the config, tokenizer, data manifest hash and commit with every checkpoint.

## Diagnosing

| Symptom | Check first |
|---|---|
| Loss falls but WER stays near 100% | Tokenizer mismatch, or text normalization that differs between training and scoring |
| Hindi improves and English collapses | English share in the batch mix, and encoder learning rate |
| Output is English written in Devanagari | Transcript conventions in the Hinglish data |
| Good dev WER, poor real-world results | Dev set too clean, or speakers leaking across splits |
| Out of memory | Long clips first: cap duration and batch by total seconds |
