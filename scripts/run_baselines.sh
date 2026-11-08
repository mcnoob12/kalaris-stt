#!/usr/bin/env bash
# Phase 0 baselines. Runs one model per process, one after another, so speed and
# peak-memory numbers aren't polluted by each other. Safe to re-run: finished
# runs are skipped.
set -euo pipefail

cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.cache/huggingface"
PY=.venv/bin/python
MIN_FREE_GB=10

check_disk() {
  local free_gb
  free_gb=$(df -g . | awk 'NR==2 {print $4}')
  if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
    echo "Stopping: only ${free_gb} GB free, and this script keeps at least ${MIN_FREE_GB} GB." >&2
    exit 3
  fi
}

run() {
  local name=$1 backend=$2 model=$3 manifest=$4 language=$5
  local out="runs/${name}.jsonl"
  if [ -f "runs/${name}.bench.json" ]; then
    echo "== skip ${name}: already finished"
    return
  fi
  check_disk
  echo "== ${name} $(date +%H:%M:%S)"
  "$PY" -m stt_eval.cli transcribe --backend "$backend" --model "$model" \
    --manifest "$manifest" --out "$out" --language "$language" > "runs/${name}.log"
}

mkdir -p runs models
check_disk
[ -f manifests/librispeech_test_clean.jsonl ] || "$PY" scripts/fetch_librispeech.py
[ -f manifests/fleurs_hi_test.jsonl ] || "$PY" scripts/fetch_fleurs.py

SHERPA=sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8
if [ ! -d "models/$SHERPA" ]; then
  check_disk
  curl -fsSL "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$SHERPA.tar.bz2" -o "models/$SHERPA.tar.bz2"
  tar -xjf "models/$SHERPA.tar.bz2" -C models
  rm "models/$SHERPA.tar.bz2"
fi

# ONNX Runtime refuses external weights behind a cache symlink, so this model
# needs real files in its own folder.
INDIC=models/indicconformer-600m-onnx
if [ ! -f "$INDIC/encoder-model.onnx.data" ]; then
  check_disk
  .venv/bin/hf download christopherthompson81/indicconformer-600m-onnx --local-dir "$INDIC"
fi

[ -f manifests/hi_monsoon.jsonl ] || "$PY" scripts/fetch_benchmark.py hi_monsoon

EN=manifests/librispeech_test_clean.jsonl
HI=manifests/fleurs_hi_test.jsonl
MONSOON=manifests/hi_monsoon.jsonl
PARAKEET=mlx-community/parakeet-tdt-0.6b-v3
WHISPER=mlx-community/whisper-large-v3-turbo

run parakeet_mlx_en       parakeet-mlx           "$PARAKEET"      "$EN"      en
run parakeet_onnx_en      sherpa-onnx-transducer "models/$SHERPA" "$EN"      en
run parakeet_mlx_hi       parakeet-mlx           "$PARAKEET"      "$HI"      hi
run indicconformer_monsoon indicconformer-onnx   "$INDIC"         "$MONSOON" hi
run whisper_turbo_monsoon whisper-mlx            "$WHISPER"       "$MONSOON" hi
run indicconformer_hi     indicconformer-onnx    "$INDIC"         "$HI"      hi
run whisper_turbo_hi      whisper-mlx            "$WHISPER"       "$HI"      hi
run whisper_turbo_en      whisper-mlx            "$WHISPER"       "$EN"      en

echo "== all baselines finished $(date +%H:%M:%S)"
