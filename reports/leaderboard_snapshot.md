# Open ASR Leaderboard snapshot

Taken 2026-11-07 from the [leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard). These are the leaderboard's numbers, not ours. Refresh with `scripts/snapshot_leaderboard.py`.

## Hindi (Monsoon hi-IN, OIWER)

| Rank | Model | OIWER |
|---|---|---|
| 1 | fonada/kapture-v3 | 6.54 |
| 2 | meta/muse-voice-transcribe | 7.16 |
| 3 | elevenlabs/scribe_v2 | 8.45 |
| 4 | openai/gpt-transcribe | 8.59 |
| 5 | ai4bharat/indic-conformer-600m-multilingual | 8.78 |
| 6 | soniox/stt-async-v5 | 8.96 |
| 7 | microsoft/azure-speech-09-2026 | 9.14 |
| 8 | nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B | 9.6 |
| 9 | ARTPARK-IISc/SraVaani-1.0 | 11.38 |
| 10 | sprag/symphony | 11.62 |
| 11 | assemblyai/universal-3-5-pro | 12.45 |
| 12 | Qwen/Qwen3-ASR-1.7B-hf | 12.74 |
| 13 | nvidia/nemotron-3.5-asr-streaming-0.6b | 14.18 |
| 14 | google/gemma-4-E4B-it | 14.31 |
| 15 | facebook/omniASR-LLM-7B-v2 | 14.83 |
| 16 | facebook/omniASR-LLM-1B-v2 | 15.36 |
| 17 | facebook/omniASR-LLM-3B-v2 | 15.48 |
| 18 | speechmatics/enhanced | 16.64 |
| 19 | facebook/omniASR-LLM-300M-v2 | 18.28 |
| 20 | Qwen/Qwen3-ASR-0.6B-hf | 18.57 |
| 21 | bharatgenai/Shrutam-2 | 18.92 |
| 22 | assemblyai/universal-3-pro | 20.96 |
| 23 | mistralai/Voxtral-Small-24B-2507 | 22.26 |
| 24 | facebook/omniASR-CTC-7B-v2 | 22.56 |
| 25 | facebook/omniASR-CTC-3B-v2 | 24.86 |
| 26 | mistralai/Voxtral-Mini-3B-2507 | 25.26 |
| 27 | mistralai/Voxtral-Mini-4B-Realtime-2602 | 25.38 |
| 28 | facebook/omniASR-CTC-1B-v2 | 25.4 |
| 29 | modulate/multilingual | 29.62 |
| 30 | openai/whisper-large-v3 | 29.86 |
| 31 | facebook/omniASR-CTC-300M-v2 | 37.58 |
| 32 | microsoft/VibeVoice-ASR-HF | 39.94 |
| 33 | openai/whisper-large-v3-turbo | 41.55 |

## English average (main tab)

| Rank | Model | Average WER | RTFx | Monsoon en-IN |
|---|---|---|---|---|
| 1 | zoom/scribe_v2.1_pro | 4.87 | None | 2.78 |
| 2 | OtterAI/speech-v3.1 | 5.21 | None | 2.97 |
| 3 | microsoft/azure-speech-09-2026 | 5.55 | None | 3.18 |
| 4 | p99lab/audrey-0.7 | 5.7 | None | 3.22 |
| 5 | elevenlabs/scribe_v2 | 5.93 | None | 3.32 |
| 6 | modulate/multilingual | 5.96 | None | 3.26 |
| 7 | reson8/resonant-1 | 5.99 | None | 3.04 |
| 8 | sophea/asr-k1 | 6.11 | None | 3.25 |
| 9 | nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B | 6.13 | 472.62 | 3.63 |
| 10 | Qwen/Qwen3-ASR-1.7B-hf | 6.15 | 835.62 | 3.5 |
| 11 | assemblyai/universal-3-5-pro | 6.21 | None | 3.49 |
| 12 | smallestai/pulse | 6.57 | None | 3.52 |
| 13 | ibm-granite/granite-speech-5.0-2b | 6.68 | 548.43 | 3.94 |
| 14 | CohereLabs/cohere-transcribe-03-2026 | 6.72 | 923.71 | 3.44 |
| 15 | HojoAI/Hojo-ASR-V1 | 6.74 | 73.39 | 4.63 |
| 16 | TheStageAI/thewhisper-large-v3-turbo | 6.74 | 1183.92 | 3.72 |
| 17 | Edge0/ARK-ASR-0.6B | 6.75 | 664.56 | 3.43 |
| 18 | openai/gpt-transcribe | 6.8 | None | 3.63 |
| 19 | sprag/symphony | 6.85 | None | 3.55 |
| 20 | gladia/solaria-3 | 6.86 | None | 3.7 |
| 21 | nvidia/parakeet-tdt-0.6b-v2 (fast-gpu-asr) | 6.91 | 17254.19 | 3.72 |
| 22 | OpenMOSS-Team/MOSS-Transcribe-Diarize | 6.91 | 387.46 | 3.87 |
| 23 | nvidia/canary-qwen-2.5b | 6.92 | 881.67 | 3.6 |
| 24 | Edge0/ARK-ASR-3B | 6.92 | 480.33 | 3.48 |
| 25 | nvidia/parakeet-tdt-0.6b-v2 | 6.93 | 6148.32 | 3.89 |
| 26 | nvidia/parakeet-tdt-0.6b-v3 (fast-gpu-asr) | 6.95 | 14319.58 | 3.96 |
| 27 | ibm-granite/granite-speech-5.0-470m-turboctc-nc | 6.98 | 21082.03 | 4.3 |
| 28 | bosonai/higgs-audio-v3-stt | 6.98 | 109.65 | 3.97 |
| 29 | meta/muse-voice-transcribe | 7.0 | None | 4.16 |
| 30 | ibm-granite/granite-speech-4.1-2b | 7.03 | 546.84 | 4.12 |
| 31 | nvidia/parakeet-tdt-0.6b-v3 | 7.04 | 6321.9 | 4.14 |
| 32 | fonada/kapture-v3 | 7.05 | None | 3.61 |
| 33 | Qwen/Qwen3-ASR-0.6B-hf | 7.07 | 749.52 | 4.23 |
| 34 | ibm-granite/granite-speech-4.1-2b-nar | 7.16 | 2112.58 | 4.24 |
| 35 | ibm-granite/granite-speech-5.0-470m-turboctc | 7.41 | 21434.51 | 4.75 |
| 36 | mistralai/Voxtral-Small-24B-2507 | 7.44 | 101.46 | 3.49 |
| 37 | nvidia/nemotron-speech-streaming-en-0.6b | 7.47 | 1115.95 | 4.44 |
| 38 | distil-whisper/distil-large-v3.5 | 7.57 | 898.21 | 3.6 |
| 39 | ibm-granite/granite-4.0-1b-speech | 7.69 | 666.99 | 4.84 |
| 40 | FermionResearch/Phonon-2 | 7.86 | 1976.52 | 4.91 |
| 41 | soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M (fast-gpu-asr, transducer_modified_beam_search) | 8.07 | 22109.49 | 5.57 |
| 42 | openai/whisper-large-v3 | 8.24 | 470.74 | 3.95 |
| 43 | mistralai/Voxtral-Mini-3B-2507 | 8.32 | 179.78 | 4.9 |
| 44 | nvidia/canary-1b-v2 | 8.43 | 1853.68 | 4.97 |
| 45 | FunAudioLLM/Fun-ASR-Nano-2512-hf | 8.54 | 506.37 | 5.32 |
| 46 | OpenLLM-France/Luciole-8B-Audio-1.0 | 8.73 | 999.2 | 4.78 |
| 47 | openai/whisper-large-v3-turbo | 8.75 | 808.92 | 4.77 |
| 48 | espnet/owsm_ctc_v4_1B | 9.26 | 783.3 | 6.66 |
| 49 | Edge0/Audio8-ASR-0.1B | 9.3 | 725.78 | 6.18 |
| 50 | facebook/omniASR-LLM-7B-v2 | 9.45 | 129.8 | 4.87 |
| 51 | microsoft/VibeVoice-ASR-HF | 9.54 | 217.89 | 4.74 |
| 52 | mistralai/Voxtral-Mini-4B-Realtime-2602 | 9.72 | 103.48 | 8.86 |
| 53 | soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M (fast-gpu-asr, ctc_greedy_search) | 9.95 | 24659.16 | 7.76 |
| 54 | futo-org/asr4all-l | 9.97 | 20971.29 | 7.42 |
| 55 | abr-ai/niagara-84m-batch.en | 10.9 | 1940.83 | 8.82 |
| 56 | nvidia/nemotron-3.5-asr-streaming-0.6b | 11.09 | 1353.11 | 7.3 |
| 57 | futo-org/asr4all-m | 11.29 | 28913.05 | 9.01 |
| 58 | abr-ai/niagara-38m-batch.en | 12.45 | 4584.99 | 10.41 |
| 59 | google/gemma-4-E4B-it | 13.2 | 160.69 | 6.19 |
| 60 | facebook/omniASR-CTC-7B-v2 | 13.27 | 522.26 | 7.08 |
| 61 | futo-org/asr4all-s | 13.97 | 40785.8 | 12.92 |
| 62 | abr-ai/niagara-19m-batch.en | 14.43 | 4373.0 | 13.15 |
| 63 | abr-ai/niagara-9m-batch.en | 17.88 | 7217.02 | 17.11 |
