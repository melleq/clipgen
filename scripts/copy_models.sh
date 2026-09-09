#!/usr/bin/env bash
# Copy the model weights clipgen needs into ./models so the project is self-contained.
#
# Sources (on this host):
#   - faster-whisper large-v3 (ASR)     : the curl-fetched local dir in clips_autom
#   - pyannote diarization (optional)   : the shared HuggingFace hub cache
#   - torchaudio English align model    : the shared torch hub cache
#
# After running, point .env at the copied dirs:
#   TRANSCRIBE_MODEL=./models/faster-whisper-large-v3
#   HF_HOME=./models/hf-cache
#   TORCH_HOME=./models/torch-cache
#
# Idempotent: existing destinations are left as-is (delete a dir to force a re-copy).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models="${here}/models"
mkdir -p "$models"

# --- 1) ASR: faster-whisper large-v3 (the complete local dir, incl. model.bin) ---------------
asr_src="${CLIPGEN_ASR_SRC:-${HOME}/clips_autom/data/models/faster-whisper-large-v3}"
asr_dst="${models}/faster-whisper-large-v3"
if [ -d "$asr_dst" ]; then
  echo "skip ASR — ${asr_dst} already exists"
elif [ -f "${asr_src}/model.bin" ]; then
  echo "copying ASR model  ${asr_src} -> ${asr_dst}"
  cp -a "$asr_src" "$asr_dst"
else
  echo "WARN: faster-whisper model.bin not found at ${asr_src}."
  echo "      Set CLIPGEN_ASR_SRC to a complete faster-whisper-large-v3 dir, or leave"
  echo "      TRANSCRIBE_MODEL=large-v3 to let WhisperX download it into HF_HOME on first run."
fi

# --- 2) Diarization: pyannote (optional; only needed when HF_TOKEN is set) --------------------
hf_hub="${HOME}/.cache/huggingface/hub"
hf_dst="${models}/hf-cache/hub"
pyannote="models--pyannote--speaker-diarization-community-1"
if [ -d "${hf_dst}/${pyannote}" ]; then
  echo "skip pyannote — ${hf_dst}/${pyannote} already exists"
elif [ -d "${hf_hub}/${pyannote}" ]; then
  echo "copying pyannote   ${hf_hub}/${pyannote} -> ${hf_dst}/"
  mkdir -p "$hf_dst"
  cp -a "${hf_hub}/${pyannote}" "${hf_dst}/"
else
  echo "note: pyannote not cached at ${hf_hub}/${pyannote} — diarization will fetch it on"
  echo "      first run (needs HF_TOKEN + accepted model terms). Non-fatal."
fi

# --- 3) Alignment: torchaudio English wav2vec2 model -----------------------------------------
torch_ckpt="${HOME}/.cache/torch/hub/checkpoints"
torch_dst="${models}/torch-cache/hub/checkpoints"
align="wav2vec2_fairseq_base_ls960_asr_ls960.pth"
if [ -f "${torch_dst}/${align}" ]; then
  echo "skip align — ${torch_dst}/${align} already exists"
elif [ -f "${torch_ckpt}/${align}" ]; then
  echo "copying align      ${torch_ckpt}/${align} -> ${torch_dst}/"
  mkdir -p "$torch_dst"
  cp -a "${torch_ckpt}/${align}" "${torch_dst}/"
else
  echo "note: English align model not cached — WhisperX will download it on first run. Non-fatal."
fi

echo
echo "Done. Copied into ${models}:"
du -sh "${models}"/* 2>/dev/null || true
