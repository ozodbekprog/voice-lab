#!/usr/bin/env bash
# Qwen3-TTS 1.7B fine-tune on the speaker dataset (Uzbek).
set -e
cd "$HOME/voice-lab"
source env.sh

TOK=$(ls -d "$HOME"/models/hf-cache/hub/models--Qwen--Qwen3-TTS-Tokenizer-12Hz/snapshots/*/)
INIT=$(ls -d "$HOME"/models/hf-cache/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*/)

echo "tokenizer: $TOK"
echo "init:      $INIT"

~/voice-lab/venvs/qwen3tts/bin/python Qwen3-TTS/finetuning/prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path "$TOK" \
  --input_jsonl ft/train_raw.jsonl \
  --output_jsonl ft/train_with_codes.jsonl

~/voice-lab/venvs/qwen3tts/bin/python Qwen3-TTS/finetuning/sft_12hz_local.py \
  --init_model_path "$INIT" \
  --output_model_path ft/out_1.7b \
  --train_jsonl ft/train_with_codes.jsonl \
  --batch_size 2 \
  --lr 2e-5 \
  --num_epochs 3 \
  --speaker_name miralieva
