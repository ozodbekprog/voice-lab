#!/usr/bin/env python3
"""Diagnose ASR: force uz language, check reversed audio, multi-speaker windows."""
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/tony/voice-lab/checkpoints/hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import librosa
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path("/home/tony/voice-lab")
SNAP = sorted((ROOT / "checkpoints" / "hf_cache" / "hub" / "models--openai--whisper-large-v3-turbo" / "snapshots").glob("*"))[0]
FILES = [
    ("Miralieva", "33659_Miralieva.ogg"),
    ("Miralieva", "26861_Miralieva.ogg"),
    ("dot", "50344_dot.ogg"),
    ("dot", "53429_dot.ogg"),
]


def load(path, rev=False):
    wav, _ = librosa.load(path, sr=16000, mono=True)
    wav = wav[: 28 * 16000]
    return wav[::-1].copy() if rev else wav


def main():
    processor = WhisperProcessor.from_pretrained(str(SNAP), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        str(SNAP), local_files_only=True, torch_dtype=torch.float16
    ).to("cuda").eval()

    def run(wav, lang="uz"):
        feats = processor(wav, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", dtype=torch.float16)
        with torch.no_grad():
            ids = model.generate(feats, max_new_tokens=200, language=lang, task="transcribe",
                                 temperature=0.0, condition_on_prev_tokens=False)
        return processor.tokenizer.decode(ids[0], skip_special_tokens=True).strip()

    for group, name in FILES:
        path = ROOT / "data" / "data" / "ogg" / group / name
        print(f"\n=== {group}/{name}")
        print("  uz :", run(load(path), "uz")[:280], flush=True)
        print("  ru :", run(load(path), "ru")[:200], flush=True)
        print("  REV:", run(load(path, rev=True), "uz")[:200], flush=True)


if __name__ == "__main__":
    main()
