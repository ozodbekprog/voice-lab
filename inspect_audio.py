#!/usr/bin/env python3
"""Diagnose language + pitch of sample clips using whisper-small (CPU fp32)."""
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
SNAP = sorted((ROOT / "checkpoints" / "hf_cache" / "models--openai--whisper-small" / "snapshots").glob("*"))[0]

FILES = [
    ("Miralieva", "33659_Miralieva.ogg"),
    ("Miralieva", "26861_Miralieva.ogg"),
    ("dot", "53418_dot.ogg"),
    ("dot", "50344_dot.ogg"),
]


def main():
    processor = WhisperProcessor.from_pretrained(str(SNAP), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(str(SNAP), local_files_only=True).float().eval()
    tok = processor.tokenizer

    for group, name in FILES:
        path = ROOT / "data" / "data" / "ogg" / group / name
        wav, _ = librosa.load(path, sr=16000, mono=True)
        wav = wav[5 * 16000: 25 * 16000]  # middle slice, skip intro
        if len(wav) < 16000:
            print(f"{name}: too short"); continue

        f0, voiced, _ = librosa.pyin(wav, fmin=70, fmax=400, sr=16000, frame_length=1024)
        f0v = f0[np.isfinite(f0)]
        cent = librosa.feature.spectral_centroid(y=wav, sr=16000).mean()

        feats = processor(wav, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            out = model.generate(feats, max_new_tokens=120, return_dict_in_generate=True)
        ids = out.sequences[0].tolist()
        lang_id = ids[1] if len(ids) > 1 else -1
        lang = tok.decode([lang_id])
        text = tok.decode(ids, skip_special_tokens=True).strip()
        print(f"\n[{group}/{name}] f0_med={np.median(f0v) if len(f0v) else 0:.0f}Hz "
              f"voiced={float(np.mean(voiced)):.2f} centroid={cent:.0f}Hz")
        print(f"  detected_lang={lang!r} first_ids={ids[:5]}")
        print(f"  text: {text[:300]}", flush=True)


if __name__ == "__main__":
    main()
