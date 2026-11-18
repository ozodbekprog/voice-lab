#!/usr/bin/env python3
"""Transcribe dataset segments with local whisper-large-v3-turbo (offline)."""
import csv
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/tony/voice-lab/checkpoints/hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path("/home/tony/voice-lab")
CACHE = ROOT / "checkpoints" / "hf_cache" / "hub" / "models--openai--whisper-large-v3-turbo" / "snapshots"


def main():
    snap = sorted(CACHE.glob("*"))[0]
    processor = WhisperProcessor.from_pretrained(str(snap), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        str(snap), local_files_only=True, torch_dtype=torch.float16
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    rows = list(csv.DictReader((ROOT / "dataset" / "metadata.csv").open()))
    langs = Counter()
    for i, r in enumerate(rows, 1):
        wav, _ = librosa.load(r["path"], sr=16000, mono=True)
        feats = processor(wav, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(device, dtype=torch.float16 if device == "cuda" else torch.float32)
        with torch.no_grad():
            ids = model.generate(
                feats, max_new_tokens=220, language=None,
                condition_on_prev_tokens=False,
            )
        seq = ids[0].tolist()
        lang = processor.tokenizer.decode([seq[1]]) if len(seq) > 1 else "?"
        text = processor.tokenizer.decode(seq, skip_special_tokens=True).strip()
        r["text"] = text
        r["lang"] = lang.strip("<|>")
        langs[r["lang"]] += 1
        print(f"{i:3d}/{len(rows)} [{r['lang']:>3}] {r['path'].split('/')[-1]}: {text[:110]}", flush=True)

    out = ROOT / "dataset" / "metadata_asr.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    (ROOT / "analysis" / "asr-summary.json").write_text(json.dumps({
        "segments": len(rows),
        "languages": dict(langs),
        "empty": sum(1 for r in rows if not r["text"]),
        "total_chars": sum(len(r["text"]) for r in rows),
    }, indent=2) + "\n")
    print("\nlanguages:", dict(langs))


if __name__ == "__main__":
    main()
