#!/usr/bin/env python3
"""Transcribe representative clips with local whisper-small (offline)."""
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/tony/voice-lab/checkpoints/hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path("/home/tony/voice-lab")
CACHE = ROOT / "checkpoints" / "hf_cache" / "models--openai--whisper-small" / "snapshots"


def main():
    snap = sorted(CACHE.glob("*"))[0]
    processor = WhisperProcessor.from_pretrained(str(snap), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(str(snap), local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device, dtype=torch.float16 if device == "cuda" else torch.float32).eval()

    report = json.loads((ROOT / "analysis" / "report.json").read_text())["rows"]
    mir = sorted(
        [r for r in report if r["group"] == "Miralieva" and r["bandwidth_khz"] >= 6],
        key=lambda r: -r["duration_s"],
    )[:3]
    dot = sorted(
        [r for r in report if r["group"] == "dot" and r["bandwidth_khz"] >= 6],
        key=lambda r: -r["duration_s"],
    )[:3]
    narrow = [r for r in report if r["bandwidth_khz"] < 2 and r["duration_s"] > 20][:1]
    picks = mir[:2] + dot[:2] + mir[2:3] + narrow

    out = []
    for r in picks:
        path = ROOT / "data" / "data" / "ogg" / r["group"] / r["file"]
        wav, _ = librosa.load(path, sr=16000, mono=True)
        wav = wav[: 30 * 16000]
        feats = processor(wav, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(device, dtype=torch.float16 if device == "cuda" else torch.float32)
        with torch.no_grad():
            ids = model.generate(feats, max_new_tokens=220)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        entry = {"group": r["group"], "file": r["file"], "bw": r["bandwidth_khz"], "text": text}
        out.append(entry)
        print(f"[{r['group']}/{r['file']} bw={r['bandwidth_khz']}k]\n{text}\n", flush=True)

    (ROOT / "analysis" / "transcripts.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
