#!/usr/bin/env python3
"""Rank Miralieva segments by speaker similarity and f0 for a calmer reference."""

import glob
import json
import sys

import librosa
import numpy as np
import torch

sys.path.insert(0, "/home/tony/voice-lab")
from pair_cos import CAMPPlus, embed  # noqa: E402

REF = "/home/tony/voice-lab/data/data/ogg/Miralieva/26854_Miralieva.ogg"
SEG = "/home/tony/voice-lab/dataset/segments/Miralieva/*.wav"


def main():
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(
        torch.load("/home/tony/voice-lab/seed-vc/campplus_cn_common.bin", map_location="cpu")
    )
    model = model.to("cuda").eval()
    ref = embed(REF, model)
    rows = []
    for path in sorted(glob.glob(SEG)):
        y, sr = librosa.load(path, sr=16000, mono=True)
        dur = len(y) / sr
        if dur < 2.0:
            continue
        f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=400, sr=16000, frame_length=1024)
        f0v = f0[np.isfinite(f0)]
        rows.append(
            {
                "file": path,
                "name": path.rsplit("/", 1)[-1],
                "dur": round(dur, 2),
                "f0": round(float(np.median(f0v)) if len(f0v) else 0.0, 1),
                "voiced": round(float(np.mean(voiced)), 3),
                "cos": round(float(embed(path, model) @ ref), 3),
            }
        )
        print(rows[-1], flush=True)
    json.dump(rows, open("/home/tony/voice-lab/refs/seg_stats.json", "w"), indent=1)
    good = [r for r in rows if 180 <= r["f0"] <= 212 and r["dur"] >= 3.0 and r["cos"] >= 0.55]
    good.sort(key=lambda r: (-r["cos"], abs(r["f0"] - 200)))
    print("\nTOP")
    for r in good[:30]:
        print(r)


if __name__ == "__main__":
    main()
