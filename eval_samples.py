#!/usr/bin/env python3
"""Evaluate generated samples: speaker similarity vs reference + basic audio stats."""
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi

ROOT = Path("/home/tony/voice-lab")
sys.path.insert(0, str(ROOT / "seed-vc"))
from modules.campplus.DTDNN import CAMPPlus  # noqa: E402

REF = ROOT / "dataset" / "full" / "Miralieva" / "26854_Miralieva.wav"
OUT = ROOT / "out" / "baseline"


def embed(path, model):
    wav, _ = librosa.load(path, sr=16000, mono=True)
    if len(wav) < 16000 * 2:
        return None, None
    w = torch.from_numpy(wav).float().unsqueeze(0)
    feat = torchaudio.compliance.kaldi.fbank(w, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    with torch.no_grad():
        e = model(feat.unsqueeze(0).to("cuda")).cpu().numpy().ravel()
    return e / np.linalg.norm(e), wav


def main():
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(torch.load(ROOT / "seed-vc" / "campplus_cn_common.bin", map_location="cpu"))
    model = model.to("cuda").eval()

    ref_e, _ = embed(REF, model)
    rows, embs = [], {}
    for f in sorted(OUT.glob("*.wav")):
        e, wav = embed(f, model)
        if e is None:
            print(f"{f.name}: too short")
            continue
        dur = len(wav) / 16000
        rms = 20 * np.log10(np.sqrt(np.mean(wav ** 2)) + 1e-9)
        cos = float(e @ ref_e)
        rows.append((f.name, dur, rms, cos))
        embs[f.name] = e

    rows.sort(key=lambda r: -r[3])
    print(f"reference: {REF.name}  ({librosa.get_duration(path=str(REF)):.1f}s)\n")
    print(f"{'sample':<26}{'dur_s':>7}{'dBFS':>8}{'cos_to_ref':>12}")
    for name, dur, rms, cos in rows:
        print(f"{name:<26}{dur:>7.2f}{rms:>8.1f}{cos:>12.3f}")

    print("\npairwise cos:")
    names = [r[0] for r in rows]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            print(f"  {a} <-> {b}: {float(embs[a] @ embs[b]):.3f}")


if __name__ == "__main__":
    main()
