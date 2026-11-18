#!/usr/bin/env python3
"""Evaluate a directory of generated samples: CAMPPlus speaker cosine vs reference."""

import argparse
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

DEFAULT_REF = ROOT / "dataset/full/Miralieva/26854_Miralieva.wav"


def embed(path, model):
    wav, _ = librosa.load(path, sr=16000, mono=True)
    if len(wav) < 16000:
        return None, None
    w = torch.from_numpy(wav).float().unsqueeze(0)
    feat = torchaudio.compliance.kaldi.fbank(w, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    with torch.no_grad():
        e = model(feat.unsqueeze(0).to("cuda")).cpu().numpy().ravel()
    return e / np.linalg.norm(e), wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--ref", default=str(DEFAULT_REF))
    ap.add_argument("--refs", default="", help="extra reference wavs, comma separated")
    args = ap.parse_args()

    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(torch.load(ROOT / "seed-vc/campplus_cn_common.bin", map_location="cpu"))
    model = model.to("cuda").eval()

    refs = [Path(p) for p in [args.ref] + [x for x in args.refs.split(",") if x.strip()]]
    ref_embs = {}
    for p in refs:
        e, _ = embed(p, model)
        ref_embs[p.name] = e

    rows = []
    for f in sorted(Path(args.dir).glob("*.wav")):
        e, wav = embed(f, model)
        if e is None:
            print(f"{f.name}: too short")
            continue
        dur = len(wav) / 16000
        rms = 20 * np.log10(np.sqrt(np.mean(wav ** 2)) + 1e-9)
        cos = {k: float(e @ v) for k, v in ref_embs.items()}
        rows.append((f.name, dur, rms, cos))

    header = f"{'sample':<34}{'dur_s':>7}{'dBFS':>7}" + "".join(f"{('cos:' + k)[:22]:>24}" for k in ref_embs)
    print(header)
    for name, dur, rms, cos in sorted(rows, key=lambda r: -max(r[3].values())):
        line = f"{name:<34}{dur:>7.2f}{rms:>7.1f}"
        line += "".join(f"{v:>24.3f}" for v in cos.values())
        print(line)


if __name__ == "__main__":
    main()
