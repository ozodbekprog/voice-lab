#!/usr/bin/env python3
"""Inter-recording CAMPPlus cosine among Miralieva clips (same-speaker ceiling)."""

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


def embed(path, model):
    wav, _ = librosa.load(path, sr=16000, mono=True)
    w = torch.from_numpy(wav).float().unsqueeze(0)
    feat = torchaudio.compliance.kaldi.fbank(w, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    with torch.no_grad():
        e = model(feat.unsqueeze(0).to("cuda")).cpu().numpy().ravel()
    return e / np.linalg.norm(e)


def main():
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(torch.load(ROOT / "seed-vc" / "campplus_cn_common.bin", map_location="cpu"))
    model = model.to("cuda").eval()

    files = sys.argv[1:]
    embs = {Path(f).stem: embed(f, model) for f in files}
    names = list(embs)
    print(f"{'pair':<44}{'cos':>6}")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            print(f"{a + ' <-> ' + b:<44}{float(embs[a] @ embs[b]):>6.3f}")


if __name__ == "__main__":
    main()
