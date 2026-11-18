#!/usr/bin/env python3
"""Speaker cosine overall and per 3s window against the 26854 reference."""

import sys

import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi

sys.path.insert(0, "/home/tony/voice-lab")
from pair_cos import CAMPPlus, embed  # noqa: E402

REF = "/home/tony/voice-lab/data/data/ogg/Miralieva/26854_Miralieva.ogg"


def embed_array(y, model):
    w = torch.from_numpy(y).float().unsqueeze(0)
    feat = torchaudio.compliance.kaldi.fbank(
        w, num_mel_bins=80, dither=0, sample_frequency=16000
    )
    feat = feat - feat.mean(dim=0, keepdim=True)
    with torch.no_grad():
        e = model(feat.unsqueeze(0).to("cuda")).cpu().numpy().ravel()
    return e / np.linalg.norm(e)


def main():
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(
        torch.load("/home/tony/voice-lab/seed-vc/campplus_cn_common.bin", map_location="cpu")
    )
    model = model.to("cuda").eval()
    ref = embed(REF, model)
    for path in sys.argv[1:]:
        y, sr = librosa.load(path, sr=16000, mono=True)
        win, hop = int(3.0 * sr), int(1.0 * sr)
        windows = []
        for i in range(0, max(1, len(y) - win + 1), hop):
            seg = y[i : i + win]
            if len(seg) < int(1.5 * sr):
                continue
            windows.append(round(float(embed_array(seg, model) @ ref), 3))
        print(
            f"{path} dur={len(y)/sr:.2f}s cos={float(embed(path, model) @ ref):.3f} "
            f"win_min={min(windows):.3f} win_mean={np.mean(windows):.3f} windows={windows}"
        )


if __name__ == "__main__":
    main()
