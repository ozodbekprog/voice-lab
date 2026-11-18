#!/usr/bin/env python3
"""Check SNR and possible multi-speaker content in sample files."""
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

ROOT = Path("/home/tony/voice-lab")
sys.path.insert(0, str(ROOT / "seed-vc"))
from modules.campplus.DTDNN import CAMPPlus  # noqa: E402

FILES = [
    ("Miralieva", "33659_Miralieva.ogg"),
    ("Miralieva", "26861_Miralieva.ogg"),
    ("dot", "50344_dot.ogg"),
    ("dot", "53418_dot.ogg"),
]


def snr_db(wav, sr=16000):
    ints = librosa.effects.split(wav, top_db=30)
    mask = np.zeros(len(wav), dtype=bool)
    for s, e in ints:
        mask[s:e] = True
    speech = wav[mask]
    noise = wav[~mask]
    if len(noise) < sr // 4 or len(speech) < sr // 4:
        return None
    s_rms = np.sqrt(np.mean(speech ** 2))
    n_rms = np.sqrt(np.mean(noise ** 2))
    return 20 * np.log10(s_rms / (n_rms + 1e-12) + 1e-12)


def main():
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(torch.load(ROOT / "seed-vc" / "campplus_cn_common.bin", map_location="cpu"))
    model = model.to("cuda").eval()

    for group, name in FILES:
        wav, _ = librosa.load(ROOT / "data" / "data" / "ogg" / group / name, sr=16000, mono=True)
        s = snr_db(wav)
        embs, times = [], []
        win, hop = 3 * 16000, int(1.5 * 16000)
        for start in range(0, max(1, len(wav) - win), hop):
            seg = wav[start:start + win]
            if np.sqrt(np.mean(seg ** 2)) < 0.004:
                continue
            t = torch.from_numpy(seg).float().unsqueeze(0)
            feat = torchaudio_fbank(t)
            if feat.shape[0] < 40:
                continue
            with torch.no_grad():
                e = model(feat.unsqueeze(0).to("cuda")).cpu().numpy().ravel()
            embs.append(e / np.linalg.norm(e))
            times.append(start / 16000)
        if not embs:
            print(f"{group}/{name}: no windows")
            continue
        E = np.stack(embs)
        sim = E @ E.T
        iu = np.triu_indices(len(E), 1)
        pair = sim[iu]
        # k-means k=2 on embeddings (cosine -> normalize)
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(E)
        sizes = np.bincount(km.labels_, minlength=2)
        c0, c1 = km.cluster_centers_
        cc = float(c0 @ c1 / (np.linalg.norm(c0) * np.linalg.norm(c1)))
        print(f"\n{group}/{name}: snr={s if s is not None else -1:.1f}dB windows={len(E)} "
              f"pair_sim mean={pair.mean():.2f} p10={np.percentile(pair,10):.2f}")
        print(f"  kmeans sizes={sizes.tolist()} cluster_cos={cc:.3f}")


def torchaudio_fbank(w):
    import torchaudio
    import torchaudio.compliance.kaldi
    feat = torchaudio.compliance.kaldi.fbank(w, num_mel_bins=80, dither=0, sample_frequency=16000)
    return feat - feat.mean(dim=0, keepdim=True)


if __name__ == "__main__":
    import torchaudio  # noqa: F401
    main()
