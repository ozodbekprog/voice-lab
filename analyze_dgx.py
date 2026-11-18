#!/usr/bin/env python3
"""Analyze voice-lab source clips: quality metrics + CAMPPlus speaker embeddings."""
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi

ROOT = Path("/home/tony/voice-lab")
OUT = ROOT / "analysis"
SR = 16000


def analyze_file(path: Path, model, device):
    wav, _ = librosa.load(path, sr=SR, mono=True)
    dur = len(wav) / SR if len(wav) else 0.0

    rms = float(np.sqrt(np.mean(wav ** 2))) if len(wav) else 0.0
    dbfs = 20 * np.log10(rms + 1e-9)
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    clip_pct = 100.0 * float(np.mean(np.abs(wav) > 0.999)) if len(wav) else 0.0

    intervals = librosa.effects.split(wav, top_db=30) if len(wav) else np.zeros((0, 2))
    speech_s = float(sum(e - s for s, e in intervals) / SR)
    speech_ratio = speech_s / dur if dur else 0.0

    # effective bandwidth: highest freq above -40 dB of mean-spectrum peak
    bw_khz = 0.0
    if len(wav) > 512:
        S = np.abs(librosa.stft(wav, n_fft=1024, hop_length=256))
        freqs = librosa.fft_frequencies(sr=SR, n_fft=1024)
        spec = S.mean(axis=1)
        idx = np.where(spec > spec.max() * 0.01)[0]
        if len(idx):
            bw_khz = float(freqs[idx[-1]] / 1000.0)

    emb = None
    if dur >= 0.5:
        w = torch.from_numpy(wav).float().unsqueeze(0)
        feat = torchaudio.compliance.kaldi.fbank(
            w, num_mel_bins=80, dither=0, sample_frequency=SR
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        with torch.no_grad():
            emb = model(feat.unsqueeze(0).to(device)).cpu().numpy().ravel()

    return {
        "group": path.parent.name,
        "file": path.name,
        "duration_s": round(dur, 2),
        "dbfs": round(dbfs, 1),
        "peak": round(peak, 3),
        "clip_pct": round(clip_pct, 3),
        "speech_ratio": round(speech_ratio, 2),
        "bandwidth_khz": round(bw_khz, 1),
    }, emb


def main():
    OUT.mkdir(exist_ok=True)
    sys.path.insert(0, str(ROOT / "seed-vc"))
    from modules.campplus.DTDNN import CAMPPlus

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(torch.load(ROOT / "seed-vc/campplus_cn_common.bin", map_location="cpu"))
    model.eval().to(device)

    files = sorted((ROOT / "data" / "data" / "ogg").glob("*/*.ogg"))
    rows, embs = [], []
    for f in files:
        r, e = analyze_file(f, model, device)
        rows.append(r)
        embs.append(e)
        print(f"{r['group']:<10} {r['file']:<24} {r['duration_s']:>6.1f}s  "
              f"{r['dbfs']:>6.1f}dBFS  speech={r['speech_ratio']:.2f}  bw={r['bandwidth_khz']:.1f}k", flush=True)

    valid = [i for i, e in enumerate(embs) if e is not None]
    E = np.stack([embs[i] for i in valid])
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    groups = np.array([rows[i]["group"] for i in valid])

    centroids = {g: E[groups == g].mean(axis=0) for g in set(groups)}
    for g in centroids:
        centroids[g] /= np.linalg.norm(centroids[g])

    sim = E @ E.T
    within, between = [], []
    for i in range(len(E)):
        for j in range(i + 1, len(E)):
            (within if groups[i] == groups[j] else between).append(sim[i, j])

    for i, idx in enumerate(valid):
        r = rows[idx]
        own = float(E[i] @ centroids[r["group"]])
        other = [float(E[i] @ c) for g, c in centroids.items() if g != r["group"]]
        r["sim_own_centroid"] = round(own, 3)
        r["sim_other_centroid"] = round(max(other), 3) if other else None
        r["embedding_ok"] = True
    for i, r in enumerate(rows):
        r.setdefault("embedding_ok", False)

    cross = float(np.mean([centroids[a] @ centroids[b] for a in centroids for b in centroids if a < b]))
    report = {
        "files": len(rows),
        "embedded": len(valid),
        "groups": {
            g: {
                "files": int(sum(1 for r in rows if r["group"] == g)),
                "seconds": round(sum(r["duration_s"] for r in rows if r["group"] == g), 1),
            } for g in sorted(set(r["group"] for r in rows))
        },
        "within_group_mean_cos": round(float(np.mean(within)), 3),
        "between_group_mean_cos": round(float(np.mean(between)), 3) if between else None,
        "centroid_cross_cos": round(cross, 3),
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== SPEAKER SIMILARITY")
    print(f"within-group mean cos:  {report['within_group_mean_cos']}")
    print(f"between-group mean cos: {report['between_group_mean_cos']}")
    print(f"centroid cross cos:     {report['centroid_cross_cos']}")
    print("\n=== LOWEST own-centroid similarity (possible outliers)")
    for r in sorted([r for r in rows if r["embedding_ok"]], key=lambda x: x["sim_own_centroid"])[:8]:
        print(f"{r['group']:<10} {r['file']:<24} own={r['sim_own_centroid']:.3f} other={r['sim_other_centroid']:.3f} dur={r['duration_s']}s")
    print("\n=== QUALITY OUTLIERS")
    for r in sorted(rows, key=lambda x: x["dbfs"])[:5]:
        print(f"quiet  {r['group']:<10} {r['file']:<24} {r['dbfs']}dBFS dur={r['duration_s']}s")
    for r in sorted(rows, key=lambda x: x["speech_ratio"])[:5]:
        print(f"sparse {r['group']:<10} {r['file']:<24} speech={r['speech_ratio']} dur={r['duration_s']}s")


if __name__ == "__main__":
    main()
