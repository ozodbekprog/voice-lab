#!/usr/bin/env python3
"""Build cleaned dataset: VAD trim, split into 3-12s segments, loudness normalize."""
import csv
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path("/home/tony/voice-lab")
SRC = ROOT / "data" / "data" / "ogg"
OUT = ROOT / "dataset"
SEG_MIN = 1.0
SEG_MAX = 12.0
TARGET_DBFS = -20.0

SRC_SR = 48000


def vad_intervals(wav16: np.ndarray):
    ints = librosa.effects.split(wav16, top_db=30)
    merged = []
    for s, e in ints:
        if merged and (s - merged[-1][1]) < int(0.25 * 16000):
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def pack(intervals, max_len):
    """Greedy pack VAD intervals into chunks <= max_len; split long ones evenly."""
    chunks = []
    cur = None
    for s, e in intervals:
        dur = (e - s) / 16000
        if dur > max_len:
            if cur:
                chunks.append(cur)
                cur = None
            n = int(np.ceil(dur / max_len))
            step = (e - s) / n
            for i in range(n):
                chunks.append((int(s + i * step), int(min(e, s + (i + 1) * step))))
            continue
        if cur is None:
            cur = (s, e)
        elif (e - cur[0]) / 16000 <= max_len:
            cur = (cur[0], e)
        else:
            chunks.append(cur)
            cur = (s, e)
    if cur:
        chunks.append(cur)
    return chunks


def norm_gain(wav: np.ndarray, target_dbfs: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(wav ** 2))) if len(wav) else 0.0
    if rms <= 1e-9:
        return wav
    gain = 10 ** ((target_dbfs - 20 * np.log10(rms)) / 20)
    peak = float(np.max(np.abs(wav))) * gain
    if peak > 0.95:
        gain *= 0.95 / peak
    return wav * gain


def main():
    (OUT / "segments").mkdir(parents=True, exist_ok=True)
    (OUT / "full").mkdir(parents=True, exist_ok=True)
    rows = []
    for group in ["Miralieva", "dot"]:
        gdir = OUT / "segments" / group
        gdir.mkdir(parents=True, exist_ok=True)
        (OUT / "full" / group).mkdir(parents=True, exist_ok=True)
        for f in sorted((SRC / group).glob("*.ogg")):
            wav48, _ = librosa.load(f, sr=SRC_SR, mono=True)
            wav16 = librosa.resample(wav48, orig_sr=SRC_SR, target_sr=16000)
            ints = vad_intervals(wav16)
            if not ints:
                print(f"SKIP no speech: {f.name}")
                continue

            # full trimmed version
            s16, e16 = ints[0][0], ints[-1][1]
            trim = wav48[int(s16 / 16000 * SRC_SR): int(e16 / 16000 * SRC_SR)]
            sf.write(OUT / "full" / group / f"{f.stem}.wav", norm_gain(trim, TARGET_DBFS), SRC_SR, subtype="PCM_16")

            for i, (a, b) in enumerate(pack(ints, SEG_MAX)):
                if (b - a) / 16000 < SEG_MIN:
                    continue
                a48, b48 = int(a / 16000 * SRC_SR), int(b / 16000 * SRC_SR)
                seg = wav48[a48:b48]
                if len(seg) < int(SEG_MIN * SRC_SR):
                    continue
                seg = norm_gain(seg, TARGET_DBFS)
                name = f"{f.stem}_{i:02d}.wav"
                sf.write(gdir / name, seg, SRC_SR, subtype="PCM_16")
                rows.append({
                    "path": str(gdir / name),
                    "group": group,
                    "source": f.name,
                    "start_s": round(a / 16000, 2),
                    "end_s": round(b / 16000, 2),
                    "duration_s": round((b - a) / 16000, 2),
                    "text": "",
                })
        print(f"{group}: done", flush=True)

    with (OUT / "metadata.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    total = sum(r["duration_s"] for r in rows)
    summary = {
        "segments": len(rows),
        "total_min": round(total / 60, 1),
        "by_group": {
            g: {
                "segments": sum(1 for r in rows if r["group"] == g),
                "minutes": round(sum(r["duration_s"] for r in rows if r["group"] == g) / 60, 1),
            } for g in ("Miralieva", "dot")
        },
    }
    (ROOT / "analysis" / "dataset-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
