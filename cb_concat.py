#!/usr/bin/env python3
"""Concatenate selected Chatterbox candidates with trim and loudness normalization."""

import argparse
import json
import os

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gap", type=float, default=0.25)
    ap.add_argument("--trim-db", type=float, default=40.0)
    ap.add_argument("--lufs", type=float, default=-16.0)
    args = ap.parse_args()

    picks = json.load(open(args.picks))
    pieces, sr = [], None
    for p in picks:
        audio, sr = sf.read(os.path.join(args.dir, p["file"]), dtype="float32")
        trimmed, _ = librosa.effects.trim(audio, top_db=args.trim_db)
        fade = max(int(0.01 * sr), 1)
        if len(trimmed) > 2 * fade:
            trimmed[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            trimmed[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        print(f"{p['file']}: {len(audio) / sr:.2f}s -> {len(trimmed) / sr:.2f}s")
        pieces.append(trimmed)
        pieces.append(np.zeros(int(args.gap * sr), dtype=np.float32))
    final = np.concatenate(pieces[:-1]) if pieces else np.zeros(1, dtype=np.float32)

    meter = pyln.Meter(sr)
    loud = meter.integrated_loudness(final)
    final = pyln.normalize.loudness(final, loud, args.lufs)
    peak = float(np.max(np.abs(final)))
    if peak > 0.99:
        final = final * (0.99 / peak)

    sf.write(args.out, final, sr)
    print(f"wrote {args.out} {len(final) / sr:.2f}s {sr}Hz lufs {loud:.1f}->{args.lufs}")


if __name__ == "__main__":
    main()
