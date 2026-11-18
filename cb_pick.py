#!/usr/bin/env python3
"""Score Chatterbox candidates by ASR character error rate and pick the best."""

import argparse
import glob
import json
import os
import re
import wave

import numpy as np
from faster_whisper import WhisperModel

SR = 16000
MODEL = ("/home/tony/models/hf-cache/hub/models--hostmepanda--whisper-large-v3-turbo-uzbek-ct2/snapshots")


def resolve_model(path):
    p = __import__("pathlib").Path(path)
    if (p / "model.bin").exists():
        return str(p)
    for s in sorted(p.glob("*/")):
        if (s / "model.bin").exists():
            return str(s)
    raise SystemExit("model.bin not found")


def load16k(path):
    with wave.open(path, "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        data = w.readframes(w.getnframes())
    dtype = {2: np.int16, 4: np.int32}[sw]
    scale = 32768.0 if sw == 2 else 2147483648.0
    a = np.frombuffer(data, dtype=dtype).astype(np.float32) / scale
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        n = int(round(len(a) * SR / sr))
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a)
    return np.ascontiguousarray(a.astype(np.float32))


def norm(s):
    s = s.lower()
    for ch in ("‘", "’", "`", "ʼ", "ʻ"):
        s = s.replace(ch, "'")
    s = re.sub(r"[^a-z' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev = d.copy()
        d[0] = i
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + cost)
    return d[len(h)] / max(len(r), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--score-out", required=True)
    ap.add_argument("--pick-out", required=True)
    args = ap.parse_args()

    items = json.load(open(args.items))
    model = WhisperModel(resolve_model(MODEL), device="cpu", compute_type="int8", cpu_threads=16)
    picks, report = [], []
    for item in items:
        best = None
        for path in sorted(glob.glob(os.path.join(args.dir, f"{item['id']}_*.wav"))):
            segments, _ = model.transcribe(load16k(path), language="uz", beam_size=5,
                                           condition_on_previous_text=False, vad_filter=False,
                                           word_timestamps=False, temperature=0.0)
            text = " ".join(s.text.strip() for s in segments).strip()
            score = cer(item["text"], text)
            report.append({"id": item["id"], "file": os.path.basename(path), "cer": round(score, 4), "asr": text})
            print(f"{item['id']} {os.path.basename(path):28s} cer={score:.3f} {text[:80]}", flush=True)
            if best is None or score < best[0]:
                best = (score, path, text)
        picks.append({"id": item["id"], "file": os.path.basename(best[1]), "cer": round(best[0], 4), "asr": best[2]})
        print(f"PICK {item['id']} -> {os.path.basename(best[1])} cer={best[0]:.3f}", flush=True)

    json.dump(report, open(args.score_out, "w"), ensure_ascii=False, indent=1)
    json.dump(picks, open(args.pick_out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
