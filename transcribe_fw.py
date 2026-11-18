"""Transcribe Uzbek dataset segments with hostmepanda whisper-large-v3-turbo-uzbek (ct2)."""

import argparse
import csv
import json
import wave
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

SR = 16000


def load16k(path: str) -> np.ndarray:
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

DEFAULT_MODEL = (
    "/home/tony/models/hf-cache/hub/models--hostmepanda--whisper-large-v3-turbo-uzbek-ct2/"
    "snapshots"
)


def resolve_model(path: str) -> str:
    p = Path(path)
    if (p / "model.bin").exists():
        return str(p)
    snaps = sorted(p.glob("*/"))
    for s in snaps:
        if (s / "model.bin").exists():
            return str(s)
    raise SystemExit(f"model.bin not found under {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/tony/voice-lab/dataset/metadata.csv")
    ap.add_argument("--out", default="/home/tony/voice-lab/dataset/metadata_asr.csv")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute", default="int8")
    ap.add_argument("--fallback", action="store_true", help="enable whisper temperature fallback")
    args = ap.parse_args()

    model_path = resolve_model(args.model)
    model = WhisperModel(model_path, device=args.device, compute_type=args.compute,
                         cpu_threads=16)

    rows = list(csv.DictReader(open(args.csv)))
    if args.ids:
        keys = [k.strip() for k in args.ids.split(",") if k.strip()]
        rows = [r for r in rows if any(k in r["path"] for k in keys)]
    if args.limit:
        rows = rows[: args.limit]

    results = {}
    total_audio = 0.0
    for i, r in enumerate(rows, 1):
        segments, info = model.transcribe(
            load16k(r["path"]), language="uz", beam_size=args.beam,
            condition_on_previous_text=False, vad_filter=False,
            word_timestamps=False,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if args.fallback else 0.0,
            compression_ratio_threshold=None if not args.fallback else 2.4,
            log_prob_threshold=None if not args.fallback else -1.0,
            no_speech_threshold=None if not args.fallback else 0.6,
        )
        segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()
        logprob = sum(s.avg_logprob for s in segs) / len(segs) if segs else -9.0
        nsp = max((s.no_speech_prob for s in segs), default=1.0)
        results[r["path"]] = (text, logprob, nsp)
        total_audio += float(r["duration_s"])
        print(f"[{i}/{len(rows)}] {Path(r['path']).name:32s} lp={logprob:+.2f} {text[:90]}", flush=True)

    if not args.limit:
        fields = list(rows[0].keys())
        for extra in ("text_asr", "asr_logprob", "asr_nospeech", "asr_model"):
            if extra not in fields:
                fields.append(extra)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                t, lp, nsp = results[r["path"]]
                r = dict(r)
                r.update(text_asr=t, asr_logprob=f"{lp:.3f}", asr_nospeech=f"{nsp:.3f}",
                         asr_model="hostmepanda/whisper-large-v3-turbo-uzbek-ct2")
                w.writerow(r)
        print(f"written {args.out}")

    lps = [v[1] for v in results.values()]
    print(json.dumps({
        "rows": len(rows),
        "audio_min": round(total_audio / 60, 2),
        "empty": sum(1 for v in results.values() if not v[0]),
        "mean_logprob": round(sum(lps) / len(lps), 3) if lps else None,
        "low_logprob_under_-1": sum(1 for x in lps if x < -1.0),
    }))


if __name__ == "__main__":
    main()
