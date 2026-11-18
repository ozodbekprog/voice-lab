"""Transcribe Uzbek dataset segments with sarahai/uzbek-stt-3 (Wav2Vec2 CTC)."""

import argparse
import csv
import json
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForCTC

DEFAULT_MODEL = (
    "/home/tony/voice-lab/checkpoints/hf_cache/hub/models--sarahai--uzbek-stt-3/"
    "snapshots/1dd15c1ab04f851bda3e2978760fa1cd9299d413"
)
SR = 16000


def load_wav(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/tony/voice-lab/dataset/metadata.csv")
    ap.add_argument("--out", default="/home/tony/voice-lab/dataset/metadata_asr.csv")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="", help="comma separated substrings to select rows")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCTC.from_pretrained(args.model, torch_dtype=torch.float32).to(device).eval()
    tok = proc.tokenizer

    rows = list(csv.DictReader(open(args.csv)))
    if args.ids:
        keys = [k.strip() for k in args.ids.split(",") if k.strip()]
        rows = [r for r in rows if any(k in r["path"] for k in keys)]
    if args.limit:
        rows = rows[: args.limit]

    out_rows = list(rows) if args.limit or args.ids else []
    results = {}
    for start in range(0, len(rows), args.batch):
        chunk = rows[start:start + args.batch]
        wavs = [load_wav(r["path"]) for r in chunk]
        inputs = proc(
            wavs, sampling_rate=SR, return_tensors="pt", padding=True,
            return_attention_mask=True,
        )
        with torch.no_grad():
            logits = model(inputs.input_values.to(device),
                           attention_mask=inputs.attention_mask.to(device)).logits
        probs = torch.softmax(logits.float(), dim=-1)
        ids = probs.argmax(dim=-1).cpu()
        conf = probs.max(dim=-1).values.cpu()
        attn = inputs.attention_mask.cpu()
        for i, r in enumerate(chunk):
            n = int(attn[i].sum())
            text = tok.decode(ids[i][:n], skip_special_tokens=True).strip()
            text = text.replace("[UNK]", "ʼ")
            mean_conf = float(conf[i][:n].mean())
            results[r["path"]] = (text, mean_conf)
            print(f"{Path(r['path']).name:34s} {r['duration_s']:>5}s  conf={mean_conf:.2f}  {text}")
        print(f"-- {min(start + args.batch, len(rows))}/{len(rows)}", flush=True)

    if out_rows:
        fields = list(out_rows[0].keys())
        if "text_asr" not in fields:
            fields += ["text_asr", "asr_conf", "asr_model"]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in out_rows:
                t, c = results.get(r["path"], ("", 0.0))
                r = dict(r)
                r["text_asr"] = t
                r["asr_conf"] = f"{c:.4f}"
                r["asr_model"] = "sarahai/uzbek-stt-3"
                w.writerow(r)
        print(f"written {args.out}")

    n_empty = sum(1 for t, _ in results.values() if not t)
    confs = [c for _, c in results.values()]
    summary = {
        "rows": len(rows),
        "empty": n_empty,
        "mean_conf": round(float(np.mean(confs)), 4) if confs else None,
        "low_conf_under_0.5": sum(1 for c in confs if c < 0.5),
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
