#!/usr/bin/env python3
"""Generate multiple Chatterbox candidates per sentence for CER-based selection."""

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.expanduser("~/voice-lab"))
from cb_uz import load_engine, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="JSON list of {id,text}")
    ap.add_argument("--audio_prompt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seeds", default="42-49")
    ap.add_argument("--temps", default="0.6")
    ap.add_argument("--exaggeration", type=float, default=0.5)
    ap.add_argument("--cfg-weight", type=float, default=0.5)
    args = ap.parse_args()

    def parse_range(s):
        if "-" in s:
            a, b = s.split("-")
            return list(range(int(a), int(b) + 1))
        return [int(x) for x in s.split(",")]

    seeds = parse_range(args.seeds)
    temps = [float(x) for x in args.temps.split(",")]
    items = json.load(open(args.items))
    os.makedirs(args.outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = load_engine(os.path.expanduser("~/models/chatterbox-base"),
                         os.path.expanduser("~/models/chatterbox-uz/t3_finetuned_merged.safetensors"),
                         os.path.expanduser("~/models/chatterbox-uz/tokenizer.json"), device)
    print("engine loaded", flush=True)

    for item in items:
        for temp in temps:
            for seed in seeds:
                path = os.path.join(args.outdir, f"{item['id']}_s{seed}_t{temp}.wav")
                if os.path.exists(path):
                    continue
                set_seed(seed)
                with torch.no_grad():
                    wav = engine.generate(text=item["text"], audio_prompt_path=args.audio_prompt,
                                          exaggeration=args.exaggeration, cfg_weight=args.cfg_weight,
                                          repetition_penalty=2.0, temperature=temp)
                if isinstance(wav, tuple):
                    wav = wav[0]
                arr = wav.squeeze().cpu().numpy().astype(np.float32)
                sf.write(path, arr, engine.sr)
                print(f"ok {path} {len(arr) / engine.sr:.2f}s", flush=True)


if __name__ == "__main__":
    main()
