#!/usr/bin/env python3
"""F5-TTS Uzbek with sentence-level synthesis, pauses and speed control."""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from f5_tts.api import F5TTS

ROOT = Path("/home/tony/voice-lab")


def normalize(text: str) -> str:
    for ch in ("ʻ", "ʼ", "’", "‘", "`", "´"):
        text = text.replace(ch, "'")
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=str(ROOT / "out/f5s"))
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--speed", type=float, default=0.9)
    ap.add_argument("--gap", type=float, default=0.25)
    ap.add_argument("--split", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tts = F5TTS(model="F5TTS_v1_Base",
                ckpt_file=str(Path.home() / "models/f5-tts-uzbek/model.safetensors"),
                vocab_file=str(Path.home() / "models/f5-tts-uzbek/vocab.txt"),
                device="cuda")

    sentences = split_sentences(normalize(args.text)) if args.split else [normalize(args.text)]
    pieces, sr = [], 24000
    t0 = time.time()
    for i, sent in enumerate(sentences):
        wav, sr, _ = tts.infer(
            ref_file=args.ref, ref_text=normalize(args.ref_text), gen_text=sent,
            nfe_step=args.nfe, cfg_strength=args.cfg, speed=args.speed,
            seed=0, show_info=lambda *a, **k: None,
        )
        pieces.append(np.asarray(wav, dtype=np.float32))
        print(f"  [{i + 1}/{len(sentences)}] {len(wav) / sr:.2f}s  {sent[:60]}", flush=True)
        if i < len(sentences) - 1:
            pieces.append(np.zeros(int(sr * args.gap), dtype=np.float32))

    audio = np.concatenate(pieces)
    path = out / f"{args.tag}.wav"
    sf.write(path, audio, sr)
    print(f"OK {path.name}: {time.time() - t0:.1f}s gen, {len(audio) / sr:.1f}s audio", flush=True)


if __name__ == "__main__":
    main()
