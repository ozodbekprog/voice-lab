#!/usr/bin/env python3
"""Evaluate F5-TTS Uzbek zero-shot cloning with a local reference clip."""

import argparse
import time
from pathlib import Path

import soundfile as sf
from f5_tts.api import F5TTS

ROOT = Path("/home/tony/voice-lab")

TEXTS = [
    ("uz1", "Assalomu alaykum! Bugun havo juda yaxshi, shuning uchun biz sayrga chiqamiz."),
    ("uz2", "Men bu loyihada yangi imkoniyatlarni sinab ko'ryapman va natija menga yoqadi."),
    ("uz3", "Qani, aytingchi, bu ishni qanday qilib tezroq va osonroq bajarsa bo'ladi?"),
]


def fold(text: str) -> str:
    return text.replace("ʻ", "'").replace("’", "'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(Path.home() / "models/f5-tts-uzbek/model.safetensors"))
    ap.add_argument("--vocab", default=str(Path.home() / "models/f5-tts-uzbek/vocab.txt"))
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--out", default=str(ROOT / "out/f5"))
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--tag", default="f5")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tts = F5TTS(model="F5TTS_v1_Base", ckpt_file=args.ckpt, vocab_file=args.vocab,
                device="cuda" if __import__("torch").cuda.is_available() else "cpu")
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    for name, text in TEXTS:
        t0 = time.time()
        path = out / f"{args.tag}_{name}.wav"
        wav, sr, _ = tts.infer(
            ref_file=args.ref, ref_text=fold(args.ref_text), gen_text=fold(text),
            nfe_step=args.nfe, file_wave=str(path), seed=0,
        )
        elapsed = time.time() - t0
        dur = len(wav) / sr
        print(f"OK {name}: {elapsed:.1f}s audio {dur:.1f}s RTF {elapsed / max(dur, 0.1):.2f} -> {path.name}", flush=True)


if __name__ == "__main__":
    main()
