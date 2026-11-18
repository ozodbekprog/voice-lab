#!/usr/bin/env python3
"""Generate Uzbek speech from a fine-tuned Qwen3-TTS custom voice, sentence by sentence."""

import argparse
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

ROOT = Path("/home/tony/voice-lab")


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--speaker", default="miralieva")
    ap.add_argument("--gap", type=float, default=0.25)
    ap.add_argument("--out", default=str(ROOT / "out/qwens"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = Qwen3TTSModel.from_pretrained(
        args.model, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    pieces, sr = [], 24000
    for i, sent in enumerate(split_sentences(args.text)):
        wavs, sr = model.generate_custom_voice(text=sent, speaker=args.speaker, language="Auto")
        wav = np.asarray(wavs[0], dtype=np.float32)
        pieces.append(wav)
        print(f"  [{i + 1}] {len(wav) / sr:.2f}s  {sent[:55]}", flush=True)
        pieces.append(np.zeros(int(sr * args.gap), dtype=np.float32))

    audio = np.concatenate(pieces[:-1])
    path = out / f"{args.tag}.wav"
    sf.write(path, audio, sr)
    print(f"OK {path.name}: {len(audio) / sr:.1f}s audio", flush=True)


if __name__ == "__main__":
    main()
