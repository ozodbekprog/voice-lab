#!/usr/bin/env python3
"""Generate a single Qwen3-TTS zero-shot sample (language selectable)."""

import argparse
import time
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

ROOT = Path("/home/tony/voice-lab")
REF = ROOT / "dataset" / "full" / "Miralieva" / "26854_Miralieva.wav"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--lang", default="Russian")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=str(ROOT / "out/final"))
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = Qwen3TTSModel.from_pretrained(
        args.model, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    t0 = time.time()
    wavs, sr = model.generate_voice_clone(
        text=args.text, language=args.lang, ref_audio=str(REF),
        x_vector_only_mode=True,
    )
    path = out / f"{args.tag}.wav"
    sf.write(path, wavs[0], sr)
    print(f"OK {path.name}: {time.time() - t0:.1f}s for {len(wavs[0]) / sr:.1f}s audio", flush=True)


if __name__ == "__main__":
    main()
