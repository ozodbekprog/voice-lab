#!/usr/bin/env python3
"""Generate evaluation samples from a fine-tuned Qwen3-TTS checkpoint."""

import argparse
import time
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

ROOT = Path("/home/tony/voice-lab")
REF = ROOT / "dataset" / "full" / "Miralieva" / "26854_Miralieva.wav"

TEXTS = [
    ("uz", "Assalomu alaykum! Bugun havo juda yaxshi, shuning uchun biz sayrga chiqamiz."),
    ("uz2", "Men bu loyihada yangi imkoniyatlarni sinab ko'ryapman va natija menga yoqadi."),
    ("ru", "Привет! Это тестовый голос. Сегодня хорошая погода, и мы начинаем работу над проектом."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--speaker", default="speaker_test")
    ap.add_argument("--out", default="/home/tony/voice-lab/out/ft")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--mode", default="custom", choices=["custom", "clone", "both"])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        args.model, device_map="cuda:0", dtype=dtype, attn_implementation="sdpa",
    )
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    for name, text in TEXTS:
        if args.mode in ("custom", "both"):
            t0 = time.time()
            wavs, sr = model.generate_custom_voice(text=text, speaker=args.speaker)
            path = out / f"{args.model.rstrip('/').split('/')[-1]}_{name}_custom.wav"
            sf.write(path, wavs[0], sr)
            print(f"OK custom {name}: {time.time() - t0:.1f}s -> {path.name} {len(wavs[0]) / sr:.2f}s", flush=True)
        if args.mode in ("clone", "both"):
            t0 = time.time()
            wavs, sr = model.generate_voice_clone(
                text=text, language="Auto", ref_audio=str(REF),
                x_vector_only_mode=True, non_streaming_mode=True,
            )
            path = out / f"{args.model.rstrip('/').split('/')[-1]}_{name}_clone.wav"
            sf.write(path, wavs[0], sr)
            print(f"OK clone  {name}: {time.time() - t0:.1f}s -> {path.name} {len(wavs[0]) / sr:.2f}s", flush=True)


if __name__ == "__main__":
    main()
