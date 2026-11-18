#!/usr/bin/env python3
"""Zero-shot Qwen3-TTS baseline: Russian + auto-language Uzbek attempt."""
import time
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

ROOT = Path("/home/tony/voice-lab")
REF = ROOT / "dataset" / "full" / "Miralieva" / "26854_Miralieva.wav"
OUT = ROOT / "out" / "baseline"

TEXTS = [
    ("ru", "Russian", "Привет! Это тестовый голос. Сегодня хорошая погода, и мы начинаем работу над проектом."),
    ("uz_auto", "Auto", "Salom! Bu sinov ovozi. Bugun havo yaxshi va biz ishni boshlaymiz."),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    for name, lang, text in TEXTS:
        t0 = time.time()
        try:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=lang,
                ref_audio=str(REF),
                x_vector_only_mode=True,
            )
            out = OUT / f"qwen_{name}.wav"
            sf.write(out, wavs[0], sr)
            print(f"OK {name} ({lang}): {time.time() - t0:.1f}s -> {out.name} {len(wavs[0]) / sr:.2f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {name} ({lang}): {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
