#!/usr/bin/env python3
"""Zero-shot XTTS v2 baseline: Russian + Uzbek attempts with Miralieva reference."""
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from TTS.api import TTS

ROOT = Path("/home/tony/voice-lab")
REF = ROOT / "dataset" / "full" / "Miralieva" / "26854_Miralieva.wav"
OUT = ROOT / "out" / "baseline"

TEXTS = [
    ("ru", "ru", "Привет! Это тестовый голос. Сегодня хорошая погода, и мы начинаем работу над проектом."),
    ("uz_via_en", "en", "Salom! Bu sinov ovozi. Bugun havo yaxshi va biz ishni boshlaymiz."),
    ("uz_via_tr", "tr", "Salom! Bu sinov ovozi. Bugun havo yaxshi va biz ishni boshlaymiz."),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    for name, lang, text in TEXTS:
        t0 = time.time()
        out = OUT / f"xtts_{name}.wav"
        try:
            tts.tts_to_file(text=text, speaker_wav=str(REF), language=lang, file_path=str(out))
            dur = len(sf.read(str(out))[0]) / sf.info(str(out)).samplerate
            print(f"OK {name} ({lang}): {time.time() - t0:.1f}s -> {out.name} {dur:.2f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {name} ({lang}): {exc}", flush=True)


if __name__ == "__main__":
    main()
