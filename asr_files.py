#!/usr/bin/env python3
"""Back-transcribe wav files with hostmepanda whisper-large-v3-turbo-uzbek (ct2)."""

import sys
import wave
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

SR = 16000
DEFAULT_MODEL = (
    "/home/tony/models/hf-cache/hub/models--hostmepanda--whisper-large-v3-turbo-uzbek-ct2/"
    "snapshots"
)


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


def resolve_model(path: str) -> str:
    p = Path(path)
    if (p / "model.bin").exists():
        return str(p)
    for s in sorted(p.glob("*/")):
        if (s / "model.bin").exists():
            return str(s)
    raise SystemExit(f"model.bin not found under {path}")


def main() -> None:
    model = WhisperModel(resolve_model(DEFAULT_MODEL), device="cpu", compute_type="int8",
                         cpu_threads=16)
    for f in sys.argv[1:]:
        segments, _info = model.transcribe(
            load16k(f), language="uz", beam_size=5,
            condition_on_previous_text=False, vad_filter=False,
            word_timestamps=False, temperature=0.0,
            compression_ratio_threshold=None, log_prob_threshold=None,
            no_speech_threshold=None,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        print(f"{Path(f).name}: {text}", flush=True)


if __name__ == "__main__":
    main()
