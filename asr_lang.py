#!/usr/bin/env python3
"""Back-transcribe wav files with openai/whisper-large-v3-turbo (any language)."""

import os
import sys
import wave
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/tony/models/hf-cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: E402

SR = 16000


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


def main() -> None:
    lang, files = sys.argv[1], sys.argv[2:]
    root = Path(os.environ["HF_HOME"]) / "hub" / "models--openai--whisper-large-v3-turbo" / "snapshots"
    snap = sorted(p for p in root.glob("*") if (p / "model.safetensors").exists())[0]
    processor = WhisperProcessor.from_pretrained(str(snap), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        str(snap), local_files_only=True, torch_dtype=torch.float16,
    ).to("cuda").eval()

    for f in files:
        wav = load16k(f)
        feats = processor(wav, sampling_rate=SR, return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(
                feats.to("cuda", dtype=torch.float16), max_new_tokens=400,
                language=lang, task="transcribe",
            )
        print(f"{Path(f).name}: {processor.batch_decode(ids, skip_special_tokens=True)[0]}", flush=True)


if __name__ == "__main__":
    main()
