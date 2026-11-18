#!/usr/bin/env python3
"""Generate a single F5-TTS Uzbek sample from a text argument."""

import argparse
import time
from pathlib import Path

from f5_tts.api import F5TTS

ROOT = Path("/home/tony/voice-lab")


def fold(text: str) -> str:
    return text.replace("ʻ", "'").replace("’", "'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--tag", default="long")
    ap.add_argument("--out", default=str(ROOT / "out/f5"))
    ap.add_argument("--nfe", type=int, default=32)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tts = F5TTS(model="F5TTS_v1_Base",
                ckpt_file=str(Path.home() / "models/f5-tts-uzbek/model.safetensors"),
                vocab_file=str(Path.home() / "models/f5-tts-uzbek/vocab.txt"),
                device="cuda")
    t0 = time.time()
    path = out / f"{args.tag}.wav"
    wav, sr, _ = tts.infer(
        ref_file=args.ref, ref_text=fold(args.ref_text), gen_text=fold(args.text),
        nfe_step=args.nfe, file_wave=str(path), seed=0,
    )
    dur = len(wav) / sr
    print(f"OK {path.name}: {time.time() - t0:.1f}s for {dur:.1f}s audio", flush=True)


if __name__ == "__main__":
    main()
