#!/usr/bin/env python3
"""Uzbek Chatterbox fine-tune inference without base t3_cfg weights."""

import argparse
import os
import random
import re
import sys

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file

TOOLKIT = os.path.expanduser("~/voice-lab/chatterbox-ft")
sys.path.insert(0, TOOLKIT)

from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.models.t3.t3 import T3
from src.chatterbox_.models.t3.modules.t3_config import T3Config
from src.chatterbox_.models.s3gen import S3Gen
from src.chatterbox_.models.voice_encoder import VoiceEncoder
from src.chatterbox_.models.tokenizers import EnTokenizer
from src.utils import trim_silence_with_vad

NEW_VOCAB_SIZE = 2454


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_sentences(text, max_chars=140):
    text = text.strip()
    if not text:
        return [text]

    def pack(parts, sep=" "):
        chunks, cur = [], ""
        for p in parts:
            if not p:
                continue
            cand = (cur + sep + p).strip() if cur else p
            if len(cand) <= max_chars:
                cur = cand
            else:
                if cur:
                    chunks.append(cur)
                cur = p
        if cur:
            chunks.append(cur)
        return chunks

    chunks = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= max_chars:
            chunks.append(sent)
            continue
        for part in pack(re.split(r"(?<=,)\s+", sent)):
            if len(part) <= max_chars:
                chunks.append(part)
            else:
                chunks.extend(pack(part.split()))
    return chunks or [text]


def load_engine(base_dir, merged, tok_path, device):
    ve = VoiceEncoder()
    ve.load_state_dict(load_file(os.path.join(base_dir, "ve.safetensors")))
    ve.to(device).eval()

    s3gen = S3Gen()
    s3gen.load_state_dict(load_file(os.path.join(base_dir, "s3gen.safetensors")), strict=False)
    s3gen.to(device).eval()

    t3 = T3(T3Config(text_tokens_dict_size=NEW_VOCAB_SIZE))
    missing, unexpected = t3.load_state_dict(load_file(merged), strict=False)
    if missing:
        raise RuntimeError(f"missing T3 keys: {missing[:8]}")
    if unexpected:
        print(f"unexpected keys ignored: {unexpected[:5]}", flush=True)
    t3.to(device).eval()

    tokenizer = EnTokenizer(tok_path)
    return ChatterboxTTS(t3, s3gen, ve, tokenizer, device)


def synthesize(engine, text, prompt, exaggeration, cfg_weight, rep_pen, temperature,
               seed, gap=0.15, max_chars=140):
    chunks = split_sentences(text, max_chars)
    sr = engine.sr
    silence = np.zeros(int(gap * sr), dtype=np.float32)
    pieces = []
    for i, chunk in enumerate(chunks):
        set_seed(seed + i)
        with torch.no_grad():
            wav = engine.generate(
                text=chunk, audio_prompt_path=prompt, exaggeration=exaggeration,
                cfg_weight=cfg_weight, repetition_penalty=rep_pen, temperature=temperature,
            )
        if isinstance(wav, tuple):
            wav = wav[0]
        arr = wav.squeeze().cpu().numpy().astype(np.float32)
        arr = trim_silence_with_vad(arr, sr)
        print(f"chunk {i + 1}/{len(chunks)}: {len(arr) / sr:.2f}s | {chunk[:70]}", flush=True)
        pieces.append(arr)
        pieces.append(silence)
    return np.concatenate(pieces[:-1]) if pieces else np.zeros(1, dtype=np.float32), sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--audio_prompt", required=True)
    ap.add_argument("--base_dir", default=os.path.expanduser("~/models/chatterbox-base"))
    ap.add_argument("--merged", default=os.path.expanduser("~/models/chatterbox-uz/t3_finetuned_merged.safetensors"))
    ap.add_argument("--tokenizer", default=os.path.expanduser("~/models/chatterbox-uz/tokenizer.json"))
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--exaggeration", type=float, default=0.5)
    ap.add_argument("--cfg", type=float, default=0.5)
    ap.add_argument("--rep", type=float, default=2.0)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gap", type=float, default=0.15)
    ap.add_argument("--max_chars", type=int, default=140)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    engine = load_engine(args.base_dir, args.merged, args.tokenizer, device)
    print("engine loaded", flush=True)
    audio, sr = synthesize(engine, args.text, args.audio_prompt, args.exaggeration,
                           args.cfg, args.rep, args.temp, args.seed, args.gap, args.max_chars)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sf.write(args.out, audio, sr)
    print(f"saved {args.out} {len(audio) / sr:.2f}s", flush=True)


if __name__ == "__main__":
    main()
