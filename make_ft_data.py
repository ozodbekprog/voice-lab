"""Build Qwen3-TTS fine-tuning JSONL from ASR transcripts of the Uzbek dataset."""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def clean_text(text: str) -> str:
    t = text.strip().lower()
    t = t.replace("’", "ʼ").replace("'", "ʼ").replace("`", "ʼ")
    t = re.sub(r"[^\w\sʼ.,!?—–-]", "", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def repeated(text: str, limit: int = 3) -> bool:
    words = text.replace("—", " ").replace("-", " ").split()
    if len(words) >= 4 and Counter(words).most_common(1)[0][1] >= 4:
        return True
    if len(words) >= 5:
        grams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
        if Counter(grams).most_common(1)[0][1] >= limit:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/tony/voice-lab/dataset/metadata_asr.csv")
    ap.add_argument("--ref-audio", default="/home/tony/voice-lab/dataset/full/Miralieva/26855_Miralieva.wav")
    ap.add_argument("--out", default="/home/tony/voice-lab/ft/train_raw.jsonl")
    ap.add_argument("--min-logprob", type=float, default=-1.0)
    ap.add_argument("--max-nospeech", type=float, default=0.85)
    ap.add_argument("--min-dur", type=float, default=1.5)
    ap.add_argument("--max-dur", type=float, default=12.0)
    args = ap.parse_args()

    ref = Path(args.ref_audio)
    assert ref.exists(), ref
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept, dropped = [], Counter()
    total_dur = 0.0
    for r in csv.DictReader(open(args.csv)):
        dur = float(r["duration_s"])
        lp = float(r.get("asr_logprob") or -9)
        nsp = float(r.get("asr_nospeech") or 0)
        text = clean_text(r.get("text_asr") or r.get("text") or "")
        if not Path(r["path"]).exists():
            dropped["missing"] += 1
            continue
        if dur < args.min_dur or dur > args.max_dur:
            dropped["duration"] += 1
        elif lp < args.min_logprob:
            dropped["low_logprob"] += 1
        elif nsp > args.max_nospeech:
            dropped["no_speech"] += 1
        elif len(text.split()) < 2 or len(text) < 8:
            dropped["short_text"] += 1
        elif repeated(text):
            dropped["repetition"] += 1
        else:
            kept.append({"audio": r["path"], "text": text, "ref_audio": str(ref),
                         "duration_s": dur, "asr_logprob": lp})
            total_dur += dur

    with open(out, "w") as f:
        for k in kept:
            f.write(json.dumps({k2: k[k2] for k2 in ("audio", "text", "ref_audio")}, ensure_ascii=False) + "\n")

    print(f"kept {len(kept)} clips, {total_dur / 60:.1f} min; dropped {dict(dropped)}")
    print(f"written {out}")
    for k in kept[:6]:
        print(f"  {Path(k['audio']).name:32s} {k['duration_s']:>5.1f}s  {k['text'][:80]}")


if __name__ == "__main__":
    main()
