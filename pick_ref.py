#!/usr/bin/env python3
"""List clean Miralieva segments suitable for building an F5 reference clip."""

import csv
import re
from pathlib import Path

CSV = Path("/home/tony/voice-lab/dataset/metadata_asr.csv")


def clean(text: str) -> bool:
    words = text.split()
    if len(words) < 4:
        return False
    if len(set(words)) / len(words) < 0.6:
        return False
    if re.search(r"\b(\w+)( \1\b){2,}", text):
        return False
    return True


def main():
    rows = list(csv.DictReader(open(CSV)))
    cands = []
    for r in rows:
        if r["group"] != "Miralieva":
            continue
        try:
            lp = float(r["asr_logprob"])
            ns = float(r["asr_nospeech"])
            dur = float(r["duration_s"])
        except (KeyError, ValueError):
            continue
        text = r["text_asr"].strip()
        if 4.0 <= dur <= 11.0 and lp >= -0.5 and ns <= 0.6 and clean(text):
            cands.append((lp, ns, dur, r["path"], text))
    cands.sort(key=lambda x: -x[0])
    for c in cands[:30]:
        print(f"lp={c[0]:.2f} ns={c[1]:.2f} dur={c[2]:.1f} {Path(c[3]).name} | {c[4]}")


if __name__ == "__main__":
    main()
