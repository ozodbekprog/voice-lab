#!/usr/bin/env python3
"""Move rejected RVC slices out of every experiment subdir so index/training stay clean."""

import csv
import os
import shutil

rows = list(csv.DictReader(open("/home/tony/voice-lab/rvc_data/slice_cos.csv")))
keep = {
    os.path.splitext(r["file"])[0]
    for r in rows
    if float(r["cos"]) >= 0.45 and float(r["dur"]) >= 2.5
}
root = "/home/tony/voice-lab/RVC/logs/voice_her"
rejected = "/home/tony/voice-lab/rvc_data/rejected_slices"
os.makedirs(rejected, exist_ok=True)

subdirs = [
    ("0_gt_wavs", ".wav"),
    ("1_16k_wavs", ".wav"),
    ("2a_f0", ".wav.npy"),
    ("2b-f0nsf", ".wav.npy"),
    ("3_feature768", ".npy"),
]
moved = 0
for name in sorted(os.listdir(os.path.join(root, "0_gt_wavs"))):
    stem = name[:-4]
    if stem in keep:
        continue
    for sub, suffix in subdirs:
        src = os.path.join(root, sub, stem + suffix)
        if os.path.exists(src):
            shutil.move(src, os.path.join(rejected, sub + "__" + stem + suffix))
            moved += 1
print(f"kept={len(keep)} moved={moved}")
for sub, _ in subdirs:
    print(sub, len(os.listdir(os.path.join(root, sub))))
