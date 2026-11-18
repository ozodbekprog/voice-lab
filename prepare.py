#!/usr/bin/env python3
"""Prepare source audio: dedup by md5, group by speaker tag, build manifest."""
import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

SRC = Path("/Users/probox/Documents/Qoder/2026-11-16/4c2104e8/audio-from-saved")
OUT = Path("/Users/probox/Documents/Qoder/2026-11-16/f09f7a44/voice-work")
OGG_DIR = OUT / "data" / "ogg"
LOG = OUT / "logs"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return round(float(out), 3)


def main() -> None:
    OGG_DIR.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(SRC.glob("*.ogg")):
        stem = path.stem
        m = re.match(r"^(\d+)_(\w+)$", stem)
        if not m:
            print(f"SKIP unexpected name: {path.name}")
            continue
        clip_id, group = m.group(1), m.group(2)
        rows.append({
            "file": path.name,
            "clip_id": int(clip_id),
            "group": group,
            "bytes": path.stat().st_size,
            "duration_s": duration(path),
            "md5": md5(path),
        })

    # canonical = first (lowest clip_id) file per md5
    rows.sort(key=lambda r: (r["group"], r["clip_id"]))
    canonical = {}
    for r in rows:
        if r["md5"] in canonical:
            r["canonical"] = False
            r["duplicate_of"] = canonical[r["md5"]]["file"]
        else:
            r["canonical"] = True
            r["duplicate_of"] = ""
            canonical[r["md5"]] = r

    kept = [r for r in rows if r["canonical"]]
    for r in kept:
        dst_dir = OGG_DIR / r["group"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC / r["file"], dst_dir / r["file"])

    with (OUT / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "clip_id", "file", "duration_s", "bytes", "md5", "canonical", "duplicate_of"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    summary = {}
    for r in kept:
        s = summary.setdefault(r["group"], {"files": 0, "seconds": 0.0})
        s["files"] += 1
        s["seconds"] = round(s["seconds"] + r["duration_s"], 2)

    (OUT / "logs" / "prepare-summary.json").write_text(json.dumps({
        "total_files": len(rows),
        "kept": len(kept),
        "duplicates_removed": len(rows) - len(kept),
        "groups": summary,
    }, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"kept {len(kept)} / {len(rows)} (dupes: {len(rows) - len(kept)})")


if __name__ == "__main__":
    main()
