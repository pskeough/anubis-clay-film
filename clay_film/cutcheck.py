"""Finds clips where H3 invented a cut inside a single shot.

A cut is one frame-to-frame luma jump many times the clip's median jump. Smooth camera or
subject motion raises the median along with the peak, so it does not trip the test. Calibrated
on Anubis 2026-09-24: both bad s005 renders (a cut to a room of strangers, and a cut in to a lap
close-up) peaked at 53 with the peak 9-10x the median; the six clean clips peaked at 1.3-11 and
at most 2.6x. A flagged clip usually means the motion prompt asked for something the still
cannot show: people or objects that are not in it, or hand-level action on a small figure.
"""
from __future__ import annotations

from pathlib import Path

MIN_JUMP = 12.0     # mean absolute luma difference (0-255) at 160x90
MIN_RATIO = 6.0     # peak jump over the clip's median jump


def jumps(clip: Path):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(clip))
    prev, out = None, []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype(float)
        if prev is not None:
            out.append(float(np.abs(g - prev).mean()))
        prev = g
    cap.release()
    return out


def check(clip: Path) -> dict:
    d = jumps(clip)
    if len(d) < 3:
        return {"clip": clip.stem, "cut": False, "frames": len(d) + 1}
    k = max(range(len(d)), key=d.__getitem__)
    med = sorted(d)[len(d) // 2]
    ratio = d[k] / max(med, 0.3)
    return {"clip": clip.stem, "cut": d[k] > MIN_JUMP and ratio > MIN_RATIO,
            "frame": k + 1, "jump": round(d[k], 1), "ratio": round(ratio, 1)}


def scan(piece: Path, only: set | None = None) -> list[dict]:
    """Every clip in piece/clips (or those in `only`), with the suspected cuts marked."""
    return [check(c) for c in sorted((Path(piece) / "clips").glob("s*.mp4"))
            if not only or c.stem in only]
