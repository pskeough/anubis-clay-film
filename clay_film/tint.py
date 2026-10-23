"""Tint each person's mannequin in the people clay the colour of their outer clothes.

The mannequins are what places people (stages.stills edits <sid>_people.png). With all of them the
same grey, the model matched people to mannequins by the prompt's words alone and swapped the woman
and the narrator, and doubled a dealer seen from behind (gardening p13 s000 and s005, 2026-10-11).
A green mannequin for the woman in the bottle-green dress is a binding it can see.

    tint_people(blockout, sid, people_descs) -> path of <sid>_people_tint.png, or None
"""
from __future__ import annotations

import re
from pathlib import Path

# earliest match inside the clothing clause wins; specific words before general ones
COLOURS = [
    (r"bottle-green|olive|green", (70, 105, 70)),
    (r"navy", (45, 55, 90)),
    (r"camel|tan\b|khaki|beige|sand", (175, 140, 95)),
    (r"mustard|ochre|yellow", (190, 155, 60)),
    (r"teal", (50, 110, 110)),
    (r"pink", (200, 150, 160)),
    (r"red|crimson|maroon|burgundy", (140, 50, 45)),
    (r"blue", (60, 80, 130)),
    (r"brown|leather", (115, 85, 60)),
    (r"black|charcoal", (45, 45, 48)),
    (r"white|cream|ivory|pale", (215, 210, 195)),
    (r"grey|gray|silver", (130, 130, 130)),
]


def clothes_colour(desc: str):
    m = re.search(r"\b(in|wearing) (a |an |the )?", desc)
    part = desc[m.start():] if m else desc
    best = None
    for pat, rgb in COLOURS:
        for hit in re.finditer(rf"\b(?:{pat})", part, re.I):
            if best is None or hit.start() < best[0]:
                best = (hit.start(), rgb)
            break
    return best[1] if best else None


def tint_people(blk: Path, sid: str, descs: list[str], shown: list[int] | None = None) -> Path | None:
    from PIL import Image
    import numpy as np
    src = blk / f"{sid}_people.png"
    if not src.exists():
        return None
    im = np.asarray(Image.open(src).convert("RGB"), float)
    lum = im.mean(axis=2, keepdims=True)
    out = im.copy()
    done = False
    for k, d in enumerate(descs):
        mp = blk / f"{sid}_fig{k}.png"
        rgb = clothes_colour(d)
        if not mp.exists() or rgb is None or (shown is not None and k not in shown):
            continue
        m = np.asarray(Image.open(mp).convert("L").resize((im.shape[1], im.shape[0])), float)[..., None] / 255
        col = lum / 160.0 * np.array(rgb, float)
        out = out * (1 - m) + np.clip(col, 0, 255) * m
        done = True
    if not done:
        return None
    dst = blk / f"{sid}_people_tint.png"
    Image.fromarray(out.astype("uint8")).save(dst)
    return dst
