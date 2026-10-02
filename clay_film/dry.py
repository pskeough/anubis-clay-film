"""A fake GPU for testing the studio and the per-shot operations without the farm.

CLAY_FILM_DRY=1 (or `webui/server.py --dry`) routes every GPU job here. Stills are fixture
stills from a finished piece with a label drawn on them (an edit draws on the image it was
given), clips are fixture clips, narration is silence timed at ~2.4 words a second. Each takes
a few seconds so the queue can be watched. Clay, compile and assembly are not faked.

    CLAY_FILM_DRY_FIXTURE   piece to copy from (default pieces/basilisk_day2_v2, read only).
                            Without one, stills are flat placeholders; dry clips still need it.
    CLAY_FILM_DRY_SCALE     multiplies the delays (default 1; 0 for tests)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from . import paths, schema

FIXTURE = Path(os.environ.get("CLAY_FILM_DRY_FIXTURE") or paths.PIECES / "basilisk_day2_v2")
SCALE = float(os.environ.get("CLAY_FILM_DRY_SCALE") or 1.0)
DELAYS = {"clay": 4.0, "free": 5.0, "edit": 3.0, "redo": 4.0, "motion": 12.0, "narrate": 4.0, "farm": 3.0}
_state = {"farm": "off"}


def _wait(kind: str, log=None) -> None:
    t = DELAYS.get(kind, 3.0) * SCALE
    if log and t:
        log(f"[dry] {kind}: {t:.0f} s")
    time.sleep(t)


def farm(mode: str, log=None) -> None:
    if _state["farm"] != mode:
        if log:
            log(f"[dry] farm {_state['farm']} -> {mode}")
        _wait("farm")
        _state["farm"] = mode


def farm_mode() -> str:
    return _state["farm"]


def _fixture(folder: str, ext: str, i: int) -> Path:
    files = sorted((FIXTURE / folder).glob(f"s*{ext}"))
    if not files:
        raise FileNotFoundError(f"dry fixture has no {folder}/*{ext}: {FIXTURE}")
    return files[i % len(files)]


def still(kind: str, out: Path, i: int, seed: int, prompt: str, src: Path | None = None, log=None) -> Path:
    from PIL import Image, ImageDraw, ImageOps
    _wait(kind, log)
    w, h = 1344, 768
    try:
        im = Image.open(src or _fixture("stills", ".png", i + (seed or 0))).convert("RGB")
    except FileNotFoundError:
        im = Image.new("RGB", (w, h), (90 + 20 * (i % 4), 90, 100))   # no fixture piece: a flat placeholder
    if im.size != (w, h):
        im = im.resize((w, h))
    if kind == "free":
        im = ImageOps.posterize(ImageOps.autocontrast(im), 3)
    elif kind == "edit":
        im = ImageOps.colorize(ImageOps.grayscale(im), (30, 20, 10), (240, 220, 190))
    d = ImageDraw.Draw(im)
    label = f"DRY {kind}  seed {seed}"
    text = " ".join((prompt or "").split())[:110]
    d.rectangle([0, 0, w, 58], fill=(0, 0, 0))
    d.text((12, 8), label, fill=(255, 210, 60))
    d.text((12, 30), text, fill=(220, 220, 220))
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    return out


def clip(out: Path, i: int, log=None) -> Path:
    _wait("motion", log)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_fixture("clips", ".mp4", i), out)
    return out


def _ffmpeg() -> str | None:
    cand = paths.PODCAST_PY.parent.parent / "Lib" / "site-packages" / "imageio_ffmpeg" / "binaries"
    if cand.is_dir():
        for exe in sorted(cand.glob("ffmpeg-win*.exe"), reverse=True):
            return str(exe)
    return shutil.which("ffmpeg")


def narrate(piece: Path, log=None) -> None:
    """timeline.json at ~2.4 words a second and a silent narration.mp3 of that length."""
    _wait("narrate", log)
    beats = json.loads((piece / "beats.json").read_text(encoding="utf-8"))
    scenes, t = [], 0.6
    for i, b in enumerate(beats):
        d = max(schema.est_seconds(b), 1.5) + 0.3
        scenes.append({"index": i, "id": f"s{i:03d}", "beat": i, "narration": b,
                       "speech_start": round(t, 3), "speech_end": round(t + d - 0.3, 3),
                       "shot_start": round(0.0 if i == 0 else t, 3), "shot_end": round(t + d, 3),
                       "duration": round(d + (0.6 if i == 0 else 0.0), 3)})
        t += d
    total = round(t + 0.8, 3)
    scenes[-1]["shot_end"] = total
    scenes[-1]["duration"] = round(total - scenes[-1]["shot_start"], 3)
    (piece / "timeline.json").write_text(json.dumps(
        {"title": piece.name, "engine": "dry", "take": "single", "audio": "narration.mp3",
         "sample_rate": 24000, "duration": total, "scenes": scenes}, indent=2), encoding="utf-8")
    ff = _ffmpeg()
    if ff:
        subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "anullsrc=r=24000:cl=mono", "-t", f"{total:.3f}", "-q:a", "6",
                        str(piece / "narration.mp3")], check=False)
    if log:
        log(f"[dry] timeline {total:.1f} s, {len(scenes)} shots")
