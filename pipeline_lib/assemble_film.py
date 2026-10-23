"""Stage 5 for director pieces - grade, letterbox, dissolve, mux.

    <podcast venv python> assemble_film.py <piece_dir> [--dissolve 0.5] [--hard]

Differs from assemble.py in three ways, each for a reason
---------------------------------------------------------
1. Dissolves, not cuts. assemble.py stream-copies a concat list, which is a
   hard cut between every shot. On a literal shot list that is correct: the
   picture changes because the sentence changed. This shot list is associative
   rather than literal - a groove, a hand, gold lifting off a surface - and
   eleven hard cuts between eleven unrelated surfaces is what "disjointed shot
   to shot" describes. A dissolve is the edit grammar for images that argue
   with each other instead of following each other.

   Sync is preserved exactly: every segment except the last is encoded
   `want + D` long, and the xfade offsets are the unmodified cumulative `want`
   times, so each shot still STARTS on its own narration and the film is
   exactly as long as the timeline says.

2. The grade comes from the treatment, not from a per-shot key. The director
   writes a register once for the whole piece; assemble.py's `shot["grade"]`
   field belongs to the old planner and does not exist here. Movement is the
   only thing that shifts the grade, per config["grades"].

3. No upscale. Wan renders 1280x704 and the delivery is 1280x544, so the
   letterbox is a crop of native pixels. assemble.py scaled everything to
   1280x720 first, which on 832x480 clips was a 1.54x blow-up.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import os  # noqa: E402
sys.path.insert(0, os.environ.get("CLAY_FILM_TTS_ENGINE") or str(Path(__file__).resolve().parent.parent / "tts_engine"))
import narrate  # noqa: E402

W, LB, FPS = 1280, 544, 24
OVER = 1.14                      # headroom a still's crop window travels inside

# From treatment.register.grade: "cold, low contrast in the shadows, slight
# lift in the blacks so the stone reads as stone and not as void; gold is the
# only saturated note and it is always brief".
BASE_GRADE = (
    "curves=all='0/0.045 0.25/0.26 0.75/0.74 1/0.965',"
    "eq=saturation=0.74:contrast=1.03:gamma=0.99"
)
TINTS = {
    # cold shadows, neutral highs; the default state of the room
    "ash": "colorbalance=rs=-0.05:gs=-0.015:bs=0.065:rm=-0.02:bm=0.03",
    # the same room conceding a little warmth, for the last movement
    "ochre": "colorbalance=rs=0.035:gs=0.012:bs=-0.045:rm=0.02:bm=-0.02",
    "neutral": "colorbalance=rs=0:gs=0:bs=0",
    # flat-colour and painted pieces: BASE_GRADE's desaturation greys out a drawn palette,
    # so "vivid" replaces the base grade rather than tinting on top of it
    "vivid": "colorbalance=rs=0:gs=0:bs=0",
}
VIVID_BASE = "eq=saturation=1.06:contrast=1.02"
FINISH = f"crop={W}:{LB}:0:(ih-{LB})/2,noise=alls=4:allf=t+u,vignette=PI/6"

# Applied to Wan clips only, and only because the loss was measured. Laplacian
# variance of a clip's first frame is 50 +/- 4% of the still it was
# conditioned on, on every one of eleven shots - the conditioning image is
# VAE-encoded and decoded before anything is sampled, and that round-trip is
# where the softness comes from. Stills never go through it, so they do not get
# this and would only get crunchy.
# 5:5:0.45 measured back to 96-104% of the source still's Laplacian
# variance on four shots spanning a 5x range of source detail. 7:7:0.9
# overshot to 170%, which is halo, not detail.
UNSHARP = "unsharp=5:5:0.45:5:5:0.0"


def ff() -> str:
    return narrate._ffmpeg_exe()


def run(args: list[str], cwd: Path | None = None) -> None:
    r = subprocess.run(args, capture_output=True, text=True,
                       cwd=str(cwd) if cwd else None)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{' '.join(args[:16])}\n"
                           f"{r.stderr[-2500:]}")


def duration(path: Path) -> float:
    r = subprocess.run([ff(), "-i", str(path), "-hide_banner"],
                       capture_output=True, text=True)
    for line in r.stderr.splitlines():
        if "Duration:" in line:
            h, m, s = line.split("Duration:")[1].split(",")[0].strip().split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise RuntimeError(f"no duration for {path}")


def still_move(shot: dict, secs: float, src_w: int, src_h: int) -> str:
    """A travelling crop over a still, for any shot Wan did not cover.

    Direction is keyed on the shot's own index and subject so that two held
    frames in a row never drift the same way; one uniform move applied to every
    still is what read as "this weird moving base animation" on an earlier cut.
    """
    big_w, big_h = int(W * OVER), int(src_h * OVER)
    dx, dy = big_w - W, big_h - src_h
    t = f"(t/{max(secs, 0.1):.3f})"
    zoom = f"scale={big_w}:{big_h}:flags=lanczos"
    lane = shot["i"] % 4
    if lane == 0:
        x, y = f"{dx}*{t}", f"{dy}/2"
    elif lane == 1:
        x, y = f"{dx}*(1-{t})", f"{dy}/2"
    elif lane == 2:
        x, y = f"{dx}/2", f"{dy}*{t}"
    else:
        x, y = f"{dx}/2*(1+{t}/2)", f"{dy}*(1-{t})"
    return f"{zoom},crop={W}:{src_h}:{x}:{y}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("piece", type=Path)
    ap.add_argument("--dissolve", type=float, default=0.5)
    ap.add_argument("--hard", action="store_true", help="cuts instead of dissolves")
    ap.add_argument("--unsharp", type=float, default=0.45,
                    help="0 disables; 0.45 measured back to source detail")
    a = ap.parse_args()
    piece = a.piece.resolve()
    D = 0.0 if a.hard else a.dissolve

    cfg = json.loads((piece / "config.json").read_text(encoding="utf-8"))
    shots = json.loads((piece / "shots.json").read_text(encoding="utf-8"))
    tl = json.loads((piece / "timeline.json").read_text(encoding="utf-8"))
    treatment = json.loads((piece / "treatment.json").read_text(encoding="utf-8"))
    by_i = {s["i"]: s for s in shots}

    movements = [m["name"] for m in treatment["movements"]]
    grades = cfg.get("grades") or ["ash"] * len(movements)
    name_of = {m: grades[k] if k < len(grades) else "ash" for k, m in enumerate(movements)}
    tint_of = {m: TINTS.get(g, TINTS["ash"]) for m, g in name_of.items()}

    work = piece / "_assemble"
    work.mkdir(parents=True, exist_ok=True)
    for old in work.glob("*.mp4"):
        old.unlink()

    scenes = sorted(tl["scenes"], key=lambda s: s["index"])
    segs, wants, missing = [], [], []
    n_clip = n_still = 0

    print(f"\n=== {cfg['title']}: {len(scenes)} shots, "
          f"{'cuts' if a.hard else f'{D:.2f}s dissolves'} ===")
    for k, sc in enumerate(scenes):
        i = sc["index"]
        sid, shot = f"s{i:03d}", by_i[i]
        want = sc["duration"]
        # every segment but the last carries D extra, which the next xfade eats
        length = want + (D if k < len(scenes) - 1 else 0.0)
        base = VIVID_BASE if name_of.get(shot.get("movement")) == "vivid" else BASE_GRADE
        grade = f"{base},{tint_of.get(shot.get('movement'), TINTS['ash'])}"
        dst = work / f"{sid}.mp4"
        clip, still = piece / "clips" / f"{sid}.mp4", piece / "stills" / f"{sid}.png"

        # _assemble_keep/sNNN.mp4: an already graded, slot-length segment from an earlier assembly,
        # used as-is. Recovery path for clips whose raw file was lost (oceanus 2026-10-03); it
        # is only valid while the timeline is unchanged, so the length is checked.
        keep = piece / "_assemble_keep" / f"{sid}.mp4"
        if keep.exists() and abs(duration(keep) - length) < 0.1:
            shutil.copy2(keep, dst)
            n_clip += 1
            segs.append(dst)
            wants.append(want)
            print(f"  {sid} {want:5.2f}s  kept segment")
            continue
        if clip.exists():
            have = duration(clip)
            sharp = (f"unsharp=5:5:{a.unsharp:.2f}:5:5:0.0,"
                     if a.unsharp > 0 else "")
            vf = (f"setpts={length / have:.6f}*PTS,{sharp}{grade},"
                  f"{FINISH}")
            run([ff(), "-y", "-loglevel", "error", "-i", str(clip),
                 "-vf", vf, "-r", str(FPS), "-an", "-c:v", "libx264",
                 "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
                 str(dst)])
            n_clip += 1
            note = f"clip x{length / have:.2f}"
        elif still.exists():
            vf = (f"{still_move(shot, length, W, 720)},{grade},{FINISH}")
            run([ff(), "-y", "-loglevel", "error", "-loop", "1",
                 "-t", f"{length:.3f}", "-i", str(still), "-vf", vf,
                 "-r", str(FPS), "-an", "-c:v", "libx264", "-preset", "medium",
                 "-crf", "17", "-pix_fmt", "yuv420p", str(dst)])
            n_still += 1
            note = "still"
        else:
            missing.append(sid)
            continue
        segs.append(dst)
        wants.append(want)
        print(f"  {sid} {want:5.2f}s  {note:<12} {shot['relation']}/"
              f"{shot['subject_class']}")

    if not segs:
        print("  nothing to assemble")
        return 1

    silent = work / "_silent.mp4"
    total = sum(wants)
    if D <= 0 or len(segs) == 1:
        (work / "concat.txt").write_text(
            "".join(f"file '{p.name}'\n" for p in segs), encoding="utf-8")
        run([ff(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", "concat.txt", "-c", "copy", silent.name], cwd=work)
    else:
        args = [ff(), "-y", "-loglevel", "error"]
        for p in segs:
            args += ["-i", str(p)]
        chain, prev, off = [], "0:v", 0.0
        for k in range(1, len(segs)):
            off += wants[k - 1]
            label = f"v{k}"
            chain.append(f"[{prev}][{k}:v]xfade=transition=fade:"
                         f"duration={D:.3f}:offset={off:.3f}[{label}]")
            prev = label
        # the audio already leads with silence and ends on a tail, so the
        # picture fades match the narration's own margins
        chain.append(f"[{prev}]fade=t=in:st=0:d=0.8,"
                     f"fade=t=out:st={max(total - 1.4, 0):.3f}:d=1.4[vo]")
        args += ["-filter_complex", ";".join(chain), "-map", "[vo]",
                 "-r", str(FPS), "-c:v", "libx264", "-preset", "medium",
                 "-crf", "17", "-pix_fmt", "yuv420p", str(silent)]
        run(args)

    out = piece / f"{piece.name}.mp4"
    run([ff(), "-y", "-loglevel", "error", "-i", str(silent),
         "-i", str(piece / "narration.mp3"), "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
         "-ac", "2", "-shortest", "-movflags", "+faststart", str(out)])

    got = duration(out)
    last = max(s["speech_end"] for s in scenes)
    print(f"\n  {out.name}  {got:.1f}s  {out.stat().st_size / 1024**2:.1f} MB  "
          f"{W}x{LB}  {n_clip} generated / {n_still} held")
    print(f"  timeline {total:.1f}s, last word {last:.1f}s, "
          f"headroom {got - last:.2f}s")
    if missing:
        print(f"  INCOMPLETE: missing {', '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
