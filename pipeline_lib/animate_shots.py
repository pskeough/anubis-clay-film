"""Animate a director shot list with MiniMax H3, using ComfyUI's own graph.

    python animate_shots.py <piece_dir> [--only 0,3] [--no-gate]

This file was Wan 2.2, and then it was H3 through a graph assembled from prose
documentation. Both are gone. The graph is now `graphs.minimax_h3_official`,
transcribed node for node from the template ComfyUI ships, and everything below
follows from what that template does rather than from what the docs describe.

WHAT ACTUALLY FIXED THE FROZEN CLIPS. Not the sampler, the step count, the
sigma shift or the LoRA strength - all of which were tuned at length and none
of which mattered. Two things:

  the WEIGHTS   `minimax_h3_fl2va_nvfp4` does not execute camera moves. Same
                graph, same prompt, same seed, only the weights differ:
                int8_convrot moved 4.33 px per 3 frames, nvfp4 moved 0.53 -
                the same as a static control. The subject still animates under
                nvfp4; the camera simply never moves. The turbo LoRA is
                distilled against the full model and under-drives a 4-bit
                repack, and camera motion is what fails first. nvfp4 was only
                ever chosen because 12.5 GB fits one 16 GB card, and that one
                decision produced every frozen clip in this project's history.
                int8_convrot is 21 GB and needs `farm.ps1 start -Big`.

  the PROMPT    the old builder wrote "keeping every subject's appearance,
                clothing, position and the spatial layout unchanged" and asked
                for a move "with small amplitude at slow speed". Three separate
                instructions to hold still. The model obeyed.

THE SCENE NO LONGER LEADS, AND NEITHER DOES THE MOVE. The shipped template
opens with style, names the subject as "<subject> from <Picture 1> in its
original scene:", states that the environment is constant, and then writes
SHOT 1 as prose in which the action and the camera move arrive together. That
is what `h3_prompt.i2va` builds and there is no longer a MOVE_STRENGTH dial:
strength was a Wan-era workaround for a model that would not move, and on H3
turning it down measurably STOPPED the motion.

CUTS. H3 cuts between shots inside one generation. A shot carrying `cuts`
becomes SHOT 2 and SHOT 3 in the same render, so three shots cost one render
and share character, lighting and grade exactly. Only the parent shot is
conditioned on a still.

SPARSE ATTENTION. Sol-Attn at tau 1.3, measured at 305 s against ~465 s dense
for the same clip with no visible change. On by default; `--no-sparse` off.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / "farm"))

import graphs  # noqa: E402
import h3_prompt  # noqa: E402
from dispatch import Farm, Job  # noqa: E402

FPS = 24
# H3's grid is 17k+5, not Wan's 4n+1, and its trained range is ~124-362 frames
# (5.2 s to 15.1 s at 24 fps). 124 is one beat; a shot carrying cuts gets more
# room so each cut has time to read.
MAX_FRAMES = 362
MIN_FRAMES = 124
CUT_SECONDS = 3.4         # extra length per cut block
STEPS = 6                 # the template's turbo path: 8-step LoRA at 6 steps
SEED = 20260917
W, H = 1344, 768          # H3's official 768p, 0.98 MP
SCENE_CHARS = 320

# There is deliberately no negative prompt. The shipped template drives H3
# through BasicGuider, which carries POSITIVE conditioning only - no negative
# input, no cfg field - and MiniMaxH3ImageToVideo emits no negative either.
# The Wan-era string that used to live here was inert in any case: it ran at
# cfg 1.0, where classifier-free guidance is skipped and the unconditional pass
# never happens. Quality is controlled by the weights and the prompt.



def plan_stamp(piece: Path) -> str:
    """Hash of the shot list, so derived artefacts can be checked against it."""
    import hashlib
    f = Path(piece) / "shots.json"
    if not f.exists():
        return ""
    return hashlib.sha1(f.read_bytes()).hexdigest()[:16]


def clean_scene(scene: str) -> str:
    """Trim a schema-truncated string back to a complete clause."""
    s = " ".join(scene.split())
    if not s.endswith("."):
        cut = max(s.rfind(". "), s.rfind(", "), s.rfind("; "))
        if cut > 60:
            s = s[:cut]
    return s.rstrip(" ,;.") + "."


def frames_for(seconds: float, shot: dict | None = None,
               cap: int = MAX_FRAMES) -> int:
    """Seconds -> a frame count on H3's grid, with room for any cuts.

    H3 snaps UP to 17k+5, so asking off-grid silently lengthens the clip and
    desynchronises it from the narration. `graphs.h3_length` does the snapping;
    this decides what to ask for. A shot carrying cuts needs CUT_SECONDS more
    per cut, because three shots crammed into 5.2 s give each one 1.7 s and
    none of them read.
    """
    want = seconds + CUT_SECONDS * len(((shot or {}).get("cuts") or []))
    n = graphs.h3_length(want, FPS, cap=cap)
    return max(MIN_FRAMES, min(cap, n))


# The lock names no body part. The first version said "simplified painted faces and hands, mouths
# closed": H3 read the nouns as content, painted two cartoon men into an empty underwater shot
# (oceanus s036) and cut to a close-up of hands (s038), 2026-10-03.
STYLE_LOCK = ("every frame a painting, never photographic, the brush texture stays the same from the "
              "first frame to the last")
LOCK_PEOPLE = "the figures stay loose painted figures and do not speak"
LOCK_EMPTY = "no people or figures appear at any point"
# No scenery nouns here either: "the water and sky included" sent two interiors out to open sea
# (oceanus s023 galley -> porthole view, s025 tavern -> table in the surf), 2026-10-03.
HOLD = "Every frame stays a hand-painted image in the same strokes as image 1."


def has_people(shot: dict) -> bool:
    p = shot.get("people")
    if isinstance(p, str):
        return p.strip() not in ("", "[]")
    return bool(p)


def end_frame(piece: Path, sid: str) -> Path | None:
    """The shot's painted last keyframe (the endframes stage writes it), if there is one."""
    f = Path(piece) / "endframes" / f"{sid}.png"
    return f if f.exists() else None


PROFILE = "minimax_h3"   # prompt_profiles/<name>/profile.json; the model's own prompting guide as data


def style_bible(piece: Path) -> dict | None:
    """config.json style_bible (style / texture / prohibitions), if the piece has one."""
    f = Path(piece) / "config.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8")).get("style_bible")
    return None


def prompt_for(shot: dict, scene_chars: int = SCENE_CHARS, end: bool = False,
               seconds: float | None = None, bible: dict | None = None, check: bool = True) -> str:
    """The H3 prompt for one shot.

    With a style bible (config.json) the prompt is built by prompt_profile in the model's own
    documented structure: alignment line, blank line, integrated_multimodal_description /
    overall_soundscape / non_diegetic_music, style first in [Shot 1], texture and prohibition
    blocks, camera grammar from the guide. It is linted against the profile's rules and a lint
    failure raises, so a malformed prompt never reaches the GPU. Without a bible the legacy
    free-paragraph template below is used unchanged (older pieces).

    Legacy notes: the shipped template's prompt shape, via h3_prompt.i2va.

    There is no longer a move-strength dial. `MOVE_STRENGTH` existed because
    Wan would not move and the fix was to shout the camera clause louder; on
    H3 the same idea measured backwards - turning the LoRA down to make the
    picture "cleaner" cut motion from 2.35 px to 0.94 and produced a clip with
    nothing happening in it. H3 moves when the weights are right and the prompt
    does not tell it to stop.

    The register tail render_shots.py appends to the STILL prompt is still not
    repeated here: the still already looks the way it looks, and restating
    palette and grain spends tokens the action needs.
    """
    if bible:
        import prompt_profile as pp
        prof = pp.load(PROFILE)
        ctx = {"mode": "fl2va" if end else "i2va", "seconds": seconds or 5.0, "style_bible": bible}
        text = pp.build(prof, shot, ctx)
        if check:
            bad = pp.lint(prof, text, ctx)
            if bad:
                raise ValueError(f"prompt for shot {shot.get('i')} fails {PROFILE} lint: " + "; ".join(bad))
        return text
    scene = clean_scene(shot["scene"])
    # drop a leading framing clause; the still already fixed the framing
    body = scene.split(". ", 1)[-1] if scene.count(". ") else scene
    # A drawn or painted piece sets motion_style; the i2va default ("Live-action cinematic
    # film") would otherwise ask H3 to turn every frame into photography.
    # Style lock (2026-10-03): on a painted piece H3 turned close-ups and newly revealed ground
    # photographic within the clip. The look is restated in the style line and again after the
    # action, and a painted last keyframe (end=True) gives the clip a painting to land on.
    kw = {}
    if shot.get("motion_style"):
        who = LOCK_PEOPLE if has_people(shot) else LOCK_EMPTY
        kw = {"style": f"{shot['motion_style'].rstrip('. ')}; {STYLE_LOCK}; {who}", "hold": HOLD,
              "soundscape": shot.get("soundscape") or "quiet room tone"}
    return h3_prompt.i2va(
        body[:scene_chars],
        shot.get("motion") or "a slow forward push into the frame",
        **kw,
        subject=shot.get("grounded_in") or "the subject",
        action=shot.get("action", ""),
        shots=shot.get("cuts") or None,
        move_target=shot.get("move_target", ""), end_frame=end,
        **({"soundscape": shot.get("soundscape", "")} if "soundscape" not in kw else {}))


def find_ffmpeg() -> str | None:
    """ffmpeg lives in the Podcast venv, and this stage runs on system Python.

    `narrate._ffmpeg_exe()` returns None outside that venv rather than raising,
    which turned the first run of the motion gate into a TypeError instead of a
    skip. Probe the venv's bundled binary directly, then PATH.
    """
    import shutil
    import os
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return shutil.which("ffmpeg")


def gate(clips: Path, ff: str | None = None) -> list[str]:
    """Report clips in which nothing moved. Ported from andermensch/strip.py.

    That file existed because "the only honest check on a video model's output,
    short of watching it, is whether the frame at t=0 and the frame at t=end
    differ in the way the shot asked for". It was dropped in the rewrite and
    the next batch was eleven frozen clips.

    Displacement, not difference: the failed batch scored 4-41 on mean absolute
    difference while being visually identical, because a global exposure shift
    moves every pixel a little. This correlates blocks between the first and
    last frame instead, so a uniform brightness change scores zero.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        print("  (motion gate needs numpy+pillow; skipped)")
        return []
    ff = ff or find_ffmpeg()
    if not ff:
        print("  (no ffmpeg found; motion gate skipped)")
        return []

    tmp = clips / "_gate"
    tmp.mkdir(exist_ok=True)
    frozen = []
    print("\n  motion gate (block displacement, first frame vs last):")
    for mp4 in sorted(clips.glob("s*.mp4")):
        a, b = tmp / f"{mp4.stem}_a.png", tmp / f"{mp4.stem}_b.png"
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(mp4),
                        "-vf", "select=eq(n\\,0),scale=320:-1", "-frames:v", "1",
                        "-update", "1", str(a)], check=False)
        subprocess.run([ff, "-y", "-loglevel", "error", "-sseof", "-0.2",
                        "-i", str(mp4), "-vf", "scale=320:-1", "-frames:v", "1",
                        "-update", "1", str(b)], check=False)
        if not (a.exists() and b.exists()):
            continue
        A = np.asarray(Image.open(a).convert("L"), dtype=np.float32)
        B = np.asarray(Image.open(b).convert("L"), dtype=np.float32)
        A -= A.mean(); B -= B.mean()          # kill the exposure component
        h, w = A.shape
        best = []
        for y in range(0, h - 48, 48):
            for x in range(0, w - 48, 48):
                pa = A[y:y + 48, x:x + 48]
                ref = pa - pa.mean()
                rn = float(np.sqrt((ref ** 2).sum())) or 1.0
                if rn < 12.0:           # flat block; it cannot report motion
                    continue
                sc = []
                for dy in (-8, -4, 0, 4, 8):
                    for dx in (-8, -4, 0, 4, 8):
                        yy, xx = y + dy, x + dx
                        if yy < 0 or xx < 0 or yy + 48 > h or xx + 48 > w:
                            continue
                        pb = B[yy:yy + 48, xx:xx + 48] 
                        q = pb - pb.mean()
                        qn = float(np.sqrt((q ** 2).sum())) or 1.0
                        sc.append((float((ref * q).sum()) / (rn * qn),
                                   abs(dy) + abs(dx)))
                if sc:
                    best.append(max(sc)[1])
        disp = float(np.mean(best)) if best else 0.0
        flag = "  FROZEN" if disp < 1.5 else ""
        print(f"    {mp4.stem}  displacement {disp:5.2f} px{flag}")
        if disp < 1.5:
            frozen.append(mp4.stem)
        a.unlink(missing_ok=True); b.unlink(missing_ok=True)
    tmp.rmdir()
    if frozen:
        print(f"    {len(frozen)} clip(s) did not move: {', '.join(frozen)}")
    return frozen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("piece", type=Path)
    ap.add_argument("--only", default="")
    # 1344x768 is H3's official 768p. NOT 1280x704: MiniMaxH3ImageToVideo
    # resizes the still with crop="disabled", a plain stretch, so a canvas
    # whose aspect differs from the still widens every face in it. 1344x768 is
    # 1.75, which is what render_shots.py writes.
    ap.add_argument("--width", type=int, default=W)
    ap.add_argument("--height", type=int, default=H)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--no-end", action="store_true", help="ignore endframes/ (first-frame-only H3)")
    ap.add_argument("--scene-chars", type=int, default=SCENE_CHARS)
    # Sol-Attn. 305 s against ~465 s dense on the same clip, motion and detail
    # unchanged. tau is in score-distribution sigmas and higher is sparser;
    # 1.3 is H3-tuned, and reports of duplicated subjects start around 1.8.
    ap.add_argument("--no-sparse", action="store_true")
    ap.add_argument("--tau", type=float, default=1.3)
    a = ap.parse_args()

    piece = a.piece.resolve()
    shots = json.loads((piece / "shots.json").read_text(encoding="utf-8"))
    # timeline.json is written by the narration stage and carries the real
    # per-shot durations. Without it this used to raise FileNotFoundError and
    # take the whole clips stage with it - which means a TTS failure, or simply
    # running the stages out of order, costs every clip in the piece. Fall back
    # to the piece's own beat_seconds instead: the clips are then uniform and
    # the assembler re-times them against the audio anyway.
    tl_path = piece / "timeline.json"
    if tl_path.exists():
        tl = json.loads(tl_path.read_text(encoding="utf-8"))
        dur = {s["index"]: s["duration"] for s in tl["scenes"]}
    else:
        cfg = json.loads((piece / "config.json").read_text(encoding="utf-8"))
        fallback = float(cfg.get("beat_seconds") or 5.0)
        dur = {s["i"]: fallback for s in shots}
        print(f"  no timeline.json; using beat_seconds={fallback:.2f}s for all "
              f"{len(shots)} shots (narration will re-time on assemble)")

    stills, clips = piece / "stills", piece / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    only = {int(x) for x in a.only.split(",") if x.strip()} if a.only else None

    jobs, total_frames = [], 0
    for s in shots:
        i = s["i"]
        if only is not None and i not in only:
            continue
        sid = f"s{i:03d}"
        still = stills / f"{sid}.png"
        if not still.exists():
            print(f"  {sid}: no still, skipping")
            continue
        if (clips / f"{sid}.mp4").exists():
            continue
        n = frames_for(dur.get(i, 5.0), s, a.max_frames)
        total_frames += n
        cuts = s.get("cuts") or []
        # a painted last keyframe pins the clip's end; its name must differ from the still's
        # because both go to the worker's input dir
        end = None if (a.no_end or cuts) else end_frame(piece, sid)
        inputs = [still]
        if end:
            last = piece / "endframes" / f"{sid}_end.png"
            if not last.exists() or last.stat().st_mtime < end.stat().st_mtime:
                import shutil
                shutil.copy2(end, last)
            inputs.append(last)
        prompt = s.get("motion_prompt") or prompt_for(
            s, a.scene_chars, end=bool(end), seconds=n / FPS, bible=style_bible(piece))
        jobs.append(Job(
            job_id=sid, model_key="minimax_h3", out_name=sid, inputs=inputs,
            graph=graphs.minimax_h3_official(
                prompt, still.name, width=a.width, height=a.height,
                seconds=n / FPS, fps=FPS, steps=a.steps, seed=SEED + i,
                sparse=None if a.no_sparse else {"tau": a.tau,
                                                 "start_percent": 0.2},
                last_image=inputs[1].name if end else None,
                prefix=f"{piece.name}_{sid}"),
            meta={"frames": n, "seconds": round(n / FPS, 2),
                  "want": dur.get(i), "motion": s.get("motion"),
                  "action": s.get("action"), "cuts": len(cuts),
                  "end_frame": bool(end), "prompt": prompt}))

    shots_total = len(jobs) + sum(j.meta["cuts"] for j in jobs)
    print(f"\n=== {piece.name}: {len(jobs)} renders carrying {shots_total} "
          f"shots, {a.width}x{a.height}, {a.steps} steps, "
          f"{'dense' if a.no_sparse else f'sol-attn tau {a.tau}'}, "
          f"{total_frames} frames ({total_frames / FPS:.1f} s of video) ===")
    for j in jobs[:3]:
        print(f"    {j.job_id} ({j.meta['cuts']} cuts)  "
              f"{j.meta['prompt'][:120]}...")
    if jobs:
        # The big worker, not the pinned pair. int8_convrot is 21 GB and only
        # a process started with --cuda-device all can load it; the pair see
        # one card each. `farm.ps1 start -Big`.
        import config as C
        Farm(clips, workers=[C.BIG_WORKER], job_timeout=5400).run(jobs)

    for f in clips.glob("*_?.mp4"):
        want = clips / (re.sub(r"_\d+$", "", f.stem) + ".mp4")
        if not want.exists():
            f.replace(want)
    for junk in list(clips.glob("*_?.png")) + list(clips.glob("*.webp")):
        junk.unlink()

    made = sorted(p.name for p in clips.glob("s*.mp4"))
    (piece / "tiers.json").write_text(json.dumps(
        {f"s{s['i']:03d}": ("T3" if f"s{s['i']:03d}.mp4" in made else "T1")
         for s in shots}, indent=2), encoding="utf-8")
    (clips / "plan.stamp").write_text(plan_stamp(piece), encoding="utf-8")
    print(f"  {len(made)}/{len(shots)} clips on disk")

    if not a.no_gate and made:
        frozen = gate(clips)
        if frozen:
            print(f"\n  Re-run those with a stronger move or a lower --cfg "
                  f"before assembling.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
