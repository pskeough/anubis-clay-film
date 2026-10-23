"""Stretch or trim a clip to an exact frame count, and optionally dissolve into it from the previous shot or fade its
end to black.

A clip longer than the target loses its tail. A clip shorter than the target is slowed evenly from its first frame
to its last, with the new in-between frames drawn by RIFE; the slow-down is capped (--maxslow) and any frames still
missing are held on the last frame. Frame blending smears hands and faces above about x1.3, so RIFE is the default;
--interp blend is a fallback for machines without the RIFE weights.

    python retime.py clip SRC.mp4 OUT.mp4 N [--start F] [--end F] [--xfrom PREV.mp4 --xfade 12] [--black 72]
    python retime.py job JOB.json        a sequence of clips, each dissolving from the one before, optionally joined

RIFE needs torch, the IFNet definition from ComfyUI-Frame-Interpolation (RIFE_ARCH_DIR, the folder holding
rife_arch.py) and the 4.26 weights (RIFE_WEIGHTS, rife426.pth). CPU by default, capped at --threads cores.
"""
import argparse
import math
import os
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from _shared import cap_cpu, count_frames, load_job, probe, read_frames, resolve, run_ff, write_video  # noqa: E402

MAXSLOW = 2.2     # above this the motion reads as slow motion; hold the last frame instead


class Rife:
    """RIFE 4.x frame interpolation through ComfyUI-Frame-Interpolation's IFNet, without ComfyUI itself."""

    def __init__(self, device="cpu", threads=4):
        import torch
        import torch.nn.functional as F
        self.torch, self.F = torch, F
        arch_dir = os.environ.get("RIFE_ARCH_DIR")
        weights = os.environ.get("RIFE_WEIGHTS")
        if not arch_dir or not (Path(arch_dir) / "rife_arch.py").exists():
            raise SystemExit("set RIFE_ARCH_DIR to the folder holding rife_arch.py "
                             "(ComfyUI-Frame-Interpolation/vfi_models/rife), or use --interp blend")
        if not weights or not Path(weights).exists():
            raise SystemExit("set RIFE_WEIGHTS to the RIFE weights file (rife426.pth), or use --interp blend")
        torch.set_num_threads(threads)
        self.dev = torch.device(device)
        self.half = self.dev.type == "cuda"
        # rife_arch imports comfy.model_management only to ask for the device
        mm = types.ModuleType("comfy.model_management")
        mm.get_torch_device = lambda: self.dev
        sys.modules.setdefault("comfy", types.ModuleType("comfy"))
        sys.modules["comfy.model_management"] = mm
        sys.path.insert(0, str(arch_dir))
        from rife_arch import IFNet
        model = IFNet(arch_ver=os.environ.get("RIFE_ARCH_VER", "4.26"))
        model.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"), strict=False)
        self.model = (model.half() if self.half else model.float()).eval().to(self.dev)
        self.scales = [16, 8, 4, 2, 1]

    def _tensor(self, f):
        x = self.torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1)[None].to(self.dev)
        x = x.half() / 255 if self.half else x.float() / 255
        h, w = f.shape[:2]
        return self.F.pad(x, (0, (64 - w % 64) % 64, 0, (64 - h % 64) % 64), mode="replicate")

    def __call__(self, a, b, t):
        h, w = a.shape[:2]
        ts = self.torch.full((1, 1, 1, 1), t, device=self.dev)
        ts = ts.half() if self.half else ts
        with self.torch.inference_mode():
            m = self.model(self._tensor(a), self._tensor(b), ts, self.scales, True, False)
        return (m[0, :, :h, :w].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)


def blend(a, b, t):
    return (a.astype(np.float32) * (1 - t) + b.astype(np.float32) * t).round().astype(np.uint8)


def retime(frames, n, maxslow=MAXSLOW, interp=blend):
    """frames -> exactly n frames. Returns (frames, slow factor, held frames)."""
    n_in = len(frames)
    if n <= n_in:
        return list(frames[:n]), 1.0, 0
    if n_in == 1:
        return [frames[0]] * n, 1.0, n - 1
    slow = min(n / n_in, maxslow)
    n_move = max(2, round(n_in * slow))
    out = []
    for k in range(n_move):
        s = k * (n_in - 1) / (n_move - 1)
        i = min(int(math.floor(s)), n_in - 2)
        f = s - i
        # positions within 2 % of a source frame take that frame; RIFE is not asked for near-copies
        out.append(frames[i] if f < 0.02 else frames[i + 1] if f > 0.98 else interp(frames[i], frames[i + 1], f))
    hold = n - n_move
    return out + [out[-1]] * hold, slow, hold


def dissolve_from(frames, last, k):
    """Dissolve from a still frame (the previous shot's last frame) into the first k frames."""
    last = last.astype(np.float32)
    for j in range(min(k, len(frames))):
        a = (j + 1) / (k + 1)
        frames[j] = (last * (1 - a) + frames[j].astype(np.float32) * a).round().astype(np.uint8)
    return frames


def fade_to_black(frames, n):
    """The last n frames fade down; the final frame is black."""
    n = min(n, len(frames))
    for k in range(n):
        j = len(frames) - n + k
        frames[j] = (frames[j].astype(np.float32) * (1 - (k + 1) / n)).astype(np.uint8)
    return frames


def build_clip(src, out, n, start=0, end=None, xfrom=None, xfade=0, black=0, maxslow=MAXSLOW, interp=blend,
               fps=None, crf=16, threads=4, prev_frame=None):
    t0 = time.time()
    w, h, src_fps, _ = probe(src)
    fps = fps or src_fps or 24
    frames = read_frames(src, pix_fmt="rgb24")[start:end]
    if not len(frames):
        raise SystemExit(f"{src}: no frames in [{start}:{end}]")
    n_in = len(frames)
    main, slow, hold = retime(frames, n, maxslow, interp)
    main = [np.array(f) for f in main]
    if xfade:
        prev = prev_frame if prev_frame is not None else read_frames(xfrom, pix_fmt="rgb24")[-1]
        if prev.shape != main[0].shape:
            raise SystemExit(f"{xfrom}: frame size {prev.shape[1]}x{prev.shape[0]} differs from {w}x{h}")
        main = dissolve_from(main, prev, xfade)
    if black:
        main = fade_to_black(main, black)
    assert len(main) == n, (len(main), n)
    write_video(main, out, fps, crf=crf, pix_fmt="rgb24", threads=threads)
    print(f"{Path(src).name}[{start}:{end}] {n_in}f -> {n}f x{slow:.2f} hold {hold}f"
          f"{f' xfade {xfade}' if xfade else ''}{f' black {black}' if black else ''} {w}x{h} "
          f"in {time.time() - t0:.0f}s -> {out}", flush=True)
    return main[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interp", choices=["rife", "blend"], default="rife",
                    help="how in-between frames are made (default rife)")
    ap.add_argument("--device", default="cpu", help="torch device for RIFE: cpu (default) or cuda:0")
    ap.add_argument("--threads", type=int, default=4, help="CPU cores to use (default 4)")
    ap.add_argument("--maxslow", type=float, default=MAXSLOW, help=f"largest slow-down (default {MAXSLOW})")
    ap.add_argument("--fps", type=float, default=None, help="output frame rate (default: the source's)")
    ap.add_argument("--crf", type=int, default=16, help="x264 quality of the output (default 16)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clip", help="retime one clip")
    c.add_argument("src")
    c.add_argument("out")
    c.add_argument("n", type=int, help="target frame count")
    c.add_argument("--start", type=int, default=0, help="first source frame to use")
    c.add_argument("--end", type=int, default=None, help="source frame to stop before")
    c.add_argument("--xfrom", help="previous clip: dissolve from its last frame")
    c.add_argument("--xfade", type=int, default=12, help="dissolve length in frames when --xfrom is given (default 12)")
    c.add_argument("--black", type=int, default=0, help="fade the last N frames to black")
    j = sub.add_parser("job", help="retime a sequence of clips from a JSON job file (schema in README.md)")
    j.add_argument("job")
    a = ap.parse_args()

    cap_cpu(a.threads)
    interp = Rife(a.device, a.threads) if a.interp == "rife" else blend
    common = dict(maxslow=a.maxslow, interp=interp, fps=a.fps, crf=a.crf, threads=a.threads)
    if a.cmd == "clip":
        build_clip(a.src, a.out, a.n, a.start, a.end, a.xfrom, a.xfade if a.xfrom else 0, a.black, **common)
        return
    job, base = load_job(a.job)
    out_dir = resolve(base, job.get("out_dir", "retimed"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if job.get("fps"):
        common["fps"] = job["fps"]
    common["maxslow"] = job.get("maxslow", common["maxslow"])
    outs, prev = [], None
    for k, cl in enumerate(job["clips"]):
        src = resolve(base, cl["src"])
        out = out_dir / cl.get("out", f"{k:03d}_{src.stem}.mp4")
        xf = cl.get("xfade", 0)
        if xf and prev is None and not cl.get("xfrom"):
            raise SystemExit(f"clip {k}: xfade needs a clip before it or an explicit xfrom")
        pf = None if cl.get("xfrom") else prev
        prev = build_clip(src, out, cl["frames"], cl.get("start", 0), cl.get("end"), resolve(base, cl.get("xfrom")),
                          xf, cl.get("black", 0), prev_frame=pf, **common)
        outs.append(out)
    if job.get("concat"):
        lst = out_dir / "concat.txt"
        lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in outs), encoding="utf-8")
        dst = resolve(base, job["concat"])
        run_ff("-f", "concat", "-safe", 0, "-i", lst, "-c", "copy", "-movflags", "+faststart", dst)
        want = sum(cl["frames"] for cl in job["clips"])
        got = count_frames(dst)
        print(f"{dst}: {got} frames, job asks for {want}", "OK" if got == want else "MISMATCH")


if __name__ == "__main__":
    main()
