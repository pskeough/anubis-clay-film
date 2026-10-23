"""Lip-sync quoted lines in a finished film to its narration, one speaker at a time, as patches laid over the film.

    python lipsync.py quotes JOB.json     exact start and end of each quote from whisper word times (faster-whisper)
    python lipsync.py sync JOB.json       find the speaker, run LatentSync on a crop around them, paste the mouth back
    python lipsync.py check JOB.json      does each patch's mouth follow the voice better than the original does
    python lipsync.py preview JOB.json    side-by-side review clips, original left and synced right
    python lipsync.py overlay JOB.json --out FILM_synced.mp4 [--approved approved.json]

The speaker is found by face recognition (insightface buffalo_l), so a wide shot with several people syncs the right
mouth: reference embeddings come from a clear frame of each speaker, and on each frame of a quote only the face that
matches the quote's speaker (cosine >= --sim) is synced. Each run of frames with the speaker on screen is cut out as a
square crop around them with every other face masked, so LatentSync can only take that face, synced over the run's
audio, and pasted back as a feathered ellipse over the mouth only (from the 106-point landmarks). LatentSync fails a
whole run when one frame fails its own face test, so runs are split around frames that would fail it.

Needs a LatentSync checkout with its own venv (LATENTSYNC_ROOT, LATENTSYNC_PY) and insightface in the Python that
runs this script. The `quotes` step needs faster-whisper instead. Job file schema in README.md.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from _shared import (FrameWriter, cap_cpu, ffmpeg, load_job, probe, read_audio, read_frames, resolve,  # noqa: E402
                     run_ff, write_wav)

MOUTH = list(range(52, 72))        # mouth points of insightface's 2d106 landmarks
LS_FPS = 25                        # LatentSync writes 25 fps whatever it is given
SIM = 0.25                         # same speaker across shots scored 0.35-0.4 in testing; other people under 0.1
UP = 512                           # crop side handed to LatentSync
PAD = 4                            # frames added before and after each quote
MIN_RUN = 12                       # shortest run worth syncing, frames
BRIDGE = 4                         # gaps in detection up to this many frames are bridged


# ---------------------------------------------------------------- setup

def ls_root():
    r = os.environ.get("LATENTSYNC_ROOT")
    if not r or not (Path(r) / "scripts" / "inference.py").exists():
        raise SystemExit("set LATENTSYNC_ROOT to a LatentSync checkout (the folder with scripts/inference.py)")
    return Path(r)


def ls_python(root):
    if os.environ.get("LATENTSYNC_PY"):
        return os.environ["LATENTSYNC_PY"]
    for c in (root / ".venv" / "Scripts" / "python.exe", root / ".venv" / "bin" / "python"):
        if c.exists():
            return str(c)
    return sys.executable


def face_app(recognition=True, cpu=False):
    from insightface.app import FaceAnalysis
    root = os.environ.get("INSIGHTFACE_ROOT")
    if not root:
        root = str(ls_root() / "checkpoints" / "auxiliary")     # LatentSync ships buffalo_l here
    mods = ["detection", "landmark_2d_106"] + (["recognition"] if recognition else [])
    prov = ["CPUExecutionProvider"] if cpu else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    app = FaceAnalysis(allowed_modules=mods, root=root, providers=prov)
    app.prepare(ctx_id=-1 if cpu else 0, det_size=(640, 640))
    return app


class Film:
    def __init__(self, job, base):
        self.video = resolve(base, job["video"])
        self.audio = resolve(base, job.get("audio")) or self.video     # the voice the mouths follow
        self.audio_offset = float(job.get("audio_offset", 0.0))       # film seconds where `audio` starts
        self.W, self.H, fps, _ = probe(self.video)
        self.fps = float(job.get("fps") or fps)
        self.out = resolve(base, job.get("out", "lipsync"))

    def frames(self, f0, n):
        return list(read_frames(self.video, max(0, f0), n, self.fps))

    def voice(self, t0, dur, sr):
        return read_audio(self.audio, sr, start=max(0.0, t0 - self.audio_offset), duration=dur)


def quotes_of(job, base):
    q = job["quotes"]
    if isinstance(q, str):
        q = json.loads(resolve(base, q).read_text(encoding="utf-8"))
    return q


# ---------------------------------------------------------------- quotes

def norm(s):
    return re.sub(r"[^a-z0-9' ]", "", s.lower().replace("’", "'")).split()


def srt_cues(path):
    sec = lambda x: int(x[:2]) * 3600 + int(x[3:5]) * 60 + float(x[6:].replace(",", "."))
    cues = []
    for block in Path(path).read_text(encoding="utf-8-sig").strip().split("\n\n"):
        ls = block.strip().splitlines()
        if len(ls) < 3 or " --> " not in ls[1]:
            continue
        a, b = ls[1].split(" --> ")
        cues.append((sec(a.strip()), sec(b.strip()), norm(" ".join(ls[2:]))))
    return cues


def cmd_quotes(a):
    """Rough window per quote (given, or found in the subtitles from its first and last three words), then the
    first and last word of the quote from whisper word times, so the narration around it is not synced."""
    from faster_whisper import WhisperModel
    job, base = load_job(a.job)
    film = Film(job, base)
    if "quotes_in" not in job:
        raise SystemExit("the quotes step reads `quotes_in` (id, who, text, optional window) from the job")
    src = job["quotes_in"]
    qs = json.loads(resolve(base, src).read_text(encoding="utf-8")) if isinstance(src, str) else src
    cues = srt_cues(resolve(base, job["srt"])) if job.get("srt") else []
    model = WhisperModel(a.whisper, device="cpu", compute_type="int8", cpu_threads=a.threads)
    has = lambda c, w: any(c[2][i:i + len(w)] == w for i in range(len(c[2])))
    out = []
    for q in qs:
        qw = norm(q["text"])
        head, tail = qw[:3], qw[-3:]
        if "window" in q:
            w0, w1 = q["window"]
        else:
            c0 = next((c for c in cues if has(c, head)), None)
            if c0 is None:
                print(f"{q['id']}: no window given and its first words are not in the subtitles; skipped")
                continue
            c1 = next((c for c in cues if c[0] >= c0[0] and has(c, tail)), c0)
            w0, w1 = c0[0], c1[1]
        w0, w1 = max(0.0, w0 - 1.0), w1 + 1.0
        wav = film.out / "_quote.wav"
        film.out.mkdir(parents=True, exist_ok=True)
        write_wav(wav, film.voice(w0, w1 - w0, 16000), 16000)
        segs, _ = model.transcribe(str(wav), word_timestamps=True, beam_size=5, vad_filter=False)
        words = [((norm(w.word) or [""])[0], w0 + w.start, w0 + w.end) for s in segs for w in s.words]
        seq = [w[0] for w in words]
        find = lambda w, frm=0: next((i for i in range(frm, len(seq)) if seq[i:i + len(w)] == w), None)
        i0 = find(head)
        i1 = find(tail, i0 or 0)
        t0 = words[i0][1] if i0 is not None else w0 + 1.0
        t1 = words[i1 + len(tail) - 1][2] if i1 is not None else w1 - 1.0
        out.append({**{k: v for k, v in q.items() if k != "window"}, "t0": round(t0, 3), "t1": round(t1, 3),
                    "found": [i0 is not None, i1 is not None]})
        print(f"{q['id']} {q['who']:8s} {t0:8.2f}-{t1:8.2f} ({t1 - t0:4.1f}s) found {i0 is not None},"
              f"{i1 is not None}  {q['text'][:60]}", flush=True)
        wav.unlink(missing_ok=True)
    dst = resolve(base, job["quotes"]) if isinstance(job.get("quotes"), str) else film.out / "quotes.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(dst)


# ---------------------------------------------------------------- sync

def runs_of(on, min_run=MIN_RUN, bridge=BRIDGE):
    """Runs of True, bridging gaps of up to `bridge` frames, at least min_run long."""
    runs, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            k = i
            while k < n and (on[k] or any(on[k:k + bridge + 1])):
                k += 1
            if k - i >= min_run:
                runs.append((i, k))
            i = k
        else:
            i += 1
    return runs


def ref_embeddings(app, film, refs, base):
    import cv2
    out = {}
    for who, r in refs.items():
        if isinstance(r, (int, float)):
            im = film.frames(int(round(r * film.fps)), 1)[0]
        else:
            im = cv2.imread(str(resolve(base, r)))
        fs = app.get(im)
        if not fs:
            raise SystemExit(f"reference for {who!r}: no face found")
        out[who] = max(fs, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])).normed_embedding
    return out


def latent_ok(app, crop):
    """LatentSync's own face test (latentsync/utils/face_detector.py): one frame failing it fails the whole run."""
    for f in app.get(crop):
        w_, h_ = f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]
        if w_ >= 50 and h_ >= 80 and 0.2 <= w_ / h_ <= 1.5 and f.det_score >= 0.5:
            return True
    return False


def layout(fr, det, a, b, W, H):
    """Crop corners per frame (one size per run, centre smoothed over 9 frames, interpolated over bridged gaps),
    the size, and the UP-px crops with every other face masked out."""
    import cv2
    idx = [i for i in range(a, b) if det[i][0] is not None]
    cxs = np.interp(range(a, b), idx, [(det[i][0].bbox[0] + det[i][0].bbox[2]) / 2 for i in idx])
    cys = np.interp(range(a, b), idx, [(det[i][0].bbox[1] + det[i][0].bbox[3]) / 2 for i in idx])
    size = int(np.clip(2.2 * np.median([det[i][0].bbox[3] - det[i][0].bbox[1] for i in idx]), 200, min(W, H)))
    k = 9
    sm = lambda v: np.convolve(np.pad(v, k // 2, mode="edge"), np.ones(k) / k, "valid")
    cxs, cys = sm(cxs), sm(cys)
    boxes = [(int(np.clip(x - size / 2, 0, W - size)), int(np.clip(y - size / 2, 0, H - size))) for x, y in zip(cxs, cys)]
    crops = []
    for i, (x0, y0) in zip(range(a, b), boxes):
        c = fr[i][y0:y0 + size, x0:x0 + size].copy()
        fill = np.median(c.reshape(-1, 3), 0).astype(np.uint8)
        for f in det[i][1]:
            bx = f.bbox.astype(int)
            c[max(bx[1] - y0 - 10, 0):max(bx[3] - y0 + 10, 0), max(bx[0] - x0 - 10, 0):max(bx[2] - x0 + 10, 0)] = fill
        crops.append(cv2.resize(c, (UP, UP), interpolation=cv2.INTER_CUBIC))
    return boxes, size, crops


def latentsync_args(a, root):
    if a.version == "1.6":
        cfg, ckpt = "configs/unet/stage2_512.yaml", os.environ.get("LATENTSYNC_CKPT", "checkpoints_1.6/latentsync_unet.pt")
    else:
        cfg, ckpt = "configs/unet/stage2.yaml", os.environ.get("LATENTSYNC_CKPT", "checkpoints/latentsync_unet.pt")
    args = ["--unet_config_path", cfg, "--inference_ckpt_path", ckpt, "--inference_steps", str(a.steps),
            "--guidance_scale", str(a.guidance), "--seed", str(a.seed)]
    # some checkouts add --vae_slicing to scripts/inference.py to fit 1.6 on a 16 GB card; pass it when present
    if a.version == "1.6" and "vae_slicing" in (root / "scripts" / "inference.py").read_text(encoding="utf-8"):
        args.append("--vae_slicing")
    return args


def cmd_sync(a):
    import cv2
    job, base = load_job(a.job)
    film = Film(job, base)
    W, H, FPS = film.W, film.H, film.fps
    root = None if a.check_only else ls_root()
    app = face_app(recognition=True, cpu=a.cpu)
    refs = ref_embeddings(app, film, job["refs"], base)
    only = set(a.ids or [])
    for q in quotes_of(job, base):
        if only and q["id"] not in only:
            continue
        D = film.out / str(q["id"])
        D.mkdir(parents=True, exist_ok=True)
        f0 = max(0, int(round(q["t0"] * FPS)) - a.pad)
        n = int(round(q["t1"] * FPS)) + a.pad - f0
        fr = film.frames(f0, n)
        n = len(fr)
        det = []
        for im in fr:
            sims = [(float(np.dot(f.normed_embedding, refs[q["who"]])), f) for f in app.get(im)]
            best = max(sims, key=lambda s: s[0], default=(0.0, None))
            others = [f for s, f in sims if f is not best[1]]
            det.append((best[1] if best[0] >= a.sim else None, others, best[0]))
        on = [d[0] is not None for d in det]
        runs = runs_of(on, a.min_run, BRIDGE)
        print(f"{q['id']} {q['who']}: {n} frames, speaker on {sum(on)}, runs {runs}, "
              f"best sim median {np.median([d[2] for d in det]):.2f}", flush=True)
        # check sheet: first, middle and last frame of each run; the speaker boxed green, other faces red
        tiles = []
        tw, th = 672, int(round(672 * H / W))
        for (ra, rb) in runs or [(0, n)]:
            for i in (ra, (ra + rb) // 2, rb - 1):
                im = fr[i].copy()
                sp, oth, s = det[i]
                for f in oth:
                    x0, y0, x1, y1 = f.bbox.astype(int)
                    cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 3)
                if sp is not None:
                    x0, y0, x1, y1 = sp.bbox.astype(int)
                    cv2.rectangle(im, (x0, y0), (x1, y1), (0, 255, 0), 3)
                cv2.putText(im, f"{q['id']} {q['who']} f{f0 + i} sim {s:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.1, (0, 255, 255), 3)
                tiles.append(cv2.resize(im, (tw, th)))
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(D / "check.jpg"), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
        # split runs around frames that would fail LatentSync's face test; long pieces lose 2 frames each end
        split = []
        for (ra, rb) in runs:
            good = [latent_ok(app, c) for c in layout(fr, det, ra, rb, W, H)[2]]
            i = 0
            while i < len(good):
                if good[i]:
                    k = i
                    while k < len(good) and good[k]:
                        k += 1
                    if k - i >= a.min_run:
                        split.append((ra + i + 2, ra + k - 2) if k - i >= a.min_run + 4 else (ra + i, ra + k))
                    i = k
                else:
                    i += 1
        if split != runs:
            print(f"  runs after LatentSync's face test (long runs lose 2 frames each end): {runs} -> {split}",
                  flush=True)
        if a.check_only or not split:
            continue
        for r, (ra, rb) in enumerate(split):
            boxes, size, crops = layout(fr, det, ra, rb, W, H)
            raw = D / f"crop_r{r}.mp4"
            wr = FrameWriter(raw, UP, UP, FPS, crf=12)
            for c in crops:
                wr.write(c)
            wr.close()
            wav = D / f"voice_r{r}.wav"
            write_wav(wav, film.voice((f0 + ra) / FPS, (rb - ra) / FPS, 16000), 16000)
            synced = D / f"crop_r{r}_synced.mp4"
            env = dict(os.environ, PATH=str(Path(ffmpeg()).parent) + os.pathsep + os.environ.get("PATH", ""))
            res = subprocess.run([ls_python(root), "-u", "-m", "scripts.inference", *latentsync_args(a, root),
                                  "--video_path", str(raw), "--audio_path", str(wav), "--video_out_path", str(synced)],
                                 cwd=root, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                 errors="replace")
            (D / f"latentsync_r{r}.log").write_text(res.stdout + res.stderr, encoding="utf-8")
            if res.returncode or not synced.exists():
                print(f"  run {r}: LatentSync failed, see {D / f'latentsync_r{r}.log'}", flush=True)
                continue
            syn = list(read_frames(synced))
            outp = D / f"patch_r{r}.mp4"
            wr = FrameWriter(outp, W, H, FPS, crf=14)
            before_after = []
            for jj, (i, (x0, y0)) in enumerate(zip(range(ra, rb), boxes)):
                im = fr[i].copy()
                sp = det[i][0]
                kk = min(int(round(jj / FPS * LS_FPS)), len(syn) - 1)
                if sp is not None and syn:
                    s = cv2.resize(syn[kk], (size, size), interpolation=cv2.INTER_AREA)
                    m_pts = sp.landmark_2d_106[MOUTH]
                    (mx, my), (mw, mh) = m_pts.mean(0), m_pts.max(0) - m_pts.min(0)
                    m = np.zeros((size, size), np.float32)
                    cv2.ellipse(m, (int(mx - x0), int(my - y0 + mh * 0.25)), (int(mw * 0.95), int(mh * 1.6)),
                                0, 0, 360, 1, -1)
                    m = cv2.GaussianBlur(m, (0, 0), max(mw, mh) / 3)[..., None]
                    reg = im[y0:y0 + size, x0:x0 + size].astype(np.float32)
                    im[y0:y0 + size, x0:x0 + size] = (reg * (1 - m) + s.astype(np.float32) * m).astype(np.uint8)
                    if jj in ((rb - ra) // 4, (rb - ra) // 2, 3 * (rb - ra) // 4):
                        zx, zy = int(mx - mw * 1.5), int(my - mh * 2.5)
                        zs = int(max(mw * 3, mh * 5))
                        cut = lambda z: cv2.resize(z[max(zy, 0):zy + zs, max(zx, 0):zx + zs], (240, 240))
                        before_after.append(np.hstack([cut(fr[i]), cut(im)]))
                wr.write(im)
            wr.close()
            (D / f"patch_r{r}.json").write_text(json.dumps({"id": q["id"], "who": q["who"], "first_frame": f0 + ra,
                                                            "frames": rb - ra, "fps": FPS}), encoding="utf-8")
            if before_after:
                cv2.imwrite(str(D / f"mouth_r{r}.jpg"), np.vstack(before_after))
            print(f"  run {r}: patch frames {f0 + ra}..{f0 + rb - 1} written", flush=True)


# ---------------------------------------------------------------- check / preview / overlay

def patches(film, only=None):
    out = []
    for js in sorted(glob.glob(str(film.out / "*" / "patch_r*.json"))):
        m = json.loads(Path(js).read_text(encoding="utf-8"))
        name = f"{Path(js).parent.name}/{Path(js).stem}"
        if only is None or name in only or str(m["id"]) in only:
            out.append((name, m, Path(js).with_suffix(".mp4")))
    return out


def changed_centre(pa, ma, W, H):
    """Centre and spread of where a patch changed the picture."""
    ys, xs = np.nonzero(np.abs(pa.astype(np.int16) - ma).mean((0, 3)) > 6)
    if not len(xs):
        return np.array([W / 2, H / 2]), (0, 0)
    return np.array([xs.mean(), ys.mean()]), (np.ptp(xs), np.ptp(ys))


def cmd_check(a):
    """Per frame, the mouth opening (2d106 mouth height over face height) on the patch and on the original, each
    correlated with the voice's loudness per frame (best of lags 0-4 frames, mouth after sound). A good patch
    correlates clearly better than the original."""
    job, base = load_job(a.job)
    film = Film(job, base)
    app = face_app(recognition=False, cpu=a.cpu)

    def opening(frames, centre):
        out = []
        for f in frames:
            fs = app.get(f)
            if not fs:
                out.append(np.nan)
                continue
            c = min(fs, key=lambda q: np.hypot(*(((q.bbox[:2] + q.bbox[2:]) / 2) - centre)))
            m = c.landmark_2d_106[MOUTH]
            out.append((m[:, 1].max() - m[:, 1].min()) / (c.bbox[3] - c.bbox[1]))
        return np.array(out)

    def corr(x, y):
        ok = ~np.isnan(x) & ~np.isnan(y)
        return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 10 else float("nan")

    sr = 24000
    hop = int(round(sr / film.fps))
    for name, m, mp4 in patches(film, set(a.ids) if a.ids else None):
        n = m["frames"]
        pa = read_frames(mp4, 0, n)
        ma = read_frames(film.video, m["first_frame"], n, film.fps)
        n = min(len(pa), len(ma))
        pa, ma = pa[:n], ma[:n]
        au = film.voice(m["first_frame"] / film.fps, n / film.fps, sr)
        au = np.pad(au, (0, max(0, n * hop - len(au))))
        env = np.sqrt(np.array([np.mean(au[i * hop:(i + 1) * hop] ** 2) for i in range(n)]) + 1e-9)
        centre, _ = changed_centre(pa, ma, film.W, film.H)
        op, om = opening(pa, centre), opening(ma, centre)
        best = lambda o: max(((corr(o[k:], env[:n - k]), k) for k in range(5)),
                             key=lambda t: -2 if np.isnan(t[0]) else t[0])
        (cp, kp), (cm, km) = best(op), best(om)
        print(f"{name} {m['who']}: {n} frames  mouth~voice r patch {cp:+.2f} (lag {kp}) vs original {cm:+.2f} "
              f"(lag {km})   opening spread patch {np.nanstd(op):.3f} original {np.nanstd(om):.3f}", flush=True)


def cmd_preview(a):
    import cv2
    job, base = load_job(a.job)
    film = Film(job, base)
    rev = film.out / "_review"
    rev.mkdir(parents=True, exist_ok=True)
    W, H = film.W, film.H
    for name, m, mp4 in patches(film, set(a.ids) if a.ids else None):
        n, t0 = m["frames"], m["first_frame"] / film.fps
        pa, ma = read_frames(mp4, 0, n), read_frames(film.video, m["first_frame"], n, film.fps)
        n = min(len(pa), len(ma))
        (cx, cy), (sx, sy) = changed_centre(pa[:n], ma[:n], W, H)
        s = int(np.clip(max(sx, sy) * 3.2 if sx else 300, 160, min(W, H)))
        x0, y0 = int(np.clip(cx - s / 2, 0, W - s)), int(np.clip(cy - s * 0.45, 0, H - s))
        tag = name.replace("/", "_")
        raw = rev / f"_{tag}.mp4"
        wr = FrameWriter(raw, 1080, 540, film.fps, crf=18)
        for o, p in zip(ma[:n], pa[:n]):
            L = cv2.resize(o[y0:y0 + s, x0:x0 + s], (540, 540), interpolation=cv2.INTER_CUBIC)
            R = cv2.resize(p[y0:y0 + s, x0:x0 + s], (540, 540), interpolation=cv2.INTER_CUBIC)
            cv2.putText(L, "original", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(R, f"synced {m['id']}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            wr.write(np.hstack([L, R]))
        wr.close()
        wav = rev / f"_{tag}.wav"
        write_wav(wav, film.voice(t0, n / film.fps, 48000), 48000)
        run_ff("-i", raw, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
               "-shortest", rev / f"{tag}.mp4")
        raw.unlink()
        wav.unlink()
        print(rev / f"{tag}.mp4")


def cmd_overlay(a):
    """Lay patches over the film at their frames: each patch is shifted to its first frame
    (setpts=PTS-STARTPTS+t0/TB) and shown only between its first and last frame. Audio is copied."""
    job, base = load_job(a.job)
    film = Film(job, base)
    only = None
    if a.approved:
        only = set(json.loads(Path(a.approved).read_text(encoding="utf-8")))
    pts = patches(film, only)
    if not pts:
        raise SystemExit("no patches to lay")
    lay, last = "", "[0:v]"
    for k, (_, m, _) in enumerate(pts):
        fps = m.get("fps", film.fps)
        t0, t1 = m["first_frame"] / fps, (m["first_frame"] + m["frames"]) / fps
        lay += (f"[{k + 1}:v]setpts=PTS-STARTPTS+{t0:.4f}/TB[p{k}];"
                f"{last}[p{k}]overlay=enable='between(t,{t0:.4f},{t1 - 0.001:.4f})':eof_action=pass[m{k}];")
        last = f"[m{k}]"
    fc = lay + f"{last}format=yuv420p[v]"
    run_ff("-i", film.video, *[x for _, _, p in pts for x in ("-i", p)], "-filter_complex", fc, "-map", "[v]",
           "-map", "0:a?", "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf), "-c:a", "copy",
           "-movflags", "+faststart", a.out)
    print(len(pts), "patches laid;", a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu", action="store_true", help="run face detection on the CPU")
    ap.add_argument("--threads", type=int, default=4, help="CPU cores to use (default 4)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quotes", help="exact quote spans from whisper word times")
    q.add_argument("job")
    q.add_argument("--whisper", default="small.en", help="faster-whisper model (default small.en)")
    s = sub.add_parser("sync", help="lip-sync each quote's speaker into patch files")
    s.add_argument("job")
    s.add_argument("ids", nargs="*", help="only these quote ids")
    s.add_argument("--check-only", action="store_true", help="find faces and runs, write check.jpg, skip LatentSync")
    s.add_argument("--version", choices=["1.5", "1.6"], default="1.5",
                   help="LatentSync 1.5 (256 px mouth, default) or 1.6 (512 px, sharper, slower)")
    s.add_argument("--guidance", type=float, default=1.5, help="LatentSync audio guidance (default 1.5)")
    s.add_argument("--steps", type=int, default=20)
    s.add_argument("--seed", type=int, default=1247)
    s.add_argument("--sim", type=float, default=SIM, help=f"face match threshold, cosine (default {SIM})")
    s.add_argument("--pad", type=int, default=PAD, help=f"frames added each side of a quote (default {PAD})")
    s.add_argument("--min-run", type=int, default=MIN_RUN, help=f"shortest run synced, frames (default {MIN_RUN})")
    c = sub.add_parser("check", help="mouth opening vs voice loudness, patch against original")
    c.add_argument("job")
    c.add_argument("ids", nargs="*")
    p = sub.add_parser("preview", help="side-by-side review clips")
    p.add_argument("job")
    p.add_argument("ids", nargs="*")
    o = sub.add_parser("overlay", help="lay the patches over the film")
    o.add_argument("job")
    o.add_argument("--out", required=True)
    o.add_argument("--approved", help='JSON list of patches to lay, e.g. ["q01/patch_r0"] (default: all)')
    o.add_argument("--crf", type=int, default=16)
    a = ap.parse_args()
    cap_cpu(a.threads)
    {"quotes": cmd_quotes, "sync": cmd_sync, "check": cmd_check, "preview": cmd_preview, "overlay": cmd_overlay}[a.cmd](a)


if __name__ == "__main__":
    main()
