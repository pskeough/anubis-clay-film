"""Voice Editor, the booth's second window (/editor): lay the recorded takes against the picture by hand.

A project is a list of clips on the output clock (the film slowed to the booth's rate, the clock you heard
while recording). Each clip plays one take's WAV from wav_in to wav_out at 1x, starting at output second
`at`. Where clips overlap, the later-recorded take wins; the page flattens that and sends the flat list
to render(). Saved per piece as voice/edit_project.json (previous save kept as edit_project.prev.json).

Optional starting cut, per piece: voice/seed_cues.csv (columns take, offset_s, src_in, src_out; one row
per piece of a take the automatic later-take-wins cut kept) and voice/film_words.json (word timings
per take, {take: [{w0, w1, raw}]}) beside it. VOICE_EDIT_SEED_CUES overrides the csv's path. Without
them a new project starts with no clips.
"""
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

S = None                     # the server module, set by server.py on import
RATE = 0.95
XF = 0.008                   # seconds of fade either side of every cut, so abutting pieces crossfade


def _wav_t(segs, m):
    s = segs[0]
    for g in segs:
        if m >= g[0]:
            s = g
    return m / RATE + s[2] + s[3] * m


def _seed_cues(d) -> Path:
    return Path(os.environ.get("VOICE_EDIT_SEED_CUES") or d / "voice" / "seed_cues.csv")


def seed(d):
    """Clips from the automatic later-take-wins cut. Each clip carries `speed` (WAV seconds per output
    second) from the take's sync fit, so it stays on the picture the way it was recorded; a clip is split
    only where the picture stalled while recording (a new sync segment)."""
    cues = _seed_cues(d)
    if not cues.exists():
        return []
    by_id = {t["id"]: t for t in S.takes(d, with_map=True)}
    clips = []
    for r in csv.DictReader(open(cues, newline="", encoding="utf-8")):
        t = by_id.get(r["take"])
        if not t:
            continue
        segs = S.segments(t, t.get("rate", RATE))
        off, m0, m1 = float(r["offset_s"]), float(r["src_in"]), float(r["src_out"])
        cuts = sorted({m0, m1, *[g[0] for g in segs if m0 < g[0] < m1]})
        for a, b in zip(cuts, cuts[1:]):
            if b - a < 0.02:
                continue
            w_a, w_b = _wav_t(segs, a), _wav_t(segs, b - 1e-6)
            clips.append({"take": t["id"], "at": round((a + off) / RATE, 4), "wav_in": round(w_a, 4),
                          "wav_out": round(w_b, 4), "speed": round((w_b - w_a) / ((b - a) / RATE), 6)})
    return clips


def _words(d):
    words = _seed_cues(d).parent / "film_words.json"
    if not words.exists():
        return {}
    w = json.loads(words.read_text(encoding="utf-8"))
    return {t: [[round(x["w0"], 3), round(x["w1"], 3), x["raw"]] for x in ws if x.get("w1", 0) > x.get("w0", 0)]
            for t, ws in w.items()}


def project(d):
    p = d / "voice" / "edit_project.json"
    tl = S.timeline(d)
    takes = []
    for t in S.takes(d, with_map=True):
        if abs(t.get("rate", 1.0) - RATE) > 1e-3:
            continue
        segs = S.segments(t, t["rate"])
        takes.append({"id": t["id"], "created": t.get("created", ""), "dur": t.get("duration_s") or None,
                      "speed": round(1 + segs[0][3] * RATE, 6),

                      "rec_at": round(-segs[0][2], 3),          # output second where the WAV's 0 s sat when recorded
                      "range": [round(t.get("range_start", t["video_start"]) / RATE, 2), round(t["video_end"] / RATE, 2)]})
    saved = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return {"rate": RATE, "duration": round(tl["duration"] / RATE, 3), "takes": takes,
            "lines": [{"id": s["id"], "text": s["narration"], "shot": [round(s["shot_start"] / RATE, 3), round(s["shot_end"] / RATE, 3)],
                       "speech": [round(s["speech_start"] / RATE, 3), round(s["speech_end"] / RATE, 3)]} for s in tl["scenes"]],
            "words": _words(d), "clips": saved["clips"] if saved else seed(d), "saved": bool(saved),
            "seed": seed(d)}


def save(d, clips):
    v = d / "voice"
    p = v / "edit_project.json"
    if p.exists():
        p.replace(v / "edit_project.prev.json")
    p.write_text(json.dumps({"saved": time.strftime("%Y-%m-%d %H:%M:%S"), "clips": clips}, indent=0), encoding="utf-8")
    return {"ok": True, "n": len(clips)}


def render(d, flat, mp4=False, preset="broadcast"):
    """flat: [{take, at, wav_in, wav_out}] with no overlaps (the page resolved them). Denoised takes,
    8 ms crossfades at every cut, then the booth's polish on the whole track; optional MP4 at the booth rate."""
    sr = S.SR
    dur = S.timeline(d)["duration"] / RATE
    track = np.zeros(int(round(dur * sr)) + sr, np.float32)
    by_id = {t["id"]: t for t in S.takes(d, with_map=True)}
    audio = {}
    xf = int(XF * sr)
    for c in flat:
        tid = c["take"]
        if tid not in audio:
            audio[tid] = S.take_audio(d, by_id[tid], "raw" if preset == "raw" else "dn")
        x = audio[tid]
        sp = float(c.get("speed", 1.0))
        olen = (c["wav_out"] - c["wav_in"]) / sp
        j0 = max(int(round((c["at"] - XF) * sr)), 0)
        j1 = min(int(round((c["at"] + olen + XF) * sr)), len(track))
        if j1 <= j0:
            continue
        src = (c["wav_in"] + (np.arange(j0, j1) / sr - c["at"]) * sp) * sr
        seg = np.interp(src, np.arange(len(x)), x, left=0.0, right=0.0).astype(np.float32)
        n = len(seg)
        f = min(2 * xf, n // 2)
        if f:
            seg[:f] *= np.linspace(0, 1, f, dtype=np.float32)
            seg[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
        track[j0:j0 + n] += seg
    track = track[:int(round(dur * sr))]
    with S._polish_lock:
        track = S.polish(track, preset if preset in S.PRESETS else "broadcast")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    v = d / "voice"
    wav = v / f"{d.name}_edit_x{RATE:.2f}_{stamp}.wav"
    S.write_wav(wav, track)
    out = {"wav": wav.name, "wav_path": str(wav), "voice": S.analyze(track, sr, drops=False)}
    if mp4:
        pic = d / "_assemble" / "_silent.mp4"
        fps = S.video_fps(pic)
        m = v / f"{d.name}_edit_x{RATE:.2f}_{stamp}.mp4"
        base = [S.ffmpeg(), "-v", "error", "-y", "-threads", "4", "-filter_threads", "4", "-i", str(pic), "-i", str(wav),
                "-map", "0:v:0", "-map", "1:a:0", "-vf", f"setpts=PTS/{RATE},framerate=fps={fps:g}:scene=8.2", "-r", f"{fps:g}"]
        tail = ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(m)]
        try:
            subprocess.run(base + ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19"] + tail, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            subprocess.run(base + ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast"] + tail, check=True, capture_output=True)
        out.update({"mp4": m.name, "mp4_path": str(m)})
    return out
