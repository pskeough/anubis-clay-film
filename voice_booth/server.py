"""Voice Booth: record your own narration against a finished piece's picture.

VoiceBooth.bat starts this with the Podcast_Audio_Gen venv (faster-whisper, numpy, imageio-ffmpeg)
and opens http://127.0.0.1:7090. Bound to 127.0.0.1 only; 127.0.0.1 is a secure context, so the
browser allows the microphone without https. No GPU: whisper runs on CPU once per piece, ffmpeg on CPU.

Per piece (any folder under Agentic_Slop_MP4/pieces with timeline.json and _assemble/_silent.mp4):
  picture   _assemble/_silent.mp4, the final graded film with no audio
  guide     narration.mp3, the AI read, optional in the page
  words     words.json, word timings of the AI read (align.py, cached)
  takes     voice/take_*.wav + .json (rate, sync map, range, capture report)
  raw       voice/_raw/take_*.worklet.wav + .rec.webm, the two captures a take was made from
  builds    voice/<piece>_voice_*.wav and <piece>_voiced_*.mp4 (picture retimed to the rate + takes)

Capture. The page records two ways at once. The AudioWorklet copy is on the audio-context clock, which
the sync is measured against, but Chromium drops a few samples about once a second on it when the mic
and the output device run on different clocks (a USB mic always does); each drop is a click. The
MediaRecorder copy is on the mic's own clock and has no drops. The server finds where the recorder copy
sits on the worklet timeline by cross-correlation, fits the small clock drift, and keeps the recorder
copy. If that fails, it keeps the worklet copy with the drops repaired.

Sync map. During a take the page logs (seconds into the worklet WAV, media time of the frame on screen)
for every presented frame. The build places the voice from that map, so a playback rate other than 1,
a stall in the picture, or clock drift all land correctly.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
PIECES = Path(os.environ.get("CLAY_FILM_PIECES") or HERE.parent / "pieces")
PORT = int(os.environ.get("VOICE_BOOTH_PORT", "7090"))
SR = 48000
TAIL = 0.3          # media seconds a shot-by-shot take may spill past its shot, where nothing else is laid
FADE = 0.02
PRESETS = ("raw", "clean", "broadcast")
MOTION = ("blend", "repeat", "smooth")
sys.path.insert(0, str(HERE))
import align  # noqa: E402
import captions  # noqa: E402

_align_lock = threading.Lock()
_caption_lock = threading.Lock()
_polish_lock = threading.Lock()


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def pieces():
    out = []
    for d in PIECES.iterdir():
        if d.name.startswith("_") or not d.is_dir():
            continue
        if (d / "timeline.json").exists() and (d / "_assemble" / "_silent.mp4").exists():
            out.append(d)
    return sorted(out, key=lambda d: (d / "_assemble" / "_silent.mp4").stat().st_mtime, reverse=True)


def piece_dir(name):
    d = PIECES / name
    if d not in pieces():
        raise KeyError(name)
    return d


def title_of(d):
    try:
        return json.loads((d / "plan.json").read_text(encoding="utf-8"))["title"]
    except Exception:
        return d.name


def timeline(d):
    return json.loads((d / "timeline.json").read_text(encoding="utf-8"))


def takes(d, with_map=False):
    v = d / "voice"
    out = []
    for m in sorted(v.glob("take_*.json")) if v.exists() else []:
        meta = json.loads(m.read_text(encoding="utf-8"))
        if (v / meta["file"]).exists():
            if not with_map:   # the page gets the fitted sync segments instead of thousands of map points
                meta["segs"] = segments(meta, meta.get("rate", 1.0))
                meta = {k: x for k, x in meta.items() if k != "map"}
            out.append(meta)
    return out


def builds(d):
    v = d / "voice"
    return [p.name for p in sorted(v.glob(f"{d.name}_voiced_*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)] if v.exists() else []


# ---------------------------------------------------------------- audio io
def decode(src, sr=SR):
    """Any audio file or bytes -> mono float32 at sr."""
    args = [ffmpeg(), "-v", "error", "-i", "-" if isinstance(src, bytes) else str(src),
            "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(args, input=src if isinstance(src, bytes) else None, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def read_wav16(data):
    """The page's own 16-bit mono WAV -> (float32, sample rate), without resampling."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV")
    sr = int.from_bytes(data[24:28], "little")
    i = data.find(b"data", 12)
    return np.frombuffer(data[i + 8:], "<i2").astype(np.float32) / 32768.0, sr


def resample(x, sr_in):
    if sr_in == SR:
        return x
    raw = subprocess.run([ffmpeg(), "-v", "error", "-f", "f32le", "-ar", str(sr_in), "-ac", "1", "-i", "-",
                          "-ar", str(SR), "-f", "f32le", "-"], input=x.astype(np.float32).tobytes(),
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def write_wav(path, x):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    subprocess.run([ffmpeg(), "-v", "error", "-y", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-", str(path)],
                   input=pcm, check=True)


# ---------------------------------------------------------------- analysis and repair
def frame_db(x, sr, win=0.05):
    n = int(sr * win)
    k = len(x) // n
    if k == 0:
        return np.array([-120.0])
    r = np.sqrt((x[:k * n].reshape(k, n) ** 2).mean(1))
    r = r[r > 0]                      # exact zeros are unrecorded gaps in a built track, not room tone
    return 20 * np.log10(r + 1e-9) if len(r) else np.array([-120.0])


def find_drops(x, sr):
    """Positions of single-sample steps (dropped samples). Chromium's drift correction drops at a fixed
    phase of its 10 ms output callback, so when most candidates share one phase, look there harder."""
    if len(x) < sr:
        return np.array([], int)
    d1 = np.abs(np.diff(x.astype(np.float64)))
    blk = 480
    nb = len(d1) // blk
    med = np.repeat(np.median(d1[:nb * blk].reshape(nb, blk), 1), blk)
    med = np.concatenate([med, np.full(len(d1) - len(med), med[-1] if len(med) else 1e-4)]) + 1e-5
    pad = np.pad(d1, 3)
    nbr = np.max(np.stack([pad[1:-5], pad[2:-4], pad[4:-2], pad[5:-1]]), 0)
    ratio = d1 / med
    strong = np.where((ratio > 25) & (d1 > 3 * nbr) & (d1 > 0.004))[0]
    period = int(round(sr / 100))
    if len(strong) >= 5:
        h = np.bincount(strong % period, minlength=period).astype(float)
        h3 = h + np.roll(h, 1) + np.roll(h, -1)
        ph = int(np.argmax(h3))
        if h3[ph] >= 0.4 * len(strong):
            near = np.abs(((np.arange(len(d1)) - ph + period // 2) % period) - period // 2) <= 1
            return np.where(near & (ratio > 8) & (d1 > 2 * nbr) & (d1 > 0.002))[0] + 1
    return np.where((ratio > 60) & (d1 > 4 * nbr) & (d1 > 0.004))[0] + 1


def repair_drops(x, pos, half=240):
    """Remove each step by spreading it over 2*half samples with a raised-cosine offset: the click
    becomes a sub-audible 5 ms nudge."""
    y = x.astype(np.float64).copy()
    k = np.arange(half)
    up = 0.5 * (1 - np.cos(np.pi * (k + 0.5) / half))           # 0 -> 1
    for p in pos:
        if p < 3 or p + half >= len(y) or p - half < 0:
            continue
        pred = 2 * y[p - 1] - y[p - 2]
        delta = y[p] - pred
        y[p - half:p] += 0.5 * delta * up
        y[p:p + half] -= 0.5 * delta * up[::-1]
    return y.astype(np.float32)


def analyze(x, sr, drops=True):
    db = frame_db(x, sr)
    a = np.abs(x)
    out = {"peak_db": round(float(20 * np.log10(a.max() + 1e-9)), 1) if len(a) else -120.0,
           "clipped": int((a >= 0.999).sum()),
           "noise_db": round(float(np.percentile(db, 10)), 1),
           "speech_db": round(float(np.percentile(db, 90)), 1)}
    if drops:                         # only meaningful on a capture; compression makes false hits
        out["drops"] = int(len(find_drops(x, sr)))
    return out


# ---------------------------------------------------------------- recorder alignment
def best_lag(a, b, lo, hi):
    """Lag L in [lo, hi] maximising the normalised correlation of a with b[L:L+len(a)]."""
    n = len(a)
    lo, hi = max(lo, 0), min(hi, len(b) - n)
    if hi < lo or n == 0:
        return None, 0.0
    seg = b[lo:hi + n].astype(np.float64)
    N = 1 << int(np.ceil(np.log2(len(seg) + n)))
    cc = np.fft.irfft(np.fft.rfft(seg, N) * np.conj(np.fft.rfft(a.astype(np.float64), N)), N)[:hi - lo + 1]
    e = np.concatenate([[0.0], np.cumsum(seg ** 2)])
    win = e[n:n + hi - lo + 1] - e[:hi - lo + 1]
    nc = cc / np.sqrt(win * float((a.astype(np.float64) ** 2).sum()) + 1e-12)
    k = int(np.argmax(nc))
    return lo + k, float(nc[k])


def align_recorder(ref, rec):
    """Place the MediaRecorder copy on the worklet timeline. Returns (aligned, report) or (None, report)."""
    W = int(1.0 * SR)
    step = int((2.0 if len(ref) < 20 * SR else 6.0) * SR)
    starts = list(range(int(0.3 * SR), max(len(ref) - W, 0), step))
    if not starts:
        return None, {"why": "take too short to align"}
    rms = np.array([np.sqrt((ref[s:s + W] ** 2).mean()) for s in starts])
    keep = [s for s, r in zip(starts, rms) if r >= max(np.median(rms), 10 ** (-50 / 20))]
    pts = []
    pred = None
    for s in keep:
        span = int(3 * SR) if pred is None else int(0.06 * SR)
        c0 = s + (0 if pred is None else int(round(pred(s))))
        L, c = best_lag(ref[s:s + W], rec, c0 - span, c0 + span)
        if L is None or c < 0.5:
            continue
        pts.append((s, L - s, c))
        if len(pts) >= 3:
            t_, o_ = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
            b_, a_ = np.polyfit(t_, o_, 1)
            pred = lambda t, a_=a_, b_=b_: a_ + b_ * t
        else:
            off = float(np.median([p[1] for p in pts]))
            pred = lambda t, off=off: off
    if not pts:
        return None, {"why": "recorder copy did not match the worklet copy"}
    t_, o_ = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
    if len(pts) >= 4 and t_[-1] - t_[0] > 10 * SR:
        mask = np.ones(len(t_), bool)
        for _ in range(3):
            b_, a_ = np.polyfit(t_[mask], o_[mask], 1)
            res = o_ - (a_ + b_ * t_)
            mask = np.abs(res) < max(0.003 * SR, 3 * np.std(res[mask]))
            if mask.sum() < 3:
                return None, {"why": "offsets do not fit a line"}
    else:
        a_, b_, mask = float(np.median(o_)), 0.0, np.ones(len(t_), bool)
    res_ms = float(np.sqrt(np.mean((o_[mask] - (a_ + b_ * t_[mask])) ** 2)) / SR * 1000)
    idx = np.arange(len(ref), dtype=np.float64)
    out = np.interp(idx + a_ + b_ * idx, np.arange(len(rec)), rec, left=0.0, right=0.0).astype(np.float32)
    rep = {"windows": int(mask.sum()), "corr": round(float(np.median([p[2] for p in pts])), 3),
           "offset_ms": round(a_ / SR * 1000, 1), "drift_ppm": round(b_ * 1e6, 1), "fit_ms": round(res_ms, 2)}
    if res_ms > 4 or rep["corr"] < 0.6:
        return None, {**rep, "why": "alignment too loose"}
    return out, rep


# ---------------------------------------------------------------- voice polish
def _box(a, k, axis):
    """Moving average of odd width k along axis, edges padded by repetition."""
    pad = [(0, 0)] * a.ndim
    pad[axis] = (k // 2, k // 2)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
    c = np.concatenate([np.zeros_like(np.take(c, [0], axis=axis)), c], axis=axis)
    n = a.shape[axis]
    return ((np.take(c, np.arange(k, k + n), axis=axis) - np.take(c, np.arange(0, n), axis=axis)) / k).astype(np.float32)


def denoise(x, noise=None, reduce_db=14.0, n=2048, hop=512):
    """Spectral gate from a noise print (the take's count-in room tone, else its quietest tenth).
    Per frequency bin, anything under mean+1.5 sd of the print is pulled down by reduce_db; the keep
    mask is smoothed over ~50 ms and ~200 Hz so words keep their tails. Stationary noise only (fans,
    hum, hiss); a car going past stays."""
    if len(x) < n * 4:
        return x
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)
    xp = np.pad(x.astype(np.float32), (n, n + hop))

    def spec(sig):
        fr = np.lib.stride_tricks.sliding_window_view(sig, n)[::hop]
        return np.fft.rfft(fr * win, axis=1)

    X = spec(xp)
    mag = 20 * np.log10(np.abs(X) + 1e-9).astype(np.float32)
    if noise is not None and len(noise) >= SR // 2:
        nm = 20 * np.log10(np.abs(spec(noise.astype(np.float32))) + 1e-9)
    else:
        e = mag.mean(1)
        nm = mag[e <= np.percentile(e, 10)]
    thresh = nm.mean(0) + 1.5 * nm.std(0)
    keep = (mag > thresh).astype(np.float32)
    keep = _box(_box(keep, 5, 0), 9, 1)
    floor = 10 ** (-reduce_db / 20)
    G = floor + (1 - floor) * np.clip(keep * 1.5, 0, 1)
    fr = np.fft.irfft(X * G, n, axis=1).astype(np.float32) * win
    out = np.zeros(len(xp), np.float32)
    for i in range(fr.shape[0]):
        out[i * hop:i * hop + n] += fr[i]
    return (out[n:n + len(x)] / 1.5).astype(np.float32)


def room_tone(x, t):
    """The stretch before the first word: count-in plus any pre-roll, minus margins."""
    rate = t.get("rate", 1.0)
    quiet_until = t.get("lead_in", 0) + (t.get("range_start", t.get("video_start", 0)) - t.get("video_start", 0)) / rate
    a, b = int(0.25 * SR), int((quiet_until - 0.35) * SR)
    return x[a:b] if b - a >= SR else None


def _run_chain(x, chain, measure=False):
    args = [ffmpeg(), "-hide_banner", "-nostats", "-v", "info" if measure else "error", "-f", "f32le", "-ar", str(SR),
            "-ac", "1", "-i", "-", "-af", chain]
    args += ["-f", "null", "-"] if measure else ["-ar", str(SR), "-ac", "1", "-f", "f32le", "-"]
    p = subprocess.run(args, input=x.astype(np.float32).tobytes(), capture_output=True, check=True)
    return p.stderr.decode("utf-8", "replace") if measure else np.frombuffer(p.stdout, np.float32).copy()


def polish(x, preset, target_lufs=-16.0):
    """Tone and loudness, applied to audio that is already denoised (take_audio stage "dn").
    raw: untouched. clean: rumble cut and loudness. broadcast: rumble cut, mud cut, presence and air,
    de-ess, gentle 3:1 compression, loudness. Loudness is measured (EBU R128) and set with a plain gain
    and a true-peak limiter with its lookahead compensated; none of these filters shifts the audio."""
    if preset == "raw" or not len(x) or not np.any(x):
        return x
    chain = ["highpass=f=70:poles=2"]
    if preset == "broadcast":
        chain += ["equalizer=f=220:t=q:w=1.0:g=-2.5",
                  "equalizer=f=3200:t=q:w=1.2:g=2.5",
                  "highshelf=f=9000:g=1.5",
                  "deesser=i=0.35:m=0.5:f=0.5",
                  "acompressor=threshold=0.1:ratio=3:attack=6:release=120:knee=3:makeup=1"]
    chain = ",".join(chain)
    log = _run_chain(x, chain + ",ebur128=framelog=quiet", measure=True)
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", log)
    gain = min(max(target_lufs - float(m[-1]), -20.0), 30.0) if m else 0.0
    y = _run_chain(x, chain + f",volume={gain:.2f}dB,alimiter=limit=0.84:attack=5:release=60:level=disabled:latency=1")
    if len(y) >= len(x):
        return y[:len(x)]
    return np.concatenate([y, np.zeros(len(x) - len(y), np.float32)])


def take_audio(d, t, stage="raw"):
    """A take's audio at SR, cached in voice/_cache.
    raw: as saved (drops repaired for takes from before the recorder path). dn: raw, denoised from its
    own room tone. clean / broadcast: dn plus polish(), for reviewing one take on its own."""
    v = d / "voice"
    src = v / t["file"]
    if stage == "raw" and t.get("clean"):
        return decode(src)
    cache = v / "_cache" / f"{t['id']}.{stage}.wav"
    if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        return decode(cache)
    if stage == "raw":
        x = decode(src)
        y = repair_drops(x, find_drops(x, SR))
    elif stage == "dn":
        x = take_audio(d, t, "raw")
        y = denoise(x, room_tone(x, t))
    else:
        with _polish_lock:
            y = polish(take_audio(d, t, "dn"), stage)
    cache.parent.mkdir(exist_ok=True)
    write_wav(cache, y)
    return y


# ---------------------------------------------------------------- placement
def segments(t, rate):
    """[(m_a, m_b, a, b)]: over media time m in [m_a, m_b], the take's audio for m sits at
    WAV seconds m/rate + a + b*m. Split where the picture stalled; each piece fitted as a line."""
    mp = t.get("map") or []
    if len(mp) < 8:
        return [(t["video_start"], t["video_end"], t["lead_in"] - t["video_start"] / rate, 0.0)]
    w = np.array([p[0] for p in mp], float)
    m = np.array([p[1] for p in mp], float)
    o = np.argsort(m, kind="stable")
    w, m = w[o], m[o]
    off = w - m / rate
    k = 7
    padded = np.pad(off, k // 2, mode="edge")
    sm = np.array([np.median(padded[i:i + k]) for i in range(len(off))])
    cuts = list(np.where(np.abs(np.diff(sm)) > 0.06)[0] + 1)
    bounds = [0, *cuts, len(m)]
    segs = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        mm, oo = m[s:e], off[s:e]
        a, b = float(np.median(oo)), 0.0
        if len(mm) >= 16 and mm[-1] - mm[0] > 3:
            mask = np.ones(len(mm), bool)
            for _ in range(3):
                b, a = np.polyfit(mm[mask], oo[mask], 1)
                mask = np.abs(oo - (a + b * mm)) < 0.015
                if mask.sum() < 8:
                    a, b = float(np.median(oo)), 0.0
                    break
        m_a = t["video_start"] if i == 0 else float(mm[0])
        m_b = t["video_end"] if e == len(m) else float(m[e])
        segs.append((m_a, m_b, float(a), float(b)))
    return segs


def lay(track, audio, segs, lo_m, hi_m, rate, nudge):
    fade = int(FADE * SR)
    idx = np.arange(len(audio), dtype=np.float64)
    for m_a, m_b, a, b in segs:
        s_m, e_m = max(m_a, lo_m), min(m_b, hi_m)
        if e_m - s_m < 0.01:
            continue
        j0 = int(np.ceil(s_m / rate * SR))
        j1 = min(int(np.floor(e_m / rate * SR)), len(track))
        for c0 in range(j0, j1, 10 * SR):
            c1 = min(c0 + 10 * SR, j1)
            j = np.arange(c0, c1, dtype=np.float64)
            src = (j / SR + a + b * (j / SR * rate) + nudge) * SR
            seg = np.interp(src, idx, audio, left=0.0, right=0.0).astype(np.float32)
            ramp = np.ones(c1 - c0, np.float32)
            if c0 == j0:
                f = min(fade, (j1 - j0) // 2)
                ramp[:f] = np.linspace(0, 1, f)
            if c1 == j1:
                f = min(fade, (j1 - j0) // 2)
                if f:
                    ramp[-f:] = np.minimum(ramp[-f:], np.linspace(1, 0, f))
            track[c0:c1] = track[c0:c1] * (1 - ramp) + seg * ramp


def video_fps(path):
    err = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True).stderr.decode("utf-8", "replace")
    m = re.search(r"([\d.]+) fps", err)
    return float(m.group(1)) if m else 24.0


def build(d, ids, nudge_ms, rate=1.0, preset="broadcast", motion="blend"):
    """Lay the chosen takes (those recorded at this rate) on one track the length of the retimed film.
    Later takes win where they overlap, 20 ms crossfades at every seam; shot-by-shot spill (TAIL) goes
    down first so it never covers another take. nudge_ms > 0 moves the voice earlier."""
    rate = float(rate)
    dur = timeline(d)["duration"]
    out_dur = dur / rate
    track = np.zeros(int(round(out_dur * SR)), np.float32)
    by_id = {t["id"]: t for t in takes(d, with_map=True)}
    chosen = [by_id[i] for i in ids if i in by_id and abs(by_id[i].get("rate", 1.0) - rate) < 1e-3]
    skipped = [i for i in ids if i in by_id and by_id[i] not in chosen]
    if not chosen:
        raise ValueError(f"no ticked takes were recorded at x{rate:.2f}")
    nudge = nudge_ms / 1000.0
    audio = {t["id"]: take_audio(d, t, "raw" if preset == "raw" else "dn") for t in chosen}
    segs = {t["id"]: segments(t, rate) for t in chosen}
    for t in chosen:                                       # spill first
        if t.get("range_end") is not None:
            lay(track, audio[t["id"]], segs[t["id"]], t["range_end"], min(t["range_end"] + TAIL, t["video_end"]), rate, nudge)
    for t in chosen:                                       # then the takes proper, later wins
        r0 = t.get("range_start", t["video_start"])
        r1 = t["range_end"] if t.get("range_end") is not None else t["video_end"]
        lay(track, audio[t["id"]], segs[t["id"]], r0, r1, rate, nudge)
    with _polish_lock:
        track = polish(track, preset if preset in PRESETS else "broadcast")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = "" if abs(rate - 1) < 1e-3 else f"x{rate:.2f}_"
    v = d / "voice"
    wav = v / f"{d.name}_voice_{tag}{stamp}.wav"
    write_wav(wav, track)
    mp4 = v / f"{d.name}_voiced_{tag}{stamp}.mp4"
    pic = d / "_assemble" / "_silent.mp4"
    cmd = [ffmpeg(), "-v", "error", "-y", "-i", str(pic), "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0"]
    if abs(rate - 1) < 1e-3:
        cmd += ["-c:v", "copy"]
    else:
        fps = video_fps(pic)
        vf = {"repeat": f"setpts=PTS/{rate}",
              "blend": f"setpts=PTS/{rate},framerate=fps={fps:g}:scene=8.2",
              "smooth": f"setpts=PTS/{rate},minterpolate=fps={fps:g}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=fdiff",
              }[motion if motion in MOTION else "blend"]
        cmd += ["-vf", vf, "-r", f"{fps:g}", "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(mp4)]
    subprocess.run(cmd, check=True, capture_output=True)
    return {"wav": wav.name, "mp4": mp4.name, "path": str(mp4), "used": len(chosen), "skipped": skipped,
            "duration": round(out_dur, 2), "voice": analyze(track, SR, drops=False)}


# ---------------------------------------------------------------- capture intake
def split_parts(data, q):
    lens = [int(x) for x in q.get("parts", f"{len(data)},0,0").split(",")]
    wav = data[:lens[0]]
    rec = data[lens[0]:lens[0] + lens[1]]
    mp = data[lens[0] + lens[1]:lens[0] + lens[1] + lens[2]]
    return wav, rec, (json.loads(mp.decode("utf-8")) if mp else [])


def clean_capture(wav, rec):
    """-> (audio at SR on the worklet timeline, report)."""
    x, sr = read_wav16(wav)
    drops = find_drops(x, sr)
    rep = {"worklet_drops": int(len(drops)), "worklet_sr": sr}
    fixed = resample(repair_drops(x, drops), sr)
    if rec:
        try:
            r = decode(rec)
            out, arep = align_recorder(resample(x, sr), r)
            rep["recorder"] = arep
            if out is not None:
                rep["capture"] = "recorder"
                return out, rep
        except subprocess.CalledProcessError as e:
            rep["recorder"] = {"why": "could not decode: " + (e.stderr or b"").decode("utf-8", "replace")[-200:]}
    rep["capture"] = "worklet, drops repaired"
    return fixed, rep


def advice(a, rep):
    tips = []
    if a["clipped"] or a["peak_db"] > -1:
        tips.append(("bad", f"Clipping (peak {a['peak_db']} dBFS). Turn the mic gain down until your loudest words peak near -10."))
    elif a["speech_db"] < -32:
        tips.append(("warn", f"Quiet: speech sits at {a['speech_db']} dBFS. Get closer (two fists from the grille) or raise gain; aim for peaks around -12 to -8."))
    else:
        tips.append(("good", f"Level fine: peaks {a['peak_db']} dBFS, speech {a['speech_db']} dBFS."))
    snr = a["speech_db"] - a["noise_db"]
    after = f" The denoiser takes it to {a['noise_after_db']} dBFS." if "noise_after_db" in a else " The denoiser removes about 12 dB more."
    if a["noise_db"] > -58 or snr < 35:
        tips.append(("warn", f"Room noise at {a['noise_db']} dBFS, only {snr:.0f} dB under your voice. Getting closer to the mic helps most "
                             "(half the distance is 6 dB). Then fans, the PC, aircon: turn off what you can and put something soft behind you." + after))
    else:
        tips.append(("good", f"Room quiet: noise {a['noise_db']} dBFS, {snr:.0f} dB under your voice."))
    wd = rep.get("worklet_drops", 0)
    if rep.get("capture") == "recorder":
        tips.append(("good", f"Clean capture: the browser's live path dropped {wd} times, the recorder path is used and has none."))
    elif wd:
        tips.append(("warn", f"The browser dropped samples {wd} times and the recorder copy could not be used "
                             f"({rep.get('recorder', {}).get('why', 'not recorded')}). Drops are repaired, but close heavy apps while recording."))
    return tips


def save_take(d, data, q):
    wav, rec, mp = split_parts(data, q)
    audio, rep = clean_capture(wav, rec)
    v = d / "voice"
    v.mkdir(exist_ok=True)
    base = "take_" + time.strftime("%Y%m%d_%H%M%S")
    tid, k = base, 1
    while (v / f"{tid}.wav").exists():   # two takes in one second must not overwrite
        k += 1
        tid = f"{base}_{k}"
    write_wav(v / f"{tid}.wav", audio)
    raw = v / "_raw"
    raw.mkdir(exist_ok=True)
    (raw / f"{tid}.worklet.wav").write_bytes(wav)
    if rec:
        (raw / f"{tid}.rec.webm").write_bytes(rec)
    a = analyze(audio, SR)
    meta = {"id": tid, "file": f"{tid}.wav", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rate": float(q.get("rate", 1)), "video_start": float(q["video_start"]), "video_end": float(q["video_end"]),
            "lead_in": float(q["lead_in"]), "range_start": float(q.get("range_start", q["video_start"])),
            "range_end": float(q["range_end"]) if q.get("range_end") else None, "shot": q.get("shot") or None,
            "sample_rate": SR, "device": q.get("device", ""), "sync": q.get("sync", ""), "clean": True,
            **a, **rep, "tips": advice(a, rep), "map": mp}
    (v / f"{tid}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    meta["segs"] = segments(meta, meta["rate"])
    return {k: x for k, x in meta.items() if k != "map"}


def mic_check(data, q):
    """The page asks for ~2 s of silence, then ~5 s of talking. Room tone is measured on the silence."""
    wav, rec, _ = split_parts(data, q)
    audio, rep = clean_capture(wav, rec)
    quiet = float(q.get("quiet", 2.0))
    room, talk = audio[int(0.3 * SR):int((quiet - 0.1) * SR)], audio[int((quiet + 0.3) * SR):]
    a = analyze(talk, SR)
    rms = lambda z: round(float(20 * np.log10(np.sqrt(np.mean(z.astype(np.float64) ** 2)) + 1e-12)), 1)
    if len(room) > SR // 2:
        a["noise_db"] = rms(room)
        a["noise_after_db"] = rms(denoise(audio, room)[int(0.3 * SR):int((quiet - 0.1) * SR)])
    return {**a, **rep, "tips": advice(a, rep)}


# ---------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    server_version = "VoiceBooth/2"

    def log_message(self, fmt, *args):
        if "/api/" in self.path and not self.path.startswith("/api/pieces"):
            sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {fmt % args}\n")

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, ctype):
        size = path.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            if a:
                start = int(a)
                end = int(b) if b else size - 1
            else:
                start = max(size - int(b), 0)
            end = min(end, size - 1)
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                pass

    def body(self):
        n = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(n) if n else b""

    def do_GET(self):
        u = urlparse(self.path)
        parts = [unquote(p) for p in u.path.strip("/").split("/") if p]
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if not parts:
                return self.send_file(HERE / "index.html", "text/html; charset=utf-8")
            if parts == ["recorder-worklet.js"]:
                return self.send_file(HERE / "recorder-worklet.js", "text/javascript")
            if parts == ["api", "pieces"]:
                return self.send_json([{"name": d.name, "title": title_of(d), "duration": timeline(d)["duration"],
                                        "aligned": (d / "words.json").exists(), "takes": len(takes(d))}
                                       for d in pieces()])
            if len(parts) == 3 and parts[:2] == ["api", "piece"]:
                d = piece_dir(parts[2])
                with _align_lock:
                    w = align.align(d)
                return self.send_json({"name": d.name, "title": title_of(d), **w,
                                       "takes": takes(d), "builds": builds(d), "folder": str(d / "voice")})
            if len(parts) == 3 and parts[:2] == ["api", "map"]:
                d = piece_dir(parts[2])
                return self.send_json({t["id"]: segments(t, t.get("rate", 1.0)) for t in takes(d, with_map=True)})
            if len(parts) >= 3 and parts[0] == "media":
                d = piece_dir(parts[1])
                if parts[2] == "picture":
                    return self.send_file(d / "_assemble" / "_silent.mp4", "video/mp4")
                if parts[2] == "guide":
                    return self.send_file(d / timeline(d).get("audio", "narration.mp3"), "audio/mpeg")
                if parts[2] == "take" and len(parts) == 4:
                    t = {x["id"]: x for x in takes(d)}[parts[3]]
                    preset = q.get("preset", "raw")
                    if preset not in PRESETS:
                        return self.send_error(400)
                    if preset == "raw" and t.get("clean"):
                        return self.send_file(d / "voice" / t["file"], "audio/wav")
                    take_audio(d, t, preset)
                    return self.send_file(d / "voice" / "_cache" / f"{t['id']}.{preset}.wav", "audio/wav")
                if parts[2] == "voice" and len(parts) == 4 and re.fullmatch(r"[\w.-]+\.(wav|mp4|srt)", parts[3]):
                    p = d / "voice" / parts[3]
                    return self.send_file(p, {".mp4": "video/mp4", ".srt": "text/plain; charset=utf-8"}.get(p.suffix, "audio/wav"))
            self.send_error(404)
        except KeyError:
            self.send_error(404, "no such piece or take")
        except FileNotFoundError:
            self.send_error(404)
        except subprocess.CalledProcessError as e:
            self.send_json({"error": (e.stderr or b"").decode("utf-8", "replace")[-800:] or str(e)}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        parts = [unquote(p) for p in u.path.strip("/").split("/") if p]
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if len(parts) == 3 and parts[:2] == ["api", "take"]:
                d = piece_dir(parts[2])
                return self.send_json(save_take(d, self.body(), q))
            if parts == ["api", "miccheck"]:
                return self.send_json(mic_check(self.body(), q))
            if len(parts) == 3 and parts[:2] == ["api", "discard"]:
                d = piece_dir(parts[2])
                tid = json.loads(self.body())["id"]
                if not re.fullmatch(r"take_\d{8}_\d{6}(_\d+)?", tid):
                    return self.send_json({"error": "bad id"}, 400)
                bin_ = d / "voice" / "_discarded"
                bin_.mkdir(parents=True, exist_ok=True)
                for p in [d / "voice" / f"{tid}.wav", d / "voice" / f"{tid}.json",
                          *(d / "voice" / "_raw").glob(f"{tid}.*"), *(d / "voice" / "_cache").glob(f"{tid}.*")]:
                    if p.exists():
                        shutil.move(str(p), str(bin_ / p.name))
                return self.send_json({"ok": True})
            if len(parts) == 3 and parts[:2] == ["api", "build"]:
                d = piece_dir(parts[2])
                req = json.loads(self.body())
                return self.send_json(build(d, req["ids"], float(req.get("nudge_ms", 0)), float(req.get("rate", 1.0)),
                                            req.get("preset", "broadcast"), req.get("motion", "blend")))
            if len(parts) == 3 and parts[:2] == ["api", "captions"]:
                d = piece_dir(parts[2])
                req = json.loads(self.body())
                if not re.fullmatch(r"[\w.-]+_voiced_[\w.-]+\.mp4", req["mp4"]):
                    return self.send_json({"error": "bad build name"}, 400)
                with _caption_lock:
                    return self.send_json(captions.caption(d, req["mp4"], req.get("font", captions.DEFAULT_FONT)))
            if len(parts) == 3 and parts[:2] == ["api", "realign"]:
                d = piece_dir(parts[2])
                with _align_lock:
                    w = align.align(d, force=True)
                return self.send_json({"method": w["method"]})
            if len(parts) == 3 and parts[:2] == ["api", "open"]:
                d = piece_dir(parts[2])
                (d / "voice").mkdir(exist_ok=True)
                os.startfile(str(d / "voice"))
                return self.send_json({"ok": True})
            self.send_error(404)
        except KeyError as e:
            self.send_json({"error": f"missing {e}"}, 400)
        except FileNotFoundError as e:
            self.send_json({"error": f"not found: {e}"}, 404)
        except ValueError as e:
            self.send_json({"error": str(e)}, 400)
        except subprocess.CalledProcessError as e:
            self.send_json({"error": (e.stderr or b"").decode("utf-8", "replace")[-800:] or str(e)}, 500)


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Voice Booth on http://127.0.0.1:{PORT}  (Ctrl+C or close this window to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
