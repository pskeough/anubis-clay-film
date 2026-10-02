"""Burned-in captions for a voiced build. Text is the script, never a transcription.

python captions.py PIECE_DIR VOICED_MP4_NAME [FONT_KEY]

Timing. faster-whisper small.en (GPU when available) runs on the build's voice WAV for word times only.
Its words are matched to the script with difflib; a matched script word takes the heard time, and
unmatched script words are spread by character length between matched neighbours. Whatever whisper
thought it heard never reaches the screen. guide_cues() is the no-transcription alternative (guide
read plus a voice-onset lag); it was too loose to use, see its docstring.

Cues are at most MAX_CHARS, broken at sentence ends, scene ends and, once half full, at commas. The ASS
is rendered by libass (imageio-ffmpeg ships it); the picture is re-encoded at CRF 17 with the audio
copied. The .srt is the same cues for YouTube's caption upload, if wanted.
"""
import difflib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import align

MAX_CHARS = 42
MAX_DUR = 4.5
MIN_DUR = 0.8
HOLD = 0.3
SR = 16000
FONTS = {   # key: (family, file, size as fraction of frame height, bold)
    "lato": ("Lato", "LatoWeb-Bold.ttf", 0.060, True),
    "garamond": ("Garamond", "GARABD.TTF", 0.066, True),
    "georgia": ("Georgia", "georgiab.ttf", 0.058, True),
    "segoe": ("Segoe UI Semibold", "seguisb.ttf", 0.060, False),
}
DEFAULT_FONT = "garamond"   # Patrick's blind pick, 2026-10-02
FONTS_DIR = __import__("os").environ.get("CAPTION_FONTS_DIR", r"C:\Windows\Fonts")


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def voice_wav(d: Path, mp4_name: str):
    m = re.fullmatch(rf"{re.escape(d.name)}_voiced_(.+)\.mp4", mp4_name)
    if not m:
        raise ValueError(f"not a voiced build: {mp4_name}")
    tag = m.group(1)
    wav = d / "voice" / f"{d.name}_voice_{tag}.wav"
    if not wav.exists():
        raise FileNotFoundError(wav.name)
    r = re.match(r"x(\d+\.\d+)_", tag)
    return wav, float(r.group(1)) if r else 1.0


def onsets(wav: Path):
    """Times where speech starts after at least 0.25 s of quiet, from a 20 ms energy gate."""
    raw = subprocess.run([ffmpeg(), "-v", "error", "-i", str(wav), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32)
    n = SR // 50
    k = len(x) // n
    db = 20 * np.log10(np.sqrt((x[:k * n].reshape(k, n) ** 2).mean(1)) + 1e-9)
    floor, loud = np.percentile(db, 10), np.percentile(db, 95)
    on = db > floor + 0.35 * (loud - floor)
    out, quiet = [], 99
    for i, v in enumerate(on):
        if v and quiet >= 12:
            out.append(i * n / SR)
        quiet = 0 if v else quiet + 1
    return np.array(out)


def group(words):
    out, cur = [], []
    for w in words:
        if cur:
            text = " ".join(x["w"] for x in cur)
            prev = cur[-1]["w"]
            if (len(text) + 1 + len(w["w"]) > MAX_CHARS or w["scene"] != cur[-1]["scene"]
                    or w["t1"] - cur[0]["t0"] > MAX_DUR
                    or re.search(r"[.?!…]['\"’”)]*$", prev)
                    or (re.search(r"[,;:—–]['\"’”)]*$", prev) and len(text) >= MAX_CHARS // 2)):
                out.append(cur)
                cur = []
        cur.append(w)
    out += [cur] if cur else []
    merged = []   # a lone short tail ("denim.") rides on the cue before it, up to 12 chars over
    for g in out:
        tail = " ".join(x["w"] for x in g)
        if merged and len(tail) < 12 and g[0]["scene"] == merged[-1][-1]["scene"] \
                and len(" ".join(x["w"] for x in merged[-1])) + 1 + len(tail) <= MAX_CHARS + 12:
            merged[-1] = merged[-1] + g
        else:
            merged.append(g)
    return merged


def heard_words(wav: Path):
    """[(text, t0, t1)] from faster-whisper small.en, GPU if ctranslate2 sees one. Used for timing only.
    Cached by the WAV's size and mtime."""
    cache = wav.parent / "_cache" / f"{wav.stem}.words.json"
    key = f"{wav.stat().st_size}:{int(wav.stat().st_mtime)}"
    if cache.exists():
        c = json.loads(cache.read_text(encoding="utf-8"))
        if c.get("key") == key:
            return c["words"]
    from faster_whisper import WhisperModel
    try:
        model = WhisperModel("small.en", device="cuda", compute_type="float16")
    except Exception:
        model = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=8)
    segs, _ = model.transcribe(str(wav), language="en", beam_size=5, word_timestamps=True,
                               vad_filter=False, condition_on_previous_text=False)
    words = [(w.word.strip(), round(w.start, 3), round(w.end, 3)) for s in segs for w in (s.words or []) if w.word.strip()]
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps({"key": key, "words": words}), encoding="utf-8")
    return words


def voice_timed_words(d: Path, wav: Path):
    """Script words with t0/t1 on the build's timeline. Words whisper heard as the script word take its
    time; the rest are spread by character length between the matched neighbours."""
    words = [dict(w) for w in align.align(d)["words"]]
    heard = heard_words(wav)
    a = [align.norm(w["w"]) for w in words]
    b = [align.norm(h[0]) for h in heard]
    hit = np.zeros(len(words), bool)
    for blk in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            words[blk.a + k]["t0"], words[blk.a + k]["t1"] = heard[blk.b + k][1], heard[blk.b + k][2]
            hit[blk.a + k] = True
    if hit.sum() < 2:
        raise ValueError("could not match the voice to the script")
    pos = np.cumsum([0] + [len(w["w"]) + 1 for w in words])          # character position of each word
    t0 = np.interp(pos[:-1], pos[:-1][hit], [w["t0"] for w, h in zip(words, hit) if h])
    t1 = np.interp(pos[1:] - 1, pos[1:][hit] - 1, [w["t1"] for w, h in zip(words, hit) if h])
    for i, w in enumerate(words):
        if not hit[i]:
            w["t0"], w["t1"] = float(t0[i]), float(max(t1[i], t0[i] + 0.05))
    return words, int(hit.sum())


def timed_cues(d: Path, wav: Path, rate: float):
    words, matched = voice_timed_words(d, wav)
    cues = []
    groups = group(words)
    for i, g in enumerate(groups):
        s, e = g[0]["t0"], g[-1]["t1"]
        nxt = groups[i + 1][0]["t0"] - 0.04 if i + 1 < len(groups) else e + 2
        end = min(max(e + HOLD, s + MIN_DUR), nxt)
        cues.append((max(s, 0.0), max(end, s + 0.2), " ".join(w["w"] for w in g)))
    return cues, {"matched_words": matched, "script_words": len(words)}


def guide_cues(d: Path, wav: Path, rate: float):
    """The no-transcription timing: guide read plus a lag measured from voice onsets. Kept for the
    timing check; off by more than 0.5 s on 19 of 111 cues on anubis, so not used for builds."""
    words = align.align(d)["words"]
    groups = group(words)
    starts = np.array([g[0]["t0"] / rate for g in groups])
    ends = np.array([g[-1]["t1"] / rate for g in groups])
    # a cue follows a pause if the guide had >= 0.35 s of silence before it
    after_pause = np.r_[True, starts[1:] - ends[:-1] >= 0.35]
    ons = onsets(wav)
    pairs = []
    for i in np.where(after_pause)[0]:
        if len(ons):
            j = np.argmin(np.abs(ons - starts[i]))
            if abs(ons[j] - starts[i]) <= 1.2:
                pairs.append((starts[i], ons[j] - starts[i]))
    if pairs:
        pt, lag = np.array(pairs).T
        sm = np.array([np.median(lag[max(0, i - 2):i + 3]) for i in range(len(lag))])   # one bad pairing can't move the rest
        shift = np.interp(starts, pt, sm)
    else:
        shift = np.zeros(len(starts))
    s, e = starts + shift, ends + shift
    cues = []
    for i, g in enumerate(groups):
        nxt = s[i + 1] - 0.04 if i + 1 < len(groups) else e[i] + 2
        end = min(max(e[i] + HOLD, s[i] + MIN_DUR), nxt)
        cues.append((max(s[i], 0.0), max(end, s[i] + 0.2), " ".join(w["w"] for w in g)))
    return cues, {"anchors": len(pairs), "lag_ms": int(round(np.median(shift) * 1000)) if len(shift) else 0}


def ts_srt(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def ts_ass(t):
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def video_size(path):
    err = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True).stderr.decode("utf-8", "replace")
    m = re.search(r"Video:.*?(\d{3,5})x(\d{3,5})", err)
    return (int(m.group(1)), int(m.group(2))) if m else (1920, 1080)


def write_ass(path, cues, w, h, font=DEFAULT_FONT):
    fam, _, frac, bold = FONTS[font]
    size = round(h * frac)
    outline = max(1.5, round(size * 0.06, 1))
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{fam},{size},&H00F2F2F2,&H00F2F2F2,&H00101010,&H8C000000,{-1 if bold else 0},0,0,0,100,100,0.3,0,1,{outline},{outline},2,{round(w * 0.08)},{round(w * 0.08)},{round(h * 0.06)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    body = [f"Dialogue: 0,{ts_ass(a)},{ts_ass(b)},Cap,,0,0,0,,{{\\blur0.6}}{t.replace('{', '(').replace('}', ')')}"
            for a, b, t in cues]
    path.write_text(head + "\n".join(body) + "\n", encoding="utf-8-sig")


def write_srt(path, cues):
    path.write_text("\n".join(f"{i}\n{ts_srt(a)} --> {ts_srt(b)}\n{t}\n" for i, (a, b, t) in enumerate(cues, 1)),
                    encoding="utf-8")


def burn(v: Path, src: str, ass: str, out: str, extra=()):
    fonts = FONTS_DIR.replace("\\", "/").replace(":", "\\:")
    cmd = [ffmpeg(), "-v", "error", "-y", *extra, "-i", src, "-vf", f"ass={ass}:fontsdir='{fonts}'",
           "-c:v", "libx264", "-crf", "17", "-preset", "fast", "-pix_fmt", "yuv420p",
           "-c:a", "copy", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True, capture_output=True, cwd=str(v))   # relative names: no drive colons to escape


def caption(d: Path, mp4_name: str, font=DEFAULT_FONT):
    d = Path(d)
    v = d / "voice"
    mp4 = v / mp4_name
    if not mp4.exists():
        raise FileNotFoundError(mp4_name)
    if font not in FONTS:
        raise ValueError(f"font must be one of {', '.join(FONTS)}")
    wav, rate = voice_wav(d, mp4_name)
    cues, info = timed_cues(d, wav, rate)
    stem = mp4.stem.replace("_voiced_", "_captioned_")
    ass, srt = v / f"{stem}.ass", v / f"{stem}.srt"
    w, h = video_size(mp4)
    write_ass(ass, cues, w, h, font)
    write_srt(srt, cues)
    out = v / f"{stem}.mp4"
    burn(v, mp4.name, ass.name, out.name)
    return {"mp4": out.name, "srt": srt.name, "path": str(out), "cues": len(cues), "font": font, **info}


if __name__ == "__main__":
    t0 = time.time()
    r = caption(Path(sys.argv[1]), sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else DEFAULT_FONT)
    print(json.dumps(r, indent=1), f"{time.time() - t0:.0f} s")
