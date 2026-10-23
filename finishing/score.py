"""Make a film score with Stable Audio 3 Medium and mix it under a narration.

    python score.py generate JOB.json            score.wav and score_movements.json from movements on the film clock
    python score.py repaint --score score.wav --movements movements.json --name NAME --prompt "..."
                                                  regenerate one movement (or --from/--to seconds) in place
    python score.py cue --prompt "..." --length 14 --out cue.wav
                                                  a standalone cue, e.g. under a title before the narration starts
    python score.py mix --voice FILM.mp4 --score score.wav --out FILM_scored.mp4 [--movements movements.json]
                                                  EQ, level, duck the score under the voice, remux with the picture

generate, repaint and cue need the stable-audio-3 package (SA3_MODEL, default "medium"; the weights come from
Hugging Face into HF_HOME). mix needs only numpy and ffmpeg. Job file schemas are in README.md.

Each generated piece runs TAIL seconds past what is kept and the tail is dropped, because SA3 ends a piece with a
fade. Long scores are built in chunks, each an inpainted continuation of the last CONTEXT seconds before it, and any
stretch that falls quiet (1 s windows under --hole-db) is repainted from the music either side, up to three rounds.
"""
import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from _shared import cap_cpu, load_job, lufs, probe, read_audio, resolve, run_ff, write_wav  # noqa: E402

# generation defaults
CONTEXT, CHUNK, TAIL, SEED, HOLE_DB, MAX_S = 20.0, 340.0, 12.0, 2310, -45.0, 380.0
NEGATIVE = ("vocals, singing, choir, voice, drums, percussion, beat, electronic, synth lead, bright, happy, upbeat, "
            "guitar strumming, silence, pauses, rests, fade out, ending, outro")
# mix defaults
SR = 48000
EQ = ("acompressor=threshold=-30dB:ratio=3:attack=30:release=400,highpass=f=35,"
      "equalizer=f=250:t=o:w=1:g=-3,equalizer=f=2500:t=o:w=1.5:g=-6")


# ---------------------------------------------------------------- generation

class SA3:
    def __init__(self, model_id=None, max_s=MAX_S):
        from stable_audio_3 import StableAudioModel
        self.model = StableAudioModel.from_pretrained(model_id or os.environ.get("SA3_MODEL", "medium"))
        self.sr = getattr(self.model, "sample_rate", 44100)
        self.max_s = max_s

    def gen(self, prompt, duration, negative=NEGATIVE, seed=SEED, inpaint=None, mask=None):
        """(samples, channels) float32. inpaint: audio (samples, channels) to keep outside mask=(start s, end s)."""
        kw = {}
        if inpaint is not None:
            # SA3 takes the inpaint audio as (sr, numpy (samples, channels) float32)
            kw = dict(inpaint_audio=(self.sr, np.ascontiguousarray(inpaint, np.float32)),
                      inpaint_mask_start_seconds=float(mask[0]), inpaint_mask_end_seconds=float(mask[1]))
        # sample_size caps the output length (default 120 s); raise it to the model's ceiling
        a = self.model.generate(prompt=prompt, negative_prompt=negative, seed=int(seed), duration=float(duration),
                                sample_size=int(self.max_s * self.sr), **kw)
        a = a.float().cpu()
        a = a[0] if a.dim() == 3 else a
        return a.numpy().T.astype(np.float32)


def holes(x, sr, hole_db):
    """Sample ranges where 1 s windows sit under hole_db, merged across gaps under 3 s."""
    w = sr
    r = 20 * np.log10(np.sqrt((x[:len(x) // w * w].reshape(-1, w, x.shape[1]) ** 2).mean((1, 2))) + 1e-9)
    out = []
    for i in np.where(r < hole_db)[0]:
        if out and i - out[-1][1] <= 3:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return [(a * w, b * w) for a, b in out]


def splice(score, a, b, fill, xf):
    """Lay fill over score[a:b] with xf-sample ramps at both ends."""
    n = b - a
    fill = np.pad(fill, ((0, max(0, n - len(fill))), (0, 0)))[:n]
    ramp = np.ones(n, np.float32)
    xf = min(xf, n // 2)
    if xf:
        ramp[:xf], ramp[-xf:] = np.linspace(0, 1, xf), np.linspace(1, 0, xf)
    score[a:b] = score[a:b] * (1 - ramp[:, None]) + fill * ramp[:, None]


def apply_stops(score, sr, stops, fade_out):
    """Dead stops (music cut to silence for a moment) and the closing fade."""
    env = np.ones(len(score), np.float32)
    for s in stops:
        i = int(s["at"] * sr)
        down, hold, back = (int(s.get(k, d) * sr) for k, d in (("fade", 0.15), ("length", 1.5), ("return", 2.5)))
        env[max(0, i - down):i] = np.linspace(1, 0, i - max(0, i - down))
        env[i:i + hold] = 0
        seg = env[i + hold:i + hold + back]
        env[i + hold:i + hold + back] = np.linspace(0, 1, back)[:len(seg)]
    if fade_out:
        k = min(len(env), int(fade_out * sr))
        env[-k:] *= np.linspace(1, 0, k)
    return score * env[:, None]


def spans_of(job):
    """Movements with their end times: the next movement's start, the last one's the score length."""
    mv = sorted(job["movements"], key=lambda m: m["start"])
    pal = job.get("palette", "")
    out = []
    for k, m in enumerate(mv):
        end = mv[k + 1]["start"] if k + 1 < len(mv) else job["duration"]
        out.append({"name": m["name"], "start": float(m["start"]), "end": float(end),
                    "prompt": m["prompt"].replace("{palette}", pal), "level": float(m.get("level", 0)),
                    "seed": m.get("seed")})
    if out and out[0]["start"] > 0:
        out[0]["start"] = 0.0
    return out


def cmd_generate(a):
    job, base = load_job(a.job)
    out = resolve(base, job.get("out", "score.wav"))
    ctx_s, chunk, tail = job.get("context", CONTEXT), job.get("chunk", CHUNK), job.get("tail", TAIL)
    seed, neg = job.get("seed", SEED), job.get("negative", NEGATIVE)
    hole_db, rounds = job.get("hole_db", HOLE_DB), job.get("repair_rounds", 3)
    max_s = job.get("max_seconds", MAX_S)
    if chunk + ctx_s + tail > max_s:
        raise SystemExit(f"chunk + context + tail = {chunk + ctx_s + tail} s is over the model's {max_s} s")
    spans = spans_of(job)
    t0 = time.time()
    m = SA3(job.get("model"), max_s)
    sr = m.sr
    score = None
    for sp in spans:
        n = max(1, int(math.ceil((sp["end"] - sp["start"]) / chunk)))
        step = (sp["end"] - sp["start"]) / n
        for j in range(n):
            s = sp["seed"] if sp["seed"] is not None else seed
            if score is None:
                new = m.gen(sp["prompt"], step + tail, neg, s)[:int(step * sr)]
            else:
                c = min(ctx_s, len(score) / sr)
                ctx = score[-int(c * sr):]
                got = m.gen(sp["prompt"], c + step + tail, neg, s, inpaint=ctx, mask=(c, c + step + tail))
                ic = len(ctx)
                new = got[ic:ic + int(step * sr)]
                xf = min(int(0.5 * sr), ic)                     # half-second crossfade at the seam
                ramp = np.linspace(0, 1, xf)[:, None]
                score[-xf:] = score[-xf:] * (1 - ramp) + got[ic - xf:ic] * ramp
            score = new if score is None else np.concatenate([score, new])
            print(f"{sp['name']} {j + 1}/{n}: {len(score) / sr:7.1f}s  ({time.time() - t0:.0f}s)", flush=True)
    L = int(job["duration"] * sr)
    score = np.pad(score, ((0, max(0, L - len(score))), (0, 0)))[:L]
    for rnd in range(rounds):
        hs = holes(score, sr, hole_db)
        print(f"repair pass {rnd + 1}: {len(hs)} holes", [round(x / sr) for x, _ in hs], flush=True)
        if not hs:
            break
        for k, (ha, hb) in enumerate(hs):
            ma, mb = max(0, ha - 2 * sr), min(L, hb + 2 * sr)
            wa, wb = max(0, ma - 40 * sr), min(L, mb + 20 * sr)
            if (wb - wa) / sr > max_s:
                print(f"  hole at {ha / sr:.0f}s is too long to repaint in one window; left as is", flush=True)
                continue
            prompt = next(sp["prompt"] for sp in spans if sp["start"] * sr <= ha < sp["end"] * sr)
            got = m.gen(prompt, (wb - wa) / sr, neg, seed + 100 * (rnd + 1) + k, inpaint=score[wa:wb],
                        mask=((ma - wa) / sr, (mb - wa) / sr))
            splice(score, ma, mb, got[ma - wa:mb - wa], int(0.3 * sr))
    stops, fade_out = job.get("stops", []), job.get("fade_out", 6.0)
    score = apply_stops(score, sr, stops, fade_out)
    score = score / max(1e-6, float(np.abs(score).max())) * 0.95
    write_wav(out, score, sr)
    meta = {"sr": sr, "duration": job["duration"], "stops": stops, "fade_out": fade_out,
            "movements": [{k: sp[k] for k in ("name", "start", "end", "level", "prompt")} for sp in spans]}
    mpath = out.with_name(out.stem + "_movements.json") if a.movements is None else Path(a.movements)
    mpath.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{out} {len(score) / sr:.1f}s, {mpath}, total {time.time() - t0:.0f}s")


def cmd_repaint(a):
    meta = json.loads(Path(a.movements).read_text(encoding="utf-8")) if a.movements else {"stops": [], "movements": []}
    if a.name:
        mv = next((m for m in meta["movements"] if m["name"] == a.name), None)
        if not mv:
            raise SystemExit(f"no movement {a.name!r} in {a.movements}")
        s0, s1 = mv["start"], mv["end"]
    else:
        s0, s1 = a.from_s, a.to_s
    if s0 is None or s1 is None:
        raise SystemExit("give --name (with --movements) or --from and --to")
    m = SA3(a.model, a.max_seconds)
    sr = m.sr
    score = read_audio(a.score, sr, channels=2)
    room = a.max_seconds - 2 * a.context
    parts = a.parts or max(1, int(math.ceil((s1 - s0) / room)))
    edges = np.linspace(s0, s1, parts + 1)
    for p0, p1 in zip(edges[:-1], edges[1:]):
        w0, w1 = max(0, int((p0 - a.context) * sr)), min(len(score), int((p1 + a.context) * sr))
        i0, i1 = int(p0 * sr), int(p1 * sr)
        got = m.gen(a.prompt, (w1 - w0) / sr, a.negative, a.seed, inpaint=score[w0:w1],
                    mask=((i0 - w0) / sr, (i1 - w0) / sr))
        splice(score, i0, i1, got[i0 - w0:i1 - w0], int(0.5 * sr))
        print(f"repainted {p0:.1f}-{p1:.1f}s", flush=True)
    # the closing fade is already in the score; only the stops inside the repainted range are cut again
    inside = [s for s in meta.get("stops", []) if s0 - 3 <= s["at"] <= s1 + 3]
    score = apply_stops(score, sr, inside, 0)
    score = score / max(1e-6, float(np.abs(score).max())) * 0.95
    out = Path(a.out) if a.out else Path(a.score)
    if out.resolve() == Path(a.score).resolve():
        bak = out.with_name(out.stem + "_before_repaint.wav")
        shutil.copy(out, bak)
        print("previous score kept as", bak)
    write_wav(out, score, sr)
    print(out, "written")


def cmd_cue(a):
    m = SA3(a.model)
    sr = m.sr
    got = m.gen(a.prompt, max(a.generate, a.length), a.negative, a.seed)
    cue = got[:int(a.length * sr)]
    k = int(0.05 * sr)
    cue[:k] *= np.linspace(0, 1, k)[:, None]                     # 50 ms fade-in against a click
    write_wav(a.out, cue, sr)
    w = sr // 2
    print(f"{a.out} {len(cue) / sr:.2f}s; 0.5 s RMS dBFS:",
          [round(float(20 * np.log10(np.sqrt((cue[i:i + w] ** 2).mean()) + 1e-9)), 1) for i in range(0, len(cue) - w, w)])


# ---------------------------------------------------------------- mix

def rms_db(x):
    return 10 * np.log10(float((x ** 2).mean()) + 1e-12)


def running_mean(x, k):
    """Centred running mean over k samples, by cumulative sum (a direct convolve over a film is far too slow)."""
    c = np.concatenate([[0], np.cumsum(np.pad(x, (k // 2, k - k // 2), mode="edge"), dtype=np.float64)])
    return ((c[k:] - c[:-k]) / k)[:len(x)].astype(np.float32)


def level_music(music, meta, sr, leveller=True):
    """Slow leveller around each movement's median, then each movement to its planned level with 4 s glides."""
    spans = meta.get("movements") or [{"name": "all", "start": 0.0, "end": len(music) / sr, "level": 0.0}]
    if leveller:
        # half the swells and dips around each movement's median level (3 s window, 2.5 s glide), leaving the
        # dead stops and the closing fade as the score has them
        fh = sr // 2
        nfm = len(music) // fh
        fe = 10 * np.log10(np.convolve((music[:nfm * fh].reshape(nfm, fh) ** 2).mean(1), np.ones(6) / 6, "same")
                           + 1e-12)
        lg = np.zeros(nfm, np.float32)
        for sp in spans:
            f0, f1 = int(sp["start"] * 2), min(nfm, int(sp["end"] * 2))
            if f1 > f0:
                lg[f0:f1] = np.clip(0.5 * (np.median(fe[f0:f1]) - fe[f0:f1]), -6, 8)
        for s in meta.get("stops", []):
            lg[max(0, int((s["at"] - 1) * 2)):int((s["at"] + 5) * 2)] = 0
        lg[max(0, nfm - int((meta.get("fade_out", 6.0) + 2) * 2)):] = 0
        lg = np.convolve(np.pad(lg, 2, mode="edge"), np.ones(5) / 5, "valid")
        music = music * 10 ** (np.interp(np.arange(len(music)), np.arange(nfm) * fh + fh / 2, lg) / 20).astype(np.float32)
    ref = rms_db(music)
    lev = np.zeros(len(music), np.float32)
    for sp in spans:
        i0, i1 = int(sp["start"] * sr), min(len(music), int(sp["end"] * sr))
        if i1 > i0:
            lev[i0:i1] = ref + sp.get("level", 0.0) - rms_db(music[i0:i1])
    lev = running_mean(lev, 4 * sr)
    return music * 10 ** (lev / 20)


def duck_gain(voice, sr, duck_db, vad_db, lookahead, attack, release):
    """Per-sample gain that dips duck_db while the voice speaks: activity on 10 ms frames, gaps under 150 ms
    bridged, look-ahead so the dip starts before a line, attack/release smoothing. Also returns the activity."""
    hop = sr // 100
    nf = len(voice) // hop
    vrms = np.sqrt((voice[:nf * hop].reshape(nf, hop) ** 2).mean(1) + 1e-12)
    act = (20 * np.log10(vrms) > vad_db).astype(np.float32)
    act = (np.convolve(act, np.ones(15), "same") > 0).astype(np.float32)
    la = int(round(lookahead * 100))
    if la:
        act = np.concatenate([act[la:], np.zeros(la, np.float32)])
    a_c, r_c = np.exp(-1 / (attack * 100)), np.exp(-1 / (release * 100))
    env = np.zeros(nf, np.float32)
    e = 0.0
    for i in range(nf):
        c = a_c if act[i] > e else r_c
        e = c * e + (1 - c) * act[i]
        env[i] = e
    gain = 10 ** (-duck_db * env / 20)
    gain = np.interp(np.arange(len(voice)), np.arange(nf) * hop + hop / 2, gain).astype(np.float32)
    speech = np.repeat(act > 0, hop)
    speech = np.pad(speech, (0, len(voice) - len(speech)))
    return gain, speech


def cmd_mix(a):
    cap_cpu(a.threads)
    work = Path(tempfile.mkdtemp(prefix="score_mix_"))
    meta = json.loads(Path(a.movements).read_text(encoding="utf-8")) if a.movements else {}
    voice = read_audio(a.voice, SR)
    music = read_audio(a.score, SR, af=a.eq or None)
    music = level_music(music, meta, SR, leveller=not a.no_leveller)
    write_wav(work / "voice.wav", voice, SR)
    write_wav(work / "music_lev.wav", music, SR)
    v_lufs, m_lufs = lufs(work / "voice.wav"), lufs(work / "music_lev.wav")
    music *= 10 ** ((v_lufs - a.bed - m_lufs) / 20)                # the bed: --bed LU under the voice's loudness
    gain, speech = duck_gain(voice, SR, a.duck, a.vad_db, a.lookahead, a.attack, a.release)

    off = int(round(a.offset * SR))
    bed = np.zeros_like(voice)
    n = max(0, min(len(music), len(voice) - off))
    bed[off:off + n] = music[:n]
    if a.cue:
        # the cue plays --intro dB over the bed before the score starts, glides down to the bed over the last
        # --glide s, then crossfades into the score over its first --cue-xfade s
        if off <= 0:
            raise SystemExit("--cue needs --offset (where the score starts)")
        cue = read_audio(a.cue, SR)
        g, xs = int(a.glide * SR), int(a.cue_xfade * SR)
        cue = cue[:off + xs]
        cue = np.pad(cue, (0, off + xs - len(cue)))
        cue = cue * 10 ** ((rms_db(music[:60 * SR]) - rms_db(cue[:off])) / 20)
        glide = np.full(len(cue), a.intro, np.float32)
        g0 = max(0, off - g)
        glide[g0:off] = np.linspace(a.intro, 0, off - g0)
        glide[off:] = 0
        xf = np.ones(len(cue), np.float32)
        xf[off:] = np.linspace(1, 0, len(cue) - off)
        m_end = min(len(bed), off + xs)
        bed[off:m_end] *= 1 - xf[off:off + (m_end - off)]
        bed[:m_end] += (cue * 10 ** (glide / 20) * xf)[:m_end]
    bed *= gain
    mix = voice + bed
    write_wav(work / "mix.wav", mix, SR)
    write_wav(work / "bed.wav", bed, SR)

    scored = np.zeros(len(voice), bool)
    scored[off:off + n] = True
    sp, gp = speech & scored, ~speech & scored
    under = rms_db(voice[sp]) - rms_db(bed[sp]) if sp.any() else float("nan")
    gaps = rms_db(voice[gp]) - rms_db(bed[gp]) if gp.any() else float("nan")
    print(f"voice {v_lufs:.1f} LUFS, bed {lufs(work / 'bed.wav'):.1f} LUFS, mix {lufs(work / 'mix.wav'):.1f} LUFS, "
          f"peak {float(np.abs(mix).max()):.2f}")
    print(f"music sits {under:.1f} dB under the voice while it speaks; in the gaps voice minus music is {gaps:.1f} dB")

    out = Path(a.out)
    limiter = f"alimiter=limit={a.limit}:level=disabled"           # peaks are caught here; scaling the whole mix to
    if out.suffix.lower() in (".wav", ".flac"):                     # one peak would drop the voice
        run_ff("-i", work / "mix.wav", "-af", limiter, "-c:a", "pcm_f32le" if out.suffix.lower() == ".wav" else "flac",
               out)
    else:
        video = a.video or a.voice
        if probe(video)[0] is None:
            raise SystemExit(f"{video} has no picture: give --video, or write a .wav")
        run_ff("-i", video, "-i", work / "mix.wav", "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-af", limiter,
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out)
    if a.stems:
        Path(a.stems).mkdir(parents=True, exist_ok=True)
        for f in ("bed.wav", "mix.wav"):
            shutil.copy(work / f, Path(a.stems) / f)
    shutil.rmtree(work, ignore_errors=True)
    print(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a score from a job file")
    g.add_argument("job")
    g.add_argument("--movements", help="where to write the movements file (default: next to the score)")

    r = sub.add_parser("repaint", help="regenerate one stretch of an existing score")
    r.add_argument("--score", required=True)
    r.add_argument("--movements", help="movements file written by generate")
    r.add_argument("--name", help="movement to repaint")
    r.add_argument("--from", dest="from_s", type=float, help="start of the stretch in seconds (instead of --name)")
    r.add_argument("--to", dest="to_s", type=float, help="end of the stretch in seconds")
    r.add_argument("--prompt", required=True)
    r.add_argument("--negative", default=NEGATIVE)
    r.add_argument("--seed", type=int, default=SEED + 67)
    r.add_argument("--context", type=float, default=CONTEXT, help="seconds kept on each side of each window (20)")
    r.add_argument("--parts", type=int, default=None, help="windows to split the stretch into (default: as few as fit)")
    r.add_argument("--out", help="output file (default: overwrite --score, keeping a copy of the old one)")
    r.add_argument("--model", default=None)
    r.add_argument("--max-seconds", type=float, default=MAX_S)

    c = sub.add_parser("cue", help="generate a standalone cue")
    c.add_argument("--prompt", required=True)
    c.add_argument("--length", type=float, required=True,
                   help="seconds kept; for a title cue, the title's length plus the mix's --cue-xfade")
    c.add_argument("--generate", type=float, default=20.0, help="seconds generated before trimming (20)")
    c.add_argument("--negative", default="vocals, singing, choir, voice, drums, percussion, beat, electronic, "
                                         "synth lead, bright, happy, upbeat, guitar, silence, fade in")
    c.add_argument("--seed", type=int, default=4411)
    c.add_argument("--out", required=True)
    c.add_argument("--model", default=None)

    m = sub.add_parser("mix", help="mix a score under a narration")
    m.add_argument("--voice", required=True, help="the narration (a WAV, or the film whose audio is the voice)")
    m.add_argument("--score", required=True)
    m.add_argument("--out", required=True, help=".mp4/.mov/.mkv remuxes the picture; .wav/.flac writes audio only")
    m.add_argument("--video", help="picture to remux (default: --voice)")
    m.add_argument("--movements", help="movements file from generate: per-movement levels, stops, fade")
    m.add_argument("--offset", type=float, default=0.0, help="seconds into the voice track where the score starts")
    m.add_argument("--bed", type=float, default=10.0, help="score loudness, LU under the voice's (default 10)")
    m.add_argument("--duck", type=float, default=4.0, help="extra dB down while the voice speaks (default 4)")
    m.add_argument("--cue", help="cue to play before the score starts (needs --offset)")
    m.add_argument("--intro", type=float, default=12.0, help="cue level, dB over the bed (default 12)")
    m.add_argument("--glide", type=float, default=2.0, help="seconds the cue glides down to the bed (2)")
    m.add_argument("--cue-xfade", type=float, default=2.0, help="seconds the cue crossfades into the score (2)")
    m.add_argument("--vad-db", type=float, default=-42.0, help="voice activity threshold, dBFS RMS per 10 ms (-42)")
    m.add_argument("--lookahead", type=float, default=0.15, help="seconds the duck starts before speech (0.15)")
    m.add_argument("--attack", type=float, default=0.25, help="duck attack, seconds (0.25)")
    m.add_argument("--release", type=float, default=1.5, help="duck release, seconds (1.5)")
    m.add_argument("--eq", default=EQ, help="ffmpeg filter chain on the score before levelling ('' for none)")
    m.add_argument("--no-leveller", action="store_true", help="skip the slow leveller")
    m.add_argument("--limit", type=float, default=0.95, help="limiter ceiling on the final mix (0.95)")
    m.add_argument("--stems", help="folder to keep bed.wav and mix.wav in")
    m.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    {"generate": cmd_generate, "repaint": cmd_repaint, "cue": cmd_cue, "mix": cmd_mix}[a.cmd](a)


if __name__ == "__main__":
    main()
