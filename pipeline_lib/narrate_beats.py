"""Stage 1 for director pieces - ONE narration take, then hang the shots on it.

    <podcast venv python> narrate_beats.py <piece_dir> [--device 1] [--per-beat]

Two bugs are fixed here and they are different bugs.

ONE VOICE. Qwen3-TTS VoiceDesign reports `supports_speaker_reference=false`:
there is no way to pin a timbre with a reference clip, because the voice is
sampled afresh from the prose description on every request. Synthesising beat
by beat therefore draws a new speaker eight times, and the last cut audibly
changed voice between segments. There is no seed that fixes this, because the
requests differ in their text as well.

So the passage is spoken in a SINGLE request. One request, one sampling of the
voice, one speaker for the whole film. `--per-beat` restores the old behaviour
for a passage too long to fit one call, and says loudly that it is doing so.

The cost is that a single take has no beat boundaries in it. They are recovered
by looking for the pauses the model already puts at punctuation: a proportional
prior from the character counts says roughly where each boundary should fall,
and the nearest real silence inside a window of that is taken as the cut. A
proportional split alone would drift; a silence search alone would happily pick
a dramatic mid-sentence pause. `supports_timestamps=false`, and there is no
alignment model on this machine, so this is the available method.

ONE TAKE PER BEAT, NOT PER SHOT. The director makes more shots than beats and
several shots can cover one beat. `s1_narrate.py` synthesises `shot["narration"]`
and duly spoke three sentences twice in a row - 187 words of audio for a
127-word passage. The audio spine is the beat list, which is the writing; shots
are laid over it, and a beat covered by k shots splits its window k ways.

LONG PASSAGES (2026-09-24). A single take of 3.8k characters (Anubis, ~290 s)
ran the 16 GB card out of memory allocating the talker step graph. Letting the
engine chunk internally fits, but it re-draws the voice at chunk boundaries:
the first ~90 s came out ~20 Hz higher in median pitch and 4.7-5.4 dB off the
rest of the take in long-term spectrum, with the step at the boundary. So a
passage over MAX_TAKE_CHARS, or any piece given `--voice-ref-json`, is spoken
by the Qwen3-TTS Base model in clone mode: several takes, each cloned from the
same reference clip, joined at beat boundaries. Without a given reference, one
is made first by VoiceDesign from the opening beats, so the voice is still the
design's.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import os  # noqa: E402
sys.path.insert(0, os.environ.get("CLAY_FILM_TTS_ENGINE") or str(Path(__file__).resolve().parent.parent / "tts_engine"))

import numpy as np  # noqa: E402

import s1_narrate as S  # noqa: E402  (CLI paths, DESIGN, SAMPLE_RATE)

SR = S.SAMPLE_RATE
LEAD_S, TAIL_S = 0.8, 1.6
MIN_PAUSE_S = 0.09        # shorter than this is articulation, not a boundary
SNAP_WINDOW_S = 2.0       # how far from the proportional guess to look
BASE = Path(os.environ.get("QWEN3TTS_BASE_GGUF", "models/qwen3-tts-12hz-1.7b-base-q8_0.gguf"))
MAX_TAKE_CHARS = 1400     # one take this long fits the 16 GB card; 3.8k did not
TAKE_GAP_S = 0.55         # silence between cloned takes, a paragraph-sized breath
REF_CHARS = 260           # opening text VoiceDesign speaks to make a reference


def beat_text(b) -> str:
    return b if isinstance(b, str) else b.get("text", "")


def synth_one(text: str, design: str, device: int, scratch: Path,
              seed: int = 20260917) -> np.ndarray:
    """The whole passage, one request, one voice."""
    import wave

    scratch.mkdir(parents=True, exist_ok=True)
    out = scratch / "one.wav"
    out.unlink(missing_ok=True)
    r = subprocess.run(
        [str(S.CLI), "--task", "vdes", "--family", "qwen3_tts",
         "--model", str(S.VOICEDESIGN), "--backend", "cuda",
         "--device", str(device), "--instruct", design,
         "--language", "english", "--seed", str(seed),
         # big enough that the model does not chunk internally: a chunk
         # boundary is another place the voice could be re-drawn
         "--text-chunk-size", str(max(2000, len(text) + 200)),
         # asked for opportunistically. `--inspect` says
         # supports_timestamps=false for this weight, so these usually do not
         # appear; when they do they beat any silence heuristic and
         # `segments_of()` picks them up.
         "--segments-out", str(scratch / "segments.json"),
         "--text", text, "--out", str(out), "--metrics"],
        capture_output=True, text=True, timeout=7200)
    if not out.exists():
        tail = (r.stderr or r.stdout).strip().splitlines()[-8:]
        raise SystemExit("single-take synthesis produced no file:\n  "
                         + "\n  ".join(tail))
    return read_wav(out)


def read_wav(path: Path) -> np.ndarray:
    """Mono float32 at SR."""
    import wave

    with wave.open(str(path)) as w:
        pcm = (np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
               .astype(np.float32) / 32767.0)
        if w.getnchannels() > 1:
            pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1)
        sr = w.getframerate()
    if sr != SR:
        idx = np.linspace(0, len(pcm) - 1, int(len(pcm) * SR / sr))
        pcm = np.interp(idx, np.arange(len(pcm)), pcm).astype(np.float32)
    return pcm


def synth_clone(text: str, ref_wav: Path, ref_text: str, device: int,
                out: Path, seed: int = 20260917) -> np.ndarray:
    """One take in the reference's voice (Qwen3-TTS Base, clone mode)."""
    out.unlink(missing_ok=True)
    r = subprocess.run(
        # the Base model refuses `--task clon` ("only supports the Tts task"); a voice
        # reference on the plain tts task is how it clones
        [str(S.CLI), "--task", "tts", "--family", "qwen3_tts",
         "--model", str(BASE), "--backend", "cuda", "--device", str(device),
         "--voice-ref", str(ref_wav), "--reference-text", ref_text,
         "--language", "english", "--seed", str(seed),
         "--text-chunk-size", str(max(2000, len(text) + 200)),
         "--text", text, "--out", str(out), "--metrics"],
        capture_output=True, text=True, timeout=7200)
    if not out.exists():
        tail = (r.stderr or r.stdout).strip().splitlines()[-8:]
        raise SystemExit(f"clone take {out.name} produced no file:\n  "
                         + "\n  ".join(tail))
    return read_wav(out)


def make_ref(spec: dict | None, beats: list[str], design: str, device: int,
             scratch: Path) -> tuple[Path, str]:
    """The clip every take is cloned from, and its transcript.
    spec {wav, text, start, end} cuts a given narration; None has VoiceDesign
    speak the opening beats, so the voice is the design's."""
    import narrate

    ref = scratch / "ref.wav"
    if spec:
        pcm = read_wav(Path(spec["wav"]))
        a = int(float(spec.get("start", 0)) * SR)
        b = int(float(spec["end"]) * SR) if spec.get("end") else len(pcm)
        narrate.write_wav(ref, pcm[a:b])
        return ref, " ".join(spec["text"].split())
    words, n = [], 0
    for b in beats:
        words.append(" ".join(b.split()))
        n += len(b)
        if n >= REF_CHARS:
            break
    text = " ".join(words)
    narrate.write_wav(ref, synth_one(text, design, device, scratch))
    return ref, text


def take_groups(beats: list[str], cap: int) -> list[list[int]]:
    """Contiguous runs of beat indices, balanced by characters, each <= cap
    where a single beat allows it. A beat is never split across takes."""
    lens = [len(b) + 1 for b in beats]
    total = sum(lens)
    n = max(1, -(-total // cap))
    while True:
        target, groups, cum = total / n, [[] for _ in range(n)], 0
        for i, L in enumerate(lens):
            groups[min(n - 1, int((cum + L / 2) // target))].append(i)
            cum += L
        groups = [g for g in groups if g]
        over = [g for g in groups if len(g) > 1 and sum(lens[i] for i in g) > cap]
        if not over:
            return groups
        n += 1


def synth_cloned(beats: list[str], ref: Path, ref_text: str, device: int,
                 scratch: Path) -> tuple[np.ndarray, list[float]]:
    """Every take cloned from one reference; returns speech and beat ends."""
    groups = take_groups(beats, MAX_TAKE_CHARS)
    gap = np.zeros(int(TAKE_GAP_S * SR), dtype=np.float32)
    parts, ends, offset = [], [], 0.0
    for k, g in enumerate(groups):
        tb = [" ".join(beats[i].split()) for i in g]
        text = " ".join(tb)
        pcm = synth_clone(text, ref, ref_text, device, scratch / f"take{k:02d}.wav")
        # Takes come back at different levels (Anubis: RMS 0.056, 0.086, 0.074), which is an
        # audible step at each join. Level each take's voiced frames to one RMS.
        voiced = pcm[np.abs(pcm) > 0.02]
        if len(voiced):
            pcm = np.clip(pcm * (0.09 / float(np.sqrt((voiced ** 2).mean()))), -1, 1)
        print(f"  take {k + 1}/{len(groups)}: beats {g[0]}-{g[-1]}, "
              f"{len(text)} chars, {len(pcm) / SR:.1f}s")
        if k:
            parts.append(gap)
            offset += TAKE_GAP_S
        parts.append(pcm)
        ends += [offset + e for e in boundaries(pcm, tb)]
        offset += len(pcm) / SR
    return np.concatenate(parts), ends


def pauses(pcm: np.ndarray, hop: int = 240) -> list[tuple[float, float]]:
    """Silence runs, as (start_s, end_s). Threshold is relative to the take."""
    n = len(pcm) // hop
    rms = np.sqrt(np.array([(pcm[i * hop:(i + 1) * hop] ** 2).mean()
                            for i in range(n)]) + 1e-12)
    # 18% of the median voiced level: low enough to ignore breath, high enough
    # to catch a real stop. Absolute thresholds do not survive a level change.
    thr = 0.18 * float(np.median(rms[rms > np.percentile(rms, 40)]))
    quiet = rms < thr
    runs, i = [], 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) * hop / SR >= MIN_PAUSE_S:
                runs.append((i * hop / SR, j * hop / SR))
            i = j
        else:
            i += 1
    return runs


def segments_of(scratch: Path, n: int) -> list[float] | None:
    """Real segment ends from the engine, if it gave us any."""
    f = scratch / "segments.json"
    if not f.exists():
        return None
    try:
        raw = json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    segs = raw.get("segments", raw) if isinstance(raw, dict) else raw
    ends = [float(s["end"]) for s in segs
            if isinstance(s, dict) and "end" in s]
    return ends if len(ends) == n else None


def boundaries(pcm: np.ndarray, beats: list[str]) -> list[float]:
    """Where each beat ends, in seconds. len == len(beats); last is the end."""
    dur = len(pcm) / SR
    chars = np.cumsum([len(b) for b in beats], dtype=np.float64)
    guess = (chars / chars[-1] * dur)[:-1]          # n-1 interior boundaries
    runs = pauses(pcm)
    mids = [(a + b) / 2 for a, b in runs]
    lens = [b - a for a, b in runs]

    out, floor = [], 0.0
    for g in guess:
        best, best_len = None, 0.0
        for m, L in zip(mids, lens):
            if m <= floor + 0.2 or abs(m - g) > SNAP_WINDOW_S:
                continue
            if L > best_len:                         # prefer the longer stop
                best, best_len = m, L
        chosen = best if best is not None else max(g, floor + 0.2)
        out.append(chosen)
        floor = chosen
    out.append(dur)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("piece", type=Path)
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--design", default=S.DESIGN)
    ap.add_argument("--per-beat", action="store_true",
                    help="one request per beat; the voice will drift")
    ap.add_argument("--reuse", action="store_true",
                    help="re-time an existing narration.wav against a new "
                         "shot list; synthesises nothing")
    ap.add_argument("--voice-ref-json", default="",
                    help='clone this narrator: {"wav", "text", "start", "end"}')
    a = ap.parse_args()

    piece = a.piece.resolve()
    beats = [beat_text(b) for b in
             json.loads((piece / "beats.json").read_text(encoding="utf-8"))]
    shots = json.loads((piece / "shots.json").read_text(encoding="utf-8"))

    owner: dict[int, list[int]] = {b: [] for b in range(len(beats))}
    for s in shots:
        cov = s.get("covers_beats") or [0]
        owner.setdefault(min(cov), []).append(s["i"])

    words = sum(len(b.split()) for b in beats)
    print(f"\n=== {piece.name}: {len(beats)} beats, {words} words, "
          f"{len(shots)} shots ===")
    for b in range(len(beats)):
        print(f"  b{b} -> {owner[b] or '(held)'}")

    engine, take = "voicedesign", "per-beat" if a.per_beat else "single"
    if a.reuse and (piece / "narration.wav").exists():
        # The audio does not depend on the shot list, only on the beats, so a
        # loop that re-plans does not have to re-speak. It only has to work out
        # which shots now sit in which beat's window.
        import wave
        with wave.open(str(piece / "narration.wav")) as w:
            full0 = (np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
                     .astype(np.float32) / 32767.0)
        speech = full0[int(LEAD_S * SR):len(full0) - int(TAIL_S * SR)]
        ends = segments_of(piece / "_tts", len(beats)) or boundaries(speech, beats)
        print(f"  --reuse: {len(speech) / SR:.1f}s of existing narration re-timed")
    elif a.per_beat:
        print("  --per-beat: one request per beat. The voice WILL change "
              "between segments; this model re-samples it every call.")
        takes = S.synth_voicedesign(beats, a.design, a.device, piece / "_tts")
        speech = np.concatenate(
            [np.concatenate([t, np.zeros(int(0.45 * SR), dtype=np.float32)])
             for t in takes])
        ends, cur = [], 0.0
        for t in takes:
            cur += (len(t) + int(0.45 * SR)) / SR
            ends.append(cur)
    elif a.voice_ref_json or len(" ".join(beats)) > MAX_TAKE_CHARS:
        spec = json.loads(a.voice_ref_json) if a.voice_ref_json else None
        scratch = piece / "_tts"
        scratch.mkdir(parents=True, exist_ok=True)
        ref, ref_text = make_ref(spec, beats, a.design, a.device, scratch)
        print(f"  cloned takes from {'the given narrator' if spec else 'a VoiceDesign reference'}"
              f" ({len(read_wav(ref)) / SR:.1f}s): \"{ref_text[:60]}...\"")
        speech, ends = synth_cloned(beats, ref, ref_text, a.device, scratch)
        engine, take = "qwen3_base_clone", "cloned"
    else:
        text = " ".join(" ".join(b.split()) for b in beats)
        print(f"  one request, {len(text)} chars, one voice for the whole take")
        speech = synth_one(text, a.design, a.device, piece / "_tts")
        ends = segments_of(piece / "_tts", len(beats))
        if ends:
            print(f"  {len(speech) / SR:.1f}s spoken, boundaries taken from "
                  f"the engine's own segment timings")
        else:
            ends = boundaries(speech, beats)
            gaps = pauses(speech)
            print(f"  {len(speech) / SR:.1f}s spoken, {len(gaps)} pauses "
                  f"found, {len(beats) - 1} boundaries snapped to them")

    full = np.concatenate([np.zeros(int(LEAD_S * SR), dtype=np.float32),
                           speech,
                           np.zeros(int(TAIL_S * SR), dtype=np.float32)])
    total = len(full) / SR

    scenes: list[dict] = []
    prev_end = 0.0
    for b, end in enumerate(ends):
        win_start = 0.0 if b == 0 else LEAD_S + prev_end
        win_end = LEAD_S + end if b < len(ends) - 1 else total
        ids = owner.get(b) or []
        if not ids:
            if scenes:
                scenes[-1]["shot_end"] = round(win_end, 3)
                scenes[-1]["duration"] = round(
                    win_end - scenes[-1]["shot_start"], 3)
            prev_end = end
            continue
        span = (win_end - win_start) / len(ids)
        for k, i in enumerate(ids):
            st = win_start + k * span
            scenes.append({
                "index": i, "id": f"s{i:03d}", "beat": b,
                "narration": beats[b],
                "speech_start": round(LEAD_S + prev_end, 3),
                "speech_end": round(LEAD_S + end, 3),
                "shot_start": round(st, 3),
                "shot_end": round(st + span, 3),
                "duration": round(span, 3),
            })
        prev_end = end

    import narrate
    wav = piece / "narration.wav"
    narrate.write_wav(wav, full)
    mp3 = piece / "narration.mp3"
    subprocess.run([narrate._ffmpeg_exe(), "-y", "-loglevel", "error",
                    "-i", str(wav), "-codec:a", "libmp3lame", "-qscale:a", "2",
                    str(mp3)], check=True)

    scenes.sort(key=lambda s: s["index"])
    (piece / "timeline.json").write_text(json.dumps({
        "title": piece.name,
        "engine": engine, "take": take,
        "audio": "narration.mp3", "sample_rate": SR,
        "duration": round(total, 3), "scenes": scenes,
    }, indent=2), encoding="utf-8")

    durs = [s["duration"] for s in scenes]
    print(f"  {total:.1f}s total, {len(scenes)} shots "
          f"{min(durs):.2f}-{max(durs):.2f}s, ASL {sum(durs) / len(durs):.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
