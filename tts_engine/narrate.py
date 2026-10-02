"""
narrate.py — Local text-to-audio using Kokoro-82M on the GPU.

Two modes, auto-detected:
  * single   : one voice narrates the whole text (research reports, overviews).
  * dialogue : a two-host "audio overview" — 'Speaker 1:' / 'Alex:' turns are
               mapped to alternating voices and stitched into a conversation.

Text is run through tts_prep.clean() (markdown stripped, numbers/dates/units spoken)
and sentence-chunked so no segment overruns Kokoro's token limit (the main cause of
looping/rushing). Output is a 24 kHz WAV (optional MP3 if ffmpeg is on PATH).

Examples:
    python narrate.py --in report.md --out overview.wav
    python narrate.py --in report.md --voice am_michael --speed 1.05
    python narrate.py --in dialogue.txt --dialogue --voices af_heart,am_michael
    echo "Hello there." | python narrate.py --out hi.wav
    python narrate.py --list-voices
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

import tts_prep
import voices as voice_reg

SAMPLE_RATE = 24000
HERE = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = HERE.parent / "output"


# ---------------------------------------------------------------------------
# Pipeline management (one KPipeline per language, created lazily)
# ---------------------------------------------------------------------------

class Engine:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self._pipes = {}
        self._KPipeline = None
        self.device = "cpu"
        self._init_torch()

    def _log(self, *a):
        if self.verbose:
            print(*a, file=sys.stderr, flush=True)

    def _init_torch(self):
        import torch
        self.torch = torch
        if torch.cuda.is_available():
            self.device = "cuda"
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            self._log(f"[engine] CUDA on: {name} (sm_{cap[0]}{cap[1]}), torch {torch.__version__}")
            torch.cuda.reset_peak_memory_stats()
        else:
            self._log("[engine] WARNING: CUDA not available — running on CPU (much slower). "
                      "Check that torch was installed from the cu128 index.")

    def _pipeline(self, lang_code: str):
        if lang_code in self._pipes:
            return self._pipes[lang_code]
        if self._KPipeline is None:
            from kokoro import KPipeline
            self._KPipeline = KPipeline
        self._log(f"[engine] loading Kokoro pipeline (lang='{lang_code}') ...")
        pipe = self._KPipeline(lang_code=lang_code)
        self._pipes[lang_code] = pipe
        return pipe

    def warmup(self, voice: str):
        """Load the pipeline for a voice and run a tiny synth to trigger kernel
        compilation, so later timing reflects steady-state synthesis, not startup."""
        try:
            for _ in self._pipeline(voice_reg.lang_of(voice))(
                    "Ready.", voice=voice, speed=1.0, split_pattern=r"\n+"):
                break
        except Exception as e:  # pragma: no cover
            self._log(f"[engine] warmup note: {e}")

    def synth(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        """Synthesize one chunk of already-cleaned text. Returns float32 mono @24k."""
        lang = voice_reg.lang_of(voice)
        pipe = self._pipeline(lang)
        segments = []
        for result in pipe(text, voice=voice, speed=speed, split_pattern=r"\n+"):
            audio = getattr(result, "audio", None)
            if audio is None and isinstance(result, (tuple, list)):
                audio = result[-1]
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            segments.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        if not segments:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(segments)

    def peak_vram_mb(self) -> float:
        if self.device == "cuda":
            return self.torch.cuda.max_memory_allocated() / (1024 ** 2)
        return 0.0


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def write_wav(path: Path, audio: np.ndarray):
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def _ffmpeg_exe() -> str | None:
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:  # pip-bundled ffmpeg (no system install / admin needed)
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def to_mp3(wav_path: Path) -> Path | None:
    ff = _ffmpeg_exe()
    if not ff:
        return None
    mp3_path = wav_path.with_suffix(".mp3")
    if mp3_path == wav_path:
        print("[narrate] mp3 target is the source file, skipping conversion", file=sys.stderr)
        return None
    try:
        subprocess.run(
            [ff, "-y", "-loglevel", "error", "-i", str(wav_path),
             "-codec:a", "libmp3lame", "-qscale:a", "2", str(mp3_path)],
            check=True,
        )
        return mp3_path
    except Exception as e:  # report the real reason rather than blaming a missing ffmpeg
        print(f"[narrate] mp3 conversion failed: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_single(engine: Engine, raw_text: str, voice: str, speed: float,
                    gap: float = 0.2) -> np.ndarray:
    cleaned = tts_prep.clean(raw_text)
    chunks = tts_prep.to_chunks(cleaned, max_chars=400)
    engine._log(f"[narrate] single voice={voice} chunks={len(chunks)}")
    pieces = []
    for i, ch in enumerate(chunks, 1):
        audio = engine.synth(ch, voice=voice, speed=speed)
        pieces.append(audio)
        pieces.append(silence(gap))
        engine._log(f"[narrate]   chunk {i}/{len(chunks)} -> {len(audio)/SAMPLE_RATE:.1f}s")
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def generate_dialogue(engine: Engine, raw_text: str, voice_list, speed: float,
                     turn_gap: float = 0.35) -> np.ndarray:
    turns = tts_prep.parse_dialogue(raw_text)
    if not turns:
        raise ValueError("dialogue mode requested but no 'Speaker:'/'Name:' turns were found")
    # Map speaker labels to voices in order of first appearance.
    speaker_voice = {}
    order = []
    for sp, _ in turns:
        if sp not in speaker_voice:
            speaker_voice[sp] = voice_list[len(order) % len(voice_list)]
            order.append(sp)
    engine._log("[narrate] dialogue speakers -> " +
                ", ".join(f"{sp}={speaker_voice[sp]}" for sp in order))
    pieces = []
    for i, (sp, line) in enumerate(turns, 1):
        v = speaker_voice[sp]
        for ch in tts_prep.to_chunks(line, max_chars=400):
            audio = engine.synth(ch, voice=v, speed=speed)
            pieces.append(audio)
            pieces.append(silence(0.15))
        pieces.append(silence(turn_gap))
        engine._log(f"[narrate]   turn {i}/{len(turns)} [{sp}/{v}]")
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_out(args, infile: Path | None) -> Path:
    if args.out:
        p = Path(args.out)
        return p if p.is_absolute() else (Path.cwd() / p)
    stem = infile.stem if infile else "overview"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUT_DIR / f"{stem}_{ts}.wav"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Kokoro local text-to-audio.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--in", dest="infile", help="input text/markdown file")
    src.add_argument("--text", help="inline text")
    ap.add_argument("--out", help="output audio path (default: output/<name>_<ts>.wav)")
    ap.add_argument("--voice", default=voice_reg.DEFAULT_VOICE, help="voice id (single mode)")
    ap.add_argument("--voices", help="comma-separated voice ids for dialogue mode")
    ap.add_argument("--mode", choices=["auto", "single", "dialogue"], default="auto")
    ap.add_argument("--speed", type=float, default=1.0, help="0.5-2.0, 1.0 = normal")
    ap.add_argument("--format", choices=["wav", "mp3"], default="wav")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--list-voices", action="store_true")
    args = ap.parse_args(argv)

    if args.list_voices:
        print("Kokoro voices:\n" + voice_reg.pretty_list())
        print(f"\nDefault single: {voice_reg.DEFAULT_VOICE}")
        print("Default dialogue pair: " + ", ".join(voice_reg.DEFAULT_DIALOGUE_VOICES))
        return 0

    # Load input
    infile = None
    if args.infile:
        infile = Path(args.infile)
        raw = infile.read_text(encoding="utf-8")
    elif args.text:
        raw = args.text
    else:
        raw = sys.stdin.read()
    if not raw.strip():
        print("ERROR: no input text", file=sys.stderr)
        return 2

    # Decide mode
    mode = args.mode
    if mode == "auto":
        mode = "dialogue" if tts_prep.parse_dialogue(raw) else "single"

    engine = Engine(verbose=not args.quiet)

    # Resolve voices, warm the model (timed separately from synthesis).
    if mode == "dialogue":
        vlist = ([v.strip() for v in args.voices.split(",")] if args.voices
                 else list(voice_reg.DEFAULT_DIALOGUE_VOICES))
        primary_voice = vlist[0]
    else:
        primary_voice = args.voice
    t_load = time.time()
    engine.warmup(primary_voice)
    load_s = time.time() - t_load

    t0 = time.time()
    if mode == "dialogue":
        audio = generate_dialogue(engine, raw, vlist, args.speed)
    else:
        audio = generate_single(engine, raw, args.voice, args.speed)
    gen_s = time.time() - t0

    if audio.size == 0:
        print("ERROR: no audio produced (empty after cleaning?)", file=sys.stderr)
        return 3

    out = resolve_out(args, infile)
    # The WAV is always written to a .wav path, even when the user asked for an
    # .mp3 output path. Writing the WAV to the requested .mp3 name and then
    # converting made ffmpeg's input and output the same file, so it refused, the
    # fallback branch fired, and the run finished "successfully" with a WAV
    # carrying an .mp3 extension. Found 2026-09-06.
    wav_out = out.with_suffix(".wav") if args.format == "mp3" else out
    write_wav(wav_out, audio)
    final = wav_out
    if args.format == "mp3":
        mp3 = to_mp3(wav_out)
        if mp3:
            if mp3 != wav_out:
                wav_out.unlink(missing_ok=True)
            final = mp3
        else:
            engine._log("[narrate] mp3 conversion unavailable, keeping WAV at " + str(wav_out))

    dur_s = audio.size / SAMPLE_RATE
    rtf = gen_s / dur_s if dur_s else 0.0
    summary = {
        "output": str(final),
        "mode": mode,
        "audio_seconds": round(dur_s, 1),
        "load_seconds": round(load_s, 1),
        "gen_seconds": round(gen_s, 1),
        "rtf": round(rtf, 3),
        "realtime_x": round(1 / rtf, 1) if rtf else None,
        "device": engine.device,
        "peak_vram_mb": round(engine.peak_vram_mb(), 1),
    }
    engine._log(
        f"[narrate] DONE {summary['audio_seconds']}s audio | load {summary['load_seconds']}s + "
        f"synth {summary['gen_seconds']}s ({summary['realtime_x']}x realtime) on {engine.device}, "
        f"peak VRAM {summary['peak_vram_mb']}MB"
    )
    # Machine-readable line for the skill to parse.
    print("RESULT " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
