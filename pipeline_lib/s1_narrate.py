"""Stage 1 - narration and the timing spine, for any piece.

    <podcast venv python> s1_narrate.py <piece_dir> [--engine kokoro|gptsovits]

Reads shots.json, writes narration.wav / narration.mp3 / timeline.json.

NOT for director pieces. This assumes one shot == one narration line. The
director layer makes more shots than beats, so two shots can carry the same
beat text and this file synthesises that beat once per shot - the take is then
SPOKEN TWICE. Measured on gift_of_silence: 187 words of audio for a 127-word
passage, three sentences repeated back to back. Use narrate_beats.py, which
narrates the beat list and splits each beat's window across its shots.

The timeline is what every later stage reads: a shot is exactly as long as its
line takes to say, so nothing downstream ever has to freeze a frame to fill
time. Asserts the device is cuda; the one previously logged Kokoro run fell
back to CPU silently and ran at 1.0x realtime for a month before anyone noticed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
import os  # noqa: E402
ENGINE_DIR = Path(os.environ.get("CLAY_FILM_TTS_ENGINE") or HERE.parent / "tts_engine")
sys.path.insert(0, str(ENGINE_DIR))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

SAMPLE_RATE = 24000
GAP_S = 0.45
LEAD_S = 0.6
TAIL_S = 1.2


def synth_kokoro(lines: list[str], voice: str) -> list[np.ndarray]:
    import torch
    import narrate
    if not torch.cuda.is_available():
        raise SystemExit("FAIL: torch reports no CUDA; Kokoro would run at 1.0x")
    print(f"  kokoro on cuda - {torch.cuda.get_device_name(0)}")
    eng = narrate.Engine()
    eng.warmup(voice)
    out = []
    for i, t in enumerate(lines):
        out.append(np.asarray(eng.synth(t, voice=voice, speed=1.0),
                              dtype=np.float32))
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(lines)}", flush=True)
    return out


def synth_gptsovits(lines: list[str], ref_wav: Path, ref_text: str,
                    api: str) -> list[np.ndarray]:
    """Drive the GPT-SoVITS api_v2 server. One request per beat.

    Each beat is re-anchored to the same reference clip, so drift is bounded by
    construction. The cost is that prosody does not carry across a beat
    boundary, which is the documented trade this model makes.
    """
    import urllib.parse
    import urllib.request
    import io
    import wave

    out = []
    for i, t in enumerate(lines):
        q = urllib.parse.urlencode({
            "text": t, "text_lang": "en",
            "ref_audio_path": str(ref_wav), "prompt_text": ref_text,
            "prompt_lang": "en", "text_split_method": "cut0",
            "media_type": "wav", "streaming_mode": "false",
        })
        with urllib.request.urlopen(f"{api}/tts?{q}", timeout=600) as r:
            raw = r.read()
        with wave.open(io.BytesIO(raw)) as w:
            n = w.getnframes()
            sr = w.getframerate()
            pcm = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32767.0
            if w.getnchannels() > 1:
                pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1)
        if sr != SAMPLE_RATE:
            # Linear resample; these are narration takes, not music.
            idx = np.linspace(0, len(pcm) - 1, int(len(pcm) * SAMPLE_RATE / sr))
            pcm = np.interp(idx, np.arange(len(pcm)), pcm).astype(np.float32)
        out.append(pcm)
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(lines)}", flush=True)
    return out


CLI = Path(os.environ.get("AUDIOCPP_CLI", "audiocpp_cli.exe"))
VOICEDESIGN = Path(os.environ.get("QWEN3TTS_VOICEDESIGN_GGUF", "models/qwen3-tts-voicedesign-q8_0.gguf"))

# The production voice, chosen blind against VibeVoice and then against two
# deeper takes. Qwen3-TTS VoiceDesign builds a voice from this description
# rather than from a reference clip, so it clones nobody, and "deeper" is a
# prose edit rather than a retrain. Measured 1.3x faster than realtime.
DESIGN = (
    "A very deep male bass-baritone, around fifty. Low, dark and chesty, with "
    "heavy resonance and a slight gravel, the register of a late-night radio "
    "narrator. Calm, slow and unhurried audiobook narration. Restrained and "
    "serious, never theatrical."
)


def synth_voicedesign(lines: list[str], design: str, device: int,
                      scratch: Path) -> list[np.ndarray]:
    """One request per beat, but one model load for the whole piece.

    --batch-text-file takes one line per request and --out-dir names each
    output, so the 2.6 GB of weights are paged in once instead of once per
    beat. Beats must therefore survive being flattened to a single line, which
    segment.py already guarantees.
    """
    import wave

    scratch.mkdir(parents=True, exist_ok=True)
    listing = scratch / "beats.txt"
    outs = scratch / "wav"
    outs.mkdir(exist_ok=True)
    for f in outs.glob("*.wav"):
        f.unlink()
    flat = [" ".join(l.split()) for l in lines]
    listing.write_text("\n".join(flat) + "\n", encoding="utf-8")

    r = subprocess.run(
        [str(CLI), "--task", "vdes", "--family", "qwen3_tts",
         "--model", str(VOICEDESIGN), "--backend", "cuda",
         "--device", str(device), "--instruct", design, "--language",
         "english", "--batch-text-file", str(listing), "--out-dir", str(outs),
         "--metrics"],
        capture_output=True, text=True, timeout=7200)
    got = sorted(outs.glob("*.wav"))
    if r.returncode != 0 or len(got) != len(lines):
        tail = (r.stderr or r.stdout).strip().splitlines()[-6:]
        raise SystemExit(f"voicedesign produced {len(got)} of {len(lines)} "
                         f"beats:\n  " + "\n  ".join(tail))

    audio = []
    for f in got:
        with wave.open(str(f)) as w:
            pcm = (np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
                   .astype(np.float32) / 32767.0)
            if w.getnchannels() > 1:
                pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1)
            sr = w.getframerate()
        if sr != SAMPLE_RATE:
            idx = np.linspace(0, len(pcm) - 1,
                              int(len(pcm) * SAMPLE_RATE / sr))
            pcm = np.interp(idx, np.arange(len(pcm)), pcm).astype(np.float32)
        audio.append(pcm)
    return audio


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("piece", type=Path)
    ap.add_argument("--engine", default="voicedesign",
                    choices=["voicedesign", "kokoro", "gptsovits"])
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--design", default=DESIGN)
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--ref-wav", type=Path)
    ap.add_argument("--ref-text", default="")
    ap.add_argument("--api", default="http://127.0.0.1:9880")
    args = ap.parse_args()

    piece = args.piece.resolve()
    shots = json.loads((piece / "shots.json").read_text(encoding="utf-8"))
    lines = [s["narration"] for s in shots]
    print(f"\n=== {piece.name}: {len(lines)} beats, "
          f"{sum(len(l.split()) for l in lines)} words ===")

    if args.engine == "voicedesign":
        audio = synth_voicedesign(lines, args.design, args.device,
                                  piece / "_tts")
    elif args.engine == "kokoro":
        audio = synth_kokoro(lines, args.voice)
    else:
        if not args.ref_wav or not args.ref_wav.exists():
            raise SystemExit("--ref-wav is required for gptsovits")
        audio = synth_gptsovits(lines, args.ref_wav, args.ref_text, args.api)

    parts: list[np.ndarray] = []
    timeline: list[dict] = []
    cursor = 0

    def silence(sec: float) -> None:
        nonlocal cursor
        n = int(sec * SAMPLE_RATE)
        parts.append(np.zeros(n, dtype=np.float32))
        cursor += n

    silence(LEAD_S)
    for i, (shot, a) in enumerate(zip(shots, audio)):
        start = cursor / SAMPLE_RATE
        parts.append(a)
        cursor += len(a)
        end = cursor / SAMPLE_RATE
        silence(GAP_S if i < len(shots) - 1 else TAIL_S)
        shot_end = cursor / SAMPLE_RATE
        timeline.append({
            "index": i, "id": f"s{i:03d}",
            "narration": shot["narration"],
            "speech_start": round(start, 3), "speech_end": round(end, 3),
            "shot_start": round(start if i else 0.0, 3),
            "shot_end": round(shot_end, 3),
            "duration": round(shot_end - (start if i else 0.0), 3),
        })

    full = np.concatenate(parts)
    total = len(full) / SAMPLE_RATE

    import narrate
    wav = piece / "narration.wav"
    narrate.write_wav(wav, full)
    mp3 = piece / "narration.mp3"
    subprocess.run([narrate._ffmpeg_exe(), "-y", "-loglevel", "error",
                    "-i", str(wav), "-codec:a", "libmp3lame", "-qscale:a", "2",
                    str(mp3)], check=True)

    (piece / "timeline.json").write_text(json.dumps({
        "title": piece.name, "engine": args.engine,
        "voice": args.voice if args.engine == "kokoro" else str(args.ref_wav),
        "audio": "narration.mp3", "sample_rate": SAMPLE_RATE,
        "duration": round(total, 3), "scenes": timeline,
    }, indent=2), encoding="utf-8")

    durs = [t["duration"] for t in timeline]
    print(f"  {total:.1f}s total, shots {min(durs):.2f}-{max(durs):.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
