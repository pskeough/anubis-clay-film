"""Word timings for a piece's narration, so the booth can highlight the script at the AI read's pace.

python align.py PIECE_DIR          (writes PIECE_DIR/words.json, prints a summary)

Runs faster-whisper small.en (cached locally, CPU int8, no VAD, so onnxruntime is never loaded)
on narration.mp3 with word timestamps, then maps the recognised words back onto the script text
from timeline.json with difflib. Script words whisper missed are interpolated by character length
between their matched neighbours, inside their own shot's speech window. If whisper is unavailable
the whole thing falls back to that interpolation, shot by shot. The cache key is the narration's
size and mtime, so a re-narrated piece is realigned on the next load.
"""
import difflib
import json
import re
import sys
from pathlib import Path

VERSION = 2


def norm(w):
    return re.sub(r"[^a-z0-9]", "", w.lower().replace("’", "'"))


def script_words(timeline):
    out = []
    for sc in timeline["scenes"]:
        for tok in sc["narration"].split():
            out.append({"w": tok, "shot": sc["id"], "scene": sc["index"]})
    return out


def whisper_words(audio):
    from faster_whisper import WhisperModel
    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    segs, _ = model.transcribe(str(audio), language="en", beam_size=5, word_timestamps=True,
                               vad_filter=False, condition_on_previous_text=False)
    return [(w.word.strip(), w.start, w.end) for s in segs for w in (s.words or [])]


def fill_gaps(words, timeline):
    """Interpolate words with no time by character length, inside their scene's speech window."""
    scenes = {sc["index"]: sc for sc in timeline["scenes"]}
    i = 0
    while i < len(words):
        if words[i].get("t0") is not None:
            i += 1
            continue
        j = i
        while j < len(words) and words[j].get("t0") is None and words[j]["scene"] == words[i]["scene"]:
            j += 1
        sc = scenes[words[i]["scene"]]
        lo = words[i - 1]["t1"] if i > 0 and words[i - 1]["scene"] == words[i]["scene"] else sc["speech_start"]
        hi = words[j]["t0"] if j < len(words) and words[j].get("t0") is not None \
            and words[j]["scene"] == words[i]["scene"] else sc["speech_end"]
        hi = max(hi, lo + 0.05 * (j - i))
        weights = [max(len(norm(w["w"])), 1) + 1 for w in words[i:j]]
        tot, t = sum(weights), lo
        for w, k in zip(words[i:j], weights):
            w["t0"], t = t, t + (hi - lo) * k / tot
            w["t1"], w["est"] = t, True
        i = j
    return words


def align(piece: Path, force=False):
    piece = Path(piece)
    timeline = json.loads((piece / "timeline.json").read_text(encoding="utf-8"))
    audio = piece / timeline.get("audio", "narration.mp3")
    key = f"{VERSION}:{audio.stat().st_size}:{int(audio.stat().st_mtime)}"
    cache = piece / "words.json"
    if cache.exists() and not force:
        c = json.loads(cache.read_text(encoding="utf-8"))
        if c.get("key") == key:
            return c
    words = script_words(timeline)
    method = "interpolated"
    try:
        heard = whisper_words(audio)
        a = [norm(w["w"]) for w in words]
        b = [norm(h[0]) for h in heard]
        for blk in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
            for k in range(blk.size):
                words[blk.a + k]["t0"] = round(heard[blk.b + k][1], 3)
                words[blk.a + k]["t1"] = round(heard[blk.b + k][2], 3)
        method = f"whisper small.en ({sum(1 for w in words if 't0' in w)}/{len(words)} words matched)"
    except Exception as e:  # no whisper, no model: interpolation still gives a usable pace
        method = f"interpolated (whisper failed: {type(e).__name__}: {e})"
    words = fill_gaps(words, timeline)
    for w in words:
        w["t0"], w["t1"] = round(w["t0"], 3), round(w["t1"], 3)
    out = {"key": key, "method": method, "duration": timeline["duration"],
           "scenes": [{k: sc[k] for k in ("index", "id", "narration", "speech_start", "speech_end",
                                          "shot_start", "shot_end")} for sc in timeline["scenes"]],
           "words": words}
    cache.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    return out


if __name__ == "__main__":
    r = align(Path(sys.argv[1]), force="--force" in sys.argv)
    print(r["method"])
    for w in r["words"][:12]:
        print(f"{w['t0']:7.2f} {w['t1']:7.2f} {w['shot']} {'~' if w.get('est') else ' '} {w['w']}")
