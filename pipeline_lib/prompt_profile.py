"""Model prompt profiles: each video model's own prompting guide, turned into data the pipeline uses.

A profile lives in prompt_profiles/<model>/profile.json, next to the guide documents it was written
from (prompt_profiles/<model>/guide/). It says how that model wants a prompt laid out: the fixed
keyframe-alignment line, the named fields and their order, where the style goes, the camera
vocabulary the model was trained on, whether it takes a negative prompt, length, and lint rules.
Deploying a different model means adding a profile, not editing the renderer.

Three entry points:

    build(profile, shot, ctx)   -> prompt text in the model's own structure
    lint(profile, prompt, ctx)  -> list of problems (empty = passes)
    writer_brief(profile, ...)  -> instructions for an LLM or a person writing the per-shot path

Why this exists (2026-10-03, oceanus): H3 was given an 80-word free paragraph with no alignment
line and no fields, while its guide prescribes a structured ~250-600 word prompt. Between pinned
keyframes it fell back on its photographic prior. Several lock sentences also carried content
nouns ("faces and hands", "water and sky") that the model painted in as content.

The per-shot path text (what happens between the keyframes) comes from shot["video_path"] when a
writer has supplied one, else it is assembled from the shot's action and camera fields.

The piece supplies a style bible in config.json:
    "style_bible": {"style": "...",        # one sentence, goes first in the description
                    "texture": "...",      # how the medium looks and moves, every frame
                    "prohibitions": "..."} # 'No ...' sentences, observable, no content nouns
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent / "prompt_profiles"


def available() -> list[str]:
    return sorted(p.parent.name for p in ROOT.glob("*/profile.json"))


def load(model: str) -> dict:
    f = ROOT / model / "profile.json"
    if not f.exists():
        raise FileNotFoundError(f"no prompt profile for {model!r}; have {available()}")
    prof = json.loads(f.read_text(encoding="utf-8"))
    prof["_dir"] = str(f.parent)
    return prof


def guide_text(profile: dict) -> str:
    """The guide documents behind a profile, for an LLM writer's context."""
    d = Path(profile["_dir"]) / "guide"
    parts = []
    for name in ("VIDEO_PROMPT_WRITING_GUIDE_base_en.md",):
        if (d / name).exists():
            parts.append((d / name).read_text(encoding="utf-8"))
    return "\n\n".join(parts)


# ------------------------------------------------------------------ camera

def camera_sentence(profile: dict, motion: str, toward: str = "") -> str:
    """Director motion text -> the model's trained camera grammar (type + amplitude + speed)."""
    cam = profile.get("camera") or {}
    m = (motion or "").lower()
    verb = None
    for key in sorted(cam.get("verbs", {}), key=len, reverse=True):
        if re.search(r"\b" + re.escape(key) + r"\b", m):
            verb = cam["verbs"][key]
            break
    if verb is None:
        verb = cam.get("verbs", {}).get("push in", "The camera pushes in")
    if "static" in verb:
        return verb + "."
    if verb.endswith("trucks") and "left" not in verb and "right" not in verb:
        verb += " right" if "right" in m else (" left" if "left" in m else " slightly")
    amp = "large" if re.search(r"\b(sweep|large|wide|fast)\b", m) else cam.get("default_amplitude", "small")
    speed = "fast" if "fast" in m else cam.get("default_speed", "slow")
    tw = f" toward {toward}" if toward and re.search(r"push|zoom|truck|track", verb.lower()) else ""
    return cam.get("grammar", "{verb} with {amplitude} amplitude at {speed} speed{toward}").format(
        verb=verb, amplitude=amp, speed=speed, toward=tw) + "."


# ------------------------------------------------------------------ build

def _sent(s: str) -> str:
    s = " ".join((s or "").split()).strip()
    if not s:
        return ""
    s = s[0].upper() + s[1:]
    return s if s.endswith((".", "!", "?")) else s + "."


def build(profile: dict, shot: dict, ctx: dict) -> str:
    """ctx: mode ('i2va'|'fl2va'|'l2va'), seconds (float), style_bible (dict).
    shot: scene, subject/grounded_in, action, motion, move_target, soundscape, music, video_path."""
    if profile.get("builder") != "sectioned":
        raise NotImplementedError(f"builder {profile.get('builder')!r}")
    mode = ctx.get("mode", "i2va")
    seconds = float(ctx.get("seconds") or 5.0)
    bible = ctx.get("style_bible") or {}
    subject = (shot.get("grounded_in") or shot.get("subject") or "the scene").strip()
    scene = " ".join((shot.get("scene") or "").split()).rstrip(". ")

    al = profile["alignment"]
    head = al[mode].format(last_shot=1, seconds=seconds)

    d = profile["description"]
    # what the keyframe's surface is made of: "brushwork" on a painted piece; a photographic piece's
    # bible sets "surface": "light" (gardening 2026-10-10, H3 was told to keep brushwork on film)
    surface = bible.get("surface", "brushwork")
    parts = {
        "style": _sent(bible.get("style") or shot.get("motion_style") or ""),
        "anchor": _sent(d["anchor"][mode].format(subject=subject[0].upper() + subject[1:], scene=scene,
                                                 surface=surface)),
        "texture": _sent(bible.get("texture", "")),
        "path": " ".join(_sent(x) for x in re.split(r"(?<=[.!?])\s+", shot.get("video_path") or "") if x)
                or " ".join(filter(None, [
                    camera_sentence(profile, shot.get("motion", ""), shot.get("move_target", "")),
                    _sent(shot.get("action", ""))])),
        "landing": _sent(d["landing"][mode].format(surface=surface)),
        "prohibitions": " ".join(_sent(x) for x in re.split(r"(?<=\.)\s+", bible.get("prohibitions", "")) if x),
    }
    body = " ".join(p for p in (parts[k] for k in d["order"]) if p)
    desc = f"{d['shot_label'].format(n=1)} {body}"

    vals = {"description": desc, "soundscape": _sent(shot.get("soundscape", "")),
            "music": _sent(shot.get("music", ""))}
    fields = []
    for f in profile["fields"]:
        v = vals.get(f["content"]) or f.get("empty", "")
        fields.append(f"{f['name']}: {v}")
    return head + ("\n\n" if al.get("blank_line_after") else "\n") + profile.get("field_separator", "\n\n").join(fields)


# ------------------------------------------------------------------ lint

def _lock_text(prompt: str, ctx: dict) -> str:
    b = ctx.get("style_bible") or {}
    return " ".join([b.get("style", ""), b.get("texture", ""), b.get("prohibitions", "")])


def lint(profile: dict, prompt: str, ctx: dict) -> list[str]:
    out = []
    mode = ctx.get("mode", "i2va")
    lines = prompt.split("\n")
    for r in profile.get("rules", []):
        c = r["check"]
        if c == "first_line_is_alignment":
            want = profile["alignment"][mode].split("{")[0].split("<Picture")[0][:40]
            if not lines or not lines[0].startswith(want[:30]):
                out.append(f"{r['id']}: first line is not the {mode} alignment instruction")
            elif profile["alignment"].get("blank_line_after") and (len(lines) < 2 or lines[1].strip()):
                out.append(f"{r['id']}: no blank line after the alignment instruction")
        elif c == "fields_in_order":
            pos = [prompt.find(f["name"] + ":") for f in profile["fields"]]
            if -1 in pos or pos != sorted(pos):
                out.append(f"{r['id']}: fields missing or out of order")
        elif c == "picture_count_matches":
            n = {"i2va": 1, "fl2va": 2, "l2va": 1}[mode]
            named = set(re.findall(r"Picture (\d)", prompt))
            if named != {str(i) for i in range(1, n + 1)}:
                out.append(f"{r['id']}: names pictures {sorted(named)} but {n} keyframe(s) are attached")
        elif c == "style_at_shot1_start":
            style = (ctx.get("style_bible") or {}).get("style", "")
            i = prompt.find("[Shot 1]")
            if style and (i < 0 or style[:25] not in prompt[i:i + 80 + len(style)]):
                out.append(f"{r['id']}: style sentence is not at the start of [Shot 1]")
        elif c == "lock_lines_have_no_content_nouns":
            lock = _lock_text(prompt, ctx).lower()
            hit = [n for n in r["nouns"] if re.search(r"\b" + n + r"\b", lock)]
            if hit:
                out.append(f"{r['id']}: style/texture/prohibition lines name content: {hit}")
        elif c == "regex_absent":
            m = re.search(r["pattern"], prompt, re.I)
            if m:
                out.append(f"{r['id']}: {m.group(0)!r}")
        elif c == "word_count_in_range":
            n = len(prompt.split())
            L = profile.get("length", {})
            if n < L.get("min_words", 0) or n > L.get("max_words", 10**6):
                out.append(f"{r['id']}: {n} words, want {L.get('min_words')}-{L.get('max_words')}")
    return out


# ------------------------------------------------------------------ writer brief

def writer_brief(profile: dict, shot: dict, ctx: dict) -> str:
    """What an LLM (or a person) needs in order to write shot['video_path'] for this model."""
    rules = "\n".join(f"- {r['id']}: {r.get('why', '')}" for r in profile.get("rules", []))
    return (f"Write the motion path for one shot, for {profile['display']}.\n"
            f"Mode {ctx.get('mode')}, {float(ctx.get('seconds', 5)):.2f} s. Describe only what changes between the "
            "keyframes, in the model's camera grammar (type + amplitude + speed), in present tense, "
            "4-8 sentences, every detail visible or audible. Do not restate the style, texture or prohibitions; "
            "the builder adds them.\n\n"
            f"Shot: {json.dumps({k: shot.get(k) for k in ('scene', 'action', 'motion', 'people', 'narration')}, ensure_ascii=False)}\n\n"
            f"Rules:\n{rules}\n\nGuide:\n{guide_text(profile)}")
