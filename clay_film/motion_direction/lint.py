"""Lint for H3 motion prompts and motion plans.

Two layers:
  lint_prompt(prompt, ctx)        text-only rules; run on today's prompts and on translated ones
  lint_plan(mp, shot, ctx)        rules that need the director's plan and the shot's geometry

Each finding is (level, rule_id, message). level: error (do not send), warn (look first), note.
Every rule cites where the lesson came from.
"""
from __future__ import annotations

import re
import sys

from ..paths import LIB  # noqa: E402

if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))
import prompt_profile as pp  # noqa: E402

MAX_S, MIN_S = 15.1, 5.2   # animate_shots MAX_FRAMES 362 / MIN_FRAMES 124 at 24 fps

# Copied from _lib/director.py STILL_IN_ACTION (importing director pulls in llm_router).
STILL_IN_ACTION = re.compile(
    r"\b(hold(s|ing)?\s+(still|steady|motionless)|stay(s|ing)?\s+"
    r"(still|fixed|put|unchanged|the\s+same)|remain(s|ing)?\s+"
    r"(still|fixed|unchanged|motionless)|motionless|unmoving|"
    r"without\s+moving|no\s+movement|nothing\s+moves|unchanged|"
    r"perfectly\s+still|frozen|static)\b", re.I)

_STOP = set("""a an the of to in on at by for from with and or but as is are be her his their its
them they he she it this that these those over under into onto across toward towards behind ahead
beside near far small large slow slowly one two each every all some just still very more less up down
back front left right side end""".split())

HAND_SCALE = re.compile(r"\b(hands?|fingers?|fingertips?|lips|mouth|eyes?|eyelids?|wrists?|knuckles?)\b", re.I)
SMALL_SIZES = {"full_body", "wide", "very_wide"}

CAMERA_IN_TEXT = [  # what a prompt's camera clause actually asks for
    (r"push-in|pushes in|creeps forward", "push_in"),
    (r"pulls (steadily )?back|pulls out", "pull_out"),
    (r"tracks steadily behind|follows in a tracking shot|tracking shot", "tracking"),
    (r"holds (steady|a static shot)", "static"),
    (r"glides steadily to the right|trucks right|pans right", "right"),
    (r"glides steadily to the left|trucks left|pans left", "left"),
    (r"cranes up|pedestals up|tilts up", "up"),
    (r"cranes down|pedestals down|tilts down", "down"),
    (r"orbits|arc shot", "arc"),
]
PLAN_TO_TEXT = {"push_in": "push_in", "zoom_in": "push_in", "pull_out": "pull_out", "zoom_out": "pull_out",
                "tracking_forward": "tracking", "tracking_backward": "pull_out", "static": "static",
                "truck_right": "right", "pan_right": "right", "truck_left": "left", "pan_left": "left",
                "tilt_up": "up", "pedestal_up": "up", "tilt_down": "down", "pedestal_down": "down",
                "arc": "arc"}


def _description(prompt: str) -> str:
    m = re.search(r"integrated_multimodal_description:(.*?)(?:\n\noverall_soundscape:|$)", prompt, re.S)
    return m.group(1) if m else prompt


def camera_asked(prompt: str) -> str | None:
    for pat, kind in CAMERA_IN_TEXT:
        if re.search(pat, prompt, re.I):
            return kind
    return None


def lint_prompt(prompt: str, ctx: dict) -> list[tuple[str, str, str]]:
    """Text-only rules. ctx: mode, seconds, style_bible, still_text (the director's still sentence),
    structured (True when the prompt claims the guide's structure)."""
    out = []
    if ctx.get("structured", True):
        prof = pp.load("minimax_h3")
        for msg in pp.lint(prof, prompt, ctx):     # alignment, fields, pictures, style-first, locks, length
            out.append(("error", "h3-structure", msg))
    else:
        if not prompt.startswith(("For the target video", "How the reference pictures")):
            out.append(("warn", "h3-structure", "no alignment line and no three fields: free paragraph, "
                        "not the guide's structure (guide 2.1-2.2; STYLE_HOLD_PLAN.md)"))
    if re.search(r"\bclose on\b", prompt, re.I):
        out.append(("error", "close-on", "'close on' in the camera clause: iris-out on empty frames "
                    "(redbook s004) and cut to a face on far targets (basilisk s001)"))
    for m in re.finditer(r"\b(close-up|close shot|extreme close|wide shot|medium shot|full[- ]length|"
                         r"\d+\s?mm lens)\b", prompt, re.I):
        out.append(("warn", "framing-words", f"{m.group(0)!r}: the still already fixed the framing; a "
                    "framing word it did not deliver makes H3 reframe with a cut (Anubis s015)"))
        break
    desc = _description(prompt)
    if re.search(r"\b(fades?|fading|vanish\w*|disappear\w*)\b", desc, re.I):
        out.append(("error", "fade", "H3 renders 'X fades' as X appearing (Anubis s034); write the "
                    "appearance and reverse the clip"))
    if re.search(r"\b(light|glint|reflection|highlight|shine|glow)\w*\b[^.]{0,50}\b(slides?|moves?|"
                 r"ripples?|shifts?|sweeps?|crosses|travels?)\b", desc, re.I):
        out.append(("warn", "moving-light", "light moving over a surface: on a glossy print it smeared "
                    "into white streaks over a face (Anubis s007); check the still's surface"))
    if re.search(r"\b(like an?|as if|as though)\b", desc, re.I):
        out.append(("warn", "simile", "a simile in the description can be drawn literally "
                    "(Aminita s013 'between two giants'); say what is seen"))
    if re.search(r"SHOT 2|\[Shot 2\]|cuts to", prompt):
        out.append(("error", "one-take", "a second shot is named; every directed shot is one take"))
    if "no cuts" not in prompt:
        out.append(("warn", "one-take", "no 'one continuous take ... no cuts' sentence (clay_film CLAUDE.md)"))
    still = " ".join((ctx.get("still_text") or "").split())
    m = re.search(r"in its original scene: (.*?)\. The environment", prompt, re.S)
    if still and m:
        part = m.group(1).strip()
        i = still.find(part)
        if i >= 0 and i + len(part) < len(still) and still[i + len(part)] not in ".;,:":
            nxt = still[i + len(part)]
            out.append(("error", "scene-truncated", "the scene text stops "
                        + ("inside a word" if nxt.isalnum() else "inside a sentence")
                        + f" ('...{part[-30:]}'); animate_shots.prompt_for slices it at SCENE_CHARS"))
    sec = float(ctx.get("seconds") or 0)
    if sec > MAX_S:
        out.append(("error", "duration", f"{sec:.2f} s is over H3's 15.1 s single take"))
    return out


def _content_words(s: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z]+", (s or "").lower()) if len(w) > 2 and w not in _STOP}


def lint_plan(mp: dict, shot: dict, ctx: dict, prompt: str = "") -> list[tuple[str, str, str]]:
    """Rules that need the plan and the shot geometry. ctx: scene (what the still shows), seconds."""
    out = []
    scene_words = _content_words(ctx.get("scene") or shot.get("still", ""))
    cam = mp.get("camera") or {}
    camspec = shot.get("camera") or {}
    interp = mp.get("interp") or {}
    e = float(interp.get("escalation") or 0)

    # nothing the still does not hold (Anubis s005 'rows of clerks', s030 the counter)
    for label, text in [("camera.toward", cam.get("toward", ""))] + \
                       [(f"mover {m.get('what')!r}", m.get("what", "")) for m in mp.get("movers") or []]:
        words = _content_words(text)
        if words and not words & scene_words:
            out.append(("error", "off-still", f"{label} names nothing the still shows; H3 cuts to find it"))
    subj = mp.get("anchor") or shot.get("subject", "")
    if re.search(r"\b\w+ing\b", subj) and not re.search(r"\b(clothing|building|ceiling|railing|wiring)\b", subj):
        out.append(("warn", "subject-label", f"subject {subj!r} carries a verb; it goes verbatim into the "
                    "anchor (Aminita s013), so it should name what the still shows"))
    if len(subj.split()) > 12:
        out.append(("warn", "subject-label", "subject over 12 words; it is the anchor's noun phrase"))
    # action at the still's scale (Anubis s005 lap close-up)
    small = camspec.get("distance") in SMALL_SIZES or camspec.get("frame") == "small_in_frame"
    act = " ".join([mp.get("beat", {}).get("develop", "")] + [m.get("does", "") for m in mp.get("movers") or []])
    if small and HAND_SCALE.search(act):
        out.append(("error", "action-scale", f"hand- or face-level business ({HAND_SCALE.search(act).group(0)!r}) "
                    "on a small figure: H3 cuts in to show it (Anubis s005)"))
    if camspec.get("frame") == "small_in_frame" and cam.get("move") != "static":
        out.append(("warn", "lone-small-figure", "a moving camera on a small figure: H3 pushes in on it "
                    "whatever the prompt says (Anubis s005); lock off or hold the still"))
    m = STILL_IN_ACTION.search(act)
    if m:
        out.append(("error", "stillness", f"stillness word {m.group(0)!r} in the action freezes the clip "
                    "(director.lint_action)"))
    # endings
    ef = (mp.get("ending") or {}).get("end_frame", "none")
    if cam.get("move") in ("pull_out", "zoom_out", "tracking_backward") and "crop" in ef:
        out.append(("error", "end-frame", "a pull-out ends on ground the still does not show; a crop "
                    "cannot be its end frame"))
    if cam.get("move") == "arc" and "crop" in ef:
        out.append(("error", "end-frame", "a cropped end fought the arc and H3 invented a second cook "
                    "(oceanus s023)"))
    # does the prompt ask for the move the director planned (catches regex mis-maps)
    if prompt:
        want, got = PLAN_TO_TEXT.get(cam.get("move")), camera_asked(prompt)
        if want and got and want != got:
            out.append(("error", "camera-mismatch", f"the director planned {cam.get('move')} but the "
                        f"prompt asks for {got}"))
    # interpretation layer
    band = interp.get("reality") or ("literal" if e < 0.25 else "heightened" if e < 0.6 else "abstract")
    if e >= 0.25 and not mp.get("wrongness"):
        out.append(("warn", "escalation", f"escalation {e} but no wrongness written; the prompt adds none"))
    if e < 0.25 and mp.get("wrongness"):
        out.append(("note", "escalation", "wrongness written at escalation under 0.25; the translator drops it"))
    if band == "literal" and mp.get("wrongness") and e >= 0.25:
        out.append(("warn", "reality", "reality is literal but the shot carries wrongness"))
    if e >= 0.7 and interp.get("source") == "clay":
        out.append(("note", "source", "escalation 0.7 or more on a clay shot; the brief expects more free shots here"))
    if mp.get("wrongness") and not _content_words(mp["wrongness"]) & scene_words:
        out.append(("error", "off-still", "the wrongness names nothing the still shows"))
    nouns = next((r["nouns"] for r in pp.load("minimax_h3")["rules"] if r["id"] == "no-content-in-locks"), [])
    hit = [n for n in nouns if re.search(r"\b" + n + r"\b", " ".join(mp.get("avoid") or []).lower())]
    if hit:
        out.append(("error", "no-content-in-locks", f"avoid lines name content {hit}; H3 paints named "
                    "nouns in (oceanus 2026-10-03)"))
    sec = float(ctx.get("seconds") or 0)
    if 0 < sec < MIN_S:
        out.append(("note", "duration", f"{sec:.2f} s is under 5.2 s; H3 renders 124 frames and assembly "
                    "stretches it"))
    return out
