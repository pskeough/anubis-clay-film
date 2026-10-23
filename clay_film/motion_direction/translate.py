"""Motion plan -> an H3 prompt in the official guide's structure. Rules compiler, no model call.

Used by stages.motion_one for shots that carry a motion_plan.

The fixed strings (alignment lines, field names, the anchor and landing sentences) come from
pipeline_lib/prompt_profiles/minimax_h3/profile.json through prompt_profile.load,
so the profile stays the one source of truth for the guide's wording. What this file adds is the
order the guide asks for inside [Shot 1] (guide 3.1 and 3.2):

    I2VA:  first-frame anchor -> action onset -> continuous development -> result or reaction
    FL2VA: first-frame state -> observable intermediate changes -> narrowing differences -> last frame

and the camera sentence in the guide's grammar (4.3): motion type + amplitude + speed, a target,
and a reason, written as one natural sentence.
"""
from __future__ import annotations

import re
import sys

from ..paths import LIB  # noqa: E402

if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))
import prompt_profile as pp  # noqa: E402

PROFILE = "minimax_h3"
ONE_TAKE = "The whole clip is one continuous take on this framing, with no cuts."
# v2: an anchor, not a ban. New space may come into view; it has to be the same place with the same people.
HOLD = ("Everyone keeps the same face, hair and clothes as in the opening frames and stays in their place unless "
        "they are seen to move there; whatever comes into view as the camera moves belongs to the same place, "
        "with the same walls, light, furniture and weather as the opening frames. Parked cars and other objects "
        "stay where they are, and anything a person is holding stays in their hands unless they are seen to put "
        "it down.")   # p17: the empty car drove off and an umbrella vanished (2026-10-12)
# 2026-10-13: HOLD's "whatever comes into view as the camera moves" invites a reveal; a contained or static shot
# gets the same anchor without it
HOLD_STILL = ("Everyone keeps the same face, hair and clothes as in the opening frames and stays in their place unless "
              "they are seen to move there, and the room keeps the same walls, light and furniture. Anything a person "
              "is holding stays in their hands unless they are seen to put it down.")
NO_SPEECH = "No one delivers lines or makes a speech; faces stay natural, breathing and reacting."
# Patrick, second full-run review (2026-10-12): about one shot in four hallucinated, almost always when the camera
# travelled, turned or re-widened. The camera stays contained; the animation is inside the frame.
CONTAIN = ("The camera stays centred on this same view the whole time: it eases slowly forward and settles, and "
           "never turns, pans, tilts or pulls back, so nothing outside the opening frame ever comes into view. All the life "
           "of the shot is inside the frame: faces, eyes, hands, breathing and the light.")   # "the rain" dropped
           # 2026-10-13: it named rain in the windowless Eden rooms
# 2026-10-12 Patrick: nobody should look like they are speaking the voice-over's lines. Natural faces are fine;
# a conversation reads mostly through listening, with the odd murmur, never a speech.
TALK = ("They are in a quiet conversation seen mostly through listening: nods, glances, small changes of "
        "expression, the odd brief murmur, never a long speech.")

# ------------------------------------------------------------------ the still, as text

_LEAD_FRAMING = re.compile(
    r"^(?:an? )?(?:very |extreme )?(?:wide|close|medium|full[- ]length|long|low|high)"
    r"(?:[- ]up| shot)?\s+(?:from|on|of|at)?\s*", re.I)
_LENS = re.compile(r",?\s*(?:(?:on|through|with|from) )?(?:an? )?(?:long |short |wide |telephoto )?"
                   r"\d+\s?mm(?: lens)?\b", re.I)
_SIMILE = re.compile(r",?\s*\b(?:like an?|as if|as though)\b[^,.;]*", re.I)


_STOP = {"the", "and", "her", "his", "its", "she", "him", "they", "them", "with", "for", "all", "one",
         "two", "each", "from", "into", "over", "that", "this", "slow", "slowly"}


def _content(s: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 2 and w not in _STOP}


def plan_scene(mp: dict, shot: dict, max_words: int = 140) -> str:
    """The scene text for this plan: the still, keeping every sentence that names what moves."""
    keep = [m.get("what", "") for m in mp.get("movers") or []]
    keep += [(mp.get("camera") or {}).get("toward", ""), mp.get("wrongness", "")]
    return scene_from_still(shot.get("still", ""), max_words, [k for k in keep if k])


def scene_from_still(still: str, max_words: int = 140, keep: list[str] | None = None) -> str:
    """What the still shows, without the lens and shot-size words the still already fixed.

    Lessons kept: framing words the still may not deliver make H3 reframe and cut (Anubis s015);
    a mid-word cut leaves a fragment (animate_shots.prompt_for slices scene at 320 characters).
    Sentences are kept whole, never cut inside. Over the word budget, sentences are dropped from
    the end, except any sentence that names a thing in `keep` (the planned movers, the camera
    target, the wrongness): the prompt must still show H3 what is going to move.
    """
    text = " ".join((still or "").split())
    text = _LENS.sub("", text)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if sents:
        sents[0] = re.sub(r"^(?:an?|the)\s+(?=at |from |of )", "", sents[0].lstrip(" ,"), flags=re.I)
        m = _LEAD_FRAMING.match(sents[0])
        if m:
            prep = re.search(r"\b(from|at|of|on)\s*$", m.group(0), re.I)
            rest = sents[0][m.end():]
            sents[0] = (f"Seen {prep.group(1).lower()} {rest}" if prep and prep.group(1).lower() in ("from", "at")
                        else rest[:1].upper() + rest[1:])
    want = set().union(*[_content(k) for k in keep or []]) if keep else set()
    must = [i == 0 or bool(want & _content(s)) for i, s in enumerate(sents)]   # the first sentence is the composition
    budget = max_words - sum(len(s.split()) for s, m_ in zip(sents, must) if m_)
    out = []
    for s, m_ in zip(sents, must):
        w = len(s.split())
        if m_ or w <= budget:
            out.append(s)
            if not m_:
                budget -= w
        else:
            budget = -1          # keep order: once a free sentence is dropped, drop later free ones
    return " ".join(out).rstrip(". ")


# ------------------------------------------------------------------ the camera sentence

def _amp(c: dict) -> str:
    return {"small": " with small amplitude", "large": " with large amplitude"}.get(c.get("amplitude"), "")


def _speed(c: dict) -> str:
    return {"slow": " at slow speed", "fast": " at fast speed"}.get(c.get("speed"), "")


def camera_sentence(c: dict, subject: str) -> str:
    """The guide's camera grammar (4.3). Never writes 'close on' (iris-out, cut to a face)."""
    mv = c.get("move", "static")
    tw = (c.get("toward") or "").strip()
    why = (c.get("because") or "").strip().rstrip(".")
    why = f", {why}" if why else ""
    a, s = _amp(c), _speed(c)
    if mv == "static":
        return f"The camera holds a static shot{why}."
    if c.get("amplitude") == "slight" and mv in ("push_in", "zoom_in", "tracking_forward"):
        # 16 Oct H3 A/B (h3ab/FINDINGS.md): "small amplitude at slow speed" ran to 1.4x in 7 s and 1.6-1.9x over
        # 11 s; this wording gave 1.04-1.06x and kept faces. A fully static camera on a still figure made him turn
        # to the lens, so the drift is the floor.
        return (f"The camera creeps forward very slightly" + (f" toward {tw}" if tw else "") + f"{why}, "
                "so the last frame is almost the same framing as the first.")
    if mv in ("push_in", "zoom_in"):
        verb = "pushes in" if mv == "push_in" else "zooms in"
        return f"The camera {verb}{a}{s}" + (f" toward {tw}" if tw else "") + f"{why}."
    if mv in ("pull_out", "zoom_out"):
        verb = "pulls out" if mv == "pull_out" else "zooms out"
        return f"The camera {verb}{a}{s}{why}."
    if mv == "tracking_forward":
        return (f"The camera follows in a tracking shot{s}, moving forward behind {subject}"
                + (f" toward {tw}" if tw else "") + f"{why}.")
    if mv == "tracking_backward":
        return (f"The camera pulls out{s} in a tracking shot, moving backward ahead of {subject} "
                f"at the same pace{why}.")
    if mv == "arc":
        return f"The camera moves in an arc shot{a}{s} around {tw or subject}{why}."
    if mv == "shake":
        return f"The camera shakes slightly{why}."
    kind, side = mv.split("_")
    verb = {"pan": "pans", "truck": "trucks", "tilt": "tilts", "pedestal": "pedestals"}[kind]
    return f"The camera {verb} {side}{a}{s}{why}."


def restraint_sentence(c: dict) -> str:
    """How far a small move gets. H3 reads 'small amplitude' as a direction, not a distance: on the
    2026-10-11 pod test every small push ran the whole clip and ended in an extreme close-up (C s001,
    s003, s006). The sentence states the end of the move in framing terms the still can be held to."""
    mv = c.get("move", "static")
    if c.get("amplitude") == "slight":
        return ""                # the drift sentence already states the end framing
    if mv == "arc":
        # p01 smoke 3: an unbounded arc orbited ~180 degrees, crossed the axis and swapped who sat where
        return ("The arc is gentle: by the last frame the camera has travelled only a short way around, about "
                "thirty degrees, staying on the same side of the table, and everyone keeps their place and the room "
                "keeps its layout.")
    if c.get("amplitude") == "medium" and mv in ("push_in", "zoom_in", "tracking_forward"):
        # a real move with an end: one shot size, so it neither crawls nor runs into an extreme close-up
        # p05 smoke 3: without "steadily for the whole clip" H3 got there in 3 s and then sat dead
        return ("The camera keeps moving steadily for the whole clip, never stopping, and only at the last frame "
                "has it come about one shot size closer than Picture 1, the people it moves toward still in the frame.")
    if c.get("amplitude") == "medium" and mv in ("pull_out", "zoom_out", "tracking_backward"):
        return ("The camera keeps moving steadily for the whole clip, never stopping, and only at the last frame "
                "has it drawn back about one shot size wider than Picture 1.")
    if c.get("amplitude") == "medium" and mv != "static":
        return "The camera keeps moving steadily for the whole clip and is still moving at the last frame."
    if c.get("amplitude") != "small" or mv == "static":
        return ""
    if mv in ("push_in", "zoom_in", "tracking_forward"):
        return ("The move is slight and even: at the last frame the framing is only a little tighter than "
                "Picture 1, and everyone in Picture 1 is still in the frame.")
    if mv in ("pull_out", "zoom_out", "tracking_backward"):
        return "The move is slight and even: at the last frame the framing is only a little wider than Picture 1."
    return "The move is slight and even, and everyone in Picture 1 stays in the frame."


# ------------------------------------------------------------------ escalation

def escalation_band(e: float) -> str:
    """0-0.25 literal, 0.25-0.6 heightened, 0.6-1 abstract. The director's interp.reality wins
    when it is set; the band is what the translator uses for wording and lint keying."""
    return "literal" if e < 0.25 else "heightened" if e < 0.6 else "abstract"


def _wrongness(mp: dict, e: float) -> str:
    w = (mp.get("wrongness") or "").strip().rstrip(".")
    if not w or e < 0.25:
        return ""
    w = w[0].upper() + w[1:]
    if e < 0.6:
        return f"{w}, so slowly that it is hard to be sure it is happening."
    return f"{w}, without stopping." if re.search(r"\bslow", w, re.I) else f"{w}, slowly and without stopping."


# ------------------------------------------------------------------ people outside the still

WIDENS = {"arc", "pull_out", "zoom_out", "tracking_backward", "pan_left", "pan_right", "truck_left", "truck_right"}


def _offscreen(mp: dict, cam: dict) -> str:
    """People in the scene but outside Picture 1 come into view when the move widens or swings round to them
    (p05 smoke 3: the arc showed an empty seat where the woman sits, because 'no new characters enter')."""
    if cam.get("move") not in WIDENS:
        return ""
    return " ".join(_sent(f"As the camera moves, {p['who']}, {p['where']}, comes into view and stays in place there")
                    for p in mp.get("offscreen") or [] if p.get("who") and p.get("where"))


def _prohibit(x: str, mp: dict, cam: dict) -> str:
    if mp.get("offscreen") and cam.get("move") in WIDENS and re.search(r"new characters enter", x, re.I):
        return "No one else enters the frame."
    return x


# ------------------------------------------------------------------ build

def _sent(s: str) -> str:
    return pp._sent(s)


def translate(mp: dict, shot: dict, ctx: dict) -> str:
    """mp: motion plan. shot: the direction shot (still, subject). ctx: mode ('i2va'|'fl2va'),
    seconds, style_bible, strand (optional look per escalation band), voiceover (bool)."""
    prof = pp.load(PROFILE)
    mode = ctx.get("mode", "i2va")
    seconds = float(ctx.get("seconds") or 5.2)
    bible = ctx.get("style_bible") or {}
    interp = mp.get("interp") or {}
    e = float(interp.get("escalation") or 0.0)
    band = interp.get("reality") or escalation_band(e)
    subject = (mp.get("anchor") or shot.get("subject") or "the scene").strip()   # anchor: optional plain noun phrase
    scene = ctx.get("scene") or plan_scene(mp, shot)
    surface = bible.get("surface", "brushwork")

    head = prof["alignment"][mode].format(last_shot=1, seconds=seconds)
    d = prof["description"]
    strand = (ctx.get("strand") or {}).get(band, "")

    beat = mp.get("beat") or {}
    develop = beat.get("develop", "")
    movers = [f"{m['what']} {m['does']}" for m in mp.get("movers") or []
              if m.get("does") and m["does"].lower() not in develop.lower()
              and m["what"].lower() not in develop.lower()]
    cam = mp.get("camera") or {}
    end = mp.get("ending") or {}
    if mode == "fl2va":
        ending = d["landing"]["fl2va"].format(surface=surface)
    else:
        ending = {"settles": "In the last second the movement slows and the frame settles.",
                  "still_moving": "The movement is still under way at the last frame.",
                  "lands_on_target": (f"The move ends a little nearer {cam['toward']}, in the same continuous framing."
                                      if cam.get("toward") else ""),
                  "match": ""}.get(end.get("kind"), "")

    parts = [
        _sent(bible.get("style", "")),
        _sent(d["anchor"][mode].format(subject=subject[0].upper() + subject[1:], scene=scene,
                                       surface=surface)),
        _sent(bible.get("texture", "")),
        _sent(strand),
        _sent(beat.get("start", "")),                      # action onset: what is already under way
        camera_sentence(cam, cam.get("subject") or "the subject"),
        restraint_sentence(cam) if mode == "i2va" else "",   # FL2VA: the end frame fixes the distance
        _sent(develop),                                    # continuous development
        " ".join(_sent(m) for m in movers),
        # everyone in frame gets their own business: a figure with no verb freezes (p05 smoke 2)
        " ".join(_sent(f"{p['who']} {p['does']}") for p in mp.get("everyone") or []
                 if p.get("who") and p.get("does")),
        _wrongness(mp, e),
        _sent(beat.get("result", "")),                     # result or reaction
        ending,
        ONE_TAKE,
        _offscreen(mp, cam),
        # 2026-10-12 p01/p17 review: a continuation that travelled into unseen space re-invented faces and moved
        # people; it stays inside what the opening frames already show
        ((HOLD_STILL if mp.get("contained") or (mp.get("camera") or {}).get("move") == "static" else HOLD)
         if mp.get("hold_geography") else ""),
        CONTAIN if mp.get("contained") else "",
        " ".join(_sent(_prohibit(x, mp, cam)) for x in re.split(r"(?<=\.)\s+", bible.get("prohibitions", "")) if x),
        # p05 smoke 3: people stared into the lens
        "No one looks into the camera; their eyes stay on each other and the room.",
        " ".join(_sent(x) for x in mp.get("avoid") or []),
        # a conversation under the voice-over: mouths move (frozen faces read as dead), no words heard
        (TALK if mp.get("talk") else NO_SPEECH) if ctx.get("voiceover", True) else "",
    ]
    desc = f"{d['shot_label'].format(n=1)} " + " ".join(p for p in parts if p)
    sound = _sent(mp.get("sound", "")) or prof["fields"][1].get("empty", "")
    fields = {"description": desc, "soundscape": sound, "music": ""}
    body = [f"{f['name']}: {fields.get(f['content']) or f.get('empty', '')}" for f in prof["fields"]]
    sep = "\n\n" if prof["alignment"].get("blank_line_after") else "\n"
    return head + sep + prof.get("field_separator", "\n\n").join(body)
