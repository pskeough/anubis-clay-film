"""The director's per-shot motion plan: the schema, a validator, and a bootstrap from today's fields.

Used by stages.motion_one for shots that carry a motion_plan.

A motion plan is written by the director in film language. It never contains prompt text for a
model. `translate.py` turns it into an H3 prompt; `lint.py` checks both.

    "motion_plan": {
      "intent":   "what the viewer must take from this shot (never sent to the model)",
      "beat":     {"start": "what the still already shows under way",
                   "develop": "the one continuous visible event",
                   "result": "where the event has got to at the last frame"},
      "movers":   [{"what": "a thing the still shows", "does": "physical verb phrase, mid-action"}],
      "camera":   {"move": "static | push_in | pull_out | pan_left | pan_right | truck_left |
                            truck_right | tilt_up | tilt_down | pedestal_up | pedestal_down | arc |
                            tracking_forward | tracking_backward | zoom_in | zoom_out | shake",
                   "amplitude": "small | medium | large", "speed": "slow | normal | fast",
                   "toward": "a thing the still shows (push, zoom, tracking)",
                   "because": "short in-still clause that motivates the move, e.g. 'as she leans closer'",
                   "motivation": "hold | follow | reveal | approach | withdraw | carry (director_kb/03, 14)"},
      "continuity": {"from_prev": "how this shot picks up from the end of the shot before it",
                     "into_next": "what the last frame hands the next shot (a match, a contrast, a settle)"},
      "caption":  "what the approved still shows, written from looking at it (never the plan)",
      "ending":   {"kind": "settles | still_moving | lands_on_target | match",
                   "end_frame": "none | state | target | crop | state+crop", "change": "one edit"},
      "narration": {"relation": "illustrates | runs_against | anticipates | echoes",
                    "lands_on": "the word in the narration where the result should arrive"},
      "sound":    "diegetic sound only, concrete",
      "interp":   {"mode": "beat | place | detail", "reality": "literal | heightened | abstract",
                   "source": "clay | free", "escalation": 0.0},
      "wrongness": "for escalation > 0: what is wrong, on something the still shows",
      "avoid":    ["optional 'No ...' sentences, observable"]
    }
"""
from __future__ import annotations

import re

MOVES = {"static", "push_in", "pull_out", "pan_left", "pan_right", "truck_left", "truck_right",
         "tilt_up", "tilt_down", "pedestal_up", "pedestal_down", "arc", "tracking_forward",
         "tracking_backward", "zoom_in", "zoom_out", "shake"}
AMPLITUDES = {"slight", "small", "medium", "large"}   # slight: 16 Oct, a drift that holds past 10 s
SPEEDS = {"slow", "normal", "fast"}
ENDINGS = {"settles", "still_moving", "lands_on_target", "match"}
END_FRAMES = {"none", "state", "target", "crop", "state+crop"}
RELATIONS = {"illustrates", "runs_against", "anticipates", "echoes"}
MOTIVATIONS = {"hold", "follow", "reveal", "approach", "withdraw", "carry"}
MODES = {"beat", "place", "detail"}
REALITIES = {"literal", "heightened", "abstract"}
SOURCES = {"clay", "free"}


def validate(mp: dict) -> list[str]:
    """Schema problems in a motion plan (empty list = valid)."""
    out = []
    for k in ("intent", "beat", "movers", "camera", "ending", "interp"):
        if k not in mp:
            out.append(f"missing {k}")
    b = mp.get("beat") or {}
    for k in ("start", "develop", "result"):
        if not (b.get(k) or "").strip():
            out.append(f"beat.{k} is empty")
    if not mp.get("movers") and not (b.get("develop") or "").strip():
        out.append("nothing moves: give beat.develop or at least one mover")
    for m in mp.get("movers") or []:
        if not m.get("what") or not m.get("does"):
            out.append(f"mover needs what and does: {m}")
    c = mp.get("camera") or {}
    if c.get("move") not in MOVES:
        out.append(f"camera.move {c.get('move')!r} is not in {sorted(MOVES)}")
    if c.get("amplitude", "medium") not in AMPLITUDES:
        out.append(f"camera.amplitude {c.get('amplitude')!r}")
    if c.get("speed", "normal") not in SPEEDS:
        out.append(f"camera.speed {c.get('speed')!r}")
    mot = c.get("motivation")
    if mot is not None:   # required by the sequence pass (sequence.lint_sequence); older plans lack it
        if mot not in MOTIVATIONS:
            out.append(f"camera.motivation {mot!r} is not in {sorted(MOTIVATIONS)}")
        elif (mot == "hold") != (c.get("move") == "static"):
            out.append("camera.motivation 'hold' goes with move 'static' and only with it")
    e = mp.get("ending") or {}
    if e.get("kind") not in ENDINGS:
        out.append(f"ending.kind {e.get('kind')!r} is not in {sorted(ENDINGS)}")
    if e.get("end_frame", "none") not in END_FRAMES:
        out.append(f"ending.end_frame {e.get('end_frame')!r}")
    n = mp.get("narration") or {}
    if n and n.get("relation") not in RELATIONS:
        out.append(f"narration.relation {n.get('relation')!r} is not in {sorted(RELATIONS)}")
    i = mp.get("interp") or {}
    for k, allowed in (("mode", MODES), ("reality", REALITIES), ("source", SOURCES)):
        if i.get(k) not in allowed:
            out.append(f"interp.{k} {i.get(k)!r} is not in {sorted(allowed)}")
    esc = i.get("escalation", 0.0)
    if not isinstance(esc, (int, float)) or not 0.0 <= esc <= 1.0:
        out.append(f"interp.escalation {esc!r} must be a number from 0 to 1")
    return out


# ------------------------------------------------------------------ bootstrap
# Until the director writes motion plans, one can be guessed from today's direction fields.
# The guess is only a starting point: it cannot recover intent the director never wrote down.

_MOVE_WORDS = [  # order matters: the most specific phrase first
    (r"track\w*[^.]*\b(back|backward|ahead of)\b|moving back", "tracking_backward"),
    (r"track\w*|follow", "tracking_forward"),
    (r"pull(s|ing)? (back|out)|widen", "pull_out"),
    (r"push(es|ing)? in|push\b", "push_in"),
    (r"zoom(s|ing)? in", "zoom_in"), (r"zoom(s|ing)? out", "zoom_out"),
    (r"pan\w* left", "pan_left"), (r"pan\w* right", "pan_right"),
    (r"truck\w* left", "truck_left"), (r"truck\w* right", "truck_right"),
    (r"tilt\w* up", "tilt_up"), (r"tilt\w* down", "tilt_down"),
    (r"crane\w* up|pedestal\w* up|\brise", "pedestal_up"),
    (r"crane\w* down|pedestal\w* down", "pedestal_down"),
    (r"\barc\b|orbit", "arc"),
    (r"static|locked[- ]off|\bhold", "static"),
]


def guess_move(motion: str) -> str:
    m = (motion or "").lower()
    for pat, mv in _MOVE_WORDS:
        if re.search(pat, m):
            return mv
    return "static"


def bootstrap(shot: dict, transition_out: str = "cut") -> dict:
    """A first motion plan from a rebuild direction shot (direction/<scene>.json)."""
    motion = shot.get("motion") or ""
    m = motion.lower()
    amp = "large" if re.search(r"\blarge\b|sweep", m) else "small" if "small" in m else "medium"
    speed = "fast" if "fast" in m else "slow" if "slow" in m else "normal"
    movers = [{"what": c.get("who"), "does": c.get("action", "")} for c in shot.get("cast") or []
              if c.get("action")]
    return {
        "intent": shot.get("why", ""),
        "beat": {"start": shot.get("action", ""), "develop": shot.get("action", ""), "result": ""},
        "movers": movers or [{"what": shot.get("subject", ""), "does": shot.get("action", "")}],
        "camera": {"move": guess_move(motion), "amplitude": amp, "speed": speed,
                   "toward": shot.get("move_target") or ""},
        "ending": {"kind": "settles" if transition_out in ("dissolve", "fade") else "still_moving",
                   "end_frame": (shot.get("end") or {}).get("kind", "none")},
        "narration": {"relation": "illustrates", "lands_on": ""},
        "interp": {"mode": "beat", "reality": "literal", "source": "clay", "escalation": 0.0},
    }
