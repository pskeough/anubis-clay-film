"""The plan: the one file a person or a model writes. Everything downstream is compiled from it.

A plan names places by mark ("mid-centre"), people by cast id, and cameras by Roko vocabulary.
It never contains coordinates a model would have to invent. `compile_plan` turns it into the
files the Agentic_Slop stages already read (beats.json, shots.json, config.json, treatment.json)
plus clay_job.json for Roko's src/_filmprep.ts.

One shot = one narration sentence = one beat. The narrator reads ~2.4 words a second (measured
2026-09-24, 92 words = 41 s), and H3 clips run about 5 s, so 8-14 words per shot is the target.
"""
import json
from pathlib import Path

from . import cast as castmod
from . import locations

ANGLES = ["frontal", "three_quarter", "profile_left", "profile_right", "rear", "three_quarter_back",
          "over_shoulder"]
ELEVATIONS = ["eye", "low", "high", "ground", "overhead"]
DISTANCES = ["extreme_close", "close", "medium", "full_body", "wide", "very_wide"]
LENSES = [18, 24, 35, 50, 85, 135]
POSES = ["standing", "standing_hands_in_pockets", "arms_crossed", "seated", "seated_hunched",
         "crouching", "kneeling", "leaning_against_wall", "leaning_on_furniture", "walking",
         "mid_stride", "turning_away", "looking_back", "head_down", "head_tilted_back", "reaching",
         "holding_something", "hands_on_face", "lying_down", "sitting_on_floor"]
FRAMES = ["center", "left_third", "right_third", "far_from_camera", "close_to_edge", "small_in_frame"]
# Roko figure yaw: 0 faces +z (away from the near end), 180 faces z = 0. Left/right are left out
# until their handedness is checked against a render; a number in degrees is accepted as-is.
FACING = {"far": 0, "near": 180}
SUBJECT_CLASSES = ["figure", "place", "object"]
GRADES = ["ash", "ochre", "neutral", "vivid"]
WORDS_PER_SECOND = 2.3
READ_WPS = 2.4          # what the narrator actually reads; a shot lasts its narration at this rate
MODES = ["clay", "free"]

PERSON_SCHEMA = {
    "type": "object",
    "required": ["who", "at", "facing", "pose", "action", "placement"],
    "properties": {
        "who": {"type": "string", "description": "cast id"},
        "at": {"type": "string", "description": "a mark of the shot's location, e.g. mid-centre"},
        "facing": {"type": "string", "enum": list(FACING),
                   "description": "far = away from the near/camera end of the location, near = toward it"},
        "pose": {"type": "string", "enum": POSES},
        "action": {"type": "string", "description": "what they are doing, one clause, present participle"},
        "placement": {"type": "string", "description": "where they sit in the frame, plain words"},
    },
}
SHOT_SCHEMA = {
    "type": "object",
    "required": ["narration", "camera", "people", "subject", "scene", "action", "motion"],
    "properties": {
        "narration": {"type": "string", "description": "the one sentence read over this shot"},
        "location": {"type": "string", "description": "only when different from the plan's location"},
        "camera": {
            "type": "object", "required": ["angle", "elevation", "distance", "lens"],
            "properties": {
                "angle": {"type": "string", "enum": ANGLES},
                "elevation": {"type": "string", "enum": ELEVATIONS},
                "distance": {"type": "string", "enum": DISTANCES},
                "lens": {"type": "integer", "enum": LENSES},
                "frame": {"type": "string", "enum": FRAMES},
            },
        },
        "people": {"type": "array", "items": PERSON_SCHEMA, "maxItems": 3,
                   "description": "first person is the one the camera is built around; [] for an empty shot"},
        "look_at": {"type": "string", "description": "empty shots only: a look_at prop id to frame"},
        "subject": {"type": "string", "description": "short noun phrase for what the shot is about, e.g. 'the hooded man'"},
        "subject_class": {"type": "string", "enum": SUBJECT_CLASSES},
        "scene": {"type": "string", "description": "one or two sentences describing the still as a viewer sees it"},
        "action": {"type": "string", "description": "what moves during the ~5 s clip"},
        "motion": {"type": "string", "description": "the camera move, e.g. 'a slow push in', 'a locked-off hold'"},
        "move_target": {"type": "string", "description": "what the camera moves toward, if not the subject"},
        "soundscape": {"type": "string"},
        "movement": {"type": "string", "description": "section name, for grading; default 'main'"},
    },
}
PLAN_SCHEMA = {
    "type": "object",
    "required": ["title", "location", "shots"],
    "properties": {
        "title": {"type": "string"},
        "location": {"type": "string"},
        "seed": {"type": "integer"},
        "grade": {"type": "string", "enum": GRADES},
        "narrator": {"type": "string", "description": "optional TTS voice design text"},
        "voice_ref": {"type": "object", "description": "optional narrator to clone: wav, text (its "
                      "transcript), start and end in seconds. Long passages are spoken in several "
                      "takes, and cloning one reference is what keeps them one voice"},
        "cast": {"type": "object", "additionalProperties": {"type": "string"},
                 "description": "id -> full appearance, for people not in the bible or to override one"},
        "shots": {"type": "array", "items": SHOT_SCHEMA, "minItems": 1},
    },
}

# A `free` shot has no Roko geometry: its still is text-to-image from still_prompt, with the
# plan's `style` clause in front and the plan's seed, so abstract or stylised passages that are
# not set in a Roko location can still be filmed. People are cast ids only, for their appearance.
FREE_SHOT_SCHEMA = {
    "type": "object",
    "required": ["mode", "narration", "still_prompt", "subject", "scene", "action", "motion"],
    "properties": {
        "mode": {"type": "string", "enum": ["free"]},
        "narration": {"type": "string", "description": "the one sentence read over this shot"},
        "still_prompt": {"type": "string", "description": "the picture, described as a photograph or "
                         "image a viewer sees: medium, subject, setting, light. Present tense, no instructions"},
        "people": {"type": "array", "maxItems": 3, "items": {
            "type": "object", "required": ["who"],
            "properties": {"who": {"type": "string", "description": "cast id"},
                           "action": {"type": "string"}, "placement": {"type": "string"}}}},
        "camera": {"type": "object", "properties": {
            "angle": {"type": "string", "enum": ANGLES},
            "elevation": {"type": "string", "enum": ELEVATIONS},
            "distance": {"type": "string", "enum": DISTANCES},
            "lens": {"type": "integer", "enum": LENSES}}},
        "subject": {"type": "string"},
        "subject_class": {"type": "string", "enum": SUBJECT_CLASSES},
        "scene": {"type": "string", "description": "one or two sentences describing the still"},
        "action": {"type": "string", "description": "what moves during the ~5 s clip"},
        "motion": {"type": "string", "description": "the camera move"},
        "move_target": {"type": "string"},
        "soundscape": {"type": "string"},
        "movement": {"type": "string"},
    },
}
FREE_PLAN_SCHEMA = {
    "type": "object",
    "required": ["title", "style", "shots"],
    "properties": {
        "title": {"type": "string"},
        "style": {"type": "string", "description": "one style clause put in front of every still prompt"},
        "motion_style": {"type": "string", "description": "replaces 'Live-action cinematic film' at the head of every H3 prompt"},
        "seed": {"type": "integer"},
        "grade": {"type": "string", "enum": GRADES},
        "cast": {"type": "object", "additionalProperties": {"type": "string"}},
        "shots": {"type": "array", "items": FREE_SHOT_SCHEMA, "minItems": 1},
    },
}


def _enum(errs, where, val, allowed):
    if val not in allowed:
        errs.append(f"{where}: {val!r} is not one of {allowed}")


def shot_mode(s: dict) -> str:
    return s.get("mode") or "clay"


def _narration(errs: list, w: str, s: dict, passage: str | None) -> int:
    narr = (s.get("narration") or "").strip()
    cap = 34 if passage is not None else 22  # an author's long sentence cannot be split
    if len(narr.split()) > cap:
        errs.append(f"{w}.narration: {len(narr.split())} words is too long for one shot; "
                    f"keep it under {cap}" + ("" if passage is not None else " or split it into two shots"))
    if passage is not None and narr and " ".join(narr.split()) not in passage:
        errs.append(f"{w}.narration: must be copied verbatim from the passage; {narr[:60]!r} is not in it")
    return len(narr.split())


def _check_free(errs: list, w: str, s: dict, reg: dict, passage: str | None) -> int:
    """A free shot: no location, marks or look_at; people are cast ids for appearance only."""
    for k in ("narration", "subject", "scene", "action", "motion"):
        if k not in s:
            errs.append(f"{w}: missing '{k}'")
    if not (s.get("still_prompt") or s.get("scene") or "").strip():
        errs.append(f"{w}.still_prompt: a free shot needs a still_prompt (or a scene) to draw")
    words = _narration(errs, w, s, passage)
    cam = s.get("camera") or {}
    if not isinstance(cam, dict):
        errs.append(f"{w}.camera: must be an object")
        cam = {}
    for k, allowed in (("angle", ANGLES), ("elevation", ELEVATIONS), ("distance", DISTANCES),
                       ("lens", LENSES)):
        if cam.get(k) is not None:
            _enum(errs, f"{w}.camera.{k}", cam[k], allowed)
    people = s.get("people") or []
    if not isinstance(people, list):
        errs.append(f"{w}.people: must be a list")
        people = []
    if len(people) > 3:
        errs.append(f"{w}.people: at most 3 people per shot")
    for j, p in enumerate(people):
        if not isinstance(p, dict) or p.get("who") not in reg:
            errs.append(f"{w}.people[{j}].who: {(p or {}).get('who')!r} is not in the cast; add it to "
                        f"plan.cast with a full appearance")
    if s.get("subject_class") is not None:
        _enum(errs, f"{w}.subject_class", s["subject_class"], SUBJECT_CLASSES)
    return words


def validate(plan: dict, passage: str | None = None, target_seconds: float | None = None) -> list[str]:
    """Every problem in the plan, phrased so a model can fix it. [] means it compiles."""
    errs: list[str] = []
    if passage is not None:
        passage = " ".join(passage.split())
    shots_in = plan.get("shots") if isinstance(plan.get("shots"), list) else []
    # plan.location is needed only by clay shots that do not name their own location
    needs_loc = not shots_in or any(shot_mode(s) != "free" and not s.get("location")
                                    for s in shots_in if isinstance(s, dict))
    for k in PLAN_SCHEMA["required"]:
        if k not in plan and (k != "location" or needs_loc):
            errs.append(f"plan: missing '{k}'")
    if errs:
        return errs
    locs = set(locations.ids())
    if "location" in plan and plan["location"] not in locs:
        errs.append(f"plan.location: {plan['location']!r} unknown; use one of the listed locations")
    if plan.get("grade") is not None:
        _enum(errs, "plan.grade", plan["grade"], GRADES)
    reg = castmod.registry(plan.get("cast"))
    shots = plan["shots"]
    if not isinstance(shots, list) or not shots:
        return errs + ["plan.shots: must be a non-empty list"]
    words = 0
    for i, s in enumerate(shots):
        w = f"shots[{i}]"
        if shot_mode(s) == "free":
            words += _check_free(errs, w, s, reg, passage)
            continue
        if shot_mode(s) != "clay":
            errs.append(f"{w}.mode: {s.get('mode')!r} is not one of {MODES}")
            continue
        for k in SHOT_SCHEMA["required"]:
            if k not in s:
                errs.append(f"{w}: missing '{k}'")
        loc = s.get("location") or plan["location"]
        if loc not in locs:
            errs.append(f"{w}.location: {loc!r} unknown")
            continue
        narr = (s.get("narration") or "").strip()
        words += len(narr.split())
        cap = 34 if passage is not None else 22  # an author's long sentence cannot be split
        if len(narr.split()) > cap:
            errs.append(f"{w}.narration: {len(narr.split())} words is too long for one shot; "
                        f"keep it under {cap}" + ("" if passage is not None else " or split it into two shots"))
        if passage is not None and narr and narr not in passage:
            errs.append(f"{w}.narration: must be copied verbatim from the passage; {narr[:60]!r} is not in it")
        cam = s.get("camera") or {}
        for k, allowed in (("angle", ANGLES), ("elevation", ELEVATIONS), ("distance", DISTANCES),
                           ("lens", LENSES)):
            _enum(errs, f"{w}.camera.{k}", cam.get(k), allowed)
        if cam.get("frame") is not None:
            _enum(errs, f"{w}.camera.frame", cam["frame"], FRAMES)
        if cam.get("distance") in ("wide", "very_wide") and not locations.describe(loc, False)["exterior"]:
            errs.append(f"{w}.camera.distance: {cam['distance']} needs an exterior; {loc} is an interior, "
                        f"use full_body or tighter")
        people = s.get("people")
        if not isinstance(people, list):
            errs.append(f"{w}.people: must be a list ([] for an empty shot)")
            continue
        if len(people) > 3:
            errs.append(f"{w}.people: at most 3 people per shot")
        for j, p in enumerate(people):
            pw = f"{w}.people[{j}]"
            for k in PERSON_SCHEMA["required"]:
                if k not in p:
                    errs.append(f"{pw}: missing '{k}'")
            if p.get("who") not in reg:
                errs.append(f"{pw}.who: {p.get('who')!r} is not in the cast; add it to plan.cast with a full appearance")
            if "at" in p:
                try:
                    locations.resolve_mark(loc, p["at"])
                except (KeyError, ValueError, TypeError) as e:
                    errs.append(f"{pw}.at: {e}")
            if "facing" in p and p["facing"] not in FACING and not isinstance(p["facing"], (int, float)):
                errs.append(f"{pw}.facing: {p['facing']!r} is not one of {list(FACING)}")
            if "pose" in p:
                _enum(errs, f"{pw}.pose", p["pose"], POSES)
        if s.get("look_at") and s["look_at"] not in locations.prop_ids(loc):
            errs.append(f"{w}.look_at: {s['look_at']!r} is not a prop id in {loc}")
        if s.get("subject_class") is not None:
            _enum(errs, f"{w}.subject_class", s["subject_class"], SUBJECT_CLASSES)
    if target_seconds:
        budget = target_seconds * WORDS_PER_SECOND
        if words > budget * 1.2 or words < budget * 0.6:
            errs.append(f"plan: narration is {words} words; {target_seconds:.0f} s needs about "
                        f"{budget:.0f} (the narrator reads ~2.4 words a second)")
    return errs


# The clay carries no figure, so pose reaches the still only as words. Until 2026-09-24 it
# reached it not at all: Anubis s004/s005 asked for a seated clerk and got him standing, or
# perched on the desk top. Skipped when the action already says it.
POSE_WORDS = {
    "standing": "standing", "standing_hands_in_pockets": "standing with hands in pockets",
    "arms_crossed": "standing with arms crossed", "seated": "sitting on a chair",
    "seated_hunched": "sitting hunched forward on a chair", "crouching": "crouching low",
    "kneeling": "kneeling", "leaning_against_wall": "leaning back against the wall",
    "leaning_on_furniture": "leaning on the furniture beside them", "walking": "walking",
    "mid_stride": "caught mid-stride", "turning_away": "turning away",
    "looking_back": "looking back over the shoulder", "head_down": "head bowed",
    "head_tilted_back": "head tilted back", "reaching": "reaching out", "holding_something": "",
    "hands_on_face": "hands over the face", "lying_down": "lying down",
    "sitting_on_floor": "sitting on the floor",
}


# How big the lead person is in frame. The clay has no figure and the camera line gives only
# height and field of view, so a `close` on a seated clerk came back as a small standing man in
# the aisle behind the desks (Anubis s009, s015, s019, s025, first stills pass).
FIGURE_SCALE = {
    "extreme_close": "framed as an extreme close-up, the face filling the frame",
    "close": "framed as a close-up, head and shoulders filling most of the frame",
    "medium": "framed from the waist up, large in the frame",
    "full_body": "seen full length",
}


def _people_line(p: dict, reg: dict) -> dict:
    action = p["action"].strip().rstrip(".")
    pose = POSE_WORDS.get(p.get("pose", ""), "")
    if pose and pose.split()[0] not in action.lower():
        action = f"{pose}, {action}"
    return {"desc": reg[p["who"]], "action": action,
            "placement": p["placement"].strip().rstrip(".")}


def _yaw(f) -> float:
    return float(f) if isinstance(f, (int, float)) else float(FACING[f])


CAMERA_WORDS = {
    "distance": {"extreme_close": "an extreme close-up", "close": "a close-up",
                 "medium": "a medium shot, waist up", "full_body": "a full-length shot",
                 "wide": "a wide shot", "very_wide": "a very wide establishing shot"},
    "elevation": {"eye": "at eye level", "low": "from a low angle looking up",
                  "high": "from a high angle looking down", "ground": "from ground level",
                  "overhead": "from directly overhead"},
    "angle": {"frontal": "facing the subject", "three_quarter": "from a three-quarter angle",
              "profile_left": "in profile", "profile_right": "in profile", "rear": "from behind",
              "three_quarter_back": "from behind and to one side",
              "over_shoulder": "over the shoulder"},
}


def camera_words(cam: dict | None) -> str:
    """Camera vocabulary as one plain sentence, for free shots that have no solver."""
    cam = cam or {}
    bits = [CAMERA_WORDS[k][cam[k]] for k in ("distance", "angle", "elevation")
            if cam.get(k) in CAMERA_WORDS[k]]
    if cam.get("lens") in LENSES:
        bits.append(f"on a {cam['lens']} mm lens")
    if not bits:
        return ""
    s = ", ".join(bits)
    return s[0].upper() + s[1:] + "."


def free_prompt(plan: dict, s: dict) -> str:
    """The text-to-image prompt of a free shot: style clause, the picture, each person's canon
    appearance (the same words in every shot, as for clay), then the camera."""
    reg = castmod.registry(plan.get("cast"))
    parts = []
    style = " ".join((plan.get("style") or "").split()).rstrip(". ")
    if style:
        parts.append(style + ".")
    body = " ".join((s.get("still_prompt") or s.get("scene") or "").split()).rstrip(". ")
    if body:
        parts.append(body[0].upper() + body[1:] + ".")
    for p in s.get("people") or []:
        bits = [reg.get(p.get("who"), "")] + [" ".join((p.get(k) or "").split()).rstrip(". ")
                                              for k in ("action", "placement")]
        line = ", ".join(b for b in bits if b)
        if line:
            parts.append(line[0].upper() + line[1:] + ".")
    cam = camera_words(s.get("camera"))
    if cam:
        parts.append(cam)
    return " ".join(parts)


def est_seconds(narration: str) -> float:
    return round(len((narration or "").split()) / READ_WPS, 2)


def coverage(plan: dict, source: str) -> dict:
    """Where each shot's narration sits in the source, and what is left over.

    Passage mode wants the narration of all shots, joined in order, to equal the source
    (whitespace-normalised). Each shot is looked up from where the previous one ended; a shot
    not found there is looked up from the start, which is an overlap or an out-of-order shot."""
    src = " ".join((source or "").split())
    spans, covered, cursor, errs = [], [0] * len(src), 0, []
    for i, s in enumerate(plan.get("shots") or []):
        n = " ".join((s.get("narration") or "").split())
        a = src.find(n, cursor) if n else -1
        if a < 0 and n:
            a = src.find(n)
        if a < 0:
            spans.append({"i": i, "start": None, "end": None, "found": False})
            errs.append(f"shots[{i}].narration: not in the source" if n else f"shots[{i}].narration: empty")
            continue
        b = a + len(n)
        overlap = any(covered[a:b])
        if a < cursor or overlap:
            errs.append(f"shots[{i}].narration: overlaps or repeats source text already used")
        for k in range(a, b):
            covered[k] += 1
        spans.append({"i": i, "start": a, "end": b, "found": True, "overlap": overlap})
        cursor = max(cursor, b)
    gaps, overlaps, k = [], [], 0
    while k < len(src):
        if covered[k] == 0:
            j = k
            while j < len(src) and covered[j] == 0:
                j += 1
            if src[k:j].strip():
                gaps.append([k, j])
            k = j
        elif covered[k] > 1:
            j = k
            while j < len(src) and covered[j] > 1:
                j += 1
            overlaps.append([k, j])
            k = j
        else:
            k += 1
    for a, b in gaps:
        errs.append(f"source: not narrated by any shot: {src[a:b].strip()[:60]!r}")
    joined = " ".join(" ".join((s.get("narration") or "").split()) for s in plan.get("shots") or [])
    return {"source": src, "spans": spans, "gaps": gaps, "overlaps": overlaps,
            "exact": joined == src, "errors": errs}


def compile_plan(plan: dict, piece: Path) -> dict:
    """Write the stage files for a validated plan. Returns clay_job.json's contents.

    Free shots go into beats/shots but not into clay_job.json, so the clay stage skips them.
    Optional per-shot fields (still_prompt, motion_prompt, mode) reach shots.json only when set,
    so a plan without them compiles byte-identically to before they existed."""
    piece = Path(piece)
    piece.mkdir(parents=True, exist_ok=True)
    reg = castmod.registry(plan.get("cast"))
    beats, shots, job_shots, movements = [], [], [], []
    for i, s in enumerate(plan["shots"]):
        mv = s.get("movement") or "main"
        if not movements or movements[-1]["name"] != mv:
            movements.append({"name": mv, "shots": 0})
        movements[-1]["shots"] += 1
        if shot_mode(s) == "free":
            people = s.get("people") or []
            beats.append(s["narration"].strip())
            row = {
                "i": i, "covers_beats": [i], "movement": mv, "relation": "literal",
                "subject_class": s.get("subject_class") or ("figure" if people else "place"),
                "narration": s["narration"].strip(), "location": s.get("location") or plan.get("location", ""),
                "mode": "free",
                "people": [{k: p.get(k, "") for k in ("who", "action", "placement")} for p in people],
                "scene": s["scene"], "grounded_in": s["subject"], "action": s["action"],
                "motion": s["motion"], "move_target": s.get("move_target", ""),
                "soundscape": s.get("soundscape", ""), "still_prompt": free_prompt(plan, s),
            }
            for k in ("motion_prompt", "tail_s"):
                if s.get(k):
                    row[k] = s[k]
            shots.append(row)
            continue
        loc = s.get("location") or plan["location"]
        people = s["people"]
        cam = s["camera"]
        roko = {"angle": cam["angle"], "elevation": cam["elevation"], "distance": cam["distance"],
                "lens": cam["lens"]}
        if cam.get("frame"):
            roko["placement"] = cam["frame"]
        job = {"sid": f"s{i:03d}", "location": loc, "people": [_people_line(p, reg) for p in people]}
        scale = FIGURE_SCALE.get(cam["distance"], "")
        if people and scale:
            job["people"][0]["placement"] += f", {scale}"
        if people:
            lead = people[0]
            roko["pose"] = lead["pose"]
            roko["figureAt"] = list(locations.resolve_mark(loc, lead["at"]))
            roko["figureYaw"] = _yaw(lead["facing"])
            roko["extraFigures"] = [{"at": list(locations.resolve_mark(loc, p["at"])),
                                     "yaw": _yaw(p["facing"]), "pose": p["pose"], "id": p["who"]}
                                    for p in people[1:]]
        else:
            roko["emptySubject"] = s.get("look_at")
        job.update({"shot": {k: roko[k] for k in ("angle", "elevation", "distance", "lens")}
                    | ({"pose": roko["pose"]} if "pose" in roko else {})
                    | ({"placement": roko["placement"]} if "placement" in roko else {})
                    | ({"action": people[0]["action"]} if people else {}),
                    "figureAt": roko.get("figureAt"), "figureYaw": roko.get("figureYaw", 0),
                    "extraFigures": roko.get("extraFigures", []),
                    "emptySubject": roko.get("emptySubject")})
        job_shots.append(job)
        beats.append(s["narration"].strip())
        shots.append({
            "i": i, "covers_beats": [i], "movement": mv, "relation": "literal",
            "subject_class": s.get("subject_class") or ("figure" if people else "place"),
            "narration": s["narration"].strip(), "location": loc, "roko": roko,
            "people": [{k: p[k] for k in ("who", "action", "placement")} for p in people],
            "scene": s["scene"], "grounded_in": s["subject"], "action": s["action"],
            "motion": s["motion"], "move_target": s.get("move_target", ""),
            "soundscape": s.get("soundscape", ""),
        })
        for k in ("still_prompt", "motion_prompt", "tail_s"):
            if s.get(k):
                shots[-1][k] = s[k]
    if plan.get("motion_style"):
        for row in shots:
            row["motion_style"] = plan["motion_style"]
    grade = plan.get("grade") or "ash"
    cfg = {"title": plan["title"], "location": plan.get("location", ""), "seed": plan.get("seed", 31337),
           "beat_seconds": 5.2, "grades": [grade] * len(movements), "compiled_by": "clay_film"}
    if plan.get("narrator"):
        cfg["narrator"] = plan["narrator"]
    if plan.get("voice_ref"):
        cfg["voice_ref"] = plan["voice_ref"]
    if plan.get("style"):
        cfg["style"] = plan["style"]
    wr = lambda name, obj: (piece / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False),
                                                     encoding="utf-8")
    wr("beats.json", beats)
    wr("shots.json", shots)
    wr("config.json", cfg)
    wr("treatment.json", {"movements": movements})
    job = {"out": str(piece / "blockout"), "width": 1344, "height": 768, "shots": job_shots}
    wr("clay_job.json", job)
    return job


def example() -> dict:
    """A small valid plan in the Basilisk's lakeside location, for prompts and the selftest."""
    return {
        "title": "Lakeside, the stranger",
        "location": "lakeside-pine-path",
        "seed": 31337,
        "grade": "ash",
        "cast": {"stranger": "a man in a dark hooded jacket with the hood pulled far forward, the "
                             "inside of the hood a void of solid black shadow where a face would be"},
        "shots": [
            {"narration": "Another twig snapped, this time closer.",
             "camera": {"angle": "frontal", "elevation": "low", "distance": "full_body", "lens": 35},
             "people": [], "look_at": "path_mid", "subject": "the empty pine path",
             "scene": "A flagstone trail runs down through tall pines toward a grey lake.",
             "action": "nothing moves on the trail; a few needles drift down",
             "motion": "a very slow push forward down the path"},
            {"narration": "There, in the distance, was the outline of a human.",
             "camera": {"angle": "rear", "elevation": "low", "distance": "medium", "lens": 35,
                        "frame": "left_third"},
             "people": [
                 {"who": "menote", "at": "near-centre", "facing": "far", "pose": "kneeling",
                  "action": "kneeling on the flagstones with his back to the camera",
                  "placement": "large in the left foreground, seen from behind"},
                 {"who": "stranger", "at": "far-centre", "facing": "near", "pose": "walking",
                  "action": "walking up the trail toward the camera",
                  "placement": "small and far down the path"}],
             "subject": "the kneeling young man",
             "scene": "Over the shoulder of a young man kneeling on a forest path, a distant hooded figure walks toward him.",
             "action": "far down the trail the hooded figure keeps walking closer",
             "motion": "a slow push in over his shoulder", "move_target": "the distant hooded figure"},
        ],
    }
