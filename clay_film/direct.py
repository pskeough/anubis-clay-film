"""Director mode: a frontier model directs, Python enforces.

    python -m clay_film direct check      NAME     errors and warnings; exit 1 on any error
    python -m clay_film direct continuity NAME     adjacent-shot report -> <adaptation>/CONTINUITY.md
    python -m clay_film direct board      NAME     the shot board as a markdown table
    python -m clay_film direct refs       NAME     reference-image plan -> <adaptation>/refs_plan.json
    python -m clay_film direct compile    NAME     direction.json -> <adaptation>/plan.directed.json
    python -m clay_film direct animatic   NAME     timed cards (or stills) + narration -> animatic.mp4
    python -m clay_film direct signoff    NAME --watched "what you noticed"

NAME is an adaptation folder (adaptations/NAME/direction.json) or a path to a direction.json.

Why this exists. The Oceanus board (adaptations/oceanus/build_plan.py, 78 shots) was written by
hand, one literal picture per narration clause, and bypassed both planners. Neither planner would
have saved it: clay_film's planner is "one shot per narration sentence" by rule, and
Agentic_Slop's director.py has movements and a ledger but no idea of a location, a character's
state, a story moment or screen direction, so every failure Patrick named on the cut (the cook
in the water, then indoors, then on deck; Jacob and Betty at the bar twice with the roles
swapped; a screen-direction flip on the first cut; stones in a boat for "no stones cast") passes
its checks. AUDIT.md in adaptations/oceanus has the line-by-line version.

So the decisions move to a document the main Claude session writes as the director: Pass A (the
read) and Pass B (movements, then shots against a running ledger) are sections of
direction.json, and the continuity bible is its spine. This module does not have an opinion
about any shot. It checks the document against itself and against the source text:

  BIBLE      every shot's location, register, stage, characters and their states exist in the
             bible and agree with it; a character's blocking in a location is fixed there
  MOMENTS    a story moment appears once, in adjacent shots; any later return is a declared rhyme
  REGISTERS  every change of register (present / memory / myth) carries a transition and a bridge
  SPINE      a location marked monotonic (the water) only degrades: its stages never go backward
  ARCS       a character's state in one register only moves forward along its arc
  ANCHORS    each shot starts at a quoted phrase found in the source, in order, so the board
             survives Patrick's edit of the text; a lost phrase reports its nearest match
  FIGURES    a phrase the read marked figurative is never played under a literal depiction
  ORPHANS    a shot with no character says why, and shares a location with a neighbour, carries a
             motif, or is myth; the "divorced from anything" shots fail here
  SCREEN     horizontal travel matches the location's axis and does not flip between neighbours
  RHYTHM     shot count and average shot length inside the targets; clip lengths reported

The animatic gate: compile stamps the plan with a hash of the direction, and the stills stage
refuses to render a stamped piece until an animatic of that exact direction has been built and
signed off. Sequence is judged in time, before anyone looks at a contact sheet.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

from . import paths

# The same ten as Agentic_Slop's director.RELATIONS (selftest checks they agree).
RELATIONS = ["depict", "detail", "motif", "displace", "aftermath", "absence", "counterpoint",
             "abstract", "rhyme", "hold"]
SIZES = ["extreme_close", "close", "medium", "full_body", "wide", "very_wide"]
SCREEN = ["L>R", "R>L", "up", "down", "toward", "away", "none"]
TRANSITIONS = ["cut", "match_cut", "dissolve", "swirl", "fade"]
HORIZONTAL = {"L>R", "R>L"}
# Under 3 s a picture is a flash, not a shot: Patrick saw "an unplanned still" before the cups
# shot and "two flashing stills" before "I had always loved the sea" on the Oceanus v1 cut.
MIN_SHOT_S = 3.0
# People the bible does not name. v1 filled the tavern and deck with anonymous sailors and drinkers
# where the story has Jacob, Allam, Betty and the owner ("use the main characters", v1 s043).
EXTRAS = re.compile(r"\b(sailors?|crewmen|crew|drinkers?|strangers?|men|women|people|figures?|"
                    r"villagers|passengers|onlookers)\b", re.I)
REQUIRED = ["id", "movement", "register", "location", "cast", "moment", "under", "relation",
            "size", "screen", "picture", "action", "motion"]

ADAPT = paths.ADAPTATIONS


# -- loading ---------------------------------------------------------------------------------

def find(name: str) -> Path:
    p = Path(name)
    if p.suffix == ".json" and p.is_file():
        return p.resolve()
    p = ADAPT / name / "direction.json"
    if p.is_file():
        return p
    raise FileNotFoundError(f"no direction.json for {name!r} (looked in {p})")


def load(name: str) -> tuple[dict, Path]:
    f = find(name)
    return json.loads(f.read_text(encoding="utf-8")), f


def source_text(d: dict, base: Path) -> str:
    """The narration text, whitespace-normalised. `source` is a text file, or a markdown file and
    one of its `## ` sections ("EDIT_PROPOSAL.md#Proposed text"), so a board can be planned
    against an edit before the edit is applied. In a section, an editor's marked wording
    `**[mine: think]**` is read as the word itself and other bold markers are dropped."""
    ref, _, section = d["source"].partition("#")
    src = Path(ref)
    if not src.is_absolute():
        src = base.parent / src
    raw = src.read_text(encoding="utf-8")
    if section:
        m = re.search(r"^##\s+" + re.escape(section) + r"\s*$(.*?)(?=^##\s|\Z)", raw, re.M | re.S)
        if not m:
            raise ValueError(f"{src.name} has no '## {section}' section")
        raw = re.sub(r"\*\*\[mine:\s*([^\]]+)\]\*\*", r"\1", m.group(1)).replace("**", "")
    return " ".join(raw.split())


def write_source(d: dict, base: Path, src: str) -> Path:
    """The exact text a compiled plan narrates, frozen next to the direction. clay_film's own
    verbatim checks (validate, compile_stage) read plan.source, so it has to be a plain file."""
    out = base.parent / "narration_source.txt"
    out.write_text(src + "\n", encoding="utf-8")
    return out


def digest(d: dict) -> str:
    """What the animatic sign-off is bound to: the bible and the shots, nothing cosmetic."""
    body = json.dumps({"bible": d.get("bible"), "shots": d.get("shots")}, sort_keys=True,
                      ensure_ascii=False)
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]


# -- text ------------------------------------------------------------------------------------

_FOLD = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                       "–": "-", "—": "-"})


def norm(text: str) -> str:
    """Lower case, straight quotes, single spaces. Same length as `" ".join(text.split())`, so an
    index into the normalised text is an index into the whitespace-normalised source."""
    return " ".join((text or "").split()).translate(_FOLD).lower()


def nearest(phrase: str, src: str) -> str:
    """The run of source words most like a lost anchor, to re-anchor after an edit."""
    words, k = src.split(" "), max(1, len(phrase.split()))
    best, at = 0.0, ""
    for i in range(0, max(1, len(words) - k + 1)):
        cand = " ".join(words[i:i + k])
        r = difflib.SequenceMatcher(None, norm(phrase), norm(cand)).ratio()
        if r > best:
            best, at = r, cand
    return f"{at!r} ({best:.0%} alike)" if at else "nothing close"


def anchors(d: dict, src: str) -> tuple[list[int | None], list[str]]:
    """Where each shot's `under` phrase starts in the source, searched forward from the last."""
    n, out, errs, cursor = norm(src), [], [], 0
    for s in d.get("shots", []):
        ph = norm(s.get("under", ""))
        a = n.find(ph, cursor) if ph else -1
        if a < 0:
            back = n.find(ph) if ph else -1
            why = ("comes before the shot ahead of it; shots follow the text in order" if back >= 0
                   else f"is not in the source; nearest is {nearest(s.get('under', ''), src)}")
            errs.append(f"{s.get('id')}: under {s.get('under', '')!r} {why}")
            out.append(None)
            continue
        if out and a == next((x for x in reversed(out) if x is not None), -1):
            errs.append(f"{s.get('id')}: starts at the same words as the shot before it; every "
                        f"shot carries some narration (use hold_s to hold a picture longer)")
        out.append(a)
        cursor = a + 1
    if out and out[0] not in (None, 0):
        errs.append(f"{d['shots'][0].get('id')}: the first shot must start at the first words of "
                    f"the source ({src[:40]!r}...)")
    return out, errs


def narration(d: dict, src: str) -> list[str]:
    """Each shot's narration: the source from its anchor to the next shot's. Joined, they are the
    source exactly, which is what clay_film's verbatim check wants."""
    pos, _ = anchors(d, src)
    if any(p is None for p in pos):
        return []
    ends = pos[1:] + [len(src)]
    return [src[a:b].strip() for a, b in zip(pos, ends)]


def seconds(d: dict, s: dict, text: str) -> float:
    wps = float(d.get("targets", {}).get("wps", 2.4))
    return round(len(text.split()) / wps + float(s.get("hold_s") or 0), 1)


# -- the checks ------------------------------------------------------------------------------

# Copied from Agentic_Slop's director.py (lint_scene / lint_action), which is not shipped here:
# importing it pulls in that pipeline's LLM planner. Both were learned on finished films.
# Readable lettering: no image model here can spell, and the negative prompt is inert at cfg 1.0.
TEXT_IN_SCENE = re.compile(
    r"\b(inscription|inscribed|lettering|letters|lettered|wording|"
    r"writing|written|script|engraved|signage|signboard|placard|plaque|"
    r"nameplate|label|labelled|caption|banner|legible|spelled|spelling|"
    r"typography|text)\b", re.I)
# Stillness written into an action: H3 obeys it literally and returns a frozen clip.
STILL_IN_ACTION = re.compile(
    r"\b(hold(s|ing)?\s+(still|steady|motionless)|stay(s|ing)?\s+"
    r"(still|fixed|put|unchanged|the\s+same)|remain(s|ing)?\s+"
    r"(still|fixed|unchanged|motionless)|motionless|unmoving|"
    r"without\s+moving|no\s+movement|nothing\s+moves|unchanged|"
    r"perfectly\s+still|frozen|static)\b", re.I)


def _lint_scene(scene: str) -> str | None:
    """The reason this scene cannot be photographed, or None."""
    m = TEXT_IN_SCENE.search(scene or "")
    if not m:
        return None
    return (f"it asks for readable lettering ({m.group(0)!r}); no image model here can spell and the "
            f"negative prompt is inert at cfg 1.0, so this renders as a sign covered in nonsense")


def _lint_action(action: str) -> str | None:
    """The reason this action will produce a frozen clip, or None."""
    m = STILL_IN_ACTION.search(action or "")
    if m:
        return (f"it asks for stillness ({m.group(0)!r}); the video model obeys that literally and "
                f"returns a clip with nothing happening in it")
    if len((action or "").split()) < 4:
        return "it is too short to name an event; the camera move alone only slides across a photograph"
    return None


def _lints():
    """Lettering in a picture, stillness in an action (see TEXT_IN_SCENE, STILL_IN_ACTION)."""
    return _lint_scene, _lint_action


def check(d: dict, src: str) -> tuple[list[str], list[str]]:
    """(errors, warnings). An error blocks compile; a warning goes in the report."""
    E, W = [], []
    bible = d.get("bible") or {}
    regs, locs, chars = (bible.get("registers") or {}, bible.get("locations") or {},
                         bible.get("characters") or {})
    read = d.get("read") or {}
    motifs = {m["id"] for m in read.get("motifs", []) if isinstance(m, dict) and m.get("id")}
    shots = d.get("shots") or []
    tg = d.get("targets") or {}
    lint_scene, lint_action = _lints()

    if not shots:
        return ["direction has no shots"], []
    for k in ("subject", "motifs", "rules", "figurative"):
        if not read.get(k):
            E.append(f"read.{k}: Pass A is incomplete without it")
    mv_names = [m.get("name") for m in d.get("movements") or []]
    if not mv_names:
        E.append("movements: Pass B needs its movements declared before its shots")

    ids = [s.get("id") for s in shots]
    for x in {i for i in ids if ids.count(i) > 1}:
        E.append(f"{x}: id used twice")

    text = narration(d, src)
    _, aerrs = anchors(d, src)
    E += aerrs
    fig = [norm(f) for f in read.get("figurative", [])]

    seen_moment: dict[str, int] = {}
    stage_hi: dict[str, int] = {}
    arc_at: dict[tuple[str, str], int] = {}
    for i, s in enumerate(shots):
        sid = s.get("id", f"#{i}")
        for k in REQUIRED:
            if k not in s:
                E.append(f"{sid}: missing '{k}'")
        reg, loc = s.get("register"), s.get("location")
        L = locs.get(loc)
        if reg not in regs:
            E.append(f"{sid}: register {reg!r} is not in the bible ({list(regs)})")
        if L is None:
            E.append(f"{sid}: location {loc!r} is not in the bible")
            L = {}
        elif L.get("register") != reg:
            E.append(f"{sid}: {loc} belongs to the {L.get('register')} register, the shot says {reg}")
        if mv_names and s.get("movement") not in mv_names:
            E.append(f"{sid}: movement {s.get('movement')!r} is not declared")
        if s.get("relation") not in RELATIONS:
            E.append(f"{sid}: relation {s.get('relation')!r} is not one of {RELATIONS}")
        if s.get("size") not in SIZES:
            E.append(f"{sid}: size {s.get('size')!r} is not one of {SIZES}")
        if s.get("screen") not in SCREEN:
            E.append(f"{sid}: screen {s.get('screen')!r} is not one of {SCREEN}")
        if s.get("motif") and s["motif"] not in motifs:
            E.append(f"{sid}: motif {s['motif']!r} is not in read.motifs")

        # the spine degrades by degrees and never recovers
        stages = L.get("stages") or []
        if stages:
            st = s.get("stage")
            if st not in stages:
                E.append(f"{sid}: {loc} has stages {stages}; give this shot one of them")
            elif L.get("monotonic"):
                k = stages.index(st)
                if k < stage_hi.get(loc, 0):
                    E.append(f"{sid}: {loc} is back at {st!r} after {stages[stage_hi[loc]]!r}; "
                             f"this location only degrades")
                stage_hi[loc] = max(stage_hi.get(loc, 0), k)

        # who, in what state, where they stand
        allows, blocking = L.get("allows"), L.get("blocking") or {}
        for c in s.get("cast") or []:
            who, state = c.get("who"), c.get("state")
            C = chars.get(who)
            if C is None:
                E.append(f"{sid}: {who!r} is not in the bible")
                continue
            if state not in (C.get("states") or {}):
                E.append(f"{sid}: {who} has no state {state!r} (bible: {list(C.get('states') or {})})")
            if allows is not None:
                if who not in allows:
                    E.append(f"{sid}: {who} never appears in {loc} (bible allows {list(allows)})")
                elif state not in allows[who]:
                    E.append(f"{sid}: {who} is {state!r} in {loc}; the bible allows only "
                             f"{allows[who]} there")
            if c.get("blocking") and who in blocking and norm(c["blocking"]) != norm(blocking[who]):
                E.append(f"{sid}: {who} is {c['blocking']!r} but in {loc} always "
                         f"{blocking[who]!r}")
            arc = C.get("arc") or []
            if state in arc:
                key, k = (who, reg), arc.index(state)
                if k < arc_at.get(key, 0):
                    E.append(f"{sid}: {who} goes back to {state!r} in the {reg} register after "
                             f"{arc[arc_at[key]]!r}")
                arc_at[key] = max(arc_at.get(key, 0), k)

        # one moment, shown once
        m = s.get("moment")
        if m in seen_moment and seen_moment[m] != i - 1 and not s.get("rhymes_with"):
            E.append(f"{sid}: moment {m!r} was already shown at {shots[seen_moment[m]].get('id')}; "
                     f"show a moment once, or declare rhymes_with if the return is the point")
        if s.get("rhymes_with") and s["rhymes_with"] not in ids[:i]:
            E.append(f"{sid}: rhymes_with {s['rhymes_with']!r} is not an earlier shot")
        if m:
            seen_moment[m] = i

        # register changes are decisions, and so is any change of place
        prev = shots[i - 1] if i else None
        if prev and prev.get("location") != loc and s.get("transition") not in TRANSITIONS:
            E.append(f"{sid}: {prev.get('location')} -> {loc} needs a declared transition "
                     f"({TRANSITIONS}); v1's abrupt cuts (Allam to the cook, s047->s048, "
                     f"s071->s072) were changes of place nobody chose")
        if (prev and s.get("transition") in ("match_cut", "dissolve", "swirl")
                and len((s.get("bridge") or "").split()) < 4):
            E.append(f"{sid}: a {s['transition']} needs a bridge: what carries across it")
        if prev and prev.get("register") != reg:
            if s.get("transition") not in TRANSITIONS:
                E.append(f"{sid}: {prev.get('register')} -> {reg} needs a transition "
                         f"({TRANSITIONS})")
            if len((s.get("bridge") or "").split()) < 4:
                E.append(f"{sid}: {prev.get('register')} -> {reg} needs a bridge: what carries the "
                         f"eye or ear across")
        if (prev and i + 1 < len(shots) and prev.get("register") != reg
                and shots[i + 1].get("register") == prev.get("register") and text
                and seconds(d, s, text[i]) < 6):
            W.append(f"{sid}: a {seconds(d, s, text[i])} s visit to {reg} between two "
                     f"{prev.get('register')} shots reads as a flicker")

        # shots with nobody in them
        if not s.get("cast"):
            if len((s.get("why") or "").split()) < 6:
                E.append(f"{sid}: no character in frame, so say why (relation and what it does "
                         f"against the words) in 'why'")
            near = {x.get("location") for x in (prev, shots[i + 1] if i + 1 < len(shots) else None)
                    if x}
            if loc not in near and not s.get("motif") and reg != "myth":
                E.append(f"{sid}: an empty frame in a place neither neighbour visits and carrying no "
                         f"motif; it is divorced from the film")

        # words that name what this film never shows (bible.forbid, and a location's own forbid)
        body = norm(s.get("picture", "") + " " + s.get("action", ""))
        for w_ in list(bible.get("forbid") or []) + list(L.get("forbid") or []):
            if re.search(r"\b" + re.escape(norm(w_)) + r"\b", body):
                E.append(f"{sid}: the picture names {w_!r}, which the bible forbids"
                         + (f" in {loc}" if w_ in (L.get("forbid") or []) else ""))
        if s.get("cast") and reg != "myth" and not s.get("extras_ok"):
            mx = EXTRAS.search(s.get("picture", ""))
            if mx:
                E.append(f"{sid}: unnamed people ({mx.group(0)!r}) beside the cast; name them from "
                         f"the bible or set extras_ok with the reason")
        if text and seconds(d, s, text[i]) < MIN_SHOT_S:
            E.append(f"{sid}: {seconds(d, s, text[i])} s is a flash; give it hold_s or merge it")

        # figurative lines are not illustrated
        if text and s.get("relation") == "depict" and reg != "myth":
            hit = [f for f in fig if f and f in norm(text[i])]
            if hit:
                E.append(f"{sid}: depicts narration the read marked figurative ({hit[0]!r}); "
                         f"choose another relation or the myth register")

        # screen direction against the location's axis
        travel = (L.get("screen") or {}).get("travel")
        if travel in HORIZONTAL and s.get("screen") in HORIZONTAL and s["screen"] != travel \
                and not s.get("crosses_line"):
            E.append(f"{sid}: travels {s['screen']} but {loc} runs {travel}; set crosses_line "
                     f"with the reason if crossing is the point")

        for k, fn in (("picture", lint_scene), ("action", lint_action)):
            why = fn(s.get(k, ""))
            if why:
                E.append(f"{sid}: {k} {why}")
        if re.search(r"\blocked[- ]off\b|\bdoes not move\b|\bstatic\b", s.get("motion", ""), re.I):
            W.append(f"{sid}: the camera move is a hold; H3 clips from held cameras read as stills")

    # adjacent continuity, the part a contact sheet hides
    for row in continuity(d, src):
        E += row["errors"]
        W += row["warnings"]

    # rhythm
    n = len(shots)
    lo, hi = (tg.get("shots") or [1, 10 ** 6])[:2]
    if not lo <= n <= hi:
        E.append(f"targets: {n} shots; the target is {lo}-{hi}")
    if text:
        durs = [seconds(d, s, t) for s, t in zip(shots, text)]
        asl = sum(durs) / n
        alo, ahi = (tg.get("asl_s") or [0, 10 ** 6])[:2]
        if not alo <= asl <= ahi:
            E.append(f"targets: average shot length {asl:.1f} s; the target is {alo}-{ahi} s "
                     f"(source is {len(src.split())} words at {tg.get('wps', 2.4)} words/s)")
        clo, chi = (tg.get("clip_s") or [0, 10 ** 6])[:2]
        for s, t in zip(shots, durs):
            if not clo <= t <= chi:
                W.append(f"{s.get('id')}: {t} s is outside one H3 clip ({clo}-{chi} s); "
                         + ("set hold_s or merge with a neighbour" if t < clo
                            else "this shot needs two clips or a shorter span"))
    for mv in d.get("movements") or []:
        got = sum(1 for s in shots if s.get("movement") == mv.get("name"))
        if mv.get("shots") is not None and got != mv["shots"]:
            W.append(f"movement {mv.get('name')!r}: declared {mv['shots']} shots, board has {got}")
    for f in read.get("figurative", []):
        if norm(f) not in norm(src):
            W.append(f"read.figurative {f!r} is no longer in the source; drop it or re-quote it")
    used = {s.get("motif") for s in shots}
    for m in sorted(motifs - used):
        W.append(f"motif {m!r} is declared in the read and never used")
    return E, W


def continuity(d: dict, src: str) -> list[dict]:
    """One row per cut: what changes across it, and what is wrong with that."""
    shots = d.get("shots") or []
    locs = (d.get("bible") or {}).get("locations") or {}
    text = narration(d, src)
    rows = []
    for a, b in zip(shots, shots[1:]):
        E, W, notes = [], [], []
        same_place = a.get("location") == b.get("location") and a.get("register") == b.get("register")
        if a.get("register") != b.get("register"):
            notes.append(f"{a.get('register')} -> {b.get('register')} by {b.get('transition', '?')}")
        if same_place:
            if a.get("stage") != b.get("stage"):
                notes.append(f"stage {a.get('stage')} -> {b.get('stage')}")
            if {a.get("screen"), b.get("screen")} == HORIZONTAL and not b.get("crosses_line"):
                E.append(f"{a.get('id')}->{b.get('id')}: screen direction flips "
                         f"{a['screen']} -> {b['screen']} inside {b.get('location')}")
            if a.get("size") == b.get("size") and a.get("moment") == b.get("moment"):
                W.append(f"{a.get('id')}->{b.get('id')}: same size on the same moment is a jump "
                         f"cut; change size or angle")
            sa = {c.get("who"): c.get("state") for c in a.get("cast") or []}
            for c in b.get("cast") or []:
                if c.get("who") in sa and sa[c["who"]] != c.get("state") \
                        and a.get("moment") == b.get("moment"):
                    E.append(f"{a.get('id')}->{b.get('id')}: {c['who']} is {sa[c['who']]!r} then "
                             f"{c.get('state')!r} inside one moment")
        else:
            notes.append(f"{a.get('location')} -> {b.get('location')}")
        if a.get("motion") and a.get("motion") == b.get("motion"):
            W.append(f"{a.get('id')}->{b.get('id')}: the same camera move twice running")
        dur = (seconds(d, a, text[shots.index(a)]), seconds(d, b, text[shots.index(b)])) if text else ("?", "?")
        rows.append({"cut": f"{a.get('id')}->{b.get('id')}", "notes": notes, "errors": E,
                     "warnings": W, "seconds": dur,
                     "screen": f"{a.get('screen')} -> {b.get('screen')}",
                     "size": f"{a.get('size')} -> {b.get('size')}",
                     "axis": (locs.get(b.get("location")) or {}).get("screen", {}).get("travel", "")})
    return rows


def continuity_md(d: dict, src: str) -> str:
    rows = continuity(d, src)
    out = [f"# Continuity: {d.get('title', '')}", "",
           f"Direction {digest(d)}, {len(d.get('shots') or [])} shots, {len(rows)} cuts. "
           f"Generated by `python -m clay_film direct continuity`; do not edit by hand.", "",
           "| cut | seconds | across the cut | screen | size | flags |", "|---|---|---|---|---|---|"]
    for r in rows:
        flags = "; ".join(["ERROR " + e.split(": ", 1)[-1] for e in r["errors"]]
                          + [w.split(": ", 1)[-1] for w in r["warnings"]]) or "ok"
        out.append(f"| {r['cut']} | {r['seconds'][0]} / {r['seconds'][1]} | "
                   f"{'; '.join(r['notes'])} | {r['screen']} | {r['size']} | {flags} |")
    return "\n".join(out) + "\n"


def board_md(d: dict, src: str) -> str:
    text = narration(d, src) or [""] * len(d.get("shots") or [])
    out = ["| # | reg | location | who (state) | under | rel | frame | picture | camera; H3 action | s |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for s, t in zip(d.get("shots") or [], text):
        loc = s.get("location", "") + (f" @{s['stage']}" if s.get("stage") else "")
        who = ", ".join(f"{c['who']} ({c['state']})" for c in s.get("cast") or []) or "-"
        under = f"\"{s.get('under', '')}...\""
        rel = s.get("relation", "") + (f", motif {s['motif']}" if s.get("motif") else "") \
            + (f", rhymes {s['rhymes_with']}" if s.get("rhymes_with") else "")
        frame = f"{s.get('size', '')}, {s.get('screen', '')}"
        if s.get("transition") and s.get("bridge"):
            frame += f"; in by {s['transition']}"
        mv = f"{s.get('motion', '')}; {s.get('action', '')}"
        out.append(f"| {s.get('id')} | {s.get('register')} | {loc} | {who} | {under} | {rel} | "
                   f"{frame} | {s.get('picture', '')} | {mv} | {seconds(d, s, t) if t else ''} |")
    return "\n".join(out) + "\n"


# -- references and compile -------------------------------------------------------------------

def ref_keys(d: dict) -> list[tuple[str, str]]:
    """(key, kind) for every reference image the bible needs, in a fixed order: characters, then
    locations, with one plate per stage for a staged location."""
    b = d.get("bible") or {}
    keys = [(f"char:{k}", "char") for k in (b.get("characters") or {})]
    for k, L in (b.get("locations") or {}).items():
        if L.get("stages"):
            keys += [(f"loc:{k}@{st}", "loc") for st in L["stages"]]
        else:
            keys.append((f"loc:{k}", "loc"))
    return keys


def refs_piece(d: dict) -> str:
    return f"{d.get('piece', 'directed')}_refs"


def ref_path(d: dict, key: str) -> Path:
    """An explicit `ref` in the bible wins; otherwise where `refs` + `clay_film run` put it."""
    b = d.get("bible") or {}
    kind, _, rest = key.partition(":")
    name, _, stage = rest.partition("@")
    entry = (b.get("characters") if kind == "char" else b.get("locations") or {}).get(name) or {}
    explicit = (entry.get("refs") or {}).get(stage) if stage else entry.get("ref")
    if explicit:
        return Path(explicit)
    k = [x for x, _ in ref_keys(d)].index(key)
    return paths.PIECES / refs_piece(d) / "stills" / f"s{k:03d}.png"


def refs_plan(d: dict) -> dict:
    """A free clay_film plan whose stills ARE the references: one character sheet per person and
    one plate per location (per stage). Rendered once, approved once, reused by every shot."""
    b = d.get("bible") or {}
    shots = []
    for key, kind in ref_keys(d):
        name, _, stage = key.split(":", 1)[1].partition("@")
        if kind == "char":
            C = b["characters"][name]
            # the backdrop was Oceanus's; a direction may name its own (gardening: plain dark grey)
            backdrop = d.get("ref_backdrop", "a plain dark sea-green backdrop")
            prompt = (f"{C['look']}, standing full length in the centre of the frame, facing the "
                      f"viewer three-quarter on, lit evenly against {backdrop}")
            people = []
        else:
            L = b["locations"][name]
            # The stage leads: on the Oceanus plates the shared look ("planks adrift") outvoted
            # "below the surface", and every underwater plate came back as the surface. A plate is
            # a set with nobody on it, so say so; the tavern and galley plates grew extras.
            st = L["stage_looks"][stage] if stage and L.get("stage_looks") else ""
            # Gardening (photoreal): "no people" still drew a man into the foreground, since the
            # negation names a person and the style asks for skin; describe the emptiness instead.
            prompt = ((f"{st}; {L['look']}" if st else L["look"])
                      + "; the place is deserted, every seat and table empty, an interior photograph"
                        " of an empty set")
            people = []
        shots.append({"mode": "free", "id": key.replace(":", "_").replace("@", "_"),
                      "narration": f"reference {key}", "still_prompt": prompt, "people": people,
                      "subject": key, "scene": prompt, "action": "the light shifts slowly",
                      "motion": "a slow push in", "movement": kind})
    return {"title": f"{d.get('title', '')} references", "style": d.get("style", ""),
            "seed": d.get("seed", 31337), "grade": d.get("grade", "vivid"), "shots": shots}


def to_plan(d: dict, src: str, source_path: Path) -> tuple[dict, list[str]]:
    """The clay_film plan: free shots, narration cut at the anchors, references where they exist."""
    b = d["bible"]
    regs, locs, chars = b["registers"], b["locations"], b["characters"]
    text = narration(d, src)
    notes, shots = [], []
    for s, t in zip(d["shots"], text):
        L = locs[s["location"]]
        place = L["look"] + (f"; {L['stage_looks'][s['stage']]}"
                             if s.get("stage") and L.get("stage_looks") else "")
        reg_clause = (regs[s["register"]] or {}).get("still", "") if isinstance(regs[s["register"]], dict) else ""
        people, ref_keys_used = [], [f"loc:{s['location']}" + (f"@{s['stage']}" if s.get("stage") else "")]
        # An insert is of a hand, a wrist, a bag: the full character line ("a heavy-set unshaven man...")
        # and the identity pass both pull a whole person into it (gardening g07, the tattooed wrist,
        # came back as a medium shot of the dealer). The picture already says what is visible.
        insert = s.get("size") == "extreme_close"
        # the plan schema takes at most 3 people; a room wide with six names the rest in its picture
        for c in ([] if insert else (s.get("cast") or [])[:3]):
            C = chars[c["who"]]
            bits = [C["states"][c["state"]]]
            blk = (L.get("blocking") or {}).get(c["who"]) or c.get("blocking")
            if blk:
                bits.append(blk)
            people.append({"who": c["who"], "action": ", ".join(bits)})
            ref_keys_used.append(f"char:{c['who']}")
        # The picture leads: it opens with the shot size and angle. With the place first, t2i framed a
        # room and its people whatever size the shot asked for (gardening, 2026-10-10).
        pic = " ".join(s["picture"].split()).rstrip(". ")
        still = f"{pic[0].upper() + pic[1:]}. The setting: {place}." + (f" {reg_clause}" if reg_clause else "")
        shot = {"mode": "free", "id": s["id"], "movement": s["movement"], "narration": t,
                "still_prompt": still, "people": people,
                "camera": {"distance": s["size"]}, "subject": s.get("subject") or s["picture"].split(",")[0][:60],
                "scene": s["picture"], "action": s["action"], "motion": s["motion"],
                "relation": s["relation"], "register": s["register"]}
        if s.get("move_target"):
            shot["move_target"] = s["move_target"]
        if s.get("hold_s"):
            shot["tail_s"] = s["hold_s"]
        if s.get("end"):   # the designed last frame, made by stages.endframes (director_kb 12)
            shot["end"] = s["end"]
        refs = [ref_path(d, k) for k in ref_keys_used]
        if all(r.is_file() for r in refs):
            names = []
            for k, (key, r) in enumerate(zip(ref_keys_used, refs), start=1):
                if key.startswith("char:"):
                    names.append(f"the {chars[key[5:]].get('name', key[5:])} is the person in <image{k}>")
            shot["refs"] = [str(r) for r in refs]
            shot["ref_prompt"] = (f"{d.get('style', '').rstrip('. ')}. The place is the one in <image1>. "
                                  + (" ".join(n[0].upper() + n[1:] + "." for n in names) + " " if names else "")
                                  + s["picture"] + (f" {reg_clause}" if reg_clause else ""))
        else:
            missing = [str(r) for r in refs if not r.is_file()]
            notes.append(f"{s['id']}: text-to-image, references not rendered yet ({len(missing)} missing)")
        shots.append(shot)
    cast = {k: C["look"] for k, C in chars.items()}
    plan = {"title": d["title"], "source": str(source_path), "style": d.get("style", ""),
            "motion_style": d.get("motion_style", ""), "seed": d.get("seed", 31337),
            "grade": d.get("grade", "vivid"), "cast": cast, "directed": True,
            "direction_hash": digest(d), "shots": shots}
    if d.get("voice_ref"):
        plan["voice_ref"] = d["voice_ref"]
    return plan, notes


# -- the animatic gate ------------------------------------------------------------------------

def _font(size: int):
    from PIL import ImageFont
    for f in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


REG_COLOUR = {"present": (16, 40, 52), "memory": (70, 46, 20), "myth": (48, 24, 64)}


def card(d: dict, s: dict, text: str, secs: float, out: Path) -> None:
    """A storyboard card: register colour, place, who and in what state, the picture, the words."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (1344, 768), REG_COLOUR.get(s.get("register"), (30, 30, 30)))
    g = ImageDraw.Draw(im)
    big, mid, small = _font(40), _font(28), _font(24)
    who = ", ".join(f"{c['who']} ({c['state']})" for c in s.get("cast") or []) or "nobody in frame"
    g.text((48, 36), f"{s['id']}  {s.get('register', '').upper()}  {secs:.1f} s", fill=(240, 230, 210), font=big)
    g.text((48, 96), f"{s.get('location')}{' @' + s['stage'] if s.get('stage') else ''}  |  {who}",
           fill=(220, 210, 190), font=mid)
    g.text((48, 140), f"{s.get('size')}, screen {s.get('screen')}, {s.get('relation')}"
           + (f"  |  in by {s['transition']}" if s.get("transition") else ""), fill=(190, 180, 160), font=small)
    y = 200
    for line in textwrap.wrap(s.get("picture", ""), 80)[:6]:
        g.text((48, y), line, fill=(250, 245, 235), font=mid)
        y += 38
    y += 10
    for line in textwrap.wrap(s.get("action", "") + "  /  " + s.get("motion", ""), 90)[:3]:
        g.text((48, y), line, fill=(200, 220, 230), font=small)
        y += 32
    y = max(y + 20, 560)
    for line in textwrap.wrap(f"“{text}”", 100)[:5]:
        g.text((48, y), line, fill=(170, 165, 150), font=small)
        y += 32
    im.save(out)


def animatic(d: dict, src: str, piece: Path) -> Path:
    """Cards (or the shot's still, when one exists) for each shot's length, over the narration
    when the piece has one. CPU only."""
    from . import dry
    from PIL import Image
    ff = dry._ffmpeg()
    if not ff:
        raise RuntimeError("no ffmpeg (set FFMPEG, install imageio-ffmpeg, or put ffmpeg on PATH)")
    piece.mkdir(parents=True, exist_ok=True)
    work = piece / "_animatic"
    work.mkdir(exist_ok=True)
    text = narration(d, src)
    tl = {}
    if (piece / "timeline.json").is_file():
        tl = {x["index"]: x for x in json.loads((piece / "timeline.json").read_text(encoding="utf-8"))
              .get("scenes", [])}
    lines, total, with_stills = [], 0.0, 0
    for i, (s, t) in enumerate(zip(d["shots"], text)):
        secs = float(tl[i]["duration"]) if i in tl else seconds(d, s, t)
        f = work / f"{i:03d}.png"
        still = piece / "stills" / f"s{i:03d}.png"
        if still.is_file():
            Image.open(still).convert("RGB").resize((1344, 768)).save(f)
            with_stills += 1
        else:
            card(d, s, t, secs, f)
        lines += [f"file '{f.as_posix()}'", f"duration {secs:.3f}"]
        total += secs
    lines.append(f"file '{(work / f'{len(text) - 1:03d}.png').as_posix()}'")
    (work / "list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = piece / "animatic.mp4"
    audio = piece / "narration.mp3" if tl and (piece / "narration.mp3").is_file() else None
    cmd = [ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(work / "list.txt")]
    if audio:
        cmd += ["-i", str(audio), "-c:a", "aac", "-shortest"]
    cmd += ["-vf", "fps=12,scale=1344:768,format=yuv420p", "-c:v", "libx264", "-crf", "26", str(out)]
    subprocess.run(cmd, check=True)
    (piece / "animatic.json").write_text(json.dumps(
        {"direction_hash": digest(d), "built": datetime.now().isoformat(timespec="seconds"),
         "seconds": round(total, 1), "shots": len(text), "with_stills": with_stills,
         "narration": bool(audio)}, indent=1), encoding="utf-8")
    return out


def signoff(d: dict, piece: Path, watched: str) -> None:
    a = piece / "animatic.json"
    if not a.is_file():
        raise RuntimeError(f"no animatic in {piece}; build it with `direct animatic` and watch it")
    meta = json.loads(a.read_text(encoding="utf-8"))
    if meta.get("direction_hash") != digest(d):
        raise RuntimeError("the animatic was built from an older direction; rebuild and rewatch it")
    if len(watched.split()) < 5:
        raise RuntimeError("--watched needs a sentence on what the sequence did, not a tick")
    (piece / "animatic_signoff.json").write_text(json.dumps(
        {"direction_hash": meta["direction_hash"], "at": datetime.now().isoformat(timespec="seconds"),
         "watched": watched}, indent=1), encoding="utf-8")


def animatic_gate(piece: Path, h: str) -> str | None:
    """Why stills may not render yet, or None."""
    f = Path(piece) / "animatic_signoff.json"
    if not f.is_file():
        return (f"{Path(piece).name} is a directed piece and its animatic has not been signed off. "
                f"Run `python -m clay_film direct animatic NAME`, watch it, then `direct signoff`.")
    if json.loads(f.read_text(encoding="utf-8")).get("direction_hash") != h:
        return "the direction changed since the animatic was signed off; rebuild and rewatch it"
    return None


# -- command line ------------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="clay_film direct")
    ap.add_argument("op", choices=["check", "continuity", "board", "refs", "compile", "animatic",
                                   "signoff", "selftest"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--watched", default="")
    ap.add_argument("--piece", help="piece folder name (default: direction's 'piece')")
    a = ap.parse_args(argv)
    if a.op == "selftest":
        return selftest()
    d, f = load(a.name)
    src = source_text(d, f)
    piece = paths.PIECES / (a.piece or d.get("piece") or f.parent.name)
    if a.op == "check":
        E, W = check(d, src)
        for w in W:
            print("warn  " + w)
        for e in E:
            print("ERROR " + e)
        text = narration(d, src)
        if text:
            durs = [seconds(d, s, t) for s, t in zip(d["shots"], text)]
            print(f"{len(durs)} shots, {sum(durs):.0f} s, ASL {sum(durs) / len(durs):.1f} s, "
                  f"source {len(src.split())} words, direction {digest(d)}")
        print(f"{len(E)} error(s), {len(W)} warning(s)")
        return 1 if E else 0
    if a.op == "continuity":
        out = f.parent / "CONTINUITY.md"
        out.write_text(continuity_md(d, src), encoding="utf-8")
        print(f"wrote {out}")
        return 0
    if a.op == "board":
        print(board_md(d, src))
        return 0
    if a.op == "refs":
        out = f.parent / "refs_plan.json"
        out.write_text(json.dumps(refs_plan(d), indent=1, ensure_ascii=False), encoding="utf-8")
        for k, (key, _) in enumerate(ref_keys(d)):
            print(f"s{k:03d}  {key:34s} -> {ref_path(d, key)}")
        print(f"wrote {out}\nrender: python -m clay_film new {refs_piece(d)} {out}\n"
              f"        python -m clay_film run {refs_piece(d)} --stages compile,stills")
        return 0
    E, W = check(d, src)
    if E and a.op in ("compile", "animatic", "signoff"):
        print("the direction does not check; fix these first:\n- " + "\n- ".join(E[:30]))
        return 1
    if a.op == "compile":
        plan, notes = to_plan(d, src, write_source(d, f, src))
        joined = " ".join(s["narration"] for s in plan["shots"])
        if joined != src:
            print("refusing to compile: the shots' narration, joined, is not the source text "
                  f"({d['source']}); re-anchor the shots")
            return 1
        out = f.parent / "plan.directed.json"
        out.write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
        for n in notes:
            print("note  " + n)
        print(f"wrote {out} ({len(plan['shots'])} shots, direction {plan['direction_hash']})\n"
              f"install: python -m clay_film new {piece.name} {out}")
        return 0
    if a.op == "animatic":
        print(f"wrote {animatic(d, src, piece)}")
        return 0
    if a.op == "signoff":
        try:
            signoff(d, piece, a.watched)
        except RuntimeError as e:
            print(f"not signed off: {e}")
            return 1
        print(f"signed off {digest(d)} for {piece.name}; stills may render")
        return 0
    return 2


def selftest() -> int:
    """The checks catch each failure Patrick named on the hand-written Oceanus board."""
    src = ("Jacob lies below. Miss Betty smiled. Boats capsize. Jacob lingered at the bar. "
           "No stones cast here.")
    bible = {"registers": {"present": {}, "memory": {}, "myth": {}}, "forbid": ["barrel"],
             "characters": {"cook": {"look": "an aged cook", "states": {"in_water": "x", "remembered": "y"},
                                     "arc": ["in_water"]},
                            "jacob": {"look": "a young sailor", "states": {"at_bar": "x"}},
                            "betty": {"look": "a barmaid", "states": {"behind_bar": "x"}}},
             "locations": {"water": {"register": "present", "look": "sea", "stages": ["night", "dawn"],
                                     "monotonic": True, "allows": {"cook": ["in_water"]},
                                     "screen": {"travel": "R>L"}},
                           "tavern": {"register": "memory", "look": "bar",
                                      "allows": {"jacob": ["at_bar"], "betty": ["behind_bar"],
                                                 "cook": ["remembered"]},
                                      "blocking": {"betty": "behind the bar"}},
                           "port": {"register": "memory", "look": "a port"}}}
    base = {"movement": "m", "relation": "detail", "size": "medium", "screen": "none",
            "picture": "a picture", "action": "the water rises and falls slowly", "motion": "a slow push in"}

    def mk(**kw):
        return {**base, **kw}
    d = {"title": "t", "read": {"subject": "s", "motifs": [{"id": "m1"}], "rules": ["r"],
                                "figurative": ["no stones cast"]},
         "movements": [{"name": "m"}], "bible": bible, "targets": {"shots": [1, 99]},
         "shots": [
             mk(id="a", register="present", location="water", stage="dawn", moment="m0",
                cast=[{"who": "cook", "state": "in_water"}], under="Jacob lies below", screen="L>R"),
             mk(id="b", register="memory", location="tavern", moment="bar", picture="sailors drink",
                cast=[{"who": "cook", "state": "in_water"}, {"who": "betty", "state": "behind_bar",
                                                             "blocking": "in front of the bar"}],
                under="Miss Betty"),
             mk(id="c", register="memory", location="port", moment="cap", cast=[], under="Boats capsize"),
             mk(id="d", register="present", location="water", stage="night", moment="bar",
                cast=[{"who": "cook", "state": "in_water"}], under="Jacob lingered", transition="cut",
                bridge="the bar dissolves into water"),
             mk(id="e", register="present", location="water", stage="night", moment="x",
                cast=[{"who": "cook", "state": "in_water"}], under="No stones", relation="depict",
                picture="the cook hugs a barrel")]}
    E, _ = check(d, src)
    want = {"screen travel": "runs R>L", "state not allowed here": "allows only",
            "blocking": "always", "register change without bridge": "needs a bridge",
            "orphan empty frame": "divorced", "spine goes backward": "only degrades",
            "repeated moment": "already shown", "figurative depicted": "figurative",
            "forbidden prop": "forbids", "unnamed extras": "unnamed people",
            "flash": "is a flash", "undeclared change of place": "declared transition"}
    bad = [k for k, frag in want.items() if not any(frag in e for e in E)]
    sys.path.insert(0, str(paths.LIB))
    try:
        import director
        if list(director.RELATIONS) != RELATIONS:
            bad.append("RELATIONS drifted from director.py")
    except Exception:                                           # noqa: BLE001
        pass
    print("\n".join(E))
    print("direct selftest: " + ("ok" if not bad else "MISSED " + ", ".join(bad)))
    return 1 if bad else 0
