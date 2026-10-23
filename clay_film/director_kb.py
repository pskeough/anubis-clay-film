"""The director's knowledge base: load, retrieve, brief, and advisory lint.

    python -m clay_film.director_kb index                      files and sections in director_kb/
    python -m clay_film.director_kb ask "dissolve between memory and present"
    python -m clay_film.director_kb brief PASSAGE.txt [--style "Dore engraving"] [--words-per-s 2.4]
    python -m clay_film.director_kb lint NAME                  advisory warnings on a direction.json

The KB is markdown in `director_kb/` (one topic per file, YAML-ish front matter with `triggers`,
sections split on `## ` with a `Use when:` line). Nothing here renders or touches the GPU, and
nothing here blocks: `direct check` stays the gate. `lint` is the KB's opinion, reported.

The KB itself is not shipped with this repo. Point CLAY_FILM_DIRECTOR_KB at your own folder
(default: director_kb/ at the repo root). `index`, `ask` and `brief` need it; `lint` does not.

`brief` is the input to the treatment stage (director_kb/00_treatment.md): passage facts, breath
points for `under` anchors, figurative candidates, targets, and the KB sections the passage calls
for. The director (the main Claude session) writes treatment.md and direction.json from it.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .paths import DIRECTOR_KB as KB_DIR  # noqa: E402
WPS = 2.4

_WORD = re.compile(r"[a-z][a-z'\-]+")
_STOP = set("""a an the and or but if of to in on at by for with from into onto over under as is are was
were be been being it its this that these those he she they we i you his her their our my your him
them me us not no so do does did have has had which who whom what when where why how than then there
here also very can will would should could may might must shall just only any all each every some""".split())


def _words(text: str) -> list[str]:
    return [w[:-2] if w.endswith("'s") else w for w in _WORD.findall((text or "").lower()) if w not in _STOP]


@dataclass
class Section:
    file: str
    file_id: str
    heading: str
    use_when: str
    body: str
    triggers: list[str] = field(default_factory=list)

    def ref(self) -> str:
        return f"{self.file}#{self.heading}"


def _front(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    meta = {}
    for line in text[3:end].strip().splitlines():
        k, _, v = line.partition(":")
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            meta[k.strip()] = [x.strip().strip("'\"").lower() for x in v[1:-1].split(",") if x.strip()]
        else:
            meta[k.strip()] = v
    return meta, text[end + 4:]


def load(kb: Path = KB_DIR) -> list[Section]:
    if not kb.is_dir() or not any(kb.glob("*.md")):
        raise FileNotFoundError(f"no director knowledge base at {kb}; set CLAY_FILM_DIRECTOR_KB to a "
                                "folder of markdown topic files")
    out = []
    for f in sorted(kb.glob("*.md")):
        if f.name in ("INDEX.md",):
            continue
        meta, body = _front(f.read_text(encoding="utf-8"))
        trig = meta.get("triggers") or []
        parts = re.split(r"^## ", body, flags=re.M)
        head = parts[0].strip()
        if head:
            out.append(Section(f.name, meta.get("id", f.stem), "(intro)", meta.get("use_when", ""), head, trig))
        for p in parts[1:]:
            h, _, b = p.partition("\n")
            m = re.search(r"^Use when:\s*(.+)$", b, re.M)
            out.append(Section(f.name, meta.get("id", f.stem), h.strip(), m.group(1).strip() if m else "",
                               b.strip(), trig))
    return out


def ask(query: str | list[str], k: int = 6, sections: list[Section] | None = None) -> list[tuple[float, Section]]:
    """Sections ranked for a question or a list of tags. Trigger phrases count most, then the
    heading and its Use-when line, then the body."""
    secs = sections if sections is not None else load()
    q = " ".join(query) if isinstance(query, list) else query
    ql = q.lower()
    qw = set(_words(q))
    scored = []
    for s in secs:
        trig = sum(2.0 for t in s.triggers if t and t in ql)
        head = len(qw & set(_words(s.heading + " " + s.use_when))) * 1.5
        bw = _words(s.body)
        body = len(qw & set(bw)) * 0.4
        score = trig + head + body
        if score > 0:
            scored.append((round(score, 2), s))
    scored.sort(key=lambda x: -x[0])
    return scored[:k]


# -- passage facts ------------------------------------------------------------------------------

_SIMILE = re.compile(r"\b(like a|like an|like the|as if|as though|akin to|as \w+ as)\b[^.;:,]*", re.I)
_TURN = re.compile(r"\b(however|but|yet|until|still|though|although|instead|now|then|once|before|after)\b", re.I)
_PRESENT = re.compile(r"\b(am|is|are|lies|has|have|do|does|cannot|can|will)\b", re.I)
_PAST = re.compile(r"\b(was|were|had|did|could|would|\w+ed)\b", re.I)


def sentences(text: str) -> list[str]:
    t = " ".join((text or "").split())
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'“])", t) if s]


def breath_points(text: str) -> list[dict]:
    """Candidate `under` anchors: sentence starts and clause starts after , ; : or a dash, each with
    the seconds from the top at WPS. A director picks a subset; these are where a cut can land
    without splitting a phrase."""
    t = " ".join((text or "").split())
    out, words_before = [], 0
    for m in re.finditer(r"(^|[.!?;:,—-]\s+)([A-Za-z\"'‘“][^.!?;:,—]*)", t):
        start = m.start(2)
        words_before = len(t[:start].split())
        kind = ("sentence" if m.group(1).strip()[:1] in ".!?" or start == 0 else "clause")
        phrase = " ".join(m.group(2).split()[:5])
        if out and out[-1]["at_word"] == words_before:
            continue
        out.append({"at_word": words_before, "t": round(words_before / WPS, 1), "kind": kind,
                    "phrase": phrase, "turn": bool(_TURN.match(phrase))})
    return out


def facts(text: str, wps: float = WPS) -> dict:
    t = " ".join((text or "").split())
    sents = sentences(t)
    n = len(t.split())
    secs = n / wps
    pres, past = len(_PRESENT.findall(t)), len(_PAST.findall(t))
    return {
        "words": n, "seconds": round(secs, 1), "sentences": len(sents),
        "longest_sentence_words": max((len(s.split()) for s in sents), default=0),
        "similes": sorted({m.group(0).strip() for m in _SIMILE.finditer(t)}),
        "first_person": bool(re.search(r"\b(I|we|my|our|me|us)\b", t)),
        "second_person": bool(re.search(r"\byou\b", t, re.I)),
        "dialogue": bool(re.search(r"[\"“][^\"”]{3,}[\"”]", t)),
        "tense_mix": "mixed" if pres and past and min(pres, past) / max(pres, past) > 0.2 else
                     ("present" if pres > past else "past"),
        "names": sorted({w for w in re.findall(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b", t)}
                        - {"The", "And", "But", "His", "Her", "They", "When", "Then", "There", "This"}),
        # ASL targets: narrated painted films here run 8-15 s (Oceanus targets); 5.2-15.1 s is one H3 clip
        "shots_range": [max(1, round(secs / 15)), max(1, round(secs / 8))],
    }


TREATMENT_HEADINGS = [
    "1. What the passage is about (one sentence) and what it is not about",
    "2. The narrator's position: who speaks, from when, to whom; registers (present / memory / myth ...)",
    "3. Medium and why (07_style_media.md); the style bible as a finished surface",
    "4. Visual structure: contrast and affinity per movement (08_visual_structure.md), value and colour arc",
    "5. Image-text stance per movement: how often to depict, where to counterpoint, what is never shown (06)",
    "6. Movements: name, text span, intensity, the one strong image, how it opens and closes",
    "7. Camera grammar: move budget, holds, the one big move, where stillness lives (03)",
    "8. Size and angle plan: the progression per movement, where the close-ups go (01, 02)",
    "9. Join grammar: what each transition type means in THIS film, section breaks, match cuts planned (04)",
    "10. Timing: cut-to-word policy, where hold_s goes, breaths at movement ends (05)",
    "11. Risks against measured limits (10_pipeline_mapping.md): small figures, empty frames, figurative lines",
]


def brief(text: str, style: str = "", wps: float = WPS, k_per_topic: int = 2) -> str:
    """The director brief for a passage, as markdown."""
    f = facts(text, wps)
    secs = load()
    topics = ["treatment visual approach", "voice-over image relation illustration counterpoint figurative",
              "transitions dissolve match cut fade section break", "rhythm timing cut on phrase hold breathing",
              "shot size progression close-up", "composition placement frame thirds negative space depth",
              "camera move motivation static hold", "visual structure contrast affinity intensity colour"]
    if style:
        topics.insert(1, f"style medium {style}")
    if f["similes"]:
        topics.append("figurative metaphor simile never depict literally")
    if f["tense_mix"] == "mixed":
        topics.append("register memory present transition bridge")
    lines = ["# Director brief", "", "## Passage facts", "",
             "```json", json.dumps(f, indent=1, ensure_ascii=False), "```", "",
             f"At {wps} words/s the narration runs about {f['seconds']} s. One H3 clip is 5.2-15.1 s, "
             f"so the passage wants about {f['shots_range'][0]}-{f['shots_range'][1]} shots before holds.", ""]
    bp = breath_points(text)
    lines += ["## Breath points (candidate `under` anchors)", "",
              "| at s | kind | turn | phrase |", "|---|---|---|---|"]
    for b in bp:
        lines.append(f"| {b['t']} | {b['kind']} | {'yes' if b['turn'] else ''} | {b['phrase']} |")
    lines += ["", "## Treatment to write (director_kb/00_treatment.md)", ""]
    lines += [f"- {h}" for h in TREATMENT_HEADINGS]
    lines += ["", "## KB sections this passage calls for", ""]
    seen = set()
    for tp in topics:
        for score, s in ask(tp, k=k_per_topic, sections=secs):
            if s.ref() in seen:
                continue
            seen.add(s.ref())
            lines.append(f"- `{s.ref()}` ({score}): {s.use_when or tp}")
    lines += ["", "## Before writing shots", "", "Read `10_pipeline_mapping.md` section 7 (the checklist).", ""]
    return "\n".join(lines)


# -- advisory lint on a direction ----------------------------------------------------------------

MOVE_TYPES = [("push", r"push|toward|into the frame|close in"), ("pull", r"pull|widen|withdraw"),
              ("pan", r"\bpan"), ("tilt", r"tilt"), ("truck", r"track|truck|lateral|glide|slide"),
              ("crane", r"crane|rise|pedestal|lift"), ("arc", r"\barc|orbit|around"),
              ("drift", r"drift"), ("static", r"static|locked|hold|still")]
POSITION = re.compile(r"\b(left|right|centre|center|middle|foreground|background|third|corner|edge|top|"
                      r"bottom|upper|lower|behind|in front of|beyond)\b", re.I)
LIGHT = re.compile(r"\b(light|lamp|lit|glow|shadow|sun|moon|dawn|dusk|candle|fire|backlit|silhouette)\w*", re.I)
SIZES = ["extreme_close", "close", "medium", "full_body", "wide", "very_wide"]


def move_type(motion: str) -> str:
    m = (motion or "").lower()
    for k, pat in MOVE_TYPES:
        if re.search(pat, m):
            return k
    return "other"


def lint(d: dict) -> list[str]:
    """KB opinions on a direction.json. Advisory; each names the KB file behind it."""
    W = []
    shots = d.get("shots") or []
    n = len(shots) or 1
    types = [move_type(s.get("motion", "")) for s in shots]
    for k in ("push",):
        share = types.count(k) / n
        if share > 0.3:
            W.append(f"camera: {types.count(k)} of {n} moves are push-ins ({share:.0%}); vary type and "
                     f"direction (03_camera_movement.md#4)")
    if "static" not in types and n >= 8:
        W.append("camera: no locked-off shots; stillness is a tool and the fix for small lone figures "
                 "(03_camera_movement.md#3)")
    trans = [s.get("transition") for s in shots[1:] if s.get("transition")]
    if trans and len(set(trans)) == 1 and len(trans) > 4:
        W.append(f"joins: every declared transition is {trans[0]!r}; a join type should mean one thing "
                 f"in this film (04_transitions.md)")
    run_s, run_m = 1, 1
    for i, s in enumerate(shots):
        sid = s.get("id", f"#{i}")
        pic = s.get("picture", "")
        if not POSITION.search(pic):
            W.append(f"{sid}: picture names no frame position (11_prompt_still.md#1)")
        if not LIGHT.search(pic):
            W.append(f"{sid}: picture names no light source or direction (11_prompt_still.md#1)")
        if i:
            p = shots[i - 1]
            run_s = run_s + 1 if s.get("size") == p.get("size") else 1
            run_m = run_m + 1 if types[i] == types[i - 1] else 1
            if run_s == 3:
                W.append(f"{sid}: third {s.get('size')} in a row; change size by a step "
                         f"(01_shot_grammar.md#4)")
            if run_m == 3:
                W.append(f"{sid}: third {types[i]} move in a row (03_camera_movement.md#4)")
            if s.get("movement") != p.get("movement") and s.get("transition") in (None, "cut") \
                    and not p.get("hold_s"):
                W.append(f"{sid}: a new movement enters by a plain cut with no hold before it; a section "
                         f"break wants a fade, a dip or a held breath (04_transitions.md, 05_rhythm_timing.md)")
            if s.get("transition") == "match_cut" and not s.get("match"):
                W.append(f"{sid}: match_cut with no `match` spec; the outgoing end frame must be designed "
                         f"(12_prompt_endframe.md#4)")
        txt = " ".join([s.get("action", ""), s.get("motion", ""), s.get("subject", "")]).lower()
        if not s.get("cast") and re.search(r"\b(close|closes|closing|shut|shuts)\b", txt):
            W.append(f"{sid}: 'close'/'shut' on an empty frame read as an iris-out (10_pipeline_mapping.md#4)")
        if re.search(r"\bfades?\b|\bvanish|\bdisappear|\bgoes out\b|\bgo out\b", s.get("action", ""), re.I):
            W.append(f"{sid}: H3 renders vanishing as appearing; write the arrival and reverse "
                     f"(13_prompt_motion.md#2)")
        if re.search(r"\b(turns|looks|glances)\b[^.]{0,40}\btoward (the )?(porthole|window|door|horizon)",
                     s.get("action", ""), re.I):
            W.append(f"{sid}: a look toward an opening cues a cut to the view (13_prompt_motion.md#2)")
        if re.search(r"\b(blue|red|green|grey|gray)\b[^.]{0,30}\b(lips|skin|cheeks|face)\b",
                     pic + " " + s.get("action", ""), re.I):
            W.append(f"{sid}: colour word on skin is painted literally (10_pipeline_mapping.md#4)")
        subj = s.get("subject", "")
        if subj and not set(_words(subj)) & set(_words(pic)):
            W.append(f"{sid}: subject {subj!r} shares no word with the picture; it goes verbatim into H3 "
                     f"(13_prompt_motion.md#4)")
    return W


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="clay_film.director_kb")
    ap.add_argument("op", choices=["index", "ask", "brief", "lint"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--style", default="")
    ap.add_argument("--words-per-s", type=float, default=WPS)
    ap.add_argument("-k", type=int, default=6)
    a = ap.parse_args(argv)
    if a.op != "lint" and not (KB_DIR.is_dir() and any(KB_DIR.glob("*.md"))):
        print(f"director_kb: no knowledge base at {KB_DIR}; set CLAY_FILM_DIRECTOR_KB to a folder of "
              "markdown topic files (`lint` works without one)", file=sys.stderr)
        return 2
    if a.op == "index":
        for s in load():
            print(f"{s.file:34s} {s.heading[:60]}")
        return 0
    if a.op == "ask":
        for score, s in ask(a.arg or "", k=a.k):
            print(f"{score:5.1f}  {s.ref()}\n       {s.use_when}")
        return 0
    if a.op == "brief":
        p = Path(a.arg)
        print(brief(p.read_text(encoding="utf-8"), style=a.style, wps=a.words_per_s))
        return 0
    if a.op == "lint":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from clay_film import direct
        d, _ = direct.load(a.arg)
        ws = lint(d)
        for w in ws:
            print("kb    " + w)
        print(f"{len(ws)} advisory note(s)")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
