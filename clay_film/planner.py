"""Turn a brief or a passage into a plan with any OpenAI-compatible chat endpoint.

    llama-server  --endpoint http://127.0.0.1:8080/v1   (json_schema constrained decoding)
    OpenRouter    --endpoint https://openrouter.ai/api/v1 --model <id>  (key: OPENROUTER_API_KEY)

The model writes only what a director decides: narration, camera words, who stands at which
mark, what moves. Every answer goes through schema.validate, and the errors go back to the model
verbatim for up to `rounds` repairs. A plan that still fails is returned with its errors rather
than patched silently.

`guide()` is the same instruction text an agent gets from the MCP `guide` tool, so a local agent
driving the MCP tools and this one-shot planner work from one rulebook.
"""
import json
import os
import re
import urllib.error
import urllib.request

from . import cast as castmod
from . import locations, schema

RULES = """You are the director of a short narrated film made from still photographs that are then
animated into ~5 second clips. You write a JSON plan. You never write coordinates: people stand at
named marks, cameras are described in fixed vocabulary, and the pipeline solves the geometry.

SHOTS AND NARRATION
- One shot per narration sentence. Each shot is about 5 seconds of film.
- The narrator reads about 2.4 words a second, so the whole plan should hold about
  {budget} words of narration for {seconds} seconds. 8 to 14 words per shot.
- {mode_rule}

PEOPLE
- people[0] is the person the camera is built around. Use [] for a shot with nobody in it, and
  then set look_at to one of the location's look_at prop ids.
- who is a cast id. Never describe what someone looks like in action or placement; the cast
  description is added for you, identical in every shot, which is what keeps a person consistent.
  A person not in the cast goes in plan.cast with one full appearance description.
- at is a mark of the shot's location. A mark is <depth>-<lateral>: near/mid/far is distance from
  the near end of the location (where the default camera side is), left/centre/right is across.
- facing: far = looking away from the near end, near = looking toward it.
- Two people in one shot must stand at different marks.

CAMERA
- angle is relative to where people[0] is looking: frontal sees their face, rear is behind them,
  over_shoulder looks past them, profile_left/right is side-on.
- distance: extreme_close, close, medium (waist up), full_body, wide, very_wide. Indoors, stay at
  full_body or tighter; wide and very_wide are for exteriors only.
- lens in mm: 18, 24, 35, 50, 85, 135. Long lenses (85, 135) flatten and isolate; 35 is neutral.

MOTION (for the clip made from the still)
- scene: one or two plain sentences describing the still as a viewer would see it.
- action: the one continuous thing that moves during the 5 seconds. There are no cuts in a clip.
- motion: one camera move in plain words ("a slow push in", "a locked-off hold", "a slow pull back",
  "a slow tilt down onto the knife").
- move_target: set this when the camera moves toward something other than the subject, for example
  an over-shoulder push toward a distant figure. Without it the camera closes on the subject.
- subject: a short noun phrase, "the hooded man", "the empty pine path".

Answer with the JSON plan only."""


def split_sentences(text: str) -> list[str]:
    """Sentences of a passage, whitespace-normalised, exactly as they appear in it."""
    flat = " ".join(text.split())
    closers, openers = "\"'”’", "\"'“‘"
    out, start = [], 0
    for m in re.finditer(r" ", flat):
        before, after = flat[:m.start()].rstrip(closers), flat[m.end():].lstrip(openers)
        if before.endswith((".", "!", "?")) and after[:1].isupper():
            out.append(flat[start:m.start()])
            start = m.end()
    out.append(flat[start:])
    return [s for s in out if s]


def chunk_passage(text: str, cap: int = 34) -> list[str]:
    """Every sentence of a passage, with any sentence over `cap` words cut at the clause break
    nearest its middle (a comma, semicolon, colon or dash; the middle word if there is none).
    Joined with single spaces the chunks equal the whitespace-normalised passage exactly, which is
    what verbatim narration needs: nothing is skipped, nothing is reworded."""
    out = []

    def cut(s: str) -> None:
        words = s.split(" ")
        if len(words) <= cap:
            out.append(s)
            return
        mid = len(words) / 2
        breaks = [k for k in range(1, len(words)) if words[k - 1].endswith((",", ";", ":", "-"))]
        k = min(breaks, key=lambda k: abs(k - mid)) if breaks else int(mid)
        cut(" ".join(words[:k]))
        cut(" ".join(words[k:]))

    for s in split_sentences(text):
        cut(s)
    return out


def passage_schema(n: int, base: dict | None = None) -> dict:
    """PLAN_SCHEMA with each shot naming a sentence number instead of writing narration.

    Qwen3-14B asked to copy sentences verbatim cut them into fragments and re-capitalised them
    (4 of 7 shots failed the substring check after 3 repair rounds, 2026-09-24). Picking a number
    cannot misquote."""
    sc = json.loads(json.dumps(base or schema.PLAN_SCHEMA))
    shot = sc["properties"]["shots"]["items"]
    shot["properties"].pop("narration")
    shot["properties"] = {"sentence": {"type": "integer", "minimum": 1, "maximum": n,
                                       "description": "number of the sentence read over this shot"},
                          **shot["properties"]}
    shot["required"] = ["sentence"] + [k for k in shot["required"] if k != "narration"]
    sc["properties"]["shots"]["minItems"] = sc["properties"]["shots"]["maxItems"] = n
    return sc


def fill_sentences(p: dict, sents: list[str]) -> list[str]:
    """Replace each shot's sentence number with its text. Returns problems with the numbers."""
    errs, last = [], 0
    for i, s in enumerate(p.get("shots") or []):
        k = s.pop("sentence", None)
        if not isinstance(k, int) or not 1 <= k <= len(sents):
            errs.append(f"shots[{i}].sentence: {k!r} must be a sentence number from 1 to {len(sents)}")
            continue
        if k <= last:
            errs.append(f"shots[{i}].sentence: {k} comes after {last}; use sentences in order, each once")
        last = max(last, k)
        s["narration"] = sents[k - 1]
    if len(p.get("shots") or []) != len(sents):
        errs.append(f"make exactly {len(sents)} shots, one per sentence 1 to {len(sents)}")
    return errs


def guide(location: str | None = None, passage: bool = False, seconds: float = 30,
          sentences: list[str] | None = None) -> str:
    budget = round(seconds * schema.WORDS_PER_SECOND)
    if sentences:
        mode = ("The narration is already chosen: the numbered SENTENCES below. Make exactly one shot "
                "per sentence, in order. Instead of writing narration, each shot gives "
                "\"sentence\": <number>.")
    elif passage:
        mode = ("Narration sentences are copied word for word from the passage, in order. You may "
                "skip sentences to fit the time; never paraphrase.")
    else:
        mode = "Write the narration yourself: plain past-tense prose, concrete, no rhetorical flourishes."
    parts = [RULES.format(budget=budget, seconds=round(seconds), mode_rule=mode)]
    if sentences:
        parts.append("SENTENCES\n" + "\n".join(f"{k}. [{len(t.split())}] {t}"
                                                for k, t in enumerate(sentences, 1)))
    reg = castmod.registry()
    parts.append("CAST (id: appearance; these are already defined, never put them in plan.cast)\n"
                 + "\n".join(f"- {k}: {v}" for k, v in reg.items()))
    if location:
        d = locations.describe(location)
        parts.append("LOCATION\n" + json.dumps(d, ensure_ascii=False))
    else:
        parts.append("LOCATIONS (id: name)\n" + "\n".join(
            f"- {r['id']}: {r['name']}{' (exterior)' if r['exterior'] else ''}"
            for r in locations.catalogue()))
    ex = schema.example()
    if sentences:
        for k, s in enumerate(ex["shots"], 1):
            s.pop("narration")
            s["sentence"] = k
    parts.append("PLAN JSON SCHEMA\n" + json.dumps(passage_schema(len(sentences)) if sentences
                                                   else schema.PLAN_SCHEMA))
    parts.append("EXAMPLE PLAN\n" + json.dumps(ex, ensure_ascii=False))
    return "\n\n".join(parts)


FREE_RULES = """You are the director of a short narrated film made from still images that are then
animated into ~5 second clips. You write a JSON plan. These shots are FREE shots: there is no 3D
set, so each still is generated from text alone. The film may be abstract, stylised or dreamlike.

SHOTS AND NARRATION
- One shot per narration sentence. Each shot is about 5 seconds of film.
- The narrator reads about 2.4 words a second, so the whole plan should hold about
  {budget} words of narration for {seconds} seconds. 8 to 14 words per shot.
- {mode_rule}

THE STILL
- still_prompt describes the picture as a viewer sees it: the medium (photograph, painting,
  etching...), the subject and where it sits in the frame, the setting, the light. Present tense,
  third person, plain words. Never an instruction ("create", "make sure"), no quality words.
- Do not restate the style; plan.style is put in front of every still_prompt for you.
- People: list cast ids in people with an action and placement. Never describe what a cast
  member looks like in still_prompt; their description is added for you, identical in every shot.
- camera is optional: angle, elevation, distance, lens in the fixed vocabulary.

MOTION (for the clip made from the still)
- scene: one or two plain sentences describing the still.
- action: the one continuous thing that moves during the 5 seconds. No cuts.
- motion: one camera move in plain words ("a slow push in", "a locked-off hold").
- move_target: set when the camera moves toward something other than the subject.
- subject: a short noun phrase.

Every shot has "mode": "free". Answer with the JSON plan only."""


def guide_free(passage: bool = False, seconds: float = 30, sentences: list[str] | None = None,
               style: str = "") -> str:
    budget = round(seconds * schema.WORDS_PER_SECOND)
    if sentences:
        mode = ("The narration is already chosen: the numbered SENTENCES below. Make exactly one shot "
                "per sentence, in order. Instead of writing narration, each shot gives "
                "\"sentence\": <number>.")
    elif passage:
        mode = "Narration sentences are copied word for word from the passage, in order."
    else:
        mode = "Write the narration yourself: plain past-tense prose, concrete, no rhetorical flourishes."
    parts = [FREE_RULES.format(budget=budget, seconds=round(seconds), mode_rule=mode)]
    if style:
        parts.append(f"STYLE (already set as plan.style)\n{style}")
    if sentences:
        parts.append("SENTENCES\n" + "\n".join(f"{k}. [{len(t.split())}] {t}"
                                                for k, t in enumerate(sentences, 1)))
    reg = castmod.registry()
    parts.append("CAST (id: appearance; these are already defined, never put them in plan.cast)\n"
                 + "\n".join(f"- {k}: {v}" for k, v in reg.items()))
    parts.append("PLAN JSON SCHEMA\n" + json.dumps(passage_schema(len(sentences), schema.FREE_PLAN_SCHEMA)
                                                   if sentences else schema.FREE_PLAN_SCHEMA))
    return "\n\n".join(parts)


def _key(endpoint: str, api_key: str | None = None) -> str | None:
    if api_key:
        return api_key
    if os.environ.get("CLAY_FILM_API_KEY"):
        return os.environ["CLAY_FILM_API_KEY"]
    if "openrouter" in endpoint:
        return os.environ.get("OPENROUTER_API_KEY")
    return None


def _post(endpoint: str, body: dict, timeout: float, api_key: str | None = None) -> dict:
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    k = _key(endpoint, api_key)
    if k:
        req.add_header("Authorization", f"Bearer {k}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def chat(endpoint: str, model: str, messages: list[dict], json_schema: dict | None,
         temperature: float = 0.4, timeout: float = 600, api_key: str | None = None) -> str:
    """One completion. Tries schema-constrained output, then plain JSON mode, then nothing."""
    formats = []
    if json_schema is not None:
        formats.append({"type": "json_schema",
                        "json_schema": {"name": "plan", "schema": json_schema, "strict": False}})
    formats += [{"type": "json_object"}, None]
    last = None
    for fmt in formats:
        body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": 8192}
        if fmt:
            body["response_format"] = fmt
        try:
            out = _post(endpoint, body, timeout, api_key)
            return out["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:400]}"
            if e.code not in (400, 422, 500):
                break
    raise RuntimeError(f"endpoint refused every response format; last: {last}")


def parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < a:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[a:b + 1])


def pick_location(endpoint: str, model: str, request: str, api_key: str | None = None) -> str:
    ids = locations.ids()
    cat = "\n".join(f"- {r['id']}: {r['name']}" for r in locations.catalogue())
    msg = [{"role": "system", "content": "Pick the one location that best fits the film request. "
                                         "Answer as JSON {\"location\": \"<id>\"}.\n" + cat},
           {"role": "user", "content": request}]
    js = {"type": "object", "required": ["location"],
          "properties": {"location": {"type": "string", "enum": ids}}}
    loc = parse_json(chat(endpoint, model, msg, js, temperature=0.0, api_key=api_key)).get("location")
    if loc not in ids:
        raise ValueError(f"model picked unknown location {loc!r}")
    return loc


def select_sentences(endpoint: str, model: str, sents: list[str], seconds: float,
                     rounds: int = 2, log=print, api_key: str | None = None) -> list[int]:
    """Step one of passage mode: which sentences to narrate, as 1-based numbers in order.

    The model scores every sentence and code fits the best ones into the word budget. Asking
    Qwen3-14B to choose directly failed twice on 2026-09-24: given selection and staging together
    it used all 24 sentences of a 518-word passage for a 30 s film (budget 69), and given selection
    alone it returned all 24 again in four rounds of repair. Scoring each item is a task it does.
    Falls back to an in-order fill if the scores never come back well-formed."""
    budget = seconds * schema.WORDS_PER_SECOND
    n = len(sents)
    listing = "\n".join(f"{k}. {t}" for k, t in enumerate(sents, 1))
    msgs = [{"role": "system", "content":
             "Score each numbered sentence of this passage from 1 to 5 for how much it matters to a "
             "short film of the scene: 5 = a key story event or a strong visible action, 1 = inner "
             f"reflection with nothing to show. Answer as JSON {{\"scores\": [{n} integers, one per "
             "sentence, in order]}."},
            {"role": "user", "content": listing}]
    js = {"type": "object", "required": ["scores"],
          "properties": {"scores": {"type": "array", "minItems": n, "maxItems": n,
                                    "items": {"type": "integer", "minimum": 1, "maximum": 5}}}}
    scores = None
    for r in range(rounds + 1):
        reply = chat(endpoint, model, msgs, js, temperature=0.2, api_key=api_key)
        try:
            got = parse_json(reply).get("scores")
        except (ValueError, json.JSONDecodeError):
            got = None
        if isinstance(got, list) and len(got) == n and all(isinstance(x, int) for x in got):
            scores = got
            break
        log(f"planner: scores round {r} malformed")
        msgs += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": f"Give exactly {n} integer scores from 1 to 5."}]
    order = (sorted(range(n), key=lambda i: (-scores[i], i)) if scores else list(range(n)))
    pick, words = [], 0
    for i in order:
        w = len(sents[i].split())
        if w <= 34 and words + w <= budget * 1.1:
            pick.append(i + 1)
            words += w
    pick.sort()
    log(f"planner: {'scored' if scores else 'in-order fallback'} selection {pick} = {words} words"
        + (f" (scores {scores})" if scores else ""))
    return pick


def plan(endpoint: str, model: str, *, brief: str | None = None, passage: str | None = None,
         seconds: float = 30, location: str | None = None, title: str | None = None,
         rounds: int = 3, log=print, mode: str = "clay", style: str = "", select: bool = True,
         api_key: str | None = None) -> tuple[dict, list[str]]:
    """Returns (plan, remaining_errors). remaining_errors == [] means the plan compiles.

    mode 'free' writes free shots (text-to-image, no location). select=False narrates the whole
    passage verbatim (chunk_passage) instead of scoring sentences into the time budget; seconds
    then follows the passage's length."""
    if not brief and not passage:
        raise ValueError("give a brief or a passage")
    free = mode == "free"
    request = brief or passage
    if not location and not free:
        location = pick_location(endpoint, model, request, api_key=api_key)
        log(f"planner: picked location {location}")
    sents = None
    if passage:
        if select:
            allsents = split_sentences(passage)
            sents = [allsents[k - 1] for k in select_sentences(endpoint, model, allsents, seconds,
                                                                log=log, api_key=api_key)]
        else:
            sents = chunk_passage(passage)
            seconds = max(5.0, len(" ".join(sents).split()) / schema.WORDS_PER_SECOND)
            log(f"planner: whole passage, {len(sents)} shots")
    user = ((f"Make a {round(seconds)} second film." if free
             else f"Make a {round(seconds)} second film in location {location}.")
            + (f" Title: {title}." if title else "")
            + (f" Visual style: {style}." if style else "")
            + ("\n\nNarrate from the numbered SENTENCES in your instructions." if passage
               else f"\n\nBRIEF:\n{brief}"))
    system = (guide_free(bool(passage), seconds, sents, style) if free
              else guide(location, passage=bool(passage), seconds=seconds, sentences=sents))
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    base = schema.FREE_PLAN_SCHEMA if free else schema.PLAN_SCHEMA
    js = passage_schema(len(sents), base) if sents else base
    p, errs = {}, ["no answer yet"]
    for r in range(rounds + 1):
        reply = chat(endpoint, model, msgs, js, api_key=api_key)
        try:
            p = parse_json(reply)
            if free:
                if style:
                    p["style"] = style
                for s in p.get("shots") or []:
                    if isinstance(s, dict):
                        s["mode"] = "free"
            else:
                p.setdefault("location", location)
            num_errs = fill_sentences(p, sents) if sents else []
            # A model copying the cast list into plan.cast would replace canon with its own paraphrase
            # (Qwen3-14B did, 2026-09-24). Bible ids are never redefined by a model-written plan.
            bible = castmod.bible()
            if isinstance(p.get("cast"), dict):
                p["cast"] = {k: v for k, v in p["cast"].items() if k not in bible}
            errs = num_errs + schema.validate(p, passage=" ".join(passage.split()) if passage else None,
                                              target_seconds=seconds)
        except (ValueError, json.JSONDecodeError) as e:
            errs = [f"reply was not valid JSON: {e}"]
        log(f"planner: round {r}: {len(errs)} problem(s)")
        if not errs:
            break
        msgs += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": "Fix these problems and answer with the whole corrected "
                                             "plan as JSON:\n- " + "\n- ".join(errs[:25])}]
    return p, errs


# -- revising what is already there (the studio's chat boxes) ---------------------------------

# Fields a model never writes: the studio's own bookkeeping and hand overrides.
KEEP_FIELDS = ("id", "still_prompt", "motion_prompt", "tail_s", "video_path", "music", "end_crop")


def _shot_errs(errs: list[str], i: int) -> list[str]:
    return [e for e in errs if e.startswith(f"shots[{i}]")]


def revise_shot(endpoint: str, model: str, plan: dict, i: int, request: str, *,
                lock_narration: bool = False, passage: str | None = None, rounds: int = 2,
                api_key: str | None = None) -> tuple[dict, list[str]]:
    """One shot rewritten to a plain-language request ("make this a low angle"). Returns
    (revised_shot, errors). The rest of the plan is context only and is never changed."""
    shot = plan["shots"][i]
    free = schema.shot_mode(shot) == "free"
    loc = shot.get("location") or plan.get("location")
    system = (guide_free(False, 30) if free else guide(loc)) + (
        "\n\nYOU ARE REVISING ONE SHOT of an existing plan. Answer with that one shot as JSON, "
        "every field, changed only as far as the request asks. Keep the same people unless asked.")
    if lock_narration:
        system += " Never change the narration; it is the author's text."
    ctx = {"title": plan.get("title"), "location": plan.get("location"), "style": plan.get("style"),
           "cast": plan.get("cast") or {},
           "previous_shot_narration": plan["shots"][i - 1].get("narration") if i > 0 else None,
           "next_shot_narration": plan["shots"][i + 1].get("narration")
           if i + 1 < len(plan["shots"]) else None}
    model_shot = {k: v for k, v in shot.items() if k not in KEEP_FIELDS}
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": "CONTEXT\n" + json.dumps(ctx, ensure_ascii=False)
             + "\n\nSHOT\n" + json.dumps(model_shot, ensure_ascii=False) + "\n\nREQUEST\n" + request}]
    js = schema.FREE_SHOT_SCHEMA if free else schema.SHOT_SCHEMA
    new, errs = dict(shot), ["no answer yet"]
    for _ in range(rounds + 1):
        reply = chat(endpoint, model, msgs, js, api_key=api_key)
        try:
            got = parse_json(reply)
            new = {**got, **{k: shot[k] for k in KEEP_FIELDS if k in shot}}
            if free:
                new["mode"] = "free"
            elif "mode" in shot:
                new["mode"] = shot["mode"]
            if lock_narration:
                new["narration"] = shot.get("narration", "")
            trial = {**plan, "shots": plan["shots"][:i] + [new] + plan["shots"][i + 1:]}
            errs = _shot_errs(schema.validate(trial, passage), i)
        except (ValueError, json.JSONDecodeError) as e:
            errs = [f"reply was not valid JSON: {e}"]
        if not errs:
            break
        msgs += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": "Fix these problems and answer with the whole corrected "
                                             "shot as JSON:\n- " + "\n- ".join(errs[:15])}]
    return new, errs


def revise_plan(endpoint: str, model: str, plan: dict, request: str, *,
                lock_narration: bool = False, passage: str | None = None, rounds: int = 2,
                api_key: str | None = None) -> tuple[dict, list[str]]:
    """The whole plan rewritten to a request. Shot ids and hand overrides are carried over by
    position when the shot count is unchanged, otherwise by identical narration."""
    free = all(schema.shot_mode(s) == "free" for s in plan["shots"])
    system = (guide_free(False, 30, style=plan.get("style", "")) if free else guide(plan.get("location"))) + (
        "\n\nYOU ARE REVISING AN EXISTING PLAN. Answer with the whole plan as JSON, changed only as "
        "far as the request asks. Shots with \"mode\": \"free\" keep that mode and its fields.")
    if lock_narration:
        system += (" Never change, split, merge, add or drop narration; it is the author's text and "
                   "every shot keeps its narration exactly.")
    strip = {**plan, "shots": [{k: v for k, v in s.items() if k not in KEEP_FIELDS} for s in plan["shots"]]}
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": "PLAN\n" + json.dumps(strip, ensure_ascii=False)
             + "\n\nREQUEST\n" + request}]
    new, errs = dict(plan), ["no answer yet"]
    for _ in range(rounds + 1):
        reply = chat(endpoint, model, msgs, None if not free else schema.FREE_PLAN_SCHEMA, api_key=api_key)
        try:
            got = parse_json(reply)
            shots = got.get("shots") or []
            old = plan["shots"]
            by_narr = {" ".join((s.get("narration") or "").split()): s for s in old}
            for k, s in enumerate(shots):
                src = old[k] if len(shots) == len(old) else by_narr.get(" ".join((s.get("narration") or "").split()))
                if src:
                    for f in KEEP_FIELDS:
                        if f in src:
                            s[f] = src[f]
                    if lock_narration and len(shots) == len(old):
                        s["narration"] = src.get("narration", "")
            for f, v in plan.items():      # a plan-level field the model left out is kept
                got.setdefault(f, v)
            new = got
            errs = schema.validate(new, passage)
        except (ValueError, json.JSONDecodeError) as e:
            errs = [f"reply was not valid JSON: {e}"]
        if not errs:
            break
        msgs += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": "Fix these problems and answer with the whole corrected "
                                             "plan as JSON:\n- " + "\n- ".join(errs[:25])}]
    return new, errs
