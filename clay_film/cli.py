"""python -m clay_film <command>

    locations [--id ID]                 the catalogue, or one location's marks and look_at props
    cast                                who can appear
    guide [--location ID] [--passage] [--seconds N]   the rulebook a planner model gets
    validate PLAN.json [--passage FILE] [--seconds N]
    plan NAME (--brief TEXT | --passage-file FILE) [--seconds 30] [--location ID]
         --endpoint URL --model ID      write pieces/NAME/plan.json with a model
    new NAME PLAN.json                  validate and install a hand-written plan
    run NAME [--stages a,b] [--only s000,s003] [--seeds 1,2] [--force] [--keep-farm]
    status NAME
    shot NAME still|edit|motion|clay SID [--seed N] [--prompt TEXT] [--base V]
                                        one shot; the output becomes a new active version
    versions NAME                       every shot's stills and clips, active ones and staleness
    activate NAME SID still|clip V      choose which version stills/ and clips/ carry
    selftest                            schema + marks + compile, no GPU

CLAY_FILM_DRY=1 fakes every GPU job (see dry.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

from . import cast, locations, planner, schema, stages


def _install(name: str, plan: dict) -> Path:
    piece = stages.piece_dir(name)
    piece.mkdir(parents=True, exist_ok=True)
    (piece / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    schema.compile_plan(plan, piece)
    return piece


def selftest() -> int:
    ex = schema.example()
    errs = schema.validate(ex)
    assert not errs, errs
    bad = json.loads(json.dumps(ex))
    bad["shots"][1]["people"][0]["at"] = "behind-the-moon"
    bad["shots"][1]["camera"]["lens"] = 40
    bad["shots"][1]["people"][1]["who"] = "nobody"
    e2 = schema.validate(bad)
    assert len(e2) == 3, e2
    m = locations.marks("lakeside-pine-path")
    assert m["mid-centre"] == [0.4, 12.0], m["mid-centre"]
    assert locations.resolve_mark("lakeside-pine-path", "mid center") == (0.4, 12.0)
    assert "menote" in cast.registry() and " or " not in cast.registry()["menote"]
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        job = schema.compile_plan(ex, Path(t))
        shots = json.loads((Path(t) / "shots.json").read_text(encoding="utf-8"))
        assert shots[1]["move_target"] == "the distant hooded figure"
        assert job["shots"][1]["figureAt"] == [1.0, 5.0] and job["shots"][1]["extraFigures"][0]["at"] == [-0.6, 19.0]
        assert job["shots"][0]["emptySubject"] == "path_mid"
    g = planner.guide("lakeside-pine-path", seconds=30)
    assert "69 words" in g, "budget line"
    # free shots: no location or marks, skipped by the clay job, prompt carries style and cast
    free = json.loads(json.dumps(ex))
    free["style"] = "A grainy black-and-white photograph"
    free["shots"].append({"mode": "free", "narration": "The lake remembered nothing.",
                          "still_prompt": "a flat grey lake under fog, a single post in the water",
                          "people": [{"who": "stranger", "action": "standing still"}],
                          "subject": "the lake", "scene": "Fog over a grey lake.",
                          "action": "fog drifts", "motion": "a locked-off hold"})
    assert not schema.validate(free), schema.validate(free)
    only_free = {"title": "t", "style": "ink", "cast": ex["cast"], "shots": [free["shots"][-1]]}
    assert not schema.validate(only_free), schema.validate(only_free)
    bad_free = json.loads(json.dumps(only_free))
    bad_free["shots"][0]["people"][0]["who"] = "nobody"
    assert len(schema.validate(bad_free)) == 1
    with tempfile.TemporaryDirectory() as t:
        job = schema.compile_plan(free, Path(t))
        shots = json.loads((Path(t) / "shots.json").read_text(encoding="utf-8"))
        assert len(job["shots"]) == 2 and shots[2]["mode"] == "free"
        assert shots[2]["still_prompt"].startswith("A grainy black-and-white photograph. A flat grey lake")
        assert "void of solid black shadow" in shots[2]["still_prompt"]
    # verbatim passage coverage and chunking
    text = "One two three. Four five, six seven eight nine ten. Eleven."
    chunks = planner.chunk_passage(text, cap=4)
    assert " ".join(chunks) == " ".join(text.split()), chunks
    cov = schema.coverage({"shots": [{"narration": c} for c in chunks]}, text)
    assert cov["exact"] and not cov["errors"], cov
    cov = schema.coverage({"shots": [{"narration": "One two three."}, {"narration": "Eleven."}]}, text)
    assert not cov["exact"] and len(cov["gaps"]) == 1, cov
    print(f"selftest ok: {len(locations.ids())} locations, {len(cast.registry())} cast, "
          f"guide {len(g)} chars, free shots and coverage ok")
    return 0


def stop_all() -> int:
    """Free the GPUs: end every clay_film run or shot job and the Agentic_Slop GPU scripts they
    start (with their children), then stop the farm. Finished stills and clips are kept, and
    `run NAME --stages ...` resumes, because each stage skips shots whose output exists.
    The studio server is left up; its queued jobs fail and can be retried."""
    import psutil
    from . import paths, stages
    # Only clay_film's own runners. The TTS, H3 and assembly scripts are their children and go
    # with them; matching those by name would also kill another session's Agentic_Slop job.
    marks = ("clay_film run", "clay_film shot")
    me = psutil.Process().pid
    victims = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        cmd = " ".join(p.info["cmdline"] or []).replace("-m clay_film", "clay_film")
        if p.info["pid"] != me and any(m in cmd for m in marks):
            victims.append(p)
    for p in victims:
        try:
            for c in p.children(recursive=True):
                c.kill()
            p.kill()
            print(f"stopped {p.pid}")
        except psutil.Error:
            pass
    # Queued chains (adaptations/*/..._chain.py) read this to tell a stop from a failed step:
    # both end a run with a non-zero code, but only a stop means Patrick wants the GPUs.
    (paths.ROOT / "last_stop.txt").write_text(f"{time.time()}\n", encoding="utf-8")
    print(f"{len(victims)} process(es) stopped; stopping the farm")
    stages._run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                 paths.FARM / "farm.ps1", "stop"], cwd=paths.FARM)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="clay_film")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("locations"); a.add_argument("--id")
    sub.add_parser("cast")
    a = sub.add_parser("guide"); a.add_argument("--location"); a.add_argument("--passage", action="store_true")
    a.add_argument("--seconds", type=float, default=30)
    a = sub.add_parser("validate"); a.add_argument("plan"); a.add_argument("--passage"); a.add_argument("--seconds", type=float)
    a = sub.add_parser("plan"); a.add_argument("name"); a.add_argument("--brief"); a.add_argument("--passage-file")
    a.add_argument("--seconds", type=float, default=30); a.add_argument("--location"); a.add_argument("--title")
    a.add_argument("--endpoint", default="http://127.0.0.1:8080/v1"); a.add_argument("--model", default="local")
    a.add_argument("--rounds", type=int, default=3)
    a = sub.add_parser("new"); a.add_argument("name"); a.add_argument("plan")
    a = sub.add_parser("run"); a.add_argument("name"); a.add_argument("--stages"); a.add_argument("--only")
    a.add_argument("--seeds"); a.add_argument("--force", action="store_true"); a.add_argument("--keep-farm", action="store_true")
    a.add_argument("--device", type=int, default=0)
    a = sub.add_parser("status"); a.add_argument("name")
    a = sub.add_parser("shot"); a.add_argument("name"); a.add_argument("op", choices=["still", "edit", "motion", "clay"])
    a.add_argument("sid"); a.add_argument("--seed"); a.add_argument("--prompt"); a.add_argument("--base", type=int)
    a = sub.add_parser("versions"); a.add_argument("name")
    a = sub.add_parser("activate"); a.add_argument("name"); a.add_argument("sid")
    a.add_argument("kind", choices=["still", "clip"]); a.add_argument("v", type=int)
    sub.add_parser("selftest")
    sub.add_parser("stop")
    a = sub.add_parser("cuts"); a.add_argument("name")
    a = ap.parse_args(argv)

    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "stop":
        return stop_all()
    if a.cmd == "cuts":
        from . import cutcheck
        rows = cutcheck.scan(stages.piece_dir(a.name))
        for r in rows:
            print(f"{r['clip']}  jump {r.get('jump', '-'):>5}  x{r.get('ratio', '-'):<5} "
                  f"f{r.get('frame', '-'):<4} {'CUT?' if r['cut'] else ''}")
        return 1 if any(r["cut"] for r in rows) else 0
    if a.cmd == "locations":
        print(json.dumps(locations.describe(a.id) if a.id else locations.catalogue(), indent=1, ensure_ascii=False))
    elif a.cmd == "cast":
        print(json.dumps(cast.registry(), indent=1, ensure_ascii=False))
    elif a.cmd == "guide":
        print(planner.guide(a.location, a.passage, a.seconds))
    elif a.cmd == "validate":
        plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
        passage = Path(a.passage).read_text(encoding="utf-8") if a.passage else None
        errs = schema.validate(plan, passage, a.seconds)
        print("\n".join(errs) or "ok")
        return 1 if errs else 0
    elif a.cmd == "plan":
        passage = Path(a.passage_file).read_text(encoding="utf-8") if a.passage_file else None
        plan, errs = planner.plan(a.endpoint, a.model, brief=a.brief, passage=passage, seconds=a.seconds,
                                  location=a.location, title=a.title, rounds=a.rounds)
        piece = stages.piece_dir(a.name)
        piece.mkdir(parents=True, exist_ok=True)
        (piece / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
        if errs:
            print("plan written but does not validate:\n- " + "\n- ".join(errs))
            return 1
        schema.compile_plan(plan, piece)
        print(f"plan ok: {piece / 'plan.json'} ({len(plan['shots'])} shots)")
    elif a.cmd == "new":
        plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
        src = plan.get("source")
        passage = Path(src).read_text(encoding="utf-8") if src and Path(src).is_file() else None
        errs = schema.validate(plan, passage)
        if errs:
            print("\n".join(errs))
            return 1
        print(f"installed {_install(a.name, plan)}")
    elif a.cmd == "run":
        stages.run(stages.piece_dir(a.name),
                   stages=a.stages.split(",") if a.stages else None,
                   only=set(a.only.split(",")) if a.only else None,
                   seeds=[int(x) for x in a.seeds.split(",")] if a.seeds else None,
                   force=a.force, device=a.device, farm_after="keep" if a.keep_farm else "off")
    elif a.cmd == "status":
        print(json.dumps(stages.status(stages.piece_dir(a.name)), indent=1))
    elif a.cmd == "shot":
        if a.op == "still":
            out = stages.still_one(a.name, a.sid, seed=a.seed, prompt=a.prompt)
        elif a.op == "edit":
            if not a.prompt:
                print("edit needs --prompt")
                return 2
            out = stages.edit_still(a.name, a.sid, a.prompt, seed=a.seed, base_v=a.base)
        elif a.op == "motion":
            out = stages.motion_one(a.name, a.sid, prompt_override=a.prompt, seed=a.seed)
        else:
            out = stages.clay_shots(a.name, [a.sid])
        print(json.dumps(out, indent=1, ensure_ascii=False))
    elif a.cmd == "versions":
        from . import versions
        piece, plan = stages.prepare(a.name)
        rows = versions.shot_info(piece, plan)
        for r in rows:
            print(f"{r['sid']} {r['id']} {r['mode']:4} stills {[x['v'] for x in r['stills']]} "
                  f"active {r['active_still']}{' STALE' if r['still_stale'] else ''} | clips "
                  f"{[x['v'] for x in r['clips']]} active {r['active_clip']}"
                  f"{' STALE: ' + ', '.join(r['clip_stale']) if r['clip_stale'] else ''}")
    elif a.cmd == "activate":
        from . import versions
        piece, plan = stages.prepare(a.name)
        i, s = stages.resolve(plan, a.sid)
        versions.set_active(piece, s["id"], a.kind, a.v)
        print(f"{a.sid} active {a.kind} -> v{a.v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
