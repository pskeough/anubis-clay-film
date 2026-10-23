"""The stages, in order: compile -> narrate -> clay -> stills -> motion -> assemble.

Each stage reads what the one before wrote in the piece folder and records itself in status.json,
so a run can stop anywhere and resume with --stages. GPU ownership is explicit:

    narrate   Qwen3-TTS on GPU 0, farm stopped (TTS and a loaded ComfyUI do not share a 16 GB card)
    stills    farm pair, gpu0 8190 + gpu1 8191, one Qwen-Image-2.1 edit per shot
    motion    farm big worker 8192 spanning both cards (H3 int8 is 21 GB)

The clay stage repairs two gate failures itself and reports the rest:
    fov > 85   -> the solver widened past a real lens. wide/very_wide tighten one step; otherwise
                  the camera is backed against a wall, so the people move 1.2 m away from it
                  (dorm-room, someone at a near mark facing near: 134 deg at every distance)
    void       -> step the people back toward the camera until the model extends 6 m past them
cover and sparse are reported, not repaired: they mean the shot needs a director's eye.
"""
import copy
import json
import re
import os
import queue as _queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from . import paths, schema

STAGES = ["compile", "narrate", "clay", "stills", "endframes", "motion", "assemble"]
STILL_STEPS = 18
T2I_STEPS = 25            # free shots start from noise, not from a clay pass
STILL_W, STILL_H = 1344, 768
# Which Qwen-Image-2.1 transformer renders stills, edits and end frames (farm graphs.QWEN21_VARIANTS):
# "base" at the step counts above, "turbo" (official Qwen-Image-2.1-Turbo, 8 steps on its saved
# sigmas) or "turbo_simple" (same model, euler/simple at 8). Chosen by, in order: the model
# argument (CLI --model), CLAY_FILM_STILL_MODEL, the plan's still_model (config.json), "base".
STILL_MODELS = schema.STILL_MODELS
# CLAY_FILM_DRY=1 (or the studio's --dry) fakes every GPU job from fixture files. Clay, compile
# and assembly still run for real: they are CPU work.
DRY = os.environ.get("CLAY_FILM_DRY") == "1"
VOID_M = 6.0
FOV_STEP_M = 1.2
PAIR_PORTS = (8190, 8191)
BIG_PORT = 8192


# -- bookkeeping ----------------------------------------------------------------

def piece_dir(name_or_path: str) -> Path:
    p = Path(name_or_path)
    return p if p.is_absolute() or p.exists() else paths.PIECES / name_or_path


def status(piece: Path) -> dict:
    f = piece / "status.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def still_model(piece: Path, model: str | None = None) -> str:
    f = Path(piece) / "config.json"
    cfg = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    m = model or os.environ.get("CLAY_FILM_STILL_MODEL") or cfg.get("still_model") or "base"
    if m not in STILL_MODELS:
        raise ValueError(f"still model {m!r}: use one of {STILL_MODELS}")
    return m


def _mark(piece: Path, stage: str, state: str, **detail) -> None:
    st = status(piece)
    row = st.get(stage, {})
    row.update({"state": state, "at": datetime.now().isoformat(timespec="seconds"), **detail})
    st[stage] = row
    (piece / "status.json").write_text(json.dumps(st, indent=1), encoding="utf-8")


_TLS = threading.local()


def set_sink(fn) -> None:
    """Send this thread's progress lines to fn as well as stdout (the studio's job log)."""
    _TLS.sink = fn


def _say(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)
    sink = getattr(_TLS, "sink", None)
    if sink:
        sink(msg)


def _run(cmd: list, cwd: Path | None = None) -> None:
    _say("$ " + " ".join(str(c) for c in cmd))
    if getattr(_TLS, "sink", None) is None:
        r = subprocess.run([str(c) for c in cmd], cwd=cwd)
        rc = r.returncode
    else:
        # the studio wants the child's lines in its job log
        # The pipe is read on a side thread and the loop ends when the child exits, not at EOF.
        # farm.ps1 start launches ComfyUI workers that inherit the pipe handle and keep it open
        # for their whole life, so EOF never comes and a plain `for line in p.stdout` hung the
        # studio's first real still job. The pump thread is left blocked on that handle; the
        # workers log to their own files, so nothing is ever written to it.
        p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        lines: _queue.Queue = _queue.Queue()

        def pump():
            for ln in p.stdout:
                lines.put(ln)
            lines.put(None)

        threading.Thread(target=pump, daemon=True).start()
        exited_at = None
        while True:
            try:
                line = lines.get(timeout=0.5)
            except _queue.Empty:
                if p.poll() is None:
                    continue
                exited_at = exited_at or time.monotonic()
                if time.monotonic() - exited_at > 1.0:
                    break
                continue
            if line is None:
                break
            if line.strip():
                _say(line.rstrip())
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"exit {rc}: {cmd[1] if len(cmd) > 1 else cmd[0]}")


# -- farm -----------------------------------------------------------------------

def _up(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=3):
            return True
    except OSError:
        return False


def farm(mode: str) -> None:
    """mode: 'pair', 'big' or 'off'. Restarts only when the wrong shape is running."""
    if DRY:
        from . import dry
        return dry.farm(mode, _say)
    pair = [_up(p) for p in PAIR_PORTS]
    big_up = _up(BIG_PORT)
    anything = big_up or any(pair)
    if (mode == "pair" and all(pair) and not big_up) or (mode == "big" and big_up) \
            or (mode == "off" and not anything):
        return
    ps = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", paths.FARM / "farm.ps1"]
    if anything:
        _run(ps + ["stop"], cwd=paths.FARM)
    if mode == "pair":
        _run(ps + ["start", "-Skip"], cwd=paths.FARM)
    elif mode == "big":
        _run(ps + ["start", "-Big", "-Skip"], cwd=paths.FARM)


def _gate_first_load_only(f) -> None:
    """The farm's RAM gate runs before every job, but only a worker's first Qwen-2.1 job loads
    weights; later ones are warm. With the two warm workers resident, free RAM sits under 8 GB,
    so each later still waited the full 300 s (Anubis, 2026-09-24: 6.5 GB free, 7 stills gated).
    Gate each worker's first job only."""
    seen, orig = set(), f._load_gate

    def gate(name: str) -> None:
        if name not in seen:
            seen.add(name)
            orig(name)
    f._load_gate = gate


# -- stages ---------------------------------------------------------------------

def compile_stage(piece: Path, **_) -> None:
    plan = json.loads((piece / "plan.json").read_text(encoding="utf-8"))
    src = plan.get("source")
    passage = Path(src).read_text(encoding="utf-8") if src and Path(src).is_file() else None
    errs = schema.validate(plan, passage)
    # A directed plan narrates its whole source and nothing else, in order. The per-shot substring
    # check passes a plan that skips or repeats text, so check the join (clay_film.direct).
    if plan.get("directed"):
        if passage is None:
            errs.append("plan.source: a directed plan needs its narration source file")
        elif not schema.coverage(plan, passage)["exact"]:
            errs.append("plan: the shots' narration joined is not plan.source; recompile the "
                        "direction against the current text")
    if errs:
        raise ValueError("plan.json does not validate:\n- " + "\n- ".join(errs))
    job = schema.compile_plan(plan, piece)
    _say(f"compiled {len(job['shots'])} shots")


def narrate(piece: Path, device: int = 0, **_) -> None:
    if DRY:
        from . import dry
        return dry.narrate(piece, _say)
    farm("off")
    cfg = json.loads((piece / "config.json").read_text(encoding="utf-8"))
    cmd = [paths.PODCAST_PY, paths.LIB / "narrate_beats.py", piece, "--device", str(device)]
    if cfg.get("narrator"):
        cmd += ["--design", cfg["narrator"]]
    if cfg.get("voice_ref"):
        cmd += ["--voice-ref-json", json.dumps(cfg["voice_ref"], ensure_ascii=False)]
    _run(cmd, cwd=paths.LIB)
    tl = json.loads((piece / "timeline.json").read_text(encoding="utf-8"))
    _say(f"narration done: {json.dumps(tl)[:200]}")


def _prep(job: dict, piece: Path) -> list[dict]:
    f = piece / "clay_job_run.json"
    f.write_text(json.dumps(job, indent=1, ensure_ascii=False), encoding="utf-8")
    npx = shutil.which("npx.cmd") or shutil.which("npx") or "npx"
    _say(f"$ npx tsx src/_filmprep.ts ({len(job['shots'])} shots)")
    r = subprocess.run([npx, "tsx", "src/_filmprep.ts", str(f)], cwd=paths.ROKO,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError("clay prep failed:\n" + (r.stderr or r.stdout)[-2000:])
    return [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]


def _shift(s: dict, piece: Path, dx: float, dz: float) -> None:
    """Move every person in a shot by (dx, dz), each landing on a free spot."""
    from . import locations
    loc = s["location"]
    s["figureAt"] = list(locations.nudge(loc, s["figureAt"][0] + dx, s["figureAt"][1] + dz))
    for e in s.get("extraFigures", []):
        e["at"] = list(locations.nudge(loc, e["at"][0] + dx, e["at"][1] + dz))


def clay(piece: Path, only: set | None = None, rounds: int = 4, **_) -> None:
    job = json.loads((piece / "clay_job.json").read_text(encoding="utf-8"))
    shots = json.loads((piece / "shots.json").read_text(encoding="utf-8"))
    todo = [s for s in job["shots"] if not only or s["sid"] in only]
    report, repairs = {}, []
    for rnd in range(rounds + 1):
        rows = _prep({**job, "shots": todo}, piece)
        by = {r["sid"]: r for r in rows}
        report.update(by)
        retry = []
        for s in todo:
            g = by[s["sid"]]["gates"]
            fix = None
            # the geometry this round actually rendered; a fix found in the final round is never
            # rendered, so it is undone below rather than written back (gardening audit 2026-10-11)
            snap = (s["shot"]["distance"], copy.deepcopy(s.get("figureAt")),
                    copy.deepcopy(s.get("extraFigures")), s.get("_moved"))
            ux, uz = by[s["sid"]]["view"]
            d = schema.DISTANCES.index(s["shot"]["distance"])
            if any(x.startswith("fov") for x in g) and d > 3:
                # wide / very_wide: the frame asks for more throw than the room has
                s["shot"]["distance"] = schema.DISTANCES[d - 1]
                fix = f"distance -> {s['shot']['distance']}"
            elif any(x.startswith("fov") for x in g) and s.get("figureAt") \
                    and not (s.get("_moved") and d > 0):
                # the camera is backed against a wall: give it room by moving the people away from it.
                # Alternates with tightening, because in a 3 x 4 m room the far wall stops the move.
                _shift(s, piece, ux * FOV_STEP_M, uz * FOV_STEP_M)
                s["_moved"] = True
                fix = f"people moved {FOV_STEP_M} m away from the camera -> {s['figureAt']}"
            elif any(x.startswith("fov") for x in g) and d > 0:
                s["shot"]["distance"] = schema.DISTANCES[d - 1]
                s["_moved"] = False
                fix = f"distance -> {s['shot']['distance']}"
            elif any(x.startswith("void") for x in g) and s.get("figureAt"):
                step = VOID_M + 1.0 - (by[s["sid"]]["reach"] or 0)
                _shift(s, piece, -ux * step, -uz * step)
                fix = f"people moved {step:.1f} m toward camera -> {s['figureAt']}"
            if fix and rnd < rounds:
                repairs.append(f"{s['sid']} round {rnd}: {', '.join(g)}; {fix}")
                _say(f"repair {repairs[-1]}")
                retry.append(s)
            elif fix:
                s["shot"]["distance"], fa, ef, mv = snap
                for k, v in (("figureAt", fa), ("extraFigures", ef), ("_moved", mv)):
                    if v is None:
                        s.pop(k, None)
                    else:
                        s[k] = v
                _say(f"{s['sid']}: gates still open after {rounds} repairs ({', '.join(g)}); kept the rendered geometry")
        if not retry:
            break
        todo = retry
    # write the repaired geometry back so later runs and the shot list agree
    for s in job["shots"]:
        sh = shots[int(s["sid"][1:])]["roko"]
        sh["distance"] = s["shot"]["distance"]
        if s.get("figureAt"):
            sh["figureAt"] = s["figureAt"]
            sh["extraFigures"] = s.get("extraFigures", [])
    (piece / "clay_job.json").write_text(json.dumps(job, indent=1, ensure_ascii=False), encoding="utf-8")
    (piece / "shots.json").write_text(json.dumps(shots, indent=1, ensure_ascii=False), encoding="utf-8")
    failed = {k: v["gates"] for k, v in report.items() if v["gates"]}
    _mark(piece, "clay", "running", report=report, repairs=repairs, open_gates=failed)
    for k, v in sorted(report.items()):
        _say(f"{k} fov {v['fov']} visible {v['visible']} cover {v['maxCover']} reach {v['reach']} "
             f"{'GATES ' + ', '.join(v['gates']) if v['gates'] else 'ok'}")


def _sheet(files: list[Path], out: Path) -> None:
    from PIL import Image
    ims = [Image.open(f).convert("RGB") for f in files]
    if not ims:
        return
    tw = 448
    th = round(tw * ims[0].height / ims[0].width)
    cols = min(3, len(ims))
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), (20, 20, 20))
    for k, im in enumerate(ims):
        sheet.paste(im.resize((tw, th)), ((k % cols) * tw, (k // cols) * th))
    sheet.save(out, quality=88)


# -- Qwen-2.1 edit hygiene (gardening_2min, 2026-10-10) ------------------------------------------
# Every edit of a photograph came back relit and over-sharpened (contrast x1.3, detail x2.3 against
# its input). Two causes, both measured in qwen21_turbo_bench/grid_sweep.py on one canvas:
#  1. noise replay (diffusers #14824): an edit sampled with the seed that generated its input re-adds
#     that input's own noise. The pieces used one seed for everything, so the refs plate, every
#     shot and every end frame shared 2207. Same seed x1.31/x2.29; a different seed x0.98/x1.25.
#  2. the resolved latent grid (ComfyUI #16435, open): resolution=0 on 1344x768 is grid 84x48 and
#     still noisier than 960 (1280x736, grid 80x46: x0.96/x1.04). Edits run at EDIT_RES and are
#     resized back to the still size.
EDIT_RES = 960


def edit_seed(seed: int, salt: int) -> int:
    """A seed for an edit that can never equal the (small, per-piece) seed its input was made with."""
    return (int(seed) * 1_000_003 + (int(salt) + 1) * 7_919 + 104_729) % 2_147_483_647


def _fit(p: Path) -> None:
    """Resize an edit output back to the still size (EDIT_RES resolves 1344x768 to 1280x736)."""
    from PIL import Image
    im = Image.open(p)
    if im.size != (STILL_W, STILL_H):
        im.convert("RGB").resize((STILL_W, STILL_H), Image.LANCZOS).save(p)


_PEOPLE_REF = re.compile(r"The (?:the |a |an )?(.+?) is the person in <image(\d+)>")


def identity_prompt(names: list[str]) -> str:
    """Pass 2 of a reference shot: image_1 is the shot's own t2i frame, image_2.. the character sheets.
    The location plate is never the canvas (it made every gardening shot the empty diner)."""
    # face and hair only: a character sheet shows one costume (the diner coat), and the shot's own
    # prompt already dresses them for where they are (lab suit, base layer)
    who = " ".join(f"The {n} in <image1> is the person in <image{k}>: give the {n} that person's face "
                   f"and hair." for k, n in enumerate(names, start=2))
    return ("Keep <image1> exactly as it is: its framing, camera angle, composition, the setting, every "
            f"pose, all the clothing, the light and the colours. {who} Change nothing else.")


def people_refs(s: dict) -> list[tuple[str, str]]:
    """(name, ref path) for the people in a compiled shot's ref_prompt, in image order; places dropped."""
    out = []
    for name, k in sorted(_PEOPLE_REF.findall(s.get("ref_prompt", "")), key=lambda x: int(x[1])):
        k = int(k)
        if 1 <= k <= len(s.get("refs") or []):
            out.append((name, s["refs"][k - 1]))
    return out


def stills(piece: Path, only: set | None = None, seeds: list[int] | None = None,
           force: bool = False, model: str | None = None, **_) -> None:
    from PIL import Image
    sys.path.insert(0, str(paths.FARM))
    import graphs  # noqa: E402  (farm modules)
    from dispatch import Farm, Job  # noqa: E402
    cfg = json.loads((piece / "config.json").read_text(encoding="utf-8"))
    _direction_gate(piece, cfg)
    variant = still_model(piece, model)
    _say(f"stills: qwen21 {variant}")
    seeds = seeds or [cfg.get("seed", 31337)]
    blk, raw = piece / "blockout", piece / "stills_raw"
    final = piece / ("stills" if len(seeds) == 1 else "stills_cand")
    final.mkdir(parents=True, exist_ok=True)
    compiled = {f"s{s['i']:03d}": s for s in
                json.loads((piece / "shots.json").read_text(encoding="utf-8"))}
    jobs = []
    # the clay job (people descs per shot) and the prep rows (who is shown), for mannequin tinting
    jf, pf = piece / "clay_job_run.json", blk / "prep_last.json"
    tint_jobs = ({s["sid"]: s for s in json.loads(jf.read_text(encoding="utf-8"))["shots"]}
                 if jf.exists() else {})
    tint_prep = ({r["sid"]: r for r in json.loads(pf.read_text(encoding="utf-8"))} if pf.exists() else {})
    for clay_png in sorted(blk.glob("s*_clay.png")):
        sid = clay_png.name[:4]
        if only and sid not in only:
            continue
        if compiled.get(sid, {}).get("mode") == "free":
            continue    # a stale clay pass from before the shot became free
        # a hand-written still_prompt in the plan replaces the one _filmprep.ts wrote
        prompt = (compiled.get(sid, {}).get("still_prompt")
                  or (blk / f"{sid}_prompt.txt").read_text(encoding="utf-8"))
        # The edit input is the clay with a lit mannequin per visible person (_filmprep writes
        # <sid>_people.png after the clay). The empty clay left placement to the prompt, and the model
        # cloned people and swapped their sides (gardening p01, 2026-10-11). An older people pass than
        # the clay is stale: that shot has nobody in frame now.
        people = blk / f"{sid}_people.png"
        src = people if people.exists() and people.stat().st_mtime >= clay_png.stat().st_mtime else clay_png
        newest_in = max(src.stat().st_mtime, (blk / f"{sid}_prompt.txt").stat().st_mtime)
        if src == people:
            # each mannequin tinted its person's clothes colour (clay_film/tint.py): grey ones let the
            # model swap two people and double a third (gardening p13, 2026-10-11). The tinted file is
            # derived, so freshness is still judged on the people pass.
            tinted = blk / f"{sid}_people_tint.png"
            if not tinted.exists() or tinted.stat().st_mtime < people.stat().st_mtime:
                from .tint import tint_people
                tinted = tint_people(blk, sid, [p["desc"] for p in tint_jobs.get(sid, {}).get("people", [])],
                                     tint_prep.get(sid, {}).get("shown"))
            if tinted is not None:
                src = tinted
        # a per-shot reroll pin: <sid>_seed.txt in blockout/ (one integer) overrides the piece seed
        pin = blk / f"{sid}_seed.txt"
        if pin.exists() and len(seeds) == 1:
            seeds_here = [int(pin.read_text().strip())]
            newest_in = max(newest_in, pin.stat().st_mtime)
        else:
            seeds_here = seeds
        for seed in seeds_here:
            name = sid if len(seeds) == 1 else f"{sid}_{seed}"
            # Done only when a still is newer than its clay pass and prompt. Skipping on any
            # existing stills/ file meant a re-rendered clay pass never reached the still
            # (Anubis, 2026-09-24: 12 fixed shots, "0 rendered", motion started on the old ones).
            if not force and any((d / f"{name}.png").exists()
                                 and (d / f"{name}.png").stat().st_mtime > newest_in
                                 for d in (final, raw)):
                continue
            # A unique SaveImage prefix: an identical graph is a ComfyUI cache hit that reports the
            # previous output's filename, which the farm has already moved away (HTTP 404 on /view).
            g = graphs.qwen21_edit(prompt, [src.name], steps=STILL_STEPS, resolution=EDIT_RES,
                                   seed=seed, prefix=f"{name}_{int(time.time())}", variant=variant)
            jobs.append(Job(job_id=name, graph=g, model_key="qwen21", out_name=name,
                            meta={"sid": sid, "seed": seed, "edit": True}, inputs=[src]))
    # free shots: text-to-image from the compiled prompt (style clause + still_prompt + cast). A shot
    # with people in its refs then gets pass 2, an identity edit of that frame (identity_prompt).
    t2i_dir = raw / "_t2i"
    second = {}   # name -> (shot, seed) for pass 2
    for sid, s in compiled.items():
        if s.get("mode") != "free" or (only and sid not in only):
            continue
        for seed in seeds:
            name = sid if len(seeds) == 1 else f"{sid}_{seed}"
            if not force and (final / f"{name}.png").exists():
                continue
            g = graphs.qwen21_t2i(s["still_prompt"], width=STILL_W, height=STILL_H, steps=T2I_STEPS,
                                  seed=seed + s["i"], prefix=f"{name}_{int(time.time())}", variant=variant)
            people = people_refs(s) if s.get("refs") else []
            jobs.append(Job(job_id=name, graph=g, model_key="qwen21", out_name=name,
                            meta={"sid": sid, "seed": seed + s["i"], "mode": "free", "two_pass": bool(people)}))
            if people:
                second[name] = (s, seed)
    if jobs and DRY:
        from . import dry
        raw.mkdir(parents=True, exist_ok=True)
        for j in jobs:
            dry.still("free" if j.meta.get("mode") == "free" else "clay", raw / f"{j.out_name}.png",
                      int(j.meta["sid"][1:]), j.meta["seed"], "", log=_say)
        jobs, second = [], {}
    if jobs:
        farm("pair")

        def _farm(into: Path):
            f = Farm(into, job_timeout=1800)
            # The farm's 18 GB RAM gate is sized for the Qwen3-VL-32B encoder. Qwen-2.1 reads the 8B
            # w4a8 one; with 18 GB the first run waited out the full 300 s gate on a box at 9.8 GB free.
            f.LOAD_MIN_FREE_GB = 8.0
            # The 90 s stagger is held on every job, which serialises the pair. Only the first job per
            # worker loads weights here; later ones hit the cache and sample in ~30 s.
            f.LOAD_STAGGER_S = 20.0
            _gate_first_load_only(f)
            return f
        # pass 1: two-pass shots land in raw/_t2i/ (kept for review), the rest straight in raw/
        one = [j for j in jobs if not j.meta.get("two_pass")]
        pre = [j for j in jobs if j.meta.get("two_pass")]
        done = _farm(raw).run(one) if one else []
        if pre:
            t2i_dir.mkdir(parents=True, exist_ok=True)
            done += _farm(t2i_dir).run(pre)
        for d in done:
            if d.status == "ok" and d.files and next((j for j in jobs if j.job_id == d.job_id), None).meta.get("edit"):
                _fit(raw / d.files[0])
        bad = [d.job_id for d in done if d.status != "ok"]
        # pass 2: identity edit, canvas = the shot's own t2i frame, never the location plate
        jobs2 = []
        for name, (s, seed) in second.items():
            canvas = t2i_dir / f"{name}.png"
            if name in bad or not canvas.exists():
                continue
            people = people_refs(s)
            inputs = [canvas] + _stage_refs([p for _, p in people], piece / "_refs", name)
            g = graphs.qwen21_edit(identity_prompt([n for n, _ in people]), [x.name for x in inputs],
                                   steps=T2I_STEPS, resolution=EDIT_RES, seed=edit_seed(seed, s["i"]),
                                   prefix=f"{name}_id_{int(time.time())}", variant=variant)
            jobs2.append(Job(job_id=name, graph=g, model_key="qwen21", out_name=name, inputs=inputs,
                             meta={"sid": s and f"s{s['i']:03d}", "mode": "identity"}))
        if jobs2:
            _say(f"stills: identity pass on {len(jobs2)} shots")
            for d in _farm(raw).run(jobs2):
                if d.status == "ok" and d.files:
                    _fit(raw / d.files[0])
                else:
                    # the clean t2i frame stands in; the shot keeps its composition, loses only identity
                    _say(f"stills: identity pass failed for {d.job_id} ({d.error}); using the t2i frame")
                    shutil.copy2(t2i_dir / f"{d.job_id}.png", raw / f"{d.job_id}.png")
        if bad:
            raise RuntimeError(f"stills failed: {bad} (rerun the stage; finished stills are kept)")
    for src in sorted(raw.glob("s*.png")) if raw.exists() else []:
        dest = final / src.name
        if force or not dest.exists() or dest.stat().st_mtime < src.stat().st_mtime:
            if not only or src.name[:4] in only:
                Image.open(src).convert("RGB").save(dest)
    _sheet(sorted(final.glob("s*.png")), piece / f"{final.name}_sheet.jpg")
    _say(f"stills: {len(jobs)} rendered into {final.name}/, sheet {final.name}_sheet.jpg")


def motion(piece: Path, only: set | None = None, **_) -> None:
    if _remote_motion():
        # rented pods (farm/config.py remote_workers): every shot in one batch over the tunnels,
        # no local farm
        from . import motion_batch
        motion_batch.motion_stage(piece, only)
        return
    compiled = json.loads((piece / "shots.json").read_text(encoding="utf-8"))
    if DRY or any(s.get("motion_prompt") or s.get("motion_plan") for s in compiled):
        # a hand-edited H3 prompt, a director's motion plan (or the dry backend): shot by shot through
        # motion_one, which builds or reads the prompt. Without any, animate_shots runs unchanged.
        for s in compiled:
            sid = f"s{s['i']:03d}"
            if (only and sid not in only) or (piece / "clips" / f"{sid}.mp4").exists():
                continue
            if (piece / "stills" / f"{sid}.png").exists():
                motion_one(piece, sid)
        return
    farm("big")
    cmd = [paths.SYS_PY, paths.LIB / "animate_shots.py", piece]
    if only:
        cmd += ["--only", ",".join(str(int(s[1:])) for s in sorted(only))]
    t0 = time.time()
    try:
        _run(cmd, cwd=paths.LIB)
    finally:
        _register_batch_clips(piece, t0)
    _report_cuts(piece, only)


def _register_batch_clips(piece: Path, since: float) -> None:
    """Record clips animate_shots wrote into versions.json as the active clip.

    animate_shots writes clips/ directly. Any later prepare() rematerializes clips/ from
    versions.json, so an unregistered clip is overwritten by the shot's older active version:
    on 2026-10-03 one `shot ... motion` call reverted 25 restyled oceanus clips to the overnight
    ones. Registering each new clip as it lands closes that."""
    from . import versions
    plan = load_plan(piece)
    for i, s in enumerate(plan["shots"]):
        f = piece / "clips" / f"s{i:03d}.mp4"
        if s.get("id") and f.exists() and f.stat().st_mtime >= since:
            versions.add_clip(piece, s["id"], f, {"kind": "h3", "backend": "batch"}, sync=False)


def _report_cuts(piece: Path, only: set | None = None) -> None:
    """A report, not a gate: flagged clips stay in place for a person to judge."""
    from . import cutcheck
    try:
        found = [r for r in cutcheck.scan(piece, only) if r["cut"]]
    except Exception as e:  # noqa: BLE001  cv2 missing or a clip unreadable
        _say(f"cut check skipped: {e}")
        return
    _mark(piece, "motion", "running", suspected_cuts=found)   # run() marks it done after
    _say("cut check: " + (", ".join(f"{r['clip']} at f{r['frame']} (x{r['ratio']})" for r in found)
                          if found else "no invented cuts"))


ENDFRAME_LOOK = "loose painted figures, no photographic detail"
# (pattern on the shot's motion, (crop size, x0, y0) as fractions of the still). First match wins.
# A size of 1.0 means the camera ends where it started: the still itself is the end frame.
# None means no end frame; "out" means paint the still outward (a pull back).
END_CROPS = [
    # A pull back ends on ground the still does not show, so its end frame is the still shrunk into
    # a wider canvas that Qwen paints out ("out"). With no end frame H3 cut to a photographic
    # close-up of hands (oceanus s038, 2026-10-03).
    (r"pull(s|ing)? (back|out)|widen", "out"),
    # an arc ends on a side of the subject the still does not show; a cropped end frame fought
    # the move and H3 invented a second cook and a steamship to bridge them (oceanus s023).
    # The still itself as end frame keeps the look and shrinks the arc to a sway.
    (r"\barc|orbit|around (him|her|it|the)", (1.0, 0.0, 0.0)),
    (r"static|locked|hold", (1.0, 0.0, 0.0)),
    (r"pan.*left|left\b.*(drift|glide|track)|(drift|glide|track).*left", (0.80, 0.0, 0.10)),
    (r"pan.*right|right\b.*(drift|glide|track)|(drift|glide|track).*right|drift|across", (0.80, 0.20, 0.10)),
    (r"tilt up|crane up|rise|up the", (0.80, 0.10, 0.0)),
    (r"tilt down|crane down|down from|sink|following him down", (0.80, 0.10, 0.20)),
    (r"push|close|toward|in on|into", (0.70, 0.15, 0.15)),
]


def end_crop(shot: dict) -> tuple:
    """Where the camera ends, as a crop of the still. The plan can set end_crop: [size, x0, y0]."""
    if shot.get("end_crop"):
        return tuple(shot["end_crop"])
    m = (shot.get("motion") or "").lower()
    for pat, c in END_CROPS:
        if re.search(pat, m):
            return c
    return (0.85, 0.075, 0.075)


MID_GUIDE_MIN_FRAMES = 160   # ~6.7 s; shorter clips held between first and last alone


def mid_frame(n: int) -> int:
    return n // 2


def mid_keyframe(still: Path, shot: dict, out: Path) -> Path:
    """A middle keyframe halfway along the camera move: half the end crop of the painted still.
    No repaint: a crop of a painting is still a painting, and this only has to stop H3 wandering
    (a tavern turned into a beach mid-clip, oceanus s025). Pull backs and arcs use the still."""
    crop = end_crop(shot)
    ff = _animate().find_ffmpeg()
    if not isinstance(crop, tuple) or crop[0] >= 1.0:
        shutil.copy2(still, out)
        return out
    size, x0, y0 = crop
    hs, hx, hy = (1 + size) / 2, x0 / 2, y0 / 2
    _run([ff, "-y", "-v", "error", "-i", still,
          "-vf", f"crop=iw*{hs}:ih*{hs}:iw*{hx}:ih*{hy},scale={STILL_W}:{STILL_H}:flags=lanczos", out])
    return out


def endframes(piece: Path, only: set | None = None, model: str | None = None, **_) -> None:
    """A painted LAST keyframe per shot, at endframes/sNNN.png, for H3's last_frame input.

    Measured 2026-10-03 (oceanus_directed): with the first frame only, H3 turned close-ups and
    newly revealed ground photographic inside the clip on a painted piece; s010, s017 and s035
    re-rendered with a pinned painted end frame held the look to the last frame. The still is
    cropped to where the camera ends and a Qwen-2.1 edit repaints the crop in the piece's look.
    Only pieces with a motion_style (drawn or painted) get them. Shots with an end frame on disk
    are skipped, so a stopped run resumes; delete a file to remake it.
    """
    compiled = json.loads((piece / "shots.json").read_text(encoding="utf-8"))
    out = piece / "endframes"
    out.mkdir(exist_ok=True)
    ff = _animate().find_ffmpeg()
    bible = _animate().style_bible(piece)   # the same look the H3 prompt asks for, when set
    work = piece / "_work" / f"endframes_{_stamp()}"
    work.mkdir(parents=True, exist_ok=True)
    todo = []
    for s in compiled:
        sid = f"s{s['i']:03d}"
        still = piece / "stills" / f"{sid}.png"
        if (only and sid not in only) or s.get("cuts") or (out / f"{sid}.png").exists() \
                or not still.exists():
            continue
        if s.get("end"):
            job = _designed_end(s, still, work, ff)
            if job:
                todo.append((sid, *job))
            continue
        if not s.get("motion_style"):
            continue
        crop = end_crop(s)
        if crop is None:
            continue
        look = (f"{bible['style']}. {bible['texture']}" if bible else
                (s.get("still_prompt") or s["motion_style"])).strip().rstrip(",. ")
        if crop == "out":
            c = work / f"{sid}_out.png"
            k = 0.72
            _run([ff, "-y", "-v", "error", "-i", still, "-vf",
                  f"scale=iw*{k}:ih*{k},pad={STILL_W}:{STILL_H}:(ow-iw)/2:(oh-ih)/2:color=0x808080", c])
            todo.append((sid, c, f"This painting sits in the middle of a flat grey border. Paint over the grey "
                                 "border by extending the scene outward on every side, as the same scene seen "
                                 f"from further back, as {look}, {ENDFRAME_LOOK}. Keep the middle exactly as "
                                 "it is. No grey left, no border, no picture frame, no lettering."))
            continue
        size, x0, y0 = crop
        if size >= 1.0:
            shutil.copy2(still, out / f"{sid}.png")
            continue
        c = work / f"{sid}_crop.png"
        _run([ff, "-y", "-v", "error", "-i", still,
              "-vf", f"crop=iw*{size}:ih*{size}:iw*{x0}:ih*{y0},scale={STILL_W}:{STILL_H}:flags=lanczos", c])
        todo.append((sid, c, f"Repaint this image as {look}, {ENDFRAME_LOOK}. Keep the composition, the "
                             "figures, their poses and the colours exactly as they are; only the surface "
                             "becomes painted. No border, no picture frame, no lettering."))
    if not todo:
        shutil.rmtree(work, ignore_errors=True)
        _say("endframes: nothing to paint")
        return
    if DRY:
        for sid, c, _p in todo:
            shutil.copy2(c, out / f"{sid}.png")
        shutil.rmtree(work, ignore_errors=True)
        return
    if str(paths.FARM) not in sys.path:
        sys.path.insert(0, str(paths.FARM))
    import graphs  # noqa: E402
    from dispatch import Farm, Job  # noqa: E402
    cfg = json.loads((piece / "config.json").read_text(encoding="utf-8"))
    base = int(cfg.get("seed", 31337))
    # seed: never the still's own (noise replay, see EDIT_RES); the old fixed seed=7 was one piece-wide
    jobs = [Job(job_id=f"{sid}_end", model_key="qwen21", out_name=f"{sid}_end", inputs=[c],
                graph=graphs.qwen21_edit(p, [c.name], steps=25, resolution=EDIT_RES,
                                         seed=edit_seed(base, 1000 + int(sid[1:])),
                                         prefix=f"{sid}_end_{int(time.time())}",
                                         variant=still_model(piece, model)))
            for sid, c, p in todo]
    _say(f"endframes: painting {len(jobs)}")
    farm("pair")
    f = Farm(work, job_timeout=1800)
    f.LOAD_MIN_FREE_GB = 8.0
    # without this every end frame waited the RAM gate's full 300 s for a 36 s render (gardening,
    # 7.4 GB free with the worker warm, 2026-10-11), the same trap stills() closed on 2026-09-24
    f.LOAD_STAGGER_S = 20.0
    _gate_first_load_only(f)
    designed = {s_["i"] for s_ in compiled if s_.get("end")}
    for d in f.run(jobs):
        if d.status == "ok" and d.files:
            sid = d.job_id[:-4]
            dest = out / f"{sid}.png"
            shutil.copy2(work / d.files[0], dest)
            _fit(dest)
            if int(sid[1:]) in designed:
                _match_colour(piece / "stills" / f"{sid}.png", dest)
                p = next(p for s_, _c, p in todo if s_ == sid)
                (out / f"{sid}.json").write_text(json.dumps({"prompt": p, "seconds": round(d.seconds, 1)},
                                                            indent=1, ensure_ascii=False), encoding="utf-8")
        else:
            _say(f"endframe {d.job_id} failed: {d.error}")
    shutil.rmtree(work, ignore_errors=True)


def _designed_end(s: dict, still: Path, work: Path, ff: str) -> tuple | None:
    """The shot's planned last frame (director_kb/12_prompt_endframe.md), as (input png, edit prompt).

    end: {"kind": "none" | "state" | "target" | "crop" | "state+crop", "change": str, "crop": [size, x0, y0]}
    A crop is cut first (where the camera ends) and the one edit then does the change and restores
    full detail, so a crop never reaches H3 soft. The preservation line names the medium: a painted
    piece (motion_style set) is "this painting", anything else "this photograph"."""
    e = s["end"]
    kind = e.get("kind", "state")
    if kind == "none":
        return None
    medium = "painting" if s.get("motion_style") else "photograph"
    sid = f"s{s['i']:03d}"
    c = work / f"{sid}_endin.png"
    crop = e.get("crop") or (end_crop(s) if "crop" in kind else None)
    if "crop" in kind and isinstance(crop, (list, tuple)) and crop[0] < 1.0:
        size, x0, y0 = crop
        _run([ff, "-y", "-v", "error", "-i", still,
              "-vf", f"crop=iw*{size}:ih*{size}:iw*{x0}:ih*{y0},scale={STILL_W}:{STILL_H}:flags=lanczos", c])
    else:
        shutil.copy2(still, c)
    change = " ".join((e.get("change") or "").split()).rstrip(". ")
    if change:
        p = (f"{change}. Keep everything else exactly as in this {medium}: the framing, the people's "
             "identity and clothing, the setting, the light and the colour.")
    else:
        p = (f"This {medium} is a crop of a larger frame; render it at full sharp detail as the same "
             f"{medium}. Change nothing: keep the framing, the people's identity, clothing and pose, the "
             "setting, the light and the colour exactly as they are.")
    return c, p


def _match_colour(ref: Path, img: Path) -> None:
    """Per-channel mean and spread of img moved to ref's in YCbCr, so the last frame carries the
    still's exposure and colour (the Oceanus v3 pulse tracked a luma/saturation gap, endframe_mismatch.py)."""
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(ref).convert("YCbCr"), dtype=np.float32)
    b = np.asarray(Image.open(img).convert("YCbCr"), dtype=np.float32)
    for ch in range(3):
        bs = b[..., ch].std() or 1.0
        b[..., ch] = (b[..., ch] - b[..., ch].mean()) / bs * a[..., ch].std() + a[..., ch].mean()
    Image.fromarray(np.clip(b, 0, 255).astype(np.uint8), "YCbCr").convert("RGB").save(img)


def assemble(piece: Path, **_) -> None:
    _run([paths.PODCAST_PY, paths.LIB / "assemble_film.py", piece], cwd=paths.LIB)


FUNCS = {"compile": compile_stage, "narrate": narrate, "clay": clay, "stills": stills,
         "endframes": endframes, "motion": motion, "assemble": assemble}
GPU_STAGES = {"narrate", "stills", "endframes", "motion"}


# -- one GPU user at a time ---------------------------------------------------------------------
# The studio's queue, a CLI run and an MCP-started job all take this before touching the farm or
# the TTS. It is a file holding the owner's pid, so a crashed owner does not hold it forever.

LOCK_FILE = paths.ROOT / ".gpu.lock"
_lock_state = {"depth": 0}
_lock_mutex = threading.Lock()


def _pid_alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))  # QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(h)
    return bool(ok) and code.value == 259  # STILL_ACTIVE


def gpu_lock_holder() -> dict | None:
    """Who holds the GPU lock, if it is a live process other than this one."""
    try:
        info = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    pid = info.get("pid")
    if not pid or pid == os.getpid() or not _pid_alive(pid):
        return None
    return info


@contextmanager
def gpu_lock(what: str):
    """Re-entrant within one process; waits (and says so) while another process holds it."""
    if DRY:
        yield
        return
    with _lock_mutex:
        mine = _lock_state["depth"] > 0
        if mine:
            _lock_state["depth"] += 1
    if not mine:
        told = False
        while True:
            try:
                fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, json.dumps({"pid": os.getpid(), "what": what,
                                         "at": datetime.now().isoformat(timespec="seconds")}).encode())
                os.close(fd)
                break
            except FileExistsError:
                if gpu_lock_holder() is None:
                    LOCK_FILE.unlink(missing_ok=True)
                    continue
                if not told:
                    _say(f"waiting for the GPU lock, held by {gpu_lock_holder()}")
                    told = True
                time.sleep(5)
        with _lock_mutex:
            _lock_state["depth"] = 1
    try:
        yield
    finally:
        with _lock_mutex:
            _lock_state["depth"] -= 1
            last = _lock_state["depth"] == 0
        if last:
            LOCK_FILE.unlink(missing_ok=True)


# -- per-shot operations ------------------------------------------------------------------------
# What the studio (webui/) queues, and what the CLI's `shot` command and the MCP tools call.
# Each one takes a shot as its position ("s003") or its stable id, reads plan.json at the moment
# it starts (so an edit made while the job was queued is used), and records its output as a new
# version of that shot (versions.py), which becomes the active one.

_piece_locks: dict = {}


def piece_lock(piece) -> threading.RLock:
    """Held while plan.json, the compiled stage files or a clay pass are written, so two threads
    working on one piece (the studio runs a GPU lane and a CPU lane) never compile over each other.
    Not held while the farm renders."""
    key = str(Path(piece).resolve()).lower()
    with _lock_mutex:
        return _piece_locks.setdefault(key, threading.RLock())


def load_source(piece: Path) -> dict:
    """The piece's brief or passage: source.json (the studio writes it), else a plan whose
    top-level `source` names a text file (adaptations/*/build_plan.py writes that), else {}."""
    f = Path(piece) / "source.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    try:
        plan = load_plan(piece)
    except (OSError, ValueError):
        return {}
    src = plan.get("source")
    if isinstance(src, str) and src.strip() and Path(src).is_file():
        return {"mode": "passage", "text": Path(src).read_text(encoding="utf-8"), "file": src,
                "title": plan.get("title", ""), "style": plan.get("style", ""),
                "location": plan.get("location", "")}
    return {}


def blocking_errors(plan: dict) -> list[str]:
    """Validation errors that stop compile_plan. Narration length and wording are left to the
    person: they matter to the film, not to the geometry."""
    return [e for e in schema.validate(plan)
            if ".narration:" not in e and not e.startswith("plan: narration is")]


def load_plan(piece: Path) -> dict:
    return json.loads((Path(piece) / "plan.json").read_text(encoding="utf-8"))


def save_plan(piece: Path, plan: dict) -> None:
    f = Path(piece) / "plan.json"
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(f)


def compile_piece(piece: Path) -> dict:
    plan = load_plan(piece)
    errs = blocking_errors(plan)
    if errs:
        raise ValueError("plan does not compile:\n- " + "\n- ".join(errs[:12]))
    schema.compile_plan(plan, Path(piece))
    return plan


def prepare(piece) -> tuple[Path, dict]:
    """Ids, versions.json and fresh stage files. The first call on a piece imports what is on disk."""
    from . import versions
    piece = piece_dir(str(piece))
    with piece_lock(piece):
        if not versions.path(piece).exists():
            versions.import_piece(piece)
        plan = load_plan(piece)
        if versions.ensure_ids(plan):
            save_plan(piece, plan)
        return piece, compile_piece(piece)


def resolve(plan: dict, ref) -> tuple[int, dict]:
    shots = plan["shots"]
    if isinstance(ref, int):
        return ref, shots[ref]
    ref = str(ref)
    if ref[:1] == "s" and ref[1:].isdigit() and int(ref[1:]) < len(shots):
        return int(ref[1:]), shots[int(ref[1:])]
    for i, s in enumerate(shots):
        if s.get("id") == ref:
            return i, s
    raise KeyError(f"no shot {ref!r}")


def _stamp() -> str:
    return f"{int(time.time() * 1000) % 10**9:09d}"


def ensure_clay(piece: Path, plan: dict, i: int, force: bool = False) -> dict | None:
    """The shot's clay pass, rendered again only if its geometry changed since (or force)."""
    from . import versions
    s = plan["shots"][i]
    if schema.shot_mode(s) == "free":
        return None
    key = versions.clay_key(plan, s)
    have = (versions.load(piece)["shots"].get(s["id"]) or {}).get("clay")
    if not force and have and have["key"] == key and (piece / have["file"]).exists():
        return have
    sid = f"s{i:03d}"
    with piece_lock(piece):
        clay(piece, only={sid})
        _mark(piece, "clay", "done")
        return versions.snapshot_clay(piece, s["id"], sid, key)


def clay_shots(piece, refs: list, force: bool = True) -> list[dict]:
    piece = piece_dir(str(piece))
    with piece_lock(piece):
        piece, plan = prepare(piece)
        out = []
        for ref in refs:
            i, s = resolve(plan, ref)
            if schema.shot_mode(s) == "free":
                _say(f"s{i:03d} is a free shot: no clay pass")
                continue
            out.append({"sid": f"s{i:03d}", "id": s["id"], **ensure_clay(piece, plan, i, force=force)})
        return out


EDIT_TAIL = ("Keep everything else exactly as in the input photograph: the people and their identity, "
             "clothing and pose, the camera position and framing, the setting, the light and the colour.")


def edit_prompt(instruction: str) -> str:
    """An instruction for a Qwen-2.1 edit of a finished still: the operation first, then one
    blanket preservation clause (QWEN_IMAGE_2.1_PROMPTING.md section 4), on one line."""
    ins = " ".join((instruction or "").split()).rstrip(". ")
    if not ins:
        raise ValueError("an edit needs an instruction")
    return f"{ins[0].upper()}{ins[1:]}. {EDIT_TAIL}"


def render_stills(piece, specs: list[dict], on_start=None, strict: bool = True,
                  model: str | None = None) -> list[dict]:
    """Render several stills in one farm batch (the pair runs two at once).

    spec: {"ref": "s003" or id, "kind": "redo" | "edit", "seed": int|None, "prompt": str|None,
           "base_v": int|None (edit only: the version to edit, default the active one)}
    redo on a clay shot re-uses the clay pass (re-rendering it first if the geometry changed);
    redo on a free shot is text-to-image; edit runs a Qwen-2.1 edit on an existing still.
    Each returned item carries k, its spec's index. strict=False returns failed specs as
    {"k", "sid", "id", "error"} items instead of raising. model: see still_model()."""
    piece = piece_dir(str(piece))
    with piece_lock(piece):
        todo, work = _still_todo(piece, specs)
    return _render_todo(piece, todo, work, on_start, strict, still_model(piece, model))


def _direction_gate(piece: Path, cfg: dict) -> None:
    """A directed piece renders no stills until its animatic has been watched and signed off."""
    if not cfg.get("direction_hash"):
        return
    from . import direct
    why = direct.animatic_gate(piece, cfg["direction_hash"])
    if why:
        raise RuntimeError(why)


def _stage_refs(refs: list, into: Path, name: str) -> list[Path]:
    """Copy reference images under names unique to this job, so the farm's input dir never
    serves one shot's location plate to another."""
    into.mkdir(parents=True, exist_ok=True)
    out = []
    for k, r in enumerate(refs):
        src = Path(r)
        dest = into / f"{name}_r{k}{src.suffix.lower() or '.png'}"
        shutil.copy2(src, dest)
        out.append(dest)
    return out


def _still_todo(piece: Path, specs: list[dict]) -> tuple[list[dict], Path]:
    from . import versions
    piece, plan = prepare(piece)
    if plan.get("direction_hash"):
        _direction_gate(piece, {"direction_hash": plan["direction_hash"]})
    base_seed = int(plan.get("seed", 31337))
    work = piece / "_work" / f"stills_{_stamp()}"
    work.mkdir(parents=True, exist_ok=True)
    todo = []
    for k, sp in enumerate(specs):
        i, s = resolve(plan, sp["ref"])
        uid, sid = s["id"], f"s{i:03d}"
        seed = base_seed if sp.get("seed") in (None, "") else int(sp["seed"])
        name = f"{uid}_{_stamp()}_{k}"
        t = {"k": k, "i": i, "sid": sid, "uid": uid, "name": name, "seed": seed, "inputs": []}
        if sp.get("kind") == "edit":
            row = versions.load(piece)["shots"].get(uid) or {"stills": [], "active_still": None}
            want = sp.get("base_v") or row.get("active_still")
            base = next((x for x in row["stills"] if x["v"] == want), None)
            if not base:
                raise ValueError(f"{sid} has no still to edit")
            src = work / f"{name}_in.png"
            shutil.copy2(piece / base["file"], src)
            t.update(kind="edit", prompt=edit_prompt(sp.get("prompt", "")), src=src, inputs=[src],
                     meta={"instruction": sp.get("prompt", ""), "parent": base["v"],
                           "shot_key": base.get("shot_key")})
        elif schema.shot_mode(s) == "free" and s.get("refs") and not sp.get("prompt") and False:
            # Disabled 2026-10-10: this one-pass refs edit puts the location plate in image_1, the
            # edit canvas, so the shot comes back as the empty plate. Studio redos of reference shots
            # fall through to text-to-image; the batch stills() stage does the two-pass version.
            inputs = _stage_refs(s["refs"], work, name)
            t.update(kind="refs", prompt=s["ref_prompt"], src=inputs[0], inputs=inputs,
                     meta={"shot_key": versions.still_key(plan, s)})
        elif schema.shot_mode(s) == "free":
            t.update(kind="free", prompt=sp.get("prompt") or schema.free_prompt(plan, s), src=None,
                     meta={"shot_key": versions.still_key(plan, s)})
        else:
            c = ensure_clay(piece, plan, i)
            src = work / f"{name}_clay.png"
            shutil.copy2(piece / c["file"], src)
            prompt = (sp.get("prompt") or s.get("still_prompt")
                      or (piece / c["prompt_file"]).read_text(encoding="utf-8"))
            t.update(kind="clay", prompt=prompt, src=src, inputs=[src],
                     meta={"shot_key": versions.still_key(plan, s), "clay_at": c["at"]})
        todo.append(t)
    return todo, work


def _render_todo(piece: Path, todo: list[dict], work: Path, on_start=None, strict: bool = True,
                 variant: str = "base") -> list[dict]:
    from . import versions
    if on_start:
        on_start([{"sid": t["sid"], "id": t["uid"], "kind": t["kind"], "seed": t["seed"],
                   "prompt": t["prompt"]} for t in todo])
    results, errors = {}, {}
    if DRY:
        from . import dry
        for t in todo:
            out = work / f"{t['name']}.png"
            dry.still(t["kind"], out, t["i"], t["seed"], t["prompt"], src=t["src"] if t["kind"] == "edit" else None,
                      log=_say)
            results[t["name"]] = out
    else:
        sys.path.insert(0, str(paths.FARM))
        import graphs  # noqa: E402
        from dispatch import Farm, Job  # noqa: E402
        jobs = []
        for t in todo:
            if t["kind"] == "free":
                g = graphs.qwen21_t2i(t["prompt"], width=STILL_W, height=STILL_H, steps=T2I_STEPS,
                                      seed=t["seed"], prefix=t["name"], variant=variant)
            elif t["kind"] == "refs":
                g = graphs.qwen21_edit(t["prompt"], [x.name for x in t["inputs"]], steps=T2I_STEPS,
                                       resolution=EDIT_RES, seed=edit_seed(t["seed"], t["i"]),
                                       prefix=t["name"], variant=variant)
            else:
                # an edit of an existing still or a clay pass: a seed of its own (noise replay, EDIT_RES)
                g = graphs.qwen21_edit(t["prompt"], [t["src"].name], steps=STILL_STEPS, resolution=EDIT_RES,
                                       seed=edit_seed(t["seed"], t["i"]) if t["kind"] == "edit" else t["seed"],
                                       prefix=t["name"], variant=variant)
            jobs.append(Job(job_id=t["name"], graph=g, model_key="qwen21", out_name=t["name"],
                            inputs=t["inputs"], meta={"sid": t["sid"], "kind": t["kind"]}))
        with gpu_lock(f"stills {piece.name}"):
            farm("pair")
            import dispatch  # noqa: E402  (already imported above; its C is farm/config.py)
            farm_config = dispatch.C
            # One job needs one card. With both workers the RAM gate still applies, and after a
            # batch the two warm workers keep free RAM under 8 GB, so a lone studio edit waited
            # the gate's full 300 s before a 59 s render (2026-09-24).
            workers = farm_config.WORKERS[:1] if len(jobs) == 1 else None
            f = Farm(work, workers=workers, job_timeout=1800)
            f.LOAD_MIN_FREE_GB = 8.0   # see stills(): the 18 GB gate is sized for a 32B encoder
            f.LOAD_STAGGER_S = 20.0
            _gate_first_load_only(f)
            sink = getattr(_TLS, "sink", None)
            if sink:   # the farm's worker threads have no sink of their own
                f._say = lambda m, _s=sink: (print(f"[farm] {m}", flush=True), _s(m))
            for d in f.run(jobs):
                if d.status == "ok" and d.files:
                    results[d.job_id] = work / d.files[0]
                    _fit(results[d.job_id])   # edits come back at EDIT_RES's size
                else:
                    errors[d.job_id] = d.error or "no output"
    items, failed = [], []
    for t in todo:
        f = results.get(t["name"])
        if not f or not f.exists():
            err = errors.get(t["name"], "no output")[:300]
            failed.append(f"{t['sid']}: {err}")
            items.append({"k": t["k"], "sid": t["sid"], "id": t["uid"], "error": err})
            continue
        meta = {"kind": t["kind"], "seed": t["seed"], "prompt": t["prompt"],
                "backend": "dry" if DRY else "farm", "model": variant, **t["meta"]}
        item = versions.add_still(piece, t["uid"], f, meta)
        items.append({"k": t["k"], "sid": t["sid"], "id": t["uid"], **item})
        _say(f"{t['sid']} still v{item['v']} ({t['kind']}, seed {t['seed']})")
    shutil.rmtree(work, ignore_errors=True)
    if failed and strict:
        raise RuntimeError("stills failed: " + "; ".join(failed))
    return items


def still_one(piece, sid, seed=None, prompt=None, on_start=None, model=None) -> dict:
    """A new still for one shot from its clay pass (clay mode) or from text (free mode)."""
    return render_stills(piece, [{"ref": sid, "kind": "redo", "seed": seed, "prompt": prompt}],
                         on_start=on_start, model=model)[0]


def edit_still(piece, sid, prompt, seed=None, base_v=None, on_start=None, model=None) -> dict:
    """A Qwen-2.1 edit of one shot's active still (or version base_v) by a plain instruction."""
    return render_stills(piece, [{"ref": sid, "kind": "edit", "seed": seed, "prompt": prompt,
                                  "base_v": base_v}], on_start=on_start, model=model)[0]


def _animate():
    if str(paths.LIB) not in sys.path:
        sys.path.insert(0, str(paths.LIB))
    import animate_shots  # noqa: E402  (Agentic_Slop _lib; puts farm/ on sys.path itself)
    return animate_shots


def compiled_shot(piece: Path, i: int) -> dict:
    shots = json.loads((Path(piece) / "shots.json").read_text(encoding="utf-8"))
    return next(s for s in shots if s["i"] == i)


def auto_motion_prompt(compiled: dict) -> str:
    """What would be sent H3 for this shot, ignoring any hand-written override: the director's motion
    plan through the translator when the shot has one, else what animate_shots builds."""
    if compiled.get("motion_plan"):
        from .motion_direction import compile_prompt
        return compile_prompt(compiled)[0]
    return _animate().prompt_for({k: v for k, v in compiled.items() if k != "motion_prompt"})


def shot_seconds(piece: Path, i: int, narration: str, tail_s: float = 0.0) -> float:
    """The narration's real length if timeline.json has this shot's words, else ~2.4 words/s,
    plus the shot's tail_s (silence held after the narration) when the timeline lacks it."""
    f = Path(piece) / "timeline.json"
    if f.exists():
        tl = json.loads(f.read_text(encoding="utf-8"))
        sc = next((x for x in tl.get("scenes", []) if x.get("index") == i), None)
        if sc and " ".join(sc.get("narration", "").split()) == " ".join(narration.split()):
            return float(sc["duration"])
    return max(schema.est_seconds(narration), 1.0) + float(tail_s or 0)


def motion_one(piece, sid, prompt_override: str | None = None, seed=None, on_start=None) -> dict:
    """One H3 clip from the shot's active still. The prompt is, in order: prompt_override, the
    plan's motion_prompt, what animate_shots builds. It is read when the job starts.
    The body lives in motion_batch.py (2026-10-11), split into prepare / run / register helpers that
    the remote batch path shares. With FARM_REMOTE_WORKERS unset it runs exactly as before."""
    from . import motion_batch
    return motion_batch.motion_one(piece, sid, prompt_override, seed, on_start)


def narrate_piece(piece) -> None:
    from . import versions
    piece, _ = prepare(piece)
    versions.materialize(piece)
    with gpu_lock(f"narrate {piece.name}"):
        narrate(piece)
    _mark(piece, "narrate", "done")


def assemble_piece(piece) -> Path:
    from . import versions
    piece, _ = prepare(piece)
    versions.materialize(piece)
    assemble(piece)
    _mark(piece, "assemble", "done")
    return piece / f"{piece.name}.mp4"


def _remote_motion() -> bool:
    """Motion on rented pods (FARM_REMOTE_WORKERS, farm/config.py). It takes no local GPU lock and
    leaves the local farm alone, so a stills queue on GPU1 keeps running."""
    return bool(os.environ.get("FARM_REMOTE_WORKERS", "").strip()) and not DRY


def run(piece: Path, stages: list[str] | None = None, farm_after: str = "off", **kw) -> None:
    stages = stages or STAGES
    for st in stages:
        if st not in FUNCS:
            raise ValueError(f"unknown stage {st}; stages are {STAGES}")
    t0 = time.time()
    try:
        for st in stages:
            _say(f"=== {st} ===")
            _mark(piece, st, "running")
            t = time.time()
            if st in GPU_STAGES and not (st == "motion" and _remote_motion()):
                with gpu_lock(f"run {piece.name} {st}"):
                    FUNCS[st](piece, **kw)
            else:
                FUNCS[st](piece, **kw)
            _mark(piece, st, "done", seconds=round(time.time() - t, 1))
    except Exception as e:
        _mark(piece, st, "failed", error=str(e)[-1500:])
        raise
    finally:
        local = ("stills", "endframes") + (() if _remote_motion() else ("motion",))
        if farm_after == "off" and any(s in stages for s in local):
            farm("off")
    _say(f"run finished in {time.time() - t0:.0f} s")
