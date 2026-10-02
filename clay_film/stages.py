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
import json
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

STAGES = ["compile", "narrate", "clay", "stills", "motion", "assemble"]
STILL_STEPS = 18
T2I_STEPS = 25            # free shots start from noise, not from a clay pass
STILL_W, STILL_H = 1344, 768
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


def stills(piece: Path, only: set | None = None, seeds: list[int] | None = None,
           force: bool = False, **_) -> None:
    from PIL import Image
    sys.path.insert(0, str(paths.FARM))
    import graphs  # noqa: E402  (farm modules)
    from dispatch import Farm, Job  # noqa: E402
    cfg = json.loads((piece / "config.json").read_text(encoding="utf-8"))
    seeds = seeds or [cfg.get("seed", 31337)]
    blk, raw = piece / "blockout", piece / "stills_raw"
    final = piece / ("stills" if len(seeds) == 1 else "stills_cand")
    final.mkdir(parents=True, exist_ok=True)
    compiled = {f"s{s['i']:03d}": s for s in
                json.loads((piece / "shots.json").read_text(encoding="utf-8"))}
    jobs = []
    for clay_png in sorted(blk.glob("s*_clay.png")):
        sid = clay_png.name[:4]
        if only and sid not in only:
            continue
        if compiled.get(sid, {}).get("mode") == "free":
            continue    # a stale clay pass from before the shot became free
        # a hand-written still_prompt in the plan replaces the one _filmprep.ts wrote
        prompt = (compiled.get(sid, {}).get("still_prompt")
                  or (blk / f"{sid}_prompt.txt").read_text(encoding="utf-8"))
        newest_in = max(clay_png.stat().st_mtime, (blk / f"{sid}_prompt.txt").stat().st_mtime)
        for seed in seeds:
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
            g = graphs.qwen21_edit(prompt, [clay_png.name], steps=STILL_STEPS, resolution=0,
                                   seed=seed, prefix=f"{name}_{int(time.time())}")
            jobs.append(Job(job_id=name, graph=g, model_key="qwen21", out_name=name,
                            meta={"sid": sid, "seed": seed}, inputs=[clay_png]))
    for sid, s in compiled.items():
        # free shots: text-to-image from the compiled prompt (style clause + still_prompt + cast)
        if s.get("mode") != "free" or (only and sid not in only):
            continue
        for seed in seeds:
            name = sid if len(seeds) == 1 else f"{sid}_{seed}"
            if not force and (final / f"{name}.png").exists():
                continue
            g = graphs.qwen21_t2i(s["still_prompt"], width=STILL_W, height=STILL_H, steps=T2I_STEPS,
                                  seed=seed, prefix=f"{name}_{int(time.time())}")
            jobs.append(Job(job_id=name, graph=g, model_key="qwen21", out_name=name,
                            meta={"sid": sid, "seed": seed, "mode": "free"}))
    if jobs and DRY:
        from . import dry
        raw.mkdir(parents=True, exist_ok=True)
        for j in jobs:
            dry.still("free" if j.meta.get("mode") == "free" else "clay", raw / f"{j.out_name}.png",
                      int(j.meta["sid"][1:]), j.meta["seed"], "", log=_say)
        jobs = []
    if jobs:
        farm("pair")
        f = Farm(raw, job_timeout=1800)
        # The farm's 18 GB RAM gate is sized for the Qwen3-VL-32B encoder. Qwen-2.1 reads the 8B
        # w4a8 one; with 18 GB the first run waited out the full 300 s gate on a box at 9.8 GB free.
        f.LOAD_MIN_FREE_GB = 8.0
        # The 90 s stagger is held on every job, which serialises the pair. Only the first job per
        # worker loads weights here; later ones hit the cache and sample in ~30 s.
        f.LOAD_STAGGER_S = 20.0
        _gate_first_load_only(f)
        done = f.run(jobs)
        bad = [d.job_id for d in done if d.status != "ok"]
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
    compiled = json.loads((piece / "shots.json").read_text(encoding="utf-8"))
    if DRY or any(s.get("motion_prompt") for s in compiled):
        # a hand-edited H3 prompt (or the dry backend): shot by shot through motion_one, which
        # reads the override. Without either, animate_shots runs unchanged.
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
    _run(cmd, cwd=paths.LIB)
    _report_cuts(piece, only)


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


def assemble(piece: Path, **_) -> None:
    _run([paths.PODCAST_PY, paths.LIB / "assemble_film.py", piece], cwd=paths.LIB)


FUNCS = {"compile": compile_stage, "narrate": narrate, "clay": clay, "stills": stills,
         "motion": motion, "assemble": assemble}
GPU_STAGES = {"narrate", "stills", "motion"}


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


def render_stills(piece, specs: list[dict], on_start=None, strict: bool = True) -> list[dict]:
    """Render several stills in one farm batch (the pair runs two at once).

    spec: {"ref": "s003" or id, "kind": "redo" | "edit", "seed": int|None, "prompt": str|None,
           "base_v": int|None (edit only: the version to edit, default the active one)}
    redo on a clay shot re-uses the clay pass (re-rendering it first if the geometry changed);
    redo on a free shot is text-to-image; edit runs a Qwen-2.1 edit on an existing still.
    Each returned item carries k, its spec's index. strict=False returns failed specs as
    {"k", "sid", "id", "error"} items instead of raising."""
    piece = piece_dir(str(piece))
    with piece_lock(piece):
        todo, work = _still_todo(piece, specs)
    return _render_todo(piece, todo, work, on_start, strict)


def _still_todo(piece: Path, specs: list[dict]) -> tuple[list[dict], Path]:
    from . import versions
    piece, plan = prepare(piece)
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


def _render_todo(piece: Path, todo: list[dict], work: Path, on_start=None, strict: bool = True) -> list[dict]:
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
                                      seed=t["seed"], prefix=t["name"])
            else:
                g = graphs.qwen21_edit(t["prompt"], [t["src"].name], steps=STILL_STEPS, resolution=0,
                                       seed=t["seed"], prefix=t["name"])
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
                "backend": "dry" if DRY else "farm", **t["meta"]}
        item = versions.add_still(piece, t["uid"], f, meta)
        items.append({"k": t["k"], "sid": t["sid"], "id": t["uid"], **item})
        _say(f"{t['sid']} still v{item['v']} ({t['kind']}, seed {t['seed']})")
    shutil.rmtree(work, ignore_errors=True)
    if failed and strict:
        raise RuntimeError("stills failed: " + "; ".join(failed))
    return items


def still_one(piece, sid, seed=None, prompt=None, on_start=None) -> dict:
    """A new still for one shot from its clay pass (clay mode) or from text (free mode)."""
    return render_stills(piece, [{"ref": sid, "kind": "redo", "seed": seed, "prompt": prompt}],
                         on_start=on_start)[0]


def edit_still(piece, sid, prompt, seed=None, base_v=None, on_start=None) -> dict:
    """A Qwen-2.1 edit of one shot's active still (or version base_v) by a plain instruction."""
    return render_stills(piece, [{"ref": sid, "kind": "edit", "seed": seed, "prompt": prompt,
                                  "base_v": base_v}], on_start=on_start)[0]


def _animate():
    if str(paths.LIB) not in sys.path:
        sys.path.insert(0, str(paths.LIB))
    import animate_shots  # noqa: E402  (Agentic_Slop _lib; puts farm/ on sys.path itself)
    return animate_shots


def compiled_shot(piece: Path, i: int) -> dict:
    shots = json.loads((Path(piece) / "shots.json").read_text(encoding="utf-8"))
    return next(s for s in shots if s["i"] == i)


def auto_motion_prompt(compiled: dict) -> str:
    """What animate_shots would send H3 for this shot, ignoring any hand-written override."""
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
    plan's motion_prompt, what animate_shots builds. It is read when the job starts."""
    from . import versions
    piece = piece_dir(str(piece))
    with piece_lock(piece):
        piece, plan = prepare(piece)
        i, s = resolve(plan, sid)
        comp = compiled_shot(piece, i)
    uid, sid = s["id"], f"s{i:03d}"
    row = versions.load(piece)["shots"].get(uid) or {}
    st = versions.active(row, "still") if row else None
    if not st:
        raise ValueError(f"{sid} has no active still; make one first")
    an = _animate()
    prompt = prompt_override or comp.get("motion_prompt") or an.prompt_for(comp)
    secs = shot_seconds(piece, i, s.get("narration", ""), s.get("tail_s", 0))
    n = an.frames_for(secs, comp)
    seed = an.SEED + i if seed in (None, "") else int(seed)
    if on_start:
        on_start([{"sid": sid, "id": uid, "kind": "motion", "seed": seed, "prompt": prompt,
                   "still_v": st["v"]}])
    work = piece / "_work" / f"motion_{_stamp()}"
    work.mkdir(parents=True, exist_ok=True)
    name = f"{uid}_{_stamp()}"
    src = work / f"{name}_still.png"
    shutil.copy2(piece / st["file"], src)
    meta = {"kind": "h3", "seed": seed, "prompt": prompt, "frames": n, "seconds": round(n / an.FPS, 2),
            "want": round(secs, 2), "still_ref": st["v"],
            "prompt_overridden": bool(prompt_override or comp.get("motion_prompt")),
            "backend": "dry" if DRY else "farm"}
    if DRY:
        from . import dry
        out = work / f"{name}.mp4"
        dry.clip(out, i, log=_say)
    else:
        import config as C  # noqa: E402  (farm/, on sys.path via animate_shots)
        from dispatch import Farm, Job  # noqa: E402
        g = an.graphs.minimax_h3_official(
            prompt, src.name, width=an.W, height=an.H, seconds=n / an.FPS, fps=an.FPS,
            steps=an.STEPS, seed=seed, sparse={"tau": 1.3, "start_percent": 0.2},
            prefix=f"{piece.name}_{name}")
        _say(f"{sid}: H3 {n} frames, seed {seed}")
        with gpu_lock(f"motion {piece.name} {sid}"):
            farm("big")
            done = Farm(work, workers=[C.BIG_WORKER], job_timeout=5400).run(
                [Job(job_id=name, model_key="minimax_h3", out_name=name, inputs=[src], graph=g)])
        if not done or done[0].status != "ok":
            shutil.rmtree(work, ignore_errors=True)
            raise RuntimeError(f"motion failed: {done[0].error if done else 'no result'}")
        out = next(iter(sorted(work.glob(f"{name}*.mp4"))), None)
        if out is None:
            raise RuntimeError("motion finished but wrote no mp4")
        gate_dir = work / "gate"
        gate_dir.mkdir()
        shutil.copy2(out, gate_dir / f"{sid}.mp4")
        try:
            meta["frozen"] = bool(an.gate(gate_dir))
        except Exception as e:  # noqa: BLE001  the gate is a report, not a requirement
            _say(f"motion gate skipped: {e}")
    item = versions.add_clip(piece, uid, out, meta)
    shutil.rmtree(work, ignore_errors=True)
    _say(f"{sid} clip v{item['v']}")
    return {"sid": sid, "id": uid, **item}


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
            if st in GPU_STAGES:
                with gpu_lock(f"run {piece.name} {st}"):
                    FUNCS[st](piece, **kw)
            else:
                FUNCS[st](piece, **kw)
            _mark(piece, st, "done", seconds=round(time.time() - t, 1))
    except Exception as e:
        _mark(piece, st, "failed", error=str(e)[-1500:])
        raise
    finally:
        if farm_after == "off" and any(s in stages for s in ("stills", "motion")):
            farm("off")
    _say(f"run finished in {time.time() - t0:.0f} s")
