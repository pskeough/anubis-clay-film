"""Clay Film Studio: a local web UI over clay_film, for working a film shot by shot.

    python webui/server.py [--port 8761] [--dry] [--open]

--dry fakes every GPU job (clay_film/dry.py) so the whole flow can be tried without the farm.
The server binds 127.0.0.1 only. See webui/README.md.

Jobs go into a queue persisted per piece at <piece>/_webui/queue.json, so a restart loses
nothing (a job that was running when the server stopped is queued again). Two lanes run:

    gpu   still / edit (farm pair), motion (farm big), narrate (farm off). One at a time; the
          next job is the oldest one that suits the farm as it is now, then the oldest overall.
          Two queued still or edit jobs of one piece run as one farm batch (the pair renders two).
    cpu   clay, assemble, plan (an LLM call). Beside the GPU lane.

A GPU job waits while clay_film/.gpu.lock is held by another process, while the farm is busy
with work this server did not send, or while the farm is down and a card has memory in use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from collections import deque
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402
from fastapi import Body, FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from clay_film import cast as castmod  # noqa: E402
from clay_film import locations, paths, planner, schema, stages, versions  # noqa: E402

PORT = 8761
CONFIG = HERE / "config.local.json"
STATIC = HERE / "static"
PROTECTED = {"basilisk_day2_v2", "basilisk_day2_demo"}   # finished pieces: copy, never write
NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_\-]{0,63}$")
FARM_IDLE_S = 600          # stop a farm this server started after this long with no GPU work
BUSY_MIB = 3500            # a card using more than this with the farm down is someone else's
EST = {"still": 45, "edit": 40, "motion": 330, "narrate": 40, "clay": 12, "assemble": 60, "plan": 120}
SWITCH_S = {"pair": 120, "big": 150, "off": 15}
FARM_OF = {"still": "pair", "edit": "pair", "motion": "big", "narrate": "off"}
GPU_KINDS = set(FARM_OF)
CPU_KINDS = {"clay", "assemble", "plan"}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# -- config (the planner endpoint and its key) ------------------------------------------------

def cfg_load() -> dict:
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def cfg_save(c: dict) -> None:
    tmp = CONFIG.with_suffix(".tmp")
    tmp.write_text(json.dumps(c, indent=1), encoding="utf-8")
    tmp.replace(CONFIG)


def api_key(endpoint: str) -> str | None:
    c = cfg_load()
    if c.get("api_key"):
        return c["api_key"]
    return planner._key(endpoint, None)   # CLAY_FILM_API_KEY, or OPENROUTER_API_KEY for openrouter


def cfg_public() -> dict:
    c = cfg_load()
    ep = c.get("endpoint") or "http://127.0.0.1:8080/v1"
    src = "config" if c.get("api_key") else (
        "env CLAY_FILM_API_KEY" if os.environ.get("CLAY_FILM_API_KEY") else
        "env OPENROUTER_API_KEY" if "openrouter" in ep and os.environ.get("OPENROUTER_API_KEY") else None)
    return {"endpoint": ep, "model": c.get("model") or "local", "has_key": bool(src), "key_source": src}


# -- pieces -----------------------------------------------------------------------------------

def piece_path(name: str, must_exist: bool = True) -> Path:
    if not NAME_RE.match(name or ""):
        raise HTTPException(400, f"bad piece name {name!r}: letters, digits, _ and - only")
    p = paths.PIECES / name
    if must_exist and not p.is_dir():
        raise HTTPException(404, f"no piece {name}")
    return p


def writable(name: str) -> None:
    if name in PROTECTED:
        raise HTTPException(403, f"{name} is a finished piece and is read only here; copy it to work on it")


def imported(piece: Path) -> bool:
    return versions.path(piece).exists()


def need_imported(name: str) -> Path:
    writable(name)
    p = piece_path(name)
    if not imported(p):
        raise HTTPException(409, f"{name} is not imported into the studio yet; press Import first")
    return p


def webdir(piece: Path) -> Path:
    d = piece / "_webui"
    d.mkdir(exist_ok=True)
    return d


def open_piece(piece: Path) -> dict:
    """Import on first use: ids, versions.json, and a compile when the plan allows one."""
    with stages.piece_lock(piece):
        if not imported(piece):
            versions.import_piece(piece)
        plan = stages.load_plan(piece)
        if versions.ensure_ids(plan):
            stages.save_plan(piece, plan)
        if not stages.blocking_errors(plan):
            schema.compile_plan(plan, piece)
            versions.materialize(piece)
        return plan


# -- the queue --------------------------------------------------------------------------------

def _qfile(piece: Path) -> Path:
    return piece / "_webui" / "queue.json"


def _log_file(piece: Path, jid: str) -> Path:
    d = piece / "_webui" / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{jid}.log"


def _ports_up() -> dict:
    return {p: stages._up(p) for p in (*stages.PAIR_PORTS, stages.BIG_PORT)}


def _comfy_busy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/queue", timeout=3) as r:
            q = json.loads(r.read().decode("utf-8"))
        return bool(q.get("queue_running") or q.get("queue_pending"))
    except (OSError, ValueError):
        return False


def _gpu_used() -> list[int]:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15)
        return [int(x) for x in r.stdout.split() if x.strip().isdigit()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


class Queue:
    def __init__(self, dry: bool):
        self.dry = dry
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.running: dict[str, list] = {"gpu": [], "cpu": []}   # [(piece, job id)]
        self.logs: dict[str, deque] = {}
        self.block = None             # why the GPU lane is waiting, for the UI
        self.block_checked = 0.0
        self.farm_started_by_us = False
        self.gpu_idle_since = time.time()
        self.farm_state = "off"
        self.stop = False

    # persistence
    def load(self, piece: Path) -> list[dict]:
        try:
            return json.loads(_qfile(piece).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def save(self, piece: Path, jobs: list[dict]) -> None:
        f = _qfile(piece)
        f.parent.mkdir(exist_ok=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(jobs, indent=1, ensure_ascii=False), encoding="utf-8")
        # Windows refuses the swap while any other handle has queue.json open (a lock-free
        # read in loop() or /api/queue, an indexer, antivirus). That is brief, so retry.
        for attempt in range(40):
            try:
                tmp.replace(f)
                return
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(0.05)

    def pieces(self) -> list[Path]:
        return sorted(p.parent.parent for p in paths.PIECES.glob("*/_webui/queue.json"))

    def update(self, piece: Path, jid: str, **fields) -> dict | None:
        with self.lock:
            jobs = self.load(piece)
            for j in jobs:
                if j["id"] == jid:
                    j.update(fields)
                    self.save(piece, jobs)
                    return j
        return None

    def recover(self) -> None:
        """A job that was running when the server stopped goes back in the queue."""
        with self.lock:
            for piece in self.pieces():
                jobs = self.load(piece)
                touched = False
                for j in jobs:
                    if j["state"] == "running":
                        j.update(state="queued", started=None, note="queued again after a server restart")
                        touched = True
                if touched:
                    self.save(piece, jobs)

    # adding and removing
    def add(self, piece: Path, kind: str, shot: dict | None, params: dict) -> dict:
        lane = "gpu" if kind in GPU_KINDS else "cpu"
        jid = f"j{int(time.time() * 1000) % 10**10:010d}{os.urandom(2).hex()}"
        job = {"id": jid, "piece": piece.name, "kind": kind, "lane": lane, "state": "queued",
               "shot_id": (shot or {}).get("id"), "sid": (shot or {}).get("sid"),
               "params": params, "created": now(), "started": None, "finished": None,
               "est_s": EST.get(kind, 60), "error": None, "result": None, "used": None, "note": None,
               "dry": self.dry}
        with self.lock:
            jobs = self.load(piece)
            jobs.append(job)
            self.save(piece, jobs)
        self.wake.set()
        return job

    def cancel(self, piece: Path, jid: str) -> dict:
        with self.lock:
            jobs = self.load(piece)
            j = next((x for x in jobs if x["id"] == jid), None)
            if not j:
                raise HTTPException(404, "no such job")
            if j["state"] == "running":
                raise HTTPException(409, "this job is running and cannot be stopped from the studio; "
                                         "it will finish on its own")
            if j["state"] != "queued":
                raise HTTPException(409, f"job is {j['state']}")
            j.update(state="cancelled", finished=now())
            self.save(piece, jobs)
            return j

    def edit(self, piece: Path, jid: str, params: dict) -> dict:
        with self.lock:
            jobs = self.load(piece)
            j = next((x for x in jobs if x["id"] == jid), None)
            if not j:
                raise HTTPException(404, "no such job")
            if j["state"] == "running":
                raise HTTPException(409, "this job is running; it read its prompt when it started, so a "
                                         "change now would not reach it. Queue another render instead.")
            if j["state"] != "queued":
                raise HTTPException(409, f"job is {j['state']}")
            for k in ("prompt", "seed"):
                if k in params:
                    j["params"][k] = params[k]
            self.save(piece, jobs)
            return j

    def clear_finished(self, piece: Path) -> int:
        with self.lock:
            jobs = self.load(piece)
            keep = [j for j in jobs if j["state"] in ("queued", "running")]
            self.save(piece, keep)
            return len(jobs) - len(keep)

    # logs
    def log(self, jid: str, piece: Path, msg: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        self.logs.setdefault(jid, deque(maxlen=400)).append(line)
        try:
            with open(_log_file(piece, jid), "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def log_tail(self, piece: Path, jid: str, n: int = 200) -> list[str]:
        if jid in self.logs:
            return list(self.logs[jid])[-n:]
        f = piece / "_webui" / "logs" / f"{jid}.log"
        return f.read_text(encoding="utf-8", errors="replace").splitlines()[-n:] if f.exists() else []

    # the farm
    def farm_mode(self) -> str:
        if self.dry:
            from clay_film import dry
            return dry.farm_mode()
        up = _ports_up()
        if up[stages.BIG_PORT]:
            return "big"
        if all(up[p] for p in stages.PAIR_PORTS):
            return "pair"
        return "partial" if any(up.values()) else "off"

    def gpu_block(self) -> str | None:
        """Why a GPU job cannot start now, or None."""
        if self.dry:
            return None
        h = stages.gpu_lock_holder()
        if h:
            return f"another clay_film process holds the GPU lock (pid {h.get('pid')}: {h.get('what')})"
        up = _ports_up()
        live = [p for p, ok in up.items() if ok]
        if live:
            busy = [p for p in live if _comfy_busy(p)]
            if busy:
                return f"the farm is busy with work this studio did not send (ports {busy})"
            return None
        used = _gpu_used()
        hot = [f"GPU {i}: {m} MiB" for i, m in enumerate(used) if m > BUSY_MIB]
        if hot:
            return "a card is in use by something else with the farm down (" + ", ".join(hot) + ")"
        return None

    def _pick(self, lane: str) -> list[tuple[Path, dict]]:
        with self.lock:
            cands = []
            for piece in self.pieces():
                for j in self.load(piece):
                    # a job queued under --dry never runs on a real server, and the reverse
                    if j["state"] == "queued" and j["lane"] == lane and bool(j.get("dry")) == self.dry:
                        cands.append((piece, j))
            if not cands:
                return []
            cands.sort(key=lambda pj: pj[1]["created"])
            if lane == "gpu":
                mode = self.farm_state
                first = next((pj for pj in cands if FARM_OF.get(pj[1]["kind"]) == mode), cands[0])
                batch = [first]
                if first[1]["kind"] in ("still", "edit"):
                    for pj in cands:
                        if len(batch) >= 2:
                            break
                        if pj is not first and pj[0] == first[0] and pj[1]["kind"] in ("still", "edit"):
                            batch.append(pj)
            else:
                batch = [cands[0]]
            for piece, j in batch:
                self.update(piece, j["id"], state="running", started=now(), note=None)
                j.update(state="running", started=now())
            self.running[lane] = [(p, j["id"]) for p, j in batch]
            return batch

    def loop(self, lane: str) -> None:
        while not self.stop:
            if lane == "gpu" and not self.dry:
                has_work = any(j["state"] == "queued" and j["lane"] == "gpu" and bool(j.get("dry")) == self.dry
                               for p in self.pieces() for j in self.load(p))
                if has_work:
                    self.block = self.gpu_block()
                    if self.block:
                        self.wake.wait(15)
                        self.wake.clear()
                        continue
                else:
                    self.block = None
                    self._idle_farm()
            batch = self._pick(lane)
            if not batch:
                self.wake.wait(2)
                self.wake.clear()
                continue
            try:
                self._run(lane, batch)
            except Exception:  # noqa: BLE001  a lane thread that dies stops the whole queue
                traceback.print_exc()
                for piece, j in batch:
                    try:
                        self.update(piece, j["id"], state="failed", finished=now(),
                                    error="studio error: " + traceback.format_exc(limit=3)[-1200:])
                    except Exception:  # noqa: BLE001
                        pass
            finally:
                self.running[lane] = []
                if lane == "gpu":
                    self.gpu_idle_since = time.time()
                    try:
                        self.farm_state = self.farm_mode()
                    except Exception:  # noqa: BLE001
                        pass

    def _idle_farm(self) -> None:
        if self.farm_started_by_us and time.time() - self.gpu_idle_since > FARM_IDLE_S:
            if stages.gpu_lock_holder() is None:
                stages.set_sink(None)
                stages.farm("off")
                self.farm_started_by_us = False
                self.farm_state = "off"

    def stop_farm(self) -> str:
        if self.running["gpu"]:
            raise HTTPException(409, "a GPU job is running; the farm stops after it (or after "
                                     f"{FARM_IDLE_S // 60} idle minutes)")
        if stages.gpu_lock_holder():
            raise HTTPException(409, "another process holds the GPU lock; not touching the farm")
        stages.set_sink(None)
        stages.farm("off")
        self.farm_started_by_us = False
        self.farm_state = "off"
        return "farm stopped"

    def _run(self, lane: str, batch: list[tuple[Path, dict]]) -> None:
        piece = batch[0][0]
        lead = batch[0][1]
        ids = [j["id"] for _, j in batch]

        def sink(msg: str) -> None:
            for jid in ids:
                self.log(jid, piece, msg)
        stages.set_sink(sink)
        # the job record as it is now: a prompt edited while it was queued is read here
        fresh = {j["id"]: j for j in self.load(piece)}
        jobs = [fresh.get(jid, j) for (_, j), jid in zip(batch, ids)]
        sink(f"start {', '.join(j['kind'] + ' ' + (j.get('sid') or '') for j in jobs)}")
        try:
            kind = lead["kind"]
            if kind in GPU_KINDS and not self.dry and FARM_OF[kind] != self.farm_state:
                self.farm_started_by_us = True
                for jid in ids:
                    self.update(piece, jid, est_s=EST[kind] + SWITCH_S.get(FARM_OF[kind], 60))
            if kind in ("still", "edit"):
                specs = [{"ref": j["shot_id"], "kind": "edit" if j["kind"] == "edit" else "redo",
                          "seed": j["params"].get("seed"), "prompt": j["params"].get("prompt") or None,
                          "base_v": j["params"].get("base_v")} for j in jobs]

                def on_start(info):
                    for j, inf in zip(jobs, info):
                        self.update(piece, j["id"], used=inf)
                items = stages.render_stills(piece, specs, on_start=on_start, strict=False)
                by_k = {it["k"]: it for it in items}
                for k, j in enumerate(jobs):
                    it = by_k.get(k)
                    if it and "error" not in it:
                        self.update(piece, j["id"], state="done", finished=now(),
                                    result={"v": it["v"], "file": it["file"]})
                    else:
                        self.update(piece, j["id"], state="failed", finished=now(),
                                    error=(it or {}).get("error", "no result"))
                return
            if kind == "motion":
                def on_start(info):
                    self.update(piece, lead["id"], used=info[0])
                it = stages.motion_one(piece, jobs[0]["shot_id"], prompt_override=jobs[0]["params"].get("prompt") or None,
                                       seed=jobs[0]["params"].get("seed"), on_start=on_start)
                res = {"v": it["v"], "file": it["file"]}
            elif kind == "narrate":
                stages.narrate_piece(piece)
                res = {"timeline": "timeline.json"}
            elif kind == "clay":
                out = stages.clay_shots(piece, [jobs[0]["shot_id"]], force=True)
                res = {"clay": out}
            elif kind == "assemble":
                f = stages.assemble_piece(piece)
                res = {"file": f.relative_to(piece).as_posix() if f.exists() else None}
            elif kind == "plan":
                res = self._plan_job(piece, jobs[0], sink)
            else:
                raise ValueError(f"unknown job kind {kind}")
            self.update(piece, lead["id"], state="done", finished=now(), result=res)
        except Exception as e:  # noqa: BLE001  a failed job is reported, the runner goes on
            # the child's last lines say why (assemble_film: "INCOMPLETE: missing s005")
            said = [l.split("] ", 1)[-1] for l in self.log_tail(piece, lead["id"], 6)
                    if not l.split("] ", 1)[-1].startswith(("$ ", "start "))]
            sink("FAILED: " + str(e))
            sink(traceback.format_exc()[-3000:])
            err = str(e) + ("\n" + "\n".join(said[-4:]) if said else "")
            for jid in ids:
                j = next((x for x in self.load(piece) if x["id"] == jid), {})
                if j.get("state") == "running":
                    self.update(piece, jid, state="failed", finished=now(), error=err[-1500:])
        finally:
            stages.set_sink(None)
            for jid in ids:            # the log file keeps it; memory only holds running jobs
                self.logs.pop(jid, None)

    def _plan_job(self, piece: Path, job: dict, log) -> dict:
        src = stages.load_source(piece)
        p = job["params"]
        ep, model = p.get("endpoint") or cfg_public()["endpoint"], p.get("model") or cfg_public()["model"]
        passage = src.get("text") if src.get("mode") == "passage" else None
        brief = src.get("text") if src.get("mode") != "passage" else None
        if not (passage or brief):
            raise ValueError("the piece has no brief or passage; fill in the Input page first")
        log(f"planner: {model} at {ep}")
        plan, errs = planner.plan(ep, model, brief=brief, passage=passage,
                                  seconds=float(src.get("seconds") or 30),
                                  location=src.get("location") or None, title=src.get("title") or None,
                                  mode=src.get("shot_mode") or "clay", style=src.get("style") or "",
                                  select=not src.get("whole", True), log=log, api_key=api_key(ep))
        with stages.piece_lock(piece):
            old = piece / "plan.json"
            if old.exists():
                b = webdir(piece) / "plan_backups"
                b.mkdir(exist_ok=True)
                shutil.copy2(old, b / f"plan_{datetime.now():%Y%m%dT%H%M%S}.json")
            if src.get("title") and not plan.get("title"):
                plan["title"] = src["title"]
            versions.ensure_ids(plan)
            stages.save_plan(piece, plan)
            if not imported(piece):
                versions.import_piece(piece)
            if not stages.blocking_errors(plan):
                schema.compile_plan(plan, piece)
                versions.materialize(piece)
        return {"shots": len(plan.get("shots") or []), "errors": errs[:30]}

    def snapshot(self) -> dict:
        out, t = [], time.time()
        for piece in self.pieces():
            for j in self.load(piece):
                if j["state"] in ("queued", "running") or (j.get("finished") or "") >= \
                        datetime.fromtimestamp(t - 3600).isoformat(timespec="seconds"):
                    j = dict(j)
                    if j["state"] == "queued" and bool(j.get("dry")) != self.dry:
                        j["note"] = ("queued under --dry; runs only when the server runs dry" if j.get("dry")
                                     else "queued on a real server; runs only when the server is not --dry")
                    if j["state"] == "running" and j.get("started"):
                        el = t - datetime.fromisoformat(j["started"]).timestamp()
                        j["elapsed_s"] = round(el)
                        j["progress"] = round(min(0.97, el / max(j.get("est_s") or 60, 1)), 3)
                        j["log_tail"] = self.log_tail(piece, j["id"], 6)
                    out.append(j)
        out.sort(key=lambda j: ({"running": 0, "queued": 1}.get(j["state"], 2), j["created"]))
        fin = [j for j in out if j["state"] not in ("running", "queued")]
        fin.sort(key=lambda j: j.get("finished") or "", reverse=True)
        live = [j for j in out if j["state"] in ("running", "queued")]
        return {"dry": self.dry, "farm": self.farm_state, "farm_started_by_studio": self.farm_started_by_us,
                "gpu_block": self.block, "jobs": live + fin[:25]}


Q: Queue | None = None


# -- what a page needs about a piece ----------------------------------------------------------

_meta_cache: dict = {}


def meta() -> dict:
    if not _meta_cache:
        locs = []
        for r in locations.catalogue():
            d = locations.describe(r["id"])
            locs.append({"id": r["id"], "name": r.get("name", r["id"]), "exterior": r.get("exterior"),
                         "marks": sorted(d.get("marks", {})), "props": d.get("look_at") or {}})
        _meta_cache.update({
            "locations": locs, "cast": castmod.registry(),
            "enums": {"angle": schema.ANGLES, "elevation": schema.ELEVATIONS, "distance": schema.DISTANCES,
                      "lens": schema.LENSES, "frame": schema.FRAMES, "pose": schema.POSES,
                      "facing": list(schema.FACING), "subject_class": schema.SUBJECT_CLASSES,
                      "grade": schema.GRADES, "mode": list(schema.MODES)},
            "read_wps": schema.READ_WPS, "protected": sorted(PROTECTED)})
    return {**_meta_cache, "dry": Q.dry if Q else False}


def _passage(src: dict) -> str | None:
    return " ".join(src["text"].split()) if src.get("mode") == "passage" and src.get("text") else None


def plan_report(piece: Path, plan: dict) -> dict:
    src = stages.load_source(piece)
    passage = _passage(src)
    try:
        errs = schema.validate(plan, passage)
        blocking = set(stages.blocking_errors(plan))
    except Exception as e:  # noqa: BLE001  a half-typed plan must not break the page
        errs = [f"plan: cannot validate: {type(e).__name__}: {e}"]
        blocking = set(errs)
    cov = schema.coverage(plan, src["text"]) if passage else None
    if cov:
        cov.pop("source", None)
    words = sum(len((s.get("narration") or "").split()) for s in plan.get("shots") or [] if isinstance(s, dict))
    return {"errors": errs, "blocking": [e for e in errs if e in blocking], "coverage": cov,
            "words": words, "est_seconds": round(words / schema.READ_WPS
                                                 + sum(float(s.get("tail_s") or 0) for s in plan.get("shots") or []
                                                       if isinstance(s, dict)), 1)}


def _read(p: Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return None


def prompts(piece: Path, plan: dict, info: list[dict]) -> list[dict]:
    """Per shot: the still prompt a redo would use and the H3 prompt a motion job would send."""
    try:
        compiled = json.loads((piece / "shots.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        compiled = []
    out = []
    for i, s in enumerate(plan.get("shots") or []):
        r = info[i] if i < len(info) else {}
        if schema.shot_mode(s) == "free":
            still = schema.free_prompt(plan, s)
        else:
            clay = r.get("clay") or {}
            still = _read(piece / clay["prompt_file"]) if clay.get("prompt_file") else None
        comp = compiled[i] if i < len(compiled) and compiled[i].get("i") == i else None
        auto = None
        if comp and " ".join(comp.get("narration", "").split()) == " ".join((s.get("narration") or "").split()):
            try:
                auto = stages.auto_motion_prompt({**comp, "scene": s.get("scene", comp.get("scene", "")),
                                                  "motion": s.get("motion", comp.get("motion")),
                                                  "action": s.get("action", comp.get("action")),
                                                  "grounded_in": s.get("subject", comp.get("grounded_in")),
                                                  "move_target": s.get("move_target", ""),
                                                  "soundscape": s.get("soundscape", "")})
            except Exception as e:  # noqa: BLE001
                auto = f"(could not build: {e})"
        out.append({"still_auto": still, "still_override": s.get("still_prompt") or None,
                    "motion_auto": auto, "motion_override": s.get("motion_prompt") or None})
    return out


def piece_state(name: str) -> dict:
    piece = piece_path(name)
    plan = stages.load_plan(piece) if (piece / "plan.json").exists() else None
    src = stages.load_source(piece)
    out = {"name": name, "protected": name in PROTECTED, "imported": imported(piece), "plan": plan,
           "source": src, "config": cfg_public()}
    if plan is None:
        return out
    out.update(plan_report(piece, plan))
    if out["imported"]:
        info = versions.shot_info(piece, plan)
        pr = prompts(piece, plan, info)
        for r, p in zip(info, pr):
            cl = next((x for x in r["clips"] if x["v"] == r["active_clip"]), None)
            want = p["motion_override"] or p["motion_auto"]
            if cl and cl.get("prompt") and want and cl["prompt"] != want:
                r["clip_stale"] = r["clip_stale"] + ["the motion prompt changed since this clip"]
        out.update(shots=info, prompts=pr)
    else:
        out.update(shots=[], prompts=[])
    films = sorted(piece.glob(f"{piece.name}*.mp4"))
    out["films"] = [{"file": f.name, "size": f.stat().st_size, "at": f.stat().st_mtime} for f in films]
    out["narrated"] = (piece / "timeline.json").exists()
    out["status"] = stages.status(piece)
    return out


# -- the app ----------------------------------------------------------------------------------

app = FastAPI(title="Clay Film Studio")


@app.get("/api/meta")
def api_meta():
    return meta()


@app.get("/api/pieces")
def api_pieces():
    rows = []
    for d in sorted(paths.PIECES.iterdir()):
        if not d.is_dir() or d.name.startswith("_lib") or not NAME_RE.match(d.name):
            continue
        pf = d / "plan.json"
        if not pf.exists() and not (d / "source.json").exists():
            continue
        title, n = "", 0
        try:
            p = json.loads(pf.read_text(encoding="utf-8"))
            title, n = p.get("title", ""), len(p.get("shots") or [])
        except (OSError, ValueError):
            pass
        rows.append({"name": d.name, "title": title, "shots": n, "imported": imported(d),
                     "protected": d.name in PROTECTED, "at": (pf if pf.exists() else d).stat().st_mtime})
    return rows


@app.post("/api/pieces")
def api_new_piece(body: dict = Body(...)):
    name = body.get("name", "")
    dst = piece_path(name, must_exist=False)
    if dst.exists():
        raise HTTPException(409, f"{name} already exists")
    how = body.get("from", "blank")
    if how == "copy":
        src = piece_path(body.get("source", ""))
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("_work", "_webui", "*.tmp"))
        for f in ("versions.json",):
            (dst / f).unlink(missing_ok=True)
        if (dst / "versions").exists():
            shutil.rmtree(dst / "versions")
    elif how == "plan_file":
        f = Path(body.get("path", ""))
        if not f.is_file() or f.suffix.lower() != ".json":
            raise HTTPException(400, f"not a plan file: {f}")
        plan = json.loads(f.read_text(encoding="utf-8"))
        dst.mkdir(parents=True)
        stages.save_plan(dst, plan)
    else:
        dst.mkdir(parents=True)
        (dst / "source.json").write_text(json.dumps({"mode": body.get("mode", "brief"), "text": "",
                                                     "seconds": 30, "shot_mode": "clay"}, indent=1),
                                         encoding="utf-8")
    if (dst / "plan.json").exists():
        open_piece(dst)
    return piece_state(name)


@app.get("/api/piece/{name}")
def api_piece(name: str):
    return piece_state(name)


@app.post("/api/piece/{name}/import")
def api_import(name: str):
    writable(name)
    piece = piece_path(name)
    if not (piece / "plan.json").exists():
        raise HTTPException(409, "no plan.json to import")
    open_piece(piece)
    return piece_state(name)


@app.post("/api/piece/{name}/rescan")
def api_rescan(name: str):
    piece = need_imported(name)
    found = versions.sync_external(piece)
    return {"found": found, **piece_state(name)}


@app.put("/api/piece/{name}/source")
def api_source(name: str, body: dict = Body(...)):
    writable(name)
    piece = piece_path(name)
    keep = ("mode", "text", "seconds", "style", "title", "location", "shot_mode", "whole", "file")
    src = {k: body[k] for k in keep if k in body}
    (piece / "source.json").write_text(json.dumps(src, indent=1, ensure_ascii=False), encoding="utf-8")
    return piece_state(name)


@app.put("/api/piece/{name}/plan")
def api_plan(name: str, body: dict = Body(...)):
    """Save the plan as typed, report every problem, compile when nothing blocks it."""
    piece = need_imported(name)
    plan = body.get("plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("shots"), list):
        raise HTTPException(400, "body.plan must be a plan object with a shots list")
    compiled = False
    with stages.piece_lock(piece):
        versions.ensure_ids(plan)
        stages.save_plan(piece, plan)
        rep = plan_report(piece, plan)
        if not rep["blocking"]:
            try:
                schema.compile_plan(plan, piece)
                compiled = True
            except Exception as e:  # noqa: BLE001  a plan valid to the checker can still trip compile
                rep["blocking"].append(f"plan: compile failed: {e}")
                rep["errors"].append(f"plan: compile failed: {e}")
        versions.materialize(piece)
    st = piece_state(name)
    return {**st, "compiled": compiled, "blocking": rep["blocking"], "errors": rep["errors"]}


def _chat_ctx(name: str, body: dict):
    piece = need_imported(name)
    plan = body.get("plan") or stages.load_plan(piece)
    src = stages.load_source(piece)
    passage = _passage(src)
    c = cfg_public()
    ep, model = body.get("endpoint") or c["endpoint"], body.get("model") or c["model"]
    return piece, plan, passage, ep, model


@app.post("/api/piece/{name}/chat/shot")
def api_chat_shot(name: str, body: dict = Body(...)):
    """A proposed revision of one shot. Nothing is saved: the page shows it as a diff."""
    piece, plan, passage, ep, model = _chat_ctx(name, body)
    i = int(body["index"])
    try:
        shot, errs = planner.revise_shot(ep, model, plan, i, body.get("request", ""),
                                         lock_narration=passage is not None, passage=passage,
                                         api_key=api_key(ep))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"planner model failed: {e}")
    return {"index": i, "shot": shot, "errors": errs}


@app.post("/api/piece/{name}/chat/plan")
def api_chat_plan(name: str, body: dict = Body(...)):
    piece, plan, passage, ep, model = _chat_ctx(name, body)
    try:
        new, errs = planner.revise_plan(ep, model, plan, body.get("request", ""),
                                        lock_narration=passage is not None, passage=passage,
                                        api_key=api_key(ep))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"planner model failed: {e}")
    return {"plan": new, "errors": errs}


@app.post("/api/piece/{name}/jobs")
def api_enqueue(name: str, body: dict = Body(...)):
    """body: {kind, shots: [id or sNNN, ...] (per-shot kinds), seed, prompt, base_v}."""
    kind = body.get("kind")
    if kind not in GPU_KINDS | CPU_KINDS:
        raise HTTPException(400, f"kind is one of {sorted(GPU_KINDS | CPU_KINDS)}")
    if kind == "plan":
        writable(name)
        piece = piece_path(name)
        c = cfg_public()
        return [Q.add(piece, "plan", None, {"endpoint": body.get("endpoint") or c["endpoint"],
                                            "model": body.get("model") or c["model"]})]
    piece = need_imported(name)
    if kind in ("narrate", "assemble"):
        return [Q.add(piece, kind, None, {})]
    plan = stages.load_plan(piece)
    refs = body.get("shots") or []
    if not refs:
        raise HTTPException(400, "which shots? body.shots is empty")
    if kind == "edit" and not (body.get("prompt") or "").strip():
        raise HTTPException(400, "an edit needs an instruction")
    todo = []
    for ref in refs:
        try:
            i, s = stages.resolve(plan, ref)
        except KeyError as e:
            raise HTTPException(404, str(e))
        if kind == "clay" and schema.shot_mode(s) == "free":
            continue
        params = {k: body.get(k) for k in ("seed", "prompt", "base_v") if body.get(k) not in (None, "")}
        todo.append(({"id": s["id"], "sid": f"s{i:03d}"}, params))
    with Q.lock:        # all of one request land together, so stills of one request can share a batch
        return [Q.add(piece, kind, shot, params) for shot, params in todo]


@app.get("/api/queue")
def api_queue():
    return Q.snapshot()


@app.get("/api/jobs/{name}/{jid}/log")
def api_job_log(name: str, jid: str, n: int = 300):
    return {"lines": Q.log_tail(piece_path(name), jid, n)}


@app.patch("/api/jobs/{name}/{jid}")
def api_job_edit(name: str, jid: str, body: dict = Body(...)):
    return Q.edit(piece_path(name), jid, body)


@app.delete("/api/jobs/{name}/{jid}")
def api_job_cancel(name: str, jid: str):
    return Q.cancel(piece_path(name), jid)


@app.post("/api/piece/{name}/jobs/clear")
def api_jobs_clear(name: str):
    return {"removed": Q.clear_finished(piece_path(name))}


@app.post("/api/piece/{name}/activate")
def api_activate(name: str, body: dict = Body(...)):
    piece = need_imported(name)
    try:
        versions.set_active(piece, body["id"], body["kind"], body.get("v"))
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    return piece_state(name)


@app.get("/api/config")
def api_config():
    return cfg_public()


@app.put("/api/config")
def api_config_put(body: dict = Body(...)):
    c = cfg_load()
    for k in ("endpoint", "model"):
        if body.get(k):
            c[k] = str(body[k]).strip()
    if body.get("clear_key"):
        c.pop("api_key", None)
    elif body.get("api_key"):
        c["api_key"] = str(body["api_key"]).strip()
    cfg_save(c)
    return cfg_public()


@app.post("/api/farm/stop")
def api_farm_stop():
    return {"message": Q.stop_farm()}


def _safe(piece: Path, rel: str) -> Path:
    f = (piece / rel).resolve()
    if piece.resolve() not in f.parents or not f.is_file():
        raise HTTPException(404, "no such file")
    return f


@app.get("/api/piece/{name}/file")
def api_file(name: str, path: str):
    f = _safe(piece_path(name), path)
    return FileResponse(f, headers={"Cache-Control": "no-cache"})


@app.get("/api/piece/{name}/thumb")
def api_thumb(name: str, path: str, w: int = 448):
    from PIL import Image
    piece = piece_path(name)
    f = _safe(piece, path)
    w = max(64, min(int(w), 1344))
    st = f.stat()
    key = hashlib.sha1(f"{f}|{st.st_mtime_ns}|{st.st_size}|{w}".encode()).hexdigest()[:16]
    d = webdir(piece) / "thumbs"
    d.mkdir(exist_ok=True)
    out = d / f"{key}.jpg"
    if not out.exists():
        im = Image.open(f).convert("RGB")
        im.thumbnail((w, w * 4))
        tmp = out.with_suffix(".tmp.jpg")
        im.save(tmp, quality=85)
        tmp.replace(out)
    return FileResponse(out, headers={"Cache-Control": "max-age=86400"})


@app.exception_handler(HTTPException)
def _http_err(_req, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main() -> int:
    global Q
    ap = argparse.ArgumentParser(description="Clay Film Studio")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--dry", action="store_true", help="fake every GPU job (clay_film/dry.py)")
    ap.add_argument("--open", action="store_true", help="open the browser")
    a = ap.parse_args()
    if a.dry:
        os.environ["CLAY_FILM_DRY"] = "1"
        stages.DRY = True
    Q = Queue(dry=stages.DRY)
    Q.recover()
    try:
        Q.farm_state = Q.farm_mode()
    except Exception:  # noqa: BLE001
        Q.farm_state = "off"
    for lane in ("gpu", "cpu"):
        threading.Thread(target=Q.loop, args=(lane,), daemon=True, name=f"lane-{lane}").start()
    url = f"http://127.0.0.1:{a.port}/"
    print(f"Clay Film Studio on {url}{'  (DRY: GPU jobs are faked)' if Q.dry else ''}", flush=True)
    if a.open:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
