"""H3 motion, factored out of stages.motion_one, plus a batch path for several workers (2026-10-11).

`stages.motion_one` delegates here and behaves exactly as before. With no remote workers configured
nothing else changes: the local path still takes the GPU lock, starts the big worker and sends the
job to farm BIG_WORKER (8192).

Remote mode is opt-in through the farm's FARM_REMOTE_WORKERS (see farm/config.py remote_workers).
Each entry is a ComfyUI with the H3 weights reached through an SSH tunnel. When it is set and
CLAY_FILM_DRY is not:
- `stages.motion` calls `motion_stage`, which prepares every shot's job exactly as motion_one does
  (prompt, translator, seed, frames, end frame, mid guide), submits them all to one `Farm.run`
  across the remote workers, and registers each clip in versions.json with motion_one's meta;
- `motion_one` sends its single job to the remote workers instead of the local farm;
- neither takes the local GPU lock or starts or stops the local farm;
- the graph is `graphs.minimax_h3_single_gpu` (also chosen locally by CLAY_FILM_H3_SINGLE_GPU=1).

The helpers, in the order motion_one runs them: prepare_shot, then run_local (or one Farm.run over
the remote workers), collect, register.

The studio (webui/server.py) calls stages.farm("off") after its GPU jobs, which stops the local
farm whatever FARM_REMOTE_WORKERS says. Run remote motion from the CLI while a local stills queue
owns a card.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from . import stages as S

REMOTE_ENV = "FARM_REMOTE_WORKERS"


def remote_on() -> bool:
    return bool(os.environ.get(REMOTE_ENV, "").strip()) and not S.DRY


def single_gpu() -> bool:
    return remote_on() or os.environ.get("CLAY_FILM_H3_SINGLE_GPU") == "1"


def remote_workers() -> list[dict]:
    S._animate()                      # puts farm/ on sys.path
    import config as C  # noqa: E402  (farm/)
    return C.remote_workers()


def prepare_shot(piece, sid, prompt_override: str | None = None, seed=None, on_start=None,
                 work: Path | None = None) -> dict:
    """Everything motion_one does before the job is sent: the prompt, seed and frame count, the
    work folder with the still, end frame and mid guide copied in, the meta and the graph."""
    from . import versions
    piece = S.piece_dir(str(piece))
    with S.piece_lock(piece):
        piece, plan = S.prepare(piece)
        i, s = S.resolve(plan, sid)
        comp = S.compiled_shot(piece, i)
    uid, sid = s["id"], f"s{i:03d}"
    row = versions.load(piece)["shots"].get(uid) or {}
    st = versions.active(row, "still") if row else None
    if not st:
        raise ValueError(f"{sid} has no active still; make one first")
    an = S._animate()
    # A painted last keyframe made H3 flash white and pulse on some shots with no way to predict which
    # (Oceanus A/B 2026-10-04); a piece turns it off with "end_frames": false in plan.json.
    cfg = json.loads((piece / "config.json").read_text(encoding="utf-8")) if (piece / "config.json").exists() else {}
    end = None if comp.get("cuts") or cfg.get("end_frames") is False else an.end_frame(piece, sid)
    secs = S.shot_seconds(piece, i, s.get("narration", ""), s.get("tail_s", 0))
    n = an.frames_for(secs, comp)
    source = "override" if prompt_override else "motion_prompt" if comp.get("motion_prompt") else \
        "motion_plan" if comp.get("motion_plan") else "auto"
    if source == "motion_plan":
        # the director's plan, through the translator; a lint error refuses the clip before any GPU time
        from .motion_direction import checked_prompt
        prompt = checked_prompt(comp, end=bool(end), seconds=n / an.FPS, bible=an.style_bible(piece))
    else:
        prompt = prompt_override or comp.get("motion_prompt") or an.prompt_for(
            comp, end=bool(end), seconds=n / an.FPS, bible=an.style_bible(piece))
    seed = an.SEED + i if seed in (None, "") else int(seed)
    if on_start:
        on_start([{"sid": sid, "id": uid, "kind": "motion", "seed": seed, "prompt": prompt,
                   "still_v": st["v"]}])
    work = work or piece / "_work" / f"motion_{S._stamp()}"
    work.mkdir(parents=True, exist_ok=True)
    name = f"{uid}_{S._stamp()}"
    src = work / f"{name}_still.png"
    shutil.copy2(piece / st["file"], src)
    last = mid = None
    if end:
        last = work / f"{name}_last.png"
        shutil.copy2(end, last)
        if n >= S.MID_GUIDE_MIN_FRAMES:
            mid = work / f"{name}_mid.png"
            S.mid_keyframe(piece / st["file"], comp, mid)
    meta = {"kind": "h3", "seed": seed, "prompt": prompt, "frames": n, "seconds": round(n / an.FPS, 2),
            "end_frame": bool(end), "mid_guide": bool(mid),
            "want": round(secs, 2), "still_ref": st["v"],
            "prompt_overridden": bool(prompt_override or comp.get("motion_prompt")), "prompt_source": source,
            "backend": "dry" if S.DRY else "farm"}
    prep = {"piece": piece, "i": i, "sid": sid, "uid": uid, "n": n, "seed": seed, "prompt": prompt,
            "work": work, "name": name, "src": src, "last": last, "mid": mid, "meta": meta, "graph": None}
    if not S.DRY:
        build = an.graphs.minimax_h3_single_gpu if single_gpu() else an.graphs.minimax_h3_official
        prep["graph"] = build(
            prompt, src.name, width=an.W, height=an.H, seconds=n / an.FPS, fps=an.FPS,
            steps=an.STEPS, seed=seed, sparse={"tau": 1.3, "start_percent": 0.2},
            last_image=last.name if last else None,
            guides=[(mid.name, S.mid_frame(n))] if mid else None, prefix=f"{piece.name}_{name}")
    return prep


def job(prep: dict):
    from dispatch import Job  # noqa: E402  (farm/, on sys.path via animate_shots)
    p = prep
    return Job(job_id=p["name"], model_key="minimax_h3", out_name=p["name"],
               meta={"sid": p["sid"], "frames": p["n"]},   # the farm manifest only (batch_report.py)
               inputs=[p["src"]] + ([p["last"]] if p["last"] else []) + ([p["mid"]] if p["mid"] else []),
               graph=p["graph"])


def _announce(prep: dict) -> None:
    n, mid = prep["n"], prep["mid"]
    S._say(f"{prep['sid']}: H3 {n} frames, seed {prep['seed']}" + (", painted end frame" if prep["last"] else "")
           + (f", mid guide at f{S.mid_frame(n)}" if mid else ""))


def run_local(prep: dict):
    """The path motion_one always took: GPU lock, big worker, one job on BIG_WORKER."""
    import config as C  # noqa: E402  (farm/, on sys.path via animate_shots)
    from dispatch import Farm  # noqa: E402
    _announce(prep)
    with S.gpu_lock(f"motion {prep['piece'].name} {prep['sid']}"):
        S.farm("big")
        return Farm(prep["work"], workers=[C.BIG_WORKER], job_timeout=5400).run([job(prep)])


def collect(prep: dict, done, out_dir: Path | None = None) -> Path:
    """The clip the farm wrote for this job; raises as motion_one did when there is none."""
    out_dir = out_dir or prep["work"]
    if not done or done.status != "ok":
        shutil.rmtree(prep["work"], ignore_errors=True)
        raise RuntimeError(f"motion failed: {done.error if done else 'no result'}")
    out = next(iter(sorted(out_dir.glob(f"{prep['name']}*.mp4"))), None)
    if out is None:
        raise RuntimeError("motion finished but wrote no mp4")
    return out


def register(prep: dict, out: Path, gate: bool = True) -> dict:
    """The frozen-clip gate (farm runs only), then the clip as the shot's new active version."""
    from . import versions
    meta = prep["meta"]
    if gate:
        an = S._animate()
        gate_dir = prep["work"] / "gate"
        gate_dir.mkdir()
        shutil.copy2(out, gate_dir / f"{prep['sid']}.mp4")
        try:
            meta["frozen"] = bool(an.gate(gate_dir))
        except Exception as e:  # noqa: BLE001  the gate is a report, not a requirement
            S._say(f"motion gate skipped: {e}")
    item = versions.add_clip(prep["piece"], prep["uid"], out, meta)
    shutil.rmtree(prep["work"], ignore_errors=True)
    S._say(f"{prep['sid']} clip v{item['v']}")
    return {"sid": prep["sid"], "id": prep["uid"], **item}


def motion_one(piece, sid, prompt_override: str | None = None, seed=None, on_start=None) -> dict:
    """stages.motion_one. Local unless FARM_REMOTE_WORKERS is set."""
    prep = prepare_shot(piece, sid, prompt_override, seed, on_start)
    if S.DRY:
        from . import dry
        out = prep["work"] / f"{prep['name']}.mp4"
        dry.clip(out, prep["i"], log=S._say)
        return register(prep, out, gate=False)
    if remote_on():
        from dispatch import Farm  # noqa: E402
        _announce(prep)
        done = Farm(prep["work"], workers=remote_workers(), job_timeout=5400).run([job(prep)])
    else:
        done = run_local(prep)
    out = collect(prep, done[0] if done else None)
    return register(prep, out)


def motion_stage(piece: Path, only: set | None = None) -> list[dict]:
    """stages.motion in remote mode: every shot with a still and no clip, in one Farm.run.

    Shots are chosen as stages.motion chooses them. A shot whose job cannot be prepared (no active
    still, a motion plan the lint refuses) is reported and skipped, the rest render, and the stage
    raises at the end naming every shot that did not get a clip.
    """
    piece = S.piece_dir(str(piece))
    S._animate()                      # puts farm/ on sys.path
    from dispatch import Farm  # noqa: E402
    workers = remote_workers()
    compiled = json.loads((piece / "shots.json").read_text(encoding="utf-8"))
    batch = piece / "_work" / f"motion_batch_{S._stamp()}"
    batch.mkdir(parents=True, exist_ok=True)
    preps, failed = [], {}
    for s in compiled:
        sid = f"s{s['i']:03d}"
        if (only and sid not in only) or (piece / "clips" / f"{sid}.mp4").exists():
            continue
        if not (piece / "stills" / f"{sid}.png").exists():
            continue
        try:
            preps.append(prepare_shot(piece, sid, work=batch / sid))
        except Exception as e:  # noqa: BLE001  one bad shot must not cost the batch
            failed[sid] = f"not prepared: {type(e).__name__}: {str(e)[:300]}"
            S._say(f"{sid}: {failed[sid]}")
    S._say(f"motion batch: {len(preps)} jobs over {len(workers)} remote workers "
           f"({', '.join(w['name'] for w in workers)})")
    for p in preps:
        _announce(p)
    results = Farm(batch, workers=workers, job_timeout=5400).run([job(p) for p in preps]) if preps else []
    by_name = {r.job_id: r for r in results}
    made = []
    for p in preps:
        done = by_name.get(p["name"])
        try:
            out = collect(p, done, out_dir=batch)
            p["meta"]["worker"] = done.worker
            made.append(register(p, out))
        except Exception as e:  # noqa: BLE001
            failed[p["sid"]] = str(e)[:300]
            S._say(f"{p['sid']}: {failed[p['sid']]}")
    if (batch / "manifest.json").exists():
        shutil.copy2(batch / "manifest.json", batch.parent / f"{batch.name}_manifest.json")
    shutil.rmtree(batch, ignore_errors=True)
    S._say(f"motion batch: {len(made)} clips registered, {len(failed)} failed")
    S._report_cuts(piece, only)
    if failed:
        raise RuntimeError("motion batch: no clip for " + "; ".join(f"{k}: {v}" for k, v in sorted(failed.items())))
    return made
