"""clay_film as an MCP server (stdio), for any agent that speaks MCP: Claude Code, OpenCode, a local model.

    python -m clay_film.mcp_server

The intended loop for an agent: guide -> (location) -> write a plan -> validate_plan until it
returns ok -> new_piece -> run_piece -> status until done. plan_with_llm does the whole planning
half in one call against an OpenAI-compatible endpoint, for agents that would rather delegate it.
run_piece starts a background process and returns at once; the GPU stages take minutes to hours.

Shot by shot: shot_still / shot_edit / shot_motion start one background job each and add a new
active version; shot_versions lists them and activate_version chooses one. All GPU jobs share
clay_film/.gpu.lock with the studio (webui/), so they wait for each other rather than collide.
"""
import json
import subprocess
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from . import cast, locations, paths, planner, schema, stages

mcp = FastMCP("clay_film")


def _plan(plan_json) -> dict:
    return json.loads(plan_json) if isinstance(plan_json, str) else plan_json


@mcp.tool()
def guide(location: str = "", passage: bool = False, seconds: float = 30) -> str:
    """The planning rulebook, schema and an example plan. Read this before writing a plan.
    Pass location to get its marks (where people can stand) and look_at props."""
    return planner.guide(location or None, passage, seconds)


@mcp.tool()
def list_locations() -> list[dict]:
    """Every location: id, name, description, size in metres, exterior or not."""
    return locations.catalogue()


@mcp.tool()
def location(location_id: str) -> dict:
    """One location with its marks and look_at prop ids."""
    return locations.describe(location_id)


@mcp.tool()
def list_cast() -> dict:
    """Cast id -> appearance. People not listed go in the plan's cast block."""
    return cast.registry()


@mcp.tool()
def validate_plan(plan_json: str, passage: str = "", seconds: float = 0) -> str:
    """Check a plan. Returns 'ok' or one problem per line, phrased to be fixed directly.
    passage: when narrating a text, narration must be verbatim from it. seconds: target length."""
    try:
        errs = schema.validate(_plan(plan_json), passage or None, seconds or None)
    except json.JSONDecodeError as e:
        return f"not valid JSON: {e}"
    return "\n".join(errs) or "ok"


@mcp.tool()
def new_piece(name: str, plan_json: str) -> str:
    """Validate a plan and install it as pieces/<name>/plan.json with its compiled stage files."""
    plan = _plan(plan_json)
    src = plan.get("source")
    passage = Path(src).read_text(encoding="utf-8") if src and Path(src).is_file() else None
    errs = schema.validate(plan, passage)
    if errs:
        return "not installed:\n" + "\n".join(errs)
    piece = stages.piece_dir(name)
    piece.mkdir(parents=True, exist_ok=True)
    (piece / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    schema.compile_plan(plan, piece)
    return f"installed {piece} ({len(plan['shots'])} shots)"


@mcp.tool()
def plan_with_llm(name: str, brief: str = "", passage: str = "", seconds: float = 30,
                  location: str = "", endpoint: str = "http://127.0.0.1:8080/v1",
                  model: str = "local") -> str:
    """Have a model at an OpenAI-compatible endpoint write the plan, with validate-and-repair
    rounds, and install it. Give a brief (the model writes narration) or a passage (verbatim)."""
    plan, errs = planner.plan(endpoint, model, brief=brief or None, passage=passage or None,
                              seconds=seconds, location=location or None, log=lambda m: None)
    piece = stages.piece_dir(name)
    piece.mkdir(parents=True, exist_ok=True)
    (piece / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    if errs:
        return "plan written but does not validate; fix with validate_plan/new_piece:\n" + "\n".join(errs)
    schema.compile_plan(plan, piece)
    return f"installed {piece}:\n" + json.dumps(plan, indent=1, ensure_ascii=False)


@mcp.tool()
def run_piece(name: str, stages_csv: str = "", only: str = "") -> str:
    """Start the pipeline in the background. stages_csv from compile,narrate,clay,stills,motion,
    assemble (default all). only: shot ids like s000,s003. Poll status(name)."""
    piece = stages.piece_dir(name)
    if not (piece / "plan.json").exists() and not (piece / "shots.json").exists():
        return f"no plan at {piece}"
    cmd = [sys.executable, "-m", "clay_film", "run", name]
    if stages_csv:
        cmd += ["--stages", stages_csv]
    if only:
        cmd += ["--only", only]
    log = open(piece / "clay_film.log", "a", encoding="utf-8")
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    p = subprocess.Popen(cmd, cwd=paths.ROOT, stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, creationflags=flags)
    return f"started pid {p.pid}; log {piece / 'clay_film.log'}"


def _spawn(name: str, args: list[str]) -> str:
    piece = stages.piece_dir(name)
    if not (piece / "plan.json").exists():
        return f"no plan at {piece}"
    cmd = [sys.executable, "-m", "clay_film"] + args
    log = open(piece / "clay_film.log", "a", encoding="utf-8")
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    p = subprocess.Popen(cmd, cwd=paths.ROOT, stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, creationflags=flags)
    return f"started pid {p.pid}: {' '.join(args)}; log {piece / 'clay_film.log'}; see shot_versions(name)"


def _opt(args: list[str], flag: str, val) -> list[str]:
    return args if val is None or val == "" else args + [flag, str(val)]


@mcp.tool()
def shot_still(name: str, shot: str, seed: int = -1, prompt: str = "") -> str:
    """Render a new still for one shot in the background; it becomes the active version.
    shot: position like s003 or the shot's id. A clay shot renders from its clay pass (made
    again first if the geometry changed); a free shot is text-to-image. seed -1 = the plan's seed.
    prompt replaces the still prompt for this render only."""
    return _spawn(name, _opt(_opt(["shot", name, "still", shot], "--seed", None if seed < 0 else seed),
                             "--prompt", prompt))


@mcp.tool()
def shot_edit(name: str, shot: str, instruction: str, seed: int = -1, base_version: int = 0) -> str:
    """Qwen-2.1 edit of one shot's active still (or base_version) by a plain instruction, e.g.
    'make the coat dark green'. Runs in the background; the result becomes the active version."""
    args = ["shot", name, "edit", shot, "--prompt", instruction]
    args = _opt(args, "--seed", None if seed < 0 else seed)
    return _spawn(name, _opt(args, "--base", base_version or None))


@mcp.tool()
def shot_motion(name: str, shot: str, prompt: str = "", seed: int = -1) -> str:
    """One H3 clip from the shot's active still, in the background (about 5-6 minutes). prompt
    replaces the H3 prompt for this clip; empty uses the plan's motion_prompt or the built one."""
    return _spawn(name, _opt(_opt(["shot", name, "motion", shot], "--prompt", prompt),
                             "--seed", None if seed < 0 else seed))


@mcp.tool()
def shot_clay(name: str, shot: str) -> str:
    """Render one shot's clay pass again (CPU, a few seconds). Returns when done."""
    try:
        return json.dumps(stages.clay_shots(name, [shot]), indent=1)
    except Exception as e:  # noqa: BLE001  report, do not crash the server
        return f"failed: {e}"


@mcp.tool()
def shot_versions(name: str) -> str:
    """Every shot's stills and clips with the active version and what is stale."""
    from . import versions
    piece, plan = stages.prepare(name)
    rows = versions.shot_info(piece, plan)
    keep = ("i", "sid", "id", "mode", "active_still", "active_clip", "still_stale", "clay_stale",
            "clip_stale", "est_seconds")
    return json.dumps([{**{k: r[k] for k in keep},
                        "stills": [{k: x.get(k) for k in ("v", "kind", "seed", "file", "at")} for x in r["stills"]],
                        "clips": [{k: x.get(k) for k in ("v", "seed", "still_ref", "file", "at")} for x in r["clips"]]}
                       for r in rows], indent=1)


@mcp.tool()
def activate_version(name: str, shot: str, kind: str, version: int) -> str:
    """Choose which still or clip version a shot uses (kind: still or clip). The positional
    stills/sNNN.png and clips/sNNN.mp4 files are rewritten to match."""
    from . import versions
    piece, plan = stages.prepare(name)
    i, s = stages.resolve(plan, shot)
    versions.set_active(piece, s["id"], kind, version)
    return f"s{i:03d} active {kind} -> v{version}"


@mcp.tool()
def status(name: str, log_lines: int = 20) -> str:
    """Per-stage state and the tail of the run log."""
    piece = stages.piece_dir(name)
    st = stages.status(piece)
    lf = piece / "clay_film.log"
    tail = lf.read_text(encoding="utf-8", errors="replace").splitlines()[-log_lines:] if lf.exists() else []
    films = [str(p) for p in piece.glob(f"{piece.name}*.mp4")]
    return json.dumps({"stages": st, "films": films, "log": tail}, indent=1)


if __name__ == "__main__":
    mcp.run()
