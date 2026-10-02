"""Roko locations as a planner sees them: id, name, size, and named marks where a person can stand.

A mark is "<depth>-<lateral>": depth near/mid/far is 20/50/80% of the location's depth from the
camera side (z = 0); lateral left/centre/right is -w/6, 0, +w/6. A computed mark that lands inside
a solid prop is nudged to the nearest free spot, so a planner can never stand someone inside a
tree. marks/<location>.json overrides any mark with coordinates proven in a real shot.
"""
import json
import math
from functools import lru_cache

from . import paths

DEPTHS = {"near": 0.2, "mid": 0.5, "far": 0.8}
LATERALS = {"left": -1 / 6, "centre": 0.0, "right": 1 / 6}
CLEAR_M = 0.35


def _space(loc: str) -> dict:
    return json.loads((paths.LOCATIONS / loc / "space.json").read_text(encoding="utf-8"))


def _meta(loc: str) -> dict:
    p = paths.LOCATIONS / loc / "location.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _solids(space: dict) -> list[tuple[float, float, float, float]]:
    """Plan-view footprints (x0, x1, z0, z1) of props a person cannot stand inside."""
    out = []
    for pr in space.get("props", []):
        pos, size = pr.get("pos"), pr.get("size")
        if not pos or not size or len(size) < 3:
            continue
        bottom = pos[1] - size[1] / 2
        if size[1] < 0.3 or bottom > 1.5:
            continue  # flat ground cover, or something overhead
        out.append((pos[0] - size[0] / 2, pos[0] + size[0] / 2, pos[2] - size[2] / 2, pos[2] + size[2] / 2))
    return out


def _free(x: float, z: float, solids, w: float, d: float) -> bool:
    if not (-w / 2 + CLEAR_M <= x <= w / 2 - CLEAR_M and CLEAR_M <= z <= d - CLEAR_M):
        return False
    return all(not (x0 - CLEAR_M < x < x1 + CLEAR_M and z0 - CLEAR_M < z < z1 + CLEAR_M)
               for x0, x1, z0, z1 in solids)


def nudge(loc: str, x: float, z: float) -> tuple[float, float]:
    """The nearest free standing spot to (x, z), searched in 0.25 m rings out to 5 m."""
    sp = _space(loc)
    w, d = sp["room"]["w"], sp["room"]["d"]
    solids = _solids(sp)
    if _free(x, z, solids, w, d):
        return round(x, 2), round(z, 2)
    for k in range(1, 21):
        r = 0.25 * k
        for a in range(24):
            t = 2 * math.pi * a / 24
            cx, cz = x + r * math.cos(t), z + r * math.sin(t)
            if _free(cx, cz, solids, w, d):
                return round(cx, 2), round(cz, 2)
    return round(x, 2), round(z, 2)


@lru_cache(maxsize=None)
def marks(loc: str) -> dict[str, list[float]]:
    sp = _space(loc)
    w, d = sp["room"]["w"], sp["room"]["d"]
    out = {f"{dn}-{ln}": list(nudge(loc, lf * w, df * d))
           for dn, df in DEPTHS.items() for ln, lf in LATERALS.items()}
    over = paths.MARKS / f"{loc}.json"
    if over.exists():
        out.update({k: v for k, v in json.loads(over.read_text(encoding="utf-8")).items()
                    if not k.startswith("_")})
    return out


def resolve_mark(loc: str, at) -> tuple[float, float]:
    """A mark name or explicit [x, z] -> free coordinates."""
    if isinstance(at, (list, tuple)) and len(at) == 2:
        return nudge(loc, float(at[0]), float(at[1]))
    key = str(at).strip().lower().replace("center", "centre").replace("_", "-").replace(" ", "-")
    m = marks(loc)
    if key not in m:
        raise KeyError(f"unknown mark {at!r} in {loc}; marks are {sorted(m)}")
    return tuple(m[key])


def ids() -> list[str]:
    return sorted(p.name for p in paths.LOCATIONS.iterdir() if (p / "space.json").exists())


def describe(loc: str, with_marks: bool = True) -> dict:
    sp, meta = _space(loc), _meta(loc)
    row = {"id": loc, "name": meta.get("name", loc),
           "desc": meta.get("desc") or (sp.get("look") or "")[:300],
           "size_m": [sp["room"]["w"], sp["room"]["d"]],
           "exterior": bool(sp.get("openToSky")),
           "props": len(sp.get("props", []))}
    if with_marks:
        row["marks"] = sorted(marks(loc))
        # what an empty shot can frame: one prop id per distinct label
        seen, looks = set(), {}
        for pr in sp.get("props", []):
            lab = pr.get("label")
            if lab and lab not in seen:
                seen.add(lab)
                looks[pr["id"]] = lab
        row["look_at"] = looks
    return row


def prop_ids(loc: str) -> set[str]:
    return {p["id"] for p in _space(loc).get("props", []) if "id" in p}


def catalogue(with_marks: bool = False) -> list[dict]:
    rows = []
    for loc in ids():
        try:
            rows.append(describe(loc, with_marks))
        except (KeyError, ValueError):
            continue
    return rows
