"""Every still and clip a shot has had, which one is active, and what has gone stale.

The Agentic_Slop stages read positional files: stills/s003.png, clips/s003.mp4, blockout/s003_clay.png.
A shot's position changes when shots are added, split or reordered, so versions are keyed by the
shot's stable `id` in plan.json instead, and the positional files are a copy of each shot's active
version, rewritten by `materialize()`.

    <piece>/versions.json
    <piece>/versions/<id>/clay.png, clay_depth.png, clay_prompt.txt   latest clay pass
    <piece>/versions/<id>/still_v001.png ...
    <piece>/versions/<id>/clip_v001.mp4 ...
    <piece>/versions/_orphans/                                         positional files moved aside

Staleness is a comparison of hashes, never a regeneration:
    still   the shot's still-relevant plan fields changed after the still was made
    clay    the same, for the geometry fields, against the clay pass
    clip    the active still is no longer the one the clip was made from
"""
from __future__ import annotations

import hashlib
import json
import secrets
import shutil
import threading
from datetime import datetime
from pathlib import Path

from . import cast as castmod
from . import schema

_LOCK = threading.RLock()


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _h(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


def sid_of(i: int) -> str:
    return f"s{i:03d}"


# -- shot identity ------------------------------------------------------------------------------

def new_id() -> str:
    return "k" + secrets.token_hex(3)


def ensure_ids(plan: dict) -> bool:
    """Give every shot a stable id. Returns True when the plan changed."""
    seen, changed = set(), False
    for s in plan.get("shots") or []:
        if not s.get("id") or s["id"] in seen:
            s["id"] = new_id()
            changed = True
        seen.add(s["id"])
    return changed


def _people_descs(plan: dict, shot: dict) -> list[str]:
    reg = castmod.registry(plan.get("cast"))
    return [reg.get(p.get("who"), "") for p in shot.get("people") or [] if isinstance(p, dict)]


def clay_key(plan: dict, shot: dict) -> str:
    """What the clay pass depends on."""
    if schema.shot_mode(shot) == "free":
        return ""
    return _h({"loc": shot.get("location") or plan.get("location"), "camera": shot.get("camera"),
               "people": shot.get("people"), "look_at": shot.get("look_at"),
               "cast": _people_descs(plan, shot)})


def still_key(plan: dict, shot: dict) -> str:
    """What a still made from the plan depends on (not what an edit on top of it depends on)."""
    if schema.shot_mode(shot) == "free":
        return _h({"mode": "free", "prompt": schema.free_prompt(plan, shot)})
    return _h({"clay": clay_key(plan, shot), "still_prompt": shot.get("still_prompt") or ""})


# -- the record ---------------------------------------------------------------------------------

def path(piece: Path) -> Path:
    return Path(piece) / "versions.json"


def load(piece: Path) -> dict:
    f = path(piece)
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return {"shots": {}, "materialized": {}}


def save(piece: Path, rec: dict) -> None:
    f = path(piece)
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(f)


def _row(rec: dict, uid: str) -> dict:
    return rec["shots"].setdefault(uid, {"stills": [], "active_still": None,
                                         "clips": [], "active_clip": None, "clay": None})


def _stat(p: Path) -> list:
    st = p.stat()
    return [st.st_size, st.st_mtime_ns]


def _plan(piece: Path) -> dict:
    return json.loads((Path(piece) / "plan.json").read_text(encoding="utf-8"))


def _dir(piece: Path, uid: str) -> Path:
    d = Path(piece) / "versions" / uid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _next_v(items: list[dict]) -> int:
    return max((x["v"] for x in items), default=0) + 1


def active(row: dict, kind: str) -> dict | None:
    items, a = row[kind + "s"], row["active_" + kind]
    return next((x for x in items if x["v"] == a), None)


# -- adding and choosing ------------------------------------------------------------------------

def add_still(piece: Path, uid: str, src: Path, meta: dict, activate: bool = True,
              sync: bool = True) -> dict:
    piece = Path(piece)
    with _LOCK:
        rec = load(piece)
        row = _row(rec, uid)
        v = _next_v(row["stills"])
        dest = _dir(piece, uid) / f"still_v{v:03d}.png"
        shutil.copy2(src, dest)
        item = {"v": v, "file": dest.relative_to(piece).as_posix(), "at": now(), **meta}
        row["stills"].append(item)
        if activate:
            row["active_still"] = v
        save(piece, rec)
    if activate and sync:
        materialize(piece)
    return item


def add_clip(piece: Path, uid: str, src: Path, meta: dict, activate: bool = True,
             sync: bool = True) -> dict:
    piece = Path(piece)
    with _LOCK:
        rec = load(piece)
        row = _row(rec, uid)
        v = _next_v(row["clips"])
        dest = _dir(piece, uid) / f"clip_v{v:03d}{Path(src).suffix or '.mp4'}"
        shutil.copy2(src, dest)
        item = {"v": v, "file": dest.relative_to(piece).as_posix(), "at": now(), **meta}
        row["clips"].append(item)
        if activate:
            row["active_clip"] = v
        save(piece, rec)
    if activate and sync:
        materialize(piece)
    return item


def set_active(piece: Path, uid: str, kind: str, v: int | None) -> None:
    if kind not in ("still", "clip"):
        raise ValueError("kind is still or clip")
    with _LOCK:
        rec = load(piece)
        row = _row(rec, uid)
        if v is not None and not any(x["v"] == v for x in row[kind + "s"]):
            raise ValueError(f"{uid} has no {kind} v{v}")
        row["active_" + kind] = v
        save(piece, rec)
    materialize(piece)


def snapshot_clay(piece: Path, uid: str, sid: str, key: str) -> dict:
    """Copy blockout/<sid>_clay.png (and depth, prompt) under the shot's id."""
    piece = Path(piece)
    blk, d = piece / "blockout", _dir(piece, uid)
    for suffix, name in (("_clay.png", "clay.png"), ("_depth.png", "clay_depth.png"),
                         ("_prompt.txt", "clay_prompt.txt")):
        if (blk / f"{sid}{suffix}").exists():
            shutil.copy2(blk / f"{sid}{suffix}", d / name)
    with _LOCK:
        rec = load(piece)
        row = _row(rec, uid)
        row["clay"] = {"key": key, "at": now(), "file": (d / "clay.png").relative_to(piece).as_posix(),
                       "prompt_file": (d / "clay_prompt.txt").relative_to(piece).as_posix()}
        save(piece, rec)
        return row["clay"]


# -- positional files ---------------------------------------------------------------------------

def _orphan(piece: Path, f: Path) -> None:
    d = piece / "versions" / "_orphans"
    d.mkdir(parents=True, exist_ok=True)
    f.replace(d / f"{datetime.now():%Y%m%dT%H%M%S}_{f.parent.name}_{f.name}")


def materialize(piece: Path) -> list[str]:
    """Make stills/sNNN.png and clips/sNNN.mp4 the active versions of the shot now at position N.
    A positional file with no active version behind it is moved to versions/_orphans, never deleted."""
    piece = Path(piece)
    changes = []
    with _LOCK:
        plan = _plan(piece)
        rec = load(piece)
        mat = rec.setdefault("materialized", {})
        n = len(plan.get("shots") or [])
        for kind, folder, ext in (("still", "stills", ".png"), ("clip", "clips", ".mp4")):
            (piece / folder).mkdir(exist_ok=True)
            want = {}
            for i, s in enumerate(plan.get("shots") or []):
                row = rec["shots"].get(s.get("id") or "")
                item = active(row, kind) if row else None
                want[f"{folder}/{sid_of(i)}{ext}"] = (s.get("id"), item)
            for rel, (uid, item) in want.items():
                f = piece / rel
                if item is None:
                    if f.exists():
                        _orphan(piece, f)
                        changes.append(f"orphaned {rel}")
                    mat.pop(rel, None)
                    continue
                m = mat.get(rel)
                if f.exists() and m and m[:2] == _stat(f) and m[2:] == [uid, item["v"]]:
                    continue
                shutil.copy2(piece / item["file"], f)
                mat[rel] = _stat(f) + [uid, item["v"]]
                changes.append(f"{rel} <- {uid} v{item['v']}")
            for f in sorted((piece / folder).glob(f"s*{ext}")):
                rel = f"{folder}/{f.name}"
                if rel not in want and f.stem[1:].isdigit() and int(f.stem[1:]) >= n:
                    _orphan(piece, f)
                    mat.pop(rel, None)
                    changes.append(f"orphaned {rel}")
        save(piece, rec)
    return changes


def sync_external(piece: Path) -> list[str]:
    """A positional still or clip written by something else (a CLI run, the batch stills stage)
    becomes a new active version of the shot at that position, so nothing is overwritten unseen."""
    piece = Path(piece)
    found = []
    plan = _plan(piece)
    for i, s in enumerate(plan.get("shots") or []):
        uid = s.get("id")
        if not uid:
            continue
        for kind, rel in (("still", f"stills/{sid_of(i)}.png"), ("clip", f"clips/{sid_of(i)}.mp4")):
            f = piece / rel
            if not f.exists():
                continue
            with _LOCK:
                m = load(piece).get("materialized", {}).get(rel)
            if m and m[:2] == _stat(f):
                continue
            meta = {"kind": "external", "note": f"found at {rel}"}
            if kind == "still":
                meta["shot_key"] = still_key(plan, s)
                item = add_still(piece, uid, f, meta, sync=False)
            else:
                row = load(piece)["shots"].get(uid) or {}
                st = active(row, "still") if row else None
                meta["still_ref"] = st["v"] if st else None
                item = add_clip(piece, uid, f, meta, sync=False)
            found.append(f"{rel} -> {uid} {kind} v{item['v']}")
    if found:
        # once, after every file is adopted: materializing after each one would move the
        # not-yet-adopted positional files of later shots to _orphans
        materialize(piece)
    return found


def import_piece(piece: Path) -> dict:
    """First open of a piece: give shots ids and adopt what is on disk as version 1."""
    piece = Path(piece)
    plan = _plan(piece)
    if ensure_ids(plan):
        (piece / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    with _LOCK:
        rec = load(piece)
        rec.setdefault("imported", now())
        save(piece, rec)
    for i, s in enumerate(plan["shots"]):
        sid, uid = sid_of(i), s["id"]
        if (piece / "blockout" / f"{sid}_clay.png").exists():
            snapshot_clay(piece, uid, sid, clay_key(plan, s))
    sync_external(piece)
    return load(piece)


# -- what the studio shows ----------------------------------------------------------------------

def shot_info(piece: Path, plan: dict) -> list[dict]:
    rec = load(piece)
    out = []
    for i, s in enumerate(plan.get("shots") or []):
        uid = s.get("id") or ""
        row = rec["shots"].get(uid) or {"stills": [], "active_still": None, "clips": [],
                                        "active_clip": None, "clay": None}
        st, cl = active(row, "still"), active(row, "clip")
        ck, sk = clay_key(plan, s), still_key(plan, s)
        still_stale = bool(st and st.get("shot_key") and st["shot_key"] != sk)
        if st and st.get("kind") == "edit" and st.get("base_key"):
            still_stale = st["base_key"] != sk
        clip_stale = []
        if cl and st and cl.get("still_ref") not in (None, st["v"]):
            clip_stale.append("the active still changed")
        if cl and not st:
            clip_stale.append("no active still")
        if cl and still_stale:
            clip_stale.append("its still is stale")
        out.append({"i": i, "sid": sid_of(i), "id": uid, "mode": schema.shot_mode(s),
                    "est_seconds": round(schema.est_seconds(s.get("narration", ""))
                                         + float(s.get("tail_s") or 0), 2),
                    "clay": row.get("clay"), "clay_stale": bool(row.get("clay") and ck and row["clay"]["key"] != ck),
                    "stills": row["stills"], "active_still": row["active_still"],
                    "clips": row["clips"], "active_clip": row["active_clip"],
                    "still_stale": still_stale, "clip_stale": clip_stale,
                    "still_key": sk, "clay_key": ck})
    return out
