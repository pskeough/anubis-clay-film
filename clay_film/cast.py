"""Who can appear, and the one description each person gets in every still.

The description is byte-identical in every shot a person appears in. That is the only identity
control the stills stage has: the settled recipe is clay alone into Qwen-Image-2.1, with no
reference image of the person (see the roko-scene-render ledger).

Sources, later wins: the Roko story bible's cast, then a piece's own `cast` block in plan.json.
Clauses offering alternatives ("barefoot or in scuffed black shoes") are dropped from bible text,
because the still model resolves an either/or differently in each shot.
"""
import json

from . import paths


def _clean(text: str) -> str:
    keep = [c for c in text.split(", ") if " or " not in c]
    return ", ".join(keep).strip().rstrip(".")


def bible() -> dict[str, str]:
    if not paths.BIBLE.exists():
        return {}
    b = json.loads(paths.BIBLE.read_text(encoding="utf-8"))
    out = {}
    for k, v in (b.get("cast") or {}).items():
        if isinstance(v, dict) and v.get("kind") == "human" and v.get("appearance"):
            out[k] = _clean(v["appearance"])
    return out


def registry(plan_cast: dict | None = None) -> dict[str, str]:
    reg = bible()
    reg.update({k: v.strip().rstrip(".") for k, v in (plan_cast or {}).items()})
    return reg
