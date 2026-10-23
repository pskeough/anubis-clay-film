"""The director's motion plan to an H3 prompt in the official guide's structure, and its lint.

A shot whose plan carries a `motion_plan` (see motion_plan.py) gets its H3 prompt from here instead
of animate_shots.prompt_for: stages.motion_one calls compile_prompt and refuses the clip on a lint
error.
"""
from __future__ import annotations

from . import lint, motion_plan, translate


class MotionPlanError(ValueError):
    pass


def shot_view(comp: dict, mp: dict) -> dict:
    """The direction-shot fields the translator and lint read, from a compiled shots.json row.

    `caption` is the director's description of the approved still. It wins over the still prompt
    (`scene`), because H3 must be told what the still shows, not what was asked for.
    """
    roko = comp.get("roko") or {}
    return {"still": mp.get("caption") or comp.get("scene", ""),
            "subject": comp.get("grounded_in") or comp.get("subject") or "",
            "camera": {"distance": roko.get("distance"), "placement": roko.get("placement")}}


def compile_prompt(comp: dict, end: bool = False, seconds: float | None = None, bible: dict | None = None,
                   strand: dict | None = None) -> tuple[str, list[tuple[str, str, str]]]:
    """(prompt, findings) for a compiled shot with a motion_plan. Findings are (level, rule, message)."""
    mp = comp["motion_plan"]
    shot = shot_view(comp, mp)
    ctx = {"mode": "fl2va" if end else "i2va", "seconds": seconds, "style_bible": bible or {},
           "strand": strand or {}, "voiceover": not mp.get("speaks")}
    ctx["scene"] = translate.plan_scene(mp, shot)
    prompt = translate.translate(mp, shot, ctx)
    found = [("error", "plan-schema", m) for m in motion_plan.validate(mp)]
    found += lint.lint_prompt(prompt, {**ctx, "still_text": shot["still"]})
    found += lint.lint_plan(mp, shot, ctx, prompt)
    return prompt, found


def checked_prompt(comp: dict, **kw) -> str:
    """compile_prompt, raising MotionPlanError on any lint error (warnings pass)."""
    prompt, found = compile_prompt(comp, **kw)
    errs = [f"{r}: {m}" for lv, r, m in found if lv == "error"]
    if errs:
        raise MotionPlanError(f"shot {comp.get('i')}: motion plan does not compile:\n- " + "\n- ".join(errs))
    return prompt
