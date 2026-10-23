"""Build MiniMax H3 prompts in the format the SHIPPED WORKFLOW uses.

Rewritten after reading the prompt inside ComfyUI's own H3 template
(`comfyui_workflow_templates_json/templates/video_minimax_h3_i2v.json`). The
previous version of this file invented a labelled-section format out of the
prose docs:

    integrated_multimodal_description: [Shot 1] ...
    overall_soundscape: ...
    non_diegetic_music: ...

The shipped template uses none of that. Its prompt is natural prose:

    Editorial tech product film. The transparent gaming mouse from <Picture 1>
    in its original scene: a pitch-black studio void with a dark, subtle
    reflective surface, lit by dramatic duotone vibrant blue and warm neon
    orange rim lighting... The environment is constant throughout.
    SHOT 1: The scene opens exactly on image 1, the mouse resting confidently
    on the dark surface; the blue and orange lights slowly pulse brighter,
    refracting deeply through the transparent acrylic shell as the camera
    executes a slow, deliberate push-in to reveal the intricate circuitry.
    SHOT 2: Cut to an extreme macro profile of the ridged scroll wheel...
    Audio: deep pulsing sub-bass room tone, sharp tactile mechanical clicks...

Four things that matters for:

1. STYLE LEADS. "Editorial tech product film." is the first thing in the
   prompt, before any subject.
2. The reference is "<the subject> from <Picture 1> in its original scene:",
   NOT an instruction to preserve anything. Continuity is pinned with "The
   environment is constant throughout" - a statement about the SETTING.
   The old builder wrote "keeping every subject's appearance, clothing,
   position and the spatial layout unchanged", which reads as an instruction
   to hold still, and measured exactly that: 0.936 px of motion where the
   camera move alone should have produced several times more.
3. Motion is described as something the camera DOES - "the camera executes a
   slow, deliberate push-in TO REVEAL the intricate circuitry" - with a
   reason. Not "pushes in with small amplitude at slow speed", which is the
   guide's taxonomy read out loud and asks for the smallest move available.
4. Audio is one "Audio:" line at the end. There is no separate music field.

H3 also cuts BETWEEN SHOTS inside a single generation - the template's example
runs three - which nothing in this pipeline has used.
"""

from __future__ import annotations

import re

# How the template phrases camera work: a verb, an adverb pair, and a PURPOSE.
# Keyed off the director's own prose. `reveal` completes "...to {reveal}", which
# is what makes the move read as motivated rather than as a slider value.
FROM_DIRECTOR = [
    (r"push|forward|toward|into the frame",
     ("executes a slow, deliberate push-in", "close on")),
    (r"pull back|pull out|shrink|continuous pull",
     ("pulls steadily back", "open out onto the wider scene around")),
    (r"track(ing)? to the right|lateral.*right|drift.*right",
     ("glides steadily to the right", "carry the frame across")),
    (r"track(ing)? to the left|lateral.*left|drift.*left",
     ("glides steadily to the left", "carry the frame across")),
    (r"crane down|tilt down|down from above|over the top",
     ("cranes down from above", "settle onto")),
    (r"rise|crane up|up the face|from the ground",
     ("cranes up", "rise past")),
    (r"arc|around it|orbit",
     ("orbits slowly around the subject", "turn the far side into view past")),
    (r"follow|behind the",
     ("tracks steadily behind the subject", "stay with")),
    (r"drift|across the surface",
     ("drifts steadily across", "sweep along")),
    (r"static|holds|still",
     ("holds steady", "hold on")),
]
DEFAULT_MOVE = ("executes a slow, deliberate push-in", "close on")


def camera_clause(director_motion: str, subject: str = "the subject") -> str:
    """"the camera executes a slow, deliberate push-in to close on X"."""
    m = (director_motion or "").lower()
    verb, purpose = DEFAULT_MOVE
    for pat, pair in FROM_DIRECTOR:
        if re.search(pat, m):
            verb, purpose = pair
            break
    return f"the camera {verb} to {purpose} {subject}"


def i2va(scene: str, motion: str, *, style: str = "Live-action cinematic film",
         action: str = "", soundscape: str = "", music: str = "",
         subject: str = "the subject", shots: list[str] | None = None,
         move_target: str = "", end_frame: bool = False, hold: str = "") -> str:
    """One I2V prompt in the shipped template's shape.

    `move_target` is what the camera move travels toward, when that is not the
    subject. Measured 2026-09-24 (basilisk_day2_demo s001): an over-shoulder
    push aimed at a distant figure was written "push-in to close on the kneeling
    young man", and H3 cut to a close-up of his face inside the clip.

    `shots`, when given, become additional "SHOT n: Cut to ..." blocks - H3
    generates the cuts itself inside the one clip. Left empty, this is a
    single continuous take.
    """
    scene = " ".join((scene or "").split()).rstrip(". ")
    subject = (subject or "the subject").strip()
    head = (f"{style.rstrip('. ')}. {subject[0].upper()}{subject[1:]} from "
            f"<Picture 1> in its original scene: {scene}. "
            f"The environment is constant throughout.")

    develop = " ".join((action or "").split()).rstrip(". ") or (
        "the subject continues the action already under way")
    # end_frame: a last keyframe is pinned (MiniMaxH3ImageToVideo last_frame). Saying so lets the
    # text and the keyframe agree. hold: a sentence restating the look at the end of the shot,
    # because the style line at the top alone lost to the content words on 2026-10-02 (oceanus).
    opens = ("opens exactly on image 1 and ends exactly on image 2" if end_frame
             else "opens exactly on image 1")
    shot1 = (f"SHOT 1: The scene {opens}; {develop}, as "
             f"{camera_clause(motion, (move_target or subject).strip())}.")
    if hold:
        shot1 += f" {hold.strip().rstrip('.')}."

    body = [head, shot1]
    if not shots:
        body.append("The whole clip is one continuous take on this framing, "
                    "with no cuts.")
    for i, extra in enumerate(shots or [], start=2):
        body.append(f"SHOT {i}: Cut to {extra.strip().rstrip('.')}.")

    sound = " ".join((soundscape or "").split()).rstrip(". ") or (
        "quiet room tone, faint movement of air, the soft sound of fabric")
    if music:
        sound = f"{sound}; {music.strip().rstrip('.')}"
    body.append(f"Audio: {sound}.")
    return "\n".join(body)


if __name__ == "__main__":
    print(i2va(
        "a close portrait of an older bearded man in a flat cap smoking a "
        "wooden pipe, deep lines in his skin, grey beard",
        "a slow forward push into the frame",
        subject="the bearded man",
        action="smoke curls upward from the pipe bowl and drifts across his "
               "cheek, he draws once on the pipe and the ember flares",
        soundscape="faint wind over open ground, the soft crackle of tobacco, "
                   "one slow audible breath"))
