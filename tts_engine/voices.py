"""
voices.py — Kokoro-82M voice registry and helpers.

Kokoro voice ids encode language + gender: the first letter is the language
('a'=American English, 'b'=British English), the second is gender (f/m).
Grade is Kokoro's own quality rating (A best). af_heart (grade A) is the default.

Only a curated, high-quality English subset is exposed here; the full pack ships
with the model. Use `list` on the CLI to print them.
"""

# id -> (human label, language_code, gender, quality_grade)
VOICES = {
    # American female
    "af_heart":   ("Heart (US F)",    "a", "female", "A"),
    "af_bella":   ("Bella (US F)",    "a", "female", "A-"),
    "af_nicole":  ("Nicole (US F)",   "a", "female", "B-"),
    "af_aoede":   ("Aoede (US F)",    "a", "female", "C+"),
    "af_kore":    ("Kore (US F)",     "a", "female", "C+"),
    "af_sarah":   ("Sarah (US F)",    "a", "female", "C+"),
    "af_nova":    ("Nova (US F)",     "a", "female", "C"),
    "af_sky":     ("Sky (US F)",      "a", "female", "C-"),
    # American male
    "am_michael": ("Michael (US M)",  "a", "male",   "C+"),
    "am_fenrir":  ("Fenrir (US M)",   "a", "male",   "C+"),
    "am_puck":    ("Puck (US M)",     "a", "male",   "C+"),
    "am_echo":    ("Echo (US M)",     "a", "male",   "C"),
    "am_eric":    ("Eric (US M)",     "a", "male",   "C"),
    "am_liam":    ("Liam (US M)",     "a", "male",   "C"),
    "am_onyx":    ("Onyx (US M)",     "a", "male",   "C"),
    "am_adam":    ("Adam (US M)",     "a", "male",   "F+"),
    # British female
    "bf_emma":    ("Emma (UK F)",     "b", "female", "B-"),
    "bf_isabella":("Isabella (UK F)", "b", "female", "C"),
    "bf_alice":   ("Alice (UK F)",    "b", "female", "C"),
    "bf_lily":    ("Lily (UK F)",     "b", "female", "C"),
    # British male
    "bm_george":  ("George (UK M)",   "b", "male",   "C"),
    "bm_daniel":  ("Daniel (UK M)",   "b", "male",   "C"),
    "bm_lewis":   ("Lewis (UK M)",    "b", "male",   "D+"),
    "bm_fable":   ("Fable (UK M)",    "b", "male",   "C"),
}

# Patrick's pick, set 2026-08-26 after A/B on a real explainer: af_heart. It is the only
# grade-A voice in the pack and audibly cleaner than am_puck (C+), which had been the
# default. Pass --voice am_puck for the male read.
DEFAULT_VOICE = "af_heart"

# Contrasting two-host pairing for dialogue / audio-overview mode: male lead + female co-host.
DEFAULT_DIALOGUE_VOICES = ["am_puck", "af_heart"]


def lang_of(voice: str) -> str:
    """Return the Kokoro lang_code for a voice id (falls back to American English)."""
    info = VOICES.get(voice)
    if info:
        return info[1]
    return voice[0] if voice and voice[0] in ("a", "b") else "a"


def is_known(voice: str) -> bool:
    return voice in VOICES


def describe(voice: str) -> str:
    info = VOICES.get(voice)
    return f"{voice} — {info[0]} [{info[3]}]" if info else voice


def pretty_list() -> str:
    lines = []
    for vid, (label, lang, gender, grade) in VOICES.items():
        lines.append(f"  {vid:12s} {label:16s} grade {grade}")
    return "\n".join(lines)
