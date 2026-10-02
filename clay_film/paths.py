"""Where the pieces of the pipeline live. Every root can be overridden by environment variable."""
import os
import sys
from pathlib import Path


def _p(env: str, default: str) -> Path:
    return Path(os.environ.get(env) or default)


PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
ROKO = _p("CLAY_FILM_ROKO", str(ROOT / "filmprep"))
LIB = _p("CLAY_FILM_LIB", str(ROOT / "pipeline_lib"))
FARM = _p("CLAY_FILM_FARM", str(ROOT / "farm"))
PIECES = _p("CLAY_FILM_PIECES", str(ROOT / "pieces"))
MARKS = ROOT / "marks"

# animate_shots and the farm run on system Python; TTS and assembly need the podcast venv.
SYS_PY = os.environ.get("CLAY_FILM_PY") or sys.executable
PODCAST_PY = _p("CLAY_FILM_PODCAST_PY", sys.executable)

LOCATIONS = ROKO / "locations"
BIBLE = ROKO / "bible" / "story-bible.json"
