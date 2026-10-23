"""Where the pieces of the pipeline live. Every root can be overridden by environment variable,
set in the shell or in a .env at the repo root (envfile.py; the shell wins)."""
import os
import sys
from pathlib import Path

from . import envfile

envfile.load()


def _p(env: str, default: str) -> Path:
    return Path(os.environ.get(env) or default)


PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
ROKO = _p("CLAY_FILM_ROKO", str(ROOT / "filmprep"))
LIB = _p("CLAY_FILM_LIB", str(ROOT / "pipeline_lib"))
FARM = _p("CLAY_FILM_FARM", str(ROOT / "farm"))
PIECES = _p("CLAY_FILM_PIECES", str(ROOT / "pieces"))
MARKS = ROOT / "marks"
# director mode (clay_film.direct): one folder per adaptation, each holding its direction.json
ADAPTATIONS = _p("CLAY_FILM_ADAPTATIONS", str(ROOT / "adaptations"))
# the director's knowledge base (clay_film.director_kb): markdown, one topic per file. Optional;
# not shipped with the repo, and only `director_kb index/ask/brief` need it.
DIRECTOR_KB = _p("CLAY_FILM_DIRECTOR_KB", str(ROOT / "director_kb"))

# animate_shots and the farm run on system Python; TTS and assembly need the podcast venv.
SYS_PY = os.environ.get("CLAY_FILM_PY") or sys.executable
PODCAST_PY = _p("CLAY_FILM_PODCAST_PY", sys.executable)

LOCATIONS = ROKO / "locations"
BIBLE = ROKO / "bible" / "story-bible.json"
