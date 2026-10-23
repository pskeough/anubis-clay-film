"""Read a repo-root .env into os.environ. Stdlib only and no package imports, so farm/config.py
can load this file by path under any Python without importing clay_film. paths.py and
voice_booth/server.py call it on import; farm.ps1 and phase.ps1 carry a PowerShell copy of the rules.

    KEY=value                 unquoted; a " #" starts a comment, backslashes are kept (Windows paths)
    KEY="value with # and \\n"  double quotes: \\n \\t \\" \\\\ are unescaped, the rest is literal
    KEY='value'               single quotes: literal
    export KEY=value          the export prefix is allowed
    # comment / blank line    skipped

A variable already in the process environment is never overridden, so the shell (and a parent
process that loaded the file first) always wins. An empty value leaves the variable unset, so a
copied .env.example with blank lines changes nothing. A value starting with ./ or ../ is read
relative to the .env file's folder (the repo root), not the working directory, because the stages
run their subprocesses from other folders.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ESC = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
_loaded: set[Path] = set()


def _value(raw: str) -> str:
    raw = raw.strip()
    if raw[:1] in ("'", '"'):
        q = raw[0]
        out, i = [], 1
        while i < len(raw) and raw[i] != q:
            if q == '"' and raw[i] == "\\" and i + 1 < len(raw):
                out.append(_ESC.get(raw[i + 1], "\\" + raw[i + 1]))
                i += 2
                continue
            out.append(raw[i])
            i += 1
        return "".join(out)   # anything after the closing quote (a comment) is dropped
    m = re.search(r"\s#", raw)
    return (raw[:m.start()] if m else raw).rstrip()


def parse(text: str) -> dict[str, str]:
    """KEY -> value for every well-formed line; malformed lines are skipped."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, eq, raw = line.partition("=")
        key = key.strip()
        if eq and _KEY.fullmatch(key):
            out[key] = _value(raw)
    return out


def load(path: Path | str | None = None) -> dict[str, str]:
    """Apply the .env at `path` (default: repo root). Returns what was set. Safe to call twice."""
    f = Path(path) if path else ROOT / ".env"
    f = f.resolve()
    if f in _loaded or not f.is_file():
        return {}
    _loaded.add(f)
    applied = {}
    for k, v in parse(f.read_text(encoding="utf-8-sig")).items():
        if k in os.environ or v == "":
            continue
        if v.startswith(("./", ".\\", "../", "..\\")):
            v = str((f.parent / v).resolve())
        os.environ[k] = v
        applied[k] = v
    return applied
