"""
tts_prep.py — Turn written text (Markdown reports, chat logs) into clean, speakable
text for local neural TTS engines that have little/no built-in text normalization
(Kokoro, Chatterbox, VibeVoice, etc.).

Two responsibilities:
  1. strip_markdown()      -> remove/convert Markdown & symbols to spoken prose
  2. normalize_for_speech()-> numbers, currency, dates, units, acronyms -> words

`clean()` runs the whole pipeline. `parse_dialogue()` splits a two-host script into
(speaker, line) turns. `to_chunks()` sentence-splits under a char budget for engines
with a per-call length cap.

This is a SAFETY NET. The primary lever is the model writing spoken-style text in the
first place (see the audio-overview SKILL.md). But residual "$4.2B", "e.g.", "##", or a
stray table should never reach the model, so we scrub here too.

CLI:
    python tts_prep.py --in report.md            # print cleaned text
    python tts_prep.py --in report.md --dialogue # print parsed turns
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    from num2words import num2words
except Exception:  # pragma: no cover - dependency is declared in requirements
    num2words = None


# ---------------------------------------------------------------------------
# Number spelling helpers
# ---------------------------------------------------------------------------

def _n(num, to="cardinal"):
    """num2words wrapper that degrades gracefully if the lib is missing."""
    if num2words is None:
        return str(num)
    try:
        return num2words(num, to=to, lang="en")
    except Exception:
        return str(num)


_MAGNITUDE = {
    "k": "thousand", "thousand": "thousand",
    "m": "million", "mn": "million", "million": "million",
    "b": "billion", "bn": "billion", "billion": "billion",
    "t": "trillion", "tn": "trillion", "trillion": "trillion",
}

# Common units -> spoken. Longest keys first so "km" beats "m", "ms" beats "m", etc.
_UNITS = {
    "km/h": "kilometers per hour", "mph": "miles per hour", "m/s": "meters per second",
    "kb": "kilobytes", "mb": "megabytes", "gb": "gigabytes", "tb": "terabytes",
    "kg": "kilograms", "mg": "milligrams", "km": "kilometers", "cm": "centimeters",
    "mm": "millimeters", "ghz": "gigahertz", "mhz": "megahertz", "khz": "kilohertz",
    "hz": "hertz", "ms": "milliseconds", "kw": "kilowatts", "mw": "megawatts",
    "kwh": "kilowatt hours", "wh": "watt hours", "w": "watts",
    "fps": "frames per second", "px": "pixels",
    "°c": "degrees celsius", "°f": "degrees fahrenheit",
}

# Acronyms to SPELL OUT letter-by-letter (spaces let the TTS say each letter).
_SPELL = {
    "AI", "API", "CPU", "GPU", "TTS", "STT", "LLM", "VRAM", "RAM", "SSD", "HDD",
    "URL", "HTML", "CSS", "SQL", "USB", "HDMI", "PDF", "CSV", "JSON", "XML", "HTTP",
    "HTTPS", "FAQ", "CEO", "CTO", "CFO", "GDP", "FBI", "CIA", "IRS", "USA", "UK",
    "EU", "UN", "PhD", "ID", "OS", "IP", "DNS", "VPN", "ROI", "KPI", "SDK", "CLI",
    "GUI", "UI", "UX", "ML", "NLP", "RTX", "GTX", "SSD", "DIY",
}
# Acronyms that are read as a WORD (leave untouched).
_WORDS = {"NASA", "NATO", "LASER", "RADAR", "SCUBA", "PIN", "GIF", "RAM"}

_MONTHS = ["", "January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]

# Abbreviations -> expansion (word-boundary, case-insensitive where safe).
_ABBREV = [
    (r"\be\.g\.", "for example"),
    (r"\bi\.e\.", "that is"),
    (r"\betc\.", "et cetera"),
    (r"\bvs\.?\b", "versus"),
    (r"\bapprox\.", "approximately"),
    (r"\bDr\.", "Doctor"),
    (r"\bMr\.", "Mister"),
    (r"\bMrs\.", "Missus"),
    (r"\bMs\.", "Miz"),
    (r"\bProf\.", "Professor"),
    (r"\bSt\.", "Saint"),
    (r"\bFig\.", "Figure"),
    (r"\bno\.\s*(?=\d)", "number "),
    (r"\bvol\.", "volume"),
]


# ---------------------------------------------------------------------------
# Markdown / symbol stripping
# ---------------------------------------------------------------------------

_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols & pictographs, emoji, supplemental
    "\U00002600-\U000027BF"  # misc symbols & dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicators (flags)
    "\U00002190-\U000021FF"  # arrows
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U00002B00-\U00002BFF"  # misc symbols and arrows
    "]",
    flags=re.UNICODE,
)


def _linearize_table(block_lines):
    """Turn a Markdown table into spoken rows: 'header: cell; header: cell.'"""
    rows = []
    for ln in block_lines:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        # skip the |---|---| separator row
        if all(re.fullmatch(r":?-{2,}:?", c or "-") for c in cells):
            continue
        rows.append(cells)
    if not rows:
        return ""
    header = rows[0] if len(rows) > 1 else None
    out = []
    for r in (rows[1:] if header else rows):
        if header and len(header) == len(r):
            out.append("; ".join(f"{h}: {v}" for h, v in zip(header, r) if v))
        else:
            out.append(", ".join(v for v in r if v))
    return ". ".join(s for s in out if s) + "."


def strip_markdown(text: str) -> str:
    # Normalize newlines and fancy punctuation first.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = (text.replace("’", "'").replace("‘", "'")
                .replace("“", '"').replace("”", '"')
                .replace("—", ", ").replace("–", "-")
                .replace("…", ", ").replace(" ", " "))

    # Remove fenced code blocks entirely.
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"~~~.*?~~~", " ", text, flags=re.DOTALL)

    # Handle tables block-by-block.
    out_lines, table_buf = [], []
    for line in text.split("\n"):
        if "|" in line and line.strip().startswith("|"):
            table_buf.append(line)
            continue
        if table_buf:
            out_lines.append(_linearize_table(table_buf))
            table_buf = []
        out_lines.append(line)
    if table_buf:
        out_lines.append(_linearize_table(table_buf))
    text = "\n".join(out_lines)

    # Images -> drop (keep nothing; alt text is usually decorative).
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    # Links -> keep the link text only.
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    # Bare autolinks / URLs -> drop.
    text = re.sub(r"<https?://[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", " ", text)  # emails

    # Inline code / bold / italic / strikethrough -> inner text.
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)

    # Footnote/citation markers: [^1], [12] -> drop.
    text = re.sub(r"\[\^?\d+\]", " ", text)

    # Strip HTML tags.
    text = re.sub(r"<[^>]+>", " ", text)

    # Process line-level markers; make headings and list items their own sentences.
    cleaned = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            cleaned.append("")
            continue
        if re.fullmatch(r"[-*_]{3,}", s):   # horizontal rule
            continue
        s = re.sub(r"^#{1,6}\s*", "", s)     # heading marker
        s = re.sub(r"^>\s?", "", s)          # blockquote
        s = re.sub(r"^\s*[-*+]\s+", "", s)   # bullet
        s = re.sub(r"^\s*\d+[.)]\s+", "", s) # numbered list
        s = s.strip()
        if s and s[-1] not in ".!?:;,":
            s += "."                          # give headings/bullets a stop
        cleaned.append(s)
    text = "\n".join(cleaned)

    text = _EMOJI_RE.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Number / date / unit normalization (operates on plain prose)
# ---------------------------------------------------------------------------

def _spell_int(s: str) -> str:
    s = s.replace(",", "")
    try:
        return _n(int(s))
    except ValueError:
        return s


def _spell_decimal(m: re.Match) -> str:
    whole, frac = m.group(1), m.group(2)
    whole_w = _spell_int(whole) if whole not in ("", "-") else "zero"
    if whole.startswith("-"):
        whole_w = "negative " + _spell_int(whole.lstrip("-"))
    digits = " ".join(_n(int(d)) for d in frac)
    return f"{whole_w} point {digits}"


def _currency(m: re.Match) -> str:
    num = m.group(1)
    mag = (m.group(2) or "").lower()
    if mag:
        val = num  # e.g. 4.2
        val_words = _spell_number_token(val)
        return f"{val_words} {_MAGNITUDE.get(mag, mag)} dollars"
    if "." in num:
        dollars, cents = num.replace(",", "").split(".")
        cents = (cents + "00")[:2]
        out = f"{_spell_int(dollars)} dollars"
        if cents != "00":
            out += f" and {_spell_int(cents)} cents"
        return out
    return f"{_spell_int(num)} dollars"


def _spell_number_token(tok: str) -> str:
    tok = tok.strip()
    if re.fullmatch(r"-?\d[\d,]*\.\d+", tok):
        m = re.fullmatch(r"(-?\d[\d,]*)\.(\d+)", tok)
        return _spell_decimal(m)
    return _spell_int(tok)


def _year(y: int) -> str:
    return _n(y, to="year")


def normalize_for_speech(text: str) -> str:
    # Abbreviations first (before we touch periods/numbers).
    for pat, rep in _ABBREV:
        try:
            text = re.sub(pat, rep, text, flags=re.IGNORECASE)
        except re.error:
            pass

    # ISO dates YYYY-MM-DD -> "Month Dayth, year".
    def _iso_date(m):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{_MONTHS[mo]} {_n(d, to='ordinal')}, {_year(y)}"
        return m.group(0)
    text = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", _iso_date, text)

    # Times: 3:30pm / 14:05 -> spoken.
    def _time(m):
        h, mn, ap = int(m.group(1)), m.group(2), (m.group(3) or "")
        h_w = _n(h)
        mn_w = "" if mn == "00" else (f"oh {_n(int(mn))}" if int(mn) < 10 else _n(int(mn)))
        ap_w = ""
        if ap:
            ap_w = " " + " ".join(ap.replace(".", "").upper())  # pm -> P M
        return f"{h_w} {mn_w}{ap_w}".replace("  ", " ").strip()
    text = re.sub(r"\b(\d{1,2}):(\d{2})\s*([ap]\.?m\.?)?\b", _time, text, flags=re.IGNORECASE)

    # Currency: $1,234.56 / $4.2B / $50
    text = re.sub(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(K|M|B|T|bn|mn|billion|million|thousand|trillion)?\b",
                  _currency, text, flags=re.IGNORECASE)

    # Percent: 45% / 3.5%
    text = re.sub(r"(\d[\d,]*(?:\.\d+)?)\s?%",
                  lambda m: f"{_spell_number_token(m.group(1))} percent", text)

    # Plain magnitude: 4.2B tokens, 10M users
    text = re.sub(r"\b(\d[\d,]*(?:\.\d+)?)\s?(K|M|B|T|bn|mn|billion|million|thousand|trillion)\b",
                  lambda m: f"{_spell_number_token(m.group(1))} {_MAGNITUDE.get(m.group(2).lower(), m.group(2))}",
                  text)

    # Units: 12km, 16GB, 180W, 5 kg
    unit_alt = "|".join(sorted((re.escape(u) for u in _UNITS), key=len, reverse=True))
    text = re.sub(rf"\b(\d[\d,]*(?:\.\d+)?)\s?({unit_alt})\b",
                  lambda m: f"{_spell_number_token(m.group(1))} {_UNITS[m.group(2).lower()]}",
                  text, flags=re.IGNORECASE)

    # Multipliers: 80x -> "eighty times" ( \b after x auto-excludes 1920x1080 )
    text = re.sub(r"\b(\d[\d,]*)x\b",
                  lambda m: f"{_spell_number_token(m.group(1))} times", text)

    # Ordinals: 1st, 22nd, 3rd
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b",
                  lambda m: _n(int(m.group(1)), to="ordinal"), text, flags=re.IGNORECASE)

    # Numeric ranges: 3-5 -> three to five (avoid dates already handled).
    text = re.sub(r"\b(\d[\d,]*)\s?-\s?(\d[\d,]*)\b",
                  lambda m: f"{_spell_number_token(m.group(1))} to {_spell_number_token(m.group(2))}", text)

    # Standalone years 1500-2099 -> "twenty twenty-six".
    text = re.sub(r"\b(1[5-9]\d{2}|20\d{2})\b",
                  lambda m: _year(int(m.group(1))), text)

    # Decimals then integers (with thousands separators).
    text = re.sub(r"(?<![\w.])(-?\d[\d,]*)\.(\d+)(?![\w])", _spell_decimal, text)
    text = re.sub(r"(?<![\w.])(\d[\d,]*)(?![\w.])",
                  lambda m: _spell_int(m.group(1)), text)

    # Acronyms: spell out selected ones (A P I). Leave dictionary words alone.
    def _acr(m):
        tok = m.group(0)
        if tok in _WORDS:
            return tok
        if tok in _SPELL or (tok.isupper() and 2 <= len(tok) <= 5 and tok not in _WORDS):
            return " ".join(tok)
        return tok
    text = re.sub(r"\b[A-Za-z]{2,6}\b",
                  lambda m: _acr(m) if m.group(0).isupper() else m.group(0), text)

    # Ampersand, slashes as spoken.
    text = re.sub(r"\s?&\s?", " and ", text)
    text = re.sub(r"\b24/7\b", "twenty four seven", text)

    # Collapse stacked punctuation & tidy whitespace.
    text = re.sub(r"([.!?,;:])\1+", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([.!?,;:])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean(text: str) -> str:
    """Full pipeline: markdown -> spoken prose -> normalized numbers/units."""
    return normalize_for_speech(strip_markdown(text))


# ---------------------------------------------------------------------------
# Dialogue parsing + chunking
# ---------------------------------------------------------------------------

_SPEAKER_RE = re.compile(
    r"^\s*(?:\*\*|__)?\s*"
    r"(Speaker\s*\d+|Host\s*[A-Za-z0-9]?|Guest\s*[A-Za-z0-9]?|[A-Z][A-Za-z0-9 .'-]{0,24})"
    r"\s*(?:\*\*|__)?\s*[:\)]\s+(.*\S)\s*$"
)


def parse_dialogue(text: str):
    """Parse 'Speaker 1: ...' / 'Alex: ...' turns.

    Returns list of (speaker_label, spoken_line). Lines without a speaker prefix
    are appended to the previous turn. Returns [] if no speaker labels found.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    turns = []
    found = False
    for raw in text.split("\n"):
        line = raw.rstrip()
        if not line.strip():
            continue
        m = _SPEAKER_RE.match(line)
        if m:
            found = True
            speaker = re.sub(r"\s+", " ", m.group(1).strip())
            turns.append([speaker, m.group(2).strip()])
        elif turns:
            turns[-1][1] += " " + line.strip()
    if not found:
        return []
    # clean each turn's spoken text
    return [(sp, clean(txt)) for sp, txt in turns if clean(txt)]


_SENT_RE = re.compile(r"(.+?(?:[.!?](?=\s|$)|$))", re.DOTALL)


def to_chunks(text: str, max_chars: int = 400):
    """Split cleaned text into sentence-bounded chunks under max_chars.

    Never splits mid-sentence (the #1 cause of TTS looping/repetition). A single
    over-long sentence is split at the nearest comma, else hard-split.
    """
    sentences = []
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if not para:
            continue
        for m in _SENT_RE.finditer(para):
            s = m.group(1).strip()
            if s:
                sentences.append(s)
    chunks, cur = [], ""
    for s in sentences:
        if len(s) > max_chars:
            if cur:
                chunks.append(cur); cur = ""
            parts, buf = [], ""
            for piece in re.split(r"(,\s+)", s):
                if len(buf) + len(piece) > max_chars and buf:
                    parts.append(buf.strip()); buf = ""
                buf += piece
            if buf.strip():
                parts.append(buf.strip())
            chunks.extend(parts)
            continue
        if len(cur) + len(s) + 1 > max_chars and cur:
            chunks.append(cur); cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Preview TTS text preparation.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--in", dest="infile", help="input text/markdown file")
    src.add_argument("--text", help="inline text")
    ap.add_argument("--dialogue", action="store_true", help="parse speaker turns")
    ap.add_argument("--chunks", action="store_true", help="show sentence chunks")
    args = ap.parse_args(argv)

    if args.infile:
        raw = Path(args.infile).read_text(encoding="utf-8")
    elif args.text:
        raw = args.text
    else:
        raw = sys.stdin.read()

    if args.dialogue:
        for sp, line in parse_dialogue(raw):
            print(f"[{sp}] {line}")
    elif args.chunks:
        for i, c in enumerate(to_chunks(clean(raw)), 1):
            print(f"--- chunk {i} ({len(c)} chars) ---\n{c}\n")
    else:
        print(clean(raw))


if __name__ == "__main__":
    main()
