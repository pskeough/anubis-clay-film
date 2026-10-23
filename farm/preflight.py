"""Preflight for the two-card farm. Read-only: it starts nothing and kills nothing.

Run this before every batch. Each check that fails prints the remediation
command rather than a bare FAIL, because the cost of a bad launch is a wasted
night, not a wasted minute.

    python farm/preflight.py            # human table
    python farm/preflight.py --json     # machine-readable, for the runner
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

OK, WARN, FAIL = "OK", "WARN", "FAIL"


class Report:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, name: str, status: str, detail: str, fix: str = "") -> None:
        self.rows.append({"check": name, "status": status, "detail": detail, "fix": fix})

    @property
    def failed(self) -> bool:
        return any(r["status"] == FAIL for r in self.rows)

    def render(self) -> str:
        w = max((len(r["check"]) for r in self.rows), default=10)
        lines = []
        for r in self.rows:
            lines.append(f"  {r['status']:<4} {r['check']:<{w}}  {r['detail']}")
            if r["fix"]:
                lines.append(f"       {'':<{w}}  -> {r['fix']}")
        return "\n".join(lines)


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((host, port)) == 0


def gpu_state() -> list[dict]:
    """nvidia-smi rather than ComfyUI's /system_stats, because this runs before
    any worker exists."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    cards = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            cards.append({
                "index": int(parts[0]),
                "used_mib": int(parts[1]),
                "total_mib": int(parts[2]),
                "util_pct": int(parts[3]),
                "free_mib": int(parts[2]) - int(parts[1]),
            })
    return cards


def check_install(rep: Report) -> None:
    if not C.COMFY_PYTHON.exists():
        rep.add("comfy venv", FAIL, f"missing {C.COMFY_PYTHON}")
    else:
        rep.add("comfy venv", OK, str(C.COMFY_PYTHON))
    if not C.COMFY_MAIN.exists():
        rep.add("comfy main.py", FAIL, f"missing {C.COMFY_MAIN}")
    else:
        rep.add("comfy main.py", OK, str(C.COMFY_MAIN))


def check_disk(rep: Report) -> None:
    free_gb = shutil.disk_usage(C.PROJECT.anchor).free / 1024**3
    if free_gb < 5:
        rep.add("disk", FAIL, f"{free_gb:.1f} GB free on {C.PROJECT.anchor}",
                "reclaim space; a run writes frames, clips and logs")
    elif free_gb < 25:
        rep.add("disk", WARN, f"{free_gb:.1f} GB free", "thin for an overnight batch")
    else:
        rep.add("disk", OK, f"{free_gb:.1f} GB free")


def check_ports(rep: Report) -> None:
    for w in C.WORKERS:
        if port_in_use(w["port"]):
            rep.add(f"port {w['port']} ({w['name']})", FAIL, "already listening",
                    "another process owns this port; stop it or edit farm/config.py")
        else:
            rep.add(f"port {w['port']} ({w['name']})", OK, "free")
    for port, who in C.FOREIGN_PORTS.items():
        if port_in_use(port):
            level = WARN if port == 8080 else WARN
            rep.add(f"port {port}", level, f"in use: {who}",
                    "the farm does not touch this port; noted so a collision is not a surprise")


def check_gpus(rep: Report) -> None:
    cards = gpu_state()
    if not cards:
        rep.add("gpus", FAIL, "nvidia-smi returned nothing")
        return
    if len(cards) < len(C.WORKERS):
        rep.add("gpus", FAIL, f"{len(cards)} card(s) visible, {len(C.WORKERS)} workers configured")
        return
    for w in C.WORKERS:
        card = next((c for c in cards if c["index"] == w["device"]), None)
        if card is None:
            rep.add(f"gpu{w['device']}", FAIL, "not enumerated")
            continue
        if card["free_mib"] < C.MIN_FREE_MIB:
            rep.add(f"gpu{w['device']}", FAIL,
                    f"only {card['free_mib']} MiB free of {card['total_mib']}",
                    "something else holds this card. If it is the 27B, run "
                    "`launch-dual-terminal.bat stop` (its launcher owns the lifecycle, "
                    "so do not kill llama-server directly)")
        else:
            rep.add(f"gpu{w['device']}", OK,
                    f"{card['free_mib']} MiB free, {card['util_pct']}% util")


def check_models(rep: Report) -> None:
    """Names only; the live enum check happens in verify_worker() once a worker
    is up, because only ComfyUI knows what its folder mapping actually resolves."""
    roots = [
        *([Path(__import__("os").environ["COMFY_EXTRA_MODELS"])] if __import__("os").environ.get("COMFY_EXTRA_MODELS") else []),
        C.COMFY_ROOT / "models",
    ]
    found: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.name not in found:
                found[p.name] = p
    missing = [f"{k}={v}" for k, v in C.MODELS.items() if v not in found]
    if missing:
        rep.add("models on disk", FAIL, f"{len(missing)} missing: " + ", ".join(missing[:4]),
                "download them, or correct farm/config.py MODELS")
    else:
        rep.add("models on disk", OK, f"all {len(C.MODELS)} present")


def verify_worker(port: int, needed: dict[str, tuple[str, str]]) -> list[str]:
    """Validate model names against a LIVE worker's loader enums.

    `needed` maps a config key to (node_class, input_name). Returns a list of
    human-readable problems; empty means clean.
    """
    from comfy_client import ComfyClient  # local import: only needed when live

    client = ComfyClient(port)
    problems: list[str] = []
    cache: dict[tuple[str, str], list[str]] = {}
    for key, (node_class, input_name) in needed.items():
        want = C.MODELS.get(key)
        if not want:
            problems.append(f"{key}: not in config.MODELS")
            continue
        ck = (node_class, input_name)
        if ck not in cache:
            try:
                cache[ck] = client.loader_options(node_class, input_name)
            except Exception as e:  # noqa: BLE001 - reported, not swallowed
                problems.append(f"{node_class}: could not read enum ({e})")
                cache[ck] = []
        opts = cache[ck]
        if opts and want not in opts:
            near = [o for o in opts if o.split(".")[0][:12].lower() in want.lower()][:3]
            hint = f" (close: {near})" if near else ""
            problems.append(f"{key}: {node_class}.{input_name} does not offer {want!r}{hint}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rep = Report()
    check_install(rep)
    check_disk(rep)
    check_ports(rep)
    check_gpus(rep)
    check_models(rep)

    if args.json:
        print(json.dumps({"failed": rep.failed, "checks": rep.rows}, indent=2))
    else:
        print("\nPreflight - ComfyUI farm\n")
        print(rep.render())
        print("\n  " + ("BLOCKED" if rep.failed else "clear to launch") + "\n")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
