"""Fan a job list across both workers. One thread per worker, shared queue.

Design rules taken from what has already gone wrong on this machine:

* The manifest is rewritten after EVERY job, with explicit {"failed": ...} and
  {"skipped": ...} records. A batch has to be diagnosable from its own record.
  Absence is not a diagnosis, and tracking by file mtime once cost an hour of
  wrong conclusions when a frame from a previous pass was read as stale output
  from the current one.

* Jobs are grouped by `model_key` before dispatch. A different checkpoint, or
  even a different LoRA, produces a different patched model and forces a
  reload. Ordering the batch by model is worth more than any launch flag.

* Every job is wrapped. A 45-shot batch that renders 43 is still a useful
  batch; one hung job must not take the night with it.

* Threads, not processes. The work is HTTP wait, so the GIL is irrelevant, and
  a thread can hold a live client without re-handshaking per job.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from comfy_client import ComfyClient, ComfyError  # noqa: E402


@dataclass
class Job:
    job_id: str
    graph: dict
    model_key: str = "default"
    out_name: str | None = None
    meta: dict = field(default_factory=dict)
    inputs: list[Path] = field(default_factory=list)  # uploaded to the worker first


@dataclass
class Done:
    job_id: str
    worker: str
    status: str                 # ok | failed
    seconds: float = 0.0
    files: list[str] = field(default_factory=list)
    error: str | None = None
    meta: dict = field(default_factory=dict)
    at: str = ""


class Farm:
    # Model-load stagger. See _load_gate for why this exists at all.
    #
    # MIN_FREE_GB 18: the Qwen3-VL-32B encoder reads ~16 GB, plus headroom.
    # STAGGER_S 90: measured ~75 s from submit to the first sampling step on
    #   this box (encoder read, encode, free, DiT page-in), rounded up.
    # WAIT_S 300: after five minutes, stop waiting and let it try. A batch that
    #   refuses to start is a worse failure than one that retries a job.
    LOAD_MIN_FREE_GB = 18.0
    LOAD_STAGGER_S = 90.0
    LOAD_WAIT_S = 300.0

    def __init__(self, out_dir: Path, workers: list[dict] | None = None,
                 job_timeout: float = C.JOB_TIMEOUT_S, verbose: bool = True):
        self.workers = workers or C.WORKERS
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.out_dir / "manifest.json"
        self.job_timeout = job_timeout
        self.verbose = verbose
        self.run_id = "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._lock = threading.Lock()
        # a plain Lock, not RLock: _release_gate runs on a timer thread
        self._load_lock = threading.Lock()
        self._records: list[dict] = []
        self._started = time.monotonic()

    # -- logging ------------------------------------------------------------

    def _say(self, msg: str) -> None:
        if self.verbose:
            with self._lock:
                el = time.monotonic() - self._started
                print(f"[{el:7.1f}s] {msg}", flush=True)

    # -- manifest -----------------------------------------------------------

    def _record(self, rec: dict) -> None:
        with self._lock:
            self._records.append(rec)
            payload = {
                "run_id": self.run_id,
                "updated": datetime.now(timezone.utc).isoformat(),
                "workers": [{"name": w["name"], "device": w["device"], "port": w["port"]}
                            for w in self.workers],
                "jobs": self._records,
            }
            tmp = self.manifest_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.manifest_path)

    # -- health -------------------------------------------------------------

    def live_workers(self) -> list[dict]:
        alive = []
        for w in self.workers:
            client = ComfyClient(w["port"])
            if not client.reachable():
                self._say(f"{w['name']}: not reachable on {w['port']}, skipping")
                continue
            devs = client.devices()
            if len(devs) != 1:
                self._say(f"{w['name']}: WARNING {len(devs)} devices visible, "
                          f"expected 1 - --cuda-device did not take")
            alive.append(w)
        return alive

    # -- run ----------------------------------------------------------------

    def run(self, jobs: list[Job]) -> list[Done]:
        alive = self.live_workers()
        if not alive:
            raise RuntimeError("no live workers; run farm.ps1 start")
        if len(alive) < len(self.workers):
            self._say(f"running DEGRADED on {len(alive)} of {len(self.workers)} workers")

        # Group by model so a worker does not thrash between checkpoints.
        ordered = sorted(jobs, key=lambda j: (j.model_key, j.job_id))
        q: queue.Queue[Job | None] = queue.Queue()
        for j in ordered:
            q.put(j)
        for _ in alive:
            q.put(None)

        results: list[Done] = []
        threads = []
        for w in alive:
            t = threading.Thread(target=self._worker_loop, args=(w, q, results),
                                 name=w["name"], daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        ok = sum(1 for r in results if r.status == "ok")
        bad = len(results) - ok
        total = time.monotonic() - self._started
        self._say(f"done: {ok} ok, {bad} failed, {total:.1f}s wall")
        self._record({"summary": True, "ok": ok, "failed": bad,
                      "wall_seconds": round(total, 1)})
        return results

    def free_ram_gb(self) -> float:
        """Free physical memory, or -1.0 if it cannot be read.

        Stdlib only, because farm/ has to run under system Python as well as
        the ComfyUI venv and psutil is in neither by default.
        """
        try:
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            m = MemStatus()
            m.dwLength = ctypes.sizeof(MemStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return -1.0
            return m.ullAvailPhys / (1024 ** 3)
        except Exception:  # noqa: BLE001 - a gate that cannot read ram must not block
            return -1.0

    def _load_gate(self, name: str) -> None:
        """Let only one worker be in its model-loading phase at a time.

        HOST ram, not VRAM, is what stops both cards working on H3 at once.
        The Qwen3-VL-32B text encoder needs ~16 GB of it to read, this box has
        31.9 GB, and two workers reading it together produced
        `hostbuf_file_reader_read failed`. VRAM was never the problem: each
        card has 16 GB and ComfyUI loads encoder -> frees -> loads DiT.

        So jobs are not serialised - only their LOADS are. A worker takes this
        gate, waits for enough free ram, submits, and holds the gate for
        LOAD_STAGGER_S while the weights page in; the next worker then starts
        roughly a minute and a half behind and the two run concurrently from
        there. Measured on this box: ~75 s from submit to the first sampling
        step, against ~286 s for a whole 124-frame job, so the stagger costs
        about a third of one job once and buys a second card for all of them.

        A gate that cannot read memory does not block: free_ram_gb() returning
        -1.0 falls through, because a farm that refuses to run is worse than
        one that risks a retry.
        """
        if len(self.workers) < 2:
            return
        self._load_lock.acquire()
        try:
            deadline = time.monotonic() + self.LOAD_WAIT_S
            while time.monotonic() < deadline:
                free = self.free_ram_gb()
                if free < 0 or free >= self.LOAD_MIN_FREE_GB:
                    break
                time.sleep(3.0)
            else:
                self._say(f"{name}: ram gate timed out after "
                          f"{self.LOAD_WAIT_S:.0f}s, submitting anyway")
        finally:
            # Released on a timer, NOT here. client.run() blocks for the whole
            # job, so holding the gate across the caller's submit would
            # serialise the entire batch and throw away the second card. The
            # timer releases it once the weights have had time to page in,
            # which is the only part that actually contends.
            # A plain Lock may be released by a thread that did not acquire it;
            # an RLock may not, so this must stay a Lock.
            threading.Timer(self.LOAD_STAGGER_S, self._release_gate).start()

    def _release_gate(self) -> None:
        try:
            self._load_lock.release()
        except RuntimeError:
            pass  # already released; never take the batch down over the gate

    def _drop_worker_copy(self, w: dict, out) -> None:
        """Remove the file ComfyUI wrote, now that ours is saved.

        Never raises: losing a temp file is not worth failing a job that has
        already produced its output, and a permission error here would
        otherwise mark a good render as failed.
        """
        try:
            base = C.FARM_STATE / w["name"] / "output"
            p = base / (out.subfolder or "") / out.filename
            if p.is_file():
                p.unlink()
        except Exception:
            pass

    def _worker_loop(self, w: dict, q: "queue.Queue[Job | None]",
                     results: list[Done]) -> None:
        client = ComfyClient(w["port"], client_id=f"slop-farm-{w['name']}")
        uploaded: set[str] = set()

        while True:
            job = q.get()
            if job is None:
                q.task_done()
                return
            started = time.monotonic()
            try:
                for src in job.inputs:
                    src = Path(src)
                    if src.name not in uploaded:
                        client.upload_image(src)
                        uploaded.add(src.name)

                self._load_gate(w["name"])
                self._say(f"{w['name']} -> {job.job_id}")
                res = client.run(job.graph, timeout=self.job_timeout)

                files: list[str] = []
                for i, out in enumerate(res.all_outputs):
                    stem = job.out_name or job.job_id
                    suffix = Path(out.filename).suffix
                    name = f"{stem}{suffix}" if len(res.all_outputs) == 1 else \
                           f"{stem}_{i}{suffix}"
                    dest = client.save(out, self.out_dir / name)
                    files.append(dest.name)
                    # Drop the worker's own copy once ours is on disk.
                    # ComfyUI writes into farm/_state/<worker>/output and
                    # nothing ever removed it, so every render was stored
                    # twice and the second copy was never read again. Measured
                    # on 2026-09-19: 1.8 GB of orphans across three worker
                    # directories, on a disk that was down to 13 GB free.
                    self._drop_worker_copy(w, out)

                done = Done(job_id=job.job_id, worker=w["name"], status="ok",
                            seconds=res.seconds, files=files, meta=job.meta,
                            at=datetime.now(timezone.utc).isoformat())
                self._say(f"{w['name']} ok {job.job_id} in {res.seconds:.1f}s "
                          f"-> {', '.join(files) or '(no file)'}")
            except (ComfyError, OSError, ValueError) as e:
                done = Done(job_id=job.job_id, worker=w["name"], status="failed",
                            seconds=time.monotonic() - started, error=str(e)[:1200],
                            meta=job.meta, at=datetime.now(timezone.utc).isoformat())
                self._say(f"{w['name']} FAILED {job.job_id}: {str(e)[:300]}")
            except Exception as e:  # noqa: BLE001 - never let one job kill the batch
                done = Done(job_id=job.job_id, worker=w["name"], status="failed",
                            seconds=time.monotonic() - started,
                            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}",
                            meta=job.meta, at=datetime.now(timezone.utc).isoformat())
                self._say(f"{w['name']} CRASHED {job.job_id}: {type(e).__name__}: {e}")

            with self._lock:
                results.append(done)
            self._record({
                "job_id": done.job_id, "worker": done.worker, "status": done.status,
                "seconds": round(done.seconds, 2), "files": done.files,
                "error": done.error, "meta": done.meta, "at": done.at,
            })
            q.task_done()


def main() -> int:
    """Run a JSON job file:  [{"job_id": "...", "graph": {...}, ...}, ...]"""
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", type=Path, help="JSON list of jobs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timeout", type=float, default=C.JOB_TIMEOUT_S)
    args = ap.parse_args()

    raw = json.loads(args.jobs.read_text(encoding="utf-8"))
    jobs = [Job(job_id=j["job_id"], graph=j["graph"],
                model_key=j.get("model_key", "default"),
                out_name=j.get("out_name"), meta=j.get("meta", {}),
                inputs=[Path(p) for p in j.get("inputs", [])]) for j in raw]

    farm = Farm(args.out, job_timeout=args.timeout)
    results = farm.run(jobs)
    return 0 if all(r.status == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
