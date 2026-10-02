"""Minimal ComfyUI HTTP client. Stdlib only.

Deliberate choices, each with a reason:

* The caller mints the prompt_id. ComfyUI accepts a client-supplied `prompt_id`
  on POST /prompt, which makes submission idempotent: a retry after a network
  blip cannot queue the same job twice.

* Completion is decided by polling, never by a websocket. The websocket stream
  is a live feed, not a replayable log, and it has an open host-RAM leak on
  long-lived clients. It also only emits the `executing/node=None` sentinel
  when client_id was set on the prompt, which is a silent hang if you forget.
  Hundreds of jobs a night is exactly the shape that breaks a socket.

* fetchRetry semantics are lifted from the Roko pipeline: retry network errors
  and 5xx only, never a 4xx. One ECONNRESET there once killed a 49-job run.
"""

from __future__ import annotations

import json
import mimetypes
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

RETRY_BACKOFF_S = (1.5, 3.0, 6.0)


class ComfyError(RuntimeError):
    pass


class JobFailed(ComfyError):
    def __init__(self, prompt_id: str, detail: object):
        self.prompt_id = prompt_id
        self.detail = detail
        super().__init__(f"job {prompt_id} failed: {detail!r}")


class JobTimeout(ComfyError):
    pass


@dataclass
class Output:
    filename: str
    subfolder: str
    type: str
    node_id: str


@dataclass
class JobResult:
    prompt_id: str
    seconds: float
    images: list[Output] = field(default_factory=list)
    videos: list[Output] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @property
    def all_outputs(self) -> list[Output]:
        return self.images + self.videos


class ComfyClient:
    def __init__(self, port: int, host: str = "127.0.0.1", client_id: str | None = None):
        self.base = f"http://{host}:{port}"
        self.client_id = client_id or f"slop-farm-{uuid.uuid4().hex[:8]}"

    # -- transport ----------------------------------------------------------

    def _request(self, path: str, *, data: bytes | None = None,
                 headers: dict | None = None, timeout: float = 60.0,
                 retries: int = len(RETRY_BACKOFF_S)) -> bytes:
        url = self.base + path
        last: Exception | None = None
        for attempt in range(retries + 1):
            req = urllib.request.Request(url, data=data, headers=headers or {})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.read()
            except urllib.error.HTTPError as e:
                body = e.read()[:2000].decode("utf-8", "replace")
                # 4xx is our fault: a bad graph, a missing model. Retrying a
                # malformed graph 4 times just wastes 10 seconds.
                if e.code < 500:
                    raise ComfyError(f"HTTP {e.code} {path}: {body}") from e
                last = ComfyError(f"HTTP {e.code} {path}: {body}")
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = ComfyError(f"{type(e).__name__} on {path}: {e}")
            if attempt < retries:
                time.sleep(RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)])
        raise last or ComfyError(f"unreachable: {path}")

    def _get_json(self, path: str, timeout: float = 60.0, retries: int = 3) -> dict:
        return json.loads(self._request(path, timeout=timeout, retries=retries) or b"{}")

    def _post_json(self, path: str, payload: dict, timeout: float = 60.0) -> dict:
        body = json.dumps(payload).encode("utf-8")
        raw = self._request(path, data=body,
                            headers={"Content-Type": "application/json"},
                            timeout=timeout)
        return json.loads(raw or b"{}")

    # -- health -------------------------------------------------------------

    def reachable(self, timeout: float = 3.0) -> bool:
        try:
            self._request("/system_stats", timeout=timeout, retries=0)
            return True
        except ComfyError:
            return False

    def system_stats(self) -> dict:
        """/system_stats is multi-GPU aware: `devices` is a list with per-device
        vram_total / vram_free. This is how the farm reads a card's state
        without shelling out to nvidia-smi."""
        return self._get_json("/system_stats")

    def devices(self) -> list[dict]:
        return self.system_stats().get("devices", [])

    def object_info(self, node_class: str | None = None) -> dict:
        path = "/object_info" + (f"/{urllib.parse.quote(node_class)}" if node_class else "")
        return self._get_json(path, timeout=120.0)

    def loader_options(self, node_class: str, input_name: str) -> list[str]:
        """The enum a loader will accept, read live. Validating against this
        before a long batch is what stops a 3-hour run dying on job 41 because
        a filename was spelled from memory."""
        info = self.object_info(node_class)
        node = info.get(node_class)
        if not node:
            return []
        req = node.get("input", {}).get("required", {})
        spec = req.get(input_name)
        if isinstance(spec, list) and spec and isinstance(spec[0], list):
            return list(spec[0])
        return []

    def validate_graph(self, graph: dict) -> list[str]:
        """Check a built graph against this worker's live /object_info.

        Catches the two failures that otherwise surface only as an opaque
        HTTP 400 `prompt_outputs_failed_validation` at queue time, after a
        batch has already started: an unknown node class, and a required input
        that was never wired. `ZImageFunControlnet.vae` is required, not
        optional, and omitting it killed four jobs in one run.
        """
        info = self.object_info()
        problems: list[str] = []
        for nid, node in graph.items():
            cls = node.get("class_type")
            spec = info.get(cls)
            if spec is None:
                problems.append(f"node {nid}: unknown class {cls!r}")
                continue
            required = spec.get("input", {}).get("required", {})
            missing = [k for k in required if k not in node.get("inputs", {})]
            if missing:
                problems.append(f"node {nid} ({cls}): missing required "
                                f"{', '.join(missing)}")
        return problems

    # -- inputs -------------------------------------------------------------

    def upload_image(self, path: Path, name: str | None = None,
                     subfolder: str = "", overwrite: bool = True) -> str:
        path = Path(path)
        name = name or path.name
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        boundary = f"----slopfarm{uuid.uuid4().hex}"
        parts: list[bytes] = []

        def field(key: str, value: str) -> None:
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n"
                f"{value}\r\n".encode()
            )

        field("overwrite", "true" if overwrite else "false")
        field("type", "input")
        if subfolder:
            field("subfolder", subfolder)
        parts.append(
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n".encode()
        )
        parts.append(path.read_bytes())
        parts.append(f"\r\n--{boundary}--\r\n".encode())

        raw = self._request(
            "/upload/image",
            data=b"".join(parts),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            timeout=300.0,
        )
        info = json.loads(raw or b"{}")
        sub = info.get("subfolder") or ""
        return f"{sub}/{info['name']}" if sub else info["name"]

    # -- jobs ---------------------------------------------------------------

    def queue(self, graph: dict, prompt_id: str | None = None, front: bool = False) -> str:
        pid = prompt_id or str(uuid.uuid4())
        payload = {"prompt": graph, "client_id": self.client_id, "prompt_id": pid}
        if front:
            payload["front"] = True
        info = self._post_json("/prompt", payload, timeout=120.0)
        errors = info.get("node_errors") or {}
        if errors:
            raise ComfyError(f"graph rejected: {json.dumps(errors)[:2000]}")
        return info.get("prompt_id", pid)

    def history(self, prompt_id: str) -> dict | None:
        data = self._get_json(f"/history/{urllib.parse.quote(prompt_id)}", retries=2)
        return data.get(prompt_id)

    def wait(self, prompt_id: str, timeout: float = 1800.0,
             poll: float = 1.0, on_tick=None) -> JobResult:
        started = time.monotonic()
        while True:
            entry = self.history(prompt_id)
            if entry is not None:
                status = entry.get("status", {})
                if status.get("status_str") == "error" or status.get("completed") is False:
                    if status.get("status_str") == "error":
                        raise JobFailed(prompt_id, _first_error(status))
                if status.get("completed") or entry.get("outputs"):
                    return _collect(prompt_id, entry, time.monotonic() - started)
            elapsed = time.monotonic() - started
            if elapsed > timeout:
                raise JobTimeout(f"job {prompt_id} exceeded {timeout:.0f}s")
            if on_tick:
                on_tick(elapsed)
            time.sleep(poll)

    def run(self, graph: dict, timeout: float = 1800.0, poll: float = 1.0) -> JobResult:
        return self.wait(self.queue(graph), timeout=timeout, poll=poll)

    # -- outputs ------------------------------------------------------------

    def fetch(self, out: Output) -> bytes:
        q = urllib.parse.urlencode(
            {"filename": out.filename, "subfolder": out.subfolder, "type": out.type}
        )
        return self._request(f"/view?{q}", timeout=300.0)

    def save(self, out: Output, dest: Path) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.fetch(out))
        return dest

    # -- housekeeping -------------------------------------------------------

    def free(self, unload_models: bool = True, free_memory: bool = True) -> None:
        """Only for OOM recovery. Calling this in steady state throws away the
        resident model and pays the cold-load cost again on the next job."""
        self._post_json("/free", {"unload_models": unload_models,
                                  "free_memory": free_memory})

    def clear_history(self) -> None:
        """/history accumulates for the lifetime of the process."""
        self._post_json("/history", {"clear": True})


def _first_error(status: dict) -> object:
    """Pull the execution_error payload out of a history status block.

    ComfyUI emits each message as a 2-element pair, ["execution_error", {...}],
    not the 4-tuple this originally unpacked. Because this function only runs
    when a job has ALREADY failed, the ValueError it raised replaced every real
    error message with "not enough values to unpack (expected 4, got 2)" — so
    every failure in this pipeline has been reporting the error handler's own
    crash instead of its cause. Tolerant of any shape now: the point is to
    surface the payload, never to add a second failure on top of the first.
    """
    for msg in status.get("messages", []) or []:
        if isinstance(msg, (list, tuple)) and len(msg) >= 2 \
                and msg[0] == "execution_error":
            return msg[1]
    return status


_VIDEO_EXT = (".mp4", ".webm", ".gif", ".mkv", ".mov", ".avi")


def _collect(prompt_id: str, entry: dict, seconds: float) -> JobResult:
    res = JobResult(prompt_id=prompt_id, seconds=seconds, raw=entry)
    for node_id, node_out in (entry.get("outputs") or {}).items():
        for key in ("images", "gifs", "videos"):
            for item in node_out.get(key, []) or []:
                out = Output(
                    filename=item.get("filename", ""),
                    subfolder=item.get("subfolder", "") or "",
                    type=item.get("type", "output") or "output",
                    node_id=str(node_id),
                )
                if out.filename.lower().endswith(_VIDEO_EXT):
                    res.videos.append(out)
                else:
                    res.images.append(out)
    return res
