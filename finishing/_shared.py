"""Helpers shared by the finishing tools: finding ffmpeg, probing and piping video and audio through it, reading job
files, and capping CPU use. Only the standard library and numpy are needed here, so every tool can import it from
whichever environment it runs in.
"""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np


def ffmpeg():
    """The ffmpeg executable: $FFMPEG, then imageio-ffmpeg's bundled binary, then ffmpeg on PATH."""
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    found = shutil.which("ffmpeg")
    if not found:
        raise SystemExit("ffmpeg not found: set FFMPEG, install imageio-ffmpeg, or put ffmpeg on PATH")
    return found


def run_ff(*args, quiet=True):
    cmd = [ffmpeg(), "-y", *(["-v", "error"] if quiet else []), *map(str, args)]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)


def probe(path):
    """(width, height, fps, duration seconds) of a media file, read from ffmpeg's banner. Width and height are None
    for audio-only files."""
    err = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True,
                         stdin=subprocess.DEVNULL, errors="replace").stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None
    v = re.search(r"Video:.*?, (\d{2,5})x(\d{2,5})", err)
    fps = re.search(r"([\d.]+) fps", err)
    return (int(v.group(1)) if v else None, int(v.group(2)) if v else None,
            float(fps.group(1)) if fps else None, dur)


def read_frames(path, start_frame=0, count=None, fps=None, pix_fmt="bgr24"):
    """Frames of a video as a (n, h, w, 3) uint8 array, frame-exact from start_frame."""
    w, h, src_fps, _ = probe(path)
    fps = fps or src_fps
    args = [ffmpeg(), "-v", "error"]
    if start_frame:
        args += ["-ss", f"{start_frame / fps:.4f}"]
    args += ["-i", str(path), "-fps_mode", "passthrough"]
    if count is not None:
        args += ["-frames:v", str(count)]
    args += ["-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]
    raw = subprocess.run(args, capture_output=True, check=True, stdin=subprocess.DEVNULL).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).copy()


def count_frames(path):
    """Decoded video frame count."""
    err = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
                         capture_output=True, text=True, stdin=subprocess.DEVNULL, errors="replace").stderr
    found = re.findall(r"frame=\s*(\d+)", err)
    return int(found[-1]) if found else 0


class FrameWriter:
    """Pipe raw frames into an H.264 file."""

    def __init__(self, path, w, h, fps, crf=16, pix_fmt="bgr24", threads=4):
        self.p = subprocess.Popen([ffmpeg(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{w}x{h}",
                                   "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf",
                                   str(crf), "-pix_fmt", "yuv420p", "-threads", str(threads), str(path)],
                                  stdin=subprocess.PIPE)
        self.path = path

    def write(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait():
            raise SystemExit(f"ffmpeg failed writing {self.path}")


def write_video(frames, path, fps, **kw):
    h, w = frames[0].shape[:2]
    wr = FrameWriter(path, w, h, fps, **kw)
    for f in frames:
        wr.write(f)
    wr.close()


def read_audio(path, sr, channels=1, start=None, duration=None, af=None):
    """Audio of any media file as float32, shape (samples,) for mono or (samples, channels)."""
    args = [ffmpeg(), "-v", "error"]
    if start is not None:
        args += ["-ss", f"{start:.4f}"]
    if duration is not None:
        args += ["-t", f"{duration:.4f}"]
    args += ["-i", str(path), "-vn"]
    if af:
        args += ["-af", af]
    args += ["-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(args, capture_output=True, check=True, stdin=subprocess.DEVNULL).stdout
    a = np.frombuffer(raw, np.float32).copy()
    return a if channels == 1 else a.reshape(-1, channels)


def write_wav(path, audio, sr):
    """Write float32 audio ((samples,) or (samples, channels)) as a 32-bit float WAV."""
    a = np.ascontiguousarray(audio, np.float32)
    ch = 1 if a.ndim == 1 else a.shape[1]
    p = subprocess.Popen([ffmpeg(), "-y", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", str(ch), "-i", "-",
                          "-c:a", "pcm_f32le", str(path)], stdin=subprocess.PIPE)
    p.stdin.write(a.tobytes())
    p.stdin.close()
    if p.wait():
        raise SystemExit(f"ffmpeg failed writing {path}")


def lufs(path):
    """Integrated loudness (EBU R128) of a file, in LUFS."""
    err = subprocess.run([ffmpeg(), "-nostats", "-hide_banner", "-i", str(path), "-af", "ebur128", "-f", "null", "-"],
                         capture_output=True, text=True, stdin=subprocess.DEVNULL, errors="replace").stderr
    return float(err.rsplit("I:", 1)[1].split("LUFS")[0])


def load_job(path):
    """A JSON job file, with every relative path in it resolved later against the file's own folder."""
    path = Path(path).resolve()
    return json.loads(path.read_text(encoding="utf-8")), path.parent


def resolve(base, p):
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else (base / p).resolve()


def cap_cpu(threads):
    """Keep this process to `threads` cores: thread-count variables for numpy/torch/onnxruntime, and CPU affinity
    when psutil is installed. Call before importing torch or onnxruntime."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, str(threads))
    try:
        import psutil
        n = psutil.cpu_count() or threads
        psutil.Process().cpu_affinity(list(range(min(threads, n))))
    except Exception:
        pass
