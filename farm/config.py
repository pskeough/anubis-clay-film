"""Static configuration for the two-card ComfyUI farm.

Stdlib only, on purpose: this has to run under system Python 3.14 (which has no
torch) as well as under the ComfyUI venv.

Why two processes rather than one process with two devices
----------------------------------------------------------
ComfyUI 0.34.0 added a Windows guard in main.py: when os.name == 'nt' and more
than one NVIDIA GPU is visible, it sets CUDA_VISIBLE_DEVICES=0 and warns. The
cause is ComfyUI issue #15255 (open, reported to NVIDIA), where the second GPU
merely being visible poisons the first GPU's CUDA context on the first host
transfer. Restoring both cards in one process needs
`--cuda-device all --disable-pinned-memory`, which costs async weight offload.

Two processes, each launched with `--cuda-device N`, each see exactly one GPU,
so the guard never fires and pinned memory stays on. This is also the only
layout that scales a CFG=1 workload: MultiGPUCFGSplitNode splits positive and
negative conditioning as its two work units, and comfy/samplers.py sets
uncond_ = None when cond_scale == 1.0, so at CFG 1 there is one work unit and
the second card gets nothing. Every fast route here runs CFG 1.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- roots -----------------------------------------------------------------

PROJECT = Path(__file__).resolve().parent.parent
COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", r"C:\ComfyUI"))
COMFY_PYTHON = COMFY_ROOT / "venv" / "Scripts" / "python.exe"
COMFY_MAIN = COMFY_ROOT / "main.py"

RUNS = PROJECT / "runs"
FARM_STATE = PROJECT / "farm" / "_state"

# --- workers ---------------------------------------------------------------
# 8188 is deliberately avoided. Three other codebases assume sole ownership of
# it (Roko pipeline/src/comfy.ts, a local chat UI, and
# Games\CivAI\LAUNCH_CIVAI.bat), and the latter two will start ComfyUI
# themselves if nothing is listening. The farm takes ports nothing else claims.

WORKERS = [
    {"name": "gpu0", "device": 0, "port": 8190},
    {"name": "gpu1", "device": 1, "port": 8191},
]

# The exception to everything above: ONE process that sees BOTH cards, started
# only by `farm.ps1 start -Big` and never alongside the pair.
#
# It exists because the reasoning in this docstring, while correct, quietly set
# a ceiling: if no process may see two cards, then no model larger than 16 GB
# can ever run, and every H3 checkpoint that fits one card is a pruned 4-bit
# repack. The 21 GB minimax_h3_fl2va_pruned_int8_convrot is the build the turbo
# LoRA's author actually recommends, and ComfyUI-MultiGPU's
# UNETLoaderDisTorch2MultiGPU splits it across the pair.
#
# The cost is real and is the reason this is not the default: --cuda-device all
# means ComfyUI's Windows guard for issue #15255 no longer fires, so
# --disable-pinned-memory is required, and that gives up async weight offload.
# Slower per job, and only one job at a time instead of two.
BIG_WORKER = {"name": "big", "device": "all", "port": 8192}

FOREIGN_PORTS = {
    8188: "shared ComfyUI claimed by Roko / chatui / CivAI",
    8080: "llama-server (the 27B)",
}

# --- launch flags ----------------------------------------------------------
# Deliberately NOT set: --highvram and --gpu-only. Both disable ComfyUI's
# DynamicVRAM allocator (comfy/cli_args.py enables_dynamic_vram returns False
# when either is present), which is the thing that lets a large model fit in
# 16 GB at all. --highvram's help text still reads "keeps them in GPU memory",
# which reads like what you want and is the opposite.
#
# Also not set: --disable-smart-memory and --cache-none, both bad for batch.

BASE_ARGS = [
    "--disable-auto-launch",
    "--listen", "127.0.0.1",
    "--async-offload", "2",
    "--vram-headroom", "0.5",
    # 32 GB of system RAM with two workers means RAM, not VRAM, is the ceiling.
    # --fast-disk prefers disk-backed dynamic loading over unpinned RAM.
    "--fast-disk",
]

# Blackwell allocator fragmentation produced spurious CUDA OOMs on this machine
# before; expandable_segments was the fix.
LAUNCH_ENV = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

# --- health ----------------------------------------------------------------

BOOT_TIMEOUT_S = 240          # cold start pages weights off a full NVMe
POLL_INTERVAL_S = 1.0
JOB_TIMEOUT_S = 1800          # a cold 13 GB GGUF can spend minutes paging

# --- VRAM ------------------------------------------------------------------

CARD_TOTAL_MIB = 16311
# Below this free on a card, a worker will not start.
#
# 12000 rather than 14000 on purpose. The two states this has to separate are
# "the 27B is resident" (tensor-split, ~13 GB per card, leaving 0.5-2.8 GB free)
# and "only light tenants" (display compositing on GPU0 plus Voice_Chat's
# whisper/Kokoro at ~2.8 GB, leaving ~13.6 GB free). 12000 sits cleanly between
# them. A higher floor would refuse to launch alongside Voice_Chat, which is a
# legitimate configuration.
MIN_FREE_MIB = 12000

# --- models, as ComfyUI's loader enums spell them ---------------------------
# Verified present on disk 2026-09-16. preflight.py re-checks these against a
# live /object_info rather than trusting this list.

MODELS = {
    "zimage_unet": "z_image_turbo_nvfp4.safetensors",
    "zimage_clip": "qwen_3_4b.safetensors",
    "zimage_vae": "ae.safetensors",
    "zimage_control": "Z-Image-Turbo-Fun-Controlnet-Union.safetensors",

    "qwen_edit_unet": "Qwen-Image-Edit-2509-Q4_K_M.gguf",
    "qwen_edit_clip": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "qwen_edit_vae": "qwen_image_vae.safetensors",
    "qwen_lightning_4": "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
    "qwen_lightning_8": "Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors",

    # Qwen-Image-2.1 (2026-09-21): one 7B checkpoint for text-to-image AND
    # editing, replacing Qwen-Image-Edit-2509. Qwen3-VL-8B encoder, new VAE,
    # native 2K and up to 10 reference slots. cfg MUST stay 1.0 (negative is
    # ignored), euler/simple, 25 working steps, no Lightning/angle LoRAs (those
    # are 2509-only and will not load). See Image_Edit/QWEN_IMAGE_2.1_PROMPTING.md.
    "qwen21_unet": "qwen_image_2.1_int8_convrot.safetensors",
    "qwen21_clip": "qwen3vl_8b_w4a8.safetensors",
    "qwen21_vae": "qwen_image_2.1_vae_bf16.safetensors",
    # Anything2Real Alpha, built for Qwen-Image-Edit-2509. Trained to
    # push any non-photographic style all the way to photograph while
    # holding composition, which is exactly the clay-to-photo problem:
    # the base model preserves the clay's own quad seams as if they
    # were surface features and returns a CG room.
    "qwen_a2r": "Anything2RealAlpha.safetensors",
    # dx8152/Qwen-Edit-2509-Multiple-angles. Adds camera-move control to
    # this exact base model. Needed because the Qwen-Image technical
    # report states the base "is unable to robustly perform global
    # viewpoint control operations due to the inherent complexity and
    # ambiguity in defining a common reference frame", which is exactly
    # the operation the blockout wants. No trigger word; it responds to
    # plain instructions like "move the camera back" or "turn the
    # camera to a wide-angle lens".
    "qwen_multiangle": "qwen_multiangle.safetensors",

    # Wan is gone. The 5B unet, the 2.2 VAE, the 2.1 VAE, the CausVid LoRA and
    # the 6.4 GB umt5 text encoder are all deleted from disk - 2.2 was thirteen
    # months older than what replaced it and there are no 2.7 weights anywhere
    # to upgrade to. `graphs.wan_i2v` and `wan_i2v_fp8` are left in place as a
    # record but will now fail loudly on a missing key, which is the correct
    # behaviour: a graph whose weights do not exist should not silently build.
}

# moodyProMix_collectorsEdition.safetensors is documented as the Z-Image
# production UNET in roko-scene-render/references/recipes.md and is NOT on
# disk. z_image_turbo_nvfp4 is the substitute in use here.


def worker_dirs(name: str) -> dict[str, Path]:
    """Per-worker output/temp/input, so two instances never collide on a file."""
    base = FARM_STATE / name
    return {
        "output": base / "output",
        "temp": base / "temp",
        "input": base / "input",
        "log": base / "comfy.log",
    }


def live_workers(timeout: float = 2.0) -> list[dict]:
    """Whichever workers are actually answering, big one first.

    Stages used to hardcode WORKERS, which broke the moment the farm layout
    grew a third mode: the pinned pair and the dual-card worker are mutually
    exclusive (three ComfyUI processes do not fit 31.9 GB of host RAM once the
    32B text encoder is in play), so a stage that assumes the pair dispatches
    into nothing whenever the big worker is the one serving.

    Falls back to WORKERS so the failure is still the familiar connection
    error rather than an empty job list that silently renders zero shots.
    """
    import urllib.request

    alive = []
    for w in [BIG_WORKER] + WORKERS:
        try:
            with urllib.request.urlopen(
                    "http://127.0.0.1:%d/system_stats" % w["port"],
                    timeout=timeout) as r:
                if r.status == 200:
                    alive.append(w)
        except Exception:
            continue
    # never mix: if the big worker is up it owns both cards, and handing jobs
    # to the pair as well would have them fighting over the same VRAM
    if alive and alive[0]["name"] == "big":
        return [alive[0]]
    return alive or list(WORKERS)
