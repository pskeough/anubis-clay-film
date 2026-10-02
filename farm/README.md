# farm — two ComfyUI workers, one per card

Both RTX 5060 Ti cards, one ComfyUI process each, driven from Python. Stdlib
only, so it runs under system Python 3.14 as well as the ComfyUI venv.

## Why two processes and not one

Three independent reasons, and any one of them is sufficient.

**ComfyUI already ignores the second card.** Version 0.34.0 added a Windows
guard in `main.py`: when `os.name == "nt"` and more than one NVIDIA GPU is
visible, it sets `CUDA_VISIBLE_DEVICES=0` and warns. The cause is ComfyUI issue
#15255, still open and reported to NVIDIA, where the second card merely being
*visible* poisons the first card's CUDA context on the first host transfer. It
was filed by someone running an RTX 5060 Ti as primary. This install is 0.35.0,
so it is already subject to the guard. Restoring both cards inside one process
needs `--cuda-device all --disable-pinned-memory`, which costs async weight
offload. One GPU per process means the guard never fires and pinned memory
stays on.

**CFG-split parallelism does nothing here.** `MultiGPUCFGSplitNode` ships in
0.35.0 and splits positive and negative conditioning as its two work units.
`comfy/samplers.py` sets `uncond_ = None` when `cond_scale` is 1.0, so at CFG 1
there is exactly one work unit and the second card gets nothing. Its measured
1.85-1.89x on dual 4090s was taken on full-CFG sampling. Every fast route here
runs CFG 1: Z-Image production, Qwen Lightning, Wan Lightning, Causal Forcing,
SVI.

**Layer-splitting crashes on this exact hardware.** ComfyUI-MultiGPU / DisTorch
relies on peer-to-peer DMA, which is disabled at driver level on consumer
GeForce. The CPU-staging fallback aborts with `cudaErrorIllegalAddress` from a
C++ destructor, so it is uncatchable and kills the process. Two open issues
report it verbatim on 2x 5060 Ti. The driver patch that fixes it is Linux-only.

So: **two concurrent jobs, never one fast job.** The workload is hundreds of
independent shots and clips, which makes that the better shape anyway.

## Ports

8190 (GPU 0) and 8191 (GPU 1). **8188 is deliberately avoided.** Three other
codebases assume sole ownership of it — `Roko_Analogue/pipeline/src/comfy.ts`,
a local chat UI, and `Games\CivAI\LAUNCH_CIVAI.bat` — and the
latter two will start ComfyUI themselves if nothing is listening. Sharing that
port would mean three lifecycle owners for one process.

## Usage

```powershell
.\farm\farm.ps1 start      # preflight, launch both, wait for health
.\farm\farm.ps1 status
.\farm\farm.ps1 logs gpu1
.\farm\farm.ps1 stop
```

```powershell
python farm\preflight.py          # read-only; starts and kills nothing
python farm\test_graphs.py        # offline graph structure checks, no GPU
python farm\smoke.py --verify-only
python farm\smoke.py --jobs 4     # 4 jobs on one card, then 4 on two, reports speedup
```

## The GPUs have to be free first

The 27B in tensor-split mode holds ~13 GB on **both** cards, and `Voice_Chat`
takes another ~2.8 GB when it is running. Preflight fails with the exact
remediation rather than launching into a doomed state. Phases are separated in
time:

```
A  27B across both cards      script.json + shotlist.json
B  CPU only                   clay / depth / edge for every shot
C  2x ComfyUI, one per card   stills, then clips
D  CPU only                   assemble, mux, report
```

Stop the 27B through its own launcher (`launch-dual-terminal.bat stop`), never
by killing `llama-server` directly — the launcher owns the lifecycle.

## Files

| File | What it is |
|---|---|
| `config.py` | ports, device map, launch flags, model names. Single source of truth. |
| `farm.ps1` | start / stop / status / restart / logs. PowerShell 5.1, no `&&`. |
| `preflight.py` | ports, GPUs, disk, model presence. Read-only, prints remediation. |
| `comfy_client.py` | HTTP client. Self-minted `prompt_id`, polling, retry, upload, fetch. |
| `graphs.py` | API-format graph builders, in code. Never store UI-format JSON. |
| `dispatch.py` | fan a job list across both workers, manifest after every job. |
| `smoke.py` | verify both workers, then measure one-card vs two-card wall time. |
| `test_graphs.py` | offline structural checks on the builders. |

## Launch flags, and two that are deliberately absent

`--highvram` and `--gpu-only` are **not** set. Both disable ComfyUI's
DynamicVRAM allocator (`enables_dynamic_vram()` returns False when either is
present), which is the thing that lets a large model fit in 16 GB at all.
`--highvram`'s help text still reads "keeps them in GPU memory", which reads
like what you want and is the opposite.

Set instead: `--async-offload 2`, `--vram-headroom 0.5`, and `--fast-disk`,
which prefers disk-backed dynamic loading over unpinned RAM and is aimed at
exactly this 32 GB situation. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
is set per worker; Blackwell allocator fragmentation produced spurious CUDA
OOMs on this machine before and that was the fix.

Also avoided for batch work: `--disable-smart-memory`, `--cache-none`, and
calling `/free` in steady state.

## Things that fail in a way that looks like something else

**Omitting `client_id` on `POST /prompt`** means the websocket done-sentinel
never fires. This client polls and never opens a socket, so it cannot hit that,
but anything else driving ComfyUI here should know.

**Windows Standby List thrashing.** A Wan 2.2 user on an RTX 5060 Ti 16 GB and
Windows 11 reported clean runs for a while, then severe SSD thrashing needing a
reboot. RAMMap showed ~13.3 GB in Standby (Mapped File), matching the loaded
model, with free memory at ~40 MB. Windows was not reclaiming mappings across
repeated loads of one large file. "Empty Standby List" in Sysinternals RAMMap
fixes it without rebooting. Expect this on long unattended runs at 32 GB.

**Recycle the workers every 50-100 jobs.** Several ComfyUI memory leaks are
open. `/history` also accumulates for the process lifetime;
`client.clear_history()` prunes it.

**Sort the batch by model.** `dispatch.py` groups by `model_key` before
dispatch. A different checkpoint, or even a different LoRA, produces a different
patched model and forces a reload. This is worth more than any launch flag.

## Missing weight worth knowing about

`moodyProMix_collectorsEdition.safetensors` is documented as the production
Z-Image UNET in `roko-scene-render/references/recipes.md` and **is not on
disk**. `config.py` points the Z-Image route at `z_image_turbo_nvfp4.safetensors`
instead. Any recipe copied from that ledger needs the same substitution.
