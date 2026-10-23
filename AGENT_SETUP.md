# Agent setup

Open a coding agent (Claude Code or similar) in the repository folder and give it the prompt below. It sets everything up, asks before large downloads, and ends with a report of what is ready and what is missing. It renders nothing.

The same steps work by hand: read the prompt as a checklist.

---

```text
You are setting up Scriptorium, a local text-to-film pipeline, in the current folder (a clone of
https://github.com/pskeough/scriptorium). Work through the steps in order. Do not render stills or
clips and do not start long GPU jobs. Ask me before any download over 1 GB and before installing
anything outside this folder. Never put keys in a file git tracks (.env is gitignored; check it).
Finish with the report in step 9.

1. Machine.
   - Run `nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv` and report each
     card, its VRAM, total RAM and the Windows page file size.
   - The pipeline was built on 2x 16 GB cards, 32 GB RAM, Windows 11. Stills need one 16 GB card.
     H3 motion (21 GB checkpoint) runs on a worker spanning two cards; on one card it needs
     CLAY_FILM_H3_SINGLE_GPU=1, FARM_EXTRA_ARGS=--disable-pinned-memory and a large page file.
     Tell me which layout this machine supports, and which stages can run if neither fits
     (plan, compile, clay, stills, assemble and the voice booth all run without H3).
   - On Linux or macOS, say so: farm/farm.ps1 and farm/phase.ps1 are PowerShell and would need porting.

2. Base install.
   - Run `powershell -ExecutionPolicy Bypass -File setup.ps1`. It makes .venv, installs torch (CUDA
     index cu128, right for RTX 50-series; pass -CudaIndex for older cards after checking the
     driver's CUDA version), installs requirements.txt, runs npm install in filmprep, writes .env
     and runs `python -m clay_film selftest`.
   - If Python 3.12+, Node 20+ or ffmpeg is missing, tell me and stop at that point.
   - Confirm `.venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"` prints True.

3. ComfyUI.
   - Look for an install: COMFY_ROOT if set, then C:\ComfyUI. farm/config.py expects
     COMFY_ROOT\venv\Scripts\python.exe and COMFY_ROOT\main.py. If the layout differs (portable
     build, other venv), report it and ask before editing anything.
   - If there is none, ask me, then install the current ComfyUI release into a venv at that layout.
   - Check each node the graphs in farm/graphs.py use by searching ComfyUI's source and
     custom_nodes for the class name, and report found or missing with the defining file:
       core: UNETLoader, CLIPLoader (types "qwen_image" and "minimax"), VAELoader, LoadImage,
         KSampler, KSamplerSelect, BasicScheduler, BasicGuider, RandomNoise,
         SamplerCustomAdvanced, VAEDecode, SaveImage, EmptyLatentImage, LoraLoaderModelOnly
       Qwen-Image 2.1: TextEncodeQwenImage21, QwenImage21Cache
       MiniMax H3: MiniMaxH3ImageToVideo
       BlockSparseAttention (used by motion unless --no-sparse; find which package provides it)
       VHS_VideoCombine from https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite (install if missing)
       UNETLoaderDisTorch2MultiGPU from ComfyUI-MultiGPU (only for the two-card H3 worker)
   - Set COMFY_ROOT in .env if it is not C:\ComfyUI.

4. Model weights. File names must match exactly. First list which are already present under
   COMFY_ROOT\models (or COMFY_EXTRA_MODELS). Show me the licences and the total (about 58 GB),
   and wait for my yes. URLs: https://huggingface.co/<repo>/resolve/main/<path>.
     Comfy-Org/Qwen-Image-2.1
       diffusion_models/qwen_image_2.1_int8_convrot.safetensors              7.26 GB
       text_encoders/qwen3vl_8b_w4a8.safetensors                             6.31 GB
       vae/qwen_image_2.1_vae_bf16.safetensors                               0.68 GB
     Comfy-Org/MiniMax-H3
       diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors    20.97 GB
       text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors           15.69 GB
         (save as qwen3vl32b_h3_nvfp4.safetensors, the name the graphs load)
       vae/minimax_h3_video_vae_fp16.safetensors                             5.21 GB
       loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors       1.96 GB
   Each goes in the models subfolder its loader reads (diffusion_models, text_encoders, vae, loras).
   Licences: Qwen Research licence (Qwen-Image 2.1), MiniMax H3 Community License (H3).
   farm/config.py also lists Z-Image and Qwen-Image-Edit-2509 files for older graphs. Do not
   download them; preflight will report them missing, which is expected.

5. Narration.
   - Guide narration (optional) uses the audio.cpp CLI with qwen3-tts-12hz-1.7b-base-q8_0.gguf and
     qwen3-tts-voicedesign-q8_0.gguf. Their source is not recorded here: find them, show me the
     sources, and ask before building or downloading. Set AUDIOCPP_CLI, QWEN3TTS_BASE_GGUF and
     QWEN3TTS_VOICEDESIGN_GGUF in .env.
   - Kokoro and faster-whisper fetch their own weights on first use.
   - CAPTION_FONTS_DIR must contain GARABD.TTF (Garamond Bold) for the default caption style. Report
     whether it does.

6. Planner.
   - `python -m clay_film plan` needs an OpenAI-compatible chat endpoint (--endpoint, --model;
     default http://127.0.0.1:8080/v1 and "local"). Tested: Qwen3-14B on llama-server (one card).
     Used for the films: Qwen3.8-27B Q4_K_M, about 17 GB, split across two cards. Ask me which.
   - Report whether anything is listening on the endpoint. Do not start or download a model.

7. Finishing tools (optional; ask me whether I want them). finishing/README.md has the details.
   - retime: ComfyUI-Frame-Interpolation checkout and rife426.pth; set RIFE_ARCH_DIR, RIFE_WEIGHTS.
   - score: the Stable Audio 3 repo in its own venv; Medium weights download on first use. Show me
     the model licence. SA3_MODEL defaults to medium.
   - lipsync: the LatentSync repo in its own venv plus the 1.5 (and optionally 1.6) checkpoints;
     set LATENTSYNC_ROOT (and LATENTSYNC_PY if its venv is elsewhere).
   - Install none of these without my yes; each is several GB.

8. Checks. Run each and report the output:
     .venv\Scripts\python -m clay_film selftest
     .venv\Scripts\python farm\preflight.py
     .venv\Scripts\python -m clay_film validate examples\example_plan.json
     .venv\Scripts\python tests\mcp_handshake.py
   Then a dry run with CLAY_FILM_DRY=1 set in the shell:
     .venv\Scripts\python -m clay_film new setup_test examples\example_plan.json
     .venv\Scripts\python -m clay_film run setup_test --stages compile,narrate,clay,stills
     .venv\Scripts\python -m clay_film status setup_test
   preflight is read-only; it fails if another process holds VRAM and on the unused legacy model
   names. Report those apart from real failures. Dry stills are flat placeholders; that is expected.

9. Report. One table: each step as ready, missing or check, and the one action I need to take for
   each missing item. List every file you downloaded or installed and where. Do not commit.
```
