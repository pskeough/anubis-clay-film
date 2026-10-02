# Agent setup

Clone the repository, open a coding agent (Claude Code or similar) in the repository folder, and give it the prompt below. The agent does the setup and reports what is still missing. It does not render anything.

Items marked "check" are things the repository does not pin down. The agent should confirm them on your machine rather than assume.

---

```text
You are setting up Scriptorium, a local text-to-film pipeline, in the current folder (a clone of
https://github.com/pskeough/scriptorium). Do the steps below in order. Do not render stills or
clips and do not start long GPU jobs. Ask me before any download larger than 1 GB and before
installing anything outside this folder. At the end, give me a report.

1. Hardware.
   - Run `nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv`.
   - Report each GPU, its VRAM, and total system RAM.
   - The pipeline was built on 2x 16 GB NVIDIA cards and 32 GB RAM on Windows 11.
     farm/config.py expects two workers on GPUs 0 and 1 (ports 8190, 8191) for stills and
     one worker seeing all GPUs (port 8192) for motion. It refuses to start a worker on a
     card with less than 12000 MiB free (MIN_FREE_MIB).
   - The motion checkpoint is about 21 GB. Tell me if the available VRAM cannot hold it,
     and say which stages can still run (plan, compile, clay, stills, assemble, voice booth).
   - farm/farm.ps1 and farm/phase.ps1 are PowerShell. On Linux or macOS, say so and note that
     the farm start/stop will need porting.

2. Python.
   - Python 3.12 or later. Report the version.
   - Install torch with CUDA support matching the installed driver (check the right index URL
     for the CUDA version nvidia-smi reports).
   - `pip install -r requirements.txt`.
   - Confirm `python -c "import torch; print(torch.cuda.is_available())"` prints True.

3. Node (only needed for clay staging).
   - Node 20 or later. `cd filmprep && npm install`. Confirm `npx tsx --version` works there.

4. ComfyUI.
   - Look for an existing ComfyUI install. Check COMFY_ROOT if set, then C:\ComfyUI.
   - farm/config.py expects ComfyUI's Python at COMFY_ROOT\venv\Scripts\python.exe and
     COMFY_ROOT\main.py. If ComfyUI is installed another way (portable build, Linux venv),
     report the difference; do not edit farm/config.py without asking me.
   - If none exists, ask me before installing ComfyUI. If I agree, install the current
     ComfyUI release into a venv at that layout.
   - The graphs in farm/graphs.py use these nodes. Start nothing; check each one by searching
     ComfyUI's source and custom_nodes folder for the class name:
       Core loaders and samplers: UNETLoader, CLIPLoader (types "qwen_image" and "minimax"),
         VAELoader, LoadImage, KSampler, KSamplerSelect, BasicScheduler, BasicGuider,
         RandomNoise, SamplerCustomAdvanced, VAEDecode, SaveImage, EmptyLatentImage,
         LoraLoaderModelOnly.
       Qwen-Image 2.1: TextEncodeQwenImage21, QwenImage21Cache (check whether these are in
         core ComfyUI for the installed version or need a custom node).
       MiniMax H3: MiniMaxH3ImageToVideo (check: core or custom node for the installed version).
       BlockSparseAttention, used by motion unless --no-sparse is passed (check which package
         provides it).
       VHS_VideoCombine, from ComfyUI-VideoHelperSuite
         (https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite). Install it if missing.
     Report each node as found or missing, with the file that defines it.
   - Not needed by the current stages: UNETLoaderDisTorch2MultiGPU (ComfyUI-MultiGPU) and
     UnetLoaderGGUF (ComfyUI-GGUF). They are used only by older graphs.

5. Model weights. Place each file in the ComfyUI models subfolder its loader reads
   (UNETLoader -> models/diffusion_models, CLIPLoader -> models/text_encoders,
   VAELoader -> models/vae, LoraLoaderModelOnly -> models/loras). The file names must match
   exactly; they come from farm/config.py MODELS and the defaults of
   farm/graphs.py minimax_h3_official.
     Stills (Qwen-Image 2.1):
       qwen_image_2.1_int8_convrot.safetensors      diffusion model
       qwen3vl_8b_w4a8.safetensors                  text encoder
       qwen_image_2.1_vae_bf16.safetensors          vae
     Motion (MiniMax H3):
       minimax_h3_fl2va_pruned_int8_convrot.safetensors              diffusion model
       qwen3vl32b_h3_nvfp4.safetensors                               text encoder (ComfyUI's
         H3 template names this file qwen3vl_32b_minimax_h3_nvfp4_awq; rename it to match)
       minimax_h3_video_vae_fp16.safetensors                         vae
       minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors     lora
   - Download sources are not recorded in the repository (check). Search Hugging Face and
     ComfyUI's model documentation for these exact names, show me the source URL and size of
     each, and wait for my approval before downloading.
   - First list which of these files are already present anywhere under COMFY_ROOT\models or
     COMFY_EXTRA_MODELS.
   - farm/config.py MODELS also lists Z-Image and Qwen-Image-Edit-2509 files. The current
     stages do not use them. Do not download them. farm/preflight.py will report them as
     missing; say so in the report rather than treating it as a failure.

6. Guide narration (Qwen3-TTS). Optional; the voice booth can replace it.
   - The narrate stage calls the audio.cpp CLI with Qwen3-TTS 1.7B GGUF files:
       qwen3-tts-12hz-1.7b-base-q8_0.gguf
       qwen3-tts-voicedesign-q8_0.gguf
   - Where to get audio.cpp and these GGUFs is not recorded in the repository (check). Find
     them, show me the sources, and ask before building or downloading.
   - Kokoro and faster-whisper small.en download their own weights on first use through
     the kokoro and faster-whisper packages.

7. Environment.
   - Copy .env.example to .env. Every value is optional and defaults are repo-relative.
     Set what differs on this machine:
       CLAY_FILM_PIECES, CLAY_FILM_ROKO, CLAY_FILM_LIB, CLAY_FILM_FARM, CLAY_FILM_PY,
       CLAY_FILM_PODCAST_PY, CLAY_FILM_TTS_ENGINE, COMFY_ROOT, COMFY_EXTRA_MODELS,
       AUDIOCPP_CLI, QWEN3TTS_BASE_GGUF, QWEN3TTS_VOICEDESIGN_GGUF, CLAY_FILM_LLM_SCRIPT,
       CAPTION_FONTS_DIR, FFMPEG, CLAY_FILM_API_KEY, OPENROUTER_API_KEY.
   - No code in the repository loads .env; every value is read from the process environment.
     Keep .env as the record of this machine's values, and tell me how to export them in my
     shell (or set them as user environment variables) before running the pipeline.
   - Do not put API keys in any file that git tracks. .env is in .gitignore; confirm that.
   - CAPTION_FONTS_DIR must contain GARABD.TTF (Garamond Bold) for the default caption font.
     Report whether it does.

8. Planner model.
   - `python -m clay_film plan` needs an OpenAI-compatible chat endpoint. The default is
     http://127.0.0.1:8080/v1 with model "local" (tested with Qwen3-14B on llama-server).
     Report whether anything is listening there. Do not start or download a model.

9. Tests. Run each and report the output:
     python -m clay_film selftest
     python farm/preflight.py
     python -m clay_film guide --seconds 30
     python -m clay_film validate examples/example_plan.json
   preflight.py is read-only. It will fail on the GPU check if another process holds VRAM,
   and on unused legacy model names (step 5); report those separately from real failures.

10. Dry run. With CLAY_FILM_DRY=1 set, run:
      python -m clay_film new setup_test examples/example_plan.json
      python -m clay_film run setup_test --stages compile,narrate,clay,stills
      python -m clay_film status setup_test
    The clay stage runs for real (CPU, Node). Without a fixture piece the dry stills are flat
    placeholder images; that is expected. Report any failure.

11. Report. A short table: each item above as ready, missing or check, with the one action
    I need to take for each missing item. List every file you downloaded or installed and
    where. Do not commit anything.
```
