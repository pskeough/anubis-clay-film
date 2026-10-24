# Scriptorium

Scriptorium turns a piece of writing into a narrated film on your own machine. An LLM breaks the text into shots, Qwen-Image 2.1 paints a still for each one, MiniMax H3 animates the stills, you record the narration in the voice booth, and ffmpeg puts it together with captions. No image, video or speech model runs in the cloud.

Films made with it:

- *Anubis*: https://youtu.be/npvfbjC5cDI
- *Deucalion*: https://youtu.be/BGRu-16W0GA
- *Gardening*: https://youtu.be/celER261eds (42 minutes, with lip sync and a generated score from `finishing/`)

It is a working tool built for one machine and opened up as it goes. Expect Windows scripts and some rough edges.

## What you need

- Windows 11. The GPU farm is driven by PowerShell.
- An NVIDIA card with 16 GB. It was built on two RTX 5060 Ti 16 GB cards and 32 GB of RAM.
- About 60 GB of model weights, ComfyUI, Python 3.12+, Node 20+ and ffmpeg.

Stills fit on one 16 GB card. The H3 motion checkpoint is 21 GB: by default it runs on one worker that spans both cards. On a single card, set `CLAY_FILM_H3_SINGLE_GPU=1` and `FARM_EXTRA_ARGS=--disable-pinned-memory`, and give Windows a large page file, since the weights are offloaded to system memory. Rough speeds on the machine above: 35 s per still, 5 to 6 minutes per 5 s clip.

## Setup

```
git clone https://github.com/pskeough/scriptorium.git
cd scriptorium
powershell -ExecutionPolicy Bypass -File setup.ps1
```

`setup.ps1` makes a `.venv`, installs torch and the Python requirements, runs `npm install` for the clay staging, writes a `.env` pointing at the venv, and runs the CPU self-test. It downloads no models.

ComfyUI, its custom nodes and the model weights come next. The easy way is to open a coding agent (Claude Code or similar) in the repo folder and give it the prompt in [AGENT_SETUP.md](AGENT_SETUP.md). It checks your cards, finds or installs ComfyUI, lists the weights with sizes and licences, waits for your go-ahead before downloading, and reports what is still missing. The same steps are written out there if you would rather do them by hand.

Settings live in `.env` at the repo root (copied from `.env.example`). Everything has a default. A variable set in your shell wins over the file.

## Making a film

```
.venv\Scripts\activate
python -m clay_film plan my_film --passage-file story.txt --endpoint http://127.0.0.1:8080/v1 --model local
python -m clay_film run my_film --stages compile,narrate,clay,stills,endframes
python -m clay_film run my_film --stages motion
python -m clay_film run my_film --stages assemble
```

Or start from the example: `python -m clay_film new my_film examples/example_plan.json`.

The stages, in order:

1. **plan**: any OpenAI-compatible chat model turns the text into `plan.json`, one entry per shot with its narration, what the still shows, what moves and where the camera is. Every answer is checked against the schema and sent back for repair. `python -m clay_film guide` prints the rules the model gets. You can also write or edit the plan by hand, in the studio, or through the MCP server.
2. **compile**: shots and timing.
3. **narrate**: a Qwen3-TTS guide track that sets each shot's length. Optional once you record your own voice.
4. **clay**: for shots set in a known location, a 3D clay blockout places the people and solves the camera, so the still model gets the layout right.
5. **stills**: Qwen-Image 2.1, one 1344x768 still per shot, from a prompt or as an edit of the clay pass.
6. **endframes**: an optional last frame per shot, for moves that need to land somewhere.
7. **motion**: MiniMax H3 animates each still. A check flags clips where nothing moved and clips where H3 cut to a new view mid-shot.
8. **assemble**: the clips under the narration.

Then record the narration in the voice booth (`python voice_booth/server.py`, http://127.0.0.1:7090). It plays the silent cut with the script, keeps each take, denoises and levels it, and builds captions from the script text with whisper word timing.

Redo one shot with `python -m clay_film shot my_film still|edit|motion|clay s012 --seed 7`. Every version is kept: `versions` lists them and `activate` picks one. `status`, `cuts` and `stop` do what they say.

Other ways in:

- Studio web UI: `python webui/server.py --open` (port 8761). Edit the plan and run stages per shot.
- MCP server for agents: `python -m clay_film.mcp_server`.
- Director mode for longer adaptations: `python -m clay_film direct check|continuity|board|compile|animatic NAME`.
- `CLAY_FILM_DRY=1` fakes every GPU job, for testing without models.

## Finishing

`finishing/` holds three standalone tools used on *Gardening*:

- `retime.py` fits a clip to its narration with RIFE in-betweens, dissolves in from the previous shot, fades to black.
- `score.py` generates a score with Stable Audio 3, one movement per section of the story, and mixes it under the voice with ducking.
- `lipsync.py` lip-syncs dialogue with LatentSync, finding the speaker by face recognition so the right mouth moves in a shot with several people.

Each needs its own extra install. See [finishing/README.md](finishing/README.md).

## Writing shots that H3 can animate

These came from the films above. `python -m clay_film guide` has the full set.

- Describe only what the still shows. H3 invents a cut to anything the text names that the frame does not contain.
- Keep actions at the scale of the frame. Hand business on a small full-length figure produces a cut to a close-up.
- Use a slight push-in and hold. Pans, travelling shots and pull-backs make H3 invent the rest of the room.
- Fill the shot with movement. A beat with nothing to do comes out frozen, and pinning a shot to an end frame tends to give a slow morph between two pictures.
- H3 renders "X fades" as X appearing. Write the appearance and reverse the clip.

## Models

No weights are included. These are the ones the films used; each one can be swapped by changing the names in `farm/config.py` and `farm/graphs.py`.

| Role | Model | Licence |
|---|---|---|
| Planner | Qwen3.8-27B (Q4_K_M) or Qwen3-14B on llama-server; any OpenAI-compatible model | Apache 2.0 |
| Stills | [Qwen-Image 2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1): int8 ConvRot model, Qwen3-VL 8B encoder, VAE | Qwen Research |
| Motion | [MiniMax H3](https://huggingface.co/Comfy-Org/MiniMax-H3): FL2V int8 ConvRot, Qwen3-VL 32B NVFP4 encoder, video VAE, 8-step turbo LoRA | MiniMax H3 Community |
| Guide narration | Qwen3-TTS 1.7B GGUF via audio.cpp | see the model card |
| Captions | faster-whisper small.en | MIT |
| Finishing | Stable Audio 3 Medium, LatentSync 1.5/1.6, RIFE 4.26 | see each model card |

Exact file names and sizes are in [AGENT_SETUP.md](AGENT_SETUP.md).

## Layout

| Folder | What is in it |
|---|---|
| `clay_film/` | the CLI, plan schema, planner, stage runner, versions, cut check, director mode, MCP server |
| `farm/` | ComfyUI workers (one per card), job dispatch, graphs, preflight |
| `pipeline_lib/` | narration, motion and assembly steps |
| `filmprep/` | TypeScript clay staging, two example locations |
| `voice_booth/` | recording app, take editor, captions |
| `webui/` | the studio |
| `finishing/` | retime, score, lip sync |
| `tts_engine/`, `marks/`, `examples/`, `tests/` | narration engine, standing marks per location, an example plan, tests |

## Limits

- The farm scripts assume Windows and are tuned for two 16 GB cards.
- Clay staging covers far and near framing well; very close framing is not reliably held by the still model.
- The two example locations are small. Your own sets go in `filmprep/locations/`.

## Licence

MIT, see `LICENSE`. Model weights carry their own licences.
