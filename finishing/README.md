# Finishing tools

Three command-line tools for the last pass on an assembled film. Each runs on its own and takes its inputs as arguments or as a small JSON job file. Relative paths in a job file are read from the job file's folder. All three use ffmpeg: set `FFMPEG`, or install `imageio-ffmpeg`, or put ffmpeg on PATH.

| Tool | What it does | Runs in |
|---|---|---|
| `retime.py` | Stretches or trims a clip to an exact frame count, dissolves into it from the shot before, fades its end to black | any Python with numpy; torch for RIFE |
| `score.py` | Generates a score with Stable Audio 3 Medium and mixes it under the narration | the SA3 environment for `generate`, `repaint` and `cue`; numpy only for `mix` |
| `lipsync.py` | Lip-syncs quoted lines to the narration as patches laid over the film | the LatentSync environment (`quotes` needs faster-whisper) |

Every tool takes `--threads` (default 4) and keeps itself to that many cores.

## retime.py

A clip longer than its slot loses its tail. A shorter one is slowed evenly from its first frame to its last, and the new in-between frames are drawn by RIFE. The slow-down stops at x2.2 (`--maxslow`), and any frames still missing hold the last frame.

```
python finishing/retime.py clip shot.mp4 shot_fit.mp4 187 --xfrom previous.mp4 --xfade 12
python finishing/retime.py job sequence.json
```

`--xfade N` dissolves from the previous clip's last frame over the first N frames. `--black N` fades the last N frames down to black. `--interp blend` replaces RIFE with frame blending.

Job file:

```json
{
  "fps": 24,
  "out_dir": "retimed",
  "concat": "sequence.mp4",
  "clips": [
    {"src": "shots/s01.mp4", "frames": 140},
    {"src": "shots/s02.mp4", "frames": 212, "start": 3, "xfade": 12},
    {"src": "shots/s03.mp4", "frames": 180, "xfade": 6, "black": 72}
  ]
}
```

In a job, `xfade` dissolves from the clip before it in the list, as built. `start` and `end` select source frames. `concat` joins the results and checks the frame count.

Install: torch, a checkout of ComfyUI-Frame-Interpolation (https://github.com/Fannovel16/ComfyUI-Frame-Interpolation) for the IFNet definition, and the RIFE 4.26 weights that node uses (`rife426.pth`). ComfyUI itself is not needed.

```
RIFE_ARCH_DIR=<ComfyUI-Frame-Interpolation>/vfi_models/rife     # the folder holding rife_arch.py
RIFE_WEIGHTS=<path>/rife426.pth
RIFE_ARCH_VER=4.26                                              # optional
```

Defaults and why:

- RIFE in-betweens. Frame blending smears hands and faces above about x1.3.
- `--maxslow 2.2`. Slower than that reads as slow motion, so the tool holds the last frame instead.
- CPU by default, so retiming can run while the cards render. Use `--device cuda:0` when a card is free.

## score.py

```
python finishing/score.py generate score_job.json
python finishing/score.py repaint --score score.wav --movements score_movements.json --name climax --prompt "..."
python finishing/score.py cue --prompt "..." --length 14.5 --out title_cue.wav
python finishing/score.py mix --voice film.mp4 --score score.wav --movements score_movements.json \
    --offset 12.5 --cue title_cue.wav --out film_scored.mp4
```

`generate` builds one continuous score on the film's clock. Each movement has its own prompt, and the shared `palette` keeps one sound across all of them. A long movement is built in chunks: each chunk is an inpainted continuation of the last 20 s before it, joined with a 0.5 s crossfade. `stops` cut the music dead for a moment (0.15 s down, `length` s of silence, 2.5 s back up). The score fades out over `fade_out` s and is peak-normalised to 0.95. The tool writes `score.wav` and `score_movements.json`, which `mix` and `repaint` read.

```json
{
  "duration": 1860.0,
  "out": "score.wav",
  "seed": 2310,
  "palette": "low cello, bowed double bass, slow string pads, a continuous sustained drone bed that never stops",
  "negative": "vocals, singing, drums, percussion, silence, pauses, rests, fade out, ending, outro",
  "movements": [
    {"name": "opening", "start": 0,     "prompt": "Muted and intimate, {palette}, 56 BPM", "level": 4},
    {"name": "middle",  "start": 610.5, "prompt": "Creeping tension, {palette}, 56 BPM"},
    {"name": "climax",  "start": 1402,  "prompt": "A slow build to a climax, {palette}, 56 BPM", "level": 4}
  ],
  "stops": [{"at": 1655.2, "length": 1.5}],
  "fade_out": 6
}
```

Optional keys: `chunk` (340), `context` (20), `tail` (12), `hole_db` (-45), `repair_rounds` (3), `max_seconds` (380), and `seed` per movement. `level` is the movement's dB offset in the mix.

`repaint` regenerates one movement, or `--from`/`--to` seconds, with a new prompt. It keeps 20 s of the old score on each side of each window as context, re-cuts any stop inside the range and keeps a copy of the old file. `cue` makes a standalone piece, for example under a title card before the narration starts. Give it the title's length plus the mix's 2 s crossfade.

`mix` treats the voice file's audio as the narration. A video file works too, and its picture is copied into the output. The score is processed in order: EQ and compression, a slow leveller, per-movement levels with 4 s glides, a bed level set against the voice's measured loudness, and ducking while the voice speaks. The output goes through a limiter. Afterwards `mix` prints how far the music sits under the voice, both while it speaks and in the gaps.

Install: the Stable Audio 3 repository (https://github.com/Stability-AI/stable-audio-3, `uv sync` or `pip install -e .`, torch 2.7.1). The Medium weights download from Hugging Face into `HF_HOME` on first use. Read the licence on the model card before you release music made with it. `SA3_MODEL` selects the model (default `medium`). Medium needs a CUDA card.

Defaults and why:

- `--bed 10 --duck 4`: on a test pair this put the music 12 to 13 dB under the voice while it speaks. A bed 18 to 23 dB under could not be heard, and about 12 dB under works.
- `--intro 12`: a title cue 12 dB over the bed carries the opening, then glides down to the bed over the last 2 s before the score starts.
- A compressor runs first in the EQ chain, which then cuts 3 dB around 250 Hz and 6 dB around 2.5 kHz, where the voice has its body and its consonants. Some SA3 takes pulse, with dips of about 10 dB every second or two, and under narration that read as the music cutting out. The compressor fixes it.
- Ducking: voice activity on 10 ms frames, a 150 ms look-ahead, a 0.25 s attack and a 1.5 s release. The music dips just before a line and comes back slowly in the gaps.
- Peaks are caught by `alimiter` at the remux. Scaling the whole mix down to one peak dropped the voice by 3 dB.
- Each chunk is generated 12 s longer than kept, and the extra is dropped, because SA3 ends every piece with a fade.
- `sample_size` is raised to 380 s. SA3's default caps output at 120 s, whatever `duration` asks for.
- Inpaint audio goes to SA3 as a float32 numpy array shaped (samples, channels).
- Quiet stretches are repainted after generation: 1 s windows under -45 dBFS, merged across gaps under 3 s, up to three rounds with new seeds.
- Inpainting in front of a score from a silent lead-in returned silence, so `cue` generates the title cue on its own and `mix` crossfades it into the score.

## lipsync.py

```
python finishing/lipsync.py quotes lipsync_job.json          # needs faster-whisper
python finishing/lipsync.py sync lipsync_job.json [ID ...] [--check-only]
python finishing/lipsync.py check lipsync_job.json
python finishing/lipsync.py preview lipsync_job.json
python finishing/lipsync.py overlay lipsync_job.json --approved approved.json --out film_synced.mp4
```

`quotes` finds each quote's exact first and last word from whisper word times. The rough window comes from the job's `window` or from the subtitle file. With exact spans, the narration around a quote ("she says") is not synced. `sync` finds the quote's speaker by face recognition, so a shot with several people syncs the right mouth. For each run of frames with the speaker on screen, it cuts a square crop around them, masks every other face, runs LatentSync on the crop with that stretch of the voice, and pastes back only a feathered ellipse over the mouth. It writes `ID/patch_rN.mp4` (full frames for the run), `ID/patch_rN.json` (first frame and count), `ID/check.jpg` (speaker boxed green, other faces red) and `ID/mouth_rN.jpg` (mouth before and after). `--check-only` stops after the face matching and check sheet, which is a quick way to confirm the right person is chosen.

`check` correlates the mouth opening with the voice's loudness frame by frame, for the patch and for the original. A good patch scores clearly higher than the original. `preview` writes side-by-side review clips. `overlay` lays the listed patches over the film at their frames and copies the audio. The approved list is a JSON array such as `["q01/patch_r0", "q03/patch_r0"]`.

```json
{
  "video": "film.mp4",
  "audio": null,
  "srt": "film.srt",
  "refs": {"anna": 95.0, "ben": "refs/ben_close.png"},
  "quotes_in": [
    {"id": "q01", "who": "anna", "text": "Did you lock the gate when you came in?"},
    {"id": "q02", "who": "ben", "text": "Every night, the same as always.", "window": [312.0, 318.5]}
  ],
  "quotes": "lipsync/quotes.json",
  "out": "lipsync"
}
```

`refs` gives each speaker one clear, near-frontal face: a time in seconds in the film, or an image. `audio` is an optional separate voice file, with `audio_offset` saying where it starts on the film's clock. By default the film's own audio is used. `quotes` is where `quotes` writes and `sync` reads. It can also be an inline list of `{"id", "who", "t0", "t1"}`.

Install:

```
git clone https://github.com/bytedance/LatentSync      # its own venv; torch, then requirements.txt
                                                        # (insightface is in there; on Windows it builds with MSVC)
huggingface-cli download ByteDance/LatentSync-1.5 whisper/tiny.pt latentsync_unet.pt --local-dir checkpoints
huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints_1.6   # for --version 1.6
```

insightface downloads its `buffalo_l` models into `checkpoints/auxiliary/models` on first use. LatentSync's own face detector reads the same folder. Environment:

```
LATENTSYNC_ROOT=<LatentSync checkout>
LATENTSYNC_PY=<its venv python>         # default: LATENTSYNC_ROOT/.venv
LATENTSYNC_CKPT=<unet checkpoint>       # optional override of the per-version default
INSIGHTFACE_ROOT=<folder holding models/buffalo_l>   # default: LATENTSYNC_ROOT/checkpoints/auxiliary
```

LatentSync needs a CUDA card. Face detection for `--check-only`, `check` and `preview` also runs on the CPU with `--cpu`.

Defaults and why:

- `--sim 0.25`: the same person scored 0.35 to 0.4 across shots, and other people scored under 0.1.
- Other faces are masked out of the crop. On a three-person shot, a whole-frame run synced the largest face, which was the wrong speaker.
- Runs are split around frames that fail LatentSync's own face test (a face at least 50x80 px, aspect 0.2 to 1.5, score 0.5), because one failing frame fails the whole run. Runs of 16 frames or more also lose 2 frames at each end.
- Detection gaps up to 4 frames are bridged, and runs shorter than 12 frames are skipped.
- `--version 1.5 --guidance 1.5 --steps 20 --seed 1247` made the approved patches. 1.6 (512 px) keeps lip edges and teeth sharper and runs about 2.5 times slower. If the checkout's `scripts/inference.py` has a `--vae_slicing` flag, it is passed for 1.6 to fit a 16 GB card.
- Near-profile faces do not sync, and a face around 100 px moves only a little even at guidance 2.5. Choose quotes where the speaker faces the camera.
