# Clay Film Studio

A local web UI for building a clay_film piece shot by shot. It runs on this machine only (127.0.0.1:8761).

## Start it

Double-click `ClayFilmStudio.bat` in the clay_film folder. It starts the server with system Python and opens the browser.

From a terminal:

```
python webui/server.py [--port 8761] [--dry] [--open]
```

`--dry` sends every GPU job to a fake backend that copies fixture stills and clips from basilisk_day2_v2. Use it to try the flow without the farm. A dry server and a real server keep separate queues of jobs; neither runs the other's jobs.

## Pages

1. **Input.** Start a piece from a brief, a passage, a copy of another piece, or an existing plan file. Set length, style and the planner endpoint and model.
2. **Script.** One card per shot. Edits save as you type and validation errors show under the field. Split, merge, add, delete and reorder shots. A bar shows how much of the passage the shots cover, at 2.4 words a second. Chat with the planner about one shot or the whole plan; each answer comes back as a diff to accept or reject.
3. **Stills.** A grid of shots. Tick shots to render or re-clay them together. Click a still for the large view next to its clay pass, the version strip, an edit bar (a Qwen-2.1 edit by plain instruction) and a redo bar (new seed, new prompt, or both). Every version is kept and one is active.
4. **Motion.** One row per shot with its still, clip and H3 prompt. A prompt edit to a queued job takes effect when the job starts. A running job cannot change and the page says so.
5. **Assemble.** Narrate, assemble and play the film. Anything that would stop assembly, such as a missing clip, is listed first.

## Queue

Each piece has a queue in `<piece>/_webui/queue.json` that survives a restart. One GPU job runs at a time, and jobs that need the same farm shape run first so the farm restarts less. Up to two stills or edits from one piece go to the pair together. Queued jobs can be cancelled; a running job cannot. The studio stops the farm after 10 minutes idle if it started the farm. The Stop farm button stops it now.

The studio does not start GPU work while another process holds the clay_film GPU lock, while ComfyUI is busy, or while the farm is down and a card is using more than 3.5 GB.

## Staleness

When a plan edit changes what a shot's clay, still or clip was made from, the shot is marked stale with the reason. Nothing is regenerated until you ask.

## Planner key

Set the endpoint and model on the Input page. The key comes from `webui/config.local.json` (written when you save a key on that page), else from `CLAY_FILM_API_KEY`, else `OPENROUTER_API_KEY` for an OpenRouter endpoint. The browser only learns whether a key is set and where it came from. Keys are not written to logs. Keep `config.local.json` out of any copy you share.

## Protected pieces

basilisk_day2_v2 and basilisk_day2_demo are read only here. Copy one to work on it.
