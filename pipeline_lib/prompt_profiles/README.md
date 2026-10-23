# Prompt profiles

Each video (or image) model gets a folder here holding its own prompting guide, as data the pipeline reads. To deploy a different model you add a profile; you do not edit the renderer.

```
prompt_profiles/
  minimax_h3/
    profile.json      the rules the builder and linter follow
    guide/            the official documents the profile was written from (keep them; cite them)
```

The vendor guides are not redistributed here (MiniMax H3's docs carry their own licence). For `minimax_h3`, fetch `docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md` from the [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) repo into `minimax_h3/guide/` if you want `writer_brief` to include it. `build` and `lint` read only `profile.json`.

`prompt_profile.py` (one level up) loads a profile and does three jobs:

- `build(profile, shot, ctx)` writes the prompt in the model's own structure.
- `lint(profile, prompt, ctx)` lists every rule the prompt breaks. `animate_shots.prompt_for` refuses to send a prompt that fails lint.
- `writer_brief(profile, shot, ctx)` writes the instructions an LLM or a person needs to write a shot's `video_path`, the per-shot motion text. It includes the guide text and the rules.

## Adding a model

1. Put the vendor's prompt guide(s) and example requests in `guide/`.
2. Write `profile.json`. Copy `minimax_h3/profile.json` and change these keys:
   - `alignment`: the fixed keyframe line per mode (I2V, first+last frame, last frame), with `{seconds:.2f}` / `{last_shot}` placeholders, and whether it must be the first line followed by a blank line.
   - `fields`: the named sections and their order, with the fallback text when a section is empty.
   - `description.order`: where the style, anchor, texture, path, landing and prohibitions go.
   - `camera`: the model's trained camera vocabulary and grammar.
   - `negatives`: whether a negative prompt exists. If it doesn't, exclusions are written into the prompt.
   - `length`: the word range.
   - `rules`: lint checks. Most are generic (`regex_absent`, `word_count_in_range`); model-specific checks are named in `prompt_profile.lint`.
3. If the layout is not sectioned, add a builder to `prompt_profile.build` and name it in `"builder"`.
4. Set `PROFILE` in `animate_shots.py` (or pass a profile per model key) and run a lint pass on a piece before any render.

## Per piece

The look lives in the plan's `style_bible`, which the compiler copies to `config.json`:

```json
"style_bible": {"style": "...", "texture": "...", "prohibitions": "..."}
```

Write these three without content nouns (face, hands, water, sky, window...). A video model paints whatever a lock line names; see the `no-content-in-locks` rule.

Per shot, `video_path` holds what happens between the keyframes, written in the model's camera grammar. Without it the builder falls back to the shot's `motion` + `action`.

## Why

On 2026-10-03, H3 was getting an 80-word free paragraph with no alignment line and no fields, where its guide asks for a structured prompt of about 250-600 words. Between pinned keyframes it drifted toward photorealism.
