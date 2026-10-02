/**
 * Instruction builder for image-EDIT models (Qwen-Image-Edit, Kontext-family).
 *
 * This exists because the first hand-written Qwen instruction taught the model to do the wrong thing, and
 * nothing stopped it. That prompt opened with "Image 1 is a grey untextured 3D layout render", then spent
 * three sentences on DO NOT CHANGE against one vague sentence of "replace with real materials". Edit
 * models are trained to make the MINIMUM change that satisfies the instruction, so a prompt weighted like
 * that is an instruction to preserve — and it did, faithfully, including preserving the grey plastic. The
 * geometry was excellent and the surfaces never became photographic.
 *
 * The rules below are therefore enforced in code rather than left to whoever writes the next harness:
 *
 *   1. NEVER name the input as CG. Saying "3D render" / "clay" / "untextured" tells the model the scene
 *      is made of those things, and it will helpfully keep them. Describe the input as a photograph that
 *      is missing detail — same pixels, opposite instruction.
 *   2. The CHANGE clause leads and carries the specifics. Preservation gets ONE line, about the camera.
 *   3. Materials come from the scene graph, per visible surface, not from an adjective. Every prop in
 *      space.json already has a `desc` like "a fluted stone column" — naming them is strictly more
 *      information than "real materials", and it is information we already have.
 *   4. Split the jobs across passes. One instruction that converts materials AND adds a character AND
 *      relights is three instructions, and the model will do the cheapest one.
 *
 * `lintEditPrompt` is the backstop: it hard-fails on the CG vocabulary and warns when the prompt drifts
 * back toward preservation-heavy phrasing. Call it before every queue.
 */

/** Phrases that tell an edit model its input is computer graphics. Never send these. */
const CG_WORDS = [
  "3d render", "3-d render", "3d layout", "layout render", "clay render", "clay model",
  "untextured", "grey box", "gray box", "greybox", "graybox", "blockout", "block-out",
  "massing model", "cg render", "viewport", "wireframe", "primitives", "grey plastic", "gray plastic",
];

export interface EditLint { ok: boolean; errors: string[]; warnings: string[] }

/**
 * Check an edit instruction against the rules above.
 *
 * CG vocabulary is an error, not a warning: it is the specific mistake that cost a full round of Qwen
 * renders, and there is no case where naming the input as CG helps an edit model.
 */
export function lintEditPrompt(prompt: string, visible?: { desc: string }[]): EditLint {
  const errors: string[] = [];
  const warnings: string[] = [];
  const low = prompt.toLowerCase();

  // Naming an object is asking for it. A light "source" is also a physical fixture, so describing the
  // light as "a green-glass banker lamp" put a lamp in a frame the solver says has no lamp in it.
  // When the caller supplies the visible set we can check that automatically instead of hoping.
  if (visible) {
    const seen = visible.map(v => (v.desc ?? "").toLowerCase()).join(" | ");
    for (const w of FIXTURE_WORDS) {
      // The exemption has to match how lightClause actually phrases an out-of-frame fixture, not a
      // guessed substring — the first version looked for "<noun> standing just outside", which never
      // matched once the fixture had any words after the noun ("bulb hanging from the ceiling").
      const excused = /out of frame/i.test(prompt) || low.includes(`no ${w}`);
      if (low.includes(w) && !seen.includes(w) && !excused) {
        warnings.push(`names "${w}" but no visible prop matches it — it will be drawn into frame; `
          + `use lightClause() or drop it`);
      }
    }
  }

  for (const w of CG_WORDS) {
    if (low.includes(w)) errors.push(`names the input as CG: "${w}" — describe it as a photograph instead`);
  }

  // Count preservation phrasing against change phrasing. The failure mode is not "mentions preservation",
  // it is "mentions preservation so much that preserving is the cheapest way to satisfy the prompt".
  const keeps = (low.match(/\bdo not\b|\bdon't\b|\bmust stay\b|\bkeep the same\b|\bunchanged\b|\bexactly the same\b/g) ?? []).length;
  const changes = (low.match(/\breplace\b|\badd\b|\bconvert\b|\brender\b|\bmake\b|\bturn\b|\blight\b/g) ?? []).length;
  if (keeps > changes) {
    warnings.push(`preservation phrases (${keeps}) outnumber change phrases (${changes}) — the model will preserve`);
  }
  if (keeps > 3) warnings.push(`${keeps} preservation phrases; one line about the camera is enough`);
  if (!/photograph/i.test(prompt)) warnings.push("does not assert the output is a photograph");

  return { ok: errors.length === 0, errors, warnings };
}

/** Throwing variant, for use right before queueing work. */
export function assertEditPrompt(prompt: string, label = "edit prompt"): void {
  const l = lintEditPrompt(prompt);
  if (!l.ok) throw new Error(`${label} rejected:\n  - ${l.errors.join("\n  - ")}`);
}

export interface VisibleProp { desc: string; sx: number; sy: number; mid: number; cover?: number }

const third = (v: number) => (v < 0.34 ? "on the left" : v > 0.66 ? "on the right" : "in the centre");

/**
 * Turn the solver's visible-prop list into per-surface material language.
 *
 * Deduplicated by description: a colonnade is eight `col_*` props and reads as noise if listed eight
 * times, but "fluted stone columns on the left" is exactly the sentence that stops the model inventing
 * what those grey cylinders are.
 */
export function materialsClause(visible: VisibleProp[], limit = 8): string {
  const seen = new Map<string, VisibleProp>();
  for (const p of visible) {
    if (!p.desc) continue;
    const k = p.desc.toLowerCase().trim();
    // Keep the nearest instance of each distinct thing: it is the one the eye reads material off.
    const prev = seen.get(k);
    if (!prev || p.mid < prev.mid) seen.set(k, p);
  }
  return [...seen.values()]
    .sort((a, b) => a.mid - b.mid)
    .slice(0, limit)
    .map(p => `${p.desc} ${third(p.sx)}`)
    .join(", ");
}

/**
 * Light fixtures, which are ALSO objects.
 *
 * The bug this prevents: the light was passed as "a single green-glass banker lamp" and the models
 * dutifully drew a green-glass banker lamp — into a frame the solver says the lamp is not in. There IS a
 * lamp prop in this location, it was simply off-camera for that shot, and naming a fixture in a prompt is
 * indistinguishable from asking for it.
 *
 * So: name the fixture only when the solver reports it visible; otherwise describe what the light DOES.
 */
export function lightClause(fixture: string, visible: VisibleProp[]): string {
  const descs = visible.map(v => (v.desc ?? "").toLowerCase());
  // Match on the fixture's distinctive noun rather than the whole phrase, since the prop desc and the
  // spec wording rarely agree word for word ("a green-glass banker lamp, lit" vs "a banker lamp").
  const noun = fixture.toLowerCase().replace(/^(a|an|the)\s+/, "").split(/[\s,]+/).filter(w => w.length > 3).pop() ?? "";
  const inFrame = noun.length > 0 && descs.some(d => d.includes(noun));
  // "standing just outside the frame" was the first phrasing and it assumed a floor lamp — applied to a
  // ceiling bulb it produced "a bare bulb hanging from the ceiling standing just outside the frame",
  // which is nonsense the model has to resolve somehow. Neutral wording works for any mounting.
  return inFrame
    ? `Light the room with ${fixture}, visible in frame`
    : `The room is lit by ${fixture}, which is OUT OF FRAME — its light reaches in, but do not draw the fixture itself`;
}

/** Fixture nouns that are objects as well as light sources. Naming one asks for it to be drawn. */
const FIXTURE_WORDS = ["lamp", "chandelier", "sconce", "lantern", "candle", "bulb", "torch", "window", "skylight", "fireplace"];

export interface CameraFacts { heightM: number; fovDeg: number }

/** One line, ground truth from the solver. A VLM reading the render got the height wrong tenfold. */
export function cameraLine(c: CameraFacts): string {
  const h = c.heightM;
  const where = h < 0.4 ? `${Math.round(h * 100)} cm above the floor, looking slightly up`
    : h > 2.2 ? `${h.toFixed(1)} m up, angled down`
      : `${h.toFixed(2)} m, normal standing eye height, horizon level`;
  return `The camera is at ${where}, with a ${Math.round(c.fovDeg)}-degree field of view.`;
}

export interface Stage1Opts {
  /** Per-surface materials, from materialsClause(). */
  materials: string;
  camera: CameraFacts;
  /** Free prose about the place, from the location's `look`. */
  look?: string;
  /** e.g. "a single green-glass banker lamp". */
  light: string;
  mood?: string;
}

/**
 * Pass 1 — clay to an empty photographic room.
 *
 * No character on purpose. Adding a person and inventing every surface at once are two jobs, and an edit
 * model asked for both will do whichever is cheaper. It also makes the output a REUSABLE background
 * plate: the same room, shot once, with figures added per shot downstream.
 */
export function stage1Prompt(o: Stage1Opts): string {
  const l = [
    "This is a photograph of a real room that has come out badly: flat, colourless, and missing all",
    "surface detail. Restore it into a properly exposed, photorealistic photograph of that same room.",
    "",
    "Give every surface its real material and texture:",
    o.materials + ".",
  ];
  if (o.look) l.push("", o.look);
  l.push(
    "",
    `${o.light}, with the falloff, shadows and reflections that light would really cast.`,
    o.mood ? `${o.mood}.` : "",
    "",
    "Keep the camera and the layout exactly as they are:",
    cameraLine(o.camera),
    "",
    // Emptiness stated POSITIVELY, because negation is the one thing this instruction cannot rely on.
    //
    // It used to read "The room is empty — no people." Two figures walked into campus-courtyard
    // anyway. Nothing was stopping them: `EDIT_NEGATIVE` never mentioned people, and on the Z-Image
    // path cfg is 1, where the negative prompt is inert by design (see scenespec RENDER). So the only
    // guard was a negated noun inside the POSITIVE prompt, and naming a thing in order to exclude it
    // is the classic way to summon it.
    //
    // Describing the absence as a property of the place gives the model something to render instead
    // of something to suppress. "Deserted" and "long abandoned" are scene facts; "no people" is an
    // instruction the sampler has no mechanism to obey.
    "Deserted, long abandoned, not a living soul in sight — every surface undisturbed and settled "
      + "with dust. Sharp focus, fine surface detail, no text or watermarks.",
  );
  return l.filter(s => s !== "").join("\n");
}

export interface Stage2Opts {
  subject: string;
  wardrobe: string;
  action: string;
  /** Where in frame, in plain words. */
  placement: string;
  light: string;
  mood?: string;
  /**
   * The room, described briefly, even though it is already in the input image.
   *
   * Added after the first Qwen->Qwen chain produced a neon colour blowout with the figure lost. The
   * likely cause was not the chain but the setup: that pass sampled from an EMPTY latent, so the model
   * was regenerating the whole frame, while the instruction had been written to say as little about the
   * room as possible ("leave the room alone"). It had neither the pixels nor a description to work from,
   * so it invented. Pair this with latentFromRef on a PHOTO input — the earlier finding that latent-init
   * fails was measured on CLAY input, where preserving the input was the failure; here it is the goal.
   */
  materials?: string;
}

/**
 * Pass 2 — add the figure to an already-photographic plate.
 *
 * The input is now a photograph, so the model's minimum-change instinct protects a correct room instead
 * of protecting grey plastic. That inversion is the whole reason the chain works, and it is why this
 * prompt can afford to be almost entirely about the person.
 */
export function stage2Prompt(o: Stage2Opts): string {
  return [
    `This is a photograph of a room: ${o.materials ?? "a dim interior"}.`,
    "",
    `Add one person to it: ${o.subject}, wearing ${o.wardrobe}.`,
    `She is ${o.action}, positioned ${o.placement}.`,
    "",
    "Integrate her properly into the scene: match the existing light direction and colour temperature,",
    "cast correct contact shadows on the floor and any surface she touches, and match the perspective and",
    `scale of the room. ${o.light}.`,
    o.mood ? `${o.mood}.` : "",
    "",
    "Keep the room exactly as photographed — same camera, same furniture, same walls, same colours.",
    "Photorealistic, sharp focus, natural skin texture. No text or watermarks.",
  ].filter(s => s !== "").join("\n");
}

/**
 * Single-pass variant — materials, lighting and figure in one instruction.
 *
 * Kept because the reviewed Qwen arms got the geometry right in one shot and it is worth knowing how much
 * of the surface problem was the prompt rather than the pass count. Same rules: change clause first and
 * specific, preservation reduced to the camera.
 */
export function singlePassPrompt(o: Stage1Opts & Stage2Opts): string {
  const l = [
    "This is a photograph of a real room that has come out badly: flat, colourless, and missing all",
    "surface detail. Restore it into a properly exposed, photorealistic photograph of that same room.",
    "",
    "Give every surface its real material and texture:",
    o.materials + ".",
  ];
  if (o.look) l.push("", o.look);
  l.push(
    "",
    `${o.light}, with the falloff, shadows and reflections that light would really cast.`,
    "",
    `Add one person: ${o.subject}, wearing ${o.wardrobe}, ${o.action}, positioned ${o.placement},`,
    "lit by the same source and casting correct contact shadows.",
    o.mood ? `${o.mood}.` : "",
    "",
    "Keep the camera and the layout exactly as they are:",
    cameraLine(o.camera),
    "",
    "Photorealistic, sharp focus, fine surface detail, natural skin texture. No text or watermarks.",
  );
  return l.filter(s => s !== "").join("\n");
}

export interface PersonSpec {
  /** Canon appearance, verbatim from the story bible. Do not paraphrase per shot — byte-identical
   *  subject text across frames is the only consistency lever this stack has. */
  appearance: string;
  /** What they are doing, this frame. */
  action: string;
  /** Where in frame, in plain words, derived from the solved figure position. */
  placement: string;
}

/**
 * Two people in one frame, in a single pass.
 *
 * A separate builder rather than a flag on the others, because the single-subject prompts hardcode "she"
 * and because a two-hander needs the RELATIONSHIP stated — two people described independently tend to
 * come back as two people ignoring each other in the same room.
 *
 * Ordering matters and is the lesson from the chain failure: the people lead. When the materials clause
 * was put first in the stage-2 prompt, the room description dominated and the figure was dropped from the
 * output entirely — twice, at both CFGNorm settings. So the room gets one line here, at the end.
 */
export function twoPersonPrompt(o: {
  a: PersonSpec;
  b: PersonSpec;
  /** How they relate this frame, e.g. "mid-conversation, the older man speaking". */
  between: string;
  materials: string;
  camera: CameraFacts;
  light: string;
  look?: string;
  mood?: string;
}): string {
  const l = [
    "This is a photograph of a real room that has come out badly: flat, colourless, and missing all",
    "surface detail. Restore it into a properly exposed, photorealistic photograph, and place two people",
    "in it.",
    "",
    `PERSON ONE: ${o.a.appearance}. ${o.a.action}, ${o.a.placement}.`,
    `PERSON TWO: ${o.b.appearance}. ${o.b.action}, ${o.b.placement}.`,
    "",
    `They are ${o.between}. Both are standing on the floor of this room at the correct scale for the`,
    "perspective, lit by the same light as everything else, casting contact shadows where they meet the",
    "floor. Two people only — no one else in the frame.",
    "",
    `The room: ${o.materials}.`,
  ];
  if (o.look) l.push(o.look);
  l.push(
    "",
    `${o.light}.`,
    o.mood ? `${o.mood}.` : "",
    "",
    "Keep the camera and the layout exactly as they are:",
    cameraLine(o.camera),
    "",
    "Photorealistic, sharp focus, fine surface detail, natural skin texture, correct hands.",
    "No text or watermarks.",
  );
  return l.filter(s => s !== "").join("\n");
}

/**
 * Refer to a reference slot the way the model actually sees it.
 *
 * From ComfyUI's own `comfy_extras/nodes_qwen.py`: TextEncodeQwenImageEditPlus emits literal
 * `Picture 1: <vision tokens> Picture 2: ... Picture 3: ...` ahead of the prompt text. So the slots do
 * have names and in-prompt addressing is real — but the name is "Picture N". Our first attempt said
 * "Image 2 is a reference photograph", which asks the model to resolve a label it was never given.
 */
export const pictureRef = (n: number) => `Picture ${n}`;

/**
 * Instruction for character-sheet reference slots.
 *
 * Kept separate from the scene prompt so the slot vocabulary lives in one place, and phrased to make the
 * DIVISION of labour explicit: the same encoder squeezes every reference to a 384x384 vision budget, so
 * the model has limited pixels per reference and needs telling which one is for what rather than being
 * left to blend them.
 */
export function referenceClause(scene: number, people: { slot: number; who: string }[]): string {
  const l = [
    `${pictureRef(scene)} is the room: use it ONLY for the camera, the layout and where things sit.`,
  ];
  for (const p of people) {
    l.push(`${pictureRef(p.slot)} is a reference photograph of ${p.who}: match that face, build, hair and clothing exactly.`);
  }
  l.push("Do not blend the people together and do not swap their clothing.");
  return l.join("\n");
}

/** Negative prompt for edit passes. Names the CG failure explicitly — safe here, it is what we DON'T want. */
export const EDIT_NEGATIVE =
  "3d render, clay render, untextured, grey plastic, flat shading, video game screenshot, "
  + "blurry, soft focus, low detail, washed out, deformed hands, extra limbs, text, watermark";
