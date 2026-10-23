/**
 * SceneSpec — the contract between a small local model and the image pipeline.
 *
 * The design premise, learned the hard way: a 4B-14B model asked to author image prose will drift the
 * palette, drop the LoRA trigger, invent a location that contradicts canon, and emit camera language
 * we have measured as ineffective. So it does not author prose. It fills in a spec of mostly enums,
 * and the composer in this file turns that spec into the actual prompt.
 *
 * Everything validated on 2026-08-17 lives here as a constant the model never sees:
 * turbo 8 steps / cfg 1 / shift 3, control strength 0.5, the `35mm` LoRA trigger, and the camera
 * phrasings that actually produced the requested angle. Retuning any of these is a one-line edit here
 * rather than a re-prompting exercise.
 *
 * The single most important property: SUBJECT TEXT IS BYTE-IDENTICAL ACROSS EVERY SHOT. That is the
 * only consistency lever available on this checkpoint (Omni reference latents are dead, and a shared
 * seed does not survive a camera change), so the composer emits it verbatim per shot rather than
 * letting anything paraphrase it per frame.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { HOUSE_STYLE, ANATOMY_GUARD, EMPTY_SCENE, lintBackgroundPrompt, PERSON_NOUN_RX } from "./prompts.js";

// fileURLToPath, NOT new URL().pathname: the latter percent-encodes a space in the folder path
// and silently yields a directory that does not exist, so listLocationIds() returned [] and the canon
// enum was empty — which read as the model refusing to pick canon locations.
const LOCATIONS_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "locations");

// --- vocabularies -------------------------------------------------------------
// These are the enum domains handed to the model as a JSON schema. Keep them small: every value
// here must map to a camera phrasing below that we have actually seen work.

export const ANGLES = ["three_quarter", "profile_left", "profile_right", "rear", "over_shoulder", "frontal", "three_quarter_back"] as const;
export const ELEVATIONS = ["eye", "low", "high", "ground", "overhead"] as const;
export const DISTANCES = ["extreme_close", "close", "medium", "full_body", "wide", "very_wide"] as const;
export const LENSES = [18, 24, 35, 50, 85, 135] as const;
export const CONTROLS = ["none", "depth", "pose"] as const;

/**
 * What the subject is DOING. This is the variance dimension that was missing entirely: with only
 * angle/elevation/distance to vary, every frame was a person standing still, photographed from a
 * slightly different side, and a five-shot set read as one photo repeated. Pose changes the silhouette,
 * so it changes the image far more than another 15 degrees of camera yaw does.
 */
export const POSES = [
  "standing", "standing_hands_in_pockets", "arms_crossed", "seated", "seated_hunched", "crouching",
  "kneeling", "leaning_against_wall", "leaning_on_furniture", "walking", "mid_stride", "turning_away",
  "looking_back", "head_down", "head_tilted_back", "reaching", "holding_something", "hands_on_face",
  "lying_down", "sitting_on_floor",
] as const;

/** Where the subject sits in frame. Dead-centre every time is what makes a set look mechanical. */
export const PLACEMENTS = ["center", "left_third", "right_third", "far_from_camera", "close_to_edge", "small_in_frame"] as const;
export const LIGHTS = [
  "daylight_window", "overcast_daylight", "night_sodium", "night_fluorescent",
  "single_bulb", "candlelight", "screen_glow", "dusk",
] as const;
export const MOODS = ["muted_desaturated", "cold_blue", "warm_amber", "sickly_green", "high_contrast_bw"] as const;

export type Angle = typeof ANGLES[number];
export type Elevation = typeof ELEVATIONS[number];
export type Distance = typeof DISTANCES[number];
export type Control = typeof CONTROLS[number];
export type Pose = typeof POSES[number];
export type Placement = typeof PLACEMENTS[number];

export interface ShotSpecLite {
  angle: Angle;
  elevation: Elevation;
  distance: Distance;
  lens: typeof LENSES[number];
  control: Control;
  pose?: Pose;
  placement?: Placement;
  /** One short scene-specific beat, e.g. "pulling a book off a high shelf". Bounded and optional:
   *  it is the only per-shot free text, and it is what makes a set feel like a scene rather than a
   *  turntable. Ignored entirely when there is no subject. */
  action?: string;
}

export interface SubjectSpec {
  /** Who they are, physically. Short — this is not a place for prose. */
  description: string;
  /** Named garments and colours. These are the anchors that make drift legible between frames. */
  wardrobe: string;
}

export interface SceneSpec {
  /** A canon location id, or null to describe a new place in `location_description`. */
  location_id: string | null;
  /** Only used when `location_id` is null. */
  location_description: string;
  /** null = an empty stage (no figure at all). */
  subject: SubjectSpec | null;
  light: typeof LIGHTS[number];
  mood: typeof MOODS[number];
  shots: ShotSpecLite[];
}

// --- camera phrasing ----------------------------------------------------------
// Verbatim from the strings that produced the requested angle in the _zangles run. `rear` in
// particular needs the redundant "facing away from camera" or the model quietly turns the figure
// back around.

const ANGLE_TEXT: Record<Angle, string> = {
  three_quarter:      "turned three-quarters toward camera",
  profile_left:       "full profile facing frame left",
  profile_right:      "full profile facing frame right",
  rear:               "seen from behind, facing away from camera",
  three_quarter_back: "turned three-quarters away from camera, back and shoulder toward the lens",
  over_shoulder:      "seen from behind over the shoulder, glancing back",
  frontal:            "facing camera directly",
};

const ELEVATION_TEXT: Record<Elevation, string> = {
  eye:      "at eye level",
  low:      "low angle looking up",
  high:     "high angle looking down",
  ground:   "camera at floor level looking up",
  overhead: "directly overhead looking straight down",
};

const DISTANCE_TEXT: Record<Distance, string> = {
  extreme_close: "extreme close-up, face filling the frame",
  close:         "close shot, head and shoulders",
  medium:        "medium shot, waist up",
  full_body:     "full body in frame head to feet",
  wide:          "wide shot showing the surroundings",
  very_wide:     "very wide establishing shot, the space dominating the figure",
};

const LENS_TEXT: Record<number, string> = {
  18:  "ultra-wide 18mm lens, extreme perspective distortion",
  24:  "wide 24mm lens, strong perspective",
  35:  "35mm lens, natural perspective",
  50:  "50mm lens, normal perspective",
  85:  "85mm lens, compressed perspective, shallow depth of field",
  135: "135mm telephoto, heavily compressed, very shallow depth of field",
};

const POSE_TEXT: Record<Pose, string> = {
  standing: "standing still",
  standing_hands_in_pockets: "standing with hands in pockets",
  arms_crossed: "standing with arms crossed",
  seated: "seated",
  seated_hunched: "seated and hunched forward",
  crouching: "crouching low",
  kneeling: "kneeling",
  leaning_against_wall: "leaning back against the wall",
  leaning_on_furniture: "leaning forward on the furniture",
  walking: "walking through the space",
  mid_stride: "caught mid-stride",
  turning_away: "turning away from the camera",
  looking_back: "stopped and looking back",
  head_down: "head lowered, looking at the floor",
  head_tilted_back: "head tilted back",
  reaching: "reaching out with one arm",
  holding_something: "holding something in both hands",
  hands_on_face: "one hand raised to the face",
  lying_down: "lying down",
  sitting_on_floor: "sitting on the floor",
};

const PLACEMENT_TEXT: Record<Placement, string> = {
  center: "centred in frame",
  left_third: "positioned in the left third of the frame",
  right_third: "positioned in the right third of the frame",
  far_from_camera: "far from the camera, deep in the space",
  close_to_edge: "near the edge of the frame, partially cropped",
  small_in_frame: "small in the frame, dwarfed by the surroundings",
};

const LIGHT_TEXT: Record<string, string> = {
  daylight_window:   "soft daylight through a window",
  overcast_daylight: "flat overcast daylight",
  night_sodium:      "sodium vapour lighting at night",
  night_fluorescent: "harsh overhead fluorescent light at night",
  single_bulb:       "a single bare bulb overhead",
  candlelight:       "candlelight",
  screen_glow:       "lit only by the glow of a screen",
  dusk:              "failing light at dusk",
};

const MOOD_TEXT: Record<string, string> = {
  muted_desaturated: "muted desaturated palette",
  cold_blue:         "cold blue-grey palette",
  warm_amber:        "warm amber palette",
  sickly_green:      "sickly green cast",
  high_contrast_bw:  "high contrast black and white",
};

// --- validated render constants ----------------------------------------------
// Probed on moodyProMix 2026-08-17. The model never sees these.

export const RENDER = {
  /** res_multistep is what the official Z-Image-Turbo example uses; euler was inherited and never
   *  questioned. shift 6 rather than 3: 3 renders slightly crisper but drifts off the depth map, and
   *  geometry is the one thing only the blockout can supply. Detail is recoverable, layout is not. */
  steps: 9, cfg: 1, shift: 6, sampler: "res_multistep", scheduler: "simple",
  /** Control patch strength. 0.5 keeps pose/framing while leaving identity+palette to the prompt.
   *  MEASURED 2026-08-18: PAI's own recommended 0.65-1.00 band is wrong for this stack — every arm from
   *  0.65 up went dark and mushy and hallucinated architecture absent from the depth map, WITHOUT
   *  becoming more faithful. Scheduled release and depth blur did not rescue it either. */
  controlStrength: 0.5,
  width: 1024, height: 1344,
  /**
   * NO LoRA. This is deliberate and hard-won.
   *
   * `Z-TURBO_Photography_35mmPhoto` sat here at 0.7 and was applied to every render this project ever
   * made without once being tested at 0. It is a 35mm FILM emulation, so soft falloff and grain are
   * precisely what it exists to add — it WAS the blur that four separate sweeps (sampler, shift, control
   * strength, hires refine) failed to find, because all four were measured through it.
   *
   * Left as an empty list rather than deleted so callers keep working and the reason stays attached to
   * the decision. If a stylistic LoRA is ever wanted again, A/B it against an empty list first.
   */
  loras: [] as { name: string; strength: number }[],
  /**
   * Prompt lead-in, replacing what used to be the LoRA's trigger word ("35mm photograph").
   *
   * Still worth stating that the output is a photograph — that framing does real work on a model that
   * will otherwise drift toward illustration. What is gone is the film-stock reference, which was asking
   * in words for the same softness the LoRA was baking in.
   */
  lead: "photograph, sharp focus, fine detail",
  /** Hero-frame regime. The negative prompt only functions above cfg 1, which is why it was never
   *  available before; with it, this is by a clear margin the best output measured. ~210 s/frame
   *  against ~28, so it is a per-shot opt-in, not the default. */
  hero: {
    steps: 40, cfg: 4, shift: 6, sampler: "res_multistep", scheduler: "simple",
    negative: "blurry, soft focus, out of focus, low detail, film grain, hazy, washed out, "
      + "smeared, motion blur, deformed hands, extra limbs, text, watermark",
  },
} as const;

// --- canon -------------------------------------------------------------------

/**
 * Location ids available as canon.
 *
 * Gated on location.json, NOT space.json: only 2 of 19 locations have been through the blockout stage,
 * but all 19 have an authored description. Requiring space.json here hid 17 perfectly usable locations
 * from the model. Geometry is a separate question — see locationHasGeometry, which gates control modes.
 */
export function listLocationIds(): string[] {
  try {
    return fs.readdirSync(LOCATIONS_DIR, { withFileTypes: true })
      .filter(d => d.isDirectory() && (
        fs.existsSync(path.join(LOCATIONS_DIR, d.name, "location.json")) ||
        fs.existsSync(path.join(LOCATIONS_DIR, d.name, "space.json"))))
      .map(d => d.name).sort();
  } catch { return []; }
}

/**
 * The canonical description for a location, injected verbatim so world consistency never depends on a
 * small model remembering the place.
 *
 * space.json `look` is preferred where it exists — it is written as generator prompt text (materials,
 * light, film character). location.json `desc` is the fallback and is prose written for a reader, but
 * it still pins the architecture and contents, which is the part that matters most.
 */
export function locationLook(id: string): string | null {
  try {
    const s = JSON.parse(fs.readFileSync(path.join(LOCATIONS_DIR, id, "space.json"), "utf8"));
    if (typeof s.look === "string" && s.look.trim()) return s.look.trim();
  } catch { /* fall through to location.json */ }
  try {
    const l = JSON.parse(fs.readFileSync(path.join(LOCATIONS_DIR, id, "location.json"), "utf8"));
    if (typeof l.desc === "string" && l.desc.trim()) return l.desc.trim();
  } catch { /* no canon text */ }
  return null;
}

/** Does this location have geometry we could render a real depth map from? Gates control modes. */
export function locationHasGeometry(id: string): boolean {
  try {
    const s = JSON.parse(fs.readFileSync(path.join(LOCATIONS_DIR, id, "space.json"), "utf8"));
    return Array.isArray(s.props) && s.props.length > 0;
  } catch { return false; }
}

// --- JSON schema for constrained decoding ------------------------------------

/**
 * The schema handed to llama-server as `json_schema`. This is the guidance engine: under grammar
 * constraint the model *cannot* emit an out-of-vocabulary enum or omit a field, so compliance stops
 * being a prompting problem. Keep it flat and shallow — deep nesting measurably degrades small-model
 * grammar adherence, and there is nothing here that needs depth.
 */
export interface SchemaOpts {
  /** Force an exact shot count (the operator's "how many" box). Blank/undefined = model decides. */
  shotCount?: number;
  /** Restrict the angle vocabulary to the operator's selection. */
  angles?: string[];
  /** Pin the location, removing the choice from the model entirely. */
  locationId?: string | null;
}

export function sceneSpecJsonSchema(opts: SchemaOpts = {}): object {
  const ids = listLocationIds();
  const angleEnum = opts.angles?.length ? ANGLES.filter(a => opts.angles!.includes(a)) : [...ANGLES];
  // minItems === maxItems is how a hard count is enforced: under grammar-constrained decoding the
  // model physically cannot emit a different number, which beats asking it nicely in prose.
  const shotBounds = opts.shotCount
    ? { minItems: opts.shotCount, maxItems: opts.shotCount }
    : { minItems: 2, maxItems: 10 };
  return {
    type: "object",
    additionalProperties: false,
    required: ["location_id", "location_description", "subject", "light", "mood", "shots"],
    properties: {
      // A null-able enum rather than free text: canon first, freeform only as an explicit fallback.
      location_id: opts.locationId
        ? { type: "string", enum: [opts.locationId] }
        : { anyOf: [{ type: "string", enum: ids }, { type: "null" }] },
      location_description: { type: "string", maxLength: 400 },
      subject: {
        anyOf: [
          {
            type: "object", additionalProperties: false,
            required: ["description", "wardrobe"],
            properties: {
              description: { type: "string", maxLength: 200 },
              wardrobe: { type: "string", maxLength: 200 },
            },
          },
          { type: "null" },
        ],
      },
      light: { type: "string", enum: [...LIGHTS] },
      mood: { type: "string", enum: [...MOODS] },
      shots: {
        type: "array", ...shotBounds,
        items: {
          type: "object", additionalProperties: false,
          // pose/placement/action are required so the model must make a choice per shot rather than
          // omitting them and collapsing back to a row of identical standing figures.
          required: ["angle", "elevation", "distance", "lens", "control", "pose", "placement", "action"],
          properties: {
            angle: { type: "string", enum: angleEnum },
            elevation: { type: "string", enum: [...ELEVATIONS] },
            distance: { type: "string", enum: [...DISTANCES] },
            lens: { type: "integer", enum: [...LENSES] },
            control: { type: "string", enum: [...CONTROLS] },
            pose: { type: "string", enum: [...POSES] },
            placement: { type: "string", enum: [...PLACEMENTS] },
            action: { type: "string", maxLength: 80 },
          },
        },
      },
    },
  };
}

// --- enum recovery ------------------------------------------------------------

/**
 * Map an off-vocabulary value onto the nearest legal one.
 *
 * Needed because only the llama-server path is grammar-constrained; any unconstrained backend (and
 * the eval's reference model) will invent plausible values like "night_moonlight" or "blue_glow".
 * Silently defaulting those was actively harmful — a request for a library AT NIGHT was rendering as
 * `overcast_daylight`, i.e. the validator quietly contradicted the operator. Matching on shared
 * tokens recovers the intent; genuinely unmappable values become hard errors so the model gets a
 * repair round-trip instead of a confidently wrong render.
 */
function nearestEnum(value: unknown, legal: readonly string[], synonyms: Record<string, string> = {}): string | null {
  if (typeof value !== "string") return null;
  const v = value.trim().toLowerCase();
  if (legal.includes(v)) return v;
  if (synonyms[v]) return synonyms[v];
  const tokens = v.split(/[^a-z0-9]+/).filter(Boolean);
  // Score by shared token count, then prefer the shorter candidate so "night_sodium" beats
  // "night_fluorescent" for a bare "night" rather than the tie resolving arbitrarily.
  let best: string | null = null, bestScore = 0;
  for (const cand of legal) {
    const ct = cand.split(/[^a-z0-9]+/).filter(Boolean);
    const score = tokens.filter(t => ct.includes(t)).length;
    if (score > bestScore || (score === bestScore && score > 0 && best && cand.length < best.length)) { best = cand; bestScore = score; }
  }
  return bestScore > 0 ? best : null;
}

const LIGHT_SYNONYMS: Record<string, string> = {
  night_moonlight: "night_sodium", moonlight: "night_sodium", night: "night_sodium",
  night_screen: "screen_glow", laptop_glow: "screen_glow", monitor: "screen_glow",
  fluorescent: "night_fluorescent", strip_light: "night_fluorescent", flickering_fluorescent: "night_fluorescent",
  window: "daylight_window", daylight: "overcast_daylight", overcast: "overcast_daylight",
  bulb: "single_bulb", sunset: "dusk", golden_hour: "dusk", evening: "dusk",
};

const MOOD_SYNONYMS: Record<string, string> = {
  decay_shadowed: "muted_desaturated", muted: "muted_desaturated", desaturated: "muted_desaturated",
  blue_glow: "cold_blue", cold: "cold_blue", blue: "cold_blue",
  warm: "warm_amber", amber: "warm_amber", golden: "warm_amber",
  green: "sickly_green", bw: "high_contrast_bw", monochrome: "high_contrast_bw", black_and_white: "high_contrast_bw",
};

const DISTANCE_SYNONYMS: Record<string, string> = {
  closeup: "close", close_up: "close", "extreme_closeup": "extreme_close", macro: "extreme_close",
  medium_shot: "medium", mid: "medium", waist_up: "medium",
  full: "full_body", full_shot: "full_body", long: "wide", long_shot: "wide",
  establishing: "very_wide", extreme_wide: "very_wide", wide_shot: "wide",
};

const POSE_SYNONYMS: Record<string, string> = {
  sitting: "seated", sat: "seated", slouched: "seated_hunched", hunched: "seated_hunched",
  squatting: "crouching", crouched: "crouching", kneel: "kneeling",
  leaning: "leaning_against_wall", lean: "leaning_against_wall",
  walk: "walking", striding: "mid_stride", running: "mid_stride",
  glancing_back: "looking_back", over_shoulder: "looking_back",
  looking_down: "head_down", bowed: "head_down", looking_up: "head_tilted_back",
  reach: "reaching", holding: "holding_something", carrying: "holding_something",
  lying: "lying_down", prone: "lying_down", supine: "lying_down",
  hands_in_pockets: "standing_hands_in_pockets", crossed_arms: "arms_crossed",
};

const ANGLE_SYNONYMS: Record<string, string> = {
  overhead: "frontal", top_down: "frontal", birds_eye: "frontal",
  three_quarters: "three_quarter", side: "profile_left", profile: "profile_left",
  back: "rear", behind: "rear", from_behind: "rear", back_view: "rear",
  front: "frontal", facing_camera: "frontal", shoulder: "over_shoulder",
};

// --- validation ---------------------------------------------------------------

export interface Validation { ok: boolean; errors: string[]; warnings: string[]; spec: SceneSpec }

/**
 * Validate and REPAIR a spec. Errors are things we send back to the model; warnings are things the
 * operator should know but which do not block a render.
 *
 * Repairs happen in place because a grammar-constrained model gets the shape right and the semantics
 * wrong — a duplicate shot or an impossible control mode is worth fixing silently rather than
 * burning a whole retry round-trip on.
 */
export function validateSceneSpec(raw: unknown, opts: SchemaOpts = {}): Validation {
  const errors: string[] = [];
  const warnings: string[] = [];
  const s = (raw ?? {}) as Partial<SceneSpec>;

  const ids = listLocationIds();
  let locId = typeof s.location_id === "string" ? s.location_id : null;
  // An operator-pinned location is not a suggestion: overwrite whatever the model chose. This is the
  // whole point of the dropdown — the previous build had no way to command the location at all.
  if (opts.locationId) {
    if (locId !== opts.locationId) warnings.push(`location pinned to "${opts.locationId}" by the operator`);
    locId = opts.locationId;
  }
  const locDesc = (typeof s.location_description === "string" ? s.location_description : "").trim();

  if (locId && !ids.includes(locId)) {
    // Hallucinated a canon id. Don't fail — fall back to freeform if there is anything usable.
    warnings.push(`location_id "${locId}" is not a canon location — treating as freeform`);
    locId = null;
  }
  if (!locId && locDesc.length < 12) {
    errors.push("no canon location_id given, so location_description must describe the place in at least 12 characters");
  }

  // Recover rather than default: see nearestEnum. Unmappable is a hard error, because a wrong light
  // silently inverts the whole mood of the render.
  const lightFix = nearestEnum(s.light, LIGHTS, LIGHT_SYNONYMS);
  if (lightFix && lightFix !== s.light) warnings.push(`light "${s.light}" is not in the vocabulary — mapped to nearest "${lightFix}"`);
  if (!lightFix) errors.push(`light "${s.light}" is not one of: ${LIGHTS.join(", ")}`);
  const light = (lightFix ?? "overcast_daylight") as SceneSpec["light"];

  const moodFix = nearestEnum(s.mood, MOODS, MOOD_SYNONYMS);
  if (moodFix && moodFix !== s.mood) warnings.push(`mood "${s.mood}" is not in the vocabulary — mapped to nearest "${moodFix}"`);
  if (!moodFix) errors.push(`mood "${s.mood}" is not one of: ${MOODS.join(", ")}`);
  const mood = (moodFix ?? "muted_desaturated") as SceneSpec["mood"];

  let subject: SubjectSpec | null = null;
  if (s.subject && typeof s.subject === "object") {
    const d = String((s.subject as SubjectSpec).description ?? "").trim();
    const w = String((s.subject as SubjectSpec).wardrobe ?? "").trim();
    if (d.length < 4) errors.push("subject present but description is empty");
    // Wardrobe is not optional in practice: it is the anchor set that makes identity drift legible,
    // and a bare subject gives the model nothing to agree with itself about between angles.
    if (w.length < 4) warnings.push("subject has no wardrobe anchors — cross-shot consistency will be much worse");
    subject = { description: d, wardrobe: w };
  }

  // Shots: recover malformed entries, dedupe, then backfill so the operator gets the COUNT they asked
  // for. Silently returning 2 shots when 3 were requested reads as the machine obeying while it is
  // actually short-changing the request, which is the failure mode the eval caught.
  const requested = Array.isArray(s.shots) ? s.shots.length : 0;
  const seen = new Set<string>();
  const shots: ShotSpecLite[] = [];
  for (const sh of Array.isArray(s.shots) ? s.shots : []) {
    const rawA = (sh as ShotSpecLite)?.angle, e0 = (sh as ShotSpecLite)?.elevation;
    const d = (sh as ShotSpecLite)?.distance, ln = Number((sh as ShotSpecLite)?.lens);
    let ctl = (sh as ShotSpecLite)?.control as Control;
    const a = nearestEnum(rawA, ANGLES, ANGLE_SYNONYMS);
    if (!a) { warnings.push(`dropped a shot with unrecoverable angle "${rawA}"`); continue; }
    if (a !== rawA) warnings.push(`angle "${rawA}" is not in the vocabulary — mapped to nearest "${a}"`);
    const e = nearestEnum(e0, ELEVATIONS) ?? "eye";
    if (e !== e0) warnings.push(`elevation "${e0}" is not in the vocabulary — mapped to "${e}"`);
    const dist = nearestEnum(d, DISTANCES, DISTANCE_SYNONYMS) ?? "medium";
    const lens = (LENSES as readonly number[]).includes(ln) ? ln : 35;
    if (!(CONTROLS as readonly string[]).includes(ctl)) ctl = "none";
    const pose = nearestEnum((sh as ShotSpecLite)?.pose, POSES, POSE_SYNONYMS) ?? "standing";
    const placement = nearestEnum((sh as ShotSpecLite)?.placement, PLACEMENTS) ?? "center";
    const rawAction = String((sh as ShotSpecLite)?.action ?? "").trim();
    // Truncate rather than reject: an over-long action is a formatting slip, not a semantic error.
    const action = rawAction.length > 80 ? rawAction.slice(0, 80).trim() : rawAction;

    // Control needs a geometry source. Asking for depth on a freeform location we have no blockout
    // for is not a warning, it is impossible — downgrade rather than fail at render time.
    if (ctl !== "none" && (!locId || !locationHasGeometry(locId))) {
      warnings.push(`control "${ctl}" requested but ${locId ? `location "${locId}" has no props/geometry` : "location is freeform"} — downgraded to none`);
      ctl = "none";
    }

    const key = `${a}|${e}|${dist}|${pose}`;
    if (seen.has(key)) { warnings.push(`dropped duplicate shot ${key}`); continue; }
    seen.add(key);
    shots.push({
      angle: a as Angle, elevation: e as Elevation, distance: dist as Distance, lens: lens as 35,
      control: ctl, pose: pose as Pose, placement: placement as Placement, action,
    });
  }
  // Backfill to the requested count using angles not yet used, so a dedupe or a recovery collision
  // costs the operator a repeated angle rather than a missing shot. An explicit operator count wins
  // over whatever the model emitted.
  if (opts.shotCount && shots.length > opts.shotCount) {
    warnings.push(`operator asked for ${opts.shotCount} shots; trimmed from ${shots.length}`);
    shots.length = opts.shotCount;
  }
  const target = Math.min(Math.max(opts.shotCount ?? requested, shots.length), 10);
  if (shots.length && shots.length < target) {
    const template = shots[shots.length - 1];
    for (const a of ANGLES) {
      if (shots.length >= target) break;
      const key = `${a}|${template.elevation}|${template.distance}`;
      if (seen.has(key)) continue;
      seen.add(key);
      shots.push({ ...template, angle: a });
    }
    warnings.push(`backfilled to ${shots.length} shot(s) after ${requested - shots.length + (shots.length - 1)} were dropped or deduped`);
  }
  if (!shots.length) errors.push("no valid shots — need at least one");
  if (shots.length > 10) { warnings.push(`truncated ${shots.length} shots to 10`); shots.length = 10; }

  // --- diversity repair ------------------------------------------------------
  // Measured failure: with one worked example in the prompt, three different intents all produced the
  // SAME opening pair (three_quarter/eye/medium then profile_left/eye/close/85) — the model was copying
  // the example rather than directing the scene. Better examples help, but a set that still collapses
  // to one axis should be fixed rather than shipped, because the operator's complaint is about the
  // pictures, not about whose fault it is. Rotate the offending axis deterministically so results stay
  // reproducible for a given seed.
  if (shots.length >= 3 && subject) {
    const distinct = (f: (s: ShotSpecLite) => unknown) => new Set(shots.map(f)).size;

    if (distinct(s => s.pose) === 1) {
      const pool: Pose[] = ["standing", "seated", "leaning_against_wall", "crouching", "walking", "looking_back", "head_down", "arms_crossed"];
      shots.forEach((s, i) => { s.pose = pool[i % pool.length]; });
      warnings.push("every shot had the same pose — rotated poses so the set is not one photo repeated");
    }
    if (distinct(s => s.distance) === 1) {
      const pool: Distance[] = ["close", "medium", "full_body", "wide", "extreme_close", "very_wide"];
      shots.forEach((s, i) => { s.distance = pool[i % pool.length]; });
      warnings.push("every shot had the same framing distance — varied it across the set");
    }
    if (distinct(s => s.placement) === 1 && shots.length >= 4) {
      const pool: Placement[] = ["center", "left_third", "right_third", "small_in_frame"];
      shots.forEach((s, i) => { s.placement = pool[i % pool.length]; });
      warnings.push("every shot placed the subject identically in frame — varied placement");
    }
    if (distinct(s => s.lens) === 1) {
      // Lens tracks distance: a 135mm very-wide establishing shot is incoherent.
      const byDist: Record<string, 18 | 24 | 35 | 50 | 85 | 135> = {
        extreme_close: 85, close: 85, medium: 50, full_body: 35, wide: 24, very_wide: 18,
      };
      shots.forEach(s => { s.lens = byDist[s.distance] ?? 35; });
      warnings.push("every shot used the same lens — matched lens to framing distance instead");
    }
  }

  // Measured 2026-08-17: elevation wording is only loosely obeyed without a control map. Say so
  // rather than letting the operator believe a low angle was delivered.
  const unreliable = shots.filter(sh => sh.elevation !== "eye" && sh.control === "none");
  if (unreliable.length) {
    warnings.push(`${unreliable.length} shot(s) request a non-eye-level elevation with no control map — elevation is only loosely obeyed in prompt-only mode`);
  }
  if (subject && shots.length > 1) {
    warnings.push("no character LoRA is attached, so the face WILL drift between angles (worst on profile and rear) — wardrobe and scene will hold");
  }

  return {
    ok: errors.length === 0,
    errors, warnings,
    spec: { location_id: locId, location_description: locDesc, subject, light, mood, shots },
  };
}

// --- composition --------------------------------------------------------------

/** Everything needed to render one shot: the final prompt plus the settings the model never chose. */
export interface ComposedShot {
  label: string;
  prompt: string;
  control: Control;
  lint: { scrubbed: boolean; warnings: string[] };
}

/**
 * Turn a validated spec into per-shot prompts.
 *
 * Order is deliberate and constant: style trigger, then place, then subject (verbatim), then camera,
 * then grade. Holding the order fixed across shots is part of what keeps the film character steady —
 * reordering clauses between frames measurably changes the result even at a fixed seed.
 */
export function composeShots(spec: SceneSpec): ComposedShot[] {
  let place = spec.location_id ? (locationLook(spec.location_id) ?? spec.location_description) : spec.location_description;

  // Canon descriptions were authored for a reader, not for an empty stage, so several of them mention
  // inhabitants ("a slumped figure visible at one wheel"). Injecting that into a scene the operator
  // explicitly asked to be empty reintroduces the accidental-people bug the ANATOMY_GUARD work fixed.
  // Drop whole sentences that name a person before the guards ever see the prompt — a sentence is the
  // right unit because these mentions are self-contained clauses about the inhabitant, not adjectives
  // load-bearing for the architecture.
  if (!spec.subject && place) {
    const kept = place.split(/(?<=\.)\s+/).filter(s => !PERSON_NOUN_RX.test(s));
    const rebuilt = kept.join(" ").trim();
    if (rebuilt.length >= 24) place = rebuilt;
  }
  const grade = [LIGHT_TEXT[spec.light], MOOD_TEXT[spec.mood], "natural skin texture"].join(", ");

  // Built ONCE, outside the loop, and interpolated byte-identically into every shot.
  const subjectClause = spec.subject
    ? `${spec.subject.description}, ${spec.subject.wardrobe}`.replace(/,\s*$/, "")
    : "";

  return spec.shots.map((sh, i) => {
    // Pose and action only mean anything with a figure present; on an empty stage they would be a
    // direct invitation to hallucinate one.
    const camera = [
      spec.subject ? POSE_TEXT[sh.pose ?? "standing"] : null,
      spec.subject && sh.action ? sh.action.trim() : null,
      ANGLE_TEXT[sh.angle],
      spec.subject && sh.placement ? PLACEMENT_TEXT[sh.placement] : null,
      DISTANCE_TEXT[sh.distance],
      ELEVATION_TEXT[sh.elevation],
      LENS_TEXT[sh.lens],
    ].filter(Boolean).join(", ");

    const parts = [RENDER.lead, place, subjectClause, camera, grade].filter(p => p && p.length);
    let prompt = parts.join(", ");

    // An empty stage is the case that historically produced accidental people: body-shaped prompt
    // language gets morphed into bodies. Reuse the existing guards rather than reinventing them.
    let lint = { scrubbed: false, warnings: [] as string[] };
    if (!spec.subject) {
      const l = lintBackgroundPrompt(prompt, [], { strict: true });
      prompt = `${l.prompt}\n\n${EMPTY_SCENE}\n\n${ANATOMY_GUARD}`;
      lint = { scrubbed: l.scrubbed, warnings: l.warnings };
    }

    // Pose is in the filename because it is now a primary variance axis — without it two shots that
    // differ only by pose would collide on disk and silently overwrite each other.
    const label = [
      String(i + 1).padStart(2, "0"), sh.angle, sh.elevation, sh.distance,
      spec.subject ? (sh.pose ?? "standing") : null,
    ].filter(Boolean).join("_");
    return { label, prompt, control: sh.control, lint };
  });
}

/**
 * Machine-readable description of everything this engine can do.
 *
 * Exists so a caller that is itself an AI can DISCOVER the contract instead of having it hardcoded:
 * fetch this, read the schema and vocabularies, emit a conforming spec, post it. That makes the local
 * director optional rather than load-bearing — any model anywhere can drive the renderer as long as it
 * can satisfy the schema, which is the difference between a demo and an engine.
 *
 * `render` is included as information, not as knobs: those constants are validated and callers should
 * not be inventing their own step counts.
 */
export function capabilities() {
  return {
    version: 1,
    schema: sceneSpecJsonSchema(),
    vocabularies: {
      angle: [...ANGLES], elevation: [...ELEVATIONS], distance: [...DISTANCES],
      lens: [...LENSES], control: [...CONTROLS], light: [...LIGHTS], mood: [...MOODS],
      // pose and placement are the primary variance axes; omitting them here left an API-driven
      // caller unable to discover the two fields that matter most for a set that is not one photo
      // repeated, even though the schema required them.
      pose: [...POSES], placement: [...PLACEMENTS],
    },
    locations: listLocationIds().map(id => ({
      id,
      hasGeometry: locationHasGeometry(id),
      look: (locationLook(id) ?? "").slice(0, 200),
    })),
    render: { ...RENDER },
    notes: [
      "Post a spec to /api/spec to validate it without spending GPU time. Do that before /api/render.",
      "subject.wardrobe should name specific garments and colours; it is interpolated byte-identically " +
        "into every shot and is the only cross-angle consistency anchor available.",
      "There is no character LoRA yet, so a face will drift between angles (worst on profile and rear). " +
        "Wardrobe, location and grade do hold.",
      "control modes other than 'none' require a location with geometry (hasGeometry: true); anything " +
        "else is downgraded to 'none' during validation.",
      "elevation other than 'eye' is only loosely obeyed without a control map.",
      "Set subject to null for an empty scene; person-noun sentences are then stripped from the " +
        "location description and anti-anatomy guards are appended.",
    ],
    guide: capabilityCard(),
  };
}

/**
 * Compose one shot with a geometry-derived inventory spliced in.
 *
 * Used by the blockout path. The location `look` still leads, because it is what carries material and
 * film character and is the only clause held constant across cuts — dropping it was why the first
 * geometry A/B came out murky and colourless. The projected inventory is ADDITIVE: it says which of
 * those objects are actually on screen from this camera, where in frame, and how far away.
 *
 * Camera wording is deliberately kept even though the depth map now enforces it. The two agree, and
 * agreement is reinforcement — the failure mode we care about is depth and text pulling apart, which
 * cannot happen when both derive from the same solved camera.
 */
export function composeShotWithGeometry(
  spec: SceneSpec, sh: ShotSpecLite, index: number, inventory: string,
): ComposedShot {
  const base = composeShots({ ...spec, shots: [sh] })[0];
  const place = spec.location_id ? (locationLook(spec.location_id) ?? spec.location_description) : spec.location_description;
  // Rebuild from parts rather than string-surgery on `base.prompt`: slicing that string by commas is
  // what silently dropped the subject clause and produced empty rooms.
  const subjectClause = spec.subject ? `${spec.subject.description}, ${spec.subject.wardrobe}`.replace(/,\s*$/, "") : "";
  const camera = base.prompt.split(", ").filter(p =>
    Object.values(ANGLE_TEXT).includes(p) || Object.values(ELEVATION_TEXT).includes(p) ||
    Object.values(DISTANCE_TEXT).includes(p) || Object.values(LENS_TEXT).includes(p) ||
    Object.values(POSE_TEXT).includes(p) || Object.values(PLACEMENT_TEXT).includes(p),
  ).join(", ");
  const grade = [LIGHT_TEXT[spec.light], MOOD_TEXT[spec.mood], "natural skin texture"].join(", ");

  const parts = [RENDER.lead, place, inventory, subjectClause, sh.action?.trim(), camera, grade]
    .filter(p => p && p.length);
  let prompt = parts.join(", ");
  let lint = { scrubbed: false, warnings: [] as string[] };
  if (!spec.subject) {
    const l = lintBackgroundPrompt(prompt, [], { strict: true });
    prompt = `${l.prompt}\n\n${EMPTY_SCENE}\n\n${ANATOMY_GUARD}`;
    lint = { scrubbed: l.scrubbed, warnings: l.warnings };
  }
  return { label: base.label, prompt, control: sh.control, lint };
}

/** Compact capability card for the local model's system prompt. Small models imitate examples far
 *  better than they follow rules, so this leans on one worked example over a list of instructions. */
export function capabilityCard(opts: SchemaOpts = {}): string {
  const ids = listLocationIds();
  const countLine = opts.shotCount
    ? `You MUST output exactly ${opts.shotCount} shots.`
    : "Choose the shot count yourself: 3 for a simple beat, 5-6 for a scene worth covering, 8+ only " +
      "if the space genuinely rewards it. Do not default to 4 every time.";
  const angleLine = opts.angles?.length
    ? `Use ONLY these angles: ${opts.angles.join(", ")}.`
    : `Allowed angle values: ${ANGLES.join(", ")}`;
  const locLine = opts.locationId
    ? `The location is FIXED to "${opts.locationId}". Set location_id to exactly that and leave location_description empty.`
    : "";
  return [
    "You are a shot director for a photographic image generator. You do NOT write image prompts.",
    "You output ONLY a JSON scene spec. Another system turns your spec into the actual prompt, and it",
    "already handles film stock, palette wording, lens phrasing, and quality tags. Do not describe those.",
    "",
    "Rules that matter:",
    "- ALWAYS check the canon location list FIRST and use a location_id when one plausibly fits.",
    "  A canon id pulls in that place's authored description, which is far better than anything you",
    "  can write, and it keeps the location consistent with every other scene in the project.",
    "  'library' -> library-interior. 'dorm room' -> dorm-room. 'parking lot' -> campus-parking-lot.",
    "  Only set location_id to null when NOTHING in the list fits, and then write location_description.",
    "  When you do use a location_id, leave location_description as an empty string.",
    "- light and mood MUST be copied exactly from the allowed values listed below. Do not invent",
    "  new ones like 'night_moonlight' or 'blue_glow' — pick the closest allowed value instead.",
    "- subject.wardrobe must name specific garments and colours. These anchor the character across",
    "  angles. Vague wardrobe makes the character fall apart between shots.",
    "- Set subject to null only if the scene should contain no people at all.",
    "",
    "COVERAGE is the job. You are not photographing a mannequin from six sides — you are covering a",
    "moment. Vary these independently across the set, and let the SCENE decide which values fit:",
    "  pose      — the single biggest difference between shots. A set where everyone stands still",
    "              reads as one photo repeated. Sit them, crouch them, have them walk or look back.",
    "  distance  — mix an extreme_close with a very_wide. Do not shoot everything at medium.",
    "  placement — do not centre the subject every time.",
    "  elevation — ground and overhead are available and dramatic; use them when the space earns it.",
    "  lens      — pair long lenses with close framing and wide lenses with wide framing.",
    "  action    — one short concrete beat per shot, specific to THIS scene, under 80 characters.",
    "              e.g. \"pulling a book off a high shelf\", not \"posing\".",
    "",
    "Do NOT copy the shot lists in the examples below. They differ from each other on purpose; yours",
    "should differ from both. Repeating an example verbatim is the main failure mode here.",
    countLine,
    locLine,
    "",
    `Canon location ids: ${ids.join(", ")}`,
    `Allowed light values: ${LIGHTS.join(", ")}`,
    `Allowed mood values: ${MOODS.join(", ")}`,
    angleLine,
    `Allowed elevation values: ${ELEVATIONS.join(", ")}`,
    `Allowed distance values: ${DISTANCES.join(", ")}`,
    `Allowed pose values: ${POSES.join(", ")}`,
    `Allowed placement values: ${PLACEMENTS.join(", ")}`,
    `Allowed lens values: ${LENSES.join(", ")}`,
    "",
    // Two examples, and the CANON one first: a single freeform example was enough to make the model
    // answer freeform for every request including ones that named canon places (measured, 0/10).
    "Example 1 — request \"Julee alone in the derelict library at night\".",
    "The library IS in the canon list, so use the id and leave the description empty. Note the range:",
    "an extreme close, a ground-level wide, someone seated and someone walking. Six shots because the",
    "space is big enough to earn them.",
    JSON.stringify({
      location_id: "library-interior",
      location_description: "",
      subject: { description: "a woman in her twenties, dark hair loose", wardrobe: "grey wool cardigan over a white shirt, dark corduroy trousers" },
      light: "single_bulb",
      mood: "muted_desaturated",
      shots: [
        { angle: "three_quarter", elevation: "eye", distance: "full_body", lens: 35, control: "none", pose: "walking", placement: "left_third", action: "stepping over fallen books" },
        { angle: "frontal", elevation: "eye", distance: "extreme_close", lens: 85, control: "none", pose: "hands_on_face", placement: "center", action: "rubbing dust from her eyes" },
        { angle: "rear", elevation: "ground", distance: "very_wide", lens: 18, control: "none", pose: "standing", placement: "small_in_frame", action: "facing the far reading hall" },
        { angle: "profile_right", elevation: "eye", distance: "medium", lens: 50, control: "none", pose: "seated_hunched", placement: "right_third", action: "reading by the lamp" },
        { angle: "over_shoulder", elevation: "high", distance: "close", lens: 85, control: "none", pose: "reaching", placement: "center", action: "pulling a book off a high shelf" },
        { angle: "three_quarter_back", elevation: "overhead", distance: "wide", lens: 24, control: "none", pose: "sitting_on_floor", placement: "center", action: "surrounded by open volumes" },
      ],
    }, null, 2),
    "",
    "Example 2 — request \"a woman waiting on a train platform at night, a few angles\".",
    "No canon location fits a subway platform, so location_id is null. Deliberately a DIFFERENT shape",
    "from Example 1: only three shots, different angles, different poses, different lenses.",
    JSON.stringify({
      location_id: null,
      location_description: "a near-empty subway platform at night, white tiled wall, wet concrete, yellow tactile edge strip",
      subject: { description: "a woman in her early thirties, hair tied back in a low ponytail", wardrobe: "dark green canvas jacket over a grey shirt, black jeans" },
      light: "night_sodium",
      mood: "muted_desaturated",
      shots: [
        { angle: "profile_left", elevation: "low", distance: "close", lens: 135, control: "none", pose: "leaning_against_wall", placement: "right_third", action: "watching the tunnel mouth" },
        { angle: "three_quarter_back", elevation: "high", distance: "very_wide", lens: 18, control: "none", pose: "standing_hands_in_pockets", placement: "small_in_frame", action: "alone at the platform edge" },
        { angle: "frontal", elevation: "eye", distance: "medium", lens: 50, control: "none", pose: "looking_back", placement: "left_third", action: "turning at a noise behind her" },
      ],
    }, null, 2),
    "",
    "Example 3 — request \"the parking lot, empty, no one there\". subject is null, so there are no",
    "pose or action beats to speak of; set pose to \"standing\", placement \"center\", action \"\".",
    JSON.stringify({
      location_id: "campus-parking-lot",
      location_description: "",
      subject: null,
      light: "dusk",
      mood: "sickly_green",
      shots: [
        { angle: "frontal", elevation: "eye", distance: "very_wide", lens: 18, control: "none", pose: "standing", placement: "center", action: "" },
        { angle: "three_quarter", elevation: "ground", distance: "wide", lens: 24, control: "none", pose: "standing", placement: "center", action: "" },
        { angle: "rear", elevation: "overhead", distance: "wide", lens: 35, control: "none", pose: "standing", placement: "center", action: "" },
      ],
    }, null, 2),
  ].join("\n");
}

export { HOUSE_STYLE };
