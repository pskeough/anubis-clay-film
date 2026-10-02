/**
 * Deterministic per-shot prompt construction.
 *
 * The prompt used to be: location description + shot-size words + one bearing line for the framing
 * subject. That is a description of the LOCATION, and it produced exactly the failures you'd expect
 * from telling a model "this room contains an armchair and a camp cot" on every frame regardless of
 * viewpoint — cots invented behind chairs, tables mirrored, objects appearing that the camera cannot
 * see. Nothing in the text was tied to the actual frame.
 *
 * This builds a description of THIS FRAME instead, entirely from geometry the renderer already
 * computes: which objects have visible pixels, how much of the frame each occupies, where its
 * centroid sits, how far away it is, and which of its faces is turned toward the lens. Nothing is
 * authored per shot and nothing is guessed, so it generalises to any location and any camera without
 * new code — add a prop to a space.json and it starts appearing in prompts when, and only when, it
 * is actually on screen.
 */
import type { ShotSpec, SpaceSpec, VisibleProp } from "./storyboard.ts";
import { analyzeShot } from "./storyboard.ts";

const SIZE_WORDS: Record<string, string> = {
  ELS: "extreme wide shot, the whole room, deep focus",
  LS: "wide shot, full architecture visible, deep focus",
  MLS: "medium-wide shot, furniture at full height in frame",
  MS: "medium shot, subject filling the middle of frame, background falling soft",
  MCU: "close shot, subject large in frame, shallow depth of field",
  CU: "close-up, tight on the object, very shallow depth of field",
  ECU: "extreme close-up, macro, razor-thin plane of focus",
};

const ANGLE_WORDS: Record<string, string> = {
  floor: "camera resting on the floor looking up, dramatic low angle",
  low: "low camera angle, looking slightly up",
  eye: "eye-level camera",
  high: "elevated camera looking down",
  overhead: "high angle looking steeply down",
};

/**
 * Which face of the object is turned toward the lens.
 *
 * Bearing is the angle from the prop's own forward (+Z local) to the direction of the camera. The
 * screen-space consequence was measured with src/_facing-check.ts rather than derived, because
 * getting the handedness wrong here is invisible to every other check — the geometry stays right,
 * the sentence stays grammatical, and the model is simply told the wrong side:
 *
 *   bearing > 0  ->  the object's FRONT projects to the RIGHT of its centre in frame
 *   bearing > 0  ->  its LEFT side is the nearer one
 *
 * Re-run that script if this is ever edited.
 */
function facing(bearing: number): string {
  const a = ((bearing % 360) + 360) % 360;
  if (a < 22.5 || a >= 337.5) return "facing the camera head-on";
  if (a < 67.5) return "turned three-quarters toward the camera, its left side nearer";
  if (a < 112.5) return "in profile, its left side to the camera";
  if (a < 157.5) return "turned mostly away, its left rear toward the camera";
  if (a < 202.5) return "seen from directly behind, facing away";
  if (a < 247.5) return "turned mostly away, its right rear toward the camera";
  if (a < 292.5) return "in profile, its right side to the camera";
  return "turned three-quarters toward the camera, its right side nearer";
}

/**
 * Beds don't have a face, they have a head and a foot, and that is the only thing worth saying
 * about how one is turned. Told "the cot is facing the camera head-on" a generator has no usable
 * instruction, which is half of why the cot landed at a different angle in every frame.
 */
function facingBed(bearing: number): string {
  const b = ((((bearing + 180) % 360) + 360) % 360) - 180;      // -180..180
  const m = Math.abs(b);
  if (m < 30) return "running away from the camera with its pillow end nearest the lens";
  if (m > 150) return "running toward the camera, seen from the foot, its pillow at the far end";
  const side = b > 0 ? "right" : "left";
  return m < 90
    ? `lying side-on across the frame, its pillow end to the ${side} and angled toward the camera`
    : `lying side-on across the frame, its pillow end to the ${side} and angled away from the camera`;
}

/** Orientation vocabulary by archetype. */
const facingFor = (v: VisibleProp): string =>
  v.type === "cot" ? facingBed(v.bearing) : facing(v.bearing);

/** Where in frame, in thirds — the vocabulary a director would actually use. */
function placement(sx: number, sy: number): string {
  const h = sx < 0.34 ? "on the left" : sx > 0.66 ? "on the right" : "at centre frame";
  const v = sy < 0.36 ? " in the upper frame" : sy > 0.72 ? " low in frame" : "";
  return h + v;
}

/** How far, in language rather than metres. */
function distance(mid: number): string {
  if (mid < 1.6) return "close to the lens";
  if (mid < 3.5) return "in the near middle distance";
  if (mid < 7) return "in the middle distance";
  return "far back in the room";
}

const NUMBER = ["", "", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];

/** "a fluted stone column" x3 -> "three fluted stone columns". Descs are authored with an article
 *  because they read better singular; grouping has to undo that rather than emit "3 a column s". */
function pluralise(desc: string, n: number): string {
  const stripped = desc.replace(/^(a|an|the)\s+/i, "");
  const words = stripped.split(" ");
  const last = words[words.length - 1];
  const plural = /(s|x|z|ch|sh)$/i.test(last) ? `${last}es`
    : /[^aeiou]y$/i.test(last) ? `${last.slice(0, -1)}ies`
    : `${last}s`;
  words[words.length - 1] = plural;
  return `${NUMBER[n] ?? n} ${words.join(" ")}`;
}

function clause(v: VisibleProp, dominant: boolean): string {
  const parts = [`${v.desc} ${placement(v.sx, v.sy)}`, distance(v.mid)];
  if (v.directional) parts.push(facingFor(v));           // only objects that have a front
  return (dominant ? "Foreground: " : "") + parts.join(", ");
}

export interface ShotPromptOpts {
  /** ignore anything occupying less than this share of frame */
  minCover?: number;
  /** cap the number of objects named, longest-first by screen area */
  maxProps?: number;
}

/**
 * Build the full positive prompt for one shot.
 *
 * `space.look` supplies the location's material and lighting character and is the ONLY part that is
 * constant across shots — which is what holds palette and film character steady between cuts. The
 * inventory clause is per-frame and replaces the old static "this room contains…" listing.
 */
export function buildShotPrompt(shot: ShotSpec, space: SpaceSpec, opts: ShotPromptOpts = {}): string {
  const f = shot.framing ?? ({} as NonNullable<ShotSpec["framing"]>);
  const lens = f.lens ?? 35;
  const persp = lens <= 24 ? "ultra-wide 24mm lens, strong perspective"
    : lens <= 35 ? "35mm lens, natural perspective"
    : lens <= 50 ? "50mm lens, normal perspective"
    : "85mm lens, compressed perspective";

  const visible = analyzeShot(shot, space, opts.minCover ?? 0.004);

  // Group identical descriptions ("column" x4) so the prompt reads naturally instead of repeating.
  // Grouping happens BEFORE the cap, not after. Slicing first meant four separate columns consumed
  // four of the seven slots and then collapsed into a single clause, and the cot — plainly visible
  // at 1.7% of frame — fell off the end of the list and went unmentioned. One kind of thing should
  // cost one slot.
  const seen = new Map<string, VisibleProp[]>();
  for (const v of visible) {
    const k = v.desc;
    (seen.get(k) ?? seen.set(k, []).get(k)!).push(v);
  }

  // Rank by screen area weighted toward the near ground, then again toward objects that have a
  // front. Raw coverage alone ranks a distant column row above the furniture the shot is about,
  // because architecture is large and far away; and an object whose orientation can be wrong is
  // worth more prompt budget than one that reads the same from every side, because orientation is
  // the failure the inventory exists to prevent.
  const ranked = [...seen.entries()]
    .map(([desc, group]) => {
      const cover = group.reduce((a, v) => a + v.cover, 0);
      const mid = Math.min(...group.map((v) => v.mid));
      const near = Math.max(0.6, Math.min(2, 6 / Math.max(0.5, mid)));
      const dir = group.some((v) => v.directional) ? 1.5 : 1;
      return { desc, group, weight: cover * near * dir };
    })
    .sort((a, b) => b.weight - a.weight)
    .slice(0, opts.maxProps ?? 10);

  const clauses: string[] = [];
  let first = true;
  for (const { desc, group } of ranked) {
    if (group.length > 1) {
      clauses.push(`${pluralise(desc, group.length)}, ${distance(Math.min(...group.map((v) => v.mid)))}`);
    } else {
      clauses.push(clause(group[0], first && group[0].cover > 0.12));
    }
    first = false;
  }

  const inventory = clauses.length
    ? ` In this frame: ${clauses.join(". ")}. Nothing else is in shot.`
    : "";

  return `${space.look ?? ""}` +
    ` ${SIZE_WORDS[f.size ?? "MS"] ?? ""}. ${ANGLE_WORDS[f.angle ?? "eye"] ?? ""}. Shot on a ${persp}, wide aperture.` +
    inventory;
}

/** Debug view: what the camera sees, as a table. */
export function describeShot(shot: ShotSpec, space: SpaceSpec): string {
  const vis = analyzeShot(shot, space);
  return vis
    .map((v) => `  ${v.id.padEnd(16)} ${(v.cover * 100).toFixed(1).padStart(5)}%  ` +
      `[${v.sx.toFixed(2)},${v.sy.toFixed(2)}]  ${v.mid.toFixed(1)}m  bearing ${v.bearing.toFixed(0).padStart(4)}  ${facingFor(v)}`)
    .join("\n");
}
