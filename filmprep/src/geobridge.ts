/**
 * Geometry bridge — resolve a SceneSpec shot into a real camera against a location's blockout.
 *
 * This is the join between the two halves of the pipeline that had been built separately:
 *
 *   the director  (scenespec.ts)  knows WHAT to shoot but its camera is a string, so "low angle"
 *                                 is a wish the diffusion model obeys about half the time
 *   the blockout  (storyboard.ts) knows where everything physically IS — 37 metric props for the
 *                                 library — and already solves cameras, renders clay, emits depth,
 *                                 and derives per-frame visible-object clauses by projection
 *
 * Bridging them converts every soft camera word into geometry. `elevation: "ground"` stops being
 * prose and becomes a camera at y=0.18m; `placement: "left_third"` becomes an aim correction that
 * actually lands the subject on the left third; occlusion becomes true, so a rear shot no longer
 * names the thing standing behind the lens. The depth map that falls out is the per-angle control
 * source that was the only thing gating the ControlNet path.
 *
 * Nothing here renders. It produces the two control images and the geometry-derived prompt fragment,
 * so the caller can feed either the fast prompt-only path or the depth+init path with the same spec.
 */
import fs from "node:fs/promises";
import path from "node:path";
import {
  loadSpace, solveCamera, renderPasses, renderShot, analyzeShot, depthPng,
  type SpaceSpec, type ShotSpec as BoardShot, type FigureSpec, type Framing, type VisibleProp,
} from "./storyboard.js";
import { buildShotPrompt } from "./shotprompt.js";
import type { ShotSpecLite, SceneSpec, Distance, Elevation, Angle, Placement, Pose } from "./scenespec.js";

/**
 * Should THIS shot go through geometry, or is prompt-only better for it?
 *
 * Measured on library-interior, same spec and seed both ways (_geoab, 2026-08-17):
 *
 *   geometry WINS decisively where the camera is the point. `ground` and `overhead` came out at a real
 *   18cm and 2.42m instead of drifting to eye level, `small_in_frame` actually dwarfed the figure down
 *   a colonnaded hall, and every frame agreed about the room because they were all the same room.
 *
 *   geometry LOSES on close shots. A CU/ECU depth map is a near-featureless white blob — one chair back
 *   filling the frame — so the control carries almost no gradient and the render goes muddy. This is the
 *   same failure as the long-standing wide-shot grip problem seen from the other end: what matters is
 *   how much geometric STRUCTURE the map contains, not how far away the subject is.
 *
 * So route per shot rather than picking a side. Cost is similar either way (~28s vs ~42s), which makes
 * routing nearly free.
 */
export function shouldUseGeometry(sh: ShotSpecLite): boolean {
  // Non-eye elevations are the thing prose provably cannot deliver — always worth the geometry.
  if (sh.elevation !== "eye") return true;
  // Close framing starves the depth map of structure.
  if (sh.distance === "extreme_close" || sh.distance === "close") return false;
  // Scale statements need a real room to be dwarfed by.
  if (sh.placement === "small_in_frame" || sh.placement === "far_from_camera") return true;
  return sh.distance === "wide" || sh.distance === "very_wide" || sh.distance === "full_body";
}

/** Does this location have a blockout we can solve cameras against? */
export async function hasBlockout(locationId: string | null): Promise<boolean> {
  if (!locationId) return false;
  try {
    const s = await loadSpace(locationId);
    return Array.isArray(s.props) && s.props.length > 0 && !!s.room;
  } catch { return false; }
}

// --- enum -> geometry ---------------------------------------------------------
// The director's vocabulary was designed for prose; the solver's was designed for cameras. These
// tables are the whole translation, kept in one place so the two vocabularies can evolve apart.

/** Framing distance -> the solver's subject-height-fraction buckets. */
const SIZE_OF: Record<Distance, string> = {
  extreme_close: "ECU",
  close: "CU",
  medium: "MS",
  full_body: "MLS",
  wide: "LS",
  very_wide: "ELS",
};

/** Our elevation names vs the solver's. Only "ground" differs, and it is the one that never worked
 *  as prose — it becomes a camera 18cm off the floor. */
const ANGLE_OF: Record<Elevation, string> = {
  eye: "eye", low: "low", high: "high", ground: "floor", overhead: "overhead",
};

/**
 * Where to put the camera relative to where the figure is LOOKING.
 *
 * The two conventions have to be reconciled carefully. In storyboard.ts a figure's `yaw` is degrees
 * with 0 facing +z (deeper into the room) and 180 facing the front wall, while camera `azimuth` is
 * degrees around the subject with 0 meaning "from the room's front wall". So the azimuth a figure
 * faces is (180 - yaw). A frontal shot puts the camera at that azimuth — i.e. where the eyes point —
 * and everything else is an offset from it.
 */
const AZIMUTH_OFFSET: Record<Angle, number> = {
  frontal: 0,              // camera stands where the figure is looking
  three_quarter: 40,       // off the facing axis, still showing most of the face
  profile_right: 90,
  profile_left: -90,
  three_quarter_back: 140,
  over_shoulder: 165,      // just off the spine, looking the way they look
  rear: 180,
};

/** Subject position in frame. `place` is 0..1 from left/top and the solver corrects the aim to hit it. */
const PLACE_OF: Record<Placement, [number, number]> = {
  center: [0.5, 0.5],
  left_third: [0.33, 0.5],
  right_third: [0.67, 0.5],
  far_from_camera: [0.5, 0.5],
  close_to_edge: [0.14, 0.52],
  small_in_frame: [0.5, 0.56],
};

/** Placements that mean "smaller in frame" are expressed by widening the shot, not by moving it. */
const SIZE_DEMOTE: Partial<Record<Distance, Distance>> = {
  extreme_close: "close", close: "medium", medium: "full_body", full_body: "wide", wide: "very_wide",
};

/** The rasteriser models five body poses; the director has twenty words for them. */
const FIG_POSE_OF: Record<Pose, FigureSpec["pose"]> = {
  standing: "standing", standing_hands_in_pockets: "standing", arms_crossed: "standing",
  walking: "standing", mid_stride: "standing", turning_away: "standing", looking_back: "standing",
  head_tilted_back: "standing", holding_something: "standing", hands_on_face: "standing",
  seated: "sitting", seated_hunched: "slumped", sitting_on_floor: "sitting",
  crouching: "slumped", kneeling: "slumped", head_down: "slumped", leaning_on_furniture: "slumped",
  leaning_against_wall: "standing",
  lying_down: "lying",
  reaching: "reaching",
};

export interface ResolveOpts {
  width?: number;
  height?: number;
  /** Where the figure stands, [x, z] in metres. Defaults to the room's near third, off-centre. */
  figureAt?: [number, number];
  /** Which way the figure faces, degrees (0 = deeper into the room). */
  figureYaw?: number;
  /**
   * Further bodies beyond the framing subject, for two-handers and group shots.
   *
   * Positions are [x, z] in metres in the same room frame as `figureAt`. The camera still solves on the
   * primary subject; these widen what has to fit and give the prompt real coordinates for who stands
   * where, instead of a hopeful "two people facing each other".
   */
  extraFigures?: {
    at: [number, number];
    yaw?: number;
    pose?: Pose;
    height?: number;
    id?: string;
    label?: string;
  }[];
  /** Prop id to frame when there is no figure — defaults to the largest visible solid. */
  emptySubject?: string;
  /**
   * Put the figure's proxy solid into the depth/clay passes. Default FALSE, and that default is a
   * measured result rather than a preference.
   *
   * `collectFaces` represents a figure as a 0.46 x h x 0.3 box. That box is fine for a human-readable
   * board, but as ControlNet input it is a flat slab standing where the person should be, and at any
   * useful strength the model obeys it: the first A/B run produced four frames with a good room and NO
   * PERSON AT ALL. Room geometry helps because rooms are box-shaped; bodies are not. So the room goes
   * through depth and the body stays in the prompt, where pose language already works.
   *
   * Turn this on only once the figure is drawn from the articulated bone rig rather than a box.
   */
  includeFigureInDepth?: boolean;
}

export interface ResolvedShot {
  /** Board shot, fully solved — hand this to renderShot/renderPasses. */
  board: BoardShot;
  /** Geometry-derived description of what is actually visible from THIS camera. */
  inventory: string;
  /** Props big enough to be worth DESCRIBING (>=0.4% of frame). Use for prompt text. */
  visible: VisibleProp[];
  /**
   * Everything the camera can see at all (>=0.02% of frame). Use for EXISTENCE checks.
   *
   * Separate from `visible` because those are different questions and conflating them put a lamp in a
   * prompt that said the lamp was off-screen: at 0.122% of frame it fell under the description floor
   * while still being drawn into the control image. Small-and-salient props — lamps, screens, candles —
   * are exactly where area is the wrong proxy for importance.
   */
  inFrame: VisibleProp[];
  /** PNG buffers ready to upload as control inputs or hosted reference images. */
  clay: Buffer;
  depth: Buffer;
  /** Line/edge pass. Unlike clay it carries NO tonal information, so it cannot mislead a hosted model
   *  about lighting while still stating the geometry unambiguously. */
  edge: Buffer;
}

/**
 * Choose a default subject for an empty room: the biggest thing that is not the shell.
 * Framing bare plaster is the classic failure when nothing is nominated.
 */
function defaultSubject(space: SpaceSpec): string {
  const solids = space.props.filter(p => p.pos && p.size);
  if (!solids.length) return space.props[0]?.id ?? "";
  const biggest = solids.reduce((a, b) =>
    (b.size![0] * b.size![1] * b.size![2]) > (a.size![0] * a.size![1] * a.size![2]) ? b : a);
  return biggest.id;
}

/**
 * Resolve one director shot into a solved board shot plus its clay and depth passes.
 *
 * The figure is placed in the blockout rather than left to the prompt. That is what makes `placement`
 * and `pose` geometric instead of adjectival: the depth map then contains the body as well as the
 * room, so one control image carries both and the two do not have to be composited later.
 */
export async function resolveShot(
  sh: ShotSpecLite, spec: SceneSpec, index: number, opts: ResolveOpts = {},
): Promise<ResolvedShot> {
  if (!spec.location_id) throw new Error("resolveShot needs a canon location_id with a blockout");
  const space = await loadSpace(spec.location_id);
  const W = opts.width ?? 1024, H = opts.height ?? 1344;

  const figures: FigureSpec[] = [];
  let framingSubject: Framing["subject"];
  let azimuth: number;

  if (spec.subject) {
    // Stand the figure in the near third, off-centre left — the same place the library's authored
    // notes put its "nest", which is where a person in these scenes actually is.
    const at = opts.figureAt ?? [-space.room.w * 0.16, space.room.d * 0.30];
    const yaw = opts.figureYaw ?? 180;             // facing the front wall by default
    figures.push({
      id: "subject", at, yaw,
      height: 1.7,
      pose: FIG_POSE_OF[sh.pose ?? "standing"] ?? "standing",
      label: spec.subject.description.slice(0, 40),
    });
    framingSubject = "subject";
    // (180 - yaw) is the azimuth the figure faces; offset from there per the shot's orientation.
    azimuth = ((180 - yaw) + (AZIMUTH_OFFSET[sh.angle] ?? 0) + 360) % 360;

    // Additional bodies, for two-hander scenes.
    //
    // They do not go into the control passes any more than the primary does (see includeFigureInDepth —
    // a box proxy erases the person it is meant to place), so their job is the CAMERA SOLVE: a second
    // body at a known spot is what stops the solver framing tightly on one person and cropping the other
    // out of a conversation. Placement in the prompt is then describable from real coordinates rather
    // than asserted and hoped for.
    (opts.extraFigures ?? []).forEach((f, i) => {
      figures.push({
        id: f.id ?? `figure_${i + 2}`,
        at: f.at, yaw: f.yaw ?? 180, height: f.height ?? 1.7,
        pose: FIG_POSE_OF[f.pose ?? "standing"] ?? "standing",
        label: (f.label ?? `figure ${i + 2}`).slice(0, 40),
      });
    });
  } else {
    framingSubject = opts.emptySubject ?? defaultSubject(space);
    // With no body there is no facing to be relative to, so spread empty-room coverage around the
    // space deterministically instead of shooting the same wall every time.
    //
    // The parentheses are load-bearing. `??` binds looser than `+`, so the original
    // `AZIMUTH_OFFSET[sh.angle] ?? 0 + index * 37` parsed as `AZIMUTH_OFFSET[...] ?? (0 + index*37)`,
    // and since AZIMUTH_OFFSET is a Record<Angle, number> covering every angle the left side is
    // never nullish. `index` therefore did nothing, and every empty-stage shot at a given angle
    // framed the same azimuth forever, which is exactly what the line above says it must not do.
    azimuth = ((AZIMUTH_OFFSET[sh.angle] ?? 0) + index * 37) % 360;
  }

  // "small_in_frame"/"far_from_camera" are size statements wearing placement clothing.
  const demote = sh.placement === "small_in_frame" || sh.placement === "far_from_camera";
  const dist: Distance = demote ? (SIZE_DEMOTE[sh.distance] ?? sh.distance) : sh.distance;

  const framing: Framing = {
    subject: framingSubject,
    size: SIZE_OF[dist],
    angle: ANGLE_OF[sh.elevation],
    lens: sh.lens,
    azimuth,
    place: PLACE_OF[sh.placement ?? "center"],
  };

  const board: BoardShot = {
    id: `s${String(index + 1).padStart(2, "0")}`,
    location: spec.location_id,
    framing, figures, width: W, height: H,
  };
  // Solve once and freeze it onto the shot, so the prompt analysis, the clay render and the depth
  // pass are guaranteed to describe the SAME camera. Re-solving per consumer is how those three
  // drift apart.
  board.camera = solveCamera(framing, space, figures, W, H);

  const visible = analyzeShot(board, space);
  // A SECOND pass at a far lower floor, for existence questions rather than description.
  //
  // analyzeShot culls under 0.4% of frame, which is right for "what should the prompt describe" — below
  // that a prop is noise. It is wrong for "is this thing in the picture at all", and the two got
  // conflated: the banker lamp covers 0.122% of the low 35mm frame, so it was culled from `visible`
  // while renderShot drew it anyway. The light clause then truthfully-by-its-own-inputs told the model
  // the lamp was outside the frame, and the control image disagreed.
  //
  // Small and salient is a real category — lamps, screens, candles, anything emissive — so existence
  // checks read this list and description still reads `visible`.
  const inFrame = analyzeShot(board, space, 0.0002);
  const inventory = buildShotPrompt(board, space, { maxProps: 7, minCover: 0.006 });


  // The figure is framed by the camera solve but kept OUT of the control passes by default — see
  // ResolveOpts.includeFigureInDepth for why a box proxy erases the person it is meant to place.
  // `framing` is dropped here, not just `figures`. shotCamera prefers framing over an explicit camera
  // and re-solves, which threw "subject is not a prop or figure" once the figure was removed — and
  // would silently have used a DIFFERENT camera than analyzeShot even when it didn't throw. Passing
  // only the frozen camera makes the control passes and the prompt analysis provably the same view.
  const passBoard: BoardShot = opts.includeFigureInDepth
    ? board
    : { ...board, framing: undefined, figures: [] };

  const passes = renderPasses(passBoard, space);
  // Normalise depth to what THIS camera can actually see. The room-wide range blew the floor to pure
  // white on the ground-level shot (camera 18cm up, so the near plane is centimetres away), and a
  // saturated depth map is a control image carrying no gradient at all.
  const dists = visible.map(p => p.mid).filter(n => Number.isFinite(n) && n > 0);
  const depth = dists.length >= 2
    ? depthPng(passes.passes, W, H, { near: Math.min(...dists) * 0.55, far: Math.max(...dists) * 1.45 })
    : passes.depth;
  const clay = renderShot(passBoard, space, { guides: false, mode: "clay" }) as Buffer;
  const edge = passes.edge;

  return { board, inventory, visible, inFrame, clay, depth, edge };
}

/** Write the control pair to disk so ComfyUI can be handed filenames rather than bytes. */
export async function writePair(r: ResolvedShot, dir: string, label: string): Promise<{ clay: string; depth: string; edge: string }> {
  await fs.mkdir(dir, { recursive: true });
  const clay = path.join(dir, `${label}.clay.png`);
  const depth = path.join(dir, `${label}.depth.png`);
  const edge = path.join(dir, `${label}.edge.png`);
  await fs.writeFile(clay, r.clay);
  await fs.writeFile(depth, r.depth);
  await fs.writeFile(edge, r.edge);
  return { clay, depth, edge };
}
