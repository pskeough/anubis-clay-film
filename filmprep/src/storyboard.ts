/**
 * L1.5 storyboard — shot GEOMETRY as a control layer.
 *
 * The gap this fills: shot direction is currently prose ("New camera: from across the room, low
 * angle toward the low bed"). Klein re-guesses the room's geometry on every reframe, and because
 * scene chains are sequential edits (bg_s02_02 -> _03 -> ... -> _09) the guesses compound. Seven
 * cameras into a scene the room has quietly become a different room.
 *
 * The fix is to stop describing space and start MEASURING it. Each location gets a `space.json`:
 * a coarse solid of the room and its furniture, authored ONCE (the LOCATION CANON extended from a
 * flat plate to a volume). Each shot is then just a camera on that solid. Two shots of the same
 * room are consistent because they are literally the same numbers, not because a model remembered.
 *
 * We project that solid to a clean black-on-white wireframe and hand it to Klein as an extra
 * reference. No ControlNet needed (none exists for klein 4B) — `buildKleinGraph` already chains a
 * ReferenceLatent per reference, so a board is one more entry in `refPaths`.
 *
 * Two outputs per shot, and the difference matters:
 *   - `<id>.lineart.png` — the CONTROL image. Black on white, geometry only. No text, no grid,
 *     no labels: anything written in here gets rendered as writing in the final image.
 *   - `<id>.board.svg`   — the HUMAN image. Same geometry plus labels, thirds, horizon, notes.
 *     Editable by hand or by model; it is the artifact you actually plan against.
 *
 * Both come off one primitive list through two sinks, so the board can never drift from the
 * control image it documents.
 */
import { promises as fs } from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { createCanvas } from "@napi-rs/canvas";
import { assetTriangles, assetFaceCost } from "./assets.ts";

// ---------------------------------------------------------------- vector math

export type Vec3 = [number, number, number];
export type Vec2 = [number, number];

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const norm = (a: Vec3): Vec3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

// ---------------------------------------------------------------- camera

export interface CameraSpec {
  /** world position, metres. Room is x:[-w/2,w/2] y:[0,h] z:[0,d]. */
  pos: Vec3;
  /** world point the lens is aimed at. */
  look: Vec3;
  /** horizontal field of view in degrees. 20=long lens, 40=normal, 65=wide, 85=very wide. */
  fov?: number;
  /** dutch angle in degrees, + = clockwise. This is how you get a canted "lurching" frame. */
  roll?: number;
}

export interface Basis { pos: Vec3; right: Vec3; up: Vec3; fwd: Vec3; fov: number }

function cameraBasis(c: CameraSpec): Basis {
  const fwd = norm(sub(c.look, c.pos));
  // world up, unless we're looking near-vertically (straight up at the bulb / down at the floor),
  // in which case fall back to world -Z so the basis stays defined.
  const worldUp: Vec3 = Math.abs(fwd[1]) > 0.995 ? [0, 0, -1] : [0, 1, 0];
  let right = norm(cross(fwd, worldUp));
  let up = cross(right, fwd);
  const roll = ((c.roll ?? 0) * Math.PI) / 180;
  if (roll) {
    const cs = Math.cos(roll), sn = Math.sin(roll);
    const r2: Vec3 = add(mul(right, cs), mul(up, sn));
    const u2: Vec3 = add(mul(up, cs), mul(right, -sn));
    right = r2; up = u2;
  }
  return { pos: c.pos, right, up, fwd, fov: c.fov ?? 40 };
}

/** Camera-space coords: x right, y up, z forward-depth. */
function toCam(b: Basis, p: Vec3): Vec3 {
  const d = sub(p, b.pos);
  return [dot(d, b.right), dot(d, b.up), dot(d, b.fwd)];
}

const NEAR = 0.05;

/** Perspective divide to pixels. Assumes cam-space z >= NEAR (clip first). */
function project(v: Vec3, b: Basis, w: number, h: number): Vec2 {
  const f = w / 2 / Math.tan((b.fov * Math.PI) / 360);
  return [w / 2 + (v[0] * f) / v[2], h / 2 - (v[1] * f) / v[2]];
}

/** Clip a camera-space segment against the near plane. Returns null if fully behind. */
function nearClip(a: Vec3, b: Vec3): [Vec3, Vec3] | null {
  const inA = a[2] >= NEAR, inB = b[2] >= NEAR;
  if (!inA && !inB) return null;
  if (inA && inB) return [a, b];
  const t = (NEAR - a[2]) / (b[2] - a[2]);
  const cut: Vec3 = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, NEAR];
  return inA ? [a, cut] : [cut, b];
}

// ---------------------------------------------------------------- draw sinks

export type Layer = "structure" | "prop" | "figure" | "guide" | "note";

export interface Stroke { layer: Layer; width?: number; dash?: boolean }

/** A sink consumes 2D primitives. One primitive list, several renderings. */
/** A colour is either a grey scalar or an explicit linear rgb triple, each 0..1. */
export type Col = number | [number, number, number];

interface Sink {
  poly(pts: Vec2[], s: Stroke): void;
  fill(pts: Vec2[], col: Col): void;
  circle(c: Vec2, r: number, s: Stroke): void;
  disc(c: Vec2, r: number, col: Col): void;
  text(at: Vec2, str: string, size: number): void;
}

const hex = (c: Col) => {
  const ch = (v: number) => Math.max(0, Math.min(255, Math.round(v * 255))).toString(16).padStart(2, "0");
  return typeof c === "number" ? "#" + ch(c).repeat(3) : "#" + ch(c[0]) + ch(c[1]) + ch(c[2]);
};

const LAYER_WIDTH: Record<Layer, number> = {
  structure: 3.2, prop: 2.4, figure: 2.6, guide: 1.0, note: 1.0,
};

class SvgSink implements Sink {
  private parts: string[] = [];
  constructor(private w: number, private h: number, private showGuides: boolean) {}
  private skip(s: Stroke) { return !this.showGuides && (s.layer === "guide" || s.layer === "note"); }
  private attrs(s: Stroke) {
    const col = s.layer === "guide" || s.layer === "note" ? "#9aa0a6" : "#000";
    const wd = s.width ?? LAYER_WIDTH[s.layer];
    return `stroke="${col}" stroke-width="${wd}" fill="none" stroke-linecap="round" stroke-linejoin="round"` +
      (s.dash ? ` stroke-dasharray="7 6"` : "");
  }
  poly(pts: Vec2[], s: Stroke) {
    if (this.skip(s) || pts.length < 2) return;
    const d = pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
    this.parts.push(`<polyline points="${d}" ${this.attrs(s)}/>`);
  }
  fill(pts: Vec2[], col: Col) {
    if (pts.length < 3) return;
    const d = pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
    this.parts.push(`<polygon points="${d}" fill="${hex(col)}" stroke="none"/>`);
  }
  disc(c: Vec2, r: number, col: Col) {
    if (!(r > 0.4)) return;
    this.parts.push(`<circle cx="${c[0].toFixed(1)}" cy="${c[1].toFixed(1)}" r="${r.toFixed(1)}" fill="${hex(col)}" stroke="none"/>`);
  }
  circle(c: Vec2, r: number, s: Stroke) {
    if (this.skip(s) || !(r > 0.4)) return;
    this.parts.push(`<circle cx="${c[0].toFixed(1)}" cy="${c[1].toFixed(1)}" r="${r.toFixed(1)}" ${this.attrs(s)}/>`);
  }
  text(at: Vec2, str: string, size: number) {
    if (!this.showGuides) return;
    const esc = str.replace(/[<>&]/g, (m) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" }[m] as string));
    this.parts.push(
      `<text x="${at[0].toFixed(1)}" y="${at[1].toFixed(1)}" font-family="ui-monospace,Menlo,Consolas,monospace" ` +
      `font-size="${size}" fill="#5f6368">${esc}</text>`,
    );
  }
  done(): string {
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${this.w}" height="${this.h}" viewBox="0 0 ${this.w} ${this.h}">` +
      `<rect width="${this.w}" height="${this.h}" fill="#fff"/>${this.parts.join("")}</svg>`;
  }
}

class CanvasSink implements Sink {
  private ctx: any;
  private canvas: any;
  /**
   * `bg` exists because of open-sky volumes.
   *
   * An interior's shell covers every pixel, so the clear colour never shows and pure white is
   * harmless. Once `openToSky` dropped the walls and ceiling, a fifth of an exterior frame became
   * this colour — and the clay pass is what Qwen takes as its EDIT REFERENCE. Handed a frame that is
   * 22% pure white it did exactly what that means in photographs: it returned a white-cyclorama
   * studio product shot, over-clean and floodlit, with the highway furniture posed on seamless.
   * library-interior, 0% white, came back as a photograph.
   *
   * So the sky has to be a sky tone, not a clear colour. Overcast grey, distinctly off-white.
   */
  constructor(private w: number, private h: number, private showGuides: boolean, bg = "#fff") {
    this.canvas = createCanvas(w, h);
    this.ctx = this.canvas.getContext("2d");
    this.ctx.fillStyle = bg;
    this.ctx.fillRect(0, 0, w, h);
    this.ctx.lineCap = "round";
    this.ctx.lineJoin = "round";
  }
  private skip(s: Stroke) { return !this.showGuides && (s.layer === "guide" || s.layer === "note"); }
  private set(s: Stroke) {
    this.ctx.strokeStyle = s.layer === "guide" || s.layer === "note" ? "#9aa0a6" : "#000";
    this.ctx.lineWidth = s.width ?? LAYER_WIDTH[s.layer];
    this.ctx.setLineDash(s.dash ? [7, 6] : []);
  }
  poly(pts: Vec2[], s: Stroke) {
    if (this.skip(s) || pts.length < 2) return;
    this.set(s);
    this.ctx.beginPath();
    this.ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) this.ctx.lineTo(pts[i][0], pts[i][1]);
    this.ctx.stroke();
  }
  fill(pts: Vec2[], col: Col) {
    if (pts.length < 3) return;
    this.ctx.setLineDash([]);
    this.ctx.fillStyle = hex(col);
    this.ctx.beginPath();
    this.ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) this.ctx.lineTo(pts[i][0], pts[i][1]);
    this.ctx.closePath();
    this.ctx.fill();
  }
  disc(c: Vec2, r: number, col: Col) {
    if (!(r > 0.4)) return;
    this.ctx.fillStyle = hex(col);
    this.ctx.beginPath();
    this.ctx.arc(c[0], c[1], r, 0, Math.PI * 2);
    this.ctx.fill();
  }
  circle(c: Vec2, r: number, s: Stroke) {
    if (this.skip(s) || !(r > 0.4)) return;
    this.set(s);
    this.ctx.beginPath();
    this.ctx.arc(c[0], c[1], r, 0, Math.PI * 2);
    this.ctx.stroke();
  }
  text(at: Vec2, str: string, size: number) {
    if (!this.showGuides) return;
    this.ctx.setLineDash([]);
    this.ctx.fillStyle = "#5f6368";
    this.ctx.font = `${size}px monospace`;
    this.ctx.fillText(str, at[0], at[1]);
  }
  png(): Buffer { return this.canvas.toBuffer("image/png"); }
}

// ---------------------------------------------------------------- space schema

export interface Prop {
  id: string;
  /** box = furniture solid; panel = flat on a wall (tapestry, poster, window); lamp = hanging light. */
  /** box/panel/lamp/cyl are primitives; the rest are archetypes expanded into them at render time. */
  /** model = a real mesh from the shared asset library, placed rather than authored. See assets.ts. */
  type: "box" | "panel" | "lamp" | "cyl"
    | "shelf" | "chair" | "table" | "cot" | "column" | "clutter"
    | "ladder" | "stack" | "beams" | "window"
    | "model";
  /** model only: the library slug. `python cli.py find <term>` prints a ready stanza. */
  asset?: string;
  /** model only. contain (default) fits the mesh inside `size` keeping its proportions and seats
   *  it on the box floor; stretch fills `size` exactly and distorts. A model's `size` is a BOUND,
   *  not an extent — that is the one place model props differ from every other solid. */
  fit?: "contain" | "stretch";
  /** how many repeats: debris pieces, ladder rungs, ceiling joists, window lights. */
  count?: number;
  /** Diffuse reflectance, 0..1. This is the MATERIAL, and it is what gives the clay render tonal
   *  separation between a leather chair, a limestone column and loose paper — without it every
   *  surface returns the same grey and the init image is a uniform-albedo massing model. */
  albedo?: number;
  /** cyl only: radius, defaulting to half the x extent. */
  radius?: number;
  /** cyl only: which world axis the cylinder runs along. Default "y" (upright, e.g. a column). */
  axis?: "x" | "y" | "z";
  /** Set by expandProps: the id of the authored prop this part came from. Lets the object-ID pass
   *  report "the chair is visible" rather than "chair_armr is visible". */
  owner?: string;
  /** Prompt-facing noun phrase. Falls back to `label`, then `id`. Authored per location so the
   *  same machinery generates usable prose for any scene without bespoke code. */
  desc?: string;
  label?: string;
  /** Does this object have a describable FRONT? Defaults by type — a chair and a cot do, a rug and
   *  a ladder do not, even though both need a yaw to sit right. Keeping this separate from `yaw`
   *  matters: yaw is geometry, this is language, and conflating them makes every turned object
   *  acquire a facing the prompt then asserts to the model. */
  facing?: boolean;
  // box
  pos?: Vec3;            // centre of the solid
  size?: Vec3;           // full extents
  yaw?: number;          // degrees about Y
  // panel
  wall?: "back" | "front" | "left" | "right" | "ceiling";
  u?: number;            // 0..1 along the wall
  y?: number;            // centre height (metres) for wall panels
  w?: number; h?: number;
  // lamp
  r?: number;
  cord?: number;         // cord length down from ceiling
  /** lamp only: relative output. 1 = the room's reference practical. */
  intensity?: number;
}

export interface SpaceSpec {
  id: string;
  /** One look description per LOCATION. Identical wording across every shot is what keeps palette,
   *  film character and lighting constant between cuts; per-shot phrasing is what makes them drift. */
  look?: string;
  negative?: string;
  /** metres. x = width, z = depth, y = height. Room spans x[-w/2,w/2] y[0,h] z[0,d]. */
  room: { w: number; d: number; h: number };
  /** Optional override for the depth map's fixed metric range. See depthRange(). */
  depth?: { near: number; far: number };
  /**
   * The volume is OUTDOORS: omit the ceiling quad from the room shell.
   *
   * Exteriors have to be authored as a bounded box because that is the only volume the schema and
   * the camera solver understand, but the shell then puts a lid on them. rural-highway rendered with
   * a visible ceiling grid arching over the road, and a lid is not a neutral mistake: it is stated to
   * the diffusion model as geometry, so it produces a covered street.
   *
   * With no ceiling face, upward rays hit nothing and land at the far end of the depth range, which
   * is exactly what sky should be. Defaults false, so every existing interior is unaffected.
   */
  openToSky?: boolean;
  props: Prop[];
  notes?: string;
}

export interface FigureSpec {
  id: string;
  /** floor position [x, z] in metres. */
  at: [number, number];
  /** standing height in metres. */
  height?: number;
  /** degrees; 0 faces +z (deeper into the room), 180 faces the front wall. */
  yaw?: number;
  pose?: "standing" | "slumped" | "lying" | "sitting" | "reaching";
  label?: string;
}

export interface ShotSpec {
  id: string;
  /** location id -> locations/<id>/space.json */
  location: string;
  /** Directorial form: say the shot, let the solver place the lens. Preferred over `camera`. */
  framing?: Framing;
  /** Explicit camera, for when you want to override the solver. */
  camera?: CameraSpec;
  figures?: FigureSpec[];
  /** free text carried onto the human board only, never into the control image. */
  note?: string;
  /** which asset request in plan.json this board controls. */
  targets?: string;
  width?: number;
  height?: number;
}

// ---------------------------------------------------------------- geometry

type Seg = [Vec3, Vec3];

function boxEdges(centre: Vec3, size: Vec3, yawDeg = 0): Seg[] {
  const [sx, sy, sz] = [size[0] / 2, size[1] / 2, size[2] / 2];
  const corners: Vec3[] = [
    [-sx, -sy, -sz], [sx, -sy, -sz], [sx, -sy, sz], [-sx, -sy, sz],
    [-sx, sy, -sz], [sx, sy, -sz], [sx, sy, sz], [-sx, sy, sz],
  ];
  const yaw = (yawDeg * Math.PI) / 180, cs = Math.cos(yaw), sn = Math.sin(yaw);
  const world = corners.map((c): Vec3 => [
    centre[0] + c[0] * cs + c[2] * sn,
    centre[1] + c[1],
    centre[2] - c[0] * sn + c[2] * cs,
  ]);
  const idx: [number, number][] = [
    [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  return idx.map(([a, b]): Seg => [world[a], world[b]]);
}

/** The room shell: floor, ceiling, and the four vertical corner posts. */
function roomEdges(r: SpaceSpec["room"]): Seg[] {
  const x = r.w / 2;
  const f: Vec3[] = [[-x, 0, 0], [x, 0, 0], [x, 0, r.d], [-x, 0, r.d]];
  const c: Vec3[] = f.map((p): Vec3 => [p[0], r.h, p[2]]);
  const segs: Seg[] = [];
  for (let i = 0; i < 4; i++) {
    segs.push([f[i], f[(i + 1) % 4]]);
    segs.push([c[i], c[(i + 1) % 4]]);
    segs.push([f[i], c[i]]);
  }
  return segs;
}

function panelEdges(p: Prop, r: SpaceSpec["room"]): Seg[] {
  const w = p.w ?? 0.8, h = p.h ?? 1.0, u = p.u ?? 0.5, y = p.y ?? 1.2;
  const hx = r.w / 2;
  let quad: Vec3[];
  if (p.wall === "left" || p.wall === "right") {
    const x = p.wall === "left" ? -hx + 0.02 : hx - 0.02;
    const z = u * r.d;
    quad = [[x, y - h / 2, z - w / 2], [x, y - h / 2, z + w / 2], [x, y + h / 2, z + w / 2], [x, y + h / 2, z - w / 2]];
  } else if (p.wall === "ceiling") {
    const x = (u - 0.5) * r.w, z = (p.y ?? 0.5) * r.d;
    quad = [[x - w / 2, r.h - 0.02, z - h / 2], [x + w / 2, r.h - 0.02, z - h / 2], [x + w / 2, r.h - 0.02, z + h / 2], [x - w / 2, r.h - 0.02, z + h / 2]];
  } else {
    const z = p.wall === "front" ? 0.02 : r.d - 0.02;
    const x = (u - 0.5) * r.w;
    quad = [[x - w / 2, y - h / 2, z], [x + w / 2, y - h / 2, z], [x + w / 2, y + h / 2, z], [x - w / 2, y + h / 2, z]];
  }
  return quad.map((v, i): Seg => [v, quad[(i + 1) % 4]]);
}

/** Mannequin skeleton in unit-height local space, then placed. Deliberately crude: the board is a
 *  geometry brief, not a character design — identity comes from the character sheet reference. */
const POSES: Record<string, Record<string, Vec3>> = {
  standing: {
    head: [0, 0.93, 0], neck: [0, 0.82, 0], hip: [0, 0.50, 0],
    shL: [-0.11, 0.80, 0], shR: [0.11, 0.80, 0],
    elL: [-0.15, 0.64, 0.02], elR: [0.15, 0.64, 0.02],
    hnL: [-0.16, 0.47, 0.04], hnR: [0.16, 0.47, 0.04],
    hpL: [-0.07, 0.49, 0], hpR: [0.07, 0.49, 0],
    knL: [-0.07, 0.26, 0.01], knR: [0.07, 0.26, 0.01],
    ftL: [-0.07, 0, 0.03], ftR: [0.07, 0, 0.03],
  },
  slumped: {
    head: [0.02, 0.62, 0.16], neck: [0.01, 0.54, 0.11], hip: [0, 0.26, 0],
    shL: [-0.12, 0.52, 0.10], shR: [0.12, 0.52, 0.12],
    elL: [-0.17, 0.38, 0.16], elR: [0.17, 0.38, 0.16],
    hnL: [-0.15, 0.24, 0.24], hnR: [0.15, 0.24, 0.24],
    hpL: [-0.08, 0.25, 0], hpR: [0.08, 0.25, 0],
    knL: [-0.10, 0.16, 0.30], knR: [0.10, 0.16, 0.30],
    ftL: [-0.11, 0.02, 0.52], ftR: [0.11, 0.02, 0.52],
  },
  lying: {
    head: [0, 0.09, -0.42], neck: [0, 0.08, -0.32], hip: [0, 0.07, 0],
    shL: [-0.11, 0.08, -0.29], shR: [0.11, 0.08, -0.29],
    elL: [-0.17, 0.06, -0.14], elR: [0.17, 0.06, -0.14],
    hnL: [-0.19, 0.05, 0.02], hnR: [0.19, 0.05, 0.02],
    hpL: [-0.07, 0.07, 0], hpR: [0.07, 0.07, 0],
    knL: [-0.08, 0.06, 0.24], knR: [0.08, 0.06, 0.24],
    ftL: [-0.08, 0.04, 0.47], ftR: [0.08, 0.04, 0.47],
  },
  sitting: {
    head: [0, 0.72, -0.04], neck: [0, 0.62, -0.02], hip: [0, 0.30, 0],
    shL: [-0.11, 0.60, -0.02], shR: [0.11, 0.60, -0.02],
    elL: [-0.15, 0.46, 0.02], elR: [0.15, 0.46, 0.02],
    hnL: [-0.14, 0.31, 0.10], hnR: [0.14, 0.31, 0.10],
    hpL: [-0.07, 0.29, 0], hpR: [0.07, 0.29, 0],
    knL: [-0.08, 0.28, 0.26], knR: [0.08, 0.28, 0.26],
    ftL: [-0.08, 0.02, 0.28], ftR: [0.08, 0.02, 0.28],
  },
  reaching: {
    head: [0, 0.93, 0], neck: [0, 0.82, 0], hip: [0, 0.50, 0],
    shL: [-0.11, 0.80, 0], shR: [0.11, 0.80, 0],
    elL: [-0.15, 0.64, 0.02], elR: [0.18, 0.90, 0.10],
    hnL: [-0.16, 0.47, 0.04], hnR: [0.20, 1.06, 0.22],
    hpL: [-0.07, 0.49, 0], hpR: [0.07, 0.49, 0],
    knL: [-0.07, 0.26, 0.01], knR: [0.07, 0.26, 0.01],
    ftL: [-0.07, 0, 0.03], ftR: [0.07, 0, 0.03],
  },
};

const BONES: [string, string][] = [
  ["neck", "hip"], ["shL", "shR"], ["neck", "shL"], ["neck", "shR"],
  ["shL", "elL"], ["elL", "hnL"], ["shR", "elR"], ["elR", "hnR"],
  ["hip", "hpL"], ["hip", "hpR"], ["hpL", "knL"], ["knL", "ftL"], ["hpR", "knR"], ["knR", "ftR"],
];

function figureGeom(f: FigureSpec): { bones: Seg[]; head: { c: Vec3; r: number } } {
  const H = f.height ?? 1.7;
  const pose = POSES[f.pose ?? "standing"] ?? POSES.standing;
  const yaw = ((f.yaw ?? 180) * Math.PI) / 180, cs = Math.cos(yaw), sn = Math.sin(yaw);
  const put = (p: Vec3): Vec3 => {
    const [lx, ly, lz] = [p[0] * H, p[1] * H, p[2] * H];
    return [f.at[0] + lx * cs + lz * sn, ly, f.at[1] - lx * sn + lz * cs];
  };
  const bones = BONES.map(([a, b]): Seg => [put(pose[a]), put(pose[b])]);
  return { bones, head: { c: put(pose.head), r: 0.105 * H } };
}

// ---------------------------------------------------------------- solid faces

/** A shaded surface. `n` faces the side we want visible; faces pointing away get culled.
 *  `owner` is the id of the space.json prop this surface came from (not the expanded part), which
 *  is what lets the renderer report exactly which objects a given camera can actually see. */
interface Face { pts: Vec3[]; n: Vec3; base: number; owner?: string; emit?: number }

const faceNormal = (p: Vec3[]): Vec3 => norm(cross(sub(p[1], p[0]), sub(p[2], p[0])));
const centroid = (p: Vec3[]): Vec3 =>
  mul(p.reduce((a, b) => add(a, b), [0, 0, 0] as Vec3), 1 / p.length);

/** Room shell as inward-facing quads. Floor light, ceiling dark, walls mid — the tonal skeleton a
 *  storyboard artist blocks in first, and what actually reads as depth to an image model. */
/** Orient a face's normal toward `target` (room shell: the room centre) or away from it (solids). */
function orient(pts: Vec3[], target: Vec3, toward: boolean, base: number): Face {
  let n = faceNormal(pts);
  const facing = dot(n, sub(target, centroid(pts)));
  if (toward ? facing < 0 : facing > 0) n = mul(n, -1);
  return { pts, n, base };
}

function roomFaces(r: SpaceSpec["room"], openToSky = false): Face[] {
  const x = r.w / 2, d = r.d, h = r.h;
  const mid: Vec3 = [0, h / 2, d / 2];
  // Normals must point INWARD — the camera lives inside this box. Hand-written flip flags got this
  // wrong for the floor, ceiling and both side walls, which silently culled most of the room.
  const F = (pts: Vec3[], base: number) => orient(pts, mid, true, base);
  const faces = [
    F([[-x, 0, 0], [x, 0, 0], [x, 0, d], [-x, 0, d]], 0.62),          // floor
    F([[-x, 0, d], [x, 0, d], [x, h, d], [-x, h, d]], 0.50),          // back wall (door end)
    F([[-x, 0, 0], [x, 0, 0], [x, h, 0], [-x, h, 0]], 0.50),          // front wall
    F([[-x, 0, 0], [-x, 0, d], [-x, h, d], [-x, h, 0]], 0.44),        // left wall
    F([[x, 0, 0], [x, 0, d], [x, h, d], [x, h, 0]], 0.44),            // right wall
  ];
  // An outdoor volume is still a box, because that is the only shape the camera solver understands,
  // but it must not have a lid — and, as it turns out, it must not have walls either.
  //
  // Removing only the ceiling was half a fix. A 24 x 28 m grove still rendered as two enormous tiled
  // walls meeting at a corner, which reads as a courtyard; the shell was the dominant geometry in
  // frame and no amount of prop authoring could out-vote it. Beyond the trees there should be
  // distance, and a ray that hits nothing lands at the far end of the depth range, which IS distance.
  //
  // The floor stays. Ground is real outdoors, it carries the perspective that tells the model where
  // the camera is standing, and without it the lower half of frame goes to void.
  //
  // The camera solve is unaffected: it bounds the camera by the room DIMENSIONS, not by these
  // rasterised faces, so an open volume still cannot be shot from outside itself.
  if (!openToSky) {
    return [
      faces[0],
      F([[-x, h, 0], [x, h, 0], [x, h, d], [-x, h, d]], 0.30),                     // ceiling
      ...faces.slice(1),
    ];
  }
  return [faces[0]];
}

function boxFaces(centre: Vec3, size: Vec3, yawDeg = 0, base = 0.34): Face[] {
  const [sx, sy, sz] = [size[0] / 2, size[1] / 2, size[2] / 2];
  const yaw = (yawDeg * Math.PI) / 180, cs = Math.cos(yaw), sn = Math.sin(yaw);
  const P = (lx: number, ly: number, lz: number): Vec3 => [
    centre[0] + lx * cs + lz * sn, centre[1] + ly, centre[2] - lx * sn + lz * cs,
  ];
  const quads: Vec3[][] = [
    [P(-sx, sy, -sz), P(sx, sy, -sz), P(sx, sy, sz), P(-sx, sy, sz)],       // top
    [P(-sx, -sy, -sz), P(-sx, -sy, sz), P(sx, -sy, sz), P(sx, -sy, -sz)],   // bottom
    [P(-sx, -sy, -sz), P(sx, -sy, -sz), P(sx, sy, -sz), P(-sx, sy, -sz)],
    [P(sx, -sy, sz), P(-sx, -sy, sz), P(-sx, sy, sz), P(sx, sy, sz)],
    [P(-sx, -sy, sz), P(-sx, -sy, -sz), P(-sx, sy, -sz), P(-sx, sy, sz)],
    [P(sx, -sy, -sz), P(sx, -sy, sz), P(sx, sy, sz), P(sx, sy, -sz)],
  ];
  return quads.map((pts) => orient(pts, centre, false, base));   // solids face outward
}

/** Deterministic PRNG (mulberry32). Scatter must be identical every render or the location stops
 *  being canon — the same room would have differently-placed debris in every shot. */
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const hashId = (s: string) => { let h = 2166136261 >>> 0; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; };

/** Rotate a local offset about +Y and translate to a world centre. This is the SAME convention as
 *  boxFaces, and everything that positions a sub-part must go through it — a part placed with raw
 *  world arithmetic does not move when its parent turns. */
const place = (centre: Vec3, yawDeg: number) => {
  const yr = (yawDeg * Math.PI) / 180, cs = Math.cos(yr), sn = Math.sin(yr);
  return (lx: number, ly: number, lz: number): Vec3 => [
    centre[0] + lx * cs + lz * sn, centre[1] + ly, centre[2] - lx * sn + lz * cs,
  ];
};

/** A cylinder as an n-gon prism. Columns rendered as boxes read as pillars of stacked crates; at
 *  ~14 segments the silhouette reads as round in a depth map without meaningful extra cost.
 *  `yawDeg` turns the whole cylinder about +Y, which is what a chair arm needs when the chair does
 *  not face down the z-axis — an unrotated arm on a rotated chair is a chair with its arms in the
 *  wrong place, and no amount of prompt wording fixes that. */
function cylinderFaces(centre: Vec3, radius: number, length: number, axis: "x" | "y" | "z" = "y", segs = 20, base = 0.34, yawDeg = 0): Face[] {
  // Sagitta error of an n-gon is r(1-cos(pi/n)); at 20 segments that is ~1.2% of radius, roughly
  // 1mm on a chair arm and 4mm on a column — below the perceptual threshold for "round" while
  // staying cheap. 8 segments (7.6%) reads as a faceted barrel, which is its own kind of wrong.
  const half = length / 2;
  const put = place(centre, yawDeg);
  const ringAt = (t: number): Vec3[] => Array.from({ length: segs }, (_v, i): Vec3 => {
    const a = (i / segs) * Math.PI * 2, u = Math.cos(a) * radius, v = Math.sin(a) * radius;
    if (axis === "y") return put(u, t, v);
    if (axis === "x") return put(t, u, v);
    return put(u, v, t);
  });
  const lo = ringAt(-half), hi = ringAt(half);
  const faces: Face[] = [];
  for (let i = 0; i < segs; i++) {
    const j = (i + 1) % segs;
    faces.push(orient([lo[i], lo[j], hi[j], hi[i]], centre, false, base));
  }
  // Cap normals point along the cylinder's own axis, which yaws with it — so the "away" reference
  // point for each cap has to be built through `put` as well, not from a world axis.
  const axisDir: Vec3 = axis === "y" ? [0, 1, 0] : axis === "x" ? [1, 0, 0] : [0, 0, 1];
  faces.push(orient(hi, put(-axisDir[0], -axisDir[1], -axisDir[2]), false, base));
  faces.push(orient(lo, put(axisDir[0], axisDir[1], axisDir[2]), false, base));
  return faces;
}

function panelFace(p: Prop, r: SpaceSpec["room"]): Face | null {
  const segs = panelEdges(p, r);
  if (segs.length < 4) return null;
  const pts = segs.map((s) => s[0]);
  // Orient toward the room centre like the shell does — this was left unfixed when roomFaces and
  // boxFaces were corrected, so wall panels were culled or lit on the wrong side at random.
  return orient(pts, [0, r.h / 2, r.d / 2], true, p.id === "door" ? 0.18 : 0.40);
}

/**
 * Furniture archetypes -> boxes.
 *
 * A chair authored as a single 1.05 m cube renders as a featureless white block that fills the
 * frame and tells the model nothing. Real objects are assemblies, and it is the assembly — shelf
 * boards, chair backs, table legs — that carries the mid-frequency depth structure a ControlNet
 * can actually read. Archetypes keep space.json terse while emitting geometry with shape.
 */
function expandProp(p: Prop): Prop[] {
  const [w, h, d] = p.size ?? [1, 1, 1];
  const yaw = p.yaw ?? 0;
  // EVERY sub-part is authored in the prop's OWN frame — x right, y up, z forward — and placed
  // through `at`, which rotates about the prop's centre. Building parts from world arithmetic
  // (`cz - d/2`) and then setting each part's own yaw looks equivalent and is not: the parts stay
  // where they were and merely spin in place, so a chair turned 122 degrees keeps its back at -Z
  // and its arms on the world x-axis. Every archetype was wrong that way; this is the fix.
  const at = place(p.pos ?? [0, 0, 0], yaw);
  const y0 = -h / 2;                                       // floor line, in the prop's own frame
  const alb = p.albedo;
  const box = (id: string, lp: Vec3, size: Vec3, extraYaw = 0): Prop =>
    ({ id, type: "box", pos: at(lp[0], lp[1], lp[2]), size, yaw: yaw + extraYaw, label: p.label, albedo: alb });
  const cyl = (id: string, lp: Vec3, len: number, r: number, axis: "x" | "y" | "z"): Prop => ({
    id, type: "cyl", label: p.label, pos: at(lp[0], lp[1], lp[2]), axis, yaw, albedo: alb,
    size: axis === "z" ? [r * 2, r * 2, len] : axis === "x" ? [len, r * 2, r * 2] : [r * 2, len, r * 2],
    radius: r,
  });

  switch (p.type) {
    case "shelf": {                                        // side panels + individual boards
      const parts: Prop[] = [
        box(`${p.id}_l`, [-w / 2 + 0.03, 0, 0], [0.06, h, d]),
        box(`${p.id}_r`, [w / 2 - 0.03, 0, 0], [0.06, h, d]),
        box(`${p.id}_back`, [0, 0, -d / 2 + 0.03], [w, h, 0.06]),
      ];
      const shelves = Math.max(2, Math.round(h / 0.34));
      const along = d >= w ? "z" : "x";                    // books run down the long axis
      const runLen = (along === "z" ? d : w) - 0.12;
      const rand = rng(hashId(p.id));
      for (let i = 0; i <= shelves; i++) {
        const y = y0 + (h * i) / shelves;
        parts.push(box(`${p.id}_s${i}`, [0, y, 0], [w - 0.1, 0.05, d - 0.06]));
        if (i >= shelves) continue;
        // INDIVIDUAL SPINES, not one slab. A shelf rendered as a solid block is a block; the
        // staggered depth of hundreds of spines is the single richest source of mid-frequency
        // structure in a library, and it is what makes the depth map read as books.
        const bandH = h / shelves - 0.1;
        let t = 0.06;
        while (t < runLen) {
          const bw = 0.025 + rand() * 0.045;               // spine width 2.5–7cm
          const bh = bandH * (0.72 + rand() * 0.26);       // uneven heights
          const inset = rand() * 0.05;                     // some pushed back on the shelf
          const lean = (rand() < 0.06 ? 6 + rand() * 10 : 0); // the odd toppled volume
          const c: Vec3 = along === "z"
            ? [(rand() - 0.5) * 0.02, y + 0.025 + bh / 2, -d / 2 + 0.06 + t + bw / 2]
            : [-w / 2 + 0.06 + t + bw / 2, y + 0.025 + bh / 2, (rand() - 0.5) * 0.02];
          const s: Vec3 = along === "z"
            ? [(w - 0.16) * (0.8 + rand() * 0.2) - inset, bh, bw]
            : [bw, bh, (d - 0.16) * (0.8 + rand() * 0.2) - inset];
          parts.push(box(`${p.id}_bk${i}_${t.toFixed(2)}`, c, s, lean));
          t += bw + 0.004;
        }
      }
      return parts;
    }
    case "clutter": {
      // Debris on a library floor is BOOKS: flat slabs, fanned paper, and the odd small pile. The
      // previous version scattered near-cubic lumps 6–30cm on a side, which renders as gravel and
      // reads to the model as rubble. Shape carries the noun here — silhouette is most of what
      // survives into the depth map at this scale.
      const rand = rng(hashId(p.id));
      const n = Math.max(4, Math.round(p.count ?? 26));
      const parts: Prop[] = [];
      for (let i = 0; i < n; i++) {
        const lx = (rand() - 0.5) * w, lz = (rand() - 0.5) * d;
        const roll = rand();
        if (roll < 0.22) {                                 // a small toppled stack
          const bw = 0.16 + rand() * 0.1, bd = 0.22 + rand() * 0.12;
          const k = 2 + Math.floor(rand() * 4);
          for (let j = 0; j < k; j++) {
            const t = 0.028 + rand() * 0.022;
            parts.push(box(`${p.id}_s${i}_${j}`,
              [lx + (rand() - 0.5) * 0.05, y0 + 0.015 + j * t + t / 2, lz + (rand() - 0.5) * 0.05],
              [bw, t, bd], rand() * 40 - 20));
          }
        } else {                                           // one book, lying open or shut
          const bw = 0.15 + rand() * 0.13, bd = 0.2 + rand() * 0.14;
          const t = roll < 0.62 ? 0.03 + rand() * 0.03 : 0.008 + rand() * 0.008;  // book / loose leaf
          parts.push(box(`${p.id}_c${i}`, [lx, y0 + t / 2, lz], [bw, t, bd], rand() * 180));
        }
      }
      return parts;
    }
    case "chair": {
      // ROLLED arms and a cushion with a round front edge. Furniture rendered as cuboids produces
      // cuboid furniture in the output — the model reproduces the silhouette it is given, and a
      // square-armed square-cushioned block is what "boxy and unrealistic" actually looks like.
      // Cylinders where real upholstery is round is the cheapest possible fix for that.
      const seatH = h * 0.42, armR = 0.11;
      // No full-height base block. A solid crate from floor to seat with a seat slab on top reads
      // from some angles as a chair sitting on a box — which is exactly the "split chair" artifact.
      // A real armchair is a continuous upholstered mass on a short skirt, so build it that way:
      // the body spans skirt-to-seat as ONE volume, inset from the arms, and the skirt is shallow.
      const skirt = 0.13;
      return [
        box(`${p.id}_skirt`, [0, y0 + skirt / 2, 0], [w - 0.16, skirt, d - 0.16]),
        box(`${p.id}_body`, [0, y0 + skirt + (seatH - skirt) / 2 + 0.04, 0], [w - 0.06, seatH - skirt + 0.08, d - 0.08]),
        cyl(`${p.id}_seatfront`, [0, y0 + seatH + 0.06, d / 2 - 0.13], w - 0.24, 0.10, "x"),
        box(`${p.id}_back`, [0, y0 + h * 0.70, -d / 2 + 0.13], [w - 0.16, h * 0.54, 0.22]),
        cyl(`${p.id}_backtop`, [0, y0 + h * 0.95, -d / 2 + 0.13], w - 0.16, 0.11, "x"),
        cyl(`${p.id}_arml`, [-w / 2 + armR, y0 + seatH + 0.17, 0.01], d - 0.14, armR, "z"),
        cyl(`${p.id}_armr`, [w / 2 - armR, y0 + seatH + 0.17, 0.01], d - 0.14, armR, "z"),
      ];
    }
    case "table": {
      const legs: Prop[] = [];
      for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
        legs.push(box(`${p.id}_leg${sx}${sz}`, [sx * (w / 2 - 0.06), y0 + (h - 0.05) / 2, sz * (d / 2 - 0.06)], [0.07, h - 0.05, 0.07]));
      }
      return [box(`${p.id}_top`, [0, y0 + h - 0.025, 0], [w, 0.05, d]), ...legs];
    }
    case "cot": {
      // The side table reads correctly from every angle and the cot never did. The difference is
      // silhouette legibility, not size or lighting: a table is four legs and a top, which cannot
      // be mistaken for anything else, while this was two stacked slabs on TWO DIAGONAL legs — a
      // floating bench with no bed cues at all. Told "there is a cot here" and shown an ambiguous
      // low mass, a generator puts a bed wherever it thinks a bed belongs.
      //
      // A bed is read from three things: a long low deck, the lump of a pillow at one end, and the
      // heavier mass of bedding at the other. That asymmetry is also what makes its orientation
      // legible, which is the second half of the complaint.
      const legH = h * 0.58, deck = y0 + legH;
      const parts: Prop[] = [];
      for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
        parts.push(box(`${p.id}_leg${sx > 0 ? "r" : "l"}${sz > 0 ? "h" : "f"}`,
          [sx * (w / 2 - 0.07), y0 + legH / 2, sz * (d / 2 - 0.11)], [0.06, legH, 0.06]));
      }
      for (const sx of [-1, 1]) {
        parts.push(box(`${p.id}_rail${sx > 0 ? "r" : "l"}`, [sx * (w / 2 - 0.04), deck - 0.05, 0], [0.07, 0.09, d - 0.16]));
      }
      for (const sz of [-1, 1]) {
        parts.push(box(`${p.id}_end${sz > 0 ? "h" : "f"}`, [0, deck - 0.05, sz * (d / 2 - 0.05)], [w - 0.08, 0.08, 0.07]));
      }
      parts.push(box(`${p.id}_deck`, [0, deck, 0], [w - 0.10, 0.045, d - 0.14]));
      parts.push(box(`${p.id}_roll`, [0, deck + 0.075, -0.05], [w - 0.16, 0.10, d - 0.34]));
      // Head end is local +Z, which is the prop's FRONT — so an authored yaw points the pillow.
      parts.push(box(`${p.id}_pillow`, [0, deck + 0.16, d / 2 - 0.32], [w * 0.60, 0.12, 0.40], -7));
      parts.push(box(`${p.id}_blanket0`, [0.02, deck + 0.15, -d / 2 + 0.40], [w - 0.20, 0.09, 0.46], 4));
      parts.push(box(`${p.id}_blanket1`, [-0.03, deck + 0.22, -d / 2 + 0.44], [w - 0.30, 0.07, 0.38], -6));
      return parts;
    }
    case "column": {                                       // round shaft with a base and capital
      return [
        cyl(`${p.id}_shaft`, [0, 0, 0], h, (w * 0.8) / 2, "y"),
        box(`${p.id}_base`, [0, y0 + 0.13, 0], [w, 0.26, d]),
        box(`${p.id}_cap`, [0, y0 + h - 0.16, 0], [w * 1.08, 0.32, d * 1.08]),
      ];
    }
    case "ladder": {                                       // rolling library ladder, leaning
      const rails = 0.055, rungs = Math.max(3, Math.round(p.count ?? Math.round(h / 0.3)));
      const parts: Prop[] = [
        box(`${p.id}_rl`, [-w / 2 + rails, 0, 0], [rails * 2, h, rails * 2]),
        box(`${p.id}_rr`, [w / 2 - rails, 0, 0], [rails * 2, h, rails * 2]),
      ];
      for (let i = 0; i < rungs; i++) {
        parts.push(cyl(`${p.id}_g${i}`, [0, y0 + 0.12 + (i * (h - 0.24)) / (rungs - 1), 0], w - rails * 2, 0.022, "x"));
      }
      return parts;
    }
    case "stack": {                                        // a leaning pile of volumes on the floor
      const rand = rng(hashId(p.id));
      const parts: Prop[] = [];
      let y = y0;
      let i = 0;
      while (y < y0 + h - 0.02) {
        const t = 0.032 + rand() * 0.028;
        parts.push(box(`${p.id}_v${i++}`,
          [(rand() - 0.5) * 0.05, y + t / 2, (rand() - 0.5) * 0.05],
          [w * (0.82 + rand() * 0.18), t, d * (0.82 + rand() * 0.18)], rand() * 22 - 11));
        y += t;
      }
      return parts;
    }
    case "lamp": {
      // A lamp used to contribute NO geometry — it was a light position plus a flat white disc the
      // size of the shade, which renders as a featureless glowing ball hanging in the room and is
      // duly reproduced as one. The fixture is an object: it occludes, it casts, and it is the
      // brightest thing in frame, so it needs to be built like everything else.
      const r = p.r ?? 0.09;
      if ((p.cord ?? 0.35) > 0) {                          // hung from the ceiling
        return [
          cyl(`${p.id}_cord`, [0, (p.cord ?? 0.35) / 2, 0], p.cord ?? 0.35, 0.008, "y"),
          cyl(`${p.id}_shade`, [0, 0, 0], r * 1.1, r, "y"),
        ];
      }
      const drop = Math.min(0.34, r * 1.9);                // table lamp: shade, stem, foot
      return [
        cyl(`${p.id}_foot`, [0, -drop, 0], 0.035, r * 0.72, "y"),
        cyl(`${p.id}_stem`, [0, -drop / 2, 0], drop, 0.017, "y"),
        cyl(`${p.id}_shade`, [0, 0.03, 0], r * 1.9, r * 0.60, "x"),
      ];
    }
    case "beams": {                                        // repeated joists — ceiling or floor structure
      // The ceiling is a fifth of every wide frame and renders as one featureless plane. A joist run
      // gives it rhythm and, more usefully, gives the depth map strong horizontal structure at the
      // top of frame that the model can hang a room shape on.
      const n = Math.max(2, Math.round(p.count ?? Math.round(d / 1.2)));
      const bd = Math.max(0.12, d / (n * 3));
      return Array.from({ length: n }, (_v, i) =>
        box(`${p.id}_b${i}`, [0, 0, -d / 2 + bd + (i * (d - 2 * bd)) / (n - 1)], [w, h, bd]));
    }
    case "window": {                                       // frame + glazing bars, set in a reveal
      // A window authored as a flat panel is a rectangle of wall in a slightly different grey. The
      // frame and bars are what make it read as a window from any angle, and the reveal gives the
      // opening real depth instead of a decal.
      const thin = Math.min(w, h, d);                       // the axis that goes into the wall
      const horiz = thin === w ? "x" : thin === d ? "z" : "y";
      const cols = Math.max(1, Math.round(p.count ?? 2));
      const fw = 0.09;                                      // frame member thickness
      const spanW = horiz === "x" ? d : w;                  // the opening's width in-plane
      const parts: Prop[] = [];
      const S = (a: number, b: number, c: number): Vec3 => horiz === "x" ? [thin, b, a] : [a, b, c ? c : thin];
      const P = (u: number, v: number): Vec3 => horiz === "x" ? [0, v, u] : [u, v, 0];
      parts.push(box(`${p.id}_hd`, P(0, h / 2 - fw / 2), S(spanW, fw, 0)));       // head
      parts.push(box(`${p.id}_sl`, P(0, -h / 2 + fw / 2), S(spanW, fw, 0)));      // sill
      for (const s of [-1, 1]) {
        parts.push(box(`${p.id}_j${s}`, P(s * (spanW / 2 - fw / 2), 0), horiz === "x" ? [thin, h, fw] : [fw, h, thin]));
      }
      for (let i = 1; i < cols; i++) {                                            // vertical glazing bars
        const u = -spanW / 2 + (i * spanW) / cols;
        parts.push(box(`${p.id}_m${i}`, P(u, 0), horiz === "x" ? [thin * 0.7, h - fw * 2, 0.05] : [0.05, h - fw * 2, thin * 0.7]));
      }
      for (let i = 1; i < 4; i++) {                                               // horizontal bars
        const v = -h / 2 + (i * h) / 4;
        parts.push(box(`${p.id}_t${i}`, P(0, v), S(spanW - fw * 2, 0.045, 0)));
      }
      return parts;
    }
    default:
      return [p];
  }
}

/** Sensible reflectance per archetype so a space.json need not specify one. Roughly real: limestone
 *  and paper are bright, oak and old leather are dark, and the spread between them is most of what
 *  makes a clay render look like a photograph of objects rather than a massing study. */
const ALBEDO: Partial<Record<Prop["type"], number>> = {
  column: 0.56, window: 0.50, beams: 0.26, shelf: 0.30, stack: 0.44,
  clutter: 0.60, chair: 0.24, cot: 0.42, table: 0.28, ladder: 0.32,
  lamp: 0.72,                                             // lit glass, brightest object in frame
};

/** Expand archetypes, stamping every emitted part with the id/label/desc of the prop it came from
 *  so downstream visibility reporting speaks in authored objects rather than generated parts. */
export const expandProps = (props: Prop[]): Prop[] =>
  props.flatMap((p) => expandProp(p).map((q) => ({
    ...q,
    owner: p.id,
    label: q.label ?? p.label,
    desc: q.desc ?? p.desc ?? p.label,
    albedo: q.albedo ?? p.albedo ?? ALBEDO[p.type] ?? 0.34,
  })));

/** Clip a camera-space polygon against the near plane (Sutherland-Hodgman, one plane). */
function clipPoly(poly: Vec3[]): Vec3[] {
  const out: Vec3[] = [];
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    const inA = a[2] >= NEAR, inB = b[2] >= NEAR;
    if (inA) out.push(a);
    if (inA !== inB) {
      const t = (NEAR - a[2]) / (b[2] - a[2]);
      out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, NEAR]);
    }
  }
  return out;
}

/**
 * Paint the solid: cull backfaces, shade by the room's practical light, fog toward white with
 * distance, sort back-to-front and fill. Depth cueing matters more than accuracy here — it is the
 * cue that survives being read by an image model.
 */
/** Practical light colour and the ambient it falls off into. Tuned so the clay render's own average
 *  matches a lamplit night interior — whatever bleeds through img2img is then already the right
 *  colour and exposure, instead of neutral grey laid over a dark warm photograph as milky fog. */
const KEY_RGB: [number, number, number] = [1.0, 0.78, 0.50];   // tungsten / green-glass lamp
// Ambient stands in for bounce. Near-zero made every surface turned away from the lamp render as
// flat black, so a kettle in front of the lamp became a void with no form — which is not what a
// room looks like, and gives img2img nothing to work from in half the frame. It stays COLD and very
// dim: warm ambient is what reads as grey fog, blue ambient reads as night.
const AMB_RGB: [number, number, number] = [0.022, 0.027, 0.040];
const KEY_GAIN = 0.85;                                           // exposure
const KEY_R = 1.7;                                               // metres to half-brightness

export interface Light { pos: Vec3; r: number; gain: number }
interface Shadower { c: Vec3; s: Vec3; owner: string }

/**
 * Split a large quad into a grid so shading can VARY across it.
 *
 * Flat-shading one face per surface means the floor of a 9x14 hall gets a single grey, chosen at
 * its centroid — an even slab under a lamp that in reality falls off to nothing within a few
 * metres. That flatness is a large part of what read as "milky" and "foggy": the init image had no
 * luminance gradient anywhere, so the photograph inherited none. Subdividing costs nothing on the
 * small stuff (a book spine is already under the cell size and passes straight through) and buys
 * a real falloff gradient and, with it, usable cast shadows.
 */
function subdivide(f: Face, cell: number): Face[] {
  if (f.pts.length !== 4) return [f];
  const [a, b2, c, d] = f.pts;
  const nu = Math.ceil(Math.max(Math.hypot(...sub(b2, a)), Math.hypot(...sub(c, d))) / cell);
  const nv = Math.ceil(Math.max(Math.hypot(...sub(d, a)), Math.hypot(...sub(c, b2))) / cell);
  if (nu <= 1 && nv <= 1) return [f];
  const P = (u: number, v: number): Vec3 => {
    const top = add(mul(a, 1 - u), mul(b2, u));
    const bot = add(mul(d, 1 - u), mul(c, u));
    return add(mul(top, 1 - v), mul(bot, v));
  };
  const out: Face[] = [];
  for (let i = 0; i < nu; i++) {
    for (let j = 0; j < nv; j++) {
      out.push({
        pts: [P(i / nu, j / nv), P((i + 1) / nu, j / nv), P((i + 1) / nu, (j + 1) / nv), P(i / nu, (j + 1) / nv)],
        n: f.n, base: f.base, owner: f.owner,
      });
    }
  }
  return out;
}

/** Fraction of a light visible from a point: 0 fully shadowed, 1 clear. Samples the source at
 *  three points so the edge is a penumbra rather than a stencil. */
function visibility(from: Vec3, l: Light, blockers: Shadower[], own?: string): number {
  const to = sub(l.pos, from);
  const side = norm(cross(to, [0, 1, 0]));
  const up: Vec3 = norm(cross(side, to));
  const samples: Vec3[] = [l.pos, add(l.pos, mul(side, l.r)), add(l.pos, mul(up, l.r))];
  let vis = 0;
  for (const s of samples) {
    const d = sub(s, from);
    let blocked = false;
    for (const b of blockers) {
      if (b.owner === own) continue;                                  // never shadow yourself
      let t0 = 0.02, t1 = 0.98;
      let hit = true;
      for (let k = 0; k < 3; k++) {
        const o = from[k], dir = d[k];
        const lo = b.c[k] - b.s[k] / 2, hi = b.c[k] + b.s[k] / 2;
        if (Math.abs(dir) < 1e-9) { if (o < lo || o > hi) { hit = false; break; } continue; }
        let x = (lo - o) / dir, y = (hi - o) / dir;
        if (x > y) [x, y] = [y, x];
        t0 = Math.max(t0, x); t1 = Math.min(t1, y);
        if (t0 > t1) { hit = false; break; }
      }
      if (hit) { blocked = true; break; }
    }
    if (!blocked) vis += 1 / samples.length;
  }
  return vis;
}

function emitFaces(
  sink: Sink, faces: Face[], b: Basis, W: number, H: number,
  lights: Light[], clay = false, blockers: Shadower[] = [],
) {
  const src = clay ? faces.flatMap((f) => subdivide(f, 0.30)) : faces;
  const drawable: { pts: Vec2[]; grey: Col; depth: number }[] = [];
  for (const f of src) {
    const c = centroid(f.pts);
    if (dot(f.n, sub(b.pos, c)) <= 0) continue;                       // facing away
    const cam = f.pts.map((p) => toCam(b, p));
    const clipped = clipPoly(cam);
    if (clipped.length < 3) continue;
    const depth = clipped.reduce((a, p) => a + p[2], 0) / clipped.length;

    let grey: Col;
    if (clay) {
      // CLAY RENDER — the init image for img2img, so its value structure becomes the photograph's.
      // Practicals with falloff and near-black ambient, because that IS the scene's lighting. The
      // diagram shading below (flat ambient, fog toward white) told the model "bright, flat, low
      // contrast" and img2img dutifully preserved it, producing a fogged frame.
      if (f.emit !== undefined) {
        // A lit fixture is not lit BY anything — its own source sits inside it, so the light vector
        // is degenerate and the shading model returns black. Emissive surfaces carry their value.
        drawable.push({
          pts: clipped.map((q) => project(q, b, W, H)),
          grey: [0, 1, 2].map((i) => Math.pow(Math.min(1, f.emit! * KEY_RGB[i]), 1 / 2.2)) as [number, number, number],
          depth,
        });
        continue;
      }
      let acc = 0;
      for (const l of lights) {
        const toKey = sub(l.pos, c);
        const lambert = Math.max(0, dot(f.n, norm(toKey)));
        if (lambert <= 0) continue;                                   // dark anyway; skip the rays
        // Softened inverse-square: 1/(1+(r/R)^2) rather than 1/r^2. True inverse-square blows every
        // surface within a metre of the practical to clipped white and flattens everything past it
        // to one grey — a lamp in a real room reads over a much longer range than the physics
        // suggests because the eye and the film both compress it.
        const r = Math.hypot(toKey[0], toKey[1], toKey[2]);
        const falloff = 1 / (1 + (r / KEY_R) * (r / KEY_R));
        const shade = blockers.length ? visibility(c, l, blockers, f.owner) : 1;
        if (shade <= 0) continue;
        acc += l.gain * lambert * falloff * shade;
      }
      // Hemispheric ambient: up-facing surfaces catch more bounce than down-facing ones. Flat
      // ambient leaves every unlit plane at one identical value, which is the signature of a
      // massing model rather than a photograph.
      const sky = 0.55 + 0.45 * f.n[1];
      grey = [0, 1, 2].map((i) =>
        Math.pow(Math.min(1, f.base * (AMB_RGB[i] * sky + acc * KEY_RGB[i])), 1 / 2.2),
      ) as [number, number, number];
    } else {
      const lambert = lights.length
        ? Math.max(0, dot(f.n, norm(sub(lights[0].pos, c))))
        : 0.5;
      const fog = Math.min(1, depth / 9);
      grey = Math.min(0.97, (f.base * (0.55 + 0.55 * lambert)) * (1 - fog) + 0.94 * fog);
    }
    drawable.push({ pts: clipped.map((p) => project(p, b, W, H)), grey, depth });
  }
  drawable.sort((x, y) => y.depth - x.depth);
  for (const d of drawable) sink.fill(d.pts, d.grey);
}

/** Every practical in the room. Shading from one hardcoded lamp means adding a second light to a
 *  space.json changes the prompt and the wireframe but not the image it is supposed to describe. */
export function collectLights(space: SpaceSpec): Light[] {
  const lamps = space.props.filter((p) => p.type === "lamp" && p.pos);
  if (!lamps.length) return [{ pos: [0, space.room.h - 0.3, space.room.d / 2], r: 0.1, gain: KEY_GAIN }];
  return lamps.map((l) => ({ pos: l.pos as Vec3, r: l.r ?? 0.09, gain: (l.intensity ?? 1) * KEY_GAIN }));
}

/**
 * Coarse occluders for shadow casting: the expanded parts big enough to throw a shadow worth
 * drawing. Testing every emitted solid means several thousand ray-box tests per face against book
 * spines that contribute nothing; testing only the authored props means a scattered debris field
 * casts one enormous rectangular slab, because its authored footprint is a lie about its shape.
 */
function collectShadowers(space: SpaceSpec, limit = 220): Shadower[] {
  // A fixture never occludes its own light. The lamp's shade is centred ON the light position, so
  // including it would put the source inside a blocker and shadow the entire room.
  const fixtures = new Set(space.props.filter((p) => p.type === "lamp").map((p) => p.id));
  return expandProps(space.props)
    .filter((p) => p.pos && p.size && (p.type === "box" || p.type === "cyl") &&
      !fixtures.has(p.owner ?? p.id) &&
      Math.max(p.size[0], p.size[1], p.size[2]) >= 0.25)
    .map((p) => {
      const [sw, sh, sd] = p.size as Vec3;
      const yr = ((p.yaw ?? 0) * Math.PI) / 180, cs = Math.abs(Math.cos(yr)), sn = Math.abs(Math.sin(yr));
      // A yawed box's axis-aligned bound is wider than the box. Using the unrotated extents would
      // let light leak past a turned solid at exactly the angles where its shadow matters most.
      return { c: p.pos as Vec3, s: [sw * cs + sd * sn, sh, sw * sn + sd * cs] as Vec3, owner: p.owner ?? p.id };
    })
    .sort((a, b) => b.s[0] * b.s[1] * b.s[2] - a.s[0] * a.s[1] * a.s[2])
    .slice(0, limit);
}

// ---------------------------------------------------------------- geometry passes

/**
 * Rasterise the solid to ground-truth buffers with a z-buffer.
 *
 * Why this replaces the flat-shaded blockout: depth ControlNets are trained on MiDaS-lineage
 * output, which is DISPARITY (1/z), near=white far=black — not linear distance and definitely not
 * Lambert shading. Lambert encodes surface-to-light angle, which is a lighting cue that merely
 * happens to be grey. Feeding it to a depth model is a category error.
 *
 * The upside of doing this from our own solid: a photo pipeline has to estimate depth off an RGB
 * frame and eats the estimator's error. We emit the real thing. 1/z is also exactly what perspective
 * rasterisation interpolates linearly in screen space, so the disparity buffer is free.
 */
export interface Passes {
  invZ: Float32Array; nx: Float32Array; ny: Float32Array; nz: Float32Array; hit: Uint8Array;
  /** Object-ID pass: index into `owners` per pixel, 0 = room shell / nothing. */
  id: Uint16Array;
  owners: string[];
  /** The camera these buffers were rasterised from. Kept so any pass can recover the WORLD point
   *  behind a pixel exactly (from invZ), which is what lets surface detail be anchored to the room
   *  instead of to the frame. */
  basis: Basis;
}

/** World point behind pixel (x,y), reconstructed exactly from the depth buffer. */
export function worldAt(p: Passes, x: number, y: number, W: number, H: number): Vec3 {
  const b = p.basis;
  const f = W / 2 / Math.tan((b.fov * Math.PI) / 360);
  const z = 1 / p.invZ[y * W + x];
  const cx = ((x + 0.5 - W / 2) * z) / f, cy = (-(y + 0.5 - H / 2) * z) / f;
  return add(b.pos, add(mul(b.right, cx), add(mul(b.up, cy), mul(b.fwd, z))));
}

function rasterise(faces: Face[], b: Basis, W: number, H: number): Passes {
  const invZ = new Float32Array(W * H);
  const nx = new Float32Array(W * H), ny = new Float32Array(W * H), nz = new Float32Array(W * H);
  const hit = new Uint8Array(W * H);
  const id = new Uint16Array(W * H);
  const owners: string[] = [""];
  const ownerIdx = new Map<string, number>();
  const f = W / 2 / Math.tan((b.fov * Math.PI) / 360);

  for (const face of faces) {
    let oi = 0;
    if (face.owner) {
      oi = ownerIdx.get(face.owner) ?? 0;
      if (!oi) { owners.push(face.owner); oi = owners.length - 1; ownerIdx.set(face.owner, oi); }
    }
    const c = centroid(face.pts);
    if (dot(face.n, sub(b.pos, c)) <= 0) continue;
    const cam = clipPoly(face.pts.map((p) => toCam(b, p)));
    if (cam.length < 3) continue;
    // face normal in camera space, for the normal pass
    const ncam: Vec3 = [dot(face.n, b.right), dot(face.n, b.up), dot(face.n, b.fwd)];
    const scr = cam.map((v): [number, number, number] => [W / 2 + (v[0] * f) / v[2], H / 2 - (v[1] * f) / v[2], 1 / v[2]]);

    for (let t = 1; t + 1 < scr.length; t++) {                    // fan triangulation
      const [p0, p1, p2] = [scr[0], scr[t], scr[t + 1]];
      const minX = Math.max(0, Math.floor(Math.min(p0[0], p1[0], p2[0])));
      const maxX = Math.min(W - 1, Math.ceil(Math.max(p0[0], p1[0], p2[0])));
      const minY = Math.max(0, Math.floor(Math.min(p0[1], p1[1], p2[1])));
      const maxY = Math.min(H - 1, Math.ceil(Math.max(p0[1], p1[1], p2[1])));
      const area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]);
      if (Math.abs(area) < 1e-9) continue;

      for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
          const px = x + 0.5, py = y + 0.5;
          let w0 = ((p1[0] - px) * (p2[1] - py) - (p2[0] - px) * (p1[1] - py)) / area;
          let w1 = ((p2[0] - px) * (p0[1] - py) - (p0[0] - px) * (p2[1] - py)) / area;
          let w2 = 1 - w0 - w1;
          if (w0 < 0 || w1 < 0 || w2 < 0) continue;
          const z = w0 * p0[2] + w1 * p1[2] + w2 * p2[2];          // 1/z interpolates linearly
          const i = y * W + x;
          if (z > invZ[i]) {
            invZ[i] = z; hit[i] = 1; id[i] = oi;
            nx[i] = ncam[0]; ny[i] = ncam[1]; nz[i] = ncam[2];
          }
        }
      }
    }
  }
  return { invZ, nx, ny, nz, hit, id, owners, basis: b };
}

function bufferToPng(px: Uint8ClampedArray, W: number, H: number): Buffer {
  const cv = createCanvas(W, H);
  const ctx = cv.getContext("2d");
  const img = ctx.createImageData(W, H);
  img.data.set(px);
  ctx.putImageData(img, 0, 0);
  return cv.toBuffer("image/png");
}

/** Log-depth, normalised over a FIXED metric range. Near = white, far = black. */
export function depthPng(p: Passes, W: number, H: number, range?: { near: number; far: number }): Buffer {
  // LOG-DEPTH, not raw disparity.
  //
  // Disparity (1/z) is hyperbolic, so in a deep interior it allocates almost the whole 0–255 range
  // to the nearest metre. A 14 m hall shot from 1.5 m puts everything past ~6 m into the bottom
  // tenth of the range, which clips to black and deletes the columns, the shelf runs and the far
  // doorway — precisely the architecture that makes the space legible. Normalising in log(z)
  // instead spreads depth perceptually evenly, which is much closer to what DepthAnything actually
  // produces on a photograph of a room like this.
  //
  // FIXED RANGE, not per-frame min-max. This is the difference between a depth map and a depth map
  // you can cut between. Normalising to each frame's own near/far makes grey mean something
  // different in every shot: the chair at 3 m is grey 190 in a shot that happens to see the far
  // wall and grey 120 in one that does not, so two angles of one room hand the ControlNet two
  // incompatible spatial scales and it has no way to know they are the same place. Anchoring the
  // range to the ROOM instead — near plane to room diagonal — makes grey a fixed function of world
  // distance for every camera in that location. The cost is contrast on very tight shots, which is
  // the right trade: a close-up can afford a flatter map, a sequence cannot afford a moving one.
  const near = Math.max(0.05, range?.near ?? 0.30);
  const far = Math.max(near * 1.05, range?.far ?? 20);
  const lnN = Math.log(near), span = Math.log(far) - lnN || 1;

  const grey = new Float32Array(W * H);
  for (let i = 0; i < p.invZ.length; i++) {
    if (p.hit[i] && p.invZ[i] > 1e-6) {
      const t = (Math.log(1 / p.invZ[i]) - lnN) / span;          // 0 near .. 1 far
      grey[i] = (1 - Math.max(0, Math.min(1, t))) * 255;         // near = white
    }
  }

  // SOFTEN. A z-buffer render has razor silhouettes and dead-flat plateaus; a depth map estimated
  // from a photograph does not. Handed a hard-edged CAD render, the ControlNet reproduces the boxes
  // literally instead of treating them as massing — the geometry stops guiding and starts dictating.
  // A small blur moves the map toward the estimator-like distribution these models were trained on.
  // Radius ~2px at 1536 wide: enough to take the CAD edge off a silhouette, nowhere near enough to
  // erase book spines or scattered debris. An earlier radius of 7 destroyed exactly the fine
  // structure that makes the map worth rendering.
  const soft = blur(grey, W, H, Math.max(1, Math.round(W / 760)));

  // WorldGen's fix for exactly this pipeline shape (arXiv 2511.16825): a small perturbation scaled
  // proportionally to depth, applied specifically to discourage "overly rectilinear shapes".
  //
  // Sampled in WORLD SPACE, not screen space. The old version indexed a fixed lattice by pixel
  // coordinate, so the undulation was welded to the frame: every shot of every location got the
  // identical pattern, and a given patch of wall got a different wobble from every angle. That is
  // the exact opposite of the intent — it added a view-dependent signal to a map whose whole job is
  // to be view-independent. Sampling a 3D lattice at the world point behind each pixel means one
  // patch of wall carries the same undulation from every camera, so the perturbation survives a cut.
  //
  // Amplitude is constant in grey rather than scaled by brightness. Under log normalisation the
  // grey-per-log-metre slope is constant, so a flat grey amplitude IS a perturbation proportional
  // to depth — which is what the paper prescribes. The old brightness scaling made it proportional
  // to depth twice over and left far surfaces untouched.
  const CELL = 0.55;                                                // metres per lattice cell
  const AMP = 2.0;                                                  // grey levels, ~3% of depth
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (!p.hit[i] || p.invZ[i] <= 1e-6) continue;
      const wp = worldAt(p, x, y, W, H);
      soft[i] += vnoise3(wp[0] / CELL, wp[1] / CELL, wp[2] / CELL) * AMP;
    }
  }

  // Never map to pure black. The farthest surfaces are the ceiling and the far wall, and clipping
  // them to 0 throws away the end of the room — a real estimated depth map keeps them dim, not gone.
  const FLOOR = 20;
  const out = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < soft.length; i++) {
    const v = Math.round(FLOOR + (Math.max(0, Math.min(255, soft[i])) / 255) * (255 - FLOOR));
    out[i * 4] = out[i * 4 + 1] = out[i * 4 + 2] = v;
    out[i * 4 + 3] = 255;
  }
  return bufferToPng(out, W, H);
}

/** Value noise on a 3D world lattice, trilinear with a smoothstep fade. Stateless and seeded only
 *  by position, so the same world point yields the same value in every shot, in every process. */
function vnoise3(x: number, y: number, z: number): number {
  const lat = (i: number, j: number, k: number) => {
    let h = (Math.imul(i, 374761393) + Math.imul(j, 668265263) + Math.imul(k, 1442695041)) | 0;
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    return (((h ^ (h >>> 16)) >>> 0) / 2147483648) - 1;
  };
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const s = (t: number) => t * t * (3 - 2 * t);
  const fx = s(x - xi), fy = s(y - yi), fz = s(z - zi);
  const L = (a: number, b: number, t: number) => a + (b - a) * t;
  const c00 = L(lat(xi, yi, zi), lat(xi + 1, yi, zi), fx);
  const c10 = L(lat(xi, yi + 1, zi), lat(xi + 1, yi + 1, zi), fx);
  const c01 = L(lat(xi, yi, zi + 1), lat(xi + 1, yi, zi + 1), fx);
  const c11 = L(lat(xi, yi + 1, zi + 1), lat(xi + 1, yi + 1, zi + 1), fx);
  return L(L(c00, c10, fy), L(c01, c11, fy), fz);
}

/** Separable box blur, run three times — a cheap, close approximation of a gaussian. */
function blur(src: Float32Array, W: number, H: number, r: number): Float32Array {
  let cur = src, tmp = new Float32Array(W * H);
  for (let pass = 0; pass < 3; pass++) {
    for (let y = 0; y < H; y++) {                                 // horizontal
      let acc = 0;
      for (let x = -r; x <= r; x++) acc += cur[y * W + Math.max(0, Math.min(W - 1, x))];
      for (let x = 0; x < W; x++) {
        tmp[y * W + x] = acc / (2 * r + 1);
        acc -= cur[y * W + Math.max(0, x - r)];
        acc += cur[y * W + Math.min(W - 1, x + r + 1)];
      }
    }
    const swap = cur === src ? new Float32Array(W * H) : cur;
    for (let x = 0; x < W; x++) {                                 // vertical
      let acc = 0;
      for (let y = -r; y <= r; y++) acc += tmp[Math.max(0, Math.min(H - 1, y)) * W + x];
      for (let y = 0; y < H; y++) {
        swap[y * W + x] = acc / (2 * r + 1);
        acc -= tmp[Math.max(0, y - r) * W + x];
        acc += tmp[Math.min(H - 1, y + r + 1) * W + x];
      }
    }
    cur = swap;
  }
  return cur;
}

/** Camera-space normals, the usual (n+1)/2 encoding. */
export function normalPng(p: Passes, W: number, H: number): Buffer {
  const out = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < p.hit.length; i++) {
    if (p.hit[i]) {
      out[i * 4] = Math.round((p.nx[i] * 0.5 + 0.5) * 255);
      out[i * 4 + 1] = Math.round((p.ny[i] * 0.5 + 0.5) * 255);
      out[i * 4 + 2] = Math.round((p.nz[i] * 0.5 + 0.5) * 255);
    } else { out[i * 4] = out[i * 4 + 1] = 128; out[i * 4 + 2] = 255; }
    out[i * 4 + 3] = 255;
  }
  return bufferToPng(out, W, H);
}

/**
 * Edge pass from depth + normal discontinuity. This is what a Canny preprocessor tries to recover
 * from an RGB frame; we can just read it off the geometry, with no texture noise and no misses.
 * White lines on black, the convention every canny/lineart ControlNet expects.
 */
export function edgePng(p: Passes, W: number, H: number): Buffer {
  const out = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H * 4; i += 4) out[i + 3] = 255;
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      let edge = 0;
      for (const j of [i - 1, i + 1, i - W, i + W]) {
        if (p.hit[i] !== p.hit[j]) { edge = 1; break; }
        if (!p.hit[i]) continue;
        const dz = Math.abs(p.invZ[i] - p.invZ[j]) / (Math.max(p.invZ[i], 1e-6));
        const dn = 1 - (p.nx[i] * p.nx[j] + p.ny[i] * p.ny[j] + p.nz[i] * p.nz[j]);
        if (dz > 0.02 || dn > 0.05) { edge = 1; break; }
      }
      if (edge) { const o = i * 4; out[o] = out[o + 1] = out[o + 2] = 255; }
    }
  }
  return bufferToPng(out, W, H);
}

// ---------------------------------------------------------------- shot solver

/** Subject height as a fraction of frame height, by shot size. The grammar a director actually uses. */
export const SHOT_SIZE: Record<string, number> = {
  ELS: 0.10, LS: 0.28, MLS: 0.42, MS: 0.58, MCU: 0.74, CU: 0.90, ECU: 1.30,
};
/** Camera height in metres for an eye-level-relative angle. */
export const ANGLE_HEIGHT: Record<string, number> = {
  floor: 0.18, low: 0.62, eye: 1.55, high: 2.05, overhead: 2.42,
};

export interface Framing {
  /** prop id in space.json, a figure id in this shot, or [x,y,z] world point. */
  subject: string | Vec3;
  size?: keyof typeof SHOT_SIZE | string;
  angle?: keyof typeof ANGLE_HEIGHT | string;
  /** focal length, 35mm-equivalent. 18=very wide, 35=natural, 50=normal, 85=long. */
  lens?: number;
  /** degrees around the subject; 0 = from the room's front wall, 90 = from its right. */
  azimuth?: number;
  /** where the subject sits in frame, 0..1. [0.33,0.5] = left third. */
  place?: [number, number];
  roll?: number;
}

function subjectSolid(f: Framing, space: SpaceSpec, figures: FigureSpec[]): { c: Vec3; h: number } {
  if (Array.isArray(f.subject)) return { c: f.subject, h: 1.0 };
  const fig = figures.find((x) => x.id === f.subject);
  if (fig) {
    const h = fig.height ?? 1.7;
    return { c: [fig.at[0], h * 0.55, fig.at[1]], h };
  }
  const prop = space.props.find((p) => p.id === f.subject);
  if (!prop) throw new Error(`shot subject "${f.subject}" is not a prop or figure`);
  // Any prop carrying pos+size is a solid — box AND every archetype (chair/shelf/table/cot/column).
  // Matching only on type "box" silently dropped archetypes through to the panel branch, which
  // returns a point on the back wall, so every shot of a chair was framing bare plaster.
  if (prop.pos && prop.size) return { c: prop.pos, h: Math.max(prop.size[1], prop.size[2] * 0.6) };
  if (prop.type === "lamp" && prop.pos) return { c: prop.pos, h: 0.45 };
  const face = panelFace(prop, space.room);
  if (face) return { c: centroid(face.pts), h: prop.h ?? 1.0 };
  throw new Error(`cannot solve subject "${f.subject}"`);
}

const lensToFovH = (mm: number) => (2 * Math.atan(36 / (2 * mm)) * 180) / Math.PI;

/**
 * Turn shot language into a camera. Distance comes from the requested subject size, height from the
 * angle, direction from the azimuth; then the aim is corrected so the subject lands where asked.
 * The correction loop is what makes `place` trustworthy rather than approximate.
 */
export function solveCamera(f: Framing, space: SpaceSpec, figures: FigureSpec[], W: number, H: number): CameraSpec {
  const { c: subj, h: subjH } = subjectSolid(f, space, figures);
  const frac = SHOT_SIZE[f.size ?? "MS"] ?? SHOT_SIZE.MS;
  const az = ((f.azimuth ?? 0) * Math.PI) / 180;
  const pad = 0.14;
  const camY = Math.max(pad, Math.min(space.room.h - pad, ANGLE_HEIGHT[f.angle ?? "eye"] ?? ANGLE_HEIGHT.eye));
  const dy = camY - subj[1];

  // How far back can we actually get? Interiors are small; this is the constraint that decides
  // the shot, exactly as it does on a real set.
  const lim = (o: number, dir: number, lo: number, hi: number) =>
    Math.abs(dir) < 1e-6 ? Infinity : dir > 0 ? (hi - o) / dir : (lo - o) / dir;
  const reach = (hx: number, hz: number) => Math.min(
    lim(subj[0], hx, -space.room.w / 2 + pad, space.room.w / 2 - pad),
    lim(subj[2], hz, pad, space.room.d - pad),
  );
  let hx = Math.sin(az), hz = -Math.cos(az);
  // A subject flat against a wall has no room on its far side. Asking for that azimuth would put the
  // camera through the plaster, so take the opposite side rather than silently leaving the room.
  if (reach(hx, hz) < 0.30 && reach(-hx, -hz) > reach(hx, hz)) { hx = -hx; hz = -hz; }
  let flatMax = Math.max(0.30, reach(hx, hz));

  // OCCLUSION. Distance and angle alone are not a camera position: a free-standing bookstack between
  // lens and subject fills the frame with the back of a shelf, and the framing check still passes
  // because the subject projects exactly where it was ordered — it is simply behind something.
  // Walk the azimuth outward until the sight-line to the subject is clear.
  // Cylinders block sight-lines and bodies exactly as boxes do; testing only `type === "box"` let
  // the camera stand inside a column shaft, because the shaft is a cyl. Treat both as AABBs.
  const blockers = expandProps(space.props)
    .filter((p) => (p.type === "box" || p.type === "cyl") && p.pos && p.size && !p.id.startsWith(String(f.subject)))
    .map((p) => ({ c: p.pos as Vec3, s: p.size as Vec3 }));
  /** Is the lens itself buried in something? A clear sight-line to the subject says nothing about
   *  whether the camera is standing inside a column — which renders as one surface 10cm away. */
  const CLEARANCE = 0.22;
  const inside = (px: number, py: number, pz: number) =>
    blockers.some((b) =>
      px > b.c[0] - b.s[0] / 2 - CLEARANCE && px < b.c[0] + b.s[0] / 2 + CLEARANCE &&
      py > b.c[1] - b.s[1] / 2 - CLEARANCE && py < b.c[1] + b.s[1] / 2 + CLEARANCE &&
      pz > b.c[2] - b.s[2] / 2 - CLEARANCE && pz < b.c[2] + b.s[2] / 2 + CLEARANCE);

  const clear = (px: number, py: number, pz: number) => {
    if (inside(px, py, pz)) return false;
    const d: Vec3 = [subj[0] - px, subj[1] - py, subj[2] - pz];
    for (const b of blockers) {
      let t0 = 0.04, t1 = 0.96;                       // ignore contact at either endpoint
      let hit = true;
      for (let k = 0; k < 3; k++) {
        const o = [px, py, pz][k], dir = d[k];
        const lo = b.c[k] - b.s[k] / 2, hi = b.c[k] + b.s[k] / 2;
        if (Math.abs(dir) < 1e-9) { if (o < lo || o > hi) { hit = false; break; } continue; }
        let a = (lo - o) / dir, z = (hi - o) / dir;
        if (a > z) [a, z] = [z, a];
        t0 = Math.max(t0, a); t1 = Math.min(t1, z);
        if (t0 > t1) { hit = false; break; }
      }
      if (hit) return false;
    }
    return true;
  };

  // Distance the requested lens would need, then the distance we can have.
  let fovH = lensToFovH(f.lens ?? 35);
  let fovV = (2 * Math.atan(Math.tan((fovH * Math.PI) / 360) * (H / W)) * 180) / Math.PI;
  const need = subjH / (2 * Math.tan((fovV * Math.PI) / 360) * frac);

  // Now resolve occlusion AT THE DISTANCE THE CAMERA WILL ACTUALLY SIT. Testing the sight-line at
  // the reachable distance while placing the lens at the needed distance checks a point the camera
  // never occupies, which is how every shot ended up in the same corner at 131 degrees.
  // Preference order: hold the ordered azimuth and give up stand-off first, since losing a metre of
  // distance costs less than losing the angle the shot was designed around.
  const wantFlat = (r: number) => {
    const d0 = Math.min(need, Math.hypot(r, dy));
    return Math.sqrt(Math.max(0.09, d0 * d0 - dy * dy));
  };
  outer: for (const dAz of [0, 8, -8, 16, -16, 26, -26, 38, -38, 52, -52, 70, -70, 92, -92, 120, -120, 150, -150, 180]) {
    const a2 = az + (dAz * Math.PI) / 180;
    const tx2 = Math.sin(a2), tz2 = -Math.cos(a2);
    const r = reach(tx2, tz2);
    if (r < 0.30) continue;
    const base = wantFlat(r);
    for (const scale of [1.0, 0.82, 0.66, 0.52, 0.4]) {
      // Clamp rather than reject. On any wide shot the needed distance equals the reachable
      // distance exactly, so a strict `fl > r` test is decided by floating-point noise — and losing
      // that coin flip discarded the ordered azimuth entirely and rotated the camera into an aisle.
      const fl = Math.min(r, Math.max(0.30, base * scale));
      if (clear(subj[0] + tx2 * fl, camY, subj[2] + tz2 * fl)) { hx = tx2; hz = tz2; flatMax = fl; break outer; }
    }
  }

  const haveMax = Math.hypot(flatMax, dy);
  const dist = Math.min(need, haveMax);

  // If the room won't give us the throw, widen until the subject still reads at the ordered size.
  if (dist < need - 1e-4) {
    const tanV = subjH / (2 * dist * frac);
    fovV = (2 * Math.atan(tanV) * 180) / Math.PI;
    fovH = (2 * Math.atan(Math.tan((fovV * Math.PI) / 360) * (W / H)) * 180) / Math.PI;
  }

  const flat = Math.sqrt(Math.max(0.04, dist * dist - dy * dy));
  const pos: Vec3 = [subj[0] + hx * flat, camY, subj[2] + hz * flat];

  // Aim by ANGLE, not by sliding the look point. Rotating the forward axis toward +right moves the
  // image left, so the correction is the current angle minus the ordered one.
  const [tx, ty] = f.place ?? [0.5, 0.5];
  const fpx = W / 2 / Math.tan((fovH * Math.PI) / 360);
  const wantAx = Math.atan(((tx - 0.5) * W) / fpx);
  const wantAy = Math.atan((-(ty - 0.5) * H) / fpx);
  const dist3 = Math.hypot(...sub(subj, pos));
  let fwd = norm(sub(subj, pos));
  for (let i = 0; i < 6; i++) {
    const b = cameraBasis({ pos, look: add(pos, mul(fwd, dist3)), fov: fovH, roll: f.roll });
    const cs = toCam(b, subj);
    if (cs[2] < NEAR) break;
    const curAx = Math.atan(cs[0] / cs[2]);
    const curAy = Math.atan(cs[1] / cs[2]);
    const ex = curAx - wantAx, ey = curAy - wantAy;
    if (Math.abs(ex) < 1e-4 && Math.abs(ey) < 1e-4) break;
    fwd = norm(add(fwd, add(mul(b.right, Math.tan(ex)), mul(b.up, Math.tan(ey)))));
  }
  return { pos, look: add(pos, mul(fwd, dist3)), fov: fovH, roll: f.roll };
}

// ---------------------------------------------------------------- render

function emit(sink: Sink, segs: Seg[], b: Basis, w: number, h: number, s: Stroke) {
  for (const [p, q] of segs) {
    const clipped = nearClip(toCam(b, p), toCam(b, q));
    if (!clipped) continue;
    sink.poly([project(clipped[0], b, w, h), project(clipped[1], b, w, h)], s);
  }
}

/** Billboard a sphere as a screen circle sized by its depth. */
function emitSphere(sink: Sink, c: Vec3, r: number, b: Basis, w: number, h: number, s: Stroke) {
  const cc = toCam(b, c);
  if (cc[2] < NEAR) return;
  const f = w / 2 / Math.tan((b.fov * Math.PI) / 360);
  sink.circle(project(cc, b, w, h), (r * f) / cc[2], s);
}

export type BoardMode = "wire" | "blockout" | "clay";

/** Every solid in the shot: room shell, props, and a coarse box per figure. */
export function collectFaces(shot: ShotSpec, space: SpaceSpec): Face[] {
  const faces: Face[] = [...roomFaces(space.room, space.openToSky === true)];
  const lamps = new Set(space.props.filter((q) => q.type === "lamp").map((q) => q.id));
  // expandProps fills a default albedo for every prop from the ALBEDO table, so by the time a
  // prop reaches this loop there is no way to tell an authored 0.34 from a defaulted one. A model
  // carries its OWN per-triangle reflectance from the glTF materials, and that is better
  // information than one number for the whole object — so the override must come from what the
  // author actually wrote, which is only visible here, before expansion.
  const authoredAlbedo = new Map(space.props.map((q) => [q.id, q.albedo]));
  for (const p of expandProps(space.props)) {
    // Only the shade glows. A hanging lamp's cord and a table lamp's stem and foot are ordinary
    // objects that happen to belong to a light fitting.
    const glow = lamps.has(p.owner ?? "") && p.id.endsWith("_shade") ? 0.78 : undefined;
    const own = (fs: Face[]) => { for (const f of fs) { f.owner = p.owner ?? p.id; f.emit = glow; } return fs; };
    const alb = p.albedo ?? 0.34;
    if (p.type === "box" && p.pos && p.size) faces.push(...own(boxFaces(p.pos, p.size, p.yaw ?? 0, alb)));
    else if (p.type === "cyl" && p.pos && p.size) {
      const ax = p.axis ?? "y";
      const len = ax === "y" ? p.size[1] : ax === "x" ? p.size[0] : p.size[2];
      faces.push(...own(cylinderFaces(p.pos, p.radius ?? p.size[0] / 2, len, ax, 20, alb, p.yaw ?? 0)));
    }
    else if (p.type === "panel") { const f = panelFace(p, space.room); if (f) faces.push(...own([f])); }
    else if (p.type === "model" && p.asset && p.pos && p.size) {
      const { tris, doubleSided } = assetTriangles({
        asset: p.asset, pos: p.pos, size: p.size, yaw: p.yaw ?? 0, fit: p.fit,
        albedo: authoredAlbedo.get(p.owner ?? p.id) ?? undefined,
      });
      const mf: Face[] = [];
      for (const t of tris) {
        const n = faceNormal(t.pts);
        mf.push({ pts: t.pts, n, base: t.albedo });
        // An open mesh emits BOTH windings. The rasteriser culls on dot(n, toCamera), so exactly
        // one of the pair survives at any camera angle — same pixels, same depth, no hole. A
        // watertight mesh skips this and costs half as many faces.
        if (doubleSided) mf.push({ pts: t.pts, n: mul(n, -1), base: t.albedo });
      }
      faces.push(...own(mf));
    }
  }
  for (const f of shot.figures ?? []) {
    const hgt = f.height ?? 1.7;
    faces.push(...boxFaces([f.at[0], hgt * 0.5, f.at[1]], [0.46, hgt, 0.3], f.yaw ?? 180, 0.16));
  }
  return faces;
}

/** The face budget a location's model props cost, WITHOUT loading a camera or rasterising.
 *
 *  `rasterise` is single-threaded JavaScript that loops a bounding box per face. Archetype
 *  geometry runs about 18k faces for a dense interior and that is comfortable. Model props are
 *  the first thing in this pipeline that can silently multiply it — an open mesh emits both
 *  windings, so a 4,000-triangle tree is 8,000 faces, and twelve of them is a different order of
 *  cost than the whole library-interior. Call this in pre-flight and refuse the shot rather than
 *  discovering it as a slow render.
 *
 *  Returns null when the library is absent, so a space.json with no model props never makes the
 *  library a hard dependency. */
export function modelFaceCost(space: SpaceSpec): { faces: number; props: number } | null {
  const models = space.props.filter((p) => p.type === "model" && p.asset && p.pos && p.size);
  if (!models.length) return { faces: 0, props: 0 };
  try {
    let faces = 0;
    for (const p of models) {
      faces += assetFaceCost({ asset: p.asset!, pos: p.pos!, size: p.size!, fit: p.fit });
    }
    return { faces, props: models.length };
  } catch {
    return null;
  }
}

/** What a given camera can actually SEE, measured off the object-ID pass rather than assumed. */
export interface VisibleProp {
  id: string;
  desc: string;
  /** The authored archetype, so the prompt can use the right vocabulary for the kind of thing it
   *  is. "Facing the camera head-on" is correct for a chair and meaningless for a bed. */
  type: Prop["type"];
  /** share of the frame this object occupies, 0..1 */
  cover: number;
  /** screen centroid, 0..1 from left/top */
  sx: number; sy: number;
  /** nearest and median distance from camera, metres */
  near: number; mid: number;
  /** relative bearing in degrees: 0 = we see its front, 180 = we see its back, +ve = its left */
  bearing: number;
  /** Does this object HAVE a front? A chair and a doorway do; a column, a bookshelf run and a
   *  scatter of debris do not. Describing orientation for something with no front is noise that can
   *  only mislead the model. Defaults by type, overridable per prop with `facing`. */
  directional: boolean;
}

/** Types whose orientation is worth saying out loud, when they carry a yaw. */
const FRONTED = new Set<Prop["type"]>(["chair", "cot", "panel"]);

/**
 * Analyse a shot into the objects a camera genuinely sees, with placement and orientation.
 *
 * This is what makes the prompt deterministic. Describing the LOCATION on every shot tells the
 * model a cot exists with no clue where, so it invents one — usually behind the chair. Describing
 * only what this frame contains, where it sits and which face is toward us removes the guesswork
 * that produced hallucinated furniture and reversed tables.
 */
export function analyzeShot(shot: ShotSpec, space: SpaceSpec, minCover = 0.004): VisibleProp[] {
  const W = shot.width ?? 1536, H = shot.height ?? 1024;
  const cam = shotCamera(shot, space, W, H);
  const b = cameraBasis(cam);
  const p = rasterise(collectFaces(shot, space), b, W, H);

  const acc = new Map<number, { n: number; sx: number; sy: number; zs: number[] }>();
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x, oi = p.id[i];
      if (!oi || !p.hit[i]) continue;
      let a = acc.get(oi);
      if (!a) { a = { n: 0, sx: 0, sy: 0, zs: [] }; acc.set(oi, a); }
      a.n++; a.sx += x; a.sy += y;
      if (a.zs.length < 4000) a.zs.push(1 / p.invZ[i]);
    }
  }

  const byId = new Map(space.props.map((q) => [q.id, q]));
  const out: VisibleProp[] = [];
  for (const [oi, a] of acc) {
    const cover = a.n / (W * H);
    if (cover < minCover) continue;                       // a few stray pixels is not "visible"
    const prop = byId.get(p.owners[oi]);
    a.zs.sort((u, v) => u - v);
    // bearing: the prop's own forward vs. the direction from prop to camera
    const yaw = (((prop?.yaw ?? 0) * Math.PI) / 180);
    const fwd = [Math.sin(yaw), Math.cos(yaw)];
    const pp = prop?.pos ?? [0, 0, 0];
    const toCamV = [cam.pos[0] - pp[0], cam.pos[2] - pp[2]];
    const len = Math.hypot(toCamV[0], toCamV[1]) || 1;
    const tc = [toCamV[0] / len, toCamV[1] / len];
    const bearing = (Math.atan2(fwd[0] * tc[1] - fwd[1] * tc[0], fwd[0] * tc[0] + fwd[1] * tc[1]) * 180) / Math.PI;
    out.push({
      id: p.owners[oi],
      desc: prop?.desc ?? prop?.label ?? p.owners[oi],
      type: prop?.type ?? "box",
      cover, sx: a.sx / a.n / W, sy: a.sy / a.n / H,
      near: a.zs[0] ?? 0, mid: a.zs[Math.floor(a.zs.length / 2)] ?? 0,
      bearing,
      directional: prop?.facing ?? (prop?.yaw !== undefined && FRONTED.has(prop.type)),
    });
  }
  return out.sort((x, y) => y.cover - x.cover);
}

/**
 * The metric range the depth map is normalised over, fixed per LOCATION so grey means the same
 * distance in every shot of that room.
 *
 * The obvious choice — 0.3 m to the full room diagonal — is the true bound and it wastes the range.
 * Nothing is ever at 0.3 m, only the diagonal corners reach the far end, and spending log(58) on
 * that envelope left the actual content of a normal shot inside 92 grey levels of the 235
 * available. Pulling in to 0.5 m and 80% of the diagonal clips two regions that were already
 * flat — closer than half a metre, and the last few metres of the far corner, both of which sit at
 * the ends of the curve where a real depth estimator has no precision either — and buys about 20%
 * more gradient everywhere a shot actually lives. Override per location with `depth` in space.json
 * when a room's proportions are unusual.
 */
export const depthRange = (r: SpaceSpec["room"], space?: SpaceSpec) =>
  space?.depth ?? { near: 0.5, far: 0.8 * Math.hypot(r.w, r.d, r.h) };

/** Ground-truth conditioning passes for a shot: log-depth, camera-space normals, geometric edges. */
export function renderPasses(shot: ShotSpec, space: SpaceSpec) {
  const W = shot.width ?? 1536, H = shot.height ?? 1024;
  const b = cameraBasis(shotCamera(shot, space, W, H));
  const p = rasterise(collectFaces(shot, space), b, W, H);
  const range = depthRange(space.room, space);
  return { depth: depthPng(p, W, H, range), normal: normalPng(p, W, H), edge: edgePng(p, W, H), passes: p, W, H };
}

/** Resolve a shot's camera: solved from `framing` when present, else the explicit `camera`. */
export function shotCamera(shot: ShotSpec, space: SpaceSpec, W: number, H: number): CameraSpec {
  if (shot.framing) return solveCamera(shot.framing, space, shot.figures ?? [], W, H);
  if (shot.camera) return shot.camera;
  throw new Error(`shot "${shot.id}" has neither framing nor camera`);
}

export function renderShot(
  shot: ShotSpec,
  space: SpaceSpec,
  opts: { guides: boolean; mode?: BoardMode },
) {
  const W = shot.width ?? 1536, H = shot.height ?? 1024;
  const mode = opts.mode ?? "wire";
  // Open-sky volumes have no shell above the horizon, so the clear colour IS the sky in the clay
  // pass. It must read as overcast sky rather than as a white studio backdrop. See CanvasSink.
  const sink = opts.guides
    ? new SvgSink(W, H, true)
    : new CanvasSink(W, H, false, space.openToSky ? "#9aa3ab" : "#fff");
  const cam = shotCamera(shot, space, W, H);
  const b = cameraBasis(cam);

  if (mode === "blockout" || mode === "clay") {
    const lights = collectLights(space);
    emitFaces(sink, collectFaces(shot, space), b, W, H, lights, mode === "clay",
      mode === "clay" ? collectShadowers(space) : []);
    for (const lamp of space.props.filter((p) => p.type === "lamp")) {
      const at: Vec3 = lamp.pos ?? [0, space.room.h - 0.35, space.room.d / 2];
      const cc = toCam(b, at);
      if (cc[2] >= NEAR) {
        // A small hot core, not a disc the size of the shade. The shade is real geometry now, so
        // this only has to say "the source is here"; drawn at full shade radius it was a white ball
        // covering a quarter of the frame, and the generator reproduced exactly that.
        const f = W / 2 / Math.tan((b.fov * Math.PI) / 360);
        sink.disc(project(cc, b, W, H), ((lamp.r ?? 0.09) * 0.34 * f) / cc[2], 0.96);
      }
    }
    if (!opts.guides) return (sink as CanvasSink).png();
  }

  emit(sink, roomEdges(space.room), b, W, H, { layer: "structure" });

  for (const p of expandProps(space.props)) {
    if ((p.type === "box" || p.type === "cyl") && p.pos && p.size) {
      emit(sink, boxEdges(p.pos, p.size, p.yaw ?? 0), b, W, H, { layer: "prop" });
    } else if (p.type === "panel") {
      emit(sink, panelEdges(p, space.room), b, W, H, { layer: "prop" });
    } else if (p.type === "lamp") {
      const cord = p.cord ?? 0.35;
      const at: Vec3 = p.pos ?? [0, space.room.h - cord, space.room.d / 2];
      // cord 0 = a standing/table lamp, not hung from the ceiling
      if (cord > 0) emit(sink, [[[at[0], space.room.h, at[2]], at]] as Seg[], b, W, H, { layer: "prop" });
      emitSphere(sink, at, p.r ?? 0.08, b, W, H, { layer: "prop" });
    }
  }

  for (const f of shot.figures ?? []) {
    const g = figureGeom(f);
    emit(sink, g.bones, b, W, H, { layer: "figure" });
    emitSphere(sink, g.head.c, g.head.r, b, W, H, { layer: "figure" });
  }

  if (opts.guides) {
    for (let i = 1; i < 3; i++) {
      sink.poly([[(W * i) / 3, 0], [(W * i) / 3, H]], { layer: "guide", dash: true });
      sink.poly([[0, (H * i) / 3], [W, (H * i) / 3]], { layer: "guide", dash: true });
    }
    const fr = shot.framing;
    const label =
      `${shot.id}   ${space.id}   ` +
      (fr ? `${fr.size ?? "MS"} ${fr.angle ?? "eye"} ${fr.lens ?? 35}mm on ${fr.subject}   ` : "") +
      `fov ${(cam.fov ?? 40).toFixed(0)}` + (cam.roll ? `   roll ${cam.roll}` : "") +
      `   cam [${cam.pos.map((n) => n.toFixed(1)).join(", ")}]`;
    sink.text([18, H - 40], label, 22);
    if (shot.note) sink.text([18, H - 14], shot.note.slice(0, 120), 20);
    for (const f of shot.figures ?? []) {
      const cc = toCam(b, [f.at[0], (f.height ?? 1.7) * 1.02, f.at[1]]);
      if (cc[2] >= NEAR) {
        const p = project(cc, b, W, H);
        sink.text([p[0] + 8, p[1] - 6], f.label ?? f.id, 20);
      }
    }
  }

  return sink instanceof SvgSink ? sink.done() : (sink as CanvasSink).png();
}

// ---------------------------------------------------------------- io

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

export async function loadSpace(locationId: string): Promise<SpaceSpec> {
  const p = path.join(ROOT, "locations", locationId, "space.json");
  try {
    return JSON.parse(await fs.readFile(p, "utf8"));
  } catch {
    throw new Error(`no space.json for location "${locationId}" (expected ${p}) — author the room solid first`);
  }
}

export async function renderDay(dayId: string): Promise<{ id: string; svg: string; png: string }[]> {
  const dayDir = path.join(ROOT, "days", dayId);
  const shots: ShotSpec[] = JSON.parse(await fs.readFile(path.join(dayDir, "shots.json"), "utf8")).shots;
  const outDir = path.join(dayDir, "board");
  await fs.mkdir(outDir, { recursive: true });

  const spaces = new Map<string, SpaceSpec>();
  const made: { id: string; svg: string; png: string }[] = [];

  for (const shot of shots) {
    if (!spaces.has(shot.location)) spaces.set(shot.location, await loadSpace(shot.location));
    const space = spaces.get(shot.location)!;
    const svg = renderShot(shot, space, { guides: true, mode: "wire" }) as string;
    const png = renderShot(shot, space, { guides: false, mode: "wire" }) as Buffer;
    const blk = renderShot(shot, space, { guides: false, mode: "blockout" }) as Buffer;
    const svgPath = path.join(outDir, `${shot.id}.board.svg`);
    const pngPath = path.join(outDir, `${shot.id}.lineart.png`);
    await fs.writeFile(svgPath, svg, "utf8");
    await fs.writeFile(pngPath, png);
    await fs.writeFile(path.join(outDir, `${shot.id}.blockout.png`), blk);
    await fs.writeFile(path.join(outDir, `${shot.id}.clay.png`), renderShot(shot, space, { guides: false, mode: "clay" }) as Buffer);
    const ps = renderPasses(shot, space);
    await fs.writeFile(path.join(outDir, `${shot.id}.depth.png`), ps.depth);
    await fs.writeFile(path.join(outDir, `${shot.id}.normal.png`), ps.normal);
    await fs.writeFile(path.join(outDir, `${shot.id}.edge.png`), ps.edge);
    made.push({ id: shot.id, svg: svgPath, png: pngPath });
  }
  return made;
}

/**
 * Does each solved camera actually deliver the shot that was ordered? Reprojects the subject and
 * reports the framing error in pixels, plus the blockout's tonal spread (a board with no tonal
 * range carries no depth and will condition nothing).
 */
export async function checkDay(dayId: string) {
  const dayDir = path.join(ROOT, "days", dayId);
  const shots: ShotSpec[] = JSON.parse(await fs.readFile(path.join(dayDir, "shots.json"), "utf8")).shots;
  console.log("shot           ordered        landed        err px   tones  ink");
  console.log("-".repeat(68));
  for (const shot of shots) {
    const space = await loadSpace(shot.location);
    const W = shot.width ?? 1536, H = shot.height ?? 1024;
    const cam = shotCamera(shot, space, W, H);
    const b = cameraBasis(cam);
    const { c: subj } = subjectSolid(shot.framing ?? { subject: [0, 1, 2] }, space, shot.figures ?? []);
    const cs = toCam(b, subj);
    const p = cs[2] >= NEAR ? project(cs, b, W, H) : [NaN, NaN];
    const want = shot.framing?.place ?? [0.5, 0.5];
    const err = Math.hypot(p[0] - want[0] * W, p[1] - want[1] * H);

    // tonal spread of the blockout: count distinct 16-level buckets actually used
    const blk = renderShot(shot, space, { guides: false, mode: "blockout" }) as Buffer;
    const img = await loadImageBuf(blk, W, H);
    const buckets = new Set<number>();
    let ink = 0;
    for (let i = 0; i < img.length; i += 4) {
      const v = img[i];
      buckets.add(v >> 4);
      if (v < 200) ink++;
    }
    console.log(
      `${shot.id.padEnd(14)}[${want[0]},${want[1]}]`.padEnd(30) +
      `[${(p[0] / W).toFixed(2)},${(p[1] / H).toFixed(2)}]`.padEnd(14) +
      `${err.toFixed(0).padStart(5)}   ${String(buckets.size).padStart(4)}   ${((ink / (img.length / 4)) * 100).toFixed(0)}%`,
    );
  }
}

async function loadImageBuf(buf: Buffer, W: number, H: number): Promise<Uint8ClampedArray> {
  const { loadImage } = await import("@napi-rs/canvas");
  const img = await loadImage(buf);
  const cv = createCanvas(W, H);
  const ctx = cv.getContext("2d");
  ctx.drawImage(img, 0, 0, W, H);
  return ctx.getImageData(0, 0, W, H).data;
}

if (process.argv[1]?.endsWith("storyboard.ts")) {
  const cmd = process.argv[2];
  const day = (cmd === "render" || cmd === "check" ? process.argv[3] : process.argv[2]) ?? "day_00";
  if (cmd === "check") {
    await checkDay(day);
  } else {
    const made = await renderDay(day);
    for (const m of made) console.log(`  ${m.id}  ->  ${path.basename(m.svg)} + lineart + blockout`);
    console.log(`\n${made.length} boards -> days/${day}/board/`);
  }
}
