/**
 * assets.ts — real 3D models as space.json props.
 *
 * WHY THIS EXISTS
 *   Every prop in this pipeline used to be authored as JSON geometry, because `Face` was the
 *   only thing the rasteriser understood and the only way to make one was to write boxes.
 *   Twenty-eight locations cost 18,612 lines of hand-written solids that way, and the ledger
 *   records what it bought: `column` expands to shaft + base + CAPITAL, so 19 tree trunks
 *   authored as `column` produced a ruined colonnade, and 103 props across 8 locations had to be
 *   converted to plain cylinders. There is no organic repeated form in the archetype vocabulary
 *   because a box-and-cylinder DSL cannot express one.
 *
 *   `Face` is `{ pts: Vec3[]; n: Vec3; base: number }` — an arbitrary convex polygon. A triangle
 *   has always been a legal Face. Nothing in the rasteriser needed to change; what was missing
 *   was a way to get triangles out of a file. That is this module, and nothing more.
 *
 * WHAT IT DOES NOT CHANGE
 *   The camera solve, the depth normalisation, the clay clear colour, the visible/inFrame split,
 *   the cover and background gates, owner tagging. A model prop carries pos+size like every other
 *   solid, so `subjectSolid` frames it, `expandProps` passes it through untouched, and the gates
 *   measure it from rasterised pixels exactly as before.
 *
 * THE SIZE CONTRACT, which is different from a box's
 *   A box's `size` IS its extent. A model's `size` is the BOX IT MUST FIT IN. Under the default
 *   `fit: "contain"` the mesh is uniformly scaled to fit inside that box and seated on its floor,
 *   so it usually occupies less than the authored volume. That is the point: the proportions come
 *   from the asset, which is the whole reason for using one. `fit: "stretch"` fills the box
 *   exactly and distorts — only for things with no correct proportion, like rubble or a hedge run.
 *
 * BACKFACES
 *   The rasteriser culls backfaces. Most low-poly CC0 assets are not watertight — a tree with
 *   plane-card leaves, a chair with no underside — and culling an open mesh punches holes clean
 *   through the depth pass. A hole reads to a ControlNet as background, which is the failure the
 *   `background <= 45%` gate exists to catch. The bake records watertightness per asset; open
 *   meshes emit both windings so no hole can appear, at 2x the faces. Do not override this.
 */

import { readFileSync, existsSync } from "node:fs";
import { join, resolve } from "node:path";
import type { Vec3 } from "./storyboard.ts";

/** Where the mesh library lives. Only read when a space.json prop names an `asset`; the example
 *  locations use none. Set ASSET_LIBRARY to use one; the default is filmprep/asset_library. */
export const LIBRARY_ROOT = resolve(process.env.ASSET_LIBRARY ?? "asset_library");

const TRI_MAGIC = 0x49525452;            // "RTRI" read as a little-endian u32
const TRI_VERSION = 1;
const HEADER_BYTES = 40;                 // 4 magic + 3 u32 + 6 f32
const FLOATS_PER_TRI = 10;               // ax ay az bx by bz cx cy cz albedo

export interface AssetRecord {
  slug: string;
  name: string;
  source: string;
  licence: string;
  category: string;
  tags?: string[];
  tri_count: number;
  extents: [number, number, number];
  source_size_m?: number[];
  watertight?: boolean;
  tri_path: string;
  author?: string;
  page_url?: string;
}

export interface AssetTri {
  pts: [Vec3, Vec3, Vec3];
  albedo: number;
}

interface LoadedAsset {
  rec: AssetRecord;
  /** Normalised: centred on the origin, longest axis exactly 1.0. */
  tris: Float32Array;
  lo: Vec3;
  hi: Vec3;
}

let manifest: Record<string, AssetRecord> | null = null;
const loaded = new Map<string, LoadedAsset>();

/** The manifest is a few hundred short records; read it once and keep it. */
export function assetManifest(): Record<string, AssetRecord> {
  if (manifest) return manifest;
  const p = join(LIBRARY_ROOT, "manifest.json");
  if (!existsSync(p)) {
    throw new Error(
      `no asset library at ${LIBRARY_ROOT}. Set ASSET_LIBRARY, or build one:\n` +
      `  python "${join(LIBRARY_ROOT, "cli.py")}" add "pine tree"`,
    );
  }
  const doc = JSON.parse(readFileSync(p, "utf8"));
  if (doc.version !== 1) throw new Error(`manifest version ${doc.version}, this reader expects 1`);
  manifest = doc.assets as Record<string, AssetRecord>;
  return manifest;
}

/** Slugs whose name, tags or category contain `q`. For the error message on a miss, and for
 *  an author who wants to know what vocabulary exists without shelling out to the CLI. */
export function assetSuggest(q: string, limit = 6): string[] {
  const t = q.toLowerCase();
  return Object.values(assetManifest())
    .filter((a) => `${a.slug} ${a.name} ${a.category} ${(a.tags ?? []).join(" ")}`.toLowerCase().includes(t))
    .slice(0, limit)
    .map((a) => a.slug);
}

function loadAsset(slug: string): LoadedAsset {
  const hit = loaded.get(slug);
  if (hit) return hit;

  const rec = assetManifest()[slug];
  if (!rec) {
    const near = assetSuggest(slug.split("_")[0]);
    throw new Error(
      `no asset "${slug}" in the library.` +
      (near.length ? ` Did you mean: ${near.join(", ")}?` : " Run `python cli.py list`."),
    );
  }

  const p = join(LIBRARY_ROOT, rec.tri_path);
  if (!existsSync(p)) {
    throw new Error(`asset "${slug}" is in the manifest but ${rec.tri_path} is missing. ` +
      `Run \`python cli.py verify\`.`);
  }
  const buf = readFileSync(p);
  if (buf.length < HEADER_BYTES) throw new Error(`${rec.tri_path}: truncated header`);
  if (buf.readUInt32LE(0) !== TRI_MAGIC) throw new Error(`${rec.tri_path}: not a .tri file`);
  const version = buf.readUInt32LE(4);
  if (version !== TRI_VERSION) {
    throw new Error(`${rec.tri_path}: .tri version ${version}, this reader expects ${TRI_VERSION}`);
  }
  const count = buf.readUInt32LE(8);
  const lo: Vec3 = [buf.readFloatLE(16), buf.readFloatLE(20), buf.readFloatLE(24)];
  const hi: Vec3 = [buf.readFloatLE(28), buf.readFloatLE(32), buf.readFloatLE(36)];

  const want = HEADER_BYTES + count * FLOATS_PER_TRI * 4;
  if (buf.length < want) {
    throw new Error(`${rec.tri_path}: header claims ${count} triangles, file holds ${
      Math.floor((buf.length - HEADER_BYTES) / (FLOATS_PER_TRI * 4))}`);
  }
  // slice() COPIES, which is what makes the Float32Array view safe: a Buffer from readFileSync
  // can sit at any byteOffset in a pooled ArrayBuffer, and a 4-byte-typed view over an unaligned
  // offset throws.
  const tris = new Float32Array(
    buf.buffer.slice(buf.byteOffset + HEADER_BYTES, buf.byteOffset + want),
  );

  const out: LoadedAsset = { rec, tris, lo, hi };
  loaded.set(slug, out);
  return out;
}

/** Extents of the normalised mesh. Largest is 1.0; the other two carry the aspect ratio. */
function extentsOf(a: LoadedAsset): Vec3 {
  const e: Vec3 = [a.hi[0] - a.lo[0], a.hi[1] - a.lo[1], a.hi[2] - a.lo[2]];
  return [Math.max(e[0], 1e-6), Math.max(e[1], 1e-6), Math.max(e[2], 1e-6)];
}

export interface ModelPlacement {
  asset: string;
  pos: Vec3;                    // centre of the authored box
  size: Vec3;                   // the box the mesh must fit in
  yaw?: number;                 // degrees about +Y, about the box centre
  fit?: "contain" | "stretch";
  albedo?: number;              // overrides the asset's baked per-triangle reflectance
}

/**
 * World-space triangles for one model prop.
 *
 * The local->world step is deliberately identical to `place()` in storyboard.ts: build every
 * vertex in the prop's OWN frame, then rotate about the prop's centre and translate. Composing
 * world arithmetic with a per-part yaw looks equivalent and is not — that bug turned every
 * rotated archetype into a chair with its arms in the wrong place, and a mesh would fail the
 * same way silently.
 */
export function assetTriangles(p: ModelPlacement): { tris: AssetTri[]; doubleSided: boolean } {
  const a = loadAsset(p.asset);
  const ext = extentsOf(a);
  const [sx, sy, sz] = p.size;
  const fit = p.fit ?? "contain";

  let kx: number, ky: number, kz: number, seatY: number;
  if (fit === "stretch") {
    kx = sx / ext[0]; ky = sy / ext[1]; kz = sz / ext[2];
    seatY = 0;
  } else {
    const k = Math.min(sx / ext[0], sy / ext[1], sz / ext[2]);
    kx = ky = kz = k;
    // Seat the mesh on the FLOOR of the authored box rather than leaving it centred. A prop
    // authored at pos.y = size.y/2 then stands on the room floor, which is the convention every
    // existing space.json already uses.
    seatY = -sy / 2 + (ext[1] * k) / 2;
  }

  const yr = ((p.yaw ?? 0) * Math.PI) / 180;
  const cs = Math.cos(yr), sn = Math.sin(yr);
  const [cxw, cyw, czw] = p.pos;
  const world = (lx: number, ly: number, lz: number): Vec3 =>
    [cxw + lx * cs + lz * sn, cyw + ly, czw - lx * sn + lz * cs];

  const src = a.tris;
  const n = src.length / FLOATS_PER_TRI;
  const out: AssetTri[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const o = i * FLOATS_PER_TRI;
    out[i] = {
      pts: [
        world(src[o] * kx, src[o + 1] * ky + seatY, src[o + 2] * kz),
        world(src[o + 3] * kx, src[o + 4] * ky + seatY, src[o + 5] * kz),
        world(src[o + 6] * kx, src[o + 7] * ky + seatY, src[o + 8] * kz),
      ],
      albedo: p.albedo ?? src[o + 9],
    };
  }
  return { tris: out, doubleSided: a.rec.watertight !== true };
}

/** Faces a placement will emit, without building them. For pre-flight budgeting. */
export function assetFaceCost(p: ModelPlacement): number {
  const a = loadAsset(p.asset);
  return a.rec.tri_count * (a.rec.watertight === true ? 1 : 2);
}

export function assetRecord(slug: string): AssetRecord {
  return loadAsset(slug).rec;
}
