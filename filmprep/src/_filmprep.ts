/**
 * Clay passes and still prompts for clay_film (../clay_film).
 *
 *   npx tsx src/_filmprep.ts <job.json>
 *
 * The job file is written by clay_film.stages.clay: coordinates are already resolved from marks,
 * people are already their canon descriptions. This file only does what needs the Roko geometry:
 * solve the camera, render clay and depth, build the prompt from what the camera actually sees,
 * and report the gates so the Python side can retry a failed shot with its next candidate.
 *
 * Gates (a failing shot is still written, with its reasons, so a human can look):
 *   fov    > 85 deg: the room could not give the throw and the solver widened toward fisheye.
 *   cover  > 45%: one prop fills the frame, nothing left to photograph.
 *   sparse < 4 describable props.
 *   void   open-sky locations only: the sight-line leaves the modelled volume within 6 m of the
 *          subject. The clay then shows sky where ground and trees should be, and the still model
 *          invents whatever it likes there. Measured 2026-09-24: Menote shot looking uphill from
 *          z 4.2 got a lake painted behind him, uphill, where no lake exists.
 *
 * Prompt rules are editprompt.ts's: the input is never named as CG (assertEditPrompt), people lead
 * because a room-first ordering drops them, materials come per surface from `visible`, and the
 * camera line is the solver's ground truth. Exterior wording replaces stage1Prompt's "room".
 */
import fs from "node:fs/promises";
import path from "node:path";
import { locationLook, type SceneSpec, type ShotSpecLite, type Pose } from "./scenespec.js";
import { resolveShot } from "./geobridge.js";
import { loadSpace } from "./storyboard.js";
import { materialsClause, lightClause, cameraLine, assertEditPrompt } from "./editprompt.js";

type Person = { desc: string; action: string; placement: string };
type JobShot = {
  sid: string;
  location: string;
  shot: ShotSpecLite;
  figureAt?: [number, number] | null;
  figureYaw?: number;
  extraFigures?: { at: [number, number]; yaw?: number; pose?: Pose; id?: string }[];
  emptySubject?: string | null;
  people: Person[];
};
type Job = { out: string; width: number; height: number; shots: JobShot[] };

const VOID_M = 6;
const job: Job = JSON.parse(await fs.readFile(process.argv[2], "utf8"));
await fs.mkdir(job.out, { recursive: true });
const rows: unknown[] = [];

for (const s of job.shots) {
  const space = await loadSpace(s.location);
  const look = locationLook(s.location) ?? "";
  const lead = s.people[0];
  const spec: SceneSpec = {
    location_id: s.location, location_description: "",
    subject: lead ? { description: lead.desc, wardrobe: "" } : null,
    light: "overcast_daylight", mood: "muted_desaturated", shots: [s.shot],
  };
  const r = await resolveShot({ ...s.shot, control: "depth" }, spec, Number(s.sid.slice(1)) || 0, {
    width: job.width, height: job.height,
    emptySubject: s.emptySubject ?? undefined,
    figureAt: s.figureAt ?? undefined, figureYaw: s.figureYaw,
    extraFigures: s.extraFigures,
  });
  const cam = r.board.camera!;
  const fov = cam.fov ?? 0;
  const cover = r.visible.length ? Math.max(...r.visible.map((v) => v.cover ?? 0)) : 0;

  // How far the model extends beyond the subject along the sight-line, in plan view.
  let reach: number | null = null;
  const dx = cam.look[0] - cam.pos[0], dz = cam.look[2] - cam.pos[2];
  const n = Math.hypot(dx, dz) || 1;
  const ux = dx / n, uz = dz / n;
  if ((space as { openToSky?: boolean }).openToSky) {
    const w2 = space.room.w / 2, d = space.room.d;
    const tx = ux > 1e-6 ? (w2 - cam.look[0]) / ux : ux < -1e-6 ? (-w2 - cam.look[0]) / ux : Infinity;
    const tz = uz > 1e-6 ? (d - cam.look[2]) / uz : uz < -1e-6 ? (0 - cam.look[2]) / uz : Infinity;
    reach = Math.max(0, Math.min(tx, tz));
  }
  const gates: string[] = [];
  if (fov > 85) gates.push(`fov ${fov.toFixed(0)}`);
  if (cover > 0.45) gates.push(`cover ${(cover * 100).toFixed(0)}%`);
  if (r.visible.length < 4) gates.push(`sparse ${r.visible.length}`);
  if (reach !== null && reach < VOID_M) gates.push(`void ${reach.toFixed(1)}m`);

  const fixture = r.inFrame.find((v) => /lamp|bulb|fixture|sconce|pendant|candle|lantern/i.test(v.desc));
  const light = fixture ? lightClause(fixture.desc, r.inFrame) : "";

  // Which way each person faces relative to the camera, from the geometry. The clay carries no
  // figure, so without this the still model turns everyone toward the lens: s005 of the v2 demo
  // (three_quarter_back, camera behind him) came back with Menote facing the camera.
  const places: { at?: [number, number] | null; yaw?: number }[] =
    [{ at: s.figureAt, yaw: s.figureYaw ?? 180 }, ...(s.extraFigures ?? [])];
  const orient = (k: number): string => {
    const f = places[k];
    if (!f?.at) return "";
    const yr = ((f.yaw ?? 180) * Math.PI) / 180;
    const tx = cam.pos[0] - f.at[0], tz = cam.pos[2] - f.at[1];
    const tn = Math.hypot(tx, tz) || 1;
    const dot = (Math.sin(yr) * tx + Math.cos(yr) * tz) / tn;
    return dot > 0.5 ? "facing the camera" : dot < -0.5 ? "seen from behind, facing away from the camera"
      : "side-on to the camera";
  };
  const who = s.people.map((p, k) => {
    const tag = s.people.length > 1 ? `PERSON ${["ONE", "TWO", "THREE", "FOUR"][k] ?? k + 1}: ` : "";
    const o = orient(k);
    return `${tag}${p.desc}, ${o ? o + ", " : ""}${p.action}, ${p.placement}.`;
  });

  // Drop look clauses that name something out of shot. The lakeside look says "the lake beyond
  // reading as a flat bright pewter glare"; on an uphill shot Qwen painted that lake into the sky
  // gap behind Menote (v2 demo s003), although no water was in frame.
  const words = (t: string) => new Set((t.toLowerCase().match(/[a-z]{4,}/g) ?? []));
  const inDescs = r.inFrame.map((v) => v.desc);
  const inWords = new Set<string>();
  const outWords = new Set<string>();
  // Only backdrop-sized props (6 m or more across) can veto a look clause: a small boulder out of
  // shot must not strip "granite flags" from a frame full of flagstones.
  for (const pr of (space.props ?? []) as { label?: string; desc?: string; size?: number[] }[]) {
    const bag = words(pr.label ?? "");
    if (pr.desc && inDescs.some((d) => d.includes(pr.desc!))) { bag.forEach((w) => inWords.add(w)); words(pr.desc).forEach((w) => inWords.add(w)); }
    else if (Math.max(pr.size?.[0] ?? 0, pr.size?.[2] ?? 0) >= 6) bag.forEach((w) => outWords.add(w));
  }
  const drop = [...outWords].filter((w) => !inWords.has(w));
  const dropRe = drop.length ? new RegExp(`\\b(${drop.join("|")})\\b`, "i") : null;
  const lookShot = dropRe
    ? look.split(/(?<=[.,:;])\s+/).filter((c) => !dropRe.test(c)).join(" ").replace(/[,:;]\s*$/, ".")
    : look;
  const count = ["", "one person", "two people", "three people", "four people"][s.people.length] ?? "people";
  const lines = [
    "This is a photograph of a real place that has come out badly: flat, colourless, and missing all",
    `surface detail. Restore it into a properly exposed, photorealistic film still of that same place${
      s.people.length ? `, with ${count} in it.` : "."}`,
    "",
    ...who,
    who.length ? "Light them with the light already in the scene, with correct contact shadows." : "",
    "",
    "Give every surface its real material and texture:",
    materialsClause(r.visible) + ".",
    "",
    lookShot,
    light ? `${light}, with the falloff and shadows that light would really cast.` : "",
    "",
    "Keep the camera and the layout exactly as they are:",
    cameraLine({ fovDeg: fov, heightM: cam.pos[1] }),
    "",
    s.people.length
      ? "Photorealistic, sharp focus, fine surface detail, natural skin texture, fine film grain. No text or watermarks."
      : "Deserted and silent, not a living soul in sight. Sharp focus, fine surface detail, fine film grain. No text or watermarks.",
  ];
  const prompt = lines.filter((l, k, a) => !(l === "" && a[k - 1] === "")).join("\n").trim();
  assertEditPrompt(prompt, s.sid);
  await fs.writeFile(path.join(job.out, `${s.sid}_clay.png`), r.clay);
  await fs.writeFile(path.join(job.out, `${s.sid}_depth.png`), r.depth);
  await fs.writeFile(path.join(job.out, `${s.sid}_prompt.txt`), prompt, "utf8");
  const row = { sid: s.sid, fov: +fov.toFixed(1), heightM: +cam.pos[1].toFixed(2), visible: r.visible.length,
                maxCover: +cover.toFixed(3), reach: reach === null ? null : +reach.toFixed(1),
                view: [+ux.toFixed(3), +uz.toFixed(3)], lookDropped: drop, gates };
  rows.push(row);
  console.log(JSON.stringify(row));
}
await fs.writeFile(path.join(job.out, "prep_last.json"), JSON.stringify(rows, null, 1), "utf8");
