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
import { resolveShot, IN_SHOT_COVER } from "./geobridge.js";
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
  azimuth?: number | null;
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
    azimuth: s.azimuth ?? undefined,
    reveal: true,
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
  // The edit input is the clay with a lit mannequin for every person the camera sees
  // (r.peopleClay), so the prompt names exactly those people, each tied to its mannequin by screen
  // side. A person the camera cannot see is left out: a prompt that counted three people over room
  // for two is what made the model invent a place for the third (p01 s004, a cloned woman).
  const TAGS = ["ONE", "TWO", "THREE", "FOUR"];
  const kept = s.people.map((_, k) => k).filter((k) => (r.masks[k]?.cover ?? 0) >= IN_SHOT_COVER);
  const sx = (k: number): number => {
    const m = r.masks[k];
    if (m.head && m.head[0] >= 0 && m.head[0] <= 1 && m.head[1] >= 0 && m.head[1] <= 1) return m.head[0];
    return m.bbox ? (m.bbox[0] + m.bbox[2]) / 2 : 0.5;
  };
  const sideOf = (x: number) => (x < 0.36 ? "on the left" : x > 0.64 ? "on the right" : "in the centre");
  const dist = (k: number) => {
    const at = places[k]?.at;
    return at ? Math.hypot(cam.pos[0] - at[0], cam.pos[2] - at[1]) : 0;
  };
  // Tie each person to a mannequin by left-to-right order, which is what the eye reads. Two people
  // "on the left" told apart only as nearer/further were swapped (p13 s000: the woman is behind the
  // narrator and right of him on screen, and came back left of him). Near/far is kept only for two
  // figures overlapping on screen, where order across the frame means nothing.
  const ORD = ["first", "second", "third", "fourth"];
  const where = (k: number): string => {
    const side = sideOf(sx(k));
    const overlap = kept.filter((j) => j !== k && Math.abs(sx(j) - sx(k)) < 0.06);
    if (overlap.length) {
      const near = overlap.every((j) => dist(k) < dist(j)), far = overlap.every((j) => dist(k) > dist(j));
      return `the ${near ? "nearer" : far ? "further" : "middle"} of the overlapping clay figures ${side}`;
    }
    const rank = kept.filter((j) => sx(j) < sx(k) && Math.abs(sx(j) - sx(k)) >= 0.06).length;
    return `the ${ORD[rank] ?? `${rank + 1}th`} clay figure from the left, ${side}`;
  };
  // How far each person stands from the nearest visible piece of furniture, and on which side of it,
  // so the text agrees with the mannequin about depth. Without it the model pulled a woman standing a
  // metre behind a trough up against it (p17 s004), and moved a couple from the shop door to the kerb
  // (p17 s001), 2026-10-11.
  type Solid = { desc?: string; label?: string; pos?: number[]; size?: number[]; yaw?: number };
  const solids = ((space.props ?? []) as Solid[]).filter((q) => q.pos && q.size && q.desc
    && Math.max(q.size[0], q.size[2]) < 5 && q.pos[1] - q.size[1] / 2 < 1.2
    && q.size[0] * q.size[1] * q.size[2] >= 0.15   // a whole object, not its bumper or a wheel
    && r.inFrame.some((v) => v.desc === q.desc));
  const nameOf = (q: Solid) => "the " + q.desc!.split(",")[0].replace(/^(a|an|the)\s+/i, "").trim();
  const gapTo = (q: Solid, at: [number, number]) => {
    const yr = ((q.yaw ?? 0) * Math.PI) / 180, dx = at[0] - q.pos![0], dz = at[1] - q.pos![2];
    const lx = dx * Math.cos(yr) - dz * Math.sin(yr), lz = dx * Math.sin(yr) + dz * Math.cos(yr);
    const ox = Math.max(0, Math.abs(lx) - q.size![0] / 2), oz = Math.max(0, Math.abs(lz) - q.size![2] / 2);
    return Math.hypot(ox, oz) - 0.2;
  };
  const amount = (g: number) => (g < 0.75 ? "half a metre" : g < 1.5 ? "about a metre" : g < 2.5 ? "about two metres"
    : `about ${Math.round(g)} metres`);
  const relation = (k: number): string => {
    const at = places[k]?.at;
    if (!at || !solids.length) return "";
    const q = solids.reduce((a, b) => (gapTo(b, at) < gapTo(a, at) ? b : a));
    const g = gapTo(q, at);
    if (g < 0.35) return `right at ${nameOf(q)}`;
    if (g > 4) return "";
    const camToProp = Math.hypot(cam.pos[0] - q.pos![0], cam.pos[2] - q.pos![2]);
    return camToProp < dist(k) ? `${amount(g)} beyond ${nameOf(q)}, with clear floor between them`
      : `${amount(g)} nearer the camera than ${nameOf(q)}`;
  };
  // Screen-position words ("in the left third") come out of the placement: the mannequins say where
  // people are, and the words fought them and won (p13 s000). Same rule as clay_film/placewords.py,
  // which strips the director's shot sentence: a side word goes only when punctuation, a frame noun
  // or a preposition follows it, so "his right wrist" stays.
  const SIDE = "(?:far )?(?:left|right)", FRAME = "(?: (?:third|edge|half|side)(?: of (?:the )?frame)?)?";
  // No "of" in NEXT, and eyelines are held out of the strip (audit 2026-10-12): see placewords.py.
  const NEXT = "(?=[,.;:]|$|\\s+(?:beyond|behind|by|near|in|and|with|at|across|over|under|against|foreground|background)\\b)";
  const EYELINE = new RegExp(`\\b((?:look|gaz|star|glanc|fac|turn|peer)\\w*(?: (?:off|out|away|across|back|over))?`
    + ` (?:to|toward|towards|at|into) the ${SIDE}(?: (?:third|edge|half|side))?(?: of (?:the )?frame)?)`, "gi");
  const stripPlace = (t0: string) => {
    const kept: string[] = [];
    const t = t0.replace(EYELINE, (_m, g: string) => { kept.push(g); return `\u0000${kept.length - 1}\u0000`; });
    return stripRest(t).replace(/\u0000(\d+)\u0000/g, (_m, i: string) => kept[Number(i)]);
  };
  const stripRest = (t: string) => t
    .replace(new RegExp(`\\b(in|at|across|into|through) the (?:${SIDE}|centre|center) (foreground|background)\\b`, "gi"), "$1 the $2")
    .replace(new RegExp(`,?\\s*\\b(?:in|at|on|to) the ${SIDE}(?: (?:third|edge|half|side))? of (?:the )?frame\\b`, "gi"), "")
    .replace(new RegExp(`,?\\s*\\b(?:in|at|on|to) the ${SIDE}${FRAME}${NEXT}`, "gi"), "")
    .replace(/,?\s*\b(?:just )?(?:left|right) of (?:centre|center)\b/gi, "")
    .replace(/,?\s*\bin the (?:centre|center)(?: third)?(?: of (?:the )?frame)?\b/gi, "")
    .replace(/,?\s*\bscreen (?:left|right)\b/gi, "")
    .replace(/\s+,/g, ",").replace(/,\s*,/g, ",").replace(/\s{2,}/g, " ").replace(/^[,\s]+/, "").trim();
  // "the tall woman", "the short heavy man with his back to us": for the order line and "hidden behind"
  const shortName = (k: number) => {
    const d = s.people[k].desc.split(",")[0].replace(/\s+of\s+.*$/, "").replace(/^(a|an)\s+/i, "the ");
    const o = orient(k);
    const pron = /\b(woman|girl|mother)\b/.test(d) ? "her" : "his";
    return d + (o.startsWith("seen from behind") ? ` with ${pron} back to us` : "");
  };
  const who = kept.map((k, j) => {
    const p = s.people[k];
    const tag = kept.length > 1 ? `PERSON ${TAGS[j] ?? j + 1} (${where(k)}): ` : "";
    const o = orient(k);
    // Someone the camera barely sees keeps no action. "Her bare hand resting on the dealer's hand" for
    // a woman sitting behind the narrator made the model bring her to the front on every seed, so the
    // two swapped seats (p01 s005, 2026-10-11).
    if (r.masks[k].seen < 0.3) {
      const front = kept.filter((j) => j !== k && dist(j) < dist(k) && Math.abs(sx(j) - sx(k)) < 0.2)
        .sort((x, y) => Math.abs(sx(x) - sx(k)) - Math.abs(sx(y) - sx(k)))[0];
      // The director's shot line can still give the hidden person's hand a job (p01 s005), so say where
      // that hand comes from rather than leave the model to move her forward to free it.
      const her = /\b(woman|girl|mother)\b/.test(p.desc.split(",")[0]) ? "her" : "his";
      const fp = front !== undefined && /\b(woman|girl|mother)\b/.test(s.people[front].desc.split(",")[0]) ? "her" : "him";
      const hand = /\bhand/.test(p.action) ? `; only ${her} arm and hand show, reaching past ${fp} across the table` : `, only a little of ${her === "her" ? "her" : "him"} showing`;
      const hidden = front !== undefined
        ? `further from the camera than ${shortName(front).replace(/ with (his|her) back to us$/, "")} and mostly hidden behind ${fp}, who is nearer the camera${hand}`
        : "mostly hidden, only a little showing";
      return `${tag}${p.desc}, ${o ? o + ", " : ""}${hidden}.`;
    }
    const rel = relation(k);
    const place = stripPlace(p.placement);
    return `${tag}${p.desc}, ${o ? o + ", " : ""}${p.action}${place ? ", " + place : ""}${rel ? ", " + rel : ""}.`;
  });

  // Props with a front (a car, `front` in space.json as a world direction): which way that front
  // points in this frame. A car's clay box reads the same at both ends, and p17 s001 came back with
  // the saloon reversed.
  const facingLines = ((space.props ?? []) as (Solid & { front?: number[] })[])
    .filter((q) => q.front && q.desc && r.inFrame.some((v) => v.desc === q.desc))
    .map((q) => {
      const f = q.front!, fw = [cam.look[0] - cam.pos[0], cam.look[1] - cam.pos[1], cam.look[2] - cam.pos[2]];
      const n = Math.hypot(fw[0], fw[1], fw[2]) || 1;
      const F = [fw[0] / n, fw[1] / n, fw[2] / n];
      const R = [F[1] * 0 - F[2] * 1, F[2] * 0 - F[0] * 0, F[0] * 1 - F[1] * 0];   // cross(fwd, up): screen right
      const across = f[0] * R[0] + f[2] * R[2], toward = -(f[0] * F[0] + f[2] * F[2]);
      const dir = Math.abs(across) >= Math.abs(toward)
        ? (across > 0 ? "points to the right of the frame" : "points to the left of the frame")
        : (toward > 0 ? "points toward the camera" : "points away from the camera");
      return `The front end of ${nameOf(q)} ${dir}.`;
    });
  // One plain left-to-right line. The director's sentence names the lead first ("the woman sits in the
  // left third ... the narrator beside her"), which the model read as order (p13 s000), and a person
  // seen from behind whose description lists a face came back twice, once to show it (p13 s005).
  const order = kept.length > 1
    ? `Left to right across the frame: ${[...kept].sort((a, b) => sx(a) - sx(b)).map(shortName).join(", ")}. `
      + `Each person appears exactly once; nobody is doubled and nobody else is in the frame.`
    : "";
  const dummies = (kept.length > 1
    ? "The clay mannequins in this photograph, each tinted the colour of that person's clothes, are the people: turn each one into the person described, in exactly its place, size and pose, and add nobody else."
    : "The clay mannequin in this photograph, tinted the colour of the person's clothes, is the person: turn it into the person described, in exactly its place, size and pose, and add nobody else.")
    + (order ? " " + order : "");

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
  const count = ["", "one person", "two people", "three people", "four people"][kept.length] ?? "people";
  const lines = [
    "This is a photograph of a real place that has come out badly: flat, colourless, and missing all",
    `surface detail. Restore it into a properly exposed, photorealistic film still of that same place${
      kept.length ? `, with ${count} in it.` : "."}`,
    "",
    ...who,
    who.length ? `Light them with the light already in the scene, with correct contact shadows. ${dummies}` : "",
    "",
    "Give every surface its real material and texture:",
    materialsClause(r.visible) + ".",
    ...facingLines,
    "",
    lookShot,
    light ? `${light}, with the falloff and shadows that light would really cast.` : "",
    "",
    "Keep the camera and the layout exactly as they are:",
    cameraLine({ fovDeg: fov, heightM: cam.pos[1] }),
    "",
    kept.length
      ? "Photorealistic, sharp focus, fine surface detail, natural skin texture, fine film grain. No text or watermarks."
      : "Deserted and silent, not a living soul in sight. Sharp focus, fine surface detail, fine film grain. No text or watermarks.",
  ];
  const prompt = lines.filter((l, k, a) => !(l === "" && a[k - 1] === "")).join("\n").trim();
  assertEditPrompt(prompt, s.sid);
  await fs.writeFile(path.join(job.out, `${s.sid}_clay.png`), r.clay);
  await fs.writeFile(path.join(job.out, `${s.sid}_depth.png`), r.depth);
  await fs.writeFile(path.join(job.out, `${s.sid}_prompt.txt`), prompt, "utf8");
  // Masks follow the order of s.people: the lead is the framed subject, then extraFigures.
  const figs = r.masks.map((m, k) => ({ k, id: m.id, cover: +m.cover.toFixed(4), seen: +m.seen.toFixed(3),
    bbox: m.bbox?.map((v) => +v.toFixed(3)) ?? null, head: m.head?.map((v) => +v.toFixed(3)) ?? null, headR: m.headR === null ? null : +m.headR.toFixed(4) }));
  if (r.peopleClay && kept.length) await fs.writeFile(path.join(job.out, `${s.sid}_people.png`), r.peopleClay);
  for (const [k, m] of r.masks.entries()) {
    if (k < s.people.length) await fs.writeFile(path.join(job.out, `${s.sid}_fig${k}.png`), m.png);
  }
  const row = { sid: s.sid, fov: +fov.toFixed(1), heightM: +cam.pos[1].toFixed(2), visible: r.visible.length,
                maxCover: +cover.toFixed(3), reach: reach === null ? null : +reach.toFixed(1),
                view: [+ux.toFixed(3), +uz.toFixed(3)], lookDropped: drop, gates,
                camPos: cam.pos.map((v) => +v.toFixed(3)), camLook: cam.look.map((v) => +v.toFixed(3)),
                figs: figs.slice(0, s.people.length), shown: kept, revealed: r.revealed,
                // prompt version: 3 = tinted mannequins, no screen-position words, furniture distance, car facing
                prepv: 3 };
  rows.push(row);
  console.log(JSON.stringify(row));
}
// Merge by sid: a repair round re-preps only the shots it fixed, and overwriting the file left every
// other shot looking unrendered to the shot gate (clay_film.shotgate reads one row per shot).
const prepFile = path.join(job.out, "prep_last.json");
let merged: { sid: string }[] = [];
try { merged = JSON.parse(await fs.readFile(prepFile, "utf8")); } catch { merged = []; }
const fresh = new Set((rows as { sid: string }[]).map((r) => r.sid));
merged = [...merged.filter((r) => !fresh.has(r.sid)), ...(rows as { sid: string }[])]
  .sort((a, b) => a.sid.localeCompare(b.sid));
await fs.writeFile(prepFile, JSON.stringify(merged, null, 1), "utf8");
