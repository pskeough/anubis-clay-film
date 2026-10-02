/**
 * Image prompt engineering — centralized + model-specific.  ← review this file
 *
 * Two image models with DIFFERENT text encoders, so they want different prompt styles:
 *
 *  ┌ z-image-turbo  (encoder: Lumina2 / Gemma-style)  — TEXT→IMAGE
 *  │   Backgrounds + character master sheets. Prefers NATURAL, descriptive, photographic
 *  │   language — full clauses describing subject, lighting, mood, lens/composition. It does
 *  │   NOT use weight syntax like (word:1.3) and reacts poorly to comma-spam keyword lists, so
 *  │   we write coherent descriptive sentences. cfg 1, 6 steps (distilled turbo).
 *  │
 *  └ Flux.2 Klein 4B  (encoder: qwen3-4b)  — IMAGE EDIT (Kontext-style)
 *      Compose / variant edits / motion keyframes. Prefers a clear INSTRUCTION plus an explicit
 *      PRESERVATION clause: "<change> while preserving <identity>; only modify <X>; keep the rest
 *      identical." This is how the model keeps a character's face/identity stable across edits.
 *      cfg 1, 4 steps, references scaled to ~1 MP.
 *
 * Identity preservation is the whole game for character consistency: every Klein call that touches a
 * character appends a preservation clause so poses/outfits/expressions change WITHOUT redrawing the face.
 */

// ── Resolutions (per model; all multiples of 64) ───────────────────────────────
// The live engine dithers every plate down to a ~533px buffer, so resolution above ~960px is thrown
// away. These were lowered from 1536×1024/1.6MP to ~1152×768/0.88MP (≈1.8× faster gen, ~44% fewer
// pixels) while KEEPING the invariant that compose/bgEdit MP ≈ the background plate's MP so mask-diff
// registration and motion keyframes stay same-size. All env-overridable (ROKO_BG_W/H, ROKO_*_MP) —
// push to 960×640 / 0.6MP for ~2.6× if Klein quality still holds on your plates.
export const RES = {
  background:      { width: 1152, height: 768 },   // 3:2 cinematic scene plate (z-image), /64
  characterSheet:  { width: 768, height: 1152 },   // 2:3 full-body sheet (z-image), /64
  composeMegapixels: 0.9,    // Klein bg+character compose — matches the 1152×768 plate (~0.88 MP)
  editMegapixels:    0.7,    // Klein variant edit (character sheet), inherits the sheet's aspect
  bgEditMegapixels:  0.9,    // Klein edit of a 1152×768 background/scene — match its ~0.88 MP, shots stay same-size
};

// ── z-image (Lumina2) styles ──────────────────────────────────────────────────
// The live dither REPLACES source colour with a palette ramp keyed off LUMINANCE, and the per-scene
// mood/tint comes from the engine's palette — so the plate's own colour/mood is largely thrown away.
// What matters is luminance STRUCTURE: a full, legible range of darks→mids→lights and clear depth, so
// the dither reads cleanly. So the house style fixes the RENDER (photoreal, cinematic, strong tonal
// separation, no text) and stays mood-neutral — the scene prompt + palette carry the atmosphere.
export const HOUSE_STYLE =
  "Rendered in one consistent house style for a 1-bit dithered visual novel: photographic and realistic, " +
  "composed like a deliberate cinematic film still — a clear FOCAL SUBJECT and a strong foreground anchor, " +
  "with leading lines and layered depth (foreground, midground, background) drawing the eye to one point. " +
  "Never an empty, evenly-filled record shot of a room; even an environment shot is framed around something. " +
  "Strong, legible tonal range — deep shadows, distinct mid-tones, bright highlights — so luminance reads " +
  "cleanly when reduced to a few values. Even in night, interior, or low-light scenes keep shadows lifted " +
  "enough to read texture and form across the whole frame, with visible mid-tones and at least one source of " +
  "brighter or rim light for contrast — never a flat, crushed, near-black image. Natural light that models " +
  // "fine film grain" was here, and was asking in words for the same softness the 35mm LoRA was baking
  // in — see RENDER.loras. Both removed together on 2026-08-18.
  "form and texture, crisp detail, a shallow-to-medium depth of field that separates the " +
  "subject from its surroundings. Naturalistic colour. No on-screen text, captions, watermarks, logos, UI, " +
  "or borders, and no extra people beyond those explicitly described.";

// Character master sheet: neutral, evenly lit, plain backdrop, sharp face — the better the sheet, the
// better Klein preserves identity in later edits.
export const CHAR_SHEET_STYLE =
  "Presented as a full-body character reference sheet: the figure stands front-facing in a relaxed " +
  "neutral pose, full body in frame head to feet, against a plain seamless light-grey studio backdrop. " +
  "Soft even three-point lighting, no cast shadows on the backdrop, sharp clearly-defined facial features, " +
  "consistent realistic proportions. Photographic and realistic, no props, no on-screen text or watermark.";

// Non-human entity reference (spirits, creatures, manifestations): the humanoid "standing reference
// sheet" template forces absurd anatomy on a non-human (verified: the serpent came out as a goofy
// standing snake-person). An entity sheet shows the being as it MANIFESTS, photoreal and ominous.
export const ENTITY_SHEET_STYLE =
  "Presented as a reference study of this entity in full against a plain seamless dark studio backdrop: " +
  "its complete form visible, coiled or posed as it would naturally manifest — NOT standing like a person, " +
  "no humanoid posture or limbs unless described. Photographic, realistic textures (scale, skin, matter), " +
  "dramatic but legible lighting that defines its silhouette and surface, sharp detail on the head/face. " +
  "No props, no on-screen text or watermark.";

// ── Flux.2 Klein (qwen3) edit clauses ─────────────────────────────────────────
// Appended by L2 so the model always gets the change + a preservation guarantee.
// Compose: place a reference-sheet character INTO a background plate as the integrated focal subject.
// This is how a named character (the protagonist) gets into a scene WITHOUT z-image inventing a random
// person — identity comes from the sheet, the scene from the plate, Klein marries them.
export const COMPOSE_DIRECTIVE =
  "Place the person from the second image into the scene of the first image as the clear focal subject of a " +
  "cinematic film still. Match the scene's lighting direction, colour temperature, perspective, and grain; " +
  "scale them correctly for their position in space; and ground them with believable contact shadow and " +
  "occlusion so they truly inhabit the scene — not pasted on top. Preserve the person's exact face, " +
  "hairstyle, skin tone, build, and identity so they are unmistakably the same person; you MAY change their " +
  "pose, expression, and how they are lit to fit the moment described, but do not redesign them or change " +
  "who they are. If any person is ALREADY visible in the scene image, REPLACE them with the person from the " +
  "reference — the result must contain exactly the people being placed, never a duplicate or leftover figure. " +
  "Keep the background scene otherwise unchanged apart from the integrated figure, its shadow, " +
  "and its natural contact with the environment.";

// ── No-people mode (no-Menote shot language) ───────────────────────────────────
// The Klein/z-image stack renders people badly, and the game's look is environment-and-object driven
// (Critters-style). When a day has NO character assets, every shot is generated as an EMPTY scene:
// these clauses replace the person-forcing COMPOSE/EDIT/MOTION directives so nothing invents a figure.
// NO_PEOPLE is an INSTRUCTION clause for the Klein edit/compose/motion stack (Flux.2 instruct obeys
// negation and explicit "do not render X" commands). It must NOT be appended to a z-image POSITIVE
// prompt: z-image-turbo runs at cfg=1 with a zeroed negative, so the negative channel is inert and
// every body-part noun here ("hands, arms, faces…") lands in the positive conditioning and SUMMONS a
// figure (verified: dorm/grove plates painted people from this clause). For z-image use EMPTY_SCENE.
export const NO_PEOPLE =
  " Depict NO people and no human figure of any kind, and NO body parts (no hands, arms, fingers, faces, " +
  "heads, torsos, shoulders, legs, feet). Render only the environment, objects, surfaces, light, and " +
  "atmosphere — the space is empty of any living person. Any human presence is implied, never shown.";

// EMPTY_SCENE is the z-image (cfg=1, positive-only) counterpart: it suppresses figures using POSITIVE
// vacancy adjectives (abandoned/deserted/unoccupied/empty) that counter the model's "bed → sleeper",
// "path → walker" prior, while naming NO body parts and NO "people/person/figure" tokens (which would
// summon what they forbid). This is the only lever that works on a distilled turbo model at cfg=1.
export const EMPTY_SCENE =
  " The place is abandoned, deserted, and unoccupied — completely empty and still, vacant and silent, " +
  "long uninhabited. Only environment, architecture, objects, surfaces, light, and atmosphere remain; " +
  "nothing alive is present.";

// ANATOMY_GUARD: the harder no-people failure is not a named figure but Klein MORPHING body-shaped
// imagery into a body — "a fallen pale branch lies extended, dark channels spreading through it" became
// a tattooed forearm; "fallen branch / bitten ring / serpent-mark scored" became an arm over a log
// (verified, bg_bconsume_s01_04). Once it appears, REFRAME/BG_EDIT_PRESERVE faithfully carry it down the
// whole chain. This clause forces ambiguous organic forms to resolve as literal inanimate matter. Append
// it alongside NO_PEOPLE on every no-people Klein edit/compose/motion.
export const ANATOMY_GUARD =
  " Render every branch, root, log, twig, vine, ridge, channel, mark, or carved line as literal wood, bark, " +
  "stone, soil, ash, or water ONLY — inanimate natural matter with real bark, grain, or mineral texture. Do " +
  "NOT let any pale, smooth, tapering, or rounded form read as an arm, hand, finger, leg, limb, torso, neck, " +
  "or skin, and never render tattoos, veins, sinew, or flesh. If a described shape could resemble part of a " +
  "body, render it unmistakably as the dead wood, rock, or earth it actually is.";

// Characterless "compose": there is no figure to place — render the beat as an empty environmental still
// of the same place (the single reference is the location). Used in no-people mode instead of COMPOSE_DIRECTIVE.
export const ENV_COMPOSE_DIRECTIVE =
  "Re-render the place in the reference image as a deliberate cinematic film still of the described moment: " +
  "the same real location, materials, and perspective, re-lit and re-framed for the beat with a clear focal " +
  "subject (an object, a surface, a fall of light) and layered foreground/midground/background depth. Keep it " +
  "recognisably the same place; change only the light, weather, framing, and described elements." + NO_PEOPLE + ANATOMY_GUARD;

export const EDIT_PRESERVE =
  " Preserve the character's exact facial features, hairstyle, skin tone, eye colour, and body " +
  "proportions so their identity is unmistakably the same person; change ONLY what is described above " +
  "and leave the face and everything else unchanged.";

// Background-shot edit: a "change" within the SAME location (no character present). Keep the place itself
// identical so the shot reads as the same room a moment later — only the described element changes, and it
// must be INTEGRATED, not pasted. The engine melt-dissolves between the plate and its edits, so they must
// register as one continuous place.
export const BG_EDIT_PRESERVE =
  " Keep the location's architecture, layout, camera angle, framing, and overall composition exactly the " +
  "same — this is the same place a moment later. Introduce or alter ONLY the element described above, and " +
  "INTEGRATE it fully: match the existing light direction, colour, and grain, cast correct contact shadows, " +
  "and arrange it naturally as a believable, focal part of the space — not pasted on top, not scattered in a " +
  "uniform sprinkle. Leave everything else identical; do not re-stage or re-frame. No on-screen text or watermarks.";

// Scene-shot edit: a "change" within a composed scene that HAS a character in it. Preserve BOTH the place
// AND the person's identity, and integrate the change — used when an edit chains off a compose (or off an
// earlier edit that roots in one), so the protagonist stays himself as the scene develops around him.
export const SCENE_EDIT_PRESERVE =
  " This is the same scene a moment later. Keep the location's architecture, layout, camera angle, framing, " +
  "and composition identical, AND preserve the exact identity of any person present — their face, hair, skin " +
  "tone, build, and clothing must stay unmistakably the same. Change ONLY the element described above, " +
  "integrating it with matched light, colour, grain, and contact shadows so it reads as a real part of the " +
  "scene — not pasted on. Leave everything else identical. No on-screen text or watermarks.";

export const MOTION_PRESERVE =
  " Change only the described motion. Keep the character's face, hair, expression, and outfit, and the " +
  "entire background, identical to the source image — same framing, same everything else.";

// Reframe: render the SAME location from a different camera. This is the location-canon mechanism —
// instead of re-generating a place from text (which invents a new room every time), every later shot
// of a known location derives from its master plate, so architecture/furniture/materials stay canon.
export const REFRAME_PRESERVE =
  " Render this EXACT same location from the new viewpoint described above. This is the same real " +
  "place — keep its architecture, layout, furniture, objects, materials, and overall condition identical " +
  "and recognisable; only the framing, angle, and (if described) the moment's light change. Compose the new " +
  "view as a deliberate cinematic film still with a clear focal subject and layered depth. Do not add any " +
  "people, and never depict a camera or any filming equipment — the viewpoint is a position, not an object. " +
  "No on-screen text or watermarks.";

// ── Background-prompt hygiene (identity-leak lint) ─────────────────────────────
// A background/establish prompt must be the EMPTY stage. In practice the visual director leaks the
// protagonist in via pronouns ("a mattress beside HIM") and z-image then paints a random person —
// which collides with the compose step (verified failure: bg_s01_02). Pronouns are unambiguous
// leaks, so we scrub them mechanically; person-nouns are only warned (corpses/crowds can be legit).
const PRONOUNS = "he|him|his|she|her|hers|himself|herself";
/** Exported so the scene composer can strip inhabitant sentences out of canon location descriptions
 *  before they reach an empty-stage prompt (several canon `desc` fields mention a figure). */
export const PERSON_NOUN_RX = /\b(man|woman|person|figure|protagonist|boy|girl|someone|people|crowd|bodies|body)\b/i;

export interface PromptLint { prompt: string; scrubbed: boolean; warnings: string[] }

// Body-part words are ALWAYS a leak in a `framing` instruction (a camera position in an empty
// place has no limbs) — verified live: "tight on the upturned forearms" made z-image/Klein paint
// arms onto a plate. In a scene `prompt` they can be legit scenery (corpses), so strict mode is
// opt-in for framing fields only.
const BODY_PARTS = "arm|arms|forearm|forearms|hand|hands|fist|fists|finger|fingers|torso|shoulder|shoulders|face|leg|legs|foot|feet|knee|knees|chest|head";

/** Scrub named-character leakage out of an empty-stage prompt. Removes pronoun-bearing clauses.
 *  `strict` additionally treats body-part words as leaks (use for camera/framing instructions). */
export function lintBackgroundPrompt(prompt: string, castIds: string[] = [], opts: { strict?: boolean } = {}): PromptLint {
  const warnings: string[] = [];
  let p = (prompt || "").trim();
  let scrubbed = false;
  const terms = [PRONOUNS, ...(castIds.length ? [castIds.join("|")] : []), ...(opts.strict ? [BODY_PARTS] : [])];
  const leakSrc = `\\b(${terms.join("|")})\\b`;
  const dirty = (s: string) => new RegExp(leakSrc, "i").test(s);
  if (dirty(p)) {
    // drop the offending clauses (split on , ; —) rather than the whole prompt
    const parts = p.split(/(?<=[,;—])/);
    const kept = parts.filter(part => !dirty(part));
    const cleaned = kept.join("").replace(/\s{2,}/g, " ").replace(/[,;—\s]+$/g, "").trim();
    if (cleaned.length >= 24) { p = cleaned; }
    else { p = p.replace(new RegExp(leakSrc, "gi"), "").replace(/\s{2,}/g, " ").trim(); }
    scrubbed = true;
    warnings.push("background prompt contained a named-character/pronoun leak — scrubbed");
  }
  if (PERSON_NOUN_RX.test(p)) warnings.push(`background prompt mentions a person noun ("${p.match(PERSON_NOUN_RX)?.[0]}") — verify no unintended figure`);
  return { prompt: p, scrubbed, warnings };
}

/** Join a subject/scene prompt with a z-image style block. */
export const withStyle = (prompt: string, style: string) => `${(prompt || "").trim()}\n\n${style}`;
/** Append a Klein preservation/directive clause to an edit instruction. */
export const withClause = (instruction: string, clause: string) => `${(instruction || "").trim()}${clause}`;
