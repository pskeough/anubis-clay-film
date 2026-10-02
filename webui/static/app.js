// Clay Film Studio front end. Plain JS, no build step.
"use strict";

const S = {
  meta: null, pieces: [], name: null, st: null, plan: null, page: "input",
  saving: false, savePending: false, saveTimer: null, dirty: false,
  queue: null, seenDone: new Set(), sel: new Set(), msel: new Set(), cursor: {},
  proposals: {}, planProposal: null, modal: null,
};

// -- helpers -----------------------------------------------------------------------------------

const $ = (s, r = document) => r.querySelector(s);
const clone = (o) => JSON.parse(JSON.stringify(o));
const norm = (t) => (t || "").split(/\s+/).filter(Boolean).join(" ");
const words = (t) => norm(t) ? norm(t).split(" ").length : 0;
const sidOf = (i) => "s" + String(i).padStart(3, "0");

function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style") el.style.cssText = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "value") el.value = v;
    else if (k === "checked") el.checked = !!v;
    else if (k === "text") el.textContent = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

async function api(method, url, body) {
  const r = await fetch(url, {
    method, headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  let js = null;
  try { js = await r.json(); } catch (e) { /* empty */ }
  if (!r.ok) throw new Error((js && (js.error || js.detail)) ? JSON.stringify(js.error || js.detail).replace(/^"|"$/g, "") : `HTTP ${r.status}`);
  return js;
}

let toastTimer = null;
function toast(msg, bad) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = bad ? "bad" : "";
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, bad ? 9000 : 3500);
}

async function guard(fn) {
  try { return await fn(); } catch (e) { toast(e.message, true); return null; }
}

function getPath(o, path) {
  for (const k of path.split(".")) { if (o == null) return undefined; o = o[k]; }
  return o;
}

function setPath(o, path, val, optional) {
  const ks = path.split(".");
  for (let i = 0; i < ks.length - 1; i++) {
    if (o[ks[i]] == null || typeof o[ks[i]] !== "object") o[ks[i]] = /^\d+$/.test(ks[i + 1]) ? [] : {};
    o = o[ks[i]];
  }
  const last = ks[ks.length - 1];
  if (optional && (val === "" || val === null || val === undefined)) delete o[last];
  else o[last] = val;
}

const fileUrl = (rel, bust) => `/api/piece/${encodeURIComponent(S.name)}/file?path=${encodeURIComponent(rel)}&t=${encodeURIComponent(bust || "")}`;
const thumbUrl = (rel, bust, w) => `/api/piece/${encodeURIComponent(S.name)}/thumb?path=${encodeURIComponent(rel)}&w=${w || 448}&t=${encodeURIComponent(bust || "")}`;

function locOf(id) { return (S.meta.locations || []).find((l) => l.id === id); }
function castIds() {
  return [...new Set([...Object.keys(S.meta.cast || {}), ...Object.keys((S.plan && S.plan.cast) || {})])].sort();
}
function info(i) { return (S.st.shots || [])[i] || null; }
function prompt(i) { return (S.st.prompts || [])[i] || {}; }
function shotJobs(id, kinds) {
  return ((S.queue && S.queue.jobs) || []).filter((j) => j.piece === S.name && j.shot_id === id &&
    (j.state === "queued" || j.state === "running") && (!kinds || kinds.includes(j.kind)));
}
function activeItem(r, kind) {
  if (!r) return null;
  return (r[kind + "s"] || []).find((x) => x.v === r["active_" + kind]) || null;
}
function isPassage() { return S.st && S.st.source && S.st.source.mode === "passage"; }
function editable() { return S.st && S.st.imported && !S.st.protected; }

// -- boot, pieces, tabs --------------------------------------------------------------------------

async function boot() {
  S.meta = await api("GET", "/api/meta");
  $("#drybadge").hidden = !S.meta.dry;
  document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => showPage(b.dataset.page)));
  $("#piecesel").addEventListener("change", (e) => selectPiece(e.target.value));
  $("#stopfarm").addEventListener("click", () => guard(async () => toast((await api("POST", "/api/farm/stop")).message)));
  $("#clearjobs").addEventListener("click", () => S.name && guard(async () => {
    await api("POST", `/api/piece/${S.name}/jobs/clear`); pollQueue();
  }));
  document.addEventListener("keydown", onKey);
  const hash = new URLSearchParams(location.hash.slice(1));
  await loadPieces(hash.get("piece"));
  if (hash.get("page")) S.page = hash.get("page");
  showPage(S.page);
  pollQueue();
  setInterval(pollQueue, 1500);
}

async function loadPieces(want) {
  S.pieces = await api("GET", "/api/pieces");
  const sel = $("#piecesel");
  sel.replaceChildren(h("option", { value: "" }, "(choose a piece)"),
    ...S.pieces.map((p) => h("option", { value: p.name },
      `${p.name}${p.protected ? " (read only)" : p.imported ? "" : " (not imported)"}`)));
  const name = want || S.name;
  if (name && S.pieces.some((p) => p.name === name)) {
    sel.value = name;
    await selectPiece(name, true);
  }
}

async function selectPiece(name, quiet) {
  if (S.dirty) await saveNow();
  S.name = name || null;
  S.sel.clear(); S.msel.clear(); S.proposals = {}; S.planProposal = null;
  if (!S.name) { S.st = null; S.plan = null; render(); return; }
  await reload(true);
  setHash();
  if (!quiet) render();
}

async function reload(adoptPlan) {
  if (!S.name) return;
  const st = await api("GET", `/api/piece/${encodeURIComponent(S.name)}`);
  S.st = st;
  if (adoptPlan || !S.dirty) S.plan = st.plan ? clone(st.plan) : null;
  const p = S.pieces.find((x) => x.name === S.name) || {};
  $("#pieceinfo").textContent = st.plan
    ? `${st.plan.title || ""} | ${(st.plan.shots || []).length} shots | ${st.words} words, about ${st.est_seconds} s` +
      (st.protected ? " | read only" : st.imported ? "" : " | not imported")
    : "no plan yet";
  return p;
}

function setHash() {
  history.replaceState(null, "", `#piece=${encodeURIComponent(S.name || "")}&page=${S.page}`);
}

function showPage(page) {
  S.page = page;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.page === page));
  setHash();
  render();
}

function render() {
  const m = $("#main");
  m.replaceChildren();
  if (S.page === "input") return renderInput(m);
  if (!S.name) { m.append(h("p", { class: "muted" }, "Choose or create a piece on the Input page.")); return; }
  if (!S.plan) { m.append(h("p", { class: "muted" }, "This piece has no plan yet. Write one from the Input page.")); return; }
  if (!S.st.imported || S.st.protected) m.append(importNotice());
  if (S.page === "script") return renderScript(m);
  if (S.page === "stills") return renderStills(m);
  if (S.page === "motion") return renderMotion(m);
  if (S.page === "assemble") return renderAssemble(m);
}

function importNotice() {
  if (S.st.protected) {
    const nm = h("input", { value: S.name + "_copy", size: 24 });
    return h("div", { class: "warnbox row", style: "margin-bottom:12px" },
      `${S.name} is a finished piece and is read only here. Copy it to work on it:`, nm,
      h("button", { class: "primary", onclick: () => createPiece(nm.value, "copy", S.name) }, "Copy"));
  }
  return h("div", { class: "warnbox row", style: "margin-bottom:12px" },
    "This piece is not in the studio yet. Import gives every shot an id and adopts the stills and clips on disk as version 1.",
    h("button", { class: "primary", onclick: () => guard(async () => {
      await api("POST", `/api/piece/${S.name}/import`); await loadPieces(S.name); render(); toast("imported");
    }) }, "Import"));
}

async function createPiece(name, from, source, path) {
  await guard(async () => {
    await api("POST", "/api/pieces", { name, from, source, path });
    await loadPieces(name);
    render();
    toast(`created ${name}`);
  });
}

// -- page 1: input -------------------------------------------------------------------------------

function renderInput(m) {
  m.append(h("h2", {}, "Input"));
  // new piece
  const nm = h("input", { placeholder: "new_piece_name", size: 22 });
  const from = h("select", {}, h("option", { value: "blank" }, "blank (write a plan from a brief or passage)"),
    h("option", { value: "copy" }, "copy of an existing piece"), h("option", { value: "plan_file" }, "a plan.json file"));
  const src = h("select", {}, ...S.pieces.map((p) => h("option", { value: p.name }, p.name)));
  const pth = h("input", { placeholder: "C:\\...\\plan.json", size: 50 });
  const sync = () => { src.hidden = from.value !== "copy"; pth.hidden = from.value !== "plan_file"; };
  from.addEventListener("change", sync); sync();
  m.append(h("div", { class: "panel" }, h("h3", {}, "New piece"),
    h("div", { class: "row", style: "margin-top:6px" }, nm, from, src, pth,
      h("button", { class: "primary", onclick: () => createPiece(nm.value.trim(), from.value, src.value, pth.value.trim()) }, "Create"))));
  if (!S.name) { m.append(h("p", { class: "muted" }, "Or choose a piece in the sidebar.")); return; }
  if (!S.st.imported || S.st.protected) { if (S.plan) m.append(importNotice()); }

  // source
  const s = Object.assign({ mode: "brief", text: "", seconds: 30, shot_mode: "clay", whole: true }, S.st.source || {});
  const ro = S.st.protected;
  const mode = h("select", { disabled: ro }, h("option", { value: "brief" }, "brief (the model writes narration)"),
    h("option", { value: "passage" }, "passage (narration is the text, verbatim)"));
  mode.value = s.mode;
  const text = h("textarea", { rows: 12, disabled: ro, value: s.text || "" });
  const secs = h("input", { type: "number", min: 5, step: 1, value: s.seconds || 30, disabled: ro });
  const title = h("input", { value: s.title || "", size: 30, disabled: ro });
  const style = h("textarea", { rows: 2, disabled: ro, value: s.style || "",
    placeholder: "Per-piece style clause, used at the start of every free shot's prompt" });
  const loc = h("select", { disabled: ro }, h("option", { value: "" }, "(the model picks)"),
    ...S.meta.locations.map((l) => h("option", { value: l.id }, `${l.id}: ${l.name}`)));
  loc.value = s.location || "";
  const smode = h("select", { disabled: ro }, h("option", { value: "clay" }, "clay (Roko clay pass, then Qwen edit)"),
    h("option", { value: "free" }, "free (text to image, no location)"));
  smode.value = s.shot_mode || "clay";
  const whole = h("input", { type: "checkbox", checked: s.whole !== false, disabled: ro });
  const wc = h("span", { class: "muted small" });
  const upd = () => {
    const n = words(text.value);
    wc.textContent = `${n} words, about ${(n / S.meta.read_wps).toFixed(1)} s read aloud`;
  };
  text.addEventListener("input", upd); upd();
  const saveSource = () => guard(async () => {
    S.st = await api("PUT", `/api/piece/${S.name}/source`, {
      mode: mode.value, text: text.value, seconds: Number(secs.value) || 30, title: title.value, style: style.value,
      location: loc.value, shot_mode: smode.value, whole: whole.checked, file: s.file,
    });
    toast("source saved");
  });
  m.append(h("div", { class: "panel" }, h("h3", {}, "Brief or passage"),
    s.file ? h("div", { class: "muted small" }, `Read from the plan's source file ${s.file}. Saving writes source.json in the piece.`) : null,
    h("div", { class: "fields" },
      h("label", {}, "Kind", mode), h("label", {}, "Target seconds (brief, or passage with selection)", secs),
      h("label", {}, "Title", title), h("label", {}, "Location (clay shots)", loc),
      h("label", {}, "Shot mode", smode),
      h("label", {}, "Passage: narrate the whole text", h("span", {}, whole, " whole passage, verbatim (off = pick sentences to fit the time)"))),
    h("label", { class: "stack" }, "Style clause (free shots)", style),
    h("label", { class: "stack", style: "margin-top:6px" }, "Text", text), wc,
    h("div", { class: "row", style: "margin-top:8px" },
      h("button", { onclick: saveSource, disabled: ro }, "Save source"))));

  // planner endpoint
  const c = S.st.config || {};
  const ep = h("input", { value: c.endpoint || "", size: 40 });
  const model = h("input", { value: c.model || "", size: 30 });
  const key = h("input", { type: "password", size: 30, autocomplete: "off",
    placeholder: c.has_key ? `a key is set (${c.key_source}); type to replace` : "optional API key" });
  const saveCfg = (clear) => guard(async () => {
    const r = await api("PUT", "/api/config", { endpoint: ep.value, model: model.value, api_key: key.value, clear_key: !!clear });
    S.st.config = r; key.value = "";
    toast("planner settings saved; the key is stored in webui/config.local.json and never sent back");
    render();
  });
  const writePlan = () => guard(async () => {
    if (S.plan && !confirm("Replace the current plan with a new one from the model? The old plan is kept in _webui/plan_backups.")) return;
    await saveSource();
    await api("POST", `/api/piece/${S.name}/jobs`, { kind: "plan", endpoint: ep.value, model: model.value });
    toast("plan job queued (CPU lane); watch the queue");
    pollQueue();
  });
  m.append(h("div", { class: "panel" }, h("h3", {}, "Planner model (OpenAI-compatible)"),
    h("div", { class: "fields" }, h("label", { class: "wide" }, "Endpoint URL", ep), h("label", {}, "Model", model),
      h("label", {}, "API key", key)),
    h("div", { class: "row" }, h("button", { onclick: () => saveCfg(false) }, "Save settings"),
      c.has_key && c.key_source === "config" ? h("button", { class: "danger", onclick: () => saveCfg(true) }, "Forget saved key") : null,
      h("button", { class: "primary", disabled: ro, onclick: writePlan }, S.plan ? "Write a new plan with the model" : "Write the plan with the model"))));
}

// -- page 2: script and shots ------------------------------------------------------------------

function scheduleSave(delay) {
  S.dirty = true;
  clearTimeout(S.saveTimer);
  S.saveTimer = setTimeout(saveNow, delay === undefined ? 700 : delay);
}

async function saveNow() {
  clearTimeout(S.saveTimer);
  if (!S.dirty || !S.name || !editable()) return;
  if (S.saving) { S.savePending = true; return; }
  S.saving = true; S.dirty = false;
  try {
    const r = await api("PUT", `/api/piece/${S.name}/plan`, { plan: S.plan });
    // adopt ids the server gave new shots; keep everything typed since
    (r.plan.shots || []).forEach((s, i) => { if (S.plan.shots[i] && !S.plan.shots[i].id) S.plan.shots[i].id = s.id; });
    const keep = S.plan;
    S.st = r;
    S.plan = keep;
    updateScriptStatus();
  } catch (e) {
    toast("save failed: " + e.message, true);
    S.dirty = true;
  } finally {
    S.saving = false;
    if (S.savePending) { S.savePending = false; saveNow(); }
  }
}

function structural() { scheduleSave(0); renderKeepScroll(); }

function renderKeepScroll() {
  const m = $("#main");
  const y = m.scrollTop;
  render();
  m.scrollTop = y;
}

function errsFor(i) {
  const pre = `shots[${i}]`;
  return (S.st.errors || []).filter((e) => e.startsWith(pre + ".") || e.startsWith(pre + ":") || e.startsWith(pre + " "));
}

function updateScriptStatus() {
  if (S.page !== "script") return;
  const planErrs = (S.st.errors || []).filter((e) => !e.startsWith("shots["));
  const pe = $("#planerrs");
  if (pe) pe.replaceChildren(...planErrs.map((e) => h("li", {}, e)));
  (S.plan.shots || []).forEach((s, i) => {
    const ul = document.getElementById("errs-" + i);
    if (ul) ul.replaceChildren(...errsFor(i).map((e) => h("li", {}, e)));
    const card = document.getElementById("shot-" + i);
    if (card) card.classList.toggle("bad", errsFor(i).length > 0);
    const wc = document.getElementById("wc-" + i);
    if (wc) wc.textContent = wcText(s);
  });
  const cov = $("#coverage");
  if (cov) cov.replaceChildren(...coverageView());
  const sum = $("#plansum");
  if (sum) sum.replaceChildren(...summaryView());
}

function wcText(s) {
  const n = words(s.narration);
  return `${n} words, ${(n / S.meta.read_wps + Number(s.tail_s || 0)).toFixed(1)} s`;
}

function summaryView() {
  const st = S.st;
  const blocking = (st.blocking || []).length;
  return [
    h("span", {}, `${(S.plan.shots || []).length} shots, ${st.words} words, about ${st.est_seconds} s. `),
    blocking ? h("span", { class: "badge bad" }, `${blocking} problem(s) stop compiling`) : h("span", { class: "badge ok" }, "compiles"),
    " ",
    (st.errors || []).length > blocking ? h("span", { class: "badge warn" }, `${(st.errors || []).length - blocking} narration warning(s)`) : null,
    st.coverage ? h("span", { class: "badge " + (st.coverage.exact ? "ok" : "warn"), style: "margin-left:4px" },
      st.coverage.exact ? "narration equals the source" : "narration differs from the source") : null,
  ];
}

const HUES = [205, 150, 35, 280, 95, 330, 180, 10];

function coverageView() {
  const cov = S.st.coverage;
  if (!cov) return [h("div", { class: "muted small" }, "Coverage shows in passage mode (Input page).")];
  const src = norm(S.st.source.text);
  const n = src.length || 1;
  const segs = [];
  let pos = 0;
  const spans = cov.spans.filter((s) => s.found).sort((a, b) => a.start - b.start);
  for (const sp of spans) {
    if (sp.start > pos) segs.push(h("div", { class: src.slice(pos, sp.start).trim() ? "gapseg" : "", style: `flex:${sp.start - pos}`,
      title: src.slice(pos, sp.start).trim() ? "not narrated" : "" }));
    const b = Math.max(sp.end, pos);
    if (b > Math.max(sp.start, pos)) {
      segs.push(h("div", {
        style: `flex:${b - Math.max(sp.start, pos)};background:hsl(${HUES[sp.i % HUES.length]} 45% ${sp.overlap ? 35 : 45}%)` +
          (sp.overlap ? ";outline:2px solid var(--overlap)" : ""),
        title: `${sidOf(sp.i)}`, onclick: () => scrollToShot(sp.i),
      }));
    }
    pos = Math.max(pos, sp.end);
  }
  if (pos < n && src.slice(pos).trim()) segs.push(h("div", { class: "gapseg", style: `flex:${n - pos}`, title: "not narrated" }));
  // the text with gaps marked
  const txt = [];
  let p = 0;
  const marks = [...cov.gaps.map(([a, b]) => ({ a, b, c: "g" })), ...cov.overlaps.map(([a, b]) => ({ a, b, c: "o" }))].sort((x, y) => x.a - y.a);
  for (const mk of marks) {
    if (mk.a < p) continue;
    txt.push(src.slice(p, mk.a));
    txt.push(h("span", { class: mk.c, title: mk.c === "g" ? "not narrated" : "narrated twice" }, src.slice(mk.a, mk.b)));
    p = mk.b;
  }
  txt.push(src.slice(p));
  const missing = cov.spans.filter((s) => !s.found).map((s) => sidOf(s.i));
  return [
    h("div", { class: "cov" }, segs),
    h("div", { class: "small muted", style: "margin-top:3px" },
      `${cov.gaps.length} gap(s), ${cov.overlaps.length} overlap(s)` + (missing.length ? `; not found in the source: ${missing.join(", ")}` : "")),
    h("details", {}, h("summary", { class: "small" }, "source text (red = not narrated, orange = narrated twice)"),
      h("div", { class: "covtext" }, txt)),
  ];
}

function scrollToShot(i) {
  const el = document.getElementById("shot-" + i);
  if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
}

function fieldEl(path, kind, opts) {
  opts = opts || {};
  const ro = !editable();
  let el;
  const cur = getPath(S.plan, path);
  if (kind === "select") {
    const choices = (opts.choices || []).map((c) => (Array.isArray(c) ? c : [c, c]));
    el = h("select", { disabled: ro },
      opts.blank !== undefined ? h("option", { value: "" }, opts.blank) : null,
      ...choices.map(([v, l]) => h("option", { value: String(v) }, String(l))));
    if (cur !== undefined && cur !== null && !choices.some(([v]) => String(v) === String(cur))) {
      el.append(h("option", { value: String(cur) }, `${cur} (not valid)`));
    }
    el.value = cur === undefined || cur === null ? "" : String(cur);
  } else if (kind === "area") {
    el = h("textarea", { rows: opts.rows || 2, disabled: ro, placeholder: opts.placeholder || "" });
    el.value = cur || "";
  } else {
    el = h("input", { type: kind === "num" ? "number" : "text", step: opts.step || "any", disabled: ro,
      placeholder: opts.placeholder || "" });
    el.value = cur === undefined || cur === null ? "" : cur;
  }
  const ev = kind === "select" ? "change" : "input";
  el.addEventListener(ev, () => {
    let v = el.value;
    if (opts.type === "int" && v !== "") v = parseInt(v, 10);
    if ((opts.type === "num" || kind === "num") && v !== "") v = Number(v);
    setPath(S.plan, path, v, opts.optional);
    if (opts.onchange) opts.onchange(v);
    if (opts.structural) structural(); else scheduleSave();
  });
  if (opts.label === false) return el;
  return h("label", { class: opts.wide ? "wide" : "" }, opts.label || path.split(".").pop(), el);
}

function renderScript(m) {
  const plan = S.plan;
  const E = S.meta.enums;
  m.append(h("h2", {}, "Script and shots"));
  // plan-level
  const castArea = h("textarea", { rows: 3, disabled: !editable() });
  castArea.value = JSON.stringify(plan.cast || {}, null, 1);
  castArea.addEventListener("change", () => {
    try { plan.cast = JSON.parse(castArea.value || "{}"); castArea.style.borderColor = ""; structural(); }
    catch (e) { castArea.style.borderColor = "var(--bad)"; toast("cast is not valid JSON: " + e.message, true); }
  });
  m.append(h("div", { class: "panel" },
    h("div", { class: "fields" },
      fieldEl("title", "text", { label: "Title" }),
      fieldEl("location", "select", { label: "Location (clay shots)", blank: "(none: free shots only)", optional: true,
        choices: S.meta.locations.map((l) => [l.id, l.id]), structural: true }),
      fieldEl("grade", "select", { label: "Grade", blank: "(ash)", optional: true, choices: E.grade }),
      fieldEl("seed", "num", { label: "Seed", type: "int", optional: true }),
      fieldEl("style", "area", { label: "Style clause (free shots)", wide: true, optional: true }),
      h("label", { class: "wide" }, "Extra cast (JSON id to appearance; bible cast: " + Object.keys(S.meta.cast).join(", ") + ")", castArea)),
    h("div", { id: "plansum", class: "row small" }, summaryView()),
    h("ul", { id: "planerrs", class: "errs" }, (S.st.errors || []).filter((e) => !e.startsWith("shots[")).map((e) => h("li", {}, e))),
    h("div", { id: "coverage", style: "margin-top:8px" }, coverageView()),
    planChat()));
  (plan.shots || []).forEach((s, i) => m.append(shotCard(s, i)));
  if (editable()) {
    m.append(h("button", { onclick: () => { plan.shots.push(blankShot()); structural(); } }, "Add a shot at the end"));
  }
}

function blankShot(mode) {
  if (mode === "free") return { mode: "free", narration: "", subject: "", scene: "", action: "", motion: "locked off", people: [] };
  return { narration: "", camera: { angle: "frontal", elevation: "eye", distance: "medium", lens: 35 }, people: [],
    subject: "", scene: "", action: "", motion: "a slow push in" };
}

function planChat() {
  if (!editable()) return null;
  const inp = h("input", { placeholder: "Ask for a change to the whole plan, e.g. 'make every shot a little wider'" });
  const btn = h("button", {}, "Ask");
  const box = h("div", {});
  const ask = () => guard(async () => {
    if (!inp.value.trim()) return;
    btn.disabled = true; btn.textContent = "asking...";
    try {
      await saveNow();
      const r = await api("POST", `/api/piece/${S.name}/chat/plan`, { request: inp.value, plan: S.plan });
      S.planProposal = r;
      box.replaceChildren(planProposalView(box));
    } finally { btn.disabled = false; btn.textContent = "Ask"; }
  });
  btn.addEventListener("click", ask);
  inp.addEventListener("keydown", (e) => { if (e.key === "Enter") ask(); });
  if (S.planProposal) box.append(planProposalView(box));
  return h("div", {}, h("div", { class: "chat" }, inp, btn), box);
}

function planProposalView(box) {
  const p = S.planProposal;
  const old = S.plan, neu = p.plan;
  const rows = [];
  for (const k of new Set([...Object.keys(old), ...Object.keys(neu)])) {
    if (k === "shots") continue;
    if (JSON.stringify(old[k]) !== JSON.stringify(neu[k])) rows.push(diffRow(k, old[k], neu[k]));
  }
  const n = Math.max(old.shots.length, (neu.shots || []).length);
  for (let i = 0; i < n; i++) {
    const a = old.shots[i], b = (neu.shots || [])[i];
    if (JSON.stringify(a) === JSON.stringify(b)) continue;
    if (!a || !b) { rows.push(diffRow(sidOf(i), a ? "(shot)" : "", b ? "(new shot) " + (b.narration || "") : "(removed)")); continue; }
    for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
      if (JSON.stringify(a[k]) !== JSON.stringify(b[k])) rows.push(diffRow(`${sidOf(i)}.${k}`, a[k], b[k]));
    }
  }
  return h("div", { class: "proposal" },
    h("div", { class: "small" }, `Proposed plan: ${old.shots.length} to ${(neu.shots || []).length} shots, ${rows.length} change(s).`),
    p.errors.length ? h("ul", { class: "errs" }, p.errors.slice(0, 12).map((e) => h("li", {}, e))) : null,
    h("table", { class: "diff" }, rows),
    h("div", { class: "row", style: "margin-top:6px" },
      h("button", { class: "primary", onclick: () => { S.plan = clone(neu); S.planProposal = null; structural(); } }, "Accept"),
      h("button", { onclick: () => { S.planProposal = null; box.replaceChildren(); } }, "Reject")));
}

function show(v) {
  if (v === undefined) return "";
  return typeof v === "object" ? JSON.stringify(v, null, 1) : String(v);
}

function diffRow(k, a, b) {
  return h("tr", {}, h("td", { class: "muted" }, k), h("td", { class: "old" }, show(a)), h("td", { class: "new" }, show(b)));
}

function shotCard(s, i) {
  const E = S.meta.enums;
  const free = (s.mode || "clay") === "free";
  const base = `shots.${i}`;
  const r = info(i) || {};
  const pr = prompt(i);
  const errs = errsFor(i);
  const ed = editable();
  const plan = S.plan;
  const loc = locOf(s.location || plan.location);
  const narr = fieldEl(base + ".narration", "area", { label: false, rows: 2 });
  const trackCursor = () => { S.cursor[i] = narr.selectionStart; };
  ["keyup", "click", "select", "input"].forEach((ev) => narr.addEventListener(ev, trackCursor));

  const btn = (label, fn, cls) => h("button", { class: "small " + (cls || ""), disabled: !ed, onclick: fn }, label);
  const move = (d) => { const j = i + d; if (j < 0 || j >= plan.shots.length) return;
    [plan.shots[i], plan.shots[j]] = [plan.shots[j], plan.shots[i]]; structural(); };
  const split = () => {
    const t = s.narration || "";
    let at = S.cursor[i];
    if (at === undefined || at <= 0 || at >= t.length) {
      const w = t.split(/\s+/); at = w.slice(0, Math.ceil(w.length / 2)).join(" ").length;
    }
    const a = t.slice(0, at).trim(), b = t.slice(at).trim();
    if (!a || !b) { toast("put the cursor inside the narration where the split goes", true); return; }
    const twin = clone(s); delete twin.id;
    s.narration = a; twin.narration = b;
    plan.shots.splice(i + 1, 0, twin); structural();
  };
  const merge = () => {
    const nx = plan.shots[i + 1]; if (!nx) return;
    s.narration = norm((s.narration || "") + " " + (nx.narration || ""));
    plan.shots.splice(i + 1, 1); structural();
  };
  const del = () => { if (confirm(`Delete ${sidOf(i)}? Its stills and clips stay on disk under versions/.`)) { plan.shots.splice(i, 1); structural(); } };
  const setMode = (v) => {
    if (v === "free") { s.mode = "free"; }
    else { delete s.mode; if (!s.camera) s.camera = blankShot().camera; (s.people || []).forEach((p) => { p.at = p.at || (loc && loc.marks[0]); p.facing = p.facing || "near"; p.pose = p.pose || "standing"; p.action = p.action || ""; p.placement = p.placement || ""; }); }
    structural();
  };
  const modeSel = h("select", { disabled: !ed }, h("option", { value: "clay" }, "clay"), h("option", { value: "free" }, "free"));
  modeSel.value = free ? "free" : "clay";
  modeSel.addEventListener("change", () => setMode(modeSel.value));

  const badges = [];
  if (r.active_still) badges.push(h("span", { class: "badge" + (r.still_stale ? " warn" : "") }, `still v${r.active_still}${r.still_stale ? " stale" : ""}`));
  if (r.active_clip) badges.push(h("span", { class: "badge" + ((r.clip_stale || []).length ? " warn" : "") }, `clip v${r.active_clip}${(r.clip_stale || []).length ? " stale" : ""}`));
  if (r.clay_stale) badges.push(h("span", { class: "badge warn" }, "clay pass stale"));
  for (const j of shotJobs(s.id)) badges.push(h("span", { class: "badge run" }, `${j.kind} ${j.state}`));

  const head = h("div", { class: "head" },
    h("span", { class: "sid" }, sidOf(i)), h("span", { class: "muted small" }, s.id || "(new)"), modeSel,
    h("span", { id: "wc-" + i, class: "small muted" }, wcText(s)), ...badges,
    h("span", { class: "grow" }),
    btn("Up", () => move(-1)), btn("Down", () => move(1)), btn("Split at cursor", split),
    btn("Merge with next", merge), btn("Add after", () => { plan.shots.splice(i + 1, 0, blankShot(s.mode)); structural(); }),
    btn("Delete", del, "danger"));

  const fields = [];
  if (!free) {
    fields.push(fieldEl(base + ".location", "select", { label: "Location", blank: `(plan: ${plan.location || "none"})`, optional: true,
      choices: S.meta.locations.map((l) => l.id), structural: true }));
  }
  const camOpt = free;   // free shots: camera words are optional
  fields.push(
    fieldEl(base + ".camera.distance", "select", { label: "Shot size", choices: E.distance, blank: camOpt ? "(any)" : undefined, optional: camOpt }),
    fieldEl(base + ".camera.angle", "select", { label: "Angle", choices: E.angle, blank: camOpt ? "(any)" : undefined, optional: camOpt }),
    fieldEl(base + ".camera.elevation", "select", { label: "Elevation", choices: E.elevation, blank: camOpt ? "(any)" : undefined, optional: camOpt }),
    fieldEl(base + ".camera.lens", "select", { label: "Lens mm", choices: E.lens, type: "int", blank: camOpt ? "(any)" : undefined, optional: camOpt }));
  if (!free) {
    fields.push(fieldEl(base + ".camera.frame", "select", { label: "Framing", choices: E.frame, blank: "(solver)", optional: true }),
      fieldEl(base + ".look_at", "select", { label: "Look at (empty shots)", blank: "(none)", optional: true,
        choices: loc ? Object.entries(loc.props).map(([k, v]) => [k, `${k}: ${v}`]) : [] }));
  }
  fields.push(fieldEl(base + ".subject_class", "select", { label: "Subject class", blank: "(auto)", optional: true, choices: E.subject_class }),
    fieldEl(base + ".tail_s", "num", { label: "Hold after narration s", optional: true }),
    fieldEl(base + ".subject", "text", { label: "Subject" }),
    fieldEl(base + ".action", "text", { label: "Action" }),
    fieldEl(base + ".motion", "text", { label: "Camera motion" }),
    fieldEl(base + ".move_target", "text", { label: "Move target", optional: true }),
    fieldEl(base + ".scene", "area", { label: "Scene", wide: true }));

  // people
  const ppl = s.people || [];
  const cast = castIds();
  const prow = (p, j) => {
    const pb = `${base}.people.${j}`;
    const cells = [h("td", {}, fieldEl(pb + ".who", "select", { label: false, choices: cast }))];
    if (!free) {
      cells.push(h("td", {}, fieldEl(pb + ".at", "select", { label: false, choices: loc ? loc.marks : [] })),
        h("td", {}, fieldEl(pb + ".facing", "select", { label: false, choices: E.facing })),
        h("td", {}, fieldEl(pb + ".pose", "select", { label: false, choices: E.pose })));
    }
    cells.push(h("td", {}, fieldEl(pb + ".action", "text", { label: false, placeholder: "action", optional: free })),
      h("td", {}, fieldEl(pb + ".placement", "text", { label: false, placeholder: "placement in frame", optional: free })),
      h("td", {}, h("button", { class: "small danger", disabled: !ed, onclick: () => { ppl.splice(j, 1); structural(); } }, "remove")));
    return h("tr", {}, cells);
  };
  const addPerson = () => {
    s.people = s.people || [];
    s.people.push(free ? { who: cast[0] } : { who: cast[0], at: loc ? loc.marks[0] : "mid-centre", facing: "near", pose: "standing", action: "", placement: "" });
    structural();
  };
  const peopleEl = h("div", {},
    h("div", { class: "small muted" }, free ? "People (appearance only: who, and optionally what they do)" : "People (who, mark, facing, pose, action, placement)"),
    ppl.length ? h("table", { class: "people" }, ppl.map(prow)) : h("div", { class: "small muted" }, "nobody in this shot"),
    h("button", { class: "small", disabled: !ed || ppl.length >= 3, onclick: addPerson }, "Add person"));

  const stillPh = (pr.still_auto || (free ? "" : "the clay pass writes this prompt; render the clay pass to see it")).slice(0, 600);
  const motionPh = (pr.motion_auto || "").slice(0, 600);
  const overrides = h("details", { open: !!(s.still_prompt || s.motion_prompt) || free },
    h("summary", { class: "small" }, "Prompts (empty = computed)"),
    h("div", { class: "fields" },
      fieldEl(base + ".still_prompt", "area", { label: free ? "Still prompt (what to draw; the scene is used if empty)" : "Still prompt override",
        wide: true, optional: true, rows: 3, placeholder: stillPh }),
      fieldEl(base + ".motion_prompt", "area", { label: "H3 motion prompt override", wide: true, optional: true, rows: 3, placeholder: motionPh })));

  const card = h("div", { id: "shot-" + i, class: `shot ${free ? "free" : "clay"}${errs.length ? " bad" : ""}` },
    head, h("label", { class: "stack" }, "Narration", narr),
    h("div", { class: "fields" }, fields), peopleEl, overrides,
    h("ul", { id: "errs-" + i, class: "errs" }, errs.map((e) => h("li", {}, e))),
    shotChat(i));
  return card;
}

function shotChat(i) {
  if (!editable()) return null;
  const inp = h("input", { placeholder: "Ask for a change to this shot, e.g. 'low angle, he looks at the lake'" });
  const btn = h("button", { class: "small" }, "Ask");
  const box = h("div", {});
  const ask = () => guard(async () => {
    if (!inp.value.trim()) return;
    btn.disabled = true; btn.textContent = "asking...";
    try {
      await saveNow();
      const r = await api("POST", `/api/piece/${S.name}/chat/shot`, { index: i, request: inp.value, plan: S.plan });
      S.proposals[i] = r;
      box.replaceChildren(shotProposalView(i, box));
    } finally { btn.disabled = false; btn.textContent = "Ask"; }
  });
  btn.addEventListener("click", ask);
  inp.addEventListener("keydown", (e) => { if (e.key === "Enter") ask(); });
  if (S.proposals[i]) box.append(shotProposalView(i, box));
  return h("div", {}, h("div", { class: "chat" }, inp, btn), box);
}

function shotProposalView(i, box) {
  const p = S.proposals[i];
  const a = S.plan.shots[i], b = p.shot;
  const rows = [];
  for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
    if (JSON.stringify(a[k]) !== JSON.stringify(b[k])) rows.push(diffRow(k, a[k], b[k]));
  }
  return h("div", { class: "proposal" },
    h("div", { class: "small" }, rows.length ? `Proposed change to ${sidOf(i)}:` : "The model proposed no change."),
    p.errors.length ? h("ul", { class: "errs" }, p.errors.map((e) => h("li", {}, e))) : null,
    h("table", { class: "diff" }, rows),
    h("div", { class: "row", style: "margin-top:6px" },
      h("button", { class: "primary small", onclick: () => { S.plan.shots[i] = clone(b); delete S.proposals[i]; structural(); } }, "Accept"),
      h("button", { class: "small", onclick: () => { delete S.proposals[i]; box.replaceChildren(); } }, "Reject")));
}

// -- page 3: stills ------------------------------------------------------------------------------

async function enqueue(body, what) {
  return guard(async () => {
    const r = await api("POST", `/api/piece/${S.name}/jobs`, body);
    toast(`${r.length} ${what || body.kind} job(s) queued`);
    pollQueue();
    return r;
  });
}

function seedValue(inp) {
  const v = (inp.value || "").trim();
  if (!v) return undefined;
  if (v.toLowerCase() === "r") return Math.floor(Math.random() * 2 ** 31);
  return parseInt(v, 10);
}

function renderStills(m) {
  const shots = S.plan.shots || [];
  const ed = editable();
  m.append(h("h2", {}, "Stills"));
  const seed = h("input", { size: 10, placeholder: "seed (blank = plan, r = random)" });
  const count = h("span", { class: "small muted" }, `${S.sel.size} selected`);
  const selWhere = (fn) => { S.sel = new Set(shots.map((s, i) => fn(s, i) ? s.id : null).filter(Boolean)); renderKeepScroll(); };
  m.append(h("div", { class: "panel row" },
    h("button", { class: "small", onclick: () => selWhere(() => true) }, "Select all"),
    h("button", { class: "small", onclick: () => selWhere(() => false) }, "Select none"),
    h("button", { class: "small", onclick: () => selWhere((s, i) => !(info(i) || {}).active_still || (info(i) || {}).still_stale) }, "Select missing or stale"),
    count, h("span", { class: "grow" }), seed,
    h("button", { class: "primary", disabled: !ed, onclick: () => {
      if (!S.sel.size) return toast("select some shots first", true);
      const sd = seedValue(seed);
      enqueue({ kind: "still", shots: [...S.sel], seed: sd }, "still");
    } }, "Render selected"),
    h("button", { disabled: !ed, onclick: () => {
      const ids = [...S.sel].filter((id) => { const s = shots.find((x) => x.id === id); return s && s.mode !== "free"; });
      if (!ids.length) return toast("select some clay shots first", true);
      enqueue({ kind: "clay", shots: ids }, "clay");
    } }, "Clay pass for selected")));
  const tiles = h("div", { class: "tiles" });
  shots.forEach((s, i) => tiles.append(stillTile(s, i, count)));
  m.append(tiles);
}

function stillTile(s, i, count) {
  const r = info(i) || {};
  const st = activeItem(r, "still");
  const clay = r.clay;
  let img;
  if (st) img = h("img", { src: thumbUrl(st.file, st.at + st.v), loading: "lazy", alt: sidOf(i) });
  else if (clay) img = h("img", { src: thumbUrl(clay.file, clay.at), loading: "lazy", alt: "clay pass", style: "opacity:.55" });
  else img = h("span", { class: "muted small" }, s.mode === "free" ? "no still yet (free shot)" : "no clay pass or still yet");
  const over = h("div", { class: "over" });
  if (!st && clay) over.append(h("span", { class: "badge" }, "clay pass only"));
  if (r.still_stale) over.append(h("span", { class: "badge warn" }, "stale: plan changed"));
  if (r.clay_stale) over.append(h("span", { class: "badge warn" }, "clay stale"));
  for (const j of shotJobs(s.id, ["still", "edit", "clay"])) over.append(h("span", { class: "badge run" }, `${j.kind} ${j.state}`));
  const cb = h("input", { type: "checkbox", checked: S.sel.has(s.id) });
  const tile = h("div", { class: "tile" + (S.sel.has(s.id) ? " sel" : "") },
    h("div", { class: "img", onclick: () => openStill(i) }, img, over),
    h("div", { class: "meta" }, cb, h("b", {}, sidOf(i)), h("span", { class: "badge" }, s.mode || "clay"),
      st ? h("span", { class: "muted" }, `v${st.v} of ${r.stills.length}`) : null,
      st && st.kind ? h("span", { class: "muted" }, st.kind) : null),
    h("div", { class: "narr" }, s.narration || ""));
  cb.addEventListener("change", () => {
    if (cb.checked) S.sel.add(s.id); else S.sel.delete(s.id);
    tile.classList.toggle("sel", cb.checked);
    count.textContent = `${S.sel.size} selected`;
  });
  return tile;
}

function closeModal() { $("#modal").hidden = true; $("#modal").replaceChildren(); S.modal = null; }

function onKey(e) {
  if (!S.modal) return;
  const tag = (document.activeElement || {}).tagName;
  if (e.key === "Escape") closeModal();
  if (tag === "INPUT" || tag === "TEXTAREA") return;
  if (S.modal.kind === "still" && e.key === "ArrowRight") openStill(Math.min(S.modal.i + 1, S.plan.shots.length - 1));
  if (S.modal.kind === "still" && e.key === "ArrowLeft") openStill(Math.max(S.modal.i - 1, 0));
}

function openStill(i, viewV) {
  const s = S.plan.shots[i];
  const r = info(i) || { stills: [] };
  const act = activeItem(r, "still");
  const vv = viewV || (S.modal && S.modal.kind === "still" && S.modal.i === i ? S.modal.v : null) || (act && act.v) || null;
  const view = (r.stills || []).find((x) => x.v === vv) || act;
  S.modal = { kind: "still", i, v: view ? view.v : null, active: r.active_still };
  const pr = prompt(i);
  const ed = editable();
  const free = s.mode === "free";

  const big = view ? h("img", { src: fileUrl(view.file, view.at + view.v) }) : h("div", { class: "noimg" }, "no still yet");
  const right = free
    ? h("div", {}, h("div", { class: "small muted" }, "Free shot: text to image from this prompt"), h("pre", { class: "prompt" }, pr.still_auto || ""))
    : (r.clay ? h("div", {}, h("div", { class: "small muted" }, `Clay pass ${r.clay_stale ? "(stale: geometry changed; a redo renders it again)" : ""}`),
      h("img", { src: fileUrl(r.clay.file, r.clay.at) }))
      : h("div", { class: "noimg" }, "no clay pass yet; a redo renders one"));

  const strip = h("div", { class: "strip" }, (r.stills || []).map((x) => h("div", {
    class: "v" + (view && x.v === view.v ? " view" : ""), onclick: () => openStill(i, x.v), title: `${x.kind || ""} seed ${x.seed ?? ""}`,
  }, h("img", { src: thumbUrl(x.file, x.at + x.v, 300), loading: "lazy" }),
    h("span", { class: "lab badge" + (x.v === r.active_still ? " ok" : "") }, `v${x.v}${x.v === r.active_still ? " active" : ""}`))));

  const detail = view ? h("div", { class: "small" },
    h("div", { class: "row" }, h("b", {}, `v${view.v}`), h("span", { class: "muted" }, `${view.kind || ""}, seed ${view.seed ?? "?"}, ${view.at || ""}${view.backend === "dry" ? ", DRY" : ""}`),
      view.instruction ? h("span", {}, `edit of v${view.parent}: "${view.instruction}"`) : null,
      view.v !== r.active_still && ed ? h("button", { class: "primary small", onclick: () => activate(s.id, "still", view.v, () => openStill(i, view.v)) }, "Make active") : null,
      view.v === r.active_still ? h("span", { class: "badge ok" }, "active") : null),
    view.prompt ? h("details", {}, h("summary", {}, "prompt used"), h("pre", { class: "prompt" }, view.prompt)) : null) : null;

  const ins = h("input", { placeholder: "e.g. make his coat dark green; add rain", style: "width:100%" });
  const eseed = h("input", { size: 12, placeholder: "seed (blank = plan)" });
  const doEdit = () => {
    if (!ins.value.trim()) return toast("type what to change", true);
    if (!view) return toast("no still to edit", true);
    enqueue({ kind: "edit", shots: [s.id], prompt: ins.value.trim(), seed: seedValue(eseed), base_v: view.v }, "edit");
  };
  ins.addEventListener("keydown", (e) => { if (e.key === "Enter") doEdit(); });
  const defaultPrompt = s.still_prompt || pr.still_auto || "";
  const rp = h("textarea", { rows: 4 }); rp.value = defaultPrompt;
  const rseed = h("input", { size: 12, placeholder: "seed" });
  rseed.value = String(Math.floor(Math.random() * 2 ** 31));
  const doRedo = () => {
    const body = { kind: "still", shots: [s.id], seed: seedValue(rseed) };
    if (norm(rp.value) && norm(rp.value) !== norm(defaultPrompt)) body.prompt = rp.value;
    enqueue(body, "redo");
  };
  const jobs = shotJobs(s.id, ["still", "edit", "clay"]);

  const box = h("div", { class: "mbox" },
    h("div", { class: "row" }, h("h3", {}, `${sidOf(i)} ${s.mode || "clay"}`), h("span", { class: "muted small grow" }, s.narration || ""),
      ...jobs.map((j) => h("span", { class: "badge run" }, `${j.kind} ${j.state}`)),
      r.still_stale ? h("span", { class: "badge warn" }, "active still is stale: the plan changed after it") : null,
      h("button", { onclick: () => openStill(Math.max(i - 1, 0)) }, "Prev"),
      h("button", { onclick: () => openStill(Math.min(i + 1, S.plan.shots.length - 1)) }, "Next"),
      h("button", { onclick: closeModal }, "Close")),
    h("div", { class: "mpair", style: "margin-top:8px" }, big, right),
    strip, detail,
    ed ? h("div", { class: "bars" },
      h("div", { class: "panel" }, h("h3", {}, "Edit this version (Qwen-2.1 edit)"),
        h("div", { class: "small muted" }, "Say what to change. The rest of the picture is kept."),
        h("div", { style: "margin:6px 0" }, ins),
        h("div", { class: "row" }, eseed, h("button", { class: "primary", onclick: doEdit }, `Edit v${view ? view.v : "?"}`))),
      h("div", { class: "panel" }, h("h3", {}, free ? "Redo from text" : "Redo from the clay pass"),
        h("div", { class: "small muted" }, "A new seed and, if you change it, a new prompt for this render only."),
        h("div", { style: "margin:6px 0" }, rp),
        h("div", { class: "row" }, rseed,
          h("button", { class: "small", onclick: () => { rseed.value = String(Math.floor(Math.random() * 2 ** 31)); } }, "new seed"),
          h("button", { class: "small", onclick: () => { rp.value = defaultPrompt; } }, "reset prompt"),
          h("button", { class: "primary", onclick: doRedo }, "Redo")))) : null);
  const md = $("#modal");
  md.replaceChildren(box);
  md.hidden = false;
  md.onclick = (e) => { if (e.target === md) closeModal(); };
}

async function activate(id, kind, v, after) {
  await guard(async () => {
    S.st = await api("POST", `/api/piece/${S.name}/activate`, { id, kind, v });
    toast(`${kind} v${v} is active`);
    if (after) after(); else renderKeepScroll();
  });
}

// -- page 4: motion ------------------------------------------------------------------------------

function renderMotion(m) {
  const shots = S.plan.shots || [];
  const ed = editable();
  m.append(h("h2", {}, "Motion"));
  const count = h("span", { class: "small muted" }, `${S.msel.size} selected`);
  const selWhere = (fn) => { S.msel = new Set(shots.map((s, i) => fn(s, i) ? s.id : null).filter(Boolean)); renderKeepScroll(); };
  m.append(h("div", { class: "panel" },
    h("div", { class: "row" },
      h("button", { class: "small", onclick: () => selWhere(() => true) }, "Select all"),
      h("button", { class: "small", onclick: () => selWhere(() => false) }, "Select none"),
      h("button", { class: "small", onclick: () => selWhere((s, i) => { const r = info(i) || {}; return r.active_still && (!r.active_clip || (r.clip_stale || []).length); }) }, "Select missing or stale"),
      count, h("span", { class: "grow" }),
      h("button", { class: "primary", disabled: !ed, onclick: () => {
        if (!S.msel.size) return toast("select some shots first", true);
        enqueue({ kind: "motion", shots: [...S.msel] }, "motion");
      } }, "Render clips for selected")),
    h("div", { class: "small muted", style: "margin-top:6px" },
      "A clip is made from the shot's active still, about 5 to 6 minutes each. A queued job reads the prompt when it starts, " +
      "so you can keep editing prompts here while other shots render. A job that is already running keeps the prompt it started with.")));
  shots.forEach((s, i) => m.append(motionRow(s, i, count)));
}

function motionRow(s, i, count) {
  const r = info(i) || {};
  const pr = prompt(i);
  const ed = editable();
  const st = activeItem(r, "still");
  const cl = activeItem(r, "clip");
  const jobs = shotJobs(s.id, ["motion"]);
  const running = jobs.find((j) => j.state === "running");
  const queued = jobs.filter((j) => j.state === "queued");
  const cb = h("input", { type: "checkbox", checked: S.msel.has(s.id) });
  cb.addEventListener("change", () => { if (cb.checked) S.msel.add(s.id); else S.msel.delete(s.id); count.textContent = `${S.msel.size} selected`; });

  const left = h("div", {},
    h("div", { class: "row" }, cb, h("b", {}, sidOf(i)), h("span", { class: "badge" }, s.mode || "clay"),
      h("span", { class: "small muted" }, `${(r.est_seconds || 0).toFixed(1)} s`)),
    st ? h("img", { src: thumbUrl(st.file, st.at + st.v, 440), loading: "lazy", style: "margin-top:4px" }) : h("div", { class: "noimg" }, "no active still"),
    st ? h("div", { class: "small muted" }, `still v${st.v}`) : null,
    h("div", { class: "small", style: "margin-top:4px" }, s.narration || ""));

  const vsel = h("select", { class: "small", disabled: !ed || !(r.clips || []).length },
    ...(r.clips || []).map((x) => h("option", { value: x.v }, `v${x.v} seed ${x.seed ?? "?"} from still v${x.still_ref ?? "?"}${x.backend === "dry" ? " DRY" : ""}`)));
  if (r.active_clip) vsel.value = String(r.active_clip);
  vsel.addEventListener("change", () => activate(s.id, "clip", parseInt(vsel.value, 10)));
  const mid = h("div", {},
    cl ? h("video", { src: fileUrl(cl.file, cl.at + cl.v), controls: true, preload: "metadata", loop: true }) : h("div", { class: "noimg" }, "no clip yet"),
    h("div", { class: "row small", style: "margin-top:4px" }, (r.clips || []).length ? vsel : h("span", { class: "muted" }, "no versions"),
      ...(r.clip_stale || []).map((x) => h("span", { class: "badge warn" }, x))));

  const override = s.motion_prompt || "";
  const ta = h("textarea", { disabled: !ed });
  ta.value = override || pr.motion_auto || "";
  const tag = h("span", { class: "badge" + (override ? " warn" : "") }, override ? "edited" : "computed");
  const saveP = () => {
    const v = ta.value.trim();
    if (!v || norm(v) === norm(pr.motion_auto || "")) delete s.motion_prompt; else s.motion_prompt = v;
    scheduleSave(0);
    tag.textContent = s.motion_prompt ? "edited" : "computed";
    tag.className = "badge" + (s.motion_prompt ? " warn" : "");
    toast(running ? `${sidOf(i)} prompt saved. The running render keeps its old prompt; this applies to the next one.` : `${sidOf(i)} motion prompt saved`);
  };
  const seed = h("input", { size: 10, placeholder: "seed (blank = default)" });
  const notes = [];
  if (running) {
    notes.push(h("div", { class: "warnbox small" },
      "Rendering now (progress in the queue panel). It read its prompt when it started and cannot be changed; edits here apply to the next render.",
      running.used && running.used.prompt ? h("details", {}, h("summary", {}, "prompt in use"), h("pre", { class: "prompt" }, running.used.prompt)) : null));
  }
  for (const q of queued) {
    notes.push(h("div", { class: "okbox small" },
      q.params && q.params.prompt ? "Queued with its own prompt override; it will use that." :
        "Queued. It reads the prompt above when it starts, so edits saved now reach it.",
      h("button", { class: "small", style: "margin-left:6px", onclick: () => guard(async () => { await api("DELETE", `/api/jobs/${S.name}/${q.id}`); pollQueue(); }) }, "cancel")));
  }
  const right = h("div", {},
    h("div", { class: "row small" }, h("b", {}, "H3 prompt"), tag,
      pr.motion_auto ? null : h("span", { class: "muted" }, "(compile the plan to see the computed prompt)")),
    ta,
    h("div", { class: "row", style: "margin-top:4px" },
      h("button", { class: "small", disabled: !ed, onclick: saveP }, "Save prompt"),
      h("button", { class: "small", disabled: !ed, onclick: () => { ta.value = pr.motion_auto || ""; saveP(); } }, "Reset to computed"),
      h("span", { class: "grow" }), seed,
      h("button", { class: "primary small", disabled: !ed || !st, onclick: async () => {
        if (norm(ta.value) !== norm(s.motion_prompt || pr.motion_auto || "")) saveP();
        await saveNow();
        enqueue({ kind: "motion", shots: [s.id], seed: seedValue(seed) }, "motion");
      } }, "Render clip")),
    notes);
  return h("div", { class: "mrow" }, left, mid, right);
}

// -- page 5: assemble ------------------------------------------------------------------------

function renderAssemble(m) {
  const shots = S.plan.shots || [];
  const ed = editable();
  m.append(h("h2", {}, "Assemble"));
  const noStill = [], noClip = [], stale = [];
  shots.forEach((s, i) => {
    const r = info(i) || {};
    if (!r.active_still) noStill.push(sidOf(i));
    if (!r.active_clip) noClip.push(sidOf(i));
    if (r.still_stale || (r.clip_stale || []).length) stale.push(sidOf(i));
  });
  const st = S.st.status || {};
  const row = (k) => st[k] ? `${st[k].state} ${st[k].at || ""}` : "not run";
  const busy = (kind) => ((S.queue && S.queue.jobs) || []).some((j) => j.piece === S.name && j.kind === kind && (j.state === "queued" || j.state === "running"));
  m.append(h("div", { class: "panel" },
    h("h3", {}, "Narration"),
    h("div", { class: "small muted" }, `Qwen3-TTS reads the narration of every shot (GPU, farm stopped). Last run: ${row("narrate")}.`),
    S.st.narrated ? h("audio", { src: fileUrl("narration.mp3", (st.narrate || {}).at), controls: true, preload: "none", style: "margin-top:6px;width:100%" }) : null,
    h("div", { class: "row", style: "margin-top:6px" },
      h("button", { class: "primary", disabled: !ed || busy("narrate"), onclick: () => enqueue({ kind: "narrate" }) }, busy("narrate") ? "narration queued" : "Narrate"))));
  m.append(h("div", { class: "panel" },
    h("h3", {}, "Assembly"),
    h("div", { class: "small" }, `${shots.length - noStill.length} of ${shots.length} shots have a still, ${shots.length - noClip.length} have a clip. Last assembly: ${row("assemble")}.`),
    noClip.length ? h("div", { class: "small warnbox", style: "margin-top:6px" }, `No clip yet: ${noClip.join(", ")}`) : null,
    stale.length ? h("div", { class: "small warnbox", style: "margin-top:6px" }, `Stale still or clip: ${stale.join(", ")}`) : null,
    !S.st.narrated ? h("div", { class: "small warnbox", style: "margin-top:6px" }, "No narration yet; narrate first.") : null,
    h("div", { class: "row", style: "margin-top:6px" },
      h("button", { class: "primary", disabled: !ed || busy("assemble"), onclick: () => enqueue({ kind: "assemble" }) }, busy("assemble") ? "assembly queued" : "Assemble"))));
  const films = S.st.films || [];
  m.append(h("div", { class: "panel" }, h("h3", {}, "Player"),
    films.length ? films.map((f) => h("div", { style: "margin-top:8px" },
      h("div", { class: "small muted" }, `${f.file}, ${(f.size / 1e6).toFixed(1)} MB, ${new Date(f.at * 1000).toLocaleString()}`),
      h("video", { src: fileUrl(f.file, f.at), controls: true, preload: "metadata", style: "width:100%;max-width:1100px;background:#000" })))
      : h("div", { class: "muted small" }, "No film yet.")));
}

// -- queue panel -----------------------------------------------------------------------------

async function pollQueue() {
  let q;
  try { q = await api("GET", "/api/queue"); } catch (e) { $("#farm").textContent = "server not answering"; return; }
  const prev = S.queue;
  S.queue = q;
  $("#farm").textContent = `farm: ${q.farm}${q.dry ? " (dry)" : ""}`;
  const gb = $("#gpublock");
  gb.hidden = !q.gpu_block;
  gb.textContent = q.gpu_block ? "GPU jobs waiting: " + q.gpu_block : "";
  renderQueue(q);
  // a job of this piece finished: refresh what it changed
  let changed = false;
  for (const j of q.jobs) {
    if ((j.state === "done" || j.state === "failed") && !S.seenDone.has(j.id)) {
      S.seenDone.add(j.id);
      if (prev && j.piece === S.name) {
        changed = true;
        if (j.state === "failed") toast(`${j.kind} ${j.sid || ""} failed: ${(j.error || "").slice(0, 300)}`, true);
      }
    }
  }
  const liveNow = q.jobs.filter((j) => j.piece === S.name && (j.state === "running" || j.state === "queued")).map((j) => j.id + j.state).join();
  const livePrev = prev ? prev.jobs.filter((j) => j.piece === S.name && (j.state === "running" || j.state === "queued")).map((j) => j.id + j.state).join() : liveNow;
  if (changed && S.name) {
    const planJob = q.jobs.some((j) => j.piece === S.name && j.kind === "plan" && j.state === "done");
    await reload(planJob && !S.dirty);
    refreshView();
  } else if (liveNow !== livePrev && S.name && S.page !== "script" && S.page !== "input") {
    refreshView();
  }
}

function refreshView() {
  const a = document.activeElement;
  const typing = a && (a.tagName === "INPUT" || a.tagName === "TEXTAREA") && $("#main").contains(a);
  if (S.modal && S.modal.kind === "still") {
    const ai = document.activeElement;
    const now = (info(S.modal.i) || {}).active_still;
    // a new version arrived (a finished redo or edit): show it
    const v = now !== S.modal.active ? now : S.modal.v;
    if (!(ai && (ai.tagName === "INPUT" || ai.tagName === "TEXTAREA"))) openStill(S.modal.i, v);
  }
  if (typing || S.page === "input") { if (S.page === "script") updateScriptStatus(); return; }
  renderKeepScroll();
}

function renderQueue(q) {
  const list = $("#qlist");
  const items = q.jobs.map((j) => {
    const run = j.state === "running";
    const lab = `${j.kind}${j.sid ? " " + j.sid : ""}`;
    const kids = [h("div", { class: "row" },
      h("b", {}, lab), h("span", { class: "muted" }, j.piece),
      h("span", { class: "grow" }),
      h("span", { class: "badge " + (run ? "run" : j.state === "done" ? "ok" : j.state === "failed" ? "bad" : "") }, j.state))];
    if (run) {
      kids.push(h("div", { class: "bar" }, h("div", { style: `width:${Math.round((j.progress || 0) * 100)}%` })),
        h("div", { class: "muted" }, `${j.elapsed_s || 0} s of about ${j.est_s} s`),
        h("div", { class: "log" }, (j.log_tail || []).join("\n")));
      if (j.kind === "motion") kids.push(h("div", { class: "muted" }, "running: its prompt is fixed"));
    }
    if (j.note) kids.push(h("div", { class: "muted" }, j.note));
    if (j.state === "failed" && j.error) kids.push(h("div", { style: "color:var(--bad)" }, j.error.slice(0, 240)));
    const btns = [];
    if (j.state === "queued") {
      btns.push(h("button", { class: "small", onclick: () => guard(async () => { await api("DELETE", `/api/jobs/${j.piece}/${j.id}`); pollQueue(); }) }, "cancel"));
      if (j.kind === "motion" || j.kind === "still") btns.push(h("button", { class: "small", onclick: () => editJob(j) }, "prompt"));
    }
    btns.push(h("button", { class: "small", onclick: () => showLog(j) }, "log"));
    kids.push(h("div", { class: "row" }, btns));
    return h("div", { class: "job " + j.state }, kids);
  });
  list.replaceChildren(...(items.length ? items : [h("div", { class: "muted small" }, "nothing queued")]));
}

async function showLog(j) {
  const r = await guard(() => api("GET", `/api/jobs/${j.piece}/${j.id}/log?n=400`));
  if (!r) return;
  S.modal = { kind: "log" };
  const md = $("#modal");
  md.replaceChildren(h("div", { class: "mbox" },
    h("div", { class: "row" }, h("h3", {}, `${j.kind} ${j.sid || ""} ${j.piece}`), h("span", { class: "grow" }), h("button", { onclick: closeModal }, "Close")),
    j.used && j.used.prompt ? h("details", { open: true }, h("summary", {}, "prompt used"), h("pre", { class: "prompt" }, j.used.prompt)) : null,
    h("pre", { class: "prompt", style: "max-height:70vh" }, (r.lines || []).join("\n") || "(no log yet)")));
  md.hidden = false;
  md.onclick = (e) => { if (e.target === md) closeModal(); };
}

function editJob(j) {
  S.modal = { kind: "job" };
  const ta = h("textarea", { rows: 8 });
  ta.value = (j.params && j.params.prompt) || "";
  const md = $("#modal");
  md.replaceChildren(h("div", { class: "mbox", style: "max-width:800px" },
    h("h3", {}, `Prompt override for queued ${j.kind} ${j.sid || ""}`),
    h("div", { class: "small muted" }, "Empty uses the shot's saved prompt when the job starts. This works only while the job is queued."),
    ta,
    h("div", { class: "row", style: "margin-top:6px" },
      h("button", { class: "primary", onclick: () => guard(async () => {
        await api("PATCH", `/api/jobs/${j.piece}/${j.id}`, { prompt: ta.value.trim() || null });
        closeModal(); toast("queued job updated"); pollQueue();
      }) }, "Save"),
      h("button", { onclick: closeModal }, "Cancel"))));
  md.hidden = false;
}

window.addEventListener("beforeunload", () => { if (S.dirty) saveNow(); });
boot().catch((e) => toast("could not start: " + e.message, true));
