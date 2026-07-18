/**
 * breve web UI — chat → scene → live 3D
 * + autoplay default demo, curriculum chips, shareable ?s= / ?example= links
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);

const state = {
  scene: null,
  sessionId: null,
  ws: null,
  meshes: new Map(),
  paused: false,
  shareToken: null,
};

const AI_PROMPTS = [
  "Heavy red bowling ball and light yellow ping-pong balls bouncing on a floor so I can see gravity and mass",
  "Staircase with mixed-mass balls rolling down",
  "Flock of 50 cyan birds in open space",
  "Box tower and a wrecking ball lobbed from the left",
  "Two ramps and balls of different weights racing down",
];

// --- three.js ---------------------------------------------------------------

const canvas = $("c");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
renderer.shadowMap.enabled = true;

const scene3 = new THREE.Scene();
scene3.background = new THREE.Color(0x0a0e16);
scene3.fog = new THREE.Fog(0x0a0e16, 40, 90);

const camera = new THREE.PerspectiveCamera(
  50,
  canvas.clientWidth / Math.max(canvas.clientHeight, 1),
  0.1,
  200
);
camera.position.set(10, 6, 12);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.target.set(0, 1, 0);

scene3.add(new THREE.HemisphereLight(0xb0c4ff, 0x223311, 1.1));
const dir = new THREE.DirectionalLight(0xffffff, 1.0);
dir.position.set(8, 16, 6);
dir.castShadow = true;
scene3.add(dir);

const grid = new THREE.GridHelper(40, 40, 0x3a4a60, 0x1e2838);
grid.position.y = 0.001;
scene3.add(grid);

function resize() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== w || canvas.height !== h) {
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(h, 1);
    camera.updateProjectionMatrix();
  }
}

function clearMeshes() {
  for (const m of state.meshes.values()) {
    scene3.remove(m);
    m.geometry?.dispose?.();
    if (m.material) {
      if (Array.isArray(m.material)) m.material.forEach((x) => x.dispose());
      else m.material.dispose();
    }
  }
  state.meshes.clear();
}

function colorFromArr(c) {
  if (!c || c.length < 3) return new THREE.Color(0.7, 0.7, 0.8);
  return new THREE.Color(c[0], c[1], c[2]);
}

function ensureMesh(obj) {
  let mesh = state.meshes.get(obj.id);
  if (mesh) return mesh;
  const mat = new THREE.MeshStandardMaterial({
    color: colorFromArr(obj.color),
    metalness: 0.15,
    roughness: 0.55,
  });
  let geo;
  if (obj.type === "box") {
    const s = obj.size || [1, 1, 1];
    geo = new THREE.BoxGeometry(s[0], s[1], s[2]);
  } else {
    geo = new THREE.SphereGeometry(obj.radius || 0.25, 24, 16);
  }
  mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  scene3.add(mesh);
  state.meshes.set(obj.id, mesh);
  return mesh;
}

function applyState(simState) {
  if (!simState) return;
  const live = new Set();
  for (const obj of simState.objects || []) {
    live.add(obj.id);
    if (obj.type === "box" && obj.size && Math.max(...obj.size) > 50) continue;
    const mesh = ensureMesh(obj);
    mesh.position.set(obj.pos[0], obj.pos[1], obj.pos[2]);
    mesh.material.color.copy(colorFromArr(obj.color));
  }
  for (const id of [...state.meshes.keys()]) {
    if (!live.has(id)) {
      scene3.remove(state.meshes.get(id));
      state.meshes.delete(id);
    }
  }
  if (simState.background) {
    const b = simState.background;
    scene3.background = new THREE.Color(b[0] * 0.35, b[1] * 0.35, b[2] * 0.4 + 0.05);
  }
  if (simState.camera?.target) {
    const t = simState.camera.target;
    controls.target.lerp(new THREE.Vector3(t[0], t[1], t[2]), 0.08);
  }
  $("hudTitle").textContent = simState.title || "breve";
  $("hudTime").textContent = `t=${(simState.time || 0).toFixed(1)}s · n=${(simState.objects || []).length}`;
}

function frameCamera(simState) {
  if (!simState?.camera?.target) return;
  const t = simState.camera.target;
  const z = simState.camera.zoom || 12;
  controls.target.set(t[0], t[1], t[2]);
  camera.position.set(t[0] + z * 0.6, t[1] + z * 0.35, t[2] + z * 0.7);
}

function animate() {
  requestAnimationFrame(animate);
  resize();
  controls.update();
  renderer.render(scene3, camera);
}
animate();

// fade welcome tip
setTimeout(() => $("welcome")?.classList.add("fade"), 6000);

// --- UI helpers -------------------------------------------------------------

function addBubble(text, role) {
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  div.textContent = text;
  $("chatLog").appendChild(div);
  $("chatLog").scrollTop = $("chatLog").scrollHeight;
}

function setOverlay(show, title, body) {
  $("overlay").classList.toggle("hidden", !show);
  if (title) $("overlayTitle").textContent = title;
  if (body) $("overlayBody").textContent = body;
}

function toast(msg, isError = false) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.toggle("error", !!isError);
  el.classList.remove("hidden");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.add("hidden"), 2800);
}

function getKey() {
  return $("apiKey").value.trim() || localStorage.getItem("breve_xai_key") || "";
}

$("apiKey").value = localStorage.getItem("breve_xai_key") || "";
$("apiKey").addEventListener("change", () => {
  const v = $("apiKey").value.trim();
  if (v) localStorage.setItem("breve_xai_key", v);
  else localStorage.removeItem("breve_xai_key");
  refreshStatus();
});

// AI prompt chips
const chips = $("chips");
for (const s of AI_PROMPTS) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "chip";
  b.textContent = s.length > 42 ? s.slice(0, 40) + "…" : s;
  b.title = s;
  b.addEventListener("click", () => {
    $("prompt").value = s;
    $("prompt").focus();
  });
  chips.appendChild(b);
}

async function refreshStatus() {
  try {
    const r = await fetch("/api/status");
    const j = await r.json();
    const el = $("keyStatus");
    if (j.has_server_key) {
      el.textContent = `Server key ready · v${j.version} · AI chat on`;
      el.className = "hint ok";
    } else if (getKey()) {
      el.textContent = `Browser key ready · v${j.version}`;
      el.className = "hint ok";
    } else {
      el.textContent = "No API key — examples still work; paste xAI key to chat-build";
      el.className = "hint";
    }
  } catch {
    $("keyStatus").textContent = "API unreachable";
    $("keyStatus").className = "hint bad";
  }
}

// --- curriculum + examples --------------------------------------------------

async function loadCurriculum() {
  const r = await fetch("/api/curriculum");
  const j = await r.json();
  const box = $("curriculumChips");
  box.innerHTML = "";
  for (const c of j.curriculum || []) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip curriculum";
    b.dataset.id = c.id;
    b.textContent = c.label;
    b.title = c.blurb || c.notes || c.label;
    b.addEventListener("click", () => loadExampleById(c.id, true));
    box.appendChild(b);
  }
  return j;
}

async function loadExamplesSelect() {
  const r = await fetch("/api/examples");
  const j = await r.json();
  const sel = $("exampleSelect");
  for (const ex of j.examples || []) {
    const opt = document.createElement("option");
    opt.value = ex.id;
    opt.textContent = ex.title;
    sel.appendChild(opt);
  }
}

function markCurriculumActive(id) {
  document.querySelectorAll("#curriculumChips .chip").forEach((el) => {
    el.classList.toggle("active", el.dataset.id === id);
  });
}

async function loadExampleById(id, fromChip = false) {
  setOverlay(true, "Loading demo…", "Starting physics / agents in the browser.");
  try {
    const r = await fetch(`/api/examples/${encodeURIComponent(id)}`);
    if (!r.ok) throw new Error("Example not found");
    const scene = await r.json();
    state.scene = scene;
    state.shareToken = null;
    markCurriculumActive(id);
    if (fromChip) {
      addBubble(`Curriculum: ${scene.title || id}`, "system");
      if (scene.notes) addBubble(scene.notes, "assistant");
    }
    // update URL without reload (shareable example link)
    const url = new URL(location.href);
    url.searchParams.delete("s");
    url.searchParams.set("example", id);
    history.replaceState({}, "", url);
    $("simStatus").textContent = `Demo “${scene.title || id}” — running`;
    await startSession(scene);
  } catch (e) {
    addBubble(String(e), "system");
    toast(String(e), true);
  } finally {
    setOverlay(false);
  }
}

$("loadExampleBtn").addEventListener("click", async () => {
  const id = $("exampleSelect").value;
  if (!id) return;
  await loadExampleById(id, true);
});

// --- share ------------------------------------------------------------------

async function copyShareLink() {
  if (!state.scene) {
    toast("Nothing to share yet", true);
    return;
  }
  $("shareStatus").textContent = "Creating link…";
  try {
    const r = await fetch("/api/share", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scene: state.scene }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "Share failed");
    if (!j.ok) {
      toast("Scene is large — link may break in some browsers", true);
    }
    state.shareToken = j.token;
    const full = `${location.origin}/?s=${j.token}`;
    // also set example-style clean URL when possible is harder; use s=
    await navigator.clipboard.writeText(full);
    history.replaceState({}, "", `/?s=${j.token}`);
    $("shareStatus").textContent = "Link copied to clipboard";
    $("shareStatus").className = "hint share-status ok";
    toast("Share link copied");
  } catch (e) {
    $("shareStatus").textContent = String(e);
    $("shareStatus").className = "hint share-status";
    toast(String(e), true);
  }
}

$("shareBtn").addEventListener("click", copyShareLink);

// --- chat -------------------------------------------------------------------

async function buildFromPrompt(message, refine) {
  setOverlay(true, "Building scene…", "Grok is composing floors, masses, and framing.");
  $("sendBtn").disabled = true;
  addBubble(message, "user");
  try {
    const body = {
      message,
      api_key: getKey() || null,
      scene: refine && state.scene ? state.scene : null,
    };
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) {
      const detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail || r.statusText);
      addBubble(detail || "Request failed", "system");
      toast(detail || "AI request failed", true);
      return;
    }
    state.scene = j.scene;
    state.shareToken = j.share_token || null;
    markCurriculumActive("");
    addBubble(j.explanation || "Scene ready.", "assistant");
    if (j.share_token) {
      history.replaceState({}, "", `/?s=${j.share_token}`);
    }
    $("simStatus").textContent = `“${j.title || "untitled"}” — running`;
    await startSession(state.scene);
  } catch (e) {
    addBubble(String(e), "system");
    toast(String(e), true);
  } finally {
    setOverlay(false);
    $("sendBtn").disabled = false;
  }
}

$("chatForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const msg = $("prompt").value.trim();
  if (!msg) return;
  $("prompt").value = "";
  buildFromPrompt(msg, false);
});

$("refineBtn").addEventListener("click", () => {
  const msg = $("prompt").value.trim();
  if (!msg) {
    addBubble("Type a refine instruction first (e.g. “make gravity stronger”).", "system");
    return;
  }
  if (!state.scene) {
    addBubble("Build or load a scene first, then refine.", "system");
    return;
  }
  $("prompt").value = "";
  buildFromPrompt(msg, true);
});

// --- simulation -------------------------------------------------------------

async function startSession(sceneSpec) {
  stopWs();
  clearMeshes();
  const r = await fetch("/api/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scene: sceneSpec }),
  });
  const j = await r.json();
  if (!r.ok) {
    addBubble(j.detail || "Session failed", "system");
    return;
  }
  state.sessionId = j.session_id;
  state.paused = false;
  applyState(j.state);
  frameCamera(j.state);
  connectWs(j.session_id);
  $("simStatus").textContent = `Running · session ${j.session_id}`;
}

function stopWs() {
  if (state.ws) {
    try {
      state.ws.close();
    } catch (_) {}
    state.ws = null;
  }
}

function connectWs(sessionId) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/sim/${sessionId}`);
  state.ws = ws;
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "state") applyState(msg.state);
    if (msg.type === "error") {
      addBubble(msg.error, "system");
      $("simStatus").textContent = "Error — see chat";
    }
  };
  ws.onclose = () => {
    if (state.ws === ws) $("simStatus").textContent = "Socket closed — press Run";
  };
}

function sendCmd(cmd) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ cmd }));
  }
}

$("playBtn").addEventListener("click", async () => {
  if (!state.scene) {
    addBubble("Build or load a scene first.", "system");
    return;
  }
  if (!state.sessionId) await startSession(state.scene);
  else {
    state.paused = false;
    sendCmd("resume");
    $("simStatus").textContent = "Running";
  }
});
$("pauseBtn").addEventListener("click", () => {
  state.paused = true;
  sendCmd("pause");
  $("simStatus").textContent = "Paused";
});
$("resetBtn").addEventListener("click", () => {
  sendCmd("reset");
  state.paused = false;
  $("simStatus").textContent = "Reset";
});

// --- boot: URL params → autoplay --------------------------------------------

async function boot() {
  addBubble(
    "This is the product: a live continuous 3D multi-agent world in your browser.\n\n" +
      "• Demo auto-starts (no API key)\n" +
      "• Curriculum chips teach gravity, mass, impact, flocking\n" +
      "• Paste an xAI key only when you want to invent scenes in English\n" +
      "• Share copies a link to this world\n\n" +
      "Drag the 3D view to orbit. That’s the whole point of this revival.",
    "system"
  );
  await refreshStatus();
  await loadCurriculum();
  await loadExamplesSelect();

  const params = new URLSearchParams(location.search);
  const shareToken = params.get("s");
  const exampleId = params.get("example");

  try {
    if (shareToken) {
      setOverlay(true, "Opening shared scene…", "Decoding link and starting simulation.");
      const r = await fetch(`/api/share/${encodeURIComponent(shareToken)}`);
      const scene = await r.json();
      if (!r.ok) throw new Error(scene.detail || "Bad share link");
      state.scene = scene;
      state.shareToken = shareToken;
      addBubble(`Opened shared scene “${scene.title || "untitled"}”`, "system");
      if (scene.notes) addBubble(scene.notes, "assistant");
      $("simStatus").textContent = "Shared scene — running";
      await startSession(scene);
    } else if (exampleId) {
      await loadExampleById(exampleId, true);
    } else {
      // HUGE WIN: never show an empty canvas
      await loadExampleById("example_gravity", false);
      addBubble(
        "Auto-playing “heavy vs light” gravity demo. Try curriculum chips, or Share this view.",
        "system"
      );
    }
  } catch (e) {
    addBubble(`Boot: ${e}`, "system");
    // fallback
    try {
      await loadExampleById("example_gravity", false);
    } catch (_) {
      $("simStatus").textContent = "Could not auto-start — pick a curriculum chip";
    }
  } finally {
    setOverlay(false);
  }
}

boot();
