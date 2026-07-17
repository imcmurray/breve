/**
 * breve web UI — chat → scene JSON → live 3D via WebSocket + three.js
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);

const state = {
  scene: null,
  sessionId: null,
  ws: null,
  meshes: new Map(), // id -> THREE.Object3D
  paused: false,
};

const SUGGESTIONS = [
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

const hemi = new THREE.HemisphereLight(0xb0c4ff, 0x223311, 1.1);
scene3.add(hemi);
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
    const r = obj.radius || 0.25;
    geo = new THREE.SphereGeometry(r, 24, 16);
  }
  mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.userData = { type: obj.type };
  scene3.add(mesh);
  state.meshes.set(obj.id, mesh);
  return mesh;
}

function applyState(simState) {
  if (!simState) return;
  const live = new Set();
  for (const obj of simState.objects || []) {
    live.add(obj.id);
    // skip huge ground planes for mesh spam? still show them
    if (obj.type === "box" && obj.size) {
      const maxS = Math.max(...obj.size);
      if (maxS > 50) continue;
    }
    const mesh = ensureMesh(obj);
    mesh.position.set(obj.pos[0], obj.pos[1], obj.pos[2]);
    mesh.material.color.copy(colorFromArr(obj.color));
  }
  // remove stale
  for (const id of [...state.meshes.keys()]) {
    if (!live.has(id)) {
      const m = state.meshes.get(id);
      scene3.remove(m);
      state.meshes.delete(id);
    }
  }

  if (simState.background) {
    const b = simState.background;
    scene3.background = new THREE.Color(b[0] * 0.35, b[1] * 0.35, b[2] * 0.4 + 0.05);
  }
  if (simState.camera?.target) {
    const t = simState.camera.target;
    controls.target.set(t[0], t[1], t[2]);
  }
  $("hudTitle").textContent = simState.title || "breve";
  $("hudTime").textContent = `t=${(simState.time || 0).toFixed(1)}s · objects=${(simState.objects || []).length}`;
}

function animate() {
  requestAnimationFrame(animate);
  resize();
  controls.update();
  renderer.render(scene3, camera);
}
animate();

// --- UI helpers -------------------------------------------------------------

function addBubble(text, role) {
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  div.textContent = text;
  $("chatLog").appendChild(div);
  $("chatLog").scrollTop = $("chatLog").scrollHeight;
}

function setOverlay(show) {
  $("overlay").classList.toggle("hidden", !show);
}

function getKey() {
  return $("apiKey").value.trim() || localStorage.getItem("breve_xai_key") || "";
}

$("apiKey").value = localStorage.getItem("breve_xai_key") || "";
$("apiKey").addEventListener("change", () => {
  const v = $("apiKey").value.trim();
  if (v) localStorage.setItem("breve_xai_key", v);
  else localStorage.removeItem("breve_xai_key");
});

// chips
const chips = $("chips");
for (const s of SUGGESTIONS) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "chip";
  b.textContent = s.length > 48 ? s.slice(0, 46) + "…" : s;
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
      el.textContent = `Server has XAI_API_KEY · v${j.version}`;
      el.className = "hint ok";
    } else if (getKey()) {
      el.textContent = `Using key from this browser · v${j.version}`;
      el.className = "hint ok";
    } else {
      el.textContent = "No API key — paste xAI key above or set XAI_API_KEY on the server";
      el.className = "hint bad";
    }
  } catch {
    $("keyStatus").textContent = "Cannot reach API";
    $("keyStatus").className = "hint bad";
  }
}

async function loadExamples() {
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

$("loadExampleBtn").addEventListener("click", async () => {
  const id = $("exampleSelect").value;
  if (!id) return;
  const r = await fetch(`/api/examples/${id}`);
  if (!r.ok) {
    addBubble("Failed to load example", "system");
    return;
  }
  state.scene = await r.json();
  addBubble(`Loaded example “${state.scene.title || id}”`, "system");
  if (state.scene.notes) addBubble(state.scene.notes, "assistant");
  $("simStatus").textContent = "Example loaded — press Run";
  await startSession(state.scene);
});

// --- chat -------------------------------------------------------------------

async function buildFromPrompt(message, refine) {
  setOverlay(true);
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
      addBubble(j.detail || r.statusText || "Request failed", "system");
      return;
    }
    state.scene = j.scene;
    addBubble(j.explanation || "Scene ready.", "assistant");
    $("simStatus").textContent = `Scene “${j.title || "untitled"}” ready — running…`;
    await startSession(state.scene);
  } catch (e) {
    addBubble(String(e), "system");
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
    addBubble("Build a scene first, then refine it.", "system");
    return;
  }
  $("prompt").value = "";
  buildFromPrompt(msg, true);
});

// --- simulation session -----------------------------------------------------

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
  // frame camera once
  if (j.state?.camera?.target) {
    const t = j.state.camera.target;
    const z = j.state.camera.zoom || 12;
    controls.target.set(t[0], t[1], t[2]);
    camera.position.set(t[0] + z * 0.6, t[1] + z * 0.35, t[2] + z * 0.7);
  }
  connectWs(j.session_id);
  $("simStatus").textContent = `Running session ${j.session_id}`;
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
    if (state.ws === ws) $("simStatus").textContent = "Simulation socket closed";
  };
  ws.onerror = () => addBubble("WebSocket error", "system");
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

// boot
addBubble(
  "Describe a 3D world in plain English. Grok builds a safe scene (floors, masses, flocks). Then watch it live — drag to orbit, scroll to zoom.",
  "system"
);
refreshStatus();
loadExamples();
