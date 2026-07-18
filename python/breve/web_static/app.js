/**
 * breve web UI — chat → scene → live 3D
 * + autoplay default demo, curriculum chips, shareable ?s= / ?example= links
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);

const state = {
  scene: null, // last scene sent to the server (with tweaks applied)
  baseScene: null, // original un-tweaked scene (for re-applying sliders)
  sceneKey: null, // example id or custom key — localStorage scope
  sessionId: null,
  ws: null,
  meshes: new Map(),
  paused: false,
  shareToken: null,
  tweaks: null,
  _tweakReloadTimer: null,
  _cmdQueue: [],
};

// ---------------------------------------------------------------------------
// Lab controls — accordion groups, population, mass ranges, per-demo storage
// ---------------------------------------------------------------------------

const TWEAK_DEFAULTS = {
  speed: 1,
  gravity: 1,
  extraBoxes: 0,
  extraBalls: 0,
  massAll: 1,
  ballMassMin: 0.3,
  ballMassMax: 18,
  boxMassMin: 0.2,
  boxMassMax: 2.5,
  bounce: 1,
  friction: 1,
  velocity: 1,
  randomness: 1,
  sizeJitter: 0,
  autoPause: 1,
  cullOob: 1,
};

const TWEAK_GROUPS = [
  {
    id: "world",
    title: "World",
    open: true,
    summary: (t) => `${t.speed.toFixed(1)}× · g ${t.gravity.toFixed(1)}×`,
    items: [
      { key: "speed", label: "Sim speed", min: 0.25, max: 3, step: 0.05, format: (v) => `${v.toFixed(2)}×`, live: true },
      { key: "gravity", label: "Gravity", min: 0.1, max: 3, step: 0.05, format: (v) => `${v.toFixed(2)}×` },
    ],
  },
  {
    id: "pop",
    title: "Population",
    open: true,
    summary: (t) => `+${t.extraBoxes|0} □ · +${t.extraBalls|0} ○`,
    items: [
      { key: "extraBoxes", label: "Extra boxes", min: 0, max: 40, step: 1, format: (v) => String(Math.round(v)), int: true },
      { key: "extraBalls", label: "Extra balls", min: 0, max: 24, step: 1, format: (v) => String(Math.round(v)), int: true },
      { key: "randomness", label: "Spawn scatter", min: 0, max: 2.5, step: 0.05, format: (v) => `${v.toFixed(2)}×` },
      { key: "sizeJitter", label: "Size variety", min: 0, max: 0.6, step: 0.02, format: (v) => `±${Math.round(v * 100)}%` },
    ],
  },
  {
    id: "mass",
    title: "Mass range",
    open: true,
    summary: (t) => `○ ${t.ballMassMin.toFixed(1)}–${t.ballMassMax.toFixed(1)}`,
    items: [
      { key: "massAll", label: "Global mass ×", min: 0.15, max: 5, step: 0.05, format: (v) => `${v.toFixed(2)}×` },
      { key: "ballMassMin", label: "Balls min mass", min: 0.05, max: 30, step: 0.05, format: (v) => v.toFixed(2) },
      { key: "ballMassMax", label: "Balls max mass", min: 0.05, max: 40, step: 0.05, format: (v) => v.toFixed(2) },
      { key: "boxMassMin", label: "Boxes min mass", min: 0.05, max: 12, step: 0.05, format: (v) => v.toFixed(2) },
      { key: "boxMassMax", label: "Boxes max mass", min: 0.05, max: 20, step: 0.05, format: (v) => v.toFixed(2) },
    ],
  },
  {
    id: "feel",
    title: "Feel & launch",
    open: false,
    summary: (t) => `b${t.bounce.toFixed(1)} f${t.friction.toFixed(1)}`,
    items: [
      { key: "bounce", label: "Bounce", min: 0, max: 2, step: 0.05, format: (v) => `${v.toFixed(2)}×` },
      { key: "friction", label: "Friction", min: 0, max: 2.5, step: 0.05, format: (v) => `${v.toFixed(2)}×` },
      { key: "velocity", label: "Launch velocity", min: 0, max: 3, step: 0.05, format: (v) => `${v.toFixed(2)}×` },
    ],
  },
  {
    id: "auto",
    title: "Housekeeping",
    open: false,
    summary: (t) =>
      `${t.autoPause ? "auto-pause" : "run"} · ${t.cullOob ? "cull" : "keep"}`,
    items: [
      {
        key: "autoPause",
        label: "Auto-pause when still",
        min: 0,
        max: 1,
        step: 1,
        format: (v) => (v >= 0.5 ? "on" : "off"),
        int: true,
        housekeeping: true,
      },
      {
        key: "cullOob",
        label: "Remove off-screen",
        min: 0,
        max: 1,
        step: 1,
        format: (v) => (v >= 0.5 ? "on" : "off"),
        int: true,
        housekeeping: true,
      },
    ],
  },
];

const TWEAK_DEFS = TWEAK_GROUPS.flatMap((g) => g.items);

function storageKeyFor(sceneKey) {
  return `breve_tweaks_v2:${sceneKey || "custom"}`;
}

function loadTweaks(sceneKey) {
  const base = { ...TWEAK_DEFAULTS };
  try {
    // migrate v1 if present
    let raw = localStorage.getItem(storageKeyFor(sceneKey));
    if (!raw) raw = localStorage.getItem(`breve_tweaks_v1:${sceneKey || "custom"}`);
    if (!raw) return base;
    const parsed = JSON.parse(raw);
    for (const d of TWEAK_DEFS) {
      if (typeof parsed[d.key] === "number" && Number.isFinite(parsed[d.key])) {
        base[d.key] = clamp(parsed[d.key], d.min, d.max);
      }
    }
    // legacy massBalls/massBoxes → midpoints in range if new keys missing
    if (parsed.massBalls != null && parsed.ballMassMin == null) {
      const m = Number(parsed.massBalls) || 1;
      base.ballMassMin = clamp(0.3 * m, 0.05, 30);
      base.ballMassMax = clamp(16 * m, 0.05, 40);
    }
    if (parsed.massBoxes != null && parsed.boxMassMin == null) {
      const m = Number(parsed.massBoxes) || 1;
      base.boxMassMin = clamp(0.2 * m, 0.05, 12);
      base.boxMassMax = clamp(1.2 * m, 0.05, 20);
    }
  } catch (_) {}
  // keep min <= max
  if (base.ballMassMin > base.ballMassMax) {
    const t = base.ballMassMin;
    base.ballMassMin = base.ballMassMax;
    base.ballMassMax = t;
  }
  if (base.boxMassMin > base.boxMassMax) {
    const t = base.boxMassMin;
    base.boxMassMin = base.boxMassMax;
    base.boxMassMax = t;
  }
  return base;
}

function saveTweaks(sceneKey, tweaks) {
  try {
    localStorage.setItem(storageKeyFor(sceneKey), JSON.stringify(tweaks));
  } catch (_) {}
}

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

function rand(a, b) {
  return a + Math.random() * (b - a);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function pickMassInRange(minM, maxM, globalScale, prefer) {
  const lo = Math.min(minM, maxM);
  const hi = Math.max(minM, maxM);
  let m;
  if (prefer != null && Number.isFinite(prefer) && hi > lo) {
    // bias toward original relative position if we can map it
    m = lerp(lo, hi, clamp(prefer, 0, 1));
  } else if (hi <= lo) {
    m = lo;
  } else {
    m = rand(lo, hi);
  }
  return Math.max(0.05, m * globalScale);
}

function cloneObj(o) {
  return JSON.parse(JSON.stringify(o));
}

function expandPopulation(objects, t) {
  const statics = [];
  const boxes = [];
  const balls = [];
  for (const o of objects) {
    if (o.static) {
      statics.push(o);
      continue;
    }
    const type = String(o.type || "sphere").toLowerCase();
    if (type === "box") boxes.push(o);
    else balls.push(o);
  }

  const boxTpl =
    boxes[0] ||
    {
      type: "box",
      static: false,
      pos: [0, 1.2, 0],
      size: [0.4, 0.4, 0.4],
      mass: 0.5,
      color: [0.72, 0.52, 0.3],
      restitution: 0.12,
      friction: 0.55,
    };
  const ballTpl =
    balls[0] ||
    {
      type: "sphere",
      static: false,
      pos: [-3, 1.5, 0],
      radius: 0.35,
      mass: 2,
      velocity: [6, 1, 0],
      color: [0.25, 0.4, 0.9],
      restitution: 0.35,
      friction: 0.15,
    };

  const scatter = Math.max(0, Number(t.randomness) || 0);
  const extras = [];

  const nBox = Math.round(t.extraBoxes || 0);
  for (let i = 0; i < nBox; i++) {
    const o = cloneObj(boxTpl);
    const col = i % 6;
    const row = Math.floor(i / 6);
    o.pos = [
      -1.5 + col * 0.55 + rand(-0.12, 0.12) * scatter,
      0.35 + row * 0.48 + rand(0, 0.2) * scatter,
      rand(-1.2, 1.2) * Math.max(0.35, scatter),
    ];
    o.pos_jitter = [0.08 * scatter, 0.05 * scatter, 0.08 * scatter];
    o.color = [
      clamp(0.55 + Math.random() * 0.3, 0, 1),
      clamp(0.38 + Math.random() * 0.25, 0, 1),
      clamp(0.22 + Math.random() * 0.15, 0, 1),
    ];
    delete o.velocity;
    extras.push(o);
  }

  const nBall = Math.round(t.extraBalls || 0);
  for (let i = 0; i < nBall; i++) {
    const o = cloneObj(ballTpl);
    o.pos = [
      -5.5 + rand(0, 2.5) * Math.max(0.4, scatter),
      0.8 + rand(0, 2.5) * Math.max(0.5, scatter),
      rand(-2.2, 2.2) * Math.max(0.4, scatter),
    ];
    o.pos_jitter = [0.15 * scatter, 0.12 * scatter, 0.15 * scatter];
    const speed = 4 + rand(0, 10) * Math.max(0.3, scatter);
    o.velocity = [speed * rand(0.6, 1.2), rand(0.5, 3.5), rand(-1.5, 1.5)];
    o.velocity_jitter = [0.4 * scatter, 0.3 * scatter, 0.35 * scatter];
    o.color = [
      clamp(0.15 + Math.random() * 0.7, 0, 1),
      clamp(0.2 + Math.random() * 0.6, 0, 1),
      clamp(0.4 + Math.random() * 0.55, 0, 1),
    ];
    extras.push(o);
  }

  return [...statics, ...boxes, ...balls, ...extras];
}

function applySizeJitter(o, amount) {
  if (!amount) return;
  const f = 1 + rand(-amount, amount);
  if (String(o.type || "").toLowerCase() === "box" && Array.isArray(o.size)) {
    o.size = o.size.map((s) => Math.max(0.08, Number(s) * f));
  } else if (o.radius != null) {
    o.radius = Math.max(0.06, Number(o.radius) * f);
  }
}

function applyTweaksToScene(baseScene, tweaks) {
  if (!baseScene) return null;
  const s = JSON.parse(JSON.stringify(baseScene));
  const t = tweaks || TWEAK_DEFAULTS;

  if (Array.isArray(s.gravity) && s.gravity.length >= 2) {
    s.gravity = s.gravity.map((g) => Number(g) * t.gravity);
  } else if (t.gravity !== 1) {
    s.gravity = [0, -9.8 * t.gravity, 0];
  }

  // population first (so new objects get mass/feel too)
  s.objects = expandPopulation(s.objects || [], t);

  const ballLo = Math.min(t.ballMassMin, t.ballMassMax);
  const ballHi = Math.max(t.ballMassMin, t.ballMassMax);
  const boxLo = Math.min(t.boxMassMin, t.boxMassMax);
  const boxHi = Math.max(t.boxMassMin, t.boxMassMax);
  const scatter = Math.max(0, Number(t.randomness) || 0);

  // collect mass extents of originals for optional mapping
  let ballMasses = [];
  let boxMasses = [];
  for (const o of s.objects) {
    if (o.static) continue;
    const type = String(o.type || "sphere").toLowerCase();
    const m = Number(o.mass != null ? o.mass : 1);
    if (type === "box") boxMasses.push(m);
    else ballMasses.push(m);
  }
  const ballM0 = ballMasses.length ? Math.min(...ballMasses) : 1;
  const ballM1 = ballMasses.length ? Math.max(...ballMasses) : 1;
  const boxM0 = boxMasses.length ? Math.min(...boxMasses) : 1;
  const boxM1 = boxMasses.length ? Math.max(...boxMasses) : 1;

  for (const o of s.objects) {
    if (o.static) continue;
    const type = String(o.type || "sphere").toLowerCase();
    const baseMass = Number(o.mass != null ? o.mass : 1);

    if (type === "box") {
      const u = boxM1 > boxM0 ? (baseMass - boxM0) / (boxM1 - boxM0) : Math.random();
      o.mass = pickMassInRange(boxLo, boxHi, t.massAll, clamp(u, 0, 1));
    } else {
      const u = ballM1 > ballM0 ? (baseMass - ballM0) / (ballM1 - ballM0) : Math.random();
      o.mass = pickMassInRange(ballLo, ballHi, t.massAll, clamp(u, 0, 1));
    }

    if (o.restitution != null) {
      o.restitution = clamp(Number(o.restitution) * t.bounce, 0, 1);
    } else if (t.bounce !== 1) {
      o.restitution = clamp(0.45 * t.bounce, 0, 1);
    }
    if (o.friction != null) {
      o.friction = clamp(Number(o.friction) * t.friction, 0, 3);
    } else if (t.friction !== 1) {
      o.friction = clamp(0.4 * t.friction, 0, 3);
    }

    if (Array.isArray(o.velocity)) {
      o.velocity = o.velocity.map((v) => Number(v) * t.velocity);
    }
    // jitter: scene supports pos_jitter / velocity_jitter; scale by randomness
    if (scatter !== 1 || t.velocity !== 1) {
      const pj = o.pos_jitter || [0.05, 0.04, 0.05];
      o.pos_jitter = pj.map((v) => Number(v) * scatter);
      if (Array.isArray(o.velocity) || o.velocity_jitter) {
        const vj = o.velocity_jitter || [0.2, 0.15, 0.2];
        o.velocity_jitter = vj.map((v) => Number(v) * scatter * t.velocity);
      }
    }

    applySizeJitter(o, t.sizeJitter);
  }
  return s;
}

function updateTweakTrack(input) {
  const min = Number(input.min);
  const max = Number(input.max);
  const val = Number(input.value);
  const pct = ((val - min) / (max - min)) * 100;
  input.style.setProperty("--pct", `${pct}%`);
}

function updateGroupSummaries() {
  const t = state.tweaks || TWEAK_DEFAULTS;
  for (const g of TWEAK_GROUPS) {
    const el = $(`tweak_group_meta_${g.id}`);
    if (el) el.textContent = g.summary(t);
  }
}

function syncTweakUI() {
  const t = state.tweaks || TWEAK_DEFAULTS;
  for (const d of TWEAK_DEFS) {
    const input = $(`tweak_${d.key}`);
    const valEl = $(`tweak_val_${d.key}`);
    if (!input) continue;
    input.value = String(t[d.key]);
    if (valEl) valEl.textContent = d.format(t[d.key]);
    updateTweakTrack(input);
  }
  updateGroupSummaries();
  const scope = $("tweaksScope");
  if (scope) {
    const key = state.sceneKey || "custom";
    scope.textContent = key.startsWith("example_")
      ? key.replace(/^example_/, "")
      : key === "custom"
        ? "this scene"
        : key.slice(0, 16);
    scope.title = `Saved under “${storageKeyFor(key)}”`;
  }
}

function buildTweakUI() {
  const grid = $("tweakGrid");
  if (!grid) return;
  grid.innerHTML = "";
  for (const g of TWEAK_GROUPS) {
    const det = document.createElement("details");
    det.className = "tweak-group";
    det.open = !!g.open;
    det.innerHTML = `
      <summary>
        <span class="tweak-group-title">${g.title}</span>
        <span class="tweak-group-meta" id="tweak_group_meta_${g.id}"></span>
        <span class="chev">▾</span>
      </summary>
      <div class="tweak-group-body" id="tweak_group_body_${g.id}"></div>
    `;
    const body = det.querySelector(".tweak-group-body");
    for (const d of g.items) {
      const row = document.createElement("div");
      row.className = "tweak";
      row.innerHTML = `
        <span class="tweak-label">${d.label}</span>
        <span class="tweak-value" id="tweak_val_${d.key}"></span>
        <input type="range" id="tweak_${d.key}"
          min="${d.min}" max="${d.max}" step="${d.step}"
          aria-label="${d.label}" />
      `;
      body.appendChild(row);
      const input = row.querySelector("input");
      input.addEventListener("input", () => onTweakInput(d, input));
    }
    grid.appendChild(det);
  }
  $("tweaksResetBtn")?.addEventListener("click", () => {
    state.tweaks = { ...TWEAK_DEFAULTS };
    saveTweaks(state.sceneKey, state.tweaks);
    syncTweakUI();
    applyTweaksLive(true);
    toast("Lab controls reset for this demo");
  });
}

function onTweakInput(def, input) {
  if (!state.tweaks) state.tweaks = { ...TWEAK_DEFAULTS };
  let v = clamp(Number(input.value), def.min, def.max);
  if (def.int) v = Math.round(v);
  state.tweaks[def.key] = v;

  // keep mass ranges ordered while dragging
  if (def.key === "ballMassMin" && state.tweaks.ballMassMin > state.tweaks.ballMassMax) {
    state.tweaks.ballMassMax = state.tweaks.ballMassMin;
    const other = $("tweak_ballMassMax");
    if (other) {
      other.value = String(state.tweaks.ballMassMax);
      updateTweakTrack(other);
      const ve = $("tweak_val_ballMassMax");
      if (ve) ve.textContent = TWEAK_DEFS.find((x) => x.key === "ballMassMax").format(state.tweaks.ballMassMax);
    }
  }
  if (def.key === "ballMassMax" && state.tweaks.ballMassMax < state.tweaks.ballMassMin) {
    state.tweaks.ballMassMin = state.tweaks.ballMassMax;
    const other = $("tweak_ballMassMin");
    if (other) {
      other.value = String(state.tweaks.ballMassMin);
      updateTweakTrack(other);
      const ve = $("tweak_val_ballMassMin");
      if (ve) ve.textContent = TWEAK_DEFS.find((x) => x.key === "ballMassMin").format(state.tweaks.ballMassMin);
    }
  }
  if (def.key === "boxMassMin" && state.tweaks.boxMassMin > state.tweaks.boxMassMax) {
    state.tweaks.boxMassMax = state.tweaks.boxMassMin;
    const other = $("tweak_boxMassMax");
    if (other) {
      other.value = String(state.tweaks.boxMassMax);
      updateTweakTrack(other);
      const ve = $("tweak_val_boxMassMax");
      if (ve) ve.textContent = TWEAK_DEFS.find((x) => x.key === "boxMassMax").format(state.tweaks.boxMassMax);
    }
  }
  if (def.key === "boxMassMax" && state.tweaks.boxMassMax < state.tweaks.boxMassMin) {
    state.tweaks.boxMassMin = state.tweaks.boxMassMax;
    const other = $("tweak_boxMassMin");
    if (other) {
      other.value = String(state.tweaks.boxMassMin);
      updateTweakTrack(other);
      const ve = $("tweak_val_boxMassMin");
      if (ve) ve.textContent = TWEAK_DEFS.find((x) => x.key === "boxMassMin").format(state.tweaks.boxMassMin);
    }
  }

  const valEl = $(`tweak_val_${def.key}`);
  if (valEl) valEl.textContent = def.format(v);
  updateTweakTrack(input);
  updateGroupSummaries();
  saveTweaks(state.sceneKey, state.tweaks);

  if (def.live) {
    sendCmd("set_speed", { speed: state.tweaks.speed });
  } else if (def.housekeeping) {
    sendHousekeeping();
  } else {
    clearTimeout(state._tweakReloadTimer);
    state._tweakReloadTimer = setTimeout(() => applyTweaksLive(false), 180);
  }
}

function housekeepingPrefs() {
  const t = state.tweaks || TWEAK_DEFAULTS;
  return {
    auto_pause: Number(t.autoPause ?? 1) >= 0.5,
    cull_oob: Number(t.cullOob ?? 1) >= 0.5,
    speed: Number(t.speed ?? 1),
  };
}

function sendHousekeeping() {
  const p = housekeepingPrefs();
  sendCmd("set_housekeeping", {
    auto_pause: p.auto_pause,
    cull_oob: p.cull_oob,
  });
}

function pushSessionPrefs() {
  const p = housekeepingPrefs();
  sendCmd("set_speed", { speed: p.speed });
  sendCmd("set_housekeeping", {
    auto_pause: p.auto_pause,
    cull_oob: p.cull_oob,
  });
}

async function applyTweaksLive(forceRestart) {
  if (!state.baseScene) return;
  const scene = applyTweaksToScene(state.baseScene, state.tweaks);
  state.scene = scene;
  if (!state.sessionId || !state.ws || state.ws.readyState !== WebSocket.OPEN || forceRestart) {
    await startSession(scene);
    // prefs applied in startSession body + ws.onopen
    return;
  }
  sendCmd("reload_scene", { scene });
  pushSessionPrefs();
  const n = (scene.objects || []).filter((o) => !o.static).length;
  const p = housekeepingPrefs();
  $("simStatus").textContent = p.auto_pause
    ? `Tweaks applied · ${n} bodies · running`
    : `Tweaks applied · ${n} bodies · auto-pause off`;
}

function setBaseScene(scene, sceneKey) {
  state.baseScene = JSON.parse(JSON.stringify(scene));
  state.sceneKey = sceneKey || "custom";
  state.tweaks = loadTweaks(state.sceneKey);
  syncTweakUI();
  state.scene = applyTweaksToScene(state.baseScene, state.tweaks);
  return state.scene;
}

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

function applyMaterialStyle(mat, obj) {
  const opacity =
    obj.opacity != null ? Math.min(1, Math.max(0, Number(obj.opacity))) : 1;
  mat.color.copy(colorFromArr(obj.color));
  mat.transparent = opacity < 0.999;
  mat.opacity = opacity;
  mat.depthWrite = opacity >= 0.95;
  // glass-ish sides so the cascade reads through them
  if (opacity < 0.999) {
    mat.metalness = 0.05;
    mat.roughness = 0.18;
    mat.side = THREE.DoubleSide;
  } else {
    mat.metalness = 0.15;
    mat.roughness = 0.55;
    mat.side = THREE.FrontSide;
  }
  mat.needsUpdate = true;
}

function ensureMesh(obj) {
  let mesh = state.meshes.get(obj.id);
  if (mesh) return mesh;
  const mat = new THREE.MeshStandardMaterial({
    color: colorFromArr(obj.color),
    metalness: 0.15,
    roughness: 0.55,
  });
  applyMaterialStyle(mat, obj);
  let geo;
  if (obj.type === "box") {
    const s = obj.size || [1, 1, 1];
    geo = new THREE.BoxGeometry(s[0], s[1], s[2]);
  } else {
    geo = new THREE.SphereGeometry(obj.radius || 0.25, 24, 16);
  }
  mesh = new THREE.Mesh(geo, mat);
  const opacity = obj.opacity != null ? Number(obj.opacity) : 1;
  mesh.castShadow = opacity >= 0.95;
  mesh.receiveShadow = true;
  // transparent walls render after opaque balls so you see through correctly
  mesh.renderOrder = opacity < 0.999 ? 1 : 0;
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
    applyMaterialStyle(mesh.material, obj);
    // physics quat is (w,x,y,z); three.js is (x,y,z,w)
    if (obj.quat && obj.quat.length >= 4) {
      mesh.quaternion.set(obj.quat[1], obj.quat[2], obj.quat[3], obj.quat[0]);
    }
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

// Compact AI prompt suggestions (under the compose box)
const chips = $("chips");
const SHORT_LABELS = [
  "Heavy + light balls",
  "Mixed-mass stairs",
  "Cyan flock",
  "Wrecking ball",
  "Ramp race",
];
AI_PROMPTS.forEach((s, i) => {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "chip";
  b.textContent = SHORT_LABELS[i] || s.slice(0, 28);
  b.title = s;
  b.addEventListener("click", () => {
    $("prompt").value = s;
    $("prompt").focus();
  });
  chips.appendChild(b);
});

async function refreshStatus() {
  try {
    const r = await fetch("/api/status");
    const j = await r.json();
    const el = $("keyStatus");
    const avail = !!j.numba_available;
    const on = !!j.numba_physics;
    const accel = avail ? (on ? "Numba on" : "Numba off") : "Numba n/a";
    if (j.has_server_key) {
      el.textContent = `Server key ready · v${j.version} · ${accel} · AI chat on`;
      el.className = "hint ok";
    } else if (getKey()) {
      el.textContent = `Browser key ready · v${j.version} · ${accel}`;
      el.className = "hint ok";
    } else {
      el.textContent = `No API key — demos work · ${accel} · paste xAI key to chat-build`;
      el.className = "hint";
    }
    syncNumbaToggle(j);
  } catch {
    $("keyStatus").textContent = "API unreachable";
    $("keyStatus").className = "hint bad";
  }
}

function syncNumbaToggle(status) {
  const toggle = $("numbaToggle");
  const hint = $("numbaHint");
  const st = $("numbaStatus");
  if (!toggle) return;
  const avail = !!(status && status.numba_available);
  const on = !!(status && status.numba_physics);
  toggle.disabled = !avail;
  toggle.checked = avail && on;
  if (hint) {
    hint.textContent = avail
      ? "Faster CPU JIT solver (toggle anytime)"
      : "Not installed — pip install 'breve[fast]'";
  }
  if (st) {
    if (!avail) {
      st.textContent = "Install numba on the server to enable this switch.";
      st.className = "hint";
    } else if (on) {
      st.textContent = "Using Numba JIT for integrate + contact resolve.";
      st.className = "hint ok";
    } else {
      st.textContent = "Using pure-Python physics solver.";
      st.className = "hint";
    }
  }
}

async function setNumbaPreference(enabled) {
  const st = $("numbaStatus");
  try {
    localStorage.setItem("breve_numba", enabled ? "1" : "0");
    const r = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ numba: !!enabled }),
    });
    const j = await r.json();
    if (!r.ok || j.ok === false) {
      throw new Error(j.error || j.detail || "Could not update Numba setting");
    }
    syncNumbaToggle({
      numba_available: j.numba_available,
      numba_physics: j.numba_physics,
    });
    await refreshStatus();
    toast(j.numba_physics ? "Numba physics enabled" : "Pure-Python physics");
  } catch (e) {
    if (st) {
      st.textContent = String(e.message || e);
      st.className = "hint bad";
    }
    toast(String(e.message || e), true);
    // re-sync from server
    refreshStatus();
  }
}

function wireSettingsToggles() {
  const toggle = $("numbaToggle");
  if (toggle) {
    toggle.addEventListener("change", () => {
      setNumbaPreference(toggle.checked);
    });
  }
  // Don't collapse Lab panel when clicking Defaults / scope badge
  $("tweaksResetBtn")?.addEventListener("click", (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
  });
  $("tweaksScope")?.addEventListener("click", (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
  });
}

async function applyStoredNumbaPreference() {
  const raw = localStorage.getItem("breve_numba");
  if (raw !== "0" && raw !== "1") return;
  try {
    await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ numba: raw === "1" }),
    });
  } catch (_) {}
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
    state.shareToken = null;
    // base scene + restore this demo’s saved sliders (survive re-click)
    const tuned = setBaseScene(scene, id);
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
    await startSession(tuned);
    sendCmd("set_speed", { speed: state.tweaks?.speed ?? 1 });
    sendHousekeeping();
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
    state.shareToken = j.share_token || null;
    // Chat scenes share one “custom” tweak slot (or per share token)
    const key = j.share_token ? `share_${j.share_token.slice(0, 12)}` : "custom";
    const tuned = setBaseScene(j.scene, key);
    markCurriculumActive("");
    addBubble(j.explanation || "Scene ready.", "assistant");
    if (j.share_token) {
      history.replaceState({}, "", `/?s=${j.share_token}`);
    }
    $("simStatus").textContent = `“${j.title || "untitled"}” — running`;
    await startSession(tuned);
    sendCmd("set_speed", { speed: state.tweaks?.speed ?? 1 });
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
  state._cmdQueue = [];
  const prefs = housekeepingPrefs();
  const r = await fetch("/api/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      scene: sceneSpec,
      auto_pause: prefs.auto_pause,
      cull_oob: prefs.cull_oob,
      speed: prefs.speed,
    }),
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
  const ap = j.auto_pause === false ? "auto-pause off" : "auto-pause on";
  $("simStatus").textContent = `Running · ${ap}`;
}

function stopWs() {
  if (state.ws) {
    try {
      state.ws.close();
    } catch (_) {}
    state.ws = null;
  }
  state._cmdQueue = [];
}

function connectWs(sessionId) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/sim/${sessionId}`);
  state.ws = ws;
  ws.onopen = () => {
    // Re-apply lab prefs once the socket is live (covers race after session create)
    pushSessionPrefs();
    const q = state._cmdQueue.splice(0, state._cmdQueue.length);
    for (const msg of q) {
      if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify(msg));
      }
    }
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "state") {
      applyState(msg.state);
      // Only treat as auto-pause when the server flags it AND lab still wants it
      const wantAuto = Number((state.tweaks || TWEAK_DEFAULTS).autoPause ?? 1) >= 0.5;
      if (wantAuto && (msg.just_settled || msg.auto_paused)) {
        state.paused = true;
        const culled =
          msg.culled_total != null ? msg.culled_total : msg.culled || 0;
        const parts = ["Settled — auto-paused"];
        if (culled > 0) parts.push(`${culled} removed off-screen`);
        $("simStatus").textContent = parts.join(" · ");
      } else if (msg.culled > 0 && !state.paused) {
        // brief note when debris leaves the play volume mid-run
        const n = (msg.state?.objects || []).filter((o) => !o.static).length;
        $("simStatus").textContent = `Running · ${n} bodies · culled ${msg.culled_total || msg.culled}`;
      }
    }
    if (msg.type === "error") {
      addBubble(msg.error, "system");
      $("simStatus").textContent = "Error — see chat";
    }
  };
  ws.onclose = () => {
    if (state.ws === ws) $("simStatus").textContent = "Socket closed — press Run";
  };
}

function sendCmd(cmd, extra = {}) {
  const msg = { cmd, ...extra };
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(msg));
  } else {
    // Queue until websocket opens (session start race)
    state._cmdQueue.push(msg);
  }
}

$("playBtn").addEventListener("click", async () => {
  if (!state.scene) {
    addBubble("Build or load a scene first.", "system");
    return;
  }
  if (!state.sessionId) {
    if (state.baseScene) {
      state.scene = applyTweaksToScene(state.baseScene, state.tweaks);
    }
    await startSession(state.scene);
    sendCmd("set_speed", { speed: state.tweaks?.speed ?? 1 });
    sendHousekeeping();
  } else {
    state.paused = false;
    sendCmd("resume");
    sendCmd("set_speed", { speed: state.tweaks?.speed ?? 1 });
    sendHousekeeping();
    $("simStatus").textContent = "Running";
  }
});
$("pauseBtn").addEventListener("click", () => {
  state.paused = true;
  sendCmd("pause");
  $("simStatus").textContent = "Paused";
});
$("resetBtn").addEventListener("click", () => {
  // Re-apply current tweaks so Reset keeps mass/gravity/velocity settings
  if (state.baseScene) {
    state.scene = applyTweaksToScene(state.baseScene, state.tweaks);
    sendCmd("reset", { scene: state.scene });
  } else {
    sendCmd("reset");
  }
  sendCmd("set_speed", { speed: state.tweaks?.speed ?? 1 });
  state.paused = false;
  $("simStatus").textContent = "Reset";
});

// --- boot: URL params → autoplay --------------------------------------------

async function boot() {
  buildTweakUI();
  wireSettingsToggles();
  state.tweaks = loadTweaks("example_gravity");
  syncTweakUI();

  addBubble(
    "Demo auto-starts. Lab controls (mass ranges, extra bodies, scatter) are saved per demo and restart the sim when you drag — speed is live. Open Build with Grok when you want AI scenes.",
    "system"
  );
  await applyStoredNumbaPreference();
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
      state.shareToken = shareToken;
      const tuned = setBaseScene(scene, `share_${shareToken.slice(0, 12)}`);
      addBubble(`Opened shared scene “${scene.title || "untitled"}”`, "system");
      if (scene.notes) addBubble(scene.notes, "assistant");
      $("simStatus").textContent = "Shared scene — running";
      await startSession(tuned);
      sendCmd("set_speed", { speed: state.tweaks?.speed ?? 1 });
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
