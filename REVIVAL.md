# Breve Revival — Analysis & Modernization Roadmap

**Source:** [jonklein/breve](https://github.com/jonklein/breve) (GPL-2.0)  
**Last upstream commit:** 2015-10-13 (`f83c863` — “fix configure bitrot”)  
**Declared version:** 2.7.5b2  
**This document:** living plan for a successful, modern resurrection.

---

## 1. What Breve is (and why it still matters)

Breve (Jon Klein, ~2000–2009, bitrot fixes through 2015) is a **continuous-time, continuous-space 3D multi-agent / artificial-life simulator**. Agents are scripted in **Python** or a custom language **steve**; the engine provides:

- OpenGL visualization (camera, lighting, shadows, textures, skybox)
- Rigid-body physics via **ODE**
- Collision detection (V-Clip + ODE)
- Articulated bodies / joints / springs / terrain
- Genetic algorithms, simple neural nets, patch chemistry, Push GP hooks
- Optional networking for multi-host sims

**Gap today:** NetLogo / Mesa are excellent but mostly 2D/grid. Unity ML-Agents / Isaac / MuJoCo are heavyweight and robotics-oriented. Particle-life / ALIEN are specialized. There is still no **lightweight, educational, open 3D agent sandbox** with breve’s “subclass Control + Mobile, implement `iterate` and collision handlers” UX.

---

## 2. Current repository structure

| Path | Role | Notes |
|------|------|--------|
| `kernel/` | Simulation engine core | Eval, object API, plugins, networking, multi-frontend |
| `simulation/` | 3D world + render + physics | `world.cc`, `gldraw.cc`, ODE joints, V-Clip, terrain, sensors |
| `steve/` | steve language | lex/yacc (`stevelex.l`, `steveparse.y`), ~4k LOC evaluator |
| `python/` | Python frontend bridge | Embeds CPython, exposes `breveInternal` |
| `lib/classes/` | Standard class library | `.tz` (steve) + auto-converted `.py` |
| `lib/classes/breve/` | Python agent API | Control, Mobile, Real, Shape, GA, patches, … |
| `demos/` | Showcase sims | ~54 `.py` + ~56 `.tz` (Swarm, Walker, Braitenberg, PatchLife, …) |
| `neural/` | Feedforward nets | C |
| `plugins/` | Digitizer, QuickTime music, regex | Platform-tied |
| `wx/`, `Qt/`, `OSX/` | IDE / GUI frontends | wxWidgets, Qt, Cocoa |
| `docBuild/` | Doc generation | DocBook/Doxygen/Perl — no ready-to-read `docs/` in tree |
| `configure.ac` | Autotools build | AC_PREREQ 2.57; Python 2.3–2.7 only |

**Language mix (GitHub):** ~89% Python (mostly demos + class lib + bundled stdlib), ~6% C++, rest C / ObjC++ / scripts.

**Rough engine size:** ~30k LOC across kernel + simulation + python + steve (C/C++).

### Agent model (spirit to preserve)

```
Object
├── Abstract          # no world body (controllers, data)
│   └── Control       # one per sim: init, iterate, camera, menus
└── Real              # has world representation
    ├── Mobile        # moves, collides, optional physics
    ├── Stationary
    ├── Link / MultiBody / Joint / Spring
    └── Shape, Terrain, …
```

User workflow:

1. Subclass `Control`, set up world in `init()`
2. Subclass `Mobile` (etc.), implement `iterate()` and collision handlers
3. Instantiate controller → engine runs timestep loop

Example (from `demos/Getting-Started/HelloWorld.py`):

```python
import breve

class HelloWorld(breve.Control):
    def iterate(self):
        print("Hello, world!")
        breve.Control.iterate(self)

HelloWorld()
```

---

## 3. Strengths

1. **Clean agent abstraction** — Control / Mobile / collision callbacks are still a great teaching API.
2. **Rich demos** — flocking, Braitenberg vehicles, rigid walkers, patch RD chemistry, GA, Push — excellent curriculum material.
3. **Continuous 3D + physics** — different niche from NetLogo.
4. **Multi-language design** — `brObjectType` frontend interface is conceptually sound (even if C-era).
5. **Educational DNA** — Hampshire College / ALife research lineage; fun to explore.
6. **GPL-2.0** — clear open license for a revival fork.

---

## 4. Main pain points

| Area | Reality |
|------|---------|
| **Python** | **Python 2 only** (`cPickle`, `Exception, e`, checks for 2.3–2.7; bug: 2.7 link line uses `-lpython2.6`) |
| **Build** | Ancient autoconf; empty `install-sh`; no CMake; last “bitrot” pass was OS X 10.9 (2015) |
| **Deps** | ODE (double precision custom build), GLUT, GSL, fixed-function OpenGL, optional ffmpeg/portaudio/3ds |
| **Graphics** | Pre-shader OpenGL + GLUT — painful on Wayland/modern GL |
| **Docs** | README is m4 template; no markdown; docs built offline; website bitrotted |
| **Code quality** | Author notes: “total mess”, C/C++ hybrid, void* soup; incomplete C++ migration |
| **Shipping** | Bundled python2.3–2.5 stdlibs + `.pyc`/`.pyo`/`.so` binaries; Windows DLLs in `bin/` |
| **CI / tests** | Minimal; not a modern test suite |
| **Performance** | Single-threaded agent loop; no GPU; 2000s-era integrator path |
| **Community** | ~39 stars, ~15 forks, no releases on GitHub; abandoned for research use |

**Verdict:** Getting the *legacy* tree to build on 2026 Linux is possible with effort but is a maintenance tar pit. Long-term success = **preserve API spirit + demos**, reimplement the engine on a modern stack, keep original as `legacy/` reference.

---

## 5. Modernization roadmap

### Phase 0 — Foundation ✅ done

- [x] Clone and inventory upstream
- [x] `REVIVAL.md` + modern `README.md`
- [x] Strategy locked (§9)
- [x] License / attribution (GPL-2.0-or-later, `CITATION.cff`)
- [x] Original tree moved to `legacy/`

### Phase 1 — “Hello agents today” ✅ mostly done

**Goal:** A runnable modern slice that feels like breve.

- [x] Minimal **Python 3** package `breve`:
  - `Control`, `Mobile`, `Stationary`, `Floor`, vectors, shapes
  - Fixed-timestep loop, sphere collision callbacks
  - Neighborhood queries for flocking
  - Headless mode + optional pyglet 2D view
- [x] Demos: HelloWorld, Fountain, Gatherers, **Swarm**
- [x] `pyproject.toml`, `pip install -e .`
- [x] CI workflow (pytest + headless demos)

### Phase 2 — Core fidelity (1–2 months)

- Rigid-body physics (Rapier via Rust or pure Python fallback)
- Shapes: sphere, box, cylinder, mesh import (glTF)
- Camera, lighting, selection, basic UI (menus → simple GUI or TUI)
- Spatial hash / broadphase for thousands of agents
- Port Physics-Examples (Gravity, Walker lite) and Braitenberg

### Phase 3 — Performance & scale (ongoing)

- Rust core for world step, collisions, optional GPU particles
- Parallel agent `iterate` where safe (data-oriented or ECS)
- Recording / replay, headless batch for evolution experiments
- Optional Bevy viewer process or wgpu embedded window

### Phase 4 — Ecosystem & impact

- Jupyter / notebook demos; web viewer (WASM or remote stream)
- Gymnasium / PettingZoo env wrappers for RL research
- Package on PyPI + `cargo` crates; conda-forge optional
- Curriculum: “ALife in an afternoon” workshop
- Community: discussions, showcase gallery, citation file (`CITATION.cff`)

### Parallel track (optional, lower priority)

**Legacy build archaeology:** Docker image that builds original C++ against pinned ODE/Python2 — for historians and porting reference only, not the main product.

---

## 6. Tech choices (recommendations)

| Concern | Recommendation | Why |
|---------|----------------|-----|
| **Agent API language** | **Python 3** first-class | Matches original demos, education, ML/ALife ecosystem |
| **Sim core** | Start pure Python; extract hot paths to **Rust** | Fast iteration first; Rust where profiling demands |
| **Physics** | **Rapier** (Rust) via PyO3, or **PyBullet** short-term | Modern, maintained; ODE is stagnant for our UX |
| **3D rendering** | Phase 1: **pyglet** or **moderngl** + simple camera; Phase 2+: **Bevy** (Rust) or **wgpu** | Avoid Unity lock-in; keep OSS; Bevy if we go deep Rust |
| **Bindings** | **PyO3** + **maturin** | Best Rust↔Python story in 2026 |
| **Packaging** | `pyproject.toml` + optional workspace Cargo crates | `pip install breve` dream |
| **steve language** | **Do not port initially** | Python demos cover the spirit; steve is a museum piece |
| **IDE (wx/Qt)** | Defer; use editor + live-reload or minimal egui/Bevy UI | Original IDE is huge surface area |
| **GPU** | Optional later (compute for particles / flocking) | Not blocking educational use |

### Architecture sketch (target)

```
┌─────────────────────────────────────────────┐
│  Python user sims (Control / Mobile / …)    │
├─────────────────────────────────────────────┤
│  breve Python package (API + demos)         │
├──────────────────┬──────────────────────────┤
│  Headless engine │  Viewer (pyglet / Bevy)  │
├──────────────────┴──────────────────────────┤
│  Core (Python → Rust): world, step, collide │
│  Physics: Rapier                            │
└─────────────────────────────────────────────┘
```

**Not recommended as primary path:** full rewrite of legacy OpenGL/GLUT/ODE C++ with autotools. Use it as **oracle** for behavior and demo semantics.

---

## 7. Quick wins vs bigger improvements

### Quick wins (days–1 week)

1. Modern README + this roadmap (clarity for collaborators)
2. Python 3 API skeleton + HelloWorld / Fountain without physics
3. Headless smoke tests in CI
4. Catalog demos with 1-line descriptions (`demos/INDEX.md`)
5. `CITATION.cff` + clear Jon Klein attribution
6. Strip or quarantine binary blobs from the revival track (`bin/*.dll`, `lib/python2.*`)

### Bigger improvements (weeks–months)

1. Full physics parity with Walker / MultiBody demos
2. Rust performance core + 10k–100k agent flocking
3. RL / evolutionary experiment harness
4. Web or notebook visualization
5. Sensor suite (IR, light, range image) for robotics-ish curricula
6. Optional steve interpreter only if community demands nostalgia

---

## 8. First actionable steps (ordered)

1. **Decide fork strategy** (see questions below)
2. Scaffold modern package:
   - `python/breve/` or top-level `breve_ng/` with pyproject
   - Move original sources under `legacy/` *or* keep monorepo with clear root README
3. Implement minimal engine + 2 demos
4. Add CI and a 5-minute tutorial

---

## 9. Decisions (locked 2026-07-17)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Product name | **`breve`** | Continuity, searchability, respect for lineage |
| Layout | **Modern at repo root + `legacy/`** | Installable product first; museum tree preserved |
| Rendering | **Headless-first + optional pyglet 2D**; full 3D later | Science/CI works everywhere; fun path available |
| Physics | **Kinematic now → Rapier (Rust/PyO3) next** | Avoid ODE tar pit; ship flocking demos today |
| License (new code) | **GPL-2.0-or-later** | Compatible with original GPL-2 |
| steve / wx IDE | **Not ported initially** | High cost, low revival ROI |
| Branch | **`revival`** | Non-destructive relative to upstream `master` |

---

## 10. Success criteria (revival is “working”)

- [x] `pip install -e .` on Linux; run flocking demo headless in < 5 minutes
- [x] Core demos: HelloWorld, Fountain, Gatherers, Swarm
- [ ] At least 5 more classic demos ported and documented
- [ ] Headless batch mode for GA/evolution experiments
- [x] Clear docs + roadmap (`README.md`, `REVIVAL.md`, `demos/INDEX.md`)
- [x] CI on Python 3.10 / 3.12
- [ ] Someone who never used breve can have fun in one evening (3D polish remaining)

---

## 11. Immediate next work (ordered)

1. ~~True 3D viewer (moderngl-window)~~ **done**
2. ~~Rigid-body physics (pure Python solver)~~ **done** — Gravity + Stack demos
3. Joints / MultiBody / Walker-lite (or Rapier backend when joints matter)
4. Spatial hash for neighborhoods (scale past ~500 agents)
5. Port Braitenberg + PatchLife (curriculum demos)
6. Publish package to TestPyPI
7. Better materials/lighting / optional Bevy sidecar later

---

*Original software © 2000–2007 Jonathan Klein (and later contributors). Revival notes for the open-source resurrection effort.*
