# breve

**3D multi-agent / artificial-life simulation — modern revival.**

Classic [breve](http://www.spiderland.org/) (Jon Klein, ~2000–2015) made continuous 3D agent worlds easy to script. This repository is an active resurrection: a **Python 3** engine that keeps the original *spirit* (subclass `Control` + `Mobile`, implement `iterate` / collisions) while the C++/Python 2 sources live under `legacy/` for reference.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python demos/swarm.py --steps 150
```

Optional **interactive 3D** view (orbit camera, velocity-aligned agents):

```bash
pip install -e ".[viz]"
python demos/swarm.py --viz
# drag = orbit · scroll = zoom · SPACE = pause · A = auto-orbit · ESC = quit
```

## Why breve?

| Tool | Fit |
|------|-----|
| NetLogo / Mesa | Great for 2D grids & ABM — not continuous 3D physics toys |
| Unity ML-Agents / MuJoCo | Powerful, heavy, not “script a flock in 50 lines” |
| **breve** | Lightweight continuous 3D agents, educational, fun, emergent |

## Quick example

```python
import breve
from breve.engine import Engine, set_engine

set_engine(Engine())

class Hello(breve.Control):
    def iterate(self):
        print("Hello, world!")
        super().iterate()

Hello().run(steps=3)
```

## AI scene builder (describe a world in English)

Talk to **Grok** (xAI) and it builds a safe JSON scene — floors, balls, masses, flocks — then you open it in 3D.

```bash
pip install -e ".[ai,viz]"
export XAI_API_KEY=xai-...          # https://console.x.ai
# optional: cp .env.example .env

# one-shot
breve-ai "heavy red ball and light yellow balls bouncing on stairs so I can see gravity" --viz

# interactive
breve-ai
# breve-ai> flock of 40 blue birds
# breve-ai> viz
# breve-ai> make gravity stronger
# breve-ai> save

# no AI key needed — run a hand-written scene
breve-ai --load scenes/example_gravity.json --viz
```

How it works: the model only emits **declarative JSON** (not arbitrary code). Breve validates and builds the sim. See `python/breve/scene.py` for the schema.

## Demos

| Demo | Command |
|------|---------|
| Hello World | `python demos/hello_world.py` |
| Fountain | `python demos/fountain.py --steps 50` |
| Gatherers | `python demos/gatherers.py` |
| **Swarm (boids)** | `python demos/swarm.py --steps 200` |
| Swarm + **3D** | `python demos/swarm.py --viz` |
| **Bouncy** (physics) | `python demos/bouncy.py --viz` |
| **Gravity** (staircase) | `python demos/gravity.py --viz` |
| **Stack** (wrecking ball) | `python demos/stack.py --viz` |

Full list: [`demos/INDEX.md`](demos/INDEX.md)

## Layout

```
python/breve/     # installable engine (Python 3)
demos/            # modern demos
tests/
legacy/           # original C++ / steve / Python 2 tree (museum)
REVIVAL.md        # roadmap & architecture decisions
```

## Status

| Feature | State |
|---------|--------|
| Control / Mobile / Stationary / Floor | done |
| Kinematic integration, collisions | done |
| Neighborhoods (flocking) | done |
| Swarm, Fountain, Gatherers demos | done |
| Interactive **3D** viewer (moderngl) | done |
| **Rigid-body physics** (pure Python) | done |
| Gravity + Stack demos | done |
| Joints / Walker / Rapier backend | next |
| steve language | later (legacy only) |

Roadmap and decisions: **[`REVIVAL.md`](REVIVAL.md)**

## License

GPL-2.0-or-later. Original breve © Jonathan Klein and contributors. See `LICENSE` / `GPL.txt`.

## Citation

If you use breve in research, please cite the original work, e.g. Klein, J. (2002/2003), *BREVE: a 3D environment for the simulation of decentralized systems and artificial life*, and note this revival build.
