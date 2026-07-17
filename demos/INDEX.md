# Demo index

## Modern (Python 3 — run from repo root after `pip install -e .`)

| Demo | Command | What it shows |
|------|---------|----------------|
| Hello World | `python demos/hello_world.py` | Minimal Control / iterate loop |
| Fountain | `python demos/fountain.py --steps 50` | Particles under gravity |
| Gatherers | `python demos/gatherers.py` | Collision-driven pickup |
| **Swarm** | `python demos/swarm.py --steps 200` | Boids flocking (classic showpiece) |
| Swarm **3D** | `python demos/swarm.py --viz` | Interactive 3D orbit view (`pip install -e '.[viz]'`) |
| **Bouncy** | `python demos/bouncy.py --viz` | Gravity arcs + elastic bounces (best physics demo) |
| **Gravity** | `python demos/gravity.py --viz` | Balls bounce down a staircase |
| **Stack** | `python demos/stack.py --viz` | Lobbed wrecking ball + tower |

### Swarm 3D controls (`--viz`)

| Input | Action |
|-------|--------|
| Drag | Orbit camera |
| Scroll | Zoom |
| Space | Pause / resume |
| A | Toggle auto-orbit |
| N / O / W | Flock normal / obedient / wacky |
| S | Squish birds to origin |
| R | Reset camera |
| Esc | Quit |

## Classic (legacy engine — reference only)

Hundreds of demos live under `legacy/demos/` (Python 2 / steve). Highlights:

- `Getting-Started/` — HelloWorld, Fountain, Gravity, RandomWalker
- `Swarm/` — flocking + evolution variants
- `Physics-Examples/` — Walker, Demolition, springs
- `Braitenberg/` — vehicle sensors
- `Chemistry/` — Gray-Scott, hypercycle
- `Push/` — PushGP integration
