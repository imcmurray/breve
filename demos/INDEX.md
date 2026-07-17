# Demo index

## Modern (Python 3 — run from repo root after `pip install -e .`)

| Demo | Command | What it shows |
|------|---------|----------------|
| Hello World | `python demos/hello_world.py` | Minimal Control / iterate loop |
| Fountain | `python demos/fountain.py --steps 50` | Particles under gravity |
| Gatherers | `python demos/gatherers.py` | Collision-driven pickup |
| **Swarm** | `python demos/swarm.py --steps 200` | Boids flocking (classic showpiece) |
| Swarm (viz) | `python demos/swarm.py --viz` | Top-down live view (needs `pip install -e '.[viz]'`) |

### Swarm keys (with `--viz`)

| Key | Action |
|-----|--------|
| Space | Pause / resume |
| N / O / W | Flock normal / obedient / wacky |
| S | Squish birds to origin |
| Esc | Quit |

## Classic (legacy engine — reference only)

Hundreds of demos live under `legacy/demos/` (Python 2 / steve). Highlights:

- `Getting-Started/` — HelloWorld, Fountain, Gravity, RandomWalker
- `Swarm/` — flocking + evolution variants
- `Physics-Examples/` — Walker, Demolition, springs
- `Braitenberg/` — vehicle sensors
- `Chemistry/` — Gray-Scott, hypercycle
- `Push/` — PushGP integration
