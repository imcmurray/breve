# Changelog

## [0.3.0] — 2026-07-18

### Physics — natural resting behavior
- **Boxes no longer freeze balanced on a corner or edge.** Sleeping now
  requires a *stable support configuration*: the COM's gravity projection must
  lie inside the convex hull of the body's supporting contact points
  (corner → point, edge → segment, face → polygon). An unsupported tilt keeps
  tipping under gravity until it rests on a face; a genuinely balanced edge
  pose (COM directly over the edge) may still rest.
- Rest damping only applies to stably supported bodies — it no longer stalls
  a slow tip below the sleep threshold (the frozen-on-a-corner bug).
- Sleeping bodies are solved as immovable until woken (no more gravity-bias
  skating); they wake when support is lost *or* becomes unstable, and when
  hit by energetic bodies (a tipping base re-awakens the stack above it).
- Hard depenetration vs static top surfaces (both paths): multi-body shoves
  can no longer tunnel bodies through floors, steps, or pads.
- Floor-normal orientation guard: contacts past a slab's midplane can no
  longer point the SAT normal into the floor (boxes and spheres).
- Pure-Python and Numba paths share the rest/sleep gate — behavior parity by
  construction; both covered by tests.

### Tests
- New regressions: corner-tilted cube settles on a face (both paths), edge
  tilts 25–35°, upright rest, legitimate 45° edge balance, two-box stack,
  cube dropped onto cube — all assert settled pose, rest height, low residual
  velocity, and no NaNs.

### Demos
- Pyramid/tower: larger ground slabs, slightly grippier low-bounce boxes.

## [0.2.1] — 2026-07-18

### Performance
- Optional **Numba** CPU JIT for integrate + contact resolve (`pip install 'breve[fast]'` or `breve[webfast]`)
- Disable with `BREVE_NUMBA=0`; `/api/status` reports `numba_physics`

## [0.2.0] — 2026-07-18

### Web product
- **Lab controls** in the sidebar: sim speed, gravity, population (extra boxes/balls), ball/box mass min–max, bounce, friction, launch velocity, scatter, size variety
- Per-demo **localStorage** — reopening the same curriculum demo restores your sliders
- Tweaks **restart the sim** on change (speed is live without rebuild)
- Compact accordion UI; Grok chat and API settings collapse out of the way
- **Auto-pause** when the world settles; **cull** bodies that fall out of view (both toggleable)

### Physics
- Multi-point contact manifolds so stacks settle and spin dies
- Rest-hop fix (no restitution/bias on soft contacts)
- Deep-overlap position projection so piles don’t sink into each other
- ~10× faster pure-Python solver (scalar hot path) for realtime web demos
- Edge tip from COM / torque (no special-case edge filters)

### Demos
- Multi-layer **funnel** with transparent outer walls
- Curriculum blurb and README updates for lab controls and funnel

### Package
- Version bump to **0.2.0**

## [0.1.0-alpha] — earlier

Initial revival: Python 3 engine, FastAPI + three.js web UI, declarative scenes, AI scene builder, 10 curriculum demos, fork credit for original breve.
