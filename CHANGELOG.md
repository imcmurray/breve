# Changelog

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
