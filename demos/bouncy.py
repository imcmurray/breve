#!/usr/bin/env python3
"""
Bouncy — gravity arcs + elastic bounces with *different masses*.

Under gravity alone all balls accelerate the same (Galileo). Mass shows up in
*collisions*: light balls ricochet off heavy ones; heavy balls plow through.
Size and color encode weight so you can read it in the 3D view.
"""

from __future__ import annotations

import argparse
import sys

import breve
from breve.engine import Engine, set_engine

# name, mass (kg-ish), radius, color (RGB 0-1)
# Big/dark = heavy; small/bright = light
BALL_SPECS = [
    ("feather", 0.25, 0.16, breve.vector(0.95, 0.9, 0.4)),   # light yellow
    ("ping", 0.40, 0.18, breve.vector(0.4, 0.95, 0.95)),      # light cyan
    ("tennis", 1.0, 0.24, breve.vector(0.3, 0.85, 0.35)),     # medium green
    ("baseball", 1.5, 0.26, breve.vector(0.95, 0.55, 0.25)),  # medium orange
    ("bowling", 6.0, 0.38, breve.vector(0.25, 0.35, 0.85)),  # heavy blue
    ("cannon", 12.0, 0.45, breve.vector(0.75, 0.15, 0.15)),   # very heavy red
]


class Bouncy(breve.PhysicalControl):
    def __init__(self):
        self.balls = None
        self._last_print = -1.0
        super().__init__()

    def init(self):
        self.set_integration_step(0.004)
        self.set_iteration_step(0.016)
        self.full_gravity()
        self.enable_lighting()
        self.set_background_color(breve.vector(0.08, 0.1, 0.14))

        floor = breve.Stationary()
        floor.set_shape(breve.Box().init_with(breve.vector(14, 0.2, 10)))
        floor.move(breve.vector(0, -0.1, 0))
        floor.set_color(breve.vector(0.3, 0.35, 0.4))
        _static(floor, restitution=0.82, friction=0.18)

        for x, sx in ((-6.5, 0.25), (6.5, 0.25)):
            w = breve.Stationary()
            w.set_shape(breve.Box().init_with(breve.vector(sx, 4.0, 10)))
            w.move(breve.vector(x, 1.9, 0))
            w.set_color(breve.vector(0.4, 0.45, 0.5))
            _static(w, restitution=0.8, friction=0.1)

        for z, sz in ((-4.5, 0.25), (4.5, 0.25)):
            w = breve.Stationary()
            w.set_shape(breve.Box().init_with(breve.vector(14, 4.0, sz)))
            w.move(breve.vector(0, 1.9, z))
            w.set_color(breve.vector(0.35, 0.4, 0.48))
            _static(w, restitution=0.8, friction=0.1)

        mid = breve.Stationary()
        mid.set_shape(breve.Box().init_with(breve.vector(3.0, 0.15, 2.0)))
        mid.move(breve.vector(0, 1.0, 0))
        mid.set_color(breve.vector(0.55, 0.45, 0.35))
        _static(mid, restitution=0.75, friction=0.15)

        self.balls = breve.object_list()
        self._spawn_volley(initial=True)

        self.point_camera(breve.vector(0, 1.2, 0), breve.vector(10, 6, 12))
        self.camera_zoom = 14.0
        print("Bouncy — different masses (size ≈ weight):")
        for name, mass, radius, _col in BALL_SPECS:
            print(f"  {name:10s}  mass={mass:5.2f}  radius={radius:.2f}")
        print("Heavy (big red/blue) barely budge when light ones hit them.")
        print("Light (small yellow/cyan) fly off after collisions.")

    def _spawn_volley(self, initial: bool = False):
        """Drop all mass classes with similar initial speeds so mass shows in impacts."""
        if not initial:
            for b in list(self.balls):
                b.remove()
            self.balls = breve.object_list()

        # Two groups heading toward each other → clear mass contrast on impact
        n = len(BALL_SPECS)
        for i, (name, mass, radius, color) in enumerate(BALL_SPECS):
            b = BounceBall()
            # left squad goes right, right squad goes left
            if i < n // 2:
                loc = breve.vector(-4.5 + i * 0.15, 3.2 + i * 0.35, -0.8 + i * 0.3)
                vel = breve.vector(4.5, 1.5, 0.8)
            else:
                j = i - n // 2
                loc = breve.vector(4.5 - j * 0.15, 3.0 + j * 0.35, 0.8 - j * 0.3)
                vel = breve.vector(-4.5, 1.5, -0.8)
            b.setup(loc=loc, vel=vel, color=color, radius=radius, mass=mass, name=name)
            self.balls.append(b)

    def iterate(self):
        if self.balls:
            cx = sum(b.location.x for b in self.balls) / len(self.balls)
            cy = sum(b.location.y for b in self.balls) / len(self.balls)
            self.aim_camera(breve.vector(cx * 0.25, max(0.5, cy * 0.45), 0))

        if self.engine.time - self._last_print >= 0.6:
            self._last_print = self.engine.time
            # Per-ball speeds so mass differences are visible in the log
            parts = []
            for b in self.balls:
                parts.append(f"{b.label}:{breve.length(b.velocity):4.1f}")
            print(f"t={self.engine.time:5.1f}s  speeds[name:|v|]  " + "  ".join(parts))

            avg = sum(breve.length(b.velocity) for b in self.balls) / max(len(self.balls), 1)
            if avg < 0.2 and self.engine.time > 2.5:
                self._spawn_volley()
                print("  → re-tossed (watch heavy vs light on impact)")
        super().iterate()


class BounceBall(breve.Mobile):
    def setup(self, loc, vel, color, radius=0.25, mass=1.0, name="ball"):
        self.label = name
        self.mass = float(mass)
        self.set_shape(breve.Sphere().init_with(radius))
        self.set_color(color)
        self.move(loc)
        self.set_velocity(vel)
        self.enable_physics(mass=self.mass)
        body = get_engine().physics.get_body(self)
        if body is not None:
            # Slightly bouncier light balls; heavy still elastic but not super-ball
            body.restitution = 0.92 if mass < 1.0 else (0.8 if mass < 4.0 else 0.7)
            body.friction = 0.1 if mass < 1.0 else 0.2
            body.position[:] = [loc.x, loc.y, loc.z]
            body.velocity[:] = [vel.x, vel.y, vel.z]
            body.mass = self.mass
            body.inv_mass = 0.0 if body.static else 1.0 / max(self.mass, 1e-6)
            body.awake = True
        return self


def _static(obj, restitution=0.8, friction=0.2):
    eng = get_engine()
    eng.register_physics_body(obj, static=True)
    body = eng.physics.get_body(obj)
    if body:
        body.restitution = restitution
        body.friction = friction


def get_engine():
    return breve.get_engine()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Breve bouncy balls with different masses")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    if args.seed is not None:
        import numpy as np

        np.random.seed(args.seed)

    steps = args.steps if args.steps is not None else (None if args.viz else 400)

    set_engine(Engine())
    sim = Bouncy()

    if args.viz:
        try:
            from breve.viz import run_with_viewer

            print("3D: drag=orbit  scroll=zoom  SPACE=pause  A=auto-orbit  ESC=quit")
            run_with_viewer(sim, steps=steps)
            return
        except ImportError:
            print("viz deps missing; pip install -e '.[viz]'", file=sys.stderr)

    sim.run(steps=steps if steps is not None else 400)
    print("done.")


if __name__ == "__main__":
    main()
