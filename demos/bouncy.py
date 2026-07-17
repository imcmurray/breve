#!/usr/bin/env python3
"""
Bouncy — clear gravity arcs + elastic bounces (the “yes, gravity works” demo).

Several colorful spheres drop with horizontal velocity onto a tilted floor and
walls so they keep hopping around. No auto-teleport; pure physics.
"""

from __future__ import annotations

import argparse
import sys

import breve
from breve.engine import Engine, set_engine


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

        # Floor
        floor = breve.Stationary()
        floor.set_shape(breve.Box().init_with(breve.vector(14, 0.2, 10)))
        floor.move(breve.vector(0, -0.1, 0))
        floor.set_color(breve.vector(0.3, 0.35, 0.4))
        _static(floor, restitution=0.85, friction=0.2)

        # Side walls so balls stay in view and bounce sideways
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

        # A couple of platforms / ramps (boxes at different heights)
        mid = breve.Stationary()
        mid.set_shape(breve.Box().init_with(breve.vector(3.0, 0.15, 2.0)))
        mid.move(breve.vector(0, 1.0, 0))
        mid.set_color(breve.vector(0.55, 0.45, 0.35))
        _static(mid, restitution=0.75, friction=0.15)

        self.balls = breve.object_list()
        colors = [
            breve.vector(0.95, 0.25, 0.2),
            breve.vector(0.25, 0.85, 0.4),
            breve.vector(0.3, 0.5, 1.0),
            breve.vector(1.0, 0.85, 0.2),
            breve.vector(0.9, 0.4, 0.9),
            breve.vector(0.4, 0.95, 0.95),
        ]
        for i, col in enumerate(colors):
            b = BounceBall()
            b.setup(
                loc=breve.vector(-4 + i * 0.3, 3.5 + i * 0.4, -1 + (i % 3) * 0.5),
                vel=breve.vector(2.5 + i * 0.3, 1.0, (i % 2) * 1.5 - 0.5),
                color=col,
                radius=0.22 + (i % 3) * 0.04,
            )
            self.balls.append(b)

        self.point_camera(breve.vector(0, 1.2, 0), breve.vector(10, 6, 12))
        self.camera_zoom = 14.0
        print(f"Bouncy: {len(self.balls)} balls under gravity — watch them arc and bounce")
        print("No re-spawns for a few seconds; pure physics. SPACE pauses.")

    def iterate(self):
        if self.balls:
            cx = sum(b.location.x for b in self.balls) / len(self.balls)
            cy = sum(b.location.y for b in self.balls) / len(self.balls)
            self.aim_camera(breve.vector(cx * 0.3, max(0.5, cy * 0.5), 0))

        # Soft re-drop only if everything is basically dead for a while
        if self.engine.time - self._last_print >= 0.5:
            self._last_print = self.engine.time
            speeds = [breve.length(b.velocity) for b in self.balls]
            avg = sum(speeds) / max(len(speeds), 1)
            print(
                f"t={self.engine.time:5.1f}s  speed_avg={avg:5.2f}  "
                f"y_avg={sum(b.location.y for b in self.balls)/len(self.balls):5.2f}"
            )
            if avg < 0.15 and self.engine.time > 2.0:
                for i, b in enumerate(self.balls):
                    b.setup(
                        loc=breve.vector(-3 + i * 0.4, 4.0 + (i % 3) * 0.5, (i % 3) - 1),
                        vel=breve.vector(3.0, 2.0, (i % 2) * 2 - 1),
                        color=b.color,
                        radius=b.shape.radius if b.shape else 0.25,
                    )
                print("  → re-tossed (everything had stopped)")
        super().iterate()


class BounceBall(breve.Mobile):
    def setup(self, loc, vel, color, radius=0.25):
        self.set_shape(breve.Sphere().init_with(radius))
        self.set_color(color)
        self.move(loc)
        self.set_velocity(vel)
        self.mass = 1.0
        self.enable_physics(mass=self.mass)
        body = get_engine().physics.get_body(self)
        if body is not None:
            body.restitution = 0.88
            body.friction = 0.12
            body.position[:] = [loc.x, loc.y, loc.z]
            body.velocity[:] = [vel.x, vel.y, vel.z]
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
    parser = argparse.ArgumentParser(description="Breve bouncy balls under gravity")
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
