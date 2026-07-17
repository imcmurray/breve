#!/usr/bin/env python3
"""
Fountain demo — kinematic particles under gravity (modern breve).

Inspired by demos/Getting-Started/Fountain.py from classic breve.
"""

from __future__ import annotations

import argparse
import sys

import breve
from breve.engine import Engine, set_engine


class Fountain(breve.Control):
    def __init__(self):
        self.particles = None
        super().__init__()

    def init(self):
        self.set_integration_step(0.05)
        self.set_iteration_step(0.1)
        self.enable_lighting()
        self.set_background_color(breve.vector(0, 0, 0))
        self.particles = breve.create_instances(Particle, 80)
        self.point_camera(breve.vector(0, 9, 0), breve.vector(40, 2, 0))
        print(f"Fountain: {len(self.particles)} particles")

    def iterate(self):
        alive = sum(1 for p in self.particles if p.location.y > -6)
        if int(self.engine.time * 10) % 10 == 0:
            print(f"t={self.engine.time:5.1f}s  airborne≈{alive}")
        super().iterate()


class Particle(breve.Mobile):
    def __init__(self):
        self.range = 3.0
        super().__init__()

    def init(self):
        size = breve.random_expression(breve.vector(0.5, 0.5, 0.5)) + breve.vector(0.1, 0.1, 0.1)
        self.set_shape(breve.Cube().init_with(size))
        self.set_acceleration(breve.vector(0, -9.8, 0))
        self.reset()

    def iterate(self):
        if self.location.y < -6.0:
            self.reset()

    def reset(self):
        self.set_color(breve.random_expression(breve.vector(0, 1, 1)))
        self.move(breve.vector(0, 0, 0))
        r = self.range
        self.set_velocity(
            breve.random_expression(breve.vector(2 * r, 20, 2 * r))
            + breve.vector(-r, 4, -r)
        )
        self.set_rotational_velocity(breve.random_expression(breve.vector(0.6, 0.6, 0.6)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Breve fountain demo")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--viz", action="store_true", help="Open a pyglet window if available")
    args = parser.parse_args(argv)

    set_engine(Engine())
    sim = Fountain()

    if args.viz:
        try:
            from breve.viz_pyglet import run_with_viewer

            run_with_viewer(sim, steps=args.steps)
            return
        except ImportError:
            print("pyglet not installed; run: pip install -e '.[viz]'", file=sys.stderr)
            print("Falling back to headless.", file=sys.stderr)

    sim.run(steps=args.steps)
    print("done.")


if __name__ == "__main__":
    main()
