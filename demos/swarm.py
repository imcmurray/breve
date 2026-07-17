#!/usr/bin/env python3
"""
Swarm / boids flocking — port of classic demos/Swarm/Swarm.py.

Reynolds-style separation, alignment, cohesion + world centering + wander.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import breve
from breve.engine import Engine, set_engine


class Swarm(breve.Control):
    def __init__(self, n_birds: int = 80, mode: str = "normal"):
        self.birds = breve.object_list()
        self.selection = None
        self.mode = mode
        self._n_birds = n_birds
        super().__init__()

    def init(self):
        self.enable_lighting()
        self.enable_smooth_drawing()
        self.move_light(breve.vector(0, 20, 20))
        # No Floor in the modern port: classic landing washed out flocking in
        # headless runs. Pure airborne boids are clearer for teaching.
        self.birds = breve.create_instances(Bird, self._n_birds)
        if self.mode == "obedient":
            self.flock_obediently()
        elif self.mode == "wacky":
            self.flock_wackily()
        else:
            self.flock_normally()
        self.offset_camera(breve.vector(5, 1.5, 6))
        self.set_iteration_step(0.05)
        self.set_integration_step(0.025)
        print(f"Swarm: {len(self.birds)} birds  mode={self.mode}")

    def flock_normally(self):
        self.mode = "normal"
        self.birds.flock_normally()

    def flock_obediently(self):
        self.mode = "obedient"
        self.birds.flock_obediently()

    def flock_wackily(self):
        self.mode = "wacky"
        self.birds.flock_wackily()

    def squish(self):
        self.birds.move(breve.vector(0, 2, 0))

    def iterate(self):
        self.update_neighbors()
        centroid = breve.vector(0, 0, 0)
        for bird in self.birds:
            bird.fly()
            centroid = centroid + bird.get_location()

        n = breve.length(self.birds)
        if n:
            centroid = centroid / n

        top_diff = 0.0
        for bird in self.birds:
            d = breve.length(centroid - bird.get_location())
            if d > top_diff:
                top_diff = d

        self.aim_camera(centroid)
        self.zoom_camera(0.5 * top_diff + 10)

        if int(self.engine.time * 10) % 20 == 0:
            speeds = [breve.length(b.get_velocity()) for b in self.birds]
            avg = sum(speeds) / max(len(speeds), 1)
            landed = sum(1 for b in self.birds if b.landed)
            print(
                f"t={self.engine.time:5.1f}s  centroid={centroid}  "
                f"avg_speed={avg:5.2f}  spread={top_diff:5.1f}  landed={landed}  [{self.mode}]"
            )
        super().iterate()


class Bird(breve.Mobile):
    def __init__(self):
        self.center_constant = 0.0
        self.cruise_distance = 0.0
        self.landed = 0
        self.max_acceleration = 0.0
        self.max_velocity = 0.0
        self.spacing_constant = 0.0
        self.velocity_constant = 0.0
        self.wander_constant = 0.0
        self.world_center_constant = 0.0
        super().__init__()

    def init(self):
        self.set_shape(breve.Sphere().init_with(0.25))
        # tint each bird slightly
        self.set_color(
            breve.vector(0.7, 0.75, 1.0)
            + breve.random_expression(breve.vector(0.25, 0.2, 0.0))
        )
        self.move(
            breve.random_expression(breve.vector(10, 10, 10)) - breve.vector(5, -5, 5)
        )
        self.set_velocity(
            breve.random_expression(breve.vector(20, 20, 20)) - breve.vector(10, 10, 10)
        )
        self.set_neighborhood_size(2.0)

    def check_landed(self) -> int:
        return self.landed

    def check_visibility(self, item) -> bool:
        if item is self:
            return False
        if not item.is_a("Bird"):
            return False
        if item.check_landed():
            return False
        if self.get_angle(item) > 2.0:
            return False
        return True

    def flock_normally(self):
        self.wander_constant = 4.0
        self.world_center_constant = 5.0
        self.center_constant = 2.0
        self.velocity_constant = 2.0
        self.spacing_constant = 5.0
        self.max_velocity = 15
        self.max_acceleration = 15
        self.cruise_distance = 0.4
        self.max_speed = float(self.max_velocity)

    def flock_obediently(self):
        self.wander_constant = 6.0
        self.world_center_constant = 6.0
        self.center_constant = 2.0
        self.velocity_constant = 3.0
        self.spacing_constant = 4.0
        self.max_velocity = 16
        self.max_acceleration = 20
        self.cruise_distance = 1.0
        self.max_speed = float(self.max_velocity)

    def flock_wackily(self):
        self.wander_constant = 8.0
        self.world_center_constant = 14.0
        self.center_constant = 1.0
        self.velocity_constant = 3.0
        self.spacing_constant = 4.0
        self.max_velocity = 20
        self.max_acceleration = 30
        self.cruise_distance = 0.5
        self.max_speed = float(self.max_velocity)

    def fly(self):
        neighbors = [b for b in self.get_neighbors() if self.check_visibility(b)]

        if self.landed:
            if np.random.randint(0, 40) == 1:
                self.landed = 0
                self.set_velocity(
                    breve.random_expression(breve.vector(0.1, 1.1, 0.1))
                    - breve.vector(0.05, 0, 0.05)
                )
            else:
                return

        center_urge = self.get_center_urge(neighbors)
        velocity_urge = self.get_velocity_urge(neighbors)
        spacing_urge = breve.vector(0, 0, 0)
        for bird in neighbors:
            to_neighbor = self.get_location() - bird.get_location()
            if breve.length(to_neighbor) < self.cruise_distance:
                spacing_urge = spacing_urge + to_neighbor

        world_center_urge = breve.vector(0, 0, 0)
        if breve.length(self.get_location()) > 10:
            world_center_urge = -self.get_location()

        wander_urge = breve.random_expression(breve.vector(2, 2, 2)) - breve.vector(1, 1, 1)

        def _unit(v):
            n = breve.length(v)
            return v / n if n else breve.vector(0, 0, 0)

        spacing_urge = _unit(spacing_urge)
        world_center_urge = _unit(world_center_urge)
        velocity_urge = _unit(velocity_urge)
        center_urge = _unit(center_urge)
        wander_urge = _unit(wander_urge)

        acceleration = (
            world_center_urge * self.world_center_constant
            + center_urge * self.center_constant
            + velocity_urge * self.velocity_constant
            + spacing_urge * self.spacing_constant
            + wander_urge * self.wander_constant
        )
        if breve.length(acceleration) != 0:
            acceleration = _unit(acceleration)

        self.set_acceleration(acceleration * self.max_acceleration)
        v = self.get_velocity()
        if breve.length(v) > self.max_velocity:
            self.set_velocity(_unit(v) * self.max_velocity)

    def get_center_urge(self, flock):
        if not flock:
            return breve.vector(0, 0, 0)
        center = breve.vector(0, 0, 0)
        for item in flock:
            center = center + item.get_location()
        center = center / len(flock)
        return center - self.get_location()

    def get_velocity_urge(self, flock):
        if not flock:
            return breve.vector(0, 0, 0)
        ave = breve.vector(0, 0, 0)
        for item in flock:
            ave = ave + item.get_velocity()
        ave = ave / len(flock)
        return ave - self.get_velocity()

    def land(self, _ground):
        self.set_acceleration(breve.vector(0, 0, 0))
        self.set_velocity(breve.vector(0, 0, 0))
        self.landed = 1
        self.offset(breve.vector(0, 0.01, 0))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Breve swarm / boids demo")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--birds", type=int, default=80)
    parser.add_argument("--mode", choices=("normal", "obedient", "wacky"), default="normal")
    parser.add_argument("--viz", action="store_true", help="Open 2D projection viewer (pyglet)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    if args.seed is not None:
        np.random.seed(args.seed)

    set_engine(Engine())
    sim = Swarm(n_birds=args.birds, mode=args.mode)

    if args.viz:
        try:
            from breve.viz_pyglet import run_with_viewer

            run_with_viewer(sim, steps=args.steps)
            return
        except ImportError:
            print("pyglet not installed; pip install -e '.[viz]'", file=sys.stderr)
            print("Falling back to headless.", file=sys.stderr)

    sim.run(steps=args.steps)
    print("done.")


if __name__ == "__main__":
    main()
