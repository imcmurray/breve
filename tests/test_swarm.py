"""Swarm / neighborhood tests."""

from __future__ import annotations

import breve
from breve.engine import Engine, set_engine


def test_neighbors_update():
    set_engine(Engine())

    class A(breve.Mobile):
        def init(self):
            self.set_neighborhood_size(2.0)
            self.set_shape(breve.Sphere().init_with(0.1))

    class C(breve.Control):
        def init(self):
            self.a = A()
            self.b = A()
            self.a.move(breve.vector(0, 0, 0))
            self.b.move(breve.vector(1, 0, 0))
            self.c = A()
            self.c.move(breve.vector(10, 0, 0))

    ctrl = C()
    ctrl.update_neighbors()
    n_a = ctrl.a.get_neighbors()
    assert ctrl.b in n_a
    assert ctrl.c not in n_a


def test_swarm_runs_and_moves():
    set_engine(Engine())
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "demos" / "swarm.py"
    spec = importlib.util.spec_from_file_location("swarm_demo", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    set_engine(Engine())
    sim = mod.Swarm(n_birds=15, mode="normal")
    positions_before = [b.get_location().copy() for b in sim.birds]
    sim.run(steps=10)
    moved = sum(
        1
        for before, bird in zip(positions_before, sim.birds)
        if before != bird.get_location()
    )
    assert moved > 0
    assert sim.engine.time > 0
