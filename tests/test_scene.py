"""Declarative scene builder tests (no API key required)."""

from __future__ import annotations

from breve.engine import set_engine, Engine
from breve.scene import build_and_run, validate_scene, loads_scene, SceneController


def test_validate_ok():
    spec = {
        "title": "t",
        "mode": "physics",
        "objects": [
            {"type": "box", "static": True, "pos": [0, -0.1, 0], "size": [8, 0.2, 8]},
            {"type": "sphere", "pos": [0, 3, 0], "radius": 0.3, "mass": 1.0},
        ],
    }
    assert validate_scene(spec) == []


def test_loads_fenced_json():
    raw = """Here you go.\n```json\n{"title": "x", "mode": "physics", "objects": []}\n```\n"""
    s = loads_scene(raw)
    assert s["title"] == "x"


def test_physics_scene_runs():
    set_engine(Engine())
    spec = {
        "title": "drop",
        "mode": "physics",
        "gravity": [0, -9.8, 0],
        "camera": {"target": [0, 1, 0], "zoom": 10},
        "objects": [
            {
                "type": "box",
                "static": True,
                "pos": [0, -0.1, 0],
                "size": [10, 0.2, 10],
                "color": [0.3, 0.3, 0.3],
            },
            {
                "type": "sphere",
                "static": False,
                "pos": [0, 3, 0],
                "radius": 0.25,
                "mass": 1.0,
                "velocity": [1, 0, 0],
                "color": [1, 0.2, 0.2],
                "restitution": 0.8,
            },
        ],
        "notes": "test",
    }
    sim = build_and_run(spec, viz=False, steps=40)
    assert isinstance(sim, SceneController)
    # ball should have fallen
    spheres = [
        o
        for o in sim.engine.objects
        if o.__class__.__name__ == "Mobile" and getattr(o, "physics_enabled", False)
    ]
    assert spheres
    assert spheres[0].location.y < 3.0


def test_flock_scene_runs():
    set_engine(Engine())
    spec = {
        "title": "birds",
        "mode": "kinematic",
        "agents": [{"behavior": "flock", "count": 12, "spread": 6}],
        "camera": {"target": [0, 2, 0], "zoom": 15},
    }
    sim = build_and_run(spec, viz=False, steps=15)
    assert len(sim.flock_agents) == 12
