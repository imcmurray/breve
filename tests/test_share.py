"""Share token encode/decode."""

from __future__ import annotations

from breve.share import decode_scene, encode_scene
from breve.scene import validate_scene, load_scene_file
from pathlib import Path


def test_roundtrip_small_scene():
    scene = {
        "title": "t",
        "mode": "physics",
        "objects": [
            {"type": "box", "static": True, "pos": [0, 0, 0], "size": [1, 1, 1]},
        ],
    }
    token = encode_scene(scene)
    assert isinstance(token, str) and len(token) > 10
    back = decode_scene(token)
    assert back["title"] == "t"
    assert validate_scene(back) == []


def test_example_gravity_encodes():
    root = Path(__file__).resolve().parents[1]
    scene = load_scene_file(str(root / "scenes" / "example_gravity.json"))
    token = encode_scene(scene)
    back = decode_scene(token)
    assert back["title"] == scene["title"]
    assert len(back["objects"]) == len(scene["objects"])
