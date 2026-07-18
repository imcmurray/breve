"""
Declarative scene description → runnable breve simulation.

This is the safe path for AI-generated worlds: the model emits JSON that
matches SCENE_SCHEMA; we never exec arbitrary model code.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from breve.engine import Engine, get_engine, set_engine
from breve.objects import Mobile, PhysicalControl, Stationary
from breve.shapes import Box, Sphere
from breve.vector import vector

# Human + LLM readable contract ------------------------------------------------

SCENE_SCHEMA_DOC = """
Breve scene JSON (only this format is accepted):

{
  "title": "short name",
  "mode": "physics" | "kinematic",     // physics = gravity/collisions; kinematic = scripted agents
  "gravity": [0, -9.8, 0],             // optional; default earth gravity if mode=physics
  "background": [0.1, 0.12, 0.16],     // optional RGB 0-1
  "camera": {
    "target": [0, 1, 0],               // look-at point
    "zoom": 12                         // distance
  },
  "objects": [
    {
      "type": "box" | "sphere",
      "static": true | false,          // true = floor/wall; false = dynamic body
      "pos": [x, y, z],
      "pos_jitter": [0.15, 0.1, 0.15], // optional: ±uniform random added to pos each run
      "size": [sx, sy, sz],            // box full extents (required for box)
      "radius": 0.25,                  // sphere (required for sphere)
      "mass": 1.0,                     // dynamic only; heavier = more inertia in collisions
      "velocity": [vx, vy, vz],        // optional initial velocity
      "velocity_jitter": [0.3, 0.2, 0.3], // optional: ±random added to velocity
      "color": [r, g, b],              // 0-1
      "restitution": 0.7,              // bounciness 0-1
      "friction": 0.2
    }
  ],
  "agents": [                          // optional kinematic multi-agent behaviors
    {
      "behavior": "flock" | "wander" | "gather",
      "count": 40,
      "color": [0.8, 0.8, 1.0],
      "radius": 0.25,
      "spread": 8
    }
  ],
  "notes": "one sentence for the user about what they will see"
}

Rules for good gravity demos:
- Always include a large static box as a floor near y=0.
- Drop dynamic spheres from y >= 3 so arcs are visible.
- Use different mass values (0.3 light vs 8 heavy) and sizes so collisions differ.
- Give some horizontal velocity so paths are parabolas, not vertical lines.
- Set camera.target near the action and zoom 10-16.

Rules for flocking:
- mode can be "kinematic"; agents with behavior "flock".
"""


def _v3(seq: Sequence[float], default: Tuple[float, float, float] = (0, 0, 0)):
    if not seq:
        return vector(*default)
    return vector(float(seq[0]), float(seq[1]), float(seq[2] if len(seq) > 2 else 0))


class SceneController(PhysicalControl):
    """Runtime controller built from a scene dict."""

    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec
        self.title = str(spec.get("title") or "scene")
        self.flock_agents: list = []
        self.wander_agents: list = []
        self._last_print = -1.0
        super().__init__()

    def init(self) -> None:
        mode = str(self.spec.get("mode") or "physics").lower()
        g = self.spec.get("gravity")
        if mode == "physics":
            if g:
                self.set_gravity(_v3(g, (0, -9.8, 0)))
            else:
                self.full_gravity()
            # One physics substep per display frame keeps pure-Python realtime.
            self.set_integration_step(0.016)
            self.set_iteration_step(0.016)
        else:
            # kinematic: no gravity forces, but allow agents
            self.set_gravity(vector(0, 0, 0))
            self.set_integration_step(0.02)
            self.set_iteration_step(0.05)

        bg = self.spec.get("background")
        if bg:
            self.set_background_color(_v3(bg, (0.1, 0.12, 0.16)))
        else:
            self.set_background_color(vector(0.1, 0.12, 0.16))
        self.enable_lighting()

        for obj in self.spec.get("objects") or []:
            self._spawn_object(obj, physics=(mode == "physics"))

        for agent in self.spec.get("agents") or []:
            self._spawn_agents(agent)

        cam = self.spec.get("camera") or {}
        target = _v3(cam.get("target") or [0, 1, 0], (0, 1, 0))
        self.point_camera(target, vector(8, 4, 10))
        self.camera_zoom = float(cam.get("zoom") or 12)

        notes = self.spec.get("notes") or ""
        print(f"Scene: {self.title}")
        if notes:
            print(f"  {notes}")
        n_obj = len(self.spec.get("objects") or [])
        n_ag = sum(int(a.get("count") or 0) for a in (self.spec.get("agents") or []))
        print(f"  objects={n_obj}  agents≈{n_ag}  mode={mode}")

    def _spawn_object(self, obj: Dict[str, Any], physics: bool) -> None:
        otype = str(obj.get("type") or "sphere").lower()
        static = bool(obj.get("static", False))
        pos = _v3(obj.get("pos") or [0, 1, 0])
        # Per-run randomness so demos don't look identical every Reset
        if not static and obj.get("pos_jitter"):
            pos = _apply_jitter(pos, obj.get("pos_jitter"))
        color = _v3(obj.get("color") or [0.7, 0.7, 0.8])
        mass = float(obj.get("mass") or 1.0)
        restitution = float(obj.get("restitution") if obj.get("restitution") is not None else 0.7)
        friction = float(obj.get("friction") if obj.get("friction") is not None else 0.2)
        vel_spec = obj.get("velocity")
        vel = _v3(vel_spec) if vel_spec else None
        if not static and vel is not None and obj.get("velocity_jitter"):
            vel = _apply_jitter(vel, obj.get("velocity_jitter"))

        if static:
            body = Stationary()
        else:
            body = Mobile()

        if otype == "box":
            size = _v3(obj.get("size") or [1, 1, 1], (1, 1, 1))
            body.set_shape(Box().init_with(size))
        else:
            radius = float(obj.get("radius") or 0.25)
            body.set_shape(Sphere().init_with(radius))

        body.move(pos)
        body.set_color(color)
        # Optional 0–1 opacity for web viewer (physics unchanged).
        if obj.get("opacity") is not None:
            body.opacity = float(obj["opacity"])  # type: ignore[attr-defined]
        elif obj.get("transparent") is True:
            body.opacity = 0.22  # type: ignore[attr-defined]
        else:
            body.opacity = 1.0  # type: ignore[attr-defined]

        if not static and vel is not None:
            body.set_velocity(vel)

        eng = get_engine()
        if physics or static:
            if not static:
                body.physics_enabled = True
                body.mass = mass
            eng.register_physics_body(body, static=static, mass=mass if not static else 0.0)
            pb = eng.physics.get_body(body)
            if pb is not None:
                pb.restitution = restitution
                pb.friction = friction
                pb.position[:] = [pos.x, pos.y, pos.z]
                if not static and vel is not None:
                    pb.velocity[:] = [vel.x, vel.y, vel.z]
                else:
                    pb.velocity[:] = 0.0
                pb.angular_velocity[:] = 0.0
                from breve.physics import _q_identity

                pb.orientation = _q_identity()
                if not static:
                    pb._recompute_local_inertia()
                pb.awake = True

    def _spawn_agents(self, agent: Dict[str, Any]) -> None:
        behavior = str(agent.get("behavior") or "wander").lower()
        count = int(agent.get("count") or 10)
        color = _v3(agent.get("color") or [0.85, 0.85, 1.0])
        radius = float(agent.get("radius") or 0.22)
        spread = float(agent.get("spread") or 8.0)

        for _ in range(count):
            m = Mobile()
            m.set_shape(Sphere().init_with(radius))
            m.set_color(color)
            m.set_wander_range(vector(spread, spread * 0.5, spread))
            m.randomize_location()
            if m.location.y < 0.5:
                m.location.y = 0.5 + abs(m.location.y)
            speed = 4.0 + (spread * 0.2)
            m.set_velocity(
                vector(
                    (random_unit() * 2 - 1) * speed,
                    (random_unit() * 2 - 1) * speed * 0.3,
                    (random_unit() * 2 - 1) * speed,
                )
            )
            m.neighborhood_size = 2.5
            if behavior == "flock":
                self.flock_agents.append(m)
            else:
                self.wander_agents.append(m)

    def iterate(self) -> None:
        # simple boids for flock agents
        for bird in self.flock_agents:
            self._flock_step(bird)
        for w in self.wander_agents:
            self._wander_step(w)

        # keep camera near dynamic action
        mobiles = [
            o
            for o in self.engine.objects
            if isinstance(o, Mobile) and o.enabled and getattr(o, "physics_enabled", False)
        ]
        if mobiles:
            cx = sum(m.location.x for m in mobiles) / len(mobiles)
            cy = sum(m.location.y for m in mobiles) / len(mobiles)
            self.aim_camera(vector(cx * 0.3, max(0.5, cy * 0.5), 0))
        elif self.flock_agents:
            cx = sum(m.location.x for m in self.flock_agents) / len(self.flock_agents)
            cy = sum(m.location.y for m in self.flock_agents) / len(self.flock_agents)
            self.aim_camera(vector(cx, cy, 0))
            self.camera_zoom = max(8.0, min(30.0, self.camera_zoom))

        if self.engine.time - self._last_print >= 1.0:
            self._last_print = self.engine.time
            n = sum(1 for o in self.engine.objects if o.enabled)
            print(f"  t={self.engine.time:5.1f}s  live_objects={n}")
        super().iterate()

    def _flock_step(self, bird: Mobile) -> None:
        neighbors = []
        for other in self.flock_agents:
            if other is bird:
                continue
            d = bird.location - other.location
            if d.length() < bird.neighborhood_size:
                neighbors.append(other)
        accel = vector(0, 0, 0)
        if neighbors:
            center = vector(0, 0, 0)
            avg_v = vector(0, 0, 0)
            sep = vector(0, 0, 0)
            for n in neighbors:
                center = center + n.location
                avg_v = avg_v + n.velocity
                to = bird.location - n.location
                if to.length() < 0.5:
                    sep = sep + to
            center = center / len(neighbors)
            avg_v = avg_v / len(neighbors)
            accel = (center - bird.location) * 2.0 + (avg_v - bird.velocity) * 2.0 + sep * 5.0
        # world center pull
        if bird.location.length() > 12:
            accel = accel + (vector(0, 4, 0) - bird.location) * 0.5
        # mild wander
        accel = accel + vector(random_unit() - 0.5, random_unit() - 0.5, random_unit() - 0.5) * 3.0
        if accel.length() > 0:
            accel = accel.normalize() * 12.0
        bird.set_acceleration(accel)
        v = bird.velocity
        if v.length() > 14:
            bird.set_velocity(v.normalize() * 14)

    def _wander_step(self, agent: Mobile) -> None:
        if random_unit() < 0.05:
            agent.set_velocity(
                vector(
                    (random_unit() * 2 - 1) * 5,
                    (random_unit() * 2 - 1) * 2,
                    (random_unit() * 2 - 1) * 5,
                )
            )
        # soft bounds
        r = agent.wander_range
        loc, vel = agent.location, agent.velocity
        if abs(loc.x) > r.x:
            vel.x *= -1
        if abs(loc.y) > r.y + 2:
            vel.y *= -1
        if abs(loc.z) > r.z:
            vel.z *= -1
        if loc.y < 0.3:
            loc.y = 0.3
            if vel.y < 0:
                vel.y *= -0.5
        agent.set_velocity(vel)


def random_unit() -> float:
    import numpy as np

    return float(np.random.random())


def _apply_jitter(base, jitter_spec) -> "vector":
    """base ± uniform[0, jitter] on each axis (symmetric around base)."""
    import numpy as np

    j = _v3(jitter_spec or [0, 0, 0])
    return vector(
        base.x + (np.random.random() * 2 - 1) * j.x,
        base.y + (np.random.random() * 2 - 1) * j.y,
        base.z + (np.random.random() * 2 - 1) * j.z,
    )


def validate_scene(spec: Dict[str, Any]) -> List[str]:
    """Return list of errors (empty if OK)."""
    errs: List[str] = []
    if not isinstance(spec, dict):
        return ["scene must be a JSON object"]
    mode = str(spec.get("mode") or "physics").lower()
    if mode not in ("physics", "kinematic"):
        errs.append("mode must be 'physics' or 'kinematic'")
    objects = spec.get("objects")
    if objects is not None and not isinstance(objects, list):
        errs.append("objects must be a list")
    agents = spec.get("agents")
    if agents is not None and not isinstance(agents, list):
        errs.append("agents must be a list")
    if mode == "physics":
        objs = objects or []
        if not any(bool(o.get("static")) for o in objs if isinstance(o, dict)):
            # warn-level: allow but recommend floor
            pass
        if not objs and not (agents or []):
            errs.append("physics scene needs objects and/or agents")
    for i, o in enumerate(objects or []):
        if not isinstance(o, dict):
            errs.append(f"objects[{i}] must be an object")
            continue
        t = str(o.get("type") or "sphere").lower()
        if t not in ("box", "sphere"):
            errs.append(f"objects[{i}].type must be box|sphere")
        if t == "box" and not o.get("size"):
            errs.append(f"objects[{i}] box needs size [sx,sy,sz]")
        if t == "sphere" and o.get("radius") is None and not o.get("static"):
            pass  # default radius ok
    return errs


def loads_scene(text: str) -> Dict[str, Any]:
    """Parse scene JSON, optionally extracting a fenced code block."""
    text = text.strip()
    if "```" in text:
        # extract first json fence
        parts = text.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 1:
                body = part.strip()
                if body.startswith("json"):
                    body = body[4:].strip()
                text = body
                break
    # try raw
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # try find outermost braces
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def build_scene(spec: Dict[str, Any]) -> SceneController:
    """Validate and construct a scene controller (does not run)."""
    errs = validate_scene(spec)
    if errs:
        raise ValueError("Invalid scene:\n  - " + "\n  - ".join(errs))
    set_engine(Engine())
    return SceneController(spec)


def _dynamic_physics_bodies(sim: SceneController):
    """Yield (owner, RigidBody) for enabled dynamic physics mobiles."""
    eng = sim.engine
    for obj in eng.objects:
        if not obj.enabled or not isinstance(obj, Mobile):
            continue
        if not getattr(obj, "physics_enabled", False):
            continue
        body = eng.physics.get_body(obj)
        if body is None or body.static:
            continue
        yield obj, body


def cull_out_of_view(
    sim: SceneController,
    *,
    y_min: float = -12.0,
    xz_limit: float = 48.0,
) -> int:
    """
    Remove dynamic bodies that have fallen out of the play volume.

    Used by the web sim so debris that leaves the camera never accumulates.
    """
    culled = 0
    doomed = []
    for obj, body in _dynamic_physics_bodies(sim):
        x, y, z = float(body.position[0]), float(body.position[1]), float(body.position[2])
        if y < y_min or abs(x) > xz_limit or abs(z) > xz_limit:
            doomed.append(obj)
    for obj in doomed:
        obj.remove()
        culled += 1
    return culled


def world_is_settled(
    sim: SceneController,
    *,
    speed_eps: float = 0.10,
    spin_eps: float = 0.22,
    min_bodies: int = 1,
) -> bool:
    """
    True when every dynamic physics body is nearly still.

    Empty / non-physics scenes are treated as settled so the web UI can pause.
    """
    n = 0
    for _obj, body in _dynamic_physics_bodies(sim):
        n += 1
        vx, vy, vz = (
            float(body.velocity[0]),
            float(body.velocity[1]),
            float(body.velocity[2]),
        )
        speed = (vx * vx + vy * vy + vz * vz) ** 0.5
        wx, wy, wz = (
            float(body.angular_velocity[0]),
            float(body.angular_velocity[1]),
            float(body.angular_velocity[2]),
        )
        spin = (wx * wx + wy * wy + wz * wz) ** 0.5
        if speed > speed_eps or spin > spin_eps:
            return False
    return n >= 0  # settled even if zero dynamics left


def snapshot_state(sim: SceneController) -> Dict[str, Any]:
    """Serialize current world for web / remote viewers."""
    from breve.shapes import Box, Sphere

    objects = []
    for obj in sim.engine.objects:
        if not obj.enabled:
            continue
        shape = obj.shape
        opacity = float(getattr(obj, "opacity", 1.0))
        entry: Dict[str, Any] = {
            "id": id(obj),
            "class": obj.__class__.__name__,
            "pos": [obj.location.x, obj.location.y, obj.location.z],
            "color": [obj.color.x, obj.color.y, obj.color.z],
            "opacity": opacity,
            "static": not isinstance(obj, Mobile)
            or not getattr(obj, "physics_enabled", False),
        }
        if isinstance(shape, Sphere):
            entry["type"] = "sphere"
            entry["radius"] = float(shape.radius)
        elif isinstance(shape, Box):
            entry["type"] = "box"
            entry["size"] = [shape.size.x, shape.size.y, shape.size.z]
        else:
            entry["type"] = "sphere"
            entry["radius"] = 0.25
        if isinstance(obj, Mobile):
            entry["velocity"] = [
                obj.velocity.x,
                obj.velocity.y,
                obj.velocity.z,
            ]
            entry["mass"] = float(getattr(obj, "mass", 1.0))
            entry["physics"] = bool(getattr(obj, "physics_enabled", False))
        # orientation from physics body (w,x,y,z) for three.js
        quat = getattr(obj, "_physics_quat", None)
        if quat is not None:
            entry["quat"] = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
        else:
            entry["quat"] = [1.0, 0.0, 0.0, 0.0]
        objects.append(entry)

    cam = {
        "target": [
            sim.camera_target.x,
            sim.camera_target.y,
            sim.camera_target.z,
        ],
        "zoom": float(sim.camera_zoom),
    }
    bg = sim.background_color
    return {
        "time": sim.engine.time,
        "title": sim.title,
        "background": [bg.x, bg.y, bg.z],
        "camera": cam,
        "objects": objects,
        "settled": world_is_settled(sim),
    }


def build_and_run(
    spec: Dict[str, Any],
    *,
    viz: bool = False,
    steps: Optional[int] = None,
) -> SceneController:
    """Validate, construct, and run a scene."""
    sim = build_scene(spec)

    if steps is None:
        steps = None if viz else 300

    if viz:
        from breve.viz import run_with_viewer

        print("3D: drag=orbit  scroll=zoom  SPACE=pause  ESC=quit")
        run_with_viewer(sim, steps=steps)
    else:
        sim.run(steps=steps if steps is not None else 300)
    return sim


def save_scene(spec: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
        f.write("\n")


def load_scene_file(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
