"""Rigid-body physics tests."""

from __future__ import annotations

import numpy as np

import breve
from breve.engine import Engine, set_engine


def test_ball_falls_under_gravity():
    set_engine(Engine())

    class C(breve.PhysicalControl):
        def init(self):
            self.set_integration_step(0.005)
            self.set_iteration_step(0.02)
            self.full_gravity()
            ground = breve.Stationary()
            ground.set_shape(breve.Box().init_with(breve.vector(4, 0.2, 4)))
            ground.move(breve.vector(0, -0.1, 0))
            breve.get_engine().register_physics_body(ground, static=True)
            self.ball = breve.Mobile()
            self.ball.set_shape(breve.Sphere().init_with(0.2))
            self.ball.move(breve.vector(0, 2.0, 0))
            self.ball.set_velocity(breve.vector(0, 0, 0))
            self.ball.enable_physics(mass=1.0)

    c = C()
    y0 = c.ball.location.y
    c.run(steps=40)
    assert c.ball.location.y < y0
    # should have hit the ground and settled near y≈0.2
    assert c.ball.location.y < 1.0
    assert c.ball.location.y > 0.05


def test_sphere_bounces_on_box():
    set_engine(Engine())

    class C(breve.PhysicalControl):
        def init(self):
            self.set_integration_step(0.004)
            self.set_iteration_step(0.016)
            self.full_gravity()
            step = breve.Stationary()
            step.set_shape(breve.Box().init_with(breve.vector(2, 0.1, 2)))
            step.move(breve.vector(0, 0, 0))
            breve.get_engine().register_physics_body(step, static=True)
            self.ball = breve.Mobile()
            self.ball.set_shape(breve.Sphere().init_with(0.15))
            self.ball.move(breve.vector(0, 1.5, 0))
            self.ball.set_velocity(breve.vector(0, 0, 0))
            self.ball.enable_physics(mass=1.0)

    c = C()
    c.run(steps=80)
    # not fallen through the step
    assert c.ball.location.y > 0.05


def test_box_tips_off_platform_edge():
    """COM past the rim: gravity + contact torque at the edge must tip it off."""
    set_engine(Engine())

    class C(breve.PhysicalControl):
        def init(self):
            self.set_integration_step(0.004)
            self.set_iteration_step(0.016)
            self.full_gravity()
            # platform occupies x in [-1, 1]
            floor = breve.Stationary()
            floor.set_shape(breve.Box().init_with(breve.vector(2.0, 0.2, 2.0)))
            floor.move(breve.vector(0, -0.1, 0))
            breve.get_engine().register_physics_body(floor, static=True)
            # 0.5-wide box: COM at 1.05 is past the rim (x=1); only the
            # inner face still overlaps. No initial push — weight alone tips it.
            self.box = breve.Mobile()
            self.box.set_shape(breve.Box().init_with(breve.vector(0.5, 0.5, 0.5)))
            self.box.move(breve.vector(1.05, 0.35, 0))
            self.box.set_velocity(breve.vector(0, 0, 0))
            self.box.enable_physics(mass=1.0)

    c = C()
    c.run(steps=100)
    assert c.box.location.y < -0.5


def test_box_stack_tips_off_edge():
    """Stacked boxes with COMs past the platform edge must all fall."""
    set_engine(Engine())

    class C(breve.PhysicalControl):
        def init(self):
            self.set_integration_step(0.004)
            self.set_iteration_step(0.016)
            self.full_gravity()
            floor = breve.Stationary()
            floor.set_shape(breve.Box().init_with(breve.vector(2.0, 0.2, 2.0)))
            floor.move(breve.vector(0, -0.1, 0))
            breve.get_engine().register_physics_body(floor, static=True)
            self.boxes = []
            for i in range(3):
                b = breve.Mobile()
                b.set_shape(breve.Box().init_with(breve.vector(0.5, 0.5, 0.5)))
                b.move(breve.vector(1.05, 0.35 + i * 0.52, 0))
                b.enable_physics(mass=0.5)
                self.boxes.append(b)

    c = C()
    c.run(steps=220)
    assert all(b.location.y < -0.5 for b in c.boxes)


def test_box_does_not_freeze_on_edge():
    """A cube on an edge (45°) must tip to a face — not sleep mid-pose."""
    set_engine(Engine())

    class C(breve.PhysicalControl):
        def init(self):
            self.set_integration_step(0.016)
            self.set_iteration_step(0.016)
            self.full_gravity()
            floor = breve.Stationary()
            floor.set_shape(breve.Box().init_with(breve.vector(8, 0.2, 8)))
            floor.move(breve.vector(0, -0.1, 0))
            breve.get_engine().register_physics_body(floor, static=True)
            self.box = breve.Mobile()
            self.box.set_shape(breve.Box().init_with(breve.vector(0.5, 0.5, 0.5)))
            self.box.move(breve.vector(0, 1.2, 0))
            self.box.enable_physics(mass=1.0)
            pb = breve.get_engine().physics.get_body(self.box)
            # Slightly off 45° so natural gravity tips (exact 45° is
            # metastable with symmetric contacts — no artificial tip nudge).
            a = np.deg2rad(40) / 2
            pb.orientation = np.array(
                [np.cos(a), 0.0, 0.0, np.sin(a)], dtype=np.float64
            )
            pb.restitution = 0.1
            pb.friction = 0.4

    c = C()
    c.run(steps=200)
    body = breve.get_engine().physics.get_body(c.box)
    R = body.rotation_matrix()
    face_align = max(abs(float(R[1, i])) for i in range(3))
    assert face_align > 0.90, f"box stuck on edge, face_align={face_align}"
    # half-extent 0.25 → face rest ≈0.25; allow small settle slack
    assert c.box.location.y < 0.34


def test_dynamic_support_does_not_force_face_down():
    """Tip nudge must not run when another dynamic body is supporting the pose."""
    from breve.physics import _apply_contact_rest

    set_engine(Engine())

    class C(breve.PhysicalControl):
        def init(self):
            self.set_integration_step(0.016)
            self.set_iteration_step(0.016)
            self.full_gravity()
            self.box = breve.Mobile()
            self.box.set_shape(breve.Box().init_with(breve.vector(0.5, 0.5, 0.5)))
            self.box.move(breve.vector(0, 1.0, 0))
            self.box.enable_physics(mass=1.0)
            pb = breve.get_engine().physics.get_body(self.box)
            a = np.deg2rad(40) / 2
            pb.orientation = np.array(
                [np.cos(a), 0.0, 0.0, np.sin(a)], dtype=np.float64
            )
            pb.velocity[:] = 0.0
            pb.angular_velocity[:] = 0.0

    c = C()
    body = breve.get_engine().physics.get_body(c.box)
    # Simulate rest while propped by another box (dynamic support)
    for _ in range(60):
        body.velocity[:] = 0.0
        _apply_contact_rest(body, sleep_frames_needed=4, dt=0.016, has_dynamic_support=True)
    # Must stay nearly still — no face-align spin injected
    assert float(np.linalg.norm(body.angular_velocity)) < 0.05
    R = body.rotation_matrix()
    face_align = max(abs(float(R[1, i])) for i in range(3))
    # Orientation unchanged (still tilted ~40°)
    assert face_align < 0.92


def test_offset_hit_spins_box():
    """Impulse away from COM should produce angular velocity."""
    set_engine(Engine())

    class C(breve.PhysicalControl):
        def init(self):
            self.zero_gravity()
            self.set_integration_step(0.005)
            self.set_iteration_step(0.02)
            self.box = breve.Mobile()
            self.box.set_shape(breve.Box().init_with(breve.vector(1.0, 0.4, 0.4)))
            self.box.move(breve.vector(0, 1.0, 0))
            self.box.set_velocity(breve.vector(0, 0, 0))
            self.box.enable_physics(mass=2.0)
            ball = breve.Mobile()
            ball.set_shape(breve.Sphere().init_with(0.2))
            ball.move(breve.vector(-1.2, 1.15, 0))  # hit top half of box
            ball.set_velocity(breve.vector(8, 0, 0))
            ball.enable_physics(mass=1.0)

    c = C()
    body = breve.get_engine().physics.get_body(c.box)
    c.run(steps=15)
    omega = float(np.linalg.norm(body.angular_velocity))
    assert omega > 0.05


# ---------------------------------------------------------------------------
# Resting-behavior regressions: a cube must never freeze balanced on an
# unsupported corner/edge; it must keep tipping until its COM is supported.
# ---------------------------------------------------------------------------

def _quat_axis_angle(ax, ay, az, deg):
    a = np.deg2rad(deg) / 2
    n = (ax * ax + ay * ay + az * az) ** 0.5
    s = np.sin(a) / n
    return np.array([np.cos(a), ax * s, ay * s, az * s], dtype=np.float64)


def _q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _make_floor_scene(box_specs):
    """Floor + cubes. box_specs: list of (pos, orientation_quat, mass)."""
    set_engine(Engine())

    class C(breve.PhysicalControl):
        def init(self):
            self.set_integration_step(0.016)
            self.set_iteration_step(0.016)
            self.full_gravity()
            floor = breve.Stationary()
            floor.set_shape(breve.Box().init_with(breve.vector(8, 0.2, 8)))
            floor.move(breve.vector(0, -0.1, 0))
            breve.get_engine().register_physics_body(floor, static=True)
            self.boxes = []
            for pos, quat, mass in box_specs:
                b = breve.Mobile()
                b.set_shape(breve.Box().init_with(breve.vector(0.5, 0.5, 0.5)))
                b.move(breve.vector(*pos))
                b.enable_physics(mass=mass)
                pb = breve.get_engine().physics.get_body(b)
                if quat is not None:
                    pb.orientation = np.array(quat, dtype=np.float64)
                pb.restitution = 0.05
                pb.friction = 0.5
                self.boxes.append(b)

    return C()


def _body_state(box):
    body = breve.get_engine().physics.get_body(box)
    R = body.rotation_matrix()
    face_align = max(abs(float(R[1, i])) for i in range(3))
    speed = float(np.linalg.norm(body.velocity))
    spin = float(np.linalg.norm(body.angular_velocity))
    return body, face_align, speed, spin


def _assert_sane(body):
    assert np.all(np.isfinite(body.position)), "NaN/inf position"
    assert np.all(np.isfinite(body.velocity)), "NaN/inf velocity"
    assert np.all(np.isfinite(body.orientation)), "NaN/inf orientation"
    assert np.all(np.isfinite(body.angular_velocity)), "NaN/inf angular velocity"


def _with_numba(enabled):
    from breve.physics_kernels import numba_enabled, set_numba_enabled

    prev = numba_enabled()
    set_numba_enabled(enabled)
    return prev


def _run_tilted_cube_case(use_numba, rx_deg=25, rz_deg=40, steps=600):
    """Corner-balanced cube (screenshot repro): must settle onto a face."""
    from breve.physics_kernels import set_numba_enabled

    prev = _with_numba(use_numba)
    try:
        q = _q_mul(
            _quat_axis_angle(0, 0, 1, rz_deg), _quat_axis_angle(1, 0, 0, rx_deg)
        )
        c = _make_floor_scene([((0, 0.7, 0), q, 1.0)])
        c.run(steps=steps)
        body, face_align, speed, spin = _body_state(c.boxes[0])
        _assert_sane(body)
        assert face_align > 0.97, (
            f"cube frozen at implausible angle: face_align={face_align:.3f}"
        )
        # face rest height = 0.25; allow slop but forbid mid-tip poses and sinks
        assert 0.20 < float(body.position[1]) < 0.32, (
            f"bad rest height y={float(body.position[1]):.3f}"
        )
        assert speed < 0.05 and spin < 0.10, (
            f"not settled: speed={speed:.3f} spin={spin:.3f}"
        )
    finally:
        set_numba_enabled(prev)


def test_corner_tilted_cube_settles_numba():
    _run_tilted_cube_case(use_numba=True)


def test_corner_tilted_cube_settles_python():
    _run_tilted_cube_case(use_numba=False)


def test_edge_tilted_cube_settles_both_paths():
    """25–35° single-axis edge tilts must also settle onto a face."""
    from breve.physics_kernels import set_numba_enabled

    for use_numba in (True, False):
        prev = _with_numba(use_numba)
        try:
            for rz in (25, 35):
                q = _quat_axis_angle(0, 0, 1, rz)
                c = _make_floor_scene([((0, 0.7, 0), q, 1.0)])
                c.run(steps=500)
                body, face_align, speed, spin = _body_state(c.boxes[0])
                _assert_sane(body)
                assert face_align > 0.97, (
                    f"numba={use_numba} rz={rz}: stuck face_align={face_align:.3f}"
                )
                assert speed < 0.05 and spin < 0.10
        finally:
            set_numba_enabled(prev)


def test_upright_cube_rests_quietly():
    """Upright cube must simply rest: no drift, no spin, no sink."""
    from breve.physics_kernels import set_numba_enabled

    for use_numba in (True, False):
        prev = _with_numba(use_numba)
        try:
            c = _make_floor_scene([((0, 0.26, 0), None, 1.0)])
            c.run(steps=240)
            body, face_align, speed, spin = _body_state(c.boxes[0])
            _assert_sane(body)
            assert face_align > 0.995
            assert 0.20 < float(body.position[1]) < 0.30
            assert speed < 0.02 and spin < 0.05
            assert abs(float(body.position[0])) < 0.1
            assert abs(float(body.position[2])) < 0.1
        finally:
            set_numba_enabled(prev)


def test_edge_balance_com_supported_is_legitimate():
    """Exactly 45°, COM directly over the edge: a genuinely balanced pose.

    It must either remain balanced or (numerics permitting) tip fully to a
    face — never freeze partway between edge-balance and face rest.
    """
    from breve.physics_kernels import set_numba_enabled

    for use_numba in (True, False):
        prev = _with_numba(use_numba)
        try:
            q = _quat_axis_angle(0, 0, 1, 45)
            # rest height on edge = 0.25*sqrt(2)
            c = _make_floor_scene([((0, 0.3557, 0), q, 1.0)])
            c.run(steps=400)
            body, face_align, speed, spin = _body_state(c.boxes[0])
            _assert_sane(body)
            balanced = face_align < 0.72  # still diamond (45° => 0.707)
            settled_flat = face_align > 0.97
            assert balanced or settled_flat, (
                f"numba={use_numba}: frozen mid-tip face_align={face_align:.3f}"
            )
            assert speed < 0.05 and spin < 0.10
        finally:
            set_numba_enabled(prev)


def test_two_box_stack_stays_stacked():
    from breve.physics_kernels import set_numba_enabled

    for use_numba in (True, False):
        prev = _with_numba(use_numba)
        try:
            c = _make_floor_scene(
                [((0, 0.26, 0), None, 1.0), ((0, 0.80, 0), None, 1.0)]
            )
            c.run(steps=300)
            for box in c.boxes:
                body, face_align, speed, spin = _body_state(box)
                _assert_sane(body)
                assert face_align > 0.99
                assert speed < 0.03 and spin < 0.05
            y0 = float(c.boxes[0].location.y)
            y1 = float(c.boxes[1].location.y)
            assert 0.20 < y0 < 0.30
            assert 0.68 < y1 < 0.82, f"top box at y={y1:.3f}"
        finally:
            set_numba_enabled(prev)


def test_cube_falls_onto_cube():
    """Drop a cube onto a resting cube: both must end settled, no NaNs."""
    from breve.physics_kernels import set_numba_enabled

    for use_numba in (True, False):
        prev = _with_numba(use_numba)
        try:
            c = _make_floor_scene(
                [((0, 0.26, 0), None, 1.0), ((0.1, 1.6, 0.05), None, 1.0)]
            )
            c.run(steps=600)
            for box in c.boxes:
                body, face_align, speed, spin = _body_state(box)
                _assert_sane(body)
                assert speed < 0.06 and spin < 0.12
                assert face_align > 0.95
                assert float(body.position[1]) > 0.15  # nothing through the floor
        finally:
            set_numba_enabled(prev)


def test_gravity_demo_runs():
    set_engine(Engine())
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "demos" / "gravity.py"
    spec = importlib.util.spec_from_file_location("gravity_demo", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    set_engine(Engine())
    sim = mod.Gravity()
    y0 = [b.location.y for b in sim.balls]
    sim.run(steps=50)
    y1 = [b.location.y for b in sim.balls]
    # overall, balls should have moved (fallen / rolled)
    assert any(abs(a - b) > 0.01 for a, b in zip(y0, y1))
