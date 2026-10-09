"""Тесты BeamNG.Drive Lite: физика, гараж, режимы, эффекты.

Запуск:  python -m pytest tests/ -q
Всё гоняется headless — окно и OpenGL не нужны.
"""
import numpy as np
import pytest

from beamng_lite import garage, modes
from beamng_lite.terrain import Terrain
from beamng_lite.vehicle import Vehicle
from beamng_lite.effects import Particles, Smoke

DT = 1.0 / 60.0


def drive(v, steps, throttle=1.0, steer=0.0):
    for i in range(steps):
        v.controls(throttle, 0.0, steer, 0.0, DT)
        v.step(DT)


# ----------------------------------------------------------------------
# гараж / конфиги
def test_configs_valid():
    allv = garage.load_all()
    assert len(allv) >= 4                      # седан, пикап, багги, монстр-трак
    for vid, cfg in allv.items():
        assert cfg["id"] == vid
        assert len(cfg["sections"]) >= 4
        assert len(cfg["wheel_pos"]) == 4
        assert len(cfg["gears"]) >= 3          # R, N, минимум одна вперёд
        assert cfg["body_mass"] > 0
        assert 0.2 < cfg["wheel_r"] < 1.0
        assert cfg["driven"] in ("rear", "front", "all")
        assert cfg["torque"]["peak"] > 0
        assert len(cfg["color"]) == 3


def test_default_fallback():
    cfg = garage.load(garage.DEFAULT_ID)
    assert cfg["id"] == garage.DEFAULT_ID
    with pytest.raises(KeyError):
        garage.load("no_such_car")


@pytest.mark.parametrize("vid", garage.ids())
def test_vehicle_builds(vid):
    t = Terrain(seed=7)
    v = Vehicle(t, spawn=(0.0, -6.0), cfg=garage.load(vid))
    assert len(v.sb.mass) > 30                  # узлы кузова + 4 колеса
    assert len(v.sb.rest) > 100                # балки
    assert len(v.wheels) == 4
    assert v.skin.nf > 100                     # шкура кузова
    assert v.damage() == 0.0


# ----------------------------------------------------------------------
# физика
@pytest.mark.parametrize("vid", garage.ids())
def test_physics_stable(vid):
    """120 шагов разгона на стартовой площадке: нет NaN, кузов цел."""
    t = Terrain(seed=7)
    v = Vehicle(t, spawn=(0.0, -6.0), cfg=garage.load(vid))
    drive(v, 120, throttle=1.0, steer=0.35 * np.sin(0.02))
    assert np.all(np.isfinite(v.sb.pos))
    assert np.all(np.isfinite(v.sb.vel))
    assert v.damage() < 0.15, "%s получил %.0f%% на ровной площадке" % (
        vid, v.damage() * 100)


def test_sedan_regression():
    """Седан за 300 шагов разгоняется примерно до 60 км/ч без повреждений."""
    t = Terrain(seed=7)
    v = Vehicle(t, spawn=(0.0, -6.0), cfg=garage.load("sedan"))
    drive(v, 300, throttle=1.0, steer=0.35 * np.sin(0.02))
    assert np.all(np.isfinite(v.sb.pos))
    assert 45.0 < v.speed_kmh() < 75.0
    assert v.damage() < 0.05


def test_crash_test_wall():
    """Лобовой удар в неподвижную стену на ~120 км/ч: кузов серьёзно мнётся."""
    t = Terrain(seed=7)
    v = Vehicle(t, spawn=(17.0, -20.0), heading=0.0, cfg=garage.load("sedan"))
    drive(v, 900, throttle=1.0)                # стена в (17, 26), разгон 46 м
    assert v.damage() > 0.35, "крэш-тест в стену дал %.1f%%" % (v.damage() * 100)


def test_no_stuck_in_box():
    """Удар в лёгкий ящик: ни один узел кузова не остаётся внутри ящика."""
    t = Terrain(seed=7)
    v = Vehicle(t, spawn=(-9.0, -2.0), heading=0.0, cfg=garage.load("sedan"))
    drive(v, 400, throttle=1.0)                # ящик слалома в (-9, 6)
    p = v.sb.pos[v.body_idx]
    for b in t.boxes:
        inside = np.all(np.abs(p - b[:3]) < b[3:] - 0.05, axis=1)
        assert not inside.any(), "узел застрял в ящике %s" % b[:3]


def test_reset_clears_damage():
    t = Terrain(seed=7)
    v = Vehicle(t, spawn=(17.0, -20.0), heading=0.0, cfg=garage.load("sedan"))
    drive(v, 900, throttle=1.0)
    assert v.damage() > 0.35
    v.reset(spawn=(0.0, -6.0), heading=0.0)
    assert v.damage() == 0.0
    assert int((~v.sb.alive).sum()) == 0


def test_gearbox_shifts_up():
    t = Terrain(seed=7)
    v = Vehicle(t, spawn=(0.0, -6.0), cfg=garage.load("sedan"))
    assert v.gear == 2                          # старт на 1-й
    drive(v, 600, throttle=1.0)
    assert v.gear >= 4                          # автомат поднял передачи


def test_driven_axles():
    t = Terrain(seed=7)
    v = Vehicle(t, spawn=(0.0, -6.0), cfg=garage.load("sedan"))
    assert [w.driven for w in v.wheels] == [True, True, False, False]   # задний
    v = Vehicle(t, spawn=(0.0, -6.0), cfg=garage.load("buggy"))
    assert all(w.driven for w in v.wheels)                               # полный


# ----------------------------------------------------------------------
# режимы
def _teleport(v, x, z):
    c = v.frame()[0]
    v.sb.pos += np.array([x - c[0], 0.0, z - c[2]])
    v.sb.vel[:] = 0.0


def test_time_trial_lap():
    t = Terrain(seed=7)
    v = Vehicle(t, spawn=(0.0, -6.0), cfg=garage.load("sedan"))
    tt = modes.TimeTrial()
    tt.start(type("G", (), {"vehicle_id": "sedan", "seed": 7})())
    assert not tt.running
    p = modes.ring_point(0)
    _teleport(v, p[0], p[2])
    tt.update(v, t, DT)
    assert tt.running and tt.next_cp == 1
    for i in range(1, modes.N_CPS):
        p = modes.ring_point(i)
        _teleport(v, p[0], p[2])
        tt.update(v, t, DT)
    assert tt.next_cp == 0 and tt.running      # едем к финишу
    p = modes.ring_point(0)
    _teleport(v, p[0], p[2])
    tt.update(v, t, DT)
    assert not tt.running
    assert tt.last_lap is not None and tt.last_lap > 0
    assert "Круг пройден" in tt.hud_line()


def test_drift_scores():
    t = Terrain(seed=7)
    v = Vehicle(t, spawn=(0.0, -6.0), cfg=garage.load("sedan"))
    d = modes.Drift()
    d.start(type("G", (), {"vehicle_id": "sedan", "seed": 7})())
    _, fwd, rgt, _ = v.frame()
    v.sb.vel[:] = 0.0
    v.sb.vel[v.body_idx] = np.outer(np.ones(len(v.body_idx)), rgt * 10.0)
    for _ in range(60):
        d.update(v, t, DT)
    assert d.score > 0 and d.combo > 1.0


def test_trial_spawn_on_ring():
    (x, z), heading = modes.trial_spawn()
    r = np.hypot(x, z)
    assert abs(r - modes.RING_R) < 1e-6         # спавн прямо на кольце
    assert 0.0 <= heading < 2 * np.pi


# ----------------------------------------------------------------------
# эффекты и ландшафт
def test_smoke_pool():
    t = Terrain(seed=7)
    s = Smoke()
    s.emit(20, np.array([0.0, 0.5, 0.0]))
    for _ in range(60):
        s.update(DT, t)
    pos, size, fade = s.active()
    assert len(pos) > 0
    assert np.all(fade > 0) and np.all(fade <= 1)


def test_particles_pool():
    t = Terrain(seed=7)
    fx = Particles()
    fx.emit(30, np.zeros(3), np.zeros(3), [(1, 0, 0)], spread=2.0)
    for _ in range(30):
        fx.update(DT, t)
    pos, col, size, fade = fx.active()
    assert len(pos) > 0


def test_terrain_mesh_day_night():
    t = Terrain(seed=7)
    vd, cd, _ = t.build_mesh()
    vn, cn, _ = t.build_mesh(night=True)
    assert vd.shape == vn.shape
    assert cd.mean() > cn.mean()                # ночь темнее


def test_terrain_height_normal():
    t = Terrain(seed=7)
    h = t.height(0.0, 0.0)
    assert abs(h) < 5.0                          # стартовая площадка плоская
    n = t.normal(np.array([0.0, 10.0]), np.array([0.0, -10.0]))
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0)
    assert np.all(n[:, 1] > 0.5)                 # нормали вверх
