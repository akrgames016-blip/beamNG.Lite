"""Игровые режимы: свободная езда, тайм-трайл по кольцу чекпоинтов, дрифт.

Трасса тайм-трайла — грунтовое кольцо радиусом 72 м вокруг стартовой
площадки (оно уже есть на карте). Ворота — 12 точек по кольцу; старт и
финиш — ворота №0. Рекорды хранятся в ~/.beamng_lite/records.json.
"""
import json
import os

import numpy as np

RING_R = 72.0
N_CPS = 12
CP_RADIUS = 7.0          # радиус засчитывания ворот, м

MODES = ("free", "time_trial", "drift")
NAMES = {"free": "Свободная езда",
         "time_trial": "Тайм-трайл",
         "drift": "Дрифт"}


def ring_point(i, terrain=None):
    """Точка i-й воротины на кольце (0-я — старт/финиш, нос к +Z)."""
    a = 2.0 * np.pi * i / N_CPS
    return np.array([RING_R * np.sin(a), 0.0, RING_R * np.cos(a)])


def trial_spawn():
    """Спавн на кольце за ~13 м до стартовой воротины, нос вдоль трассы."""
    a = -np.pi / 18.0                      # -10°
    x, z = RING_R * np.sin(a), RING_R * np.cos(a)
    heading = float(np.arctan2(np.cos(a), -np.sin(a)))   # вдоль касательной
    return (x, z), heading


# ----------------------------------------------------------------------
# рекорды
def _records_path():
    return os.path.join(os.path.expanduser("~"), ".beamng_lite", "records.json")


def load_best(mode, vehicle_id, seed):
    try:
        with open(_records_path(), encoding="utf-8") as f:
            data = json.load(f)
        return data.get(mode, {}).get("%s:%d" % (vehicle_id, seed))
    except (OSError, ValueError):
        return None


def save_best(mode, vehicle_id, seed, value):
    path = _records_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
        data.setdefault(mode, {})["%s:%d" % (vehicle_id, seed)] = value
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    except OSError:
        pass                                  # рекорды — не критично


# ----------------------------------------------------------------------
class FreeRoam:
    """Свободная езда по карте."""
    id = "free"

    def spawn(self):
        return (0.0, -6.0), 0.0

    def start(self, game):
        pass

    def update(self, veh, terrain, dt):
        return ""

    def hud_line(self):
        return ""

    def draw_markers(self, r, terrain):
        pass


class TimeTrial:
    """Круг по кольцу чекпоинтов на время."""
    id = "time_trial"

    def __init__(self):
        self.next_cp = 0
        self.running = False
        self.t = 0.0
        self.lap = 0
        self.last_lap = None
        self.best = None

    def spawn(self):
        return trial_spawn()

    def start(self, game):
        self.next_cp = 0
        self.running = False
        self.t = 0.0
        self.lap = 0
        self.last_lap = None
        self.best = load_best(self.id, game.vehicle_id, game.seed)

    def _near(self, c, i):
        p = ring_point(i)
        return abs(c[0] - p[0]) + abs(c[2] - p[2]) < CP_RADIUS

    def update(self, veh, terrain, dt):
        c = veh.frame()[0]
        if not self.running:
            if self._near(c, 0):              # пересекли старт — поехали
                self.running = True
                self.t = 0.0
                self.lap += 1
                self.next_cp = 1
            return
        self.t += dt
        if self._near(c, self.next_cp):
            if self.next_cp == 0:             # финишная воротина
                self.last_lap = self.t
                self.running = False
                if self.best is None or self.t < self.best:
                    self.best = self.t
                    save_best(self.id, veh.cfg["id"], terrain.seed, self.t)
            else:
                self.next_cp = (self.next_cp + 1) % N_CPS

    def hud_line(self):
        best = "%.2f с" % self.best if self.best is not None else "—"
        if self.running:
            return "ТАЙМ-ТРАЙЛ: круг %d  %6.2f с  ворота %d/%d  (лучшее %s)" % (
                self.lap, self.t, self.next_cp, N_CPS, best)
        if self.last_lap is not None:
            return "Круг пройден: %.2f с!  лучшее %s  — к старту за новым кругом" % (
                self.last_lap, best)
        return "ТАЙМ-ТРАЙЛ: к стартовой воротине  (лучшее %s)" % best

    def draw_markers(self, r, terrain):
        for i in range(N_CPS):
            p = ring_point(i)
            if not self.running and i == 0:
                col = (0.30, 0.85, 0.45)      # стартовая воротина — зелёная
            elif self.running and i == self.next_cp:
                col = (1.00, 0.62, 0.10)      # следующая — оранжевая
            elif i == 0:
                col = (0.55, 0.30, 0.75)      # финишная — фиолетовая
            else:
                col = (0.30, 0.32, 0.36)      # остальные — тёмно-серые
            r.draw_pylon(p[0], float(terrain.height(p[0], p[2])), p[2], col)


class Drift:
    """Скоринг: очки за скорость в заносе, комбо растёт пока не выровнялся."""
    id = "drift"

    def __init__(self):
        self.score = 0.0
        self.combo = 1.0
        self.best = None

    def spawn(self):
        return (0.0, -6.0), 0.0

    def start(self, game):
        self.score = 0.0
        self.combo = 1.0
        self.best = load_best(self.id, game.vehicle_id, game.seed)
        if self.best is not None:
            self.score = float(self.best)     # цель — побить рекорд

    def update(self, veh, terrain, dt):
        c, fwd, rgt, up = veh.frame()
        v = veh.velocity()
        sp = float(np.linalg.norm(v))
        if sp > 3.0:
            sin_a = abs(float(np.cross(fwd, v)[1])) / sp   # |sin угла заноса|
        else:
            sin_a = 0.0
        if sin_a > 0.22 and sp > 5.0:
            self.combo = min(self.combo + dt * 0.6, 6.0)
            self.score += sp * 3.6 * sin_a * self.combo * dt
        else:
            self.combo = max(1.0, self.combo - dt * 2.5)

    def hud_line(self):
        best = "%.0f" % self.best if self.best else "—"
        return "ДРИФТ: %d очков  комбо ×%.1f  (лучшее %s)" % (
            round(self.score), self.combo, best)

    def draw_markers(self, r, terrain):
        pass


_CLASSES = {"free": FreeRoam, "time_trial": TimeTrial, "drift": Drift}


def make(mode_id):
    return _CLASSES[mode_id]()
