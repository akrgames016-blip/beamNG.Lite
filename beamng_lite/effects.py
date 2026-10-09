"""Эффекты столкновений: обломки, пыль, тряска камеры."""
import numpy as np

MAX = 320


class Particles:
    """Простой пул частиц: обломки кузова, стекло, комья земли."""

    def __init__(self, n=MAX):
        self.pos = np.zeros((n, 3))
        self.vel = np.zeros((n, 3))
        self.life = np.zeros(n)
        self.size = np.zeros(n)
        self.col = np.zeros((n, 3))
        self.n = n
        self._next = 0

    def emit(self, count, origin, base_vel, colors, spread=6.0, life=2.2, size=0.10):
        count = int(min(count, self.n))
        if count <= 0:
            return
        idx = (np.arange(count) + self._next) % self.n
        self._next = int((self._next + count) % self.n)
        rng = np.random.default_rng()
        self.pos[idx] = origin + rng.normal(0.0, 0.22, (count, 3))
        v = rng.normal(0.0, spread, (count, 3))
        v[:, 1] = np.abs(v[:, 1]) * 0.8 + 2.0
        self.vel[idx] = v + base_vel
        self.life[idx] = life * (0.6 + 0.8 * rng.random(count))
        self.size[idx] = size * (0.5 + rng.random(count))
        pick = rng.integers(0, len(colors), count)
        self.col[idx] = np.asarray(colors)[pick]

    def update(self, dt, terrain):
        alive = self.life > 0
        if not alive.any():
            return
        self.life[alive] -= dt
        self.vel[alive, 1] -= 9.81 * dt
        self.vel[alive] *= (1.0 - 0.6 * dt)
        self.pos[alive] += self.vel[alive] * dt
        p = self.pos[alive]
        gh = terrain.height(p[:, 0], p[:, 2]) + 0.03
        low = p[:, 1] < gh
        if low.any():
            ids = np.nonzero(alive)[0][low]
            self.pos[ids, 1] = gh[low]
            self.vel[ids, 1] = np.abs(self.vel[ids, 1]) * 0.32
            self.vel[ids, 0] *= 0.72
            self.vel[ids, 2] *= 0.72
            self.life[ids] *= 0.92

    def active(self):
        m = self.life > 0
        return self.pos[m], self.col[m], self.size[m], np.clip(self.life[m], 0, 1)


class Smoke:
    """Дым от шин и выхлопа: лёгкие частицы, всплывают вверх и тают."""

    def __init__(self, n=256):
        self.pos = np.zeros((n, 3))
        self.vel = np.zeros((n, 3))
        self.life = np.zeros(n)
        self.maxlife = np.zeros(n)
        self.size = np.zeros(n)
        self.n = n
        self._next = 0

    def emit(self, count, origin):
        count = int(min(count, self.n))
        if count <= 0:
            return
        idx = (np.arange(count) + self._next) % self.n
        self._next = int((self._next + count) % self.n)
        rng = np.random.default_rng()
        self.pos[idx] = origin + rng.normal(0.0, 0.15, (count, 3))
        v = rng.normal(0.0, 0.5, (count, 3))
        v[:, 1] = rng.uniform(0.8, 2.2, count)          # всплывает
        self.vel[idx] = v
        self.maxlife[idx] = rng.uniform(1.2, 2.6, count)
        self.life[idx] = self.maxlife[idx]
        self.size[idx] = rng.uniform(0.15, 0.35, count)

    def update(self, dt, terrain):
        alive = self.life > 0
        if not alive.any():
            return
        self.life[alive] -= dt
        self.pos[alive] += self.vel[alive] * dt
        self.vel[alive, 1] *= (1.0 - 0.4 * dt)
        self.vel[alive][:, [0, 2]] *= (1.0 - 0.8 * dt)
        p = self.pos[alive]
        gh = terrain.height(p[:, 0], p[:, 2]) + 0.05
        low = p[:, 1] < gh
        if low.any():
            ids = np.nonzero(alive)[0][low]
            self.pos[ids, 1] = gh[low]

    def active(self):
        m = self.life > 0
        fade = np.clip(self.life[m] / np.maximum(self.maxlife[m], 1e-9), 0, 1)
        return self.pos[m], self.size[m] * (0.5 + 0.9 * fade), fade
