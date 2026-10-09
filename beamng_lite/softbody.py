"""Мягкотельная (soft-body) физика: узлы + балки, как в BeamNG.

Модель:
  * узел  — материальная точка (масса, позиция, скорость);
  * балка — пружина-демпфер между двумя узлами с пластической деформацией
            и порогом разрыва.
Всё считается векторизованно на numpy, интегратор — полунеявный Эйлер
с несколькими подшагами на кадр.
"""
import numpy as np

GRAVITY = np.array([0.0, -9.81, 0.0])


class SoftBody:
    def __init__(self):
        self.pos = np.zeros((0, 3))
        self.vel = np.zeros((0, 3))
        self.mass = np.zeros(0)
        self.inv_mass = np.zeros(0)
        self.node_meta = []                 # произвольные метки узлов

        self.beams = np.zeros((0, 2), dtype=np.int64)
        self.rest = np.zeros(0)
        self.rest0 = np.zeros(0)            # исходная длина (для оценки повреждений)
        self.k = np.zeros(0)                # жёсткость, Н/м
        self.d = np.zeros(0)                # демпфирование, Н·с/м
        self.deform = np.zeros(0)           # порог пластики (отн. деформация)
        self.brk = np.zeros(0)              # порог разрыва
        self.alive = np.zeros(0, dtype=bool)

    # ---------- построение ----------
    def add_node(self, p, mass, meta=None):
        self.pos = np.vstack([self.pos, np.asarray(p, dtype=np.float64)])
        self.vel = np.vstack([self.vel, np.zeros(3)])
        self.mass = np.append(self.mass, mass)
        self.inv_mass = np.append(self.inv_mass, 1.0 / mass)
        self.node_meta.append(meta or {})
        return len(self.mass) - 1

    def add_beam(self, i, j, k=1.1e6, zeta=0.25, deform=0.06, brk=0.55,
                 rest=None, d=None):
        """zeta — коэффициент затухания (0..1); демпфер считается от приведённой
        массы пары узлов, что гарантирует устойчивость явного интегратора."""
        L = float(np.linalg.norm(self.pos[i] - self.pos[j])) if rest is None else rest
        if d is None:
            mr = self.mass[i] * self.mass[j] / (self.mass[i] + self.mass[j])
            d = 2.0 * zeta * np.sqrt(k * mr)
        self.beams = np.vstack([self.beams, [i, j]])
        self.rest = np.append(self.rest, L)
        self.rest0 = np.append(self.rest0, L)
        self.k = np.append(self.k, k)
        self.d = np.append(self.d, d)
        self.deform = np.append(self.deform, deform)
        self.brk = np.append(self.brk, brk)
        self.alive = np.append(self.alive, True)
        return len(self.rest) - 1

    def finalize(self):
        self.beams = self.beams.astype(np.int64)

    # ---------- силы ----------
    def beam_forces(self, force):
        """Силы пружин-демпферов + пластика + разрывы. force: (N,3) аккумулятор."""
        if len(self.rest) == 0:
            return
        a = self.beams[:, 0]
        b = self.beams[:, 1]
        dv = self.pos[b] - self.pos[a]
        L = np.linalg.norm(dv, axis=1)
        L = np.maximum(L, 1e-9)
        dirv = dv / L[:, None]
        strain = (L - self.rest) / self.rest

        # --- разрыв ---
        broke = self.alive & (np.abs(strain) > self.brk)
        if broke.any():
            self.alive[broke] = False

        # --- пластическая деформация: остаточное изменение длины покоя ---
        over = np.abs(strain) - self.deform
        plastic = self.alive & (over > 0)
        if plastic.any():
            sgn = np.sign(strain[plastic])
            self.rest[plastic] += sgn * over[plastic] * self.rest[plastic] * 0.35

        rel = self.vel[b] - self.vel[a]
        vrel = np.einsum("ij,ij->i", rel, dirv)
        f = (self.k * (L - self.rest) + self.d * vrel) * self.alive
        fv = dirv * f[:, None]
        np.add.at(force, a, fv)
        np.add.at(force, b, -fv)

    def damage(self):
        """Повреждение 0..1: среднее по самой пострадавшей четверти балок.

        Среднее по всем балкам бесполезно — в лобовом ударе мнётся только нос,
        и метрика показывала бы 2% на разбитой вдребезги машине.
        """
        if len(self.rest) == 0:
            return 0.0
        plastic = np.abs(self.rest - self.rest0) / self.rest0
        dmg = np.clip(plastic / 0.12, 0, 1)
        dmg = np.where(self.alive, dmg, 1.0)
        k = max(1, len(dmg) // 4)
        worst = np.partition(dmg, -k)[-k:]
        return float(worst.mean())

    def beam_stress(self):
        """Относительная остаточная деформация каждой балки (для окраски)."""
        return np.clip(np.abs(self.rest - self.rest0) / (self.rest0 * 0.3), 0, 1)
