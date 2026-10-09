"""Ландшафт: процедурная карта высот + статические препятствия."""
import numpy as np


class Terrain:
    """Карта высот с билинейной интерполяцией и аналитическими нормалями."""

    def __init__(self, size=400.0, res=224, seed=7):
        self.size = size          # физический размер (метры), центр в (0,0)
        self.res = res            # число ячеек по стороне
        self.cell = size / res
        self.x0 = -size * 0.5
        self.z0 = -size * 0.5
        self.seed = seed
        rng = np.random.default_rng(seed)

        gx = np.linspace(self.x0, self.x0 + size, res + 1)
        gz = np.linspace(self.z0, self.z0 + size, res + 1)
        X, Z = np.meshgrid(gx, gz, indexing="ij")

        # --- фрактальный value-noise: своя сетка на октаву, без наложения ---
        h = np.zeros_like(X)
        amp = 15.0
        cells = 3
        for _ in range(5):        # без высокочастотной «тряски» — рельеф едется
            n = rng.random((cells + 1, cells + 1))
            h += amp * self._octave(X, Z, n)
            amp *= 0.47
            cells *= 2
        h -= h.mean()

        # --- плоская зона старта (радиус 45 м) ---
        r = np.sqrt(X ** 2 + Z ** 2)
        flat = np.clip((r - 30.0) / 40.0, 0.0, 1.0)
        flat = flat * flat * (3 - 2 * flat)          # smoothstep
        h *= flat

        # --- трамплины: наклонные "клинья" ---
        h += self._ramp(X, Z, cx=0, cz=45, w=7.0, length=14.0, height=3.2, ang=0.0)
        h += self._ramp(X, Z, cx=-40, cz=-15, w=6.0, length=12.0, height=2.4, ang=np.pi / 2)
        # --- горка-волна для проверки подвески ---
        h += 0.75 * np.exp(-((X - 25) ** 2 + (Z - 10) ** 2) / 90.0)
        h += 0.75 * np.exp(-((X - 32) ** 2 + (Z + 2) ** 2) / 90.0)
        h += 0.75 * np.exp(-((X - 18) ** 2 + (Z + 14) ** 2) / 90.0)

        self.h = h.astype(np.float64)

        # статические ящики-препятствия (центр, полуразмеры) — их можно сносить
        # --- препятствия: стена для крэш-тестов, пирамида, слалом, блоки ---
        self.boxes = []
        masses = []

        def add(bx, bz, hx, hy, hz, mass, lift=0.0):
            by = self.height(bx, bz) + hy + lift
            self.boxes.append([bx, by, bz, hx, hy, hz])
            masses.append(mass)

        # прямая по курсу (x около 0) оставлена свободной — там трамплины
        add(17, 26, 5.0, 1.4, 0.5, 1.0e12)             # стена для крэш-тестов
        add(-21, 15, 2.5, 1.0, 0.5, 1.0e12)            # вторая стена, пониже
        for x in (9.0, 11.0, 13.0):                    # пирамида из ящиков
            add(x, 8, 0.9, 0.9, 0.9, 240)
        add(10.0, 8, 0.9, 0.9, 0.9, 240, lift=1.8)
        add(12.0, 8, 0.9, 0.9, 0.9, 240, lift=1.8)
        add(11.0, 8, 0.9, 0.9, 0.9, 240, lift=3.6)
        for z in range(6, 27, 5):                      # слалом из лёгких ящиков
            add(-9.0, z, 0.5, 0.5, 0.5, 90)
        add(10, -20, 1.6, 1.2, 1.6, 2600)              # бетонные блоки
        add(-17, -26, 1.6, 1.2, 1.6, 2600)
        add(24, -8, 1.2, 1.0, 1.2, 900)
        for (bx, bz) in [(-14, 48), (-10, 54), (12, -54), (16, -50),
                         (0, 70), (34, 36), (-38, -14), (44, -44)]:
            add(bx, bz, 1.0, 1.0, 1.0, 850)            # разбросанные по карте
        self.boxes = np.array(self.boxes, dtype=np.float64)
        self.box0 = self.boxes.copy()                 # исходные позиции (для респавна)
        self.box_mass = np.array(masses, dtype=np.float64)
        self.box_vel = np.zeros((len(self.boxes), 3))
        self.box_force = np.zeros((len(self.boxes), 3))
        self.box_hit = np.zeros(len(self.boxes))      # таймер подсветки удара

    # ---------- вспомогательное ----------
    def _octave(self, X, Z, n):
        """Одна октава value-noise с кубическим сглаживанием (без швов)."""
        cells = n.shape[0] - 1
        u = (X - self.x0) / self.size * cells
        v = (Z - self.z0) / self.size * cells
        i = np.clip(np.floor(u).astype(int), 0, cells - 1)
        j = np.clip(np.floor(v).astype(int), 0, cells - 1)
        fu, fv = u - i, v - j
        su = fu * fu * (3 - 2 * fu)
        sv = fv * fv * (3 - 2 * fv)
        a = n[i, j]
        b = n[i + 1, j]
        c = n[i, j + 1]
        d = n[i + 1, j + 1]
        return (a * (1 - su) * (1 - sv) + b * su * (1 - sv)
                + c * (1 - su) * sv + d * su * sv) - 0.5

    @staticmethod
    def _ramp(X, Z, cx, cz, w, length, height, ang):
        ca, sa = np.cos(ang), np.sin(ang)
        dx, dz = X - cx, Z - cz
        lx = dx * ca + dz * sa      # поперёк
        lz = -dx * sa + dz * ca     # вдоль подъёма
        t = np.clip((lz + length * 0.5) / length, 0.0, 1.0)
        side = np.clip(1.0 - (np.abs(lx) - w * 0.5) / 1.5, 0.0, 1.0)
        inside = (np.abs(lz) <= length * 0.5)
        return height * t * side * inside

    # ---------- подвижные препятствия ----------
    def push_box(self, i, impulse):
        """impulse = сила * шаг подшага (Н·с) — иначе импульс завышается в N раз."""
        self.box_force[i] += impulse
        self.box_hit[i] = 0.4

    def step_boxes(self, dt):
        if len(self.boxes) == 0:
            return
        self.box_vel += self.box_force / self.box_mass[:, None]
        self.box_vel[:, 1] -= 9.81 * dt
        self.box_force[:] = 0.0
        self.boxes[:, :3] += self.box_vel * dt
        # приземление и трение о грунт
        gh = self.height(self.boxes[:, 0], self.boxes[:, 2]) + self.boxes[:, 4]
        low = self.boxes[:, 1] < gh + 0.02
        if low.any():
            self.boxes[low, 1] = np.maximum(self.boxes[low, 1], gh[low])
            self.box_vel[low, 1] = np.maximum(self.box_vel[low, 1], 0.0)
            # сухое трение о грунт: ящик не катится по льду сотни метров
            hv = self.box_vel[:, [0, 2]]
            sp = np.linalg.norm(hv, axis=1)
            drop = np.where(low, 0.7 * 9.81 * dt, 0.0)
            scale = np.where(sp > 1e-6, np.maximum(sp - drop, 0.0) / np.maximum(sp, 1e-6), 0.0)
            self.box_vel[:, 0] *= scale
            self.box_vel[:, 2] *= scale
        np.clip(self.box_vel, -60, 60, out=self.box_vel)
        self.box_hit = np.maximum(self.box_hit - dt, 0.0)

    def reset_boxes(self):
        self.boxes[:] = self.box0
        self.box_vel[:] = 0.0
        self.box_force[:] = 0.0
        self.box_hit[:] = 0.0

    # ---------- запросы ----------
    def height(self, x, z):
        """Высота в точке (скаляр или массивы)."""
        u = (np.asarray(x) - self.x0) / self.cell
        v = (np.asarray(z) - self.z0) / self.cell
        u = np.clip(u, 0, self.res - 1e-6)
        v = np.clip(v, 0, self.res - 1e-6)
        i = u.astype(np.int64)
        j = v.astype(np.int64)
        fu, fv = u - i, v - j
        h = self.h
        return (h[i, j] * (1 - fu) * (1 - fv) + h[i + 1, j] * fu * (1 - fv)
                + h[i, j + 1] * (1 - fu) * fv + h[i + 1, j + 1] * fu * fv)

    def normal(self, x, z):
        """Нормаль поверхности (нормированная), поддерживает массивы."""
        d = self.cell
        hx = self.height(np.asarray(x) + d, z) - self.height(np.asarray(x) - d, z)
        hz = self.height(x, np.asarray(z) + d) - self.height(x, np.asarray(z) - d)
        n = np.stack([-hx, np.full_like(np.asarray(hx, dtype=np.float64), 2 * d), -hz], axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    # ---------- меш для рендера ----------
    def build_mesh(self, night=False):
        """Возвращает (вершины Nx3 float32, цвета Nx3 float32, индексы uint32)."""
        res = self.res
        gx = self.x0 + np.arange(res + 1) * self.cell
        gz = self.z0 + np.arange(res + 1) * self.cell
        X, Z = np.meshgrid(gx, gz, indexing="ij")
        Y = self.h
        verts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

        nrm = self.normal(X, Z)
        sun = np.array([0.45, 0.8, 0.35])
        sun /= np.linalg.norm(sun)
        lam = np.clip(nrm @ sun, 0.0, 1.0) * 0.75 + 0.25

        slope = 1.0 - nrm[..., 1]
        grass_dark = np.array([0.22, 0.34, 0.15])
        grass = np.array([0.33, 0.47, 0.21])
        dirt = np.array([0.44, 0.36, 0.24])
        rock = np.array([0.44, 0.42, 0.39])

        # пятна травы низкой частоты — земля читается, но без ряби
        rng2 = np.random.default_rng(self.seed + 77)
        patch = np.clip(0.5 + 3.0 * self._octave(X, Z, rng2.random((11, 11))), 0, 1)
        base = grass_dark * (1 - patch[..., None]) + grass * patch[..., None]

        t_dirt = np.clip((slope - 0.015) * 22.0, 0, 1)[..., None]
        base = base * (1 - t_dirt) + dirt * t_dirt
        t_rock = np.clip((slope - 0.09) * 12.0, 0, 1)[..., None]
        base = base * (1 - t_rock) + rock * t_rock

        # грунтовая трасса: кольцо вокруг площадки + прямая «разгонная»
        ring = np.abs(np.sqrt(X ** 2 + Z ** 2) - 72.0)
        strip = np.maximum(np.abs(X) - 4.5, 0.0) + np.maximum(np.abs(Z) - 72.0, 0.0)
        road = np.clip(1.0 - np.minimum(ring - 4.5, strip) / 2.0, 0.0, 1.0)[..., None] * 0.85
        base = base * (1 - road) + np.array([0.46, 0.40, 0.31]) * road

        # площадка старта — укатанный грунт
        r0 = np.sqrt(X ** 2 + Z ** 2)
        pad = np.clip((13.0 - r0) / 5.0, 0.0, 1.0)[..., None] * 0.75
        base = base * (1 - pad) + np.array([0.42, 0.38, 0.30]) * pad

        base = base * (0.94 + 0.10 * np.clip(Y[..., None] / 14.0 + 0.5, 0, 1))
        if night:
            # луна с противоположной стороны: холодный свет + тёмно-синий заполняющий
            moon = np.array([-0.45, 0.75, -0.45])
            moon /= np.linalg.norm(moon)
            ml = np.clip(nrm @ moon, 0.0, 1.0) * 0.55 + 0.08
            cols = base * (ml[..., None] * 0.45) + np.array([0.05, 0.06, 0.10])
        else:
            cols = base * lam[..., None]
        cols = cols.reshape(-1, 3).astype(np.float32)

        idx = np.arange((res + 1) * (res + 1)).reshape(res + 1, res + 1)
        a = idx[:-1, :-1].ravel()
        b = idx[1:, :-1].ravel()
        c = idx[1:, 1:].ravel()
        d = idx[:-1, 1:].ravel()
        # обход против часовой при взгляде сверху — нормали вверх
        tris = np.stack([a, c, b, a, d, c], axis=1).ravel().astype(np.uint32)
        return verts, cols, tris
