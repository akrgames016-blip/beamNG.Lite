"""Плотная «шкура» кузова: гладкий меш, натянутый на узлы soft-body.

Физика считает 36 узлов, а рисуем мы ~400 вершин: каждая вершина привязана
к четырём ближайшим узлам с весами (линейный скиннинг), поэтому меш мнётся
вместе с рамой — как в настоящем BeamNG.
"""
import numpy as np

NZ = 16          # колец вдоль кузова
NR = 16          # точек в кольце (углы профиля = рёбра кузова)
K = 4            # к скольким узлам привязана вершина

BODY_COL = np.array([0.80, 0.13, 0.11])
GLASS_COL = np.array([0.13, 0.17, 0.23])
BROKEN_COL = np.array([0.05, 0.05, 0.06])     # выбитое стекло
FLOOR_COL = np.array([0.09, 0.09, 0.10])
HEAD_COL = np.array([1.00, 0.95, 0.78])
TAIL_COL = np.array([0.88, 0.10, 0.07])
TRIM_COL = np.array([0.17, 0.17, 0.19])       # бампера, зеркала
GRILLE_COL = np.array([0.07, 0.07, 0.08])


def _catmull(p0, p1, p2, p3, t):
    t2, t3 = t * t, t * t * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


def _profile(sections, t):
    """Гладкая интерполяция сечений (z, полуширина, низ, верх) по t в [0,1]."""
    arr = np.asarray(sections, dtype=np.float64)
    n = len(arr)
    x = t * (n - 1)
    i = int(np.clip(np.floor(x), 0, n - 2))
    f = x - i
    p0 = arr[max(i - 1, 0)]
    p1 = arr[i]
    p2 = arr[i + 1]
    p3 = arr[min(i + 2, n - 1)]
    return _catmull(p0, p1, p2, p3, f)


def _ring(hw, ylo, yhi, belt):
    """Сечение кузова. Точки контура сохраняются как есть — при плоской
    заливке каждая из них становится ребром: борт, линия окон, кромка крыши."""
    h = max(yhi - ylo, 1e-6)
    bt = float(np.clip((belt - ylo) / h, 0.28, 0.88))
    right = [(0.16, 0.00), (0.74, 0.015), (0.97, 0.10), (1.00, 0.30),
             (1.00, bt), (0.93, bt + (1 - bt) * 0.45),
             (0.72, 0.96), (0.26, 1.00)]
    pts = [(fx * hw, ylo + fy * h) for (fx, fy) in right]
    pts += [(-x, y) for (x, y) in reversed(pts)]
    return np.array(pts, dtype=np.float64)


class Skin:
    def __init__(self, verts_local, tri_idx, node_idx, weights, offsets, base_col,
                 face_col, cage, glass_mask):
        self.n = len(verts_local)
        self.local = verts_local
        self.tri = tri_idx
        self.node_idx = node_idx          # (V, K) индексы узлов
        self.weights = weights            # (V, K)
        self.offsets = offsets            # (V, K, 3) смещения в локальных осях
        self.base_col = base_col          # (V, 3)
        self._out = np.zeros((self.n, 3), dtype=np.float64)
        # плоская заливка: у каждого треугольника свои вершины и свой цвет
        self.flat_idx = tri_idx.ravel()
        self.nf = len(tri_idx)
        self.face_col = face_col
        self.cage_idx, self.cage_nodes, self.cage_w, self.cage_off, self.cage_r = cage
        self.glass_mask = glass_mask
        self._fv = np.zeros((self.nf * 3, 3), dtype=np.float32)
        self._fc = np.zeros((self.nf * 3, 3), dtype=np.float32)

    def deform(self, pos, R):
        """Мировые координаты вершин: сумма w * (узел + R * смещение)."""
        base = pos[self.node_idx]                       # (V, K, 3)
        off = self.offsets @ R.T                        # (V, K, 3)
        self._out[:] = np.einsum("vk,vkj->vj", self.weights, base + off)
        return self._out

    def facets(self, verts, node_damage, sun):
        """Треугольники с плоской заливкой: рёбра видно, «мыла» нет."""
        tv = verts[self.flat_idx]                       # (3F, 3)
        a = tv[0::3]
        b = tv[1::3]
        c = tv[2::3]
        n = np.cross(b - a, c - a)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        lam = 0.34 + 0.66 * np.abs(n @ sun)
        vd = np.einsum("vk,vk->v", self.weights, node_damage[self.node_idx])
        fd = vd[self.tri].mean(axis=1)[:, None]
        col = self.face_col * (1.0 - 0.55 * fd) + np.array([0.23, 0.20, 0.18]) * fd
        broken = self.glass_mask & (fd[:, 0] > 0.28)      # стекло выбито
        if broken.any():
            col[broken] = BROKEN_COL
        self._fv[:] = tv
        self._fc[:] = np.repeat(np.clip(col * lam[:, None], 0, 1), 3, axis=0)
        return self._fv, self._fc


def _box_geom(cx, cy, cz, hx, hy, hz):
    """Кубик (8 вершин, 12 треугольников) с нормалями наружу."""
    v = np.array([[cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
                  [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
                  [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
                  [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz]])
    q = [(0, 1, 2, 3), (5, 4, 7, 6), (4, 0, 3, 7),
         (1, 5, 6, 2), (4, 5, 1, 0), (3, 2, 6, 7)]
    t = []
    for (a, b, c, d) in q:
        t.append((a, b, c))
        t.append((a, c, d))
    return v, np.array(t, dtype=np.uint32)


def build_skin(veh):
    cfg = veh.cfg
    SECTIONS = [tuple(s) for s in cfg["sections"]]
    BELT_Y = float(cfg.get("belt_y", 0.97))
    BODY_COL = np.asarray(cfg.get("color", [0.80, 0.13, 0.11]), dtype=np.float64)

    # --- вершины в локальных координатах ---
    verts = []
    for i in range(NZ):
        t = i / (NZ - 1)
        z, hw, ylo, yhi = _profile(SECTIONS, t)
        # к носу и корме кузов сужается — иначе торцы выглядят обрубками
        taper = np.sin(np.pi * min(max(t, 0.0), 1.0)) ** 0.16
        belt = float(np.clip(BELT_Y, ylo + 0.18, yhi - 0.05))
        ring = _ring(hw * (0.82 + 0.18 * taper), ylo, yhi, belt)
        for (x, y) in ring:
            verts.append((x, y, z))
    verts = np.array(verts, dtype=np.float64)

    # --- треугольники (кольца замкнуты, торцы затягиваем веером к оси) ---
    tris = []
    for i in range(NZ - 1):
        for j in range(NR):
            a = i * NR + j
            b = i * NR + (j + 1) % NR
            c = (i + 1) * NR + (j + 1) % NR
            d = (i + 1) * NR + j
            tris.append((a, b, c))
            tris.append((a, c, d))
    for (ring0, sign) in ((0, -1), (NZ - 1, 1)):
        base = ring0 * NR
        for j in range(1, NR - 1):
            if sign < 0:
                tris.append((base, base + j + 1, base + j))
            else:
                tris.append((base, base + j, base + j + 1))
    tris = np.array(tris, dtype=np.uint32)

    # --- детали: боковые зеркала ---
    detail_first = len(verts)
    extra_v, extra_t, extra_mat = [], [], []
    z_mid, hw_mid = _profile(SECTIONS, 0.62)[0], _profile(SECTIONS, 0.62)[1]
    for sx in (-1.0, 1.0):
        mv, mt = _box_geom(sx * (hw_mid + 0.11), 1.06, 0.62, 0.10, 0.055, 0.035)
        extra_t.append(mt + len(verts) + sum(len(v) for v in extra_v))
        extra_v.append(mv)
        extra_mat.append(np.full(len(mt), 5))
    verts = np.vstack([verts] + extra_v)
    tris = np.vstack([tris] + extra_t)

    # --- координаты узлов кузова в тех же локальных осях ---
    node_local = {}
    for (ix, iy, iz), nid in veh.grid.items():
        z, hw, ylo, yhi = SECTIONS[iz]
        node_local[nid] = np.array([(-1 if ix == 0 else 1) * hw,
                                    veh.level_y[iz][iy], z])
    ids = np.array(sorted(node_local.keys()))
    npos = np.array([node_local[i] for i in ids])

    # --- привязка: K ближайших узлов, веса ~ 1/d² ---
    d = np.linalg.norm(verts[:, None, :] - npos[None, :, :], axis=2)
    order = np.argsort(d, axis=1)[:, :K]
    dk = np.take_along_axis(d, order, axis=1)
    w = 1.0 / (dk ** 2 + 0.04)
    w /= w.sum(axis=1, keepdims=True)
    node_idx = ids[order]
    offsets = verts[:, None, :] - npos[order]

    # --- материал по ГРАНЯМ: границы стёкол и фар получаются резкими ---
    tri_c = verts[tris].mean(axis=1)
    ta, tb, tc = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    fn = np.cross(tb - ta, tc - ta)          # наружу от кузова
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9)

    zs = np.array([_profile(SECTIONS, i / (NZ - 1))[0] for i in range(NZ)])
    roofs = np.array([_profile(SECTIONS, i / (NZ - 1))[3] for i in range(NZ)])
    hws = np.array([_profile(SECTIONS, i / (NZ - 1))[1] for i in range(NZ)])
    roof_at = np.interp(tri_c[:, 2], zs, roofs)
    hw_at = np.interp(tri_c[:, 2], zs, hws)

    fx, fy, fz = tri_c[:, 0], tri_c[:, 1], tri_c[:, 2]
    zmax, zmin = verts[:, 2].max(), verts[:, 2].min()
    face_col = np.tile(BODY_COL, (len(tris), 1))

    # стёкла: боковые окна, лобовое и заднее — по зонам, а не «поясом»
    side_win = (np.abs(fx) > 0.42 * hw_at) & (fy > BELT_Y) & \
        (fy < roof_at - 0.06) & (fz > -1.30) & (fz < 1.00)
    windshield = (fz > 0.42) & (fz < 1.30) & (fy > 0.98) & (fn[:, 1] < 0.86)
    rear_win = (fz > -1.45) & (fz < -0.42) & (fy > 1.00) & (fn[:, 1] < 0.86)
    glass = side_win | windshield | rear_win
    face_col[glass] = GLASS_COL

    # днище
    face_col[fn[:, 1] < -0.45] = FLOOR_COL
    # бампера и пороги
    rocker = fy < 0.545                                   # пороги — ровной полосой
    bumper = (np.abs(fz) > 1.72) & (fy < 0.78)            # бампера
    face_col[rocker | bumper] = TRIM_COL
    # решётка радиатора
    grille = (fz > zmax - 0.30) & (fy > 0.60) & (fy < 0.86) & (np.abs(fx) < 0.30)
    face_col[grille] = GRILLE_COL
    # фары и фонари
    head = (fz > zmax - 0.28) & (fy > 0.62) & (fy < 0.88) & (np.abs(fx) > 0.30) & (np.abs(fx) < 0.62)
    tail = (fz < zmin + 0.26) & (fy > 0.64) & (fy < 0.92) & (np.abs(fx) > 0.24) & (np.abs(fx) < 0.62)
    face_col[head] = HEAD_COL
    face_col[tail] = TAIL_COL
    # зеркала и прочие детали
    detail = np.arange(len(tris)) >= (len(tris) - sum(len(m) for m in extra_mat))
    face_col[detail] = TRIM_COL
    glass = glass & ~detail
    base_col = np.tile(BODY_COL, (len(verts), 1))

    # --- клетка контактных точек: по ней считаются удары о препятствия ---
    # (иначе небольшой ящик проскакивает между угловыми узлами кузова)
    all_idx = np.arange(len(verts))
    ring_i, ring_j = all_idx // NR, all_idx % NR
    ends = (ring_i <= 1) | (ring_i >= NZ - 2)         # нос и корма — все точки
    sel = all_idx[ends | (ring_j % 2 == 0)]
    sel = sel[sel < detail_first]        # зеркала в контактной клетке не нужны
    cage = (sel, node_idx[sel], w[sel], offsets[sel],
            np.full(len(sel), 0.09))

    return Skin(verts, tris, node_idx, w, offsets, base_col, face_col, cage, glass)
