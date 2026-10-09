"""Рендер на PyOpenGL (фиксированный конвейер + вершинные массивы)."""
import numpy as np
import pygame
from OpenGL.GL import *

try:    # «сырые» вызовы: не трогают учёт контекста PyOpenGL, работают везде
    from OpenGL.raw.GL.VERSION.GL_1_1 import (
        glVertexPointer as rawVertexPointer,
        glColorPointer as rawColorPointer,
        glDrawElements as rawDrawElements)
    import ctypes
    _NULL = ctypes.c_void_p(0)
except ImportError:                     # пусть будет запасной путь
    rawVertexPointer = None

from .body import NZ, NR

SKY_TOP = (0.36, 0.56, 0.86)
SKY_BOT = (0.74, 0.84, 0.93)
SUN = np.array([0.42, 0.82, 0.38])
SUN /= np.linalg.norm(SUN)

_CIRC = [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, 17)]


class Renderer:
    def __init__(self, width, height, terrain):
        self.w, self.h = width, height
        self.terrain = terrain
        self.night = False
        self.sky_top = SKY_TOP
        self.sky_bot = SKY_BOT
        self.fog_start, self.fog_end = 160.0, 520.0
        glClearColor(*SKY_BOT, 1.0)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)            # ландшафт двусторонний — ничего не пропадает
        glShadeModel(GL_SMOOTH)
        glEnable(GL_FOG)
        glFogi(GL_FOG_MODE, GL_LINEAR)
        glFogfv(GL_FOG_COLOR, (*SKY_BOT, 1.0))
        glFogf(GL_FOG_START, self.fog_start)
        glFogf(GL_FOG_END, self.fog_end)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        self.tex = glGenTextures(1)
        self._tex_size = (0, 0)
        self.vbo = None
        self.set_terrain(terrain)

    # ------------------------------------------------------------------
    def set_night(self, night):
        """День/ночь: небо, туман и палитра ландшафта меняются сразу."""
        self.night = bool(night)
        if self.night:
            self.sky_top = (0.015, 0.02, 0.06)
            self.sky_bot = (0.07, 0.09, 0.15)
            self.fog_start, self.fog_end = 90.0, 300.0
        else:
            self.sky_top, self.sky_bot = SKY_TOP, SKY_BOT
            self.fog_start, self.fog_end = 160.0, 520.0
        glClearColor(*self.sky_bot, 1.0)
        glFogfv(GL_FOG_COLOR, (*self.sky_bot, 1.0))
        glFogf(GL_FOG_START, self.fog_start)
        glFogf(GL_FOG_END, self.fog_end)
        self.set_terrain(self.terrain)     # перегенерировать меш с новой палитрой

    # ------------------------------------------------------------------
    def set_terrain(self, terrain):
        """Загрузить меш ландшафта (по возможности в VBO — так быстрее)."""
        self.terrain = terrain
        self.tv, self.tc, self.ti = terrain.build_mesh()
        self.n_index = len(self.ti)
        try:
            if rawVertexPointer is None:
                raise RuntimeError("нет сырых вызовов — рисуем клиентскими массивами")
            if self.vbo is None:
                self.vbo = glGenBuffers(3)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo[0])
            glBufferData(GL_ARRAY_BUFFER, self.tv.nbytes, self.tv, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo[1])
            glBufferData(GL_ARRAY_BUFFER, self.tc.nbytes, self.tc, GL_STATIC_DRAW)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo[2])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.ti.nbytes, self.ti, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        except Exception:                     # нет VBO — рисуем клиентскими массивами
            self.vbo = None

    # ------------------------------------------------------------------
    def resize(self, w, h):
        self.w, self.h = max(w, 320), max(h, 240)
        glViewport(0, 0, self.w, self.h)

    def begin_3d(self, eye, target, up=(0, 1, 0), fov=70.0):
        """Матрицы считаем сами — библиотека GLU есть не в каждой системе."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._draw_sky()
        near, far = 0.1, 900.0
        aspect = self.w / max(self.h, 1)
        fh = 1.0 / np.tan(np.radians(fov) * 0.5)
        proj = np.array([
            [fh / aspect, 0, 0, 0],
            [0, fh, 0, 0],
            [0, 0, (far + near) / (near - far), -1],
            [0, 0, (2 * far * near) / (near - far), 0]], dtype=np.float64)
        glMatrixMode(GL_PROJECTION)
        glLoadMatrixd(proj)

        e = np.asarray(eye, dtype=np.float64)
        f = np.asarray(target, dtype=np.float64) - e
        f /= max(np.linalg.norm(f), 1e-9)
        upv = np.asarray(up, dtype=np.float64)
        s_ = np.cross(f, upv)
        nl = np.linalg.norm(s_)
        if nl < 1e-6:                       # взгляд вдоль «вверх» — берём запасную ось
            s_ = np.cross(f, np.array([0.0, 0.0, 1.0]))
            nl = max(np.linalg.norm(s_), 1e-9)
        s_ /= nl
        u = np.cross(s_, f)
        view = np.array([
            [s_[0], u[0], -f[0], 0.0],
            [s_[1], u[1], -f[1], 0.0],
            [s_[2], u[2], -f[2], 0.0],
            [-s_ @ e, -u @ e, f @ e, 1.0]], dtype=np.float64)
        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixd(view)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_FOG)

    def _draw_sky(self):
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_FOG)
        glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, 1, 0, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glBegin(GL_QUADS)
        glColor3f(*SKY_BOT); glVertex2f(0, 0); glVertex2f(1, 0)
        glColor3f(*SKY_TOP); glVertex2f(1, 1); glVertex2f(0, 1)
        glEnd()
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

    # ------------------------------------------------------------------
    def draw_terrain(self):
        if self.vbo is not None:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo[0])
            rawVertexPointer(3, GL_FLOAT, 0, _NULL)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo[1])
            rawColorPointer(3, GL_FLOAT, 0, _NULL)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo[2])
            rawDrawElements(GL_TRIANGLES, self.n_index, GL_UNSIGNED_INT, _NULL)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        else:
            glVertexPointerf(self.tv)
            glColorPointerf(self.tc)
            glDrawElements(GL_TRIANGLES, self.n_index, GL_UNSIGNED_INT, self.ti)

    def draw_boxes(self):
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        hits = getattr(self.terrain, "box_hit", None)
        for i, b in enumerate(self.terrain.boxes):
            flash = float(hits[i]) if hits is not None else 0.0
            col = (0.70 + flash, 0.42 + flash * 0.6, 0.20 + flash * 0.3)
            self._box(b[:3], b[3:], col)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

    @staticmethod
    def _box(c, hs, col):
        x, y, z = c
        a, b, d = hs
        v = [(x - a, y - b, z - d), (x + a, y - b, z - d), (x + a, y + b, z - d), (x - a, y + b, z - d),
             (x - a, y - b, z + d), (x + a, y - b, z + d), (x + a, y + b, z + d), (x - a, y + b, z + d)]
        faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
        shade = [0.60, 1.0, 0.75, 0.86, 0.68, 0.68]
        glBegin(GL_QUADS)
        for fi, fq in enumerate(faces):
            s = shade[fi]
            glColor3f(col[0] * s, col[1] * s, col[2] * s)
            for i in fq:
                glVertex3f(*v[i])
        glEnd()

    # ------------------------------------------------------------------
    # тени и свет
    @staticmethod
    def _blob(x, y, z, rx, rz, rot=0.0, alpha=0.4, col=(0.02, 0.02, 0.03), n=24):
        """Мягкое пятно на рельефе: fan с градиентом альфы от центра к краю."""
        ca, sa = np.cos(rot), np.sin(rot)
        glBegin(GL_TRIANGLE_FAN)
        glColor4f(col[0], col[1], col[2], alpha)
        glVertex3f(x, y, z)
        for i in range(n + 1):
            t = 2.0 * np.pi * i / n
            px, pz = np.cos(t) * rx, np.sin(t) * rz
            glColor4f(col[0], col[1], col[2], 0.0)
            glVertex3f(x + px * ca - pz * sa, y, z + px * sa + pz * ca)
        glEnd()

    def draw_shadows(self, veh, terrain):
        """Blob-тени от машины и ящиков на рельефе, со сдвигом от солнца."""
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        c, fwd, rgt, up = veh.frame()
        sun_h = np.array([SUN[0], 0.0, SUN[2]])
        sun_h /= max(np.linalg.norm(sun_h), 1e-9)
        shift = -sun_h * 0.45
        heading = float(np.arctan2(fwd[0], fwd[2]))
        gy = float(terrain.height(c[0], c[2])) + 0.02
        self._blob(c[0] + shift[0] * 0.5, gy, c[2] + shift[2] * 0.5,
                   rx=2.4, rz=1.5, rot=heading, alpha=0.40)
        for b in terrain.boxes:
            by = float(terrain.height(b[0], b[2])) + 0.02
            self._blob(b[0] + shift[0] * b[1] * 0.6, by,
                       b[2] + shift[2] * b[1] * 0.6,
                       rx=b[3] * 1.15 + 0.15, rz=b[5] * 1.15 + 0.15,
                       rot=0.0, alpha=0.32)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

    @staticmethod
    def _cone(p, d, length, radius, col, alpha=0.13, n=20):
        """Конус света (fan): яркий у вершины, альфа тает к краю."""
        d = np.asarray(d, dtype=np.float64)
        d /= max(np.linalg.norm(d), 1e-9)
        a = np.cross(d, (0.0, 1.0, 0.0))
        if np.linalg.norm(a) < 1e-6:
            a = np.array([1.0, 0.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(d, a)
        glBegin(GL_TRIANGLE_FAN)
        glColor4f(col[0], col[1], col[2], min(alpha * 2.2, 0.5))
        glVertex3f(*p)
        for i in range(n + 1):
            t = 2.0 * np.pi * i / n
            q = p + d * length + (a * np.cos(t) + b * np.sin(t)) * radius
            glColor4f(col[0], col[1], col[2], 0.0)
            glVertex3f(*q)
        glEnd()

    def draw_headlights(self, veh):
        """Фары ночью: конусы света + тёплые пятна на земле впереди машины."""
        if not self.night:
            return
        c, fwd, rgt, up = veh.frame()
        heading = float(np.arctan2(fwd[0], fwd[2]))
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        for s in (-0.45, 0.45):
            p = c + fwd * 1.85 + rgt * s + up * 0.72
            self._cone(p, fwd, length=30.0, radius=4.6, col=(1.0, 0.93, 0.72))
            gx = c[0] + fwd[0] * 7.5 + rgt[0] * s * 1.4
            gz = c[2] + fwd[2] * 7.5 + rgt[2] * s * 1.4
            gy = float(self.terrain.height(gx, gz)) + 0.03
            self._blob(gx, gy, gz, rx=3.6, rz=2.0, rot=heading, alpha=0.20,
                       col=(1.0, 0.90, 0.62))
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

    def draw_smoke(self, smoke, eye, target):
        """Дым от шин: полупрозрачные билборды, затухают со временем жизни."""
        pos, size, fade = smoke.active()
        if len(pos) == 0:
            return
        f = np.asarray(target, float) - np.asarray(eye, float)
        f /= max(np.linalg.norm(f), 1e-9)
        r = np.cross(f, (0.0, 1.0, 0.0))
        r /= max(np.linalg.norm(r), 1e-9)
        u = np.cross(r, f)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glBegin(GL_QUADS)
        for p, sz, a in zip(pos, size, fade):
            glColor4f(0.50, 0.51, 0.54, 0.5 * a)
            glVertex3f(*(p - r * sz - u * sz))
            glVertex3f(*(p + r * sz - u * sz))
            glVertex3f(*(p + r * sz + u * sz))
            glVertex3f(*(p - r * sz + u * sz))
        glEnd()
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

    # ------------------------------------------------------------------
    def draw_vehicle(self, veh, show_beams=False, show_panels=True):
        sb = veh.sb
        p = sb.pos
        c, fwd, rgt, up = veh.frame()

        if show_panels:
            # локальные оси кузова: x — вправо по модели, y — вверх, z — вперёд
            R = np.column_stack([-rgt, up, fwd])
            verts = veh.skin.deform(p, R)
            fv, fc = veh.skin.facets(verts, self._node_damage(veh), SUN)
            if rawVertexPointer is not None:
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
                rawVertexPointer(3, GL_FLOAT, 0, fv.ctypes.data_as(ctypes.c_void_p))
                rawColorPointer(3, GL_FLOAT, 0, fc.ctypes.data_as(ctypes.c_void_p))
            else:
                glVertexPointerf(fv)
                glColorPointerf(fc)
            glDrawArrays(GL_TRIANGLES, 0, veh.skin.nf * 3)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        if show_beams:
            stress = sb.beam_stress()
            glLineWidth(2.0)
            glBegin(GL_LINES)
            for bi, (a, b) in enumerate(sb.beams):
                if not sb.alive[bi]:
                    continue
                st = float(stress[bi])
                glColor3f(0.15 + 0.85 * st, 0.95 - 0.75 * st, 0.25 * (1 - st))
                glVertex3f(*p[a]); glVertex3f(*p[b])
            glEnd()

        for w in veh.wheels:
            self._wheel(p[w.node], w, fwd, rgt, up)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

    @staticmethod
    def _node_damage(veh):
        sb = veh.sb
        plastic = np.abs(sb.rest - sb.rest0) / sb.rest0
        plastic = np.where(sb.alive, plastic, 0.4)
        d = np.zeros(len(sb.mass))
        cnt = np.zeros(len(sb.mass)) + 1e-9
        np.add.at(d, sb.beams[:, 0], plastic)
        np.add.at(d, sb.beams[:, 1], plastic)
        np.add.at(cnt, sb.beams[:, 0], 1.0)
        np.add.at(cnt, sb.beams[:, 1], 1.0)
        return np.clip(d / cnt / 0.10, 0.0, 1.0)

    @staticmethod
    def _patch(a, b, c, d, u0, u1, v0, v1):
        """Кусок четырёхугольника в билинейных координатах (фары/фонари)."""
        def P(u, v):
            return (a * (1 - u) * (1 - v) + b * u * (1 - v)
                    + c * u * v + d * (1 - u) * v)
        return P(u0, v0), P(u1, v0), P(u1, v1), P(u0, v1)

    @staticmethod
    def _quad(a, b, c, d, col, dmg=0.0, flat=False):
        if flat:
            lam = 1.0
        else:
            n = np.cross(b - a, c - a)
            nl = np.linalg.norm(n)
            n = n / nl if nl > 1e-9 else np.array([0.0, 1.0, 0.0])
            lam = 0.34 + 0.66 * abs(float(n @ SUN))
        col = col * (1.0 - 0.55 * dmg) + np.array([0.25, 0.22, 0.20]) * dmg
        glColor3f(*np.clip(col * lam, 0, 1))
        for v in (a, b, c, d):
            glVertex3f(*v)

    def _wheel(self, c, w, fwd, rgt, up, wheel_r, wheel_w):
        ang = w.steer if w.steered else 0.0
        f = fwd * np.cos(ang) - rgt * np.sin(ang)
        axis = np.cross(f, up)
        axis /= max(np.linalg.norm(axis), 1e-9)
        u = np.cross(axis, f)
        u /= max(np.linalg.norm(u), 1e-9)
        ca, sa = np.cos(w.angle), np.sin(w.angle)

        # протектор
        glBegin(GL_QUAD_STRIP)
        for (cx, cy) in _CIRC:
            r = f * cx + u * cy
            sh = 0.11 + 0.05 * (0.5 + 0.5 * (cx * ca + cy * sa))
            glColor3f(sh, sh, sh * 1.08)
            glVertex3f(*(c + r * wheel_r - axis * wheel_w))
            glVertex3f(*(c + r * wheel_r + axis * wheel_w))
        glEnd()

        for side in (-wheel_w, wheel_w):
            cc = c + axis * side
            # боковина покрышки
            glBegin(GL_QUAD_STRIP)
            glColor3f(0.10, 0.10, 0.11)
            for (cx, cy) in _CIRC:
                r = f * cx + u * cy
                glVertex3f(*(cc + r * wheel_r))
                glVertex3f(*(cc + r * wheel_r * 0.62))
            glEnd()
            # диск со спицами: чередуем светлый металл и тёмные окна
            glBegin(GL_TRIANGLE_FAN)
            glColor3f(0.62, 0.63, 0.67)
            glVertex3f(*(cc + axis * side * 0.02))
            for i, (cx, cy) in enumerate(_CIRC):
                rr = f * cx + u * cy
                spoke = 0.5 + 0.5 * np.cos(5.0 * (np.arctan2(cy, cx) + w.angle))
                shade = 0.22 + 0.42 * spoke
                glColor3f(shade, shade, shade * 1.05)
                glVertex3f(*(cc + rr * wheel_r * 0.60))
            glEnd()

    def draw_pylon(self, x, y, z, col):
        """Столб-ворота чекпоинта: колонна + яркий куб-маркер наверху."""
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        self._box(np.array([x, y + 1.5, z]), np.array([0.16, 1.5, 0.16]), col)
        top = np.array([min(c * 1.35 + 0.15, 1.0) for c in col])
        self._box(np.array([x, y + 3.05, z]), np.array([0.13, 0.13, 0.13]), top)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

    def draw_particles(self, fx, eye, target):
        """Обломки: билборды-квадраты, развёрнутые к камере."""
        pos, col, size, fade = fx.active()
        if len(pos) == 0:
            return
        f = np.asarray(target, float) - np.asarray(eye, float)
        f /= max(np.linalg.norm(f), 1e-9)
        r = np.cross(f, (0.0, 1.0, 0.0))
        r /= max(np.linalg.norm(r), 1e-9)
        u = np.cross(r, f)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glBegin(GL_QUADS)
        for p, c, sz, a in zip(pos, col, size, fade):
            k = 0.35 + 0.65 * a
            sz = sz * (0.45 + 0.55 * a)      # обломки «истаивают»
            glColor3f(c[0] * k, c[1] * k, c[2] * k)
            glVertex3f(*(p - r * sz - u * sz))
            glVertex3f(*(p + r * sz - u * sz))
            glVertex3f(*(p + r * sz + u * sz))
            glVertex3f(*(p - r * sz + u * sz))
        glEnd()

    # ------------------------------------------------------------------
    # 2D-слой: рисуем готовые pygame-поверхности
    def begin_2d(self):
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_FOG)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        glOrtho(0, self.w, self.h, 0, -1, 1)      # y вниз, как в pygame
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()

    def blit(self, surf, x, y):
        """Нарисовать pygame-поверхность в экранных координатах (левый верх)."""
        w, h = surf.get_size()
        data = pygame.image.tostring(surf, "RGBA", False)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(x, y)
        glTexCoord2f(1, 0); glVertex2f(x + w, y)
        glTexCoord2f(1, 1); glVertex2f(x + w, y + h)
        glTexCoord2f(0, 1); glVertex2f(x, y + h)
        glEnd()
        glDisable(GL_TEXTURE_2D)

    def flash(self, alpha, col=(0.9, 0.25, 0.15)):
        """Вспышка на весь экран в момент сильного удара."""
        if alpha <= 0.01:
            return
        glBegin(GL_QUADS)
        glColor4f(col[0], col[1], col[2], min(alpha, 0.55))
        glVertex2f(0, 0); glVertex2f(self.w, 0)
        glVertex2f(self.w, self.h); glVertex2f(0, self.h)
        glEnd()

    def dim(self, alpha=0.55):
        glBegin(GL_QUADS)
        glColor4f(0.03, 0.04, 0.06, alpha)
        glVertex2f(0, 0); glVertex2f(self.w, 0)
        glVertex2f(self.w, self.h); glVertex2f(0, self.h)
        glEnd()
