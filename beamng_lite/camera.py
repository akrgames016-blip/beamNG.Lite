"""Камеры: погоня (с вращением мышью), орбита, капот, свободная."""
import numpy as np


class Camera:
    MODES = ("chase", "orbit", "hood", "free")
    NAMES = {"chase": "погоня", "orbit": "орбита", "hood": "капот", "free": "свободная"}

    def __init__(self):
        self.mode = 0
        self.yaw = 0.0            # в погоне — смещение относительно машины
        self.pitch = 0.17
        self.dist = 6.6
        self.smooth = None
        self.free_pos = np.array([0.0, 14.0, -24.0])
        self.free_yaw = 0.0
        self.free_pitch = 0.35
        self.shake = 0.0          # амплитуда тряски от ударов

    def kick(self, amount):
        """Тряхнуть камеру после удара (amount — скачок скорости, м/с)."""
        self.shake = float(min(1.2, max(self.shake, amount * 0.045)))

    def _shake_off(self, dt):
        if self.shake <= 1e-3:
            return np.zeros(3)
        off = np.random.normal(0.0, self.shake * 0.42, 3)
        self.shake *= float(np.exp(-dt * 7.0))
        return off

    def cycle(self, delta=1):
        self.mode = (self.mode + delta) % len(self.MODES)
        self.smooth = None

    def name(self):
        return self.NAMES[self.MODES[self.mode]]

    def rotate(self, dx, dy):
        if self.MODES[self.mode] == "free":
            self.free_yaw -= dx * 0.005
            self.free_pitch = float(np.clip(self.free_pitch + dy * 0.004, -1.4, 1.4))
        else:
            self.yaw -= dx * 0.006
            self.pitch = float(np.clip(self.pitch + dy * 0.005, -0.35, 1.30))

    def zoom(self, wheel):
        self.dist = float(np.clip(self.dist * (1.0 - wheel * 0.12), 2.0, 60.0))

    def recenter(self):
        self.yaw = 0.0
        self.pitch = 0.17

    def update(self, veh, dt, terrain, free_keys=(0, 0, 0, 0, 0, 0)):
        c, fwd, rgt, up = veh.frame()
        mode = self.MODES[self.mode]

        if mode == "hood":
            eye = c + up * 0.62 + fwd * 0.45 + self._shake_off(dt)
            return eye, eye + fwd * 10.0 - up * 0.9, tuple(up)

        if mode == "free":
            f, b, l, r, u, d = free_keys
            sp = 30.0 * dt
            dirv = np.array([np.sin(self.free_yaw) * np.cos(self.free_pitch),
                             -np.sin(self.free_pitch),
                             np.cos(self.free_yaw) * np.cos(self.free_pitch)])
            right = np.array([np.cos(self.free_yaw), 0.0, -np.sin(self.free_yaw)])
            self.free_pos = self.free_pos + dirv * (f - b) * sp + right * (r - l) * sp
            self.free_pos[1] += (u - d) * sp
            return self.free_pos, self.free_pos + dirv, (0, 1, 0)

        # --- погоня и орбита: сферические координаты вокруг машины ---
        if mode == "chase":
            base = np.arctan2(fwd[0], fwd[2])     # курс машины
            yaw = base + self.yaw
        else:
            yaw = self.yaw
        cp = np.cos(self.pitch)
        off = np.array([-np.sin(yaw) * cp, np.sin(self.pitch) + 0.22, -np.cos(yaw) * cp])
        off /= np.linalg.norm(off)
        want = c + off * self.dist

        if self.smooth is None:
            self.smooth = want.copy()
        k = 1.0 - np.exp(-dt * (14.0 if mode == "orbit" else 8.0))
        self.smooth += (want - self.smooth) * k
        eye = self.smooth.copy()
        eye = eye + self._shake_off(dt)
        gh = float(terrain.height(eye[0], eye[2])) + 0.8
        if eye[1] < gh:
            eye[1] = gh
        return eye, c + np.array([0.0, 0.7, 0.0]), (0, 1, 0)
