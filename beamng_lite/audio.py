"""Звук: процедурно сгенерированные сэмплы через numpy + pygame.mixer.

Двигатель — три зацикленных слоя (холостые / средние / высокие обороты),
громкость слоёв зависит от оборотов и газа. Удар — низкий «бум» + шум,
громкость по силе. Шины и кузов — зацикленный шум с bandpass-фильтром.
В headless-режиме (или без аудиоустройства) Audio — no-op.
"""
import numpy as np

SR = 22050


def _fade_loop(s, fade_s=0.01):
    """Бесшовный зацикленный сэмпл: короткие fade-in/fade-out на концах."""
    n = int(SR * fade_s)
    if len(s) > 2 * n:
        s[:n] *= np.linspace(0.0, 1.0, n)
        s[-n:] *= np.linspace(1.0, 0.0, n)
    return s


def _to_int16(s):
    s = s / max(np.abs(s).max(), 1e-9)
    return (s * 32767 * 0.9).astype(np.int16)


def _engine_layer(base_hz, rough=0.35):
    """Звук выхлопа: сумма гармоник основного тона + шумовые «вспышки»."""
    dur = 0.5
    n = int(SR * dur)
    periods = max(1, int(round(base_hz * dur)))
    f0 = periods / dur                      # целое число периодов — бесшовно
    t = np.arange(n) / SR
    s = np.zeros(n)
    for h, amp in ((1, 1.0), (2, 0.55), (3, 0.32), (4, 0.18), (5, 0.10), (6, 0.06)):
        s += amp * np.sin(2 * np.pi * f0 * h * t)
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 1.0, n)
    k = 30
    noise = np.convolve(noise, np.ones(k) / k, mode="same")
    # короткие «вспышки» на каждом периоде
    env = np.zeros(n)
    for p in range(periods):
        i0 = int(p * n / periods)
        i1 = min(n, i0 + int(SR * 0.006))
        env[i0:i1] += np.exp(-np.arange(i1 - i0) / (SR * 0.002))
    s = s * 0.75 + noise * env * rough
    return _fade_loop(s)


def _hit_sound(strength):
    """Удар: низкий синус с затуханием + отфильтрованный шум."""
    dur = float(np.clip(0.18 + strength * 0.09, 0.18, 1.1))
    n = int(SR * dur)
    t = np.arange(n) / SR
    f = max(55.0, 110.0 - strength * 6.0)
    body = np.sin(2 * np.pi * f * t * np.exp(-t * 1.5)) * np.exp(-t * 7.0)
    rng = np.random.default_rng()
    noise = rng.normal(0.0, 1.0, n)
    noise = np.convolve(noise, np.ones(40) / 40, mode="same")
    noise *= np.exp(-t * 10.0)
    s = body * 1.2 + noise * 0.7
    s *= 0.35 + min(strength, 12.0) / 12.0 * 0.65
    return _to_int16(s)


def _screech_loop(low_k=8, high_k=60):
    """Шум с полосовым фильтром — визг шин / скрежет металла."""
    n = int(SR * 0.5)
    rng = np.random.default_rng(11)
    noise = rng.normal(0.0, 1.0, n)

    def ma(x, k):
        return np.convolve(x, np.ones(k) / k, mode="same")

    s = ma(noise, low_k) - ma(noise, high_k)
    return _fade_loop(s)


class Audio:
    def __init__(self, enabled=True, master=0.6):
        self.ok = False
        self.master = master
        if not enabled:
            return
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init(SR, -16, 1, 256)
            pygame.mixer.set_num_channels(8)
        except Exception:
            return                              # нет аудиоустройства — тихо
        self.ok = True
        pg = pygame
        self._pg = pg
        # двигатель: три слоя по оборотам
        self.snd = {
            "idle": pg.mixer.Sound(buffer=_to_int16(_engine_layer(38.0)).tobytes()),
            "mid": pg.mixer.Sound(buffer=_to_int16(_engine_layer(85.0, 0.45)).tobytes()),
            "high": pg.mixer.Sound(buffer=_to_int16(_engine_layer(160.0, 0.55)).tobytes()),
            "tire": pg.mixer.Sound(buffer=_to_int16(_screech_loop()).tobytes()),
            "scrape": pg.mixer.Sound(buffer=_to_int16(_screech_loop(4, 30)).tobytes()),
        }
        self.ch = {name: pg.mixer.Channel(i) for i, name in enumerate(
            ("idle", "mid", "high", "tire", "scrape"))}
        for name, ch in self.ch.items():
            ch.play(self.snd[name], loops=-1)
            ch.set_volume(0.0)
        self._hit_pool = [pg.mixer.Channel(5), pg.mixer.Channel(6), pg.mixer.Channel(7)]
        self._hit_i = 0

    # ------------------------------------------------------------------
    def _vol(self, ch, v):
        if self.ok:
            ch.set_volume(float(np.clip(v, 0.0, 1.0)) * self.master)

    def update(self, veh, dt):
        """Громкость каналов по состоянию машины (обороты, газ, слип, скрежет)."""
        if not self.ok:
            return
        max_rpm = float(veh.cfg.get("max_rpm", 7000.0))
        x = (veh.rpm - veh.idle_rpm) / max(1.0, max_rpm - veh.idle_rpm)
        load = veh.throttle
        self._vol(self.ch["idle"], max(0.0, 1.0 - abs(x - 0.12) * 9.0) * (0.30 + 0.45 * load))
        self._vol(self.ch["mid"], max(0.0, 1.0 - abs(x - 0.50) * 4.5) * (0.18 + 0.55 * load))
        self._vol(self.ch["high"], max(0.0, (x - 0.68) / 0.32) * (0.25 + 0.65 * load))
        slip = max((abs(w.slip) for w in veh.wheels if w.contact), default=0.0)
        self._vol(self.ch["tire"], min(1.0, max(0.0, (slip - 0.22) / 0.55)) * 0.55)
        self._vol(self.ch["scrape"], min(1.0, getattr(veh, "scrape", 0.0) / 9.0) * 0.45)

    def on_impact(self, strength):
        """One-shot звук удара (вызывается из эффектов столкновения)."""
        if not self.ok or strength < 2.6:
            return
        snd = self._pg.mixer.Sound(buffer=_hit_sound(strength).tobytes())
        ch = self._hit_pool[self._hit_i]
        self._hit_i = (self._hit_i + 1) % len(self._hit_pool)
        ch.play(snd)
        ch.set_volume(self.master)

    def off(self):
        if self.ok:
            try:
                self._pg.mixer.stop()
            except Exception:
                pass
