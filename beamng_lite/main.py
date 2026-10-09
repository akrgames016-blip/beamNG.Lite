"""BeamNG.Drive Lite — мини-симулятор soft-body автомобиля.

Запуск:  python run.py            |  python -m beamng_lite
         python run.py --headless 300   — тест физики без окна
         python run.py --vehicle pickup — другая машина из гаража
"""
import sys
import time
import argparse

import numpy as np

from . import garage
from . import modes
from .terrain import Terrain
from .vehicle import Vehicle


# ----------------------------------------------------------------------
def headless(steps=300, seed=7, vehicle_id=garage.DEFAULT_ID):
    """Прогон физики без графики — проверка устойчивости и скорости счёта."""
    t = Terrain(seed=seed)
    v = Vehicle(t, spawn=(0.0, -6.0), cfg=garage.load(vehicle_id))
    dt = 1.0 / 60.0
    t0 = time.time()
    for i in range(steps):
        v.controls(1.0 if i > 30 else 0.0, 0.0, 0.35 * np.sin(i * 0.02), 0.0, dt)
        v.step(dt)
        if not np.all(np.isfinite(v.sb.pos)):
            print("NaN на шаге", i)
            return 1
    c, fwd, rgt, up = v.frame()
    el = time.time() - t0
    print("машина %s (%s)" % (v.cfg["name"], vehicle_id))
    print("шагов %d за %.2f с (%.0f шаг/с, x%.1f реального времени)"
          % (steps, el, steps / el, steps * dt / el))
    print("позиция %s  скорость %.1f км/ч  передача %d  %.0f об/мин"
          % (c.round(2), v.speed_kmh(), v.gear - 1, v.rpm))
    print("повреждения %.1f%%  оборвано балок %d/%d  узлов %d  up.y=%.2f"
          % (v.damage() * 100, int((~v.sb.alive).sum()), len(v.sb.alive),
             len(v.sb.mass), up[1]))
    return 0


# ----------------------------------------------------------------------
class Game:
    SPAWN = (0.0, -6.0)

    def __init__(self, width, height, seed, vsync, vehicle_id=garage.DEFAULT_ID):
        import pygame
        from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE
        from .render import Renderer
        from .camera import Camera
        from .effects import Particles, Smoke
        from .audio import Audio
        from . import ui

        self.pygame = pygame
        self.ui = ui
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        try:
            pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL | RESIZABLE,
                                    vsync=1 if vsync else 0)
        except pygame.error:                      # нет MSAA — пробуем без него
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 0)
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 0)
            pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL | RESIZABLE)
        pygame.display.set_caption("BeamNG.Drive Lite")

        self.w, self.h = width, height
        self.seed = seed
        self.invert_steer = False
        self.vehicle_id = vehicle_id
        self.terrain = Terrain(seed=seed)
        self.vehicle = self._make_vehicle(vehicle_id)
        self.renderer = Renderer(width, height, self.terrain)
        self.camera = Camera()
        self.camera.dist = float(self.vehicle.cfg.get("camera_dist", 6.6))
        self.clock = pygame.time.Clock()

        self.state = "menu"
        self.show_beams = False
        self.show_panels = True
        self.paused = False
        self.slowmo = False
        self.toast = ""
        self.toast_t = 0.0
        self.fx = Particles()
        self.smoke = Smoke()
        self.audio = Audio()
        self.night = False
        self.flash = 0.0
        self.reverse_hold = 0.0
        self.acc = 0.0
        self.hud_surf = None
        self.hud_timer = 0.0
        self.hint_surf = None
        self.menu = self._main_menu()
        self.help_return = "menu"
        self.garage_items = []
        self.garage_sel = 0
        self.garage_rects = []
        self.mode_id = "free"
        self.mode = modes.make(self.mode_id)
        self.mode_line = ""

    def _make_vehicle(self, vehicle_id):
        return Vehicle(self.terrain, spawn=self.SPAWN, cfg=garage.load(vehicle_id))

    # ---------------- меню ----------------
    def _main_menu(self):
        return self.ui.Menu(
            "BeamNG.Drive Lite",
            [("play", "Играть"),
             ("garage", "Гараж", garage.load(self.vehicle_id)["name"]),
             ("mode", "Режим", modes.NAMES[self.mode_id]),
             ("seed", "Ландшафт (сид)", self.seed),
             ("time", "Время суток", "день"),
             ("invert", "Инверсия руля", "выкл"),
             ("help", "Управление"),
             ("quit", "Выход")],
            subtitle="soft-body физика: узлы + балки, деформация и разрывы")

    def _pause_menu(self):
        return self.ui.Menu("Пауза", [
            ("resume", "Продолжить"),
            ("respawn", "Респавн"),
            ("help", "Управление"),
            ("main", "В главное меню"),
            ("quit", "Выход")])

    def _start_game(self):
        """Выйти на трассу: спавн и сброс состояния текущего режима."""
        spawn, heading = self.mode.spawn()
        self.vehicle.reset(spawn=spawn, heading=heading)
        self.mode.start(self)
        self.camera.recenter()
        self.camera.smooth = None
        self.state = "game"
        self.pygame.mouse.get_rel()

    def _cycle_mode(self, d):
        i = modes.MODES.index(self.mode_id)
        self.mode_id = modes.MODES[(i + d) % len(modes.MODES)]
        self.mode = modes.make(self.mode_id)
        self.menu.items[2] = ("mode", "Режим", modes.NAMES[self.mode_id])

    def _cycle_time(self):
        self.night = not self.night
        self.renderer.set_night(self.night)
        self.menu.items[4] = ("time", "Время суток",
                              "ночь" if self.night else "день")
        self.say("Ночь — фары включены" if self.night else "День")

    def _menu_action(self, action):
        if action == "play":
            self._start_game()
        elif action == "resume":
            self.state = "game"
            self.pygame.mouse.get_rel()
        elif action == "help":
            self.help_return = self.state
            self.state = "help"
        elif action == "respawn":
            self._start_game()
        elif action == "main":
            self.state = "menu"
            self.menu = self._main_menu()
        elif action == "seed":
            self._change_seed(+1)
        elif action == "invert":
            self.invert_steer = not self.invert_steer
            self.menu.items[5] = ("invert", "Инверсия руля",
                                  "вкл" if self.invert_steer else "выкл")
        elif action == "mode":
            self._cycle_mode(+1)
        elif action == "time":
            self._cycle_time()
        elif action == "garage":
            allv = garage.load_all()
            self.garage_items = list(allv.items())
            self.garage_sel = max(0, [i for i, (vid, _) in enumerate(self.garage_items)
                                      if vid == self.vehicle_id][0:1] or [0])
            self.state = "garage"
        elif action == "quit":
            return False
        return True

    def _change_seed(self, d):
        self.seed = max(0, self.seed + d)
        self.terrain = Terrain(seed=self.seed)
        self.vehicle = self._make_vehicle(self.vehicle_id)
        self.renderer.set_terrain(self.terrain)
        self.camera.smooth = None
        self.camera.dist = float(self.vehicle.cfg.get("camera_dist", 6.6))
        self.mode = modes.make(self.mode_id)
        self.menu.items[3] = ("seed", "Ландшафт (сид)", self.seed)

    def _select_vehicle(self, idx):
        vid, cfg = self.garage_items[idx]
        if vid != self.vehicle_id:
            self.vehicle_id = vid
            self.vehicle = self._make_vehicle(vid)
            self.camera.dist = float(cfg.get("camera_dist", 6.6))
            self.camera.smooth = None
            self.mode = modes.make(self.mode_id)   # счёт режима — под новую машину
            self.say("Выехал: " + cfg["name"])
        self.state = "menu"
        self.menu = self._main_menu()

    def say(self, text, t=1.3):
        self.toast, self.toast_t = text, t

    # ---------------- главный цикл ----------------
    def run(self):
        pygame = self.pygame
        running = True
        while running:
            frame_dt = self.clock.tick(120) / 1000.0
            wheel = 0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.VIDEORESIZE:
                    self.w, self.h = e.w, e.h
                    self.renderer.resize(e.w, e.h)
                    self.hud_surf = self.hint_surf = None
                elif e.type == pygame.MOUSEWHEEL:
                    wheel = e.y
                elif e.type == pygame.MOUSEMOTION and self.state in ("menu", "pause"):
                    self.menu.hover(*e.pos)
                elif e.type == pygame.MOUSEMOTION and self.state == "garage":
                    for i, r in enumerate(self.garage_rects):
                        if r.collidepoint(*e.pos):
                            self.garage_sel = i
                            break
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    if self.state in ("menu", "pause"):
                        act = self.menu.click(*e.pos)
                        if act and not self._menu_action(act):
                            running = False
                    elif self.state == "garage":
                        for i, r in enumerate(self.garage_rects):
                            if r.collidepoint(*e.pos):
                                self.garage_sel = i
                                self._select_vehicle(i)
                                break
                    elif self.state == "help":
                        self.state = self.help_return
                elif e.type == pygame.KEYDOWN:
                    running = self._on_key(e)
            if not running:
                break

            if self.state == "game":
                self._update_game(frame_dt, wheel)
            else:
                pygame.mouse.get_rel()

            self._draw(frame_dt)
            pygame.display.flip()

        self.audio.off()
        pygame.quit()
        return 0

    def _on_key(self, e):
        pygame = self.pygame
        k = e.key
        if self.state in ("menu", "pause"):
            if k in (pygame.K_DOWN, pygame.K_s):
                self.menu.move(1)
            elif k in (pygame.K_UP, pygame.K_w):
                self.menu.move(-1)
            elif k in (pygame.K_LEFT, pygame.K_a) and self.menu.current() == "seed":
                self._change_seed(-1)
            elif k in (pygame.K_RIGHT, pygame.K_d) and self.menu.current() == "seed":
                self._change_seed(+1)
            elif k in (pygame.K_LEFT, pygame.K_a) and self.menu.current() == "mode":
                self._cycle_mode(-1)
            elif k in (pygame.K_RIGHT, pygame.K_d) and self.menu.current() == "mode":
                self._cycle_mode(+1)
            elif k in (pygame.K_LEFT, pygame.K_a, pygame.K_RIGHT, pygame.K_d) \
                    and self.menu.current() == "time":
                self._cycle_time()
            elif k in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                return self._menu_action(self.menu.current())
            elif k == pygame.K_ESCAPE:
                if self.state == "pause":
                    self.state = "game"
                    pygame.mouse.get_rel()
                else:
                    return False
            return True

        if self.state == "help":
            if k in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_SPACE):
                self.state = self.help_return
            return True

        if self.state == "garage":
            if k in (pygame.K_DOWN, pygame.K_s):
                self.garage_sel = (self.garage_sel + 1) % len(self.garage_items)
            elif k in (pygame.K_UP, pygame.K_w):
                self.garage_sel = (self.garage_sel - 1) % len(self.garage_items)
            elif k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._select_vehicle(self.garage_sel)
            elif k == pygame.K_ESCAPE:
                self.state = "menu"
                self.menu = self._main_menu()
            return True

        # --- в игре ---
        v = self.vehicle
        if k == pygame.K_ESCAPE:
            self.state = "pause"
            self.menu = self._pause_menu()
        elif k == pygame.K_r:
            self._start_game(); self.say("Респавн")
        elif k == pygame.K_f:
            v.recover(); self.say("Поставлено на колёса")
        elif k == pygame.K_c:
            self.camera.cycle(-1 if e.mod & pygame.KMOD_SHIFT else 1)
            self.say("Камера: " + self.camera.name())
        elif k == pygame.K_v:
            self.camera.recenter()
        elif k == pygame.K_b:
            self.show_beams = not self.show_beams
            self.say("Балки: " + ("видны" if self.show_beams else "скрыты"))
        elif k == pygame.K_n:
            self.show_panels = not self.show_panels
        elif k == pygame.K_p:
            self.paused = not self.paused
        elif k == pygame.K_t:
            self.slowmo = not self.slowmo
            self.say("Slow-mo " + ("вкл" if self.slowmo else "выкл"))
        elif k == pygame.K_x:
            v.tc = not v.tc
            self.say("Антипробуксовка " + ("вкл" if v.tc else "выкл"))
        elif k == pygame.K_e:
            v.shift(+1); self.say("Передача выше")
        elif k == pygame.K_q:
            v.shift(-1); self.say("Передача ниже")
        elif k == pygame.K_g:
            v.auto = not v.auto
            self.say("КПП: " + ("автомат" if v.auto else "ручная"))
        return True

    # ---------------- игровая логика ----------------
    def _update_game(self, frame_dt, wheel):
        pygame = self.pygame
        v = self.vehicle
        keys = pygame.key.get_pressed()

        # камера мышью
        dx, dy = pygame.mouse.get_rel()
        buttons = pygame.mouse.get_pressed()
        if buttons[0] or buttons[2]:
            self.camera.rotate(dx, dy)
        if wheel:
            self.camera.zoom(wheel)

        thr = 1.0 if (keys[pygame.K_w] or keys[pygame.K_UP]) else 0.0
        brk = 1.0 if (keys[pygame.K_s] or keys[pygame.K_DOWN]) else 0.0
        steer = (1.0 if (keys[pygame.K_a] or keys[pygame.K_LEFT]) else 0.0) - \
                (1.0 if (keys[pygame.K_d] or keys[pygame.K_RIGHT]) else 0.0)
        if self.invert_steer:
            steer = -steer
        hb = 1.0 if keys[pygame.K_SPACE] else 0.0

        # задний ход: держать «тормоз» на месте ~0.35 с
        if v.auto:
            slow = v.speed_kmh() < 3.0
            if v.gear >= 2:
                self.reverse_hold = self.reverse_hold + frame_dt if (brk > 0 and slow) else 0.0
                if self.reverse_hold > 0.35:
                    v.gear = 0
                    self.reverse_hold = 0.0
                    self.say("Задний ход")
            elif v.gear == 0:
                if thr > 0 and slow:
                    v.gear = 2
                    self.say("Вперёд")
        if v.gear == 0:
            thr, brk = brk, thr

        # физика с фиксированным шагом
        step = 1.0 / 60.0
        scale = 0.25 if self.slowmo else 1.0
        self.acc = min(self.acc + frame_dt * scale, step * 3)
        v.controls(thr, brk, steer, hb, max(frame_dt, 1e-4))
        while self.acc >= step and not self.paused:
            v.step(step)
            self.acc -= step
        if self.paused:
            self.acc = 0.0

        self._impact_fx(v, frame_dt)
        self.audio.update(v, frame_dt)

        # дым из-под буксующих колёс
        spd = v.speed_kmh()
        for w in v.wheels:
            if w.contact and abs(w.slip) > 0.30 and spd > 4.0:
                self.smoke.emit(int(min(6, abs(w.slip) * 4)),
                                v.sb.pos[w.node] + np.array([0.0, 0.12, 0.0]))
        self.smoke.update(frame_dt, self.terrain)

        free = (keys[pygame.K_i], keys[pygame.K_k], keys[pygame.K_j],
                keys[pygame.K_l], keys[pygame.K_u], keys[pygame.K_o])
        self.eye, self.target, self.up = self.camera.update(v, frame_dt, self.terrain, free)

        self.mode_line = self.mode.update(v, self.terrain, frame_dt)

        self.toast_t = max(0.0, self.toast_t - frame_dt)
        self.hud_timer -= frame_dt
        if self.hud_timer <= 0 or self.hud_surf is None:
            self.hud_surf = self.ui.build_hud(v, self.camera.name(),
                                              self.clock.get_fps(), self.paused,
                                              self.slowmo, self.w, self.mode_line)
            self.hud_timer = 0.1

    # ---------------- эффекты удара ----------------
    def _impact_fx(self, v, frame_dt):
        """Обломки, тряска камеры и вспышка — по скачку скорости узлов."""
        imp = v.impact
        self.fx_cool = max(0.0, getattr(self, "fx_cool", 0.0) - frame_dt)
        if imp > 2.6 and self.fx_cool <= 0.0:
            self.fx_cool = 0.11
            self.camera.kick(imp)
            self.audio.on_impact(imp)
            n = int(min(48, 4 + imp * 2.2 + v.impact_dmg * 380))
            cols = [(0.80, 0.13, 0.11), (0.62, 0.10, 0.09), (0.55, 0.62, 0.68),
                    (0.20, 0.20, 0.22)]
            if v.impact_dmg > 0.004:                 # выбитое стекло — светлая крошка
                cols += [(0.72, 0.82, 0.88)] * 2
            self.fx.emit(n, v.impact_pos, v.velocity() * 0.25, cols,
                         spread=min(6.0, 1.4 + imp * 0.4), life=1.5,
                         size=0.030 + 0.006 * min(imp, 8.0))
            self.flash = min(0.55, max(self.flash, (imp - 4.0) * 0.06))
            if v.impact_dmg > 0.02:
                self.say("Удар! Повреждения %d%%" % round(v.sb.damage() * 100))
        if v.lost_wheel >= 0:
            self.fx.emit(22, v.sb.pos[v.lost_wheel], v.velocity() * 0.2,
                         [(0.12, 0.12, 0.13), (0.5, 0.5, 0.52)], spread=3.5,
                         life=1.6, size=0.05)
            self.say("Оторвало колесо!")
            v.lost_wheel = -1
        self.fx.update(frame_dt, self.terrain)
        self.flash = max(0.0, self.flash - frame_dt * 1.6)

    # ---------------- отрисовка ----------------
    def _draw(self, frame_dt):
        r = self.renderer
        if not hasattr(self, "eye"):
            self.eye, self.target, self.up = self.camera.update(
                self.vehicle, 1 / 60.0, self.terrain)

        r.begin_3d(self.eye, self.target, self.up)
        r.draw_terrain()
        r.draw_shadows(self.vehicle, self.terrain)
        r.draw_boxes()
        self.mode.draw_markers(r, self.terrain)
        r.draw_vehicle(self.vehicle, self.show_beams, self.show_panels)
        r.draw_headlights(self.vehicle)
        r.draw_particles(self.fx, self.eye, self.target)
        r.draw_smoke(self.smoke, self.eye, self.target)

        r.begin_2d()
        if self.state == "game":
            r.flash(self.flash)
            if self.hud_surf:
                r.blit(self.hud_surf, 12, 12)
            if self.hint_surf is None:
                self.hint_surf = self.ui.build_hint(self.w)
            r.blit(self.hint_surf, (self.w - self.hint_surf.get_width()) // 2,
                   self.h - self.hint_surf.get_height() - 12)
            if self.toast_t > 0:
                t = self.ui.build_toast(self.toast)
                r.blit(t, (self.w - t.get_width()) // 2, 28)
        elif self.state == "help":
            r.dim(0.72)
            s, x, y = self.ui.build_help(self.w, self.h)
            r.blit(s, x, y)
        elif self.state == "garage":
            r.dim(0.78)
            s, x, y, self.garage_rects = self.ui.build_garage(
                self.w, self.h, self.garage_items, self.garage_sel, self.vehicle_id)
            r.blit(s, x, y)
        else:
            r.dim(0.62 if self.state == "pause" else 0.78)
            s, x, y = self.menu.render(self.w, self.h)
            r.blit(s, x, y)


# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="BeamNG.Drive Lite")
    ap.add_argument("--headless", type=int, default=0, metavar="STEPS",
                    help="прогнать N шагов физики без окна и выйти")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=760)
    ap.add_argument("--seed", type=int, default=7, help="сид ландшафта")
    ap.add_argument("--vehicle", default=garage.DEFAULT_ID,
                    help="машина из гаража (id JSON из каталога vehicles/)")
    ap.add_argument("--list-vehicles", action="store_true",
                    help="показать машины гаража и выйти")
    ap.add_argument("--no-vsync", action="store_true")
    a = ap.parse_args(argv)
    if a.list_vehicles:
        for vid, cfg in garage.load_all().items():
            print("%-10s %s" % (vid, cfg["name"]))
        return 0
    if a.headless:
        return headless(a.headless, a.seed, a.vehicle)
    return Game(a.width, a.height, a.seed, not a.no_vsync, a.vehicle).run()


if __name__ == "__main__":
    sys.exit(main())
