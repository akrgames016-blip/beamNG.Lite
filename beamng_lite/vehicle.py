"""Машина: soft-body кузов, подвеска, шины, трансмиссия.

Все параметры — геометрия кузова (сечения), колёса, мотор, трансмиссия,
аэродинамика — берутся из конфига гаража (garage.py), см. каталог vehicles/.
"""
import numpy as np

from .softbody import SoftBody, GRAVITY
from .body import build_skin
from .garage import DEFAULT


class Wheel:
    def __init__(self, node, steered, driven, x, z):
        self.node = node
        self.steered = steered
        self.driven = driven
        self.x, self.z = x, z
        self.omega = 0.0          # угловая скорость, рад/с
        self.angle = 0.0          # угол поворота для рендера
        self.steer = 0.0
        self.contact = False
        self.load = 0.0
        self.slip = 0.0
        self.attached = True


class Vehicle:
    def __init__(self, terrain, spawn=(0.0, 0.0), heading=0.0, cfg=None):
        self.terrain = terrain
        self.spawn = spawn
        self.heading = heading
        self.cfg = cfg if cfg is not None else dict(DEFAULT)
        self.build()

    # ------------------------------------------------------------------
    def build(self):
        cfg = self.cfg
        t = self.terrain
        sb = SoftBody()
        self.sb = sb
        x0, z0 = self.spawn
        y0 = float(t.height(x0, z0)) + 0.06        # уровень земли под машиной
        cs, sn = np.cos(self.heading), np.sin(self.heading)

        def world(lx, ly, lz):
            return np.array([x0 + lx * cs + lz * sn, y0 + ly, z0 - lx * sn + lz * cs])

        # --- параметры из конфига ---
        sections = [tuple(s) for s in cfg["sections"]]
        belt_y = float(cfg.get("belt_y", 0.97))
        body_mass = float(cfg["body_mass"])
        self.wheel_r = float(cfg["wheel_r"])
        self.wheel_w = float(cfg["wheel_w"])
        wheel_mass = float(cfg["wheel_mass"])
        self.wheel_inertia = 0.5 * wheel_mass * self.wheel_r ** 2
        self.max_steer = np.radians(float(cfg.get("max_steer_deg", 34.0)))
        self.gears = [float(g) for g in cfg["gears"]]
        self.final_drive = float(cfg["final_drive"])
        self.idle_rpm = float(cfg.get("idle_rpm", 900.0))
        self.max_rpm = float(cfg.get("max_rpm", 7000.0))
        self.engine_inertia = float(cfg.get("engine_inertia", 0.24))
        self.lsd = float(cfg.get("lsd", 140.0))
        self.tire_mu = float(cfg.get("tire_mu", 1.32))
        aero = cfg.get("aero", {})
        self.aero_cd = float(aero.get("cd", 0.62))
        self.aero_area = float(aero.get("area", 2.1))
        driven_axle = cfg.get("driven", "rear")

        # --- узлы кузова: grid[(сторона, уровень 0..2, секция)] ---
        # три уровня: днище, линия остекления (пояс), крыша
        ns = len(sections)
        nm = body_mass / (ns * 6)
        # Жёсткость кузова и подвески масштабируется с массой: иначе у лёгкой
        # машины собственная частота решётки вырастает настолько, что явный
        # интегратор (h·ω → 2) теряет устойчивость и кузов «взрывается».
        # За эталон взят седан: 1080 кг.
        k_scale = body_mass / 1080.0
        self.grid = {}
        self.level_y = []
        for iz, (z, hw, ylo, yhi) in enumerate(sections):
            ybelt = float(np.clip(belt_y, ylo + 0.16, yhi - 0.06))
            ys = (ylo, ybelt, yhi)
            self.level_y.append(ys)
            for ix, sx in enumerate((-1, 1)):
                for iy, y in enumerate(ys):
                    self.grid[(ix, iy, iz)] = sb.add_node(world(sx * hw, y, z), nm,
                                                          {"type": "body"})
        self.body_nodes = np.array(sorted(self.grid.values()))

        # --- балки кузова: каждый узел с каждым ---
        # Жёсткость и порог текучести зависят от места: нос и корма — зоны
        # смятия (мягче, текут раньше), салон — жёсткая клетка безопасности.
        node_z = {}
        for (ix, iy, iz), nid in self.grid.items():
            node_z[nid] = sections[iz][0]
        ids = self.body_nodes
        for a_i in range(len(ids)):
            for b_i in range(a_i + 1, len(ids)):
                a, b = int(ids[a_i]), int(ids[b_i])
                L = float(np.linalg.norm(sb.pos[a] - sb.pos[b]))
                if L > 2.35:
                    continue      # сквозных связей через весь кузов нет:
                                  # иначе удар обходит зону смятия стороной
                k = min(1.1e6, 7.0e5 / max(L, 0.55)) * k_scale
                zm = max(abs(node_z[a]), abs(node_z[b]))
                crumple = min(max((zm - 0.95) / 1.10, 0.0), 1.0)   # 0 — салон, 1 — бампер
                sb.add_beam(a, b,
                            k=k * (1.0 - 0.45 * crumple),
                            zeta=0.22,
                            deform=0.048 - 0.025 * crumple,
                            brk=0.57 - 0.23 * crumple)

        # --- колёса + подвеска ---
        # 4 почти горизонтальных жёстких рычага держат колесо продольно
        # и поперечно, 2 мягкие пружины к крыше дают ход подвески.
        self.wheels = []
        self.wheel_beams = []
        zs = [s[0] for s in sections]
        for (wx, wz) in cfg["wheel_pos"]:
            p = world(wx, self.wheel_r, wz)
            n = sb.add_node(p, wheel_mass, {"type": "wheel"})
            driven = (driven_axle == "all" or
                      (driven_axle == "rear" and wz < 0) or
                      (driven_axle == "front" and wz > 0))
            w = Wheel(n, steered=(wz > 0), driven=driven, x=wx, z=wz)
            self.wheels.append(w)

            mine = []
            ix = 0 if wx < 0 else 1
            ox = 1 - ix
            order = sorted(range(len(zs)), key=lambda i: abs(zs[i] - wz))
            s_a, s_b = sorted(order[:2])
            for tgt in ((ix, 0, s_a), (ix, 0, s_b), (ox, 0, s_a), (ox, 0, s_b)):
                mine.append(sb.add_beam(self.grid[tgt], n, k=1.9e6 * k_scale, zeta=0.28,
                                        deform=9.9, brk=0.34))  # рычаги (рвутся в тяжёлой аварии)
            for tgt in ((ix, 2, s_a), (ix, 2, s_b)):
                sb.add_beam(self.grid[tgt], n, k=3.0e4 * k_scale, zeta=0.45,
                            deform=9.9, brk=2.5)       # пружина + амортизатор
            self.wheel_beams.append(np.array(mine, dtype=int))

        sb.finalize()
        # индексы для быстрого расчёта осей кузова (frame() зовётся 9 раз за кадр)
        last = ns - 1
        self._i_front = np.array([self.grid[(ix, iy, last)] for ix in (0, 1) for iy in (0, 1, 2)])
        self._i_rear = np.array([self.grid[(ix, iy, 0)] for ix in (0, 1) for iy in (0, 1, 2)])
        self._i_low = np.array([self.grid[(ix, 0, iz)] for ix in (0, 1) for iz in range(ns)])
        self._i_high = np.array([self.grid[(ix, 2, iz)] for ix in (0, 1) for iz in range(ns)])
        self.wheel_idx = np.array([w.node for w in self.wheels])
        self.body_mask = np.ones(len(sb.mass), dtype=bool)
        self.body_mask[self.wheel_idx] = False
        self.body_idx = np.nonzero(self.body_mask)[0]

        # --- состояние трансмиссии/управления ---
        self.impact = 0.0            # скачок скорости узла за кадр, м/с
        self.impact_pos = np.zeros(3)
        self.impact_dmg = 0.0
        self.lost_wheel = -1
        self.gear = 2                # индекс в gears (2 == 1-я передача)
        self.rpm = self.idle_rpm
        self.throttle = 0.0
        self.brake = 0.0
        self.handbrake = 0.0
        self.steer_input = 0.0
        self.steer = 0.0
        self.auto = True
        self.tc = True               # антипробуксовка
        self.tc_scale = 1.0
        self.shift_timer = 0.0
        self.scrape = 0.0            # скорость царапания кузовом по земле, м/с
        self._scrape_now = 0.0
        self.skin = build_skin(self)

    # ------------------------------------------------------------------
    def engine_torque(self, rpm):
        """Кривая момента из конфига: базовый момент + гауссов пик + отсечка."""
        tq = self.cfg["torque"]
        r = min(max(rpm, 0.0), self.max_rpm)
        t = tq["base"] + tq["peak"] * np.exp(
            -((r - tq["peak_rpm"]) ** 2) / (2 * tq["width"] ** 2))
        t *= min(max((self.max_rpm - r) / 900.0, 0.0), 1.0)      # отсечка
        return float(max(t, 0.0))

    # ------------------------------------------------------------------
    # ориентация и кинематика
    def frame(self):
        """(центр, forward, right, up) кузова. right — настоящая правая сторона."""
        p = self.sb.pos
        fwd = p[self._i_front].mean(axis=0) - p[self._i_rear].mean(axis=0)
        up = p[self._i_high].mean(axis=0) - p[self._i_low].mean(axis=0)
        fwd /= max(np.linalg.norm(fwd), 1e-9)
        up -= fwd * (up @ fwd)
        up /= max(np.linalg.norm(up), 1e-9)
        rgt = np.cross(fwd, up)          # правая рука: Z×Y = -X
        rgt /= max(np.linalg.norm(rgt), 1e-9)
        center = p[self.body_nodes].mean(axis=0)
        return center, fwd, rgt, up

    def velocity(self):
        m = self.sb.mass[:, None]
        return (self.sb.vel * m).sum(axis=0) / self.sb.mass.sum()

    def speed_kmh(self):
        return float(np.linalg.norm(self.velocity()) * 3.6)

    # ------------------------------------------------------------------
    def step(self, dt, substeps=12):
        sb = self.sb
        v0 = sb.vel[self.body_idx].copy()
        d0 = sb.damage()
        h = dt / substeps
        for _ in range(substeps):
            self._substep(h)
        self.terrain.step_boxes(dt)
        self._gearbox(dt)

        # --- сила удара за кадр: по узлу с наибольшим скачком скорости ---
        dv = np.linalg.norm(sb.vel[self.body_idx] - v0, axis=1)
        k = int(np.argmax(dv))
        self.impact = float(dv[k])
        self.impact_pos = sb.pos[self.body_idx[k]].copy()
        self.impact_dmg = max(0.0, sb.damage() - d0)
        self._check_wheels()
        # скрежет кузова: затухающая добавка от _terrain_collide
        self.scrape += (self._scrape_now - self.scrape) * min(1.0, dt * 12.0)

    def _check_wheels(self):
        """Отрыв колеса: если все его балки подвески порваны — колесо теряется."""
        sb = self.sb
        for w, beams in zip(self.wheels, self.wheel_beams):
            if w.attached and len(beams) and int((~sb.alive[beams]).sum()) >= 3:
                w.attached = False
                self.lost_wheel = int(self.wheel_idx[self.wheels.index(w)])

    def _substep(self, dt):
        sb = self.sb
        f = sb.mass[:, None] * GRAVITY
        sb.beam_forces(f)
        self._aero(f)
        frame = self.frame()
        self._terrain_collide(f)
        self._box_collide(f, dt, frame)
        self._tires(f, dt, frame)

        sb.vel += f * sb.inv_mass[:, None] * dt
        np.clip(sb.vel, -160.0, 160.0, out=sb.vel)
        sb.pos += sb.vel * dt

    # ---- аэродинамика ----
    def _aero(self, f):
        v = self.velocity()
        sp = float(np.linalg.norm(v))
        if sp > 0.1:
            f += (-0.5 * 1.2 * self.aero_cd * self.aero_area * sp * v) / len(f)

    # ---- контакт узлов кузова с землёй (царапание, кувырки) ----
    def _terrain_collide(self, f):
        sb = self.sb
        idx = self.body_idx
        p = sb.pos[idx]
        gh = self.terrain.height(p[:, 0], p[:, 2])
        pen = gh - p[:, 1]
        hit = pen > 0
        if not hit.any():
            self._scrape_now = 0.0
            return
        ids = idx[hit]
        pen = pen[hit]
        n = self.terrain.normal(p[hit, 0], p[hit, 2])
        v = sb.vel[ids]
        vn = np.einsum("ij,ij->i", v, n)
        fn = np.clip(3.0e5 * pen - 3.0e3 * np.minimum(vn, 0.0), 0, 2.0e5)
        vt = v - n * vn[:, None]
        vts = np.linalg.norm(vt, axis=1)
        ft = -vt / np.maximum(vts, 1e-6)[:, None] * np.minimum(fn * 0.8, vts * 900.0)[:, None]
        np.add.at(f, ids, n * fn[:, None] + ft)
        # для звука скрежета: средняя тангенциальная скорость контакта
        self._scrape_now = float(vts[vts > 1.0].mean()) if (vts > 1.0).any() else 0.0

    # ---- контакт с ящиками: по контактным точкам шкуры кузова ----
    def _cage(self, frame):
        """Мировые координаты и скорости точек контактной клетки."""
        sk = self.skin
        _, fwd, rgt, up = frame
        R = np.column_stack([-rgt, up, fwd])
        base = self.sb.pos[sk.cage_nodes]
        off = sk.cage_off @ R.T
        pos = np.einsum("pk,pkj->pj", sk.cage_w, base + off)
        vel = np.einsum("pk,pkj->pj", sk.cage_w, self.sb.vel[sk.cage_nodes])
        return pos, vel

    def _box_collide(self, f, dt, frame):
        sb = self.sb
        boxes = self.terrain.boxes
        if len(boxes) == 0:
            return
        # широкая фаза: клетку считаем, только если ящик рядом
        ctr = frame[0]
        far = np.linalg.norm(boxes[:, :3] - ctr, axis=1) > (boxes[:, 3:].max(axis=1) + 3.6)
        if far.all():
            return
        sk = self.skin
        cpos, cvel = self._cage(frame)
        r = sk.cage_r[:, None]
        for bi in np.nonzero(~far)[0]:
            box = boxes[bi]
            c, hs = box[:3], box[3:]
            d = cpos - c
            inside = np.all(np.abs(d) < hs + r, axis=1)
            if not inside.any():
                continue
            ids = np.nonzero(inside)[0]
            dd = np.abs(d[ids]) - hs - sk.cage_r[ids, None]
            ax = np.argmax(dd, axis=1)
            rows = np.arange(len(ids))
            pen = -dd[rows, ax]
            n = np.zeros((len(ids), 3))
            n[rows, ax] = np.sign(d[ids, ax] + 1e-12)

            vrel = cvel[ids] - self.terrain.box_vel[bi]
            vn = np.einsum("ij,ij->i", vrel, n)
            fn = np.clip(9.0e5 * pen - 3.5e4 * np.minimum(vn, 0.0), 0, 6.0e5)
            force = n * fn[:, None]
            vt = vrel - n * vn[:, None]
            vts = np.linalg.norm(vt, axis=1)
            force -= vt / np.maximum(vts, 1e-6)[:, None] * \
                np.minimum(fn * 0.65, vts * 1200.0)[:, None]

            # раскидываем силу по узлам, к которым привязана точка шкуры
            nodes = sk.cage_nodes[ids].ravel()
            share = (sk.cage_w[ids][:, :, None] * force[:, None, :]).reshape(-1, 3)
            np.add.at(f, nodes, share)
            self.terrain.push_box(bi, -force.sum(axis=0) * dt)

            # жёсткое выталкивание, чтобы не залипать внутри ящика
            deep = pen > 0.05
            if deep.any():
                did = ids[deep]
                push = n[deep] * (pen[deep] - 0.05)[:, None]
                pnodes = sk.cage_nodes[did].ravel()
                pshare = (sk.cage_w[did][:, :, None] * push[:, None, :]).reshape(-1, 3)
                np.add.at(sb.pos, pnodes, pshare * 0.8)
                vv = sb.vel[pnodes]
                nn = np.repeat(n[deep], sk.cage_nodes.shape[1], axis=0)
                vnn = np.einsum("ij,ij->i", vv, nn)
                bad = vnn < 0
                if bad.any():
                    sb.vel[pnodes[bad]] -= nn[bad] * vnn[bad][:, None]

    # ---- шины (запросы к рельефу векторизованы: одна пачка на 4 колеса) ----
    def _tires(self, f, dt, frame):
        sb = self.sb
        _, fwd, rgt, up = frame
        idx = self.wheel_idx
        P = sb.pos[idx]
        V = sb.vel[idx]
        gh = np.asarray(self.terrain.height(P[:, 0], P[:, 2]), dtype=np.float64)
        NR = np.asarray(self.terrain.normal(P[:, 0], P[:, 2]), dtype=np.float64)
        pen = gh + self.wheel_r - P[:, 1]

        for i, w in enumerate(self.wheels):
            if not w.attached:
                continue
            w.contact = bool(pen[i] > 0)
            if not w.contact:
                w.load = 0.0
                w.slip = 0.0
                w.omega += (self._wheel_torque(w) - self._brake_torque(w, w.omega)) \
                    / self._wheel_inertia(w) * dt
                w.angle += w.omega * dt
                continue

            n = NR[i]
            v = V[i]
            vn = float(v @ n)
            fn = float(min(max(3.2e5 * pen[i] - 6.0e3 * min(vn, 0.0), 0.0), 4.0e4))
            w.load = fn

            ang = w.steer if w.steered else 0.0
            wf = fwd * np.cos(ang) - rgt * np.sin(ang)   # + = влево
            wf = wf - n * float(wf @ n)
            nl = float(np.linalg.norm(wf))
            if nl < 1e-6:
                continue
            wf /= nl
            wr = np.cross(n, wf)

            v_long = float(v @ wf)
            v_lat = float(v @ wr)
            denom = max(abs(v_long), 1.6)
            slip = (w.omega * self.wheel_r - v_long) / denom
            alpha = np.arctan2(-v_lat, denom)

            fx = fn * self.tire_mu * np.tanh(slip * 5.5)
            fy = fn * self.tire_mu * np.tanh(alpha * 6.5)
            mag = np.hypot(fx, fy)
            lim = fn * self.tire_mu
            if mag > lim > 0:                     # круг сцепления
                fx *= lim / mag
                fy *= lim / mag
            w.slip = float(min(abs(slip), 2.0))

            f[idx[i]] += n * fn + wf * fx + wr * fy

            tq = self._wheel_torque(w) - fx * self.wheel_r - self._brake_torque(w, w.omega)
            w.omega += tq / self._wheel_inertia(w) * dt
            if abs(w.omega) < 0.05 and (self.brake > 0.2 or self.handbrake > 0.5):
                w.omega = 0.0
            w.angle += w.omega * dt

    def _wheel_inertia(self, w):
        """Приведённая к колесу инерция (колесо + маховик через передачу)."""
        if w.driven and self.gears[self.gear] != 0.0:
            ratio = self.gears[self.gear] * self.final_drive
            return self.wheel_inertia + self.engine_inertia * ratio * ratio * 0.5
        return self.wheel_inertia

    def _wheel_torque(self, w):
        if not w.driven:
            return 0.0
        lock = 0.0                                  # вязкая блокировка диффа
        for o in self.wheels:
            if o.driven and o is not w and o.attached:
                lock += float(np.clip(self.lsd * (o.omega - w.omega), -2500, 2500))
        if self.gears[self.gear] == 0.0:
            return lock
        ratio = self.gears[self.gear] * self.final_drive
        te = self.engine_torque(self.rpm) * self.throttle
        te -= 22.0 * (1.0 - self.throttle) * (self.rpm / self.max_rpm)
        clutch = float(np.clip((self.rpm - self.idle_rpm * 0.6) / 700.0, 0.0, 1.0))
        return te * ratio * clutch * 0.5 + lock

    def _brake_torque(self, w, omega):
        base = 2400.0 if not w.driven else 1900.0
        t = self.brake * base
        if w.driven:
            t += self.handbrake * 2600.0
        return np.sign(omega) * t if abs(omega) > 1e-3 else float(np.clip(omega * 1e4, -t, t))

    # ---- КПП ----
    def _gearbox(self, dt):
        driven = [w for w in self.wheels if w.driven and w.attached]
        wheel_omega = float(np.mean([w.omega for w in driven])) if driven else 0.0
        ratio = self.gears[self.gear] * self.final_drive
        if ratio != 0.0:
            target = abs(wheel_omega * ratio) * 60.0 / (2 * np.pi)
        else:
            target = self.idle_rpm + self.throttle * 4000.0
        self.rpm += (np.clip(target, self.idle_rpm, self.max_rpm) - self.rpm) * min(1.0, dt * 12.0)

        self.shift_timer = max(0.0, self.shift_timer - dt)
        if self.auto and self.shift_timer == 0.0 and self.gear >= 2:
            up_at = self.max_rpm * 0.857
            down_at = self.max_rpm * 0.357
            if self.rpm > up_at and self.gear < len(self.gears) - 1:
                self.gear += 1
                self.shift_timer = 0.45
            elif self.rpm < down_at and self.gear > 2:
                self.gear -= 1
                self.shift_timer = 0.45

    # ---- управление ----
    def controls(self, throttle, brake, steer, handbrake, dt):
        thr = float(np.clip(throttle, 0, 1))
        if self.tc:                       # антипробуксовка
            slip = max((w.slip for w in self.wheels if w.driven), default=0.0)
            want = 1.0 if slip < 0.25 else max(0.15, 1.0 - (slip - 0.25) * 2.2)
            rate = dt * (6.0 if want < self.tc_scale else 2.0)
            self.tc_scale += float(np.clip(want - self.tc_scale, -rate, rate))
        else:
            self.tc_scale = 1.0
        self.throttle = thr * float(np.clip(self.tc_scale, 0.0, 1.0))
        self.brake = float(np.clip(brake, 0, 1))
        self.handbrake = float(handbrake)

        sp = self.speed_kmh()
        limit = self.max_steer * (1.0 - 0.55 * min(sp / 150.0, 1.0))
        target = float(np.clip(steer, -1, 1)) * limit
        rate = 4.5 * dt
        self.steer += float(np.clip(target - self.steer, -rate, rate))
        for w in self.wheels:
            if w.steered:
                w.steer = self.steer

    def shift(self, delta):
        self.auto = False
        self.gear = int(np.clip(self.gear + delta, 0, len(self.gears) - 1))
        self.shift_timer = 0.3

    def recover(self):
        """Поставить на колёса на текущем месте."""
        c, fwd, _, _ = self.frame()
        self.spawn = (float(c[0]), float(c[2]))
        self.heading = float(np.arctan2(fwd[0], fwd[2]))
        self.build()

    def reset(self, spawn=None, heading=None):
        if hasattr(self.terrain, "reset_boxes"):
            self.terrain.reset_boxes()
        if spawn is not None:
            self.spawn = spawn
        if heading is not None:
            self.heading = heading
        self.build()

    def damage(self):
        return self.sb.damage()
