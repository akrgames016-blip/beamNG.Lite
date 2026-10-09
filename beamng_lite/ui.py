"""HUD и меню: рисуются в pygame-поверхности, затем выводятся текстурой."""
import pygame

BG = (12, 15, 20)
ACCENT = (255, 196, 60)
WHITE = (240, 244, 250)
GREY = (168, 180, 196)
GREEN = (140, 226, 140)
RED = (255, 96, 84)

_FONTS = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _FONTS:
        f = pygame.font.SysFont("dejavusans,freesans,liberationsans,arial", size, bold=bold)
        if f is None:
            f = pygame.font.Font(None, size)
        _FONTS[key] = f
    return _FONTS[key]


def _panel(size, radius=12, alpha=170):
    s = pygame.Surface(size, pygame.SRCALPHA)
    pygame.draw.rect(s, (*BG, alpha), s.get_rect(), border_radius=radius)
    pygame.draw.rect(s, (90, 105, 125, 120), s.get_rect(), width=1, border_radius=radius)
    return s


# ----------------------------------------------------------------------
def build_hud(veh, cam_name, fps, paused, slowmo, screen_w, mode_line=""):
    """Панель приборов. Ширина считается по содержимому — текст не вылезает."""
    f_big = font(30, True)
    f = font(17)
    f_small = font(15)

    gear = {0: "R", 1: "N"}.get(veh.gear, str(veh.gear - 1))
    dmg = veh.damage() * 100
    broken = int((~veh.sb.alive).sum())
    onground = sum(1 for w in veh.wheels if w.contact)

    head = f_big.render("%3.0f км/ч" % veh.speed_kmh(), True, WHITE)
    sub = f.render("передача %s %s   %4.0f об/мин" % (
        gear, "A" if veh.auto else "M", veh.rpm), True, GREY)
    rows = [
        ("повреждения %.1f%%%s" % (dmg, "   разорвано балок: %d" % broken if broken else ""),
         RED if dmg > 20 else GREEN),
        ("колёс на земле %d/4    проскальзывание %.2f" % (
            onground, max(w.slip for w in veh.wheels)), GREY),
        ("ESP/TC: %s    камера: %s    FPS %.0f" % (
            "вкл" if veh.tc else "выкл", cam_name, fps), GREY),
    ]
    if paused or slowmo:
        rows.append(("%s%s" % ("[ПАУЗА] " if paused else "", "[SLOW-MO]" if slowmo else ""),
                     ACCENT))

    texts = [f_small.render(t, True, c) for t, c in rows]
    w = max([head.get_width() + 20, sub.get_width()] + [t.get_width() for t in texts]) + 28
    w = min(w, screen_w - 24)
    bar_h = 10
    h = head.get_height() + sub.get_height() + bar_h + 10 + sum(t.get_height() + 3 for t in texts) + 22

    s = _panel((w, h))
    s.blit(head, (14, 8))
    s.blit(sub, (14, 10 + head.get_height()))
    y = 12 + head.get_height() + sub.get_height()
    # шкала тахометра
    bw = w - 28
    pygame.draw.rect(s, (40, 46, 56), (14, y, bw, bar_h), border_radius=5)
    max_rpm = float(veh.cfg.get("max_rpm", 7000.0))
    redline = float(veh.cfg.get("redline_rpm", 0.857 * max_rpm))
    frac = min(max(veh.rpm / max_rpm, 0.0), 1.0)
    col = RED if veh.rpm > redline else (110, 210, 130)
    pygame.draw.rect(s, col, (14, y, int(bw * frac), bar_h), border_radius=5)
    pygame.draw.rect(s, (240, 120, 90), (14 + int(bw * redline / max_rpm), y, 2, bar_h))
    y += bar_h + 8
    for t in texts:
        s.blit(t, (14, y))
        y += t.get_height() + 3
    return s


def build_hint(screen_w):
    f = font(15)
    txt = "W/S газ-тормоз   A/D руль   Space ручник   R респавн   F перевернуть   C камера   ESC меню"
    t = f.render(txt, True, GREY)
    if t.get_width() > screen_w - 24:      # не влезает — режем на две строки
        a = "W/S газ-тормоз   A/D руль   Space ручник   R респавн   F перевернуть"
        b = "C камера   B балки   T slow-mo   Q/E передачи   ESC меню"
        ta, tb = f.render(a, True, GREY), f.render(b, True, GREY)
        s = _panel((max(ta.get_width(), tb.get_width()) + 24, ta.get_height() * 2 + 16))
        s.blit(ta, (12, 6)); s.blit(tb, (12, 8 + ta.get_height()))
        return s
    s = _panel((t.get_width() + 24, t.get_height() + 14))
    s.blit(t, (12, 7))
    return s


def build_toast(text):
    f = font(24, True)
    t = f.render(text, True, ACCENT)
    s = _panel((t.get_width() + 32, t.get_height() + 16), radius=10, alpha=190)
    s.blit(t, (16, 8))
    return s


# ----------------------------------------------------------------------
class Menu:
    """Простое меню: клавиши вверх/вниз/Enter или мышь."""

    def __init__(self, title, items, subtitle=""):
        self.title = title
        self.subtitle = subtitle
        self.items = items                # список (id, текст) или (id, текст, значение)
        self.sel = 0
        self.rects = []
        self._size = (0, 0)

    def move(self, d):
        self.sel = (self.sel + d) % len(self.items)

    def current(self):
        return self.items[self.sel][0]

    def hover(self, mx, my):
        for i, r in enumerate(self.rects):
            if r.collidepoint(mx, my):
                self.sel = i
                return True
        return False

    def click(self, mx, my):
        for i, r in enumerate(self.rects):
            if r.collidepoint(mx, my):
                self.sel = i
                return self.items[i][0]
        return None

    def render(self, screen_w, screen_h):
        f_title = font(46, True)
        f_sub = font(17)
        f_item = font(24, True)
        f_val = font(20)

        title = f_title.render(self.title, True, WHITE)
        sub = f_sub.render(self.subtitle, True, GREY) if self.subtitle else None

        item_surfs = []
        for it in self.items:
            label = f_item.render(it[1], True, WHITE)
            value = f_val.render(str(it[2]), True, ACCENT) if len(it) > 2 else None
            item_surfs.append((label, value))

        pad = 34
        row_h = 46
        w = max([title.get_width(), sub.get_width() if sub else 0] +
                [ls.get_width() + (vs.get_width() + 30 if vs else 0) for ls, vs in item_surfs]) + pad * 2
        w = max(w, 420)
        w = min(w, screen_w - 40)
        h = pad * 2 + title.get_height() + (sub.get_height() + 10 if sub else 0) + 18 \
            + row_h * len(self.items)
        x = (screen_w - w) // 2
        y = (screen_h - h) // 2

        s = _panel((w, h), radius=18, alpha=225)
        s.blit(title, ((w - title.get_width()) // 2, pad - 8))
        cy = pad - 8 + title.get_height() + 6
        if sub:
            s.blit(sub, ((w - sub.get_width()) // 2, cy))
            cy += sub.get_height() + 10
        cy += 10

        self.rects = []
        for i, (label, value) in enumerate(item_surfs):
            r = pygame.Rect(pad - 12, cy - 6, w - (pad - 12) * 2, row_h - 6)
            if i == self.sel:
                pygame.draw.rect(s, (54, 66, 84, 230), r, border_radius=9)
                pygame.draw.rect(s, ACCENT, (r.x, r.y + 6, 4, r.h - 12), border_radius=2)
            s.blit(label, (pad + 4, cy))
            if value:
                s.blit(value, (w - pad - value.get_width(), cy + 3))
            self.rects.append(pygame.Rect(x + r.x, y + r.y, r.w, r.h))
            cy += row_h
        return s, x, y


HELP_TEXT = [
    ("W / стрелка вверх", "газ"),
    ("S / стрелка вниз", "тормоз, на месте — задний ход"),
    ("A / D", "руль влево / вправо"),
    ("Space", "ручной тормоз"),
    ("R", "респавн на старте"),
    ("F", "поставить машину на колёса"),
    ("C / Shift+C", "следующая / предыдущая камера"),
    ("ЛКМ + мышь", "вращать камеру, колесо — зум"),
    ("V", "вернуть камеру за машину"),
    ("I J K L U O", "полёт свободной камерой"),
    ("B / N", "показать балки / панели кузова"),
    ("Q / E", "передача вниз / вверх"),
    ("G", "коробка: автомат / ручная"),
    ("X", "антипробуксовка вкл/выкл"),
    ("T / P", "slow-mo / пауза"),
    ("ESC", "меню"),
]


DRIVEN_NAMES = {"rear": "задний привод", "front": "передний привод",
                "all": "полный привод"}


def build_garage(screen_w, screen_h, vehicles, sel, current_id):
    """Экран гаража: список машин гаража с характеристиками.

    vehicles — список (id, cfg) в порядке показа. Возвращает
    (surface, x, y, rects) — rects для мыши.
    """
    f_title = font(40, True)
    f_name = font(24, True)
    f_desc = font(17)
    f_stat = font(16)
    f_hint = font(15)

    title = f_title.render("Гараж", True, WHITE)
    rows = []
    for vid, cfg in vehicles:
        drive = DRIVEN_NAMES.get(cfg.get("driven", "rear"), "?")
        n_fwd = max(0, len(cfg["gears"]) - 2)
        stats = "%d кг · %s · %d передач · до %d об/мин · колёса %.2f м" % (
            int(cfg["body_mass"]), drive, n_fwd, int(cfg.get("max_rpm", 7000)),
            2 * cfg["wheel_r"])
        sw = pygame.Surface((16, 16))
        sw.fill(tuple(int(c * 235) for c in cfg["color"]))
        rows.append((vid, cfg, f_name.render(cfg["name"], True, WHITE),
                     f_desc.render(cfg.get("description", ""), True, GREY),
                     f_stat.render(stats, True, ACCENT), sw))

    pad = 30
    row_h = 78
    w = min(max([title.get_width()] +
                [r[2].get_width() + 70 for r in rows] +
                [r[3].get_width() for r in rows] +
                [r[4].get_width() for r in rows]) + pad * 2 + 40, 520)
    w = min(w, screen_w - 40)
    h = pad * 2 + title.get_height() + 16 + row_h * len(rows) + 40
    h = min(h, screen_h - 40)
    x = (screen_w - w) // 2
    y = (screen_h - h) // 2

    s = _panel((w, h), radius=18, alpha=235)
    s.blit(title, ((w - title.get_width()) // 2, pad - 6))
    cy = pad - 6 + title.get_height() + 14

    rects = []
    for i, (vid, cfg, name_s, desc_s, stat_s, swatch) in enumerate(rows):
        r = pygame.Rect(pad - 10, cy - 8, w - (pad - 10) * 2, row_h - 8)
        if vid == current_id:
            pygame.draw.rect(s, (38, 74, 52, 230), r, border_radius=9)
        if i == sel:
            pygame.draw.rect(s, (54, 66, 84, 230), r, border_radius=9)
            pygame.draw.rect(s, ACCENT, (r.x, r.y + 8, 4, r.h - 16), border_radius=2)
        s.blit(swatch, (pad + 6, cy + 8))
        pygame.draw.rect(s, (90, 105, 125, 160),
                         swatch.get_rect(topleft=(pad + 6, cy + 8)), width=1)
        s.blit(name_s, (pad + 34, cy))
        s.blit(desc_s, (pad + 34, cy + 28))
        s.blit(stat_s, (pad + 34, cy + 50))
        if vid == current_id:
            ok = f_stat.render("✓ в гараже", True, GREEN)
            s.blit(ok, (w - pad - ok.get_width(), cy + 4))
        rects.append(pygame.Rect(x + r.x, y + r.y, r.w, r.h))
        cy += row_h

    hint = f_hint.render("↑/↓ или мышь — выбор, Enter — выехать, ESC — назад", True, GREY)
    s.blit(hint, ((w - hint.get_width()) // 2, h - 30))
    return s, x, y, rects


def build_help(screen_w, screen_h):
    f_title = font(34, True)
    f_k = font(18, True)
    f_v = font(18)
    title = f_title.render("Управление", True, WHITE)
    keys = [f_k.render(k, True, ACCENT) for k, _ in HELP_TEXT]
    vals = [f_v.render(v, True, GREY) for _, v in HELP_TEXT]
    kw = max(t.get_width() for t in keys)
    w = min(kw + 30 + max(t.get_width() for t in vals) + 68, screen_w - 40)
    h = 96 + sum(t.get_height() + 6 for t in vals) + 40
    s = _panel((w, h), radius=18, alpha=230)
    s.blit(title, ((w - title.get_width()) // 2, 24))
    y = 82
    for k, v in zip(keys, vals):
        s.blit(k, (34, y))
        s.blit(v, (34 + kw + 24, y))
        y += v.get_height() + 6
    hint = font(16).render("Enter / ESC — назад", True, GREY)
    s.blit(hint, ((w - hint.get_width()) // 2, h - 32))
    return s, (screen_w - w) // 2, (screen_h - h) // 2
