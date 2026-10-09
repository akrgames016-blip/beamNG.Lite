"""Гараж: автомобили описываются JSON-конфигами в каталоге vehicles/.

Чтобы добавить машину, достаточно дописать один JSON: физика (vehicle.py),
шкура кузова (body.py) и рендер (render.py) читают всё из конфига.
"""
import json
import os

import numpy as np

VEHICLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vehicles")

DEFAULT_ID = "sedan"

# Ключи, которые обязаны быть в каждом конфиге
_REQUIRED = ("id", "name", "sections", "wheel_pos", "wheel_r", "wheel_w",
             "wheel_mass", "body_mass", "gears", "final_drive", "torque")

# Конфиг по умолчанию — классический седан (фолбэк, если каталог пуст).
DEFAULT = {
    "id": "sedan",
    "name": "Седан",
    "description": "Классический soft-body седан: 1080 кг, задний привод, 5 передач.",
    "color": [0.80, 0.13, 0.11],
    "sections": [
        [-2.05, 0.66, 0.46, 0.92],
        [-1.35, 0.80, 0.42, 1.10],
        [-0.55, 0.82, 0.42, 1.43],
        [0.50, 0.82, 0.42, 1.45],
        [1.35, 0.78, 0.42, 1.02],
        [2.05, 0.66, 0.46, 0.90],
    ],
    "belt_y": 0.97,
    "wheel_pos": [[-0.88, -1.45], [0.88, -1.45], [-0.88, 1.45], [0.88, 1.45]],
    "wheel_r": 0.34,
    "wheel_w": 0.115,
    "wheel_mass": 26.0,
    "body_mass": 1080.0,
    "driven": "rear",
    "max_steer_deg": 34.0,
    "gears": [-3.2, 0.0, 3.20, 2.05, 1.45, 1.10, 0.85],
    "final_drive": 3.5,
    "idle_rpm": 900.0,
    "max_rpm": 7000.0,
    "redline_rpm": 6000.0,
    "engine_inertia": 0.24,
    "lsd": 140.0,
    "tire_mu": 1.32,
    "torque": {"base": 130.0, "peak": 175.0, "peak_rpm": 4300.0, "width": 2200.0},
    "aero": {"cd": 0.62, "area": 2.1},
    "camera_dist": 6.6,
}


def _normalize(cfg, source):
    """Проверка обязательных ключей и приведение типов."""
    missing = [k for k in _REQUIRED if k not in cfg]
    if missing:
        raise ValueError("%s: нет ключей: %s" % (source, ", ".join(missing)))
    out = dict(cfg)
    out["color"] = np.clip(np.asarray(cfg.get("color", DEFAULT["color"]),
                                      dtype=np.float64), 0.0, 1.0)
    out["sections"] = [tuple(map(float, s)) for s in cfg["sections"]]
    out["wheel_pos"] = [tuple(map(float, p)) for p in cfg["wheel_pos"]]
    out["gears"] = [float(g) for g in cfg["gears"]]
    out["belt_y"] = float(cfg.get("belt_y", DEFAULT["belt_y"]))
    out["driven"] = cfg.get("driven", "rear")
    if out["driven"] not in ("rear", "front", "all"):
        raise ValueError("%s: driven должен быть rear/front/all" % source)
    if source != "<default>":
        want = os.path.splitext(os.path.basename(source))[0]
        if out["id"] != want:
            raise ValueError("%s: id %r не совпадает с именем файла %r"
                             % (source, out["id"], want))
    return out


def load(vehicle_id):
    """Загрузить конфиг машины по id (имя JSON-файла без расширения)."""
    if vehicle_id == DEFAULT_ID and not os.path.isdir(VEHICLES_DIR):
        return _normalize(dict(DEFAULT), "<default>")
    path = os.path.join(VEHICLES_DIR, vehicle_id + ".json")
    if not os.path.isfile(path):
        raise KeyError("машина %r не найдена (нет файла %s)" % (vehicle_id, path))
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    return _normalize(cfg, path)


def load_all():
    """Все машины гаража: {id: cfg}, порядок — по имени файла."""
    out = {}
    if os.path.isdir(VEHICLES_DIR):
        for fn in sorted(os.listdir(VEHICLES_DIR)):
            if fn.endswith(".json"):
                cfg = load(fn[:-5])
                out[cfg["id"]] = cfg
    if DEFAULT_ID not in out:
        out[DEFAULT_ID] = _normalize(dict(DEFAULT), "<default>")
    return out


def ids():
    """Список id всех машин гаража."""
    return list(load_all().keys())
