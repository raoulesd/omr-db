import importlib
import os

import cv2
import numpy as np

# Shared grid dimensions across bubble detection and fill classification.
ROWS = 20
COLS = 27

# Bubble contour filtering thresholds.
circularity = 0.55
extent = 0.30
hull = 0.80

# General debug toggle used across pipeline modules.
debug_mode = False

# Iterative threshold convergence tolerance.
epsilon = 0.001

# Fill classification method: "percentile", "kmeans", or "otsu".
FILL_METHOD = "kmeans"

# Fixed marker IDs by sheet corner.
ID_TL = 4
ID_TR = 3
ID_BR = 2
ID_BL = 1

# Dictionary used to detect ArUco markers on forms.
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)

# Backward-compatible alias for existing naming in earlier code.
ARUDO_DICT = ARUCO_DICT

# Corner offsets for perspective transform fine-tuning.
offset_tl = np.array([0, 0], dtype=np.float32)
offset_tr = np.array([0, 0], dtype=np.float32)
offset_br = np.array([0, 0], dtype=np.float32)
offset_bl = np.array([0, 0], dtype=np.float32)

# Whether to crop to the bounded question area after ArUco transform.
has_bounded_question_area = True


_REQUIRED_KEYS = (
    "ROWS",
    "COLS",
    "circularity",
    "extent",
    "hull",
    "debug_mode",
    "epsilon",
    "FILL_METHOD",
    "ID_TL",
    "ID_TR",
    "ID_BR",
    "ID_BL",
    "ARUDO_DICT",
    "offset_tl",
    "offset_tr",
    "offset_br",
    "offset_bl",
    "has_bounded_question_area",
)

_ACTIVE_CONFIG = None
_ACTIVE_CONFIG_NAME = None


def _normalize_config_name(config_name):
    if not config_name:
        return "config"

    name = config_name.strip()
    if name.endswith(".py"):
        name = name[:-3]
    name = name.replace("\\", ".").replace("/", ".")
    return name or "config"


def _validate_config_module(config_module, config_name):
    missing = [key for key in _REQUIRED_KEYS if not hasattr(config_module, key)]
    if missing:
        raise ValueError(
            f"Config '{config_name}' is missing required keys: {', '.join(missing)}"
        )


def set_active_config(config_name="config"):
    global _ACTIVE_CONFIG
    global _ACTIVE_CONFIG_NAME

    module_name = _normalize_config_name(config_name)
    config_module = importlib.import_module(module_name)
    _validate_config_module(config_module, module_name)

    _ACTIVE_CONFIG = config_module
    _ACTIVE_CONFIG_NAME = module_name
    return _ACTIVE_CONFIG


def get_active_config():
    global _ACTIVE_CONFIG
    if _ACTIVE_CONFIG is None:
        default_name = os.getenv("OMR_CONFIG_NAME", "config")
        return set_active_config(default_name)
    return _ACTIVE_CONFIG


def get_active_config_name():
    if _ACTIVE_CONFIG_NAME is None:
        get_active_config()
    return _ACTIVE_CONFIG_NAME
