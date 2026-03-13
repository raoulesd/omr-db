import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# Path to the directory holding scanned files.
SCANNED_FILES_DIR = BASE_DIR / "process_data" / "to_process"

# Processing output locations.
PROCESSED_FILES_DIR = BASE_DIR / "process_data" / "processed"
ERRORED_FILES_DIR = BASE_DIR / "process_data" / "errored"
RESULTS_CSV_PATH = BASE_DIR / "results.csv"

# Relative coordinates (x_min, x_max, y_min, y_max) for UI cutouts.
UI_AREAS = {
	"name": (0.12, 0.64, 0.02, 0.078),
	"tickbox": (0.78, 1.00, 0.125, 0.805),
	"attempts_total": (0.65, 1.00, 0.80, 0.90),
}

# UI scaling factor - controls the ratio of original image size to displayed size.
UI_SCALE = 0.25

# Base (unscaled) form dimensions - the reference size before applying UI_SCALE.
_BASE_FRAME_WIDTH = 2486
_BASE_FRAME_HEIGHT = 3520

# Apply scale to get actual display dimensions.
FRAME_WIDTH = int(_BASE_FRAME_WIDTH * UI_SCALE)
FRAME_HEIGHT = int(_BASE_FRAME_HEIGHT * UI_SCALE)

# Derive panel dimensions from UI_AREAS ratios (automatically computed from config).
_name_coords = UI_AREAS["name"]
NAME_DATA_WIDTH = int((_name_coords[1] - _name_coords[0]) * FRAME_WIDTH)
NAME_DATA_HEIGHT = int((_name_coords[3] - _name_coords[2]) * FRAME_HEIGHT)

_tickbox_coords = UI_AREAS["tickbox"]
ZONES_AND_TOPS_WIDTH = int((_tickbox_coords[1] - _tickbox_coords[0]) * FRAME_WIDTH)
ZONES_AND_TOPS_HEIGHT = int((_tickbox_coords[3] - _tickbox_coords[2]) * FRAME_HEIGHT)

_attempts_coords = UI_AREAS["attempts_total"]
ATTEMPT_TOTALS_WIDTH = int((_attempts_coords[1] - _attempts_coords[0]) * FRAME_WIDTH)
ATTEMPT_TOTALS_HEIGHT = int((_attempts_coords[3] - _attempts_coords[2]) * FRAME_HEIGHT)

# Shared grid dimensions across bubble detection and fill classification.
ROWS = 30
COLS = 15

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

# Optional relative cutout of the detected page quad (x_min, x_max, y_min, y_max).
# When set, this is converted to dynamic per-corner offsets at runtime.
# Example: (0.04, 0.9501, 0.126, 0.805)
CORNER_CUTOUT = (0.064, 0.82, 0.043, 0.83)
# CORNER_CUTOUT = (0, 0, 0, 0)

# Whether to crop to the bounded question area after ArUco transform.
has_bounded_question_area = False
