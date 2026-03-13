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
	"name": (0.38, 0.85, 0.05, 0.16),
	"tickbox": (0.80, 1.00, 0.22, 0.50),
	"attempts_total": (0.80, 1.00, 0.67, 0.74),
}

# UI scaling and preview panel sizes.
UI_SCALE = 1.2
FRAME_WIDTH = 800
FRAME_HEIGHT = 455
ATTEMPT_TOTALS_HEIGHT = 100
ZONES_AND_TOPS_WIDTH = 180
NAME_DATA_WIDTH = 600
NAME_DATA_HEIGHT = 180

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
