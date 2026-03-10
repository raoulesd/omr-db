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
