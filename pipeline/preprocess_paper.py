"""preprocess_paper.py contains functions for preprocessing the input image of the paper, including converting it to grayscale, detecting ARUCO markers to find the corners of the paper, and rotating the image to ensure the paper is level. The main function is preprocess, which takes an input image and returns a rotated image along with the detected ARUCO marker positions for later use in warping and scoring.
"""

import cv2
import numpy as np
from pipeline import aruco_handler
from pipeline import region_extractor
import matplotlib.pyplot as plt


# Preprocess the paper by flattening it
def preprocess(image, debug_steps=None):
	"""Preprocess the input image to make sure it's level
	"""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	aruco_markers = aruco_handler.detect_aruco_markers(gray)


	if aruco_markers is None:
		print("Warning: Cannot find ARUCO markers in the full page texture data. The extracted regions may be misaligned. Attempting to extract regions using fallback method with default marker positions from config and rotation correction for the attempt_score region.")
		bubble_region_rotated = region_extractor.extract_region(image, "attempt_score")

		# Find the rotation angle of the bubble region and correct it
		full_page_angle = find_grid_rotation(bubble_region_rotated, max_angle_deg=10.0, debug_prints=False)

		# Rotate the whole page based on the detected angle and then extract the regions again using the rotated page
		rotated_image = rotate_image(image, full_page_angle)
		return rotated_image, None

	# Rotate the aruco markers to ensure they are in the correct orientation
	rotated_image, aruco_markers = aruco_handler.rotate_image_to_flatten_aruco_markers(image, aruco_markers)

	return rotated_image, aruco_markers


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
	"""Rotates the input image by the specified angle in degrees.

	:param image: The input image as a NumPy array.
	:param angle_deg: The angle to rotate the image, in degrees. Positive values rotate counter-clockwise.
	:return: The rotated image as a NumPy array.
	"""
	h, w = image.shape[:2]
	cx, cy = w / 2.0, h / 2.0
	rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle=angle_deg, scale=1.0)
	rotated = cv2.warpAffine(
		image, rotation_matrix, (w, h),
		flags=cv2.INTER_LINEAR,
		borderMode=cv2.BORDER_REPLICATE,
	)
	return rotated

def find_grid_rotation(
	image: np.ndarray,
	max_angle_deg: float = 10.0,
	debug_prints: bool = False,
) -> tuple[np.ndarray, float]:

	if not (0 < max_angle_deg < 90):
		raise ValueError("max_angle_deg must be in the open interval (0, 90).")

	is_grayscale_input = image.ndim == 2

	if is_grayscale_input:
		gray = image
	else:
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Edge detection using Canny after applying a bilateral filter to reduce noise while preserving edges.
	blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
	edges = cv2.Canny(blurred, threshold1=30, threshold2=100, apertureSize=3)

	# Use Hough Transform to detect lines in the edge-detected image. The parameters are chosen to be adaptive based on the image size.
	h, w = gray.shape
	min_line_len = max(20, min(h, w) // 20)   # at least 5 % of short side
	max_line_gap = max(5, min_line_len // 4)

	lines = cv2.HoughLinesP(
		edges,
		rho=1,
		theta=(np.pi / 180) / 4,
		threshold=50,
		minLineLength=min_line_len,
		maxLineGap=max_line_gap,
	)

	if debug_prints:
		# Draw the detected lines on a copy of the image for visualization.
		line_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if is_grayscale_input else image.copy()
		if lines is not None:
			for line in lines:
				x1, y1, x2, y2 = line[0]

				line_length = np.hypot(x2 - x1, y2 - y1)
				if line_length < 200:
					continue  # skip short lines

				cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

		print("Edge-detected image with detected lines (red):")
		plt.figure(figsize=(8, 8))
		plt.imshow(line_image)
		plt.title("Detected Lines for Rotation Correction")
		plt.show()

	if lines is None or len(lines) == 0:
		# No lines found – return original unchanged.
		if debug_prints:
			print("No lines detected for rotation correction. Returning original image.")
		return 0.0


	angles: list[float] = []

	if debug_prints:
		print(f"Detected {len(lines)} lines for rotation correction.")

	for line in lines:
		x1, y1, x2, y2 = line[0]
		if x1 == x2 and y1 == y2:
			continue  # degenerate point

		if y1 == y2:
			continue  # perfectly horizontal line, no angle contribution

		line_length = np.hypot(x2 - x1, y2 - y1)
		if line_length < 200:
			continue  # skip short lines

		angle_rad = np.arctan2(float(y2 - y1), float(x2 - x1))
		angle_deg = np.degrees(angle_rad)

		# Normalize angle to range [-90, 90]
		if angle_deg < -90:
			angle_deg += 180
		elif angle_deg > 90:
			angle_deg -= 180

		angles.append(angle_deg)

	if not angles:
		return 0.0

	# Use the median angle to be robust against outliers.
	detected_angle = float(np.median(angles))

	if debug_prints:
		print(f"Detected angles (degrees): {angles}")
		print(f"Median detected angle for correction: {detected_angle:.2f} degrees")

	# If the detected angle is larger than the maximum allowed, return the original image without correction.
	if abs(detected_angle) > max_angle_deg:
		if debug_prints:
			print(f"Detected angle {detected_angle:.2f} exceeds max_angle_deg of {max_angle_deg}. No rotation applied.")
		return 0.0
	
	return detected_angle

