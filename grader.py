import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import configs.config as config

import pipeline.preprocess_paper as preprocess_paper
import pipeline.bubble_grid as bubble_grid
import pipeline.find_filled_bubbles as find_filled_bubbles


class GradingDebugError(RuntimeError):
	def __init__(self, message, debug_steps=None):
		super().__init__(message)
		self.debug_steps = debug_steps if debug_steps is not None else []


def _append_debug_step(debug_steps, title, image):
	if debug_steps is None or image is None:
		return

	img = image.copy()
	if len(img.shape) == 2:
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	debug_steps.append((title, img))

def plot_paper(paper, title):
	plt.figure(figsize=(8, 10))
	plt.imshow(cv2.cvtColor(paper, cv2.COLOR_BGR2RGB))
	plt.title(title)
	plt.axis("off")
	plt.show()

def get_amounts_and_tries(cell_data):

	amountZT = [0, 0]
	triesZT = [0, 0]
	per_boulder_ZT = []
	num_rows = cell_data.shape[0]
	num_cols = cell_data.shape[1]
	for r in range(num_rows):
		zone_attempts = None
		top_attempts = None
		for (attempt_number, c) in enumerate(range(0, num_cols, 3)):
			is_top = cell_data[r, c+2] == 1
			is_zone = cell_data[r, c+1] == 1 or is_top
			is_attempt = cell_data[r, c+0] == 1 or is_top or is_zone

			if is_attempt:
				if is_zone and zone_attempts is None:
					zone_attempts = attempt_number + 1
				if is_top and top_attempts is None:
					top_attempts = attempt_number + 1

		amountZT[0] += 1 if zone_attempts is not None else 0
		amountZT[1] += 1 if top_attempts is not None else 0

		triesZT[0] += zone_attempts if zone_attempts is not None else 0
		triesZT[1] += top_attempts if top_attempts is not None else 0

		per_boulder_ZT.append((zone_attempts, top_attempts))

	return amountZT, triesZT, per_boulder_ZT


		

	

def grade_score_form(image_path, show_plots=False, debug_mode=False, return_debug_steps=False):
	debug_steps = [] if debug_mode else None

	# Load the image and convert it to grayscale
	image = cv2.imread(image_path)
	if image is None:
		raise FileNotFoundError(f"Could not read image: {image_path}")
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	_append_debug_step(debug_steps, "01 - Original Input", image)
	_append_debug_step(debug_steps, "02 - Input Grayscale", gray)
	if not hasattr(cv2, "aruco"):
		raise ImportError("cv2.aruco not found. Install opencv-contrib-python.")

	try:
		paper, warped = preprocess_paper.preprocess(image, gray, debug_steps=debug_steps)
		_append_debug_step(debug_steps, "03 - Preprocess Output (Paper)", paper)
		_append_debug_step(debug_steps, "04 - Preprocess Output (Warped)", warped)

		if show_plots:
			# Show the result of the question area contour detection
			plt.figure(figsize=(8, 10))
			plt.imshow(cv2.cvtColor(paper, cv2.COLOR_BGR2RGB))
			plt.title(f"Detected Question Box outline")
			plt.axis("off")
			plt.show()

		questionCnts, thresh2, warped_u8 = bubble_grid.detect_bubbles(warped, debug_steps=debug_steps)
		_append_debug_step(debug_steps, "05 - Bubble Threshold", thresh2)
		_append_debug_step(debug_steps, "06 - Warped U8", warped_u8)

		bubbles, row_centers_sorted, col_centers_sorted, med_w, med_h, crit = bubble_grid.compute_bubble_grid(questionCnts, thresh2, warped_u8, debug_steps=debug_steps)

		if show_plots:
			bubble_grid.plot_bubble_grid(paper, bubbles, row_centers_sorted, col_centers_sorted, med_w, med_h, warped_u8)

		filled_cells = find_filled_bubbles.find_filled_bubbles_alt(
			bubbles,
			row_centers_sorted,
			col_centers_sorted,
			thresh2,
			warped_u8,
			med_w,
			med_h,
			crit,
			debug_steps=debug_steps,
		)

		if debug_mode:
			final_overlay = warped.copy()
			if len(final_overlay.shape) == 2:
				final_overlay = cv2.cvtColor(final_overlay, cv2.COLOR_GRAY2BGR)
			for (r, c) in filled_cells:
				x = int(col_centers_sorted[c])
				y = int(row_centers_sorted[r])
				cv2.circle(final_overlay, (x, y), max(2, int(med_w * 0.8)), (0, 0, 255), 2)
			_append_debug_step(debug_steps, "99 - Final Filled Bubbles Overlay", final_overlay)
	except Exception as e:
		if debug_mode:
			raise GradingDebugError(str(e), debug_steps=debug_steps) from e
		raise

	result = (filled_cells, (ROWS, COLS), warped_u8, (row_centers_sorted, col_centers_sorted), (med_w, med_h), image)
	if return_debug_steps:
		return (*result, debug_steps)
	return result