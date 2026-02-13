import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours
import imutils

import pipeline.preprocess_paper as preprocess_paper
import pipeline.bubble_grid as bubble_grid
import pipeline.find_filled_bubbles as find_filled_bubbles

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


		

	

def grade_score_form(image_path, show_plots=False):
	# Load the image and convert it to grayscale
	image = cv2.imread(image_path)
	if image is None:
		raise FileNotFoundError(f"Could not read image: {image_path}")
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	if not hasattr(cv2, "aruco"):
		raise ImportError("cv2.aruco not found. Install opencv-contrib-python.")

	paper, warped = preprocess_paper.preprocess(image, gray)

	if show_plots:
		# Show the result of the question area contour detection
		plt.figure(figsize=(8, 10))
		plt.imshow(cv2.cvtColor(paper, cv2.COLOR_BGR2RGB))
		plt.title(f"Detected Question Box outline")
		plt.axis("off")
		plt.show()


	questionCnts, thresh2, warped_u8 = bubble_grid.detect_bubbles(warped)

	bubbles, row_centers_sorted, col_centers_sorted, med_w, med_h, crit = bubble_grid.compute_bubble_grid(questionCnts, thresh2, warped_u8)

	if show_plots:
		bubble_grid.plot_bubble_grid(paper, bubbles, row_centers_sorted, col_centers_sorted, med_w, med_h, warped_u8)

	filled_cells, (ROWS, COLS) = find_filled_bubbles.find_filled_bubbles_alt(bubbles, row_centers_sorted, col_centers_sorted, thresh2, warped_u8, med_w, med_h, crit)

	return filled_cells, (ROWS, COLS), warped_u8, (row_centers_sorted, col_centers_sorted), (med_w, med_h), image