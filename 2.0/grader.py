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

def grade_score_form(image_path, show_plots=False):
	# Load the image and convert it to grayscale
	image = cv2.imread(image_path)
	if image is None:
		raise FileNotFoundError(f"Could not read image: {image_path}")
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	if not hasattr(cv2, "aruco"):
		raise ImportError("cv2.aruco not found. Install opencv-contrib-python.")

	paper, warped, question_area_cnt = preprocess_paper.preprocess(image, gray)

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

	return filled_cells, (ROWS, COLS)