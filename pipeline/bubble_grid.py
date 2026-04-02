import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import configs.config as config

def compute_bubble_grid(questionCnts, bubble_area_image, debug_steps=None):
	"""
	Computes the grid layout of the detected bubble contours by clustering their centroids into ROWS and COLS using K-means.
	Also estimates the median bubble size for later use in scoring.
	
	:param questionCnts: List of contours corresponding to detected bubbles
	:param bubble_area_image: Image of the bubble area normalized to uint8 (0..255)

	:return: Tuple of (bubbles, row_centers_sorted, col_centers_sorted, median_bubble_size)
			
		- bubbles: List of dicts with keys: c (contour), cx, cy, w, h, area, row, col
		- row_centers_sorted: Sorted array of row center positions
		- col_centers_sorted: Sorted array of column center positions
		- median_bubble_size: Tuple of (median_width, median_height)
	"""

	rows = config.get_property("num_boulders")
	cols = config.get_property("num_attempts") * config.get_property("num_answers")

	# 3 groups * 9 columns

	# --- 1) Compute centroids + basic size estimate
	bubbles = []
	for c in questionCnts:
		area = cv2.contourArea(c)
		if area <= 0:
			continue
		M = cv2.moments(c)
		if M["m00"] == 0:
			continue
		cx = M["m10"] / M["m00"]
		cy = M["m01"] / M["m00"]
		x, y, w, h = cv2.boundingRect(c)
		bubbles.append({
			"c": c,
			"cx": float(cx),
			"cy": float(cy),
			"w": int(w),
			"h": int(h),
			"area": float(area),
		})

	if len(bubbles) < 10:
		raise RuntimeError("Too few bubble contours found to infer a grid.")

	med_w = int(np.median([b["w"] for b in bubbles]))
	med_h = int(np.median([b["h"] for b in bubbles]))

	median_bubble_size = (med_w, med_h)

	# K-means cluster centroids into ROWS and COLS
	ys = np.array([[b["cy"]] for b in bubbles], dtype=np.float32)
	xs = np.array([[b["cx"]] for b in bubbles], dtype=np.float32)

	crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-3)

	# Row centers
	_, row_labels, row_centers = cv2.kmeans(
		ys, rows, None, crit, 10, cv2.KMEANS_PP_CENTERS
	)
	row_centers = row_centers.flatten()
	row_order = np.argsort(row_centers)
	row_centers_sorted = row_centers[row_order]

	# Map original row-label -> ordered row index 0..ROWS-1
	row_map = {int(old): int(new) for new, old in enumerate(row_order)}

	# Col centers
	_, col_labels, col_centers = cv2.kmeans(
		xs, cols, None, crit, 10, cv2.KMEANS_PP_CENTERS
	)
	col_centers = col_centers.flatten()
	col_order = np.argsort(col_centers)
	col_centers_sorted = col_centers[col_order]
	col_map = {int(old): int(new) for new, old in enumerate(col_order)}

	# Attach (row, col) to each bubble
	for i, b in enumerate(bubbles):
		b["row"] = row_map[int(row_labels[i][0])]
		b["col"] = col_map[int(col_labels[i][0])]

	if debug_steps is not None:
		overlay = cv2.cvtColor(bubble_area_image.copy(), cv2.COLOR_GRAY2BGR)
		for y in row_centers_sorted:
			cv2.line(overlay, (0, int(y)), (overlay.shape[1], int(y)), (0, 255, 0), 1)
		for x in col_centers_sorted:
			cv2.line(overlay, (int(x), 0), (int(x), overlay.shape[0]), (255, 0, 0), 1)
		debug_steps.append(("Bubble Grid - Row/Col Centers", overlay))
	return bubbles, row_centers_sorted, col_centers_sorted, median_bubble_size




def detect_bubbles(bubble_area_image, debug_steps=None):
	"""
	Detects bubbles in the warped grayscale image
	
	:param rectified_cropped_bubble_area: Warped grayscale image of the question area
	:return: Tuple of (questionCnts, bubble_area_image)
			- questionCnts: List of contours corresponding to detected bubbles
	"""

	circularity_min = config.get_property("circularity")
	extent_min = config.get_property("extent")
	hull_min = config.get_property("hull")
	debug_mode = config.get_property("debug_mode")

	# 1) Threshold (make sure bubble_area_image_u8 is 8-bit single channel)
	if bubble_area_image.dtype != np.uint8:
		bubble_area_image = cv2.normalize(bubble_area_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

	# bubble_area_image must be single-channel grayscale for the thresholding and contour steps
	assert len(bubble_area_image.shape) == 2, "Bubble area image must be single-channel grayscale"

	# blur helps a lot for pencil texture
	enhanced = cv2.GaussianBlur(bubble_area_image, (5, 5), 0)


	thresh2 = cv2.adaptiveThreshold(
		enhanced, 255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY_INV,
		31,   # block size (odd): try 21, 31, 51
		7     # C: try 5..15
	)

	if debug_steps is not None:
		debug_steps.append(("Bubble Detection - Adaptive Threshold", cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR)))

	thresh2 = cv2.morphologyEx(
		thresh2, cv2.MORPH_CLOSE,
		cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
		iterations=1
	)


	# 2) Remove small specks + reduce accidental connections
	#    (tune kernel sizes if needed)
	thresh2 = cv2.morphologyEx(
		thresh2, cv2.MORPH_OPEN,
		cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
		iterations=1
	)
	4
	# Optional: if the page border is thick / present, clear a small margin
	# so it cannot appear as one giant contour
	h, w = thresh2.shape[:2]
	margin = 5
	thresh2[:margin, :] = 0
	thresh2[-margin:, :] = 0
	thresh2[:, :margin] = 0
	thresh2[:, -margin:] = 0

	# 3) Find contours: RETR_LIST or RETR_TREE to get internal contours

	cnts_info = cv2.findContours(thresh2.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

	cnts = imutils.grab_contours(cnts_info)

	hierarchy = cnts_info[1] if len(cnts_info) == 2 else cnts_info[2]

	hierarchy = hierarchy[0]  # (N, 4): [next, prev, first_child, parent]

	questionCnts = []

	for i, c in enumerate(cnts):

		# Keep ONLY outer contours (parent == -1), discard inner hole contours

		if hierarchy[i][3] != -1:

			continue

		area = cv2.contourArea(c)
		if area < 30:
			continue

		x, y, cw, ch = cv2.boundingRect(c)
		ar = cw / float(ch)

		# keep your size/aspect gate (tune)
		if cw < 6 or ch < 9 or not (0.6 <= ar <= 2.0):
			continue

		peri = cv2.arcLength(c, True)
		if peri == 0:
			continue

		# 1) Circularity: 1.0 is a perfect circle
		circularity_score = 4.0 * np.pi * area / (peri * peri)

		# 2) Extent: area / bounding-box-area (letters tend to be lower)
		extent_score = area / float(cw * ch)

		# 3) Solidity: area / convex-hull-area (Z/T can be less "solid")
		hull_contour = cv2.convexHull(c)
		hull_area = cv2.contourArea(hull_contour)
		if hull_area == 0:
			continue
		solidity = area / float(hull_area)
		

		# ---- Tune these thresholds ----
		# Good starting points for "0"-like blobs:
		if circularity_score < circularity_min:
			continue
		if extent_score < extent_min:
			continue
		if solidity < hull_min:
			continue

		questionCnts.append(c)

	if debug_mode:
		print(f"Detected bubble-like contours: {len(questionCnts)}")

	if debug_steps is not None:
		overlay = cv2.cvtColor(bubble_area_image.copy(), cv2.COLOR_GRAY2BGR)
		cv2.drawContours(overlay, questionCnts, -1, (0, 0, 255), 1)
		debug_steps.append(("Bubble Detection - Filtered Contours", overlay))

	return questionCnts





def plot_bubble_grid(paper, row_centers, col_centers):
	# Show the result of the grid detection by drawing a vertical line for each column and a horizontal line for each row
	paper = paper.copy()
	for row in row_centers:
		cv2.line(paper, (0, int(row)), (paper.shape[1], int(row)), (0, 255, 0), 1)
	for col in col_centers:
		cv2.line(paper, (int(col), 0), (int(col), paper.shape[0]), (0, 255, 0), 1)

	plt.figure(figsize=(8, 10))
	plt.imshow(cv2.cvtColor(paper, cv2.COLOR_BGR2RGB))
	plt.title("Detected Bubble Grid")
	plt.axis("off")
	plt.show()