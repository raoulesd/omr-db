import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours
import imutils

def grade_score_form(image_path, show_plots=False):
	# Load the image and convert it to grayscale
	image = cv2.imread(image_path)
	if image is None:
		raise FileNotFoundError(f"Could not read image: {image_path}")
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	if not hasattr(cv2, "aruco"):
		raise ImportError("cv2.aruco not found. Install opencv-contrib-python.")

	# Fixed marker IDs by sheet corner
	ID_TL = 4
	ID_TR = 3
	ID_BR = 2
	ID_BL = 1

	# Dictionary used to generate the markers
	ARUCO_DICT = cv2.aruco.getPredefinedDictionary(
		cv2.aruco.DICT_6X6_1000
	)


	def marker_outer_corner(marker_corners_4x2: np.ndarray, which: str) -> np.ndarray:
		"""
		Given the 4 corners of a detected ArUco marker (shape: (4, 2)),
		return the corner that corresponds to the OUTER sheet corner.

		which ∈ {'tl', 'tr', 'br', 'bl'}
		"""
		pts = marker_corners_4x2.astype(np.float32)

		s = pts.sum(axis=1)              # x + y
		d = np.diff(pts, axis=1).ravel() # y - x

		if which == "tl":
			return pts[np.argmin(s)]
		if which == "br":
			return pts[np.argmax(s)]
		if which == "tr":
			return pts[np.argmin(d)]
		if which == "bl":
			return pts[np.argmax(d)]

		raise ValueError("which must be one of: 'tl', 'tr', 'br', 'bl'")

	def aruco_transform(gray):
		"""
		Detect ArUco markers in the given grayscale image and perform a perspective transform to obtain a top-down view of the paper.
		
		:param gray: Grayscale image of the paper containing ArUco markers
		:return: Tuple of (warped_color, warped_gray, corners_list, ids, docCnt)
				- warped_color: Warped color image of the paper
				- warped_gray: Warped grayscale image of the paper
				- corners_list: List of detected marker corners
				- ids: List of detected marker IDs
				- docCnt: Array of the four corners of the paper in the original image
		"""
		try:
			# Newer OpenCV API
			params = cv2.aruco.DetectorParameters()
			detector = cv2.aruco.ArucoDetector(ARUCO_DICT, params)
			corners_list, ids, _ = detector.detectMarkers(gray)
		except AttributeError:
			# Older OpenCV API fallback
			params = cv2.aruco.DetectorParameters_create()
			corners_list, ids, _ = cv2.aruco.detectMarkers(
				gray, ARUCO_DICT, parameters=params
			)

		if ids is None:
			raise RuntimeError(
				"No ArUco markers detected. "
				"Check lighting, print quality, and dictionary."
			)

		ids = ids.flatten().tolist()

		# Map marker ID -> (4, 2) corner array
		id_to_corners = {
			marker_id: corners.reshape(4, 2)
			for corners, marker_id in zip(corners_list, ids)
		}

		# Ensure all required markers are present
		required_ids = [ID_TL, ID_TR, ID_BR, ID_BL]
		missing = [mid for mid in required_ids if mid not in id_to_corners]
		if missing:
			raise RuntimeError(
				f"Missing required ArUco IDs: {missing}. "
				f"Detected IDs: {sorted(ids)}"
			)

		pt_tl = marker_outer_corner(id_to_corners[ID_TL], "tl")
		pt_tr = marker_outer_corner(id_to_corners[ID_TR], "tr")
		pt_br = marker_outer_corner(id_to_corners[ID_BR], "br")
		pt_bl = marker_outer_corner(id_to_corners[ID_BL], "bl")

		docCnt = np.array(
			[pt_tl, pt_tr, pt_br, pt_bl],
			dtype=np.float32
		)

		paper = four_point_transform(image, docCnt)
		warped = four_point_transform(gray, docCnt)
		return paper, warped, corners_list, ids, docCnt

	paper, warped, corners_list, ids, docCnt = aruco_transform(gray)

	def ui_draw_aruco_corners(image, corners_list, ids, docCnt):
		"""
		Draw detected ArUco markers and the document corners on the image. For debugging and UI purposes.
		
		:param image: The input image on which to draw
		:param corners_list: List of detected marker corners
		:param ids: List of detected marker IDs
		:param docCnt: Array of the four corners of the document
		:return: Image with drawn markers and document corners
		"""
		preview = image.copy()
		cv2.aruco.drawDetectedMarkers(
			preview, corners_list, np.array(ids).reshape(-1, 1)
		)

		for (x, y), label in zip(docCnt, ["TL", "TR", "BR", "BL"]):
			cv2.circle(preview, (int(x), int(y)), 10, (0, 0, 255), -1)
			cv2.putText(
				preview,
				label,
				(int(x) + 10, int(y) - 10),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.8,
				(0, 0, 255),
				2
			)
		return preview

	def question_area_transform(warped):
		"""
		Finds the contour of the question area in the warped grayscale image and applies a perspective transform to isolate it. 
		This is useful if the question area is misaligned from the ArUco markers.
		
		:param warped: Warped grayscale image of the paper
		:return: Tuple of (paper, warped, docCnt)
				- paper: Warped color image of the question area
				- warped: Warped grayscale image of the question area
				- docCnt: Array of the four corners of the question area in the warped image
		"""
		image = warped
		gray = image

		blurred = cv2.GaussianBlur(gray, (5, 5), 0)
		edged = cv2.Canny(blurred, 75, 200)


		thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		questionCnts = []
		# loop over the contours
		for c in cnts:
			# compute the bounding box of the contour, then use the
			# bounding box to derive the aspect ratio
			(x, y, w, h) = cv2.boundingRect(c)
			ar = w / float(h)
			# in order to find the question area, the region
			# should be sufficiently wide, sufficiently tall, and
			# have an aspect ratio approximately equal to 0.6
			if w >= 180 and h >= 120 and ar >= 0.6:
				questionCnts.append(c)

		docCnt = None


		# cnts should only have 1 contour, the largest in the image and the area the questions are in
		# this then allows us to find the four corners of the question area for perspective transform
		# this might be handy if the question area is misaligned from the aruco markers
		if len(cnts) > 0:
			cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

			for c in cnts:
				# optional: reject tiny contours early
				if cv2.contourArea(c) < 1000:
					continue

				pts = c.reshape(-1, 2)

				# Tune these bands (pixels). Increase if edges are noisy.
				tol_y = 10

				minY = pts[:, 1].min()
				maxY = pts[:, 1].max()

				top_band = pts[pts[:, 1] <= (minY + tol_y)]
				bot_band = pts[pts[:, 1] >= (maxY - tol_y)]

				# Need enough points to be meaningful
				if len(top_band) < 2 or len(bot_band) < 2:
					continue

				# Top corners: safe from the bottom tail
				TL = top_band[np.argmin(top_band[:, 0])]
				TR = top_band[np.argmax(top_band[:, 0])]

				# Bottom y: robust statistic from bottom band
				y_bottom = int(np.median(bot_band[:, 1]))

				# Bottom corners: DO NOT use bottom_band minX (tail poison)
				BL = np.array([int(TL[0]), y_bottom], dtype=np.int32)
				BR = np.array([int(TR[0]), y_bottom], dtype=np.int32)

				# Build docCnt in the shape OpenCV/imutils code expects: (4,1,2)
				docCnt = np.array([TL, TR, BR, BL], dtype=np.float32).reshape(4, 1, 2)
				break


		# Apply a four point perspective transform to both the
		# original image and grayscale image to obtain a top-down view
		paper = four_point_transform(image, docCnt.reshape(4, 2))
		warped = four_point_transform(gray,  docCnt.reshape(4, 2))
		return paper, warped, docCnt

	paper, warped, question_area_cnt = question_area_transform(warped)

	if show_plots:
		# Show the result of the question area contour detection
		plt.figure(figsize=(8, 10))
		plt.imshow(cv2.cvtColor(paper, cv2.COLOR_BGR2RGB))
		plt.title(f"Detected Question Box outline")
		plt.axis("off")
		plt.show()

	def detect_bubbles(warped):
		"""
		Detects bubble contours in the warped grayscale image of the question area.
		
		:param warped: Warped grayscale image of the question area
		:return: Tuple of (questionCnts, thresh2, warped_u8)
				- questionCnts: List of contours corresponding to detected bubbles
				- thresh2: Binary image used for contour detection (white=ink/pencil)
				- warped_u8: Warped grayscale image normalized to uint8 (0..255)
		"""

		# 1) Threshold (make sure warped is 8-bit single channel)
		if warped.dtype != np.uint8:
			warped_u8 = cv2.normalize(warped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
		else:
			warped_u8 = warped

		# blur helps a lot for pencil texture
		enhanced = cv2.GaussianBlur(warped_u8, (5, 5), 0)


		thresh2 = cv2.adaptiveThreshold(
			enhanced, 255,
			cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
			cv2.THRESH_BINARY_INV,
			31,   # block size (odd): try 21, 31, 51
			7     # C: try 5..15
		)

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

		img_area = h * w

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
			circularity = 4.0 * np.pi * area / (peri * peri)

			# 2) Extent: area / bounding-box-area (letters tend to be lower)
			extent = area / float(cw * ch)

			# 3) Solidity: area / convex-hull-area (Z/T can be less "solid")
			hull = cv2.convexHull(c)
			hull_area = cv2.contourArea(hull)
			if hull_area == 0:
				continue
			solidity = area / float(hull_area)
			

			# ---- Tune these thresholds ----
			# Good starting points for "0"-like blobs:
			if circularity < 0.55:
				continue
			if extent < 0.30:
				continue
			if solidity < 0.80:
				continue

			questionCnts.append(c)
		return questionCnts, thresh2, warped_u8

	questionCnts, thresh2, warped_u8 = detect_bubbles(warped)
		
	def compute_bubble_grid(questionCnts, thresh2, warped_u8):
		"""
		Computes the grid layout of the detected bubble contours by clustering their centroids into ROWS and COLS using K-means.
		Also estimates the median bubble size for later use in scoring.
		
		:param questionCnts: List of contours corresponding to detected bubbles
		:param thresh2: Binary image used for contour detection (white=ink/pencil)
		:param warped_u8: Warped grayscale image normalized to uint8 (0..255)
		"""

		ROWS = 20
		COLS = 27  # 3 groups * 9 columns

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

		# K-means cluster centroids into ROWS and COLS
		ys = np.array([[b["cy"]] for b in bubbles], dtype=np.float32)
		xs = np.array([[b["cx"]] for b in bubbles], dtype=np.float32)

		crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-3)

		# Row centers
		_, row_labels, row_centers = cv2.kmeans(
			ys, ROWS, None, crit, 10, cv2.KMEANS_PP_CENTERS
		)
		row_centers = row_centers.flatten()
		row_order = np.argsort(row_centers)
		row_centers_sorted = row_centers[row_order]

		# Map original row-label -> ordered row index 0..ROWS-1
		row_map = {int(old): int(new) for new, old in enumerate(row_order)}

		# Col centers
		_, col_labels, col_centers = cv2.kmeans(
			xs, COLS, None, crit, 10, cv2.KMEANS_PP_CENTERS
		)
		col_centers = col_centers.flatten()
		col_order = np.argsort(col_centers)
		col_centers_sorted = col_centers[col_order]
		col_map = {int(old): int(new) for new, old in enumerate(col_order)}

		# Attach (row, col) to each bubble
		for i, b in enumerate(bubbles):
			b["row"] = row_map[int(row_labels[i][0])]
			b["col"] = col_map[int(col_labels[i][0])]
		return bubbles, row_centers_sorted, col_centers_sorted, med_w, med_h, crit

	bubbles, row_centers_sorted, col_centers_sorted, med_w, med_h, crit = compute_bubble_grid(questionCnts, thresh2, warped_u8)

	def compute_fill_scores(bubbles, thresh2, warped_u8, med_w, med_h, crit):
		"""
		Computes fill scores for each detected bubble contour to determine how filled each bubble is.
		
		:param bubbles: List of bubble contour dictionaries with grid positions
		:param thresh2: Binary image used for contour detection (white=ink/pencil)
		:param warped_u8: Warped grayscale image normalized to uint8 (0..255)
		:param med_w: Median width of detected bubbles
		:param med_h: Median height of detected bubbles
		:param crit: Termination criteria for k-means clustering
		:return: Tuple containing fill scores and a grid mapping of bubbles
					- fill_scores: 2D array of fill scores for each cell in the grid    
					- overlay: Image with detected bubbles and their fill status drawn for visualization
					- row_centers_sorted: Sorted array of row center y-coordinates
					- col_centers_sorted: Sorted array of column center x-coordinates
					- rad: Estimated radius of the bubbles for visualization

		"""

		# compute "filled" score for each bubble contour
		# thresh2 is white=ink/pencil, measure whiteness inside the bubble
		# To avoid counting the outline ring, erode the contour mask a bit
		scores = []
		grid = {}  # (r,c) -> best bubble dict

		erode_k = max(1, int(round(min(med_w, med_h) * 0.15)))  # ~15% of bubble size
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*erode_k+1, 2*erode_k+1))

		# 1) Enhance dark pencil strokes
		bh_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))  # tune 15..31
		blackhat = cv2.morphologyEx(warped_u8, cv2.MORPH_BLACKHAT, bh_kernel)

		# 2) Combine with original to boost strokes
		enhanced = cv2.add(warped_u8, blackhat)

		# 3) Contrast stretch (optional but helpful)
		enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

		# 4) Blur + adaptive threshold (invert so pencil=white)
		blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
		thresh2 = cv2.adaptiveThreshold(
			blur, 255,
			cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
			cv2.THRESH_BINARY_INV,
			31, 7
		)


		for b in bubbles:
			r, c = b["row"], b["col"]

			# mask for this contour
			mask = np.zeros(thresh2.shape, dtype=np.uint8)
			cv2.drawContours(mask, [b["c"]], -1, 255, -1)

			# erode to focus on interior (reduces outline influence)
			mask_in = cv2.erode(mask, kernel, iterations=1)

			inside = cv2.countNonZero(mask_in)
			if inside == 0:
				continue

			white_inside = cv2.countNonZero(cv2.bitwise_and(thresh2, thresh2, mask=mask_in))
			fill_score = white_inside / float(inside)  # 0..1
			b["fill_score"] = float(fill_score)
			scores.append(fill_score)

			# If multiple contours land in same cell, keep the larger area one
			key = (r, c)
			if key not in grid or b["area"] > grid[key]["area"]:
				grid[key] = b

		if len(scores) < 5:
			raise RuntimeError("Not enough fill scores to determine threshold.")

		scores_np = np.array(scores, dtype=np.float32).reshape(-1, 1)

		# --- 4) Auto-threshold filled vs unfilled using 2-cluster k-means on fill_score
		_, sc_labels, sc_centers = cv2.kmeans(
			scores_np, 2, None, crit, 10, cv2.KMEANS_PP_CENTERS
		)
		sc_centers = sc_centers.flatten()
		filled_cluster = int(np.argmax(sc_centers))  # higher fill_score = filled
		fill_threshold = float(np.mean(sc_centers))  # simple midpoint between cluster centers


		ROWS = 20
		COLS = 27

		overlay = paper.copy()

		# Bubble size estimate (use what you already computed)
		rad = int(round(min(med_w, med_h) * 0.45))          # approximate bubble radius
		pad = int(round(min(med_w, med_h) * 0.45))          # allow spill outside bubble
		r_out = max(4, rad + pad)                           # OUTER radius counts spill
		r_in  = max(2, int(round(rad * 0.35)))              # INNER hole ignore (optional)

		h, w = thresh2.shape[:2]

		def clamp(v, lo, hi):
			return max(lo, min(hi, v))

		fill_scores = np.zeros((ROWS, COLS), dtype=np.float32)

		# Precompute a circular mask for the ROI (square -> circle)
		roi_size = 2 * r_out + 1
		yy, xx = np.ogrid[-r_out:r_out+1, -r_out:r_out+1]
		disk = (xx*xx + yy*yy) <= (r_out*r_out)

		# Optional: ignore the very center (helps if printed “0” outline is thick)
		hole = (xx*xx + yy*yy) <= (r_in*r_in)
		disk_only = disk & (~hole)

		for r in range(ROWS):
			cy = int(round(row_centers_sorted[r]))
			for c in range(COLS):
				cx = int(round(col_centers_sorted[c]))

				x1 = clamp(cx - r_out, 0, w - 1)
				y1 = clamp(cy - r_out, 0, h - 1)
				x2 = clamp(cx + r_out, 0, w - 1)
				y2 = clamp(cy + r_out, 0, h - 1)

				roi = thresh2[y1:y2+1, x1:x2+1]

				# Build matching mask slice (if ROI clipped at edges)
				my1 = (y1 - (cy - r_out))
				mx1 = (x1 - (cx - r_out))
				my2 = my1 + roi.shape[0]
				mx2 = mx1 + roi.shape[1]
				mask = disk_only[my1:my2, mx1:mx2]

				# Score = fraction of white pixels in this padded circular ROI
				denom = float(np.count_nonzero(mask))
				if denom == 0:
					fill_scores[r, c] = 0.0
					continue

				white = float(np.count_nonzero(roi[mask] > 0))
				fill_scores[r, c] = white / denom
		return fill_scores, overlay, row_centers_sorted, col_centers_sorted, rad

	fill_scores, overlay, row_centers_sorted, col_centers_sorted, rad = compute_fill_scores(bubbles, thresh2, warped_u8, med_w, med_h, crit)

	def choose_fill_threshold(scores_1d: np.ndarray, method: str = "percentile") -> float:
		"""
		Different methods to choose a threshold for classifying filled vs unfilled bubbles based on the computed fill scores.
		scores_1d: array-like of fill scores in [0..1]
		method: 'percentile' | 'kmeans' | 'otsu'
		returns: threshold in [0..1] such that score >= threshold => filled
		"""
		s = np.asarray(scores_1d, dtype=np.float32).ravel()
		s = s[np.isfinite(s)]
		if s.size < 10:
			# fallback
			return float(np.mean(s)) if s.size else 1.0

		method = method.lower()

		if method == "percentile":
			low = float(np.percentile(s, 20))
			high = float(np.percentile(s, 95))
			# tune multiplier if needed
			return low + 0.20 * (high - low)

		if method == "kmeans":
			# 2-cluster kmeans on 1D scores
			samples = s.reshape(-1, 1)
			crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-4)
			_, labels, centers = cv2.kmeans(samples, 2, None, crit, 10, cv2.KMEANS_PP_CENTERS)
			centers = centers.flatten()
			# midpoint between the two centers is a good threshold
			centers.sort()
			return float((centers[0] + centers[1]) / 2.0)

		if method == "otsu":
			# Otsu expects 8-bit; scale [0..1] -> [0..255]
			s8 = np.clip(s * 255.0, 0, 255).astype(np.uint8)
			# Otsu threshold on a "fake image" vector
			t, _ = cv2.threshold(s8.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
			return float(t / 255.0)

		raise ValueError("method must be one of: 'percentile', 'kmeans', 'otsu'")

	def classify_bubble_fill(fill_scores, row_centers_sorted, col_centers_sorted, overlay, rad):
		all_scores = fill_scores.ravel()

		ROWS = fill_scores.shape[0]
		COLS = fill_scores.shape[1]

		# Choose one:
		FILL_METHOD = "kmeans"   # "percentile" or "kmeans" or "otsu"
		fill_threshold = choose_fill_threshold(all_scores, method=FILL_METHOD)

		print(f"Fill threshold method: {FILL_METHOD} | threshold={fill_threshold:.3f}")



		filled_cells = []
		for r in range(ROWS):
			for c in range(COLS):
				score = float(fill_scores[r, c])
				cx = int(round(col_centers_sorted[c]))
				cy = int(round(row_centers_sorted[r]))

				if score >= fill_threshold:
					filled_cells.append((r, c, score))
					cv2.circle(overlay, (cx, cy), rad, (0, 0, 255), 2)  # red = filled
					cv2.circle(overlay, (cx, cy), 2, (0, 0, 255), -1)
				else:
					# optional: show unfilled in light green
					cv2.circle(overlay, (cx, cy), rad, (0, 255, 0), 1)

		print(f"Threshold: {fill_threshold:.3f}")
		print(f"Filled count: {len(filled_cells)} / {ROWS*COLS}")
		return filled_cells, (ROWS, COLS)

	filled_cells, (ROWS, COLS) = classify_bubble_fill(fill_scores, row_centers_sorted, col_centers_sorted, overlay, rad)

	if show_plots:
		plt.figure(figsize=(10, 12))
		plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
		plt.title(f"Filled bubbles (red) | Filled={len(filled_cells)}")
		plt.axis("off")
		plt.show()

	return filled_cells, (ROWS, COLS)