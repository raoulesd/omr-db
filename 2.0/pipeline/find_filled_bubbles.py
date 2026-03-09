import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy

def plot_paper(paper, title):
	plt.figure(figsize=(8, 10))
	plt.imshow(cv2.cvtColor(paper, cv2.COLOR_BGR2RGB))
	plt.title(title)
	plt.axis("off")
	plt.show()

def plot_paper_gray(paper, title):
	plt.figure(figsize=(8, 10))
	plt.imshow(paper, cmap='gray', vmin=0, vmax=255)
	plt.title(title)
	plt.axis("off")
	plt.show()

def isodata_threshold(values):
	epsilon = 0.001

	threshold = np.mean(values)

	while True:

		mean_left = np.mean(values[values<=threshold])
		mean_right = np.mean(values[values>threshold])

		new_threshold = (mean_left + mean_right) / 2
		if np.abs(threshold - new_threshold) < epsilon:
			break
		threshold = new_threshold

	return threshold

def savgol_threshold(values, window_length=13, polyorder=3, multiplier=1.0):

	isodata_t = int(np.round(isodata_threshold(values)))

	values_histogram = np.histogram(values, bins=255, range=(0, 255))
	bins = values_histogram[1][:-1]
	counts = values_histogram[0]

	# Smooth the histogram counts using Savitzky-Golay filter
	smooth_counts = scipy.signal.savgol_filter(counts, window_length=window_length, polyorder=polyorder)
	#width = 7
	#y = np.convolve(counts, np.ones(width)/width, mode='same')  # simple moving average for comparison
	y = smooth_counts
	threshold = np.argmin(y[isodata_t:isodata_t+15]) + isodata_t

	#print(f"Isodata threshold: {isodata_t:.2f} | Savgol threshold: {threshold:.2f}")


	# plot the histogram and smoothed curve
	# plt.figure(figsize=(8, 6))
	# plt.bar(bins, counts, width=1, label='Histogram')
	# #plt.plot(bins, smooth_counts, color='red', label='Smoothed')
	# plt.plot(bins, y, color='blue', label='y')
	# # show the threshold
	# plt.axvline(x=isodata_t, color='green', linestyle='--', label=f'Isodata Threshold ({isodata_t:.2f})')
	# plt.axvline(x=threshold, color='orange', linestyle='--', label=f'Savgol Threshold ({threshold:.2f})')
	# plt.title("Histogram of Values with Savitzky-Golay Smoothing")
	# plt.xlabel("Value")
	# plt.ylabel("Frequency")
	# plt.legend()
	# plt.show()
	return threshold

def diff_with_offset(img1, img2, offset_x, offset_y):
	img2_offset = img2.copy()
	if offset_y > 0:
		img2_offset = np.pad(img2_offset, ((offset_y, 0), (0, 0)), mode='constant')[:-offset_y, :]
	elif offset_y < 0:
		img2_offset = np.pad(img2_offset, ((0, -offset_y), (0, 0)), mode='constant')[-offset_y:, :]
	if offset_x > 0:
		img2_offset = np.pad(img2_offset, ((0, 0), (offset_x, 0)), mode='constant')[:, :-offset_x]
	elif offset_x < 0:
		img2_offset = np.pad(img2_offset, ((0, 0), (0, -offset_x)), mode='constant')[:, -offset_x:]

def find_filled_bubbles_alt2(bubbles, row_centers_sorted, col_centers_sorted, thresh2, warped_u8, med_w, med_h, crit):
	ROWS = 30
	COLS = 15

	# experiment
	full_paper_threshold = isodata_threshold(warped_u8.flatten()) + 10
	thresholded = cv2.threshold(warped_u8, full_paper_threshold, 255, cv2.THRESH_BINARY)

	# idea: taking the mean loses a bunch of useful information about the distribution



def find_filled_bubbles_alt(bubbles, row_centers_sorted, col_centers_sorted, thresh2, warped_u8, med_w, med_h, crit):
	ROWS = 30
	COLS = 15

	neighbourhood_means = np.zeros(shape=(ROWS, COLS))

	for r in range(ROWS):
		for c in range(COLS):
			x = int(col_centers_sorted[c])
			y = int(row_centers_sorted[r])

			neighbourhood = warped_u8[y-med_h//2:y+med_h//2+1, x-med_w//2:x+med_w//2+1]
			neighbourhood_flattened = neighbourhood.flatten()
			neighbourhood_flattened_sorted = sorted(neighbourhood_flattened)
			#neighbourhood_flattened_sorted = neighbourhood_flattened_sorted[len(neighbourhood_flattened_sorted)//3:len(neighbourhood_flattened_sorted)//4*3]
			neighbourhood_flattened_sorted = neighbourhood_flattened_sorted[len(neighbourhood_flattened_sorted)//2:len(neighbourhood_flattened_sorted)//4*3]

			# plot the histogram of the neighbourhood
			# plt.figure(figsize=(8, 6))
			# plt.hist(neighbourhood_flattened, bins=255, range=(0, 255), alpha=0.5, label="Original")
			# plt.hist(neighbourhood_flattened_sorted, bins=255, range=(0, 255), alpha=0.5, label="Sorted")
			# plt.title(f"Histogram of Neighbourhood for Bubble at Row {r}, Col {c}")
			# plt.xlabel("Pixel Intensity")
			# plt.ylabel("Frequency")
			# plt.legend()
			# plt.show()

			neighbourhood_intensity = np.mean(neighbourhood_flattened_sorted)

			neighbourhood_means[r,c] = neighbourhood_intensity

	# neighbourhood_means_hist = np.histogram(neighbourhood_means.flatten(), bins=255, range=(0, 255))
	# plt.figure(figsize=(8, 6))
	# plt.bar(neighbourhood_means_hist[1][:-1], neighbourhood_means_hist[0], width=1)
	# plt.title("Histogram of Neighbourhood Means for Detected Bubbles")
	# plt.xlabel("Neighbourhood Mean Intensity")
	# plt.ylabel("Frequency")
	# plt.show()

	# idea: if the mean is very low, we are sure its filled.
	# but if its close to the threshold, we are not sure so we will segment to find only the lightest pixels and take the mean of those to estimate the background intensity

	bubbles_status_grid = np.zeros(shape=(ROWS, COLS), dtype=np.uint8)

	threshold = isodata_threshold(neighbourhood_means.flatten())
	threshold = savgol_threshold(neighbourhood_means.flatten())

	#print(f"Chosen threshold for bubble fill classification: {threshold:.2f}")

	mean_differences = []

	for r in range(ROWS):
		for c in range(COLS):
			neighbourhood_mean = neighbourhood_means[r,c]

			difference_to_threshold = threshold - neighbourhood_mean

			if difference_to_threshold > 0:
				bubbles_status_grid[r, c] = 1
			
			#mean_differences.append(np.abs(neighbourhood_mean - threshold))

	#diff_hist = np.histogram(mean_differences, bins=255, range=(0, 255))

	# plt.figure(figsize=(8, 6))
	# plt.bar(diff_hist[1][:-1], diff_hist[0], width=1)
	# plt.title("Histogram of Mean Differences for Detected Bubbles")
	# plt.xlabel("Difference from Threshold")
	# plt.ylabel("Frequency")
	# plt.show()

	

	
	# Rectification pass:
	
	for r in range(ROWS):
		for c in range(0, COLS, 3):
			neighbourhood_mean = neighbourhood_means[r,c]

			attempt = bubbles_status_grid[r, c] == 1
			zone = bubbles_status_grid[r, c+1] == 1
			top = bubbles_status_grid[r, c+2] == 1

			# If the top is filled, all should be filled
			if top:
				# Set attempt, zone and top
				bubbles_status_grid[r, c] = 1
				bubbles_status_grid[r, c+1] = 1
				bubbles_status_grid[r, c+2] = 1

			# If the zone is filled, attempt and zone should be filled
			elif zone:
				# Set attempt and zone
				bubbles_status_grid[r, c] = 1
				bubbles_status_grid[r, c+1] = 1
			
			# If any more on this row are filled, this must have been an attempt
			elif np.sum(bubbles_status_grid[r, c+3:]) > 0:
				bubbles_status_grid[r, c] = 1



	# Taking the bubble status grid and turning it into a list of indices
	filled_bubbles = []

	for r in range(ROWS):
		for c in range(COLS):
			if bubbles_status_grid[r, c] == 1:
				filled_bubbles.append((r, c))
				


	return filled_bubbles, (ROWS, COLS)

def find_filled_bubbles(bubbles, row_centers_sorted, col_centers_sorted, thresh2, warped_u8, med_w, med_h, crit):
	
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

		#plot_paper(warped_u8, "Paper after preprocessing")

		# 1) Enhance dark pencil strokes
		bh_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))  # tune 15..31
		blackhat = cv2.morphologyEx(warped_u8, cv2.MORPH_BLACKHAT, bh_kernel)

		#plot_paper(blackhat, "Paper after darkening")

		# 2) Combine with original to boost strokes
		enhanced = cv2.add(warped_u8, blackhat)

		#plot_paper(enhanced, "Paper after adding")

		# 3) Contrast stretch (optional but helpful)
		enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

		#plot_paper(enhanced, "Paper after contrast stretch")

		# 4) Blur + adaptive threshold (invert so pencil=white)
		blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

		#plot_paper(blur, "Paper after blur")
		thresh2 = cv2.adaptiveThreshold(
			blur, 255,
			cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
			cv2.THRESH_BINARY_INV,
			31, 7
		)
		#plot_paper(thresh2, "Paper after thresh2")


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


		ROWS = 30
		COLS = 15

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
					filled_cells.append((r, c))
					cv2.circle(overlay, (cx, cy), rad, (0, 0, 255), 2)  # red = filled
					cv2.circle(overlay, (cx, cy), 2, (0, 0, 255), -1)
				else:
					# optional: show unfilled in light green
					cv2.circle(overlay, (cx, cy), rad, (0, 255, 0), 1)

		print(f"Threshold: {fill_threshold:.3f}")
		print(f"Filled count: {len(filled_cells)} / {ROWS*COLS}")
		return filled_cells, (ROWS, COLS)

	filled_cells, (ROWS, COLS) = classify_bubble_fill(fill_scores, row_centers_sorted, col_centers_sorted, overlay, rad)