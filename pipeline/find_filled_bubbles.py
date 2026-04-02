import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import configs.config as config


def _render_histogram_image(values, bins=64, value_range=(0, 255), title="Histogram", size=(900, 500), color=(255, 120, 0)):
	"""Render a simple histogram into a BGR image for debug-step visualization."""
	w, h = size
	img = np.full((h, w, 3), 255, dtype=np.uint8)

	counts, edges = np.histogram(values, bins=bins, range=value_range)
	max_c = float(np.max(counts)) if np.max(counts) > 0 else 1.0

	left = 60
	right = w - 30
	top = 40
	bottom = h - 60
	plot_w = right - left
	plot_h = bottom - top

	# Axes
	cv2.rectangle(img, (left, top), (right, bottom), (220, 220, 220), 1)

	bar_w = max(1, int(plot_w / len(counts)))
	for i, c in enumerate(counts):
		x1 = left + i * bar_w
		x2 = min(right, x1 + bar_w - 1)
		bar_h = int((c / max_c) * plot_h)
		y1 = bottom - bar_h
		cv2.rectangle(img, (x1, y1), (x2, bottom), color, -1)

	cv2.putText(img, title, (left, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
	return img


def _render_hist_with_curve_image(counts, curve, marks=None, title="Histogram + Curve", size=(900, 500)):
	"""Render histogram bars and an overlay curve into a BGR image."""
	w, h = size
	img = np.full((h, w, 3), 255, dtype=np.uint8)

	counts = np.asarray(counts, dtype=np.float32)
	curve = np.asarray(curve, dtype=np.float32)
	n = len(counts)
	if n == 0:
		return img

	left = 60
	right = w - 30
	top = 40
	bottom = h - 60
	plot_w = right - left
	plot_h = bottom - top

	cv2.rectangle(img, (left, top), (right, bottom), (220, 220, 220), 1)

	max_c = float(np.max(counts)) if np.max(counts) > 0 else 1.0
	bar_w = max(1, int(plot_w / n))
	for i, c in enumerate(counts):
		x1 = left + i * bar_w
		x2 = min(right, x1 + bar_w - 1)
		bar_h = int((c / max_c) * plot_h)
		y1 = bottom - bar_h
		cv2.rectangle(img, (x1, y1), (x2, bottom), (200, 200, 200), -1)

	cur_min = float(np.min(curve))
	cur_max = float(np.max(curve))
	den = cur_max - cur_min if cur_max > cur_min else 1.0
	pts = []
	for i, y in enumerate(curve):
		x = left + int((i / max(1, n - 1)) * plot_w)
		yp = bottom - int(((float(y) - cur_min) / den) * plot_h)
		pts.append((x, yp))
	if len(pts) >= 2:
		cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, (255, 0, 0), 2)

	if marks:
		for idx, color in marks:
			x = left + int((idx / max(1, n - 1)) * plot_w)
			cv2.line(img, (x, top), (x, bottom), color, 2)

	cv2.putText(img, title, (left, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
	return img


def _render_grid_heatmap_image(grid, title="Grid Heatmap", size=(1000, 700)):
	"""Render a 2D float grid as a readable heatmap with cell boundaries."""
	w, h = size
	img = np.full((h, w, 3), 255, dtype=np.uint8)

	grid = np.asarray(grid, dtype=np.float32)
	rows, cols = grid.shape
	left = 60
	right = w - 30
	top = 50
	bottom = h - 40
	plot_w = right - left
	plot_h = bottom - top

	norm = grid - float(np.min(grid))
	den = float(np.max(norm))
	if den > 0:
		norm = norm / den
	norm_u8 = (norm * 255).astype(np.uint8)
	colored = cv2.applyColorMap(norm_u8, cv2.COLORMAP_VIRIDIS)
	resized = cv2.resize(colored, (plot_w, plot_h), interpolation=cv2.INTER_NEAREST)
	img[top:bottom, left:right] = resized

	# Draw grid lines for readability.
	for r in range(rows + 1):
		y = top + int((r / max(1, rows)) * plot_h)
		cv2.line(img, (left, y), (right, y), (255, 255, 255), 1)
	for c in range(cols + 1):
		x = left + int((c / max(1, cols)) * plot_w)
		cv2.line(img, (x, top), (x, bottom), (255, 255, 255), 1)

	cv2.rectangle(img, (left, top), (right, bottom), (220, 220, 220), 1)
	cv2.putText(img, title, (left, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
	return img


def _render_binary_grid_image(binary_grid, title="Binary Grid", size=(1000, 700)):
	"""Render a small binary grid as a large, readable cell map."""
	w, h = size
	img = np.full((h, w, 3), 255, dtype=np.uint8)

	grid = np.asarray(binary_grid, dtype=np.uint8)
	rows, cols = grid.shape
	left = 60
	right = w - 30
	top = 50
	bottom = h - 40
	plot_w = right - left
	plot_h = bottom - top

	cell_w = max(1, plot_w // max(1, cols))
	cell_h = max(1, plot_h // max(1, rows))

	for r in range(rows):
		for c in range(cols):
			x1 = left + c * cell_w
			y1 = top + r * cell_h
			x2 = min(right, x1 + cell_w)
			y2 = min(bottom, y1 + cell_h)
			if grid[r, c] == 1:
				color = (0, 0, 255)
			else:
				color = (230, 245, 230)
			cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
			cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), 1)

	cv2.rectangle(img, (left, top), (right, bottom), (120, 120, 120), 2)
	cv2.putText(img, title, (left, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
	return img

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
	threshold = np.mean(values)

	while True:

		mean_left = np.mean(values[values<=threshold])
		mean_right = np.mean(values[values>threshold])

		new_threshold = (mean_left + mean_right) / 2
		if np.abs(threshold - new_threshold) < config.get_property("isodata_epsilon"):
			break
		threshold = new_threshold

	return threshold

def savgol_threshold(values, window_length=13, polyorder=3, multiplier=1.0, debug_steps=None):

	isodata_t = int(np.round(isodata_threshold(values)))

	values_histogram = np.histogram(values, bins=255, range=(0, 255))
	counts = values_histogram[0]

	# Smooth the histogram counts using Savitzky-Golay filter
	smooth_counts = scipy.signal.savgol_filter(counts, window_length=window_length, polyorder=polyorder)
	y = smooth_counts
	threshold = np.argmin(y[isodata_t:isodata_t+15]) + isodata_t

	if debug_steps is not None:
		graph = _render_hist_with_curve_image(
			counts,
			y,
			marks=[
				(isodata_t, (0, 180, 0)),
				(threshold, (0, 140, 255)),
			],
			title="Threshold Selection - Histogram + Smoothed Curve",
		)
		debug_steps.append(("Fill Detection - Savgol Threshold Graph", graph))

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



def find_filled_bubbles_alt(bubbles, row_centers_sorted, col_centers_sorted, bubble_area_image, median_bubble_size, debug_steps=None):
	rows = config.get_property("num_boulders")
	cols = config.get_property("num_attempts") * config.get_property("num_answers")

	neighbourhood_means = np.zeros(shape=(rows, cols))
	sample_neighbourhood = None
	sample_neighbourhood_sorted = None

	med_w = median_bubble_size[0]
	med_h = median_bubble_size[1]

	for r in range(rows):
		for c in range(cols):
			x = int(col_centers_sorted[c])
			y = int(row_centers_sorted[r])

			neighbourhood = bubble_area_image[y-med_h//2:y+med_h//2+1, x-med_w//2:x+med_w//2+1]
			neighbourhood_flattened = neighbourhood.flatten()
			neighbourhood_flattened_sorted = sorted(neighbourhood_flattened)
			neighbourhood_flattened_sorted = neighbourhood_flattened_sorted[len(neighbourhood_flattened_sorted)//2:len(neighbourhood_flattened_sorted)//4*3]

			if sample_neighbourhood is None:
				sample_neighbourhood = np.array(neighbourhood_flattened, dtype=np.float32)
				sample_neighbourhood_sorted = np.array(neighbourhood_flattened_sorted, dtype=np.float32)

			neighbourhood_intensity = np.mean(neighbourhood_flattened_sorted)

			neighbourhood_means[r,c] = neighbourhood_intensity

	# idea: if the mean is very low, we are sure its filled.
	# but if its close to the threshold, we are not sure so we will segment to find only the lightest pixels and take the mean of those to estimate the background intensity

	bubbles_status_grid = np.zeros(shape=(rows, cols), dtype=np.uint8)

	threshold = isodata_threshold(neighbourhood_means.flatten())
	threshold = savgol_threshold(neighbourhood_means.flatten(), debug_steps=debug_steps)

	if debug_steps is not None and sample_neighbourhood is not None:
		# 2) Histogram of one sample neighbourhood (original vs sorted-selection)
		h1 = _render_histogram_image(
			sample_neighbourhood,
			bins=64,
			value_range=(0, 255),
			title="Neighbourhood Histogram - Original",
			color=(180, 180, 180),
		)
		h2 = _render_histogram_image(
			sample_neighbourhood_sorted,
			bins=64,
			value_range=(0, 255),
			title="Neighbourhood Histogram - Sorted Subset",
			color=(255, 120, 0),
		)
		debug_steps.append(("Fill Detection - Neighbourhood Histogram (Original)", h1))
		debug_steps.append(("Fill Detection - Neighbourhood Histogram (Sorted Subset)", h2))

		# 3) Neighbourhood means as a 2D heatmap is more informative than a flat histogram.
		nm_hist = _render_grid_heatmap_image(
			neighbourhood_means,
			title="Neighbourhood Means Heatmap",
		)
		debug_steps.append(("Fill Detection - Neighbourhood Means Histogram", nm_hist))

	mean_differences = []

	for r in range(rows):
		for c in range(cols):
			neighbourhood_mean = neighbourhood_means[r,c]

			difference_to_threshold = threshold - neighbourhood_mean

			if difference_to_threshold > 0:
				bubbles_status_grid[r, c] = 1
			
			mean_differences.append(np.abs(neighbourhood_mean - threshold))

	if debug_steps is not None and len(mean_differences) > 0:
		# 1) Histogram of mean differences to threshold
		diff_hist = _render_histogram_image(
			mean_differences,
			bins=64,
			value_range=(0, 255),
			title="Histogram of Mean Differences to Threshold",
			color=(0, 200, 120),
		)
		debug_steps.append(("Fill Detection - Mean Difference Histogram", diff_hist))

	

	
	# Rectification pass:
	
	for r in range(rows):
		for c in range(0, cols, 3):
			neighbourhood_mean = neighbourhood_means[r,c]

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

	for r in range(rows):
		for c in range(cols):
			if bubbles_status_grid[r, c] == 1:
				filled_bubbles.append((r, c))

	if debug_steps is not None:
		grid_img = _render_binary_grid_image(
			bubbles_status_grid,
			title="Final Filled Grid (Red=Filled, Green=Empty)",
		)
		debug_steps.append(("Fill Detection - Final Filled Grid", grid_img))
				


	return filled_bubbles
