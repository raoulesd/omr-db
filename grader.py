import cv2
import matplotlib.pyplot as plt
import debug_pipeline
from pipeline import region_extractor

from pipeline import preprocess_paper
from pipeline import bubble_grid
from pipeline import find_filled_bubbles


class GradingDebugError(RuntimeError):
	def __init__(self, message, debug_steps=None):
		super().__init__(message)
		self.debug_steps = debug_steps if debug_steps is not None else []

def plot_paper(paper, title):
	plt.figure(figsize=(8, 10))
	plt.imshow(cv2.cvtColor(paper, cv2.COLOR_BGR2RGB))
	plt.title(title)
	plt.axis("off")
	plt.show()

def get_amounts_and_tries(cell_data):

	amount_zones_tops = [0, 0]
	tries_zones_tops = [0, 0]
	per_boulder_zones_tops = []
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

		amount_zones_tops[0] += 1 if zone_attempts is not None else 0
		amount_zones_tops[1] += 1 if top_attempts is not None else 0

		tries_zones_tops[0] += zone_attempts if zone_attempts is not None else 0
		tries_zones_tops[1] += top_attempts if top_attempts is not None else 0

		per_boulder_zones_tops.append((zone_attempts, top_attempts))

	return amount_zones_tops, tries_zones_tops, per_boulder_zones_tops



def grade_score_form(image_path, show_plots=False, debug_mode=False):

	# Load the image and convert it to grayscale
	image = cv2.imread(image_path)
	if image is None:
		error_message = f"Could not read image: {image_path}"
		raise FileNotFoundError(error_message)
	debug_pipeline.add_debug_step(image, "Original Input")

	if not hasattr(cv2, "aruco"):
		error_message = "cv2.aruco not found. Make sure you have opencv-contrib-python installed."
		raise ImportError(error_message)

	try:
		# The scoresheet, but rectified such that the content is aligned with the axes
		scoresheet_rectified, _rectified_aruco_markers = preprocess_paper.preprocess(image)
		debug_pipeline.add_debug_step(scoresheet_rectified, "Preprocess Output (Rectified scoresheet)")

		if show_plots:
			# Show the result of the question area contour detection
			plt.figure(figsize=(8, 10))
			plt.imshow(cv2.cvtColor(scoresheet_rectified, cv2.COLOR_BGR2RGB))
			plt.title("Full rectified scoresheet")
			plt.axis("off")
			plt.show()

		# Now we need to extract only the attempt score region (the bubbles)
		bubble_area_image = region_extractor.extract_region(scoresheet_rectified, "attempt_score")
		bubble_area_image_gray = cv2.cvtColor(bubble_area_image, cv2.COLOR_BGR2GRAY)
		debug_pipeline.add_debug_step(bubble_area_image, "Extracted Bubble Area")

		if show_plots:
			# Show the extracted bubble area
			plt.figure(figsize=(8, 10))
			plt.imshow(cv2.cvtColor(bubble_area_image, cv2.COLOR_BGR2RGB))
			plt.title("Extracted Bubble Area")
			plt.axis("off")
			plt.show()

		question_contours = bubble_grid.detect_bubbles(bubble_area_image_gray)
		debug_pipeline.add_debug_step(bubble_area_image, "Bubble Area")


		bubbles, row_centers_sorted, col_centers_sorted, median_bubble_size = bubble_grid.compute_bubble_grid(question_contours, bubble_area_image_gray)

		if show_plots:
			bubble_grid.plot_bubble_grid(bubble_area_image, row_centers_sorted, col_centers_sorted)

		filled_cells = find_filled_bubbles.find_filled_bubbles_alt(
			bubbles,
			row_centers_sorted,
			col_centers_sorted,
			bubble_area_image,
			median_bubble_size
		)

		if debug_mode:
			final_overlay = bubble_area_image.copy()
			if len(final_overlay.shape) == 2:
				final_overlay = cv2.cvtColor(final_overlay, cv2.COLOR_GRAY2BGR)
			for (r, c) in filled_cells:
				x = int(col_centers_sorted[c])
				y = int(row_centers_sorted[r])
				median_bubble_width = median_bubble_size[0]
				cv2.circle(final_overlay, (x, y), max(2, int(median_bubble_width * 0.8)), (0, 0, 255), 2)
			debug_pipeline.add_debug_step(final_overlay, "Final Filled Bubbles Overlay")
	except Exception as e:
		if debug_mode:
			raise GradingDebugError(str(e), debug_pipeline.get_debug_steps()) from e
		raise

	result = (filled_cells, (row_centers_sorted, col_centers_sorted), median_bubble_size, scoresheet_rectified)
	return result
