import cv2

debug_steps = []

def reset_debug_steps():
	global debug_steps
	debug_steps = []

def add_debug_step(image, title=None):
	"""Adds a debug step with the given image and title to the global debug_steps list."""
	global debug_steps
	if debug_steps is None:
		raise ValueError("Debug steps list is not initialized")
	
	if image is None:
		raise ValueError("Image cannot be None")
	
	if title is None:
		title = f"Debug Step"

	# Add a number to the title based on the current number of debug steps
	title = f"{len(debug_steps) + 1:02d} - {title}"

	img = image.copy()
	if len(img.shape) == 2:
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

	debug_steps.append((title, img))

def get_debug_steps():
	"""Returns the list of debug steps."""
	global debug_steps
	return debug_steps