import cv2

debug_steps = []

def reset_debug_steps():
	global debug_steps
	debug_steps = []

def add_debug_step(image, title=None):
	"""Adds a debug step with the given image and title to the global debug_steps list."""
	global debug_steps
	if debug_steps is None:
		error_message = "Debug steps list is not initialized. Call reset_debug_steps() before adding debug steps."
		raise ValueError(error_message)

	if image is None:
		error_message = "Image cannot be None"
		raise ValueError(error_message)

	if title is None:
		title = "Debug Step"

	img = image.copy()
	if len(img.shape) == 2:
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

	debug_steps.append((title, img))

def get_debug_steps():
	"""Returns the list of debug steps."""
	return debug_steps
