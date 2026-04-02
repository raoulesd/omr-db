import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_aruco_markers(image):

	# First make sure the image is in grayscale
	if len(image.shape) == 3 and image.shape[2] == 3:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Then threshold the image to get a binary image
	_, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

	# Now apply some opening to fill in any gaps in the markers (markers should be solid black so gaps are white)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
	closed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

	image = closed

	# Plot the image
	if False:
		plt.figure(figsize=(8, 10))
		plt.imshow(image, cmap="gray")
		plt.title("Preprocessed Image for ARUCO Detection")
		plt.axis("off")
		plt.show()

	# Load the predefined dictionary
	aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

	# Initialize the detector parameters using default values
	parameters = cv2.aruco.DetectorParameters()

	# Detect the markers in the image
	corners, _ids, _rejected_image_points = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

	marker_centers = []
	if corners is not None:
		for corner in corners:
			corner = corner[0]
			center_x = int(np.mean(corner[:, 0]))
			center_y = int(np.mean(corner[:, 1]))
			marker_centers.append((center_x, center_y))

	if marker_centers is None or len(marker_centers) != 4:
		error_message = f"Invalid ARUCO markers detected. Found: {len(marker_centers) if marker_centers is not None else 0} markers. Expected: 4 markers."
		raise ValueError(error_message)

	# Sort the marker centers by their position
	top_left = min(marker_centers, key=lambda x: x[0] + x[1])
	top_right = min(marker_centers, key=lambda x: -x[0] + x[1])
	bottom_right = max(marker_centers, key=lambda x: x[0] + x[1])
	bottom_left = max(marker_centers, key=lambda x: -x[0] + x[1])
	marker_centers = [top_left, top_right, bottom_right, bottom_left]

	return marker_centers

def rotate_image_to_flatten_aruco_markers(image, aruco_markers):
	"""Rotate the image to flatten the ARUCO markers and ensure they are in the correct orientation.

	Will cause the first two aruco markers (top left and top right) to be horizontal, and the second two aruco markers (bottom right and bottom left) to be horizontal.

	Returns the rotated image and the updated ARUCO marker positions.
	"""
	top_left, top_right, bottom_right, bottom_left = aruco_markers

	top_slope = (top_right[1] - top_left[1]) / (top_right[0] - top_left[0])
	bottom_slope = (bottom_right[1] - bottom_left[1]) / (bottom_right[0] - bottom_left[0])

	avg_slope = (top_slope + bottom_slope) / 2

	angle = np.arctan(avg_slope) * 180 / np.pi

	(h, w) = image.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

	aruco_markers_rotated = []

	for marker in aruco_markers:
		x = int(M[0, 0] * marker[0] + M[0, 1] * marker[1] + M[0, 2])
		y = int(M[1, 0] * marker[0] + M[1, 1] * marker[1] + M[1, 2])
		aruco_markers_rotated.append((x, y))

	return rotated, aruco_markers_rotated

def img_coord_to_relative_coords(img_coord, aruco_markers):
	# Convert the image coordinates to relative coordinates based on the ARUCO markers
	top_left, top_right, bottom_right, bottom_left = aruco_markers
	img_width = np.linalg.norm(np.array(top_right) - np.array(top_left))
	img_height = np.linalg.norm(np.array(bottom_left) - np.array(top_left))

	relative_x = (img_coord[0] - top_left[0]) / img_width
	relative_y = (img_coord[1] - top_left[1]) / img_height

	return (relative_x, relative_y)

def relative_coords_to_img_coord(relative_coord, aruco_markers):
	# Convert the relative coordinates back to image coordinates based on the ARUCO markers
	top_left, top_right, _bottom_right, bottom_left = aruco_markers
	img_width = np.linalg.norm(np.array(top_right) - np.array(top_left))
	img_height = np.linalg.norm(np.array(bottom_left) - np.array(top_left))

	img_x = int(relative_coord[0] * img_width + top_left[0])
	img_y = int(relative_coord[1] * img_height + top_left[1])

	return (img_x, img_y)
