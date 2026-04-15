"""region_extractor.py contains functions for extracting specific regions from the input image based on the configuration. It uses the ARUCO markers to determine the absolute coordinates of the regions to be extracted. The main functions are extract_region for extracting a single region and extract_regions for extracting multiple regions at once.
"""

from configs import config
from pipeline import aruco_handler
import cv2
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

import numpy as np
import cv2

def extract_region(image: np.ndarray, region_name: str, predetermined_size: Optional[tuple] = None) -> np.ndarray:
	"""Extracts a specific region from the input image based on the configuration.

	:param image: The input image as a NumPy array (should be rotated and flattened based on ARUCO markers).
	:param region_name: The name of the region to extract (e.g., "name", "tickbox", "attempts_total").
	:param predetermined_size: Optional tuple specifying the size (width, height) to resize the extracted region to.
	:param attempt_rotation_on_fallback: If True, will attempt to rotate the image to be straight without using the ARUCO markers if they are not detected. Will only work for regions with many straight lines since it relies on line detection to determine the rotation angle. Use with caution as it may not work well for all regions or images.
	:return: The extracted region as a NumPy array, resized to predetermined_size if provided.
	"""

	# Get the coordinates for the specified region from the config
	region_min, region_max = config.get_property(region_name + "_region")

	# Find the aruco markers necessary for finding the absolute coordinates of the region
	markers = aruco_handler.detect_aruco_markers(image)

	# If the necessary markers are not found, perform fallback to default marker positions from the config
	aruco_markers_found = markers is not None and len(markers) == 4
	if not aruco_markers_found:
		print("Warning: ARUCO markers not detected or invalid. Falling back to default marker positions from config.")
		markers = config.get_property("default_marker_positions")

		# Convert the default marker positions to the format expected by the relative_coords_to_img_coord function
		markers = [(int(x * image.shape[1]), int(y * image.shape[0])) for x, y in markers]

	# Convert the relative coordinates to absolute image coordinates
	x_min, y_min = aruco_handler.relative_coords_to_img_coord(region_min, markers)
	x_max, y_max = aruco_handler.relative_coords_to_img_coord(region_max, markers)

	# Extract the region from the image
	extracted_region = image[y_min:y_max, x_min:x_max]

	# Resize to predetermined size if specified
	if predetermined_size is not None:
		extracted_region = cv2.resize(extracted_region, predetermined_size, interpolation=cv2.INTER_AREA)

	return extracted_region

def extract_regions(image: np.ndarray, region_names: list, predetermined_size: Optional[tuple] = None) -> dict:
	"""Extracts multiple regions from the input image based on the configuration.

	:param image: The input image as a NumPy array (should be rotated and flattened based on ARUCO markers).
	:param region_names: A list of region names to extract (e.g., ["name", "tickbox", "attempts_total"]).
	:param predetermined_size: Optional tuple specifying the size (width, height) to resize the extracted regions to.
	:return: A dictionary mapping region names to their extracted NumPy array images, resized to predetermined_size if provided.
	"""
	extracted_regions = {}
	for region_name in region_names:
		extracted_regions[region_name] = extract_region(image, region_name, predetermined_size)
	return extracted_regions
