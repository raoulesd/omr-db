"""region_extractor.py contains functions for extracting specific regions from the input image based on the configuration. It uses the ARUCO markers to determine the absolute coordinates of the regions to be extracted. The main functions are extract_region for extracting a single region and extract_regions for extracting multiple regions at once.
"""

from configs import config
from pipeline import aruco_handler
import cv2
import numpy as np
from typing import Optional

def extract_region(image: np.ndarray, region_name: str, predetermined_size: Optional[tuple] = None) -> np.ndarray:
	"""Extracts a specific region from the input image based on the configuration.

	Args:
		image: The input image as a NumPy array (should be rotated and flattened based on ARUCO markers).
		region_name: The name of the region to extract (e.g., "name", "tickbox", "attempts_total").
		predetermined_size: Optional tuple specifying the size (width, height) to resize the extracted region to.

	Returns:
		The extracted region as a NumPy array, resized to predetermined_size if provided.

	Raises:
		ValueError: If the specified region_name is not defined in the configuration.

	"""
	# Get the coordinates for the specified region from the config
	min, max = config.get_property(region_name + "_region")

	# Find the aruco markers necessary for finding the absolute coordinates of the region
	markers = aruco_handler.detect_aruco_markers(image)

	# Convert the relative coordinates to absolute image coordinates
	x_min, y_min = aruco_handler.relative_coords_to_img_coord(min, markers)
	x_max, y_max = aruco_handler.relative_coords_to_img_coord(max, markers)

	# Extract the region from the image
	extracted_region = image[y_min:y_max, x_min:x_max]

	# Resize to predetermined size if specified
	if predetermined_size is not None:
		extracted_region = cv2.resize(extracted_region, predetermined_size, interpolation=cv2.INTER_AREA)

	return extracted_region

def extract_regions(image: np.ndarray, region_names: list, predetermined_size: Optional[tuple] = None) -> dict:
	"""Extracts multiple regions from the input image based on the configuration.

	Args:
		image: The input image as a NumPy array (should be rotated and flattened based on ARUCO markers).
		region_names: A list of region names to extract.
		predetermined_size: Optional tuple specifying the size (width, height) to resize the extracted regions to.

	Returns:
		A dictionary mapping region names to their extracted NumPy array images.

	"""
	extracted_regions = {}
	for region_name in region_names:
		extracted_regions[region_name] = extract_region(image, region_name, predetermined_size)
	return extracted_regions
