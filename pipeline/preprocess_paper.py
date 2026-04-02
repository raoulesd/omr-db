"""preprocess_paper.py contains functions for preprocessing the input image of the paper, including converting it to grayscale, detecting ARUCO markers to find the corners of the paper, and rotating the image to ensure the paper is level. The main function is preprocess, which takes an input image and returns a rotated image along with the detected ARUCO marker positions for later use in warping and scoring.
"""

import cv2
from pipeline import aruco_handler


# Preprocess the paper by flattening it
def preprocess(image, debug_steps=None):
	"""Preprocess the input image to make sure it's level
	"""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	aruco_markers = aruco_handler.detect_aruco_markers(gray)

	# Rotate the aruco markers to ensure they are in the correct orientation
	rotated_image, aruco_markers = aruco_handler.rotate_image_to_flatten_aruco_markers(image, aruco_markers)

	return rotated_image, aruco_markers

