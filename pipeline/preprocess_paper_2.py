import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import configs.config as config
import pipeline.aruco_handler as aruco_handler

def plot_paper(paper, title):
	plt.figure(figsize=(8, 10))
	plt.imshow(cv2.cvtColor(paper, cv2.COLOR_BGR2RGB))
	plt.title(title)
	plt.axis("off")
	plt.show()


# Preprocess the paper by flattening it
def preprocess(image, debug_steps=None):
	"""Preprocess the input image to make sure it's level
	"""

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	aruco_markers = aruco_handler.detect_aruco_markers(gray)

	# Rotate the aruco markers to ensure they are in the correct orientation
	rotated_image, aruco_markers = aruco_handler.rotate_image_to_flatten_aruco_markers(image, aruco_markers)

	return rotated_image, aruco_markers

