import cv2
import numpy as np

def detect_aruco_markers(image):
	# Load the predefined dictionary
	aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
	
	# Initialize the detector parameters using default values
	parameters = cv2.aruco.DetectorParameters()
	
	# Detect the markers in the image
	corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

	marker_centers = []
	if corners is not None:
		for corner in corners:
			corner = corner[0]
			center_x = int(np.mean(corner[:, 0]))
			center_y = int(np.mean(corner[:, 1]))
			marker_centers.append((center_x, center_y))

	# Sort the marker centers by their position
	top_left = min(marker_centers, key=lambda x: x[0] + x[1])
	top_right = min(marker_centers, key=lambda x: -x[0] + x[1])
	bottom_right = max(marker_centers, key=lambda x: x[0] + x[1])
	bottom_left = max(marker_centers, key=lambda x: -x[0] + x[1])
	marker_centers = [top_left, top_right, bottom_right, bottom_left]
	
	return marker_centers

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
	top_left, top_right, bottom_right, bottom_left = aruco_markers
	img_width = np.linalg.norm(np.array(top_right) - np.array(top_left))
	img_height = np.linalg.norm(np.array(bottom_left) - np.array(top_left))
	
	img_x = int(relative_coord[0] * img_width + top_left[0])
	img_y = int(relative_coord[1] * img_height + top_left[1])
	
	return (img_x, img_y)