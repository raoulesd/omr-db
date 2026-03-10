import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import config as app_config

def plot_paper(paper, title):
	plt.figure(figsize=(8, 10))
	plt.imshow(cv2.cvtColor(paper, cv2.COLOR_BGR2RGB))
	plt.title(title)
	plt.axis("off")
	plt.show()

def preprocess(image, gray):
	cfg = app_config.get_active_config()

	def marker_outer_corner(marker_corners_4x2: np.ndarray, which: str) -> np.ndarray:
		"""
		Given the 4 corners of a detected ArUco marker (shape: (4, 2)),
		return the corner that corresponds to the OUTER sheet corner.

		which ∈ {'tl', 'tr', 'br', 'bl'}
		"""
		pts = marker_corners_4x2.astype(np.float32)

		s = pts.sum(axis=1)              # x + y
		d = np.diff(pts, axis=1).ravel() # y - x

		if which == "tl":
			return pts[np.argmin(s)]
		if which == "br":
			return pts[np.argmax(s)]
		if which == "tr":
			return pts[np.argmin(d)]
		if which == "bl":
			return pts[np.argmax(d)]

		raise ValueError("which must be one of: 'tl', 'tr', 'br', 'bl'")

	def aruco_transform(gray):
		"""
		Detect ArUco markers in the given grayscale image and perform a perspective transform to obtain a top-down view of the paper.
		
		:param gray: Grayscale image of the paper containing ArUco markers
		:return: Tuple of (warped_color, warped_gray, corners_list, ids, docCnt)
				- warped_color: Warped color image of the paper
				- warped_gray: Warped grayscale image of the paper
				- corners_list: List of detected marker corners
				- ids: List of detected marker IDs
				- docCnt: Array of the four corners of the paper in the original image
		"""
		try:
			# Newer OpenCV API
			params = cv2.aruco.DetectorParameters()
			detector = cv2.aruco.ArucoDetector(cfg.ARUDO_DICT, params)
			corners_list, ids, _ = detector.detectMarkers(gray)
		except AttributeError:
			# Older OpenCV API fallback
			params = cv2.aruco.DetectorParameters_create()
			corners_list, ids, _ = cv2.aruco.detectMarkers(
				gray, cfg.ARUDO_DICT, parameters=params
			)

		if ids is None:
			raise RuntimeError(
				"No ArUco markers detected. "
				"Check lighting, print quality, and dictionary."
			)

		ids = ids.flatten().tolist()

		# Map marker ID -> (4, 2) corner array
		id_to_corners = {
			marker_id: corners.reshape(4, 2)
			for corners, marker_id in zip(corners_list, ids)
		}

		# Ensure all required markers are present
		required_ids = [cfg.ID_TL, cfg.ID_TR, cfg.ID_BR, cfg.ID_BL]
		missing = [mid for mid in required_ids if mid not in id_to_corners]
		if missing:
			raise RuntimeError(
				f"Missing required ArUco IDs: {missing}. "
				f"Detected IDs: {sorted(ids)}"
			)

		pt_tl = marker_outer_corner(id_to_corners[cfg.ID_TL], "tl")
		pt_tr = marker_outer_corner(id_to_corners[cfg.ID_TR], "tr")
		pt_br = marker_outer_corner(id_to_corners[cfg.ID_BR], "br")
		pt_bl = marker_outer_corner(id_to_corners[cfg.ID_BL], "bl")

		docCnt = np.array(
			[
				pt_tl + cfg.offset_tl,
				pt_tr + cfg.offset_tr,
				pt_br + cfg.offset_br,
				pt_bl + cfg.offset_bl,
			],
			dtype=np.float32
		)

		paper = four_point_transform(image, docCnt)
		warped = four_point_transform(gray, docCnt)
		return paper, warped, corners_list, ids, docCnt

	def ui_draw_aruco_corners(image, corners_list, ids, docCnt):
		"""
		Draw detected ArUco markers and the document corners on the image. For debugging and UI purposes.
		
		:param image: The input image on which to draw
		:param corners_list: List of detected marker corners
		:param ids: List of detected marker IDs
		:param docCnt: Array of the four corners of the document
		:return: Image with drawn markers and document corners
		"""
		preview = image.copy()
		cv2.aruco.drawDetectedMarkers(
			preview, corners_list, np.array(ids).reshape(-1, 1)
		)

		for (x, y), label in zip(docCnt, ["TL", "TR", "BR", "BL"]):
			cv2.circle(preview, (int(x), int(y)), 10, (0, 0, 255), -1)
			cv2.putText(
				preview,
				label,
				(int(x) + 10, int(y) - 10),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.8,
				(0, 0, 255),
				2
			)
		return preview

	def question_area_transform(warped):
		"""
		Finds the contour of the question area in the warped grayscale image and applies a perspective transform to isolate it. 
		This is useful if the question area is misaligned from the ArUco markers.
		
		:param warped: Warped grayscale image of the paper
		:return: Tuple of (paper, warped, docCnt)
				- paper: Warped color image of the question area
				- warped: Warped grayscale image of the question area
				- docCnt: Array of the four corners of the question area in the warped image
		"""
		image = warped
		gray = image

		blurred = cv2.GaussianBlur(gray, (5, 5), 0)
		edged = cv2.Canny(blurred, 75, 200)


		thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		questionCnts = []
		# loop over the contours
		for c in cnts:
			# compute the bounding box of the contour, then use the
			# bounding box to derive the aspect ratio
			(x, y, w, h) = cv2.boundingRect(c)
			ar = w / float(h)
			# in order to find the question area, the region
			# should be sufficiently wide, sufficiently tall, and
			# have an aspect ratio approximately equal to 0.6
			if w >= 180 and h >= 120 and ar >= 0.6:
				questionCnts.append(c)

		docCnt = None


		# cnts should only have 1 contour, the largest in the image and the area the questions are in
		# this then allows us to find the four corners of the question area for perspective transform
		# this might be handy if the question area is misaligned from the aruco markers
		if len(cnts) > 0:
			cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

			for c in cnts:
				# optional: reject tiny contours early
				if cv2.contourArea(c) < 1000:
					continue

				pts = c.reshape(-1, 2)

				# Tune these bands (pixels). Increase if edges are noisy.
				tol_y = 10

				minY = pts[:, 1].min()
				maxY = pts[:, 1].max()

				top_band = pts[pts[:, 1] <= (minY + tol_y)]
				bot_band = pts[pts[:, 1] >= (maxY - tol_y)]

				# Need enough points to be meaningful
				if len(top_band) < 2 or len(bot_band) < 2:
					continue

				# Top corners: safe from the bottom tail
				TL = top_band[np.argmin(top_band[:, 0])]
				TR = top_band[np.argmax(top_band[:, 0])]

				# Bottom y: robust statistic from bottom band
				y_bottom = int(np.median(bot_band[:, 1]))

				# Bottom corners: DO NOT use bottom_band minX (tail poison)
				BL = np.array([int(TL[0]), y_bottom], dtype=np.int32)
				BR = np.array([int(TR[0]), y_bottom], dtype=np.int32)

				# Build docCnt in the shape OpenCV/imutils code expects: (4,1,2)
				docCnt = np.array([TL, TR, BR, BL], dtype=np.float32).reshape(4, 1, 2)
				break


		# Apply a four point perspective transform to both the
		# original image and grayscale image to obtain a top-down view
		paper = four_point_transform(image, docCnt.reshape(4, 2))
		warped = four_point_transform(gray,  docCnt.reshape(4, 2))
		return paper, warped

	try:
		paper, warped, corners_list, ids, docCnt = aruco_transform(gray)
	except Exception as e:
		print(f"Error in aruco_transform: {e}")
		warped = gray

	if cfg.has_bounded_question_area:
		paper, warped = question_area_transform(warped)

	return paper, warped