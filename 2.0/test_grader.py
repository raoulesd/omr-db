# Code from this file is based on PyImageSearch.com:
# https://pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/

# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
from imutils.perspective import four_point_transform

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
if image is None:
    raise FileNotFoundError(f"Could not read image: {args['image']}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if not hasattr(cv2, "aruco"):
    raise ImportError("cv2.aruco not found. Install opencv-contrib-python.")

# Your fixed IDs by *sheet* corner (physical page corners)
ID_TL, ID_TR, ID_BR, ID_BL = 4, 3, 2, 1

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)

# Detect markers
try:
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, params)
    corners_list, ids, _ = detector.detectMarkers(gray)
except AttributeError:
    params = cv2.aruco.DetectorParameters_create()
    corners_list, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=params)

if ids is None:
    raise RuntimeError("No ArUco markers detected. Check lighting/print/dictionary.")

ids = ids.flatten().tolist()
id_to_corners = {mid: c.reshape(4, 2).astype(np.float32) for c, mid in zip(corners_list, ids)}

missing = [mid for mid in [ID_TL, ID_TR, ID_BR, ID_BL] if mid not in id_to_corners]
if missing:
    raise RuntimeError(f"Missing required ArUco IDs: {missing}. Detected IDs: {sorted(ids)}")

# Compute an estimate of the sheet center from marker centers
marker_centers = np.array([id_to_corners[mid].mean(axis=0) for mid in [ID_TL, ID_TR, ID_BR, ID_BL]], dtype=np.float32)
sheet_center = marker_centers.mean(axis=0)

def outer_corner_by_center(marker_pts_4x2: np.ndarray, center_xy: np.ndarray) -> np.ndarray:
    """Pick the marker corner farthest from the sheet center (rotation-invariant)."""
    dists = np.linalg.norm(marker_pts_4x2 - center_xy, axis=1)
    return marker_pts_4x2[np.argmax(dists)]

# Pick the OUTER corner from each marker (rotation invariant)
pt_tl = outer_corner_by_center(id_to_corners[ID_TL], sheet_center)
pt_tr = outer_corner_by_center(id_to_corners[ID_TR], sheet_center)
pt_br = outer_corner_by_center(id_to_corners[ID_BR], sheet_center)
pt_bl = outer_corner_by_center(id_to_corners[ID_BL], sheet_center)

# Now order these four points as TL, TR, BR, BL in IMAGE coordinates (imutils expects this)
docPts = np.array([pt_tl, pt_tr, pt_br, pt_bl], dtype=np.float32)

s = docPts.sum(axis=1)
d = np.diff(docPts, axis=1).ravel()
ordered = np.array([
    docPts[np.argmin(s)],  # TL
    docPts[np.argmin(d)],  # TR
    docPts[np.argmax(s)],  # BR
    docPts[np.argmax(d)]   # BL
], dtype=np.float32)

paper = four_point_transform(image, ordered)
warped = four_point_transform(gray, ordered)

# Preview
preview = image.copy()
cv2.aruco.drawDetectedMarkers(preview, corners_list, np.array(ids).reshape(-1, 1))
for (x, y), label in zip(ordered, ["TL", "TR", "BR", "BL"]):
    cv2.circle(preview, (int(x), int(y)), 10, (0, 0, 255), -1)
    cv2.putText(preview, label, (int(x) + 10, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

fig, axs = plt.subplots(1, 3, figsize=(22, 7))
axs[0].imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)); axs[0].set_title("AruCo + Selected Corners"); axs[0].axis("off")
axs[1].imshow(cv2.cvtColor(paper, cv2.COLOR_BGR2RGB));   axs[1].set_title("Warped Color");           axs[1].axis("off")
axs[2].imshow(warped, cmap="gray");                      axs[2].set_title("Warped Gray");            axs[2].axis("off")
plt.show()






#----------THIS PART IS USED FOR PERSPECTIVE TRANSFORM USING PAPER EDGE DETECTION----------


# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# edged = cv2.Canny(blurred, 75, 200)



# # Big preview used OpenCV windows:
# cv2.imshow("Edged Image", edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Show a preview of the processed images
# fig, axs = plt.subplots(1, 3, figsize=(18, 6))
# axs[0].imshow(gray, cmap="gray")
# axs[0].set_title("Gray")
# axs[1].imshow(blurred, cmap="gray")
# axs[1].set_title("Blurred")
# axs[2].imshow(edged, cmap="gray")
# axs[2].set_title("Edged")

# for ax in axs:
#     ax.axis("off")

# plt.show()


# # find contours in the edge map, then initialize
# # the contour that corresponds to the document
# cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
# 	cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# docCnt = None
# # ensure that at least one contour was found
# if len(cnts) > 0:
# 	# sort the contours according to their size in
# 	# descending order
# 	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
# 	# loop over the sorted contours
# 	for c in cnts:
# 		# approximate the contour
# 		peri = cv2.arcLength(c, True)
# 		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
# 		# if our approximated contour has four points,
# 		# then we can assume we have found the paper
# 		if len(approx) == 4:
# 			docCnt = approx
# 			break

# # apply a four point perspective transform to both the
# # original image and grayscale image to obtain a top-down
# # birds eye view of the paper
# paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))

# draw the detected document contour on a copy of the image
# image_contour = image.copy()
# cv2.drawContours(image_contour, [docCnt], -1, (0, 255, 0), 3)

# plt.figure(figsize=(8, 10))
# plt.imshow(cv2.cvtColor(image_contour, cv2.COLOR_BGR2RGB))
# plt.title("Detected Document Contour")
# plt.axis("off")
# plt.show()

# --------------------------------------------------------- #