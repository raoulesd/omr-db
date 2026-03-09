# Code from this file is based on PyImageSearch.com:
# https://pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/

import argparse
import matplotlib.pyplot as plt

import grader

# Arguement parser to get the image path from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

_, _, warped, _, _, image= grader.grade_score_form(args["image"], show_plots=True)

plt.figure()

plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
plt.imshow(warped)
plt.axis("off")

plt.subplot(1, 2, 2)  # second plot
plt.imshow(image)
plt.axis("off")

plt.show()