# Code from this file is based on PyImageSearch.com:
# https://pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/

import argparse

import grader

# Arguement parser to get the image path from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

grader.grade_score_form(args["image"])