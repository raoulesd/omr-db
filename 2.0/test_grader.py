# Code from this file is based on PyImageSearch.com:
# https://pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/

import argparse

import grader


def parse_bool(v):
	if isinstance(v, bool):
		return v
	v = str(v).strip().lower()
	if v in {"1", "true", "t", "yes", "y", "on"}:
		return True
	if v in {"0", "false", "f", "no", "n", "off"}:
		return False
	raise argparse.ArgumentTypeError("debug_mode must be true/false")

# Arguement parser to get the image path from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-c", "--config", default="config-db9-13022026", help="config file/module name")
ap.add_argument("-d", "--debug_mode", type=parse_bool, default=False, help="show debug processing steps (true/false)")
args = vars(ap.parse_args())

grader.grade_score_form(args["image"], config_name=args["config"], debug_mode=args["debug_mode"])