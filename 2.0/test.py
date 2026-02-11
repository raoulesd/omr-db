import numpy as np
import grader
import argparse
import os

# Example usage:
# python test.py -d train
# python test.py -d test
# python test.py -d ./test_forms/train/score20251104_001.png

def read_ground_truth(instance_path):
	ground_truth_path = "."+"".join(instance_path.split(".")[:-1]) + ".csv"
	print(f"Reading ground truth from: {ground_truth_path}")
	squares_filled = []
	with open(ground_truth_path, "r") as f:
		for line in f:
			if line == "\n":
				continue
			x, y = line.strip().split(",")
			squares_filled.append((int(x), int(y)))
	return squares_filled

def compute_score(true_positives, false_positives, false_negatives, true_negatives):
	total = true_positives + false_positives + false_negatives
	if total == 0:
		return 0
	
	print(f"TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}, TN: {true_negatives}")
	
	accuracy = (true_positives) / total
	return accuracy

def run_on_folder(folder_path):

	true_positives = 0
	false_positives = 0
	false_negatives = 0
	true_negatives = 0

	for file in os.listdir(folder_path):
		file_name = os.fsdecode(file)
		if not file_name.endswith(".png"):
			continue

		true_pos, false_pos, false_neg, true_neg = run_instance(os.path.join(folder_path, file_name))
		true_positives += true_pos
		false_positives += false_pos
		false_negatives += false_neg
		true_negatives += true_neg

		print(f"Instance score: {compute_score(true_pos, false_pos, false_neg, true_neg)}\n")

	score = compute_score(true_positives, false_positives, false_negatives, true_negatives)

	print(f"Score: {score}")

def run_on_single_instance(instance_path):

	true_positives = 0
	false_positives = 0
	false_negatives = 0
	true_negatives = 0

	true_pos, false_pos, false_neg, true_neg = run_instance(instance_path)
	true_positives += true_pos
	false_positives += false_pos
	false_negatives += false_neg
	true_negatives += true_neg

	score = compute_score(true_positives, false_positives, false_negatives, true_negatives)

	print(f"Score: {score}")

	

def run_instance(instance_path):
	print(f"Running on instance: {instance_path}")

	(result, (num_rows, num_cols), warped_u8, (row_centers_sorted, col_centers_sorted), (med_w, med_h)) = grader.grade_score_form(instance_path, show_plots=False)

	ground_truth = read_ground_truth(instance_path)

	return compare_result_with_ground_truth(result, ground_truth, num_rows, num_cols)

def compare_result_with_ground_truth(result, ground_truth, num_rows, num_cols, print_mistakes=True):
	# Initialize counters
	true_positives = 0
	false_positives = 0
	false_negatives = 0
	true_negatives = 0

	result_grid = np.zeros(shape=(num_rows, num_cols), dtype=np.uint8)
	ground_truth_grid = np.zeros(shape=(num_rows, num_cols), dtype=np.uint8)

	for (x, y, score) in result:
		result_grid[x, y] = 1

	for (x, y) in ground_truth:
		ground_truth_grid[x, y] = 1

	true_positives = np.count_nonzero(np.bitwise_and(result_grid, ground_truth_grid))

	false_positives_grid = result_grid.copy()
	false_positives_grid[ground_truth_grid == 1] = 0
	false_positives = np.count_nonzero(false_positives_grid)

	false_negatives_grid = ground_truth_grid.copy()
	false_negatives_grid[result_grid == 1] = 0
	false_negatives = np.count_nonzero(false_negatives_grid)

	true_negatives = np.count_nonzero(np.bitwise_and(np.logical_not(result_grid), np.logical_not(ground_truth_grid)))

	if print_mistakes:
		for r in range(num_rows):
			for c in range(num_cols):
				if false_positives_grid[r,c]:
					print(f"False positive at ({r}, {c})")
				if false_negatives_grid[r,c]:
					print(f"False negative at ({r}, {c})")
	
	return (true_positives, false_positives, false_negatives, true_negatives)



# Arguement parser to know what to run
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True, help="what data to run (train/test/specific instance)")
args = vars(ap.parse_args())

if args["data"] == "train":
	run_on_folder("./test_forms/train")
elif args["data"] == "test":
	run_on_folder("./test_forms/test")
else:
	run_on_single_instance(args["data"])

	