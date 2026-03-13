import os

# Collect all fast ground truth paths
fast_ground_truth_paths = []
for root, dirs, files in os.walk("fast_ground_truth"):
	for file in files:
		if file.endswith(".txt"):
			fast_ground_truth_paths.append(file.split(".")[0])

# Determine the corresponding output path
output_paths = {}

# Check the train folder
for root, dirs, files in os.walk("../test_forms/train"):
	for file in files:
		file_name = file.split(".")[0]
		if file_name in fast_ground_truth_paths:
			output_paths[file_name] = os.path.join(root, file_name + ".csv")

# Check the test folder
for root, dirs, files in os.walk("../test_forms/test"):
	for file in files:
		file_name = file.split(".")[0]
		if file_name in fast_ground_truth_paths:
			output_paths[file_name] = os.path.join(root, file_name + ".csv")

# Generate the ground truth files
for file_name in fast_ground_truth_paths:
	output_path = output_paths.get(file_name)
	if not output_path:
		print(f"No corresponding output path found for {file_name}. Skipping.")
		continue

	input_path = os.path.join("fast_ground_truth", file_name + ".txt")
	with open(input_path, "r") as infile, open(output_path, "w") as outfile:
		lines = infile.readlines()

		for (line_index, line) in enumerate(lines):
			if line == "\n":
				continue

			for (char_index, char) in enumerate(line):
				if char == "n":
					continue

				if char == "a":
					outfile.write(f"{line_index},{char_index*3}\n")

				if char == "z":
					outfile.write(f"{line_index},{char_index*3}\n")
					outfile.write(f"{line_index},{char_index*3+1}\n")

				if char == "t":
					outfile.write(f"{line_index},{char_index*3}\n")
					outfile.write(f"{line_index},{char_index*3+1}\n")
					outfile.write(f"{line_index},{char_index*3+2}\n")