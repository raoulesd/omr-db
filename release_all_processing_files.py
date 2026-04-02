import os
folder = "./process_data/"
to_unpack_prefix = "processing_"
release_into_folder = "./process_data/to_process/"

# Move files from processing folders back to to_process
for file in os.listdir(folder):
	if file.startswith(to_unpack_prefix) or True:
		full_path = os.path.join(folder, file)
		if os.path.isfile(full_path):
			new_path = os.path.join(release_into_folder, file[len(to_unpack_prefix):])
			print(f"Releasing {full_path} to {new_path}")
			os.rename(full_path, new_path)