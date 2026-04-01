from ui_state import get_loaded_data, get_ui_state
from pathlib import Path
import ui as frontend

def claim_file_for_instance(candidate):
	source = Path(candidate)
	if not source.exists():
		return None, "File no longer exists (possibly claimed by another instance)."

	get_ui_state().processing_data_folder.mkdir(parents=True, exist_ok=True)
	claimed = get_ui_state().processing_data_folder / source.name
	if claimed.exists():
		return None, f"Could not claim file: claim target already exists ({claimed.name})"

	try:
		source.rename(claimed)
	except Exception as e:
		return None, f"Could not claim file: {e}"

	return str(claimed), None

def release_current_file_to_queue():
	filename = get_loaded_data().filename

	if filename is None:
		return True

	current_path = Path(filename)
	if not current_path.exists():
		filename = None
		return True

	try:
		restored_path = move_file_to_folder(current_path, get_ui_state().to_process_data_folder)
	except Exception as e:
		frontend.set_status(f"Could not put back current file {current_path.name}: {e}")
		return False

	frontend.set_status(f"Returned unexported file to queue: {restored_path.name}")
	filename = None
	return True


def restore_processing_folder_on_exit():
	if not get_ui_state().processing_data_folder.exists():
		return

	moved_count = 0
	failed = []
	for p in sorted(get_ui_state().processing_data_folder.iterdir()):
		if not p.is_file():
			continue
		try:
			move_file_to_folder(p, get_ui_state().to_process_data_folder)
			moved_count += 1
		except Exception as e:
			failed.append(f"{p.name}: {e}")

	if moved_count:
		print(f"Returned {moved_count} file(s) from {get_ui_state().processing_data_folder.name} to {get_ui_state().to_process_data_folder}")

	if failed:
		print("Could not restore some files from processing folder:")
		for message in failed:
			print(f" - {message}")

	try:
		get_ui_state().processing_data_folder.rmdir()
		print(f"Removed processing folder: {get_ui_state().processing_data_folder}")
	except OSError:
		# Folder may still contain files/subfolders when restoration fails.
		pass


def write_results_to_csv(name, contestant_number, gender, age_category):
	
	exportString = f"{name},"
	exportString += f"{contestant_number},"
	exportString += f"{gender},"
	exportString += f"{age_category},"
	file_path = Path(get_loaded_data().filename)
	only_file_name = file_path.name
	exportString += only_file_name
	for i in range(0, len(get_loaded_data().per_boulder_ZT)):
		(zone, top) = get_loaded_data().per_boulder_ZT[i]
		if zone is None:
			zone = 0
		if top is None:
			top = 0
		exportString += f",B{i + 1} T{top}Z{zone}"
	exportString += f",{get_loaded_data().amountZT[1]},{get_loaded_data().amountZT[0]}"
	exportString += f",{get_loaded_data().triesZT[1]},{get_loaded_data().triesZT[0]}"
	get_ui_state().output_csv_file.write(f"{exportString}\n")
	get_ui_state().output_csv_file.flush()

	# Move the claimed source file out of this instance processing folder.
	moved_file_name = move_file_to_folder(file_path, get_ui_state().processed_data_folder)

	if get_loaded_data().filename in get_ui_state().file_list:
		get_ui_state().file_list.remove(get_loaded_data().filename)
	
	get_ui_state().queue_error_map.pop(str(file_path), None)
	get_ui_state().last_failed_file = None

def export_to_ground_truth():

	cell_data = get_loaded_data().cell_data
	filename = get_loaded_data().filename

	filled_cells = []
	for row in range(cell_data.shape[0]):
		for col in range(cell_data.shape[1]):
			if cell_data[row, col] == 1:
				filled_cells.append((row, col))

	pure_file_name = Path(filename).name

	output_file_name = get_ui_state().processed_data_folder / (Path(pure_file_name).stem + ".csv")

	with open(output_file_name, "w") as f:
		for cell in filled_cells:
			f.write(f"{cell[0]},{cell[1]}\n")


	# Move the claimed source file out of this instance processing folder.
	move_file_to_folder(Path(filename), get_ui_state().processed_data_folder)
	if filename in get_ui_state().file_list:
		get_ui_state().file_list.remove(filename)
	get_ui_state().queue_error_map.pop(str(Path(filename)), None)
	get_ui_state().last_failed_file = None

	refresh_file_queue()
	get_next_file(False)


def toggle_all_bubbles_and_markers(sender=None, app_data=None):
	global cell_data

	if filename is None:
		set_status("No active file loaded. Cannot toggle all bubbles.")
		return

	if cell_data.size == 0:
		set_status("No bubble grid detected for current file.")
		return

	# Flip between two full debug states: everything enabled or everything disabled.
	enable_all = not np.all(cell_data == 1)
	fill_value = 1 if enable_all else 0
	cell_data[:, :] = fill_value
	draw_data()


def move_file_to_folder(source_path, target_folder):
	target_folder.mkdir(parents=True, exist_ok=True)
	source = Path(source_path)
	candidate = target_folder / source.name
	if not candidate.exists():
		source.rename(candidate)
		return candidate

	counter = 1
	while True:
		candidate = target_folder / f"{source.stem}__{counter}{source.suffix}"
		if not candidate.exists():
			source.rename(candidate)
			return candidate
		counter += 1