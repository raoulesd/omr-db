import numpy as np
import os
import ui_state
from ui_state import get_loaded_data, get_ui_state
from pathlib import Path
import grader
from configs import config
import tesseract_ocr

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

CONFIG_FILE_NAME = os.getenv("OMR_CONFIG_NAME", "db9-2025")

frontend = None

def setup(frontend_instance):
	global frontend
	frontend = frontend_instance
	config.set_active_system_config("system_config")
	config.set_active_config(CONFIG_FILE_NAME)

	ui_state.setup()

	tesseract_ocr.tesseract_setup()



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

	export_string = f"{name},"
	export_string += f"{contestant_number},"
	export_string += f"{gender},"
	export_string += f"{age_category},"
	file_path = Path(get_loaded_data().filename)
	only_file_name = file_path.name
	export_string += only_file_name
	for i in range(len(get_loaded_data().per_boulder_zones_tops)):
		(zone, top) = get_loaded_data().per_boulder_zones_tops[i]
		if zone is None:
			zone = 0
		if top is None:
			top = 0
		export_string += f",B{i + 1} T{top}Z{zone}"
	export_string += f",{get_loaded_data().amount_zones_tops[1]},{get_loaded_data().amount_zones_tops[0]}"
	export_string += f",{get_loaded_data().tries_zones_tops[1]},{get_loaded_data().tries_zones_tops[0]}"
	get_ui_state().output_csv_file.write(f"{export_string}\n")
	get_ui_state().output_csv_file.flush()

	# Move the claimed source file out of this instance processing folder.
	get_ui_state().moved_file_name = move_file_to_folder(file_path, get_ui_state().processed_data_folder)

	if get_loaded_data().filename in get_ui_state().file_list:
		get_ui_state().file_list.remove(get_loaded_data().filename)

	get_ui_state().queue_error_map.pop(str(file_path), None)
	get_ui_state().last_failed_file = None

	refresh_file_queue()
	get_next_file(False)

def export_to_ground_truth():

	cell_data = get_loaded_data().cell_data
	filename = get_loaded_data().filename

	filled_cells = get_loaded_data().get_filled_cells()

	pure_file_name = Path(filename).name

	output_file_name = get_ui_state().processed_data_folder / (Path(pure_file_name).stem + ".csv")

	with Path.open(output_file_name, "w") as f:
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


def get_next_file(is_initialization):

	if len(get_ui_state().file_list) == 0:
		refresh_file_queue()

	if len(get_ui_state().file_list) == 0:
		frontend.set_status("No files in queue. Add scans and press Refresh Queue.")
		ui_state.reset_loaded_data()
		frontend.update_queue_ui(get_ui_state().file_list, get_loaded_data().filename, get_ui_state().last_failed_file)
		frontend.set_export_buttons_enabled(False)
		return

	# Queue behavior: use the oldest queued file (front of list), but keep it in queue until export.
	load_file(get_ui_state().file_list[0])

def redetect_bubbles():
	if get_loaded_data().filename is None:
		frontend.set_status("No active file loaded. Cannot redetect bubbles.")
		return

	filename = get_loaded_data().filename
	set_state_from_file(filename)
	draw_textures_on_frontend()
	frontend.set_status(f"Redetected bubbles for current file: {Path(filename).name}")

def set_state_from_file(filename):
	filled_cells, (row_centers_sorted, col_centers_sorted), median_bubble_size, scoresheet_rectified = grader.grade_score_form(filename, show_plots=False)

	# We need to change the pixel coordinates to be scaled with the actual image so that they can be drawn correctly
	ui_scale = config.get_property("ui_scale")
	median_bubble_size = (median_bubble_size[0] * ui_scale, median_bubble_size[1] * ui_scale)
	row_centers_sorted = [int(y * ui_scale) for y in row_centers_sorted]
	col_centers_sorted = [int(x * ui_scale) for x in col_centers_sorted]

	get_loaded_data().filename = filename
	get_loaded_data().cell_data = np.zeros((config.get_property("num_boulders"), config.get_property("num_attempts") * config.get_property("num_answers")), dtype=np.uint8)
	for (r, c) in filled_cells:
		get_loaded_data().set_cell_value(r, c, 1, compute_derived_data=False)
	get_loaded_data().compute_derived_data()

	get_loaded_data().row_centers_sorted = row_centers_sorted
	get_loaded_data().col_centers_sorted = col_centers_sorted
	get_loaded_data().median_bubble_size = median_bubble_size

	get_loaded_data().set_textures_from_full_page_texture_data(scoresheet_rectified)





def on_bubble_image_click(clicked_x, clicked_y):

	row_centers_sorted = get_loaded_data().row_centers_sorted
	col_centers_sorted = get_loaded_data().col_centers_sorted
	cell_data = get_loaded_data().cell_data

	# Find the closest row and column
	closest_row = None
	closest_row_distance = None
	closest_col = None
	closest_col_distance = None

	for (row_index, row) in enumerate(row_centers_sorted):
		dist = np.abs(row-clicked_y)
		if closest_row is None or dist < closest_row_distance:
			closest_row = row_index
			closest_row_distance = dist

	for (col_index, col) in enumerate(col_centers_sorted):
		dist = np.abs(col-clicked_x)
		if closest_col is None or dist < closest_col_distance:
			closest_col = col_index
			closest_col_distance = dist

	if cell_data[closest_row, closest_col] == 1:
		get_loaded_data().set_cell_value(closest_row, closest_col, 0)
	else:
		get_loaded_data().set_cell_value(closest_row, closest_col, 1)


	draw_textures_on_frontend()


def toggle_all_bubbles_and_markers():

	if get_loaded_data().filename is None:
		frontend.set_status("No active file loaded. Cannot toggle all bubbles.")
		return

	if get_loaded_data().cell_data.size == 0:
		frontend.set_status("No bubble grid detected for current file.")
		return

	# Flip between two full debug states: everything enabled or everything disabled.
	enable_all = not np.all(get_loaded_data().cell_data == 1)
	fill_value = 1 if enable_all else 0
	for row in range(get_loaded_data().cell_data.shape[0]):
		for col in range(get_loaded_data().cell_data.shape[1]):
			get_loaded_data().set_cell_value(row, col, fill_value, compute_derived_data=False)

	# Compute the number of tops, zones, and tries for each boulder based on the cell data
	get_loaded_data().compute_derived_data()

	draw_textures_on_frontend()



def draw_textures_on_frontend():
	"""Draws all textures (bubble grid, zones/tops, name/category) on the frontend. Should be called after loading a file and setting textures in state.
	"""
	frontend.draw_data(
		cell_data=get_loaded_data().cell_data,
		bubble_grid_image=get_loaded_data().bubble_grid_texture_data,
		zones_and_tops_image=get_loaded_data().zones_and_tops_texture_data,
		attempts_total_image=get_loaded_data().attempts_total_texture_data,
		name_area_image=get_loaded_data().name_texture_data,
		category_area_image=get_loaded_data().category_texture_data,
	)


def load_file(candidate):

	if not Path(candidate).exists():
		frontend.set_status(f"File no longer exists in queue: {Path(candidate).name}")
		if candidate in get_ui_state().file_list:
			get_ui_state().file_list.remove(candidate)
		frontend.update_queue_ui(get_ui_state().file_list, get_loaded_data().filename, get_ui_state().last_failed_file)
		return False

	claimed_path, claim_error = claim_file_for_instance(candidate)
	if claim_error:
		frontend.set_status(f"Could not load {Path(candidate).name}: {claim_error}")
		if candidate in get_ui_state().file_list:
			get_ui_state().file_list.remove(candidate)
		refresh_file_queue()
		return False

	if candidate in get_ui_state().file_list:
		get_ui_state().file_list.remove(candidate)

	filename = claimed_path
	frontend.show_loading_state(filename)
	try:
		set_state_from_file(filename)
	except Exception as e:
		failed_path = Path(filename)
		get_ui_state().queue_error_map.pop(candidate, None)
		moved_failed_path = move_file_to_folder(failed_path, ui_state.get_ui_state().errored_data_folder)
		try:
			moved_failed_path = move_file_to_folder(failed_path, ui_state.get_ui_state().errored_data_folder)
		except Exception as move_error:
			frontend.set_status(f"Error reading {failed_path.name}: {e} | Could not move to errored: {move_error}")
			frontend.show_error_state(f"{failed_path.name}: {e}")
			ui_state.get_ui_state().last_failed_file = str(failed_path)
		else:
			frontend.set_status(f"Error reading {failed_path.name}: {e} | Moved to errored: {moved_failed_path.name}")
			frontend.show_error_state(f"{failed_path.name}: {e}")
			ui_state.get_ui_state().last_failed_file = str(moved_failed_path)

		ui_state.reset_loaded_data()
		refresh_file_queue()
		return False

	get_ui_state().last_failed_file = None
	get_ui_state().queue_error_map.pop(candidate, None)
	get_ui_state().queue_error_map.pop(filename, None)

	draw_textures_on_frontend()

	name_crop = ui_state.get_loaded_data().name_texture_data
	ocr_name, contestant_number, ocr_status = tesseract_ocr.read_name_from_image(name_crop)

	frontend.fill_candidate_name(ocr_name, contestant_number)

	category_values = None
	category_status = None

	if ui_state.get_loaded_data().has_category_area:
		cat_crop = ui_state.get_loaded_data().category_texture_data
		is_male, age_cat, _status = tesseract_ocr.read_category_from_image(cat_crop)
		frontend.set_category(is_male, age_cat)
		category_values = (is_male, age_cat)

	frontend.update_ocr_population_status(ocr_name, ocr_status, category_values, category_status)
	frontend.update_queue_ui(get_ui_state().file_list, get_loaded_data().filename, get_ui_state().last_failed_file)
	frontend.set_export_buttons_enabled(True)
	if ocr_name:
		frontend.set_status(f"Loaded: {Path(filename).name} | OCR: {ocr_name}")
	elif ocr_status:
		frontend.set_status(f"Loaded: {Path(filename).name} | {ocr_status}")
	else:
		frontend.set_status(f"Loaded: {Path(filename).name}")
	return True



def on_queue_file_selected(sender, app_data, user_data):
	selected_path = user_data
	if not selected_path:
		return

	if get_loaded_data().filename is not None:
		if not release_current_file_to_queue():
			return
		refresh_file_queue()

	load_file(selected_path)

def error_check_all_queued_files(sender=None, app_data=None):

	if len(get_ui_state().file_list) == 0:
		frontend.set_error_check_progress(0, 0, is_running=False)
		frontend.set_status("Queue is empty. Nothing to error-check.")
		return

	checked_paths = list(get_ui_state().file_list)
	failed_paths = []
	get_ui_state().queue_error_scan_has_run = True
	frontend.set_error_check_button_enabled(False)
	frontend.set_error_check_progress(0, len(checked_paths), is_running=True)
	frontend.set_status(f"Starting error check for {len(checked_paths)} queued file(s)...")
	frontend.refresh_ui_frame()

	try:
		for index, candidate in enumerate(checked_paths, start=1):
			frontend.set_error_check_progress(index, len(checked_paths), current_file=candidate, is_running=True)
			frontend.set_status(f"Checking file {index} / {len(checked_paths)}: {Path(candidate).name}")
			frontend.refresh_ui_frame()

			if not Path(candidate).exists():
				get_ui_state().queue_error_map[candidate] = "File no longer exists"
				failed_paths.append(candidate)
				continue

			try:
				grader.grade_score_form(candidate, show_plots=False)
			except Exception as e:
				get_ui_state().queue_error_map[candidate] = str(e)
				failed_paths.append(candidate)
			else:
				get_ui_state().queue_error_map.pop(candidate, None)
	finally:
		frontend.update_queue_ui(get_ui_state().file_list, get_loaded_data().filename, get_ui_state().last_failed_file)
		frontend.set_error_check_button_enabled(True)

	if failed_paths:
		failed_names = ", ".join(Path(p).name for p in failed_paths[:3])
		if len(failed_paths) > 3:
			failed_names += ", ..."
		completion_message = (
			f"Error check complete: {len(failed_paths)} of {len(checked_paths)} queued file(s) failed. {failed_names}"
		)
		frontend.set_status(completion_message)
		frontend.set_error_check_progress(
			len(checked_paths),
			len(checked_paths),
			is_running=False,
			status_label=f"Completed: {len(failed_paths)} failed of {len(checked_paths)} checked",
		)
	else:
		completion_message = f"Error check complete: all {len(checked_paths)} queued file(s) loaded successfully."
		frontend.set_status(completion_message)
		frontend.set_error_check_progress(
			len(checked_paths),
			len(checked_paths),
			is_running=False,
			status_label=f"Completed: all {len(checked_paths)} files passed",
		)


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


def refresh_file_queue(sender=None, app_data=None):
	had_empty_state = (len(get_ui_state().file_list) == 0 and get_loaded_data().filename is None)

	if not get_ui_state().to_process_data_folder.exists():
		frontend.set_status(f"Scan directory does not exist: {get_ui_state().to_process_data_folder}")
		frontend.update_queue_ui(get_ui_state().file_list, get_loaded_data().filename, get_ui_state().last_failed_file)
		return

	disk_files = []
	for p in get_ui_state().to_process_data_folder.iterdir():
		if not p.is_file():
			continue
		if p.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
			continue
		disk_files.append(str(p))

	def queue_sort_key(path_str):
		path_obj = Path(path_str)
		try:
			# Oldest scan first, then filename for stable ordering.
			return (path_obj.stat().st_mtime, path_obj.name.lower())
		except OSError:
			# Missing/inaccessible files sink to the end until next refresh.
			return (float("inf"), path_obj.name.lower())

	disk_files.sort(key=queue_sort_key)

	disk_set = set(disk_files)
	old_set = set(get_ui_state().file_list)
	get_ui_state().queue_error_map = {p: message for p, message in get_ui_state().queue_error_map.items() if p in disk_set}

	removed_files = [p for p in get_ui_state().file_list if p not in disk_set]
	added_files = [p for p in disk_files if p not in old_set]

	# Keep existing queue order for files still present, then append new files oldest-first.
	get_ui_state().file_list[:] = [p for p in get_ui_state().file_list if p in disk_set]
	added_files.sort(key=queue_sort_key)
	get_ui_state().file_list.extend(added_files)

	current_removed = False
	if get_loaded_data().filename is not None:
		current_path = Path(get_loaded_data().filename)
		# Current file may be in this instance's processing folder and should remain active.
		if not current_path.exists():
			frontend.set_status(f"Current file was removed: {current_path.name}")
			get_loaded_data().filename = None
			current_removed = True

	if get_ui_state().last_failed_file is not None and not Path(get_ui_state().last_failed_file).exists():
		get_ui_state().last_failed_file = None

	frontend.update_queue_ui(get_ui_state().file_list, get_loaded_data().filename, get_ui_state().last_failed_file)
	if not current_removed:
		if added_files or removed_files:
			frontend.set_status(
				f"Rescanned: +{len(added_files)} / -{len(removed_files)} file(s)"
			)
		else:
			frontend.set_status("Rescanned: no new files found")

	# If current item disappeared, immediately switch to oldest queued file.
	if current_removed:
		if len(get_ui_state().file_list) > 0:
			load_file(get_ui_state().file_list[0])
		else:
			frontend.set_status("Current file removed and queue is now empty.")

	# If we were empty and new files appeared, auto-load the oldest queued file.
	if had_empty_state and len(get_ui_state().file_list) > 0 and get_loaded_data().filename is None:
		load_file(get_ui_state().file_list[0])

def apply_scan_directory_and_refresh(new_directory):
	get_ui_state().to_process_data_folder = new_directory
	get_ui_state().processing_data_folder = get_ui_state().to_process_data_folder.parent / f"processing_{ui_state.get_ui_state().instance_id}"
	if not get_ui_state().to_process_data_folder.exists():
		get_ui_state().to_process_data_folder.mkdir(parents=True, exist_ok=True)
	if not get_ui_state().processing_data_folder.exists():
		get_ui_state().processing_data_folder.mkdir(parents=True, exist_ok=True)
	get_ui_state().queue_error_map = {}
	get_ui_state().queue_error_scan_has_run = False
	frontend.set_error_check_progress(0, 0, is_running=False)
	frontend.set_status(f"Using scan directory: {get_ui_state().to_process_data_folder} | Instance claim folder: {get_ui_state().processing_data_folder.name}")
	refresh_file_queue()
