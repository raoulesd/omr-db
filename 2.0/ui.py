import os
from pathlib import Path
import dearpygui.dearpygui as dpg
import cv2 as cv
import cv2 as cv2
import numpy as np
import grader
from configs import config as app_config

COLUMNS = 9
ROWS = 20
ANSWERS = 3
CONFIG_FILE_NAME = os.getenv("OMR_CONFIG_NAME", "config-db9-13022026")
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

if __name__ == '__main__':

	cfg = app_config.set_active_config(CONFIG_FILE_NAME)

	processed_data_folder = Path(cfg.PROCESSED_FILES_DIR)
	to_process_data_folder = Path(cfg.SCANNED_FILES_DIR)
	errored_data_folder = Path(cfg.ERRORED_FILES_DIR)
	results_csv_path = Path(cfg.RESULTS_CSV_PATH)
	ui_areas = cfg.UI_AREAS

	ui_scale = float(cfg.UI_SCALE)

	frame_width = int(cfg.FRAME_WIDTH * ui_scale)
	frame_height = int(cfg.FRAME_HEIGHT * ui_scale)

	attempt_totals_height = int(cfg.ATTEMPT_TOTALS_HEIGHT * ui_scale)

	zones_and_tops_width = int(cfg.ZONES_AND_TOPS_WIDTH * ui_scale)

	name_data_width = int(cfg.NAME_DATA_WIDTH * ui_scale)
	name_data_height = int(cfg.NAME_DATA_HEIGHT * ui_scale)

	paths = [processed_data_folder, to_process_data_folder, errored_data_folder]
	for p in paths:
		if not p.exists():
			p.mkdir(parents=True, exist_ok=True)
			print(f"Made dir: {p}")

	path = to_process_data_folder
	fileList = []
	filename = None
	queue_display_map = {}

	# Safe defaults so UI can boot even when queue is empty.
	frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
	full_page = frame.copy()
	cell_data = np.zeros((cfg.ROWS, cfg.COLS), dtype=np.uint8)
	row_centers_sorted = np.array([])
	col_centers_sorted = np.array([])
	med_w = 1
	med_h = 1
	amountZT = [0, 0]
	triesZT = [0, 0]
	per_boulder_ZT = []
	texture_data = np.zeros((frame_height * frame_width * 3,), dtype=np.float32)
	zones_and_tops_texture_data = np.zeros((frame_height * zones_and_tops_width * 3,), dtype=np.float32)
	name_texture_data = np.zeros((name_data_height * name_data_width * 3,), dtype=np.float32)
	attempts_total_data = np.zeros((attempt_totals_height * zones_and_tops_width * 3,), dtype=np.float32)

	csvFile = open(results_csv_path, "a")

	dpg.create_context()
	dpg.create_viewport(title='Review scores', width=1400, height=1000)
	dpg.setup_dearpygui()

	def set_status(message):
		if dpg.does_item_exist("scan_status_text"):
			dpg.set_value("scan_status_text", message)
		print(message)

	def set_export_buttons_enabled(enabled):
		if dpg.does_item_exist("export_button"):
			dpg.configure_item("export_button", enabled=enabled)
		if dpg.does_item_exist("export_ground_truth_button"):
			dpg.configure_item("export_ground_truth_button", enabled=enabled)

	def clear_display_textures():
		global frame, full_page, cell_data, row_centers_sorted, col_centers_sorted, med_w, med_h

		frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
		full_page = frame.copy()
		cell_data = np.zeros((cfg.ROWS, cfg.COLS), dtype=np.uint8)
		row_centers_sorted = np.array([])
		col_centers_sorted = np.array([])
		med_w = 1
		med_h = 1

		if dpg.does_item_exist("texture_tag"):
			dpg.set_value("texture_tag", np.zeros((frame_height * frame_width * 3,), dtype=np.float32))
		if dpg.does_item_exist("zones_and_tops_texture"):
			dpg.set_value("zones_and_tops_texture", np.zeros((frame_height * zones_and_tops_width * 3,), dtype=np.float32))
		if dpg.does_item_exist("name_texture"):
			dpg.set_value("name_texture", np.zeros((name_data_height * name_data_width * 3,), dtype=np.float32))
		if dpg.does_item_exist("attempts_total_texture"):
			dpg.set_value("attempts_total_texture", np.zeros((attempt_totals_height * zones_and_tops_width * 3,), dtype=np.float32))

	def update_queue_ui():
		global queue_display_map
		if not dpg.does_item_exist("queue_list"):
			return

		queue_display_map = {}
		queue_items = []
		for idx, p in enumerate(reversed(fileList), start=1):
			is_current = (filename is not None and p == filename)
			prefix = "* " if is_current else "  "
			label = f"{prefix}{idx:03d} | {Path(p).name}"
			queue_display_map[label] = p
			queue_items.append(label)

		if not queue_items:
			queue_items = ["<queue empty>"]

		dpg.configure_item("queue_list", items=queue_items)
		dpg.set_value("queue_count_text", f"Queue: {len(fileList)} file(s)")
		current_label = Path(filename).name if filename else "-"
		dpg.set_value("current_file_text", f"Current: {current_label}")

		# If nothing is queued/active, present a blank UI and disable exports.
		if len(fileList) == 0 and filename is None:
			clear_display_textures()
			set_export_buttons_enabled(False)
		else:
			set_export_buttons_enabled(filename is not None)

	def refresh_file_queue(sender=None, app_data=None):
		global to_process_data_folder
		global filename
		had_empty_state = (len(fileList) == 0 and filename is None)

		if not to_process_data_folder.exists():
			set_status(f"Scan directory does not exist: {to_process_data_folder}")
			update_queue_ui()
			return

		disk_files = []
		for p in sorted(to_process_data_folder.iterdir()):
			if not p.is_file():
				continue
			if p.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
				continue
			disk_files.append(str(p))

		disk_set = set(disk_files)
		old_set = set(fileList)

		removed_files = [p for p in fileList if p not in disk_set]
		added_files = [p for p in disk_files if p not in old_set]

		# Keep existing queue order for files still present, then append new files.
		fileList[:] = [p for p in fileList if p in disk_set]
		fileList.extend(added_files)

		current_removed = False
		if filename is not None and filename not in disk_set:
			set_status(f"Current file was removed: {Path(filename).name}")
			filename = None
			current_removed = True

		update_queue_ui()
		if added_files or removed_files:
			set_status(
				f"Rescanned: +{len(added_files)} / -{len(removed_files)} file(s)"
			)
		else:
			set_status("Rescanned: no new files found")

		# If current item disappeared, immediately switch to top-of-stack file.
		if current_removed:
			if len(fileList) > 0:
				load_file(fileList[-1])
			else:
				set_status("Current file removed and queue is now empty.")

		# If we were empty and new files appeared, auto-load the top-of-stack file.
		if had_empty_state and len(fileList) > 0 and filename is None:
			load_file(fileList[-1])

	def apply_scan_directory_and_refresh(sender, app_data):
		global to_process_data_folder
		new_dir = Path(dpg.get_value("scan_dir_input")).expanduser()
		to_process_data_folder = new_dir
		if not to_process_data_folder.exists():
			to_process_data_folder.mkdir(parents=True, exist_ok=True)
		set_status(f"Using scan directory: {to_process_data_folder}")
		refresh_file_queue()

	def load_file(candidate):
		global filename, amountZT, triesZT, per_boulder_ZT, frame, data, texture_data, cell_data, row_centers_sorted, col_centers_sorted, med_w, med_h, full_page

		if not Path(candidate).exists():
			set_status(f"File no longer exists: {Path(candidate).name}")
			if candidate in fileList:
				fileList.remove(candidate)
			update_queue_ui()
			return False

		filename = candidate
		try:
			filled_cells, (ROWS, COLS), warped_u8, (row_centers_sorted, col_centers_sorted), (med_w, med_h), full_page = grader.grade_score_form(filename, show_plots=False, config_name=CONFIG_FILE_NAME)
		except Exception as e:
			set_status(f"Error reading {Path(filename).name}: {e}")
			filename = None
			update_queue_ui()
			return False

		cell_data = np.zeros((ROWS, COLS), dtype=np.uint8)
		for (r, c) in filled_cells:
			cell_data[r, c] = 1

		if len(warped_u8.shape) == 2:
			warped_u8 = cv2.cvtColor(warped_u8, cv2.COLOR_GRAY2RGB)
		frame = warped_u8

		draw_data()
		update_queue_ui()
		set_export_buttons_enabled(True)
		set_status(f"Loaded: {Path(filename).name}")
		return True

	def on_queue_file_selected(sender, app_data):
		selected_label = dpg.get_value("queue_list")
		if not selected_label or selected_label == "<queue empty>":
			return

		selected_path = queue_display_map.get(selected_label)
		if not selected_path:
			set_status("Selected file could not be resolved. Refresh queue.")
			return

		load_file(selected_path)

	def get_next_file(is_initialization):
		global filename

		if len(fileList) == 0:
			refresh_file_queue()

		if len(fileList) == 0:
			set_status("No files in queue. Add scans and press Refresh Queue.")
			filename = None
			update_queue_ui()
			set_export_buttons_enabled(False)
			return

		# Stack behavior: use the newest queued file (end of list), but keep it in queue until export.
		load_file(fileList[-1])

	def draw_data():
		global cell_data, full_page, amountZT, triesZT, per_boulder_ZT, frame

		amountZT, triesZT, per_boulder_ZT = grader.get_amounts_and_tries(cell_data)

		draw_frame(frame)
		draw_zones_and_tops(full_page)
		draw_name_data(full_page)
		draw_attempts_total(full_page)


	def extract_zones_and_tops_area(frame):
		x_ratio_min, x_ratio_max, y_ratio_min, y_ratio_max = ui_areas["tickbox"]
		y_min = int(y_ratio_min * frame.shape[0])
		y_max = int(y_ratio_max * frame.shape[0])
		x_min = int(x_ratio_min * frame.shape[1])
		x_max = int(x_ratio_max * frame.shape[1])

		cutout = frame[y_min:y_max, x_min:x_max]

		return cv2.resize(cutout, (zones_and_tops_width, frame_height), interpolation=cv2.INTER_LINEAR)

	def extract_attempts_total(frame):
		x_ratio_min, x_ratio_max, y_ratio_min, y_ratio_max = ui_areas["attempts_total"]
		y_min = int(y_ratio_min * frame.shape[0])
		y_max = int(y_ratio_max * frame.shape[0])
		x_min = int(x_ratio_min * frame.shape[1])
		x_max = int(x_ratio_max * frame.shape[1])

		cutout = frame[y_min:y_max, x_min:x_max]
		
		return cv2.resize(cutout, (zones_and_tops_width, attempt_totals_height), interpolation=cv2.INTER_LINEAR)

	def extract_name_area(frame):
		x_ratio_min, x_ratio_max, y_ratio_min, y_ratio_max = ui_areas["name"]
		y_min = int(y_ratio_min * frame.shape[0])
		y_max = int(y_ratio_max * frame.shape[0])
		x_min = int(x_ratio_min * frame.shape[1])
		x_max = int(x_ratio_max * frame.shape[1])

		cutout = frame[y_min:y_max, x_min:x_max]

		return cv2.resize(cutout, (name_data_width, name_data_height), interpolation=cv2.INTER_LINEAR)

	def draw_attempts_total(frame):
		global attempts_total_data, amountZT, triesZT
		frame = frame.copy()

		frame = extract_attempts_total(frame)

		zone_x = int(zones_and_tops_width * 0.6)
		top_x = int(zones_and_tops_width * 0.8)
		y_amount = int(0.4 * frame.shape[0] * 0.99)
		y_tries = int(0.8 * frame.shape[0] * 0.99)

		if amountZT is not None:
			cv2.putText(frame, str(amountZT[0]), (zone_x, y_amount), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
			cv2.putText(frame, str(amountZT[1]), (top_x, y_amount), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

		if triesZT is not None:
			cv2.putText(frame, str(triesZT[0]), (zone_x, y_tries), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
			cv2.putText(frame, str(triesZT[1]), (top_x, y_tries), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
		

		try:
			#data = cv2.cvtColor(frame, cv2.COLOR_rGR2RGB)  # because the camera data comes in as BGR and we need RGB
			data = frame
			data = data.flatten()  # flatten camera data to a 1 d stricture
			data = np.float32(data)  # change data type to 32bit floats
			attempts_total_data = np.true_divide(data, 255.0)  # normalize image data to prepare for GPU
			dpg.set_value("attempts_total_texture", attempts_total_data)
		except Exception as e:
			print(f"Error processing file {filename}: {e}")
			return



	def draw_name_data(frame):
		global name_texture_data
		frame = frame.copy()

		frame = extract_name_area(frame)
		
		try:
			#data = cv2.cvtColor(frame, cv2.COLOR_rGR2RGB)  # because the camera data comes in as BGR and we need RGB
			data = frame
			data = data.flatten()  # flatten camera data to a 1 d stricture
			data = np.float32(data)  # change data type to 32bit floats
			name_texture_data = np.true_divide(data, 255.0)  # normalize image data to prepare for GPU
			dpg.set_value("name_texture", name_texture_data)
		except Exception as e:
			print(f"Error processing file {filename}: {e}")
			return

	def draw_zones_and_tops(frame):
		global zones_and_tops_texture_data, amountZT, triesZT, per_boulder_ZT
		frame = frame.copy()

		frame = extract_zones_and_tops_area(frame)

		# Write the zones and tops amounts on the frame
		num_boulders = len(per_boulder_ZT)
		for b in range(num_boulders):
			zone_x = int(zones_and_tops_width * 0.6)
			top_x = int(zones_and_tops_width * 0.8)
			y = int(((b+1) / num_boulders) * frame.shape[0] * 0.99)
			(zone, top) = per_boulder_ZT[b]
			if zone is not None:
				cv2.putText(frame, str(zone), (zone_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
			if top is not None:
				cv2.putText(frame, str(top), (top_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
		
		try:
			#data = cv2.cvtColor(frame, cv2.COLOR_rGR2RGB)  # because the camera data comes in as BGR and we need RGB
			data = frame
			data = data.flatten()  # flatten camera data to a 1 d stricture
			data = np.float32(data)  # change data type to 32bit floats
			zones_and_tops_texture_data = np.true_divide(data, 255.0)  # normalize image data to prepare for GPU
			dpg.set_value("zones_and_tops_texture", zones_and_tops_texture_data)
		except Exception as e:
			print(f"Error processing file {filename}: {e}")
			return
		
	def draw_frame(frame):
		global texture_data
		
		frame = frame.copy()

		frame = draw_grid(frame)

		#frame_height = int(frame.shape[0] * (frame_width / frame.shape[1]))
		#print(frame_height)
		#print(frame.shape)
		frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
		
		try:
			#data = cv2.cvtColor(frame, cv2.COLOR_rGR2RGB)  # because the camera data comes in as BGR and we need RGB
			data = frame
			data = data.flatten()  # flatten camera data to a 1 d stricture
			data = np.float32(data)  # change data type to 32bit floats
			texture_data = np.true_divide(data, 255.0)  # normalize image data to prepare for GPU
			dpg.set_value("texture_tag", texture_data)
		except Exception as e:
			print(f"Error processing file {filename}: {e}")
			return

		
	def draw_grid(frame):
		global row_centers_sorted, col_centers_sorted, med_w, med_h

		original_image = frame.copy()

		num_rows = cell_data.shape[0]
		num_cols = cell_data.shape[1]

		for row in range(num_rows):
			for col in range(num_cols):
				x = int(col_centers_sorted[col])
				y = int(row_centers_sorted[row])

				# Draw a circle around the bubble
				circle_color = (0, 255, 0) if cell_data[row, col] == 1 else (255, 0, 0)
				circle_thickness = 2 if cell_data[row, col] == 1 else 1
				cv2.circle(original_image, (x, y), int(med_w * 0.9), circle_color, circle_thickness)

		return original_image
	
	def on_main_frame_clicked(sender, app_data):
		global cell_data, frame, row_centers_sorted, col_centers_sorted, frame

		# Mouse position in screen space
		mouse_x, mouse_y = dpg.get_mouse_pos()

		# Get image position in screen space
		image_pos = dpg.get_item_rect_min("main_image")

		# Convert to local image coordinates
		local_x = mouse_x - image_pos[0]
		local_y = mouse_y + image_pos[1]

		# Convert UI-scaled coords back to original image coords
		scale_x = frame.shape[1] / frame_width
		scale_y = frame.shape[0] / frame_height

		img_x = int(local_x * scale_x)
		img_y = int(local_y * scale_y)

		# Find the closest row and column
		closest_row = None
		closest_row_distance = None
		closest_col = None
		closest_col_distance = None

		for (row_index, row) in enumerate(row_centers_sorted):
			dist = np.abs(row-img_y)
			if closest_row == None or dist < closest_row_distance:
				closest_row = row_index
				closest_row_distance = dist

		for (col_index, col) in enumerate(col_centers_sorted):
			dist = np.abs(col-img_x)
			if closest_col == None or dist < closest_col_distance:
				closest_col = col_index
				closest_col_distance = dist

		if cell_data[closest_row, closest_col] == 1:
			cell_data[closest_row, closest_col] = 0
		else:
			cell_data[closest_row, closest_col] = 1
		

		draw_data()
		

	def export_to_csv(sender, callback):
		global amountZT, triesZT, filename, per_boulder_ZT
		if filename is None:
			set_status("No active file to export.")
			return

		name = dpg.get_value("user_name")
		sex = "M" if dpg.get_value("is_male") else "V"
		exportString = f"{name},"
		exportString += f"{sex},"
		file_path = Path(filename)
		only_file_name = file_path.name
		exportString += only_file_name
		for i in range(0, len(per_boulder_ZT)):
			(zone, top) = per_boulder_ZT[i]
			if zone is None:
				zone = 0
			if top is None:
				top = 0
			exportString += f",B{i + 1} T{top}Z{zone}"
		exportString += f",{amountZT[1]},{amountZT[0]}"
		exportString += f",{triesZT[1]},{triesZT[0]}"
		csvFile.write(f"{exportString}\n")
		csvFile.flush()

		# Move the png file
		moved_file_name = processed_data_folder / only_file_name
		Path(filename).rename(moved_file_name)
		if filename in fileList:
			fileList.remove(filename)
		filename = None

		dpg.set_value("user_name", "")
		refresh_file_queue()
		get_next_file(False)

	def export_to_ground_truth(sender, callback):
		global cell_data, filename
		if filename is None:
			set_status("No active file to export.")
			return

		filled_cells = []
		for row in range(cell_data.shape[0]):
			for col in range(cell_data.shape[1]):
				if cell_data[row, col] == 1:
					filled_cells.append((row, col))

		pure_file_name = Path(filename).name

		moved_file_name = processed_data_folder / pure_file_name

		output_file_name = processed_data_folder / (Path(pure_file_name).stem + ".csv")

		with open(output_file_name, "w") as f:
			for cell in filled_cells:
				f.write(f"{cell[0]},{cell[1]}\n")


		# Move the png file
		Path(filename).rename(moved_file_name)
		if filename in fileList:
			fileList.remove(filename)
		filename = None

		refresh_file_queue()
		get_next_file(False)

	get_next_file(True)

	with dpg.texture_registry(show=False):
		dpg.add_raw_texture(frame_width, frame_height, texture_data, tag="texture_tag",
							format=dpg.mvFormat_Float_rgb)
		dpg.add_raw_texture(zones_and_tops_width, frame_height, zones_and_tops_texture_data, tag="zones_and_tops_texture",
							format=dpg.mvFormat_Float_rgb)
		dpg.add_raw_texture(name_data_width, name_data_height, name_texture_data, tag="name_texture",
							format=dpg.mvFormat_Float_rgb)
		dpg.add_raw_texture(zones_and_tops_width, attempt_totals_height, attempts_total_data, tag="attempts_total_texture",
							format=dpg.mvFormat_Float_rgb)
							
		
	with dpg.item_handler_registry(tag="image_handler"):
		dpg.add_item_clicked_handler(callback=on_main_frame_clicked)


	with dpg.window(label="resultstester", tag="mainWindow"):
		with dpg.table(header_row=False):
			dpg.add_table_column(width_stretch=True)
			dpg.add_table_column(width_fixed=True, init_width_or_weight=250.0)
			with dpg.table_row():
				with dpg.table_cell():
					dpg.add_image("texture_tag", tag="main_image")
				with dpg.table_cell():
					dpg.add_image("zones_and_tops_texture")
			with dpg.table_row():
				with dpg.table_cell():
					dpg.add_image("name_texture")
				with dpg.table_cell():
					dpg.add_image("attempts_total_texture")
			with dpg.table_row():
				with dpg.table_cell():
					dpg.add_text(f"Naam kandidaat:")
					dpg.add_input_text(tag=f"user_name")
					dpg.add_text(f"Is kandidaat man?")
					dpg.add_checkbox(tag=f"is_male", default_value = True)
					dpg.add_button(label="export", tag="export_button", callback=export_to_csv)
					dpg.add_button(label="export to ground truth", tag="export_ground_truth_button", callback=export_to_ground_truth)
				with dpg.table_cell():
					dpg.add_text("Scan directory")
					dpg.add_input_text(tag="scan_dir_input", default_value=str(to_process_data_folder), width=240)
					dpg.add_button(label="Apply + Refresh", callback=apply_scan_directory_and_refresh)
					dpg.add_button(label="Refresh Queue", callback=refresh_file_queue)
					dpg.add_text("Queue: 0 file(s)", tag="queue_count_text")
					dpg.add_text("Current: -", tag="current_file_text")
					dpg.add_text("Ready", tag="scan_status_text", wrap=240)
					dpg.add_listbox([], tag="queue_list", num_items=10, width=240, callback=on_queue_file_selected)

	refresh_file_queue()
	update_queue_ui()


	dpg.bind_item_handler_registry("main_image", "image_handler")

	dpg.show_viewport()
	dpg.maximize_viewport()
	dpg.set_primary_window("mainWindow", True)
	dpg.start_dearpygui()
	dpg.destroy_context()
