import difflib
import os
import shutil
import textwrap
from pathlib import Path
import dearpygui.dearpygui as dpg
import cv2 as cv
import cv2 as cv2
import numpy as np
import grader
from configs import config as app_config

try:
	import pytesseract
except ImportError:
	pytesseract = None

COLUMNS = 9
ROWS = 20
ANSWERS = 3
CONFIG_FILE_NAME = os.getenv("OMR_CONFIG_NAME", "config-dbiyo2025")
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TESSERACT_ENV_VAR = "TESSERACT_CMD"
COMMON_TESSERACT_PATHS = (
	r"C:\Program Files\Tesseract-OCR\tesseract.exe",
	r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


def resolve_tesseract_cmd():
	explicit_path = os.getenv(TESSERACT_ENV_VAR)
	if explicit_path:
		explicit = Path(explicit_path).expanduser()
		if explicit.exists():
			return str(explicit)

	detected = shutil.which("tesseract")
	if detected:
		return detected

	for candidate in COMMON_TESSERACT_PATHS:
		if Path(candidate).exists():
			return candidate

	return None


def normalize_ocr_name(text):
	cleaned_chars = []
	for char in text.replace("\n", " ").replace("\f", " "):
		if char.isalpha() or char in " -'":
			cleaned_chars.append(char)
		else:
			cleaned_chars.append(" ")

	return " ".join("".join(cleaned_chars).split()).strip()

if __name__ == '__main__':

	cfg = app_config.set_active_config(CONFIG_FILE_NAME)
	tesseract_cmd = resolve_tesseract_cmd()
	if pytesseract is not None and tesseract_cmd is not None:
		pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

	processed_data_folder = Path(cfg.PROCESSED_FILES_DIR)
	to_process_data_folder = Path(cfg.SCANNED_FILES_DIR)
	errored_data_folder = Path(cfg.ERRORED_FILES_DIR)
	results_csv_path = Path(cfg.RESULTS_CSV_PATH)
	ui_areas = cfg.UI_AREAS

	# Config values are already scaled via UI_SCALE and derived from UI_AREAS ratios.
	frame_width = cfg.FRAME_WIDTH
	frame_height = cfg.FRAME_HEIGHT

	attempt_totals_height = cfg.ATTEMPT_TOTALS_HEIGHT
	attempt_totals_width = cfg.ATTEMPT_TOTALS_WIDTH

	zones_and_tops_width = cfg.ZONES_AND_TOPS_WIDTH
	zones_and_tops_height = cfg.ZONES_AND_TOPS_HEIGHT
	zones_and_tops_left_padding = 48
	# Display zones/tops at main-frame height while keeping its original aspect ratio.
	zones_and_tops_display_height = frame_height
	zones_and_tops_base_display_width = max(
		1,
		int(round(zones_and_tops_width * (zones_and_tops_display_height / float(max(1, zones_and_tops_height)))))
	)
	zones_and_tops_display_width = zones_and_tops_base_display_width + zones_and_tops_left_padding
	controls_panel_width = 280
	controls_panel_gap = 16
	side_panel_width = max(zones_and_tops_display_width + controls_panel_gap + controls_panel_width, attempt_totals_width)

	name_data_width = cfg.NAME_DATA_WIDTH
	name_data_height = cfg.NAME_DATA_HEIGHT

	category_data_width = getattr(cfg, "CATEGORY_DATA_WIDTH", 0)
	category_data_height = getattr(cfg, "CATEGORY_DATA_HEIGHT", 0)
	has_category_area = "category" in ui_areas and category_data_width > 0 and category_data_height > 0

	paths = [processed_data_folder, to_process_data_folder, errored_data_folder]
	for p in paths:
		if not p.exists():
			p.mkdir(parents=True, exist_ok=True)
			print(f"Made dir: {p}")

	path = to_process_data_folder
	fileList = []
	filename = None
	last_failed_file = None
	queue_error_map = {}
	queue_error_scan_has_run = False

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
	zones_and_tops_texture_data = np.zeros((zones_and_tops_display_height * zones_and_tops_display_width * 3,), dtype=np.float32)
	name_texture_data = np.zeros((name_data_height * name_data_width * 3,), dtype=np.float32)
	category_texture_data = np.zeros((max(1, category_data_height) * max(1, category_data_width) * 3,), dtype=np.float32)
	attempts_total_data = np.zeros((attempt_totals_height * attempt_totals_width * 3,), dtype=np.float32)
	debug_texture_data = np.zeros((frame_height * frame_width * 3,), dtype=np.float32)
	debug_steps_cache = []
	debug_zoom = {'x0': 0.0, 'y0': 0.0, 'x1': 1.0, 'y1': 1.0}
	debug_drag_start_local = None
	debug_drag_current_local = None
	debug_is_dragging = False
	debug_current_step_img = None

	csvFile = open(results_csv_path, "a")

	dpg.create_context()
	dpg.create_viewport(title='Review scores', width=1400, height=1000)
	dpg.setup_dearpygui()

	with dpg.theme(tag="queue_error_theme"):
		with dpg.theme_component(dpg.mvButton):
			dpg.add_theme_color(dpg.mvThemeCol_Button, (140, 45, 45, 255))
			dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (170, 60, 60, 255))
			dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (120, 35, 35, 255))
			dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 235, 235, 255))

	with dpg.theme(tag="queue_current_theme"):
		with dpg.theme_component(dpg.mvButton):
			dpg.add_theme_color(dpg.mvThemeCol_Button, (45, 95, 135, 255))
			dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (60, 120, 165, 255))
			dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (35, 75, 110, 255))

	def set_status(message):
		if dpg.does_item_exist("scan_status_text"):
			dpg.set_value("scan_status_text", message)
		print(message)

	def set_ocr_population_status(message):
		if dpg.does_item_exist("ocr_population_status_text"):
			dpg.set_value("ocr_population_status_text", message)

	def update_ocr_population_status(name_value, name_status, category_values=None, category_status=None):
		missing_fields = []
		if not name_value:
			missing_fields.append("name")

		if has_category_area:
			is_male_value = None
			age_cat_value = None
			if category_values is not None:
				is_male_value, age_cat_value = category_values
			if is_male_value is None:
				missing_fields.append("gender")
			if not age_cat_value:
				missing_fields.append("age")

		if missing_fields:
			base_message = f"OCR autofill partial (missing: {', '.join(missing_fields)})"
		else:
			base_message = "OCR autofill complete"

		notes = []
		if name_status:
			notes.append(name_status)
		if has_category_area and category_status:
			notes.append(category_status)
		if notes:
			base_message = f"{base_message} | {'; '.join(notes)}"

		set_ocr_population_status(base_message)

	def set_export_buttons_enabled(enabled):
		if dpg.does_item_exist("export_button"):
			dpg.configure_item("export_button", enabled=enabled)
		if dpg.does_item_exist("export_ground_truth_button"):
			dpg.configure_item("export_ground_truth_button", enabled=enabled)

	def set_debug_button_enabled(enabled):
		if dpg.does_item_exist("show_debug_button"):
			dpg.configure_item("show_debug_button", enabled=enabled)

	def set_error_check_button_enabled(enabled):
		if dpg.does_item_exist("error_check_all_button"):
			dpg.configure_item("error_check_all_button", enabled=enabled)

	def refresh_ui_frame():
		if dpg.is_dearpygui_running():
			dpg.split_frame(delay=1)

	def set_error_check_progress(current, total, current_file=None, is_running=False, status_label=None):
		if not dpg.does_item_exist("error_check_progress_text"):
			return

		if total <= 0:
			progress_value = 0.0
			overlay = "0 / 0"
		else:
			progress_value = max(0.0, min(1.0, current / float(total)))
			overlay = f"{current} / {total}"

		if status_label is not None:
			message = status_label
		elif is_running:
			if current_file:
				message = f"Checking {current} / {total}: {Path(current_file).name}"
			else:
				message = f"Checking {current} / {total}"
		else:
			message = "Error check idle"

		dpg.set_value("error_check_progress_text", message)
		dpg.set_value("error_check_progress_bar", progress_value)
		dpg.configure_item("error_check_progress_bar", overlay=overlay)

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
			dpg.set_value("zones_and_tops_texture", np.zeros((zones_and_tops_display_height * zones_and_tops_display_width * 3,), dtype=np.float32))
		if dpg.does_item_exist("name_texture"):
			dpg.set_value("name_texture", np.zeros((name_data_height * name_data_width * 3,), dtype=np.float32))
		if dpg.does_item_exist("category_texture"):
			dpg.set_value("category_texture", np.zeros((max(1, category_data_height) * max(1, category_data_width) * 3,), dtype=np.float32))
		if dpg.does_item_exist("attempts_total_texture"):
			dpg.set_value("attempts_total_texture", np.zeros((attempt_totals_height * attempt_totals_width * 3,), dtype=np.float32))
		set_ocr_population_status("OCR autofill idle")

	def _render_message_image(width, height, title, subtitle=None, bg_color=(0, 0, 0), title_color=(255, 255, 255), subtitle_color=(200, 200, 200)):
		img = np.zeros((height, width, 3), dtype=np.uint8)
		img[:] = bg_color

		title_scale = max(0.8, min(width, height) / 900.0)
		title_th = 2
		(title_w, title_h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_th)
		title_x = max(10, (width - title_w) // 2)
		title_y = max(title_h + 20, height // 2 - 20)
		cv2.putText(img, title, (title_x, title_y), cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_color, title_th)

		if subtitle:
			wrapped = textwrap.wrap(subtitle, width=60)
			sub_scale = max(0.5, title_scale * 0.65)
			sub_th = 1
			line_gap = int(28 * sub_scale)
			start_y = title_y + 30
			for i, line in enumerate(wrapped[:5]):
				(line_w, line_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, sub_scale, sub_th)
				line_x = max(10, (width - line_w) // 2)
				line_y = start_y + i * line_gap
				cv2.putText(img, line, (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, subtitle_color, sub_th)

		return img

	def _set_texture_if_exists(tag, image_bgr, width, height):
		if dpg.does_item_exist(tag):
			dpg.set_value(tag, to_rgb_texture(image_bgr, width, height))

	def show_loading_state(file_path=None):
		name = Path(file_path).name if file_path else ""
		main_msg = _render_message_image(
			frame_width,
			frame_height,
			"Loading...",
			subtitle=name,
			bg_color=(25, 25, 25),
		)
		side_msg = _render_message_image(
			zones_and_tops_display_width,
			zones_and_tops_display_height,
			"Loading...",
			bg_color=(25, 25, 25),
		)
		name_msg = _render_message_image(
			name_data_width,
			name_data_height,
			"Loading...",
			bg_color=(25, 25, 25),
		)
		attempt_msg = _render_message_image(
			attempt_totals_width,
			attempt_totals_height,
			"Loading...",
			bg_color=(25, 25, 25),
		)

		_set_texture_if_exists("texture_tag", main_msg, frame_width, frame_height)
		_set_texture_if_exists("zones_and_tops_texture", side_msg, zones_and_tops_display_width, zones_and_tops_display_height)
		_set_texture_if_exists("name_texture", name_msg, name_data_width, name_data_height)
		if has_category_area:
			cat_msg = _render_message_image(category_data_width, category_data_height, "...", bg_color=(25, 25, 25))
			_set_texture_if_exists("category_texture", cat_msg, category_data_width, category_data_height)
		_set_texture_if_exists("attempts_total_texture", attempt_msg, attempt_totals_width, attempt_totals_height)

		set_export_buttons_enabled(False)
		set_ocr_population_status("OCR autofill in progress...")

	def show_error_state(error_message):
		error_main = _render_message_image(
			frame_width,
			frame_height,
			"Processing Error",
			subtitle=str(error_message),
			bg_color=(35, 35, 60),
			title_color=(220, 220, 255),
			subtitle_color=(220, 220, 220),
		)
		black_side = np.zeros((zones_and_tops_display_height, zones_and_tops_display_width, 3), dtype=np.uint8)
		black_name = np.zeros((name_data_height, name_data_width, 3), dtype=np.uint8)
		black_attempt = np.zeros((attempt_totals_height, attempt_totals_width, 3), dtype=np.uint8)

		_set_texture_if_exists("texture_tag", error_main, frame_width, frame_height)
		_set_texture_if_exists("zones_and_tops_texture", black_side, zones_and_tops_display_width, zones_and_tops_display_height)
		_set_texture_if_exists("name_texture", black_name, name_data_width, name_data_height)
		if has_category_area:
			black_cat = np.zeros((category_data_height, category_data_width, 3), dtype=np.uint8)
			_set_texture_if_exists("category_texture", black_cat, category_data_width, category_data_height)
		_set_texture_if_exists("attempts_total_texture", black_attempt, attempt_totals_width, attempt_totals_height)

		set_export_buttons_enabled(False)
		set_ocr_population_status("OCR autofill not completed (processing error)")

	def update_queue_ui():
		global queue_error_map
		global queue_error_scan_has_run
		global last_failed_file
		if not dpg.does_item_exist("queue_list_container"):
			return

		dpg.delete_item("queue_list_container", children_only=True)
		error_count = 0
		for idx, p in enumerate(reversed(fileList), start=1):
			is_current = (filename is not None and p == filename)
			is_error = p in queue_error_map
			prefix = "* " if is_current else "  "
			label = f"{prefix}{idx:03d} | {Path(p).name}"
			button_tag = f"queue_item_{idx}_{abs(hash(p))}"
			dpg.add_button(
				label=label,
				tag=button_tag,
				parent="queue_list_container",
				width=-1,
				callback=on_queue_file_selected,
				user_data=p,
			)
			if is_error:
				error_count += 1
				dpg.bind_item_theme(button_tag, "queue_error_theme")
			elif is_current:
				dpg.bind_item_theme(button_tag, "queue_current_theme")

		if len(fileList) == 0:
			dpg.add_button(label="<queue empty>", parent="queue_list_container", width=-1, enabled=False)

		if queue_error_scan_has_run:
			dpg.set_value("queue_count_text", f"Queue: {len(fileList)} file(s) | Errors: {error_count}")
		else:
			dpg.set_value("queue_count_text", f"Queue: {len(fileList)} file(s)")
		if filename:
			current_label = Path(filename).name
		elif last_failed_file:
			current_label = f"failed: {Path(last_failed_file).name}"
		else:
			current_label = "-"
		dpg.set_value("current_file_text", f"Current: {current_label}")

		# If nothing is queued/active, present a blank UI and disable exports.
		if len(fileList) == 0 and filename is None:
			clear_display_textures()
			set_export_buttons_enabled(False)
		else:
			set_export_buttons_enabled(filename is not None)

		set_debug_button_enabled(filename is not None or last_failed_file is not None)

	def refresh_file_queue(sender=None, app_data=None):
		global to_process_data_folder
		global filename
		global last_failed_file
		global queue_error_map
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
		queue_error_map = {p: message for p, message in queue_error_map.items() if p in disk_set}

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

		if last_failed_file is not None and last_failed_file not in disk_set:
			last_failed_file = None

		update_queue_ui()
		if not current_removed:
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
		global queue_error_map
		global queue_error_scan_has_run
		new_dir = Path(dpg.get_value("scan_dir_input")).expanduser()
		to_process_data_folder = new_dir
		if not to_process_data_folder.exists():
			to_process_data_folder.mkdir(parents=True, exist_ok=True)
		queue_error_map = {}
		queue_error_scan_has_run = False
		set_error_check_progress(0, 0, is_running=False)
		set_status(f"Using scan directory: {to_process_data_folder}")
		refresh_file_queue()

	def load_file(candidate):
		global filename, last_failed_file, amountZT, triesZT, per_boulder_ZT, frame, data, texture_data, cell_data, row_centers_sorted, col_centers_sorted, med_w, med_h, full_page, queue_error_map

		if not Path(candidate).exists():
			set_status(f"File no longer exists: {Path(candidate).name}")
			if candidate in fileList:
				fileList.remove(candidate)
			update_queue_ui()
			return False

		filename = candidate
		show_loading_state(candidate)
		try:
			filled_cells, (ROWS, COLS), warped_u8, (row_centers_sorted, col_centers_sorted), (med_w, med_h), full_page = grader.grade_score_form(filename, show_plots=False, config_name=CONFIG_FILE_NAME)
		except Exception as e:
			queue_error_map[candidate] = str(e)
			set_status(f"Error reading {Path(filename).name}: {e}")
			show_error_state(f"{Path(filename).name}: {e}")
			last_failed_file = candidate
			filename = None
			update_queue_ui()
			return False

		last_failed_file = None
		queue_error_map.pop(candidate, None)

		cell_data = np.zeros((ROWS, COLS), dtype=np.uint8)
		for (r, c) in filled_cells:
			cell_data[r, c] = 1

		if len(warped_u8.shape) == 2:
			warped_u8 = cv2.cvtColor(warped_u8, cv2.COLOR_GRAY2RGB)
		frame = warped_u8

		draw_data()
		ocr_name, ocr_status = autofill_name_from_frame(full_page)
		category_values = None
		category_status = None
		if has_category_area:
			is_male_value, age_cat_value, category_status = autofill_category_from_frame(full_page)
			category_values = (is_male_value, age_cat_value)
		update_ocr_population_status(ocr_name, ocr_status, category_values, category_status)
		update_queue_ui()
		set_export_buttons_enabled(True)
		if ocr_name:
			set_status(f"Loaded: {Path(filename).name} | OCR: {ocr_name}")
		elif ocr_status:
			set_status(f"Loaded: {Path(filename).name} | {ocr_status}")
		else:
			set_status(f"Loaded: {Path(filename).name}")
		return True

	def on_queue_file_selected(sender, app_data, user_data):
		selected_path = user_data
		if not selected_path:
			return

		load_file(selected_path)

	def error_check_all_queued_files(sender=None, app_data=None):
		global queue_error_map
		global queue_error_scan_has_run

		if len(fileList) == 0:
			set_error_check_progress(0, 0, is_running=False)
			set_status("Queue is empty. Nothing to error-check.")
			return

		checked_paths = list(fileList)
		failed_paths = []
		queue_error_scan_has_run = True
		set_error_check_button_enabled(False)
		set_error_check_progress(0, len(checked_paths), is_running=True)
		set_status(f"Starting error check for {len(checked_paths)} queued file(s)...")
		refresh_ui_frame()

		try:
			for index, candidate in enumerate(checked_paths, start=1):
				set_error_check_progress(index, len(checked_paths), current_file=candidate, is_running=True)
				set_status(f"Checking file {index} / {len(checked_paths)}: {Path(candidate).name}")
				refresh_ui_frame()

				if not Path(candidate).exists():
					queue_error_map[candidate] = "File no longer exists"
					failed_paths.append(candidate)
					continue

				try:
					grader.grade_score_form(candidate, show_plots=False, config_name=CONFIG_FILE_NAME)
				except Exception as e:
					queue_error_map[candidate] = str(e)
					failed_paths.append(candidate)
				else:
					queue_error_map.pop(candidate, None)
		finally:
			update_queue_ui()
			set_error_check_button_enabled(True)

		if failed_paths:
			failed_names = ", ".join(Path(p).name for p in failed_paths[:3])
			if len(failed_paths) > 3:
				failed_names += ", ..."
			completion_message = (
				f"Error check complete: {len(failed_paths)} of {len(checked_paths)} queued file(s) failed. {failed_names}"
			)
			set_status(completion_message)
			set_error_check_progress(
				len(checked_paths),
				len(checked_paths),
				is_running=False,
				status_label=f"Completed: {len(failed_paths)} failed of {len(checked_paths)} checked",
			)
		else:
			completion_message = f"Error check complete: all {len(checked_paths)} queued file(s) loaded successfully."
			set_status(completion_message)
			set_error_check_progress(
				len(checked_paths),
				len(checked_paths),
				is_running=False,
				status_label=f"Completed: all {len(checked_paths)} files passed",
			)

	def to_rgb_texture(image_bgr, width, height):
		img = image_bgr
		if len(img.shape) == 2:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

		# Fit image into the texture while preserving aspect ratio (letterbox).
		h_src, w_src = img.shape[:2]
		scale = min(width / float(max(1, w_src)), height / float(max(1, h_src)))
		new_w = max(1, int(round(w_src * scale)))
		new_h = max(1, int(round(h_src * scale)))
		resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

		canvas = np.zeros((height, width, 3), dtype=np.uint8)
		x0 = (width - new_w) // 2
		y0 = (height - new_h) // 2
		canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

		rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
		flat = rgb.flatten().astype(np.float32)
		return np.true_divide(flat, 255.0)

	def show_debug_step(step_index):
		global debug_current_step_img
		if not debug_steps_cache:
			return
		idx = max(0, min(step_index, len(debug_steps_cache) - 1))
		title, img = debug_steps_cache[idx]
		debug_current_step_img = img
		dpg.set_value("debug_step_title", title)
		render_debug_with_zoom()

	def on_debug_step_selected(sender, app_data):
		global debug_zoom
		selected_label = dpg.get_value("debug_step_list")
		if not selected_label:
			return
		for idx, (title, _) in enumerate(debug_steps_cache):
			if selected_label.endswith(title):
				debug_zoom = {'x0': 0.0, 'y0': 0.0, 'x1': 1.0, 'y1': 1.0}
				show_debug_step(idx)
				return

	def to_debug_texture(img_bgr, overlay_rect=None):
		"""Render debug image with preserved aspect ratio and optional display-space overlay."""
		if img_bgr is None:
			return np.zeros((frame_height * frame_width * 3,), dtype=np.float32)

		img = img_bgr
		if len(img.shape) == 2:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

		h_src, w_src = img.shape[:2]
		scale = min(frame_width / float(max(1, w_src)), frame_height / float(max(1, h_src)))
		new_w = max(1, int(round(w_src * scale)))
		new_h = max(1, int(round(h_src * scale)))
		resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

		canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
		x0 = (frame_width - new_w) // 2
		y0 = (frame_height - new_h) // 2
		canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

		if overlay_rect is not None:
			sx, sy, ex, ey = overlay_rect
			sx = int(max(0, min(sx, frame_width - 1)))
			sy = int(max(0, min(sy, frame_height - 1)))
			ex = int(max(0, min(ex, frame_width - 1)))
			ey = int(max(0, min(ey, frame_height - 1)))
			cv2.rectangle(canvas, (min(sx, ex), min(sy, ey)), (max(sx, ex), max(sy, ey)), (0, 255, 255), 2)

		rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
		return np.true_divide(rgb.flatten().astype(np.float32), 255.0)

	def get_debug_display_rect(img_w, img_h):
		scale = min(frame_width / float(max(1, img_w)), frame_height / float(max(1, img_h)))
		display_w = max(1.0, img_w * scale)
		display_h = max(1.0, img_h * scale)
		offset_x = (frame_width - display_w) / 2.0
		offset_y = (frame_height - display_h) / 2.0
		return offset_x, offset_y, display_w, display_h

	def map_debug_local_to_zoom(local_x, local_y, clamp_to_image=False):
		if debug_current_step_img is None:
			return None

		img_h, img_w = debug_current_step_img.shape[:2]
		x0, y0 = debug_zoom['x0'], debug_zoom['y0']
		x1, y1 = debug_zoom['x1'], debug_zoom['y1']

		crop_w = max(1, int((x1 - x0) * img_w))
		crop_h = max(1, int((y1 - y0) * img_h))
		dx0, dy0, dw, dh = get_debug_display_rect(crop_w, crop_h)

		if clamp_to_image:
			local_x = max(dx0, min(local_x, dx0 + dw))
			local_y = max(dy0, min(local_y, dy0 + dh))
		else:
			if local_x < dx0 or local_x > (dx0 + dw) or local_y < dy0 or local_y > (dy0 + dh):
				return None

		u = (local_x - dx0) / max(1e-9, dw)
		v = (local_y - dy0) / max(1e-9, dh)
		mapped_x = x0 + u * (x1 - x0)
		mapped_y = y0 + v * (y1 - y0)
		return max(0.0, min(mapped_x, 1.0)), max(0.0, min(mapped_y, 1.0))

	def update_debug_coords_text(coords=None, prefix="View"):
		if not dpg.does_item_exist("debug_coords_input"):
			return

		if coords is None:
			x_min, y_min, x_max, y_max = debug_zoom['x0'], debug_zoom['y0'], debug_zoom['x1'], debug_zoom['y1']
		else:
			x_min, y_min, x_max, y_max = coords

		x_min, x_max = min(x_min, x_max), max(x_min, x_max)
		y_min, y_max = min(y_min, y_max), max(y_min, y_max)
		coord_tuple = f"({x_min:.4f}, {x_max:.4f}, {y_min:.4f}, {y_max:.4f})"
		dpg.set_value("debug_coords_mode_input", f"{prefix} relative coords (x_min, x_max, y_min, y_max)")
		dpg.set_value("debug_coords_input", coord_tuple)

	def on_debug_coords_input_change(sender, app_data):
		global debug_zoom
		text = app_data.strip()
		# Try to parse the input as a tuple: (x_min, x_max, y_min, y_max)
		try:
			# Remove parentheses and split by comma
			text = text.strip("()")
			values = [float(v.strip()) for v in text.split(",")]
			if len(values) != 4:
				return
			x_min, x_max, y_min, y_max = values
			# Clamp to [0, 1] and ensure min < max
			x_min = max(0.0, min(x_min, 1.0))
			x_max = max(0.0, min(x_max, 1.0))
			y_min = max(0.0, min(y_min, 1.0))
			y_max = max(0.0, min(y_max, 1.0))
			if x_min >= x_max or y_min >= y_max:
				return
			debug_zoom = {'x0': x_min, 'y0': y_min, 'x1': x_max, 'y1': y_max}
			render_debug_with_zoom()
		except (ValueError, IndexError):
			# Invalid input, silently ignore
			pass

	def render_debug_with_zoom(draw_drag_rect=False):
		if debug_current_step_img is None or not dpg.does_item_exist("debug_texture"):
			return

		img = debug_current_step_img
		h, w = img.shape[:2]

		x0 = int(debug_zoom['x0'] * w)
		y0 = int(debug_zoom['y0'] * h)
		x1 = int(debug_zoom['x1'] * w)
		y1 = int(debug_zoom['y1'] * h)

		x0 = max(0, min(x0, w - 1))
		y0 = max(0, min(y0, h - 1))
		x1 = max(x0 + 1, min(x1, w))
		y1 = max(y0 + 1, min(y1, h))

		cropped = img[y0:y1, x0:x1].copy()

		overlay_rect = None
		if draw_drag_rect and debug_drag_start_local and debug_drag_current_local:
			crop_h, crop_w = cropped.shape[:2]
			dx0, dy0, dw, dh = get_debug_display_rect(crop_w, crop_h)

			sx = max(dx0, min(debug_drag_start_local[0], dx0 + dw))
			sy = max(dy0, min(debug_drag_start_local[1], dy0 + dh))
			ex = max(dx0, min(debug_drag_current_local[0], dx0 + dw))
			ey = max(dy0, min(debug_drag_current_local[1], dy0 + dh))
			overlay_rect = (sx, sy, ex, ey)

			p0 = map_debug_local_to_zoom(sx, sy, clamp_to_image=True)
			p1 = map_debug_local_to_zoom(ex, ey, clamp_to_image=True)
			if p0 is not None and p1 is not None:
				update_debug_coords_text((p0[0], p0[1], p1[0], p1[1]), prefix="Selection")
			else:
				update_debug_coords_text(prefix="View")
		else:
			update_debug_coords_text(prefix="View")

		dpg.set_value("debug_texture", to_debug_texture(cropped, overlay_rect=overlay_rect))

	def reset_debug_zoom():
		global debug_zoom
		debug_zoom = {'x0': 0.0, 'y0': 0.0, 'x1': 1.0, 'y1': 1.0}
		render_debug_with_zoom()

	def on_debug_scroll(sender, app_data):
		global debug_zoom
		if not dpg.does_item_exist("debug_image") or not dpg.is_item_hovered("debug_image"):
			return
		if debug_current_step_img is None:
			return

		wheel_delta = app_data
		scale_factor = 0.8 if wheel_delta > 0 else 1.25

		mouse_x, mouse_y = dpg.get_mouse_pos()
		img_pos = dpg.get_item_rect_min("debug_image")
		local_x = mouse_x - img_pos[0]
		local_y = mouse_y - img_pos[1]

		mapped_point = map_debug_local_to_zoom(local_x, local_y, clamp_to_image=False)
		if mapped_point is None:
			return
		ox, oy = mapped_point

		x0, y0 = debug_zoom['x0'], debug_zoom['y0']
		x1, y1 = debug_zoom['x1'], debug_zoom['y1']

		new_x0 = max(0.0, ox - (ox - x0) * scale_factor)
		new_x1 = min(1.0, ox + (x1 - ox) * scale_factor)
		new_y0 = max(0.0, oy - (oy - y0) * scale_factor)
		new_y1 = min(1.0, oy + (y1 - oy) * scale_factor)

		if (new_x1 - new_x0) < 0.01 or (new_y1 - new_y0) < 0.01:
			return

		debug_zoom = {'x0': new_x0, 'y0': new_y0, 'x1': new_x1, 'y1': new_y1}
		render_debug_with_zoom()

	def on_debug_mouse_down(sender, app_data):
		global debug_drag_start_local, debug_is_dragging, debug_drag_current_local
		if debug_is_dragging:
			return
		if not dpg.does_item_exist("debug_image") or not dpg.is_item_hovered("debug_image"):
			return
		if debug_current_step_img is None:
			return

		mouse_x, mouse_y = dpg.get_mouse_pos()
		img_pos = dpg.get_item_rect_min("debug_image")
		local_x = mouse_x - img_pos[0]
		local_y = mouse_y - img_pos[1]

		if map_debug_local_to_zoom(local_x, local_y, clamp_to_image=False) is None:
			return

		debug_drag_start_local = (local_x, local_y)
		debug_drag_current_local = (local_x, local_y)
		debug_is_dragging = True

	def on_debug_mouse_move(sender, app_data):
		global debug_drag_current_local
		if not debug_is_dragging or debug_drag_start_local is None:
			return
		if not dpg.does_item_exist("debug_image"):
			return

		mouse_x, mouse_y = dpg.get_mouse_pos()
		img_pos = dpg.get_item_rect_min("debug_image")
		local_x = mouse_x - img_pos[0]
		local_y = mouse_y - img_pos[1]

		clamped_point = map_debug_local_to_zoom(local_x, local_y, clamp_to_image=True)
		if clamped_point is None:
			return

		img_h, img_w = debug_current_step_img.shape[:2]
		x0, y0 = debug_zoom['x0'], debug_zoom['y0']
		x1, y1 = debug_zoom['x1'], debug_zoom['y1']
		crop_w = max(1, int((x1 - x0) * img_w))
		crop_h = max(1, int((y1 - y0) * img_h))
		dx0, dy0, dw, dh = get_debug_display_rect(crop_w, crop_h)
		mx, my = clamped_point
		u = (mx - x0) / max(1e-9, (x1 - x0))
		v = (my - y0) / max(1e-9, (y1 - y0))
		debug_drag_current_local = (dx0 + u * dw, dy0 + v * dh)
		render_debug_with_zoom(draw_drag_rect=True)

	def on_debug_mouse_release(sender, app_data):
		global debug_zoom, debug_drag_start_local, debug_drag_current_local, debug_is_dragging
		if not debug_is_dragging:
			return
		debug_is_dragging = False

		if debug_drag_start_local is None or debug_drag_current_local is None:
			debug_drag_start_local = None
			debug_drag_current_local = None
			return

		sx, sy = debug_drag_start_local
		ex, ey = debug_drag_current_local

		# Ignore tiny drags (treat as plain click, no zoom)
		if abs(ex - sx) < 10 or abs(ey - sy) < 10:
			debug_drag_start_local = None
			debug_drag_current_local = None
			render_debug_with_zoom()
			return

		p0 = map_debug_local_to_zoom(sx, sy, clamp_to_image=True)
		p1 = map_debug_local_to_zoom(ex, ey, clamp_to_image=True)
		if p0 is None or p1 is None:
			debug_drag_start_local = None
			debug_drag_current_local = None
			render_debug_with_zoom()
			return

		drag_x0 = min(p0[0], p1[0])
		drag_y0 = min(p0[1], p1[1])
		drag_x1 = max(p0[0], p1[0])
		drag_y1 = max(p0[1], p1[1])

		if (drag_x1 - drag_x0) < 0.01 or (drag_y1 - drag_y0) < 0.01:
			debug_drag_start_local = None
			debug_drag_current_local = None
			render_debug_with_zoom()
			return

		debug_zoom = {'x0': drag_x0, 'y0': drag_y0, 'x1': drag_x1, 'y1': drag_y1}
		debug_drag_start_local = None
		debug_drag_current_local = None
		render_debug_with_zoom()

	def show_debug_screen(sender, app_data):
		global debug_steps_cache
		target_file = filename if filename is not None else last_failed_file
		if target_file is None:
			set_status("No active file for debug view.")
			return

		try:
			_, _, _, _, _, _, debug_steps = grader.grade_score_form(
				target_file,
				show_plots=False,
				config_name=CONFIG_FILE_NAME,
				debug_mode=True,
				return_debug_steps=True,
			)
		except grader.GradingDebugError as e:
			debug_steps = e.debug_steps
			set_status(f"Debug captured from failed run: {e}")
		except Exception as e:
			set_status(f"Could not build debug view: {e}")
			return

		if not debug_steps:
			set_status("No debug steps captured for this form.")
			return

		debug_steps_cache = debug_steps
		step_labels = [f"{i+1:02d} | {title}" for i, (title, _) in enumerate(debug_steps_cache)]

		# Reset zoom for the new debug session
		debug_zoom['x0'] = 0.0
		debug_zoom['y0'] = 0.0
		debug_zoom['x1'] = 1.0
		debug_zoom['y1'] = 1.0

		if not dpg.does_item_exist("debug_window"):
			with dpg.window(label="Debug Pipeline", tag="debug_window", width=1400, height=900, show=False):
				dpg.add_text("Debug Pipeline", tag="debug_step_title")
				with dpg.group(horizontal=True):
					with dpg.group():
						dpg.add_listbox(step_labels, tag="debug_step_list", num_items=12, width=320, callback=on_debug_step_selected)
						dpg.add_button(label="Reset Zoom", callback=lambda s, a: reset_debug_zoom())
					with dpg.group():
						dpg.add_input_text(default_value="View relative coords (x_min, x_max, y_min, y_max)", tag="debug_coords_mode_input", width=900, readonly=True)
						dpg.add_input_text(default_value="(0.0000, 1.0000, 0.0000, 1.0000)", tag="debug_coords_input", width=900, callback=on_debug_coords_input_change)
						dpg.add_image("debug_texture", tag="debug_image")
		else:
			dpg.configure_item("debug_step_list", items=step_labels)

		dpg.set_value("debug_step_list", step_labels[0])
		show_debug_step(0)
		dpg.configure_item("debug_window", label=f"Debug Pipeline - {Path(target_file).name}", show=True)

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
		if has_category_area:
			draw_category_data(full_page)
		draw_attempts_total(full_page)


	def extract_zones_and_tops_area(frame):
		x_ratio_min, x_ratio_max, y_ratio_min, y_ratio_max = ui_areas["tickbox"]
		y_min = int(y_ratio_min * frame.shape[0])
		y_max = int(y_ratio_max * frame.shape[0])
		x_min = int(x_ratio_min * frame.shape[1])
		x_max = int(x_ratio_max * frame.shape[1])

		cutout = frame[y_min:y_max, x_min:x_max]
		resized_cutout = cv2.resize(
			cutout,
			(zones_and_tops_base_display_width, zones_and_tops_display_height),
			interpolation=cv2.INTER_LINEAR,
		)

		canvas = np.full((zones_and_tops_display_height, zones_and_tops_display_width, 3), 255, dtype=np.uint8)
		canvas[:, zones_and_tops_left_padding:zones_and_tops_left_padding + zones_and_tops_base_display_width] = resized_cutout
		return canvas

	def extract_attempts_total(frame):
		x_ratio_min, x_ratio_max, y_ratio_min, y_ratio_max = ui_areas["attempts_total"]
		y_min = int(y_ratio_min * frame.shape[0])
		y_max = int(y_ratio_max * frame.shape[0])
		x_min = int(x_ratio_min * frame.shape[1])
		x_max = int(x_ratio_max * frame.shape[1])

		cutout = frame[y_min:y_max, x_min:x_max]
		
		return cv2.resize(cutout, (attempt_totals_width, attempt_totals_height), interpolation=cv2.INTER_LINEAR)

	def extract_name_area(frame):
		x_ratio_min, x_ratio_max, y_ratio_min, y_ratio_max = ui_areas["name"]
		y_min = int(y_ratio_min * frame.shape[0])
		y_max = int(y_ratio_max * frame.shape[0])
		x_min = int(x_ratio_min * frame.shape[1])
		x_max = int(x_ratio_max * frame.shape[1])

		cutout = frame[y_min:y_max, x_min:x_max]

		return cv2.resize(cutout, (name_data_width, name_data_height), interpolation=cv2.INTER_LINEAR)

	def preprocess_name_for_ocr(frame):
		if len(frame.shape) == 3:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		else:
			gray = frame.copy()

		gray = cv2.GaussianBlur(gray, (3, 3), 0)
		gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
		_, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		return cv2.copyMakeBorder(thresholded, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

	def read_name_from_image(frame):
		if pytesseract is None:
			return "", "name OCR unavailable: install pytesseract"
		if tesseract_cmd is None:
			return "", f"name OCR unavailable: set {TESSERACT_ENV_VAR} or install Tesseract"

		processed = preprocess_name_for_ocr(frame)
		ocr_candidates = []
		for config in ("--oem 3 --psm 7", "--oem 3 --psm 6"):
			try:
				candidate = normalize_ocr_name(pytesseract.image_to_string(processed, config=config))
			except Exception as e:
				print(f"Name OCR failed for {filename}: {e}")
				return "", "name OCR failed"
			if candidate:
				ocr_candidates.append(candidate)

		if not ocr_candidates:
			return "", "name OCR found no text"

		best_match = max(ocr_candidates, key=len)
		return best_match, None

	def autofill_name_from_frame(frame):
		name_crop = extract_name_area(frame)
		ocr_name, ocr_status = read_name_from_image(name_crop)
		if dpg.does_item_exist("user_name"):
			dpg.set_value("user_name", ocr_name)
		return ocr_name, ocr_status

	def extract_category_area(frame):
		x_ratio_min, x_ratio_max, y_ratio_min, y_ratio_max = ui_areas["category"]
		y_min = int(y_ratio_min * frame.shape[0])
		y_max = int(y_ratio_max * frame.shape[0])
		x_min = int(x_ratio_min * frame.shape[1])
		x_max = int(x_ratio_max * frame.shape[1])
		cutout = frame[y_min:y_max, x_min:x_max]
		return cv2.resize(cutout, (category_data_width, category_data_height), interpolation=cv2.INTER_LINEAR)

	_AGE_CATEGORIES = ["U15", "U17", "U19", "U21"]

	def read_category_from_image(frame):
		"""Returns (is_male: bool|None, age_cat: str|None, status: str|None)."""
		if pytesseract is None:
			return None, None, "category OCR unavailable: install pytesseract"
		if tesseract_cmd is None:
			return None, None, f"category OCR unavailable: set {TESSERACT_ENV_VAR} or install Tesseract"

		processed = preprocess_name_for_ocr(frame)
		try:
			raw = pytesseract.image_to_string(processed, config="--oem 3 --psm 6")
		except Exception as e:
			return None, None, f"category OCR failed: {e}"

		text_upper = raw.upper()

		# Determine gender: check female first to avoid "male" matching inside "female".
		is_male = None
		if "FEMALE" in text_upper:
			is_male = False
		elif "MALE" in text_upper:
			is_male = True

		# Determine age category: exact substring match first, then closest word.
		age_cat = None
		for cat in _AGE_CATEGORIES:
			if cat in text_upper:
				age_cat = cat
				break
		if age_cat is None:
			for word in text_upper.split():
				matches = difflib.get_close_matches(word, _AGE_CATEGORIES, n=1, cutoff=0.6)
				if matches:
					age_cat = matches[0]
					break

		return is_male, age_cat, None

	def autofill_category_from_frame(frame):
		if not has_category_area:
			return
		cat_crop = extract_category_area(frame)
		is_male, age_cat, status = read_category_from_image(cat_crop)
		if is_male is not None and dpg.does_item_exist("is_male"):
			dpg.set_value("is_male", is_male)
		if age_cat is not None and dpg.does_item_exist("age_category"):
			dpg.set_value("age_category", age_cat)
		return is_male, age_cat, status

	def draw_category_data(frame):
		global category_texture_data
		if not has_category_area:
			return
		frame = frame.copy()
		frame = extract_category_area(frame)
		try:
			data = frame.flatten()
			data = np.float32(data)
			category_texture_data = np.true_divide(data, 255.0)
			dpg.set_value("category_texture", category_texture_data)
		except Exception as e:
			print(f"Error drawing category data: {e}")

	def draw_attempts_total(frame):
		global attempts_total_data, amountZT, triesZT
		frame = frame.copy()

		frame = extract_attempts_total(frame)

		zone_x = int(attempt_totals_width * 0.45)
		top_x = int(attempt_totals_width * 0.85)
		y_amount = int(0.4 * frame.shape[0] * 0.99)
		y_tries = int(0.8 * frame.shape[0] * 0.99)

		if amountZT is not None:
			cv2.putText(frame, str(amountZT[0]), (zone_x, y_amount), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
			cv2.putText(frame, str(triesZT[0]), (top_x, y_amount), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

		if triesZT is not None:
			cv2.putText(frame, str(amountZT[1]), (zone_x, y_tries), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
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
		left_label_x = max(6, int(frame.shape[1] * 0.02))

		# Write the zones and tops amounts on the frame
		num_boulders = len(per_boulder_ZT)
		for b in range(num_boulders):
			zone_x = int(zones_and_tops_width * 1.5)
			top_x = int(zones_and_tops_width * 1.7)
			y = int(((b+1) / num_boulders) * frame.shape[0] * 0.99)
			cv2.putText(frame, str(b + 1), (left_label_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
			(zone, top) = per_boulder_ZT[b]
			if zone is not None:
				cv2.putText(frame, str(zone), (zone_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
			if top is not None:
				cv2.putText(frame, str(top), (top_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
		
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
		left_label_x = max(6, int(med_w * 0.2))

		num_rows = cell_data.shape[0]
		num_cols = cell_data.shape[1]

		for row in range(num_rows):
			row_y = int(row_centers_sorted[row])
			cv2.putText(original_image, str(row + 1), (left_label_x, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
			for col in range(num_cols):
				x = int(col_centers_sorted[col])
				y = row_y

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
		global amountZT, triesZT, filename, last_failed_file, per_boulder_ZT, queue_error_map
		if filename is None:
			set_status("No active file to export.")
			return

		name = dpg.get_value("user_name")
		sex = "M" if dpg.get_value("is_male") else "V"
		age_cat = dpg.get_value("age_category") if dpg.does_item_exist("age_category") else ""
		exportString = f"{name},"
		exportString += f"{sex},"
		exportString += f"{age_cat},"
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
		queue_error_map.pop(str(file_path), None)
		filename = None
		last_failed_file = None

		dpg.set_value("user_name", "")
		refresh_file_queue()
		get_next_file(False)

	def export_to_ground_truth(sender, callback):
		global cell_data, filename, last_failed_file, queue_error_map
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
		queue_error_map.pop(str(Path(filename)), None)
		filename = None
		last_failed_file = None

		refresh_file_queue()
		get_next_file(False)

	with dpg.texture_registry(show=False):
		dpg.add_raw_texture(frame_width, frame_height, texture_data, tag="texture_tag",
							format=dpg.mvFormat_Float_rgb)
		dpg.add_raw_texture(zones_and_tops_display_width, zones_and_tops_display_height, zones_and_tops_texture_data, tag="zones_and_tops_texture",
							format=dpg.mvFormat_Float_rgb)
		dpg.add_raw_texture(name_data_width, name_data_height, name_texture_data, tag="name_texture",
							format=dpg.mvFormat_Float_rgb)
		dpg.add_raw_texture(max(1, category_data_width), max(1, category_data_height), category_texture_data, tag="category_texture",
							format=dpg.mvFormat_Float_rgb)
		dpg.add_raw_texture(attempt_totals_width, attempt_totals_height, attempts_total_data, tag="attempts_total_texture",
							format=dpg.mvFormat_Float_rgb)
		dpg.add_raw_texture(frame_width, frame_height, debug_texture_data, tag="debug_texture",
							format=dpg.mvFormat_Float_rgb)
							
		
	with dpg.item_handler_registry(tag="image_handler"):
		dpg.add_item_clicked_handler(callback=on_main_frame_clicked)


	with dpg.window(label="resultstester", tag="mainWindow"):
		with dpg.table(header_row=False):
			dpg.add_table_column(width_fixed=True, init_width_or_weight=float(frame_width))
			dpg.add_table_column(width_fixed=True, init_width_or_weight=float(side_panel_width))
			with dpg.table_row():
				with dpg.table_cell():
					dpg.add_image("texture_tag", tag="main_image")
				with dpg.table_cell():
					with dpg.group(horizontal=True):
						dpg.add_image("zones_and_tops_texture")
						dpg.add_spacer(width=controls_panel_gap)
						with dpg.group():
							dpg.add_text(f"Name contestant:")
							dpg.add_input_text(tag=f"user_name")
							with dpg.group(horizontal=True):
								dpg.add_text(f"Is contestant male?")
								dpg.add_checkbox(tag=f"is_male", default_value=True)
							with dpg.group(horizontal=True):
								dpg.add_text("Category:")
								dpg.add_combo(_AGE_CATEGORIES, tag="age_category", default_value=_AGE_CATEGORIES[0], width=80)
							dpg.add_spacer(height=15)
							dpg.add_button(label="export", tag="export_button", callback=export_to_csv)
							dpg.add_button(label="export to ground truth", tag="export_ground_truth_button", callback=export_to_ground_truth)
							dpg.add_button(label="Show Debug Screen", tag="show_debug_button", callback=show_debug_screen)
							dpg.add_spacer(height=15)
							dpg.add_text("Scan directory")
							dpg.add_input_text(tag="scan_dir_input", default_value=str(to_process_data_folder), width=240)
							dpg.add_button(label="Apply + Refresh", callback=apply_scan_directory_and_refresh)
							dpg.add_button(label="Refresh Queue", callback=refresh_file_queue)
							dpg.add_spacer(height=15)
							dpg.add_text("Queue: 0 file(s)", tag="queue_count_text")
							dpg.add_text("Ready", tag="scan_status_text", wrap=240)
							dpg.add_text("Current: -", tag="current_file_text")
							with dpg.child_window(tag="queue_list_container", width=240, height=220, border=True):
								pass
							dpg.add_spacer(height=60)
							dpg.add_text("OCR autofill status:")
							dpg.add_input_text(tag="ocr_population_status_text", default_value="OCR autofill idle", width=240, readonly=True)
							dpg.add_spacer(height=20)
							dpg.add_button(label="Error check ALL (will stall UI)", tag="error_check_all_button", callback=error_check_all_queued_files)
							dpg.add_text("Error check idle", tag="error_check_progress_text", wrap=240)
							dpg.add_progress_bar(default_value=0.0, tag="error_check_progress_bar", width=240, overlay="0 / 0")
			with dpg.table_row():
				with dpg.table_cell():
					dpg.add_image("name_texture")
					if has_category_area:
						dpg.add_image("category_texture")
				with dpg.table_cell():
					with dpg.group(horizontal=True):
						dpg.add_image("attempts_total_texture")

	show_loading_state("Starting up...")
	refresh_file_queue()
	update_queue_ui()
	if filename is None and last_failed_file is None:
		get_next_file(True)


	dpg.bind_item_handler_registry("main_image", "image_handler")

	with dpg.handler_registry():
		dpg.add_mouse_wheel_handler(callback=on_debug_scroll)
		dpg.add_mouse_down_handler(button=0, callback=on_debug_mouse_down)
		dpg.add_mouse_release_handler(button=0, callback=on_debug_mouse_release)
		dpg.add_mouse_move_handler(callback=on_debug_mouse_move)

	dpg.show_viewport()
	dpg.maximize_viewport()
	dpg.set_primary_window("mainWindow", True)
	dpg.start_dearpygui()
	dpg.destroy_context()
