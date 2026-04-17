import sys
import textwrap
from pathlib import Path
import dearpygui.dearpygui as dpg
import cv2
import numpy as np
import grader
from configs import config
import ui_state
import ui_backend
import debug_pipeline

ui_backend.setup(sys.modules[__name__])

#debug_texture_data = np.zeros((frame_height * frame_width * 3,), dtype=np.float32)
debug_zoom = {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}
debug_drag_start_local = None
debug_drag_current_local = None
debug_is_dragging = False
debug_current_step_img = None

dpg.create_context()
dpg.create_viewport(title="Review scores", width=1400, height=1000)
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

with dpg.theme(tag="ocr_status_red_theme"):
	with dpg.theme_component(dpg.mvInputText):
		dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (60, 40, 40, 255))
		dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 235, 235, 255))

with dpg.theme(tag="ocr_status_yellow_theme"):
	with dpg.theme_component(dpg.mvInputText):
		dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (60, 55, 40, 255))
		dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 245, 210, 255))

with dpg.theme(tag="ocr_status_green_theme"):
	with dpg.theme_component(dpg.mvInputText):
		dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (40, 60, 45, 255))
		dpg.add_theme_color(dpg.mvThemeCol_Text, (225, 255, 235, 255))

def set_ocr_status_bar(state):
	if not dpg.does_item_exist("ocr_population_status_text"):
		return

	if state == "in_progress":
		dpg.bind_item_theme("ocr_population_status_text", "ocr_status_red_theme")
	elif state == "success":
		dpg.bind_item_theme("ocr_population_status_text", "ocr_status_green_theme")
	else:
		dpg.bind_item_theme("ocr_population_status_text", "ocr_status_yellow_theme")

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

	if ui_state.get_loaded_data().has_category_area:
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
	if ui_state.get_loaded_data().has_category_area and category_status:
		notes.append(category_status)
	if notes:
		base_message = f"{base_message} | {'; '.join(notes)}"

	set_ocr_population_status(base_message)
	if missing_fields or notes:
		set_ocr_status_bar("warning")
	else:
		set_ocr_status_bar("success")

def set_export_buttons_enabled(enabled):
	if dpg.does_item_exist("export_button"):
		dpg.configure_item("export_button", enabled=enabled)
	if dpg.does_item_exist("export_ground_truth_button"):
		dpg.configure_item("export_ground_truth_button", enabled=enabled)

def set_debug_button_enabled(enabled):
	if dpg.does_item_exist("show_debug_button"):
		dpg.configure_item("show_debug_button", enabled=enabled)
	if dpg.does_item_exist("toggle_all_bubbles_button"):
		dpg.configure_item("toggle_all_bubbles_button", enabled=enabled)

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
	ui_state.get_loaded_data().clear_textures()

	if dpg.does_item_exist("texture_tag"):
		dpg.set_value("texture_tag", np.zeros((ui_state.get_loaded_data().bubble_grid_width * ui_state.get_loaded_data().bubble_grid_height * 3,), dtype=np.float32))
	if dpg.does_item_exist("zones_and_tops_texture"):
		dpg.set_value("zones_and_tops_texture", np.zeros((ui_state.get_loaded_data().zones_and_tops_display_height * ui_state.get_loaded_data().zones_and_tops_display_width * 3,), dtype=np.float32))
	if dpg.does_item_exist("name_texture"):
		dpg.set_value("name_texture", np.zeros((ui_state.get_loaded_data().name_data_height * ui_state.get_loaded_data().name_data_width * 3,), dtype=np.float32))
	if dpg.does_item_exist("category_texture"):
		dpg.set_value("category_texture", np.zeros((max(1, ui_state.get_loaded_data().category_data_height) * max(1, ui_state.get_loaded_data().category_data_width) * 3,), dtype=np.float32))
	if dpg.does_item_exist("attempts_total_texture"):
		dpg.set_value("attempts_total_texture", np.zeros((ui_state.get_loaded_data().attempt_totals_height * ui_state.get_loaded_data().attempt_totals_width * 3,), dtype=np.float32))
	set_ocr_population_status("OCR autofill idle")
	set_ocr_status_bar("warning")

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
			(line_w, _line_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, sub_scale, sub_th)
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
		ui_state.get_loaded_data().bubble_grid_width,
		ui_state.get_loaded_data().bubble_grid_height,
		"Loading...",
		subtitle=name,
		bg_color=(25, 25, 25),
	)
	side_msg = _render_message_image(
		ui_state.get_loaded_data().zones_and_tops_display_width,
		ui_state.get_loaded_data().zones_and_tops_display_height,
		"Loading...",
		bg_color=(25, 25, 25),
	)
	name_msg = _render_message_image(
		ui_state.get_loaded_data().name_data_width,
		ui_state.get_loaded_data().name_data_height,
		"Loading...",
		bg_color=(25, 25, 25),
	)
	attempt_msg = _render_message_image(
		ui_state.get_loaded_data().attempt_totals_width,
		ui_state.get_loaded_data().attempt_totals_height,
		"Loading...",
		bg_color=(25, 25, 25),
	)

	_set_texture_if_exists("texture_tag", main_msg, ui_state.get_loaded_data().bubble_grid_width, ui_state.get_loaded_data().bubble_grid_height)
	_set_texture_if_exists("zones_and_tops_texture", side_msg, ui_state.get_loaded_data().zones_and_tops_display_width, ui_state.get_loaded_data().zones_and_tops_display_height)
	_set_texture_if_exists("name_texture", name_msg, ui_state.get_loaded_data().name_data_width, ui_state.get_loaded_data().name_data_height)
	if ui_state.get_loaded_data().has_category_area:
		cat_msg = _render_message_image(ui_state.get_loaded_data().category_data_width, ui_state.get_loaded_data().category_data_height, "...", bg_color=(25, 25, 25))
		_set_texture_if_exists("category_texture", cat_msg, ui_state.get_loaded_data().category_data_width, ui_state.get_loaded_data().category_data_height)
	_set_texture_if_exists("attempts_total_texture", attempt_msg, ui_state.get_loaded_data().attempt_totals_width, ui_state.get_loaded_data().attempt_totals_height)

	set_export_buttons_enabled(False)
	set_ocr_population_status("OCR autofill in progress...")
	set_ocr_status_bar("in_progress")

def show_error_state(error_message):
	error_main = _render_message_image(
		ui_state.get_loaded_data().bubble_grid_width,
		ui_state.get_loaded_data().bubble_grid_height,
		"Processing Error",
		subtitle=str(error_message),
		bg_color=(35, 35, 60),
		title_color=(220, 220, 255),
		subtitle_color=(220, 220, 220),
	)
	black_side = np.zeros((ui_state.get_loaded_data().zones_and_tops_display_height, ui_state.get_loaded_data().zones_and_tops_display_width, 3), dtype=np.uint8)
	black_name = np.zeros((ui_state.get_loaded_data().name_data_height, ui_state.get_loaded_data().name_data_width, 3), dtype=np.uint8)
	black_attempt = np.zeros((ui_state.get_loaded_data().attempt_totals_height, ui_state.get_loaded_data().attempt_totals_width, 3), dtype=np.uint8)

	_set_texture_if_exists("texture_tag", error_main, ui_state.get_loaded_data().bubble_grid_width, ui_state.get_loaded_data().bubble_grid_height)
	_set_texture_if_exists("zones_and_tops_texture", black_side, ui_state.get_loaded_data().zones_and_tops_display_width, ui_state.get_loaded_data().zones_and_tops_display_height)
	_set_texture_if_exists("name_texture", black_name, ui_state.get_loaded_data().name_data_width, ui_state.get_loaded_data().name_data_height)
	if ui_state.get_loaded_data().has_category_area:
		black_cat = np.zeros((ui_state.get_loaded_data().category_data_height, ui_state.get_loaded_data().category_data_width, 3), dtype=np.uint8)
		_set_texture_if_exists("category_texture", black_cat, ui_state.get_loaded_data().category_data_width, ui_state.get_loaded_data().category_data_height)
	_set_texture_if_exists("attempts_total_texture", black_attempt, ui_state.get_loaded_data().attempt_totals_width, ui_state.get_loaded_data().attempt_totals_height)

	set_export_buttons_enabled(False)
	set_ocr_population_status("OCR autofill not completed (processing error)")
	set_ocr_status_bar("warning")

def update_queue_ui(file_list, filename, last_failed_file=None):
	if not dpg.does_item_exist("queue_list_container"):
		return

	dpg.delete_item("queue_list_container", children_only=True)
	error_count = 0
	for idx, p in enumerate(file_list, start=1):
		is_current = (filename is not None and p == filename)
		is_error = p in ui_state.get_ui_state().queue_error_map
		prefix = "* " if is_current else "  "
		label = f"{prefix}{idx:03d} | {Path(p).name}"
		button_tag = f"queue_item_{idx}_{abs(hash(p))}"
		dpg.add_button(
			label=label,
			tag=button_tag,
			parent="queue_list_container",
			width=-1,
			callback=ui_backend.on_queue_file_selected,
			user_data=p,
		)
		if is_error:
			error_count += 1
			dpg.bind_item_theme(button_tag, "queue_error_theme")
		elif is_current:
			dpg.bind_item_theme(button_tag, "queue_current_theme")

	if len(file_list) == 0:
		dpg.add_button(label="<queue empty>", parent="queue_list_container", width=-1, enabled=False)

	if ui_state.get_ui_state().queue_error_scan_has_run:
		dpg.set_value("queue_count_text", f"Queue: {len(file_list)} file(s) | Errors: {error_count}")
	else:
		dpg.set_value("queue_count_text", f"Queue: {len(file_list)} file(s)")
	if filename:
		current_label = Path(filename).name
	elif last_failed_file:
		current_label = f"failed: {Path(last_failed_file).name}"
	else:
		current_label = "-"
	dpg.set_value("current_file_text", f"Current: {current_label}")

	# If nothing is queued/active, present a blank UI and disable exports.
	if len(file_list) == 0 and filename is None:
		clear_display_textures()
		set_export_buttons_enabled(False)
	else:
		set_export_buttons_enabled(filename is not None)

	set_debug_button_enabled(filename is not None or last_failed_file is not None)


def apply_scan_directory_and_refresh(sender, app_data):
	new_dir = Path(dpg.get_value("scan_dir_input")).expanduser()
	ui_backend.apply_scan_directory_and_refresh(new_dir)




def to_rgb_texture(image_bgr, width, height):
	img = image_bgr
	if len(img.shape) == 2:
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

	# Fit image into the texture while preserving aspect ratio (letterbox).
	h_src, w_src = img.shape[:2]
	scale = min(width / float(max(1, w_src)), height / float(max(1, h_src)))
	new_w = max(1, round(w_src * scale))
	new_h = max(1, round(h_src * scale))
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

	debug_steps = debug_pipeline.get_debug_steps()

	idx = max(0, min(step_index, len(debug_steps) - 1))
	title, img = debug_steps[idx]
	debug_current_step_img = img
	dpg.set_value("debug_step_title", title)
	render_debug_with_zoom()

def on_debug_step_selected(sender, app_data):
	global debug_zoom

	debug_steps = debug_pipeline.get_debug_steps()

	selected_label = dpg.get_value("debug_step_list")
	if not selected_label:
		return
	for idx, (title, _) in enumerate(debug_steps):
		if selected_label.endswith(title):
			debug_zoom = {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}
			show_debug_step(idx)
			return

def to_debug_texture(img_bgr, overlay_rect=None):

	frame_width = ui_state.get_loaded_data().full_page_width
	frame_height = ui_state.get_loaded_data().full_page_height

	"""Render debug image with preserved aspect ratio and optional display-space overlay."""
	if img_bgr is None:
		return np.zeros((frame_height * frame_width * 3,), dtype=np.float32)

	img = img_bgr
	if len(img.shape) == 2:
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

	h_src, w_src = img.shape[:2]
	scale = min(frame_width / float(max(1, w_src)), frame_height / float(max(1, h_src)))
	new_w = max(1, round(w_src * scale))
	new_h = max(1, round(h_src * scale))
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

	frame_width = ui_state.get_loaded_data().bubble_grid_width
	frame_height = ui_state.get_loaded_data().bubble_grid_height

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
	x0, y0 = debug_zoom["x0"], debug_zoom["y0"]
	x1, y1 = debug_zoom["x1"], debug_zoom["y1"]

	crop_w = max(1, int((x1 - x0) * img_w))
	crop_h = max(1, int((y1 - y0) * img_h))
	dx0, dy0, dw, dh = get_debug_display_rect(crop_w, crop_h)

	if clamp_to_image:
		local_x = max(dx0, min(local_x, dx0 + dw))
		local_y = max(dy0, min(local_y, dy0 + dh))
	elif local_x < dx0 or local_x > (dx0 + dw) or local_y < dy0 or local_y > (dy0 + dh):
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
		x_min, y_min, x_max, y_max = debug_zoom["x0"], debug_zoom["y0"], debug_zoom["x1"], debug_zoom["y1"]
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
		debug_zoom = {"x0": x_min, "y0": y_min, "x1": x_max, "y1": y_max}
		render_debug_with_zoom()
	except (ValueError, IndexError):
		# Invalid input, silently ignore
		pass

def render_debug_with_zoom(draw_drag_rect=False):
	if debug_current_step_img is None or not dpg.does_item_exist("debug_texture"):
		return

	img = debug_current_step_img
	h, w = img.shape[:2]

	x0 = int(debug_zoom["x0"] * w)
	y0 = int(debug_zoom["y0"] * h)
	x1 = int(debug_zoom["x1"] * w)
	y1 = int(debug_zoom["y1"] * h)

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
	debug_zoom = {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}
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

	x0, y0 = debug_zoom["x0"], debug_zoom["y0"]
	x1, y1 = debug_zoom["x1"], debug_zoom["y1"]

	new_x0 = max(0.0, ox - (ox - x0) * scale_factor)
	new_x1 = min(1.0, ox + (x1 - ox) * scale_factor)
	new_y0 = max(0.0, oy - (oy - y0) * scale_factor)
	new_y1 = min(1.0, oy + (y1 - oy) * scale_factor)

	if (new_x1 - new_x0) < 0.01 or (new_y1 - new_y0) < 0.01:
		return

	debug_zoom = {"x0": new_x0, "y0": new_y0, "x1": new_x1, "y1": new_y1}
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
	x0, y0 = debug_zoom["x0"], debug_zoom["y0"]
	x1, y1 = debug_zoom["x1"], debug_zoom["y1"]
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

	debug_zoom = {"x0": drag_x0, "y0": drag_y0, "x1": drag_x1, "y1": drag_y1}
	debug_drag_start_local = None
	debug_drag_current_local = None
	render_debug_with_zoom()

def show_debug_screen(sender, app_data):
	filename = ui_state.get_loaded_data().filename
	last_failed_file = ui_state.get_ui_state().last_failed_file
	target_file = filename if filename is not None else last_failed_file
	if target_file is None:
		set_status("No active file for debug view.")
		return

	debug_steps = debug_pipeline.get_debug_steps()

	if not debug_steps:
		set_status("No debug steps captured for this form.")
		return

	step_labels = [f"{i+1:02d} | {title}" for i, (title, _) in enumerate(debug_steps)]

	# Reset zoom for the new debug session
	debug_zoom["x0"] = 0.0
	debug_zoom["y0"] = 0.0
	debug_zoom["x1"] = 1.0
	debug_zoom["y1"] = 1.0

	if not dpg.does_item_exist("debug_window"):
		with dpg.window(label="Debug Pipeline", tag="debug_window", width=1400, height=900, show=False):
			dpg.add_text("Debug Pipeline", tag="debug_step_title")
			with dpg.group(horizontal=True):
				with dpg.group():
					dpg.add_listbox(step_labels, tag="debug_step_list", num_items=12, width=320, callback=on_debug_step_selected)
					dpg.add_button(label="Reset Zoom", callback=lambda _: reset_debug_zoom())
				with dpg.group():
					dpg.add_input_text(default_value="View relative coords (x_min, x_max, y_min, y_max)", tag="debug_coords_mode_input", width=900, readonly=True)
					dpg.add_input_text(default_value="(0.0000, 1.0000, 0.0000, 1.0000)", tag="debug_coords_input", width=900, callback=on_debug_coords_input_change)
					dpg.add_image("debug_texture", tag="debug_image")
	else:
		dpg.configure_item("debug_step_list", items=step_labels)

	dpg.set_value("debug_step_list", step_labels[0])
	show_debug_step(0)
	dpg.configure_item("debug_window", label=f"Debug Pipeline - {Path(target_file).name}", show=True)

def draw_data(cell_data, bubble_grid_image, zones_and_tops_image, attempts_total_image, name_area_image, category_area_image):

	amount_zones_tops, tries_zones_tops, per_boulder_zones_tops = grader.get_amounts_and_tries(cell_data)

	bubble_grid_image = draw_grid_on_image(bubble_grid_image, cell_data)

	draw_texture(bubble_grid_image, "bubble_grid_texture")
	draw_zones_and_tops(zones_and_tops_image, per_boulder_zones_tops)
	draw_texture(name_area_image, "name_texture")
	if ui_state.get_loaded_data().has_category_area:
		draw_texture(category_area_image, "category_texture")

	attempts_total_image = draw_attempts_total_on_image(attempts_total_image, amount_zones_tops, tries_zones_tops)
	draw_texture(attempts_total_image, "attempts_total_texture")

def fill_candidate_name(name, contestant_number):
	if dpg.does_item_exist("user_name"):
		dpg.set_value("user_name", name)
	if dpg.does_item_exist("contestant_number"):
		dpg.set_value("contestant_number", contestant_number)


def set_category_and_gender(is_male, age_cat):
	if is_male is not None and dpg.does_item_exist("is_male"):
		dpg.set_value("is_male", is_male)
	if age_cat is not None and dpg.does_item_exist("age_category"):
		dpg.set_value("age_category", age_cat)

def draw_category_data(frame):
	if not ui_state.get_loaded_data().has_category_area:
		return

	category_area_image = ui_state.get_loaded_data().category_texture_data

	try:
		data = category_area_image.flatten()
		data = np.float32(data)
		category_texture_data = np.true_divide(data, 255.0)
		dpg.set_value("category_texture", category_texture_data)
	except Exception as e:
		print(f"Error drawing category data: {e}")

def get_score_property_text(property, amount_zones_tops, tries_zones_tops):
	if property == "zones":
		return amount_zones_tops[0] if len(amount_zones_tops) > 0 else 0
	if property == "tops":
		return amount_zones_tops[1] if len(amount_zones_tops) > 1 else 0
	if property == "att_z":
		return tries_zones_tops[0] if len(tries_zones_tops) > 0 else 0
	if property == "att_t":
		return tries_zones_tops[1] if len(tries_zones_tops) > 1 else 0

	error_message = f"Unknown property: {property}"
	raise ValueError(error_message)

def draw_attempts_total_on_image(attempts_total_image, amount_zones_tops, tries_zones_tops):

	zones_and_tops_indication = config.get_property("zones_and_tops_indication")

	attempts_total_image = attempts_total_image.copy()

	width = attempts_total_image.shape[1]
	height = attempts_total_image.shape[0]

	left_x = int(width * 0.45)
	right_x = int(width * 0.85)
	top_y = int(0.4 * height * 0.99)
	bottom_y = int(0.8 * height * 0.99)

	cv2.putText(attempts_total_image, str(get_score_property_text(zones_and_tops_indication[0], amount_zones_tops, tries_zones_tops)), (left_x, top_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
	cv2.putText(attempts_total_image, str(get_score_property_text(zones_and_tops_indication[1], amount_zones_tops, tries_zones_tops)), (right_x, top_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

	cv2.putText(attempts_total_image, str(get_score_property_text(zones_and_tops_indication[2], amount_zones_tops, tries_zones_tops)), (left_x, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
	cv2.putText(attempts_total_image, str(get_score_property_text(zones_and_tops_indication[3], amount_zones_tops, tries_zones_tops)), (right_x, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

	return attempts_total_image


def draw_zones_and_tops(zones_and_tops_image, per_boulder_zones_tops):

	# Make a copy to draw the numbers on without modifying the original texture data
	zones_and_tops_image = zones_and_tops_image.copy()

	zones_and_tops_width = zones_and_tops_image.shape[1]
	zones_and_tops_height = zones_and_tops_image.shape[0]

	left_label_x = max(6, int(zones_and_tops_width * 0.02))

	# Write the zones and tops amounts on the frame
	num_boulders = len(per_boulder_zones_tops)
	for b in range(num_boulders):
		zone_x = int(zones_and_tops_width * 0.72)
		top_x = int(zones_and_tops_width * 0.85)
		y = int(((b+1) / num_boulders) * zones_and_tops_height * 0.99)
		cv2.putText(zones_and_tops_image, str(b + 1), (left_label_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
		(zone, top) = per_boulder_zones_tops[b]
		if zone is not None:
			cv2.putText(zones_and_tops_image, str(zone), (zone_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
		if top is not None:
			cv2.putText(zones_and_tops_image, str(top), (top_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

	draw_texture(zones_and_tops_image, "zones_and_tops_texture")

def draw_texture(texture_data, texture_tag):
	try:
		data = texture_data.flatten()  # flatten camera data to a 1 d stricture
		data = np.float32(data)  # change data type to 32bit floats
		normalized_data = np.true_divide(data, 255.0)  # normalize image data to prepare for GPU
		dpg.set_value(texture_tag, normalized_data)
	except Exception as e:
		print(f"Error processing file {ui_state.get_loaded_data().filename}: {e}")
		return


def draw_grid_on_image(frame, cell_data):

	median_bubble_size = ui_state.get_loaded_data().median_bubble_size
	med_w = median_bubble_size[0]
	row_centers_sorted = ui_state.get_loaded_data().row_centers_sorted
	col_centers_sorted = ui_state.get_loaded_data().col_centers_sorted

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

	# Mouse position in screen space
	mouse_x, mouse_y = dpg.get_mouse_pos()

	# Get image position in screen space
	image_pos = dpg.get_item_rect_min("main_image")

	# Convert to local image coordinates
	local_x = mouse_x - image_pos[0]
	local_y = mouse_y + image_pos[1]

	img_x = int(local_x)
	img_y = int(local_y)

	ui_backend.on_bubble_image_click(img_x, img_y)

def toggle_all_bubbles_and_markers(sender=None, app_data=None):
	ui_backend.toggle_all_bubbles_and_markers()


def export_to_csv(sender, callback):

	name = dpg.get_value("user_name")
	contestant_number = dpg.get_value("contestant_number") if dpg.does_item_exist("contestant_number") else ""
	sex = "M" if dpg.get_value("is_male") else "V"
	age_cat = dpg.get_value("age_category") if dpg.does_item_exist("age_category") else ""

	ui_backend.write_results_to_csv(name, contestant_number, sex, age_cat)

	dpg.set_value("user_name", "")
	if dpg.does_item_exist("contestant_number"):
		dpg.set_value("contestant_number", "")

def refresh_file_queue(sender, callback):
	ui_backend.refresh_file_queue()

def export_to_ground_truth(sender, callback):
	ui_backend.export_to_ground_truth()

def error_check_all_queued_files(sender, callback):
	ui_backend.error_check_all_queued_files()

def redetect_bubbles(sender, callback):
	ui_backend.redetect_bubbles()

def add_raw_texture(tag, data, format=dpg.mvFormat_Float_rgb):
	try:
		data_f32 = np.float32(data)
		data_f32 = np.true_divide(data_f32, 255.0)
		data_f32 = data_f32.flatten()
		dpg.add_raw_texture(data.shape[1], data.shape[0], data_f32, tag=tag, format=format)
	except Exception as e:
		print(f"Error adding raw texture {tag}: {e}")

def on_scan_directory_chosen(sender, app_data):
	ui_backend.apply_scan_directory_and_refresh(Path(app_data['file_path_name']))
	# Update the text display for the chosen directory
	dpg.set_value("scan_dir_text", config.get_property("scanning_data_folder"))

def on_processed_directory_chosen(sender, app_data):
	ui_backend.apply_processed_directory_and_refresh(Path(app_data['file_path_name']))
	dpg.set_value("processed_dir_text", config.get_property("processed_data_folder"))

def on_errored_directory_chosen(sender, app_data):
	ui_backend.apply_errored_directory_and_refresh(Path(app_data['file_path_name']))
	dpg.set_value("errored_dir_text", config.get_property("errored_data_folder"))


with dpg.texture_registry(show=False):
	add_raw_texture("bubble_grid_texture", ui_state.get_loaded_data().bubble_grid_texture_data, format=dpg.mvFormat_Float_rgb)

	add_raw_texture("zones_and_tops_texture", ui_state.get_loaded_data().zones_and_tops_texture_data, format=dpg.mvFormat_Float_rgb)

	add_raw_texture("name_texture", ui_state.get_loaded_data().name_texture_data, format=dpg.mvFormat_Float_rgb)

	add_raw_texture("category_texture", ui_state.get_loaded_data().category_texture_data, format=dpg.mvFormat_Float_rgb)

	add_raw_texture("attempts_total_texture", ui_state.get_loaded_data().attempts_total_texture_data, format=dpg.mvFormat_Float_rgb)

	print(f"Creating a debug texture with size ({ui_state.get_loaded_data().full_page_width}x{ui_state.get_loaded_data().full_page_height})")
	add_raw_texture("debug_texture", ui_state.get_loaded_data().debug_texture_data, format=dpg.mvFormat_Float_rgb)


with dpg.item_handler_registry(tag="image_handler"):
	dpg.add_item_clicked_handler(callback=on_main_frame_clicked)

dpg.add_file_dialog(
    directory_selector=True, show=False, callback=on_scan_directory_chosen, tag="to_process_file_dialog_id", width=700 ,height=400)

dpg.add_file_dialog(
    directory_selector=True, show=False, callback=on_processed_directory_chosen, tag="processed_file_dialog_id", width=700 ,height=400)

dpg.add_file_dialog(
    directory_selector=True, show=False, callback=on_errored_directory_chosen, tag="errored_file_dialog_id", width=700 ,height=400)

with dpg.window(label="resultstester", tag="mainWindow"):
	with dpg.table(header_row=False):
		dpg.add_table_column(width_fixed=True, init_width_or_weight=float(ui_state.get_loaded_data().bubble_grid_width))
		dpg.add_table_column(width_fixed=True, init_width_or_weight=float(ui_state.get_loaded_data().side_panel_width))
		with dpg.table_row():
			with dpg.table_cell():
				dpg.add_image("bubble_grid_texture", tag="main_image")
			with dpg.table_cell():
				with dpg.group(horizontal=True):
					dpg.add_image("zones_and_tops_texture")
					dpg.add_spacer(width=ui_state.get_loaded_data().controls_panel_gap)
					with dpg.group():
						dpg.add_text(f"UI Instance: {ui_state.get_ui_state().instance_id}")
						dpg.add_spacer(height=8)
						dpg.add_text("Name contestant:")
						dpg.add_input_text(tag="user_name")
						dpg.add_text("Contestant number:")
						dpg.add_input_text(tag="contestant_number")
						with dpg.group(horizontal=True):
							dpg.add_text("Is contestant male?")
							dpg.add_checkbox(tag="is_male", default_value=True)
						with dpg.group(horizontal=True):
							dpg.add_text("Category:")
							dpg.add_combo(config.get_property("age_categories"), tag="age_category", default_value=config.get_property("age_categories")[0], width=80)
						dpg.add_spacer(height=15)
						dpg.add_button(label="export", tag="export_button", callback=export_to_csv)
						dpg.add_button(label="export to ground truth", tag="export_ground_truth_button", callback=export_to_ground_truth)
						dpg.add_button(label="Show Debug Screen", tag="show_debug_button", callback=show_debug_screen)
						dpg.add_spacer(height=15)
						dpg.add_text(str(ui_state.get_ui_state().to_process_data_folder), tag="scan_dir_text")
						dpg.add_button(label="Choose scanning directory", callback=lambda: dpg.show_item("to_process_file_dialog_id"))
						dpg.add_text(str(ui_state.get_ui_state().processed_data_folder), tag="processed_dir_text")
						dpg.add_button(label="Choose processed directory", callback=lambda: dpg.show_item("processed_file_dialog_id"))
						dpg.add_text(str(ui_state.get_ui_state().errored_data_folder), tag="errored_dir_text")
						dpg.add_button(label="Choose errored directory", callback=lambda: dpg.show_item("errored_file_dialog_id"))
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
						dpg.add_text("DANGER ZONE BELOW", color=(255, 0, 0))
						dpg.add_button(label="Toggle ALL bubbles", tag="toggle_all_bubbles_button", callback=toggle_all_bubbles_and_markers)
						dpg.add_button(label="Redetect bubbles", tag="redetect_bubbles_button", callback=redetect_bubbles)
						dpg.add_spacer(height=10)
						dpg.add_button(label="Error check ALL (will stall UI)", tag="error_check_all_button", callback=error_check_all_queued_files)
						dpg.add_text("Error check idle", tag="error_check_progress_text", wrap=240)
						dpg.add_progress_bar(default_value=0.0, tag="error_check_progress_bar", width=240, overlay="0 / 0")
		with dpg.table_row():
			with dpg.table_cell():
				dpg.add_image("name_texture")
				if "category" in config.get_property("included_regions"):
					dpg.add_image("category_texture")
			with dpg.table_cell():
				with dpg.group(horizontal=True):
					dpg.add_image("attempts_total_texture")

show_loading_state("Starting up...")
set_status(f"Instance {ui_state.get_ui_state().instance_id} using claim folder: {ui_state.get_ui_state().processing_data_folder.name}")
ui_backend.refresh_file_queue()

#if ui_state.get_loaded_data().filename is None and ui_state.get_ui_state().last_failed_file is None:
#	ui_backend.get_next_file(True)


dpg.bind_item_handler_registry("main_image", "image_handler")

with dpg.handler_registry():
	dpg.add_mouse_wheel_handler(callback=on_debug_scroll)
	dpg.add_mouse_down_handler(button=0, callback=on_debug_mouse_down)
	dpg.add_mouse_release_handler(button=0, callback=on_debug_mouse_release)
	dpg.add_mouse_move_handler(callback=on_debug_mouse_move)

try:
    dpg.show_viewport()
    dpg.maximize_viewport()
    dpg.set_primary_window("mainWindow", True)
    dpg.start_dearpygui()
except Exception as e:
	print(f"Error in main UI loop: {e}")
finally:
    dpg.destroy_context()
    ui_backend.restore_processing_folder_on_exit()
