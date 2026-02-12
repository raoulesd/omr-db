import os
import shutil
from os import listdir
from os.path import isfile, join
from genericpath import isfile
import dearpygui.dearpygui as dpg
import cv2 as cv
import cv2 as cv2
import numpy as np
import grader
import numpy as np

COLUMNS = 9
ROWS = 20
ANSWERS = 3

if __name__ == '__main__':
	processed_data_folder = "process_data/processed"
	to_process_data_folder = "process_data/to_process"
	errored_data_folder = "process_data/errored"

	ui_scale = 1.2
	
	frame_width = int(800 * ui_scale)
	frame_height = int(455 * ui_scale)

	zones_and_tops_width = int(180 * ui_scale)

	paths = [processed_data_folder, to_process_data_folder, errored_data_folder]
	for p in paths:
		isExist = os.path.exists(p)
		if not isExist:
			os.makedirs(p)
			print(f"Made dir: {p}")

	path = to_process_data_folder
	fileList = [join(path, f) for f in os.listdir(path) if isfile(join(path, f))]
	print(fileList)

	csvFile = open("results.csv", "a")

	dpg.create_context()
	dpg.create_viewport(title='Review scores', width=1400, height=1000)
	dpg.setup_dearpygui()

	def get_next_file(is_initialization):
		global filename, amountZT, triesZT, per_boulder_ZT, frame, data, texture_data, cell_data, row_centers_sorted, col_centers_sorted, med_w, med_h, full_page

		if len(fileList) == 0:
			print("No more files to process.")
			return

		filename = fileList.pop()
		
		filled_cells, (ROWS, COLS), warped_u8, (row_centers_sorted, col_centers_sorted), (med_w, med_h), full_page = grader.grade_score_form(filename, show_plots=False)

		cell_data = np.zeros((ROWS, COLS), dtype=np.uint8)
		for (r, c) in filled_cells:
			cell_data[r, c] = 1

		if len(warped_u8.shape) == 2:
			warped_u8 = cv2.cvtColor(warped_u8, cv2.COLOR_GRAY2RGB)
		frame = warped_u8

		draw_data()

	def draw_data():
		global cell_data, full_page, amountZT, triesZT, per_boulder_ZT, frame

		amountZT, triesZT, per_boulder_ZT = grader.get_amounts_and_tries(cell_data)

		draw_frame(frame)
		draw_zones_and_tops(full_page)


	def extract_zones_and_tops_area(frame):
		y_min = int(0.22 * frame.shape[0])
		y_max = int(0.5 * frame.shape[0])
		x_min = int(0.8 * frame.shape[1])
		x_max = int(0.915 * frame.shape[1])
		x_max = int(1 * frame.shape[1])

		cutout = frame[y_min:y_max, x_min:x_max]

		return cv2.resize(cutout, (zones_and_tops_width, frame_height), interpolation=cv2.INTER_LINEAR)

	def draw_zones_and_tops(frame):
		global zones_and_tops_texture_data, amountZT, triesZT, per_boulder_ZT
		frame = frame.copy()

		frame = extract_zones_and_tops_area(frame)

		# Write the zones and tops amounts on the frame
		num_boulders = len(per_boulder_ZT)
		print(per_boulder_ZT)
		for b in range(num_boulders):
			zone_x = int(zones_and_tops_width * 0.6)
			top_x = int(zones_and_tops_width * 0.8)
			y = int(((b+1) / num_boulders) * frame.shape[0] * 0.99)
			(zone, top) = per_boulder_ZT[b]
			print(f"Boulder {b}: zone={zone}, top={top}")
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
		global cell_data, frame, row_centers_sorted, col_centers_sorted

		# Mouse position in screen space
		mouse_x, mouse_y = dpg.get_mouse_pos()

		# Get image position in screen space
		image_pos = dpg.get_item_rect_min("main_image")

		# Convert to local image coordinates
		local_x = mouse_x - image_pos[0]
		local_y = mouse_y - image_pos[1]

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
				closest_row = row_index+1
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
		global amountZT, triesZT, filename
		get_next_file(False)


	get_next_file(True)

	with dpg.texture_registry(show=False):
		dpg.add_raw_texture(frame_width, frame_height, texture_data, tag="texture_tag",
							format=dpg.mvFormat_Float_rgb)
		dpg.add_raw_texture(zones_and_tops_width, frame_height, zones_and_tops_texture_data, tag="zones_and_tops_texture",
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
					dpg.add_text(f"Naam kandidaat:")
					dpg.add_input_text(tag=f"user_name")
					dpg.add_button(label="export", callback=export_to_csv)

	dpg.bind_item_handler_registry("main_image", "image_handler")

	dpg.show_viewport()
	dpg.set_primary_window("mainWindow", True)
	dpg.start_dearpygui()
	dpg.destroy_context()
