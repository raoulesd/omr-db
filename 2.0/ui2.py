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
		global filename, amountZT, triesZT, frame, data, texture_data, cell_data

		if len(fileList) == 0:
			print("No more files to process.")
			return

		filename = fileList.pop()
		
		filled_cells, (ROWS, COLS), warped_u8, (row_centers_sorted, col_centers_sorted), (med_w, med_h) = grader.grade_score_form(filename, show_plots=False)

		cell_data = np.zeros((ROWS, COLS), dtype=np.uint8)
		for (r, c, _) in filled_cells:
			cell_data[r, c] = 1

		amountZT, triesZT = grader.get_amounts_and_tries(filled_cells, ROWS, COLS)

		warped_u8 = cv2.resize(warped_u8, (warped_u8.shape[1] // 2, warped_u8.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
		
		frame = warped_u8
		
		try:
			data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # because the camera data comes in as BGR and we need RGB
			data = data.flatten()  # flatten camera data to a 1 d stricture
			data = np.float32(data)  # change data type to 32bit floats
			texture_data = np.true_divide(data, 255.0)  # normalize image data to prepare for GPU
			dpg.set_value("texture_tag", texture_data)
		except Exception as e:
			print(f"Error processing file {filename}: {e}")
			return


	def export_to_csv(sender, callback):
		global amountZT, triesZT, filename
		get_next_file(False)


	get_next_file(True)

	with dpg.texture_registry(show=False):
		dpg.add_raw_texture(frame.shape[1], frame.shape[0], texture_data, tag="texture_tag",
							format=dpg.mvFormat_Float_rgb)

	with dpg.window(label="resultstester", tag="mainWindow"):
		with dpg.table(header_row=False):
			dpg.add_table_column(width_stretch=True)
			dpg.add_table_column(width_fixed=True, init_width_or_weight=250.0)
			with dpg.table_row():
				with dpg.table_cell():
					dpg.add_image("texture_tag")
				with dpg.table_cell():
					dpg.add_text(f"Naam kandidaat:")
					dpg.add_input_text(tag=f"user_name")
					with dpg.table(header_row=False):
						dpg.add_table_column(width_fixed=True)
						dpg.add_table_column(width_fixed=True)
						dpg.add_table_column(width_fixed=True, init_width_or_weight=40)
						dpg.add_table_column(width_fixed=True)
						dpg.add_table_column(width_fixed=True, init_width_or_weight=40)
						for i in range(1, ROWS + 1):
							with dpg.table_row():
								dpg.add_text(f"B{i}")
								dpg.add_text("Z")
								dpg.add_text("0", tag=f"zone_{i}")
								dpg.add_text("T")
								dpg.add_text("0", tag=f"tops_{i}")
						with dpg.table_row():
							dpg.add_text("Aantal")
							dpg.add_text("Z")
							dpg.add_input_text(tag=f"zone_total", default_value=str(amountZT[0]))
							dpg.add_text("T")
							dpg.add_input_text(tag=f"tops_total", default_value=str(amountZT[1]))
						with dpg.table_row():
							dpg.add_text("Pogingen")
							dpg.add_text("Z")
							dpg.add_input_text(tag=f"zone_tries", default_value=str(triesZT[0]))
							dpg.add_text("T")
							dpg.add_input_text(tag=f"tops_tries", default_value=str(triesZT[1]))
					dpg.add_button(label="export", callback=export_to_csv)

	dpg.show_viewport()
	dpg.set_primary_window("mainWindow", True)
	dpg.start_dearpygui()
	dpg.destroy_context()
