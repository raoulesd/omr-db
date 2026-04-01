import os
import uuid
import cv2
import numpy as np
import pipeline.preprocess_paper_2 as preprocess_paper
import grader
import pipeline.region_extractor as region_extractor
import configs.config as config
from pathlib import Path

class LoadedScoreSheetData:

	def __init__(self):
		self.filename = None
		self.rectified_full_page_image = None
		self.rectified_bubble_area_image = None
		self.bubble_grid = None
		self.median_bubble_size = None
		self.row_centers_sorted = None
		self.col_centers_sorted = None
		self.amountZT = None
		self.triesZT = None
		self.per_boulder_ZT = None
		self.cell_data = None

		# Display data (textures)

		ui_scale = config.get_property("ui_scale")
		self.full_page_width = int(config.get_property("region_original_sizes")["full_page"][0] * ui_scale)
		self.full_page_height = int(config.get_property("region_original_sizes")["full_page"][1] * ui_scale)

		self.bubble_grid_width = int(config.get_property("region_original_sizes")["attempt_score"][0] * ui_scale)
		self.bubble_grid_height = int(config.get_property("region_original_sizes")["attempt_score"][1] * ui_scale)

		self.attempt_totals_width = int(config.get_property("region_original_sizes")["boulder_score"][0] * ui_scale)
		self.attempt_totals_height = int(config.get_property("region_original_sizes")["boulder_score"][1] * ui_scale)

		self.zones_and_tops_width = int(config.get_property("region_original_sizes")["total_scores"][0] * ui_scale)
		self.zones_and_tops_height = int(config.get_property("region_original_sizes")["total_scores"][1] * ui_scale)

		zones_and_tops_left_padding = 48
		# Display zones/tops at main-frame height while keeping its original aspect ratio.
		self.zones_and_tops_display_height = self.full_page_height
		self.zones_and_tops_base_display_width = max(
			1,
			int(round(self.zones_and_tops_width * (self.zones_and_tops_display_height / float(max(1, self.zones_and_tops_height)))))
		)
		self.zones_and_tops_display_width = self.zones_and_tops_base_display_width + zones_and_tops_left_padding
		self.controls_panel_width = 280
		self.controls_panel_gap = 16
		self.side_panel_width = max(self.zones_and_tops_display_width + self.controls_panel_gap + self.controls_panel_width, self.attempt_totals_width)

		self.name_data_width = int(config.get_property("region_original_sizes")["user_info"][0] * ui_scale)
		self.name_data_height = int(config.get_property("region_original_sizes")["user_info"][1] * ui_scale)

		self.category_data_width = 0
		self.category_data_height = 0
		self.has_category_area = "category" in config.get_property("included_regions")
		if self.has_category_area:
			self.category_data_width = int(config.get_property("region_original_sizes")["category"][0] * ui_scale)
			self.category_data_height = int(config.get_property("region_original_sizes")["category"][1] * ui_scale)
			
		self.full_page_texture_data = np.zeros((self.full_page_height, self.full_page_width, 3), dtype=np.uint8)
		self.debug_texture_data = np.zeros((self.full_page_height, self.full_page_width, 3), dtype=np.uint8)
		self.zones_and_tops_texture_data = np.zeros((self.zones_and_tops_display_height, self.zones_and_tops_base_display_width, 3), dtype=np.uint8)
		self.name_texture_data = np.zeros((self.name_data_height, self.name_data_width, 3), dtype=np.uint8)
		self.category_texture_data = np.zeros((self.category_data_height, self.category_data_width, 3), dtype=np.uint8)
		self.bubble_grid_texture_data = np.zeros((self.bubble_grid_height, self.bubble_grid_width, 3), dtype=np.uint8)
		self.attempts_total_texture_data = np.zeros((self.attempt_totals_height, self.attempt_totals_width, 3), dtype=np.uint8)

	def clear_textures(self):
		if self.full_page_texture_data is not None:
			self.full_page_texture_data.fill(0)

		if self.debug_texture_data is not None:
			self.debug_texture_data.fill(0)

		if self.zones_and_tops_texture_data is not None:
			self.zones_and_tops_texture_data.fill(0)

		if self.name_texture_data is not None:
			self.name_texture_data.fill(0)

		if self.category_texture_data is not None:
			self.category_texture_data.fill(0)

		if self.bubble_grid_texture_data is not None:
			self.bubble_grid_texture_data.fill(0)

		if self.attempts_total_texture_data is not None:
			self.attempts_total_texture_data.fill(0)

class UIState:

	def __init__(self):

		self.processed_data_folder = Path(config.get_property("processed_data_folder"))
		self.to_process_data_folder = Path(config.get_property("scanning_data_folder"))
		self.errored_data_folder = Path(config.get_property("errored_data_folder"))

		# Generate a unique instance ID for this UI session for claiming files in the queue
		instance_id = f"{os.getpid():x}{uuid.uuid4().hex[:2]}"

		processing_data_folder = self.to_process_data_folder.parent / f"processing_{instance_id}"

		paths = [self.processed_data_folder, self.to_process_data_folder, self.errored_data_folder, processing_data_folder]
		for p in paths:
			if not p.exists():
				p.mkdir(parents=True, exist_ok=True)
				print(f"Made dir: {p}")


		results_csv_path = Path(config.get_property("results_csv_path"))
		self.output_csv_file = open(results_csv_path, "a")


		self.file_list = []
		self.last_failed_file = None
		self.queue_error_map = {}
		self.queue_error_scan_has_run = False

loaded_data = None
ui_state = None

def setup():
	global loaded_data, ui_state
	loaded_data = LoadedScoreSheetData()
	ui_state = UIState()


def get_loaded_data() -> LoadedScoreSheetData:
	return loaded_data

def get_ui_state() -> UIState:
	return ui_state


def load_score_sheet_data(filename):
	try:
		filled_cells, (row_centers_sorted, col_centers_sorted), median_bubble_size, scoresheet_rectified = grader.grade_score_form(filename, show_plots=False)

		bubble_area_image = region_extractor.extract_region(scoresheet_rectified, "attempt_score")

		loaded_data.filename = filename
		loaded_data.rectified_full_page_image = scoresheet_rectified
		loaded_data.rectified_bubble_area_image = bubble_area_image
		loaded_data.bubble_grid = filled_cells
		loaded_data.median_bubble_size = median_bubble_size
		loaded_data.row_centers_sorted = row_centers_sorted
		loaded_data.col_centers_sorted = col_centers_sorted
		return True
	except Exception as e:
		print(f"Error loading score sheet data: {e}")
		return False
	


def reset_loaded_data():
	loaded_data.filename = None
	loaded_data.rectified_full_page_image = None
	loaded_data.rectified_bubble_area_image = None
	loaded_data.bubble_grid = None
	loaded_data.median_bubble_size = None
	loaded_data.row_centers_sorted = None
	loaded_data.col_centers_sorted = None

	# Display data (textures)
	if loaded_data.full_page_texture_data is not None:
		loaded_data.full_page_texture_data.fill(0)

	if loaded_data.zones_and_tops_texture_data is not None:
		loaded_data.zones_and_tops_texture_data.fill(0)

	if loaded_data.name_texture_data is not None:
		loaded_data.name_texture_data.fill(0)

	if loaded_data.category_texture_data is not None:
		loaded_data.category_texture_data.fill(0)

	if loaded_data.bubble_grid_texture_data is not None:
		loaded_data.bubble_grid_texture_data.fill(0)

	if loaded_data.attempts_total_texture_data is not None:
		loaded_data.attempts_total_texture_data.fill(0)