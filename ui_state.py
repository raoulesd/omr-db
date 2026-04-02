import os
import uuid
import cv2
import numpy as np
import grader
from pipeline import region_extractor
from configs import config
from pathlib import Path

class LoadedScoreSheetData:
	"""LoadedScoreSheetData holds the data related to the currently loaded score sheet, including both the raw extracted data and the processed textures for display in the UI. This class is designed to be a single source of truth for all data related to the currently loaded score sheet.
	"""

	def __init__(self):
		self.filename = None
		self.bubble_grid = None
		self.median_bubble_size = None
		self.row_centers_sorted = None
		self.col_centers_sorted = None
		self.amount_zones_tops = None
		self.tries_zones_tops = None
		self.per_boulder_zones_tops = None
		self.cell_data = None

		# Display data (textures)

		ui_scale = config.get_property("ui_scale")
		self.full_page_width = int(config.get_property("region_original_sizes")["full_page"][0] * ui_scale)
		self.full_page_height = int(config.get_property("region_original_sizes")["full_page"][1] * ui_scale)

		self.bubble_grid_width = int(config.get_property("region_original_sizes")["attempt_score"][0] * ui_scale)
		self.bubble_grid_height = int(config.get_property("region_original_sizes")["attempt_score"][1] * ui_scale)

		self.attempt_totals_width = int(config.get_property("region_original_sizes")["total_scores"][0] * ui_scale)
		self.attempt_totals_height = int(config.get_property("region_original_sizes")["total_scores"][1] * ui_scale)

		self.zones_and_tops_width = int(config.get_property("region_original_sizes")["boulder_score"][0] * ui_scale)
		self.zones_and_tops_height = int(config.get_property("region_original_sizes")["boulder_score"][1] * ui_scale)

		print(f"Calculated zones and tops display width: {self.zones_and_tops_width} and height: {self.zones_and_tops_height}")

		zones_and_tops_left_padding = 48
		# Display zones/tops at main-frame height while keeping its original aspect ratio.
		self.zones_and_tops_display_height = self.bubble_grid_height
		self.zones_and_tops_base_display_width = max(
			1,
			round(self.zones_and_tops_width * (self.zones_and_tops_display_height / float(max(1, self.zones_and_tops_height))))
		)
		self.zones_and_tops_display_width = self.zones_and_tops_base_display_width + zones_and_tops_left_padding

		self.zones_and_tops_height = self.zones_and_tops_display_height
		self.zones_and_tops_width = self.zones_and_tops_base_display_width

		self.controls_panel_width = 280
		self.controls_panel_gap = 16
		self.side_panel_width = max(self.zones_and_tops_display_width + self.controls_panel_gap + self.controls_panel_width, self.attempt_totals_width)

		self.name_data_width = int(config.get_property("region_original_sizes")["user_info"][0] * ui_scale)
		self.name_data_height = int(config.get_property("region_original_sizes")["user_info"][1] * ui_scale)

		self.category_data_width = 1
		self.category_data_height = 1
		self.has_category_area = "category" in config.get_property("included_regions")
		if self.has_category_area:
			self.category_data_width = int(config.get_property("region_original_sizes")["category"][0] * ui_scale)
			self.category_data_height = int(config.get_property("region_original_sizes")["category"][1] * ui_scale)

		self.full_page_texture_data = np.zeros((self.full_page_height, self.full_page_width, 3), dtype=np.uint8)
		self.debug_texture_data = np.zeros((self.full_page_height, self.full_page_width, 3), dtype=np.uint8)
		self.zones_and_tops_texture_data = np.zeros((self.zones_and_tops_height, self.zones_and_tops_width, 3), dtype=np.uint8)
		self.name_texture_data = np.zeros((self.name_data_height, self.name_data_width, 3), dtype=np.uint8)
		self.category_texture_data = np.zeros((self.category_data_height, self.category_data_width, 3), dtype=np.uint8)
		self.bubble_grid_texture_data = np.zeros((self.bubble_grid_height, self.bubble_grid_width, 3), dtype=np.uint8)
		self.attempts_total_texture_data = np.zeros((self.attempt_totals_height, self.attempt_totals_width, 3), dtype=np.uint8)


	def set_cell_value(self, row, col, value, compute_derived_data=True):
		"""Sets the value of a specific cell in the cell_data grid and optionally recomputes derived data like amount_zones_tops and tries_zones_tops.

		:param row: The row index of the cell to update.
		:param col: The column index of the cell to update.
		:param value: The new value to set for the specified cell (e.g., 0 or 1).
		:param compute_derived_data: If True, recomputes derived data like amount_zones_tops and tries_zones_tops after updating the cell value. Defaults to True.
		Note that recomputing derived data can be computationally expensive, so it may be desirable to set this to False if updating multiple cells in a batch and then call compute_derived_data() once at the end.
		"""
		if self.cell_data is not None:
			self.cell_data[row, col] = value

		# Recompute the derived ZT amounts and tries whenever a cell value changes
		if compute_derived_data:
			self.amount_zones_tops, self.tries_zones_tops, self.per_boulder_zones_tops = grader.get_amounts_and_tries(self.cell_data)

	def compute_derived_data(self):
		"""Recomputes derived data like amount_zones_tops and tries_zones_tops based on the current state of cell_data. This can be called after multiple cell updates if compute_derived_data was set to False in those updates to avoid redundant computations."""
		if self.cell_data is not None:
			self.amount_zones_tops, self.tries_zones_tops, self.per_boulder_zones_tops = grader.get_amounts_and_tries(self.cell_data)

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

	def set_textures_from_full_page_texture_data(self, full_page_texture_data):

		self.bubble_grid_texture_data = region_extractor.extract_region(full_page_texture_data, "attempt_score", predetermined_size=(self.bubble_grid_width, self.bubble_grid_height))
		self.attempts_total_texture_data = region_extractor.extract_region(full_page_texture_data, "total_scores", predetermined_size=(self.attempt_totals_width, self.attempt_totals_height))
		self.zones_and_tops_texture_data = region_extractor.extract_region(full_page_texture_data, "boulder_score", predetermined_size=(self.zones_and_tops_width, self.zones_and_tops_height))
		self.name_texture_data = region_extractor.extract_region(full_page_texture_data, "user_info", predetermined_size=(self.name_data_width, self.name_data_height))
		if self.has_category_area:
			self.category_texture_data = region_extractor.extract_region(full_page_texture_data, "category", predetermined_size=(self.category_data_width, self.category_data_height))

		self.full_page_texture_data = cv2.resize(full_page_texture_data, (self.full_page_width, self.full_page_height), interpolation=cv2.INTER_AREA)

	def get_filled_cells(self):
		"""Returns a list of (row, col) tuples representing the filled cells in the bubble grid based on the current cell_data."""
		filled_cells = []
		if self.cell_data is not None:
			rows, cols = self.cell_data.shape
			filled_cells = [
				(r, c)
				for r in range(rows)
				for c in range(cols)
				if self.cell_data[r, c] == 1
			]
		return filled_cells



class UIState:
	"""UIState manages the state of the user interface, in terms of things that are not specific to a single loaded score sheet.
	"""

	def __init__(self):

		self.processed_data_folder = Path(config.get_property("processed_data_folder"))
		self.to_process_data_folder = Path(config.get_property("scanning_data_folder"))
		self.errored_data_folder = Path(config.get_property("errored_data_folder"))

		# Generate a unique instance ID for this UI session for claiming files in the queue
		self.instance_id = f"{os.getpid():x}{uuid.uuid4().hex[:2]}"

		self.processing_data_folder = self.to_process_data_folder.parent / f"processing_{self.instance_id}"

		paths = [self.processed_data_folder, self.to_process_data_folder, self.errored_data_folder, self.processing_data_folder]
		for p in paths:
			if not p.exists():
				p.mkdir(parents=True, exist_ok=True)
				print(f"Made dir: {p}")


		self.results_csv_path = Path(config.get_property("results_csv_path"))
		self.output_csv_file = Path.open(self.results_csv_path, "a")


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
