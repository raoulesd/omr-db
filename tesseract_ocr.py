import os
import shutil
from pathlib import Path

try:
	import pytesseract
except ImportError:
	pytesseract = None

tesseract_cmd = None

TESSERACT_ENV_VAR = "TESSERACT_CMD"
COMMON_TESSERACT_PATHS = (
 	r"C:\Program Files\Tesseract-OCR\tesseract.exe",
 	r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)
OCR_GENDER_CHAR_SUBS = str.maketrans({
	"0": "O",
	"1": "I",
	"3": "E",
	"4": "A",
	"5": "S",
	"6": "G",
	"7": "T",
	"8": "B",
	"9": "G",
	"|": "I",
	"!": "I",
})

def tesseract_setup():
	tesseract_cmd = resolve_tesseract_cmd()
	if pytesseract is not None and tesseract_cmd is not None:
		pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

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


def extract_contestant_number(text):
	digit_groups = re.findall(r"\d+", text.replace("\n", " ").replace("\f", " "))
	if not digit_groups:
		return ""
	return max(digit_groups, key=len)


def tokenize_gender_ocr(text):
	normalized = text.upper().translate(OCR_GENDER_CHAR_SUBS).replace("\n", " ").replace("\f", " ")
	cleaned_chars = []
	for char in normalized:
		cleaned_chars.append(char if char.isalpha() else " ")
	return [token for token in "".join(cleaned_chars).split() if token]


def detect_gender_from_ocr_texts(text_candidates):
	joined = ""
	for raw in text_candidates:
		joined += "".join(tokenize_gender_ocr(raw))

	if "F" in joined:
		return False
	return True


def read_name_from_image(frame):
	if pytesseract is None:
		return "", "", "name OCR unavailable: install pytesseract"
	if tesseract_cmd is None:
		return "", "", f"name OCR unavailable: set {TESSERACT_ENV_VAR} or install Tesseract"

	processed = preprocess_name_for_ocr(frame)
	ocr_candidates = []
	number_candidates = []
	for config in ("--oem 3 --psm 7", "--oem 3 --psm 6"):
		try:
			raw = pytesseract.image_to_string(processed, config=config)
			candidate = normalize_ocr_name(raw)
			number_candidate = extract_contestant_number(raw)
		except Exception as e:
			print(f"Name OCR failed for {filename}: {e}")
			return "", "", "name OCR failed"
		if candidate:
			ocr_candidates.append(candidate)
		if number_candidate:
			number_candidates.append(number_candidate)

	if not ocr_candidates:
		return "", "", "name OCR found no text"

	best_match = max(ocr_candidates, key=len)
	contestant_number = max(number_candidates, key=len) if number_candidates else ""
	return best_match, contestant_number, None


def read_category_from_image(frame):
	"""Returns (is_male: bool|None, age_cat: str|None, status: str|None)."""
	if pytesseract is None:
		return None, None, "category OCR unavailable: install pytesseract"
	if tesseract_cmd is None:
		return None, None, f"category OCR unavailable: set {TESSERACT_ENV_VAR} or install Tesseract"

	processed = preprocess_name_for_ocr(frame)
	ocr_images = [
		processed,
		cv2.bitwise_not(processed),
	]
	char_whitelist = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/ "
	ocr_configs = (
		f"--oem 3 --psm 6 {char_whitelist}",
		f"--oem 3 --psm 7 {char_whitelist}",
	)
	ocr_texts = []

	try:
		for image in ocr_images:
			for config in ocr_configs:
				raw = pytesseract.image_to_string(image, config=config)
				if raw and raw.strip():
					ocr_texts.append(raw)
	except Exception as e:
		return None, None, f"category OCR failed: {e}"

	if not ocr_texts:
		return None, None, "category OCR found no text"

	text_upper = "\n".join(ocr_texts).upper()
	is_male = detect_gender_from_ocr_texts(ocr_texts)
	gender_tokens = []
	for raw in ocr_texts:
		gender_tokens.extend(tokenize_gender_ocr(raw))

	# Determine age category: exact substring match first, then closest word.
	age_cat = None
	_AGE_CATEGORIES = ["U15", "U17", "U19", "U21"]
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

	status = None
	if is_male is None:
		preview_tokens = gender_tokens[:8]
		if preview_tokens:
			status = f"gender OCR uncertain (detected: {' '.join(preview_tokens)})"
		else:
			status = "gender OCR uncertain (detected: no usable gender text)"

	return is_male, age_cat, status