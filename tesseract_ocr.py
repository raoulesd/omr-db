import os
import shutil
from pathlib import Path

try:
	import pytesseract
except ImportError:
	pytesseract = None


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