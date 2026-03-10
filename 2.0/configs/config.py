import importlib
import importlib.util
import os
import pathlib

_REQUIRED_KEYS = (
    "ROWS",
    "COLS",
    "circularity",
    "extent",
    "hull",
    "debug_mode",
    "epsilon",
    "FILL_METHOD",
    "ID_TL",
    "ID_TR",
    "ID_BR",
    "ID_BL",
    "ARUDO_DICT",
    "offset_tl",
    "offset_tr",
    "offset_br",
    "offset_bl",
    "has_bounded_question_area",
    "SCANNED_FILES_DIR",
    "PROCESSED_FILES_DIR",
    "ERRORED_FILES_DIR",
    "RESULTS_CSV_PATH",
    "UI_AREAS",
    "UI_SCALE",
    "FRAME_WIDTH",
    "FRAME_HEIGHT",
    "ATTEMPT_TOTALS_HEIGHT",
    "ZONES_AND_TOPS_WIDTH",
    "NAME_DATA_WIDTH",
    "NAME_DATA_HEIGHT",
)

_ACTIVE_CONFIG = None
_ACTIVE_CONFIG_NAME = None


def _normalize_config_name(config_name):
    if not config_name:
        return "config-db9-13022026"

    return config_name.strip()


def _validate_config_module(config_module, config_name):
    missing = [key for key in _REQUIRED_KEYS if not hasattr(config_module, key)]
    if missing:
        raise ValueError(
            f"Config '{config_name}' is missing required keys: {', '.join(missing)}"
        )


def _load_module_from_python_file(file_path):
    module_id = f"dynamic_config_{hash(file_path)}"
    spec = importlib.util.spec_from_file_location(module_id, file_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load config from file: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_config_module(config_name):
    normalized = _normalize_config_name(config_name)
    cwd = pathlib.Path(__file__).resolve().parent

    # Allow bare names like "config-dbiyo2026".
    if normalized.endswith(".py"):
        file_candidate = cwd / normalized
    else:
        file_candidate = cwd / f"{normalized}.py"

    if file_candidate.exists():
        return _load_module_from_python_file(str(file_candidate.resolve())), file_candidate.stem

    # Fallback for module-style names like "configs.db9".
    module_name = normalized
    if module_name.endswith(".py"):
        module_name = module_name[:-3]
    module_name = module_name.replace("\\", ".").replace("/", ".")
    module = importlib.import_module(module_name)
    return module, module_name


def set_active_config(config_name="config-db9-13022026"):
    global _ACTIVE_CONFIG
    global _ACTIVE_CONFIG_NAME

    config_module, resolved_name = _load_config_module(config_name)
    _validate_config_module(config_module, resolved_name)

    _ACTIVE_CONFIG = config_module
    _ACTIVE_CONFIG_NAME = resolved_name
    return _ACTIVE_CONFIG


def get_active_config():
    global _ACTIVE_CONFIG
    if _ACTIVE_CONFIG is None:
        default_name = os.getenv("OMR_CONFIG_NAME", "config-db9-13022026")
        return set_active_config(default_name)
    return _ACTIVE_CONFIG


def get_active_config_name():
    if _ACTIVE_CONFIG_NAME is None:
        get_active_config()
    return _ACTIVE_CONFIG_NAME
