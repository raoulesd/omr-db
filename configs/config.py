import os
import json
from pathlib import Path

"""
This module manages the configuration files for the system.
It has a slot for a format-specific config (ACTIVE_CONFIG) and a slot for a system-wide config (ACTIVE_SYSTEM_CONFIG).
The format-specific config is meant to hold properties specific to a particular sheet layout or format (e.g. db9-2024),
while the system-wide config is meant to hold properties that are shared across all formats.
The format-specific config will override the system-wide config if there are any overlapping properties.
"""

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "configs"

ACTIVE_CONFIG = None
ACTIVE_CONFIG_NAME = None
ACTIVE_SYSTEM_CONFIG = None
ACTIVE_SYSTEM_CONFIG_NAME = None

def set_active_system_config(config_name: str):
    """Sets the active system config by loading it from a file.

    Raises an error if the config file is not found or if the config name is empty.
    """
    global ACTIVE_SYSTEM_CONFIG, ACTIVE_SYSTEM_CONFIG_NAME
    config = get_config_from_disk(config_name)
    ACTIVE_SYSTEM_CONFIG = config
    ACTIVE_SYSTEM_CONFIG_NAME = config_name


def set_active_config(config_name: str):
    """Sets the active config by loading it from a file.

    Raises an error if the config file is not found or if the config name is empty.
    """
    global ACTIVE_CONFIG, ACTIVE_CONFIG_NAME
    config = get_config_from_disk(config_name)
    ACTIVE_CONFIG = config
    ACTIVE_CONFIG_NAME = config_name


def get_config_from_disk(config_name: str):
    """Gets the config data from the disk

    Raises an error if the config file is not found or if the config name is empty.
    """
    if not config_name:
        error_message = "Config name cannot be empty"
        raise ValueError(error_message)

    # Load the config from the file
    if not config_name.endswith(".json"):
        config_path = config_name + ".json"
    else:
        config_path = config_name
    config_path = CONFIG_DIR / (config_name + ".json")

    # Check if the config file exists
    if not config_path.is_file():
        error_message = f'Config file "{config_path}" not found'
        raise FileNotFoundError(error_message)

    # Load the config file
    with config_path.open("r") as f:
        config_data = json.load(f)

    return config_data

def get_property(property_name: str):
    """Gets a specific property from the active config.

    Raises an error if the property is not found
    """

    # First check the active config for the property, then check the system config if it's not found in the active config
    if ACTIVE_CONFIG is not None and property_name in ACTIVE_CONFIG:
        return ACTIVE_CONFIG[property_name]

    if ACTIVE_SYSTEM_CONFIG is not None and property_name in ACTIVE_SYSTEM_CONFIG:
        return ACTIVE_SYSTEM_CONFIG[property_name]

    # If neither the active config nor the system config has the property, raise an error
    error_message = f'Config property "{property_name}" not found in active config or system config'
    raise ValueError(error_message)

def has_property(property_name: str):
    """Checks if a specific property exists in the active config or the system config.
    """
    if ACTIVE_CONFIG is not None and property_name in ACTIVE_CONFIG:
        return True

    if ACTIVE_SYSTEM_CONFIG is not None and property_name in ACTIVE_SYSTEM_CONFIG:
        return True

    return False

def print_properties():
    """Prints all properties from the active config and the system config, with the active config properties taking precedence over the system config properties in case of overlap.
    """
    print("Active Config Properties:")
    if ACTIVE_CONFIG is not None:
        for key, value in ACTIVE_CONFIG.items():
            print(f"{key}: {value}")
    else:
        print("No active config set")

    print("\nSystem Config Properties:")
    if ACTIVE_SYSTEM_CONFIG is not None:
        for key, value in ACTIVE_SYSTEM_CONFIG.items():
            print(f"{key}: {value}")
    else:
        print("No active system config set")
