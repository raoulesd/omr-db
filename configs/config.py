import os
import json

"""
This module manages the configuration files for the system.
It has a slot for a format-specific config (ACTIVE_CONFIG) and a slot for a system-wide config (ACTIVE_SYSTEM_CONFIG).
The format-specific config is meant to hold properties specific to a particular sheet layout or format (e.g. db9-2024),
while the system-wide config is meant to hold properties that are shared across all formats.
The format-specific config will override the system-wide config if there are any overlapping properties.
"""

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
        raise ValueError("Config name cannot be empty")

    global ACTIVE_CONFIGS, ACTIVE_CONFIG_NAMES

    # Load the config from the file
    if not config_name.endswith(".json"):
        config_path = config_name + ".json"
    else:
        config_path = config_name
    config_path = "./configs/" + config_name + ".json"

    # Check if the config file exists
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found")
    
    # Load the config file
    with open(config_path, "r") as f:
        config_data = json.load(f)

    return config_data

def get_property(property_name: str):
    """
    Gets a specific property from the active config.

    Raises an error if the property is not found
    """
    global ACTIVE_CONFIG, ACTIVE_SYSTEM_CONFIG
    

    # First check the active config for the property, then check the system config if it's not found in the active config
    if ACTIVE_CONFIG is not None and property_name in ACTIVE_CONFIG:
        return ACTIVE_CONFIG[property_name]
    
    if ACTIVE_SYSTEM_CONFIG is not None and property_name in ACTIVE_SYSTEM_CONFIG:
        return ACTIVE_SYSTEM_CONFIG[property_name]
    
    
    # If neither the active config nor the system config has the property, raise an error
    raise ValueError(f"Config property '{property_name}' not found in active config or system config")

def has_property(property_name: str):
    """
    Checks if a specific property exists in the active config or the system config.
    """
    global ACTIVE_CONFIG, ACTIVE_SYSTEM_CONFIG

    if ACTIVE_CONFIG is not None and property_name in ACTIVE_CONFIG:
        return True
    
    if ACTIVE_SYSTEM_CONFIG is not None and property_name in ACTIVE_SYSTEM_CONFIG:
        return True
    
    return False