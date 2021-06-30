import os

import modules.helpers as helpers_module

SOFTWARE_VERSION = "0.0.1"

APP_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
APP_CONFIG = helpers_module.read_config(APP_DIR_PATH)
OUTPUT_DIR_PATH = f"{APP_DIR_PATH}/{APP_CONFIG['toolPathGenerator']['outputDirectory']}"

SECONDS_PER_MINUTE = 60
MM_PER_INCH = 25.4
INCH_PER_FT = 12

# Convert feedrate from mm per minute to mm per second
FEEDRATE_PER_SECOND = APP_CONFIG["toolPathGenerator"]["scans"]["feedrate"] / SECONDS_PER_MINUTE
SPACING_PER_X_INDEX = FEEDRATE_PER_SECOND / APP_CONFIG["toolPathGenerator"]["scanner"]["frequency"]
SPACING_PER_Z_INDEX = APP_CONFIG["toolPathGenerator"]["scanner"]["zSpacing"]

DEFAULT_SCANNER_ACCURACY_RANGE=APP_CONFIG["toolPathGenerator"]["scanner"]["yAxisAccuracy"]