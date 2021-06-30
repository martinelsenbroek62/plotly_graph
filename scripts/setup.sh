#!/bin/bash 

RED='\033[0;31m'
GREEN='\033[0;32m'
BROWN='\033[0;33m'
YELLOW='\033[1;33m'
NC='\033[0;0m' # No Color

# Helpers
function set_source {
	WINDOWS_SCRIPT_DIR="venv/scripts/activate"
	UNIX_SCRIPT_DIR="venv/bin/activate"

	if [ -d "$WINDOWS_SCRIPT_DIR" ]; then
		ACTIVATE_SCRIPT_DIR="$WINDOWS_SCRIPT_DIR"
	else 
		ACTIVATE_SCRIPT_DIR="$UNIX_SCRIPT_DIR"
	fi

	source "$ACTIVATE_SCRIPT_DIR"
}

# Script start
echo -e "${GREEN}==================================="
echo -e "Creating Python virtual environment"
echo -e "===================================${NC}"
python -m venv venv
set_source
echo -e "Done"

echo -e "${GREEN}========================="
echo -e "Installing Python modules"
echo -e "=========================${NC}"
pip install -r requirements.txt
echo "Done"

echo -e "${GREEN}===================="
echo -e "Creating config file"
echo -e "====================${NC}"
cp ./config.example.json ./config.json
echo -e "Done"
echo -e "${BROWN}WARNING: Make sure to review and update the config.json${NC}"
echo -e "Setup complete"
