#!/bin/bash 

# Helpers
function set_source {
	WINDOWS_SCRIPT_DIR="venv/Scripts"
	UNIX_SCRIPT_DIR="venv/bin"

	if [ -d "$WINDOWS_SCRIPT_DIR" ]; then
		ACTIVATE_SCRIPT_DIR="$WINDOWS_SCRIPT_DIR"
	else 
		ACTIVATE_SCRIPT_DIR="$UNIX_SCRIPT_DIR"
	fi

	source "$ACTIVATE_SCRIPT_DIR/activate"
}

# Script start
set_source
python app.py
