VENV_DIR = venv_api
PYTHON_VERSION = python3.12

.PHONY: venv install run

venv:
	# Check if the virtual environment exists before creating it
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating a new virtual environment with $(PYTHON_VERSION)"; \
		$(PYTHON_VERSION) -m venv $(VENV_DIR); \
	else \
		echo "Virtual environment already exists"; \
	fi

install: venv
	# Activate the virtual environment and install dependencies
	@echo "Activating the virtual environment and installing dependencies"
	. $(VENV_DIR)/bin/activate && pip install -r requirements_api.txt
	# Download the SpaCy model
	@echo "Downloading SpaCy model 'en_core_web_sm'"
	. $(VENV_DIR)/bin/activate && python -m spacy download en_core_web_sm

run:
	# Activate the virtual environment, set environment variables, and run the Flask server
	@echo "Running the Flask server"
	. $(VENV_DIR)/bin/activate && \
	export FLASK_APP=app.py && \
	export FLASK_ENV=development && \
	flask run
