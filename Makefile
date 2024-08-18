VENV_DIR = venv_api

.PHONY: venv install run

venv:
	# Deactivate any existing virtual environment
	@echo "Deactivating any existing virtual environment"
	@if [ -n "$$VIRTUAL_ENV" ]; then deactivate; fi
	# Remove existing virtual environment if it exists
	@echo "Removing existing virtual environment if it exists"
	rm -rf $(VENV_DIR)
	# Create a new virtual environment
	@echo "Creating a new virtual environment"
	python3 -m venv $(VENV_DIR)

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
