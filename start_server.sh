#!/bin/bash

# Navigate to the api directory
cd "$(dirname "$0")"

# Check if the virtual environment exists
if [ ! -d "venv_api" ]; then
    # Create a virtual environment
    python3 -m venv venv_api
fi

# Activate the virtual environment
source venv_api/bin/activate

# Install dependencies
pip install -r requirements_api.txt

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development

# Run the Flask server
flask run
