#!/bin/bash

# Define the venv directory
VENV_DIR="./venv"

# Check if the virtual environment already exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
    echo "Virtual environment created."
else
    echo "Virtual environment already exists. Activating..."
fi

# Activate the virtual environment in the current shell
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "No requirements.txt found."
fi

echo "Virtual environment is activated."
