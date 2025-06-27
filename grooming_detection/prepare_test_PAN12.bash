#!/bin/bash

# Set the project root directory
PROJECT_ROOT="/home/ak562fx/bac/grooming_detection"
LOG_DIR="$PROJECT_ROOT/logs/PAN12"
OUTPUT_DIR="$PROJECT_ROOT/outputs"

# Directory paths for metadata and data
METADATA_PATH="$PROJECT_ROOT/translated_data/PAN12/csv/PAN12-test-exp-sk.csv"

# Clean __pycache__ files
echo "Cleaning up python bytecode (__pycache__ directories)..."
find $PROJECT_ROOT -name "__pycache__" -exec rm -rf {} +
echo "Done."

echo "Clearing SLURM-specific log files..."
rm -f $LOG_DIR/*test*.err  
rm -f $LOG_DIR/*test*.out  # Remove all test logs
echo "Done."


# Verify data and metadata paths
echo "Checking if metadata and data paths exist..."
if [ ! -f "$METADATA_PATH" ]; then
    echo "Error: Metadata file not found at $METADATA_PATH"
    exit 1
fi
echo "Paths are valid."
