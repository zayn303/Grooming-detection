#!/bin/bash

# Set the project root directory
PROJECT_ROOT="/home/ak562fx/bac/voice_detection"
LOG_DIR="$PROJECT_ROOT/logs"
OUTPUT_DIR="$PROJECT_ROOT/outputs"

# Directory paths for metadata
TEST_METADATA_PATH="$PROJECT_ROOT/data/metadata/metadata_cross_balanced.csv" 

# Clean __pycache__ files
echo "Cleaning up python bytecode (__pycache__ directories)..."
find "$PROJECT_ROOT" -name "__pycache__" -exec rm -rf {} +
echo "Done."

echo "Clearing SLURM-specific log files..."
rm -f $LOG_DIR/*test*.err
rm -f $LOG_DIR/*test*.out
echo "Done."


# Verify data and metadata paths
echo "Checking if metadata paths exist..."
if [ ! -f "$TEST_METADATA_PATH" ]; then
    echo "Error: Testing metadata file not found at $TEST_METADATA_PATH"
    exit 1
fi

echo "Metadata paths are valid."
