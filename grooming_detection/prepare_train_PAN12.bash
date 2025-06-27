#!/bin/bash

# Set the project root directory
PROJECT_ROOT="/home/ak562fx/bac/grooming_detection"
LOG_DIR="$PROJECT_ROOT/logs/PAN12"
OUTPUT_DIR="$PROJECT_ROOT/outputs/PAN12"

# Directory paths for metadata
TRAIN_METADATA_PATH="$PROJECT_ROOT/translated_data/PAN12/csv/PAN12-train-exp-sk.csv" 
VAL_METADATA_PATH="$PROJECT_ROOT/translated_data/PAN12/csv/PAN12-val-exp-sk.csv"   

# Clean __pycache__ files
echo "Cleaning up python bytecode (__pycache__ directories)..."
find "$PROJECT_ROOT" -name "__pycache__" -exec rm -rf {} +
echo "Done."

echo "Clearing SLURM-specific log files..."
rm -f $LOG_DIR/*train*.err
rm -f $LOG_DIR/*train*.out
echo "Done."

# Remove previous model checkpoints
echo "Clearing previous model checkpoints..."
rm -rf $OUTPUT_DIR/*
echo "Done."

# Verify data and metadata paths
echo "Checking if metadata paths exist..."
if [ ! -f "$TRAIN_METADATA_PATH" ]; then
    echo "Error: Training metadata file not found at $TRAIN_METADATA_PATH"
    exit 1
fi
if [ ! -f "$VAL_METADATA_PATH" ]; then
    echo "Error: Validation metadata file not found at $VAL_METADATA_PATH"
    exit 1
fi
echo "Metadata paths are valid."
