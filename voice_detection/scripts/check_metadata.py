import argparse
import pandas as pd
from collections import Counter

# Setup argument parser
parser = argparse.ArgumentParser(description="Check metadata balance")
parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset CSV file (e.g., metadata_cross.csv)")

# Parse arguments
args = parser.parse_args()

print("=== Lightweight Metadata Checker ===")

# Load the metadata CSV file based on the provided dataset path
try:
    df = pd.read_csv(f"/home/ak562fx/bac/voice_detection/data/metadata/{args.dataset}.csv", usecols=["label", "split"])
except Exception as e:
    print(f"Failed to load {args.dataset}.csv:", e)
    exit()

# Process the 'label' and 'split' columns
df['label'] = df['label'].str.lower().str.strip()
df['split'] = df['split'].str.lower().str.strip()

# Print split summary
print("\n=== Split Summary ===")
split_summary = df.groupby(['split', 'label']).size().unstack(fill_value=0)
print(split_summary)

# Print total samples per split
print("\n=== Total Samples per Split ===")
print(df['split'].value_counts())

# Checking for unknown splits or labels
print("\n=== Checking for Unknown Splits or Labels ===")
print("Unique splits:", df['split'].unique())
print("Unique labels:", df['label'].unique())
