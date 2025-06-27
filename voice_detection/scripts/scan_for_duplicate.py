
import os
import csv
import hashlib
import pandas as pd

def hash_file(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Step 1: Check for duplicate file content
hashes = {}
duplicates = []

for root, _, files in os.walk("data/processed"):
    for f in files:
        if f.endswith(".npy"):
            path = os.path.join(root, f)
            h = hash_file(path)
            if h in hashes:
                duplicates.append({
                    "duplicate_path": path,
                    "original_path": hashes[h],
                    "hash": h
                })
            else:
                hashes[h] = path

# Write duplicates to CSV
output_path = "data/metadata/duplicates_by_content.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", newline="") as csvfile:
    fieldnames = ["hash", "original_path", "duplicate_path"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in duplicates:
        writer.writerow(row)

print(f"✅ Step 1: Wrote {len(duplicates)} duplicates to {output_path}")

# Step 2: Check for overlapping hashes across train/val/test
df = pd.read_csv("data/metadata/metadata_cross_balanced.csv")
df["hash"] = df["filename"].apply(lambda x: os.path.splitext(x)[0])
split_hashes = df.groupby("split")["hash"].apply(set)

print("\n🔍 Step 2: Checking for hash overlaps between splits:")
for s1 in split_hashes.index:
    for s2 in split_hashes.index:
        if s1 != s2:
            overlap = split_hashes[s1] & split_hashes[s2]
            if overlap:
                print(f"❌ Overlap between {s1} and {s2}: {len(overlap)} hashes")
            else:
                print(f"✅ No overlap between {s1} and {s2}")
