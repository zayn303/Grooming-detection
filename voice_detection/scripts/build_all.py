import os
import csv

BASE_DIR = "data/processed"
OUTPUT_CSV = "data/metadata/all_files.csv"

entries = []

for root, _, files in os.walk(BASE_DIR):
    for file in files:
        if not file.endswith(".npy"):
            continue
        full_path = os.path.join(root, file)
        rel_path = os.path.relpath(full_path, BASE_DIR)
        path_parts = rel_path.replace("\\", "/").split("/")

        label = None
        split = None
        for part in path_parts:
            part = part.lower()
            if part in ["real", "bonafide"]:
                label = "real"
            elif part in ["fake", "spoof"]:
                label = "fake"

            if part in ["train", "training"]:
                split = "train"
            elif part in ["val", "validation", "dev"]:
                split = "val"
            elif part in ["test", "testing", "eval"]:
                split = "test"

        if label and split:
            entries.append({
                "path": rel_path,
                "filename": file,
                "label": label,
                "split": split,
            })

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["path", "filename", "label", "split"])
    writer.writeheader()
    writer.writerows(entries)

print(f"Saved {len(entries)} entries to {OUTPUT_CSV}")
