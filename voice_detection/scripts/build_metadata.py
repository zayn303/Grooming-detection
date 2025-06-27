import os
import csv
import hashlib
import random
from collections import defaultdict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
INCLUDE_DIRS = [
    os.path.join(PROCESSED_DIR, "fake_or_real", "for-2sec", "for-2seconds"),
    os.path.join(PROCESSED_DIR, "fake_or_real", "for-norm"),
    os.path.join(PROCESSED_DIR, "fake_or_real", "for-rerec", "for-rerecorded")
]
METADATA_PATH = os.path.join(ROOT_DIR, "data", "metadata", "metadata.csv")
RENAME_MAP_PATH = os.path.join(ROOT_DIR, "data", "metadata", "renamed_files.csv")
DUPLICATES_LOG_PATH = os.path.join(ROOT_DIR, "data", "metadata", "skipped_duplicates.csv")

def hash_file_content(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def hash_path(path):
    return hashlib.md5(path.encode("utf-8")).hexdigest()

def infer_label(path_parts):
    for part in reversed(path_parts):
        p = part.lower()
        if p in ["real", "bonafide"]:
            return "real"
        elif p in ["fake", "spoof"]:
            return "fake"
    return "unknown"

def scan_all_files(include_dirs):
    seen_hashes = {}
    duplicates = []
    entries = []

    for root_dir in include_dirs:
        for root, _, files in os.walk(root_dir):
            for fname in files:
                if not fname.endswith(".npy"):
                    continue

                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, PROCESSED_DIR)
                path_parts = rel_path.replace("\\", "/").split("/")

                label = infer_label(path_parts)
                if label == "unknown":
                    continue

                content_hash = hash_file_content(full_path)
                if content_hash in seen_hashes:
                    duplicates.append({
                        "hash": content_hash,
                        "original_path": seen_hashes[content_hash],
                        "duplicate_path": rel_path
                    })
                    continue

                seen_hashes[content_hash] = rel_path
                entries.append({
                    "original_path": full_path,
                    "rel_path": rel_path,
                    "hash": content_hash,
                    "label": label
                })

    return entries, duplicates

def stratified_split(entries, ratio):
    label_groups = defaultdict(list)
    for e in entries:
        label_groups[e["label"]].append(e)

    final = {"train": [], "val": [], "test": []}
    for label, items in label_groups.items():
        random.shuffle(items)
        n_total = len(items)
        n_train = int(n_total * ratio["train"])
        n_val = int(n_total * ratio["val"])
        n_test = n_total - n_train - n_val

        final["train"] += items[:n_train]
        final["val"] += items[n_train:n_train + n_val]
        final["test"] += items[n_train + n_val:]

    return final

def rename_and_write(entries_by_split):
    metadata_rows = []
    rename_map = []

    for split, items in entries_by_split.items():
        for e in items:
            new_fname = f"{hash_path(e['rel_path'])}.npy"
            new_full_path = os.path.join(os.path.dirname(e["original_path"]), new_fname)
            os.rename(e["original_path"], new_full_path)
            rel_new_path = os.path.relpath(new_full_path, PROCESSED_DIR)

            metadata_rows.append({
                "path": rel_new_path,
                "filename": new_fname,
                "label": e["label"],
                "split": split
            })
            rename_map.append({
                "original_path": e["rel_path"],
                "new_filename": new_fname,
                "new_path": rel_new_path
            })

    return metadata_rows, rename_map

def save_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    print(" Scanning files and hashing contents...")
    all_entries, duplicates = scan_all_files(INCLUDE_DIRS)
    print(f" Total unique: {len(all_entries)} — 🧠 Duplicates: {len(duplicates)}")

    print(" Performing 70/15/15 split per label...")
    split_entries = stratified_split(all_entries, {"train": 0.7, "val": 0.15, "test": 0.15})

    print(" Renaming files and creating metadata...")
    metadata_rows, rename_map = rename_and_write(split_entries)

    print(" Saving CSVs...")
    save_csv(METADATA_PATH, metadata_rows, ["path", "filename", "label", "split"])
    save_csv(RENAME_MAP_PATH, rename_map, ["original_path", "new_filename", "new_path"])
    save_csv(DUPLICATES_LOG_PATH, duplicates, ["hash", "original_path", "duplicate_path"])

    print(f" Done: {len(metadata_rows)} samples, {len(duplicates)} duplicates removed")
