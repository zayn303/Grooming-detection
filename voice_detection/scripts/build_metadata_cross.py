import os
import csv
import hashlib
import random
from collections import defaultdict
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED = os.path.join(ROOT, "data", "processed")
METADATA = os.path.join(ROOT, "data", "metadata", "metadata_cross_balanced.csv")
RENAME_MAP = os.path.join(ROOT, "data", "metadata", "renamed_cross_balanced.csv")
DUPLICATES = os.path.join(ROOT, "data", "metadata", "duplicates_cross_balanced.csv")

INCLUDE = {
    "source": os.path.join(PROCESSED, "scene_fake"),
    "target": os.path.join(PROCESSED, "fake_or_real", "for-norm"),
}

def hash_content(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def hash_path(path):
    return hashlib.md5(path.encode("utf-8")).hexdigest()

def infer_label(parts):
    for p in parts[::-1]:
        p = p.lower()
        if p in ["real", "bonafide"]:
            return "real"
        elif p in ["fake", "spoof"]:
            return "fake"
    return "unknown"

def scan(include_dirs):
    seen = {}
    dups = []
    source, test = [], []

    for split_type, folder in include_dirs.items():
        print(f"Checking files in: {folder}")

        for root, _, files in os.walk(folder):
            for fname in files:
                if not fname.endswith(".npy"):
                    continue

                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, PROCESSED)
                parts = rel_path.replace("\\", "/").split("/")

                label = infer_label(parts)
                if label == "unknown":
                    continue

                # Determine split based on folder name
                if "scene_fake" in parts:
                    if "eval" in parts:
                        split = "test"
                    elif "dev" in parts:
                        split = "val"
                    elif "train" in parts:
                        split = "train"
                    else:
                        print(f"[WARNING] Skipping unrecognized scene_fake subfolder: {rel_path}")
                        continue
                else:
                    # For fake_or_real: try to detect split from path
                    if "validation" in parts:
                        split = "val"
                    elif "testing" in parts:
                        split = "test"
                    elif "training" in parts:
                        split = "train"
                    else:
                        print(f"[WARNING] Unrecognized folder in fake_or_real: {rel_path}")
                        continue


                content_hash = hash_content(full_path)
                if content_hash in seen:
                    dups.append({
                        "hash": content_hash,
                        "original": seen[content_hash],
                        "duplicate": rel_path,
                        "split": split
                    })
                    continue

                seen[content_hash] = rel_path
                entry = {
                    "orig_path": full_path,
                    "rel_path": rel_path,
                    "hash": content_hash,
                    "label": label,
                    "split": split
                }

                if split == "test":
                    test.append(entry)
                elif split in ["train", "val"]:
                    source.append(entry)
                else:
                    print(f"[WARNING] Skipping file with unknown split: {rel_path}")

    print(f"Found {len(source)} source entries and {len(test)} test entries.")
    return source, test, dups

def balance_test_data(test_data):
    fake_samples = [d for d in test_data if d["label"] == "fake"]
    real_samples = [d for d in test_data if d["label"] == "real"]

    print(f"Fake samples: {len(fake_samples)}, Real samples: {len(real_samples)}")

    min_samples = min(len(fake_samples), len(real_samples))
    balanced_test = fake_samples[:min_samples] + real_samples[:min_samples]
    random.shuffle(balanced_test)

    print(f"Balanced test set size: {len(balanced_test)}")
    return balanced_test

def split_train_val(data, ratio=0.8):
    out = {"train": [], "val": [], "test": []}

    # Separate by label
    fake = [d for d in data if d["label"] == "fake"]
    real = [d for d in data if d["label"] == "real"]

    # Determine how many samples to use (based on min of both)
    min_samples = min(len(fake), len(real))

    # Truncate both to match
    fake = fake[:min_samples]
    real = real[:min_samples]

    # Shuffle both
    random.shuffle(fake)
    random.shuffle(real)

    # Split into train/val
    train_cutoff = int(min_samples * ratio)

    out["train"] = fake[:train_cutoff] + real[:train_cutoff]
    out["val"] = fake[train_cutoff:] + real[train_cutoff:]

    # Shuffle final train/val
    random.shuffle(out["train"])
    random.shuffle(out["val"])

    # test will be added separately
    return out

def rename(entries_by_split):
    all_rows = []
    rename_map = []

    for split, entries in entries_by_split.items():
        for e in entries:
            new_fname = f"{hash_path(e['rel_path'])}.npy"
            new_path = os.path.join(os.path.dirname(e["orig_path"]), new_fname)
            os.rename(e["orig_path"], new_path)
            rel_new_path = os.path.relpath(new_path, PROCESSED)

            all_rows.append({
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

    print(f"Renamed {len(all_rows)} files.")
    return all_rows, rename_map

def save_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved metadata to {path}.")

if __name__ == "__main__":
    print("Scanning...")
    source, test, dups = scan(INCLUDE)

    splits = split_train_val(source)
    splits["test"] = balance_test_data(test)

    print("Renaming and saving...")
    meta, renames = rename(splits)

    # Shuffle metadata
    meta = pd.DataFrame(meta).sample(frac=1, random_state=42).to_dict("records")

    save_csv(METADATA, meta, ["path", "filename", "label", "split"])
    save_csv(RENAME_MAP, renames, ["original_path", "new_filename", "new_path"])
    save_csv(DUPLICATES, dups, ["hash", "original", "duplicate", "split"])
    print(f"✔ Done: {len(meta)} entries, {len(dups)} duplicates")
