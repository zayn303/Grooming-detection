import os
import pandas as pd

PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
METADATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "metadata", "metadata.csv"))

def infer_split_from_path(path):
    path = path.lower()
    if "dev" in path:
        return "val"
    elif "eval" in path:
        return "eval"
    elif "val" in path or "validation" in path:
        return "val"
    elif "test" in path or "testing" in path:
        return "test"
    elif "train" in path or "training" in path:
        return "train"
    return "unknown"

def count_npy_files(directory):
    total = 0
    for _, _, files in os.walk(directory):
        total += sum(1 for f in files if f.endswith(".npy"))
    return total

def main():
    df = pd.read_csv(METADATA_PATH)

    print("=== Real/Fake Count per Split ===")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))

    print("\n=== Total Samples per Split ===")
    print(df["split"].value_counts())

    # Total .npy files in processed/
    npy_count = count_npy_files(PROCESSED_DIR)
    print(f"\n=== Total .npy files in data/processed/: {npy_count}")
    print(f"=== Total rows in metadata.csv: {len(df)}")

    # Check for mismatches
    df["split_inferred"] = df["path"].apply(infer_split_from_path)
    mismatches = (df["split"] != df["split_inferred"]).sum()
    print(f"=== Rows with split mismatch: {mismatches}")

if __name__ == "__main__":
    main()
