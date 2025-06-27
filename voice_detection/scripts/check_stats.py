import pandas as pd

CSV_PATH = "data/metadata/all_files.csv"

print("=== Distribution Stats from all_files.csv ===")

try:
    df = pd.read_csv(CSV_PATH, usecols=["label", "split"])
except Exception as e:
    print(" Failed to load all_files.csv:", e)
    exit()

# Normalize
df["label"] = df["label"].str.strip().str.lower()
df["split"] = df["split"].str.strip().str.lower()

print("\n→ Split × Label Summary:")
summary = df.groupby(["split", "label"]).size().unstack(fill_value=0)
print(summary)

print("\n→ Total Samples per Split:")
print(df["split"].value_counts())

print("\n→ Unique Splits and Labels:")
print("Splits:", sorted(df["split"].unique()))
print("Labels:", sorted(df["label"].unique()))
