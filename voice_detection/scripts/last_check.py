import pandas as pd
import hashlib
import os
from collections import defaultdict

def hash_file(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

df = pd.read_csv("data/metadata/metadata.csv")
df["full_path"] = df["path"].apply(lambda p: os.path.join("data/processed", p))
df["hash"] = df["full_path"].apply(hash_file)

hash_to_splits = defaultdict(set)
for _, row in df.iterrows():
    hash_to_splits[row["hash"]].add(row["split"])

leaked = {h: s for h, s in hash_to_splits.items() if len(s) > 1}

if leaked:
    print(f" Found {len(leaked)} leaks across splits!")
    for h, splits in leaked.items():
        print(f" - Hash {h} occurs in: {splits}")
else:
    print(" No data leakage detected between splits.")
