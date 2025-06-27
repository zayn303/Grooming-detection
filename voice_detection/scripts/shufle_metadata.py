import pandas as pd
df = pd.read_csv("data/metadata/metadata_cross.csv")
df["label"] = df["label"].sample(frac=1.0).reset_index(drop=True)
df.to_csv("data/metadata/metadata_cross_random.csv", index=False)