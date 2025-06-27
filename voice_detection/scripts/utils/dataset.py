import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import random
import librosa

LABEL2ID = {"real": 1, "fake": 0}

class VoiceDeepfakeDataset(Dataset):
    def __init__(self, metadata_path, data_dir, split="train", augment=None, verbose=False):
        self.data_dir = data_dir
        self.split = split
        self.augment = (split == "train") if augment is None else augment

        # Load metadata CSV
        df = pd.read_csv(metadata_path)
        df["label"] = df["label"].str.strip().str.lower()
        df["split"] = df["split"].str.strip().str.lower()
        df = df[df["split"] == split]
        df = df[df["label"].isin(LABEL2ID.keys())]

        if verbose:
            print(f"[{split.upper()}] Loaded {len(df)} samples")

        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row["path"]
        label_str = row["label"]

        npy_path = os.path.join(self.data_dir, rel_path)

        # Check if the file exists
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Missing file: {npy_path}")

        mel = np.load(npy_path).astype(np.float32)

        if mel.ndim == 2:
            mel = np.expand_dims(mel, axis=0)
        elif mel.ndim != 3:
            raise ValueError(f"Bad shape {mel.shape} in file: {npy_path}")

        # DEBUG PRINT — only once per session
        if idx == 0 and self.split == "val":
            print(f"[DEBUG] MEL SHAPE: {mel.shape}, MIN: {mel.min():.4f}, MAX: {mel.max():.4f}")

        if self.augment:
            mel = self.apply_augmentations(mel)

        label = LABEL2ID[label_str]
        return torch.tensor(mel), torch.tensor(label, dtype=torch.long)

    def apply_augmentations(self, mel):
        mel = mel.copy()

        # 1. Add Gaussian noise
        if random.random() < 0.5:
            mel += np.random.normal(0, 0.01, size=mel.shape).astype(np.float32)

        # 2. Random time shift (roll)
        if random.random() < 0.5:
            shift = random.randint(1, mel.shape[2] // 8)
            mel = np.roll(mel, shift, axis=2)

        # 3. Frequency masking (SpecAugment)
        if random.random() < 0.5:
            f = random.randint(0, 15)
            f0 = random.randint(0, max(1, mel.shape[1] - f))
            mel[:, f0:f0+f, :] = 0

        # 4. Time masking (SpecAugment)
        if random.random() < 0.5:
            t = random.randint(0, 20)
            t0 = random.randint(0, max(1, mel.shape[2] - t))
            mel[:, :, t0:t0+t] = 0

        return mel
