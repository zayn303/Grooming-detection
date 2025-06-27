import pandas as pd
import torch
from transformers import AutoTokenizer

class GroomingDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer=None, max_len=512, verbose=False):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.max_len = max_len
        if verbose:
            print(f"[DATASET] Loaded {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        chat_text = str(row["segment"])
        if not chat_text or chat_text.strip().lower() == 'nan':
            chat_text = "[EMPTY]"  # or some safe default fallback

        encoding = self.tokenizer(
            chat_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )


        label_str = row["label"]
        label = 0 if label_str == "non-predator" else 1

        return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0), torch.tensor(label, dtype=torch.long)
