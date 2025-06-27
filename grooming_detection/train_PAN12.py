import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from scripts.utils.dataset import GroomingDetectionDataset

# Seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Config
SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", f"local_{int(time.time())}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_CSV = "/home/ak562fx/bac/grooming_detection/translated_data/PAN12/csv/PAN12-train-exp-sk.csv"
VAL_CSV = "/home/ak562fx/bac/grooming_detection/translated_data/PAN12/csv/PAN12-val-exp-sk.csv"
MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 512
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EVAL_EVERY = 1
EARLY_STOPPING_PATIENCE = 5
OUTPUT_DIR = "outputs/PAN12"
os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def get_dataloaders():
    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_train['label'] = df_train['label'].str.strip().str.lower()
    df_val['label'] = df_val['label'].str.strip().str.lower()
    label_map = {'non-predator': 0, 'predator': 1}
    df_train['label_id'] = df_train['label'].map(label_map)
    df_val['label_id'] = df_val['label'].map(label_map)
    counts = df_train['label_id'].value_counts().sort_index()
    total = counts.sum()
    weight_0 = total / (2.0 * counts.get(0, 1))
    weight_1 = total / (2.0 * counts.get(1, 1))
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float).to(DEVICE)
    train_dataset = GroomingDetectionDataset(df_train, tokenizer, MAX_LEN)
    val_dataset = GroomingDetectionDataset(df_val, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return train_loader, val_loader, class_weights

def evaluate(model, loader, criterion):
    model.eval()
    all_preds, all_labels = [], []
    loss_total = 0
    with torch.no_grad():
        for batch in loader:
            ids, mask, labels = [x.to(DEVICE) for x in batch]
            outputs = model(ids, attention_mask=mask, labels=labels)
            loss_total += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    loss_avg = loss_total / len(loader)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    print("\n--- Eval Report ---")
    print(classification_report(all_labels, all_preds, target_names=['non-predator', 'predator']))
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    return loss_avg, acc, p, r, f1

def train():
    print(f"Using device: {DEVICE}")
    train_loader, val_loader, class_weights = get_dataloaders()
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader) * NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(DEVICE)
    best_f1, patience = 0.0, 0
    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        loss_sum = 0
        for batch in train_loader:
            optimizer.zero_grad()
            ids, mask, labels = [x.to(DEVICE) for x in batch]
            outputs = model(ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
        avg_loss = loss_sum / len(train_loader)
        print(f"[Epoch {epoch}/{NUM_EPOCHS}] Train Loss: {avg_loss:.4f}")
        # Save model for every epoch (optional)
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"model_epoch_{epoch}_{SLURM_JOB_ID}.pth"))

        if epoch % EVAL_EVERY == 0:
            val_loss, val_acc, val_p, val_r, val_f1 = evaluate(model, val_loader, criterion)
            print(f"Eval: Loss={val_loss:.4f} | Acc={val_acc:.4f} | F1={val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience = 0
                best_path = os.path.join(OUTPUT_DIR, f"best_model_{SLURM_JOB_ID}.pth")
                torch.save(model.state_dict(), best_path)
                print(f"✓ Best model saved: {best_path}")
                # Save metrics as JSON
                metrics_path = os.path.join(OUTPUT_DIR, f"metrics_{SLURM_JOB_ID}.json")
                with open(metrics_path, "w") as f:
                    json.dump({
                        "epoch": epoch,
                        "f1": val_f1,
                        "accuracy": val_acc,
                        "precision": val_p,
                        "recall": val_r,
                        "loss": val_loss
                    }, f, indent=2)
            else:
                patience += 1
                print(f"No improvement. Patience: {patience}/{EARLY_STOPPING_PATIENCE}")
                if patience >= EARLY_STOPPING_PATIENCE:
                    print("Early stopping.")
                    break
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"final_model_{SLURM_JOB_ID}.pth"))

if __name__ == "__main__":
    train()
