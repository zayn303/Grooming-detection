import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from scripts.utils.dataset import VoiceDeepfakeDataset
from models.cnn_lstm import CNN_LSTM

torch.backends.cudnn.benchmark = True

# Seed setup
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Config
SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "nojob")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 100
EVAL_EVERY = 2
EARLY_STOPPING = 5
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_dataloaders():
    metadata_path = "data/metadata/metadata_cross_balanced.csv"
    data_dir = "data/processed"
    train_ds = VoiceDeepfakeDataset(metadata_path, data_dir, split="train", verbose=True)
    val_ds = VoiceDeepfakeDataset(metadata_path, data_dir, split="val", augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, worker_init_fn=lambda _: np.random.seed())
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return train_loader, val_loader

def get_criterion(model_output):
    return nn.BCEWithLogitsLoss() if model_output.shape[1] == 1 else nn.CrossEntropyLoss()

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    dumped = False

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = model(X)
            if outputs.shape[1] == 1:
                loss = criterion(outputs.squeeze(), y.float())
                preds = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
            else:
                loss = criterion(outputs, y)
                preds = outputs.argmax(1)

            if not dumped:
                print("=== VAL DUMP SAMPLE ===")
                for i in range(min(5, X.size(0))):
                    print(f"  [REAL: {y[i].item()} | PRED: {preds[i].item()}]")
                dumped = True

            total_loss += loss.item() * X.size(0)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    loss_avg = total_loss / total
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    print("\n--- Eval Report ---")
    print(classification_report(all_labels, all_preds, target_names=['fake', 'real']))
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    return loss_avg, acc, p, r, f1

def train():
    train_loader, val_loader = get_dataloaders()
    model = CNN_LSTM(in_channels=1, num_classes=2).to(DEVICE)

    dummy_input = torch.randn(1, 1, 128, 94).to(DEVICE)
    criterion = get_criterion(model(dummy_input))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_f1, patience = 0.0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs.squeeze() if outputs.shape[1] == 1 else outputs,
                             y.float() if outputs.shape[1] == 1 else y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * X.size(0)

        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"[Epoch {epoch}/{NUM_EPOCHS}] Train Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"model_epoch_{epoch}_{SLURM_JOB_ID}.pth"))

        if epoch % EVAL_EVERY == 0:
            val_loss, val_acc, val_p, val_r, val_f1 = evaluate(model, val_loader, criterion)
            print(f"  → Eval on val: Loss = {val_loss:.4f} | Accuracy = {val_acc:.4f} | F1 = {val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience = 0
                best_model_path = os.path.join(OUTPUT_DIR, f"best_model_{SLURM_JOB_ID}.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"  ✓ Best model saved: {best_model_path}")
                with open(os.path.join(OUTPUT_DIR, f"metrics_{SLURM_JOB_ID}.json"), "w") as f:
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
                print(f"  No improvement for {patience} evals")
                if patience >= EARLY_STOPPING:
                    print(" Early stopping triggered.")
                    break

    final_path = os.path.join(OUTPUT_DIR, f"final_model_{SLURM_JOB_ID}.pth")
    torch.save(model.state_dict(), final_path)
    print("Final model saved.")

if __name__ == "__main__":
    train()