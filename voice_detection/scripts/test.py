import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from scripts.utils.dataset import VoiceDeepfakeDataset
from models.cnn_lstm import CNN_LSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
OUTPUT_DIR = "outputs/"

def get_best_model_path():
    files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("best_model_") and f.endswith(".pth")]
    if not files:
        raise FileNotFoundError("No best_model_*.pth found.")
    files.sort(key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)), reverse=True)
    return os.path.join(OUTPUT_DIR, files[0])

def test():
    print("Loading test set...")
    test_ds = VoiceDeepfakeDataset("data/metadata/metadata_cross_balanced.csv", "data/processed", split="test", augment=False, verbose=True)
    if len(test_ds) == 0:
        print("No samples found in test set.")
        return

    dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    model = CNN_LSTM(in_channels=1, num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(get_best_model_path(), map_location=DEVICE))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_logits = [], [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = model(X)
            loss = criterion(outputs, y)
            preds = outputs.argmax(1)

            total_loss += loss.item() * X.size(0)
            correct += (preds == y).sum().item()
            total += y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_logits.extend(outputs.softmax(dim=1)[:, 1].cpu().numpy())

    acc = correct / total
    avg_loss = total_loss / total

    print(f"\n Final Test Accuracy: {acc:.4f}")
    print(f" Final Test Loss: {avg_loss:.4f}")
    print(f" F1 Score: {f1_score(all_labels, all_preds):.4f}")
    print(f" ROC-AUC: {roc_auc_score(all_labels, all_logits):.4f}")
    print("\n Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\n Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["fake", "real"]))

if __name__ == "__main__":
    test()
