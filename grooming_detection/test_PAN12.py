import os
import sys
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from scripts.utils.dataset import GroomingDetectionDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_CSV = "/home/ak562fx/bac/grooming_detection/translated_data/PAN12/csv/PAN12-test-exp-sk.csv"
MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 512
BATCH_SIZE = 16
OUTPUT_DIR = "outputs/PAN12"

def get_best_model_path():
    files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("best_model_") and f.endswith(".pth")]
    if not files:
        raise FileNotFoundError("No best_model_*.pth found.")
    files.sort(key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)), reverse=True)
    return os.path.join(OUTPUT_DIR, files[0])


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids, mask, lab = [x.to(DEVICE) for x in batch]
            output = model(ids, attention_mask=mask)
            pred = torch.argmax(output.logits, dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(lab.cpu().numpy())
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    print("\n--- Test Report ---")
    print(classification_report(labels, preds, target_names=['non-predator', 'predator']))
    print("Confusion Matrix:\n", confusion_matrix(labels, preds))
    print(f"Accuracy={acc:.4f}, Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}")

def main():
    print("Evaluating model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    df = pd.read_csv(TEST_CSV)
    df['label'] = df['label'].str.strip().str.lower()
    df['label_id'] = df['label'].map({'non-predator': 0, 'predator': 1})
    dataset = GroomingDetectionDataset(df, tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    model_path = get_best_model_path()
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    evaluate(model, loader)

if __name__ == "__main__":
    main()