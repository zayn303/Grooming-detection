import os
import sys
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.cnn_lstm import CNN_LSTM
from scripts.utils.dataset import VoiceDeepfakeDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(split="val"):
    dataset = VoiceDeepfakeDataset("data/metadata/metadata_cross_balanced.csv", "data/processed", split=split)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    model = CNN_LSTM(in_channels=1, num_classes=2).to(DEVICE)
    model_path = "outputs/final_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print("No final model found.")
        return

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = model(X)
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)

    print(f"[{split.upper()} Accuracy] {correct / total:.4f} on {total} samples")

if __name__ == "__main__":
    evaluate(split="val")
