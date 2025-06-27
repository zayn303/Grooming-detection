import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_GRU(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, hidden_size=64, num_layers=1, bidirectional=True):
        super(CNN_GRU, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.gru = nn.GRU(
            input_size=128 * 11,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_classes)

    def forward(self, x):
        x = self.conv(x)  # (B,128,16,11)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # (B,16,128,11)
        x = x.view(B, H, C * W)  # (B,16,128*11)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)
