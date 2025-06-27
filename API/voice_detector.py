import torch
import torchaudio
import io
import numpy as np
from voice_detection.models.cnn_lstm import CNN_LSTM

# Load model once
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/home/ak562fx/bac/voice_detection/outputs/best_model_2121.pth"

model = CNN_LSTM(in_channels=1, num_classes=2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def preprocess_audio(audio_bytes: bytes) -> torch.Tensor:
    waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))

    # Force mono (keep 1st channel only)
    if waveform.shape[0] > 1:
        waveform = waveform[:1, :]  # drop all but first channel

    # Resample to 16kHz if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    # ... (rest of code unchanged)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=128
    )(waveform)

    mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
    mel_db = mel_db / mel_db.abs().max()

    if mel_db.shape[-1] < 94:
        pad = 94 - mel_db.shape[-1]
        mel_db = torch.nn.functional.pad(mel_db, (0, pad))
    else:
        mel_db = mel_db[:, :, :94]

    return mel_db.unsqueeze(0).to(DEVICE)  # shape: [1, 1, 128, 94]


def detect_synthetic_voice(audio_bytes: bytes) -> float:
    input_tensor = preprocess_audio(audio_bytes)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        return probs[0][0].item()  # Class 0 = fake, 1 = real — return prob(fake)
