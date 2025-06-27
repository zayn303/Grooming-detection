import os
import librosa
import numpy as np
from tqdm import tqdm

# === CONFIG ===
RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
SAMPLE_RATE = 16000
DURATION = 3.0  # seconds
N_MELS = 128
VALID_AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac"]

# Dynamic n_fft based on signal length
def get_n_fft(signal_len):
    if signal_len < 512:
        return 512
    elif signal_len < 1024:
        return 1024
    else:
        return 2048

def extract_mel(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    target_len = int(DURATION * SAMPLE_RATE)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # Get dynamic n_fft value based on signal length
    n_fft = get_n_fft(len(y))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=n_fft)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def save_feature(mel, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, mel)

def process_all():
    print(" Scanning:", RAW_DIR)
    for root, _, files in os.walk(RAW_DIR):
        for fname in tqdm(files, desc=f"[{os.path.relpath(root, RAW_DIR)}]"):
            ext = os.path.splitext(fname)[-1].lower()
            if ext not in VALID_AUDIO_EXTENSIONS:
                continue

            full_path = os.path.join(root, fname)
            try:
                mel = extract_mel(full_path)

                # Compute relative path and mirror it under `processed/`
                rel_path = os.path.relpath(full_path, RAW_DIR)
                new_path = os.path.join(PROCESSED_DIR, os.path.splitext(rel_path)[0] + ".npy")
                save_feature(mel, new_path)

            except Exception as e:
                print(f" Skipping {fname}: {e}")

if __name__ == "__main__":
    process_all()
    print(" Preprocessing completed.")
