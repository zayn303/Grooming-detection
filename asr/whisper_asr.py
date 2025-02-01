import whisper
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default model to use (can be changed in .env)
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "medium")

def load_whisper_model():
    """Loads the Whisper ASR model (Medium by default)."""
    print(f"🔹 Loading Whisper model: {WHISPER_MODEL_PATH}...")
    model = whisper.load_model(WHISPER_MODEL_PATH)
    return model

if __name__ == "__main__":
    # Test the model loading
    model = load_whisper_model()
    print("✅ Whisper model loaded successfully!")
