import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Access environment variables
DEEP_L_API_KEY = os.getenv("DEEP_L_API_KEY")
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH")

print(f"Whisper Model Path: {WHISPER_MODEL_PATH}")
