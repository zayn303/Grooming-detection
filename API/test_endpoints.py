import requests
import json
import os

LOG_FILE = "logs/api_test.log"
os.makedirs("logs", exist_ok=True)

#Rewrite log file from the start
open(LOG_FILE, "w").close()

def write_log(message):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{message}\n")

BASE_URL = "http://localhost:8000"

# Test: /analyze-text
text_payload = {
    "text": "Hej, vyzeráš mladý, koľko máš vlastne rokov?"
}
try:
    response = requests.post(
        f"{BASE_URL}/analyze-text?model_key=pan12",
        json=text_payload
    )
    write_log("\n--- /analyze-text response ---")
    write_log(json.dumps(response.json(), indent=2, ensure_ascii=False))
except Exception as e:
    write_log(f"Error during /analyze-text: {e}")

# Test: /analyze-audio
AUDIO_PATH = "sample.mp3"
write_log("\n--- /analyze-audio response ---")
try:
    with open(AUDIO_PATH, "rb") as f:
        files = {"file": (AUDIO_PATH, f, "audio/mpeg")}
        response = requests.post(f"{BASE_URL}/analyze-audio?model_key=vtpan", files=files)

        write_log(f"Status Code: {response.status_code}")
        write_log(f"Raw Content: {response.text}")

        write_log(json.dumps(response.json(), indent=2, ensure_ascii=False))
except FileNotFoundError:
    write_log(f"Audio file {AUDIO_PATH} not found.")
except Exception as e:
    write_log(f"Error during /analyze-audio: {e}")