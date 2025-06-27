import os
import requests

API_URL = "http://localhost:8000/analyze-combined"
DATA_DIR = "data"

def load_audio_files():
    files = []
    print("Loading audio files from:", DATA_DIR)
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".mp3"):
            file_path = os.path.join(DATA_DIR, filename)
            print("Queued:", filename)
            files.append(("files", (filename, open(file_path, "rb"), "audio/mpeg")))
    return files

def test_combined_multimodal():
    print("Starting combined audio test...")
    files = load_audio_files()
    if not files:
        print("No audio files found.")
        return

    response = requests.post(API_URL, files=files)
    print("Status Code:", response.status_code)
    try:
        print("Response JSON:")
        print(response.json())
    except Exception as e:
        print("Error parsing response:", e)
        print(response.text)
    print("Combined audio test completed.")

if __name__ == "__main__":
    test_combined_multimodal()
