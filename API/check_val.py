import requests
import json
import os

LOG_FILE = "logs/api_check.log"
os.makedirs("logs", exist_ok=True)

# Rewrite log file from the start
open(LOG_FILE, "w").close()

def write_log(message):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{message}\n")

BASE_URL = "http://localhost:8000"

def run_check_val():
    examples = [
        "Hej, ako sa máš, chceš sa stretnúť dnes večer?",
        "Si krásna, máš veľmi pekný úsmev. Ako stará si?",
        "Ahoj, ako sa ti darí v škole?",
        "Martin Kukučín bol slovenský spisovateľ realizmu.",
        "Chcem ti povedať tajomstvo, ale nikomu to nehovor, dobre?",
        "Ukážeš mi niečo, ale len my dvaja to budeme vedieť."
    ]

    for model_key in ["pan12", "vtpan"]:
        write_log(f"\n--- Model: {model_key.upper()} ---")
        for text in examples:
            try:
                response = requests.post(
                    f"{BASE_URL}/analyze-text?model_key={model_key}",
                    json={"text": text}
                )
                if response.status_code == 200:
                    data = response.json()
                    write_log(f"Text: {text}\n → Score: {data['grooming_score']:.4f} | Alert: {data['grooming_alert']}")
                else:
                    write_log(f"Text: {text}\n → ERROR: {response.status_code} {response.text}")
            except Exception as e:
                write_log(f"Text: {text}\n → EXCEPTION: {e}")

if __name__ == "__main__":
    run_check_val()
