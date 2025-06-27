import time
import json
import os
from typing import List
from voice_detector import detect_synthetic_voice
from text_detector import classify_text
from whisper_asr.main import transcribe_audio

GROOMING_ALERT_THRESHOLD = 0.8
LOG_FILE = "logs/test_combined.log"

# Ensure log folder exists
os.makedirs("logs", exist_ok=True)

def write_log(line: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def analyze_multimodal(audio_segments: List[bytes], model_key: str = "pan12"):
    start_time = time.time()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    synthetic_scores = []
    transcript_parts = []

    # Reset log file
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")

    write_log("")
    write_log(f"Starting combined analysis for {len(audio_segments)} audio segments...")

    for idx, audio_bytes in enumerate(audio_segments):
        write_log(f"Processing segment {idx + 1}/{len(audio_segments)}")

        synthetic_score = detect_synthetic_voice(audio_bytes)
        synthetic_scores.append(synthetic_score)
        write_log(f"Deepfake score (segment {idx + 1}): {synthetic_score:.4f}")

        result = transcribe_audio(audio_bytes)
        transcript = result[0] if isinstance(result, tuple) else result
        transcript_parts.append(transcript)
        write_log(f"Transcript (segment {idx + 1}): {transcript.strip()}")

    full_transcript = " ".join(transcript_parts).strip()
    write_log(f"Combined transcript length: {len(full_transcript)} characters")

    grooming_score = classify_text(full_transcript, model_key=model_key)
    grooming_alert = grooming_score >= GROOMING_ALERT_THRESHOLD
    write_log(f"Grooming score: {grooming_score:.4f} | Alert: {grooming_alert}")

    fake_count = sum(1 for score in synthetic_scores if score > 0.5)
    fake_ratio = fake_count / len(synthetic_scores)

    adjusted_score = grooming_score
    if fake_ratio > 0.5:
        adjusted_score = min(grooming_score + 0.1, 1.0)

    processing_time = round(time.time() - start_time, 2)

    result = {
        "transcript": full_transcript,
        "grooming_score": round(grooming_score, 18),
        "grooming_alert": grooming_alert,
        "voice_deepfake": {
            "count": fake_count,
            "total": len(synthetic_scores),
            "ratio": round(fake_ratio, 4)
        },
        "timestamp": timestamp,
        "processing_time": processing_time,
        "model_key": model_key
    }

    write_log("")
    write_log("--- /analyze-combined response ---")
    write_log("Status Code: 200")
    write_log(json.dumps(result, indent=2, ensure_ascii=False))
    write_log("")

    return result
