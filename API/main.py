from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import time

from text_detector import classify_text
from voice_detector import detect_synthetic_voice
from whisper_asr.main import transcribe_audio
from combined_analysis import analyze_multimodal

app = FastAPI()

GROOMING_ALERT_THRESHOLD = 0.8

# ------------------------------
# Models for response schemas
# ------------------------------
class TextAnalysisRequest(BaseModel):
    text: str

class TextAnalysisResponse(BaseModel):
    grooming_score: float
    grooming_alert: bool
    model_key: str
    timestamp: str

class AudioAnalysisResponse(BaseModel):
    is_synthetic_voice: float
    transcript: str
    grooming_score: float
    grooming_alert: bool
    model_key: str
    processing_time: dict
    timestamp: str
    warnings: List[str] = []
    confidence: dict | None = None

# ------------------------------
# /analyze-text
# ------------------------------
@app.post("/analyze-text", response_model=TextAnalysisResponse)
def analyze_text(payload: TextAnalysisRequest, model_key: str = Query("pan12")):
    grooming_score = max(0.0, min(1.0, classify_text(payload.text, model_key=model_key)))
    grooming_alert = grooming_score >= GROOMING_ALERT_THRESHOLD
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    return TextAnalysisResponse(
        grooming_score=grooming_score,
        grooming_alert=grooming_alert,
        model_key=model_key,
        timestamp=timestamp
    )

# ------------------------------
# /analyze-audio
# ------------------------------
@app.post("/analyze-audio", response_model=AudioAnalysisResponse)
async def analyze_audio(file: UploadFile = File(...), model_key: str = Query("pan12")):
    start_time = time.time()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    audio_bytes = await file.read()

    # Voice deepfake detection
    synthetic_score = detect_synthetic_voice(audio_bytes)

    # ASR transcription
    transcript = transcribe_audio(audio_bytes)
    if isinstance(transcript, tuple):  # in case whisper returns (text, segments)
        transcript = transcript[0]

    # Grooming score
    grooming_score = classify_text(transcript, model_key=model_key)
    grooming_score = max(0.0, min(1.0, grooming_score))
    grooming_alert = grooming_score >= GROOMING_ALERT_THRESHOLD

    response = {
        "is_synthetic_voice": round(synthetic_score, 4),
        "transcript": transcript,
        "grooming_score": round(grooming_score, 4),
        "grooming_alert": grooming_alert,
        "model_key": model_key,
        "timestamp": timestamp,
        "processing_time": {"total": round(time.time() - start_time, 2)},
        "warnings": [],
    }

    confidence = None  # Optional field
    if confidence is not None:
        response["confidence"] = confidence

    return JSONResponse(content=response)

# ------------------------------
# /analyze-combined (multi-audio)
# ------------------------------
@app.post("/analyze-combined")
async def analyze_combined(files: List[UploadFile] = File(...), model_key: str = Query("pan12")):
    try:
        audio_segments = [await f.read() for f in files]
        result = analyze_multimodal(audio_segments, model_key=model_key)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
