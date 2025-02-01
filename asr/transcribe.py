import whisper
from whisper_asr import load_whisper_model 

# Load Whisper model
model = load_whisper_model()

def transcribe_audio(audio_file):
    """Transcribes Slovak speech from an audio file using Whisper."""
    print(f"🎙️ Transcribing: {audio_file}")
    result = model.transcribe(audio_file, language="sk")
    return result["text"]

if __name__ == "__main__":
    # Example usage
    audio_path = "data/raw/Michael speaking Slovak 30 sec.mp3"
    transcription = transcribe_audio(audio_path)
    print(f"📝 Transcription: {transcription}")
