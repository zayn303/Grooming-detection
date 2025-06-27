import whisper
import os
import tempfile

os.environ["PATH"] = "/home/ak562fx/tools/ffmpeg-7.0.2-i686-static:" + os.environ["PATH"]

model = whisper.load_model("medium")
print(" Whisper model loaded!")

def transcribe_audio(audio_bytes, filename="audio.wav"):
    suffix = os.path.splitext(filename)[1] or ".wav"  # get extension or default .wav
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmpfile:
        tmpfile.write(audio_bytes)
        tmpfile.flush()
        result = model.transcribe(tmpfile.name, language="sk")
    return result["text"]

if __name__ == "__main__":
    # audio_path = "data/Michael speaking Slovak 30 sec.mp3"  
    transcription = transcribe_audio(audio_path)
    print(f" Transcription:\n{transcription}")
