import os
import subprocess
from multiprocessing import Pool

# Path to your ffmpeg binary
ffmpeg_path = "/home/ak562fx/tools/ffmpeg-7.0.2-i686-static/ffmpeg" 

# Function to convert MP3 to WAV
def convert_mp3_to_wav(input_file):
    output_file = input_file.replace(".mp3", ".wav")
    command = [
        ffmpeg_path, "-i", input_file, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_file
    ]
    try:
        subprocess.run(command, check=True, timeout=30)
        print(f"Successfully converted {input_file} to {output_file}")
        os.remove(input_file)  
        print(f"Deleted {input_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_file}: {e}")
    except subprocess.TimeoutExpired:
        print(f"Timeout expired for {input_file}. Skipping...")

def process_directory(base_directory):
    mp3_files = []
    for root, _, files in os.walk(base_directory):
        for file_name in files:
            if file_name.endswith(".mp3"):
                mp3_files.append(os.path.join(root, file_name))

    with Pool(processes=8) as pool:  
        pool.map(convert_mp3_to_wav, mp3_files)

if __name__ == "__main__":
    base_directory = "/home/ak562fx/bac/voice_detection/data/raw/fake_or_real"  
    process_directory(base_directory)
