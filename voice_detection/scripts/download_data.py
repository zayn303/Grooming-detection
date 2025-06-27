import os
import zipfile
import subprocess

# Correct absolute path handling
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

DATASETS = {
    "scene_fake": "mohammedabdeldayem/scenefake",
    "fake_or_real": "mohammedabdeldayem/the-fake-or-real-dataset"
}

def download_and_extract(dataset_name, kaggle_id):
    print(f"Downloading {dataset_name}...")
    dataset_dir = os.path.join(RAW_DIR, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Download via Kaggle API
    subprocess.run([
        "kaggle", "datasets", "download", "-d", kaggle_id, "-p", dataset_dir
    ], check=True)

    # Find the downloaded zip
    zip_files = [f for f in os.listdir(dataset_dir) if f.endswith(".zip")]
    for zip_file in zip_files:
        zip_path = os.path.join(dataset_dir, zip_file)
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        os.remove(zip_path)

    print(f" {dataset_name} ready at {dataset_dir}")

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    for name, kaggle_id in DATASETS.items():
        download_and_extract(name, kaggle_id)

if __name__ == "__main__":
    main()
