#  Grooming Detection using Neural Networks and API

This project provides a multimodal system for detecting grooming behavior using both **voice** and **text** inputs. It includes a REST API built in Python, running on a virtual environment and deployable via SLURM for GPU acceleration.

##  Project Structure

```
final/
├── creating_venv.bash              # Script to create a Python virtual environment
├── requirements.txt                # Python dependencies
├── venv.bash                       # Script to activate the environment
├── API/
│   ├── main.py                     # Entry point for REST API
│   ├── check_val.py                # Validation helper
│   ├── combined_analysis.py        # Core logic for voice+text inference
│   ├── sample.mp3                  # Example audio input
│   ├── run_api.slurm               # SLURM script for API deployment
│   ├── test_combined_analysis.py   # Unit test for analysis logic
│   └── test_endpoints.py           # API endpoint tests
```

##  Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/zayn303/grooming-detection-api.git
cd grooming-detection-api/final
```

### 2. Set Up Virtual Environment

```bash
bash creating_venv.bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the API Locally

```bash
cd API
python3 main.py
```

The API will start on `http://127.0.0.1:5000/`.

### 5. Run Tests

```bash
pytest test_endpoints.py
pytest test_combined_analysis.py
```

##  Sample Usage

Use the provided `sample.mp3` file to test the grooming detection endpoint once the API is running.

```
POST /analyze
Content-Type: multipart/form-data
Payload:
- audio: sample.mp3
- transcript: "example conversation text"
```

##  SLURM Deployment

For running on a SLURM-managed GPU cluster:

```bash
sbatch run_api.slurm
```

Make sure the cluster has access to the needed Python environment and GPU.

##  Dependencies

Key libraries include:
- `Flask` – for REST API
- `torch`, `transformers` – for running models
- `librosa`, `torchaudio` – for audio preprocessing
- `pytest` – for testing

Full list in `requirements.txt`.

##  Authors

- Andrii Kolomiiets  
Bachelor's thesis, Technical University of Košice

##  License

This project is licensed under the MIT License.
