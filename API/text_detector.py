import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define known model directories (update if needed)
MODEL_PATHS = {
    "pan12": "/home/ak562fx/bac/grooming_detection/outputs/PAN12/best_model_2085.pth",  
    "vtpan": "/home/ak562fx/bac/grooming_detection/outputs/VTPAN/best_model_2086.pth"
}

MODEL_NAME = "xlm-roberta-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_loaded_models = {}


def load_text_model(model_key: str):
    if model_key in _loaded_models:
        return _loaded_models[model_key]

    if model_key not in MODEL_PATHS:
        raise ValueError(f"Unknown model key: {model_key}")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.load_state_dict(torch.load(MODEL_PATHS[model_key], map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    _loaded_models[model_key] = model
    return model


def classify_text(text: str, model_key: str = "pan12") -> float:
    model = load_text_model(model_key)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        return probs[0][1].item()  # Return probability of class 1 (predator)
