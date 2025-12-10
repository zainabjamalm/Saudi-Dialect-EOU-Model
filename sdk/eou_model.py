# sdk/eou_model.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SaudiEOUModel:
    def __init__(self, model_dir: str = "../model/eou_model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict_eou(self, text: str) -> float:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=1)
        return float(probs[0, 1].cpu().item())
