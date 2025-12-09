import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List

MODEL_DIR = "saved_models/eou_model/best"

class EOUModel:
    def __init__(self, model_dir=MODEL_DIR):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, texts: List[str]):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
            logits = out.logits.detach().cpu().numpy()
        # softmax and probabilities for class 1
        import numpy as np
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exps / exps.sum(axis=1, keepdims=True)
        return probs[:,1].tolist()

    def predict(self, text):
        return self.predict_proba([text])[0]

if __name__ == "__main__":
    m = EOUModel()
    print(m.predict("A: طيب بس خليني أجي بعدين"))
