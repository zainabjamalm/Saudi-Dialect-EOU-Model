import os
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig


BASE_MODEL = "asafaya/bert-medium-arabic"
ADAPTER_DIR = "saved_model/eou_model"
# -------------------------

class SaudiEOUModel:
    def __init__(self, adapter_dir: str = ADAPTER_DIR, base_model: str = BASE_MODEL, device: str = None):
        """
        Loads tokenizer from adapter_dir and loads base_model then applies the adapter.
        - adapter_dir: folder containing adapter_config + adapter_model.safetensors + tokenizer files
        - base_model: name of the huggingface base model
        """
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.adapter_dir = adapter_dir
        self.base_model_name = base_model

        # 1) Load tokenizer (from adapter folder if available, else from base)
        tokenizer_loaded = False
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
            tokenizer_loaded = True
        except Exception as e:
            print(f"[sdk] Warning: tokenizer not found in adapter dir ({adapter_dir}). Falling back to base model tokenizer.")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        # 2) Load base model (sequence classification)
        print(f"[sdk] Loading base model {self.base_model_name} ...")
        # We create model config with num_labels=2 (binary)
        config = AutoConfig.from_pretrained(self.base_model_name, num_labels=2)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.base_model_name, config=config)

        # 3) Load PEFT adapter on top using PeftModel.from_pretrained
        # This will apply the adapter weights in adapter_dir (safetensors) into the model
        try:
            print(f"[sdk] Applying adapter from {adapter_dir} ...")
            self.model = PeftModel.from_pretrained(self.model, adapter_dir, device_map="auto")
        except Exception as e:
            # if device_map auto fails in CPU-only env, try fallback
            print(f"[sdk] PeftModel.from_pretrained failed with: {e}")
            print("[sdk] Retrying without device_map...")
            self.model = PeftModel.from_pretrained(self.model, adapter_dir)

        # Move to device
        self.model.to(self.device)
        self.model.eval()
        print(f"[sdk] Model loaded and ready on {self.device}.")

    def predict_proba(self, text: str) -> float:
        """
        Returns probability (float) of class 1 = EOU.
        Input: string (transcript)
        """
        # Basic defensive: empty text -> zero probability
        if not text or text.strip() == "":
            return 0.0

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)

        # Forward
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # shape (1, 2)
            probs = torch.softmax(logits, dim=-1)
            proba = float(probs[0, 1].cpu().item())  # prob of label 1 (EOU)
        return proba

    def predict_batch(self, texts):
        """
        texts: list of strings
        returns: list of probabilities
        """
        if not isinstance(texts, (list, tuple)):
            raise ValueError("predict_batch expects a list of strings")
        # naive batching
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist()
        return probs

# quick test when called directly
if __name__ == "__main__":
    m = SaudiEOUModel()
    print(m.predict_proba("طيب خلاص شكراً"))
    print(m.predict_proba("اممم انا لسه بقول"))
