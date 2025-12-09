import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
from utils.metrics import compute_metrics

MODEL_DIR = "saved_models/eou_model/best"
PROCESSED_PATH = "data/processed"
MODEL_NAME = MODEL_DIR  # load from saved dir

def evaluate():
    print("Loading dataset from:", PROCESSED_PATH)
    ds = load_from_disk(PROCESSED_PATH)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    print("Loading model from:", MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Use HF trainer's predict? For simplicity: batch loop
    dataloader = torch.utils.data.DataLoader(ds["test"], batch_size=32)
    preds = []
    labels = []
    for batch in dataloader:
        inputs = {k: v.to(device) for k,v in batch.items() if k in ["input_ids","attention_mask"]}
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits.detach().cpu().numpy()
        preds.append(logits)
        labels.append(batch["labels"].numpy())

    import numpy as np
    preds = np.vstack(preds)
    labels = np.concatenate(labels)
    from utils.metrics import softmax
    probs = softmax(preds, axis=1)[:,1]
    pred_labels = preds.argmax(axis=1)
    class_result = {"predictions": preds, "label_ids": labels}
    metrics = compute_metrics(type("P", (), {"predictions": preds, "label_ids": labels}))
    print("Evaluation metrics:", metrics)

if __name__ == "__main__":
    evaluate()
