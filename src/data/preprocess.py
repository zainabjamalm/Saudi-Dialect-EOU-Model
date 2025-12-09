from transformers import AutoTokenizer
from datasets import load_from_disk
import os

MODEL_NAME = "aubmindlab/bert-base-arabertv2" 
PROCESSED_PATH = "data/processed"

def get_text(example):
    text = example.get("text_with_pause") or example.get("text") or example.get("raw_text") or ""
    speaker = example.get("speaker_id")
    if speaker:
        return f"{speaker}: {text}"
    return text

def preprocess_and_tokenize(save_path=PROCESSED_PATH, model_name=MODEL_NAME, max_length=128):
    print("Loading processed dataset from disk:", save_path)
    ds = load_from_disk(save_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare(example):
        text = get_text(example)
        return {"text_for_model": text}

    print("Mapping text_for_model")
    ds = ds.map(prepare)

    print("Tokenizing dataset")
    def tokenize_fn(batch):
        return tokenizer(batch["text_for_model"], truncation=True, padding="max_length", max_length=max_length)

    ds = ds.map(tokenize_fn, batched=True, remove_columns=[c for c in ds["train"].column_names if c not in ["label", "text_for_model"]])
    # Keep label and input_ids, attention_mask
    os.makedirs(save_path, exist_ok=True)
    ds.save_to_disk(save_path)  # overwrite processed tokenized dataset
    print("Tokenized dataset saved to:", save_path)
    return ds

if __name__ == "__main__":
    preprocess_and_tokenize()