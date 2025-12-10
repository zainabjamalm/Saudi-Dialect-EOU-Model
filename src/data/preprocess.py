from transformers import AutoTokenizer
from datasets import load_from_disk
import os
from src.data.load_data import load_and_filter

MODEL_NAME = "asafaya/bert-medium-arabic" 
PROCESSED_PATH = "data/processed"

def get_text(example):
    text = example.get("text_with_pause") or example.get("text") or example.get("raw_text") or ""
    speaker = example.get("speaker_id")
    if speaker:
        return f"{speaker}: {text}"
    return text

def preprocess_and_tokenize(save_path=PROCESSED_PATH, model_name=MODEL_NAME, max_length=128, force_retokenize=False):
    # Check if processed data exists, if not load and filter it first
    if not os.path.exists(save_path):
        print(f"Directory {save_path} not found. Loading and filtering data first...")
        load_and_filter("nihad-ask/arabic_eou_sada_curated", save_path=save_path)
    
    # Generate a model-specific output path
    model_hash = model_name.replace("/", "_")
    save_path_processed = f"{save_path}_processed_{model_hash}"
    
    # If tokenized data already exists for this model and not forcing retokenization, load it
    if os.path.exists(save_path_processed) and not force_retokenize:
        print(f"Loading pre-tokenized dataset for {model_name} from:", save_path_processed)
        return load_from_disk(save_path_processed)
    
    print("Loading processed dataset from disk:", save_path)
    ds = load_from_disk(save_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare(example):
        text = get_text(example)
        return {"text_for_model": text}

    print("Mapping text_for_model")
    ds = ds.map(prepare)

    print(f"Tokenizing dataset with {model_name}")
    def tokenize_fn(batch):
        return tokenizer(batch["text_for_model"], truncation=True, padding="max_length", max_length=max_length)

    ds = ds.map(tokenize_fn, batched=True, remove_columns=[c for c in ds["train"].column_names if c not in ["label", "text_for_model"]])
    # Remove text_for_model and keep only tokenized features
    ds = ds.remove_columns(["text_for_model"])
    # Rename label to labels for model compatibility
    ds = ds.rename_column("label", "labels")
    # Keep labels, input_ids, attention_mask
    os.makedirs(save_path_processed, exist_ok=True)
    ds.save_to_disk(save_path_processed)  
    print(f"Tokenized dataset for {model_name} saved to:", save_path_processed)
    return ds

if __name__ == "__main__":
    preprocess_and_tokenize()