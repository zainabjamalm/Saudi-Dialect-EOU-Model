
from src.data.load_data import load_and_filter
from src.data.preprocess import preprocess_and_tokenize
from src.model.train import main as train_main

if __name__ == "__main__":
    # 1) load and filter dataset (only do once)
    # Replace dataset id with your HF dataset name if calling directly
    load_and_filter("nihad-ask/arabic_eou_sada_curated", save_path="data/processed")

    # 2) preprocess & tokenize
    preprocess_and_tokenize(save_path="data/processed", model_name="aubmindlab/bert-base-arabertv2", max_length=128)

    # 3) run training
    train_main()
