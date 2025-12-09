from datasets import load_dataset, DatasetDict
import os

SAUDI_DIALECTS = ["Najdi", "Hijazi", "Janubi", "Shamali"]  # primary Saudi list
# SAUDI_DIALECTS = ["Najdi","Hijazi","Janubi","Shamali","Khaliji"]

def load_and_filter(dataset_name: str, saudi_dialects=None, save_path: str = "data/processed"):
    saudi_dialects = saudi_dialects or SAUDI_DIALECTS

    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)  # expects the HF dataset id or local path
    print("Dataset splits:", ds.keys())

    def is_saudi(example):
        d = example.get("dialect", None)
        # dialect might be None or differently-cased; guard against that
        return d in saudi_dialects

    print("Filtering to Saudi dialects:", saudi_dialects)
    ds_filtered = DatasetDict()
    for split in ds.keys():
        print(f"Filtering split: {split}")
        ds_filtered[split] = ds[split].filter(is_saudi)

    # Save to disk so subsequent steps can load quickly
    os.makedirs(save_path, exist_ok=True)
    ds_filtered.save_to_disk(save_path)
    print(f"Saved processed dataset to: {save_path}")
    return ds_filtered

if __name__ == "__main__":
    ds = load_and_filter("nihad-ask/arabic_eou_sada_curated", save_path="data/processed")
