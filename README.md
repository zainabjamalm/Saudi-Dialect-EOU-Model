# Saudi Dialect EOU Model

## Project Overview

This project aims to fine-tune transformer-based models to classify Saudi dialects in Arabic text. The dataset is curated to focus on specific dialects, and the training pipeline incorporates techniques like LoRA (Low-Rank Adaptation) to optimize training efficiency.

## Models and Experiments

### 1. AraBERT

- **Model**: `aubmindlab/bert-base-arabertv2`
- **Training Time**: Approximately 25 minutes per epoch
- **Results** (after 1 epoch):

   ```text
   {
      'eval_loss': 0.16674670577049255,
      'eval_accuracy': 0.9180535455861071,
      'eval_precision': 0.9992522432701895,
      'eval_recall': 0.8990805113254093,
      'eval_f1': 0.946523432888679,
      'eval_roc_auc': 0.9737775063561132,
      'eval_runtime': 13.553,
      'eval_samples_per_second': 407.88,
      'eval_steps_per_second': 50.985,
      'epoch': 1.0
  }
  ```

### 2. DistilBERT

- **Model**: `bert-medium-arabic`
- **Training Time**: TBD
- **Results**: TBD

## Key Changes and Optimizations

### 1. LoRA Fine-Tuning

- **Why**: LoRA reduces the number of trainable parameters, making training faster and more memory-efficient.
- **How**: Applied LoRA to the attention layers of the model using the `peft` library.

### 2. Dataset Preprocessing

- **Steps**:
  1. Filtered the dataset to include only Saudi dialects.
  2. Tokenized the text using the model tokenizer.
  3. Saved the processed dataset for efficient loading.
- **Why**: Ensures the dataset is clean, consistent, and ready for training.

### 3. Weighted Loss

- **Why**: To handle class imbalance in the dataset.
- **How**: Computed class weights and applied them to the loss function during training.

### 4. Compatibility Fixes

- Updated the `WeightedTrainer` class to handle additional arguments passed by newer versions of the `transformers` library.
- Ensured the dataset columns align with the model's expected input format.

## How to Run

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Preprocess the Dataset**:

   ```bash
   python -m src.data.preprocess
   ```

3. **Train the Model**:

   ```bash
   python main.py
   ```

4. **Evaluate the Model**:

   ```bash
   python -m src.model.evaluate
   ```

## Future Work

- Experiment with DistilBERT to compare training time and performance.
- Explore other transformer architectures for further optimization.
- Fine-tune hyperparameters to improve results.

---

**Note**: Update this README with DistilBERT results once the experiment is complete.
