import os
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from src.utils.metrics import compute_metrics
from src.utils.imbalance import compute_weights_from_labels

# Paths & hyperparams
PROCESSED_PATH = "data/processed_processed"  # Use the tokenized dataset
MODEL_NAME = "aubmindlab/bert-base-arabertv2"
OUT_DIR = "saved_models/eou_model"
NUM_LABELS = 2

def load_data():
    print("Loading dataset from:", PROCESSED_PATH)
    ds = load_from_disk(PROCESSED_PATH)
    # Expect ds to have tokenized fields like input_ids, attention_mask, and label
    return ds

class WeightedTrainer(Trainer):
    # Override compute_loss to apply class weights
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        if class_weights is not None:
            self.class_weights_tensor = torch.tensor([class_weights.get(0,1.0), class_weights.get(1,1.0)], device=self.model.device, dtype=torch.float)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights_tensor) if self.class_weights is not None else torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def prepare_model_and_trainer():
    ds = load_data()
    # Extract all labels to compute weights
    all_train_labels = np.array(ds["train"]["labels"])
    class_weights = compute_weights_from_labels(all_train_labels)
    print("Computed class weights:", class_weights)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    
    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=8,  
        lora_alpha=16,  
        target_modules=["query", "value"],  # Modules to apply LoRA to
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",  # Sequence classification task
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Make sure dataset features align with model inputs
    # Dataset should already have input_ids, attention_mask, and labels from preprocessing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Training on:", device)
    
    # set format for pytorch - only include required columns
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
    output_dir=OUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    fp16=True,  
    gradient_checkpointing=False,
    label_names=["labels"],
)
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    return trainer

def main():
    trainer = prepare_model_and_trainer()
    torch.utils.checkpoint.use_reentrant = False
    print("Start Training...")
    trainer.train()
    trainer.save_model("saved_models/eou_model/best")
    print("Training has finished successfully.")
    print("Model saved to saved_models/eou_model/best")

if __name__ == "__main__":
    main()
