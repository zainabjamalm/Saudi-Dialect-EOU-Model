# Saudi Dialect EOU Model

## Dataset

### Overview

The dataset used for this project focuses on Saudi dialects in Arabic text. It has been curated to ensure high-quality data for training and evaluation.

### Preprocessing

- Filtered to include only Saudi dialects.
- Tokenized using the model tokenizer.
- Saved in an efficient format for training.

### Class Imbalance Handling

- Computed class weights to address imbalance.
- Applied weighted loss during training.

## Model

### Base Models

1. **AraBERT**: `aubmindlab/bert-base-arabertv2`
   - Training Time: ~25 minutes per epoch
   - Accuracy: 91.8%

2. **BERT Medium**: `asafaya/bert-medium-arabic`
   - Training Time: ~14 minutes per epoch
   - Accuracy: 91.8%

### LoRA Fine-Tuning

- Reduced trainable parameters for efficiency.
- Applied to attention layers using the `peft` library.

## Fine-Tuning

### Training Details

- Optimized hyperparameters for performance.
- Used weighted loss to handle class imbalance.
- Updated `WeightedTrainer` for compatibility with `transformers`.

### Results

- **AraBERT**: F1 Score: 94.7%
- **BERT Medium**: F1 Score: 94.7% but faster

## Deployment

### SDK Features

- Easy integration with LiveKit.
- Real-time EOU detection.
- Configurable thresholds and model variants.

### Quick Start

```python
from sdk import SaudiEOUModel

model = SaudiEOUModel(model_name="asafaya/bert-medium-arabic")
probability = model.predict_eou("السلام عليكم ورحمة الله")
```

### LiveKit Integration

```python
from sdk.livekit_integration import LiveKitTurnDetectionModule, EOUConfig

config = EOUConfig(model_key="bert-medium", eou_threshold=0.6)
detector = LiveKitTurnDetectionModule(config=config)
result = detector.process_text("أنا جاهز")
```

### Examples

- `examples/livekit_example.py`: LiveKit integration.
- `examples/evaluation.py`: Model evaluation.
- `agent/agent.py`: Interactive demo.

### Testing

Run the evaluation suite:

```bash
python examples/evaluation.py
```

---

For more details, refer to the documentation files in the repository.
