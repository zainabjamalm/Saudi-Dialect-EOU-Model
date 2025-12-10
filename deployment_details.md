# Deployment Details

## SDK Features

- **Easy integration with LiveKit**: The SDK provides a seamless way to integrate the Saudi Dialect EOU model with LiveKit for real-time applications.
- **Real-time EOU detection**: The model is optimized for detecting the end of utterances in real-time conversations.
- **Configurable thresholds and model variants**: Users can adjust thresholds and choose from multiple model variants to suit their use case.

## Quick Start

### Using the SDK

The SDK simplifies the process of using the Saudi Dialect EOU model. Below is an example of how to use it:

```python
from sdk import SaudiEOUModel

# Initialize the model with the desired configuration
model = SaudiEOUModel(model_name="asafaya/bert-medium-arabic")

# Predict the probability of an end-of-utterance (EOU) in the given text
probability = model.predict_eou("السلام عليكم ورحمة الله")
```

#### Explanation:

1. **`SaudiEOUModel`**: This is the main class for interacting with the model. It handles initialization, loading the model, and making predictions.
2. **`predict_eou`**: This function takes a text input and returns the probability that the text marks the end of an utterance.

### LiveKit Integration

The SDK includes a module for integrating with LiveKit, enabling real-time turn detection in conversations.

```python
from sdk.livekit_integration import LiveKitTurnDetectionModule, EOUConfig

# Configure the LiveKit integration
config = EOUConfig(model_key="bert-medium", eou_threshold=0.6)

# Initialize the LiveKit turn detection module

detector = LiveKitTurnDetectionModule(config=config)

# Process a text input to determine if it triggers an agent response
result = detector.process_text("أنا جاهز")
```

#### Explanation:

1. **`EOUConfig`**: This class is used to configure the LiveKit integration, including the model key and the threshold for EOU detection.
2. **`LiveKitTurnDetectionModule`**: This module handles the integration with LiveKit. It processes text inputs and determines whether the agent should respond based on the EOU probability.
3. **`process_text`**: This function takes a text input and returns a result indicating whether the agent should respond.

## Examples

The repository includes several examples to demonstrate the usage of the SDK:

- **`examples/livekit_example.py`**: Shows how to integrate the SDK with LiveKit for real-time turn detection.
- **`examples/evaluation.py`**: Provides an evaluation suite to test the model's performance on various test cases.
- **`agent/agent.py`**: An interactive demo application for testing EOU detection in real-time conversations.

## Testing

To ensure the model and SDK are working as expected, run the evaluation suite:

```bash
python examples/evaluation.py
```

#### Explanation:

1. **Evaluation Suite**: This script evaluates the model's performance on a set of predefined test cases.
2. **Metrics**: The evaluation suite calculates metrics such as accuracy, precision, recall, and F1 score to assess the model's effectiveness.

