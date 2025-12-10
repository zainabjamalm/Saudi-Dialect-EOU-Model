# agent_example/agent.py

from sdk.eou_model import SaudiEOUModel

# Instantiate your EOU model
eou = SaudiEOUModel()

def handle_user_utterance(text: str):
    prob = eou.predict_eou(text)
    print(f"EOU probability: {prob:.3f}")
    if prob > 0.6:  # chosen threshold
        print("User has likely finished — agent can respond now.")
        # Here, trigger agent's reply (e.g. TTS or text)
    else:
        print("Wait for user to continue...")

if __name__ == "__main__":
    # Example usage:
    while True:
        user_input = input("User says: ")
        handle_user_utterance(user_input)
