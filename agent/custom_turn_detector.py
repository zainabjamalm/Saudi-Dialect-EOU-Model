# agent/custom_turn_detector.py
import time
from sdk.eou_model import SaudiEOUModel

class CustomTurnDetector:
    def __init__(self, threshold=0.50):
        self.model = SaudiEOUModel()
        self.threshold = threshold

    def detect(self, transcript_chunk):
        """Return True if end of utterance detected."""
        if not transcript_chunk or transcript_chunk.strip() == "":
            return False

        prob = self.model.predict_proba(transcript_chunk)
        print(f"[turn-detector] text='{transcript_chunk}' prob={prob:.3f}")
        return prob >= self.threshold
