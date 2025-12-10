import asyncio
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 
from sdk.eou_model import SaudiEOUModel
from agent.custom_turn_detector import CustomTurnDetector
from livekit.agents import WorkerOptions, AutoSubscribe, JobContext
EOU_BASE_MODEL = "asafaya/bert-medium-arabic"
ADAPTER_DIR = "saved_model/eou_model"
class LiveKitEOUAgent:
    def __init__(self):
        print("[agent] Loading EOU model...")
        self.eou = SaudiEOUModel(adapter_dir=ADAPTER_DIR, base_model=EOU_BASE_MODEL)
        self.detector = CustomTurnDetector(self.eou)
        print("[agent] EOU Ready!")

    async def handle_room(self, ctx: JobContext):
        room = ctx.room
        print("[agent] Ready. Listening for transcription...")

        async for event in room.on_transcription():
            text = event.text
            speaker = event.participant.identity if event.participant else "unknown"
            silence_sec = getattr(event, "silence_after_seconds", None)

            decision, combined, model_p, thresh = self.detector.detect(
                text=text,
                speaker_id=speaker,
                silence_after_seconds=silence_sec
            )

            print(f"[EOU] '{text}'  => EOU={decision}  P={model_p:.3f}  Combined={combined:.3f}  Thresh={thresh:.3f}")

            if decision:
                await room.send_chat_message("تم الانتهاء من كلامك وسأقوم بالرد الآن.")

async def main():
    agent = LiveKitEOUAgent()
    opts = WorkerOptions(
        auto_subscribe=AutoSubscribe.AUDIO,
        turn_detection=False  # Disable default detector
    )

    await opts.run(agent.handle_room)


if __name__ == "__main__":
    asyncio.run(main())
