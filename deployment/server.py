import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sdk.eou_model import SaudiEOUModel

app = FastAPI(title="EOU Model Server")

# load once at startup
MODEL_DIR ="saved_model/eou_model"
BASE_MODEL = "asafaya/bert-medium-arabic"
# instantiate model (will load adapter + base)
print("[server] Loading model... (this may take some seconds)")
eou = SaudiEOUModel(adapter_dir=MODEL_DIR, base_model=BASE_MODEL)
print("[server] Model loaded")

class PredictRequest(BaseModel):
    text: str
    participant_id: str = None

class PredictResponse(BaseModel):
    eou_proba: float

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    p = eou.predict_proba(req.text)
    return PredictResponse(eou_proba=p)

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    # run with: python deployment/server.py
    uvicorn.run("deployment.server:app", host="0.0.0.0", port=8000, reload=False)
