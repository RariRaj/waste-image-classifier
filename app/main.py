"""Phase 1: a running API; model integration comes after training."""
from fastapi import FastAPI

app = FastAPI(title="Waste Image Classifier", version="0.1.0")

@app.get("/")
def root():
    return {"message": "Waste Image Classifier API", "docs": "/docs"}

@app.get("/health")
def health():
    """API liveness only. No trained model is loaded in this phase."""
    return {"status": "ok", "model_ready": False, "stage": "project_setup"}
