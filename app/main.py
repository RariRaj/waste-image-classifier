"""FastAPI application for waste image classification."""

from pathlib import Path
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile

INDEX_PATH = Path(__file__).resolve().parent / "static" / "index.html"
from app.inference import (
    MAX_FILE_BYTES,
    InvalidImageError,
    WastePredictor,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once per server process, not once per request.
    app.state.predictor = WastePredictor()
    yield
    app.state.predictor = None


app = FastAPI(
    title="Waste Image Classifier",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/", response_class=FileResponse)
def root():
    return FileResponse(INDEX_PATH, media_type="text/html")


@app.get("/health")
def health(request: Request):
    ready = getattr(request.app.state, "predictor", None) is not None

    return {
        "status": "ok" if ready else "not_ready",
        "model_ready": ready,
        "stage": "model_integration",
    }


@app.post(
    "/predict",
    responses={
        400: {"description": "Empty, invalid, or unsupported image"},
        413: {"description": "Uploaded file exceeds 5 MiB"},
        503: {"description": "Model is not ready"},
    },
)
def predict(
    request: Request,
    file: UploadFile = File(...),
):
    predictor = getattr(request.app.state, "predictor", None)

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready.",
        )

    try:
        # Read at most the limit plus one byte to detect oversize files.
        image_bytes = file.file.read(MAX_FILE_BYTES + 1)
    finally:
        file.file.close()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(image_bytes) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Maximum supported file size is 5 MiB.",
        )

    try:
        return predictor.predict(image_bytes)
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
