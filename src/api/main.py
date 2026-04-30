"""
Brain Tumor Detection — REST API.

Serves the trained model over HTTP so the .NET backend (or any HTTP
client) can run predictions without needing a Python runtime.

Endpoints:
    GET  /health   — service health check
    GET  /         — basic service info
    POST /predict  — multipart/form-data upload of an MRI image, returns
                     predicted class, confidence, and per-class probabilities

Run:
    python scripts/run_api.py
    # or
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src.inference.predictor import BrainTumorPredictor


MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/models/best_model.pth")
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/bmp", "image/tiff"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB upload cap


# Loaded once at startup, reused for every request.
state: dict = {"predictor": None}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state["predictor"] = BrainTumorPredictor(MODEL_PATH)
    yield
    state["predictor"] = None


app = FastAPI(
    title="Brain Tumor Detection API",
    description="Classifies brain MRI images into glioma, meningioma, notumor, or pituitary.",
    version="1.0.0",
    lifespan=lifespan,
)

# Permissive CORS so the .NET dev environment can call this from anywhere
# during integration. Tighten allow_origins for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/", tags=["meta"])
def root():
    return {
        "service": "Brain Tumor Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {"health": "GET /health", "predict": "POST /predict"},
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(status="ok", model_loaded=state["predictor"] is not None)


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(file: UploadFile = File(..., description="Brain MRI image (JPEG/PNG)")):
    predictor = state["predictor"]
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type '{file.content_type}'. "
                   f"Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit",
        )

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    result = predictor.predict(image)
    return PredictionResponse(**result)
