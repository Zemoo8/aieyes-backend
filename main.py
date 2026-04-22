import base64
import io
import logging

import easyocr
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aieyes")

app = FastAPI(title="AIEyes OCR Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load once at startup — EasyOCR downloads models on first run (~500 MB)
logger.info("Loading EasyOCR models (ar, fr)…")
reader = easyocr.Reader(["ar", "en"], gpu=False)
logger.info("EasyOCR ready.")


class OCRRequest(BaseModel):
    image: str  # base64-encoded JPEG/PNG


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr")
def ocr(req: OCRRequest):
    try:
        image_bytes = base64.b64decode(req.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(pil_img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {e}")

    results = reader.readtext(img_array)

    if not results:
        return {"text": "", "confidence": 0.0}

    # Merge all detected text blocks; weight confidence by text length
    total_chars = sum(len(r[1]) for r in results)
    if total_chars == 0:
        return {"text": "", "confidence": 0.0}

    merged_text = " ".join(r[1] for r in results)
    avg_confidence = sum(r[2] * len(r[1]) for r in results) / total_chars

    logger.info("OCR result: %d block(s), conf=%.2f", len(results), avg_confidence)
    return {"text": merged_text, "confidence": round(avg_confidence, 4)}
