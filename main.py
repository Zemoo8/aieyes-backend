import base64
import io
import os
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aieyes")

app = FastAPI(title="AIEyes OCR Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# llava-v1.5-7b-4096-preview was retired by Groq (Oct 2024).
# Override with the GROQ_VISION_MODEL env var if Groq changes models again.
VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")


class OCRRequest(BaseModel):
    image: str  # base64-encoded image


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr")
def ocr(req: OCRRequest):
    # Decode and re-encode as clean JPEG base64
    try:
        image_bytes = base64.b64decode(req.image)
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=90)
        jpeg_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{jpeg_b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": "Read every piece of text visible in this image. Arabic and French. Return only the raw text, no explanation.",
                        },
                    ],
                }
            ],
            max_tokens=1024,
            # Qwen3 is a reasoning model; turn thinking off so only the text comes back
            extra_body={"reasoning_effort": "none"},
        )
        text = response.choices[0].message.content or ""
        logger.info("OCR success: %d chars returned", len(text))
        return {"text": text, "confidence": 1.0}
    except Exception as e:
        logger.error("Groq API error: %s", e)
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")
