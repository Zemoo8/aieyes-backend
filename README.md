# AI Eyes — Backend

FastAPI service that powers OCR for the [AI Eyes](https://github.com/Zemoo8/AIEyes) accessibility app. It takes a base64 image and returns the text it finds (Arabic and French), using a Groq-hosted vision model.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Health check, returns `{"status": "ok"}` |
| `POST` | `/ocr` | Body `{"image": "<base64>"}` → returns `{"text": "...", "confidence": 1.0}` |

Errors: `400` for an invalid image, `502` if the Groq API call fails.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
uvicorn main:app --reload --port 8000
```

Quick test:

```bash
curl -X POST http://localhost:8000/ocr \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$(base64 -w0 sample.jpg)\"}"
```

## Docker

```bash
docker build -t aieyes-backend .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key_here aieyes-backend
```

## Deploy

The repo includes `railway.json` for [Railway](https://railway.app): it builds from the Dockerfile and starts uvicorn on `$PORT`. Set `GROQ_API_KEY` in the service variables.

## Stack

Python · FastAPI · Pillow · Groq API · Docker · Railway
