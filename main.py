from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import config
from models import transcribe_with_speakers
from summarizer import summarize_text

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

RECORDINGS_DIR = config.RECORDINGS_DIR
RECORDINGS_DIR.mkdir(exist_ok=True)
DOCUMENTS_DIR = config.DOCUMENTS_DIR
DOCUMENTS_DIR.mkdir(exist_ok=True)


ALLOWED_AUDIO_EXTENSIONS = config.ALLOWED_AUDIO_EXTENSIONS
MAX_UPLOAD_MB = config.MAX_UPLOAD_MB


@app.get("/upload-config")
async def upload_config():
    """Upload rules, so the frontend can check a file before sending it."""
    return {"extensions": ALLOWED_AUDIO_EXTENSIONS, "max_mb": MAX_UPLOAD_MB}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    ext = Path(audio.filename or "").suffix.lower() or ".webm"
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            415,
            f"Unsupported file type '{ext}'. Supported: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}",
        )

    data = await audio.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than the {MAX_UPLOAD_MB} MB limit.")

    path = RECORDINGS_DIR / (datetime.now().strftime("%Y%m%d_%H%M%S") + ext)
    path.write_bytes(data)
    print(f"Saved {path} ({len(data)} bytes)")
    try:
        result = transcribe_with_speakers(str(path))
    except RuntimeError as e:
        # whisper.load_audio raises RuntimeError when ffmpeg can't decode the file.
        path.unlink(missing_ok=True)
        print(f"Could not decode {path}: {e}")
        raise HTTPException(
            422, "Could not decode the audio. Is this a valid sound file with an audio track?"
        ) from e
    return {"text": result["text"], "segments": result["segments"]}


class SummarizeRequest(BaseModel):
    transcript: str


@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
    path = DOCUMENTS_DIR / filename
    path.write_text(req.transcript, encoding="utf-8")
    try:
        summary = summarize_text(req.transcript)
    except Exception as e:
        return {"error": str(e)}

    return {"summary": summary}


@app.get("/", response_class=HTMLResponse)
async def main():
    with open("index.html", encoding="utf-8") as f:
        return f.read()
