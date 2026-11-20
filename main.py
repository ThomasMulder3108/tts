from datetime import datetime

from fastapi import FastAPI, File, UploadFile
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


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".webm"
    path = RECORDINGS_DIR / filename
    path.write_bytes(await audio.read())
    print(f"Saved {path} ({path.stat().st_size} bytes)")
    result = transcribe_with_speakers(str(path))
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
