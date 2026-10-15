from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from models import main_asr, whisper_model
import os
from datetime import datetime
from pathlib import Path
from summarizer import summarize_text
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

RECORDINGS_DIR = Path("recordings")
RECORDINGS_DIR.mkdir(exist_ok=True)
DOCUMENTS_DIR = Path("documents")
DOCUMENTS_DIR.mkdir(exist_ok=True)

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".webm"
    path = RECORDINGS_DIR / filename
    path.write_bytes(await audio.read())
    print(f"Saved {path} ({path.stat().st_size} bytes)")
    text = whisper_model(str(path))
    return {"text": text}

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
    return open("index.html").read()
