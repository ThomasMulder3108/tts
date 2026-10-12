from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from models import main_asr
import os
from datetime import datetime
from pathlib import Path

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

RECORDINGS_DIR = Path("recordings")
RECORDINGS_DIR.mkdir(exist_ok=True)

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".webm"
    path = RECORDINGS_DIR / filename
    path.write_bytes(await audio.read())
    print(f"Saved {path} ({path.stat().st_size} bytes)")
    text = main_asr(str(path))
    return {"text": text}

@app.get("/", response_class=HTMLResponse)
async def main():
    return open("index.html").read()
