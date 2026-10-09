from fastapi import FastAPI, UploadFile, File
from fastapi.middleware import CORSMiddleware
from fastapi.responses import HTMLResponse
from models import main_asr

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


