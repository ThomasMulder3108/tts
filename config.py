"""Runtime configuration, read from environment variables (or a local .env file).

Copy .env.example to .env and adjust. Never commit .env.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


# Web server
CORS_ORIGINS = _list("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")

# Storage
RECORDINGS_DIR = Path(os.getenv("RECORDINGS_DIR", "recordings"))
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", "documents"))

# Speech-to-text
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# Summarization (Ollama). The ollama client reads OLLAMA_HOST itself.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "2500"))
