# tts — Meeting Transcriber

A local web app that records a meeting in the browser, transcribes it with Whisper, works out who spoke when with pyannote, and summarizes it with a local Ollama model.

## Setup

```bash
pip install -r requirements.txt     # needs ffmpeg on PATH
cp .env.example .env
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

Speaker diarization uses the gated model [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1). Accept its terms on Hugging Face and put an access token in `.env` as `HF_TOKEN`. The token is only needed for the first download; after that the model loads from the local cache.

## Run

```bash
uvicorn main:app --reload           # http://localhost:8000/
```

After transcribing, you can rename "Speaker 1", "Speaker 2", … above the transcript. The summary then uses those names for decisions and action items.
