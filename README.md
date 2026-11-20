# Meeting Transcriber

A local web app that records a meeting in the browser, transcribes it with [Whisper](https://github.com/openai/whisper), works out who spoke when with [pyannote](https://github.com/pyannote/pyannote-audio) and summarizes the transcript with a local [Ollama](https://ollama.com) model. Everything runs on your own machine; no audio or text is sent to a cloud service.

> The repository is called `tts`, but the app does speech-to-text and summarization, not text-to-speech, wasn't paying attention when I created the project.

## How it works

1. The browser records audio with the MediaRecorder API (webm).
2. The audio is posted to `/transcribe`. The backend saves it to `recordings/`, transcribes it with Whisper and runs pyannote speaker diarization on the same audio. Each word is assigned to the speaker who was talking at that moment, and speakers are labelled `Speaker 1`, `Speaker 2`, … in order of appearance.
3. You can rename the speakers above the transcript. The renamed transcript is what gets summarized, so decisions and action items use those names.
4. When you click Summarize, the transcript is posted to `/summarize`. The backend saves it to `documents/` and summarizes it with Ollama. Long transcripts are split into chunks, each chunk is summarized, and the chunk summaries are combined into one final summary. The summary is written in the language of the transcript.

## Requirements

- Python 3.11 or newer (CI runs on 3.14)
- [ffmpeg](https://ffmpeg.org) on your `PATH`, used by Whisper and pydub
- [Ollama](https://ollama.com), running locally, with the summarization model pulled

## Installation

```bash
git clone https://github.com/ThomasMulder3108/tts.git
cd tts
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull qwen3:4b-instruct-2507-q4_K_M
```

Speaker diarization uses the gated model [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1). Accept its terms on Hugging Face and put an access token in `.env` as `HF_TOKEN` (see [Configuration](#configuration)). The token is only needed for the first download; after that the model loads from the local cache.

The first transcription downloads the Whisper and pyannote models, so it takes longer than later ones.

## Usage

Make sure Ollama is running (`ollama serve`, or the Ollama desktop app), then start the server:

```bash
uvicorn main:app --reload
```

Open <http://localhost:8000/> and:

1. Click **Start recording** and allow microphone access.
2. Click **Stop and transcribe**. The transcript appears on the page, split by speaker.
3. Optionally rename "Speaker 1", "Speaker 2", … to the real names.
4. Click **Summarize** to get a summary of the transcript.

Both the transcript and the summary have a **Copy** button.

## Configuration

Settings are read from environment variables or from a `.env` file in the project root. Copy the example and adjust it:

```bash
cp .env.example .env
```

| Variable | Default | Description |
| --- | --- | --- |
| `CORS_ORIGINS` | `http://localhost:8000,http://127.0.0.1:8000` | Comma-separated origins allowed to call the API |
| `RECORDINGS_DIR` | `recordings` | Where uploaded audio is stored |
| `DOCUMENTS_DIR` | `documents` | Where transcripts are stored |
| `WHISPER_MODEL` | `small` | Whisper model size: `tiny`, `base`, `small`, `medium` or `large` |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-community-1` | pyannote pipeline used for speaker diarization |
| `HF_TOKEN` | | Hugging Face token, only needed for the first download of the gated model |
| `DIARIZATION_DEVICE` | `auto` | `auto` (cuda if available, else cpu), `cpu`, `cuda` or `mps` |
| `MIN_SPEAKERS` | | Optional lower bound on the number of speakers |
| `MAX_SPEAKERS` | | Optional upper bound on the number of speakers |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Address of the Ollama server |
| `OLLAMA_MODEL` | `qwen3:4b-instruct-2507-q4_K_M` | Model used for summarization |
| `OLLAMA_NUM_CTX` | `8192` | Context window passed to Ollama |
| `OLLAMA_TEMPERATURE` | `0.3` | Sampling temperature for summarization |
| `CHUNK_WORDS` | `2500` | Transcripts longer than this are summarized in chunks |

Larger Whisper models are more accurate but slower and need more memory. If you change `OLLAMA_MODEL`, pull that model with `ollama pull` first.

## API

| Method | Path | Body | Response |
| --- | --- | --- | --- |
| `GET` | `/` | | The web UI |
| `POST` | `/transcribe` | multipart form with an `audio` file | `{"text": "Speaker 1: ...", "segments": [{"speaker", "start", "end", "text"}]}` |
| `POST` | `/summarize` | JSON `{"transcript": "..."}` | `{"summary": "..."}`, or `{"error": "..."}` if summarization fails |

`/summarize` returns HTTP 200 for errors too, so check for the `error` key.

## Project structure

| File | Purpose |
| --- | --- |
| `main.py` | FastAPI app and endpoints |
| `index.html` | Single-page frontend |
| `models.py` | Speech recognition with Whisper, combined with diarization (plus an experimental wav2vec2 path used by the notebook) |
| `diarization.py` | pyannote speaker diarization and word-to-speaker assignment |
| `summarizer.py` | Chunked summarization through Ollama |
| `config.py` | Runtime settings |
| `import_data.py` | Downloads the LibriSpeech dataset, only needed for experiments |
| `visual.ipynb` | Notebook for experimenting with the ASR models |
| `test_app.py` | Tests |

`recordings/`, `documents/`, `.env` and the LibriSpeech data are gitignored.

## Development

```bash
pytest -v                                 # run the tests (they mock Whisper, pyannote and Ollama)
ruff check . && ruff format --check .     # lint and format check
pre-commit install                        # run ruff and gitleaks on every commit
```

CI runs linting, a gitleaks secret scan and the test suite on every push to `main` and on pull requests.
