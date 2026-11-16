# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local "Meeting Transcriber" web app. Despite the repo name, it does speech-to-text (ASR) and summarization, not text-to-speech. The browser records audio, the FastAPI backend transcribes it with Whisper, and a local Ollama model summarizes the transcript.

## Commands

```bash
pip install -r requirements.txt          # also needs ffmpeg on PATH (whisper/pydub)
uvicorn main:app --reload                # serves the UI at http://localhost:8000/
pytest -v                                # run all tests (CI also installs httpx)
pytest test_app.py::test_think_block_is_stripped   # run a single test
```

Summarization requires a running Ollama server with the model pulled: `ollama pull qwen3:4b-instruct-2507-q4_K_M`.

CI (`.github/workflows/tests.yml`) runs `pytest -v` on Python 3.14 for pushes to `main` and for PRs.

## Architecture

- `index.html`: a single-page frontend served by `GET /`. It records audio with MediaRecorder (webm), POSTs it to `/transcribe`, then POSTs the transcript as JSON to `/summarize`.
- `main.py`: the FastAPI app.
  - `/transcribe` saves the upload to `recordings/<timestamp>.webm` and calls `models.whisper_model`.
  - `/summarize` saves the transcript to `documents/<timestamp>.txt` and calls `summarizer.summarize_text`. On failure it returns `{"error": ...}` with HTTP 200, not an error status.
- `models.py`: ASR.
  - `whisper_model` (Whisper "small") is the one the app uses.
  - `main_asr` (torchaudio wav2vec2 with a greedy CTC decoder) is an older, experimental path that is only used from `visual.ipynb`.
- `summarizer.py`: map-reduce summarization through `ollama.chat`. A transcript longer than `CHUNK_WORDS` is split, each chunk is summarized, and the chunk summaries are combined in a final call. `<think>` blocks are stripped from model output. The prompts tell the model to answer in the transcript's language (test data is Dutch).
- `import_data.py`: downloads and loads LibriSpeech (`train-clean-100`, `test-clean`) into the repo root.

## Gotchas

- `whisper_model()` calls `load_data()` on every request, even when it is given an audio file. `load_data()` uses `download=False`, so the extracted `LibriSpeech/` folder (with `train-clean-100` and `test-clean`) must already exist locally, or `/transcribe` raises an error. To get the data, temporarily set `download=True` or extract the tarballs into the repo root. `whisper_model()` also reloads the Whisper model on every call.
- `models.py` imports torch, torchaudio, whisper, matplotlib and IPython at module level, and `main.py` imports `models`. Importing `main` (including in tests) is therefore slow and needs the full ML stack installed.
- The tests never touch real models. They monkeypatch `main.whisper_model`, `main.summarize_text`, `summarizer._chat` or `summarizer.ollama.chat`. An autouse fixture redirects `RECORDINGS_DIR` and `DOCUMENTS_DIR` to `tmp_path`. Follow the same pattern for new endpoint tests.
- `recordings/`, `documents/`, `LibriSpeech/`, `Speech/` and the dataset tarballs are gitignored local data.
