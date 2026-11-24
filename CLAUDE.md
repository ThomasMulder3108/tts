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
ruff check . && ruff format --check .   # lint (incl. bandit "S" security rules) + format check
pre-commit install                       # one-time: run ruff + gitleaks on every commit
```

Summarization requires a running Ollama server with the model pulled: `ollama pull qwen3:4b-instruct-2507-q4_K_M`.

Speaker diarization uses the gated pyannote model `pyannote/speaker-diarization-community-1`. Accept its terms on huggingface.co once and set `HF_TOKEN` in `.env` for the first download. After that it loads offline from the HF cache.

CI (`.github/workflows/tests.yml`) runs three jobs for pushes to `main` and for PRs: `lint` (ruff check + format check), `secrets` (gitleaks over the full git history) and `test` (`pytest -v` on Python 3.14).

## Configuration

All runtime settings live in `config.py`, which reads environment variables, or `.env` through python-dotenv. `.env.example` lists every setting. Don't hardcode models, hosts, paths or origins in other modules. Add them to `config.py` and `.env.example` instead. `.env` is gitignored. CORS defaults to localhost origins only.

## Architecture

- `index.html`: a single-page frontend served by `GET /`. It records audio with MediaRecorder (webm), POSTs it to `/transcribe`, then POSTs the transcript as JSON to `/summarize`.
- `main.py`: the FastAPI app.
  - `/transcribe` handles both browser recordings and uploaded files (the frontend's "Upload audio file" button plus a confirm panel). It checks the extension against `config.ALLOWED_AUDIO_EXTENSIONS` (415), rejects empty files (400) and files over `MAX_UPLOAD_MB` (413), saves the upload to `recordings/<timestamp><original ext>` and calls `models.transcribe_with_speakers`. A `RuntimeError` from ffmpeg decoding becomes a 422 and the saved file is deleted. `GET /upload-config` exposes the allowlist and limit to the frontend. It returns `{"text": "Speaker 1: ...", "segments": [{speaker, start, end, text}]}`. The frontend renders the transcript from `segments` and lets the user rename speakers. The renamed text is what gets sent to `/summarize`.
  - `/summarize` saves the transcript to `documents/<timestamp>.txt` and calls `summarizer.summarize_text`. On failure it returns `{"error": ...}` with HTTP 200, not an error status.
- `models.py`: ASR.
  - `transcribe_with_speakers` is what the app uses. It decodes the audio once with `whisper.load_audio` (16 kHz mono), runs Whisper with `word_timestamps=True` and pyannote diarization on the same array, then merges the two with `diarization.assign_speakers`.
  - `whisper_model` (Whisper "small", no speakers) is the plain path, used for the LibriSpeech fallback.
- `diarization.py`: the pyannote pipeline, cached and imported lazily so tests don't need pyannote loaded. `assign_speakers` is a pure function: each word goes to the speaker turn it overlaps most, or to the nearest turn when it falls in a gap. Raw labels are renumbered `Speaker 1..N` by first appearance.
  - `main_asr` (torchaudio wav2vec2 with a greedy CTC decoder) is an older, experimental path that is only used from `visual.ipynb`.
- `summarizer.py`: map-reduce summarization through `ollama.chat`. A transcript longer than `CHUNK_WORDS` is split, each chunk is summarized, and the chunk summaries are combined in a final call. `<think>` blocks are stripped from model output. The prompts tell the model to answer in the transcript's language (test data is Dutch).
- `import_data.py`: downloads and loads LibriSpeech (`train-clean-100`, `test-clean`) into the repo root.

## Gotchas

- `whisper_model()` only calls `load_data()` when no audio file is passed, so the extracted `LibriSpeech/` folder is only needed for that fallback and for `main_asr`/the notebook. `load_data()` uses `download=False`, so to get the data, temporarily set `download=True` or extract the tarballs into the repo root. The Whisper model is loaded once and cached.
- `models.py` imports torch, torchaudio, whisper, matplotlib and IPython at module level, and `main.py` imports `models`. Importing `main` (including in tests) is therefore slow and needs the full ML stack installed.
- The tests never touch real models. They monkeypatch `main.transcribe_with_speakers`, `main.summarize_text`, `summarizer._chat` or `summarizer.ollama.chat`. An autouse fixture redirects `RECORDINGS_DIR` and `DOCUMENTS_DIR` to `tmp_path`. Follow the same pattern for new endpoint tests.
- `recordings/`, `documents/`, `LibriSpeech/`, `Speech/` and the dataset tarballs are gitignored local data.
