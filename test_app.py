import os
import pytest
from fastapi.testclient import TestClient
import main
import summarizer

client = TestClient(main.app)

DUTCH = (
    "Jan zei dat we de lancering naar vrijdag verplaatsen. Piet zal de factuur "
    "versturen en Anna regelt de presentatie voor de klant. We zijn het eens "
    "over het nieuwe budget en plannen volgende week een vervolgafspraak."
)


# Fixtures
@pytest.fixture(autouse=True)
def isolate_folders(tmp_path, monkeypatch):
    """Write recordings/transcripts to a temp folder"""
    monkeypatch.setattr(main, "RECORDINGS_DIR", tmp_path / "recordings")
    monkeypatch.setattr(main, "DOCUMENTS_DIR", tmp_path / "documents")
    (tmp_path / "recordings").mkdir()
    (tmp_path / "documents").mkdir()
 
 
@pytest.fixture
def fake_audio():
    return {"audio": ("recording.webm", b"fake-audio-bytes", "audio/webm")}

# API tests
def test_index_page_is_served():
    res = client.get("/")
    assert res.status_code == 200
    assert "Meeting Transcriber" in res.text
 
 
def test_transcribe_returns_text_and_saves_file(monkeypatch, fake_audio, tmp_path):
    monkeypatch.setattr(main, "whisper_model", lambda path: "hello world")
    res = client.post("/transcribe", files=fake_audio)
    assert res.status_code == 200
    assert res.json() == {"text": "hello world"}
    assert len(list((tmp_path / "recordings").iterdir())) == 1
 
 
def test_transcribe_rejects_request_without_audio():
    res = client.post("/transcribe")
    assert res.status_code == 422
 
 
def test_summarize_returns_summary_and_saves_transcript(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "summarize_text", lambda t: "a short summary")
    res = client.post("/summarize", json={"transcript": DUTCH})
    assert res.status_code == 200
    assert res.json() == {"summary": "a short summary"}
    saved = list((tmp_path / "documents").iterdir())
    assert len(saved) == 1
    assert saved[0].read_text(encoding="utf-8") == DUTCH   # non-ASCII survives
 
 
def test_summarize_requires_json_body():
    res = client.post("/summarize")
    assert res.status_code == 422
 

# Unit tests for the summarizer
def test_chunks_split_long_text():
    text = " ".join(["woord"] * 5)
    assert list(summarizer._chunks(text, 2)) == ["woord woord", "woord woord", "woord"]
 
 
def test_empty_transcript():
    assert summarizer.summarize_text("   ") == "No transcript provided."
 
 
def test_long_transcript_is_summarized_in_chunks_then_combined(monkeypatch):
    calls = []
    monkeypatch.setattr(summarizer, "_chat", lambda s, t: calls.append(t) or "part")
    words = summarizer.CHUNK_WORDS * 2 + 1          # always 3 chunks
    summarizer.summarize_text(" ".join(["word"] * words))
    assert len(calls) == 4                          # 3 partials + 1 combine
 
def test_think_block_is_stripped(monkeypatch):
    fake = lambda **kwargs: {"message": {"content": "<think>hmm</think>The summary."}}
    monkeypatch.setattr(summarizer.ollama, "chat", fake)
    assert summarizer._chat("system", "text") == "The summary."
 
 