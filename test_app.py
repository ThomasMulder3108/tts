import pytest
from fastapi.testclient import TestClient

import diarization
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


def test_transcribe_returns_text_and_segments_and_saves_file(monkeypatch, fake_audio, tmp_path):
    segments = [{"speaker": "Speaker 1", "start": 0.0, "end": 1.0, "text": "hello world"}]
    monkeypatch.setattr(
        main,
        "transcribe_with_speakers",
        lambda path: {"text": "Speaker 1: hello world", "segments": segments},
    )
    res = client.post("/transcribe", files=fake_audio)
    assert res.status_code == 200
    assert res.json() == {"text": "Speaker 1: hello world", "segments": segments}
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
    assert saved[0].read_text(encoding="utf-8") == DUTCH  # non-ASCII survives


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
    words = summarizer.CHUNK_WORDS * 2 + 1  # always 3 chunks
    summarizer.summarize_text(" ".join(["word"] * words))
    assert len(calls) == 4  # 3 partials + 1 combine


def test_think_block_is_stripped(monkeypatch):
    def fake(**kwargs):
        return {"message": {"content": "<think>hmm</think>The summary."}}

    monkeypatch.setattr(summarizer.ollama, "chat", fake)
    assert summarizer._chat("system", "text") == "The summary."


# Unit tests for speaker assignment
def _w(word, start, end):
    return {"word": word, "start": start, "end": end}


def test_words_are_split_at_speaker_change():
    words = [_w(" Hoi", 0.0, 0.4), _w(" Jan.", 0.4, 0.8), _w(" Hallo", 1.2, 1.6)]
    turns = [(0.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")]
    assert diarization.assign_speakers(words, turns) == [
        {"speaker": "Speaker 1", "start": 0.0, "end": 0.8, "text": "Hoi Jan."},
        {"speaker": "Speaker 2", "start": 1.2, "end": 1.6, "text": "Hallo"},
    ]


def test_word_spanning_two_turns_goes_to_largest_overlap():
    turns = [(0.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")]
    segs = diarization.assign_speakers([_w(" ja", 0.9, 1.5)], turns)
    assert segs[0]["speaker"] == "Speaker 1"  # first appearance, raw SPEAKER_01
    segs = diarization.assign_speakers([_w(" a", 0.0, 0.5), _w(" ja", 0.9, 1.5)], turns)
    assert [s["speaker"] for s in segs] == ["Speaker 1", "Speaker 2"]


def test_word_in_gap_goes_to_nearest_turn():
    turns = [(0.0, 1.0, "SPEAKER_00"), (3.0, 4.0, "SPEAKER_01")]
    words = [_w(" a", 0.0, 0.5), _w(" b", 2.6, 2.8)]  # gap word is closer to SPEAKER_01
    segs = diarization.assign_speakers(words, turns)
    assert [s["speaker"] for s in segs] == ["Speaker 1", "Speaker 2"]


def test_labels_are_numbered_by_first_appearance():
    turns = [(0.0, 1.0, "SPEAKER_01"), (1.0, 2.0, "SPEAKER_00"), (2.0, 3.0, "SPEAKER_01")]
    words = [_w(" a", 0.1, 0.5), _w(" b", 1.1, 1.5), _w(" c", 2.1, 2.5)]
    segs = diarization.assign_speakers(words, turns)
    assert [s["speaker"] for s in segs] == ["Speaker 1", "Speaker 2", "Speaker 1"]


def test_no_words_gives_no_segments():
    assert diarization.assign_speakers([], [(0.0, 1.0, "SPEAKER_00")]) == []


def test_format_transcript():
    segs = [
        {"speaker": "Speaker 1", "start": 0, "end": 1, "text": "Hoi."},
        {"speaker": "Speaker 2", "start": 1, "end": 2, "text": "Hallo!"},
    ]
    assert diarization.format_transcript(segs) == "Speaker 1: Hoi.\nSpeaker 2: Hallo!"
