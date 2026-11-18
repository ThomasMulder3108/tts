"""Speaker diarization with pyannote, and merging speaker turns with Whisper words."""

import functools

import numpy as np

import config

SAMPLE_RATE = 16000  # whisper.load_audio always returns 16 kHz mono


@functools.cache
def _load_pipeline():
    # Imported lazily so importing this module (e.g. in tests) doesn't need pyannote.
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(config.DIARIZATION_MODEL, token=config.HF_TOKEN)
    if pipeline is None:
        raise RuntimeError(
            f"Could not load {config.DIARIZATION_MODEL}. Accept the model's terms on "
            "huggingface.co and set HF_TOKEN in .env for the first download."
        )

    device = config.DIARIZATION_DEVICE
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))
    return pipeline


def diarize(waveform: np.ndarray) -> list[tuple[float, float, str]]:
    """Return (start, end, speaker) turns for a 16 kHz mono float32 waveform."""
    import torch

    kwargs = {}
    if config.MIN_SPEAKERS is not None:
        kwargs["min_speakers"] = config.MIN_SPEAKERS
    if config.MAX_SPEAKERS is not None:
        kwargs["max_speakers"] = config.MAX_SPEAKERS

    audio = {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": SAMPLE_RATE}
    output = _load_pipeline()(audio, **kwargs)
    # pyannote 4.x returns a DiarizeOutput; 3.x returns the Annotation directly.
    annotation = getattr(output, "speaker_diarization", output)
    return [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def _best_speaker(start: float, end: float, turns) -> str | None:
    best, best_overlap = None, 0.0
    for t_start, t_end, speaker in turns:
        overlap = min(end, t_end) - max(start, t_start)
        if overlap > best_overlap:
            best, best_overlap = speaker, overlap
    if best is not None:
        return best

    # No overlap (word falls in a gap): take the nearest turn.
    def distance(turn):
        t_start, t_end, _ = turn
        return max(t_start - end, start - t_end, 0.0)

    return min(turns, key=distance)[2] if turns else None


def assign_speakers(words: list[dict], turns: list[tuple[float, float, str]]) -> list[dict]:
    """Give each Whisper word a speaker and merge consecutive words per speaker.

    `words` are dicts with "word", "start" and "end" (Whisper word_timestamps).
    Raw labels are renamed to "Speaker 1", "Speaker 2", ... by first appearance.
    """
    labels: dict[str | None, str] = {}
    segments: list[dict] = []
    for w in words:
        raw = _best_speaker(w["start"], w["end"], turns)
        if raw not in labels:
            labels[raw] = f"Speaker {len(labels) + 1}"
        speaker = labels[raw]

        if segments and segments[-1]["speaker"] == speaker:
            seg = segments[-1]
            seg["end"] = w["end"]
            seg["text"] += w["word"]
        else:
            segments.append(
                {"speaker": speaker, "start": w["start"], "end": w["end"], "text": w["word"]}
            )

    for seg in segments:
        seg["text"] = seg["text"].strip()
    return segments


def format_transcript(segments: list[dict]) -> str:
    return "\n".join(f"{s['speaker']}: {s['text']}" for s in segments)
