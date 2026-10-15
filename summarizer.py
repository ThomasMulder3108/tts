import ollama
import re

MODEL = "qwen3:4b"
NUM_CTX = 8192
CHUNK_WORDS = 2500

SYSTEM = (
    "You summarize meeting transcripts. "
    "Write the summary in the same language as the transcript (English or Dutch), "
    "including the section headings. "
    "Structure: a short summary (2-4 sentences), then 'Key decisions' and "
    "'Action items' as bullet lists, naming the owner when the transcript does. "
    "Use only information from the transcript. If there are no decisions or "
    "action items, say so."
)

CHUNK_PROMPT = (
    "Summarize this part of a meeting transcript in the same language as the "
    "transcript. Keep names, decisions and action items. Use only the transcript."
)

def _chat(system, text):

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text}
        ],
        options={
            "num_ctx": NUM_CTX,
            "temperature": 0.3},
        think = False
    )

    out = response['message']['content']
    print(response)

    return re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()

def _chunks(text, chunk_words=CHUNK_WORDS):
    words = text.split()
    for i in range(0, len(words), chunk_words):
        yield " ".join(words[i:i + chunk_words])


def summarize_text(transcript):
    # Use Ollama to summarize
    transcript = transcript.strip()
    if not transcript:
        return "No transcript provided."

    # Split transcript into chunks
    chunks = list(_chunks(transcript))

    if len(chunks) == 1:
        return _chat(SYSTEM, f"{CHUNK_PROMPT}\n\n{chunks[0]}")

    # Summarize each chunk
    partials = [_chat(SYSTEM, f"{CHUNK_PROMPT}\n\n{chunk}") for chunk in chunks]
    combined_summary = "\n\n".join(partials)
    return _chat(SYSTEM, f"Combine these summaries into one coherent summary:\n\n{combined_summary}")



    



