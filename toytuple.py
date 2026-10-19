"""
Toy-level tuple extraction pipeline for knowledge graph construction.

Steps:
  1. Ingest document (Docs/ExecSummary.pdf)
  2. Chunk the extracted text
  3. Call a local LLM via Ollama with prompt.txt + each chunk, asking for tuples
  4. Write all tuples to a CSV

Setup:
  pip install ollama pypdf
  ollama pull llama3.2:3b   (or whatever model you set MODEL to below)
  make sure the ollama app/service is running locally

Run:
  python extract_tuples.py
"""

import csv
import json
import os
import re

from pypdf import PdfReader
import ollama

#
PROMPT_PATH = "prompt.txt"
PDF_PATH = os.path.join("Docs", "ExecSummary.pdf")
OUTPUT_CSV = "tuples.csv"
MODEL = "llama3.2:3b"       # swap for any model you've pulled with `ollama pull`
CHUNK_SIZE_WORDS = 500      # words per chunk
CHUNK_OVERLAP_WORDS = 50    # overlap between chunks, to avoid cutting relations in half


# ---------- 1. Ingest ----------
def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages)
    # light cleanup: collapse excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------- 2. Chunk ----------
def chunk_text(text: str, size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


# ---------- 3. Call LLM ----------
def build_instructions(base_prompt: str) -> str:
    # Force a strict, parseable output format on top of whatever base_prompt asks for.
    return (
        f"{base_prompt}\n\n"
        "Return ONLY a JSON array of tuples, no prose, no markdown fences. "
        'Each tuple should be an object: {"subject": "...", "relation": "...", "object": "..."}. '
        "If no tuples are found in the text, return an empty array []."
    )


def ensure_model_pulled(model: str):
    """Pull the model if it isn't already available locally."""
    try:
        ollama.show(model)
    except Exception:
        print(f"Model '{model}' not found locally, pulling it now...")
        ollama.pull(model)


def extract_tuples_from_chunk(instructions: str, chunk: str) -> list:
    resp = ollama.generate(
        model=MODEL,
        prompt=f"{instructions}\n\nTEXT:\n{chunk}",
        format="json",  # forces the model to emit syntactically valid JSON
        options={"temperature": 0.2},
    )
    if isinstance(resp, dict):
        raw = resp.get("response", "")
    else:
        raw = getattr(resp, "response", "")
    raw = raw.strip()

    # strip accidental code fences just in case
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # model may have wrapped the array in an object, e.g. {"tuples": [...]}
            for value in data.values():
                if isinstance(value, list):
                    return value
        print("  Warning: LLM did not return a JSON list (or list-containing object), skipping chunk.")
        return []
    except json.JSONDecodeError:
        print("  Warning: could not parse LLM output as JSON, skipping chunk.")
        print("  Raw output:", raw[:200])
        return []


# ---------- 4. CSV ----------
def write_csv(tuples: list, path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "relation", "object"])
        writer.writeheader()
        for t in tuples:
            writer.writerow(
                {
                    "subject": t.get("subject", ""),
                    "relation": t.get("relation", ""),
                    "object": t.get("object", ""),
                }
            )


# ---------- Main ----------
def main():
    print("Loading prompt and PDF...")
    base_prompt = load_prompt(PROMPT_PATH)
    text = load_pdf_text(PDF_PATH)
    instructions = build_instructions(base_prompt)

    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"  {len(chunks)} chunk(s) created")

    ensure_model_pulled(MODEL)

    all_tuples = []
    for i, chunk in enumerate(chunks, 1):
        print(f"Calling LLM on chunk {i}/{len(chunks)}...")
        tuples = extract_tuples_from_chunk(instructions, chunk)
        print(f"  -> {len(tuples)} tuple(s) extracted")
        all_tuples.extend(tuples)

    print(f"Writing {len(all_tuples)} total tuples to {OUTPUT_CSV}...")
    write_csv(all_tuples, OUTPUT_CSV)
    print("Done.")


if __name__ == "__main__":
    main()