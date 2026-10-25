import sys
import json
import argparse

import ollama
from pypdf import PdfReader

MODEL = "qwen3:30b"  # change to whatever you're testing
CHUNK_CHARS = 1000    # smaller chunks -> more tuples per document

SCHEMA = {
    "type": "object",
    "properties": {
        "tuples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "relation": {"type": "string"},
                    "object": {"type": "string"},
                },
                "required": ["subject", "relation", "object"],
            },
        }
    },
    "required": ["tuples"],
}


def extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join((p.extract_text() or "") for p in reader.pages).strip()


def chunk_text(text: str, size: int = CHUNK_CHARS):
    paragraphs = text.split("\n")
    chunk, chunks = "", []
    for para in paragraphs:
        if len(chunk) + len(para) > size and chunk:
            chunks.append(chunk)
            chunk = ""
        chunk += para + "\n"
    if chunk.strip():
        chunks.append(chunk)
    return chunks


def build_prompt(text: str) -> str:
    return (
        "Extract EVERY subject-relation-object triple you can find in the text below. "
        "There will usually be many (10 or more). Do not stop after one. "
        'Return a JSON object of the form {"tuples": [{"subject": "...", "relation": "...", "object": "..."}, ...]}.\n\n'
        f"TEXT:\n{text}"
    )


def load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def make_prompt(text: str, template: str | None) -> str:
    if template is None:
        return build_prompt(text)              # default prompt
    if "{text}" in template:
        return template.replace("{text}", text)
    return template + "\n\nTEXT:\n" + text     # no placeholder -> append chunk


def get_tuples(text: str, template: str | None = None):
    resp = ollama.generate(
    model=MODEL,
    prompt=make_prompt(text, template),
    format=SCHEMA,
    think=False,
    options={"temperature": 0.2, "num_predict": 2048},
    )
    raw = resp.response
    try:
        return json.loads(raw).get("tuples", [])
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
        print("Raw output:", repr(raw))
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Path to a custom prompt file. If omitted, the default prompt is used.",
    )
    args = parser.parse_args()

    template = load_prompt_template(args.prompt_file) if args.prompt_file else None
    print(f"Using {'custom prompt: ' + args.prompt_file if template else 'default prompt'}")

    text = extract_pdf_text(args.pdf_path)
    if not text:
        print("No extractable text found in PDF.")
        sys.exit(1)

    chunks = chunk_text(text)
    print(f"Processing {len(chunks)} chunk(s)...")

    all_tuples = []
    for i, chunk in enumerate(chunks, 1):
        found = get_tuples(chunk, template)
        print(f"  chunk {i}/{len(chunks)}: {len(found)} tuple(s)")
        all_tuples.extend(found)

    print(f"\nFound {len(all_tuples)} tuple(s):")
    for t in all_tuples:
        print((t.get("subject", ""), t.get("relation", ""), t.get("object", "")))
if __name__ == "__main__":
    main()