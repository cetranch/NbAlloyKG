import sys
import json
import argparse

import ollama
from pypdf import PdfReader

MODEL = "qwen2.5:7b"  # change to whatever you're testing
CHUNK_CHARS = 1000    # smaller chunks -> more tuples per document

# Ollama's format="json" forces a top-level OBJECT, so an array can't be returned.
# Passing a schema lets us ask for {"tuples": [ ... ]} explicitly.
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


# ---------------------------------------------------------------- PDF / chunking
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


# ---------------------------------------------------------------- prompts
def build_pass1_prompt(text: str) -> str:
    return (
        "Extract EVERY subject-relation-object triple you can find in the text below. "
        "There will usually be many (10 or more). Do not stop after one. "
        'Return a JSON object of the form {"tuples": [{"subject": "...", "relation": "...", "object": "..."}, ...]}.\n\n'
        f"TEXT:\n{text}"
    )


def build_pass2_prompt(text: str, candidates: list[dict]) -> str:
    return (
        "You are reviewing subject-relation-object triples that were extracted from the text below.\n"
        "Produce the FINAL, corrected list of triples for this text by doing three things:\n"
        "1. REMOVE any triple that is not clearly supported by the text.\n"
        "2. FIX triples with vague subjects/objects (e.g. pronouns like 'it' or 'they') by "
        "replacing them with the specific entity named in the text, and keep relations short and consistent.\n"
        "3. ADD any important triples that were missed.\n"
        "Do not invent facts that are not in the text.\n"
        'Return a JSON object of the form {"tuples": [{"subject": "...", "relation": "...", "object": "..."}, ...]}.\n\n'
        f"TEXT:\n{text}\n\n"
        f"CANDIDATE TRIPLES:\n{json.dumps(candidates, ensure_ascii=False, indent=1)}"
    )


def load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def make_prompt(text: str, template: str | None, default_builder) -> str:
    if template is None:
        return default_builder(text)
    if "{text}" in template:
        return template.replace("{text}", text)
    return template + "\n\nTEXT:\n" + text  # no placeholder -> append chunk


# ---------------------------------------------------------------- LLM calls
def _call(prompt: str, temperature: float):
    resp = ollama.generate(
        model=MODEL,
        prompt=prompt,
        format=SCHEMA,
        options={"temperature": temperature, "num_predict": 2048},
    )
    raw = resp.response
    try:
        return json.loads(raw).get("tuples", [])
    except json.JSONDecodeError as e:
        print("  Failed to parse JSON:", e)
        print("  Raw output:", repr(raw))
        return None  # None = failure (distinct from a legit empty list)


def pass1_extract(text: str, template: str | None):
    """Pass 1: recall-oriented extraction."""
    prompt = make_prompt(text, template, build_pass1_prompt)
    return _call(prompt, temperature=0.2) or []


def pass2_refine(text: str, candidates: list[dict], template: str | None):
    """Pass 2: verify / clean / fill gaps, given the chunk and pass-1 output."""
    if template is None:
        prompt = build_pass2_prompt(text, candidates)
    else:
        # Custom pass-2 template: {text} and {tuples} placeholders are supported.
        prompt = template.replace("{text}", text).replace(
            "{tuples}", json.dumps(candidates, ensure_ascii=False, indent=1)
        )
        if "{tuples}" not in template:
            prompt += "\n\nCANDIDATE TRIPLES:\n" + json.dumps(candidates, ensure_ascii=False, indent=1)
    refined = _call(prompt, temperature=0.0)
    # If pass 2 fails or wipes everything out, fall back to pass-1 results.
    if not refined:
        return candidates
    return refined


# ---------------------------------------------------------------- helpers
def clean(t: dict) -> tuple[str, str, str]:
    return (
        str(t.get("subject", "")).strip(),
        str(t.get("relation", "")).strip(),
        str(t.get("object", "")).strip(),
    )


def dedupe(tuples: list[dict]) -> list[tuple[str, str, str]]:
    seen, out = set(), []
    for t in tuples:
        s, r, o = clean(t)
        if not (s and r and o):
            continue
        key = (s.lower(), r.lower(), o.lower())
        if key not in seen:
            seen.add(key)
            out.append((s, r, o))
    return out


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--prompt-file", default=None,
                        help="Custom prompt for pass 1. Use {text} as the chunk placeholder.")
    parser.add_argument("--pass2-prompt-file", default=None,
                        help="Custom prompt for pass 2. Use {text} and {tuples} placeholders.")
    parser.add_argument("--no-pass2", action="store_true",
                        help="Skip the refinement pass (behaves like the original single-pass script).")
    args = parser.parse_args()

    t1 = load_prompt_template(args.prompt_file) if args.prompt_file else None
    t2 = load_prompt_template(args.pass2_prompt_file) if args.pass2_prompt_file else None
    print(f"Pass 1 prompt: {'custom: ' + args.prompt_file if t1 else 'default'}")
    if not args.no_pass2:
        print(f"Pass 2 prompt: {'custom: ' + args.pass2_prompt_file if t2 else 'default'}")

    text = extract_pdf_text(args.pdf_path)
    if not text:
        print("No extractable text found in PDF.")
        sys.exit(1)

    chunks = chunk_text(text)
    print(f"Processing {len(chunks)} chunk(s)...")

    all_tuples = []
    total_p1 = 0
    for i, chunk in enumerate(chunks, 1):
        first = pass1_extract(chunk, t1)
        total_p1 += len(first)

        if args.no_pass2:
            final = first
            print(f"  chunk {i}/{len(chunks)}: pass1={len(first)}")
        else:
            final = pass2_refine(chunk, first, t2)
            print(f"  chunk {i}/{len(chunks)}: pass1={len(first)} -> pass2={len(final)}")

        all_tuples.extend(final)

    unique = dedupe(all_tuples)
    print(f"\nPass 1 total: {total_p1} | after refinement: {len(all_tuples)} | unique: {len(unique)}")
    print(f"\nFound {len(unique)} tuple(s):")
    for t in unique:
        print(t)


if __name__ == "__main__":
    main()