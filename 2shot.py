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

# ---------------------------------------------------------------------------
# Two worked examples (the "2 shots"). Each is (input text, expected tuples).
# Swap these for examples closer to your own documents for best results.
# ---------------------------------------------------------------------------
EXAMPLES = [
  {
    "text": "The development of high-entropy alloys (HEAs) or complex concetrated alloys (CCAs) challenges the traditional alloying strategy consisting of only one or two predominant elements [1,2]. The greatly broadened alloy composition space by the idea of HEAs results in unexpected solid-solution related structures and hence outstanding integrated properties, which attracts growing attentions in the material science community over the past decades." [
      {"subject": "HEAs", "relation": "have", "object": "broadened alloy composition"},
      {"subject": "HEAs", "relation": "have", "object": "unexpected solid solution structures"},
      {"subject": "HEAs", "relation": "have", "object": "outstanding properties"}
    ]
  },
  {
    "text": "Fig. 1 shows the X-ray diffraction profiles of the as-cast Ti2ZrHf0.5VNb0.5Alx RHEAs. Only single disordered BCC phase peaks can be found in Al0, Al0.25 and Al0.5 alloys in Fig. 1(a), while in the RHEAs with higher Al content, i.e., Al0.75 and Al1 alloys, minor peaks belonging to the B2 crystal identified as the (100)B2, (111)B2 and (210)B2 appear. This suggests that the Al addition triggers the B2-type ordering in the Ti2ZrHf0.5VNb0.5 RHEA that show basically BCC crystalline structure. Fig. 1(b) shows the details of the XRD profiles from 25 to 40 degree. The pattern is smooth without any extra weak peaks beside those above from B2 and BCC. The lattice parameters were calculated decreasing from 3.3655 Å to 3.3269 Å as the Al content increases from x =0 to x =1.0. This could be roughly attributed to the smaller atom radius of Al (1.43 Å) compared to the average atomic radius (1.437 Å) of the Ti2ZrHf0.5VNb0.5 matrix. " [
      {"subject": "higher Al content", "relation": "had", "object": "B2 type ordering"},
      {"subject": "XRD", "relation": "shows", "object": "no other peaks than B2 and BCC"},
      {"subject": "lattice parameters", "relation": "decreased", "object": "from 3.3655 Å to 3.3269 Å as the Al content increases from x =0 to x =1.0"},
      {"subject": "decreasing lattice parameters", "relation": "attributed", "object": "smaller atomic radius of Al"},

    ]
  }
]


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


def build_messages(text: str, template: str | None):
    """Build a 2-shot conversation: example user/assistant pairs, then the real chunk.

    The same prompt wrapper (default or custom template) is applied to the
    examples and to the real chunk so the model sees a consistent format.
    """
    messages = []
    for ex_text, ex_tuples in EXAMPLES:
        messages.append({"role": "user", "content": make_prompt(ex_text, template)})
        messages.append({"role": "assistant", "content": json.dumps({"tuples": ex_tuples})})
    messages.append({"role": "user", "content": make_prompt(text, template)})
    return messages


def get_tuples(text: str, template: str | None = None):
    resp = ollama.chat(
        model=MODEL,
        messages=build_messages(text, template),
        format=SCHEMA,
        options={"temperature": 0.2, "num_predict": 2048},
    )
    raw = resp.message.content
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
    print(f"Using {'custom prompt: ' + args.prompt_file if template else 'default prompt'} (2-shot)")

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