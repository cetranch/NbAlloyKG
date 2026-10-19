"""
Minimal sanity check for the Ollama Python client.
No PDF, no chunking, no ontology — just: send a tiny prompt, print what comes back.

Run:
  python ollama_sanity_check.py
"""

import ollama

MODEL = "llama3.2:3b"  # change to whatever you're testing

prompt = (
    "Return ONLY a JSON array of tuples, no prose, no markdown fences. "
    'Each tuple should be an object: {"subject": "...", "relation": "...", "object": "..."}. '
    "TEXT: The cat sat on the mat. The dog chased the cat."
)

resp = ollama.generate(model=MODEL, prompt=prompt, format="json", options={"temperature": 0.2})

try:
    print(repr(getattr(resp, "response", None)))
except Exception as e:
    print("failed:", e)

