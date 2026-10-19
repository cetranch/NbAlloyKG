#The purpose of this code is to iterate through each abstract and select the optimal one to be hand annotated

#Step one read in files

#step two iterate through and trim to only abstract, no abstract elimitate

#step3 call model and apply prompt: prompt should be asking for paper most realted to my topics of
# niobium alloy oxidtion
# I want the Model ("qwen2.5:7b") to explain the rational for why this is the best one for me to annotate

#Step 4 print name of paper and abstract

# The purpose of this code is to iterate through each paper and select the optimal one
# to be hand annotated (for building the Nb-Ti-Al oxidation knowledge graph).
#
# Setup:
#   pip install ollama pypdf
#   ollama pull qwen2.5:7b      (and make sure the Ollama app/server is running)
#
# Usage:
#   python abstractSelector.py path/to/papers_folder

import json
import re
import sys
from pathlib import Path

import ollama
from pypdf import PdfReader

MODEL = "qwen2.5:7b"
PAPERS_DIR = Path(r"C:\Users\cetra\PycharmProjects\NbAlloyKG\Docs")

TOPIC = "oxidation of niobium alloys (especially Nb-Ti-Al systems and the role of each alloying element)"

# Where an abstract ends: next section heading or keywords line
END_MARKERS = r"(keywords?\s*[:\-–—]|1\.?\s+introduction|\bintroduction\b)"


# ---------------------------------------------------------------- Step 1: read in files
def read_first_pages(path: Path, max_pages: int = 2) -> str:
    """Return text from the first few pages of a PDF (or the whole file if .txt/.md)."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        pages = reader.pages[:max_pages]
        return "\n".join((p.extract_text() or "") for p in pages)
    return path.read_text(errors="ignore")


# ------------------------------------------- Step 2: trim to only abstract, none -> eliminate
def extract_abstract(text: str):
    """Return the abstract text, or None if the paper has no abstract."""
    start = re.search(r"\babstract\b\s*[:.\-–—]?", text, flags=re.IGNORECASE)
    if not start:
        return None
    rest = text[start.end():]
    end = re.search(END_MARKERS, rest, flags=re.IGNORECASE)
    abstract = rest[: end.start()] if end else rest[:2500]  # fall back to a fixed window
    abstract = re.sub(r"\s+", " ", abstract).strip()
    return abstract if len(abstract) > 100 else None  # too short = probably a false match


def load_abstracts(folder: Path) -> dict:
    abstracts = {}
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".pdf", ".txt", ".md"})
    for f in files:
        try:
            abstract = extract_abstract(read_first_pages(f))
        except Exception as e:
            print(f"  [skip] {f.name}: could not read ({e})")
            continue
        if abstract is None:
            print(f"  [skip] {f.name}: no abstract found")
            continue
        abstracts[f.name] = abstract
    return abstracts


# ---------------------------------------------------------------- Step 3: call model
def score_abstract(name: str, abstract: str) -> dict:
    """Ask the model to rate how relevant one abstract is (0-10) with a short reason."""
    prompt = f"""You are helping a researcher choose which paper to hand-annotate first.
Her project: build a causal knowledge graph from papers about {TOPIC}.
The best paper to annotate explicitly discusses how specific alloying elements (Ti, Al, Cr, Hf, etc.)
affect oxidation behavior or mechanisms in niobium-based alloys, and states clear cause-and-effect findings.

Abstract:
{abstract}

Respond ONLY with JSON: {{"score": <integer 0-10>, "reason": "<one sentence>"}}"""
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0, "num_ctx": 4096},
    )
    try:
        data = json.loads(resp["message"]["content"])
        return {"score": float(data.get("score", 0)), "reason": str(data.get("reason", ""))}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"score": 0.0, "reason": "model returned unparseable output"}


def explain_choice(best: str, abstracts: dict, ranked: list) -> str:
    """Ask the model to explain why the top paper is the best one to annotate."""
    runners_up = "\n".join(f"- {n}" for n, _ in ranked[1:4])
    prompt = f"""A researcher is building a causal knowledge graph on {TOPIC}.
Out of {len(abstracts)} papers, this one scored highest as the best to hand-annotate first.

Paper: {best}
Abstract: {abstracts[best]}

Runner-up papers: 
{runners_up or "(none)"}

Explain in 3-5 sentences why this is the best paper for her to annotate. Focus on topic fit
(niobium alloy oxidation), explicit alloying-element effects, and how useful it would be for
extracting cause-effect relationships. Use only information from the abstract."""
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2, "num_ctx": 4096},
    )
    return resp["message"]["content"].strip()



def main():
    if not PAPERS_DIR.is_dir():
        sys.exit(f"Folder not found: {PAPERS_DIR}")

    print(f"Reading papers from {PAPERS_DIR} ...")
    abstracts = load_abstracts(PAPERS_DIR)
    if not abstracts:
        sys.exit("No abstracts found.")
    print(f"Found {len(abstracts)} papers with abstracts.\n")


    print(f"Scoring with {MODEL} ...")
    scores = {}
    for i, (name, abstract) in enumerate(abstracts.items(), 1):
        scores[name] = score_abstract(name, abstract)
        print(f"  {i}/{len(abstracts)}  {scores[name]['score']:>4.0f}  {name}")

    ranked = sorted(abstracts.items(), key=lambda kv: scores[kv[0]]["score"], reverse=True)
    best = ranked[0][0]

    rationale = explain_choice(best, abstracts, ranked)

    # ------------------------------------------------ Step 4: print name of paper and abstract
    print("\n" + "=" * 80)
    print(f"BEST PAPER TO ANNOTATE: {best}")
    print("=" * 80)
    print(f"\nABSTRACT:\n{abstracts[best]}")
    print(f"\nWHY THIS ONE ({MODEL}):\n{rationale}")

    print("\nTop 5 overall:")
    for name, _ in ranked[:5]:
        print(f"  {scores[name]['score']:>4.0f}  {name} - {scores[name]['reason']}")


if __name__ == "__main__":
    main()