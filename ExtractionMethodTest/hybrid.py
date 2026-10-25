#READ ME
#to run    python ExtractionMethodTest\hybrid.py ExecSummary.pdf --prompt-file ExtractionMethodTest\hybridPrompt.txt

import re
import sys
import json
import difflib
import argparse
from pathlib import Path

import ollama
from pypdf import PdfReader

MODEL = "qwen3-vl:32b"          # change to whatever you're testing
CHUNK_CHARS = 1000            # smaller chunks -> more triples per document
NUM_CTX = 8192                # the prompt is long; Ollama's default ctx can silently truncate it
NUM_PREDICT = 4096            # the JSON output is much larger than plain tuples

# --- ontology-extension guardrails -------------------------------------------
MAX_NEW_PER_CHUNK = 3         # matches the limit stated in the prompt
SIMILARITY_THRESHOLD = 0.85   # proposed name this close to an existing one -> reuse existing
MAX_PREDICATE_LEN = 40

ENTITY_TYPES = [
    "Study", "System", "Material", "Process", "Property",
    "Performance", "Measurement", "Structure", "Characterization",
]

BASE_PREDICATES = [
    "has_system", "has_material", "has_process", "has_property",
    "has_performance", "has_structure", "has_characterization",
    "has_measurement", "acts_on", "determines",
]

# --- structured-output schema (matches prompt.txt) ---------------------------
PROVENANCE = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "source_id": {"type": "string"},
            "page": {"type": ["integer", "null"]},
            "text_span": {"type": "string"},
        },
        "required": ["text_span"],
    },
}

SCHEMA = {
    "type": "object",
    "properties": {
        "study": {
            "type": "object",
            "properties": {
                "title": {"type": ["string", "null"]},
                "doi": {"type": ["string", "null"]},
                "year": {"type": ["integer", "null"]},
            },
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string", "enum": ENTITY_TYPES},
                    "name": {"type": "string"},
                    "attributes": {"type": "object"},
                    "provenance": PROVENANCE,
                },
                "required": ["id", "type", "name", "provenance"],
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    # Can't be an enum: the predicate list grows at runtime.
                    "predicate": {"type": "string"},
                    "predicate_status": {"type": "string", "enum": ["ontology", "proposed"]},
                    "target": {"type": "string"},
                    "provenance": PROVENANCE,
                },
                "required": ["source", "predicate", "predicate_status", "target", "provenance"],
            },
        },
        "proposed_predicates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "definition": {"type": "string"},
                    "source_type": {"type": "string", "enum": ENTITY_TYPES},
                    "target_type": {"type": "string", "enum": ENTITY_TYPES},
                    "closest_existing_predicate": {"type": "string"},
                    "why_existing_do_not_fit": {"type": "string"},
                    "provenance": PROVENANCE,
                },
                "required": [
                    "name", "definition", "source_type", "target_type",
                    "closest_existing_predicate", "why_existing_do_not_fit",
                ],
            },
        },
    },
    "required": ["study", "entities", "relationships", "proposed_predicates"],
}


# --- relation registry -------------------------------------------------------
def normalize_predicate(name: str) -> str:
    name = (name or "").strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


class RelationRegistry:
    """Base ontology predicates + learned ones, persisted to JSON so the
    vocabulary carries over between documents/runs."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.relations = {n: {"origin": "base"} for n in BASE_PREDICATES}
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            for name, meta in saved.items():
                self.relations.setdefault(name, meta)

    def save(self):
        self.path.write_text(json.dumps(self.relations, indent=2, ensure_ascii=False), encoding="utf-8")

    def __contains__(self, name):
        return name in self.relations

    def is_base(self, name):
        return self.relations[name]["origin"] == "base"

    def prompt_block(self) -> str:
        lines = []
        for name, meta in self.relations.items():
            if meta["origin"] == "base":
                lines.append(f"- {name}")
            else:
                lines.append(
                    f"- {name}: {meta.get('definition', '')} "
                    f"[{meta.get('source_type', '?')} -> {meta.get('target_type', '?')}]"
                )
        return "\n".join(lines)

    def find_similar(self, name: str):
        best, best_score = None, 0.0
        for existing in self.relations:
            score = difflib.SequenceMatcher(None, name, existing).ratio()
            if score > best_score:
                best, best_score = existing, score
        return best if best_score >= SIMILARITY_THRESHOLD else None

    def add(self, name: str, proposal: dict, source_id: str, page):
        self.relations[name] = {
            "origin": "learned",
            "definition": proposal.get("definition", ""),
            "source_type": proposal.get("source_type"),
            "target_type": proposal.get("target_type"),
            "closest_existing_predicate": proposal.get("closest_existing_predicate"),
            "why_existing_do_not_fit": proposal.get("why_existing_do_not_fit"),
            "first_seen": {"source_id": source_id, "page": page},
            "uses": 0,
        }
        self.save()

    def bump(self, name: str):
        if not self.is_base(name):
            self.relations[name]["uses"] = self.relations[name].get("uses", 0) + 1


# --- PDF handling ------------------------------------------------------------
def extract_pdf_pages(pdf_path: str):
    reader = PdfReader(pdf_path)
    return [(i, (p.extract_text() or "").strip()) for i, p in enumerate(reader.pages, 1)]


def chunk_text(text: str, size: int = CHUNK_CHARS):
    chunk, chunks = "", []
    for para in text.split("\n"):
        if len(chunk) + len(para) > size and chunk:
            chunks.append(chunk)
            chunk = ""
        chunk += para + "\n"
    if chunk.strip():
        chunks.append(chunk)
    return chunks


# --- prompting ---------------------------------------------------------------
def load_prompt_template(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def make_prompt(template: str, text: str, registry: RelationRegistry, source_id: str, page) -> str:
    # str.replace (not str.format) because the prompt is full of literal JSON braces.
    prompt = template.replace("{predicates}", registry.prompt_block())
    prompt = prompt.replace("{source_id}", source_id).replace("{page}", str(page))
    if "{text}" in prompt:
        return prompt.replace("{text}", text)
    return prompt + "\n\nTEXT:\n" + text


def call_model(prompt: str):
    resp = ollama.generate(
    model=MODEL,
    prompt=prompt,
    format=SCHEMA,
    think=False,
    options={"temperature": 0.2, "num_predict": 4096},
    )   
    raw = resp.response
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
        print("Raw output:", repr(raw))
        return None


# --- validation / reconciliation --------------------------------------------
def fill_provenance(item: dict, source_id: str, page):
    for p in item.get("provenance", []) or []:
        if not p.get("source_id"):
            p["source_id"] = source_id
        if p.get("page") is None:
            p["page"] = page


def reconcile(data: dict, registry: RelationRegistry, tag: str, source_id: str, page):
    """Validate one chunk's output. Returns (entities, relationships, accepted_new, rejected).
    Adds legitimately proposed predicates to the registry as a side effect."""
    entities, id_map = [], {}
    for e in data.get("entities", []):
        old = e.get("id")
        if not old or e.get("type") not in ENTITY_TYPES:
            continue
        new = f"{tag}_{old}"        # E1 from different chunks would otherwise collide
        id_map[old] = new
        e["id"] = new
        fill_provenance(e, source_id, page)
        entities.append(e)

    proposals = {}
    for p in data.get("proposed_predicates", []):
        key = normalize_predicate(p.get("name", ""))
        if key:
            proposals[key] = p

    budget = MAX_NEW_PER_CHUNK
    relationships, accepted, rejected = [], [], []

    for r in data.get("relationships", []):
        src, tgt = id_map.get(r.get("source")), id_map.get(r.get("target"))
        if not src or not tgt:
            rejected.append({"relationship": r, "reason": "source/target is not an extracted entity"})
            continue

        pred = normalize_predicate(r.get("predicate", ""))

        if pred in registry:
            pass  # already known (base or learned earlier)
        else:
            similar = registry.find_similar(pred)
            proposal = proposals.get(pred)
            if similar:
                r["remapped_from"] = pred     # near-duplicate of an existing predicate
                pred = similar
            elif r.get("predicate_status") != "proposed" or proposal is None:
                rejected.append({"relationship": r, "reason": "unknown predicate without a proposal"})
                continue
            elif not (proposal.get("definition") and proposal.get("why_existing_do_not_fit")):
                rejected.append({"relationship": r, "reason": "proposal missing definition/justification"})
                continue
            elif len(pred) > MAX_PREDICATE_LEN or any(c.isdigit() for c in pred):
                rejected.append({"relationship": r, "reason": "predicate name not general/reusable"})
                continue
            elif budget <= 0:
                rejected.append({"relationship": r, "reason": "per-chunk new-predicate limit reached"})
                continue
            else:
                fill_provenance(proposal, source_id, page)
                registry.add(pred, proposal, source_id, page)
                accepted.append({"name": pred, **registry.relations[pred]})
                budget -= 1

        registry.bump(pred)
        r.update({
            "source": src,
            "target": tgt,
            "predicate": pred,
            "predicate_status": "ontology" if registry.is_base(pred) else "extension",
        })
        fill_provenance(r, source_id, page)
        relationships.append(r)

    return entities, relationships, accepted, rejected


# --- main --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--prompt-file", default="prompt.txt", help="Prompt template (see prompt.txt).")
    parser.add_argument("--registry", default="relations.json",
                        help="JSON file holding base + learned predicates (created if missing).")
    parser.add_argument("--out", default="extraction.json")
    args = parser.parse_args()

    template = load_prompt_template(args.prompt_file)
    registry = RelationRegistry(args.registry)
    source_id = Path(args.pdf_path).name
    print(f"Using prompt: {args.prompt_file}")
    print(f"Registry: {len(registry.relations)} predicate(s) loaded from {args.registry}")

    pages = extract_pdf_pages(args.pdf_path)
    if not any(t for _, t in pages):
        print("No extractable text found in PDF.")
        sys.exit(1)

    jobs = [(pg, chunk) for pg, text in pages for chunk in chunk_text(text)]
    print(f"Processing {len(jobs)} chunk(s)...")

    study = {"title": None, "doi": None, "year": None}
    all_entities, all_rels, new_predicates, all_rejected = [], [], [], []

    for i, (page, chunk) in enumerate(jobs, 1):
        # registry.prompt_block() is rebuilt every call, so a predicate learned
        # in chunk N is visible to the model from chunk N+1 onward.
        data = call_model(make_prompt(template, chunk, registry, source_id, page))
        if data is None:
            continue

        for k, v in (data.get("study") or {}).items():
            if k in study and study[k] is None and v is not None:
                study[k] = v

        ents, rels, accepted, rejected = reconcile(data, registry, f"c{i}", source_id, page)
        all_entities += ents
        all_rels += rels
        new_predicates += accepted
        all_rejected += rejected

        msg = f"  chunk {i}/{len(jobs)} (p.{page}): {len(ents)} entities, {len(rels)} relationships"
        if accepted:
            msg += f", NEW predicate(s): {', '.join(a['name'] for a in accepted)}"
        print(msg)

    result = {
        "study": study,
        "entities": all_entities,
        "relationships": all_rels,
        "new_predicates": new_predicates,
        "rejected": all_rejected,
    }
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    names = {e["id"]: e["name"] for e in all_entities}
    print(f"\nFound {len(all_rels)} relationship(s), {len(new_predicates)} new predicate(s), "
          f"{len(all_rejected)} rejected. Saved to {args.out}")
    for r in all_rels:
        print((names[r["source"]], r["predicate"], names[r["target"]]))


if __name__ == "__main__":
    main()