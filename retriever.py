import os, re, json, glob, uuid, math
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import tiktoken                
import yaml                     
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions                                                                                                      

ENCODER     = tiktoken.get_encoding("cl100k_base")
CHUNK_TOKENS = 350

def n_tokens(text: str) -> int:
    return len(ENCODER.encode(text))

def chunk_body(body: str, max_tokens: int = CHUNK_TOKENS) -> list[str]:
    """Greedy chunking at sentence boundaries ≈ max_tokens."""
    sentences = re.split(r'(?<=[.!?])\s+', body)
    chunks = []
    current_sentences = []
    current_sentences_tokens = 0
    for s in sentences:
        t = n_tokens(s)
        if current_sentences_tokens + t > max_tokens and current_sentences:
            chunks.append(" ".join(current_sentences))
            current_sentences, current_sentences_tokens = [], 0
        current_sentences.append(s)
        current_sentences_tokens += t
    if current_sentences:
        chunks.append(" ".join(current_sentences))
    return chunks or [""]       # handle empty body edge‑case

def read_frontmatter_md(path):
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("---"):
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.S)
        meta = yaml.safe_load(m.group(1))
        body = m.group(2).strip()
    else:
        # if there is no header
        h1 = re.search(r"^#\s+(.*)", text, re.M)
        title = h1.group(1).strip() if h1 else path.stem
        meta = {
            "id"      : f"{path.stem}_v1",
            "title"   : title,
            "category": "unspecified",
            "tags"    : [],
            "updated" : datetime.utcnow().date().isoformat()
        }
        body = text.strip()
    return {"meta": meta, "body": body}

def load_md_dir(folder):
    docs = []
    for p in Path(folder).glob("*.md"):
        fm = read_frontmatter_md(p)
        docs.append({
            "id"      : fm["meta"]["id"],
            "meta"    : fm["meta"],
            "body"    : fm["body"],
            "source"  : str(p)
        })
    return docs

def load_installation_guides(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))["software_guides"]
    docs = []
    for app, info in data.items():
        doc_id = info.get("id") or f"{app.lower()}_install_v1"
        body   = (
            f"{info['title']}\n\n"
            f"Steps:\n" + "\n".join(f"- {s}" for s in info["steps"]) + "\n\n"
            f"Common issues:\n" +
            "\n".join(f"- {i['issue']}: {i['solution']}" for i in info["common_issues"]) +
            f"\n\nSupport Contact: {info['support_contact']}"
        )
        docs.append({
            "id": doc_id,
            "meta": {
                "title"   : info["title"],
                "category": "installation_guide",
                "tags"    : [app, "installation"],
                "updated" : datetime.utcnow().date().isoformat()
            },
            "body"   : body,
            "source" : path
        })
    return docs
# C:\Users\danie\Downloads\AI_ML\knowledge\company_it_policies.md
def load_categories(path):
    cats = json.loads(Path(path).read_text(encoding="utf-8"))["categories"]
    docs = []
    for key, meta in cats.items():
        docs.append({
            "id": f"{key}_category_v1",
            "meta": {
                "title"   : key,
                "category": "category_definition",
                "tags"    : ["taxonomy"],
                "updated" : datetime.utcnow().date().isoformat(),
                **meta     # description, typical_resolution_time, escalation_triggers
            },
            "body": meta["description"],
            "source": path
        })
    return docs


def load_troubleshooting(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    # Handle optional top‑level wrapper
    records = data.get("troubleshooting_steps", data)

    docs = []
    for key, rec in records.items():
        title = key.replace("_", " ").title()           # "password_reset" -> "Password Reset"
        category            = rec.get("category", "General")
        steps               = rec.get("steps", [])
        escalation_trigger  = rec.get("escalation_trigger", "")
        escalation_contact  = rec.get("escalation_contact", "N/A")

        body = (
            f"Issue Key: {key}\n"
            f"Category: {category}\n"
            f"Escalation Trigger: {escalation_trigger}\n\n"
            "Resolution Steps:\n" +
            "\n".join(f"- {s}" for s in steps) +
            f"\n\nEscalation Contact: {escalation_contact}"
        )

        doc_id = f"{key}_troubleshoot_v1"
        docs.append({
            "id": doc_id,
            "meta": {
                "title"   : title,
                "category": "troubleshooting",
                "tags"    : [category],
                "updated" : datetime.utcnow().date().isoformat()
            },
            "body"   : body,
            "source" : path
        })

    return docs


docs = []

# 2‑A  Markdown sources that already contain YAML front‑matter
for p in [r"C:\Users\danie\Downloads\AI_ML\knowledge\company_it_policies.md", r"C:\Users\danie\Downloads\AI_ML\knowledge\knowledge_base.md"]:
    fm = read_frontmatter_md(Path(p))
    docs.append({
        "id"     : fm["meta"]["id"],
        "meta"   : fm["meta"],
        "body"   : fm["body"],
        "source" : p
    })

# 2‑B  JSON sources
docs += load_installation_guides(r"C:\Users\danie\Downloads\AI_ML\knowledge\installation_guides.json")
docs += load_categories(r"C:\Users\danie\Downloads\AI_ML\knowledge\categories.json")
docs += load_troubleshooting(r"C:\Users\danie\Downloads\AI_ML\knowledge\troubleshooting_database.json")

def sanitize_meta(meta: dict) -> dict:
    """Remove any fields whose value is not a primitive type."""
    allowed_types = (str, int, float, bool, type(None))
    return {k: v for k, v in meta.items() if isinstance(v, allowed_types)}
# 2‑C  Chunk each doc’s body to ≈ 350 tokens

def build_vector():
    chunks, texts, metadatas, ids = [], [], [], []
    for doc in docs:
        for i, chunk_text in enumerate(chunk_body(doc["body"])):
            chunk_id = f"{doc['id']}#{i}"
            meta_raw = {**doc["meta"], "parent_id": doc["id"], "source": doc["source"]}
            meta = sanitize_meta(meta_raw)
            chunks.append({"id": chunk_id, "text": chunk_text, "meta": meta})
            ids.append(chunk_id)
            texts.append(chunk_text)
            metadatas.append(meta)

    print(f"Loaded {len(docs)} docs → {len(chunks)} chunks")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

    DB_DIR = "chroma_store"                     

    chroma_client = chromadb.PersistentClient(
        path=DB_DIR,                          
        settings=Settings(anonymized_telemetry=False) 
    )
    collection = chroma_client.get_or_create_collection(name="helpdesk_knowledge")

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )
    return model, collection

print("✅ Embeddings upserted & collection persisted.")

if __name__ == "__main__":
    # Run this script to build the vector store
    # It will create a chroma_store folder with the vector database
    build_vector()
    print("Vector store built successfully.")