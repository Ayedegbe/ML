
# Imports for file handling, data processing, embeddings, and vector DB
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


# Tokenizer and chunk size for splitting documents
ENCODER     = tiktoken.get_encoding("cl100k_base")
CHUNK_TOKENS = 350

# Count tokens in a string
def n_tokens(text: str) -> int:
    return len(ENCODER.encode(text))

# Split a document body into chunks of ~max_tokens, at sentence boundaries
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

# Read markdown file and extract YAML frontmatter and body
def read_frontmatter_md(path):
    try: 
        text = path.read_text(encoding="utf-8")
        if text.lstrip().startswith("---"):
            m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.S)
            meta = yaml.safe_load(m.group(1))
            body = m.group(2).strip()
        else:
            # If there is no header, use defaults
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
    except Exception as e:
        print(f"Error reading markdown frontmatter from {path}: {e}")
        return {"meta": {}, "body": ""}

# Load all markdown files in a folder as documents
def load_md_dir(folder):
    
    docs = []
    for p in Path(folder).glob("*.md"):
        try:
            fm = read_frontmatter_md(p)
            docs.append({
                "id"      : fm["meta"]["id"],
                "meta"    : fm["meta"],
                "body"    : fm["body"],
                "source"  : str(p)
            })
        except Exception as e:
                print(f"Error loading markdown file {p}: {e}")

    return docs

    # ...existing code...
# Load installation guides from JSON
def load_installation_guides(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))["software_guides"]
    except Exception as e:
        print(f"Error loading installation guides from {path}: {e}")
        return []
    docs = []
    for app, info in data.items():
        try:
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
        except Exception as e:
            print(f"Error processing installation guide for {app}: {e}")
    return docs
    
# C:\Users\danie\Downloads\AI_ML\knowledge\company_it_policies.md
    # ...existing code...
# Load category definitions from JSON
def load_categories(path):
    try:
        cats = json.loads(Path(path).read_text(encoding="utf-8"))["categories"]
    except Exception as e:
        print(f"Error loading categories from {path}: {e}")
        return []
    docs = []
    for key, meta in cats.items():
        try:
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
        except Exception as e:
            print(f"Error processing category {key}: {e}")
    return docs


    # ...existing code...
# Load troubleshooting steps from JSON
def load_troubleshooting(path: str) -> list[dict]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error loading troubleshooting data from {path}: {e}")
        return []
    # Handle optional top‑level wrapper
    records = data.get("troubleshooting_steps", data)

    docs = []
    for key, rec in records.items():
        try:
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
        except Exception as e:
            print(f"Error processing troubleshooting record {key}: {e}")
    return docs


docs = []  # List to hold all loaded documents

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

# Remove any fields whose value is not a primitive type
def sanitize_meta(meta: dict) -> dict:
    allowed_types = (str, int, float, bool, type(None))
    return {k: v for k, v in meta.items() if isinstance(v, allowed_types)}
# 2‑C  Chunk each doc’s body to ≈ 350 tokens

# Build the vector store: chunk docs, embed, and upsert into ChromaDB
def build_vector():
    try:
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

        # Embed all text chunks
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

        DB_DIR = "chroma_store"                     

        # Connect to ChromaDB and upsert embeddings
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
    except Exception as e:
        print(f"Error building vector store: {e}")
        return None, None      

# Print status after upserting embeddings
print("✅ Embeddings upserted & collection persisted.")


# Main entry point: build the vector store if run as a script
if __name__ == "__main__":
    build_vector()
    print("Vector store built successfully.")