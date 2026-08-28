import os
import pickle
import re
import json
import numpy as np
import faiss
from flask import Flask, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
DOC_STORE_PATH = os.path.join(INDEX_DIR, "doc_store.pkl")
FAISS_STORE_PATH = os.path.join(INDEX_DIR, "faiss.index")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_PRIORITY_CHUNK_LEN = 80

app = Flask(__name__)
CORS(app)

print("Loading embedding model:", EMBED_MODEL)
embedder = SentenceTransformer(EMBED_MODEL)
print("Embedding model loaded successfully.")


def ingest_structured_law(text, filename):
    sections = re.split(r"=+\n", text)
    chunks = []
    for sec in sections:
        sec = sec.strip()
        if len(sec) < 200:
            continue
        law_title = re.search(r"LAW TITLE:\s*(.*)", sec)
        section_bns = re.search(r"SECTION \(BNS\):\s*(.*)", sec)
        section_ipc = re.search(r"SECTION \(IPC\):\s*(.*)", sec)
        category = re.search(r"CATEGORY:\s*(.*)", sec)
        keywords = re.search(r"KEYWORDS:\s*(.*)", sec)
        chunks.append({
            "text": sec,
            "metadata": {
                "type": "structured_law",
                "law_title": law_title.group(1).strip() if law_title else "Unknown",
                "section_bns": section_bns.group(1).strip() if section_bns else "Unknown",
                "section_ipc": section_ipc.group(1).strip() if section_ipc else "Unknown",
                "category": category.group(1).strip() if category else "Unknown",
                "keywords": keywords.group(1).strip() if keywords else "",
                "source_file": filename,
            },
        })
    return chunks


def ingest_faq_json(text, filename):
    chunks = []
    try:
        data = json.loads(text)
        for item in data:
            combined_text = f"Question: {item.get('question')}\n\nAnswer:\n{item.get('answer')}"
            chunks.append({
                "text": combined_text.strip(),
                "metadata": {
                    "type": "faq",
                    "law": item.get("law"),
                    "category": item.get("category"),
                    "keywords": item.get("keyword"),
                    "source_file": filename,
                },
            })
    except Exception as exc:
        print("FAQ JSON parsing error in", filename, ":", exc)
    return chunks


def ingest_priority_text(text, filename):
    major_sections = re.split(r"=+\n", text)
    chunks = []
    for section in major_sections:
        section = section.strip()
        if not section:
            continue
        sub_blocks = re.split(r"-{3,}\n", section)
        pending = ""
        for block in sub_blocks:
            block = block.strip()
            if not block:
                continue
            combined = (pending + "\n\n" + block).strip() if pending else block
            if len(combined) < MIN_PRIORITY_CHUNK_LEN:
                pending = combined
                continue
            chunks.append({
                "text": combined,
                "metadata": {"type": "priority_knowledge", "source_file": filename},
            })
            pending = ""
        if pending and len(pending) >= 40:
            chunks.append({
                "text": pending,
                "metadata": {"type": "priority_knowledge", "source_file": filename},
            })
    return chunks


def ingest_all_files():
    print("Starting structured ingestion...")
    if not os.path.exists(DATA_DIR):
        return {"error": "DATA folder not found!"}

    all_chunks = []
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(DATA_DIR, filename)
        print(f"Processing file: {filename}")
        try:
            with open(path, "r", encoding="utf-8") as file:
                text = file.read().strip()
            if text.startswith("["):
                chunks = ingest_faq_json(text, filename)
            elif "LAW TITLE:" in text:
                chunks = ingest_structured_law(text, filename)
            else:
                chunks = ingest_priority_text(text, filename)
            all_chunks.extend(chunks)
        except Exception as exc:
            print("Error processing file:", filename, exc)

    if not all_chunks:
        return {"error": "No chunks created!"}

    os.makedirs(INDEX_DIR, exist_ok=True)
    print(f"Total chunks created: {len(all_chunks)}")
    with open(DOC_STORE_PATH, "wb") as file:
        pickle.dump(all_chunks, file)
    print("doc_store.pkl saved.")

    print("Creating embeddings...")
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    # Unit-normalized vectors make inner product equal cosine similarity.
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_STORE_PATH)

    print("FAISS index built with:", index.ntotal, "vectors.")
    print("faiss.index saved successfully.")
    return {"message": "Structured ingestion completed successfully!", "chunks": len(all_chunks)}


@app.route("/ingest", methods=["POST"])
def ingest_route():
    return jsonify(ingest_all_files())


if __name__ == "__main__":
    print("Ingestion backend running at http://127.0.0.1:9000")
    app.run(host="127.0.0.1", port=9000, debug=True)
