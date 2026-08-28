# backend/app.py
"""
RAG backend for your project.

Features:
- Loads FAISS index (tries multiple filenames / formats)
- Loads doc_store.pkl (your text chunks)
- Uses sentence-transformers/all-MiniLM-L6-v2 to embed queries
- Searches FAISS for top-k contexts
- Sends prompt + contexts to LLaMA 3.1 via ollama.chat for generation
- Exposes two endpoints:
    POST /chat            -> { "message": "<user question>" }  returns {"reply": "..."}
    POST /test_retrieval  -> { "message": "<user question>", "top_k": 3 } returns retrieval debug info
"""

import os
import pickle
import json
from typing import List, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import ollama

# ---------- Configuration ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # project/
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")
# possible FAISS filenames you mentioned; order matters (first found will be used)
POSSIBLE_FAISS_FILES = [
    "faiss.index"
]
DOC_STORE_FILE = "doc_store.pkl"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # exact model you used
LLM_MODEL_NAME = "llama3.2:3b"
 # used with ollama.chat; you can append size tag if needed (eg "llama3.1:8b")

DEFAULT_TOP_K = 8

# ---------- Prompt Templates ----------

LEGAL_ASSISTANT_PROMPT = """
You are Shakti – a compassionate and knowledgeable Personal Legal Advocate for women.
Answer strictly using only the provided context. If the answer is not present in the context, respond with 'Information not available in the provided legal documents.' Do not add external knowledge.
Your role is to:
- Help a woman feel heard and supported.
- Clearly explain what legal protection applies.
- Guide her on practical next steps.
- Use only the information provided in the CONTEXT.

Response Structure (follow strictly):

1) Start with 1–2 gentle sentences acknowledging her situation in a calm and supportive way.
   Do NOT over-dramatize. Do NOT be overly emotional.

2) Add a short heading:
   Legal Protection Available

   Under this heading:
   - Clearly mention the relevant law names.
   - Mention IPC or BNS section numbers if available in context.
   - Briefly explain what each law protects her from (in simple language).
   - If multiple laws apply, explain how they work together.

3) Add a short heading:
   What You Can Do

   Under this heading:
   - Give 3–5 practical steps.
   - Keep steps short and clear.
   - Examples: file complaint, approach police, seek protection order, collect evidence, contact legal aid.

4) Add a short heading:
   Immediate Help

   - List available helpline numbers from the context only.
   - Do NOT invent numbers.
   - Do NOT leave placeholders.
   - If no numbers are available in context, say:
     "You can contact your nearest police station or local women’s helpline for immediate assistance."

Important Rules:

- Use only the information given in the CONTEXT.
- Do NOT invent sections, punishments, or procedures.
- Do NOT copy text word-for-word from the context.
- Do NOT mention labels like “LAW TITLE” or “KEYWORDS”.
- Keep response medium-length (2–4 short sections).
- Keep language simple and empowering.
- Avoid heavy legal jargon.
- The woman should feel supported, informed, and not alone.
"""

LEARNING_COMPANION_PROMPT = """
You are a calm, friendly Learning Companion chatbot for students.

Your purpose is to help students understand laws in a simple, reassuring way,
using only the study material provided to you.

How you should respond:
- Answer only what the student asks.
- Use clear, simple, student-friendly language.
- Do not sound like a teacher, examiner, or lawyer.
- Do not give legal advice or suggest actions.
- Explain legal terms gently if needed.
- Keep responses calm, supportive, and non-intimidating.

Rules:
- Use ONLY the information given in the context.
- Do NOT invent laws, sections, punishments, or procedures.
- If the answer is not found in the context, say:
  "I don’t have that information in the study material I’m using."

Tone:
- Warm, calm, and respectful.
- Like a study companion sitting beside the student.

If the student greets casually, respond warmly and invite a question.
"""
EMERGENCY_CONTEXT = """
Emergency Helplines in India:

112 – Police Emergency
181 – Women Helpline
1091 – Women Helpline
1098 – Child Helpline
"""
# ---------- Initialize Flask ----------
app = Flask(__name__)
CORS(app, origins=["http://localhost:8080", "http://127.0.0.1:8080"]) 
# ^^^ This explicitly allows connections from your frontend server.

# ---------- Load embedding model (MiniLM) ----------
print("Loading embedding model:", EMBEDDING_MODEL_NAME)
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)  # will auto-download if needed
print("Embedding model loaded.")

# ---------- Utility: load doc store ----------
def load_doc_store(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"doc store not found at: {path}")
    with open(path, "rb") as f:
        doc_store = pickle.load(f)
    # doc_store should be list-like or dict-like. Normalize to list:
    if isinstance(doc_store, dict):
        # attempt to convert mapping of idx->text to list by sorted keys
        try:
            # if keys are ints 0..N-1
            items = sorted(doc_store.items(), key=lambda x: int(x[0]))
            docs = [v for k, v in items]
        except Exception:
            # fallback: just take values
            docs = list(doc_store.values())
    elif isinstance(doc_store, list):
        docs = doc_store
    else:
        # unknown type, attempt direct use
        docs = list(doc_store)
    return docs

# ---------- Utility: load faiss index (robust) ----------
def try_load_vector_store(directory: str) -> faiss.Index:
    # try each candidate path
    for fname in POSSIBLE_FAISS_FILES:
        path = os.path.join(directory, fname)
        if os.path.exists(path):
            print(f"Attempting to load FAISS index from: {path}")
            # If file looks like a binary index created by faiss.write_index, read_index should work
            try:
                idx = faiss.read_index(path)
                print("Loaded FAISS index via faiss.read_index() from", path)
                return idx
            except Exception as e:
                print(f"faiss.read_index failed for {path}: {e}")

            # If file is a pickled object (maybe index was pickled), try pickle.load
            try:
                with open(path, "rb") as f:
                    obj = pickle.load(f)
                # If the object is a faiss index instance, return it
                if isinstance(obj, faiss.Index):
                    print("Loaded FAISS index from pickle:", path)
                    return obj
                # If obj has an 'index' key or attribute, try to extract
                if isinstance(obj, dict) and "index" in obj and isinstance(obj["index"], faiss.Index):
                    print("Loaded faiss index from dict obj in pickle:", path)
                    return obj["index"]
                # If obj is bytes of serialized index, try faiss.deserialize_index
                if isinstance(obj, (bytes, bytearray)):
                    try:
                        idx = faiss.deserialize_index(np.frombuffer(obj, dtype='uint8'))
                        return idx
                    except Exception:
                        pass

                # otherwise, not a faiss index
                print(f"Pickle at {path} is not a faiss.Index instance; continuing search.")
            except Exception as e:
                print(f"pickle.load failed for {path}: {e}")

    # If none found / loaded
    raise FileNotFoundError("No usable FAISS index file found in directory: " + directory)

# ---------- Load resources at startup ----------
# 1) Doc store
doc_store_path = os.path.join(FAISS_DIR, DOC_STORE_FILE)
print("Loading doc_store from:", doc_store_path)
documents = load_doc_store(doc_store_path)
print(f"Loaded doc_store: {len(documents)} documents/chunks.")

# 2) FAISS index
faiss_index = try_load_vector_store(FAISS_DIR)
print("FAISS index loaded successfully.")

# ---------- Embedding helper ----------
def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Returns numpy array shape (n, dim) dtype float32
    """
    # sentence-transformers returns float32 by default when convert_to_numpy=True
    embs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # ensure float32
    if embs.dtype != np.float32:
        embs = embs.astype("float32")
    return embs

def embed_query(text: str) -> np.ndarray:
    vec = embed_texts([text])[0]

    # Normalize query: normalized vectors + IndexFlatIP = cosine similarity
    vec = np.asarray([vec], dtype="float32")
    faiss.normalize_L2(vec)

    return vec[0]

# ---------- Retrieval helper ----------
def retrieve_top_k(query: str, top_k: int = DEFAULT_TOP_K) -> Tuple[List[str], List[float], List[int]]:
    """
    Returns (retrieved_texts, distances, indices)
    """
    qvec = embed_query(query)
    # FAISS expects shape (n, dim)
    qvec = np.asarray([qvec], dtype="float32")

    distances, indices = faiss_index.search(qvec, top_k)
    distances = distances[0].tolist()
    indices = indices[0].tolist()

    retrieved = []
    for idx in indices:
        if idx == -1:
            retrieved.append("")  # no result

        elif 0 <= idx < len(documents):
            doc = documents[idx]

            # FIX: Extract the text from a dict
            if isinstance(doc, dict) and "text" in doc:
                retrieved.append(doc["text"])
            else:
                retrieved.append(str(doc))

        else:
            retrieved.append("")  # out of range

    return retrieved, distances, indices

# ---------- Mode Detection ----------
def detect_mode_from_query(query: str) -> str:
    q = query.lower().strip()

    companion_keywords = [
        "explain", "explanation", "what is", "meaning", "define",
        "definition", "simple", "in simple words",
        "difference", "compare",
        "example", "examples",
        "understand", "confused",
        "why", "how does",
        "overview", "summary", "notes"
    ]

    for kw in companion_keywords:
        if kw in q:
            return "companion"

    # Default: Legal Assistant
    return "assistant"

# ---------- Prompt builder ----------
def build_prompt(context_chunks: List[str], question: str, mode: str = "assistant") -> str:
    joined_context = "\n\n---\n\n".join(context_chunks)

    # Inject emergency numbers only for serious harm-related queries
    harm_keywords = ["beat", "violence", "harass", "dowry", "threat", "suicide", "abuse"]

    if mode == "assistant" and any(word in question.lower() for word in harm_keywords):
        final_context = EMERGENCY_CONTEXT + "\n\n" + joined_context
    else:
        final_context = joined_context

    # Build prompt after context decision
    if mode == "companion":
        system_prompt = LEARNING_COMPANION_PROMPT
        prompt = f"{system_prompt}\n\nCONTEXT:\n{final_context}\n\nUSER QUESTION:\n{question}"
    else:
        system_prompt = LEGAL_ASSISTANT_PROMPT
        prompt = f"{system_prompt}\n\nCONTEXT:\n{final_context}\n\nUSER QUESTION:\n{question}"

    return prompt

# ---------- LLaMA call via Ollama ----------
def generate_answer_with_llama(prompt: str, llm_model: str = LLM_MODEL_NAME, max_tokens: int = 512) -> str:
    """
    Sends the prompt to the local Ollama LLM and returns the generated text.
    """
    # Use ollama.chat which matches earlier examples
    response = ollama.chat(
    model=llm_model,
    messages=[{"role": "user", "content": prompt}],
    options={
        "num_predict": max_tokens,
        "temperature": 0.0,
        "top_p": 0.7
    }
)
    # response structure: {"message": {"content": "<text>"} , ...}
    try:
        return response["message"]["content"]
    except Exception:
        # fallback to string representation
        return str(response)

# ---------- Flask endpoints ----------
@app.route("/chat", methods=["POST"])
def chat_endpoint():
    print("Received request to /chat")   # DEBUG
    
    payload = request.get_json(force=True)
    user_message = payload.get("message", "").strip()
    top_k = int(payload.get("top_k", DEFAULT_TOP_K))

    print("User message:", user_message)   # DEBUG

    # ---------- Retrieval ----------
    contexts, distances, indices = retrieve_top_k(user_message, top_k=top_k)

    print("Retrieved contexts")   # DEBUG
    print("Distances:", distances)   # DEBUG

    # ---------- Confidence Guard ----------
    # Cosine similarity: higher score means a better match.
# Adjust 0.35 later after testing your questions.
    CONFIDENCE_THRESHOLD = 0.35

    best_cosine_score = max(distances)
    print("Best cosine similarity:", best_cosine_score)  # DEBUG

    if best_cosine_score < CONFIDENCE_THRESHOLD:
        print("Confidence too low. Skipping LLaMA.")  # DEBUG
        return jsonify({
            "reply": "Information not available in the provided legal documents.",
            "retrieved": contexts,
            "cosine_scores": distances,
            "indices": indices
        })

    print("Confidence acceptable. Calling LLaMA...")

    # ---------- Continue if confident ----------
    mode = detect_mode_from_query(user_message)
    prompt = build_prompt(contexts, user_message, mode=mode)

    answer = generate_answer_with_llama(prompt, llm_model=LLM_MODEL_NAME)

    print("LLaMA responded.")  # DEBUG

    return jsonify({
        "reply": answer,
        "retrieved": contexts,
        "cosine_scores": distances,
        "indices": indices
    })


# ---------- Run server ----------
if __name__ == "__main__":
    print("Starting Flask server on http://127.0.0.1:8000")
    app.run(host="127.0.0.1", port=8000, debug=True)