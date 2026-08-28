# Legal Sakhi

## AI-Powered Legal Awareness Chatbot for Women in India

Legal Sakhi is an AI-powered legal awareness chatbot designed to help women in India understand relevant legal information in a simple and accessible way.

The system combines semantic embeddings, FAISS vector retrieval, and a locally hosted LLaMA model through Ollama. User questions are converted into semantic embeddings, relevant legal information is retrieved from the stored knowledge base, and the retrieved context is used to generate a legal-awareness response.

> **Disclaimer:** Legal Sakhi is an educational and legal-awareness system. It does not provide legal advice, does not replace a qualified lawyer, and should not be used as a substitute for professional legal consultation.

---

## Objectives

- Provide accessible legal awareness through a conversational interface.
- Help users find legal information relevant to their questions.
- Retrieve relevant information from a curated legal knowledge base.
- Present retrieved legal information in a simple and understandable manner.
- Explore the use of NLP, semantic retrieval, and generative AI for legal-awareness applications.
- Evaluate the quality of the retrieval component.

---

## Key Features

- **Natural-Language Questions** – Users can ask legal-awareness questions in everyday language.
- **Semantic Embeddings** – Questions are represented using the `all-MiniLM-L6-v2` Sentence Transformer model.
- **FAISS Vector Retrieval** – FAISS is used to search for semantically relevant stored documents/chunks.
- **Retrieval-Augmented Generation** – Retrieved context is supplied to the language model before generating the response.
- **Local LLM Integration** – LLaMA 3.2 3B is accessed through Ollama for response generation.
- **Legal Knowledge Store** – Retrieved text is stored alongside the FAISS index in the backend.
- **Retrieval Evaluation** – The project includes an evaluation script and test dataset for assessing retrieval performance.
- **Web Interface** – A frontend interface allows users to communicate with the chatbot.

---

## Technology Stack

### Backend

- Python
- Flask
- Flask-CORS
- NumPy
- FAISS
- Pickle

### NLP / AI

- Sentence Transformers
- `all-MiniLM-L6-v2`
- Semantic embeddings
- Vector similarity search
- Retrieval-Augmented Generation (RAG)
- LLaMA 3.2 3B
- Ollama

### Frontend

- HTML
- CSS
- JavaScript

---

## Project Structure

```text
legal-sakhi/
│
├── backend/
│   ├── app.py
│   │
│   ├── embedding/
│   │   ├── ingest.py
│   │   └── run_ingest.py
│   │
│   ├── evaluation/
│   │   ├── rag_evaluation.py
│   │   └── test_dataset_evaluation.json
│   │
│   └── faiss_index/
│       ├── doc_store.pkl
│       └── faiss.index
│
├── frontend/
│   └── index2.html
│
├── requirement.txt
├── .gitignore
└── README.md
```

> The `faiss_index` directory contains the stored FAISS index and document store used by the application.

---

## System Architecture

```text
                    User
                      │
                      ▼
              Web Frontend
                      │
                      ▼
                Flask Backend
                      │
                      ▼
            User Question / Query
                      │
                      ▼
        Sentence Transformer Embedding
             all-MiniLM-L6-v2
                      │
                      ▼
                FAISS Search
                      │
                      ▼
           Relevant Legal Context
                      │
                      ▼
          Prompt + Retrieved Context
                      │
                      ▼
             Ollama / LLaMA 3.2 3B
                      │
                      ▼
             Legal Awareness Response
                      │
                      ▼
                 User Interface
```

---

## How It Works

### 1. User Query

The user enters a legal-awareness question through the web interface.

### 2. Query Embedding

The question is converted into a numerical semantic representation using the Sentence Transformer model:

```text
all-MiniLM-L6-v2
```

### 3. Vector Retrieval

The query embedding is compared against the stored vectors in the FAISS index.

The system retrieves the most relevant document/chunk contexts.

### 4. Context Preparation

The retrieved legal information is combined with the user's question to form the context supplied to the language model.

### 5. Response Generation

The application sends the prompt and retrieved context to LLaMA 3.2 3B through Ollama.

### 6. Legal-Awareness Response

The generated response is returned through the Flask backend and displayed to the user.

---

## Backend API

The Flask backend provides the following endpoints:

### `POST /chat`

Used for normal chatbot interaction.

Example request:

```json
{
  "message": "What legal protection is available against domestic violence?"
}
```

### `POST /test_retrieval`

Used to test the retrieval component with a user question.

Example request:

```json
{
  "message": "What legal protection is available against domestic violence?"
}
```

---

## Embedding and Indexing

The `backend/embedding/` directory contains scripts responsible for preparing the information used for semantic retrieval.

- `ingest.py` – Handles the ingestion/indexing process.
- `run_ingest.py` – Runs the ingestion process.

The resulting retrieval artifacts are stored in:

```text
backend/faiss_index/
```

including:

```text
faiss.index
doc_store.pkl
```

---

## Evaluation

The project contains a retrieval evaluation module:

```text
backend/evaluation/
├── rag_evaluation.py
└── test_dataset_evaluation.json
```

The evaluation component is intended to test the quality of the retrieval process using a predefined test dataset.

---

## Legal Scope

Legal Sakhi focuses on legal awareness relevant to women in India.

The knowledge base may include information related to areas such as:

- Domestic violence
- Dowry-related offences
- Workplace sexual harassment
- Protection of children
- Maternity-related rights
- Relevant criminal-law provisions
- Other legal protections applicable to women

The exact information available to the chatbot depends on the legal resources included in its knowledge base.

---

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd legal-sakhi
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirement.txt
```

### 4. Install and run Ollama

The application uses Ollama to access the configured LLaMA model.

Make sure Ollama is installed and the required model is available locally before starting the Flask application.

### 5. Run the Flask backend

From the project root:

```bash
python backend/app.py
```

The application is configured to run locally through Flask.

---

## Important Notes

- The Sentence Transformer model may be downloaded automatically when it is first loaded, depending on the local environment.
- The FAISS index and document store are required by the current backend configuration.
- The LLaMA model is accessed locally through Ollama.
- Legal information can change because of amendments, new judgments, and changes in interpretation. The system should therefore be treated as an educational legal-awareness tool rather than a source of professional legal advice.

---

## Future Enhancements

- Multilingual support for Indian languages.
- Improved embedding and retrieval models.
- Integration with additional verified legal sources.
- Better citation and source tracking for retrieved information.
- Voice-based interaction.
- WhatsApp / Telegram integration.
- Continuous retrieval evaluation.
- Cloud deployment.
- Improved user interface and accessibility.

---

## Academic Project

**Project:** Legal Sakhi  
**Domain:** Artificial Intelligence / Natural Language Processing / Legal Technology  
**Application:** Legal Awareness for Women in India

Legal Sakhi is developed as an academic project exploring how NLP, semantic retrieval, vector databases, and generative AI can be combined to improve access to legal information.

---

## Disclaimer

Legal Sakhi provides general legal awareness and educational information only. The information provided by the system may not reflect the latest amendments, judicial interpretations, or the specific circumstances of an individual case.

**For legal advice or action on a specific matter, users should consult a qualified legal professional or an appropriate official legal-aid service.**
