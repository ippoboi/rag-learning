## RAG Learning – FastAPI + Chroma + Frontend

This project explores building a Retrieval-Augmented Generation (RAG) system
on top of the FastAPI documentation, and then exposing it through a small
FastAPI backend plus a minimal HTML/JavaScript frontend.

You can:

- index parts of the FastAPI docs into Chroma
- query them via a Groq-powered RAG pipeline
- use a simple web UI to ask questions and see answers with sources

---

## Project layout

All of the app-specific code lives inside this `rag-learning/` directory:

- **`fast-api-rag.ipynb`**

  - Prototyping notebook where the RAG pipeline was first built:
    - loads the FastAPI docs from the `fastapi/` repo
    - enriches documents with mkdocs navigation metadata
    - builds two Chroma collections:
      - `fastapi_dense_docs` under `chroma_fastapi/dense`
      - `fastapi_faq_docs` under `chroma_fastapi/faq`
    - defines the routing, multi-query reformulation, and final RAG chain

- **`backend/`**

  - `__init__.py` – backend package marker.
  - `rag_service.py` – production-ready RAG service that:
    - loads the persisted Chroma vector stores under `chroma_fastapi/`
    - recreates the same routing + multi-query retrieval logic as the notebook
    - defines `answer_question(question: str)` which:
      - generates multiple query variants
      - routes each variant to the correct corpus (`dense_docs`, `faq_docs`, or both)
      - deduplicates retrieved documents
      - calls the Groq model to generate a final answer
      - returns the answer text and a list of unique source metadata dicts
  - `app.py` – FastAPI application that:
    - exposes `POST /ask` – accepts `{"question": "..."}` and returns:
      - `answer: str`
      - `sources: [{ source, corpus, section, category_path, ... }]`
    - exposes `GET /health` – simple health check
    - mounts the `frontend/` directory as static files at `/`

- **`frontend/`**
  - `index.html` – minimal UI:
    - textarea for the question
    - “Ask” button
    - sections for the generated answer and the list of sources
  - `style.css` – small, modern dark theme for the UI
  - `main.js` – calls the backend API and renders the response:
    - sends `POST /ask` with `{ question }`
    - handles loading and error states
    - pretty-prints the answer (`white-space: pre-wrap`)
    - renders source paths and basic metadata

---

## Prerequisites

Before using the API or UI, you need to:

1. **Generate the Chroma vector stores** by running the notebook `fast-api-rag.ipynb`:

   - Execute at least the cells that build and persist the vector stores
   - This will create `chroma_fastapi/dense/` and `chroma_fastapi/faq/` directories
   - Note: These directories are gitignored (they contain generated artifacts)

2. **Set up environment variables**:

   - Create a `.env` file with your `GROQ_API_KEY`
   - Optionally add LangSmith variables (`LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, etc.)

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the RAG backend

From the `rag-learning/` directory, activate your virtual environment and start the backend:

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate

uvicorn backend.app:app --reload
```

By default this will start the server at `http://127.0.0.1:8000`.

Environment variables such as `GROQ_API_KEY`, `LANGSMITH_*`, etc. are loaded
via `python-dotenv` in `backend/rag_service.py`, mirroring the notebook setup.

---

## Using the web UI

Once the FastAPI server is running, open:

- `http://127.0.0.1:8000/`

You should see the **FastAPI RAG Assistant** page. Typical flow:

1. Type a question about FastAPI docs, for example:
   - “What is dependency injection in FastAPI and how do I use it?”
2. Click **Ask** (or press `Cmd+Enter` / `Ctrl+Enter`).
3. The UI:
   - sends a `POST /ask` request to the backend
   - displays the generated answer
   - shows a list of source documents with basic metadata:
     - absolute path to the markdown file
     - corpus (`dense_docs` or `faq_docs`)
     - navigation metadata (`category_path`, `top_level_category`, etc.)

You can also open `frontend/index.html` directly from disk. In that case,
`main.js` falls back to calling `http://localhost:8000/ask`, so make sure
the backend is running on that host and port.

---

## Calling the API directly

You don’t need the frontend to use the RAG service; you can call the API
directly, for example with `curl`:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I enable CORS in FastAPI?"}'
```

Example JSON response shape:

```json
{
  "answer": "...generated answer text...",
  "sources": [
    {
      "source": "/abs/path/to/fastapi/docs/en/docs/tutorial/cors.md",
      "corpus": "dense_docs",
      "section": null,
      "category_path": "Learn > Tutorial - User Guide",
      "top_level_category": "Learn",
      "subcategory": "Tutorial - User Guide"
    }
  ]
}
```

---

## How this relates to the upstream FastAPI repo

In the parent directory of this app you have a full clone of the FastAPI
project. The notebook and RAG pipeline rely on that checkout to load the
documentation markdown files and `mkdocs.yml`.

This `rag-learning/` app is intentionally kept separate:

- it **reads** the FastAPI docs from the sibling `fastapi/` directory
- it **stores** vector indexes locally in `chroma_fastapi/`
- it **serves** a small, self-contained RAG API and frontend for exploration

You can iterate on the RAG logic either in the notebook or in
`backend/rag_service.py`, and immediately test changes through the `/ask`
endpoint or the web UI.
