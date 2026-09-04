# Corrective RAG (CRAG) Assistant

A retrieval-augmented generation system that grades its own retrieved context, falls back to live web search when local documents are insufficient, and refines context down to only the sentences that matter before generating an answer — implemented as a LangGraph state machine served over FastAPI.

## How it works

```
question
   │
   ▼
retrieve_node  ──► doc_eval_node ──► CORRECT ──► refine ──► generate ──► answer
                        │
                        ├──► INCORRECT ──► rewrite_query_node ──► web_search_node ──► refine
                        │
                        └──► AMBIGUOUS ──► ambiguous_node
```

1. **Retrieve** — top-k similarity search against a local FAISS index built from PDF documents.
2. **Grade** — an LLM scores each retrieved chunk's relevance (0.0–1.0). Verdict is `CORRECT` (at least one strong chunk), `INCORRECT` (all chunks weak), or `AMBIGUOUS` (mixed signal).
3. **Correct** → refine local chunks down to individually relevant sentences, then answer.
4. **Incorrect** → rewrite the question into a search query, pull results from Tavily, refine, then answer.
5. **Refine** — every candidate block of text is split into sentences and each sentence is independently kept/dropped by an LLM filter, so the final context fed to generation has no irrelevant filler.

## Stack

- **Orchestration:** LangGraph
- **LLM / embeddings:** Ollama (`llama3.1:8b`, `nomic-embed-text`) — fully local
- **Vector store:** FAISS
- **Web fallback:** Tavily Search API
- **API:** FastAPI, with both a standard JSON endpoint and a streaming (SSE) endpoint
- **Deployment:** Docker Compose (app + Ollama)

## Setup

### Local (without Docker)

```bash
pip install -r requirements.txt
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

Place source PDFs in `./documents/` (expects `book1.pdf`, `book2.pdf`, `book3.pdf` — adjust `retrieval.py` for your own files).

Create a `.env`:
```
TAVILY_API_KEY=your_key_here
```

Run:
```bash
uvicorn app.main:app --reload
```

### Docker

```bash
docker compose up --build
```

First run only — pull models into the Ollama container:
```bash
docker exec -it <ollama_container_name> ollama pull llama3.1:8b
docker exec -it <ollama_container_name> ollama pull nomic-embed-text
```

## API

**Health check**
```bash
curl http://localhost:8000/health
```

**Ask a question**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "what is a session in requests?"}'
```

**Stream the answer token-by-token**
```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "what is a session in requests?"}'
```

## Project structure

```
app/
├── main.py         # FastAPI app + endpoints
├── graph.py         # LangGraph state machine wiring
├── nodes.py         # Retrieval, grading, refinement, web search, generation nodes
├── state.py         # Shared LangGraph state schema
├── config.py        # LLM, embedding, and threshold config
└── retrieval.py      # PDF loading, chunking, FAISS index construction
documents/            # Source PDFs (not committed)
Dockerfile
docker-compose.yml
requirements.txt
```

## Known limitations / roadmap

- Doc grading and sentence filtering currently run as sequential LLM calls per item — batching would meaningfully cut latency.
- FAISS index rebuilds from source PDFs on every startup; not yet persisted to disk.
- No automated eval harness yet (hit rate / hallucination rate / latency vs. plain RAG baseline).
- No request-level error handling for Ollama or Tavily failures.
