# Multimodal Agentic RAG System

This repository contains a production-oriented implementation scaffold for a Multimodal Retrieval-Augmented Generation (RAG) system. The implementation is based on the provided design document and includes a FastAPI backend, a Reflex frontend scaffold, and orchestration via Docker Compose. The system supports multimodal ingestion (PDF, DOCX, images, audio, code, text), hybrid retrieval (vector + BM25), reranking, graph storage (Neo4j), and conversational memory (Redis).

This README outlines how the project is organized, how to configure and run it locally, and what to change for production.

## Repository layout

```
multimodal-rag/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── main.py
│   │   ├── models/
│   │   │   └── document.py
│   │   └── services/
│   │       ├── ingestion.py
│   │       ├── embedding.py
│   │       ├── search.py
│   │       ├── reranker.py
│   │       ├── memory.py
│   │       ├── generation.py
│   │       └── graph.py
│   └── requirements.txt
├── frontend/
│   ├── rxconfig.py
│   ├── frontend/
│   │   └── frontend.py
│   └── requirements.txt
├── docker-compose.yml
├── .env.example
└── README.md
```

## What I implemented from the design (high level)

- Core FastAPI backend scaffold with endpoints to upload documents and chat (/api/v1/documents/upload and /api/v1/chat).
- Document model and chunk/search result types in `backend/app/models/document.py`.
- Ingestion service with file-type-aware extraction (PDF, DOCX, text, code, images, audio) in `backend/app/services/ingestion.py` (placeholders for advanced vision/audio models).
- Embedding service using `sentence-transformers`.
- Hybrid search service combining Qdrant vector search and an in-memory BM25 index.
- Cross-encoder reranker service.
- Redis-backed conversational memory service.
- Neo4j graph service for document/entity relationships.
- LLM generation wrapper with placeholders for OpenAI/Anthropic.
- Docker Compose orchestration with Qdrant, Neo4j, and Redis services.
- Minimal Reflex frontend scaffold (placeholder UI) and Dockerfile.

Notes and assumptions
- The implementation follows the instructions document. Where the document contained placeholders or pseudo-code, I adapted them into runnable Python modules and added minimal, safe placeholders (for example, image/audio transcription and LLM client calls) so the project is scaffolded and can be iterated on.
- Some code paths assume third-party services (Qdrant, Neo4j, Redis, LLM APIs). Those services must be available and configured via environment variables before full functionality works.

## Quick start (local, development)

1. Copy environment variables:

```bash
cp .env.example .env
# Edit .env to add your API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY) and passwords (NEO4J_PASSWORD)
```

2. Build and start the stack with Docker Compose:

```bash
docker-compose up --build -d
```

3. Access services:

- Frontend (placeholder): http://localhost:3000
- Backend API docs: http://localhost:8000/docs
- Qdrant dashboard: http://localhost:6333
- Neo4j browser: http://localhost:7474

## Config and important files

- `backend/app/config.py` — central settings via pydantic-settings, reads `.env`.
- `backend/app/main.py` — FastAPI application and endpoints.
- `backend/app/services/` — service modules (ingestion, embedding, search, reranker, memory, generation, graph).
- `backend/requirements.txt` — Python dependencies for the backend.
- `frontend/` — Reflex app scaffold and its requirements.
- `docker-compose.yml` — local orchestration for backend and infrastructure (Qdrant, Neo4j, Redis).
- `.env.example` — example environment variables; copy to `.env` and populate before running.

## What you must change before production

1. SECRET_KEY: change `SECRET_KEY` in the `.env` to a long random value.
2. NEO4J_PASSWORD: update the Neo4j password in `.env` and in `docker-compose.yml` if you keep Neo4j in compose.
3. API keys: populate `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.
4. Storage: mount persistent volumes for uploads and DB storage rather than local folders.
5. HTTPS/TLS: place a reverse proxy (e.g., Traefik or Nginx) with TLS in front of the services.
6. Secrets management: do not store secrets in environment files in production; use a secret manager.
7. Scale services: use Kubernetes or managed services for Qdrant/Neo4j/Redis for production scale.

## How to use the backend API (examples)

- Upload a document (curl):

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload?user_id=example_user" \
  -F "file=@/path/to/file.pdf"
```

- Chat endpoint (curl):

```bash
curl -X POST "http://localhost:8000/api/v1/chat?query=What+is+in+my+docs&user_id=example_user"
```

## Running tests

There are placeholder tests in `backend/tests/`. To run tests locally (inside a Python environment where `backend/requirements.txt` is installed):

```bash
cd backend
pytest -q
```

## Next steps / recommendations

1. Implement robust auth (JWT), user isolation for document stores, and access control.
2. Replace placeholder vision/audio/LLM calls with production-grade integrations (e.g., GPT-4V, Whisper, Claude 3.x or local models via Ollama).
3. Add CI to run tests and linters.
4. Add observability: Prometheus + Grafana, and structured logs.
5. Add database migrations and persistent storage for metadata (Postgres + SQLAlchemy + Alembic).

