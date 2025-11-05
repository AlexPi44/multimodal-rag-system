from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uuid

from app.config import settings
from app.services.embedding import EmbeddingService
from app.services.search import SearchService
from app.services.reranker import RerankerService
from app.services.memory import MemoryService
from app.services.generation import GenerationService, LLMProvider
from app.services.graph import GraphService
from app.services.ingestion import IngestionService

app = FastAPI(title=settings.PROJECT_NAME)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
embedding_service = EmbeddingService(settings.EMBEDDING_MODEL)
search_service = SearchService(
    settings.QDRANT_HOST,
    settings.QDRANT_PORT,
    settings.QDRANT_COLLECTION,
    embedding_service
)
reranker_service = RerankerService()
memory_service = MemoryService(settings.REDIS_HOST, settings.REDIS_PORT)
generation_service = GenerationService(settings.OPENAI_API_KEY, settings.ANTHROPIC_API_KEY)
graph_service = GraphService(settings.NEO4J_URI, settings.NEO4J_USER, settings.NEO4J_PASSWORD)
ingestion_service = IngestionService()


@app.post("/api/v1/documents/upload")
async def upload_document(file: UploadFile = File(...), user_id: str = "default_user"):
    """Upload and process document"""
    try:
        document, chunks = await ingestion_service.process_file(file.file, file.filename, user_id)

        await search_service.index_chunks(chunks, document.chunks, user_id, document.id, document.metadata)

        graph_service.create_document_node(document)

        return {"document_id": document.id, "filename": document.filename, "num_chunks": len(chunks), "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/chat")
async def chat(query: str, session_id: Optional[str] = None, user_id: str = "default_user", top_k_retrieval: int = 20, top_k_rerank: int = 5, provider: LLMProvider = LLMProvider.ANTHROPIC):
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        history = memory_service.get_conversation_history(user_id, session_id)

        search_results = await search_service.hybrid_search(query, user_id, top_k=top_k_retrieval)

        reranked_results = reranker_service.rerank(query, search_results, top_k=top_k_rerank)

        response = await generation_service.generate_response(query, reranked_results, history, provider=provider)

        memory_service.store_conversation(user_id, session_id, {"role": "user", "content": query})
        memory_service.store_conversation(user_id, session_id, {"role": "assistant", "content": response})

        return {
            "response": response,
            "sources": [
                {"content": r.content[:200], "score": r.score, "document_id": r.document_id} for r in reranked_results
            ],
            "session_id": session_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents")
async def list_documents(user_id: str = "default_user"):
    # TODO: implement document listing
    return {"documents": []}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
