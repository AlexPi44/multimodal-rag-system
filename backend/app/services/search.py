from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi
from typing import List, Dict
import numpy as np
from app.models.document import SearchResult


class SearchService:
    def __init__(
        self,
        qdrant_host: str,
        qdrant_port: int,
        collection_name: str,
        embedding_service,
    ):
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.embedding_service = embedding_service

        # Create collection if not exists
        try:
            self.qdrant.get_collection(collection_name)
        except Exception:
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_service.dimension,
                    distance=Distance.COSINE
                )
            )

        # BM25 index (in-memory for simplicity)
        self.bm25_corpus = []
        self.bm25_index = None

    async def index_chunks(self, chunks: List[str], chunk_ids: List[str], user_id: str, document_id: str, metadata: Dict):
        """Index document chunks in vector DB and BM25"""
        embeddings = self.embedding_service.embed_texts(chunks)

        points = [
            PointStruct(
                id=chunk_id,
                vector=embedding.tolist(),
                payload={
                    'content': chunk,
                    'user_id': user_id,
                    'document_id': document_id,
                    **metadata
                }
            )
            for chunk_id, chunk, embedding in zip(chunk_ids, chunks, embeddings)
        ]

        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points
        )

        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        self.bm25_corpus.extend(list(zip(chunk_ids, tokenized_chunks, chunks)))
        self._rebuild_bm25()

    def _rebuild_bm25(self):
        """Rebuild BM25 index"""
        if self.bm25_corpus:
            self.bm25_index = BM25Okapi([c[1] for c in self.bm25_corpus])

    async def hybrid_search(self, query: str, user_id: str, top_k: int = 20, alpha: float = 0.5) -> List[SearchResult]:
        """Perform hybrid search combining vector and BM25"""
        query_embedding = self.embedding_service.embed_query(query)
        vector_results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            query_filter={
                "must": [{"key": "user_id", "match": {"value": user_id}}]
            },
            limit=top_k
        )

        bm25_results = []
        if self.bm25_index:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25_index.get_scores(tokenized_query)

            # naive top-k selection across corpus
            for i, (chunk_id, _, content) in enumerate(self.bm25_corpus):
                bm25_results.append({'chunk_id': chunk_id, 'content': content, 'score': bm25_scores[i]})

        combined = {}

        max_vec_score = max([r.score for r in vector_results]) if vector_results else 1
        for result in vector_results:
            chunk_id = result.id
            normalized_score = (result.score / max_vec_score) if max_vec_score > 0 else 0
            combined[chunk_id] = {
                'vector_score': normalized_score * (1 - alpha),
                'bm25_score': 0,
                'content': result.payload['content'],
                'metadata': result.payload
            }

        max_bm25_score = max([r['score'] for r in bm25_results]) if bm25_results else 1
        for result in bm25_results:
            chunk_id = result['chunk_id']
            normalized_score = (result['score'] / max_bm25_score) if max_bm25_score > 0 else 0
            if chunk_id in combined:
                combined[chunk_id]['bm25_score'] = normalized_score * alpha
            else:
                combined[chunk_id] = {
                    'vector_score': 0,
                    'bm25_score': normalized_score * alpha,
                    'content': result['content'],
                    'metadata': {}
                }

        ranked = sorted(
            combined.items(),
            key=lambda x: x[1]['vector_score'] + x[1]['bm25_score'],
            reverse=True
        )

        results = [
            SearchResult(
                chunk_id=chunk_id,
                content=data['content'],
                score=data['vector_score'] + data['bm25_score'],
                metadata=data['metadata'],
                document_id=data['metadata'].get('document_id', '')
            )
            for chunk_id, data in ranked[:top_k]
        ]

        return results
