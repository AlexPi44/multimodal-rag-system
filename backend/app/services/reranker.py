from typing import List
from sentence_transformers import CrossEncoder
from app.models.document import SearchResult


class RerankerService:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: List[SearchResult], top_k: int = 5) -> List[SearchResult]:
        """Rerank search results using cross-encoder"""
        if not results:
            return []

        pairs = [[query, result.content] for result in results]
        scores = self.model.predict(pairs)

        for result, score in zip(results, scores):
            result.score = float(score)

        reranked = sorted(results, key=lambda x: x.score, reverse=True)
        return reranked[:top_k]
