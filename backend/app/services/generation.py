from typing import List, Dict, Optional
import openai
import anthropic
from enum import Enum
from app.models.document import SearchResult


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class GenerationService:
    def __init__(self, openai_key: Optional[str] = None, anthropic_key: Optional[str] = None):
        self.openai_key = openai_key
        self.anthropic_key = anthropic_key

        if openai_key:
            openai.api_key = openai_key
        if anthropic_key:
            self.anthropic_client = anthropic.Client(api_key=anthropic_key)

    async def generate_response(self, query: str, context: List[SearchResult], conversation_history: List[Dict], provider: LLMProvider = LLMProvider.ANTHROPIC, model: Optional[str] = None) -> str:
        """Generate response using LLM with retrieved context"""
        context_str = "\n\n".join([f"[Source {i+1}] {result.content}" for i, result in enumerate(context)])
        history_str = "\n".join([f"{msg.get('role')}: {msg.get('content')}" for msg in conversation_history[-5:]])

        system_prompt = """You are a highly knowledgeable AI assistant with access to a comprehensive knowledge base. Use the provided context to answer questions accurately and in-depth. Cite sources when possible. If information is not in the context, say so clearly."""

        user_prompt = f"""Conversation History:\n{history_str}\n\nRetrieved Context:\n{context_str}\n\nUser Question: {query}\n\nPlease provide a detailed, accurate response based on the context above."""

        if provider == LLMProvider.ANTHROPIC and self.anthropic_key:
            response = self.anthropic_client.create(prompt=user_prompt)
            # Depending on anthropic client, adapt; placeholder extraction
            return getattr(response, 'text', str(response))
        elif provider == LLMProvider.OPENAI and self.openai_key:
            resp = openai.ChatCompletion.create(
                model=model or "gpt-4o",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_tokens=1024
            )
            return resp.choices[0].message.content
        else:
            return "LLM provider not configured"
