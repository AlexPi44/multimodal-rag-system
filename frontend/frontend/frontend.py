import reflex as rx
from typing import List, Optional
import httpx


class ChatState(rx.State):
    messages: List[dict] = []
    current_query: str = ""
    is_loading: bool = False
    session_id: Optional[str] = None
    uploaded_documents: List[dict] = []
    upload_status: str = ""
    search_top_k: int = 20
    rerank_top_k: int = 5
    llm_provider: str = "anthropic"
    backend_url: str = "http://localhost:8000"

    async def send_message(self):
        if not self.current_query.strip():
            return
        self.messages.append({"role": "user", "content": self.current_query})
        query = self.current_query
        self.current_query = ""
        self.is_loading = True
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.backend_url}/api/v1/chat",
                    params={
                        "query": query,
                        "session_id": self.session_id,
                        "user_id": "default_user",
                        "top_k_retrieval": self.search_top_k,
                        "top_k_rerank": self.rerank_top_k,
                        "provider": self.llm_provider,
                    },
                )
                if response.status_code == 200:
                    data = response.json()
                    self.session_id = data["session_id"]
                    self.messages.append({"role": "assistant", "content": data["response"], "sources": data.get("sources", [])})
                else:
                    self.messages.append({"role": "assistant", "content": f"Error: {response.text}"})
        except Exception as e:
            self.messages.append({"role": "assistant", "content": f"Connection error: {str(e)}"})
        finally:
            self.is_loading = False


def index() -> rx.Component:
    return rx.box(rx.text("Multimodal RAG System - Placeholder Frontend"))


app = rx.App(style={}, theme=rx.theme(appearance="dark", accent_color="purple"))
app.add_page(index, route="/")
