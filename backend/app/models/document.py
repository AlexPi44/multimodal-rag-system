from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    TEXT = "text"
    PDF = "pdf"
    CODE = "code"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCX = "docx"


class Chunk(BaseModel):
    id: str
    content: str
    metadata: Dict
    embedding: Optional[List[float]] = None


class Document(BaseModel):
    id: str
    user_id: str
    filename: str
    file_type: DocumentType
    size: int
    chunks: List[str] = []
    metadata: Dict = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SearchResult(BaseModel):
    chunk_id: str
    content: str
    score: float
    metadata: Dict
    document_id: str
