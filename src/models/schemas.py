"""
Data models and schemas for the application.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class MessageRole(Enum):
    """Chat message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class DocumentType(Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"
    MARKDOWN = "md"


@dataclass
class DocumentMetadata:
    """Metadata extracted from a document."""
    filename: str
    file_type: DocumentType
    file_size_bytes: int
    page_count: int = 0
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    abstract: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    sections: Dict[str, str] = field(default_factory=dict)


@dataclass
class TextChunk:
    """A chunk of text with metadata."""
    content: str
    chunk_id: int
    document_id: str
    section: Optional[str] = None
    page_number: Optional[int] = None
    start_char: int = 0
    end_char: int = 0


@dataclass
class ChatMessage:
    """A single chat message."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    sources: List[TextChunk] = field(default_factory=list)
    confidence: Optional[float] = None


@dataclass
class ConversationHistory:
    """Complete conversation history."""
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    document_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AnalyticsEvent:
    """Analytics event for tracking."""
    event_type: str
    session_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentStats:
    """Statistics for a single document."""
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    page_count: int = 0
    avg_sentence_length: float = 0.0
    reading_time_minutes: int = 0
    sections: Dict[str, int] = field(default_factory=dict)  # section -> word count
    top_keywords: List[tuple] = field(default_factory=list)  # (word, frequency)
    named_entities: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class DashboardData:
    """Data for the visual dashboard."""
    document_metadata: DocumentMetadata
    stats: DocumentStats
    section_distribution: Dict[str, float] = field(default_factory=dict)
    keyword_cloud_data: List[Dict[str, Any]] = field(default_factory=list)
    readability_score: float = 0.0
    summary: str = ""
    key_findings: List[str] = field(default_factory=list)
    research_questions: List[str] = field(default_factory=list)
