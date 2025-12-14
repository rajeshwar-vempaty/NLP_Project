"""
Application configuration settings.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """Application settings with sensible defaults."""

    # API Keys
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    # LLM Settings
    default_llm_provider: str = "openai"
    default_model: str = "gpt-3.5-turbo"
    available_models: dict = field(default_factory=lambda: {
        "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
    })

    # Embedding Settings
    default_embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-ada-002"

    # Document Processing
    chunk_size: int = 1500
    chunk_overlap: int = 200
    max_file_size_mb: int = 50
    supported_formats: List[str] = field(default_factory=lambda: ["pdf", "docx", "txt", "html", "md"])

    # Vector Store
    vector_store_path: str = "./data/vector_stores"
    use_persistent_storage: bool = True

    # Section Headers for Academic Papers
    section_headers: tuple = (
        'Abstract', 'Introduction', 'Background', 'Literature Review',
        'Methods', 'Methodology', 'Materials and Methods',
        'Results', 'Findings', 'Discussion', 'Conclusion',
        'Conclusions', 'References', 'Acknowledgements', 'Appendix'
    )

    # UI Settings
    app_title: str = "ResearchAI: Interactive Research Paper Analysis"
    app_icon: str = ":books:"
    default_theme: str = "light"

    # Analytics
    enable_analytics: bool = True
    analytics_db_path: str = "./data/analytics.db"

    # Rate Limiting
    max_queries_per_minute: int = 20
    max_documents_per_session: int = 10


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
