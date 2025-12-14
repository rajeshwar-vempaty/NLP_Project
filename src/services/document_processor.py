"""
Document processing service for multi-format document handling.
"""

import io
import logging
import os
import re
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

import pdfplumber

from ..config.settings import get_settings
from ..models.schemas import DocumentMetadata, TextChunk, DocumentStats, DocumentType
from ..utils.text_processing import TextProcessor

logger = logging.getLogger(__name__)
settings = get_settings()


class DocumentProcessor:
    """Handles document extraction, processing, and analysis for multiple formats."""

    def __init__(self):
        self.settings = get_settings()
        self.text_processor = TextProcessor()

    def process_document(self, file_obj, filename: str) -> Tuple[str, DocumentMetadata, DocumentStats]:
        """
        Process a document and extract text, metadata, and statistics.

        Args:
            file_obj: File object or path.
            filename: Original filename.

        Returns:
            Tuple of (raw_text, metadata, stats).
        """
        file_ext = Path(filename).suffix.lower().lstrip('.')

        try:
            doc_type = DocumentType(file_ext)
        except ValueError:
            raise ValueError(f"Unsupported file format: {file_ext}")

        # Extract text based on document type
        if doc_type == DocumentType.PDF:
            raw_text, page_count = self._extract_pdf(file_obj)
        elif doc_type == DocumentType.DOCX:
            raw_text, page_count = self._extract_docx(file_obj)
        elif doc_type == DocumentType.TXT:
            raw_text, page_count = self._extract_txt(file_obj)
        elif doc_type == DocumentType.HTML:
            raw_text, page_count = self._extract_html(file_obj)
        elif doc_type == DocumentType.MARKDOWN:
            raw_text, page_count = self._extract_markdown(file_obj)
        else:
            raise ValueError(f"Unsupported document type: {doc_type}")

        # Get file size
        file_obj.seek(0, 2)
        file_size = file_obj.tell()
        file_obj.seek(0)

        # Extract sections
        sections = TextProcessor.extract_sections(raw_text, self.settings.section_headers)

        # Extract metadata
        metadata = self._extract_metadata(raw_text, filename, doc_type, file_size, page_count, sections)

        # Calculate statistics
        stats = self._calculate_stats(raw_text, sections)

        return raw_text, metadata, stats

    def _extract_pdf(self, file_obj) -> Tuple[str, int]:
        """Extract text from PDF."""
        text_parts = []
        page_count = 0

        try:
            with pdfplumber.open(file_obj) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except Exception as e:
            logger.error(f"Failed to extract PDF: {e}")
            raise ValueError(f"Error extracting PDF: {e}")

        return '\n\n'.join(text_parts), page_count

    def _extract_docx(self, file_obj) -> Tuple[str, int]:
        """Extract text from DOCX."""
        try:
            from docx import Document
            doc = Document(file_obj)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            # Estimate page count (roughly 500 words per page)
            word_count = sum(len(p.split()) for p in paragraphs)
            page_count = max(1, word_count // 500)
            return '\n\n'.join(paragraphs), page_count
        except ImportError:
            logger.warning("python-docx not installed, DOCX support unavailable")
            raise ValueError("DOCX support requires python-docx library")
        except Exception as e:
            logger.error(f"Failed to extract DOCX: {e}")
            raise ValueError(f"Error extracting DOCX: {e}")

    def _extract_txt(self, file_obj) -> Tuple[str, int]:
        """Extract text from TXT file."""
        try:
            content = file_obj.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='ignore')
            word_count = len(content.split())
            page_count = max(1, word_count // 500)
            return content, page_count
        except Exception as e:
            logger.error(f"Failed to extract TXT: {e}")
            raise ValueError(f"Error extracting TXT: {e}")

    def _extract_html(self, file_obj) -> Tuple[str, int]:
        """Extract text from HTML."""
        try:
            from bs4 import BeautifulSoup
            content = file_obj.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='ignore')
            soup = BeautifulSoup(content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text(separator='\n')
            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = '\n'.join(lines)
            word_count = len(text.split())
            page_count = max(1, word_count // 500)
            return text, page_count
        except ImportError:
            logger.warning("beautifulsoup4 not installed, HTML support unavailable")
            raise ValueError("HTML support requires beautifulsoup4 library")
        except Exception as e:
            logger.error(f"Failed to extract HTML: {e}")
            raise ValueError(f"Error extracting HTML: {e}")

    def _extract_markdown(self, file_obj) -> Tuple[str, int]:
        """Extract text from Markdown."""
        try:
            content = file_obj.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='ignore')

            # Remove markdown formatting but keep text
            # Remove headers markers
            content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)
            # Remove bold/italic markers
            content = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', content)
            content = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', content)
            # Remove links but keep text
            content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
            # Remove images
            content = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', content)
            # Remove code blocks
            content = re.sub(r'```[\s\S]*?```', '', content)
            content = re.sub(r'`[^`]+`', '', content)

            word_count = len(content.split())
            page_count = max(1, word_count // 500)
            return content, page_count
        except Exception as e:
            logger.error(f"Failed to extract Markdown: {e}")
            raise ValueError(f"Error extracting Markdown: {e}")

    def _extract_metadata(
        self,
        text: str,
        filename: str,
        doc_type: DocumentType,
        file_size: int,
        page_count: int,
        sections: Dict[str, str]
    ) -> DocumentMetadata:
        """Extract document metadata."""
        # Try to extract title (usually first non-empty line)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        title = lines[0] if lines else filename

        # Extract abstract if available
        abstract = sections.get('Abstract', '')[:500] if sections else ''

        # Extract keywords from text
        keywords = [word for word, _ in TextProcessor.extract_keywords(text, 10)]

        return DocumentMetadata(
            filename=filename,
            file_type=doc_type,
            file_size_bytes=file_size,
            page_count=page_count,
            title=title[:200] if title else None,
            abstract=abstract if abstract else None,
            keywords=keywords,
            sections=sections
        )

    def _calculate_stats(self, text: str, sections: Dict[str, str]) -> DocumentStats:
        """Calculate document statistics."""
        base_stats = TextProcessor.get_text_statistics(text)

        # Section word counts
        section_word_counts = {
            name: TextProcessor.count_words(content)
            for name, content in sections.items()
        }

        # Keywords
        top_keywords = TextProcessor.extract_keywords(text, 20)

        # Named entities
        entities = TextProcessor.extract_named_entities_simple(text)

        return DocumentStats(
            word_count=base_stats['word_count'],
            sentence_count=base_stats['sentence_count'],
            paragraph_count=base_stats['paragraph_count'],
            page_count=max(1, base_stats['word_count'] // 500),
            avg_sentence_length=base_stats['avg_sentence_length'],
            reading_time_minutes=base_stats['reading_time_minutes'],
            sections=section_word_counts,
            top_keywords=top_keywords,
            named_entities=entities
        )

    def create_chunks(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[TextChunk]:
        """
        Split text into chunks for embedding.

        Args:
            text: Text to split.
            chunk_size: Maximum chunk size.
            chunk_overlap: Overlap between chunks.

        Returns:
            List of TextChunk objects.
        """
        chunk_size = chunk_size or self.settings.chunk_size
        chunk_overlap = chunk_overlap or self.settings.chunk_overlap

        # Clean text first
        cleaned_text = TextProcessor.clean_text(text, lowercase=False)

        # Split by section headers first
        sections = TextProcessor.extract_sections(text, self.settings.section_headers)

        chunks = []
        chunk_id = 0

        if sections:
            # Process each section
            for section_name, section_content in sections.items():
                section_chunks = self._split_text_into_chunks(
                    section_content, chunk_size, chunk_overlap
                )
                for chunk_text in section_chunks:
                    chunks.append(TextChunk(
                        content=chunk_text,
                        chunk_id=chunk_id,
                        document_id="",  # Will be set later
                        section=section_name
                    ))
                    chunk_id += 1
        else:
            # No sections found, split entire text
            text_chunks = self._split_text_into_chunks(cleaned_text, chunk_size, chunk_overlap)
            for chunk_text in text_chunks:
                chunks.append(TextChunk(
                    content=chunk_text,
                    chunk_id=chunk_id,
                    document_id=""
                ))
                chunk_id += 1

        return chunks

    def _split_text_into_chunks(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []

        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0

        for word in words:
            word_len = len(word) + 1

            if current_length + word_len > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))

                # Calculate overlap
                overlap_words = []
                overlap_length = 0
                for w in reversed(current_chunk):
                    if overlap_length + len(w) + 1 <= overlap:
                        overlap_words.insert(0, w)
                        overlap_length += len(w) + 1
                    else:
                        break

                current_chunk = overlap_words
                current_length = overlap_length

            current_chunk.append(word)
            current_length += word_len

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks
