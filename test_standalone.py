"""
Standalone unit tests for ResearchAI text processing functions.

These tests don't require heavy dependencies like pdfplumber/langchain.
Run with: python -m unittest test_standalone -v
"""

import re
import unittest
import numpy as np
import faiss

# Constants (copied from app.py)
CHUNK_SIZE = 1500
SECTION_HEADERS = (
    'Abstract', 'Introduction', 'Methods', 'Methodology',
    'Results', 'Discussion', 'Conclusion'
)


def clean_text(text: str) -> str:
    """Clean and normalize extracted text by removing noise."""
    patterns_to_remove = [
        r'\b[\w.-]+?@\w+?\.\w+?\b',      # emails
        r'\[[^\]]*\]',                     # text in square brackets
        r'Figure \d+: [^\n]+',             # figure captions
        r'Table \d+: [^\n]+',              # table captions
        r'^Source:.*$',                    # source lines
        r'[^\x00-\x7F]+',                  # non-ASCII characters
        r'\bSee Figure \d+\b',             # references to figures
        r'\bEq\.\s*\d+\b',                 # equation references
        r'\b(Table|Fig)\.\s*\d+\b',        # other ref styles
        r'<[^>]+>',                        # HTML tags
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = text.lower()
    return re.sub(r'\s+', ' ', text).strip()


def get_text_chunks(text: str) -> list:
    """Split text into manageable chunks for embedding."""
    headers_pattern = '|'.join(SECTION_HEADERS)
    header_regex = re.compile(
        rf'\n\s*({headers_pattern})\s*\n',
        flags=re.IGNORECASE
    )
    section_regex = re.compile(
        rf'^({headers_pattern})$',
        flags=re.IGNORECASE
    )

    sections = header_regex.split(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for section in sections:
        if section_regex.match(section):
            if current_chunk:
                chunks.append(' '.join(current_chunk).strip())
                current_chunk = []
                current_length = 0
            current_chunk.append(section)
            current_length = len(section) + 1
        else:
            words = section.split()
            for word in words:
                word_len = len(word) + 1
                if current_length + word_len > CHUNK_SIZE:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk).strip())
                    current_chunk = [word]
                    current_length = word_len
                else:
                    current_chunk.append(word)
                    current_length += word_len

    if current_chunk:
        chunks.append(' '.join(current_chunk).strip())

    return chunks


class TestCleanText(unittest.TestCase):
    """Test cases for the clean_text function."""

    def test_removes_emails(self):
        """Test that email addresses are removed."""
        text = "Contact us at test@example.com for more info."
        result = clean_text(text)
        self.assertNotIn("test@example.com", result)

    def test_removes_html_tags(self):
        """Test that HTML tags are removed."""
        text = "This is <b>bold</b> and <i>italic</i> text."
        result = clean_text(text)
        self.assertNotIn("<b>", result)
        self.assertNotIn("</b>", result)

    def test_removes_figure_captions(self):
        """Test that figure captions are removed."""
        text = "Some text.\nFigure 1: This is a caption\nMore text."
        result = clean_text(text)
        self.assertNotIn("figure 1:", result)

    def test_removes_brackets(self):
        """Test that text in square brackets is removed."""
        text = "Reference [1] and citation [Smith et al., 2020]."
        result = clean_text(text)
        self.assertNotIn("[1]", result)
        self.assertNotIn("[Smith et al., 2020]", result)

    def test_normalizes_whitespace(self):
        """Test that multiple whitespaces are normalized."""
        text = "Too    many   spaces   here."
        result = clean_text(text)
        self.assertNotIn("  ", result)

    def test_converts_to_lowercase(self):
        """Test that text is converted to lowercase."""
        text = "UPPERCASE and MixedCase"
        result = clean_text(text)
        self.assertEqual(result, "uppercase and mixedcase")


class TestGetTextChunks(unittest.TestCase):
    """Test cases for the get_text_chunks function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        result = get_text_chunks("Sample text")
        self.assertIsInstance(result, list)

    def test_handles_empty_text(self):
        """Test handling of empty text."""
        result = get_text_chunks("")
        self.assertEqual(result, [])

    def test_respects_chunk_size(self):
        """Test that chunks do not exceed maximum size."""
        long_text = "word " * 1000
        result = get_text_chunks(long_text)
        for chunk in result:
            self.assertLessEqual(len(chunk), CHUNK_SIZE)

    def test_splits_on_section_headers(self):
        """Test that text is split on section headers."""
        text = "Intro text.\n Abstract \nAbstract content.\n Introduction \nIntro content."
        result = get_text_chunks(text)
        self.assertGreater(len(result), 1)

    def test_preserves_content(self):
        """Test that all words are preserved in chunks."""
        text = "This is a test sentence with multiple words."
        result = get_text_chunks(text)
        joined = ' '.join(result).lower()
        for word in text.lower().split():
            self.assertIn(word, joined)


class TestFaissVectorStore(unittest.TestCase):
    """Test cases for FAISS vector store functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_texts = [
            '''Self-attention, sometimes called intra-attention is an attention
            mechanism relating different positions of a single sequence.''',
            '''End-to-end memory networks are based on a recurrent attention
            mechanism instead of sequence-aligned recurrence.''',
            '''Scaled Dot-Product Attention consists of queries and keys
            of dimension dk, and values of dimension dv.'''
        ]
        self.dimension = 384

    def test_faiss_index_creation(self):
        """Test that FAISS index can be created."""
        index = faiss.IndexFlatL2(self.dimension)
        self.assertEqual(index.d, self.dimension)

    def test_faiss_add_embeddings(self):
        """Test that embeddings can be added to FAISS index."""
        index = faiss.IndexFlatL2(self.dimension)
        embeddings = np.random.rand(
            len(self.sample_texts), self.dimension
        ).astype('float32')
        index.add(embeddings)
        self.assertEqual(index.ntotal, len(self.sample_texts))

    def test_faiss_search(self):
        """Test that FAISS search returns results."""
        index = faiss.IndexFlatL2(self.dimension)
        embeddings = np.random.rand(
            len(self.sample_texts), self.dimension
        ).astype('float32')
        index.add(embeddings)

        query = np.random.rand(1, self.dimension).astype('float32')
        distances, indices = index.search(query, k=2)

        self.assertEqual(len(indices[0]), 2)
        self.assertTrue(all(idx < len(self.sample_texts) for idx in indices[0]))


if __name__ == '__main__':
    unittest.main()
