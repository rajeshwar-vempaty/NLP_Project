"""
Unit tests for the ResearchAI text processing functions.

Tests cover text cleaning, chunking, and vector store creation.
"""

import unittest
import numpy as np
import faiss

from app import clean_text, get_text_chunks, CHUNK_SIZE


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
            mechanism relating different positions of a single sequence in order
            to compute a representation of the sequence.''',
            '''End-to-end memory networks are based on a recurrent attention
            mechanism instead of sequence-aligned recurrence and have been shown
            to perform well on simple-language question answering.''',
            '''We call our particular attention "Scaled Dot-Product Attention".
            The input consists of queries and keys of dimension dk, and values
            of dimension dv.'''
        ]
        self.dimension = 384  # Typical embedding dimension

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

        # Search for similar vectors
        query = np.random.rand(1, self.dimension).astype('float32')
        distances, indices = index.search(query, k=2)

        self.assertEqual(len(indices[0]), 2)
        self.assertTrue(all(idx < len(self.sample_texts) for idx in indices[0]))


if __name__ == '__main__':
    unittest.main()
