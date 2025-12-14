"""
Text processing utilities for document analysis.
"""

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """Utilities for text cleaning, analysis, and processing."""

    # Common English stopwords
    STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
        'have', 'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'than', 'too', 'very', 'can', 'just', 'should',
        'now', 'also', 'into', 'only', 'over', 'after', 'before', 'between'
    }

    PATTERNS_TO_REMOVE = [
        (r'\b[\w.-]+?@\w+?\.\w+?\b', ''),           # emails
        (r'\[[^\]]*\]', ''),                         # text in square brackets
        (r'Figure \d+[.:][^\n]+', ''),               # figure captions
        (r'Table \d+[.:][^\n]+', ''),                # table captions
        (r'^Source:.*$', ''),                        # source lines
        (r'[^\x00-\x7F]+', ' '),                     # non-ASCII characters
        (r'\bSee Figure \d+\b', ''),                 # references to figures
        (r'\bEq\.\s*\d+\b', ''),                     # equation references
        (r'\b(Table|Fig)\.\s*\d+\b', ''),            # other ref styles
        (r'<[^>]+>', ''),                            # HTML tags
        (r'https?://\S+', ''),                       # URLs
        (r'\d{1,2}/\d{1,2}/\d{2,4}', ''),           # dates
    ]

    @classmethod
    def clean_text(cls, text: str, lowercase: bool = True) -> str:
        """
        Clean and normalize text by removing noise.

        Args:
            text: Raw text to clean.
            lowercase: Whether to convert to lowercase.

        Returns:
            Cleaned and normalized text.
        """
        if lowercase:
            text = text.lower()

        for pattern, replacement in cls.PATTERNS_TO_REMOVE:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE | re.IGNORECASE)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @classmethod
    def extract_sections(cls, text: str, section_headers: Tuple[str, ...]) -> Dict[str, str]:
        """
        Extract sections from academic paper text.

        Args:
            text: Full document text.
            section_headers: Tuple of section header names.

        Returns:
            Dictionary mapping section names to their content.
        """
        sections = {}
        headers_pattern = '|'.join(re.escape(h) for h in section_headers)
        pattern = rf'(?:^|\n)\s*({headers_pattern})\s*(?:\n|:)'

        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))

        for i, match in enumerate(matches):
            section_name = match.group(1).strip().title()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_content = text[start:end].strip()
            sections[section_name] = section_content

        return sections

    @classmethod
    def count_words(cls, text: str) -> int:
        """Count words in text."""
        return len(text.split())

    @classmethod
    def count_sentences(cls, text: str) -> int:
        """Count sentences in text."""
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])

    @classmethod
    def count_paragraphs(cls, text: str) -> int:
        """Count paragraphs in text."""
        paragraphs = re.split(r'\n\s*\n', text)
        return len([p for p in paragraphs if p.strip()])

    @classmethod
    def calculate_reading_time(cls, text: str, wpm: int = 200) -> int:
        """
        Calculate estimated reading time in minutes.

        Args:
            text: Text to analyze.
            wpm: Words per minute reading speed.

        Returns:
            Estimated reading time in minutes.
        """
        word_count = cls.count_words(text)
        return max(1, round(word_count / wpm))

    @classmethod
    def extract_keywords(cls, text: str, top_n: int = 20) -> List[Tuple[str, int]]:
        """
        Extract top keywords from text.

        Args:
            text: Text to analyze.
            top_n: Number of top keywords to return.

        Returns:
            List of (word, frequency) tuples.
        """
        # Clean and tokenize
        cleaned = cls.clean_text(text, lowercase=True)
        words = re.findall(r'\b[a-z]{3,}\b', cleaned)

        # Remove stopwords
        words = [w for w in words if w not in cls.STOPWORDS]

        # Count frequencies
        counter = Counter(words)
        return counter.most_common(top_n)

    @classmethod
    def extract_named_entities_simple(cls, text: str) -> Dict[str, List[str]]:
        """
        Simple named entity extraction using patterns.

        Args:
            text: Text to analyze.

        Returns:
            Dictionary of entity types to lists of entities.
        """
        entities = {
            'organizations': [],
            'locations': [],
            'dates': [],
            'percentages': [],
            'numbers': []
        }

        # Find capitalized phrases (potential organizations/names)
        caps_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        entities['organizations'] = list(set(re.findall(caps_pattern, text)))[:10]

        # Find dates
        date_pattern = r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})\b'
        entities['dates'] = list(set(re.findall(date_pattern, text)))[:10]

        # Find percentages
        pct_pattern = r'\b(\d+(?:\.\d+)?%)\b'
        entities['percentages'] = list(set(re.findall(pct_pattern, text)))[:10]

        return entities

    @classmethod
    def calculate_readability_score(cls, text: str) -> float:
        """
        Calculate Flesch Reading Ease score.

        Args:
            text: Text to analyze.

        Returns:
            Readability score (0-100, higher = easier to read).
        """
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]

        if not words or not sentences:
            return 0.0

        # Count syllables (simplified)
        def count_syllables(word: str) -> int:
            word = word.lower()
            vowels = 'aeiouy'
            count = 0
            prev_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            if word.endswith('e'):
                count -= 1
            return max(1, count)

        total_syllables = sum(count_syllables(w) for w in words)
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = total_syllables / len(words)

        # Flesch Reading Ease formula
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return max(0, min(100, score))

    @classmethod
    def get_text_statistics(cls, text: str) -> Dict[str, any]:
        """
        Get comprehensive text statistics.

        Args:
            text: Text to analyze.

        Returns:
            Dictionary of statistics.
        """
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]

        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'paragraph_count': cls.count_paragraphs(text),
            'avg_sentence_length': len(words) / max(1, len(sentences)),
            'avg_word_length': sum(len(w) for w in words) / max(1, len(words)),
            'reading_time_minutes': cls.calculate_reading_time(text),
            'readability_score': cls.calculate_readability_score(text)
        }
