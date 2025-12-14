"""
Visualization utilities for creating charts and graphs.
"""

from typing import Dict, List, Any, Tuple
import json


class VisualizationUtils:
    """Utilities for creating visualization data."""

    @staticmethod
    def create_word_cloud_data(keywords: List[Tuple[str, int]]) -> List[Dict[str, Any]]:
        """
        Create data structure for word cloud visualization.

        Args:
            keywords: List of (word, frequency) tuples.

        Returns:
            List of dicts with word and size for visualization.
        """
        if not keywords:
            return []

        max_freq = max(freq for _, freq in keywords)
        min_size, max_size = 12, 48

        return [
            {
                'word': word,
                'frequency': freq,
                'size': min_size + (freq / max_freq) * (max_size - min_size)
            }
            for word, freq in keywords
        ]

    @staticmethod
    def create_section_chart_data(sections: Dict[str, int]) -> Dict[str, Any]:
        """
        Create data for section distribution pie/bar chart.

        Args:
            sections: Dictionary of section names to word counts.

        Returns:
            Chart configuration data.
        """
        total = sum(sections.values())
        if total == 0:
            return {'labels': [], 'values': [], 'percentages': []}

        return {
            'labels': list(sections.keys()),
            'values': list(sections.values()),
            'percentages': [round(v / total * 100, 1) for v in sections.values()]
        }

    @staticmethod
    def create_metrics_cards_data(stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create data for metrics display cards.

        Args:
            stats: Statistics dictionary.

        Returns:
            List of metric card configurations.
        """
        return [
            {
                'label': 'Word Count',
                'value': f"{stats.get('word_count', 0):,}",
                'icon': '📝'
            },
            {
                'label': 'Reading Time',
                'value': f"{stats.get('reading_time_minutes', 0)} min",
                'icon': '⏱️'
            },
            {
                'label': 'Sentences',
                'value': f"{stats.get('sentence_count', 0):,}",
                'icon': '📊'
            },
            {
                'label': 'Readability',
                'value': VisualizationUtils.get_readability_label(stats.get('readability_score', 0)),
                'icon': '📖'
            }
        ]

    @staticmethod
    def get_readability_label(score: float) -> str:
        """Convert readability score to human-readable label."""
        if score >= 90:
            return "Very Easy"
        elif score >= 80:
            return "Easy"
        elif score >= 70:
            return "Fairly Easy"
        elif score >= 60:
            return "Standard"
        elif score >= 50:
            return "Fairly Hard"
        elif score >= 30:
            return "Hard"
        else:
            return "Very Hard"

    @staticmethod
    def create_readability_gauge_data(score: float) -> Dict[str, Any]:
        """
        Create data for readability gauge visualization.

        Args:
            score: Readability score (0-100).

        Returns:
            Gauge configuration data.
        """
        return {
            'value': round(score, 1),
            'min': 0,
            'max': 100,
            'label': VisualizationUtils.get_readability_label(score),
            'color': VisualizationUtils._get_score_color(score)
        }

    @staticmethod
    def _get_score_color(score: float) -> str:
        """Get color based on readability score."""
        if score >= 70:
            return '#22c55e'  # green
        elif score >= 50:
            return '#eab308'  # yellow
        else:
            return '#ef4444'  # red

    @staticmethod
    def create_entity_summary(entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Create summary of named entities for display.

        Args:
            entities: Dictionary of entity types to lists.

        Returns:
            Summary data for visualization.
        """
        return {
            entity_type: {
                'count': len(items),
                'items': items[:5],  # Top 5 of each type
                'has_more': len(items) > 5
            }
            for entity_type, items in entities.items()
            if items
        }
