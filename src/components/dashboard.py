"""
Dashboard component for visual research paper summary.
"""

import streamlit as st
from typing import Dict, List, Any, Optional
import json

from ..models.schemas import DocumentMetadata, DocumentStats, DashboardData
from ..utils.visualization import VisualizationUtils


class DashboardComponent:
    """Component for rendering the research paper dashboard."""

    def __init__(self, theme: str = 'light'):
        self.theme = theme

    def render(self, dashboard_data: DashboardData) -> None:
        """
        Render the complete dashboard.

        Args:
            dashboard_data: DashboardData object with all visualization data.
        """
        st.markdown("## 📊 Document Analysis Dashboard")

        # Document Info Header
        self._render_document_header(dashboard_data.document_metadata)

        # Metrics Row
        self._render_metrics_row(dashboard_data.stats)

        # Two-column layout for main content
        col1, col2 = st.columns(2)

        with col1:
            # Summary Section
            self._render_summary_section(dashboard_data.summary)

            # Key Findings
            self._render_key_findings(dashboard_data.key_findings)

        with col2:
            # Section Distribution
            self._render_section_distribution(dashboard_data.stats.sections)

            # Readability Score
            self._render_readability_gauge(
                dashboard_data.stats,
                dashboard_data.readability_score
            )

        # Keywords Cloud (full width)
        self._render_keywords_cloud(dashboard_data.stats.top_keywords)

        # Suggested Questions
        self._render_suggested_questions(dashboard_data.research_questions)

        # Named Entities (if available)
        if dashboard_data.stats.named_entities:
            self._render_named_entities(dashboard_data.stats.named_entities)

    def _render_document_header(self, metadata: DocumentMetadata) -> None:
        """Render document header with title and metadata."""
        st.markdown(f"""
        <div class="section-card">
            <h2 style="margin:0; color: var(--text-primary);">📄 {metadata.title or metadata.filename}</h2>
            <p style="color: var(--text-secondary); margin-top: 0.5rem;">
                <strong>File:</strong> {metadata.filename} |
                <strong>Type:</strong> {metadata.file_type.value.upper()} |
                <strong>Pages:</strong> {metadata.page_count}
            </p>
        </div>
        """, unsafe_allow_html=True)

    def _render_metrics_row(self, stats: DocumentStats) -> None:
        """Render metrics cards row."""
        cols = st.columns(4)

        metrics = [
            ("📝", f"{stats.word_count:,}", "Total Words"),
            ("⏱️", f"{stats.reading_time_minutes} min", "Reading Time"),
            ("📊", f"{stats.sentence_count:,}", "Sentences"),
            ("📄", f"{stats.page_count}", "Pages"),
        ]

        for col, (icon, value, label) in zip(cols, metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="icon">{icon}</div>
                    <div class="value">{value}</div>
                    <div class="label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

    def _render_summary_section(self, summary: str) -> None:
        """Render AI-generated summary."""
        st.markdown("### 📋 Summary")
        if summary:
            st.markdown(f"""
            <div class="summary-box">
                <p>{summary}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Summary will be generated after processing.")

    def _render_key_findings(self, findings: List[str]) -> None:
        """Render key findings list."""
        st.markdown("### 🎯 Key Findings")

        if findings:
            findings_html = "".join([
                f'<li>{finding}</li>'
                for finding in findings
            ])
            st.markdown(f"""
            <ul class="findings-list">
                {findings_html}
            </ul>
            """, unsafe_allow_html=True)
        else:
            st.info("Key findings will be extracted after processing.")

    def _render_section_distribution(self, sections: Dict[str, int]) -> None:
        """Render section distribution chart."""
        st.markdown("### 📈 Section Distribution")

        if sections:
            # Create a simple bar chart using Streamlit
            import pandas as pd
            df = pd.DataFrame({
                'Section': list(sections.keys()),
                'Word Count': list(sections.values())
            })
            st.bar_chart(df.set_index('Section'))
        else:
            st.info("No sections detected in the document.")

    def _render_readability_gauge(self, stats: DocumentStats, score: float) -> None:
        """Render readability score gauge."""
        st.markdown("### 📖 Readability")

        # Determine color based on score
        if score >= 70:
            color = "#22c55e"  # green
            label = "Easy to Read"
        elif score >= 50:
            color = "#eab308"  # yellow
            label = "Moderate"
        else:
            color = "#ef4444"  # red
            label = "Complex"

        st.markdown(f"""
        <div class="gauge-container">
            <div class="gauge-value" style="color: {color};">{score:.0f}</div>
            <div class="gauge-label">{label}</div>
            <div class="progress-container" style="margin-top: 1rem;">
                <div class="progress-bar" style="width: {score}%;"></div>
            </div>
            <small style="color: var(--text-secondary);">
                Flesch Reading Ease Score (0-100)
            </small>
        </div>
        """, unsafe_allow_html=True)

        # Additional stats
        st.markdown(f"""
        <div style="text-align: center; margin-top: 1rem; color: var(--text-secondary);">
            <small>
                Avg. sentence length: <strong>{stats.avg_sentence_length:.1f}</strong> words
            </small>
        </div>
        """, unsafe_allow_html=True)

    def _render_keywords_cloud(self, keywords: List[tuple]) -> None:
        """Render keywords visualization."""
        st.markdown("### 🏷️ Top Keywords")

        if keywords:
            # Create visual keyword cloud
            cloud_data = VisualizationUtils.create_word_cloud_data(keywords)

            keywords_html = "".join([
                f'<span class="keyword-tag" style="font-size: {item["size"]}px;">'
                f'{item["word"]} ({item["frequency"]})</span>'
                for item in cloud_data[:15]  # Top 15 keywords
            ])

            st.markdown(f"""
            <div class="keyword-cloud">
                {keywords_html}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No keywords extracted.")

    def _render_suggested_questions(self, questions: List[str]) -> None:
        """Render suggested research questions."""
        st.markdown("### ❓ Suggested Questions")

        if questions:
            st.markdown(
                "<p style='color: var(--text-secondary);'>"
                "Click a question to ask it:</p>",
                unsafe_allow_html=True
            )

            for i, question in enumerate(questions):
                if st.button(f"📌 {question}", key=f"question_{i}"):
                    st.session_state.suggested_question = question
                    st.rerun()
        else:
            st.info("Questions will be generated after processing.")

    def _render_named_entities(self, entities: Dict[str, List[str]]) -> None:
        """Render named entities section."""
        st.markdown("### 🔍 Detected Entities")

        entity_summary = VisualizationUtils.create_entity_summary(entities)

        if entity_summary:
            cols = st.columns(len(entity_summary))

            for col, (entity_type, data) in zip(cols, entity_summary.items()):
                with col:
                    st.markdown(f"**{entity_type.title()}** ({data['count']})")
                    for item in data['items']:
                        st.markdown(f"• {item}")
                    if data['has_more']:
                        st.markdown(f"*+{data['count'] - 5} more*")
        else:
            st.info("No entities detected.")

    def render_mini_dashboard(self, stats: DocumentStats) -> None:
        """Render a compact mini dashboard for the sidebar."""
        st.markdown("#### 📊 Quick Stats")

        st.metric("Words", f"{stats.word_count:,}")
        st.metric("Reading Time", f"{stats.reading_time_minutes} min")
        st.metric("Sentences", f"{stats.sentence_count:,}")

        if stats.sections:
            st.markdown("**Sections:**")
            for section, count in list(stats.sections.items())[:5]:
                st.markdown(f"• {section}: {count} words")
