"""
Sidebar component for document upload and settings.
"""

import streamlit as st
from typing import List, Optional, Callable, Any
from pathlib import Path

from ..config.settings import get_settings
from ..models.schemas import DocumentStats


class SidebarComponent:
    """Component for rendering the sidebar with upload and settings."""

    def __init__(self):
        self.settings = get_settings()

    def render(
        self,
        on_process: Callable,
        on_clear: Optional[Callable] = None,
        stats: Optional[DocumentStats] = None
    ) -> None:
        """
        Render the complete sidebar.

        Args:
            on_process: Callback when process button clicked.
            on_clear: Callback when clear button clicked.
            stats: Optional document statistics to display.
        """
        with st.sidebar:
            self._render_header()
            self._render_document_upload(on_process)
            self._render_settings()

            if stats:
                self._render_quick_stats(stats)

            self._render_actions(on_clear)
            self._render_footer()

    def _render_header(self) -> None:
        """Render sidebar header."""
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="font-size: 1.5rem; margin: 0;">📚 ResearchAI</h1>
            <p style="font-size: 0.875rem; color: #64748b; margin-top: 0.5rem;">
                Interactive Research Analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

    def _render_document_upload(self, on_process: Callable) -> List[Any]:
        """Render document upload section."""
        st.markdown("### 📤 Upload Documents")

        # Get supported formats
        formats = self.settings.supported_formats
        formats_str = ", ".join([f".{f}" for f in formats])

        uploaded_files = st.file_uploader(
            f"Supported: {formats_str}",
            accept_multiple_files=True,
            type=formats,
            key="file_uploader"
        )

        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
            for file in uploaded_files:
                file_size = file.size / 1024  # KB
                size_str = f"{file_size:.1f} KB" if file_size < 1024 else f"{file_size/1024:.1f} MB"
                st.markdown(f"• {file.name} ({size_str})")

        col1, col2 = st.columns(2)

        with col1:
            process_btn = st.button(
                "🔄 Process",
                disabled=not uploaded_files,
                use_container_width=True
            )

        with col2:
            clear_upload = st.button(
                "🗑️ Clear",
                use_container_width=True
            )

        if process_btn and uploaded_files:
            on_process(uploaded_files)

        if clear_upload:
            st.session_state.file_uploader = None
            st.rerun()

        return uploaded_files

    def _render_settings(self) -> None:
        """Render settings section."""
        with st.expander("⚙️ Settings", expanded=False):
            # Theme selection
            theme = st.selectbox(
                "Theme",
                ["Light", "Dark"],
                index=0 if st.session_state.get('theme', 'light') == 'light' else 1,
                key="theme_select"
            )

            if theme.lower() != st.session_state.get('theme', 'light'):
                st.session_state.theme = theme.lower()
                st.rerun()

            # Model selection
            st.markdown("**AI Model**")
            provider = st.selectbox(
                "Provider",
                ["OpenAI", "Anthropic"],
                key="llm_provider"
            )

            if provider == "OpenAI":
                models = self.settings.available_models.get('openai', [])
            else:
                models = self.settings.available_models.get('anthropic', [])

            model = st.selectbox(
                "Model",
                models,
                key="llm_model"
            )

            # Response settings
            st.markdown("**Response Settings**")
            temperature = st.slider(
                "Creativity",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                key="temperature"
            )

            show_sources = st.checkbox(
                "Show source citations",
                value=True,
                key="show_sources"
            )

            streaming = st.checkbox(
                "Enable streaming responses",
                value=True,
                key="streaming"
            )

    def _render_quick_stats(self, stats: DocumentStats) -> None:
        """Render quick statistics section."""
        st.divider()
        st.markdown("### 📊 Document Stats")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Words", f"{stats.word_count:,}")
            st.metric("Sentences", f"{stats.sentence_count:,}")
        with col2:
            st.metric("Read Time", f"{stats.reading_time_minutes} min")
            st.metric("Pages", stats.page_count)

        if stats.sections:
            with st.expander("Sections", expanded=False):
                for section, count in stats.sections.items():
                    st.markdown(f"• **{section}:** {count} words")

    def _render_actions(self, on_clear: Optional[Callable]) -> None:
        """Render action buttons."""
        st.divider()
        st.markdown("### 🔧 Actions")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                if on_clear:
                    on_clear()
                st.session_state.chat_history = []
                st.session_state.messages = []
                st.rerun()

        with col2:
            if st.button("📤 Export", use_container_width=True):
                st.session_state.show_export = True

    def _render_footer(self) -> None:
        """Render sidebar footer."""
        st.divider()
        st.markdown("""
        <div style="text-align: center; color: #64748b; font-size: 0.75rem;">
            <p>ResearchAI v2.0</p>
            <p>Powered by LangChain & OpenAI</p>
        </div>
        """, unsafe_allow_html=True)

    def render_model_selector(self) -> tuple:
        """
        Render standalone model selector.

        Returns:
            Tuple of (provider, model_name).
        """
        provider = st.selectbox(
            "AI Provider",
            ["OpenAI", "Anthropic"],
            key="provider_select"
        )

        if provider == "OpenAI":
            models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
        else:
            models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]

        model = st.selectbox(
            "Model",
            models,
            key="model_select"
        )

        return provider.lower(), model

    def render_processing_status(self, status: str, progress: float = 0) -> None:
        """Render document processing status."""
        st.markdown(f"**Status:** {status}")
        st.progress(progress)
