"""
Chat component for conversational interface.
"""

import streamlit as st
from typing import List, Optional, Callable
from datetime import datetime

from ..models.schemas import ChatMessage, MessageRole
from ..services.llm_service import SourceDocument
from ..templates.styles import Styles


class ChatComponent:
    """Component for rendering the chat interface."""

    def __init__(self, theme: str = 'light'):
        self.theme = theme
        self.styles = Styles()

    def render_chat_history(
        self,
        messages: List[ChatMessage],
        show_sources: bool = True
    ) -> None:
        """
        Render the complete chat history.

        Args:
            messages: List of ChatMessage objects.
            show_sources: Whether to show source citations.
        """
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        for message in messages:
            if message.role == MessageRole.USER:
                self._render_user_message(message.content)
            else:
                sources = message.sources if show_sources else []
                self._render_bot_message(message.content, sources)

        st.markdown('</div>', unsafe_allow_html=True)

    def _render_user_message(self, content: str) -> None:
        """Render a user message."""
        st.markdown(f'''
        <div class="chat-message user">
            <div class="avatar">👤</div>
            <div class="message">{self._escape_html(content)}</div>
        </div>
        ''', unsafe_allow_html=True)

    def _render_bot_message(
        self,
        content: str,
        sources: Optional[List[SourceDocument]] = None
    ) -> None:
        """Render a bot message with optional sources."""
        sources_html = ""
        if sources:
            source_chips = "".join([
                f'<span class="source-chip">📄 {source.section or "Document"}</span>'
                for source in sources[:5]  # Limit to 5 sources
            ])
            sources_html = f'''
            <div class="sources-container">
                <small><strong>Sources:</strong></small>
                {source_chips}
            </div>
            '''

        st.markdown(f'''
        <div class="chat-message bot">
            <div class="avatar">🤖</div>
            <div class="message">
                {self._format_response(content)}
                {sources_html}
            </div>
        </div>
        ''', unsafe_allow_html=True)

    def render_streaming_message(self, content_generator) -> str:
        """
        Render a streaming bot message.

        Args:
            content_generator: Generator yielding content chunks.

        Returns:
            Complete message content.
        """
        message_placeholder = st.empty()
        full_response = ""

        for chunk in content_generator:
            full_response += chunk
            message_placeholder.markdown(f'''
            <div class="chat-message bot">
                <div class="avatar">🤖</div>
                <div class="message">{self._format_response(full_response)}▌</div>
            </div>
            ''', unsafe_allow_html=True)

        # Final render without cursor
        message_placeholder.markdown(f'''
        <div class="chat-message bot">
            <div class="avatar">🤖</div>
            <div class="message">{self._format_response(full_response)}</div>
        </div>
        ''', unsafe_allow_html=True)

        return full_response

    def render_input(
        self,
        placeholder: str = "Ask a question about your documents...",
        key: str = "user_input"
    ) -> Optional[str]:
        """
        Render chat input field.

        Args:
            placeholder: Input placeholder text.
            key: Streamlit widget key.

        Returns:
            User input or None.
        """
        # Check for suggested question
        if 'suggested_question' in st.session_state and st.session_state.suggested_question:
            question = st.session_state.suggested_question
            st.session_state.suggested_question = None
            return question

        return st.chat_input(placeholder, key=key)

    def render_typing_indicator(self) -> None:
        """Render typing indicator while AI is processing."""
        st.markdown('''
        <div class="chat-message bot">
            <div class="avatar">🤖</div>
            <div class="message">
                <span class="loading-spinner"></span>
                <span style="margin-left: 0.5rem;">Thinking...</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    def render_error_message(self, error: str) -> None:
        """Render an error message in chat."""
        st.markdown(f'''
        <div class="chat-message bot" style="border-left: 4px solid #ef4444;">
            <div class="avatar">⚠️</div>
            <div class="message" style="color: #ef4444;">
                <strong>Error:</strong> {self._escape_html(error)}
            </div>
        </div>
        ''', unsafe_allow_html=True)

    def render_welcome_message(self) -> None:
        """Render welcome message for new sessions."""
        st.markdown('''
        <div class="chat-message bot">
            <div class="avatar">🤖</div>
            <div class="message">
                <strong>Welcome to ResearchAI!</strong><br><br>
                I'm here to help you analyze and understand research papers.
                Here's how to get started:<br><br>
                1. 📤 Upload your PDF documents using the sidebar<br>
                2. 🔄 Click "Process" to analyze the documents<br>
                3. 💬 Ask me any questions about the content<br><br>
                <em>Tip: I work best with academic papers, but I can help with any PDF document!</em>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    def _format_response(self, content: str) -> str:
        """Format response content with proper HTML."""
        # Convert markdown-like formatting
        content = self._escape_html(content)

        # Bold text
        import re
        content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)

        # Italic text
        content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content)

        # Code blocks
        content = re.sub(r'`(.*?)`', r'<code>\1</code>', content)

        # Line breaks
        content = content.replace('\n', '<br>')

        return content

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))

    def render_export_buttons(
        self,
        messages: List[ChatMessage],
        on_export_pdf: Optional[Callable] = None,
        on_export_md: Optional[Callable] = None
    ) -> None:
        """Render export buttons for chat history."""
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📄 Export as PDF", key="export_pdf"):
                if on_export_pdf:
                    on_export_pdf(messages)
                else:
                    st.info("PDF export coming soon!")

        with col2:
            if st.button("📝 Export as Markdown", key="export_md"):
                if on_export_md:
                    content = on_export_md(messages)
                    st.download_button(
                        "Download Markdown",
                        content,
                        "chat_history.md",
                        "text/markdown"
                    )
                else:
                    # Default markdown export
                    content = self._export_to_markdown(messages)
                    st.download_button(
                        "Download Markdown",
                        content,
                        "chat_history.md",
                        "text/markdown"
                    )

    def _export_to_markdown(self, messages: List[ChatMessage]) -> str:
        """Export chat history to markdown format."""
        lines = ["# Chat History", "", f"*Exported on {datetime.now().strftime('%Y-%m-%d %H:%M')}*", ""]

        for message in messages:
            role = "**You:**" if message.role == MessageRole.USER else "**AI:**"
            lines.append(f"{role}")
            lines.append(message.content)
            lines.append("")

        return "\n".join(lines)
