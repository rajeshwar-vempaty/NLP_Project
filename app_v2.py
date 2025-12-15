"""
ResearchAI v2.0: Interactive PDF Knowledge Extraction System

A production-ready Streamlit application for document analysis and Q&A
with visual dashboards, multi-format support, and advanced features.
"""

import logging
import os
import uuid
from typing import List, Optional
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from src.config.settings import get_settings
from src.services.document_processor import DocumentProcessor
from src.services.llm_service import LLMService
from src.services.analytics import AnalyticsService
from src.models.schemas import (
    DocumentMetadata, DocumentStats, DashboardData,
    ChatMessage, MessageRole, AnalyticsEvent
)
from src.components.dashboard import DashboardComponent
from src.components.chat import ChatComponent
from src.components.sidebar import SidebarComponent
from src.templates.styles import Styles
from src.utils.text_processing import TextProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get settings
settings = get_settings()


def initialize_session_state() -> None:
    """Initialize all session state variables."""
    defaults = {
        'session_id': str(uuid.uuid4()),
        'conversation': None,
        'chat_history': [],
        'messages': [],
        'documents_processed': False,
        'current_stats': None,
        'current_metadata': None,
        'dashboard_data': None,
        'raw_text': None,
        'theme': 'light',
        'show_dashboard': True,
        'suggested_question': None,
        'llm_service': None,
        'analytics': None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Initialize services
    if st.session_state.analytics is None:
        st.session_state.analytics = AnalyticsService()


def process_documents(uploaded_files: List) -> None:
    """
    Process uploaded documents and prepare for Q&A.

    Args:
        uploaded_files: List of uploaded file objects.
    """
    if not uploaded_files:
        st.warning("Please upload at least one document.")
        return

    # Note: OpenAI API key is optional - will fall back to HuggingFace embeddings if not set
    if not settings.openai_api_key:
        st.info("OpenAI API key not set. Using HuggingFace embeddings (this may be slower on first run).")

    processor = DocumentProcessor()
    llm_service = LLMService()
    analytics = st.session_state.analytics

    all_text = []
    all_chunks = []
    combined_stats = None
    primary_metadata = None

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Processing {file.name}...")
            progress_bar.progress((i + 1) / (len(uploaded_files) + 2))

            # Process document
            raw_text, metadata, stats = processor.process_document(file, file.name)
            all_text.append(raw_text)

            # Create chunks
            chunks = processor.create_chunks(raw_text)
            for chunk in chunks:
                chunk.document_id = file.name
            all_chunks.extend(chunks)

            # Track first document as primary
            if primary_metadata is None:
                primary_metadata = metadata
                combined_stats = stats
            else:
                # Combine stats
                combined_stats.word_count += stats.word_count
                combined_stats.sentence_count += stats.sentence_count
                combined_stats.page_count += stats.page_count

            # Track analytics
            analytics.track_document_processed(
                st.session_state.session_id,
                file.name,
                metadata.file_type.value,
                stats.word_count
            )

        # Create vector store
        status_text.text("Creating embeddings...")
        progress_bar.progress(0.8)

        vectorstore = llm_service.create_vectorstore(all_chunks)

        # Create conversation chain
        model_name = st.session_state.get('llm_model', 'gpt-3.5-turbo')
        llm_service.create_conversation_chain(model_name=model_name)

        # Store in session
        st.session_state.llm_service = llm_service
        st.session_state.raw_text = '\n\n'.join(all_text)
        st.session_state.current_stats = combined_stats
        st.session_state.current_metadata = primary_metadata
        st.session_state.documents_processed = True

        # Generate dashboard data
        status_text.text("Generating insights...")
        progress_bar.progress(0.9)

        dashboard_data = generate_dashboard_data(
            st.session_state.raw_text,
            primary_metadata,
            combined_stats,
            llm_service
        )
        st.session_state.dashboard_data = dashboard_data

        progress_bar.progress(1.0)
        status_text.text("Processing complete!")

        logger.info(f"Processed {len(uploaded_files)} documents with {len(all_chunks)} chunks")
        st.success(f"Successfully processed {len(uploaded_files)} document(s)!")

    except Exception as e:
        logger.error(f"Error processing documents: {e}", exc_info=True)
        error_msg = str(e)
        if "api_key" in error_msg.lower():
            st.error("API Key Error: Please ensure your OpenAI API key is correctly configured in the .env file.")
        elif "validation error" in error_msg.lower():
            st.error(f"Configuration Error: {error_msg}. Please check your settings and try again.")
        else:
            st.error(f"Error processing documents: {error_msg}")
    finally:
        progress_bar.empty()
        status_text.empty()


def generate_dashboard_data(
    text: str,
    metadata: DocumentMetadata,
    stats: DocumentStats,
    llm_service: LLMService
) -> DashboardData:
    """
    Generate dashboard data including AI-powered insights.

    Args:
        text: Combined document text.
        metadata: Document metadata.
        stats: Document statistics.
        llm_service: LLM service for generating insights.

    Returns:
        DashboardData object.
    """
    # Calculate readability
    readability = TextProcessor.calculate_readability_score(text)

    # Generate AI insights (with fallbacks for API failures)
    try:
        summary = llm_service.generate_summary(text)
    except Exception as e:
        logger.warning(f"Failed to generate summary: {e}")
        summary = "Summary generation unavailable. Please check your API key."

    try:
        key_findings = llm_service.extract_key_findings(text)
    except Exception as e:
        logger.warning(f"Failed to extract findings: {e}")
        key_findings = []

    try:
        questions = llm_service.generate_research_questions(text)
    except Exception as e:
        logger.warning(f"Failed to generate questions: {e}")
        questions = []

    return DashboardData(
        document_metadata=metadata,
        stats=stats,
        section_distribution={k: v for k, v in stats.sections.items()},
        readability_score=readability,
        summary=summary,
        key_findings=key_findings,
        research_questions=questions
    )


def handle_user_question(question: str) -> None:
    """
    Handle user question and generate response.

    Args:
        question: User's question.
    """
    if not st.session_state.llm_service:
        st.error("Please upload and process documents first.")
        return

    llm_service = st.session_state.llm_service
    analytics = st.session_state.analytics

    # Add user message
    user_message = ChatMessage(
        role=MessageRole.USER,
        content=question
    )
    st.session_state.messages.append(user_message)

    # Get response
    try:
        show_sources = st.session_state.get('show_sources', True)
        use_streaming = st.session_state.get('streaming', True)

        if use_streaming:
            # Streaming response
            chat_component = ChatComponent(st.session_state.theme)

            with st.chat_message("assistant"):
                response = ""
                for chunk in llm_service.query_streaming(question):
                    response += chunk
                    st.write(response)

            # Get sources separately
            _, sources = llm_service.query(question, include_sources=True)

        else:
            # Non-streaming response
            response, sources = llm_service.query(question, include_sources=show_sources)

        # Create assistant message
        assistant_message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=response,
            sources=sources if show_sources else []
        )
        st.session_state.messages.append(assistant_message)

        # Track query
        analytics.track_query(
            st.session_state.session_id,
            question,
            len(response),
            len(sources) if sources else 0
        )

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        error_message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=f"I encountered an error: {str(e)}. Please try again."
        )
        st.session_state.messages.append(error_message)


def render_chat_interface() -> None:
    """Render the chat interface."""
    chat_component = ChatComponent(st.session_state.theme)

    # Render message history
    for message in st.session_state.messages:
        with st.chat_message("user" if message.role == MessageRole.USER else "assistant"):
            st.write(message.content)

            # Show sources for assistant messages
            if message.role == MessageRole.ASSISTANT and message.sources:
                with st.expander("📚 Sources"):
                    for source in message.sources:
                        st.markdown(f"**{source.section or 'Document'}:**")
                        st.markdown(f"> {source.content}")
                        st.divider()

    # Chat input
    if question := st.chat_input("Ask a question about your documents..."):
        handle_user_question(question)
        st.rerun()

    # Handle suggested questions
    if st.session_state.suggested_question:
        question = st.session_state.suggested_question
        st.session_state.suggested_question = None
        handle_user_question(question)
        st.rerun()


def clear_session() -> None:
    """Clear the current session."""
    keys_to_clear = [
        'conversation', 'chat_history', 'messages',
        'documents_processed', 'current_stats', 'current_metadata',
        'dashboard_data', 'raw_text', 'llm_service'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            st.session_state[key] = None if key != 'messages' else []

    st.session_state.documents_processed = False
    st.session_state.messages = []


def export_to_markdown() -> str:
    """Export chat history to markdown."""
    lines = [
        "# ResearchAI Chat Export",
        f"*Exported on {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        ""
    ]

    if st.session_state.current_metadata:
        lines.extend([
            "## Document Information",
            f"- **File:** {st.session_state.current_metadata.filename}",
            f"- **Pages:** {st.session_state.current_metadata.page_count}",
            ""
        ])

    lines.append("## Conversation")
    lines.append("")

    for msg in st.session_state.messages:
        role = "**You:**" if msg.role == MessageRole.USER else "**AI:**"
        lines.append(role)
        lines.append(msg.content)
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Main application entry point."""
    # Page config
    st.set_page_config(
        page_title=settings.app_title,
        page_icon=settings.app_icon,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    initialize_session_state()

    # Apply theme styles
    theme = st.session_state.get('theme', 'light')
    st.markdown(Styles.get_main_css(theme), unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        sidebar = SidebarComponent()
        sidebar.render(
            on_process=process_documents,
            on_clear=clear_session,
            stats=st.session_state.current_stats
        )

        # Export button
        if st.session_state.messages:
            st.divider()
            md_content = export_to_markdown()
            st.download_button(
                "📥 Export Chat (Markdown)",
                md_content,
                "chat_export.md",
                "text/markdown",
                use_container_width=True
            )

    # Main content
    st.title(f"{settings.app_icon} {settings.app_title}")

    # Show dashboard or chat based on state
    if st.session_state.documents_processed and st.session_state.dashboard_data:
        # Tab navigation
        tab1, tab2 = st.tabs(["📊 Dashboard", "💬 Chat"])

        with tab1:
            dashboard = DashboardComponent(theme)
            dashboard.render(st.session_state.dashboard_data)

        with tab2:
            render_chat_interface()

    elif st.session_state.documents_processed:
        # Documents processed but no dashboard
        render_chat_interface()

    else:
        # Welcome state
        st.markdown("""
        ### Welcome to ResearchAI! 👋

        Upload your research papers or documents to get started.

        **Features:**
        - 📊 **Visual Dashboard** - Get instant insights with charts and statistics
        - 💬 **AI Chat** - Ask questions about your documents
        - 📚 **Source Citations** - See exactly where answers come from
        - 🔄 **Multi-Format Support** - PDF, DOCX, TXT, HTML, Markdown
        - 🌙 **Dark Mode** - Easy on the eyes

        **How to use:**
        1. Upload documents using the sidebar
        2. Click "Process" to analyze
        3. Explore the dashboard or start chatting!
        """)


if __name__ == '__main__':
    main()
