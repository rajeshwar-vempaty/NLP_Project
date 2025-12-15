"""
LLM service for multi-provider language model support.

Uses a modern approach compatible with langchain 0.2.0+ that avoids
deprecated memory and chain APIs.
"""

import logging
import os
from typing import Generator, List, Optional, Tuple
from dataclasses import dataclass

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..config.settings import get_settings
from ..models.schemas import TextChunk

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class SourceDocument:
    """Represents a source document used in response."""
    content: str
    section: Optional[str]
    relevance_score: float


class LLMService:
    """Service for LLM operations with multi-provider support."""

    def __init__(self):
        self.settings = get_settings()
        self.vectorstore = None
        self.chat_history: List = []  # Manual chat history management
        self._llm = None

    def _get_openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from settings or environment."""
        return self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")

    def _create_chat_llm(self, model_name: Optional[str] = None, temperature: float = 0.7,
                         max_tokens: int = 1000, streaming: bool = False) -> ChatOpenAI:
        """Create a ChatOpenAI instance with proper API key handling."""
        api_key = self._get_openai_api_key()
        if not api_key:
            raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY in your environment or .env file.")
        return ChatOpenAI(
            model_name=model_name or self.settings.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            openai_api_key=api_key
        )

    def create_embeddings(self, provider: str = "openai", model_name: Optional[str] = None):
        """
        Create embedding model based on provider.

        Args:
            provider: Embedding provider ('openai' or 'huggingface').
            model_name: Optional model name for HuggingFace.

        Returns:
            Embedding model instance.
        """
        # Check if OpenAI API key is available
        openai_api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")

        if provider == "huggingface" or not openai_api_key:
            # Use HuggingFace embeddings as fallback
            hf_model = model_name or "all-MiniLM-L6-v2"
            logger.info(f"Using HuggingFace embeddings: {hf_model}")
            try:
                return HuggingFaceEmbeddings(model_name=hf_model)
            except Exception as e:
                logger.warning(f"Failed to load HuggingFace embeddings: {e}")
                raise
        else:
            logger.info("Using OpenAI embeddings")
            return OpenAIEmbeddings(openai_api_key=openai_api_key)

    def create_vectorstore(
        self,
        chunks: List[TextChunk],
        provider: str = "openai",
        persist_path: Optional[str] = None
    ) -> FAISS:
        """
        Create FAISS vector store from text chunks.

        Args:
            chunks: List of TextChunk objects.
            provider: Embedding provider.
            persist_path: Optional path to persist the vector store.

        Returns:
            FAISS vector store.
        """
        embeddings = self.create_embeddings(provider)

        # Extract text content and metadata
        texts = [chunk.content for chunk in chunks]
        metadatas = [
            {
                'chunk_id': chunk.chunk_id,
                'section': chunk.section or 'Unknown',
                'document_id': chunk.document_id
            }
            for chunk in chunks
        ]

        self.vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas
        )

        logger.info(f"Created vector store with {len(chunks)} chunks")

        # Persist if path provided
        if persist_path:
            self.save_vectorstore(persist_path)

        return self.vectorstore

    def save_vectorstore(self, path: str) -> None:
        """Save vector store to disk."""
        if self.vectorstore:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.vectorstore.save_local(path)
            logger.info(f"Vector store saved to {path}")

    def load_vectorstore(self, path: str, provider: str = "openai") -> FAISS:
        """Load vector store from disk."""
        embeddings = self.create_embeddings(provider)
        self.vectorstore = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info(f"Vector store loaded from {path}")
        return self.vectorstore

    def create_conversation_chain(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> None:
        """
        Initialize the LLM for conversation.

        Args:
            model_name: LLM model name.
            temperature: Response temperature.
            max_tokens: Maximum response tokens.
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")

        self._llm = self._create_chat_llm(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True
        )

        # Clear chat history when creating new chain
        self.chat_history = []

        logger.info(f"Initialized conversation with model {model_name}")

    def _build_context_prompt(self, question: str, context: str) -> List:
        """Build the prompt messages for the LLM."""
        system_message = SystemMessage(content="""You are a helpful research assistant. Answer questions based on the provided context from research documents.
If you cannot answer based on the context, say so clearly. Always be accurate and cite relevant parts of the context when possible.
Keep responses concise but comprehensive.""")

        # Build conversation history
        messages = [system_message]

        # Add chat history (last 6 exchanges max to avoid token limits)
        for msg in self.chat_history[-12:]:
            messages.append(msg)

        # Add current question with context
        user_prompt = f"""Context from documents:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above."""

        messages.append(HumanMessage(content=user_prompt))

        return messages

    def query(
        self,
        question: str,
        include_sources: bool = True
    ) -> Tuple[str, List[SourceDocument]]:
        """
        Query the documents with a question.

        Args:
            question: User question.
            include_sources: Whether to include source documents.

        Returns:
            Tuple of (answer, source_documents).
        """
        if not self._llm or not self.vectorstore:
            raise ValueError("Conversation not initialized. Call create_conversation_chain first.")

        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search(question, k=4)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

        # Build messages and get response
        messages = self._build_context_prompt(question, context)
        response = self._llm.invoke(messages)

        answer = response.content if hasattr(response, 'content') else str(response)

        # Update chat history
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        # Build sources
        sources = []
        if include_sources:
            for doc in docs:
                sources.append(SourceDocument(
                    content=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    section=doc.metadata.get('section'),
                    relevance_score=0.0
                ))

        return answer, sources

    def query_streaming(
        self,
        question: str
    ) -> Generator[str, None, None]:
        """
        Query with streaming response.

        Args:
            question: User question.

        Yields:
            Response chunks.
        """
        if not self._llm or not self.vectorstore:
            raise ValueError("Conversation not initialized. Call create_conversation_chain first.")

        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search(question, k=4)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

        # Build messages
        messages = self._build_context_prompt(question, context)

        # Stream response
        full_response = ""
        for chunk in self._llm.stream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                full_response += chunk.content
                yield chunk.content

        # Update chat history after streaming completes
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=full_response))

    def generate_summary(self, text: str, max_length: int = 300) -> str:
        """
        Generate a summary of the document.

        Args:
            text: Document text.
            max_length: Maximum summary length in words.

        Returns:
            Summary text.
        """
        llm = self._create_chat_llm(temperature=0.3)

        prompt = f"""Summarize the following research paper in {max_length} words or less.
Focus on the main objective, methodology, key findings, and conclusions.

Text:
{text[:8000]}

Summary:"""

        response = llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)

    def extract_key_findings(self, text: str, num_findings: int = 5) -> List[str]:
        """
        Extract key findings from the document.

        Args:
            text: Document text.
            num_findings: Number of findings to extract.

        Returns:
            List of key findings.
        """
        llm = self._create_chat_llm(temperature=0.3)

        prompt = f"""Extract the {num_findings} most important findings or contributions from this research paper.
Return them as a numbered list, one finding per line.

Text:
{text[:8000]}

Key Findings:"""

        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)

        # Parse findings
        findings = []
        for line in content.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering
                finding = line.lstrip('0123456789.-) ').strip()
                if finding:
                    findings.append(finding)

        return findings[:num_findings]

    def generate_research_questions(self, text: str, num_questions: int = 5) -> List[str]:
        """
        Generate research questions based on the document.

        Args:
            text: Document text.
            num_questions: Number of questions to generate.

        Returns:
            List of suggested questions.
        """
        llm = self._create_chat_llm(temperature=0.5)

        prompt = f"""Based on this research paper, generate {num_questions} insightful questions
that a reader might want to ask about the content. Focus on methodology,
findings, implications, and potential applications.

Text:
{text[:6000]}

Questions:"""

        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)

        # Parse questions
        questions = []
        for line in content.split('\n'):
            line = line.strip()
            if line and '?' in line:
                question = line.lstrip('0123456789.-) ').strip()
                if question:
                    questions.append(question)

        return questions[:num_questions]

    def clear_memory(self) -> None:
        """Clear conversation memory."""
        self.chat_history = []
        logger.info("Conversation memory cleared")
