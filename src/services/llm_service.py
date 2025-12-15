"""
LLM service for multi-provider language model support.
"""

import logging
import os
from typing import Generator, List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from ..config.settings import get_settings
from ..models.schemas import TextChunk, ChatMessage, MessageRole

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
        self.conversation_chain = None
        self.memory = None

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
            hf_model = model_name or "hkunlp/instructor-base"
            logger.info(f"Using HuggingFace embeddings: {hf_model}")
            try:
                return HuggingFaceInstructEmbeddings(model_name=hf_model)
            except Exception as e:
                logger.warning(f"Failed to load instructor embeddings: {e}, trying sentence-transformers")
                from langchain_community.embeddings import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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
    ) -> ConversationalRetrievalChain:
        """
        Create conversational retrieval chain.

        Args:
            model_name: LLM model name.
            temperature: Response temperature.
            max_tokens: Maximum response tokens.

        Returns:
            Conversational retrieval chain.
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")

        llm = self._create_chat_llm(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True
        )

        self.memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )

        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            ),
            memory=self.memory,
            return_source_documents=True,
            verbose=False
        )

        logger.info(f"Created conversation chain with model {model_name}")
        return self.conversation_chain

    def query(
        self,
        question: str,
        include_sources: bool = True
    ) -> Tuple[str, List[SourceDocument]]:
        """
        Query the conversation chain.

        Args:
            question: User question.
            include_sources: Whether to include source documents.

        Returns:
            Tuple of (answer, source_documents).
        """
        if not self.conversation_chain:
            raise ValueError("Conversation chain not initialized.")

        result = self.conversation_chain({"question": question})

        answer = result.get('answer', '')
        sources = []

        if include_sources and 'source_documents' in result:
            for doc in result['source_documents']:
                sources.append(SourceDocument(
                    content=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    section=doc.metadata.get('section'),
                    relevance_score=0.0  # FAISS doesn't return scores by default
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
        if not self.conversation_chain:
            raise ValueError("Conversation chain not initialized.")

        # For streaming, we need to use the LLM directly with context
        docs = self.vectorstore.similarity_search(question, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""Based on the following context, answer the question.
If you cannot answer based on the context, say so.

Context:
{context}

Question: {question}

Answer:"""

        llm = self._create_chat_llm(streaming=True)

        for chunk in llm.stream(prompt):
            if hasattr(chunk, 'content'):
                yield chunk.content

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
        if self.memory:
            self.memory.clear()
            logger.info("Conversation memory cleared")
