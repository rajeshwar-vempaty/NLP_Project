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

    def create_embeddings(self, provider: str = "openai", model_name: Optional[str] = None):
        """
        Create embedding model based on provider.

        Args:
            provider: Embedding provider ('openai' or 'huggingface').
            model_name: Optional model name for HuggingFace.

        Returns:
            Embedding model instance.
        """
        if provider == "huggingface" and model_name:
            logger.info(f"Using HuggingFace embeddings: {model_name}")
            return HuggingFaceInstructEmbeddings(model_name=model_name)
        else:
            logger.info("Using OpenAI embeddings")
            return OpenAIEmbeddings()

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
        self.vectorstore = FAISS.load_local(path, embeddings)
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

        llm = ChatOpenAI(
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

        llm = ChatOpenAI(
            model_name=self.settings.default_model,
            streaming=True
        )

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
        llm = ChatOpenAI(model_name=self.settings.default_model, temperature=0.3)

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
        llm = ChatOpenAI(model_name=self.settings.default_model, temperature=0.3)

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
        llm = ChatOpenAI(model_name=self.settings.default_model, temperature=0.5)

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
