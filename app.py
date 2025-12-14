"""
ResearchAI: Interactive PDF Knowledge Extraction System

A Streamlit application that enables users to upload PDF documents and
ask natural language questions about their content using LLM-powered
retrieval-augmented generation.
"""

import logging
import re
from typing import List, Optional

import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI

from htmlTemplates import bot_template, css, user_template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 1500
SECTION_HEADERS = (
    'Abstract', 'Introduction', 'Methods', 'Methodology',
    'Results', 'Discussion', 'Conclusion'
)


def get_pdf_text(pdf_docs: List) -> str:
    """
    Extract text content from uploaded PDF documents.

    Args:
        pdf_docs: List of uploaded PDF file objects from Streamlit.

    Returns:
        Concatenated text content from all PDF pages.
    """
    text_parts = []
    for pdf in pdf_docs:
        try:
            with pdfplumber.open(pdf) as pdf_reader:
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except Exception as e:
            logger.error("Failed to process PDF %s: %s", pdf.name, str(e))
            st.error(
                f"Error processing {pdf.name}. "
                "Make sure it's not corrupted and is in a supported format."
            )
    return '\n'.join(text_parts)


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text by removing noise.

    Removes emails, HTML tags, figure/table captions, equation references,
    non-ASCII characters, and normalizes whitespace.

    Args:
        text: Raw text to clean.

    Returns:
        Cleaned and normalized text.
    """
    text = text.lower()
    patterns_to_remove = [
        r'\b[\w.-]+?@\w+?\.\w+?\b',      # emails
        r'\[[^\]]*\]',                     # text in square brackets
        r'Figure \d+: [^\n]+',             # figure captions
        r'Table \d+: [^\n]+',              # table captions
        r'^Source:.*$',                    # source lines
        r'[^\x00-\x7F]+',                  # non-ASCII characters
        r'\bSee Figure \d+\b',             # references to figures
        r'\bEq\.\s*\d+\b',                 # equation references
        r'\b(Table|Fig)\.\s*\d+\b',        # other ref styles
        r'<[^>]+>',                        # HTML tags
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.MULTILINE)
    return re.sub(r'\s+', ' ', text).strip()


def get_text_chunks(text: str) -> List[str]:
    """
    Split text into manageable chunks for embedding.

    Respects document section boundaries (Abstract, Introduction, etc.)
    while enforcing a maximum chunk size.

    Args:
        text: Text to split into chunks.

    Returns:
        List of text chunks.
    """
    headers_pattern = '|'.join(SECTION_HEADERS)
    header_regex = re.compile(
        rf'\n\s*({headers_pattern})\s*\n',
        flags=re.IGNORECASE
    )
    section_regex = re.compile(
        rf'^({headers_pattern})$',
        flags=re.IGNORECASE
    )

    sections = header_regex.split(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for section in sections:
        if section_regex.match(section):
            if current_chunk:
                chunks.append(' '.join(current_chunk).strip())
                current_chunk = []
                current_length = 0
            current_chunk.append(section)
            current_length = len(section) + 1
        else:
            words = section.split()
            for word in words:
                word_len = len(word) + 1
                if current_length + word_len > CHUNK_SIZE:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk).strip())
                    current_chunk = [word]
                    current_length = word_len
                else:
                    current_chunk.append(word)
                    current_length += word_len

    if current_chunk:
        chunks.append(' '.join(current_chunk).strip())

    return chunks


def get_vectorstore(
    text_chunks: List[str],
    model_type: str = 'openai',
    model_name: Optional[str] = None
) -> FAISS:
    """
    Create a FAISS vector store from text chunks.

    Args:
        text_chunks: List of text chunks to embed.
        model_type: Type of embedding model ('openai' or 'huggingface').
        model_name: Name of HuggingFace model (required if model_type='huggingface').

    Returns:
        FAISS vector store containing embedded text chunks.

    Raises:
        ValueError: If vector store creation fails.
    """
    try:
        if model_type == 'huggingface' and model_name:
            embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)
            logger.info("Using HuggingFace model: %s", model_name)
        else:
            embeddings = OpenAIEmbeddings()
            logger.info("Using OpenAI embeddings")

        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        logger.info("Vector store created successfully with %d chunks", len(text_chunks))
        return vectorstore
    except Exception as e:
        logger.error("Failed to create vector store: %s", str(e))
        raise ValueError(
            "Error creating vector store. Please check your API keys and model details."
        ) from e


def get_conversation_chain(vectorstore: FAISS) -> ConversationalRetrievalChain:
    """
    Create a conversational retrieval chain for Q&A.

    Args:
        vectorstore: FAISS vector store to use as retriever.

    Returns:
        Configured conversational retrieval chain.
    """
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )


def handle_userinput(user_question: str) -> None:
    """
    Process user question and display the conversation.

    Args:
        user_question: The question asked by the user.
    """
    if not st.session_state.get('conversation'):
        st.error("Please upload and process your PDF documents before asking questions.")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(
            template.replace("{{MSG}}", message.content),
            unsafe_allow_html=True
        )


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None


def process_documents(pdf_docs: List) -> None:
    """
    Process uploaded PDF documents and create conversation chain.

    Args:
        pdf_docs: List of uploaded PDF files.
    """
    if not pdf_docs:
        st.warning("Please upload at least one PDF document.")
        return

    with st.spinner("Processing documents..."):
        raw_text = get_pdf_text(pdf_docs)
        if not raw_text.strip():
            st.error("No text could be extracted from the uploaded documents.")
            return

        cleaned_text = clean_text(raw_text)
        text_chunks = get_text_chunks(cleaned_text)

        if not text_chunks:
            st.error("No text chunks were created. The documents may be empty.")
            return

        logger.info("Created %d text chunks from documents", len(text_chunks))

        try:
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.success(f"Successfully processed {len(pdf_docs)} document(s)!")
        except ValueError as e:
            st.error(str(e))


def main() -> None:
    """Main application entry point."""
    load_dotenv()

    st.set_page_config(
        page_title="ResearchAI: Answer Extraction from Research Papers",
        page_icon=":books:"
    )
    st.write(css, unsafe_allow_html=True)

    initialize_session_state()

    st.header("ResearchAI: Answer Extraction from Research Papers :books:")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
            type=['pdf']
        )
        if st.button("Process"):
            process_documents(pdf_docs)


if __name__ == '__main__':
    main()
