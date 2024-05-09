import streamlit as st
from dotenv import load_dotenv

import pdfplumber
import re
import logging
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatOpenAI
import numpy as np
import faiss
from faiss import IndexFlatL2 
from langchain.vectorstores import VectorStore

# Constants
CHUNK_SIZE = 1500

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            with pdfplumber.open(pdf) as pdf_reader:
                for page in pdf_reader.pages:
                    text += page.extract_text() or ''
        except Exception as e:
            logging.error(f"Failed to process PDF {pdf.name}: {str(e)}")
            st.error(f"Error processing {pdf.name}. Make sure it's not corrupted and is in a supported format.")
    return text


def clean_text(text):
    text = text.lower()
    patterns_to_remove = [
        r'\b[\w.-]+?@\w+?\.\w+?\b',  # emails
        r'\[[^\]]*\]',  # text in square brackets
        r'Figure \d+: [^\n]+',  # figure captions
        r'Table \d+: [^\n]+',  # table captions
        r'^Source:.*$',  # source lines
        r'[^\x00-\x7F]+',  # non-ASCII characters
        r'\bSee Figure \d+\b',  # references to figures
        r'\bEq\.\s*\d+\b',  # equation references
        r'\b(Table|Fig)\.\s*\d+\b',  # other ref styles
        r'<[^>]+>'  # HTML tags
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

# def get_text_chunks(text):
#     """
#     Splits the text into chunks based on predefined headers and a maximum chunk size.

#     Args:
#         text (str): Text to be chunked.

#     Returns:
#         list: List of text chunks.
#     """
#     # Define header pattern for splitting and checking
#     header_pattern = re.compile(r'\n\s*(Abstract|Introduction|Methods|Methodology|Results|Discussion|Conclusion)\s*\n', flags=re.IGNORECASE)
#     section_pattern = re.compile(r'^(Abstract|Introduction|Methods|Methodology|Results|Discussion|Conclusion)$', re.IGNORECASE)
    
#     sections = header_pattern.split(text)
#     chunks = []
#     current_chunk = []
#     current_length = 0

#     for section in sections:
#         if section_pattern.match(section):
#             if current_chunk:
#                 chunks.append(' '.join(current_chunk).strip())
#                 current_chunk = []
#                 current_length = 0
#             current_chunk.append(section)
#             current_length += len(section) + 1  # +1 for the space that will be added
#         else:
#             words = section.split()
#             for word in words:
#                 if current_length + len(word) + 1 > CHUNK_SIZE:
#                     if current_chunk:
#                         chunks.append(' '.join(current_chunk).strip())
#                     current_chunk = [word]
#                     current_length = len(word) + 1  # +1 for the space
#                 else:
#                     current_chunk.append(word)
#                     current_length += len(word) + 1  # +1 for the space

#     if current_chunk:
#         chunks.append(' '.join(current_chunk).strip())

#     return chunks

def get_text_chunks(text):
    header_pattern = re.compile(r'\n\s*(Abstract|Introduction|Methods|Methodology|Results|Discussion|Conclusion)\s*\n', flags=re.IGNORECASE)
    section_pattern = re.compile(r'^(Abstract|Introduction|Methods|Methodology|Results|Discussion|Conclusion)$', re.IGNORECASE)

    sections = header_pattern.split(text)
    chunks = []
    current_chunk = []
    current_length = 0
    current_offset = 0

    for section in sections:
        if section_pattern.match(section):
            if current_chunk:
                chunks.append((section, current_offset, current_offset + current_length))
                current_chunk = []
                current_length = 0
            current_chunk.append(section)
            current_offset += len(section) + 1
        else:
            words = section.split()
            for word in words:
                if current_length + len(word) + 1 > CHUNK_SIZE:
                    if current_chunk:
                        chunks.append((' '.join(current_chunk).strip(), current_offset, current_offset + current_length)) 
                    current_chunk = [word]
                    current_length = len(word) + 1 
                else:
                    current_chunk.append(word)
                    current_length += len(word) + 1 

    if current_chunk:
        chunks.append((' '.join(current_chunk).strip(), current_offset, current_offset + current_length)) 

    return [chunk[0] for chunk in chunks]


def get_vectorstore(text_chunks, model_type='huggingface', model_name=None):

    try:
        if model_type == 'huggingface' and model_name:
            embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)
            print(f"Using Hugging Face model: {model_name}")
        else:
            embeddings = OpenAIEmbeddings()
            print("Using default OpenAI embeddings.")
        
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        print("Vector store created successfully.")
        return vectorstore
    except Exception as e:
        logging.error("Failed to create vector store: %s", str(e))
        raise ValueError("Error in creating vector store. Please check the embedding model details.")



def get_conversation_chain(vectorstore):    
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

def handle_userinput(user_question):
    if "conversation" in st.session_state and st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.error("Please upload and process your PDF documents before asking questions.")

def main():
    load_dotenv()
    st.set_page_config(page_title="ResearchAI: Answer Extraction from Research Papers", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    st.header("ResearchAI: Answer Extraction from Research Papers :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
    
















